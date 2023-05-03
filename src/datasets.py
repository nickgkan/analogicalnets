"""General Dataset class for all used datasets."""

import glob
import multiprocessing as mp
import os
import os.path as osp
import random

import numpy as np
import pandas as pd
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
import torch
from tqdm import tqdm

from src.class_utils import SPLITS, CLASSES, LEVELS
from src.loader_utils import (
    get_rotated_pcs,
    wild_parallel_augment
)
from src.PointWOLF import pointwolf
from src.tools import normalize_pc, rot_z


class MemoryDataset(Dataset):
    """
    Dataset utilities.

    This class is called with 3 splits (train/val/test) simultaneously,
    in order to share cache.
    The main script (or any other caller of this class) is responsible
    for setting the split each time, e.g.:
        self.split = 'val'
    """

    def __init__(
        self,
        # General arguments
        dset_name='partnet',
        splits=['train', 'val', 'test'],
        categories=["Chair", "Chair", "Chair"],
        levels=[0, 0, 0],
        fold=None,
        k_shot_seed=None,
        k_shot=5,
        ann_path='',
        label_path='',
        # Memory-specific arguments
        use_memory=False,
        cross_instance=False,
        retriever_mode=['random'] * 3,
        retriever_ckpt='',
        num_memories=1,
        top_mem_pool_size=20,
        feat_path='',
        # Train/test variation arguments
        return_sem_labels=False,
        same_wild_augment_train=False, same_wild_augment_eval=False,
        self_supervised_train=False, self_supervised_eval=False
    ):
        """
        Initialize dataset.

        Args:
            - dset_name (str): dataset name, e.g., 'partnet'
            - splits [str, str, str]: the splits to load
            - categories [str, str, str]: classes to load per split
                see SPLITS class_utils.py for special "categories"
            - levels [int, int, int]: levels to load per split
                give 0 for all levels jointly
            - fold (int or None): select fold for unseen categories
            - k_shot_seed (int or None): seed for selecting few-shot samples
            - k_shot (int): number of training examples for few-shot
                only useful if k_shot_seed is not None
            - ann_path (str): path to find annotations
            - label_path (str): path to find semantic labels

            - use_memory (bool): enable retriever/memory bank
            - cross_instance (bool): memory and target are different
            - retriever_mode [str, str, str]: retriever mode,
                see get_memory method for all options
            - retriever_ckpt (str): backbone to load for retrieval
            - num_memories (int): number of memories to fetch
            - top_mem_pool_size (int): number of similar memories
                to sample from
            - feat_path (str): path to store memory features

            - return_sem_labels (bool): return part semantics
            - same_wild_augment_train (bool): wildly augment input-memory
            - same_wild_augment_eval (bool): wildly augment input-memory
        """
        super().__init__()
        self.split = 'train'  # by default train, callers set this

        # Memory-agnostic part
        self.dset_name = dset_name
        self.k_shot_seed = k_shot_seed
        self.k_shot = k_shot
        self.ann_path = ann_path
        self.label_path = label_path
        print("Data loading categories ", categories, splits)
        self.categories = {}
        self.levels = {}
        self.instances = {}  # keep a pd.DataFrame per split
        for cat, lvl, split in zip(categories, levels, splits):
            self.categories[split] = self.create_fold(cat, fold)
            lvl_range = [lvl] if lvl in {1, 2, 3} else LEVELS[self.dset_name]
            self.levels[split] = sorted(lvl_range)
            self.instances[split] = self.make_splits(split)
        self.init_semantic_labels()
        self.cache = mp.Manager().dict()  # shared cache for point clouds

        # Train/test special arguments
        self.return_sem_labels = return_sem_labels
        self.same_wild_augment_train = same_wild_augment_train
        self.same_wild_augment_eval = same_wild_augment_eval
        self.self_supervised_train = self_supervised_train
        self.self_supervised_eval = self_supervised_eval

        # Memory-specific part
        self.use_memory = use_memory
        self.cross_inst = cross_instance
        self.retriever_mode = {s: m for s, m in zip(splits, retriever_mode)}
        self.num_memories = num_memories
        self.top_mem_pool_size = {
            s: m for s, m in zip(splits, top_mem_pool_size)
        }
        if use_memory and (cross_instance or num_memories > 1):
            _ckpt = retriever_ckpt.split('/')[-1].split('.')[0]
            self.feat_path = f'{feat_path}features_{dset_name}/' + _ckpt
            self.retriever_backbone = self.load_retriever(retriever_ckpt)
            self.memory_bank = {}  # store all features
            self.mem_pool = {}  # store pre-computed similarities
            for split in splits:
                feats, data = self.init_memory_bank(split)
                self.memory_bank[split] = {'feats': feats, 'data': data}
                if self.retriever_mode[split].startswith("canonical"):
                    self.mem_pool[split] = self.precompute_mem_pool(split)

    def create_fold(self, category, fold):
        classes = CLASSES[self.dset_name]
        if fold is None:
            categories = SPLITS.get(category, [str(category)])
        else:
            n_novel = len(classes) // 6
            mod = len(classes) // n_novel
            novels = [
                x for i, x in enumerate(classes)
                if i % mod == fold
            ][:n_novel]
            bases = [x for x in classes if x not in novels]
            if "novel" in category:
                categories = novels[:len(SPLITS[category])]
            elif category in SPLITS:
                categories = bases[:len(SPLITS[category])]
            else:
                categories = [str(category)]
        return categories

    def make_splits(self, split):
        # Path to data
        data_path = [
            f'{self.ann_path}{cat}-{level}-orig/'
            for cat in self.categories[split]
            for level in self.levels[split]
        ]

        # Populate a data frame with all files
        fnames, cat_names, lvls = [], [], []
        for cat_path in data_path:
            cat, lvl, _ = cat_path.replace(self.ann_path, '').split('-')
            cat_insts = self._get_per_cat_split_files(cat_path, split)
            if self.k_shot_seed is not None and split == 'train' and cat_insts:
                rng = np.random.RandomState(self.k_shot_seed)
                inds = rng.permutation(range(len(cat_insts)))[:self.k_shot]
                cat_insts = np.asarray(cat_insts)[inds].tolist()
            fnames.extend(cat_insts)
            cat_names.extend([cat] * len(cat_insts))
            lvls.extend([int(lvl)] * len(cat_insts))
        return pd.DataFrame.from_dict({
            'fname': fnames, 'cat': cat_names, 'lvl': lvls,
            'rot': np.zeros(len(fnames)).tolist()
        })

    def init_semantic_labels(self):
        # Overwrite this
        self.lvl_part_cls_mask = {
            lv: np.zeros(3) for lv in LEVELS[self.dset_name]
        }
        self.part_cls_mask = {
            c: np.zeros(3) for c in CLASSES[self.dset_name]
        }

    def _get_per_cat_split_files(self, cat_path, split):
        split_ = 'train' if split == 'train' else 'test'
        cat_insts = sorted(glob.glob(osp.join(cat_path, f"{split_}0*.h5")))
        cat_insts = [name.replace(self.ann_path, '') for name in cat_insts]
        return cat_insts

    @staticmethod
    def load_retriever(retriever_ckpt):
        # Load pretrained pointnet to compute features
        ckpnt = torch.load(retriever_ckpt)
        from models.analogical_networks import AnalogicalNetworks
        pointnet_backbone = AnalogicalNetworks(out_dim=264).pc_btlnck
        pointnet_backbone.load_state_dict(
            {
                key.replace('pc_btlnck.', ''): val
                for key, val in ckpnt["model_state_dict"].items()
                if key.startswith('pc_btlnck')
            },
            strict=True
        )
        pointnet_backbone.eval()
        return pointnet_backbone.cuda()

    def init_memory_bank(self, split):
        """
        Initialize memory bank.

        Returns:
            - a feature array (num_all_memories, F)
            - pd.DataFrame containing idx, filename, class, level
        """
        # Store all memory features (only has to be ran once)
        self._save_all_features('train')

        # Fill memory bank
        classes = CLASSES[self.dset_name]
        levels = LEVELS[self.dset_name]
        feats = {lv: {c: [] for c in classes} for lv in levels}
        data = {lv: {c: [] for c in classes} for lv in levels}
        _path = self.feat_path

        for lvl in levels:
            for cat in classes:
                # Train/val cannot see test memory
                if cat not in self.categories['train']:
                    continue
                if not osp.exists(f'{_path}/{cat}-{lvl}-train.npy'):
                    continue

                # Load features and optionally filter for few-shot
                _feats = np.load(f'{_path}/{cat}-{lvl}-train.npy')
                _data = pd.read_csv(f'{_path}/{cat}-{lvl}-train.csv')
                if self.k_shot_seed and cat in self.categories[split]:
                    rng = np.random.RandomState(self.k_shot_seed)
                    inds = rng.permutation(range(len(_feats)))[:self.k_shot]
                    _feats = _feats[inds]
                    _data = _data.iloc[inds.tolist()].reset_index()
                feats[lvl][cat] = _feats
                data[lvl][cat] = _data

            # Merge classes per level for class-agnostic retrievers
            if 'class_based' not in self.retriever_mode[split]:
                feats[lvl] = np.concatenate([
                    feats[lvl][cat] for cat in classes if len(feats[lvl][cat])
                ])  # (N, F)
                data[lvl] = pd.concat(
                    data[lvl][cat] for cat in classes if len(data[lvl][cat])
                ).reset_index()
        return feats, data

    def precompute_mem_pool(self, split):
        # Store all features (only has to be ran once)
        split_ = 'train' if split == 'train' else 'test'
        self._save_all_features(split_)

        # Fill a feature bank that corresponds to stored annotations
        levels = self.levels[split]
        cats = self.categories[split]
        feats = []
        for (cat, lvl) in [(cat, level) for cat in cats for level in levels]:
            if not osp.exists(f'{self.feat_path}/{cat}-{lvl}-{split_}.npy'):
                continue
            _feats = np.load(f'{self.feat_path}/{cat}-{lvl}-{split_}.npy')
            if self.k_shot_seed is not None and split == 'train':
                rng = np.random.RandomState(self.k_shot_seed)
                inds = rng.permutation(range(len(_feats)))[:self.k_shot]
                _feats = _feats[inds]
            feats.append(_feats)
        feats = np.concatenate(feats)

        # Class filter
        if 'class_based' not in self.retriever_mode[split]:
            files = {lv: [] for lv in levels}
        else:
            files = {lv: {c: [] for c in cats} for lv in levels}

        # Compute similarities
        for lvl in levels:
            if 'class_based' in self.retriever_mode[split]:
                for cat in cats:
                    if len(self.memory_bank[split]['feats'][lvl][cat]) == 0:
                        continue
                    m_feats = self.memory_bank[split]['feats'][lvl][cat]
                    sims = np.matmul(feats, m_feats.T)
                    _, _files = torch.topk(
                        torch.from_numpy(sims),
                        k=min(self.top_mem_pool_size[split], sims.shape[1]),
                        dim=1
                    )
                    files[lvl][cat] = _files.numpy().tolist()
            else:
                m_feats = self.memory_bank[split]['feats'][lvl]
                sims = np.matmul(feats, m_feats.T)
                _, _files = torch.topk(
                    torch.from_numpy(sims),
                    k=min(self.top_mem_pool_size[split], sims.shape[1]),
                    dim=1
                )
                files[lvl] = _files.numpy().tolist()
        return files

    @torch.no_grad()
    def _save_all_features(self, split):
        _path = self.feat_path
        os.makedirs(_path, exist_ok=True)
        data_path = [
            f'{self.ann_path}{cat}-{level}-orig/'
            for cat in CLASSES[self.dset_name]
            for level in LEVELS[self.dset_name]
        ]

        # Compute feature for every sample (only once!)
        for cat_path in tqdm(data_path):
            cat_insts = self._get_per_cat_split_files(cat_path, split)
            split_ = 'train' if split == 'train' else 'test'
            cat_file = (
                f'{_path}/'
                + cat_path.replace(self.ann_path, '')[:-1].replace('orig', '')
                + f"{split_}.npy"
            )
            if osp.exists(cat_file) or not cat_insts:
                continue
            print('Saving features for ' + cat_file)
            feats = []
            fnames, cat_names, lvls, rots = [], [], [], []
            for fname in tqdm(cat_insts):
                annos = self.load_annos(fname)
                # Batch point clouds
                angle = 360
                pts, rots_ = get_rotated_pcs(annos['pts'], angle)
                pc = torch.stack([
                    torch.from_numpy(normalize_pc(pt)).float()
                    for pt in pts
                ])
                # Store names, classes etc.
                fnames.extend([fname] * len(pc))
                cat_names.extend([annos['cat']] * len(pc))
                lvls.extend([int(annos['level'])] * len(pc))
                rots.extend(rots_.tolist())
                out_dict = self.retriever_backbone(pc.cuda())
                feats.append(out_dict['lat_feats'].mean(1).cpu())
            feats = F.normalize(torch.cat(feats), p=2, dim=1)  # (N, feat_dim)
            np.save(cat_file, feats.numpy())
            df = pd.DataFrame.from_dict({
                'fname': fnames, 'cat': cat_names, 'lvl': lvls, 'rot': rots
            })
            df.to_csv(cat_file.replace('.npy', '.csv'))

    def load_annos(self, name):
        """Load annotations given file name."""
        pass

    def augment(self, pc):
        aug_pc = pc.copy()

        # Rotation along up-axis/Z-axis
        rot_angle = np.random.random() * np.pi / 2 - np.pi / 4
        aug_pc = rot_z(aug_pc, rot_angle)

        # PointWOLF augmentation
        if np.random.random() > 0.2:
            if np.random.random() < 0.6:
                n_transform = 4
            else:
                n_transform = np.random.randint(4, 10)
            aug_pc = pointwolf(aug_pc, n_local_transformations=n_transform)

        # Add jittering noise
        if np.random.random() > 0.2:
            sigma = 0.01
            clip = 0.02
            N, C = aug_pc.shape
            aug_pc += np.clip(sigma * np.random.randn(N, C), -clip, clip)

        # Scale anisotropic
        if np.random.random() > 0.1:
            scale = np.diag(0.5 + np.random.random(3))
            aug_pc[:, :3] = np.dot(aug_pc[:, :3], scale)
        return aug_pc

    def get_memory(self, ref_dict):
        """
        Get memory given the input.

        Args:
            - ref_dict = {
                'index': int target index in the loader,
                'feat': float array (F,), the feature of the target pc,
                'aug_dict': dict, see wild_parallel_augment in loader_utils.py,
                'cat': class name,
                'lvl': level id
            }

        Retrieval options are:
            - same: fetch thyself (within-instance training)
            - random: fetch any memory
            - class_based: any memory of the same class
            - canonical: class-agnostic assuming full cloud in canonical pose
            - canonical_class_based: class-aware canonical
        """
        if ref_dict.get('index', None) is not None:
            index = ref_dict['index']
            df = self.instances[self.split].iloc[index]
            _cat, fname, _lvl = df['cat'], df['fname'], df['lvl']
        else:
            _cat, fname, _lvl = ref_dict['cat'], '', ref_dict['lvl']
        if self.cross_inst or self.num_memories > 1:
            m_df = self.memory_bank[self.split]['data'][_lvl]
            m_feats = self.memory_bank[self.split]['feats'][_lvl]
        is_cross = (
            self.cross_inst
            and not (self.split == 'train' and np.random.random() > 0.7)
        )

        # Within-instance single-memory
        if not is_cross and self.num_memories == 1:
            files = []

        # Any memory
        elif self.retriever_mode[self.split] == "random":
            files = list(range(len(m_df)))

        # Any memory of the same class
        elif self.retriever_mode[self.split] == "class_based":
            m_df = m_df[_cat]
            files = list(range(len(m_df)))

        # Use retriever for cross-instance or multi-memory
        elif self.retriever_mode[self.split].startswith("can"):
            if self.retriever_mode[self.split].endswith("class_based"):
                m_feats = m_feats[_cat]
                m_df = m_df[_cat]
            # pre-computed
            files = self.mem_pool[self.split][_lvl]
            if self.retriever_mode[self.split].endswith("class_based"):
                files = files[_cat]
            files = files[index]

        # Load memory annotations (h5 files)
        if files != []:
            files = random.sample(files, min(self.num_memories, len(files)))
            fnames = [m_df.fname[f] for f in files]
            rots = [m_df.rot[f] for f in files]
        else:
            fnames, rots = [], []
        if not is_cross:  # within-instance, self+memories
            fnames = [fname] + fnames[:-1]
            rots = [0] + rots[:-1]
        annos_mem = [self.load_annos(cf) for cf in fnames]
        pc_a = [
            rot_z(a['pts'], rot) if rot > 0 else a['pts']
            for a, rot in zip(annos_mem, rots)
        ]

        # Equally (wildy) augment both point cloud streams
        _same_wild_augment = (
            (
                self.same_wild_augment_train and self.split == 'train'
                or self.same_wild_augment_eval and self.split != 'train'
            )
        )
        if _same_wild_augment:
            pc_a, _ = wild_parallel_augment(
                pc_a,
                flip=self.split == 'train' and _same_wild_augment,
                translate=False,
                rotate_x=False,
                rotate_y=False,
                rotate_z=_same_wild_augment,
                aug_dict=ref_dict['aug_dict']
            )

        # Standard augmentation
        if self.split == 'train':
            pc_a = [self.augment(pc) for pc in pc_a]

        labels_mem = np.zeros((
            len(annos_mem),
            len(annos_mem[0]['gt_instance_mask'].toarray()),
            max(a['gt_instance_mask'].toarray().shape[1] for a in annos_mem)
        ))
        for a, ann in enumerate(annos_mem):
            labels_mem[a, :, :ann['gt_instance_mask'].toarray().shape[1]] = \
                ann['gt_instance_mask'].toarray()
        ret_dict = {
            'pc_mem': np.stack(pc_a).astype(np.float32),
            'pc_labels_mem': labels_mem.astype(np.int64),
            'class_names_mem': annos_mem[0]['cat'],
            'is_cross': is_cross
        }
        if self.return_sem_labels:
            parts_mem = np.zeros((
                len(annos_mem),
                max(len(a['gt_label_parts']) for a in annos_mem)
            ))
            for a, ann in enumerate(annos_mem):
                parts_mem[a, :len(ann['gt_label_parts'])] = \
                    ann['gt_label_parts']
            ret_dict.update({
                'part_sem_labels_mem': parts_mem
            })
        return ret_dict

    def get_target(self, index):
        """Get current batch for input index."""
        # Point cloud sample name
        df = self.instances[self.split].iloc[index]

        # Load the h5 file
        annos = self.load_annos(df['fname'])
        pc = annos['pts']
        pc_label = annos['gt_instance_mask'].toarray()

        # Equally (wildy) augment both point cloud streams
        _same_wild_augment = (
            self.same_wild_augment_train and self.split == 'train'
            or self.same_wild_augment_eval and self.split != 'train'
        )
        pc_aug, aug_dict = wild_parallel_augment(
            [pc.copy()],
            flip=self.split == 'train' and _same_wild_augment,
            translate=False,
            rotate_x=False,
            rotate_y=False,
            rotate_z=_same_wild_augment,
            aug_dict=None
        )
        pc_aug = pc_aug[0]

        # Standard augmentation
        if (not self.cross_inst and self.use_memory) or self.split == 'train':
            pc_aug = self.augment(pc_aug)

        ret_dict = {
            'pc': pc_aug.astype(np.float32),
            'pc_labels': pc_label.astype(np.int64),
            'class_names': annos['cat'],
            'level': df['lvl'],
            'index': index
        }
        if self.return_sem_labels:
            ret_dict.update({
                'part_sem_labels': annos['gt_label_parts'],  # (Np,)
                'sem_lvl_mask': self.lvl_part_cls_mask[df['lvl']],
                'sem_cls_mask': self.part_cls_mask[
                    CLASSES[self.dset_name].index(df['cat'])
                ]
            })
        ret_dict.update(aug_dict)
        return ret_dict, aug_dict

    def __getitem__(self, index):
        ret_dict, aug_dict = self.get_target(index)
        if self.use_memory:
            ret_dict.update(self.get_memory({
                'index': index,
                'aug_dict': aug_dict
            }))
        return ret_dict

    def __len__(self):
        """Return number of instances."""
        return len(self.instances[self.split])


def custom_collate(batch):
    if 'pc_labels' in batch[0]:
        max_num_parts = max(item['pc_labels'].shape[1] for item in batch)
    qs = ['pc_labels', 'part_sem_labels']
    for item in batch:
        for q in qs:
            if q in item:
                item[q] = _pad_array(item[q], max_num_parts)
    if 'pc_labels_mem' in batch[0]:
        max_mem_parts = max(item['pc_labels_mem'].shape[-1] for item in batch)
    qs = ['pc_labels_mem', 'part_sem_labels_mem']
    for item in batch:
        for q in qs:
            if q in item:
                item[q] = _pad_array(item[q], max_mem_parts)
    return default_collate(batch)


def _pad_array(arr, pad_limit):
    return np.concatenate((
        arr,
        np.zeros(arr.shape[:-1] + (pad_limit - arr.shape[-1],))
    ), -1)
