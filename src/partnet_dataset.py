"""General Dataset class for PartNet."""

import h5py
import os
import os.path as osp

import numpy as np
from scipy.sparse import coo_matrix

from src.class_utils import CLASSES, LEVELS
from src.tools import normalize_pc, rot_x
from src.datasets import MemoryDataset


class PartNetDataset(MemoryDataset):
    """
    Dataset utilities for PartNet.

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
        """Initialize dataset (see MemoryDataset's docstring)."""
        super().__init__(
            dset_name=dset_name,
            splits=splits,
            categories=categories,
            levels=levels,
            fold=fold,
            k_shot_seed=k_shot_seed,
            k_shot=k_shot,
            ann_path=ann_path,
            label_path=label_path,
            use_memory=use_memory,
            cross_instance=cross_instance,
            retriever_mode=retriever_mode,
            retriever_ckpt=retriever_ckpt,
            num_memories=num_memories,
            top_mem_pool_size=top_mem_pool_size,
            feat_path=feat_path,
            return_sem_labels=return_sem_labels,
            same_wild_augment_train=same_wild_augment_train,
            same_wild_augment_eval=same_wild_augment_eval,
            self_supervised_train=self_supervised_train,
            self_supervised_eval=self_supervised_eval
        )

    def init_semantic_labels(self):
        # init semantic part label for all categories
        self.partcls_offset = [0]  # parts of all classes sorted order
        self.num_partcls = 1  # 0 idx for no_object cls
        self.partcls_names = ["ignore"]
        for idx, cat in enumerate(CLASSES[self.dset_name]):
            # load meta data for whole category
            stat_cls = osp.join(self.label_path, f'{cat}.txt')
            with open(stat_cls, 'r') as fin:
                for item in fin.readlines():
                    part_idx = int(item.rstrip().split()[0])
                    self.partcls_names.append(str(item.rstrip().split()[1]))
                self.num_partcls += part_idx
                self.partcls_offset.append(part_idx + self.partcls_offset[-1])

        self.lvl_part_cls_mask = {
            lv: np.zeros(self.num_partcls)
            for lv in LEVELS[self.dset_name]
        }
        for lv in self.lvl_part_cls_mask:
            self.lvl_part_cls_mask[lv][0] = 1  # set no_object cls to be active

        # Mask for semantic parts of each class
        self.part_cls_mask = {}
        for cls_i in range(len(CLASSES[self.dset_name])):
            self.part_cls_mask[cls_i] = np.zeros(self.num_partcls)
            self.part_cls_mask[cls_i][0] = 1

        for idx, cat in enumerate(CLASSES[self.dset_name]):
            for level_id in LEVELS[self.dset_name]:
                # Load meta data files category level wise
                stat_cls_level = osp.join(
                    self.label_path,
                    f'{cat}-level-{level_id}.txt'
                )
                if not os.path.isfile(stat_cls_level):
                    continue
                with open(stat_cls_level, 'r') as fin:
                    for item in fin.readlines():
                        part_idx = int(item.rstrip().split()[0])

                        # mask populate for each level
                        self.lvl_part_cls_mask[level_id][
                            self.partcls_offset[idx] + part_idx
                        ] = 1

                        # mask populate for each cls
                        self.part_cls_mask[idx][
                            self.partcls_offset[idx] + part_idx
                        ] = 1

    def load_annos(self, name):
        """
        Load annotations.

        Returns: {
            'pts': points (P, 3),
            'gt_label': semantic labels (P,) in [0, num_class-1],
            'gt_label_parts': semantic labels (Np,) in [0, num_class-1],
            'gt_instance_mask': which part a point belongs to (P, Np),
            'cat': (str) name of class,
            'level': (int) PartNet level
        }
        where: P are points and Np are parts
        """
        if name in self.cache:
            return self.cache[name]

        # Load from h5py
        annos = {'gt_label', 'gt_mask', 'gt_valid', 'pts'}
        annos = {item: [] for item in annos}
        with h5py.File(self.ann_path + name, 'r') as fid:
            for key in annos:
                annos[key].append(fid[key][:])
        for key in annos:
            annos[key] = np.concatenate(annos[key], 0)

        # Align point cloud and normalize
        annos['pts'] = normalize_pc(annos['pts'][0])
        annos['pts'] = rot_x(annos['pts'], np.pi / 2)

        # Get instance labels
        _mask = annos['gt_mask'].astype(bool)[0]
        _valid = annos['gt_valid'].astype(bool)[0]
        tgt = np.zeros_like(_mask.T)
        tid = 0
        for k, v in enumerate(_valid):
            if not v:
                continue
            tgt[:, tid] = _mask[k]
            tid += 1
        tgt = tgt[:, :tid]

        # Pre-process the labels into one fat one hot vector
        # across all partnet part's sem cls
        annos['gt_label'] = annos['gt_label'].astype(np.uint8)
        cat, level, _ = name.split('-')
        cat_idx = CLASSES[self.dset_name].index(cat)
        offset_ = self.partcls_offset[cat_idx]
        no_obj_mask = annos['gt_label'][0] == 0
        gt_sem_label_pc = np.array(
            annos['gt_label'][0] + (~no_obj_mask * offset_)
        ).astype(int)
        gt_sem_label_parts = np.zeros(tgt.shape[1]).astype(int)
        gt_sem_label_parts_ = np.array([
            (
                np.max(gt_sem_label_pc[g_m])
                if len(gt_sem_label_pc[g_m]) > 0
                else 0
            )
            for g_m, v in zip(_mask, _valid)
            if v
        ]).astype(int)
        gt_sem_label_parts[:len(gt_sem_label_parts_)] = gt_sem_label_parts_

        # Cache and return
        annos = {
            'pts': annos['pts'],
            'gt_label': gt_sem_label_pc,
            'gt_label_parts': gt_sem_label_parts,
            'gt_instance_mask': coo_matrix(tgt),
            'cat': cat,
            'level': int(level),
            'fname': name
        }
        self.cache[name] = annos
        return annos
