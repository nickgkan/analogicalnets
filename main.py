import argparse
import os
import os.path as osp

import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import wandb

from src.matching_loss import SetCriterion
from src.metrics import (
    AccuracyEvaluator,
    APEvaluator, ARIEvaluator,
    MemoryPropAccEvaluator, IoUEvaluator,
    PerShapePanopticEvaluator
)
from src.tools import visualize_mesh, COLORS
from src.datasets import custom_collate

print("PID : ", os.getpid())
ALL_CLASSES = 481


class Trainer:
    """Train/test models on correspondence."""

    def __init__(self, model, data_loaders, args):
        self.model = model
        self.data_loaders = data_loaders
        self.args = args

        self.optimizer = AdamW(
            model.parameters(),
            lr=args.lr, weight_decay=args.weight_decay
        )

        self.criterion = SetCriterion(
            use_identity=not args.hungarian,
            no_obj_coef=args.negative_obj_weight,
            cls_cost=1.0 if args.model == 'xy_3ddetr' else 4.0,
            background_coef=args.background_coef
        )

        if not args.eval or args.ft_epoch > 0:
            self.writer = SummaryWriter(f'runs/{args.run_name}')
        if self.args.eval and self.args.visualize_wandb:
            wandb.init(name=self.args.run_name)

        self.val_freq = 1

    def run(self):
        # Set
        start_epoch = 0
        val_acc_prev_best = -1.0

        # Load
        if self.args.bootstrap is not None:
            self._bootstrap()
        if osp.exists(self.args.ckpnt):
            start_epoch, val_acc_prev_best = self._load_ckpnt()

        # Eval?
        if self.args.eval or start_epoch >= self.args.epochs:
            mode = 'val'
            if self.args.eval_train:
                mode = 'train'
            if self.args.eval_test:
                mode = 'test'
            self.model.eval()
            self.multiseed_eval(mode)
            return self.model

        # Go!
        for epoch in range(start_epoch, self.args.epochs):
            print("Epoch: %d/%d" % (epoch + 1, self.args.epochs))
            self.model.train()
            # Train
            self.train_test_loop('train', epoch)
            # Validate
            if (epoch + 1) % self.val_freq == 0:
                print("\nValidation")
                self.model.eval()
                with torch.no_grad():
                    val_acc = self.train_test_loop('val', epoch)[0]
                    # val_acc = 0

                # Store
                if val_acc >= val_acc_prev_best:
                    print("Saving Checkpoint")
                    torch.save({
                        "epoch": epoch + 1,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "best_acc": val_acc,
                        "best_acc_epoch": epoch + 1,
                        "args": self.args
                    }, self.args.ckpnt)
                    val_acc_prev_best = val_acc
                else:
                    print("Updating Checkpoint")
                    checkpoint = torch.load(self.args.ckpnt)
                    checkpoint["epoch"] += self.val_freq
                    torch.save(checkpoint, self.args.ckpnt)

        self.writer.close()
        return self.model

    def average_weights(self):
        ckpnt = torch.load(self.args.bootstrap)
        sdict = dict(ckpnt["model_state_dict"])
        params = dict(self.model.state_dict())

        for name, param in params.items():
            if name in sdict:
                params[name].data.copy_(
                    0.5 * param.data + 0.5 * sdict[name].data
                )
        self.model.load_state_dict(params)

    def _bootstrap(self):
        ckpnt = torch.load(self.args.bootstrap)
        self.model.load_state_dict(ckpnt["model_state_dict"], strict=False)
        print(f'Bootstrapping from {self.args.bootstrap} at {ckpnt["epoch"]}')

    def _load_ckpnt(self):
        ckpnt = torch.load(self.args.ckpnt)
        self.model.load_state_dict(ckpnt["model_state_dict"], strict=False)
        if not self.args.eval:
            self.optimizer.load_state_dict(ckpnt["optimizer_state_dict"])
        start_epoch = ckpnt["epoch"]
        val_acc_prev_best = ckpnt['best_acc']
        print(
            f'Loading checkpoint {self.args.ckpnt} ',
            f'at {ckpnt["best_acc_epoch"]} with acc {val_acc_prev_best}'
        )

        if self.args.train_with_semantics and self.args.ft_epoch > 0:
            print("!!! Re-init the last cls head")
            for i in range(len(self.model.sem_cls_heads)):
                for j in range(len(self.model.sem_cls_heads[i])):
                    m = self.model.sem_cls_heads[i][j]  # last layer for cls
                    if isinstance(m, torch.nn.Conv1d):
                        print("zeroing out ", m)
                        torch.nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            torch.nn.init.zeros_(m.bias)
        return start_epoch, val_acc_prev_best

    def _prepare_inputs(self, batch):
        device = self.args.device
        ret = {
            'pc_2': batch['pc'].to(device).float(),
            'mask_2': batch['pc_labels'].to(device).bool(),
            'pad_2': batch['pc_labels'].any(-2).to(device).long(),
            'names': batch['class_names'],
            'level': batch['level'].to(device).long().reshape(-1),
            'is_cross': batch.get(
                'is_cross', torch.ones(len(batch['pc']))
            ).to(device).bool()
        }
        if 'pc_mem' in batch:
            ret.update({
                'pc_1': [
                    b[:, 0].to(device).float()
                    for b in batch['pc_mem'].split(1, dim=1)
                ],
                'mask_1': [
                    b[:, 0].to(device).bool()
                    for b in batch['pc_labels_mem'].split(1, dim=1)
                ],
                'pad_1': batch['pc_labels_mem'].any(-2).to(device).long(),
                'names_mem': batch['class_names_mem']
            })
        # Inputs for semantic segmentation
        if self.args.use_semantics:
            ret.update({
                'part_sem_labels':
                    batch['part_sem_labels'].long().to(device),
                'sem_lvl_mask': batch['sem_lvl_mask'].to(device),
                'sem_cls_mask': batch['sem_cls_mask'].to(device)
            })
            if 'part_sem_labels_mem' in batch:
                ret.update({
                    'part_sem_labels_mem': [
                        b[:, 0].long().to(device)
                        for b in
                        batch['part_sem_labels_mem'].split(1, dim=1)
                    ]
                })
        if 'pc_1' not in ret:
            ret['pc_1'] = [ret['pc_2']]
            ret['mask_1'] = [ret['mask_2']]
            ret['pad_1'] = ret['pad_2'][:, None]
            if self.args.use_semantics:
                ret['part_sem_labels_mem'] = [ret['part_sem_labels']]
        return ret

    def multiseed_eval(self, mode):
        # Do eval/few-shot-eval over multiple seeds
        _macro = []
        _chkpt = str(self.args.ckpnt)
        self.val_freq = 90
        for seed_i in range(self.args.eval_multitask):
            print(f"\ntask {seed_i} out of {self.args.eval_multitask - 1}")
            if seed_i > 0 and (not self.args.eval_forgetting or self.args.eval_test):
                # Load a new task ie new set of fewshot train samples
                rng = np.random.RandomState(seed_i)
                rand_seed = rng.randint(10000)
                self.args.k_shot_seed = rand_seed
                print("Loading new seed fewshot task ", seed_i, rand_seed)
                self.data_loaders = fetch_loaders(self.args)

            if self.args.ft_epoch:
                # Make model finetune-ready
                self.args.epochs = self.args.ft_epoch
                self.args.eval = False
                self.args.ckpnt = _chkpt.replace('.pt', f'{seed_i}.pt')
                self.model = init_model(self.args).to(self.args.device)
                self.optimizer = AdamW(
                    self.model.parameters(),
                    lr=self.args.lr, weight_decay=self.args.weight_decay
                )
                self.run()
                # Disable finetuning mode
                self.args.eval = True
                self._load_ckpnt()
                self.model.eval()

            if self.args.eval_forgetting or self.args.use_finetuned:
                self.args.ckpnt = _chkpt.replace('.pt', f'{seed_i}.pt')
                self.model = init_model(self.args).to(self.args.device)
                self._load_ckpnt()
                # self.average_weights()
                self.model.eval()

            with torch.no_grad():
                metrics = self.train_test_loop(mode)
            _macro.append(metrics)

        _macro = np.array(_macro)
        print(f'eval_multitask : Macro {_macro.mean(0)} std {_macro.std(0)}')

    def train_test_loop(self, mode='train', epoch=1000):
        # Set the mode of the dataset!
        self.data_loaders.dataset.split = mode

        # Counters to store metrics
        self._init_counters()
        total_loss = 0

        # Main loop
        for step, ex in tqdm(enumerate(self.data_loaders)):
            # Batch form
            inputs = self._prepare_inputs(ex)

            # Forward pass
            # scores [(B, P, Q)], objectness [(B, Q)], sem_logits [(B, Q, S)]
            scores, objectness, sem_logits = self.model(
                inputs['pc_1'], inputs['mask_1'], inputs['pc_2'],
                level_id=inputs['level'] - 1
            )

            # Semantic pred process: level + propagate labels
            if self.args.use_semantics:
                sem_logits = self._process_sem_logits(sem_logits, inputs)

            # Losses
            loss, inds = self._compute_loss(
                scores, objectness, sem_logits,
                self.criterion, inputs['mask_2'].float(),
                inputs.get('part_sem_labels', None),
                torch.cat(inputs['part_sem_labels_mem'], -1)
                if 'part_sem_labels_mem' in inputs else None,
                cls_mask=(
                    inputs['sem_cls_mask'] * inputs['sem_lvl_mask']
                    if 'sem_cls_mask' in inputs else None
                ),
                is_cross=inputs['is_cross']
            )
            total_loss += loss.mean().item()

            # Update
            if mode == 'train' and not self.args.eval:
                self.optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                self.optimizer.zero_grad()

            # Classless evaluation (ARI)
            self.ari_evaluator.step(
                scores[-1].sigmoid() * objectness[-1][:, None].sigmoid(),
                inputs['mask_2'],
                inputs['names'], (inputs['level'] - 1).tolist()
            )

            # Semantic evaluation
            if self.args.val_with_semantics and mode != 'train':
                # Label propagation accuracy on part level
                self.mem_prop_acc_evaluator.step(
                    torch.cat(inputs['part_sem_labels_mem'], -1),
                    inputs['part_sem_labels'],
                    inds
                )

                # Panoptic Quality
                if self.args.compute_pq:
                    self.panop_evaluator.step(
                        (
                            scores[-1].sigmoid()
                            * objectness[-1][:, None].sigmoid()
                        ),
                        sem_logits[-1],
                        inputs['mask_2'].float(),
                        torch.matmul(
                            inputs['mask_2'].float(),
                            inputs['part_sem_labels'][..., None].float()
                        ).squeeze(-1),  # (B, P)
                        inputs['names'], (inputs['level'] - 1).tolist()
                    )

                # Instance segmentation mAP
                self.ap_evaluator.step(
                    scores=(
                        scores[-1].sigmoid()
                        * objectness[-1][:, None].sigmoid()
                    ),
                    sem_logits=sem_logits[-1].softmax(-1),
                    objectness=objectness[-1],
                    target_masks=inputs['mask_2'].float(),
                    per_part_sem_labels=inputs['part_sem_labels'],
                    class_names=inputs['names'],
                    levels=inputs['level'].tolist()
                )

                # Semantic segmentation mIoU/mAcc
                self.iou_evaluator.step(
                    scores=(
                        scores[-1].sigmoid()
                        * objectness[-1][:, None].sigmoid()
                    ),
                    sem_logits=sem_logits[-1],
                    per_point_sem_labels=torch.matmul(
                        inputs['mask_2'].float(),
                        inputs['part_sem_labels'][..., None].float()
                    ).squeeze(-1),  # (B, P)
                    class_names=inputs['names'],
                    levels=inputs['level'].tolist(),
                    cls_mask=inputs['sem_cls_mask'] * inputs['sem_lvl_mask']
                )

            # Retriever evaluation
            if self.args.use_memory:
                self.retr_acc_evaluator.step(
                    inputs['names_mem'],
                    inputs['names']
                )

            # Visualization
            if self.args.visualize_wandb:
                # for b in range(len(inputs['pc_1'][0])):
                #     if sample_aris[b] > 0.7:
                #         continue
                b = 0
                for a in range(len(inputs['pc_1'])):
                    self._visualize(
                        inputs, scores, objectness, inds,
                        step * len(inputs['pc_1']) + a,
                        mem_id=a, b_id=b
                    )
                    if self.args.use_semantics:
                        self._visualize_sem(
                            inputs, scores, objectness, sem_logits,
                            step * len(inputs['pc_1']) + a,
                            mem_id=a, b_id=b
                        )

        # Post-loop: summarize metrics
        if not self.args.eval:
            # Losses
            self.writer.add_scalar(
                f'loss/{mode}', total_loss / len(self.data_loaders),
                epoch
            )
            # ARI
            self.writer.add_scalar(
                f'ARI/{mode}', self.ari_evaluator.get_mean_cls_ari(),
                epoch
            )
            # mAP
            if self.args.use_semantics:
                self.writer.add_scalar(
                    f'mAP/{mode}', self.ap_evaluator.get_map(),
                    epoch
                )

        # Post-loop: summarize metrics
        if self.args.eval:
            self.ari_evaluator.print_class_stats()
            if self.args.val_with_semantics:
                self.ap_evaluator.print_class_stats()
                self.iou_evaluator.print_class_stats()

        return self.print_metrics(mode)

    def print_metrics(self, mode):
        if self.args.ft_epoch == 0 or mode in ["test", "val"]:
            print('-' * 20)
            print('Mode ', mode)
            print(self.ari_evaluator)
            print('-' * 20)
            print("Retriever classification accuracy:")
            print(self.retr_acc_evaluator)
            print('-' * 20)
            print(self.mem_prop_acc_evaluator)
            print('-' * 20)
            print(self.panop_evaluator)
            print('-' * 20)
            print(self.ap_evaluator)
            print('-' * 20)
            print(self.iou_evaluator)
        elif self.args.ft_epoch:
            print("Train ARI:", self.ari_evaluator.get_mean_cls_ari())
        return [
            self.ari_evaluator.get_mean_cls_ari(),
            self.ap_evaluator.get_map(),
            self.iou_evaluator.get_macc(),
            self.iou_evaluator.get_miou(),
            self.panop_evaluator.get_macro_iou(),
            self.panop_evaluator.get_macro_pq()
        ]

    def _init_counters(self):
        self.retr_acc_evaluator = AccuracyEvaluator()
        self.ap_evaluator = APEvaluator(ALL_CLASSES)
        self.ari_evaluator = ARIEvaluator()
        self.mem_prop_acc_evaluator = MemoryPropAccEvaluator()
        self.iou_evaluator = IoUEvaluator(ALL_CLASSES)
        self.panop_evaluator = PerShapePanopticEvaluator(ALL_CLASSES)

    def _process_sem_logits(self, sem_logits, inputs):
        # sem_logits: list of (B, Q, S)
        processed_logits = []
        for layer_i in range(len(sem_logits)):
            logits = sem_logits[layer_i]
            # Keep only the classes of the current level
            logits = (
                logits
                * inputs['sem_lvl_mask'].unsqueeze(1)
                - 1e7 * (1 - inputs['sem_lvl_mask']).unsqueeze(1)
            )
            # Allow class-specific semantic fine-tuning
            if self.args.ft_epoch > 0:
                logits = (
                    logits
                    * inputs['sem_cls_mask'].unsqueeze(1)
                    - 1e7 * (1 - inputs['sem_cls_mask']).unsqueeze(1)
                )
            # Propagate the mem init detr query labels using mem labels
            if self.args.use_memory and not self.args.no_mem_decoding:
                mem_part_labels = F.one_hot(
                    torch.cat(inputs['part_sem_labels_mem'], -1),
                    num_classes=ALL_CLASSES
                )  # [B, memQ, S]
                from_mem_mask = torch.zeros_like(logits)
                from_mem_mask[:, :mem_part_labels.size(1)] = 1.0
                from_mem_logits = torch.zeros_like(logits)
                from_mem_logits[:, :mem_part_labels.size(1)] = mem_part_labels
                logits = (
                    logits
                    * (1 - from_mem_mask)
                    + 1e7 * from_mem_logits * from_mem_mask
                )
            processed_logits.append(logits)
        _grad = sem_logits[-1].requires_grad
        assert all(lgt.requires_grad == _grad for lgt in processed_logits)
        return processed_logits  # list of (B, Q, S)

    def _compute_loss(self, scores, objectness, sem_logits,
                      criterion, mask_tgt, label_tgt, label_mem, cls_mask,
                      is_cross):
        if self.args.supervise_last_only:
            scores = [scores[-1]]
            objectness = [objectness[-1]]
            sem_logits = (
                [sem_logits[-1]]
                if self.args.train_with_semantics
                else None
            )
        loss = 0
        inds = None
        for layer_i, (sc, obj) in enumerate(zip(scores, objectness)):
            if self.args.train_with_semantics:
                s_logit = sem_logits[layer_i]
            else:
                s_logit = None
            loss_, inds = criterion(
                sc, obj, mask_tgt,
                sem_logits=s_logit,
                tgt_labels=label_tgt,
                cls_mask=cls_mask,
                is_cross=is_cross,
                use_identity_for_within=self.args.model in (
                    'analogical_nets', 'multi_mem'
                )
            )
            loss = loss + loss_
        return loss, inds  # only last inds for visualization

    def _visualize_lbl_pc(self, pc, scores, inds=None,
                          filter_objectness=False, objectness=None,
                          use_only_mem=False, mem_id=0,
                          filter_score=False, add_anchor_offset=False,
                          parts_per_mem=(0, 16)):
        color_ids = scores.argmax(1).cpu()
        if filter_score:
            # Show only confident points
            color_ids[scores.max(1)[0].cpu() < 0.2] = -1
        if filter_objectness and objectness is not None:
            # Show only confident queries
            for o, obj in enumerate(objectness):
                if obj < 0.5:
                    color_ids[color_ids == o] = 31
        if inds is not None:
            # Plot target with parsed colors
            new_colors = -torch.ones_like(color_ids)
            for matched, gt in zip(inds[0], inds[1]):
                new_colors[color_ids == gt] = matched
            color_ids = new_colors
        elif use_only_mem:
            # Use only the memory queries
            color_ids[color_ids < parts_per_mem[0]] = -1
            color_ids[color_ids > parts_per_mem[1] - 1] = -1
        elif add_anchor_offset:
            # Plot as is - add offset for correspondences
            color_ids += mem_id * parts_per_mem[0]

        return visualize_mesh(
            pc=pc.cpu().numpy(),
            color_ids=color_ids.cpu().numpy()
        )

    def _visualize(
        self, inputs, scores,
        objectness, inds, step, mem_id=0, b_id=0
    ):
        # Parse inputs/predictions
        pc1 = inputs['pc_1'][mem_id][b_id][..., :3]
        pc2 = inputs['pc_2'][b_id][..., :3]

        p1 = inputs['pad_1'][b_id][mem_id].sum()
        p2 = inputs['pad_2'][b_id].sum()
        parts_per_mem = (
            inputs['pad_1'][b_id][:mem_id].sum().item() if mem_id else 0,
            p1.item(),  # parts of current memory
        )
        tgt = inputs['mask_2'].float()
        tgt1 = inputs['mask_1'][mem_id].float()
        logits = (
            scores[-1][b_id].sigmoid()
            * objectness[-1][b_id][None].sigmoid()
        )

        # Raw input
        vislist_target = self._visualize_lbl_pc(
            pc2, tgt[b_id][:, :p2],
            filter_objectness=True, objectness=torch.zeros(p2)
        )

        # Memory ground-truth
        vislist_gt1 = self._visualize_lbl_pc(
            pc1, tgt1[b_id][:, :p1], mem_id=mem_id, filter_score=True,
            add_anchor_offset=True, parts_per_mem=parts_per_mem
        )

        # Predictions of memory queries alone
        vislist_pred_init = self._visualize_lbl_pc(
            pc2,
            logits,
            mem_id=mem_id,
            use_only_mem=True,
            parts_per_mem=parts_per_mem
        )

        # All predictions
        vislist_pred = self._visualize_lbl_pc(
            pc2,
            logits,
            mem_id=mem_id
        )

        # Target ground-truth
        vislist_gt2 = self._visualize_lbl_pc(
            pc2, tgt[b_id][:, :p2],
            inds=inds[b_id],
            filter_score=True
        )
        color = torch.zeros_like(pc2).cpu()
        color_ids = tgt[b_id][:, :p2].argmax(1)
        color = COLORS[color_ids.cpu()]
        color = (color * 255).astype(int)
        color = torch.from_numpy(color)
        labelled_pc_plot = torch.cat((pc2.cpu(), color.cpu()), -1)
        wandb.log({'target': wandb.Object3D({
            "type": "lidar/beta",
            "points": labelled_pc_plot.cpu().numpy()
        })}, step=step)

        wandb.log({
            "input_mem_pred_gt": [
                wandb.Image(np.concatenate(vislist_target)),
                wandb.Image(np.concatenate(vislist_gt1)),
                wandb.Image(np.concatenate(vislist_pred_init)),
                wandb.Image(np.concatenate(vislist_pred)),
                wandb.Image(np.concatenate(vislist_gt2))
            ]
        }, step=step)

    def _visualize_sem(
        self, inputs, scores,
        objectness, sem_logits, step, mem_id=0, b_id=0
    ):
        # Parse inputs/predictions
        pc1 = inputs['pc_1'][mem_id][b_id][..., :3]
        pc2 = inputs['pc_2'][b_id][..., :3]

        tgt = torch.matmul(
            inputs['mask_2'][b_id].float(),  # (P, Nparts)
            inputs['part_sem_labels'][b_id][..., None].float()  # (Nparts, 1)
        ).squeeze(-1)  # (P,)
        tgt = F.one_hot(tgt.long(), ALL_CLASSES)  # (P, C)
        tgt1 = torch.matmul(
            inputs['mask_1'][mem_id][b_id].float(),
            inputs['part_sem_labels_mem'][mem_id][b_id][..., None].float()
        ).squeeze(-1)  # (P,)
        tgt1 = F.one_hot(tgt1.long(), ALL_CLASSES)  # (P, C)
        mask_logits = (
            scores[-1][b_id].sigmoid()
            * objectness[-1][b_id][None].sigmoid()
        )  # (P, Q)
        scores_1hot = F.one_hot(mask_logits.argmax(-1), mask_logits.size(-1))
        sem_pred = sem_logits[-1].argmax(-1).float()  # (B, Q)
        pc_sem_pred = torch.matmul(
            scores_1hot.float(),  # (P, Q)
            sem_pred[b_id].unsqueeze(-1)  # (Q, 1)
        ).squeeze(-1)  # (P,)
        logits = F.one_hot(pc_sem_pred.long(), ALL_CLASSES)  # (P, C)

        # Raw input
        vislist_target = self._visualize_lbl_pc(
            pc2, tgt,
            filter_objectness=True, objectness=torch.zeros(ALL_CLASSES)
        )

        # Memory ground-truth
        vislist_gt1 = self._visualize_lbl_pc(
            pc1, tgt1, filter_score=True
        )

        # All predictions
        vislist_pred = self._visualize_lbl_pc(
            pc2,
            logits,
            mem_id=mem_id
        )

        # Target ground-truth
        vislist_gt2 = self._visualize_lbl_pc(
            pc2, tgt,
            filter_score=True
        )
        wandb.log({
            "input_mem_pred_gt_sem": [
                wandb.Image(np.concatenate(vislist_target)),
                wandb.Image(np.concatenate(vislist_gt1)),
                wandb.Image(np.concatenate(vislist_pred)),
                wandb.Image(np.concatenate(vislist_gt2))
            ]
        }, step=step)


def parse_args():
    # Parse arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--run_name", default="analogical")
    argparser.add_argument("--checkpoint", default="shapenet_ss.pt")

    # Paths
    argparser.add_argument("--checkpoint_path", default="checkpoints/")
    argparser.add_argument("--feat_path",
                           default="/projects/katefgroup/part_based/")
    argparser.add_argument(
        "--label_path",
        default="/projects/katefgroup/part_based/partnet_dataset/stats/after_merging_label_ids/"
    )
    argparser.add_argument(
        "--anno_path",
        default=(
            '/projects/katefgroup/datasets/partnet/partnet_analogical/'
        )
    )

    # Training arguments
    argparser.add_argument("--epochs", default=200, type=int)
    argparser.add_argument("--batch_size", default=32, type=int)
    argparser.add_argument("--lr", default=1e-4, type=float)
    argparser.add_argument("--weight_decay", default=0.0, type=float)
    argparser.add_argument("--bootstrap", default=None)

    # Loss arguments
    argparser.add_argument("--hungarian", action='store_true')
    argparser.add_argument("--supervise_last_only", action='store_true')
    argparser.add_argument("--negative_obj_weight", default=1.0, type=float)
    argparser.add_argument("--background_coef", default=0.1, type=float)
    argparser.add_argument("--train_with_semantics", action='store_true')

    # Dataset arguments
    argparser.add_argument("--dataset", default="partnet")
    argparser.add_argument("--train_split", default="multicat20")
    argparser.add_argument("--val_split", default=None)
    argparser.add_argument("--test_split", default="multicatnovel4")
    argparser.add_argument("--fold", default=None, type=int)
    argparser.add_argument("--k_shot", default=5, type=int)
    argparser.add_argument("--k_shot_seed", default=None, type=int)
    argparser.add_argument("--cross_instance", action='store_true')
    argparser.add_argument("--same_wild_augment_train", action='store_true')

    # Memory arguments
    argparser.add_argument("--retriever_ckpt", default='', type=str)
    argparser.add_argument("--retriever_train_mode", default='random')
    argparser.add_argument("--retriever_val_mode", default='random')
    argparser.add_argument("--train_top_mem_pool_size", default=20, type=int)
    argparser.add_argument("--val_top_mem_pool_size", default=1, type=int)

    # Evaluation arguments
    argparser.add_argument("--eval", action='store_true')
    argparser.add_argument("--eval_train", action='store_true')
    argparser.add_argument("--eval_test", action='store_true')
    argparser.add_argument("--eval_multitask", default=1, type=int,
                           help='repeat eval n times with diff fewshot set')
    argparser.add_argument("--ft_epoch", default=0, type=int,
                           help='finetune n epochs on training fewshot set')
    argparser.add_argument("--visualize_wandb", action='store_true')
    argparser.add_argument("--val_with_semantics", action='store_true')
    argparser.add_argument("--compute_pq", action='store_true')
    argparser.add_argument("--eval_forgetting", action='store_true')
    argparser.add_argument("--use_finetuned", action='store_true')

    # Model variants
    argparser.add_argument("--model", default='baseline', type=str)
    argparser.add_argument("--feat_dim", default=256, type=int)
    argparser.add_argument("--queries", default=0, type=int)
    argparser.add_argument("--pre_norm", action='store_true')
    argparser.add_argument("--num_memories", default=1, type=int)
    argparser.add_argument("--rotary_pe", action='store_true')
    argparser.add_argument("--no_mem_decoding", action='store_true')

    args = argparser.parse_args()
    args.ckpnt = osp.join(args.checkpoint_path, args.checkpoint)
    if args.bootstrap is not None:
        args.bootstrap = osp.join(args.checkpoint_path, args.bootstrap)
    args.retriever_ckpt = osp.join(args.checkpoint_path, args.retriever_ckpt)
    args.eval = args.eval or args.eval_train or args.eval_test
    args.hungarian = args.hungarian or args.queries > 0
    args.use_memory = args.model not in {'partnet_model', 'xy_3ddetr'}
    if args.val_split is None:
        args.val_split = args.train_split
    if args.k_shot_seed:
        args.train_top_mem_pool_size = args.k_shot
    args.use_semantics = args.train_with_semantics or args.val_with_semantics

    # Other variables
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args.device = device
    os.makedirs(args.checkpoint_path, exist_ok=True)
    assert args.k_shot_seed != 0, "Choose any other seed id"
    return args


def fetch_loaders(args):
    # Path has all_categories of partnet with max parts in each sample <=16
    if args.dataset == 'partnet':
        from src.partnet_dataset import PartNetDataset
        DatasetClass = PartNetDataset
        PATH = args.anno_path
        PATH_LABEL = args.label_path

    datasets = DatasetClass(
        categories=[
            args.test_split if args.k_shot_seed and not args.eval_forgetting
            else args.train_split,
            args.test_split if args.k_shot_seed and not args.eval_forgetting
            else args.val_split,
            args.test_split
        ],
        fold=args.fold,
        k_shot_seed=args.k_shot_seed,
        k_shot=args.k_shot,
        ann_path=PATH,
        label_path=PATH_LABEL,
        use_memory=args.use_memory,
        cross_instance=args.cross_instance,
        retriever_mode=[
            args.retriever_train_mode,
            args.retriever_val_mode,
            args.retriever_val_mode
        ],
        retriever_ckpt=args.retriever_ckpt,
        num_memories=args.num_memories,
        top_mem_pool_size=[
            args.train_top_mem_pool_size,
            args.val_top_mem_pool_size,
            args.val_top_mem_pool_size
        ],
        feat_path=args.feat_path,
        return_sem_labels=args.use_semantics,
        same_wild_augment_train=args.same_wild_augment_train
    )
    print(
        "Dataloaders sample nums : train, val, test ",
        len(datasets.instances['train']),
        len(datasets.instances['val']),
        len(datasets.instances['test'])
    )

    data_loaders = DataLoader(
        datasets,
        batch_size=args.batch_size,
        shuffle=(not args.eval or args.ft_epoch > 0),
        drop_last=False,
        num_workers=4,
        collate_fn=custom_collate
    )
    return data_loaders


def init_model(args):
    # Models
    if args.model == 'xy_3ddetr':
        from models.xy3ddetr import XY3DDETR
        model = XY3DDETR(
            in_dim=0,
            out_dim=args.feat_dim,
            num_query=args.queries,
            num_classes=ALL_CLASSES,
            predict_classes=args.use_semantics,
            rotary_pe=args.rotary_pe,
            pre_norm=args.pre_norm
        )
    elif args.model == 'analogical_nets':
        from models.analogical_networks import AnalogicalNetworks
        model = AnalogicalNetworks(
            in_dim=0,
            out_dim=args.feat_dim,
            num_query=args.queries,
            mem_decodes=not args.no_mem_decoding,
            num_classes=ALL_CLASSES,
            predict_classes=args.use_semantics,
            rotary_pe=args.rotary_pe,
            pre_norm=args.pre_norm
        )
    elif args.model == 'multi_mem' and args.num_memories > 1:
        from models.analogical_networks_mm import AnalogicalNetworksMultiMem
        model = AnalogicalNetworksMultiMem(
            in_dim=0,
            out_dim=args.feat_dim,
            num_query=args.queries,
            mem_decodes=not args.no_mem_decoding,
            num_classes=ALL_CLASSES,
            predict_classes=args.use_semantics,
            rotary_pe=args.rotary_pe,
            pre_norm=args.pre_norm
        )
    else:
        assert False, 'unknown model name'
    return model


def main():
    """Run main training/test pipeline."""
    args = parse_args()
    print("Args : ", args)
    data_loaders = fetch_loaders(args)
    model = init_model(args)
    trainer = Trainer(model.to(args.device), data_loaders, args)

    # Train/test
    trainer.run()


if __name__ == "__main__":
    main()
