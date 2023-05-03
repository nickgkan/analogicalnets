from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from .eval_pq import PanopticEval


class AccuracyEvaluator:
    """General micro/macro-accuracy evaluator."""

    def __init__(self, class_names=None):
        self.class_names = class_names
        self._pred_per_class = defaultdict(int)
        self._gt_per_class = defaultdict(int)

    @torch.no_grad()
    def step(self, predictions, labels):
        """
        Forward pass.

        Args:
            predictions: (B,)
            labels: (B,)
        """
        for pred, gt in zip(predictions, labels):
            key = gt if self.class_names is None else self.class_names[gt]
            self._gt_per_class[key] += 1
            key = pred if self.class_names is None else self.class_names[pred]
            self._pred_per_class[key] += 1 * (pred == gt)

    def __str__(self):
        micro_acc = (
            sum(self._pred_per_class[k] for k in self._pred_per_class)
            / max(1, sum(self._gt_per_class[k] for k in self._gt_per_class))
        )
        macro_acc = (
            sum(
                self._pred_per_class[k] / self._gt_per_class[k]
                for k in self._gt_per_class
            )
            / max(1, len(self._gt_per_class.keys()))
        )
        for k in sorted(self._gt_per_class.keys()):
            print(k, self._pred_per_class[k] / self._gt_per_class[k])
        return f"Micro-ACC: {micro_acc}\nMacro-ACC: {macro_acc}"


class APEvaluator:
    """Evaluate average precision."""

    def __init__(self, n_classes, iou_threshold=0.5):
        self.helpers = {}
        self.n_classes = n_classes
        self.iou_thr = iou_threshold

    @torch.no_grad()
    def step(self, scores, sem_logits, objectness,
             target_masks, per_part_sem_labels,
             class_names, levels):
        """
        Forward pass.

        Args:
            scores: (B, points, Q) query scores per point
            sem_logits: (B, Q, C) class distribution of each query
            objectness: (B, Q) confidence of each query
            target_masks: (B, points, Nparts) ground-truth masks
            per_part_sem_labels: (B, Np) semantic class ids per part
            class_names: [str], len=B
            levels: [int], len=B
        """
        # Un-assign points that belong to class 0
        target_masks *= (~(per_part_sem_labels == 0)).float()[:, None]

        # Assign points to queries
        scores_1hot = F.one_hot(scores.argmax(-1), scores.size(-1))

        # Merge objectness and class confidence
        objectness = objectness * sem_logits.max(-1).values

        # Fill AP stats
        for b in range(len(scores)):
            nm = class_names[b]
            lvl = levels[b]
            if nm not in self.helpers:
                self.helpers[nm] = {}
            if lvl not in self.helpers[nm]:
                self.helpers[nm][lvl] = APHelper(self.n_classes, self.iou_thr)
            self.helpers[nm][lvl].step(
                scores_1hot[b].transpose(0, 1).bool().cpu().numpy(),
                pred_labels=sem_logits[b].argmax(-1).cpu().numpy(),
                pred_conf=objectness[b].cpu().numpy(),
                gt_masks=target_masks[b].transpose(0, 1).bool().cpu().numpy(),
                gt_labels=per_part_sem_labels[b].cpu().numpy(),
                gt_valid=target_masks[b].any(0).cpu().numpy(),
                gt_other=~target_masks[b].any(1).cpu().numpy()
            )

    def get_per_level_maps(self, lvl):
        maps = []
        for nm in sorted(self.helpers.keys()):
            if lvl not in self.helpers[nm]:
                continue
            map_ = self.helpers[nm][lvl].get_map()
            print(nm, map_)
            maps.append(map_)
        print('Total', np.mean(np.array(maps)))

    def get_per_class_maps(self):
        maps = []
        for nm in sorted(self.helpers.keys()):
            c_maps = []
            for lvl in self.helpers[nm]:
                c_maps.append(self.helpers[nm][lvl].get_map())
            map_ = np.mean(np.array(c_maps))
            print(nm, map_)
            maps.append(map_)
        print('Total', np.mean(np.array(maps)))

    def print_class_stats(self):
        print("-" * 20)
        print("Instance Segmentation Results:")
        for lvl in range(3):
            print(f'\nLevel {lvl + 1}: (class, mAP)')
            self.get_per_level_maps(lvl + 1)
        print('\nAll levels: (class, mAP)')
        self.get_per_class_maps()

    def get_map(self):
        maps = []
        for nm in sorted(self.helpers.keys()):
            c_maps = []
            for lvl in self.helpers[nm]:
                c_maps.append(self.helpers[nm][lvl].get_map())
            map_ = np.mean(np.array(c_maps))
            maps.append(map_)
        return np.mean(np.array(maps))

    def __str__(self):
        return f'Total mAP: {self.get_map()}'


class ARIEvaluator:
    """Compute classless ARI segmentation metric."""

    def __init__(self, n_levels=3, thres=0.25):
        self.thres = thres
        self._total_ari = defaultdict(int)
        self._ari_level = [defaultdict(int) for _ in range(n_levels)]
        self._total_p_ari = defaultdict(int)
        self._p_ari_level = [defaultdict(int) for _ in range(n_levels)]
        self._samples = defaultdict(int)
        self._samples_level = [defaultdict(int) for _ in range(n_levels)]

    @torch.no_grad()
    def step(self, scores, target_masks, class_names, levels):
        """
        Forward pass.

        Args:
            scores: (B, points, Q) query scores per point
            target_masks: (B, points, Nparts) ground-truth masks
            class_names: [str], len=B
            levels: [int], len=B
        """
        pad = target_masks.any(-2)
        _aris = []
        for b in range(len(scores)):
            sc = scores[b]
            _ari = adjusted_rand_index(
                target_masks[b][:, :pad[b].sum()][None].float(),
                sc[None]
            )
            p_ari = pessimistic_ari(
                target_masks[b][:, :pad[b].sum()].float(),
                sc,
                self.thres
            )
            if not torch.isnan(_ari):
                self._total_ari[class_names[b]] += _ari.item()
                self._ari_level[levels[b]][class_names[b]] += _ari.item()
                self._total_p_ari[class_names[b]] += p_ari.item()
                self._p_ari_level[levels[b]][class_names[b]] += p_ari.item()
                _aris.append(_ari)
            self._samples[class_names[b]] += 1
            self._samples_level[levels[b]][class_names[b]] += 1
        if len(_aris):
            _aris = torch.stack(_aris)
        return _aris

    def get_mean_ins_ari(self):
        return (
            sum(self._total_ari.values())
            / sum(self._samples.values())
        )

    def get_mean_cls_ari(self):
        return (
            sum(self._total_ari[k] / self._samples[k] for k in self._samples)
            / len(self._samples.keys())
        )

    def get_mean_ins_p_ari(self):
        return (
            sum(self._total_p_ari.values())
            / sum(self._samples.values())
        )

    def get_mean_cls_p_ari(self):
        return (
            sum(self._total_p_ari[k] / self._samples[k] for k in self._samples)
            / len(self._samples.keys())
        )

    def get_per_level_ari(self, lv):
        if len(self._ari_level[lv]) == 0:
            return False, 0.
        ari_l = (
            sum(self._ari_level[lv].values())
            / sum(self._samples_level[lv].values())
        )
        return True, ari_l

    def print_class_stats(self):
        print("-" * 20)
        print("Clustering Results:")
        for key in self._total_ari:
            print(key, self._total_ari[key] / self._samples[key])

    def __str__(self):
        mean_ins_ari = self.get_mean_ins_ari()
        mean_cls_ari = self.get_mean_cls_ari()
        mean_ins_p_ari = self.get_mean_ins_p_ari()
        mean_cls_p_ari = self.get_mean_cls_p_ari()
        return (
            "Segmentation Results w/o part labels:"
            f"\nInstance-averaged ARI: {mean_ins_ari}"
            f"\nObject class-averaged ARI: {mean_cls_ari}"
            f"\nInstance-averaged pessimistic ARI: {mean_ins_p_ari}"
            f"\nObject class-averaged pessimistic ARI: {mean_cls_p_ari}"
        )


class MemoryPropAccEvaluator:
    """Compute how often a memory query decodes a target part of same label."""

    def __init__(self):
        self._corres_prop_acc = []
        self._corres_prop_inst = {'count_pos': 0, 'count': 0}
        self._mem_query_usage = {'mem_q': 0, 'any_q': 0}

    @torch.no_grad()
    def step(self, mem_part_labels, tgt_part_labels, inds):
        """
        Forward pass.

        Args:
            mem_part_labels: (B, N_mem_part) semantic label ids of memory
            tgt_part_labels: (B, N_tgt_part) semantic label ids of target
            inds: List of len=B, each element is ([det_idx], [gt_idx])
                indices from Hungarian matching
        """
        for b, ind in enumerate(inds):
            count, count_pos = 0, 0
            det_idx, gt_idx = ind
            for i in range(len(gt_idx)):
                self._mem_query_usage['any_q'] += 1.
                if det_idx[i] < len(mem_part_labels[b]):
                    self._mem_query_usage['mem_q'] += 1.
                    match = float(
                        tgt_part_labels[b, gt_idx[i]]
                        == mem_part_labels[b, det_idx[i]]
                    )
                    count_pos += match
                    self._corres_prop_inst['count_pos'] += match
                    count += 1.
                    self._corres_prop_inst['count'] += 1.
            self._corres_prop_acc.append(
                count_pos / count if count != 0 else 1.
            )  # per sample prop acc

    def __str__(self):
        mean_prop_acc = (
            sum(self._corres_prop_acc)
            / max(1, len(self._corres_prop_acc))
        )
        mean_prop_inst_acc = (
            self._corres_prop_inst['count_pos']
            / max(1, self._corres_prop_inst['count'])
        )
        mean_mem_q_used = (
            self._mem_query_usage['mem_q']
            / max(1, self._mem_query_usage['any_q'])
        )
        return (
            "Memory Propagation Results:"
            f"\nParts decoded by memory query ratio: {mean_mem_q_used}"
            f"\nPropagated parts Label ACC: {mean_prop_acc}"
            f"\nPropagated parts Label inst ACC: {mean_prop_inst_acc}"
        )


class IoUEvaluator:
    """Evaluate cross-dataset IoU."""

    def __init__(self, n_classes):
        self.helpers = {}
        self.n_classes = n_classes

    @torch.no_grad()
    def step(self, scores, sem_logits, per_point_sem_labels,
             class_names, levels, cls_mask):
        """
        Forward pass.

        Args:
            scores: (B, points, Q) query scores per point
            sem_logits: (B, Q, C) class distribution of each query (normalized)
            per_point_sem_labels: (B, Np) semantic class ids per part
            class_names: [str], len=B
            levels: [int], len=B
            cls_mask: (B, n_classes) classes to consider for this cls-lvl
        """
        pc_sem_gt = per_point_sem_labels  # (B, Npoints)
        # Per-point semantic prediction
        scores_1hot = F.one_hot(scores.argmax(-1), scores.size(-1)).float()
        sem_pred = sem_logits.argmax(-1).float()  # (B, Q)
        pc_sem_pred = torch.matmul(
            scores_1hot,  # (B, Npoints, Q)
            sem_pred.unsqueeze(-1)  # (B, Q, 1)
        ).squeeze(-1)  # (B, Npoints)

        for b in range(len(pc_sem_gt)):
            nm = class_names[b]
            lvl = levels[b]
            if nm not in self.helpers:
                self.helpers[nm] = {}
            if lvl not in self.helpers[nm]:
                self.helpers[nm][lvl] = IoUHelper(
                    self.n_classes,
                    ignore=[0] + np.where(cls_mask[b].cpu() == 0)[0].tolist()
                )
            # st()
            self.helpers[nm][lvl].add_(
                pc_sem_pred[[b]].cpu().numpy().astype(np.int64),
                pc_sem_gt[[b]].cpu().numpy().astype(np.int64)
            )

    def get_per_level_ious(self, lvl):
        accs = []
        ious = []
        for nm in sorted(self.helpers.keys()):
            if lvl not in self.helpers[nm]:
                continue
            acc = self.helpers[nm][lvl].get_acc()
            iou = self.helpers[nm][lvl].get_iou()
            print(nm, iou, acc)
            accs.append(acc)
            ious.append(iou)
        print('Total', np.mean(np.array(ious)), np.mean(np.array(accs)))

    def get_per_class_ious(self):
        accs = []
        ious = []
        for nm in sorted(self.helpers.keys()):
            c_accs, c_ious = [], []
            for lvl in self.helpers[nm]:
                c_accs.append(self.helpers[nm][lvl].get_acc())
                c_ious.append(self.helpers[nm][lvl].get_iou())
            acc = np.mean(np.array(c_accs))
            iou = np.mean(np.array(c_ious))
            print(nm, iou, acc)
            accs.append(acc)
            ious.append(iou)
        print('Total', np.mean(np.array(ious)), np.mean(np.array(accs)))

    def print_class_stats(self):
        print("-" * 20)
        print("Semantic Segmentation Results:")
        for lvl in range(3):
            print(f'\nLevel {lvl + 1}: (class, IoU, mAcc)')
            self.get_per_level_ious(lvl + 1)
        print('\nAll levels: (class, IoU, mAcc)')
        self.get_per_class_ious()

    def get_miou(self):
        ious = []
        for nm in sorted(self.helpers.keys()):
            c_ious = []
            for lvl in self.helpers[nm]:
                c_ious.append(self.helpers[nm][lvl].get_iou())
            iou = np.mean(np.array(c_ious))
            ious.append(iou)
        return np.mean(np.array(ious))

    def get_macc(self):
        accs = []
        for nm in sorted(self.helpers.keys()):
            c_accs = []
            for lvl in self.helpers[nm]:
                c_accs.append(self.helpers[nm][lvl].get_acc())
            acc = np.mean(np.array(c_accs))
            accs.append(acc)
        return np.mean(np.array(accs))

    def __str__(self):
        return (
            f'Total mIoU: {self.get_miou()}'
            f'\nTotal mAcc: {self.get_macc()}'
        )


class PerShapePanopticEvaluator:
    """Evaluate panoptic quality and IoU per-shape."""

    def __init__(self, n_classes):
        self._panop_eval = defaultdict(int)
        self._total_iou = defaultdict(int)
        self._total_panop = defaultdict(int)
        self._samples = defaultdict(int)
        self._samples_level = [defaultdict(int) for _ in range(3)]
        self.pq_evaluator = PanopticEval(n_classes, ignore=[0], min_points=30)

    def step(self, scores, sem_logits,
             target_masks, per_point_sem_labels,
             class_names, levels):
        """
        Forward pass.

        Args:
            scores: (B, points, Q) query scores per point
            sem_logits: (B, Q, C) class distribution of each query
            target_masks: (B, points, Nparts) ground-truth masks
            per_point_sem_labels: (B, Np) semantic class ids per part
            n_mem_parts (int): number of parts in memory
            class_names: [str], len=B
            levels: [int], len=B
        """
        pc_sem_gt = per_point_sem_labels  # (B, Npoints)
        pc_inst_gt, pc_sem_pred, pc_inst_pred = semantic_preprocess(
            scores, sem_logits, target_masks
        )  # all are (B, Npoints)

        for b in range(len(pc_sem_gt)):
            sem_gt = pc_sem_gt[[b]].cpu().numpy().astype(np.int64)
            self.pq_evaluator.addBatch(
                pc_sem_pred[[b]].cpu().numpy().astype(np.int64),
                pc_inst_pred[[b]].cpu().numpy().astype(np.int64),
                pc_sem_gt[[b]].cpu().numpy().astype(np.int64),
                pc_inst_gt[[b]].cpu().numpy().astype(np.int64)
            )
            _, _, _, all_pq, all_sq, all_rq = self.pq_evaluator.getPQ()
            _, all_iou = self.pq_evaluator.getSemIoU()
            self.pq_evaluator.reset()

            batch_cls_mask = np.zeros((len(sem_gt), len(all_pq)))
            for i, cls_i in enumerate(sem_gt):
                cls_unique = np.unique(cls_i)
                batch_cls_mask[i][cls_unique] = 1.
                # Ensure no_obj cat is always 0
                batch_cls_mask[i][0] = 0.

            cur_pq = self._agg_val_cls_mask(all_pq, batch_cls_mask)
            cur_sq = self._agg_val_cls_mask(all_sq, batch_cls_mask)
            cur_rq = self._agg_val_cls_mask(all_rq, batch_cls_mask)
            cur_iou = self._agg_val_cls_mask(all_iou, batch_cls_mask)
            self._panop_eval['PQ'] += cur_pq if not np.isnan(cur_pq) else 0.
            self._panop_eval['SQ'] += cur_sq if not np.isnan(cur_sq) else 0.
            self._panop_eval['RQ'] += cur_rq if not np.isnan(cur_rq) else 0.
            self._panop_eval['IoU'] += cur_iou if not np.isnan(cur_iou) else 0.
            self._panop_eval['samples'] += 1.
            # populate macro pq and iou
            n = class_names[b]
            self._total_panop[n] += cur_pq if not np.isnan(cur_pq) else 0.
            self._total_iou[n] += cur_iou if not np.isnan(cur_iou) else 0.
            self._samples[n] += 1
            self._samples_level[levels[b]][n] += 1

    def _agg_val_cls_mask(self, all_v, batch_cls_mask):
        val = (all_v * batch_cls_mask).sum(1) / batch_cls_mask.sum(1)
        return val.mean()

    def get_macro_pq(self):
        return (
            sum(
                self._total_panop[k] / max(1, self._samples[k])
                for k in self._samples
            )
            / max(1, len(self._samples.keys()))
        )

    def get_macro_iou(self):
        return (
            sum(
                self._total_iou[k] / max(1, self._samples[k])
                for k in self._samples
            )
            / max(1, len(self._samples.keys()))
        )

    def __str__(self):
        mean_pq = (
            self._panop_eval['PQ']
            / max(1, self._panop_eval['samples'])
        )
        macro_pq = self.get_macro_pq()
        mean_sq = (
            self._panop_eval['SQ']
            / max(1, self._panop_eval['samples'])
        )
        mean_rq = (
            self._panop_eval['RQ']
            / max(1, self._panop_eval['samples'])
        )
        mean_iou = (
            self._panop_eval['IoU']
            / max(1, self._panop_eval['samples'])
        )
        macro_iou = self.get_macro_iou()
        return (
            "Panoptic Segmentation Results:"
            f"\nPanoptic: total PQ {mean_pq}, macro PQ {macro_pq}"
            f"\nSQ: {mean_sq}"
            f"\nRQ: {mean_rq}"
            f"\nTotal IoU {mean_iou}, macro IoU {macro_iou}"
        )


def adjusted_rand_index(true_mask, pred_mask):
    """
    Compute the adjusted Rand index (ARI), a clustering similarity score.

    This implementation ignores points with no cluster label in `true_mask`
    (those points for which `true_mask` is a zero vector).
    In the context of segmentation, that means this function can ignore
    points in an image corresponding to the background.

    Args:
        true_mask: `Tensor` of shape [batch_size, n_points, n_true_groups].
            The true cluster assignment encoded as one-hot.
        pred_mask: `Tensor` of shape [batch_size, n_points, n_pred_groups].
            The predicted cluster assignment encoded as
            categorical probabilities.
        This function works on the argmax over axis 2.

    Returns:
        ARI scores as a`Tensor` of shape [batch_size].
    """
    if true_mask.size(-1) == 1:
        # Create two fake points that belong to a new part
        true_mask = torch.cat((true_mask, torch.zeros_like(true_mask)), -1)
        true_mask = torch.cat((
            true_mask,
            torch.zeros(len(true_mask), 2, 2, device=true_mask.device)
        ), 1)
        true_mask[:, -2:, -1] = 1
        # Supposedly we found those two new points
        pred_mask = torch.cat((
            pred_mask,
            torch.zeros_like(pred_mask[..., :1])
        ), -1)
        pred_mask = torch.cat((
            pred_mask,
            torch.zeros_like(pred_mask[:, :2])
        ), 1)
        pred_mask[:, -2:, -1] = 1

    pred_group_ids = torch.argmax(pred_mask, -1)
    true_mask_oh = true_mask.float()  # already one-hot
    pred_mask_oh = F.one_hot(pred_group_ids, pred_mask.shape[-1]).float()

    n_points = torch.sum(true_mask_oh, axis=[1, 2]).float()

    nij = torch.einsum('bji,bjk->bki', pred_mask_oh, true_mask_oh)
    a = torch.sum(nij, axis=1)
    b = torch.sum(nij, axis=2)

    rindex = torch.sum(nij * (nij - 1), axis=[1, 2])
    aindex = torch.sum(a * (a - 1), axis=1)
    bindex = torch.sum(b * (b - 1), axis=1)
    expected_rindex = aindex * bindex / (n_points * (n_points - 1))
    max_rindex = (aindex + bindex) / 2
    ari = (rindex - expected_rindex) / (max_rindex - expected_rindex)

    return ari


def pessimistic_ari(true_mask, pred_mask, thres=0.25):
    """
    Compute the adjusted Rand index (ARI), a clustering similarity score.

    This implementation does not ignore background points.

    Args:
        true_mask: `Tensor` of shape [n_points, n_true_groups].
            The true cluster assignment encoded as one-hot.
        pred_mask: `Tensor` of shape [n_points, n_pred_groups].
            The predicted cluster assignment encoded as
            categorical probabilities.

    Returns:
        ARI scores as a`Tensor` of shape [1].
    """
    # Pad true_mask with one more category for background
    if not true_mask.any(-1).all():
        zeros = torch.zeros(len(true_mask), 1).to(true_mask.device)
        true_mask = torch.cat((true_mask, zeros), -1)
        true_mask[~true_mask.any(-1), -1] = 1

    # Pad pred_mask with one more category for background
    zeros = torch.zeros(len(pred_mask), 1).to(pred_mask.device)
    pred_mask = torch.cat((pred_mask.float(), zeros), -1)
    pred_mask[~(pred_mask > thres).any(-1), -1] = 1

    return adjusted_rand_index(true_mask[None], pred_mask[None])


def semantic_preprocess(scores, sem_logits, target_masks):
    """
    Forward pass.

    Args:
        scores: (B, points, Q) query scores per point
        sem_logits: (B, Q, C) class distribution of each query
        target_masks: (B, points, Nparts) ground-truth masks
    """
    pad = target_masks.any(-2)

    # Per-point instance label (assign random instance ids)
    range_ = torch.arange(2, 2 + pad.shape[-1]).to(pad.device).float()
    pc_inst_gt = torch.matmul(
        (pad * range_)[:, None],
        target_masks.transpose(1, 2).float()
    ).squeeze(1)  # (B, Npoints)

    # Per-point semantic prediction
    scores_1hot = F.one_hot(scores.argmax(-1), scores.size(-1)).float()
    sem_pred = sem_logits.argmax(-1).float()  # (B, Q)
    pc_sem_pred = torch.matmul(
        scores_1hot,  # (B, Npoints, Q)
        sem_pred.unsqueeze(-1)  # (B, Q, 1)
    ).squeeze(-1)  # (B, Npoints)

    # Per-point instance prediction
    range_ = torch.arange(800, 800 + scores.size(-1)).to(pad.device).float()
    pc_inst_pred = torch.matmul(
        range_[None, None],  # (B, 1, Q)
        scores_1hot.transpose(1, 2).float()  # (B, Q, Npoints)
    ).squeeze(1)  # (B, Npoints)

    return pc_inst_gt, pc_sem_pred, pc_inst_pred


class APHelper:

    def __init__(self, n_classes, iou_threshold=0.5):
        self.n_classes = n_classes
        self.iou_threshold = iou_threshold
        self.n_samples = np.zeros(n_classes)
        self.true_pos_list = [[] for _ in range(n_classes)]
        self.false_pos_list = [[] for _ in range(n_classes)]
        self.conf_score_list = [[] for _ in range(n_classes)]

    def step(
        self,
        pred_masks,  # (Q, P)
        pred_labels,  # (Q,)
        pred_conf,  # (Q,)
        gt_masks,  # (Npart, P)
        gt_labels,  # (Npart,)
        gt_valid,  # (Npart,)
        gt_other  # (P,)
    ):
        # classify all valid gt masks by part categories
        gt_mask_per_cat = [[] for _ in range(self.n_classes)]
        for j in range(len(gt_labels)):
            if not gt_valid[j]:
                continue
            sem_id = gt_labels[j]
            gt_mask_per_cat[sem_id].append(j)
            self.n_samples[sem_id] += 1

        # sort prediction and match iou to gt masks
        order = np.argsort(-pred_conf)
        gt_used = np.zeros((len(gt_labels)), dtype=np.bool)
        for j in range(len(pred_labels)):
            # get order in original query list
            idx = order[j]

            # check if mask is empty - skip if yes
            if not pred_masks[idx].any():
                continue

            # search for unused gt mask with max overlap and same class
            sem_id = pred_labels[idx]
            iou_max = 0.0
            cor_gt_id = -1
            for k in gt_mask_per_cat[sem_id]:
                if gt_used[k]:
                    continue
                # Remove points with gt label *other* from the prediction
                clean_pred_mask = (pred_masks[idx] & (~gt_other))
                # IoU
                intersect = np.sum(gt_masks[k] & clean_pred_mask)
                union = np.sum(gt_masks[k] | clean_pred_mask)
                iou = intersect * 1.0 / union
                # Best fit?
                if iou > iou_max:
                    iou_max = iou
                    cor_gt_id = k

            gt_used[cor_gt_id] = iou_max > self.iou_threshold
            self.true_pos_list[sem_id].append(iou_max > self.iou_threshold)
            self.false_pos_list[sem_id].append(iou_max <= self.iou_threshold)
            self.conf_score_list[sem_id].append(pred_conf[idx])

    def get_map(self):
        # compute per-part-category AP
        aps = np.zeros((self.n_classes), dtype=np.float32)
        ap_valids = np.ones((self.n_classes), dtype=np.bool)
        for i in range(self.n_classes):
            has_pred = len(self.true_pos_list[i]) > 0
            has_gt = self.n_samples[i] > 0

            if not has_gt:
                ap_valids[i] = False
                continue

            if has_gt and not has_pred:
                continue

            true_pos = np.array(self.true_pos_list[i], dtype=np.float32)
            false_pos = np.array(self.false_pos_list[i], dtype=np.float32)
            conf_score = np.array(self.conf_score_list[i], dtype=np.float32)

            # sort according to confidence score again
            order = np.argsort(-conf_score)
            true_pos = true_pos[order]
            false_pos = false_pos[order]

            aps[i] = self._compute_ap(true_pos, false_pos, self.n_samples[i])

        # compute mean AP
        mean_ap = np.sum(aps * ap_valids) / np.sum(ap_valids)

        return mean_ap

    @staticmethod
    def _compute_ap(tp, fp, gt_npos, n_bins=100):
        tp = np.cumsum(tp)
        fp = np.cumsum(fp)

        rec = tp / gt_npos
        prec = tp / (fp + tp)

        rec = np.insert(rec, 0, 0.0)
        prec = np.insert(prec, 0, 1.0)

        ap = 0.
        delta = 1.0 / n_bins

        out_rec = np.arange(0, 1 + delta, delta)
        out_prec = np.zeros((n_bins + 1), dtype=np.float32)

        for idx, t in enumerate(out_rec):
            prec1 = prec[rec >= t]
            if len(prec1) == 0:
                p = 0.
            else:
                p = max(prec1)

            out_prec[idx] = p
            ap = ap + p / (n_bins + 1)

        return ap


class IoUHelper:

    def __init__(self, n_classes, ignore=None):
        self.n_classes = n_classes
        self.ignore = np.array(ignore, dtype=np.int64)
        self.include = np.array([
            n for n in range(self.n_classes) if n not in self.ignore
        ], dtype=np.int64)
        self.reset()
        self.eps = 1e-15

    def reset(self):
        self.px_iou_conf_matrix = np.zeros((self.n_classes, self.n_classes))

    def add_(self, x_sem, y_sem):
        # x_sem: predicted, y_sem: gt
        idxs = np.stack([x_sem, y_sem], axis=0)

        # make confusion matrix (cols = gt, rows = pred)
        np.add.at(self.px_iou_conf_matrix, tuple(idxs), 1)

    def get_iou(self):
        # Confusion matrix with no fp for ignore
        conf = self.px_iou_conf_matrix.copy()
        conf[:, self.ignore] = 0
        tp = conf.diagonal()
        fp = conf.sum(axis=1) - tp
        fn = conf.sum(axis=0) - tp
        # IoU
        intersection = tp
        union = tp + fp + fn
        union = np.maximum(union, self.eps)
        keep = union[self.include] > 1
        iou_mean = (
            intersection[self.include][keep].astype(float)
            / union[self.include][keep].astype(float)
        ).mean()

        return iou_mean

    def get_acc(self):
        # Confusion matrix with no fp for ignore
        conf = self.px_iou_conf_matrix.copy()
        _all = conf.sum(axis=0)
        conf[:, self.ignore] = 0
        for i in self.ignore:
            conf[i, i] = _all[i]
        tp = conf.diagonal()
        fp = conf.sum(axis=1) - tp
        # Acc
        total_tp = tp.sum()
        total = tp.sum() + fp.sum()
        total = np.maximum(total, self.eps)
        return float(total_tp) / float(total)
