from scipy.optimize import linear_sum_assignment
import torch
import torch.nn as nn
import torch.nn.functional as F


def mask_iou(scores, targets):
    """Compute mask iou, scores are (B, P, Q), targets (B, P, C)."""
    scores = scores.sigmoid()
    isect = torch.matmul(scores.transpose(1, 2), targets)
    union = scores.sum(1)[:, :, None] + targets.sum(1)[:, None] - isect
    return isect / union  # B Q C


class HungarianMatcher(nn.Module):
    """Assign targets to predictions."""

    def __init__(self, cls_cost=4.0):
        """Initialize matcher."""
        super().__init__()
        self.cls_cost = cls_cost

    @torch.no_grad()
    def forward(self, scores, objectness, targets,
                sem_logits=None, tgt_labels=None):
        """
        Perform the matching.

        Args:
            scores: (B, P, Q)
            objectness: (B, Q) or (B, Q, num_sem_classes)
            targets: (B, P, C)
            sem_logits: (B, Q, num_classes)
            tgt_labels: (B, C)
                where B: batch, P: points, C: gt parts, Q: queries

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j):
                - index_i is the indices of the selected predictions
                - index_j is the indices of the corresponding selected targets
            For each batch element, it holds:
            len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        C = -mask_iou(scores, targets) - 4 * objectness.sigmoid().unsqueeze(-1)

        if tgt_labels is not None and sem_logits is not None:
            # Use semantic part labels
            out_prob = sem_logits.softmax(-1)  # [B, Q, num_part_classes]
            cost_class = torch.stack([
                out_prob[i, :, gt_l] for i, gt_l in enumerate(tgt_labels)
            ])  # (B, Q, C)
            C = C - self.cls_cost * cost_class

        C = C.detach().cpu()

        sizes = [v.max(0)[0].sum().long() for v in targets.long()]
        indices = [
            linear_sum_assignment(c[:, :sizes[i]])
            for i, c in enumerate(C)
        ]
        return [
            (
                torch.as_tensor(i, dtype=torch.int64),  # matched pred
                torch.as_tensor(j, dtype=torch.int64)  # corresponding gt
            )
            for i, j in indices
        ]


class IdentityMatcher(nn.Module):
    """Assign targets to predictions in an 1-1 fashion."""

    def __init__(self):
        """Initialize matcher."""
        super().__init__()

    @torch.no_grad()
    def forward(self, scores, objectness, targets,
                sem_logits=None, tgt_labels=None):
        """
        Perform the matching.

        Args:
            targets: (B, P, C)
                where B: batch, P: points, C: gt parts

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j):
                - index_i is the indices of the selected predictions
                - index_j is the indices of the corresponding selected targets
            For each batch element, it holds:
            len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        sizes = [v.max(0)[0].sum().long() for v in targets.long()]

        return [
            (
                torch.arange(sizes[i]),  # matched pred
                torch.arange(sizes[i])  # corresponding gt
            )
            for i in range(len(sizes))
        ]


class SetCriterion(nn.Module):
    """
    Computes the loss in two steps:
        1) compute assignment between ground truth and outputs
        2) supervise each pair of matched ground-truth / prediction
    """

    def __init__(self, no_obj_coef=1.0, use_identity=False,
                 cls_cost=4.0, background_coef=0.1):
        """
        Parameters:
            no_obj_coef: weight of the no-object category
            use_identity: use identity matcher (for within-instance)
            cls_cost: semantic class cost in Hungrarian matcher
            background_coef: weight of background points in the loss
        """
        super().__init__()
        if use_identity:
            self.matcher = IdentityMatcher()
        else:
            self.matcher = HungarianMatcher(cls_cost)
        self.id_matcher = IdentityMatcher()
        self.no_obj_coef = no_obj_coef
        self.background_coef = background_coef

    def loss_mask(self, outputs, targets, indices):
        """Compute mask losses."""
        # outputs (B, P, Q)
        # targets (B, P, Nparts)
        # indices [(qid, pid)]
        outputs = torch.cat([
            outputs[b, :, qid] for b, (qid, _) in enumerate(indices)
        ], -1)
        targets = torch.cat([
            targets[b, :, tid] for b, (_, tid) in enumerate(indices)
        ], -1)

        # Loss is (P, total_matched_queries)
        loss = F.binary_cross_entropy_with_logits(
            outputs, targets, reduction='none'
        )
        _mask = torch.ones_like(loss, device=loss.device)
        _mask[targets == 0] = self.background_coef
        loss = 10 * (loss * _mask).mean()
        return loss

    def loss_objectness(self, outputs, indices):
        """Compute objectness losses."""
        # outputs (B, Q)
        # indices [(qid, pid)]
        tgt = torch.zeros_like(outputs, device=outputs.device)
        for k, (qid, _) in enumerate(indices):
            tgt[k][qid] = 1.0
        loss = F.binary_cross_entropy_with_logits(
            outputs, tgt, reduction='none'
        )
        _mask = torch.ones_like(loss, device=loss.device)
        _mask[tgt == 0] = self.no_obj_coef
        loss = (loss * _mask).mean()
        return loss

    def loss_semantic(self, outputs=None, targets=None, indices=None,
                      cls_mask=None):
        """Compute semantic losses."""
        # indices [(qid, pid)]
        # outputs (B, Q, n_sem_classes)
        # tgt_labels (B, gt_parts)
        # cls_mask (B, n_sem_classes) classes to be considered
        if targets is None or outputs is None:
            return 0.
        # Fill target with gt part classes
        tgt_sem = torch.zeros(
            outputs.shape[:-1],
            dtype=torch.int64,
            device=outputs.device
        )  # (B, Q)
        for k, (qid, pid) in enumerate(indices):
            tgt_sem[k][qid] = targets[k][pid]
        # Queries with no assignment are not supervised
        mask = (tgt_sem > 0).float()
        # Only classes that appear in the batch are supervised (+5% chance)
        weight = (cls_mask.sum(0) + torch.rand_like(cls_mask[0]) > .95).float()
        weight[0] = 0
        loss = F.cross_entropy(
            outputs.transpose(1, 2), tgt_sem, reduction='none',
            weight=weight.double()
        )
        loss = (mask * loss).mean()
        return loss

    def forward(self, scores, objectness, targets,
                sem_logits=None, tgt_labels=None,
                cls_mask=None, is_cross=None, use_identity_for_within=False):
        """
        Perform the loss computation.

        Args:
            scores: (B, P, Q)
            objectness: (B, Q)
            targets: (B, P, C)
            sem_logits: (B, Q, num_classes)
            tgt_labels: (B, C)
            cls_mask: (B, num_classes)
                where B: batch, P: points, C: gt parts, Q: queries
            is_cross: (B,)
        """
        # Retrieve the matching between outputs and targets
        indices = self.matcher(
            scores, objectness, targets,
            sem_logits=sem_logits, tgt_labels=tgt_labels
        )
        if use_identity_for_within:  # within-scene merged into cross-scene
            id_indices = self.id_matcher(
                scores, objectness, targets,
                sem_logits=sem_logits, tgt_labels=tgt_labels
            )
            indices = [
                indices[b] if is_cross[b] else id_indices[b]
                for b in range(len(scores))
            ]
        # Compute all the requested losses
        loss = (
            self.loss_mask(scores, targets, indices)
            + self.loss_objectness(objectness, indices)
            + self.loss_semantic(sem_logits, tgt_labels, indices, cls_mask)
        )

        return loss, indices
