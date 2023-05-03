import torch
from torch import nn
from torch.nn import functional as F

from .backbones import PCAutoEncoder, UpSample
from .encoder_decoder_layers import ParallelAttentionLayer
from .positional_embeddings import (
    PositionEmbeddingLearned, VolumetricPositionEncoding
)


class AnalogicalNetworks(nn.Module):
    """Multi-anchor analogical correspondence network."""

    def __init__(self, in_dim=0, out_dim=256, num_layers=6, num_query=0,
                 mem_decodes=True, num_classes=1, predict_classes=False,
                 rotary_pe=False, pre_norm=False, no_grad=False,
                 add_pos_once=False):
        super().__init__()
        self.num_query = num_query
        self.mem_decodes = mem_decodes
        self.predict_classes = predict_classes
        self.rotary_pe = rotary_pe
        self.no_grad = no_grad
        self.add_pos_once = add_pos_once

        # Point bottleneck
        self.pc_btlnck = PCAutoEncoder(in_dim, out_dim)

        # Positional embeddings
        if rotary_pe and not add_pos_once:
            assert out_dim % 6 == 0
            self.pos_emb = VolumetricPositionEncoding(out_dim)
        else:
            self.pos_emb = PositionEmbeddingLearned(3, out_dim, True)

        # "Object" queries
        if num_query:
            self.queries = nn.Embedding(num_query, out_dim)
            print(f"model with {num_query} queries")
        else:
            self.queries = None
        if not mem_decodes:
            assert self.queries is not None, "learnable queries should be used"

        # Transformer layers
        self.ca = nn.ModuleList([
            ParallelAttentionLayer(
                out_dim,
                self_attention2=True,  # var_with_sa,
                rotary_pe=rotary_pe,
                pre_norm=pre_norm
            )
            for _ in range(num_layers)
        ])

        # Projection heads
        self.upsamplers = nn.ModuleList([
            UpSample(out_dim, in_dim=in_dim, lat_dim=128)
            for _ in range(num_layers)
        ])

        # Sem cls part prediction
        out_dim_2 = 4 * out_dim
        self.sem_cls_heads = nn.ModuleList([nn.Sequential(
            nn.Conv1d(out_dim, out_dim_2, 1, bias=False),
            nn.BatchNorm1d(out_dim_2), nn.ReLU(), nn.Dropout(0.1),
            nn.Conv1d(out_dim_2, out_dim_2, 1, bias=False),
            nn.BatchNorm1d(out_dim_2), nn.ReLU(), nn.Dropout(0.1),
            nn.Conv1d(out_dim_2, num_classes, 1)
        ) for _ in range(num_layers)])

        # Objectness prediction
        self.obj_heads = nn.ModuleList([nn.Sequential(
            nn.Conv1d(out_dim, out_dim, 1, bias=False),
            nn.BatchNorm1d(out_dim), nn.ReLU(), nn.Dropout(0.3),
            nn.Conv1d(out_dim, out_dim, 1, bias=False),
            nn.BatchNorm1d(out_dim), nn.ReLU(), nn.Dropout(0.3),
            nn.Conv1d(out_dim, 1, 1)
        ) for _ in range(num_layers)])

    def forward(self, pc_1, mask_1, pc_2, level_id=None):
        """Forward pass, pc (B, P, F), mask (B, P, n_clusters)."""
        # Encode anchor point clouds
        if self.no_grad:
            _, _, anchor_ys, anchor_ypos = self._encode_anchors(pc_1, mask_1)
        else:
            _, _, anchor_ys, anchor_ypos = self._encode_anchors_(pc_1, mask_1)

        # Encode target point cloud
        res_dict, x, pos = self._encode_target(pc_2)

        # Initialize target ys
        y, y_pos = self._construct_queries(
            anchor_ys, anchor_ypos, res_dict['lat_xyz']
        )

        # Construct attention masks
        x_mask, y_mask = self._construct_pad_masks(mask_1, x)

        # Decode
        return self._decode(x, y, x_mask, y_mask, pos, y_pos, res_dict, None)

    def _encode_anchors_(self, pc_list, mask_list):
        anchor_xs = []
        anchor_pos = []
        anchor_ys = []
        anchor_ypos = []
        for pc, mask in zip(pc_list, mask_list):
            res_dict = self.pc_btlnck(pc, True)
            # Latent features of anchor k
            xs = res_dict['lat_feats']
            anchor_xs.append(xs)
            pos = self.pos_emb(res_dict['lat_xyz'])
            anchor_pos.append(pos)
            # Upsampled features
            feats = res_dict['ups_feats']
            # Anchor k queries
            ys = torch.matmul(mask.float().transpose(1, 2), feats)
            ys = ys / (mask.sum(1).unsqueeze(-1) + 1e-8)
            anchor_ys.append(ys)
            pos = torch.matmul(
                mask.float().transpose(1, 2), res_dict['sa0_xyz']
            )  # (B, N, 3)
            pos = pos / (mask.sum(1).unsqueeze(-1) + 1e-8)
            pos = self.pos_emb(pos)
            anchor_ypos.append(pos)
        return anchor_xs, anchor_pos, anchor_ys, anchor_ypos

    @torch.no_grad()
    def _encode_anchors(self, pc_list, mask_list):
        anchor_xs = []
        anchor_pos = []
        anchor_ys = []
        anchor_ypos = []
        for pc, mask in zip(pc_list, mask_list):
            res_dict = self.pc_btlnck(pc, False)
            # Latent features of anchor k
            xs = res_dict['lat_feats']
            anchor_xs.append(xs)
            pos = self.pos_emb(res_dict['lat_xyz'])
            anchor_pos.append(pos)
            # Latent features
            feats = res_dict['lat_feats']
            # Anchor k queries
            mask = torch.gather(
                mask, 1,
                res_dict['lat_inds'][..., None].repeat(1, 1, mask.size(-1))
            )
            ys = torch.matmul(mask.float().transpose(1, 2), feats)
            ys = ys / (mask.sum(1).unsqueeze(-1) + 1e-8)
            anchor_ys.append(ys)
            pos = torch.matmul(
                mask.float().transpose(1, 2), res_dict['lat_xyz']
            )  # (B, N, 3)
            pos = pos / (mask.sum(1).unsqueeze(-1) + 1e-8)
            pos = self.pos_emb(pos)
            anchor_ypos.append(pos)
        return anchor_xs, anchor_pos, anchor_ys, anchor_ypos

    def _encode_target(self, pc):
        res_dict = self.pc_btlnck(pc, False)
        x = res_dict['lat_feats']
        pos = self.pos_emb(res_dict['lat_xyz'])
        return res_dict, x, pos

    def _construct_queries(self, anchor_ys, anchor_ypos, xyz):
        if self.queries is None:
            return torch.cat(anchor_ys, 1), torch.cat(anchor_ypos, 1)
        learnable = self.queries.weight[None].repeat(len(anchor_ys[0]), 1, 1)
        if self.rotary_pe:
            pos = self.pos_emb(
                xyz.mean(1)[:, None].repeat(1, learnable.size(1), 1)
            )
        else:
            pos = torch.zeros_like(learnable).to(learnable.device)
        return (
            torch.cat(anchor_ys + [learnable], 1),
            torch.cat(anchor_ypos + [pos], 1)
        )

    def _construct_pad_masks(self, anchor_masks, x):
        x_mask = torch.zeros(len(x), x.size(1)).bool().to(x.device)
        y_masks = []
        for mask in anchor_masks:
            cross_mask = ~mask.bool().any(1)
            y_masks.append(cross_mask)
        y_mask = torch.cat(y_masks, 1)
        if self.queries is not None:
            y_mask = torch.cat((
                y_mask,
                torch.zeros(len(x), self.num_query).bool().to(x.device)
            ), 1)
        return x_mask, y_mask

    def _decode(self, x, y, x_mask, y_mask, x_pos, y_pos, res_dict, lvl_ids):
        scores = []
        objectness = []
        sem_logits = []
        if self.add_pos_once:
            x = x + x_pos
            y = y + y_pos
            x_pos, y_pos = None, None
        for k in range(len(self.ca)):
            # Parallel Self-/Cross-attention among x and y
            x, y = self.ca[k](
                x, x_mask, y, y_mask, x_pos, y_pos,
                seq1_sem_pos=lvl_ids, seq2_sem_pos=lvl_ids
            )
            # Score every point (B, P, n_queries)
            _, feats, _ = self.upsamplers[k](x, res_dict)
            y_score = y if self.mem_decodes else y[:, -self.num_query:]
            y_score = y_score.transpose(1, 2)
            scores.append(6 * torch.matmul(
                F.normalize(feats, dim=-1),
                F.normalize(y_score, dim=1)
            ))
            # Compute objectness (B, n_queries)
            objectness.append(self.obj_heads[k](y_score).squeeze(1))
            # Compute Semantic labels (B, n_queries, num_classes)
            if self.predict_classes:
                sem_logits.append(
                    self.sem_cls_heads[k](y_score).transpose(1, 2)
                )
        return scores, objectness, sem_logits
