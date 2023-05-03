import torch
from torch import nn
from torch.nn import functional as F

from .encoder_decoder_layers import ParallelAttentionLayer
from .analogical_networks import AnalogicalNetworks


class AnalogicalNetworksMultiMem(AnalogicalNetworks):
    """Multi-anchor analogical correspondence network."""

    def __init__(self, in_dim=0, out_dim=256, num_layers=6, num_query=64,
                 mem_decodes=True, num_classes=1, predict_classes=False,
                 rotary_pe=False, pre_norm=False, no_grad=False):
        super().__init__(
            in_dim=in_dim,
            out_dim=out_dim,
            num_layers=num_layers,
            num_query=num_query,
            mem_decodes=mem_decodes,
            num_classes=num_classes,
            predict_classes=predict_classes,
            rotary_pe=rotary_pe,
            pre_norm=pre_norm,
            no_grad=no_grad
        )

        # Transformer layers
        self.ca = nn.ModuleList([
            ParallelAttentionLayer(
                out_dim,
                self_attention1=False,
                self_attention2=False,
                rotary_pe=rotary_pe,
                pre_norm=pre_norm,
                apply_ffn=False
            )
            for _ in range(num_layers)
        ])
        self.wsa = nn.ModuleList([
            ParallelAttentionLayer(
                out_dim,
                self_attention2=False,
                cross_attention1=False,
                cross_attention2=False,
                rotary_pe=rotary_pe,
                pre_norm=pre_norm,
                apply_ffn=False
            )
            for _ in range(num_layers)
        ])
        self.sa = nn.ModuleList([
            ParallelAttentionLayer(
                out_dim,
                cross_attention1=False,
                cross_attention2=False,
                rotary_pe=rotary_pe,
                pre_norm=pre_norm
            )
            for _ in range(num_layers)
        ])

        # Memory encoder layers
        self.mem_encoder = nn.ModuleList([
            ParallelAttentionLayer(
                out_dim,
                cross_attention2=False,
                self_attention2=False,  # var_with_sa,
                rotary_pe=rotary_pe,
                pre_norm=pre_norm
            )
            for _ in range(num_layers)
        ])

    def forward(self, pc_1, mask_1, pc_2, level_id=None):
        """Forward pass, pc (B, P, F), mask (B, P, n_clusters)."""
        # Encode anchor point clouds
        _, _, anchor_ys, anchor_ypos = self._encode_anchors_(pc_1, mask_1)

        # Encode target point cloud
        res_dict, x, pos = self._encode_target(pc_2)

        # Initialize target ys
        y, y_pos = self._construct_queries(
            anchor_ys, anchor_ypos, res_dict['lat_xyz']
        )

        # Construct attention masks
        x_mask, y_mask = self._construct_pad_masks(mask_1, x)

        # Contextualize each memory with input separately
        y = [
            self._mem_att_encode(y[m], y_pos[m], y_mask[m], x, pos)
            for m in range(len(y))
        ]

        # Decode
        return self._decode(x, y, x_mask, y_mask, pos, y_pos, res_dict)

    def _construct_queries(self, anchor_ys, anchor_ypos, xyz):
        if self.queries is None:
            return anchor_ys, anchor_ypos
        learnable = self.queries.weight[None].repeat(len(anchor_ys[0]), 1, 1)
        if self.rotary_pe:
            pos = self.pos_emb(
                xyz.mean(1)[:, None].repeat(1, learnable.size(1), 1)
            )
        else:
            pos = torch.zeros_like(learnable).to(learnable.device)
        return anchor_ys + [learnable], anchor_ypos + [pos]

    def _construct_pad_masks(self, anchor_masks, x):
        x_mask = torch.zeros(len(x), x.size(1)).bool().to(x.device)
        y_masks = []
        for mask in anchor_masks:
            cross_mask = ~mask.bool().any(1)
            y_masks.append(cross_mask)
        if self.queries is not None:
            y_masks.append(
                torch.zeros(len(x), self.num_query).bool().to(x.device)
            )
        return x_mask, y_masks

    def _mem_att_encode(self, mem, mem_pos, mem_mask, x, pos):
        # mem (B, Nparts, F), mem_pos wrt mem, mem_mask (B, P, Nparts)
        # x (B, P', F) point features, pos wrt x
        for layer in self.mem_encoder:
            mem, _ = layer(mem, mem_mask, x, None, mem_pos, pos)
        return mem

    def _decode(
        self,
        x,  # (B, P', F)
        y_list,  # [(B, Q_i, F)]
        x_mask,  # (B, P')
        y_mask,  # [(B, Q_i)]
        x_pos,
        y_pos,  # [pos_wrt_y]
        res_dict
    ):
        scores = []
        objectness = []
        sem_logits = []
        for k in range(len(self.ca)):
            # Parallel Cross-attention among x and y
            x, y = self.ca[k](
                x, x_mask,
                torch.cat(y_list, 1), torch.cat(y_mask, 1),
                x_pos, torch.cat(y_pos, 1)
            )
            if not self.mem_decodes:  # only update parametric queries
                y_list[-1] = y[:, -y_list[-1].size(1):]
            else:
                offset = 0
                _y_l = []
                for m in range(len(y_list)):
                    _y_l.append(y[:, offset:offset + y_list[m].size(1)])
                    offset += y_list[m].size(1)
                y_list = _y_l
            # Self-attention within each memory parts
            if self.mem_decodes:
                for m in range(len(y_list)):
                    y_list[m], _ = self.wsa[k](
                        y_list[m], y_mask[m],
                        y_list[m], y_mask[m],
                        y_pos[m], y_pos[m]
                    )
            # Parallel Self-attention + FFN, updates x and y
            x, y = self.sa[k](
                x, x_mask,
                torch.cat(y_list, 1),
                torch.cat(y_mask, 1),
                x_pos, torch.cat(y_pos, 1)
            )
            if not self.mem_decodes:  # only update parametric queries
                y_list[-1] = y[:, -y_list[-1].size(1):]
            else:
                offset = 0
                _y_l = []
                for m in range(len(y_list)):
                    _y_l.append(y[:, offset:offset + y_list[m].size(1)])
                    offset += y_list[m].size(1)
                y_list = _y_l
            # Score every point (B, P, n_queries)
            _, feats, _ = self.upsamplers[k](x, res_dict)
            y_score = torch.cat(y_list, 1) if self.mem_decodes else y_list[-1]
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
