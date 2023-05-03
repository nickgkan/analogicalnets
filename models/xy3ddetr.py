import torch
from torch import nn

from .analogical_networks import AnalogicalNetworks


class XY3DDETR(AnalogicalNetworks):
    """Standard X->Y baseline."""

    def __init__(self, in_dim=0, out_dim=256, num_layers=6, num_query=64,
                 num_classes=1,
                 rotary_pe=False, pre_norm=False, predict_classes=False):
        super().__init__(
            in_dim=in_dim, out_dim=out_dim, num_layers=num_layers,
            num_query=num_query,
            num_classes=num_classes,
            rotary_pe=rotary_pe, pre_norm=pre_norm, no_grad=False,
            predict_classes=predict_classes
        )

        # Level-id embeddings
        self.lvl_ids = nn.Embedding(3, out_dim)

    def forward(self, _a1, _a2, pc, level_id=None):
        """Forward pass, pc (B, P, F)."""
        # Encode target point cloud
        res_dict, x, pos = self._encode_target(pc)

        # Featurize clusters (B, n_clusters, F)
        y, y_pos = self._construct_queries(
            res_dict['lat_xyz'], res_dict['lat_feats']
        )

        # Level ids
        lvl_ids = None
        if level_id is not None:
            lvl_ids = self.lvl_ids(level_id).unsqueeze(1)

        # Decode
        return self._decode(x, y, None, None, pos, y_pos, res_dict, lvl_ids)

    def _construct_queries(self, xyz, feats):
        learnable = self.queries.weight[None].repeat(len(xyz), 1, 1)
        if self.rotary_pe:
            pos = self.pos_emb(
                xyz.mean(1)[:, None].repeat(1, learnable.size(1), 1)
            )
        else:
            pos = torch.zeros_like(learnable).to(learnable.device)
        return learnable, pos
