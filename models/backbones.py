import torch
from torch import nn

from pointnet2.pointnet2_modules import PointnetSAModuleVotes, PointnetFPModule


def _break_up_pc(pc):
    """Split xyz-features, input (B, P, F), out (B, P, 3)-(B, P, F-3)."""
    xyz = pc[..., :3].contiguous()
    features = pc[..., 3:].contiguous() if pc.size(-1) > 3 else None
    return xyz, features


class PCAutoEncoder(nn.Module):
    """Downsample and then upsample point cloud."""

    def __init__(self, in_dim=0, out_dim=256, depth=2,
                 mlp_last_dims=[128, 256, 256],
                 mlp_hid_dims=[64, 128, 128],
                 npoint_list=[1024, 512, 256]):
        super().__init__()

        self.sa1 = PointnetSAModuleVotes(
            npoint=npoint_list[0],
            radius=0.1,
            nsample=64,
            mlp=(
                [in_dim]
                + [mlp_hid_dims[0] for _ in range(depth)]
                + [mlp_last_dims[0]]
            ),
            use_xyz=True,
            normalize_xyz=True
        )

        self.sa2 = PointnetSAModuleVotes(
            npoint=npoint_list[1],
            radius=0.2,
            nsample=32,
            mlp=(
                [mlp_last_dims[0]]
                + [mlp_hid_dims[1] for _ in range(depth)]
                + [mlp_last_dims[1]]
            ),
            use_xyz=True,
            normalize_xyz=True
        )

        self.sa3 = PointnetSAModuleVotes(
            npoint=npoint_list[2],
            radius=0.4,
            nsample=16,
            mlp=(
                [mlp_last_dims[1]]
                + [mlp_hid_dims[2] for _ in range(depth)]
                + [mlp_last_dims[2]]
            ),
            use_xyz=True,
            normalize_xyz=True
        )

        self.fp2 = PointnetFPModule(
            mlp=[mlp_last_dims[1] + mlp_last_dims[2], 256, out_dim]
        )

        self.fp1 = PointnetFPModule(
            mlp=[mlp_last_dims[0] + out_dim, 256, mlp_last_dims[1]]
        )

        self.fp0 = PointnetFPModule(
            mlp=[in_dim + mlp_last_dims[1], 256, out_dim]
        )

    def forward(self, pc, upsample=False):
        """
        Forward pass.

        Args:
            pc: (xyz (B, P, 3), features (B, P, in_dim))

        Returns:
            latent: xyz (B, s, 3), features (B, s, out_dim), inds (B, s)
                s = 1024 sampled points
            if upsample, additionally returns:
                xyz (B, P, 3), features (B, P, out_dim)
        """
        # Downsampling layers
        sa0_xyz, sa0_feats = _break_up_pc(pc)
        if sa0_feats is not None:
            sa0_feats = sa0_feats.transpose(1, 2).contiguous()
        sa1_xyz, sa1_feats, sa1_inds = self.sa1(sa0_xyz, sa0_feats)
        sa2_xyz, sa2_feats, _ = self.sa2(sa1_xyz, sa1_feats)
        sa3_xyz, sa3_feats, _ = self.sa3(sa2_xyz, sa2_feats)

        # Upsampling layers
        fp2_feats = self.fp2(sa2_xyz, sa3_xyz, sa2_feats, sa3_feats)
        fp2_xyz = sa2_xyz
        fp2_inds = sa1_inds[:, :fp2_xyz.shape[1]]
        res_dict = {
            'sa0_xyz': sa0_xyz,
            'sa0_feats': sa0_feats,
            'sa1_xyz': sa1_xyz,
            'sa1_feats': sa1_feats,
            'sa2_xyz': sa2_xyz,
            'sa2_feats': sa2_feats,
            'lat_xyz': fp2_xyz,
            'lat_feats': fp2_feats.transpose(1, 2).contiguous(),
            'lat_inds': fp2_inds.long()
        }
        if not upsample:  # don't fully upsample
            return res_dict
        fp1_feats = self.fp1(sa1_xyz, sa2_xyz, sa1_feats, fp2_feats)
        fp0_feats = self.fp0(sa0_xyz, sa1_xyz, sa0_feats, fp1_feats)
        res_dict['ups_feats'] = fp0_feats.transpose(1, 2).contiguous()
        return res_dict


class UpSample(nn.Module):

    def __init__(self, out_dim, modulated=0, lat_dim=128, in_dim=0):
        super().__init__()
        _dim = out_dim + modulated
        self.fp1 = PointnetFPModule(mlp=[lat_dim + _dim, 256, 256])
        self.fp0 = PointnetFPModule(mlp=[256 + in_dim, 256, out_dim])

    def forward(self, feats, res_dict, modulator=None):
        if modulator is not None:  # modulate feature upsampling
            feats = torch.cat((feats, modulator), -1)
        feats = feats.transpose(1, 2).contiguous()
        sa0_xyz, sa0_feats = res_dict['sa0_xyz'], res_dict['sa0_feats']
        sa1_xyz, sa1_feats = res_dict['sa1_xyz'], res_dict['sa1_feats']
        sa2_xyz = res_dict['sa2_xyz']
        fp1_feats = self.fp1(sa1_xyz, sa2_xyz, sa1_feats, feats)
        fp0_feats = self.fp0(sa0_xyz, sa1_xyz, sa0_feats, fp1_feats)
        res_dict['attn_sa0_feats'] = fp0_feats

        return sa0_xyz, fp0_feats.transpose(1, 2).contiguous(), res_dict
