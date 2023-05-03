"""
Adapted from:
@origin : PointWOLF.py by {Sanghyeok Lee, Sihyeon Kim}
@Contact: {cat0626, sh_bs15}@korea.ac.kr
@Time: 2021.09.30
"""

import numpy as np


def pointwolf(pos, n_local_transformations=4,
              r_range=(-15., 15.), s_range=(1., 3.), t_tange=(-1, 1),
              center=True, normalize=True, transform_2d=False):
    """Apply point-wolf augmentation on pos (N, 3)."""
    # Sample anchor points (centers of local transformations)
    idx = fps(pos, n_local_transformations)  # (M,)

    # Keep
    pos_anchor = pos[idx]  # (M, 3)

    # Move to canonical space
    pos_normalize = pos[None] - pos_anchor[:, None]  # (M, N, 3)

    # Local transformation at anchor point
    pos_transformed = local_transformation(
        pos_normalize, r_range, s_range, t_tange, transform_2d=transform_2d
    )  # (M, N, 3)

    # Move to origin space
    pos_transformed = pos_transformed + pos_anchor[:, None]  # (M, N, 3)

    # Aggregate transformations
    pos = kernel_regression(pos, pos_anchor, pos_transformed)

    # Normalize
    if center or normalize:
        pos = pos - pos.mean(axis=-2, keepdims=True)
    if normalize:
        scale = (1 / np.sqrt((pos ** 2).sum(1)).max()) * 0.999999
        pos = scale * pos

    return pos.astype('float32')


def fps(pc, npoint):
    """
    Farthest point sampling.

    Args:
        pc: tensor (N, 3) of xyz
        npoint: int, points to keep

    Returns:
        centroids: tensor (npoint,) index list wrt pc
    """
    N, _ = pc.shape
    centroids = np.zeros(npoint, dtype=np.int_)  # (M)
    distance = np.ones(N, dtype=np.float64) * 1e10  # (N)
    farthest = np.random.randint(0, N, (1,), dtype=np.int_)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = pc[farthest, :]
        dist = ((pc - centroid) ** 2).sum(-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = distance.argmax()
    return centroids


def get_random_axis(n_axis):
    """
    Return a binary tensor (n_axis, 3) mapping to:
    1(001):z, 2(010):y, 3(011):yz, 4(100):x, 5(101):xz, 6(110):xy, 7(111):xyz
    """
    axis = np.random.randint(1, 8, (n_axis))
    axis = ((axis[:, None] & 1 << np.arange(3)) > 0).astype(int)
    return axis


def kernel_regression(pos, pos_anchor, pos_transformed, sigma=0.5):
    """
    Weight all points based on local affinity to anchors.

    Args:
        pos: tensor (N, 3)
        pos_anchor: tensor (M, 3)
        pos_transformed: tensor (M, N, 3)

    output :
        pos_new: tensor (N, 3), points after weighted local transformation
    """
    # Distance between anchor points & all points
    sub = pos_anchor[:, None] - pos[None]  # (M, N, 3)

    # Project distance
    project_axis = get_random_axis(1)
    projection = np.expand_dims(project_axis, axis=1) * np.eye(3)
    sub = sub @ projection  # (M, N, 3)
    sub = np.sqrt((sub ** 2).sum(2))  # (M, N)

    # Kernel regression
    weight = np.exp(-0.5 * (sub ** 2) / (sigma ** 2))  # (M, N)
    pos_new = (weight[..., None] * pos_transformed).sum(0)  # (N, 3)
    pos_new = (pos_new / weight.sum(0, keepdims=True).T)  # normalize by weight
    return pos_new


def local_transformation(
    anchor_pcs,
    r_range=(-10., 10.), s_range=(1., 3.), t_tange=(-0.25, 0.25),
    transform_2d=False
):
    """Apply local transformations, anchor_pcs is (n_anchor, n_points, 3)."""
    M, _, _ = anchor_pcs.shape

    # Select axes to affect
    transformation_dropout = np.random.binomial(1, 0.5, (M, 3))  # (M, 3)
    transformation_axis = get_random_axis(M)  # (M, 3)

    # Sample transformations
    degree = (
        np.pi * np.random.uniform(*r_range, size=(M, 3)) / 180.0
        * transformation_dropout[:, :1]
    )  # (M, 3), sampling from (-r_range, r_range)
    if transform_2d:
        degree[:, :2] = 0
    scale = (
        np.random.uniform(*s_range, size=(M, 3))
        * transformation_dropout[:, 1:2]
    )  # (M, 3), sampling from (1, s_range)
    scale = scale * transformation_axis
    scale = scale + 1 * (scale == 0)  # scaling factor must be positive
    trl = (
        np.random.uniform(*t_tange, size=(M, 3))
        * transformation_dropout[:, 2:3]
    )  # (M, 3), sampling from (-t_tange, t_tange)
    trl = trl * transformation_axis
    if transform_2d:
        trl[:, -1] = 0

    # Scaling Matrix
    S = np.expand_dims(scale, axis=1) * np.eye(3)
    # Rotation Matrix
    sin = np.sin(degree)
    cos = np.cos(degree)
    sx, sy, sz = sin[:, 0], sin[:, 1], sin[:, 2]
    cx, cy, cz = cos[:, 0], cos[:, 1], cos[:, 2]
    R = np.stack([
        cz * cy,
        cz * sy * sx - sz * cx,
        cz * sy * cx + sz * sx,
        sz * cy,
        sz * sy * sx + cz * cy,
        sz * sy * cx - cz * sx,
        -sy,
        cy * sx,
        cy * cx
    ], axis=1).reshape(M, 3, 3)

    # Apply rotation, scaling, translation
    anchor_pcs = anchor_pcs @ R @ S + trl.reshape(M, 1, 3)
    return anchor_pcs
