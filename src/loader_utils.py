import numpy as np

from .tools import rot_x, rot_y, rot_z


def get_rotated_pcs(pc, step):
    """Rotate pc (N, 3) by 0:step:360 (step is in deg)."""
    rotations = np.arange(0, 360, step=step) * np.pi / 180
    rotated = [rot_z(pc, rot) for rot in rotations]
    return rotated, rotations


def wild_parallel_augment(pc, flip=True, translate=True,
                          rotate_z=True, rotate_y=True, rotate_x=True,
                          aug_dict=None):
    """Apply the same (wild) augmentations on a list of point clouds."""
    if aug_dict is None:
        aug_dict = {
            'flip_yz': flip and np.random.random() > 0.5,
            'flip_xz': flip and np.random.random() > 0.5,
            'translate': 2.5 * np.random.uniform(-1, 1, 3) * translate,
            'rot_z': (np.random.random() * np.pi * 2 - np.pi) * rotate_z,
            'rot_y': (np.random.random() * np.pi * 2 - np.pi) * rotate_y,
            'rot_x': (np.random.random() * np.pi * 2 - np.pi) * rotate_x
        }

    # Flip
    pc = np.stack(pc)
    if aug_dict['flip_yz']:
        # Flipping along the YZ plane
        pc[..., 0] = -pc[..., 0]
    if aug_dict['flip_xz']:
        # Flipping along the XZ plane
        pc[..., 1] = -pc[..., 1]
    pc = [pc for pc in pc]

    # Rotations
    rot_angle = aug_dict['rot_z']
    if rot_angle > 0:
        pc = [rot_z(pc, rot_angle) for pc in pc]
    rot_angle = aug_dict['rot_y']
    if rot_angle > 0:
        pc = [rot_y(pc, rot_angle) for pc in pc]
    rot_angle = aug_dict['rot_x']
    if rot_angle > 0:
        pc = [rot_x(pc, rot_angle) for pc in pc]

    # Translation
    pc = np.stack(pc)
    pc += aug_dict['translate'][None, None]
    pc = [pc_ for pc_ in pc]
    return pc, aug_dict
