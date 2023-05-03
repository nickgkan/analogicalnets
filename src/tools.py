import numpy as np
import torch
import trimesh
from pytorch3d.structures import Meshes
from pytorch3d.structures.meshes import join_meshes_as_scene
from pytorch3d.renderer import (
    TexturesVertex,
    look_at_view_transform,
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader
)


COLORS = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 1, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 1],
    [1, 0.5, 0],
    [0.5, 1, 0],
    [0, 1, 0.5],
    [0, 0.5, 1],
    [0.5, 0, 1],
    [1, 0, 0.5],
    [0.5, 1, 1],
    [1, 0.5, 1],
    [1, 1, 0.5]
])
rng = np.random.RandomState(313)
COLORS = np.concatenate([COLORS, COLORS * 0.5])
COLORS = np.concatenate([COLORS, rng.rand(256, 3) * 2])
COLORS = np.clip(COLORS, a_min=0, a_max=1)


def rot_x(pc, theta):
    """Rotate along x-axis."""
    return np.matmul(
        np.array([
            [1.0, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)]
        ]),
        pc.T
    ).T


def rot_y(pc, theta):
    """Rotate along y-axis."""
    return np.matmul(
        np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1.0, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ]),
        pc.T
    ).T


def rot_z(pc, theta):
    """Rotate along z-axis."""
    return np.matmul(
        np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1.0]
        ]),
        pc.T
    ).T


def normalize_pc(pc):
    """Normalize point cloud (P, 3) to a unit sphere."""
    # Center along mean
    point_set = pc - np.expand_dims(np.mean(pc, axis=0), 0)
    # Find 'radius'
    dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
    return point_set / dist  # scale


def visualize_mesh(pc, color_ids):
    # pc (N, 3), color_ids (N,) are numpy arrays
    # create mesh for each part
    pc = rot_x(pc, -np.pi / 2)
    mesh_list = []
    for color_uid in np.unique(color_ids):
        # marching cubes to convert pc to mesh
        part_pc = pc[color_ids == color_uid]
        part_mesh = trimesh.voxel.ops.points_to_marching_cubes(
            part_pc, pitch=0.05
        )

        # create PyTorch3D mesh
        verts = torch.from_numpy(part_mesh.vertices).float().unsqueeze(0)
        faces = torch.from_numpy(part_mesh.faces).float().unsqueeze(0)
        if color_uid < 0:
            verts_rgb = (
                torch.ones_like(verts)
                * torch.as_tensor([0.0, 0, 0]).reshape(1, 1, 3)
            )
        else:
            color_uid = color_uid % len(COLORS)
            verts_rgb = (
                torch.ones_like(verts)
                * torch.from_numpy(COLORS[color_uid]).reshape(1, 1, 3)
            )
        verts_rgb = verts_rgb.float()
        textures = TexturesVertex(verts_features=verts_rgb.cuda())
        mesh = Meshes(
            verts=verts.cuda(),
            faces=faces.cuda(),
            textures=textures
        )
        mesh_list.append(mesh)

    # combine all parts
    mesh_to_render = join_meshes_as_scene(mesh_list)

    # render views and log
    # render_params = [[2, 0, 0], [2, 120, 60]]
    render_params = [[2, 0, 0], [2, -80, 0]]
    render_params = [[2, -80, 0]]
    render_params = [[2, 0, 30]]
    vislist = []
    for render_param in render_params:
        R, T = look_at_view_transform(
            dist=render_param[0],
            elev=render_param[1],
            azim=render_param[2]
        )
        cameras = FoVPerspectiveCameras(device='cuda', R=R, T=T)
        raster_settings = RasterizationSettings(
            image_size=256,
            blur_radius=0.0,
            faces_per_pixel=1,
        )
        rasterizer = MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings,
        )
        shader = SoftPhongShader(device='cuda', cameras=cameras)
        renderer = MeshRenderer(rasterizer, shader)
        image = renderer(mesh_to_render)
        image = (image[0, :, :, :3].cpu().numpy() * 255).astype(int)

        # vislist.append(wandb.Image(image))
        vislist.append(image)
    return vislist
