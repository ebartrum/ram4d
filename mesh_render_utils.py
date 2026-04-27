"""
mesh_render_utils.py — Shared utilities for rendering dynamic textured meshes.

Used by texture_actionmesh.py and render_textured_mesh.py.
"""

import numpy as np
import torch
from torchvision import transforms

from visualise_output import apply_mesh_transforms
from mvadapter.utils import make_image_grid
from mvadapter.utils.mesh_utils import render


def load_and_transform_deformations(deformations_path, offset, scale):
    """
    Load deformations_vertices.npy, undo the save_deformation coordinate
    transform (saved as [-z, x, y]), and apply mesh transforms.

    Returns:
        deformations_orig: (T, V, 3) float32 ndarray in mesh space
    """
    deformations = np.load(deformations_path)  # (T, V, 3)
    T = deformations.shape[0]

    deformations_orig = np.zeros_like(deformations)
    deformations_orig[:, :, 0] = deformations[:, :, 1]
    deformations_orig[:, :, 1] = deformations[:, :, 2]
    deformations_orig[:, :, 2] = -deformations[:, :, 0]

    for i in range(T):
        deformations_orig[i] = apply_mesh_transforms(
            deformations_orig[i], offset, scale, front_x_to_y=True
        )

    return deformations_orig


def compute_vertex_mapping(mesh_v_pos, deform_verts_0, device):
    """
    Compute nearest-neighbour mapping from textured mesh vertices to
    deformation vertices (frame 0).

    Returns:
        mapping_idx: (V_mesh,) long tensor
    """
    v_orig = torch.from_numpy(deform_verts_0).float().to(device)
    dist   = torch.cdist(mesh_v_pos.unsqueeze(0), v_orig.unsqueeze(0)).squeeze(0)
    _, mapping_idx = torch.min(dist, dim=1)
    avg_dist = dist.min(dim=1).values.mean().item()
    print(f"  NN mapping avg distance: {avg_dist:.6f}")
    if avg_dist > 1e-4:
        print("  WARNING: High avg distance — possible misalignment.")
    return mapping_idx


def render_dynamic_mesh_video(ctx, mesh, cameras, deformations_orig,
                              mapping_idx, height, width,
                              prefix_frames=None):
    """
    Render a dynamic textured mesh as a sequence of multi-view grid frames.

    Args:
        ctx:               nvdiffrast context
        mesh:              TexturedMesh (v_pos will be updated each frame)
        cameras:           camera batch from get_orthogonal_camera
        deformations_orig: (T, V, 3) ndarray in mesh space
        mapping_idx:       (V_mesh,) long tensor mapping mesh verts to deform verts
        height, width:     render resolution per view
        prefix_frames:     optional list of T PIL images prepended to each row

    Returns:
        video_frames: list of (H, W*n_views, 3) uint8 ndarrays
    """
    device = mapping_idx.device
    T = deformations_orig.shape[0]
    video_frames = []

    for i in range(T):
        verts = torch.from_numpy(deformations_orig[i]).float().to(device)
        mesh.v_pos           = verts[mapping_idx]
        mesh._v_nrm          = None
        mesh._v_tang         = None
        mesh._stitched_v_pos = None

        render_out  = render(ctx, mesh, cameras,
                             height=height, width=width,
                             render_attr=True,
                             render_depth=False,
                             render_normal=False)
        frame_pils  = [transforms.ToPILImage()(img)
                       for img in render_out.attr.clamp(0, 1).cpu().permute(0, 3, 1, 2)]

        if prefix_frames is not None:
            frame_pils.insert(0, prefix_frames[i])

        video_frames.append(np.array(make_image_grid(frame_pils, rows=1)))

        if i % 10 == 0:
            print(f"  Frame {i}/{T}")

    return video_frames
