"""
camera_utils.py — Shared camera utilities for ram4d scripts.

Core function: make_raster_camera_at_resolution()

  Given a COLMAP camera (pose_w2c 4×4, fovx in radians) and a target render
  resolution (width, height), returns the matrices needed by
  diff_gaussian_rasterization's GaussianRasterizer.

  The horizontal FoV (fovx) is kept unchanged; fovy is recomputed so that
  pixels remain square at the target resolution:

      fovy = 2 * atan(tan(fovx / 2) * height / width)

  This exactly replicates PIL.Image.resize((width, height)) applied to the
  original camera image — the rendered output will align pixel-for-pixel with
  a reference image that was produced by that resize, regardless of the
  original camera aspect ratio.

  This is the same approach as make_wan_minicam() in run_gs_replace.py.
"""

import math
import numpy as np
import torch


def make_raster_camera_at_resolution(pose_w2c, fovx, width, height, device,
                                     znear=0.01, zfar=200.0):
    """
    Build (viewmatrix, full_proj, campos, tanfovx, tanfovy) for
    GaussianRasterizationSettings at an arbitrary target resolution.

    Keeps fovx and recomputes fovy for square pixels:
        fovy = 2 * atan(tan(fovx/2) * height / width)

    Args:
        pose_w2c : (4, 4) float64 or float32 world-to-camera matrix
        fovx     : horizontal field of view in radians (from COLMAP intrinsics)
        width    : target render width in pixels
        height   : target render height in pixels
        device   : torch device string or object

    Returns:
        viewmatrix  – (4, 4) column-major W2C, float32 tensor on device
        full_proj   – (4, 4) combined view-projection, float32 tensor on device
        campos      – (3,) camera centre in world space, float32 tensor on device
        tanfovx     – float
        tanfovy     – float
    """
    fovy = 2.0 * math.atan(math.tan(fovx / 2.0) * height / width)

    R_w2c = pose_w2c[:3, :3].astype(np.float32)
    t     = pose_w2c[:3, 3].astype(np.float32)
    R_c2w = R_w2c.T

    W2C = np.eye(4, dtype=np.float32)
    W2C[:3, :3] = R_w2c
    W2C[:3,  3] = t

    tanfovx = math.tan(fovx / 2.0)
    tanfovy = math.tan(fovy / 2.0)

    P = np.zeros((4, 4), dtype=np.float32)
    P[0, 0] =  1.0 / tanfovx
    P[1, 1] =  1.0 / tanfovy
    P[3, 2] =  1.0
    P[2, 2] =  zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)

    viewmatrix = torch.from_numpy(W2C.T).float().to(device)
    proj_t     = torch.from_numpy(P.T ).float().to(device)
    full_proj  = viewmatrix.unsqueeze(0).bmm(proj_t.unsqueeze(0)).squeeze(0)
    campos     = torch.tensor(-(R_c2w @ t), dtype=torch.float32, device=device)

    return viewmatrix, full_proj, campos, tanfovx, tanfovy
