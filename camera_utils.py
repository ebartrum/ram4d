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
import os
import numpy as np
import torch
from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary, qvec2rotmat
from utils.pose_utils import generate_ellipse_path

class _SimpleCamera:
    """Minimal stub for generate_ellipse_path — needs R (C2W) and T (W2C translation)."""
    def __init__(self, R_c2w, tvec):
        self.R = R_c2w
        self.T = tvec

def create_orbit_cameras(scene_path, n_frames):
    """Generate an orbit trajectory from all COLMAP cameras using generate_ellipse_path."""

    images_bin  = os.path.join(scene_path, "sparse", "0", "images.bin")
    cameras_bin = os.path.join(scene_path, "sparse", "0", "cameras.bin")
    extrinsics = read_extrinsics_binary(images_bin)
    intrinsics = read_intrinsics_binary(cameras_bin)
    sorted_images = sorted(extrinsics.values(), key=lambda x: x.name)

    views = [
        _SimpleCamera(R_c2w=qvec2rotmat(img.qvec).T,
                      tvec=np.array(img.tvec, dtype=np.float64))
        for img in sorted_images
    ]

    # Use camera 0's intrinsics as reference FoV
    cam0 = intrinsics[sorted_images[0].camera_id]
    W, H = int(cam0.width), int(cam0.height)
    if cam0.model in ("PINHOLE", "OPENCV"):
        fx, fy = cam0.params[0], cam0.params[1]
    elif cam0.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
        fx = fy = cam0.params[0]
    else:
        raise ValueError(f"Unsupported COLMAP camera model: {cam0.model}")
    FoVx = 2.0 * math.atan(W / (2.0 * fx))
    FoVy = 2.0 * math.atan(H / (2.0 * fy))

    print(f"  Generating orbit from {len(views)} training cameras → {n_frames} frames")
    poses_w2c = generate_ellipse_path(views, n_frames=n_frames, is_circle=True)
    return poses_w2c, FoVx, FoVy, W, H


def make_minicam_at_resolution(cam_info, width, height, znear=0.01, zfar=100.0):
    """
    Create an Inpaint360GS MiniCam at (width × height) from a cam_info dict.

    Keeps cam_info["fovx"] and recomputes fovy for square pixels:
        fovy = 2 * atan(tan(fovx/2) * height / width)

    cam_info must have keys: "fovx", "R" (3×3 rotation c2w), "T" (translation w2c).

    Requires Inpaint360GS to be on sys.path (lazy import).
    Replaces make_wan_minicam() in run_gs_replace.py.
    """
    from scene.cameras import MiniCam
    from utils.graphics_utils import getWorld2View2, getProjectionMatrix

    fovx = cam_info["fovx"]
    fovy = 2.0 * math.atan(math.tan(fovx / 2.0) * height / width)

    world_view_transform = torch.tensor(
        getWorld2View2(cam_info["R"], cam_info["T"])
    ).transpose(0, 1).cuda()

    proj = getProjectionMatrix(
        znear=znear, zfar=zfar, fovX=fovx, fovY=fovy
    ).transpose(0, 1).cuda()
    full_proj = world_view_transform.unsqueeze(0).bmm(proj.unsqueeze(0)).squeeze(0)

    return MiniCam(width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj)


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
