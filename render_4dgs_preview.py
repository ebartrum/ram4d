"""
render_4dgs_preview.py — Render 4DGS output as a multi-view preview video.

Renders 4 orthogonal views side by side (matching the layout of
textured_dynamic_mesh.mp4): front, right, back, left.

Uses diff_gaussian_rasterization_inpaint360gs — the same CUDA rasterizer used by
Inpaint360GS — for standard alpha-compositing Gaussian splatting with transmittance.

Camera: azimuth_deg=[180, -90, 0, 90], elevation=0.
Camera placed at CAM_DIST=100 with tanfov=HALF_FOV/CAM_DIST for near-orthographic
projection matching the existing 4-view layout (HALF_FOV=0.55 world units).

Usage:
  python render_4dgs_preview.py --output_path output/2026.03.02/actionmesh_gs_replace_corgi
  python render_4dgs_preview.py --output_path ... --size 768 --fps 16
"""

import sys
import os
import glob
import math

if "LIBGOMP_PRELOADED" not in os.environ:
    preload_files = []
    libgomp_files = glob.glob(
        "/home/ubuntu/miniconda3/envs/mvadapter/lib/python*/site-packages/scikit_learn.libs/libgomp*.so*"
    )
    if libgomp_files:
        preload_files.append(libgomp_files[0])
    libgl_path = "/usr/lib/aarch64-linux-gnu/libGLdispatch.so.0"
    if os.path.exists(libgl_path):
        preload_files.append(libgl_path)
    if preload_files:
        os.environ["LD_PRELOAD"] = ":".join(preload_files)
        os.environ["LIBGOMP_PRELOADED"] = "1"
        os.execv(sys.executable, [sys.executable] + sys.argv)

import argparse
import numpy as np
import torch
import imageio
from plyfile import PlyData

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

C0 = 0.28209479177387814

# Camera placed far away so tanfov = HALF_FOV / CAM_DIST is tiny → near-orthographic.
CAM_DIST = 100.0
HALF_FOV  = 0.55   # world-space half-size of the view (matches get_orthogonal_camera)

# 4 views matching texture_actionmesh.py (azimuths=[270,0,90,180], mapped x-90)
DEFAULT_AZIMUTHS_DEG = [180, -90, 0, 90]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--size", type=int, default=768,
                        help="Per-view image size (square). Default 768.")
    parser.add_argument("--fps", type=int, default=16)
    return parser.parse_args()


def load_ply(path):
    v = PlyData.read(path).elements[0]
    xyz    = np.stack([v['x'],     v['y'],     v['z']    ], axis=1).astype(np.float32)
    f_dc   = np.stack([v['f_dc_0'], v['f_dc_1'], v['f_dc_2']], axis=1).astype(np.float32)
    opacity = v['opacity'].astype(np.float32)
    log_sc  = np.stack([v['scale_0'], v['scale_1'], v['scale_2']], axis=1).astype(np.float32)
    rot     = np.stack([v['rot_0'],  v['rot_1'],  v['rot_2'],  v['rot_3']], axis=1).astype(np.float32)
    return xyz, f_dc, opacity, log_sc, rot


def make_view_camera(azimuth_deg, device):
    """
    Construct camera matrices for the given azimuth (elevation=0).

    Camera at (CAM_DIST·cosθ, CAM_DIST·sinθ, 0) looking toward the origin.

    Uses 3DGS/COLMAP convention: camera +Z points INTO the scene (forward), so
    z_cam > 0 for visible points. This matches the rasterizer (P[3,2]=+1, w_clip=z_cam>0).

    Camera axes in world space:
      right:    (-sinθ,  cosθ,  0)
      up:       (0,      0,     1)   ← world Z is up
      forward:  (-cosθ,  -sinθ, 0)  ← camera +Z into scene (3DGS convention)

    Convention matches Inpaint360GS cameras.py:
      world_view_transform = W2C.T   (4×4, transposed)
      full_proj_transform  = world_view_transform @ projection_matrix.T

    Returns: viewmatrix (4,4), full_proj (4,4), campos (3,), tanfov (float)
    """
    theta = math.radians(azimuth_deg)
    D     = CAM_DIST
    tanfov = HALF_FOV / D

    # Camera-to-world rotation R  (columns = camera right / up / forward)
    # Camera +Z = forward = (-cosθ, -sinθ, 0)  [into scene, 3DGS convention]
    # Camera +Y = (0, 0, -1): rasterizer maps NDC +Y → image bottom, so we flip
    # up so that world Z+ (up) → cam -Y → NDC -Y → image top.
    R = np.array([
        [-math.sin(theta), 0.0, -math.cos(theta)],
        [ math.cos(theta), 0.0, -math.sin(theta)],
        [ 0.0,            -1.0,  0.0],
    ], dtype=np.float32)

    # World-to-camera translation: t = -R.T @ cam_pos = (0, 0, +D)
    t = np.array([0.0, 0.0, D], dtype=np.float32)

    # W2C 4×4
    W2C = np.eye(4, dtype=np.float32)
    W2C[:3, :3] = R.T
    W2C[:3,  3] = t

    # Perspective projection matrix (matches getProjectionMatrix in Inpaint360GS)
    znear, zfar = 10.0, D * 2.0
    P = np.zeros((4, 4), dtype=np.float32)
    P[0, 0] = 1.0 / tanfov
    P[1, 1] = 1.0 / tanfov
    P[3, 2] = 1.0
    P[2, 2] =   zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)

    # Both matrices stored transposed, matching cameras.py
    viewmatrix = torch.from_numpy(W2C.T).float().to(device)
    proj_t     = torch.from_numpy(P.T  ).float().to(device)
    full_proj  = (viewmatrix.unsqueeze(0).bmm(proj_t.unsqueeze(0))).squeeze(0)

    campos = torch.tensor(
        [D * math.cos(theta), D * math.sin(theta), 0.0],
        dtype=torch.float32, device=device,
    )
    return viewmatrix, full_proj, campos, tanfov


def render_view(means3D, colors, opacities, scales, rotations,
                viewmatrix, full_proj, campos, tanfov, W, H, device):
    """
    Render one view with the Inpaint360GS CUDA rasterizer.
    Returns (H, W, 3) uint8 numpy image.
    """
    N = means3D.shape[0]
    # means2D is (N, 3) zeros — screen-space placeholder (no gradient needed for preview)
    means2D = torch.zeros(N, 3, device=device, dtype=torch.float32)

    raster_settings = GaussianRasterizationSettings(
        image_height=H,
        image_width=W,
        tanfovx=float(tanfov),
        tanfovy=float(tanfov),
        bg=torch.ones(3, device=device, dtype=torch.float32),   # white background
        scale_modifier=1.0,
        viewmatrix=viewmatrix,
        projmatrix=full_proj,
        sh_degree=0,
        campos=campos,
        prefiltered=False,
        debug=False,
        antialiasing=False,
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    rendered_image, _, _ = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=None,
        colors_precomp=colors,
        opacities=opacities,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None,
    )
    # rendered_image: (3, H, W) float
    img = rendered_image.clamp(0.0, 1.0).permute(1, 2, 0)   # (H, W, 3)
    return (img.cpu().numpy() * 255).astype(np.uint8)


def main():
    args = parse_args()
    gaussians_dir = os.path.join(args.output_path, "gaussians")
    ply_path      = os.path.join(gaussians_dir, "gaussians.ply")
    offsets_path  = os.path.join(gaussians_dir, "deformation_offsets.npy")

    W = H = args.size
    azimuths = DEFAULT_AZIMUTHS_DEG

    print("\n--- Loading PLY ---")
    xyz, f_dc, opacity_raw, log_scale, rot_wxyz = load_ply(ply_path)
    N = xyz.shape[0]
    print(f"  N Gaussians: {N}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    # Activate parameters (rasterizer expects activated values)
    xyz_t    = torch.from_numpy(xyz).float().to(device)
    rgb_t    = torch.from_numpy(np.clip(f_dc * C0 + 0.5, 0.0, 1.0)).float().to(device)
    alpha_t  = torch.from_numpy(
        1.0 / (1.0 + np.exp(-opacity_raw))
    ).float().to(device).unsqueeze(1)                          # (N, 1) sigmoid
    scales_t = torch.from_numpy(np.exp(log_scale)).float().to(device)
    rot_t    = torch.from_numpy(rot_wxyz).float().to(device)   # (N, 4) wxyz, already normalised

    # Precompute per-view cameras
    print("--- Setting up cameras ---")
    cameras = []
    for az in azimuths:
        vm, fp, cp, tanfov = make_view_camera(az, device)
        cameras.append((vm, fp, cp, tanfov))
        print(f"  azimuth {az:4d}° ready  (tanfov={tanfov:.5f})")

    # Deformation offsets
    if os.path.exists(offsets_path):
        offsets = np.load(offsets_path)   # (T, N, 3)
        T = offsets.shape[0]
        print(f"\n  Deformation offsets: {offsets.shape}")
    else:
        offsets = None
        T = 1
        print("\n  No offsets found — static frame only.")

    # Render
    n_views = len(azimuths)
    print(f"\n--- Rendering {T} frames × {n_views} views at {W}×{H} each ---")
    frames = []
    for t in range(T):
        pos = xyz_t
        if offsets is not None:
            pos = xyz_t + torch.from_numpy(offsets[t]).float().to(device)

        view_imgs = []
        for az, (vm, fp, cp, tanfov) in zip(azimuths, cameras):
            img = render_view(pos, rgb_t, alpha_t, scales_t, rot_t,
                              vm, fp, cp, tanfov, W, H, device)
            view_imgs.append(img)

        row = np.concatenate(view_imgs, axis=1)   # (H, n_views*W, 3)
        frames.append(row)

        if t % 10 == 0:
            print(f"  Frame {t}/{T}")

    # Always save frame 0 as a PNG for quick visual inspection
    png_path = os.path.join(gaussians_dir, "gaussians_preview_frame0.png")
    imageio.imwrite(png_path, frames[0])
    print(f"  Frame 0 PNG: {png_path}")

    video_path = os.path.join(gaussians_dir, "gaussians_preview.mp4")
    imageio.mimsave(video_path, frames, fps=args.fps)
    print(f"\nDone. Video: {video_path}")
    print(f"  {T} frames @ {args.fps} fps, grid {W * n_views}×{H}")


if __name__ == "__main__":
    main()
