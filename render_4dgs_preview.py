"""
render_4dgs_preview.py — Render 4DGS output as a multi-view preview video.

Renders 4 orthogonal views side by side (matching the layout of
textured_dynamic_mesh.mp4): front, right, back, left.

Camera: azimuth_deg=[180, -90, 0, 90], elevation=0, distance=2.2.
Same orthographic projection as get_orthogonal_camera in mvadapter.

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

C0 = 0.28209479177387814

# Orthographic view half-size (world units), matching get_orthogonal_camera
HALF_FOV = 0.55

# 4 views matching texture_actionmesh.py (azimuths=[270,0,90,180], mapped x-90)
DEFAULT_AZIMUTHS_DEG = [180, -90, 0, 90]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--size", type=int, default=768,
                        help="Per-view image size (square). Default 768.")
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--max_r", type=int, default=10,
                        help="Max Gaussian splat radius in pixels.")
    parser.add_argument("--sigma_scale", type=float, default=1.0,
                        help="Multiplicative scale on Gaussian sigma for visibility.")
    return parser.parse_args()


def load_ply(path):
    v = PlyData.read(path).elements[0]
    xyz     = np.stack([v['x'],     v['y'],     v['z']    ], axis=1).astype(np.float32)
    f_dc    = np.stack([v['f_dc_0'], v['f_dc_1'], v['f_dc_2']], axis=1).astype(np.float32)
    opacity = v['opacity'].astype(np.float32)
    log_sc  = np.stack([v['scale_0'], v['scale_1'], v['scale_2']], axis=1).astype(np.float32)
    rot     = np.stack([v['rot_0'],  v['rot_1'],  v['rot_2'],  v['rot_3'] ], axis=1).astype(np.float32)
    return xyz, f_dc, opacity, log_sc, rot


def quaternion_to_matrix(q):
    """(N, 4) wxyz → (N, 3, 3)"""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = torch.stack([
        1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y),
        2*(x*y + w*z),      1 - 2*(x*x + z*z),   2*(y*z - w*x),
        2*(x*z - w*y),      2*(y*z + w*x),       1 - 2*(x*x + y*y),
    ], dim=-1).reshape(-1, 3, 3)
    return R


def precompute_sigma3d(R, scales, sigma_scale=1.0):
    """Σ3D = R diag(s²) R^T  →  (N, 3, 3)"""
    s = scales * sigma_scale            # (N, 3)
    RS = R * (s ** 2)[:, None, :]       # (N, 3, 3): columns of R scaled by s²
    return RS @ R.transpose(1, 2)       # (N, 3, 3)


def sigma2d_for_azimuth(sigma3d, azimuth_deg, W, H):
    """
    Compute per-Gaussian 2D pixel-space covariance for a given camera azimuth.

    For elevation=0, azimuth θ (radians):
      Camera right direction (world):  (-sin θ,  cos θ, 0)
      Camera up direction (world):     (0,       0,     1)

    Pixel projection (world xyz → pixel xy):
      px = W/2 + (-sin(θ)·x + cos(θ)·y) · kx
      py = H/2 - z · ky

    Jacobian J (2×3):
      J = [[-sin(θ)·kx,  cos(θ)·kx,  0   ],
           [ 0,           0,          -ky  ]]

    Σ2D = J · Σ3D · J^T

    Returns:
        sigma2d_inv: (N, 2, 2)
    """
    theta = math.radians(azimuth_deg)
    kx = W / (2.0 * HALF_FOV)
    ky = H / (2.0 * HALF_FOV)
    device = sigma3d.device

    J = torch.tensor(
        [[-math.sin(theta) * kx,  math.cos(theta) * kx,  0.0],
         [ 0.0,                   0.0,                   -ky]],
        dtype=torch.float32, device=device
    )  # (2, 3)

    JSigma = J @ sigma3d         # (N, 2, 3)
    sigma2d = JSigma @ J.T       # (N, 2, 2)

    # Small regularizer for numerical stability
    sigma2d = sigma2d + torch.eye(2, device=device)[None] * 1e-4

    # Batch invert 2×2
    det = (sigma2d[:, 0, 0] * sigma2d[:, 1, 1]
           - sigma2d[:, 0, 1] * sigma2d[:, 1, 0]).abs().clamp(min=1e-10)
    sigma2d_inv = torch.stack([
         sigma2d[:, 1, 1] / det, -sigma2d[:, 0, 1] / det,
        -sigma2d[:, 1, 0] / det,  sigma2d[:, 0, 0] / det,
    ], dim=-1).reshape(-1, 2, 2)

    return sigma2d_inv


def render_one_view(positions, sigma2d_inv, colors, alpha_val,
                    azimuth_deg, W, H, max_r, device):
    """
    Render a single orthographic view as an (H, W, 3) uint8 numpy image.

    Projection for azimuth θ:
      cam_x = -sin(θ)·x + cos(θ)·y
      cam_y = z
      depth  = cos(θ)·x + sin(θ)·y  (ascending = back-to-front)
      px = W/2 + cam_x · kx
      py = H/2 - cam_y · ky
    """
    theta = math.radians(azimuth_deg)
    kx = W / (2.0 * HALF_FOV)
    ky = H / (2.0 * HALF_FOV)

    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
    cam_x = -math.sin(theta) * x + math.cos(theta) * y   # (N,)
    cam_y = z                                              # (N,)
    depth = math.cos(theta) * x + math.sin(theta) * y    # (N,)

    Hp = H + 2 * max_r
    Wp = W + 2 * max_r

    cx = W / 2.0 + cam_x * kx + max_r   # (N,) padded image x
    cy = H / 2.0 - cam_y * ky + max_r   # (N,) padded image y

    cx_int = cx.round().long()
    cy_int = cy.round().long()

    # Local pixel grid: (k, k, 2) [dx, dy]
    k = 2 * max_r + 1
    offs = torch.arange(-max_r, max_r + 1, device=device, dtype=torch.float32)
    dy_grid, dx_grid = torch.meshgrid(offs, offs, indexing='ij')   # (k, k)
    d_flat = torch.stack([dx_grid.reshape(-1), dy_grid.reshape(-1)], dim=-1)  # (k², 2)
    k2 = k * k

    # Mahalanobis: d^T Σ⁻¹ d for all (Gaussian, local_pixel) pairs
    temp = torch.einsum('nij,kj->nki', sigma2d_inv, d_flat)   # (N, k², 2)
    mahal = (d_flat[None] * temp).sum(-1).clamp(min=0.0)       # (N, k²)

    gauss_w = torch.exp(-0.5 * mahal)                          # (N, k²)
    eff_alpha = alpha_val[:, None] * gauss_w                   # (N, k²)

    # Pixel indices in padded image
    px_all = (cx_int[:, None] + dx_grid.reshape(1, -1).long()).clamp(0, Wp - 1)  # (N, k²)
    py_all = (cy_int[:, None] + dy_grid.reshape(1, -1).long()).clamp(0, Hp - 1)  # (N, k²)
    flat_idx = (py_all * Wp + px_all).reshape(-1)              # (N × k²)

    # Contributions
    color_contrib = (eff_alpha[:, :, None] * colors[:, None, :]).reshape(-1, 3)
    alpha_contrib = eff_alpha.reshape(-1)

    # Sort back-to-front (ascending depth = farthest first → rendered first)
    sort_order = torch.argsort(depth, descending=False)        # farthest first
    sort_order_k2 = (sort_order[:, None] * k2 +
                     torch.arange(k2, device=device)[None]).reshape(-1)
    color_contrib = color_contrib[sort_order_k2]
    alpha_contrib = alpha_contrib[sort_order_k2]
    flat_idx      = flat_idx[sort_order_k2]

    # Scatter-accumulate
    img_flat_color = torch.zeros(Hp * Wp, 3, device=device)
    img_flat_alpha = torch.zeros(Hp * Wp,    device=device)
    img_flat_color.scatter_add_(0, flat_idx[:, None].expand(-1, 3), color_contrib)
    img_flat_alpha.scatter_add_(0, flat_idx, alpha_contrib)

    img_color = img_flat_color.view(Hp, Wp, 3)
    img_alpha = img_flat_alpha.view(Hp, Wp)

    # Composite over white background
    alpha_norm = (img_alpha / (img_alpha + 1e-8)).clamp(0, 1)
    color_norm = (img_color / (img_alpha.unsqueeze(-1) + 1e-8)).clamp(0, 1)
    result = alpha_norm.unsqueeze(-1) * color_norm + (1 - alpha_norm.unsqueeze(-1))

    # Crop padding
    result = result[max_r:max_r + H, max_r:max_r + W]
    return (result.cpu().numpy() * 255).astype(np.uint8)


def main():
    args = parse_args()
    gaussians_dir = os.path.join(args.output_path, "gaussians")
    ply_path      = os.path.join(gaussians_dir, "gaussians.ply")
    offsets_path  = os.path.join(gaussians_dir, "deformation_offsets.npy")

    W = H = args.size
    azimuths = DEFAULT_AZIMUTHS_DEG

    print(f"\n--- Loading PLY ---")
    xyz, f_dc, opacity_raw, log_scale, rot_wxyz = load_ply(ply_path)
    N = xyz.shape[0]
    print(f"  N Gaussians: {N}")

    rgb       = np.clip(f_dc * C0 + 0.5, 0.0, 1.0)              # (N, 3)
    alpha_val = 1.0 / (1.0 + np.exp(-opacity_raw))               # (N,) sigmoid

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    xyz_t    = torch.from_numpy(xyz).float().to(device)
    rgb_t    = torch.from_numpy(rgb).float().to(device)
    alpha_t  = torch.from_numpy(alpha_val).float().to(device)
    scales_t = torch.from_numpy(np.exp(log_scale)).float().to(device)
    rot_t    = torch.from_numpy(rot_wxyz).float().to(device)

    # Precompute Σ3D (fixed) and Σ2D_inv per camera azimuth
    print("--- Precomputing covariances ---")
    R_mat    = quaternion_to_matrix(rot_t)                        # (N, 3, 3)
    sigma3d  = precompute_sigma3d(R_mat, scales_t, args.sigma_scale)  # (N, 3, 3)

    sigma2d_invs = []
    for az in azimuths:
        inv = sigma2d_for_azimuth(sigma3d, az, W, H)
        sigma2d_invs.append(inv)
        print(f"  azimuth {az:4d}° → Σ2D_inv ready")

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
        for az, s2d_inv in zip(azimuths, sigma2d_invs):
            img = render_one_view(pos, s2d_inv, rgb_t, alpha_t,
                                  az, W, H, args.max_r, device)
            view_imgs.append(img)

        # Stitch horizontally (1 row × n_views columns)
        row = np.concatenate(view_imgs, axis=1)   # (H, n_views*W, 3)
        frames.append(row)

        if t % 10 == 0:
            print(f"  Frame {t}/{T}")

    video_path = os.path.join(gaussians_dir, "gaussians_preview.mp4")
    imageio.mimsave(video_path, frames, fps=args.fps)
    print(f"\nDone. Video: {video_path}")
    print(f"  {T} frames @ {args.fps} fps, grid {W*n_views}×{H}")


if __name__ == "__main__":
    main()
