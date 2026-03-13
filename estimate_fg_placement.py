"""
estimate_fg_placement.py — Estimate 3D placement of foreground 4DGS in background 3DGS scene.

Steps:
  1. Load COLMAP camera parameters for camera_idx (sorted alphabetically, matching
     Inpaint360GS Scene class convention).
  2. Project background Gaussian centres into the camera to get sparse ground-truth
     z-depths at background pixels (avoids any invdepth/depth ambiguity from the
     rasterizer).
  3. Run Depth Anything V2 on Wan video frame 0.
  4. Load SAM2 foreground mask for frame 0.
  5. Fit linear calibration  gs_z ≈ a * mono_depth + b  on background pixels.
  6. Unproject foreground pixels with calibrated depth → 3D point cloud.
  7. Estimate bounding-box centre (translation) and extent (scale).
  8. Save placement.json alongside gaussians.ply.

Usage:
  python estimate_fg_placement.py \\
      --output_path  output/2026.03.02/actionmesh_gs_replace_corgi \\
      --gs_scene_path /path/to/inpaint360gs/scene \\
      --gs_model_path /path/to/inpaint360gs/model \\
      --camera_idx 28
"""

import sys
import os
import glob

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

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, "Inpaint360GS"))

import argparse
import json
import math
import numpy as np
import torch
import torch.nn.functional as F
import imageio
from PIL import Image
from plyfile import PlyData

from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary, qvec2rotmat
from utils.point_utils import get_intrinsics, create_point_cloud
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", required=True,
                        help="ram4d output dir (contains gaussians/, sam2_masks/, final_output.mp4)")
    parser.add_argument("--gs_scene_path", required=True,
                        help="Inpaint360GS scene dir (contains sparse/0/)")
    parser.add_argument("--gs_model_path", required=True,
                        help="Inpaint360GS model dir (contains point_cloud/iteration_N/)")
    parser.add_argument("--camera_idx", type=int, default=28,
                        help="Camera index used to generate the Wan video (0-based, sorted by image name)")
    parser.add_argument("--depth_model", default="depth-anything/Depth-Anything-V2-Large-hf",
                        help="HuggingFace model ID for Depth Anything V2")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Camera loading
# ---------------------------------------------------------------------------

def load_colmap_camera(scene_path, camera_idx):
    """
    Load COLMAP camera at the given index (sorted alphabetically by image name,
    matching Inpaint360GS Scene class convention).

    Returns: R_c2w (3,3), tvec (3,), FoVx (rad), FoVy (rad), W (int), H (int)
    """
    images_bin  = os.path.join(scene_path, "sparse", "0", "images.bin")
    cameras_bin = os.path.join(scene_path, "sparse", "0", "cameras.bin")

    extrinsics = read_extrinsics_binary(images_bin)   # {colmap_id: Image}
    intrinsics = read_intrinsics_binary(cameras_bin)  # {colmap_id: Camera}

    # Sort by image name — must match Inpaint360GS dataset_readers.py
    sorted_images = sorted(extrinsics.values(), key=lambda x: x.name)

    if camera_idx >= len(sorted_images):
        raise ValueError(f"camera_idx {camera_idx} out of range ({len(sorted_images)} cameras)")

    img = sorted_images[camera_idx]
    cam = intrinsics[img.camera_id]

    print(f"  Camera {camera_idx}: '{img.name}'  COLMAP id={img.id}  model={cam.model}")

    # Extrinsics: COLMAP qvec/tvec define world-to-camera transform
    R_w2c = qvec2rotmat(img.qvec)      # (3,3)  world→camera
    tvec  = np.array(img.tvec, dtype=np.float64)  # (3,)
    R_c2w = R_w2c.T                    # (3,3)  camera→world

    W, H = int(cam.width), int(cam.height)

    # Focal lengths → FoV
    if cam.model in ("PINHOLE", "OPENCV"):
        fx, fy = cam.params[0], cam.params[1]
    elif cam.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
        fx = fy = cam.params[0]
    else:
        raise ValueError(f"Unsupported COLMAP camera model: {cam.model}")

    FoVx = 2.0 * math.atan(W / (2.0 * fx))
    FoVy = 2.0 * math.atan(H / (2.0 * fy))

    return R_c2w, tvec, FoVx, FoVy, W, H


# ---------------------------------------------------------------------------
# Background GS depth reference (via projection — no rasterizer depth needed)
# ---------------------------------------------------------------------------

def find_latest_ply(model_path):
    pc_dirs = glob.glob(os.path.join(model_path, "iteration_*"))
    if not pc_dirs:
        pc_dirs = glob.glob(os.path.join(model_path, "point_cloud", "iteration_*"))
    if not pc_dirs:
        raise FileNotFoundError(f"No iteration_* in {model_path}")
    pc_dirs.sort(key=lambda p: int(p.split("_")[-1]))
    ply = os.path.join(pc_dirs[-1], "point_cloud.ply")
    if not os.path.exists(ply):
        raise FileNotFoundError(f"PLY not found: {ply}")
    return ply


def load_gs_centers(ply_path):
    """Load only the xyz positions of background Gaussians."""
    v = PlyData.read(ply_path).elements[0]
    xyz = np.stack([v['x'], v['y'], v['z']], axis=1).astype(np.float64)
    return xyz


def project_gs_to_camera(xyz, R_c2w, tvec, FoVx, FoVy, W, H, znear=0.1):
    """
    Project GS centres (N,3) into the camera and return pixel coords + z_cam depths
    for all points that land inside the image.

    Uses standard COLMAP/pinhole convention:
      p_cam = R_w2c @ p_world + tvec   (COLMAP world→camera)
      u = fx * p_cam_x / z_cam + cx
      v = fy * p_cam_y / z_cam + cy    (y increasing downward)

    Returns: u (M,), v (M,), z_cam (M,)  — integer pixel coords and float depths
    """
    R_w2c = R_c2w.T
    fx = W / (2.0 * math.tan(FoVx * 0.5))
    fy = H / (2.0 * math.tan(FoVy * 0.5))
    cx, cy = W / 2.0, H / 2.0

    p_cam = (R_w2c @ xyz.T).T + tvec   # (N, 3)
    z = p_cam[:, 2]

    valid = z > znear
    p_cam, z = p_cam[valid], z[valid]

    u_f = p_cam[:, 0] / z * fx + cx
    v_f = p_cam[:, 1] / z * fy + cy

    in_bounds = (u_f >= 0) & (u_f < W) & (v_f >= 0) & (v_f < H)
    return u_f[in_bounds].astype(int), v_f[in_bounds].astype(int), z[in_bounds]


# ---------------------------------------------------------------------------
# Depth Anything V2
# ---------------------------------------------------------------------------

def run_depth_anything(frame_rgb, processor, da_model, device):
    """
    Run Depth Anything V2 on a single uint8 RGB frame.
    Returns mono_depth (H, W) float32 — relative depth, larger = farther.
    """
    H, W = frame_rgb.shape[:2]
    pil_img = Image.fromarray(frame_rgb)
    inputs = processor(images=pil_img, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = da_model(**inputs)
    pred = outputs.predicted_depth   # (1, H', W')
    if pred.dim() == 4:
        pred = pred.squeeze(1)
    pred_resized = F.interpolate(
        pred.unsqueeze(1).float(), size=(H, W), mode="bilinear", align_corners=False
    ).squeeze()
    return pred_resized.cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def calibrate_depth(mono_depth, u_bg, v_bg, z_bg_ref, min_pts=50):
    """
    Fit linear calibration:  z_ref ≈ a * mono_depth[v, u] + b
    using background Gaussian projection points.

    Returns: a (float), b (float)
    """
    mono_at_bg = mono_depth[v_bg, u_bg].astype(np.float64)
    z_ref = z_bg_ref.astype(np.float64)

    if len(mono_at_bg) < min_pts:
        raise RuntimeError(f"Too few background reference points: {len(mono_at_bg)}")

    A = np.stack([mono_at_bg, np.ones_like(mono_at_bg)], axis=1)
    (a, b), _, _, _ = np.linalg.lstsq(A, z_ref, rcond=None)

    corr = np.corrcoef(mono_at_bg, z_ref)[0, 1]
    print(f"  Linear fit: z = {a:.4f} * mono + {b:.4f}   r={corr:.3f}  n={len(z_ref)}")
    return float(a), float(b)


# ---------------------------------------------------------------------------
# Foreground localisation
# ---------------------------------------------------------------------------

def estimate_placement(calibrated_depth, fg_mask, R_c2w, tvec, FoVx, FoVy, W, H,
                       pct_lo=5, pct_hi=95):
    """
    Unproject foreground pixels with calibrated depth to 3D world space.
    Returns centre (3,) and extent (3,) using percentile bounding box.
    """
    K = get_intrinsics(H, W, FoVx, FoVy)   # (3,3)

    cam_center = -(R_c2w @ tvec)
    C2W = np.eye(4, dtype=np.float64)
    C2W[:3, :3] = R_c2w
    C2W[:3,  3] = cam_center

    # create_point_cloud expects (H, W) depth and (3,3) K and (4,4) C2W
    all_pts = create_point_cloud(calibrated_depth.astype(np.float64), K.astype(np.float64), C2W)
    # all_pts: (H*W, 3)

    fg_flat = fg_mask.flatten().astype(bool)
    fg_pts  = all_pts[fg_flat]

    if len(fg_pts) < 10:
        raise RuntimeError(f"Too few foreground pixels: {len(fg_pts)}")

    lo = np.percentile(fg_pts, pct_lo, axis=0)
    hi = np.percentile(fg_pts, pct_hi, axis=0)

    centre = ((lo + hi) / 2.0).astype(np.float32)
    extent = (hi - lo).astype(np.float32)

    print(f"  Foreground 3D bounding box ({pct_lo}–{pct_hi}th pct):")
    print(f"    centre: [{centre[0]:.4f}, {centre[1]:.4f}, {centre[2]:.4f}]")
    print(f"    extent: [{extent[0]:.4f}, {extent[1]:.4f}, {extent[2]:.4f}]")

    return centre, extent


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    wan_video_path = os.path.join(args.output_path, "final_output.mp4")
    masks_dir      = os.path.join(args.output_path, "sam2_masks")
    out_json       = os.path.join(args.output_path, "gaussians", "placement.json")

    # --- Camera ---
    print("\n--- Loading COLMAP camera ---")
    R_c2w, tvec, FoVx, FoVy, cam_W, cam_H = load_colmap_camera(
        args.gs_scene_path, args.camera_idx
    )
    print(f"  FoVx={math.degrees(FoVx):.1f}°  FoVy={math.degrees(FoVy):.1f}°  {cam_W}×{cam_H}")

    # --- Background GS centres ---
    print("\n--- Loading background GS centres ---")
    ply_path = find_latest_ply(args.gs_model_path)
    print(f"  PLY: {ply_path}")
    xyz = load_gs_centers(ply_path)
    print(f"  {len(xyz)} Gaussians")

    # --- Wan video frame 0 ---
    print("\n--- Loading Wan video frame 0 ---")
    frames_dir = os.path.join(args.output_path, "frames")
    if os.path.exists(wan_video_path):
        reader = imageio.get_reader(wan_video_path)
        frame0 = next(iter(reader))
        reader.close()
    elif os.path.isdir(frames_dir):
        frame_files = sorted(
            glob.glob(os.path.join(frames_dir, "*.jpg")) +
            glob.glob(os.path.join(frames_dir, "*.png"))
        )
        if not frame_files:
            raise FileNotFoundError(f"No final_output.mp4 or frame files in {args.output_path}")
        frame0 = np.array(Image.open(frame_files[0]).convert("RGB"))
    else:
        raise FileNotFoundError(f"No final_output.mp4 or frames/ in {args.output_path}")
    wan_H, wan_W = frame0.shape[:2]
    print(f"  Frame: {wan_W}×{wan_H}")

    # --- SAM2 mask frame 0 ---
    print("\n--- Loading SAM2 mask frame 0 ---")
    mask_files = sorted(glob.glob(os.path.join(masks_dir, "*.png")))
    if not mask_files:
        raise FileNotFoundError(f"No mask PNGs in {masks_dir}")
    mask0  = np.array(Image.open(mask_files[0]).convert("L"))
    fg_mask = (mask0 > 128).astype(np.uint8)
    if fg_mask.shape != (wan_H, wan_W):
        fg_mask = np.array(
            Image.fromarray(fg_mask * 255).resize((wan_W, wan_H), Image.NEAREST)
        ) > 128
        fg_mask = fg_mask.astype(np.uint8)
    print(f"  Mask shape: {fg_mask.shape}  foreground px: {fg_mask.sum()}")

    # --- Project GS centres to get reference depths ---
    print(f"\n--- Projecting GS centres to camera ({wan_W}×{wan_H}) ---")
    u_proj, v_proj, z_proj = project_gs_to_camera(
        xyz, R_c2w, tvec, FoVx, FoVy, wan_W, wan_H
    )
    # Keep only background pixels
    bg_sel = fg_mask[v_proj, u_proj] == 0
    u_bg, v_bg, z_bg = u_proj[bg_sel], v_proj[bg_sel], z_proj[bg_sel]
    print(f"  Projected {len(u_proj)} visible centres, {len(u_bg)} in background")

    # --- Depth Anything V2 ---
    print(f"\n--- Running Depth Anything V2 ({args.depth_model}) ---")
    processor = AutoImageProcessor.from_pretrained(args.depth_model)
    da_model  = AutoModelForDepthEstimation.from_pretrained(args.depth_model).to(device)
    da_model.eval()
    mono_depth = run_depth_anything(frame0, processor, da_model, device)
    print(f"  Mono depth: min={mono_depth.min():.3f}  max={mono_depth.max():.3f}")

    # --- Calibrate ---
    print("\n--- Calibrating monocular depth to world scale ---")
    a, b = calibrate_depth(mono_depth, u_bg, v_bg, z_bg)
    calibrated_depth = (a * mono_depth + b).astype(np.float32)
    calibrated_depth = np.maximum(calibrated_depth, 0.01)

    fg_depths = calibrated_depth[fg_mask == 1]
    print(f"  Calibrated fg depth: median={np.median(fg_depths):.3f}  "
          f"range=[{np.percentile(fg_depths, 5):.3f}, {np.percentile(fg_depths, 95):.3f}]")

    # --- Estimate placement ---
    print("\n--- Estimating foreground 3D placement ---")
    centre, extent = estimate_placement(
        calibrated_depth, fg_mask, R_c2w, tvec, FoVx, FoVy, wan_W, wan_H
    )

    # Scale: 4DGS is normalised to ~1 world unit; use median extent across dimensions.
    # Median is robust to objects with spread-out features (e.g. tentacles, wings) where
    # one axis is inflated while the other two capture the characteristic body size.
    scale = float(np.median(extent))

    # --- Per-frame translations ---
    # Re-use the same depth calibration (a, b) — background is static — and run
    # Depth Anything V2 on every frame to get the world-space corgi centre for each.
    print("\n--- Computing per-frame translations ---")
    if os.path.exists(wan_video_path):
        reader = imageio.get_reader(wan_video_path)
        all_frames = [f for f in reader]
        reader.close()
    else:
        all_frames = [
            np.array(Image.open(p).convert("RGB"))
            for p in frame_files
        ]
    all_masks = [
        np.array(Image.open(p).convert("L")) for p in mask_files
    ]
    # Trim to the shorter of frames / masks (should be equal, but guard against mismatches)
    n_frames = min(len(all_frames), len(all_masks))
    print(f"  Processing {n_frames} frames …")

    per_frame_translations = []
    for i in range(n_frames):
        frame_i = all_frames[i]
        raw_mask = all_masks[i]
        fg_i = (raw_mask > 128).astype(np.uint8)
        if fg_i.shape != (wan_H, wan_W):
            fg_i = (np.array(
                Image.fromarray(fg_i * 255).resize((wan_W, wan_H), Image.NEAREST)
            ) > 128).astype(np.uint8)

        if fg_i.sum() < 10:
            # No foreground visible — keep previous translation
            prev = per_frame_translations[-1] if per_frame_translations else centre.tolist()
            per_frame_translations.append(prev)
            continue

        mono_i = run_depth_anything(frame_i, processor, da_model, device)
        calib_i = np.maximum(a * mono_i + b, 0.01).astype(np.float32)
        centre_i, _ = estimate_placement(
            calib_i, fg_i, R_c2w, tvec, FoVx, FoVy, wan_W, wan_H
        )
        per_frame_translations.append(centre_i.tolist())

        if i % 10 == 0:
            print(f"  Frame {i}/{n_frames}  centre={np.round(centre_i, 3).tolist()}")

    print(f"  Done — {len(per_frame_translations)} frame translations computed.")

    placement = {
        "translation": centre.tolist(),
        "scale":       scale,
        "extent":      extent.tolist(),
        "calibration": {"a": a, "b": b},
        "camera_idx":  args.camera_idx,
        "per_frame_translations": per_frame_translations,
        "notes": (
            "translation = world-space centre of foreground bounding box (frame 0). "
            "per_frame_translations = per-frame 3D centre from depth estimation. "
            "scale = max bbox extent (4DGS normalised to ~1 unit). "
            "Apply to 4DGS: p_world = R @ (scale * p_local) + translation_t."
        ),
    }

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(placement, f, indent=2)

    print(f"\n--- Done ---")
    print(f"  translation : {centre.tolist()}")
    print(f"  scale       : {scale:.4f}")
    print(f"  Saved       : {out_json}")


if __name__ == "__main__":
    main()
