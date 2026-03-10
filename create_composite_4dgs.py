"""
create_composite_4dgs.py — Build composite 4DGS by placing the foreground in world space.

Loads placement.json (from estimate_fg_placement.py), applies the rotation/scale/translation
transform to the foreground 4DGS canonical positions and deformation offsets, and saves the
per-frame world-space foreground positions as fg_positions_world.npy.

The composite 4DGS is fully described by:
  - gaussians/fg_positions_world.npy  — (T, N_fg, 3) per-frame world-space positions
  - gaussians/gaussians.ply           — foreground Gaussian attributes (colours, opacities,
                                        scales, rotations)
  - <gs_model_path>/iteration_N/point_cloud.ply — background 3DGS (unchanged)

No rendering is performed here. Use render_composite_4dgs.py to render from any camera.

Usage:
  python create_composite_4dgs.py \\
      --output_path output/2026.03.02/actionmesh_gs_replace_corgi \\
      --gs_scene_path Inpaint360GS/data/inpaint360/bag \\
      --camera_idx 28 [--yaw_deg 0]
"""

import sys
import os
import glob
import math
import json

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
import numpy as np
from plyfile import PlyData

from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary, qvec2rotmat


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", required=True,
                        help="ram4d output dir (contains gaussians/ with placement.json)")
    parser.add_argument("--gs_scene_path", default=None,
                        help="Inpaint360GS scene dir (contains sparse/0/) — required unless "
                             "--placement_path JSON contains a 'rotation' key")
    parser.add_argument("--placement_path", default=None,
                        help="Override placement JSON (e.g. placement_refined.json from "
                             "refine_frame0.py). Defaults to <output_path>/gaussians/placement.json. "
                             "If the JSON contains a 'rotation' key (3×3 matrix), that rotation "
                             "is used directly and --gs_scene_path is not needed.")
    parser.add_argument("--camera_idx", type=int, default=28,
                        help="Camera the foreground faces toward (0-based, sorted by image name)")
    parser.add_argument("--yaw_deg", type=float, default=0.0,
                        help="Extra yaw rotation around world_up (degrees, default 0)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Camera loading
# ---------------------------------------------------------------------------

def load_colmap_cameras(scene_path, camera_idx):
    """Load world_up (from all cameras) and camera_idx centre (for fg orientation)."""
    images_bin  = os.path.join(scene_path, "sparse", "0", "images.bin")
    cameras_bin = os.path.join(scene_path, "sparse", "0", "cameras.bin")
    extrinsics = read_extrinsics_binary(images_bin)
    intrinsics = read_intrinsics_binary(cameras_bin)
    sorted_images = sorted(extrinsics.values(), key=lambda x: x.name)

    if camera_idx >= len(sorted_images):
        raise ValueError(f"camera_idx {camera_idx} out of range ({len(sorted_images)} cameras)")

    # World up: mean of -cam_down across all cameras
    downs = np.stack([qvec2rotmat(si.qvec).T[:, 1] for si in sorted_images])
    mean_down = downs.mean(axis=0)
    world_up = -mean_down / np.linalg.norm(mean_down)
    print(f"  World up (from {len(sorted_images)} cameras): {np.round(world_up, 4)}")

    img = sorted_images[camera_idx]
    cam = intrinsics[img.camera_id]
    print(f"  Camera {camera_idx}: '{img.name}'  COLMAP id={img.id}")
    R_w2c = qvec2rotmat(img.qvec)
    tvec  = np.array(img.tvec, dtype=np.float64)
    R_c2w = R_w2c.T
    cam_center = -(R_c2w @ tvec).astype(np.float32)

    return world_up, cam_center


# ---------------------------------------------------------------------------
# Foreground rotation
# ---------------------------------------------------------------------------

def compute_fg_rotation(translation, cam_center, world_up, yaw_deg=0.0):
    """
    Build rotation R such that:
      - 4DGS local Z ([0,0,1], up) → world_up
      - 4DGS local -X (corgi front) → toward camera_idx (projected onto horizontal plane)
    Plus an optional yaw around world_up.

    Returns R (3,3) float32.
    """
    world_up = world_up / np.linalg.norm(world_up)

    to_cam = np.array(cam_center, dtype=np.float64) - np.array(translation, dtype=np.float64)
    to_cam_h = to_cam - np.dot(to_cam, world_up) * world_up
    if np.linalg.norm(to_cam_h) < 1e-6:
        to_cam_h = np.array([1.0, 0.0, 0.0])
    world_fwd = to_cam_h / np.linalg.norm(to_cam_h)

    if abs(yaw_deg) > 1e-6:
        angle = math.radians(yaw_deg)
        k = world_up
        c, s = math.cos(angle), math.sin(angle)
        world_fwd = (world_fwd * c
                     + np.cross(k, world_fwd) * s
                     + k * np.dot(k, world_fwd) * (1 - c))
        world_fwd /= np.linalg.norm(world_fwd)

    world_right = np.cross(world_fwd, world_up)
    world_right /= np.linalg.norm(world_right)

    # R columns: local +X → -world_fwd,  local Y → world_right,  local Z → world_up
    R = np.stack([-world_fwd, world_right, world_up], axis=1).astype(np.float32)
    print(f"  Rotation det={np.linalg.det(R):.4f}")
    print(f"  world_fwd : {np.round(world_fwd, 4)}")
    print(f"  world_right: {np.round(world_right, 4)}")
    print(f"  world_up  : {np.round(world_up, 4)}")
    return R


# ---------------------------------------------------------------------------
# PLY loading (positions only)
# ---------------------------------------------------------------------------

def load_fg_positions(ply_path):
    v = PlyData.read(ply_path).elements[0]
    return np.stack([v['x'], v['y'], v['z']], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    gaussians_dir  = os.path.join(args.output_path, "gaussians")
    placement_path = (args.placement_path
                      if args.placement_path is not None
                      else os.path.join(gaussians_dir, "placement.json"))
    fg_ply_path    = os.path.join(gaussians_dir, "gaussians.ply")
    offsets_path   = os.path.join(gaussians_dir, "deformation_offsets.npy")
    out_path       = os.path.join(gaussians_dir, "fg_positions_world.npy")

    # --- Placement ---
    print(f"\n--- Loading placement: {placement_path} ---")
    with open(placement_path) as f:
        placement = json.load(f)
    translation = np.array(placement["translation"], dtype=np.float32)
    scale       = float(placement["scale"])
    per_frame_translations = placement.get("per_frame_translations")
    print(f"  translation: {translation.tolist()}")
    print(f"  scale:       {scale:.4f}")
    if per_frame_translations:
        print(f"  per_frame_translations: {len(per_frame_translations)} frames")

    # --- Foreground rotation ---
    # Use rotation from placement JSON if present (e.g. from refine_frame0.py),
    # otherwise compute from COLMAP cameras (requires --gs_scene_path).
    rotation_from_json = placement.get("rotation")
    if rotation_from_json is not None:
        print("\n--- Using rotation from placement JSON ---")
        R_fg = np.array(rotation_from_json, dtype=np.float32)
        print(f"  R_fg det={np.linalg.det(R_fg):.4f}")
    else:
        if args.gs_scene_path is None:
            raise ValueError(
                "--gs_scene_path is required when placement JSON has no 'rotation' key"
            )
        print("\n--- Loading COLMAP cameras ---")
        world_up, cam_center = load_colmap_cameras(args.gs_scene_path, args.camera_idx)
        print("\n--- Computing foreground orientation ---")
        R_fg = compute_fg_rotation(translation, cam_center, world_up, args.yaw_deg)
        # Write R_fg back into placement.json so downstream scripts (refine_deform.py) can read it
        placement["rotation"] = R_fg.tolist()
        with open(placement_path, "w") as f:
            json.dump(placement, f, indent=2)
        print(f"  Saved rotation to {placement_path}")

    # --- Foreground 4DGS ---
    print("\n--- Loading foreground 4DGS ---")
    xyz_fg  = load_fg_positions(fg_ply_path)
    offsets = np.load(offsets_path) if os.path.exists(offsets_path) else None
    T = offsets.shape[0] if offsets is not None else 1
    print(f"  {len(xyz_fg):,} Gaussians  |  {T} frames")

    # --- Compute per-frame world-space positions ---
    print(f"\n--- Computing {T} frames of world-space positions ---")
    xyz_fg_rot = (R_fg @ (xyz_fg * scale).T).T   # (N_fg, 3) — rotated + scaled, origin centred
    translation_const = translation

    fg_positions_world = np.zeros((T, len(xyz_fg), 3), dtype=np.float32)
    for t in range(T):
        if per_frame_translations is not None and t < len(per_frame_translations):
            trans_t = np.array(per_frame_translations[t], dtype=np.float32)
        else:
            trans_t = translation_const

        if offsets is not None:
            off_world = (R_fg @ (offsets[t] * scale).T).T
            fg_positions_world[t] = xyz_fg_rot + off_world + trans_t
        else:
            fg_positions_world[t] = xyz_fg_rot + trans_t

        if t % 20 == 0:
            print(f"  Frame {t}/{T}")

    # --- Save ---
    np.save(out_path, fg_positions_world)
    print(f"\n--- Done ---")
    print(f"  Saved: {out_path}")
    print(f"  Shape: {fg_positions_world.shape}  dtype={fg_positions_world.dtype}")
    print(f"\nComposite 4DGS:")
    print(f"  fg_positions_world.npy  — per-frame world-space positions")
    print(f"  gaussians.ply           — foreground attributes (colours, opacities, scales, rotations)")
    print(f"  <gs_model_path>/...     — background 3DGS (unchanged)")


if __name__ == "__main__":
    main()
