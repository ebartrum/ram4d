"""
appearance_refinement.py — Render inputs for Gaussian appearance refinement.

Produces:
  - <output_dir>/<prefix>_input_<camera_idx>.png   : composite RGB render (bg + fg)
  - <output_dir>/<prefix>_mask_<camera_idx>.png    : fg silhouette mask, dilated

Usage:
  python appearance_refinement.py \\
      --output_path output/2026.03.03/actionmesh_gs_replace_corgi \\
      --gs_scene_path Inpaint360GS/data/inpaint360/bag \\
      --gs_model_path output/2026.02.26/inpainted_scene/point_cloud_object_inpaint_virtual \\
      --placement_path output/2026.03.09/corgi_refined_consistent_occl/placement_refined.json \\
      --render_output_dir output/2026.03.09/corgi_appearance_refinement \\
      --camera_idx 20
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

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, "Inpaint360GS"))

import argparse
import json
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_dilation
from plyfile import PlyData

from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary, qvec2rotmat
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

C0 = 0.28209479177387814


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", required=True,
                        help="Composite dir (contains gaussians/fg_positions_world.npy)")
    parser.add_argument("--gs_scene_path", required=True,
                        help="Inpaint360GS scene dir (contains sparse/0/)")
    parser.add_argument("--gs_model_path", required=True,
                        help="Inpaint360GS model dir (contains iteration_N/point_cloud.ply)")
    parser.add_argument("--placement_path", default=None,
                        help="Path to placement_refined.json (default: gaussians/placement.json)")
    parser.add_argument("--fg_positions_path", default=None,
                        help="Override path to fg_positions_world.npy")
    parser.add_argument("--camera_idx", type=int, default=20,
                        help="COLMAP training camera index to render from (default: 20)")
    parser.add_argument("--render_output_dir", default=None,
                        help="Directory to save outputs (default: <output_path>/gaussians/)")
    parser.add_argument("--output_prefix", default="appearance_refinement",
                        help="Filename prefix (default: appearance_refinement)")
    parser.add_argument("--mask_dilation", type=int, default=3,
                        help="Mask dilation radius in pixels (default: 3)")
    parser.add_argument("--render_scale", type=float, default=1.0,
                        help="Scale factor applied to camera resolution (default: 1.0)")
    # Flux refinement
    parser.add_argument("--prompt_path", required=True,
                        help="Path to prompt .txt file for Flux inpainting refinement")
    parser.add_argument("--flux_strength", type=float, default=0.3,
                        help="Flux inpainting strength (0=no change, 1=full redraw, default: 0.3)")
    parser.add_argument("--flux_steps", type=int, default=28,
                        help="Flux number of inference steps (default: 28)")
    parser.add_argument("--flux_guidance", type=float, default=3.5,
                        help="Flux guidance scale (default: 3.5)")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Camera loading
# ---------------------------------------------------------------------------

def load_colmap_camera(scene_path, camera_idx):
    images_bin  = os.path.join(scene_path, "sparse", "0", "images.bin")
    cameras_bin = os.path.join(scene_path, "sparse", "0", "cameras.bin")
    extrinsics = read_extrinsics_binary(images_bin)
    intrinsics = read_intrinsics_binary(cameras_bin)
    sorted_images = sorted(extrinsics.values(), key=lambda x: x.name)

    if camera_idx >= len(sorted_images):
        raise ValueError(f"camera_idx {camera_idx} out of range ({len(sorted_images)} cameras)")

    img = sorted_images[camera_idx]
    cam = intrinsics[img.camera_id]
    print(f"  Camera {camera_idx}: '{img.name}'  COLMAP id={img.id}  model={cam.model}")

    R_w2c = qvec2rotmat(img.qvec)
    tvec  = np.array(img.tvec, dtype=np.float64)
    W, H  = int(cam.width), int(cam.height)

    if cam.model in ("PINHOLE", "OPENCV"):
        fx, fy = cam.params[0], cam.params[1]
    elif cam.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
        fx = fy = cam.params[0]
    else:
        raise ValueError(f"Unsupported COLMAP camera model: {cam.model}")

    FoVx = 2.0 * math.atan(W / (2.0 * fx))
    FoVy = 2.0 * math.atan(H / (2.0 * fy))

    pose_w2c = np.eye(4, dtype=np.float64)
    pose_w2c[:3, :3] = R_w2c
    pose_w2c[:3,  3] = tvec
    return pose_w2c, FoVx, FoVy, W, H


# ---------------------------------------------------------------------------
# PLY loading
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


def load_ply_gs(ply_path):
    v = PlyData.read(ply_path).elements[0]
    xyz     = np.stack([v['x'],       v['y'],       v['z']      ], axis=1).astype(np.float32)
    f_dc    = np.stack([v['f_dc_0'],  v['f_dc_1'],  v['f_dc_2'] ], axis=1).astype(np.float32)
    opacity = v['opacity'].astype(np.float32)
    log_sc  = np.stack([v['scale_0'], v['scale_1'], v['scale_2']], axis=1).astype(np.float32)
    rot     = np.stack([v['rot_0'],   v['rot_1'],   v['rot_2'],   v['rot_3']], axis=1).astype(np.float32)
    return xyz, f_dc, opacity, log_sc, rot


# ---------------------------------------------------------------------------
# Camera matrices
# ---------------------------------------------------------------------------

def make_raster_camera(pose_w2c, FoVx, FoVy, W, H, device, znear=0.01, zfar=200.0):
    R_w2c = pose_w2c[:3, :3].astype(np.float32)
    t     = pose_w2c[:3, 3].astype(np.float32)
    R_c2w = R_w2c.T

    W2C = np.eye(4, dtype=np.float32)
    W2C[:3, :3] = R_w2c
    W2C[:3,  3] = t

    tanfovx = math.tan(FoVx / 2.0)
    tanfovy = math.tan(FoVy / 2.0)

    P = np.zeros((4, 4), dtype=np.float32)
    P[0, 0] =  1.0 / tanfovx
    P[1, 1] =  1.0 / tanfovy
    P[3, 2] =  1.0
    P[2, 2] =  zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)

    viewmatrix = torch.from_numpy(W2C.T).float().to(device)
    proj_t     = torch.from_numpy(P.T  ).float().to(device)
    full_proj  = (viewmatrix.unsqueeze(0).bmm(proj_t.unsqueeze(0))).squeeze(0)
    campos     = torch.tensor(-(R_c2w @ t), dtype=torch.float32, device=device)
    return viewmatrix, full_proj, campos, tanfovx, tanfovy


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_frame(means3D, colors, opacities, scales, rotations,
                 viewmatrix, full_proj, campos, tanfovx, tanfovy, W, H, device, bg_color=None):
    N = means3D.shape[0]
    means2D = torch.zeros(N, 3, device=device, dtype=torch.float32)
    means2D.requires_grad_(True)

    raster_settings = GaussianRasterizationSettings(
        image_height=H,
        image_width=W,
        tanfovx=float(tanfovx),
        tanfovy=float(tanfovy),
        bg=bg_color if bg_color is not None else torch.zeros(3, device=device, dtype=torch.float32),
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
    rendered, _, _ = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=None,
        colors_precomp=colors,
        opacities=opacities,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None,
    )
    return rendered.clamp(0.0, 1.0).permute(1, 2, 0).detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    gaussians_dir    = os.path.join(args.output_path, "gaussians")
    fg_positions_path = args.fg_positions_path or os.path.join(gaussians_dir, "fg_positions_world.npy")
    fg_ply_path      = os.path.join(gaussians_dir, "gaussians.ply")
    placement_path   = args.placement_path or os.path.join(gaussians_dir, "placement.json")

    render_dir = args.render_output_dir or gaussians_dir
    os.makedirs(render_dir, exist_ok=True)

    out_input = os.path.join(render_dir, f"{args.output_prefix}_input_{args.camera_idx}.png")
    out_mask  = os.path.join(render_dir, f"{args.output_prefix}_mask_{args.camera_idx}.png")

    # --- Camera ---
    print("\n--- Loading camera ---")
    pose_w2c, FoVx, FoVy, cam_W, cam_H = load_colmap_camera(args.gs_scene_path, args.camera_idx)
    W = max(1, int(cam_W * args.render_scale))
    H = max(1, int(cam_H * args.render_scale))
    print(f"  Native {cam_W}×{cam_H} → render {W}×{H}  (scale={args.render_scale})")

    viewmat, full_proj, campos, tanfovx, tanfovy = make_raster_camera(
        pose_w2c, FoVx, FoVy, W, H, device
    )

    # --- Foreground ---
    print("\n--- Loading foreground Gaussians ---")
    fg_positions_world = np.load(fg_positions_path)   # (T, N_fg, 3)
    _, f_dc_fg, op_fg, log_sc_fg, rot_fg = load_ply_gs(fg_ply_path)
    with open(placement_path) as f:
        scale = float(json.load(f)["scale"])
    scales_fg = np.exp(log_sc_fg) * scale
    N_fg = fg_positions_world.shape[1]
    print(f"  {N_fg:,} Gaussians  scale={scale:.4f}")

    fg_pos_t  = torch.from_numpy(fg_positions_world[0]).float().to(device)
    rgb_fg_t  = torch.from_numpy(np.clip(f_dc_fg * C0 + 0.5, 0.0, 1.0)).float().to(device)
    alpha_fg_t = torch.from_numpy(1.0 / (1.0 + np.exp(-op_fg))).float().to(device).unsqueeze(1)
    scales_fg_t = torch.from_numpy(scales_fg).float().to(device)
    rot_fg_t  = torch.from_numpy(rot_fg).float().to(device)

    # --- Background ---
    print("\n--- Loading background 3DGS ---")
    bg_ply = find_latest_ply(args.gs_model_path)
    print(f"  PLY: {bg_ply}")
    xyz_bg, f_dc_bg, op_bg, log_sc_bg, rot_bg = load_ply_gs(bg_ply)
    print(f"  {len(xyz_bg):,} Gaussians")

    f_dc_all   = np.concatenate([f_dc_bg,  f_dc_fg ], axis=0)
    op_all     = np.concatenate([op_bg,    op_fg   ], axis=0)
    rot_all    = np.concatenate([rot_bg,   rot_fg  ], axis=0)
    scales_all = np.concatenate([np.exp(log_sc_bg), scales_fg], axis=0)

    rgb_all_t    = torch.from_numpy(np.clip(f_dc_all * C0 + 0.5, 0.0, 1.0)).float().to(device)
    alpha_all_t  = torch.from_numpy(1.0 / (1.0 + np.exp(-op_all))).float().to(device).unsqueeze(1)
    scales_all_t = torch.from_numpy(scales_all).float().to(device)
    rot_all_t    = torch.from_numpy(rot_all).float().to(device)
    xyz_bg_t     = torch.from_numpy(xyz_bg).float().to(device)
    means3D_all  = torch.cat([xyz_bg_t, fg_pos_t], dim=0)

    print(f"  Total: {means3D_all.shape[0]:,}  (bg={len(xyz_bg):,}  fg={N_fg:,})")

    # --- Render composite RGB ---
    print(f"\n--- Rendering composite RGB → {out_input} ---")
    img = render_frame(
        means3D_all, rgb_all_t, alpha_all_t, scales_all_t, rot_all_t,
        viewmat, full_proj, campos, tanfovx, tanfovy, W, H, device,
    )
    Image.fromarray((img * 255).astype(np.uint8)).save(out_input)
    print(f"  Saved: {out_input}")

    # --- Render fg silhouette mask ---
    print(f"\n--- Rendering fg mask → {out_mask} ---")
    white = torch.ones(3, device=device, dtype=torch.float32)
    white_fg_t = torch.ones_like(rgb_fg_t)
    mask_img = render_frame(
        fg_pos_t, white_fg_t, alpha_fg_t, scales_fg_t, rot_fg_t,
        viewmat, full_proj, campos, tanfovx, tanfovy, W, H, device,
        bg_color=torch.zeros(3, device=device, dtype=torch.float32),
    )
    # Binarise (threshold at 0.5) then dilate
    mask_bin = mask_img.mean(axis=2) > 0.5
    if args.mask_dilation > 0:
        struct = np.ones((args.mask_dilation * 2 + 1, args.mask_dilation * 2 + 1), dtype=bool)
        mask_bin = binary_dilation(mask_bin, structure=struct)
    mask_out = (mask_bin * 255).astype(np.uint8)
    mask_pil = Image.fromarray(mask_out, mode="L")
    mask_pil.save(out_mask)
    print(f"  Saved: {out_mask}  (dilation={args.mask_dilation}px)")

    # --- Flux inpainting refinement ---
    from diffusers import FluxInpaintPipeline

    with open(args.prompt_path) as f:
        prompt = f.read().strip()
    print(f"\n--- Flux inpainting refinement (strength={args.flux_strength}) ---")
    print(f"  Prompt: {prompt}")

    out_refined = os.path.join(render_dir, f"{args.output_prefix}_refined_{args.camera_idx}.png")

    # Free GS GPU memory before loading Flux
    del means3D_all, rgb_all_t, alpha_all_t, scales_all_t, rot_all_t
    del fg_pos_t, rgb_fg_t, alpha_fg_t, scales_fg_t, rot_fg_t
    del xyz_bg_t
    torch.cuda.empty_cache()

    input_pil = Image.open(out_input).convert("RGB")

    # Flux requires dimensions divisible by 16
    W_flux = (input_pil.width  // 16) * 16
    H_flux = (input_pil.height // 16) * 16
    if W_flux != input_pil.width or H_flux != input_pil.height:
        print(f"  Resizing {input_pil.width}×{input_pil.height} → {W_flux}×{H_flux} for Flux")
        input_pil = input_pil.resize((W_flux, H_flux), Image.LANCZOS)
        mask_pil  = mask_pil.resize((W_flux, H_flux), Image.NEAREST)

    pipe = FluxInpaintPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16
    )
    pipe.enable_model_cpu_offload()

    result = pipe(
        prompt=prompt,
        image=input_pil,
        mask_image=mask_pil,
        width=W_flux,
        height=H_flux,
        strength=args.flux_strength,
        num_inference_steps=args.flux_steps,
        guidance_scale=args.flux_guidance,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(args.seed),
    ).images[0]

    # Resize back to original render resolution if needed
    if result.width != W or result.height != H:
        result = result.resize((W, H), Image.LANCZOS)

    result.save(out_refined)
    print(f"  Saved: {out_refined}")


if __name__ == "__main__":
    main()
