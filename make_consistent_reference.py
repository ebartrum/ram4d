"""
make_consistent_reference.py — Create a background-consistent reference image.

Takes an existing reference frame (e.g. frame 0 from the Wan output video),
composites the foreground subject onto a rendered background from the 3DGS,
then runs a low-strength Flux inpainting pass to harmonise the boundary seam.

Steps:
  1. Render background 3DGS at reference frame resolution from camera_idx
     (uses make_raster_camera_at_resolution — keeps fovx, square pixels)
  2. Run LangSAM on the reference frame to get a tight foreground mask
  3. Composite: fg pixels from reference, bg pixels from 3DGS render
  4. Flux inpainting on background + boundary zone (inverted, dilated fg mask)
     with low strength to harmonise the seam without changing the interior

Usage:
  python make_consistent_reference.py \\
      --reference_frame output/2026.03.03/actionmesh_gs_replace_corgi/frames/00000.jpg \\
      --gs_scene_path Inpaint360GS/data/inpaint360/bag \\
      --gs_model_path output/2026.02.26/inpainted_scene/point_cloud_object_inpaint_virtual \\
      --seg_prompt_path data/captions/corgi_segmentation.txt \\
      --prompt_path data/captions/corgi.txt \\
      --output_path output/2026.03.06/corgi_consistent_ref.png
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
import numpy as np
import torch
from PIL import Image, ImageFilter
from plyfile import PlyData

from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary, qvec2rotmat
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from camera_utils import make_raster_camera_at_resolution
from pipeline_utils import run_langsam

C0 = 0.28209479177387814


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference_frame", required=True,
                        help="Existing reference image (e.g. frames/00000.jpg from Wan output)")
    parser.add_argument("--gs_scene_path", required=True,
                        help="Inpaint360GS scene dir (contains sparse/0/)")
    parser.add_argument("--gs_model_path", required=True,
                        help="Background 3DGS model dir (contains iteration_N/point_cloud.ply)")
    parser.add_argument("--seg_prompt_path", required=True,
                        help="LangSAM text prompt file for tight fg segmentation")
    parser.add_argument("--prompt_path", required=True,
                        help="Text prompt file for Flux background inpainting")
    parser.add_argument("--output_path", required=True,
                        help="Output path for the consistent reference image (.png)")
    parser.add_argument("--camera_idx", type=int, default=28,
                        help="COLMAP camera index for the background render (default 28)")
    parser.add_argument("--flux_strength", type=float, default=0.35,
                        help="Flux inpainting strength — lower preserves more (default 0.35)")
    parser.add_argument("--mask_dilation", type=int, default=10,
                        help="Pixels to dilate the fg mask outward to include the seam zone (default 10)")
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--num_inference_steps", type=int, default=28)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# PLY / COLMAP helpers
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


def load_colmap_camera(scene_path, camera_idx):
    """Return (pose_w2c 4×4 float64, FoVx float) for the given camera index."""
    images_bin  = os.path.join(scene_path, "sparse", "0", "images.bin")
    cameras_bin = os.path.join(scene_path, "sparse", "0", "cameras.bin")
    extrinsics  = read_extrinsics_binary(images_bin)
    intrinsics  = read_intrinsics_binary(cameras_bin)
    sorted_imgs = sorted(extrinsics.values(), key=lambda x: x.name)

    if camera_idx >= len(sorted_imgs):
        raise ValueError(f"camera_idx {camera_idx} out of range ({len(sorted_imgs)} cameras)")

    img = sorted_imgs[camera_idx]
    cam = intrinsics[img.camera_id]
    print(f"  Camera {camera_idx}: '{img.name}'  model={cam.model}  "
          f"native {cam.width}×{cam.height}")

    if cam.model in ("PINHOLE", "OPENCV"):
        fx = cam.params[0]
    elif cam.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
        fx = cam.params[0]
    else:
        raise ValueError(f"Unsupported COLMAP camera model: {cam.model}")

    FoVx = 2.0 * math.atan(int(cam.width) / (2.0 * fx))

    pose_w2c = np.eye(4, dtype=np.float64)
    pose_w2c[:3, :3] = qvec2rotmat(img.qvec)
    pose_w2c[:3,  3] = np.array(img.tvec, dtype=np.float64)
    return pose_w2c, FoVx


# ---------------------------------------------------------------------------
# Background render
# ---------------------------------------------------------------------------

def render_bg(xyz, f_dc, opacity, log_sc, rot,
              pose_w2c, fovx, W, H, device):
    """Render background 3DGS at W×H from pose_w2c, returns (H, W, 3) uint8."""
    N = len(xyz)
    viewmat, full_proj, campos, tanfovx, tanfovy = make_raster_camera_at_resolution(
        pose_w2c, fovx, W, H, device
    )

    rgb    = torch.from_numpy(np.clip(f_dc * C0 + 0.5, 0.0, 1.0)).float().to(device)
    alpha  = torch.from_numpy(1.0 / (1.0 + np.exp(-opacity))).float().to(device).unsqueeze(1)
    scales = torch.from_numpy(np.exp(log_sc)).float().to(device)
    rots   = torch.from_numpy(rot).float().to(device)
    xyz_t  = torch.from_numpy(xyz).float().to(device)

    means2D = torch.zeros(N, 3, device=device, dtype=torch.float32)
    raster_settings = GaussianRasterizationSettings(
        image_height=H, image_width=W,
        tanfovx=float(tanfovx), tanfovy=float(tanfovy),
        bg=torch.zeros(3, device=device, dtype=torch.float32),
        scale_modifier=1.0,
        viewmatrix=viewmat, projmatrix=full_proj,
        sh_degree=0, campos=campos,
        prefiltered=False, debug=False, antialiasing=False,
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    rendered, _, _ = rasterizer(
        means3D=xyz_t, means2D=means2D, shs=None,
        colors_precomp=rgb, opacities=alpha,
        scales=scales, rotations=rots, cov3D_precomp=None,
    )
    img = rendered.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return (img * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    debug_dir = os.path.join(os.path.dirname(os.path.abspath(args.output_path)), "debug_consistent_ref")
    os.makedirs(debug_dir, exist_ok=True)

    # --- Reference frame ---
    print(f"\n--- Loading reference frame: {args.reference_frame} ---")
    ref_pil = Image.open(args.reference_frame).convert("RGB")
    W, H = ref_pil.size
    print(f"  Size: {W}×{H}")

    # --- COLMAP camera ---
    print(f"\n--- Loading COLMAP camera {args.camera_idx} ---")
    pose_w2c, fovx = load_colmap_camera(args.gs_scene_path, args.camera_idx)

    # --- Background 3DGS render ---
    print("\n--- Loading background 3DGS ---")
    bg_ply = find_latest_ply(args.gs_model_path)
    print(f"  PLY: {bg_ply}")
    xyz, f_dc, opacity, log_sc, rot = load_ply_gs(bg_ply)
    print(f"  {len(xyz):,} Gaussians")

    print(f"\n--- Rendering background at {W}×{H} ---")
    bg_render_np = render_bg(xyz, f_dc, opacity, log_sc, rot,
                             pose_w2c, fovx, W, H, device)
    bg_render_pil = Image.fromarray(bg_render_np)
    bg_render_pil.save(os.path.join(debug_dir, "bg_render.png"))
    print(f"  Saved bg_render.png")

    # --- LangSAM segmentation (after render — LangSAM leaves bfloat16 autocast active) ---
    print("\n--- Running LangSAM ---")
    fg_mask_np = run_langsam(ref_pil, args.seg_prompt_path)   # (H, W) uint8, 0/255
    fg_mask_pil = Image.fromarray(fg_mask_np)
    fg_mask_pil.save(os.path.join(debug_dir, "fg_mask.png"))
    print(f"  Saved fg_mask.png")

    # --- Flux mask: fg + seam zone (white=change, black=keep background) ---
    # Dilate the fg mask outward to include the seam boundary zone.
    # The 3DGS background pixels are kept exactly; Flux re-renders fg + boundary.
    flux_mask_pil = Image.fromarray(fg_mask_np)
    if args.mask_dilation > 0:
        flux_mask_pil = flux_mask_pil.filter(
            ImageFilter.MaxFilter(2 * args.mask_dilation + 1)
        )
    flux_mask_pil.save(os.path.join(debug_dir, "flux_mask.png"))
    print(f"  Flux mask (fg + {args.mask_dilation}px seam) saved")

    # --- Composite: fg from reference, bg from 3DGS render ---
    print("\n--- Compositing ---")
    ref_np      = np.array(ref_pil)
    fg_mask_bin = (fg_mask_np > 127)[..., np.newaxis]          # (H, W, 1) bool
    composite_np = np.where(fg_mask_bin, ref_np, bg_render_np).astype(np.uint8)
    composite_pil = Image.fromarray(composite_np)
    composite_pil.save(os.path.join(debug_dir, "composite.png"))
    print(f"  Saved composite.png")

    # --- Flux inpainting ---
    print(f"\n--- Flux inpainting (strength={args.flux_strength}) ---")
    import torch as _torch
    from diffusers import FluxInpaintPipeline

    with open(args.prompt_path) as f:
        prompt = f.read().strip()
    print(f"  Prompt: '{prompt}'")

    pipe = FluxInpaintPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev", torch_dtype=_torch.bfloat16
    )
    pipe.enable_model_cpu_offload()

    result = pipe(
        prompt=prompt,
        image=composite_pil,
        mask_image=flux_mask_pil,
        height=H,
        width=W,
        strength=args.flux_strength,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        max_sequence_length=512,
        generator=_torch.Generator("cpu").manual_seed(args.seed),
    ).images[0]

    result.save(args.output_path)
    print(f"\n--- Done ---")
    print(f"  Output: {args.output_path}")
    print(f"  Debug:  {debug_dir}/")


if __name__ == "__main__":
    main()
