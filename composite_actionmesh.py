import sys
import os
import glob

# Fix static TLS block issues (libgomp + libGLdispatch for open3d) by preloading
# before the dynamic linker can lock in the TLS layout. Re-exec to apply.
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

# Add ActionMesh and its TripoSG dependency to path (use __file__ so paths
# work regardless of the working directory when the script is invoked)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, "actionmesh"))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, "actionmesh", "third_party", "TripoSG"))

import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import imageio

from visualise_output import apply_mesh_transforms
from mvadapter.utils.mesh_utils import load_mesh, get_orthogonal_camera, render
from mvadapter.pipelines.pipeline_texture import TexturePipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reverse-localise and composite dynamic mesh render onto original background."
    )
    parser.add_argument("--output_path", required=True,
                        help="ActionMesh output dir (from run_actionmesh.py).")
    parser.add_argument("--fg_anim_dir", required=True,
                        help="fg_anim output dir (has args.txt with bg_image, width, height).")
    parser.add_argument("--bg_image", default=None,
                        help="Override background image path.")
    parser.add_argument("--fps", type=int, default=16,
                        help="Output video fps.")
    parser.add_argument("--render_size", type=int, default=768,
                        help="Mesh render resolution (square).")
    return parser.parse_args()


def read_args_txt(path):
    result = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or ": " not in line:
                continue
            key, _, value = line.partition(": ")
            result[key.strip()] = value.strip()
    return result


def main():
    args = parse_args()
    output_path = args.output_path
    texture_dir = os.path.join(output_path, "texture")
    bboxes_path = os.path.join(output_path, "localised_frames", "bounding_boxes.json")
    deformations_path = os.path.join(output_path, "deformations_vertices.npy")

    # Load fg_anim args.txt for background image and resolution
    fg_args_path = os.path.join(args.fg_anim_dir, "args.txt")
    fg_args = read_args_txt(fg_args_path)
    bg_image_path = args.bg_image or fg_args.get("bg_image")
    if not bg_image_path:
        raise ValueError("No bg_image: provide --bg_image or ensure args.txt contains bg_image.")
    # Resolve relative paths against the repo root (script dir)
    if not os.path.isabs(bg_image_path):
        bg_image_path = os.path.join(_SCRIPT_DIR, bg_image_path)
    W = int(fg_args.get("width", 832))
    H = int(fg_args.get("height", 480))
    print(f"Background: {bg_image_path} -> {W}x{H}")

    # Load background
    background = Image.open(bg_image_path).convert("RGB").resize((W, H), Image.LANCZOS)
    background_np = np.array(background)

    # Load bounding boxes
    with open(bboxes_path) as f:
        bboxes = json.load(f)

    device = "cuda"

    # Step 1: Set up rasterizer context via TexturePipeline
    print("\n--- Step 1: Initialising rasterizer context ---")
    texture_pipe = TexturePipeline(
        upscaler_ckpt_path=os.path.join(_SCRIPT_DIR, "checkpoints", "RealESRGAN_x2plus.pth"),
        inpaint_ckpt_path=os.path.join(_SCRIPT_DIR, "checkpoints", "big-lama.pt"),
        device=device,
    )
    ctx = texture_pipe.ctx

    # Step 2: Load textured mesh
    print("\n--- Step 2: Loading textured mesh ---")
    textured_mesh_path = os.path.join(texture_dir, "mesh_00.obj")
    if not os.path.exists(textured_mesh_path):
        raise FileNotFoundError(f"Textured mesh not found: {textured_mesh_path}")

    # Determine UV size from texture image resolution
    texture_img_path = os.path.join(texture_dir, "mesh_00_texture.png")
    texture_img = Image.open(texture_img_path)
    uv_size = texture_img.width  # square texture map
    print(f"UV size from texture image: {uv_size}")

    mesh_textured, offset, scale = load_mesh(
        textured_mesh_path,
        rescale=True,
        move_to_center=True,
        front_x_to_y=True,
        default_uv_size=uv_size,
        device=device,
        return_transform=True,
        merge_vertices=False,
    )

    # Load texture map
    texture_tensor = torch.from_numpy(np.array(texture_img)).float() / 255.0
    if texture_tensor.shape[-1] == 4:
        texture_tensor = texture_tensor[..., :3]
    mesh_textured.texture = texture_tensor.to(device)

    # Front camera only: azimuth 270 = canonical front view
    # In texture_actionmesh.py: azimuths=[270,...], azimuth_deg=[x-90 for x in azimuths] -> 180
    camera = get_orthogonal_camera(
        elevation_deg=[0],
        distance=[2.2],
        left=-0.55, right=0.55, bottom=-0.55, top=0.55,
        azimuth_deg=[180],
        device=device,
    )

    # Step 3: Load deformations and compute NN vertex mapping
    print("\n--- Step 3: Loading deformations ---")
    deformations = np.load(deformations_path)  # (T, V, 3)
    print(f"Loaded deformations: shape {deformations.shape}")

    # Undo save_deformation transform: saved as [-z, x, y] -> restore [x, y, z]
    deformations_orig = []
    for i in range(deformations.shape[0]):
        verts = deformations[i].copy()
        v_restored = np.zeros_like(verts)
        v_restored[:, 0] = verts[:, 1]
        v_restored[:, 1] = verts[:, 2]
        v_restored[:, 2] = -verts[:, 0]
        verts = apply_mesh_transforms(v_restored, offset, scale, front_x_to_y=True)
        deformations_orig.append(verts)
    deformations_orig = np.array(deformations_orig)  # (T, V_orig, 3)

    # Compute nearest-neighbour mapping: textured mesh verts -> original mesh verts
    print("Computing vertex mapping (textured -> original)...")
    v_tex_tensor = mesh_textured.v_pos  # (V_tex, 3)
    v_orig_tensor = torch.from_numpy(deformations_orig[0]).float().to(device)  # (V_orig, 3)

    dist_matrix = torch.cdist(v_tex_tensor.unsqueeze(0), v_orig_tensor.unsqueeze(0)).squeeze(0)
    min_dist, mapping_idx = torch.min(dist_matrix, dim=1)
    avg_dist = torch.mean(min_dist).item()
    print(f"Mapping computed. Avg distance: {avg_dist:.6f}")
    if avg_dist > 1e-4:
        print("WARNING: High average distance — possible misalignment between mesh and deformations.")

    # Step 4: Render and composite
    print("\n--- Step 4: Rendering and compositing ---")
    n_frames = deformations.shape[0]
    render_size = args.render_size
    video_frames = []

    for i in range(n_frames):
        # Update mesh vertices for this frame
        verts_orig = torch.from_numpy(deformations_orig[i]).float().to(device)
        verts_tex = verts_orig[mapping_idx]
        mesh_textured.v_pos = verts_tex
        mesh_textured._v_nrm = None
        mesh_textured._v_tang = None
        mesh_textured._stitched_v_pos = None

        # Render front view
        render_out = render(
            ctx,
            mesh_textured,
            camera,
            height=render_size,
            width=render_size,
            render_attr=True,
            render_depth=False,
            render_normal=False,
        )

        rgb = render_out.attr[0].clamp(0, 1).cpu()   # (H, W, 3) float
        mask = render_out.mask[0].cpu()               # (H, W) bool

        # Reverse localisation
        frame_key = f"{i:05d}.png"
        bbox = bboxes.get(frame_key)
        if bbox is None:
            video_frames.append(background_np.copy())
            if i % 5 == 0:
                print(f"Frame {i}/{n_frames}: no bbox, using background")
            continue

        x1, y1, x2, y2 = bbox
        crop_size = x2 - x1  # always square

        # Convert render to NCHW tensors for interpolation
        rgb_t = rgb.permute(2, 0, 1).unsqueeze(0)        # (1, 3, H, W)
        mask_t = mask.float().unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

        # Resize from render_size to crop_size
        rgb_resized  = F.interpolate(rgb_t,  size=(crop_size, crop_size), mode='bilinear', align_corners=False)
        mask_resized = F.interpolate(mask_t, size=(crop_size, crop_size), mode='nearest')

        # Compute OOB padding amounts (how much the crop extended outside the image)
        pad_left   = max(0, -x1)
        pad_top    = max(0, -y1)
        pad_right  = max(0, x2 - W)
        pad_bottom = max(0, y2 - H)

        # Crop out the OOB padding to get only the in-bounds region
        slice_y_end = crop_size - pad_bottom if pad_bottom > 0 else None
        slice_x_end = crop_size - pad_right  if pad_right  > 0 else None

        rgb_crop  = rgb_resized [..., pad_top:slice_y_end, pad_left:slice_x_end]
        mask_crop = mask_resized[..., pad_top:slice_y_end, pad_left:slice_x_end]

        # Back to numpy
        rgb_crop_np  = (rgb_crop.squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        mask_crop_np = mask_crop.squeeze(0).squeeze(0).bool().numpy()  # (h_crop, w_crop)

        # Composite onto background at original bbox position
        frame_np = background_np.copy()
        xs, xe = max(0, x1), min(W, x2)
        ys, ye = max(0, y1), min(H, y2)
        frame_np[ys:ye, xs:xe][mask_crop_np] = rgb_crop_np[mask_crop_np]

        video_frames.append(frame_np)

        if i % 5 == 0:
            print(f"Rendered frame {i}/{n_frames}")

    composite_path = os.path.join(output_path, "composite.mp4")
    imageio.mimsave(composite_path, video_frames, fps=args.fps)
    print(f"\nDone. Composite video saved to {composite_path}")
    print(f"  Resolution: {W}x{H}, {len(video_frames)} frames @ {args.fps} fps")


if __name__ == "__main__":
    main()
