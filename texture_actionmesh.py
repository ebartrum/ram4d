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

# Add ActionMesh and its TripoSG dependency to path
sys.path.insert(0, os.path.abspath("actionmesh"))
sys.path.insert(0, os.path.abspath("actionmesh/third_party/TripoSG"))

import argparse
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import imageio

from visualise_output import TextureGenerator
from mvadapter.utils.mesh_utils import load_mesh, get_orthogonal_camera
from mesh_render_utils import (load_and_transform_deformations,
                                compute_vertex_mapping,
                                render_dynamic_mesh_video)


REQUIRED_CHECKPOINTS = {
    "checkpoints/RealESRGAN_x2plus.pth": (
        "wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
    ),
    "checkpoints/big-lama.pt": (
        "wget https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.pt"
    ),
}


def check_checkpoints():
    missing = []
    for path, download_cmd in REQUIRED_CHECKPOINTS.items():
        if not os.path.exists(path):
            missing.append((path, download_cmd))
    if missing:
        print("ERROR: Required checkpoints are missing:")
        for path, cmd in missing:
            print(f"  {path}")
            print(f"    Download: cd checkpoints && {cmd}")
        sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Texture ActionMesh output and render animated video.")
    parser.add_argument("--output_path", required=True,
                        help="ActionMesh output directory (from run_actionmesh.py).")
    parser.add_argument("--source_dir", default=None,
                        help="Localised frames directory. Defaults to <output_path>/localised_frames.")
    parser.add_argument("--prompt_path", default=None,
                        help="Path to a .txt file containing the text prompt for MV-Adapter texture generation.")
    parser.add_argument("--seed", type=int, default=-1,
                        help="Random seed (-1 = random).")
    parser.add_argument("--fps", type=int, default=16,
                        help="Output video fps.")
    parser.add_argument("--variant", default="sdxl", choices=["sdxl", "sd21"],
                        help="MV-Adapter variant.")
    return parser.parse_args()


def main():
    args = parse_args()
    check_checkpoints()

    # Resolve paths
    output_path = args.output_path
    source_dir = args.source_dir or os.path.join(output_path, "localised_frames")
    texture_dir = os.path.join(output_path, "texture")
    os.makedirs(texture_dir, exist_ok=True)

    # Anchor mesh (frame 0) — saved as mesh_00.glb by save_meshes()
    mesh_path = os.path.join(output_path, "meshes", "mesh_00.glb")
    if not os.path.exists(mesh_path):
        raise FileNotFoundError(f"Anchor mesh not found: {mesh_path}")

    # Reference image: first .png in source_dir, sorted by name
    png_files = sorted(glob.glob(os.path.join(source_dir, "*.png")))
    if not png_files:
        raise FileNotFoundError(f"No .png files found in {source_dir}")
    image_path = png_files[0]
    print(f"Reference image: {image_path}")
    print(f"Anchor mesh:     {mesh_path}")

    # Deformations
    deformations_path = os.path.join(output_path, "deformations_vertices.npy")
    if not os.path.exists(deformations_path):
        raise FileNotFoundError(f"deformations_vertices.npy not found at {deformations_path}")

    # Load text prompt
    if args.prompt_path:
        with open(args.prompt_path) as f:
            text_prompt = f.read().strip()
    else:
        text_prompt = "high quality"
    print(f"Text prompt: {text_prompt}")

    device = "cuda"

    # Step 1: Generate texture via MV-Adapter
    print("\n--- Step 1: Multi-view texture generation ---")
    generator = TextureGenerator(device=device, variant=args.variant, remove_bg=True)

    textured_mesh_path = generator.generate_texture(
        mesh_path=mesh_path,
        image_path=image_path,
        text_prompt=text_prompt,
        save_dir=texture_dir,
        seed=args.seed,
    )
    print(f"Textured mesh saved to: {textured_mesh_path}")

    # Step 2: Load textured mesh
    print("\n--- Step 2: Loading textured mesh ---")
    mesh_textured, offset, scale = load_mesh(
        textured_mesh_path,
        rescale=True,
        move_to_center=True,
        front_x_to_y=True,
        default_uv_size=generator.uv_size,
        device=device,
        return_transform=True,
        merge_vertices=False,
    )

    # Load texture map
    save_name = os.path.basename(mesh_path).split(".")[0]  # "mesh_00"
    texture_path = os.path.join(texture_dir, f"{save_name}_texture.png")
    if os.path.exists(texture_path):
        print(f"Loading texture from {texture_path}")
        texture_img = Image.open(texture_path)
        texture_tensor = torch.from_numpy(np.array(texture_img)).float() / 255.0
        if texture_tensor.shape[-1] == 4:
            texture_tensor = texture_tensor[..., :3]
        mesh_textured.texture = texture_tensor.to(device)
    else:
        print(f"Warning: Texture file not found at {texture_path}")

    # Cameras: front=270, right=0, back=90, left=180
    azimuths = [270, 0, 90, 180]
    cameras = get_orthogonal_camera(
        elevation_deg=[0] * 4,
        distance=[2.2] * 4,
        left=-0.55, right=0.55, bottom=-0.55, top=0.55,
        azimuth_deg=[x - 90 for x in azimuths],
        device=device,
    )

    # Step 3: Dynamic rendering
    print("\n--- Step 3: Dynamic textured mesh rendering ---")
    deformations_orig = load_and_transform_deformations(deformations_path, offset, scale)
    print(f"  Deformations: {deformations_orig.shape}")

    mapping_idx = compute_vertex_mapping(mesh_textured.v_pos, deformations_orig[0], device)

    # Load input frames for side-by-side grid
    input_frames = []
    T = deformations_orig.shape[0]
    frame_files = sorted(f for f in os.listdir(source_dir) if f.endswith(".png"))
    for fname in frame_files[:T]:
        img = Image.open(os.path.join(source_dir, fname)).convert("RGB").resize((768, 768))
        input_frames.append(img)
    if len(input_frames) < T:
        pad = input_frames[-1] if input_frames else Image.new("RGB", (768, 768))
        input_frames.extend([pad] * (T - len(input_frames)))

    video_frames = render_dynamic_mesh_video(
        generator.texture_pipe.ctx, mesh_textured, cameras,
        deformations_orig, mapping_idx,
        height=768, width=768,
        prefix_frames=input_frames,
    )

    video_path = os.path.join(texture_dir, "textured_dynamic_mesh.mp4")
    imageio.mimsave(video_path, video_frames, fps=args.fps)

    print(f"\nDone. Outputs in {texture_dir}:")
    print(f"  Multi-view images: {save_name}_mv.png")
    print(f"  Textured mesh:     {os.path.basename(textured_mesh_path)}")
    print(f"  Texture map:       {save_name}_texture.png")
    print(f"  Dynamic render:    textured_dynamic_mesh.mp4")


if __name__ == "__main__":
    main()
