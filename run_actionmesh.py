import sys
import os
import glob

# Fix sklearn TLS issue: LD_PRELOAD must be set before the dynamic linker loads
# scikit-learn's bundled libgomp. We do this by re-execing the process.
if "LIBGOMP_PRELOADED" not in os.environ:
    libgomp_files = glob.glob(
        "/home/ubuntu/miniconda3/envs/mvadapter/lib/python*/site-packages/scikit_learn.libs/libgomp*.so*"
    )
    if libgomp_files:
        os.environ["LD_PRELOAD"] = libgomp_files[0]
        os.environ["LIBGOMP_PRELOADED"] = "1"
        os.execv(sys.executable, [sys.executable] + sys.argv)

# Add ActionMesh and its TripoSG dependency to path
sys.path.insert(0, os.path.abspath("actionmesh"))
sys.path.insert(0, os.path.abspath("actionmesh/third_party/TripoSG"))

import argparse
import json
import cv2
import numpy as np
import torch
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor


def parse_args():
    parser = argparse.ArgumentParser(description="4D reconstruction from Wan output: localize then run ActionMesh.")
    parser.add_argument("--source_video_dir", required=True,
                        help="Path to run_wan_fg_anim.py output directory (must contain final_output.mp4 and args.txt).")
    parser.add_argument("--output_path", required=True,
                        help="Directory to write all outputs.")
    parser.add_argument("--mask_path", default=None,
                        help="Override initial mask for SAM2 (frame 0). Defaults to mask_path from args.txt.")
    parser.add_argument("--output_size", type=int, default=512,
                        help="Localized frame size (square). Default: 512.")
    parser.add_argument("--margin", type=int, default=30,
                        help="Bounding box pixel margin. Default: 30.")
    parser.add_argument("--seed", type=int, default=44,
                        help="Random seed for ActionMesh. Default: 44.")
    parser.add_argument("--stage_0_steps", type=int, default=None,
                        help="TripoSG inference steps (default: 100 from config).")
    parser.add_argument("--stage_1_steps", type=int, default=None,
                        help="Temporal denoiser flow-matching steps (default: 30 from config).")
    parser.add_argument("--fast", action="store_true",
                        help="Fast mode: stage_0_steps=50, stage_1_steps=15.")
    return parser.parse_args()


def read_args_txt(args_txt_path):
    """Parse key: value pairs from an args.txt file."""
    result = {}
    with open(args_txt_path) as f:
        for line in f:
            line = line.strip()
            if not line or ": " not in line:
                continue
            key, _, value = line.partition(": ")
            result[key.strip()] = value.strip()
    return result


def extract_frames(video_path, output_dir):
    """Extract all frames from a video as JPEGs (required by SAM2)."""
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(output_dir, f"{frame_count:05d}.jpg"), frame)
        frame_count += 1
    cap.release()
    print(f"Extracted {frame_count} frames to {output_dir}")
    return frame_count


def propagate_sam2_masks(frames_dir, mask_path, output_dir, checkpoint, config):
    """Run SAM2 mask propagation and save per-frame binary masks + summary video."""
    os.makedirs(output_dir, exist_ok=True)

    print("Initializing SAM2...")
    predictor = build_sam2_video_predictor(config, checkpoint)
    inference_state = predictor.init_state(video_path=frames_dir)

    # Load initial mask and resize to match the first frame
    first_frame = cv2.imread(os.path.join(frames_dir, "00000.jpg"))
    h, w = first_frame.shape[:2]

    mask_img = Image.open(mask_path).convert("L").resize((w, h), Image.NEAREST)
    binary_mask = (np.array(mask_img) > 128).astype(np.float32)

    ann_frame_idx = 0
    ann_obj_id = 1
    predictor.add_new_mask(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        mask=binary_mask,
    )

    # Video writer for quick visual check
    video_path = os.path.join(output_dir, "propagated_masks.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_video = cv2.VideoWriter(video_path, fourcc, 16.0, (w, h), isColor=False)

    print("Propagating masks...")
    masks_by_frame = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        masks_by_frame[out_frame_idx] = {
            obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, obj_id in enumerate(out_obj_ids)
        }

        if ann_obj_id in masks_by_frame[out_frame_idx]:
            mask_arr = masks_by_frame[out_frame_idx][ann_obj_id].squeeze()  # [H, W]
            mask_uint8 = (mask_arr * 255).astype(np.uint8)
            Image.fromarray(mask_uint8).save(os.path.join(output_dir, f"{out_frame_idx:05d}.png"))
            out_video.write(mask_uint8)

    out_video.release()
    print(f"SAM2 masks saved to {output_dir}")
    print(f"Mask video saved to {video_path}")


def localise_frames(frames_dir, masks_dir, output_dir, output_size, margin):
    """Crop and resize each frame to isolate the foreground subject."""
    os.makedirs(output_dir, exist_ok=True)

    frame_files = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")) + glob.glob(os.path.join(frames_dir, "*.png")))
    mask_files = sorted(glob.glob(os.path.join(masks_dir, "*.png")))
    n = min(len(frame_files), len(mask_files))
    print(f"Localising {n} frames...")

    bbox_data = {}

    for i in range(n):
        frame = cv2.imread(frame_files[i])
        mask_gray = cv2.imread(mask_files[i], cv2.IMREAD_GRAYSCALE)
        frame_name = f"{i:05d}.png"

        mask_binary = (mask_gray > 0).astype(np.float32)

        # Apply mask: black out background
        masked_frame = (frame.astype(np.float32) * mask_binary[:, :, np.newaxis]).astype(np.uint8)

        ys, xs = np.where(mask_gray > 0)

        if len(xs) == 0 or len(ys) == 0:
            print(f"  Warning: no object found in frame {i}, saving full masked frame.")
            output_image = cv2.resize(masked_frame, (output_size, output_size))
            cv2.imwrite(os.path.join(output_dir, frame_name), output_image)
            bbox_data[frame_name] = None
            continue

        min_x, max_x = int(np.min(xs)), int(np.max(xs))
        min_y, max_y = int(np.min(ys)), int(np.max(ys))

        cx = (min_x + max_x) // 2
        cy = (min_y + max_y) // 2
        max_side = max(max_x - min_x, max_y - min_y)
        crop_size = max_side + margin
        half = crop_size // 2

        x1 = cx - half
        y1 = cy - half
        x2 = x1 + crop_size
        y2 = y1 + crop_size

        bbox_data[frame_name] = [x1, y1, x2, y2]

        # Pad if the crop extends out of bounds
        pad_left = max(0, -x1)
        pad_top = max(0, -y1)
        pad_right = max(0, x2 - masked_frame.shape[1])
        pad_bottom = max(0, y2 - masked_frame.shape[0])

        if pad_left or pad_top or pad_right or pad_bottom:
            padded = cv2.copyMakeBorder(
                masked_frame, pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_CONSTANT, value=[0, 0, 0],
            )
            cx1, cy1 = x1 + pad_left, y1 + pad_top
            cx2, cy2 = x2 + pad_left, y2 + pad_top
            crop = padded[cy1:cy2, cx1:cx2]
        else:
            crop = masked_frame[y1:y2, x1:x2]

        output_image = cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(output_dir, frame_name), output_image)

    bbox_path = os.path.join(output_dir, "bounding_boxes.json")
    with open(bbox_path, "w") as f:
        json.dump(bbox_data, f, indent=4)
    print(f"Localised frames saved to {output_dir}")
    print(f"Bounding boxes saved to {bbox_path}")


def run_actionmesh(localised_dir, output_dir, seed, stage_0_steps, stage_1_steps):
    """Run ActionMesh pipeline on localised frames to produce an animated 3D mesh."""
    from actionmesh.pipeline import ActionMeshPipeline
    from actionmesh.io.video_input import load_frames
    from actionmesh.io.mesh_io import save_meshes, save_deformation

    meshes_dir = os.path.join(output_dir, "meshes")
    os.makedirs(meshes_dir, exist_ok=True)

    # Load all localised frames (black-background PNGs); RMBG inside the pipeline
    # will estimate alpha from the clean black background.
    print(f"Loading localised frames from {localised_dir}...")
    am_input = load_frames(localised_dir)
    print(f"Loaded {am_input.n_frames} frames")

    from hydra.core.global_hydra import GlobalHydra
    GlobalHydra.instance().clear()

    config_dir = os.path.abspath("actionmesh/actionmesh/configs")
    print("Initializing ActionMeshPipeline...")
    pipeline = ActionMeshPipeline(
        config_name="actionmesh.yaml",
        config_dir=config_dir,
    )
    pipeline.to("cuda")

    kwargs = dict(seed=seed)
    if stage_0_steps is not None:
        kwargs["stage_0_steps"] = stage_0_steps
    if stage_1_steps is not None:
        kwargs["stage_1_steps"] = stage_1_steps

    print(f"Running ActionMesh on {am_input.n_frames} frames (kwargs: {kwargs})...")
    meshes = pipeline(input=am_input, **kwargs)
    print(f"Generated {len(meshes)} meshes")

    print(f"Saving per-frame GLB meshes to {meshes_dir}...")
    save_meshes(meshes, output_dir=meshes_dir)

    deform_path = os.path.join(output_dir, "deformations")
    verts_path, faces_path = save_deformation(meshes, path=deform_path)
    print(f"Saved deformation arrays: {verts_path}, {faces_path}")


def main():
    args = parse_args()

    # Apply --fast shorthand
    stage_0_steps = args.stage_0_steps
    stage_1_steps = args.stage_1_steps
    if args.fast:
        stage_0_steps = stage_0_steps or 50
        stage_1_steps = stage_1_steps or 15

    corgi_dir = args.source_video_dir
    video_path = os.path.join(corgi_dir, "final_output.mp4")
    args_txt_path = os.path.join(corgi_dir, "args.txt")

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"final_output.mp4 not found in {corgi_dir}")

    # Step 1: Read args.txt to get mask_path (unless overridden)
    corgi_args = {}
    if os.path.isfile(args_txt_path):
        corgi_args = read_args_txt(args_txt_path)
        print(f"Read args.txt: {corgi_args}")
    else:
        print(f"Warning: args.txt not found at {args_txt_path}")

    mask_path = args.mask_path
    if mask_path is None:
        mask_path = corgi_args.get("mask_path")
    if mask_path is None:
        raise ValueError("No mask_path available: provide --mask_path or ensure args.txt contains mask_path.")
    if not os.path.isfile(mask_path):
        raise FileNotFoundError(f"Mask not found: {mask_path}")
    print(f"Using initial mask: {mask_path}")

    output_path = args.output_path
    frames_dir = os.path.join(output_path, "frames")
    masks_dir = os.path.join(output_path, "sam2_masks")
    localised_dir = os.path.join(output_path, "localised_frames")

    # Step 2: Extract frames
    print("\n--- Step 2: Extracting frames ---")
    if os.path.isdir(frames_dir) and os.listdir(frames_dir):
        print(f"Frames already present in {frames_dir}, skipping extraction.")
    else:
        extract_frames(video_path, frames_dir)

    # Step 3: SAM2 mask propagation
    print("\n--- Step 3: SAM2 mask propagation ---")
    if os.path.isdir(masks_dir) and os.listdir(masks_dir):
        print(f"SAM2 masks already present in {masks_dir}, skipping propagation.")
    else:
        checkpoint = "checkpoints/sam2_hiera_tiny.pt"
        config = "sam2_hiera_t.yaml"
        propagate_sam2_masks(frames_dir, mask_path, masks_dir, checkpoint, config)

    # Step 4: Localize frames
    print("\n--- Step 4: Localizing frames ---")
    if os.path.isdir(localised_dir) and os.listdir(localised_dir):
        print(f"Localised frames already present in {localised_dir}, skipping localization.")
    else:
        localise_frames(frames_dir, masks_dir, localised_dir, args.output_size, args.margin)

    # Step 5: ActionMesh inference
    print("\n--- Step 5: ActionMesh inference ---")
    run_actionmesh(localised_dir, output_path, args.seed, stage_0_steps, stage_1_steps)

    print("\nDone. Outputs:")
    print(f"  Frames:           {frames_dir}")
    print(f"  SAM2 masks:       {masks_dir}")
    print(f"  Localised frames: {localised_dir}")
    print(f"  Meshes:           {os.path.join(output_path, 'meshes')}")
    print(f"  Deformations:     {os.path.join(output_path, 'deformations_vertices.npy')}")


if __name__ == "__main__":
    main()
