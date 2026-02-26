# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a PhD research project exploring **4D inpainting**: taking a static 3D scene and inserting a dynamic 4D asset generated from a text prompt. "4D" means 3 spatial dimensions plus time — the inserted asset must be a temporally coherent animation that remains consistent across multiple viewpoints in the 3D scene.

The 3D scene representation is **Gaussian Splatting (3DGS)**. The 3DGS rendering and scene work lives in a companion repo (`~/Documents/repos/Inpaint360GS`). Images such as `gs360_bag_test_IMG_0206.png` were imported from there as inputs to video generation experiments here. This repo (`ram4d`) focuses solely on the video/inpainting side of the pipeline.

The scripts here are **experimental and exploratory**, tackling sub-problems (image inpainting, segmentation, mask propagation, video generation) that are building blocks toward the full 4D system. Not all scripts are current or part of an active pipeline — some may be outdated, duplicative, or dead ends from earlier experiments.

## Development Workflow

Code is **edited locally** (on this machine) and **executed remotely** on a GPU server.

- **Local repo:** `/home/ed/Documents/repos/ram4d`
- **Remote repo:** `~/repos/ram4d` on `lambda_instance` (SSH alias defined in `~/.ssh/config`)

### Syncing code to the server

Use `rsync` to push local changes to the server before running anything:
```bash
rsync -av --exclude='.git' --exclude='/data' --exclude='/output' \
  /home/ed/Documents/repos/ram4d/ lambda_instance:~/repos/ram4d/
```
(Data and output directories are large and live only on the server — do not sync them back.)

### Running commands

The server uses a `docker-run` helper script that launches the correct Docker container with a conda environment activated. The conda environment is always `mvadapter`; the image tag depends on the server architecture:

```bash
# Default (arm64) instance:
docker-run --env mvadapter --tag mvadapter <command>

# amd64 instance:
docker-run --env mvadapter --tag mvadapter-amd64 <command>
```

For GPU-dependent commands, the preferred approach is to send them to the persistent `run_experiments` tmux session on the server:
```bash
ssh lambda_instance "tmux send-keys -t run_experiments 'cd ~/repos/ram4d && docker-run --non-interactive --env mvadapter --tag mvadapter python run_wan_fg_anim.py --mask_method sam2' Enter"
```
This survives wifi drops and connection interruptions. The user can monitor progress by attaching: `ssh lambda_instance "tmux attach -t run_experiments"`.

Alternatively, run via SSH directly with `--non-interactive` (but this will be killed if the connection drops):
```bash
ssh lambda_instance "cd ~/repos/ram4d && docker-run --non-interactive --env mvadapter --tag mvadapter python run_wan_fg_anim.py --mask_method sam2"
```

### Syncing data and outputs

Input data (`data/images/`, `data/captions/`) must be rsynced to the server separately if not already present — they are excluded from the main code sync. Sync them with:
```bash
rsync -av /home/ed/Documents/repos/ram4d/data/images/ lambda_instance:~/repos/ram4d/data/images/
rsync -av /home/ed/Documents/repos/ram4d/data/captions/ lambda_instance:~/repos/ram4d/data/captions/
```

To retrieve results, sync back the relevant dated output subdirectory:
```bash
rsync -av lambda_instance:~/repos/ram4d/output/YYYY.MM.DD/ /home/ed/Documents/repos/ram4d/output/YYYY.MM.DD/
```

### Output organisation

All outputs are saved under `output/YYYY.MM.DD/` (e.g. `output/2026.02.23/`). Always pass `--output_path output/YYYY.MM.DD/<filename>` when running scripts, and ensure the directory exists on the server beforehand:
```bash
ssh lambda_instance "mkdir -p ~/repos/ram4d/output/YYYY.MM.DD"
```

## Running Scripts

All scripts are standalone and run directly with Python from the repo root:

```bash
# Image inpainting (Flux or SDXL)
python run_flux_inpainting.py --image_path data/images/foo.jpg --mask_path data/images/foo_mask.png --prompt_path data/captions/foo.txt --model_type flux_fill

# Segmentation with LangSAM
python run_segmentation.py --image_path data/images/foo.png --prompt_path data/captions/foo_seg.txt

# Propagate mask across video using SAM2
python propogate_mask.py

# Prepare Wan input/mask videos from image+mask
python prepare_wan_input.py

# T2V inpainting with Wan 1.3B (uses Wan2.1/ fork)
python run_wan_inpainting.py --strength 1.0 --output output/result.mp4

# I2V inpainting with Wan 14B (custom sampling loop, uses official_wan_repo/)
python run_wan_i2v_inpaint.py --image data/images/foo.png --mask data/images/foo_mask.png --prompt_path data/captions/foo.txt

# Animate foreground on static background, conditioned on inpainted image (uses official_wan_repo/)
python run_wan_fg_anim.py --mask_method attention  # or --mask_method sam2

# First-Last Frame to Video with Wan 14B FLF2V
python run_wan_flf.py
```

There are no tests, linters, or build systems configured.

**Script status**: These are research experiments — some may be outdated, duplicative, or dead ends. The two most actively developed scripts are `run_wan_fg_anim.py` (animate foreground on static background, attention + SAM2 mask blending) and `run_wan_i2v_inpaint.py` (I2V inpainting with custom denoising loop).

## Architecture

### Two Wan Repos
The project uses **two separate Wan repositories** that serve different purposes:

- **`Wan2.1/`** – A custom fork containing `wan/text2video_inpaint.py` which defines `WanT2VInpaint`. This class extends `WanT2V` with video inpainting capability (latent blending). Used only by `run_wan_inpainting.py`.
- **`official_wan_repo/`** – The official Wan repo providing `wan.WanI2V` and `wan.WanFLF2V`. Used by `run_wan_i2v_inpaint.py`, `run_wan_fg_anim.py`, and `run_wan_flf.py`. These scripts implement their own custom denoising loops (not using the built-in `.generate()`) to allow mask blending at each step.

Both repos are added to `sys.path` at the top of each script that needs them. Do not import from both simultaneously in one script.

### Typical Pipeline
1. **Image inpainting** (`run_flux_inpainting.py`): Replace the background of a source image using Flux Fill or SDXL, producing an inpainted image.
2. **Segmentation** (`run_segmentation.py`): Use LangSAM (language-guided SAM) to produce a binary mask for the foreground subject in the image.
3. **Mask propagation** (`propogate_mask.py`): Use SAM2 video predictor to propagate the per-frame mask across a video.
4. **Video input prep** (`prepare_wan_input.py`): Assembles static video (81 frames) and corresponding mask video from the image/mask.
5. **Video generation/inpainting** (`run_wan_inpainting.py` / `run_wan_i2v_inpaint.py` / `run_wan_fg_anim.py`): Generates animated video while keeping certain regions fixed using latent blending.

### Key Supporting Files
- **`taehv.py`**: Tiny AutoEncoder for Wan (TAEHV) — a fast approximate VAE for debug decoding. `WanCompatibleTAEHV` wraps `TAEHV` to match the `WanVideoVAE` interface (`.encode(List[Tensor])` / `.decode(List[Tensor])`). Requires checkpoint `checkpoints/taew2_1.pth`.
- **`utils.py`**: `preprocess_mask_image(mask_path, dilation_pixels)` — loads a mask PNG, fills holes via contour filling, and dilates using `MaxFilter`. Used before feeding masks to the Wan models.

### Mask Conventions
- **Input masks**: White (255) = area to change/inpaint, Black (0) = area to keep.
- **`WanT2VInpaint.preprocess_mask()`** inverts this internally: it stores `1 - mask` so that 1 = keep, 0 = change.
- **Latent blending formula** (used in all custom loops): `latent = foreground_mask * generated_latent + (1 - foreground_mask) * background_latent`

### Latent Space
- VAE stride: temporal 4×, spatial 8× → a 512×512×81-frame video encodes to ~64×64×21 latents.
- Wan 14B latent channels: 16.
- Default video: 81 frames at 16 fps.

### Checkpoints & Data Directories
- `checkpoints/` — model checkpoints (gitignored): `sam2_hiera_tiny.pt`, `taew2_1.pth`.
- `data/images/` — input images and masks.
- `data/captions/` — prompt `.txt` files (one prompt per file).
- `data/video_input/` — assembled input/mask videos for Wan.
- `data/videos/` — source video clips.
- `data/cache/` — extracted video frame directories for SAM2.
- `output/` — all generated outputs (videos, images, debug artifacts).

### Wan Model Download
Scripts auto-download Wan models from HuggingFace Hub via `snapshot_download()` on first run. The 14B models are large (~30GB each). The 1.3B T2V model downloads to a local `Wan2.1-T2V-1.3B/` directory; the 14B models download to the HF cache.
