#!/bin/bash
set -e
cd ~/repos/ram4d

DATE=2026.02.24
BG=data/images/gs360_bag_test_IMG_0206.png

# ── butterfly_r_s3 ──────────────────────────────────────────
NAME=butterfly_r_s3
IMG=data/images/butterfly_gs360_bag_r_s3.png
MASK=data/images/butterfly_gs360_bag_r_s3_mask.png

echo "=== [$NAME] Step 1: Segmentation ==="
python run_segmentation.py --image_path $IMG --prompt_path data/captions/butterfly_segmentation.txt

echo "=== [$NAME] Step 2: fg_anim ==="
python run_wan_fg_anim.py \
  --bg_image $BG --input_image $IMG --mask_path $MASK \
  --prompt_path data/captions/butterfly_video.txt \
  --mask_method sam2 --mask_dilation 0 \
  --output_name $DATE/${NAME}_wan --seed 42 --width 832 --height 480

echo "=== [$NAME] Step 3: ActionMesh ==="
python run_actionmesh.py \
  --source_video_dir output/$DATE/${NAME}_wan \
  --output_path output/$DATE/${NAME}_actionmesh \
  --seed 44

echo "=== [$NAME] Step 4: Texture ==="
python texture_actionmesh.py \
  --output_path output/$DATE/${NAME}_actionmesh \
  --prompt_path data/captions/butterfly_image.txt --seed 42

# ── crab ────────────────────────────────────────────────────
NAME=crab
IMG=data/images/crab_gs360_bag_s0.png
MASK=data/images/crab_gs360_bag_s0_mask.png

echo "=== [$NAME] Step 1: Segmentation ==="
python run_segmentation.py --image_path $IMG --prompt_path data/captions/crab_segmentation.txt

echo "=== [$NAME] Step 2: fg_anim ==="
python run_wan_fg_anim.py \
  --bg_image $BG --input_image $IMG --mask_path $MASK \
  --prompt_path data/captions/crab_video.txt \
  --mask_method sam2 --mask_dilation 0 \
  --output_name $DATE/${NAME}_wan --seed 42 --width 832 --height 480

echo "=== [$NAME] Step 3: ActionMesh ==="
python run_actionmesh.py \
  --source_video_dir output/$DATE/${NAME}_wan \
  --output_path output/$DATE/${NAME}_actionmesh \
  --seed 44

echo "=== [$NAME] Step 4: Texture ==="
python texture_actionmesh.py \
  --output_path output/$DATE/${NAME}_actionmesh \
  --prompt_path data/captions/crab_image.txt --seed 42

# ── octopus ─────────────────────────────────────────────────
NAME=octopus
IMG=data/images/octopus_gs360_bag_s1.png
MASK=data/images/octopus_gs360_bag_s1_mask.png

echo "=== [$NAME] Step 1: Segmentation ==="
python run_segmentation.py --image_path $IMG --prompt_path data/captions/octopus_segmentation.txt

echo "=== [$NAME] Step 2: fg_anim ==="
python run_wan_fg_anim.py \
  --bg_image $BG --input_image $IMG --mask_path $MASK \
  --prompt_path data/captions/octopus_video.txt \
  --mask_method sam2 --mask_dilation 0 \
  --output_name $DATE/${NAME}_wan --seed 42 --width 832 --height 480

echo "=== [$NAME] Step 3: ActionMesh ==="
python run_actionmesh.py \
  --source_video_dir output/$DATE/${NAME}_wan \
  --output_path output/$DATE/${NAME}_actionmesh \
  --seed 44

echo "=== [$NAME] Step 4: Texture ==="
python texture_actionmesh.py \
  --output_path output/$DATE/${NAME}_actionmesh \
  --prompt_path data/captions/octopus_image.txt --seed 42

echo "=== All done! ==="
