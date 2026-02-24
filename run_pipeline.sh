#!/bin/bash
set -e
cd ~/repos/ram4d

DATE=2026.02.24

# ── butterfly_r_s3 ──────────────────────────────────────────
NAME=butterfly_r_s3
FG_ANIM_DIR=output/2026.02.23/butterfly_r_s3

echo "=== [$NAME] Step 1: ActionMesh ==="
python run_actionmesh.py \
  --source_video_dir $FG_ANIM_DIR \
  --output_path output/$DATE/${NAME}_actionmesh \
  --seed 44

echo "=== [$NAME] Step 2: Texture ==="
python texture_actionmesh.py \
  --output_path output/$DATE/${NAME}_actionmesh \
  --prompt_path data/captions/butterfly_image.txt --seed 42

# ── crab ────────────────────────────────────────────────────
NAME=crab
FG_ANIM_DIR=output/2026.02.23/crab

echo "=== [$NAME] Step 1: ActionMesh ==="
python run_actionmesh.py \
  --source_video_dir $FG_ANIM_DIR \
  --output_path output/$DATE/${NAME}_actionmesh \
  --seed 44

echo "=== [$NAME] Step 2: Texture ==="
python texture_actionmesh.py \
  --output_path output/$DATE/${NAME}_actionmesh \
  --prompt_path data/captions/crab_image.txt --seed 42

# ── octopus ─────────────────────────────────────────────────
NAME=octopus
FG_ANIM_DIR=output/2026.02.23/octopus

echo "=== [$NAME] Step 1: ActionMesh ==="
python run_actionmesh.py \
  --source_video_dir $FG_ANIM_DIR \
  --output_path output/$DATE/${NAME}_actionmesh \
  --seed 44

echo "=== [$NAME] Step 2: Texture ==="
python texture_actionmesh.py \
  --output_path output/$DATE/${NAME}_actionmesh \
  --prompt_path data/captions/octopus_image.txt --seed 42

echo "=== All done! ==="
