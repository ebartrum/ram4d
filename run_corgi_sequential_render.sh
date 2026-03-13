#!/bin/bash
set -e
cd ~/repos/ram4d

FG_POS=output/2026.03.03/actionmesh_gs_replace_corgi/gaussians/fg_positions_world_sequential.npy
PLACEMENT=output/2026.03.09/corgi_refined_consistent_occl/placement_refined.json
SCENE=Inpaint360GS/data/inpaint360/bag
MODEL=output/2026.02.26/inpainted_scene/point_cloud_object_inpaint_virtual
OUTPUT_PATH=output/2026.03.03/actionmesh_gs_replace_corgi

# cam22 — side view (full video)
python render_composite_4dgs.py \
    --output_path $OUTPUT_PATH \
    --gs_scene_path $SCENE \
    --gs_model_path $MODEL \
    --fg_positions_path $FG_POS \
    --placement_path $PLACEMENT \
    --render_output_dir output/2026.03.13/corgi_sequential_v2_cam22 \
    --camera_idx 22 \
    --render_scale 0.5

# cam28 — source view (full video)
python render_composite_4dgs.py \
    --output_path $OUTPUT_PATH \
    --gs_scene_path $SCENE \
    --gs_model_path $MODEL \
    --fg_positions_path $FG_POS \
    --placement_path $PLACEMENT \
    --render_output_dir output/2026.03.13/corgi_sequential_v2_cam28 \
    --camera_idx 28 \
    --render_scale 0.5

# --- Baseline: frame 80 only (for DepthLab comparison) ---
mkdir -p output/2026.03.13/corgi_frame80_baseline

# cam22 frame 80 baseline
python render_composite_4dgs.py \
    --output_path $OUTPUT_PATH \
    --gs_scene_path $SCENE \
    --gs_model_path $MODEL \
    --fg_positions_path $FG_POS \
    --placement_path $PLACEMENT \
    --render_output_dir output/2026.03.13/corgi_frame80_baseline \
    --camera_idx 22 \
    --render_scale 0.5 \
    --frame_idx 80

# cam28 frame 80 baseline
python render_composite_4dgs.py \
    --output_path $OUTPUT_PATH \
    --gs_scene_path $SCENE \
    --gs_model_path $MODEL \
    --fg_positions_path $FG_POS \
    --placement_path $PLACEMENT \
    --render_output_dir output/2026.03.13/corgi_frame80_baseline \
    --camera_idx 28 \
    --render_scale 0.5 \
    --frame_idx 80
