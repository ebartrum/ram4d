#!/bin/bash
# Single-frame DepthLab depth supervision test on frame 80.
# Tests whether DepthLab can fix the corgi floating above the table in late frames.
set -e
cd ~/repos/ram4d

COMPOSITE=output/2026.03.03/actionmesh_gs_replace_corgi
PLACEMENT=output/2026.03.09/corgi_refined_consistent_occl/placement_refined.json
SCENE=Inpaint360GS/data/inpaint360/bag
MODEL=output/2026.02.26/inpainted_scene/point_cloud_object_inpaint_virtual
OUT=output/2026.03.13/corgi_frame80_dw01

mkdir -p $OUT
mkdir -p output/2026.03.13/corgi_frame80_dw01_result

# --- Optimise frame 80 with DepthLab depth supervision ---
python refine_sequential.py \
    --composite_path $COMPOSITE \
    --placement_path $PLACEMENT \
    --gs_scene_path $SCENE \
    --gs_model_path $MODEL \
    --camera_idx 28 \
    --output_path $OUT \
    --n_steps_per_frame 200 \
    --target_frame 80 \
    --rgb_weight 1.0 \
    --silhouette_weight 1.0 \
    --occlusion_weight 50.0 \
    --depth_weight 0.1 \
    --render_scale 0.5

# --- Render result: cam22 and cam28 ---
# (fg_positions_frame80.npy has shape (1, N_fg, 3); --frame_idx 0 selects it)
python render_composite_4dgs.py \
    --output_path $COMPOSITE \
    --gs_scene_path $SCENE \
    --gs_model_path $MODEL \
    --fg_positions_path $OUT/fg_positions_frame80.npy \
    --placement_path $PLACEMENT \
    --render_output_dir output/2026.03.13/corgi_frame80_dw01_result \
    --camera_idx 22 \
    --render_scale 0.5 \
    --frame_idx 0

python render_composite_4dgs.py \
    --output_path $COMPOSITE \
    --gs_scene_path $SCENE \
    --gs_model_path $MODEL \
    --fg_positions_path $OUT/fg_positions_frame80.npy \
    --placement_path $PLACEMENT \
    --render_output_dir output/2026.03.13/corgi_frame80_dw01_result \
    --camera_idx 28 \
    --render_scale 0.5 \
    --frame_idx 0

echo "Done. Results in output/2026.03.13/corgi_frame80_dw01_result/"
echo "  Compare against baseline in output/2026.03.13/corgi_frame80_baseline/"
