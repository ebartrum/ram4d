#!/bin/bash
set -e
cd ~/repos/ram4d

python render_composite_4dgs.py \
    --output_path output/2026.03.03/actionmesh_gs_replace_corgi/gaussians \
    --gs_scene_path Inpaint360GS/data/inpaint360/bag \
    --gs_model_path output/2026.02.26/inpainted_scene/point_cloud_object_inpaint_virtual \
    --fg_positions_path output/2026.03.03/actionmesh_gs_replace_corgi/gaussians/fg_positions_world_sequential.npy \
    --placement_path output/2026.03.09/corgi_refined_consistent_occl/placement_refined.json \
    --render_output_dir output/2026.03.11/corgi_sequential_orbit \
    --orbit --n_frames 240 --render_scale 0.5
