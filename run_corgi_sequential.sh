#!/bin/bash
python refine_sequential.py \
    --composite_path output/2026.03.03/actionmesh_gs_replace_corgi \
    --placement_path output/2026.03.09/corgi_refined_consistent_occl/placement_refined.json \
    --gs_scene_path Inpaint360GS/data/inpaint360/bag \
    --gs_model_path output/2026.02.26/inpainted_scene/point_cloud_object_inpaint_virtual \
    --camera_idx 28 \
    --output_path output/2026.03.13/corgi_sequential_81_depth \
    --n_steps_per_frame 50 \
    --n_frames 81 \
    --rgb_weight 1.0 --silhouette_weight 1.0 --occlusion_weight 50.0 \
    --depth_weight 0.1 \
    --render_scale 0.5
