#!/bin/bash
set -e
cd ~/repos/ram4d

COMPOSITE=output/2026.03.03/actionmesh_gs_replace_corgi
REFINED=output/2026.03.09/corgi_refined_consistent_occl
GS_SCENE=Inpaint360GS/data/inpaint360/bag
GS_MODEL=output/2026.02.26/inpainted_scene/point_cloud_object_inpaint_virtual

echo "--- Step 1: create composite with refined placement ---"
python create_composite_4dgs.py \
    --output_path ${COMPOSITE}/gaussians \
    --placement_path ${REFINED}/placement_refined.json

echo "--- Step 2: orbit render (composite) ---"
python render_composite_4dgs.py \
    --output_path ${COMPOSITE}/gaussians \
    --gs_scene_path ${GS_SCENE} \
    --gs_model_path ${GS_MODEL} \
    --placement_path ${REFINED}/placement_refined.json \
    --render_output_dir ${REFINED} \
    --orbit --static

echo "--- Step 3: orbit render (fg only) ---"
python render_composite_4dgs.py \
    --output_path ${COMPOSITE}/gaussians \
    --gs_scene_path ${GS_SCENE} \
    --gs_model_path ${GS_MODEL} \
    --placement_path ${REFINED}/placement_refined.json \
    --render_output_dir ${REFINED} \
    --orbit --static --fg_only
