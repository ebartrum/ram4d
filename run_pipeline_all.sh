#!/bin/bash
# Full alignment pipeline for all remaining objects.
# Run from ~/repos/ram4d inside the docker container.
set -e

GS_SCENE=Inpaint360GS/data/inpaint360/bag
GS_MODEL=output/2026.02.26/inpainted_scene/point_cloud_object_inpaint_virtual
DATE=output/2026.03.06

for OBJ in octopus trex; do
    echo ""
    echo "========================================"
    echo " $OBJ"
    echo "========================================"

    COMPOSITE=output/2026.03.03/actionmesh_gs_replace_${OBJ}
    REF_FRAME=${COMPOSITE}/frames/00000.jpg
    SEG_PROMPT=data/captions/${OBJ}_segmentation.txt
    IMG_PROMPT=data/captions/${OBJ}_image.txt
    CONSISTENT_REF_DIR=${DATE}/${OBJ}_consistent_ref_10px
    CONSISTENT_REF=${CONSISTENT_REF_DIR}/consistent_ref.png
    REFINED=${DATE}/${OBJ}_refined_consistent_occl
    DEFORM=${DATE}/${OBJ}_deform_occl

    mkdir -p ${CONSISTENT_REF_DIR} ${REFINED} ${DEFORM}

    echo "--- Step 1: consistent reference ---"
    python make_consistent_reference.py \
        --reference_frame ${REF_FRAME} \
        --gs_scene_path ${GS_SCENE} \
        --gs_model_path ${GS_MODEL} \
        --seg_prompt_path ${SEG_PROMPT} \
        --prompt_path ${IMG_PROMPT} \
        --output_path ${CONSISTENT_REF} \
        --camera_idx 28

    echo "--- Step 2: rigid alignment ---"
    python refine_frame0.py \
        --composite_path ${COMPOSITE}/gaussians \
        --gs_scene_path ${GS_SCENE} \
        --gs_model_path ${GS_MODEL} \
        --reference_frame ${CONSISTENT_REF} \
        --camera_idx 28 \
        --seg_prompt_path ${SEG_PROMPT} \
        --output_path ${REFINED} \
        --n_steps 500 --rgb_weight 1.0 --depth_weight 1.0 \
        --sds_type pseudo_sds --val_interval 50 --render_scale 1.0

    echo "--- Step 2.5: create composite (generate fg_positions_world.npy) ---"
    python create_composite_4dgs.py \
        --output_path ${COMPOSITE}/gaussians \
        --placement_path ${REFINED}/placement_refined.json

    echo "--- Step 3: deformation alignment ---"
    python refine_deform.py \
        --composite_path ${COMPOSITE}/gaussians \
        --placement_path ${REFINED}/placement_refined.json \
        --gs_scene_path ${GS_SCENE} \
        --gs_model_path ${GS_MODEL} \
        --reference_frame ${CONSISTENT_REF} \
        --camera_idx 28 \
        --seg_prompt_path ${SEG_PROMPT} \
        --output_path ${DEFORM} \
        --n_steps 500 --rgb_weight 1.0 --depth_weight 1.0 \
        --sds_type pseudo_sds --val_interval 50

    echo "--- Step 4: orbit render ---"
    python render_composite_4dgs.py \
        --output_path ${COMPOSITE}/gaussians \
        --gs_scene_path ${GS_SCENE} \
        --gs_model_path ${GS_MODEL} \
        --fg_positions_path ${COMPOSITE}/gaussians/fg_positions_world_deformed.npy \
        --placement_path ${REFINED}/placement_refined.json \
        --orbit --static --n_frames 240 --render_scale 0.5

    echo "=== $OBJ done ==="
done

echo ""
echo "All objects complete."
