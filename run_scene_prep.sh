#!/usr/bin/env bash
# run_scene_prep.sh — Orchestrate the full Inpaint360GS pipeline on the remote server.
#
# Runs all three stages (train object-aware Gaussians → remove object → 2D+3D inpainting)
# via SSH + docker-run. The interactive Segment-and-Track-Anything Gradio step is
# replaced by tools/bridge_masks.py (automated).
#
# Prerequisites (one-time, on the server):
#   1. cd ~/repos/ram4d/Inpaint360GS && git submodule update --init --recursive
#   2. Set up the 'inpaint360gs' conda env (see Inpaint360GS/CLAUDE.md)
#   3. Set up the 'lama' conda env (see Inpaint360GS/CLAUDE.md)
#   4. Download CropFormer weights to Inpaint360GS/seg/weight/
#   5. Download big-lama.zip weights to Inpaint360GS/LaMa/
#   6. Ensure input data exists at:
#        ~/repos/ram4d/Inpaint360GS/data/${DATASET_NAME}/${SCENE}/images/
#        ~/repos/ram4d/Inpaint360GS/data/${DATASET_NAME}/${SCENE}/sparse/0/
#      Run COLMAP (convert.py) first if sparse/0 does not exist.
#
# Usage:
#   bash run_scene_prep.sh [--stage N]   (run all stages, or start from stage N: 1, 2, or 3)

set -e

# ============================================================
# CONFIGURE THESE
# ============================================================
DATASET_NAME="inpaint360"        # dataset folder name under Inpaint360GS/data/
SCENE="bag"                      # scene name
RESOLUTION=2                     # downscale factor (1=full, 2=half, 4=quarter, 8=eighth)
TARGET_ID="26"                   # Gaussian object IDs to permanently remove (space-separated)
SURROUNDING_ID="24 10"           # Temporarily removed IDs restored during inpainting (space-sep, or "None")
# ============================================================

SSH_HOST="lambda_instance"
REMOTE_ROOT="~/repos/ram4d/Inpaint360GS"

# Docker image tag — adjust if using the amd64 server
DOCKER_TAG="mvadapter"

# Env names matching the conda environments inside docker
ENV_INPAINT="inpaint360gs"
ENV_LAMA="lama"

# ============================================================
# Helpers
# ============================================================

# Run a Python command on the server in the given conda env
run_remote() {
    local env="$1"
    shift
    echo ""
    echo ">>> [${env}] $*"
    ssh "${SSH_HOST}" "cd ${REMOTE_ROOT} && docker-run --non-interactive --env ${env} --tag ${DOCKER_TAG} $*"
}

# Run a bash command on the server (no docker-run wrapper)
run_remote_bash() {
    echo ""
    echo ">>> [bash] $*"
    ssh "${SSH_HOST}" "cd ${REMOTE_ROOT} && $*"
}

# ============================================================
# Argument parsing
# ============================================================
START_STAGE=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage)
            START_STAGE="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: bash run_scene_prep.sh [--stage N]"
            exit 1
            ;;
    esac
done

# ============================================================
# Optional: COLMAP (run if sparse/0 doesn't exist)
# ============================================================
echo ""
echo "=== Checking for COLMAP sparse model... ==="
COLMAP_EXISTS=$(ssh "${SSH_HOST}" "[ -d ${REMOTE_ROOT}/data/${DATASET_NAME}/${SCENE}/sparse/0 ] && echo yes || echo no")
if [ "${COLMAP_EXISTS}" = "no" ]; then
    echo "sparse/0 not found — running COLMAP via convert.py..."
    run_remote "${ENV_INPAINT}" python convert.py -s "data/${DATASET_NAME}/${SCENE}"
else
    echo "sparse/0 found — skipping COLMAP."
fi

# ============================================================
# Stage 1: Train Object-Aware Gaussians
# ============================================================
if [ "${START_STAGE}" -le 1 ]; then
    echo ""
    echo "============================================================"
    echo "STAGE 1: Train Object-Aware Gaussians"
    echo "============================================================"

    # 1a. Vanilla 3DGS training
    run_remote "${ENV_INPAINT}" python gaussian_splatting/train.py \
        -s "data/${DATASET_NAME}/${SCENE}" \
        -m "output/${DATASET_NAME}/${SCENE}/3dgs_output" \
        --init_mode sparse \
        --eval \
        --resolution "${RESOLUTION}"

    # 1b. 2D segmentation masks (HQ-SAM via CropFormer/detectron2)
    run_remote "${ENV_INPAINT}" bash -c \
        "PYTHONPATH=\$(pwd):\$(pwd)/seg/detectron2:\$PYTHONPATH python seg/raw_mask_sam.py \
            --dataset_path data/${DATASET_NAME}/ \
            --scene_name ${SCENE} \
            --image_folder images_${RESOLUTION} \
            --method hqsam"

    # 1c. Lift 2D masks to 3D
    run_remote "${ENV_INPAINT}" python seg/mask_associate.py \
        -s "data/${DATASET_NAME}/${SCENE}" \
        -m "output/${DATASET_NAME}/${SCENE}/3dgs_output" \
        --resolution "${RESOLUTION}" \
        --mask_generator hqsam \
        --eval

    # 1d. Label post-processing
    run_remote "${ENV_INPAINT}" python tools/add_label_num_hqsam.py \
        -s "data/${DATASET_NAME}/${SCENE}" \
        --resolution "${RESOLUTION}" \
        --mask_generator hqsam

    # 1e. Semantic distillation
    run_remote "${ENV_INPAINT}" python seg/distillation.py \
        -s "data/${DATASET_NAME}/${SCENE}" \
        -m "output/${DATASET_NAME}/${SCENE}" \
        --vanilla_3dgs_path "output/${DATASET_NAME}/${SCENE}/3dgs_output" \
        --resolution "${RESOLUTION}" \
        --object_path associated_hqsam \
        --eval

    # 1f. Render + video
    run_remote "${ENV_INPAINT}" python render.py \
        -m "output/${DATASET_NAME}/${SCENE}" \
        --render_video
fi

# ============================================================
# Stage 2: Object Removal + Automated Mask Generation
# ============================================================
if [ "${START_STAGE}" -le 2 ]; then
    echo ""
    echo "============================================================"
    echo "STAGE 2: Object Removal + Automated Mask Generation"
    echo "============================================================"

    # Convert TARGET_ID and SURROUNDING_ID from space-separated strings to comma-separated
    TARGET_ID_COMMA="${TARGET_ID// /,}"
    SURROUNDING_ID_COMMA="${SURROUNDING_ID// /,}"

    # 2a. Generate scene JSON configs
    run_remote "${ENV_INPAINT}" python tools/init_configs.py \
        --dataset_name "${DATASET_NAME}" \
        --scene "${SCENE}" \
        --target_id "${TARGET_ID_COMMA}" \
        --target_surronding_id "${SURROUNDING_ID_COMMA}"

    # 2b. Remove selected Gaussians, render virtual views
    run_remote "${ENV_INPAINT}" python edit_object_removal.py \
        -s "data/${DATASET_NAME}/${SCENE}" \
        -m "output/${DATASET_NAME}/${SCENE}" \
        --config_file "config/object_removal/${DATASET_NAME}/${SCENE}.json" \
        --render_video

    # 2c. Generate virtual camera trajectory
    run_remote "${ENV_INPAINT}" python tools/virtual_pose.py \
        -s "data/${DATASET_NAME}/${SCENE}" \
        -m "output/${DATASET_NAME}/${SCENE}" \
        --config_file "config/object_removal/${DATASET_NAME}/${SCENE}.json"

    # 2d. Auto-generate masks from objects_pred (replaces interactive SaTA Gradio step)
    run_remote "${ENV_INPAINT}" python tools/bridge_masks.py \
        --dataset_name "${DATASET_NAME}" \
        --scene "${SCENE}"
fi

# ============================================================
# Stage 3: 2D Inpainting + 3D Optimization
# ============================================================
if [ "${START_STAGE}" -le 3 ]; then
    echo ""
    echo "============================================================"
    echo "STAGE 3: 2D Inpainting + 3D Optimization"
    echo "============================================================"

    # 3a. Prepare LaMa input data (copy renders + masks into LaMa data dirs)
    run_remote "${ENV_INPAINT}" python tools/prepare_lama_data.py \
        -s "data/${DATASET_NAME}/${SCENE}" \
        -m "output/${DATASET_NAME}/${SCENE}" \
        -r "${RESOLUTION}" \
        --inpaint2lama

    # 3b. LaMa color inpainting
    run_remote "${ENV_LAMA}" bash -c \
        "cd LaMa && export TORCH_HOME=\$(pwd) && export PYTHONPATH=\$(pwd) && \
         python bin/predict_color.py --data_name 360_${SCENE}_virtual"

    # 3c. LaMa depth inpainting
    run_remote "${ENV_LAMA}" bash -c \
        "cd LaMa && export TORCH_HOME=\$(pwd) && export PYTHONPATH=\$(pwd) && \
         python bin/predict_depth.py --data_name 360_${SCENE}_virtual"

    # 3d. Copy LaMa outputs back into scene directory
    run_remote "${ENV_INPAINT}" python tools/prepare_lama_data.py \
        -s "data/${DATASET_NAME}/${SCENE}" \
        -m "output/${DATASET_NAME}/${SCENE}" \
        -r "${RESOLUTION}"

    # 3e. Colorful point cloud fusion (back-project inpainted colors into 3D)
    run_remote "${ENV_INPAINT}" python edit_object_removal_plyfusion.py \
        -s "data/${DATASET_NAME}/${SCENE}" \
        -m "output/${DATASET_NAME}/${SCENE}" \
        --config_file "config/object_removal/${DATASET_NAME}/${SCENE}.json"

    # 3f. 3DGS inpainting optimization (fills missing regions in Gaussian scene)
    run_remote "${ENV_INPAINT}" python edit_object_inpaint.py \
        -s "data/${DATASET_NAME}/${SCENE}" \
        -m "output/${DATASET_NAME}/${SCENE}" \
        --config_file "config/object_inpaint/${DATASET_NAME}/${SCENE}.json" \
        --resolution "${RESOLUTION}" \
        --render_video
fi

echo ""
echo "============================================================"
echo "Pipeline complete!"
echo "Results in: ${REMOTE_ROOT}/output/${DATASET_NAME}/${SCENE}/"
echo ""
echo "To retrieve outputs:"
echo "  rsync -av ${SSH_HOST}:${REMOTE_ROOT}/output/${DATASET_NAME}/${SCENE}/ \\"
echo "    /home/ed/Documents/repos/Inpaint360GS/output/${DATASET_NAME}/${SCENE}/"
echo "============================================================"
