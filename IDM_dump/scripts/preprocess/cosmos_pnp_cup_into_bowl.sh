#!/bin/bash
#SBATCH --job-name="cosmos_pnp_cup_preprocess"
#SBATCH --gpus=1
#SBATCH --array=0
#SBATCH --nodes=1
#SBATCH --partition=sjw_alinlab
#SBATCH --output=/virtual_lab/sjw_alinlab/suhyeok/GR00T-Dreams/slurm_results/%x_%A_%a.out
#SBATCH --error=/virtual_lab/sjw_alinlab/suhyeok/GR00T-Dreams/slurm_results/%x_%A_%a.err

# ============================================================
# Cosmos Generated Data Preprocessing Pipeline - PNP Cup into Bowl
# ============================================================
# 
# This script processes cosmos-generated videos for pnp_cup_into_bowl task.
# Steps:
#   0. Organize videos with instruction
#   1. Preprocess videos (224x224 resize, create left/right views)
#   2. Convert to LeRobot format (16fps)
#   3. Dump IDM predicted actions (observation indices: 0 4)
# ============================================================

CONDA_PATH="/sjw_alinlab3/home/suhyeok/miniconda3"    
source "$CONDA_PATH/etc/profile.d/conda.sh"
conda activate cosmos-predict1

TASK_NAME="pnp_cup_into_bowl"
INSTRUCTION="pick up the blue cup and place it into the white bowl"
SOURCE_VIDEO_DIR="/storage1/sjw_dataset/dataset/Allex/generated/cosmos-generated-for-allex-icml/${TASK_NAME}/videos/bestofn_4"
FILTER_STATS_FILE="/storage1/sjw_dataset/dataset/Allex/generated/cosmos-generated-for-allex-icml/${TASK_NAME}/filter_results/filtered_videos_stats_gemini_3_flash_preview.json"
BASE_OUTPUT_DIR="IDM_dump/data/cosmos_${TASK_NAME}_hotfix"

# Step 0: Organize videos with instruction (excluding failed videos from filtering)
echo "Step 0: Organizing videos with instruction (excluding failed videos)..."
python IDM_dump/split_video_instruction.py \
    --source_dir "${SOURCE_VIDEO_DIR}" \
    --output_dir "${BASE_OUTPUT_DIR}_split" \
    --instruction "${INSTRUCTION}" \
    --filter_stats_file "${FILTER_STATS_FILE}" \
    --recursive

# Step 1: Preprocess videos (224x224 resize, create left/right views)
echo "Step 1: Preprocessing videos..."
python IDM_dump/preprocess_video.py \
    --src_dir "${BASE_OUTPUT_DIR}_split" \
    --dst_dir "${BASE_OUTPUT_DIR}_preprocessed" \
    --dataset allex

# Step 2: Convert to lerobot format (16fps)
echo "Step 2: Converting to LeRobot format (16fps)..."
python IDM_dump/raw_to_lerobot_allex.py \
    --input_dir "${BASE_OUTPUT_DIR}_preprocessed" \
    --output_dir "${BASE_OUTPUT_DIR}_lerobot" \
    --fps 16 \
    --num_workers 16

# Step 3: Dump IDM predicted actions
# Observation indices: 0 4 (for 16fps, faster actions)
# Uses stats from checkpoint directory
echo "Step 3: Dumping IDM predicted actions (observation indices: 0 4)..."
export PYTHONPATH="/virtual_lab/sjw_alinlab/suhyeok/GR00T-Dreams:$PYTHONPATH"

python IDM_dump/dump_idm_actions.py \
    --checkpoint "/sjw_alinlab3/home/suhyeok/GR00T-Dreams/checkpoints/idm_allex_ego_0124_add_teleop_ae20_bsz64_step20k/checkpoint-20000" \
    --dataset "${BASE_OUTPUT_DIR}_lerobot" \
    --output_dir "${BASE_OUTPUT_DIR}_lerobot_idm_indice_04" \
    --num_gpus 1 \
    --batch_size 16 \
    --num_workers 8 \
    --video_indices "0 4" \
    --use_standalone_allex

echo "Pipeline complete for ${TASK_NAME}"

