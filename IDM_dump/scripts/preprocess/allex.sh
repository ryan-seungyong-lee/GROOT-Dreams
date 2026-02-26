#!/bin/bash
#SBATCH --job-name="allex_preprocess"
#SBATCH --gpus=1
#SBATCH --array=0
#SBATCH --nodes=1
#SBATCH --partition=sjw_alinlab
#SBATCH --output=/virtual_lab/sjw_alinlab/suhyeok/GR00T-Dreams/slurm_results/%x_%A_%a.out
#SBATCH --error=/virtual_lab/sjw_alinlab/suhyeok/GR00T-Dreams/slurm_results/%x_%A_%a.err

# ============================================================
# Allex Data Preprocessing Pipeline
# ============================================================
# 
# This script processes raw Allex data and applies IDM action predictions.
# All steps use global metadata from IDM training dataset for consistency.
#
# Steps:
#   0. Extract instructions from video filenames
#   1. Preprocess videos (224x224 resize, create left/right views)
#   2. Convert to LeRobot format
#   3. Dump IDM predicted actions (uses global metadata stats)
# ============================================================

# Step 0: Extract instructions from video filenames and organize structure
python IDM_dump/split_video_instruction.py \
    --source_dir "/storage1/sjw_dataset/dataset/Allex/generated/allex_vdm_ckpt-6k" \
    --output_dir "IDM_dump/data/allex_split" \
    --recursive

# # Step 1: Preprocess videos (224x224 resize, create left/right views)
python IDM_dump/preprocess_video.py \
    --src_dir "IDM_dump/data/allex_split" \
    --dst_dir "IDM_dump/data/allex_preprocessed" \
    --dataset allex

# # Step 2: Convert to lerobot format
python IDM_dump/raw_to_lerobot_allex.py \
    --input_dir "IDM_dump/data/allex_preprocessed" \
    --output_dir "IDM_dump/data/allex_lerobot" \
    --fps 16 \
    --num_workers 16

# Step 3: Dump IDM predicted actions
# 
# IMPORTANT: Both scripts now use global metadata from IDM training dataset
# (IDM_dump/global_metadata/allex/stats.json and modality.json)
# This ensures consistent normalization statistics as used during IDM training.
#
# NOTE: Observation indices selection:
#   - Training: 20fps, task duration ~20s, observation_indices = [0, 20] (1 second)
#   - Neural data: 16fps, task duration ~5s (4x faster actions)
#     -> Use --video_indices "0 4" for Neural data (faster actions)
#
# For synthetic/generated data (similar action speed to training):
#   - Auto-detection works: observation_indices will be adjusted for FPS
#   - Example: 16fps -> [0, 16] (1 second, same as training's 1 second)
#   - Omit --video_indices to use auto-detection
#
# For Neural data (faster actions):
#   - Manual specification required: --video_indices "0 4"
#   - This accounts for 4x faster action speed (5s vs 20s task duration)

export PYTHONPATH="/virtual_lab/sjw_alinlab/suhyeok/GR00T-Dreams:$PYTHONPATH"

# Option 1: Use unified script with standalone Allex mode (RECOMMENDED)
# This is the same as dump_idm_actions_allex.py but integrated into main script
python IDM_dump/dump_idm_actions.py \
    --checkpoint "/sjw_alinlab3/home/suhyeok/GR00T-Dreams/checkpoints/idm_allex_ego_0123_ae20_bsz64_step20k/checkpoint-20000" \
    --dataset "IDM_dump/data/allex_lerobot" \
    --output_dir "IDM_dump/data/allex_lerobot_idm_indice_08" \
    --num_gpus 1 \
    --batch_size 16 \
    --num_workers 8 \
    --video_indices "0 4" \
    --use_standalone_allex

