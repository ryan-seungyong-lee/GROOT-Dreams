#!/bin/bash
#SBATCH --job-name="dump-idm-allex"
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=sjw_alinlab
#SBATCH --exclude=worker-node1001
#SBATCH --output=slurm_out/%x_%j.out
#SBATCH --error=slurm_out/%x_%j.err

# ============================================================
# Dump IDM Actions for Allex Robot Dataset
# ============================================================
# 
# action_horizon은 checkpoint에서 자동으로 읽어옴!
#   - ae10: action_horizon=10, observation_indices=[0, 10]
#   - ae20: action_horizon=20, observation_indices=[0, 20]
#   - ae40: action_horizon=40, observation_indices=[0, 40]
#
# IDM 철학:
#   S_0, S_N 관찰 → a_0, a_1, ..., a_{N-1} 예측
# ============================================================

export NO_ALBUMENTATIONS_UPDATE=1
export PYTHONPATH="/sjw_alinlab3/home/suhyeok/GR00T-Dreams:$PYTHONPATH"

CONDA_PATH="/sjw_alinlab3/home/suhyeok/miniconda3"    
source "$CONDA_PATH/etc/profile.d/conda.sh"
conda activate cosmos-predict1

# ============================================================
# Configuration
# ============================================================

# IDM Checkpoint (action_horizon은 자동으로 checkpoint에서 읽어옴)
CHECKPOINT=/sjw_alinlab3/home/suhyeok/GR00T-Dreams/checkpoints/idm_allex_ego_0123_ae20_bsz64_step10k/checkpoint-10000

# Input Dataset (zero-padded actions)
DATASET=/storage1/sjw_dataset/dataset/Allex/idm/20260108_185259_cube_box_lerobot

# Output Directory (빈 문자열이면 input dataset을 덮어씀)
OUTPUT_DIR="/storage1/sjw_dataset/dataset/Allex/idm/20260108_185259_cube_box_lerobot_idm_ae20_bsz64_step10k"

# Inference settings
NUM_GPUS=1
BATCH_SIZE=16
NUM_WORKERS=8

# Observation indices: 비워두면 checkpoint의 action_horizon에서 자동 감지 [0, action_horizon]
# 수동 지정하려면: OBSERVATION_INDICES="0 20"
OBSERVATION_INDICES=""

# Sample episodes (set to empty or 0 to process all episodes)
MAX_EPISODES=""  # Process all episodes
# MAX_EPISODES=10  # Only process first 10 episodes for testing

# ============================================================
# Run IDM Inference
# ============================================================

echo "============================================================"
echo "IDM Action Dump for Allex Robot"
echo "============================================================"
echo "Checkpoint: ${CHECKPOINT}"
echo "Dataset: ${DATASET}"
echo "Output: ${OUTPUT_DIR:-'(overwrite input)'}"
echo "Observation indices: ${OBSERVATION_INDICES:-'(auto-detect from checkpoint)'}"
echo "Max episodes: ${MAX_EPISODES:-'all'}"
echo "GPUs: ${NUM_GPUS}, Batch size: ${BATCH_SIZE}"
echo "============================================================"

cd /sjw_alinlab3/home/suhyeok/GR00T-Dreams

# Build command with optional arguments
CMD="python IDM_dump/dump_idm_actions_allex.py \
    --checkpoint ${CHECKPOINT} \
    --dataset ${DATASET} \
    --num_gpus ${NUM_GPUS} \
    --batch_size ${BATCH_SIZE} \
    --num_workers ${NUM_WORKERS}"

# Add observation_indices if specified (otherwise auto-detect from checkpoint)
if [ -n "$OBSERVATION_INDICES" ]; then
    CMD="$CMD --observation_indices \"${OBSERVATION_INDICES}\""
fi

# Add output_dir if specified
if [ -n "$OUTPUT_DIR" ]; then
    CMD="$CMD --output_dir ${OUTPUT_DIR}"
fi

# Add max_episodes if specified
if [ -n "$MAX_EPISODES" ] && [ "$MAX_EPISODES" -gt 0 ] 2>/dev/null; then
    CMD="$CMD --max_episodes ${MAX_EPISODES}"
fi

echo "Running: $CMD"
eval $CMD

echo "============================================================"
echo "Done!"
echo "============================================================"

