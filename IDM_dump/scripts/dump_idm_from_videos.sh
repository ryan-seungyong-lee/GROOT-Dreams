#!/bin/bash
#SBATCH --job-name="idm-from-videos"
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=sjw_alinlab
#SBATCH --exclude=worker-node1001
#SBATCH --output=slurm_out/%x_%j.out
#SBATCH --error=slurm_out/%x_%j.err

# Note: This script can be run directly (bash dump_idm_from_videos.sh) or via sbatch

# ============================================================
# IDM Action Extraction from Video Files
# ============================================================
# This script extracts IDM-predicted actions from video files
# and creates a LeRobot format dataset.
#
# Input: Video files (will be resized to 224x224)
# Output: LeRobot dataset with:
#   - videos/ : 224x224 resized videos (h264, yuv420p)
#   - data/ : parquet with zero-padded state + IDM actions
#   - meta/ : info.json, modality.json, episodes.jsonl, tasks.jsonl
# ============================================================

export NO_ALBUMENTATIONS_UPDATE=1
export PYTHONPATH="/sjw_alinlab3/home/suhyeok/GR00T-Dreams:$PYTHONPATH"

CONDA_PATH="/sjw_alinlab3/home/suhyeok/miniconda3"    
source "$CONDA_PATH/etc/profile.d/conda.sh"
conda activate cosmos-predict1

# ============================================================
# Configuration - MODIFY THESE
# ============================================================

# IDM checkpoint path
CHECKPOINT="/sjw_alinlab3/home/suhyeok/GR00T-Dreams/checkpoints/idm_allex_ego_0123_ae20_bsz64_step10k/checkpoint-10000"

# Option 1: Single video file
VIDEO=""  # Path to single video file

# Option 2: Directory with video files (*.mp4)
# Each video = one episode, observation = frame[t] + frame[t+action_horizon]
VIDEO_DIR="/storage1/sjw_dataset/dataset/Allex/generated/allex_vdm_ckpt-6k"

# Output directory
OUTPUT_DIR="/storage1/sjw_dataset/dataset/Allex/idm/neural_sample"

# Task description
TASK_DESCRIPTION="IDM predicted manipulation task"

# Inference settings
BATCH_SIZE=16
GPU_ID=0

# Observation indices (space-separated, e.g., "0 16" or "0 20")
# Default: auto-detect based on video FPS (1 second = fps frames)
#   - For 16fps video: "0 16" (1 second span)
#   - For 20fps video: "0 20" (1 second span)
# Manual override examples:
#   - For 16fps: "0 16"
#   - For 20fps: "0 20"
#   - For ae20 model (20fps): "0 20"
#   - For ae40 model (20fps): "0 40"
OBSERVATION_INDICES=""  # Leave empty for FPS-based auto-detect

# ============================================================
# Run
# ============================================================

cd /sjw_alinlab3/home/suhyeok/GR00T-Dreams

# Create slurm output directory if it doesn't exist
mkdir -p slurm_out

# Build command
CMD="python IDM_dump/dump_idm_from_videos.py"
CMD="$CMD --checkpoint ${CHECKPOINT}"
CMD="$CMD --output_dir ${OUTPUT_DIR}"
CMD="$CMD --task_description \"${TASK_DESCRIPTION}\""
CMD="$CMD --batch_size ${BATCH_SIZE}"
CMD="$CMD --gpu_id ${GPU_ID}"

# Add observation indices if specified
if [ -n "$OBSERVATION_INDICES" ]; then
    CMD="$CMD --observation_indices \"${OBSERVATION_INDICES}\""
fi

# Add video source
if [ -n "$VIDEO_DIR" ]; then
    CMD="$CMD --video_dir ${VIDEO_DIR}"
elif [ -n "$VIDEO" ]; then
    CMD="$CMD --video ${VIDEO}"
else
    echo "ERROR: Either VIDEO_DIR or VIDEO must be set!"
    exit 1
fi

echo "============================================================"
echo "IDM Action Extraction from Videos"
echo "============================================================"
echo "Checkpoint: ${CHECKPOINT}"
if [ -n "$VIDEO_DIR" ]; then
    echo "Video Dir: ${VIDEO_DIR}"
else
    echo "Video: ${VIDEO}"
fi
if [ -n "$OBSERVATION_INDICES" ]; then
    echo "Observation Indices: ${OBSERVATION_INDICES} (user-specified)"
else
    echo "Observation Indices: auto-detect based on video FPS (1 second = fps frames)"
    echo "  - 16fps video → [0, 16] (1 second span)"
    echo "  - 20fps video → [0, 20] (1 second span)"
fi
echo "Output Dir: ${OUTPUT_DIR}"
echo "Task: ${TASK_DESCRIPTION}"
echo "Batch Size: ${BATCH_SIZE}"
echo "GPU ID: ${GPU_ID}"
echo "============================================================"
echo ""
echo "Running: $CMD"
echo ""

eval $CMD

echo ""
echo "============================================================"
echo "Done!"
echo "============================================================"

