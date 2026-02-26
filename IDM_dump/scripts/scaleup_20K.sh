#!/bin/bash
# Step 1: 10K -> 20K (diversity 1.5x of 10K)
set -e

FULL="/storage1/sjw_dataset/dataset/neural_traj/open_gr1/2025_12_11_v1_1_lerobot_merged_prompt_revised"
BASE_10K="/storage1/sjw_dataset/dataset/neural_traj/open_gr1/2025_12_11_v1_1_lerobot_merged_prompt_revised_10K"
OUT_20K="/storage1/sjw_dataset/dataset/neural_traj/open_gr1/2025_12_11_v1_1_lerobot_merged_prompt_revised_10K_scaleup_20K"
SEED=42
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=== 10K -> 20K (diversity 1.5x of 10K) ==="
python "${DIR}/scaleup_dataset.py" \
    --full-dataset "${FULL}" \
    --base-dataset "${BASE_10K}" \
    --target-size 20000 \
    --diversity-increase 0.5 \
    --output "${OUT_20K}" \
    --seed ${SEED}
