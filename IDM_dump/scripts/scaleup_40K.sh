#!/bin/bash
# Step 2: 20K -> 40K (diversity 2.0x of 10K)
# Requires 20K to be built first (scaleup_20K.sh)
set -e

FULL="/storage1/sjw_dataset/dataset/neural_traj/open_gr1/2025_12_11_v1_1_lerobot_merged_prompt_revised"
BASE_10K="/storage1/sjw_dataset/dataset/neural_traj/open_gr1/2025_12_11_v1_1_lerobot_merged_prompt_revised_10K"
OUT_20K="/storage1/sjw_dataset/dataset/neural_traj/open_gr1/2025_12_11_v1_1_lerobot_merged_prompt_revised_10K_scaleup_20K"
OUT_40K="/storage1/sjw_dataset/dataset/neural_traj/open_gr1/2025_12_11_v1_1_lerobot_merged_prompt_revised_10K_scaleup_40K"
SEED=42
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=== 20K -> 40K (diversity 2.0x of 10K) ==="
python "${DIR}/scaleup_dataset.py" \
    --full-dataset "${FULL}" \
    --base-dataset "${OUT_20K}" \
    --ref-dataset "${BASE_10K}" \
    --target-size 40000 \
    --diversity-increase 1.0 \
    --output "${OUT_40K}" \
    --seed ${SEED}
