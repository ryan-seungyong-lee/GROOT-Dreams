#!/bin/bash
# Step 1: 10K -> 20K (diversity 1.5x of 10K) for rldx_g0 v2v+i2i dataset
set -e

FULL="/storage1/sjw_dataset/dataset/neural_traj/rldx_g0_vla_pretrain_v2v+i2i_aug_v1_256x256_bug_fixed"
BASE_10K="/storage1/sjw_dataset/dataset/neural_traj/rldx_g0_vla_pretrain_v2v+i2i_aug_v1_256x256_bug_fixed_10k"
OUT_20K="/storage1/sjw_dataset/dataset/neural_traj/rldx_g0_vla_pretrain_v2v+i2i_aug_v1_256x256_bug_fixed_10k_scaleup_20K"
SEED=42
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=== 10K -> 20K (diversity 1.5x of 10K, V2V:I2I=1:2) ==="
python "${DIR}/scaleup_dataset.py" \
    --full-dataset "${FULL}" \
    --base-dataset "${BASE_10K}" \
    --target-size 20000 \
    --diversity-increase 0.5 \
    --v2v-ratio "1:2" \
    --output "${OUT_20K}" \
    --seed ${SEED}
