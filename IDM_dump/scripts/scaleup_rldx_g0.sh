#!/bin/bash
# Scale up rldx_g0 v2v+i2i dataset: 10K -> 20K -> 40K
# Priority: 1) Inclusion (10K⊂20K⊂40K) + exact count
#           2) Diversity (1.0x -> 1.5x -> 2.0x)
#           3) V2V:I2I = 1:2 (soft)
#           4) Skill distribution (soft)
set -e

FULL="/storage1/sjw_dataset/dataset/neural_traj/rldx_g0_vla_pretrain_v2v+i2i_aug_v1_256x256_bug_fixed"
BASE_10K="/storage1/sjw_dataset/dataset/neural_traj/rldx_g0_vla_pretrain_v2v+i2i_aug_v1_256x256_bug_fixed_10k"
OUT_20K="/storage1/sjw_dataset/dataset/neural_traj/rldx_g0_vla_pretrain_v2v+i2i_aug_v1_256x256_bug_fixed_10k_scaleup_20K"
OUT_40K="/storage1/sjw_dataset/dataset/neural_traj/rldx_g0_vla_pretrain_v2v+i2i_aug_v1_256x256_bug_fixed_10k_scaleup_40K"
SEED=42
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=========================================="
echo "=== Step 1: 10K -> 20K (div 1.5x, V2V:I2I=1:2) ==="
echo "=========================================="
python "${DIR}/scaleup_dataset.py" \
    --full-dataset "${FULL}" \
    --base-dataset "${BASE_10K}" \
    --target-size 20000 \
    --diversity-increase 0.5 \
    --v2v-ratio "1:2" \
    --output "${OUT_20K}" \
    --seed ${SEED}

echo ""
echo "=========================================="
echo "=== Step 2: 20K -> 40K (div 2.0x, V2V:I2I=1:2) ==="
echo "=========================================="
python "${DIR}/scaleup_dataset.py" \
    --full-dataset "${FULL}" \
    --base-dataset "${OUT_20K}" \
    --ref-dataset "${BASE_10K}" \
    --target-size 40000 \
    --diversity-increase 1.0 \
    --v2v-ratio "1:2" \
    --output "${OUT_40K}" \
    --seed ${SEED}

echo ""
echo "=========================================="
echo "=== DONE ==="
echo "=========================================="
