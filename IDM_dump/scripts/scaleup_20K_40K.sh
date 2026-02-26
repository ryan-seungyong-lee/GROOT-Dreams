#!/bin/bash
# Scale up: 10K ⊂ 20K ⊂ 40K (chained inclusion, with video copy)
# Diversity target always relative to 10K:
#   10K = 1.0x, 20K = 1.5x, 40K = 2.0x
set -e

FULL="/storage1/sjw_dataset/dataset/neural_traj/open_gr1/2025_12_11_v1_1_lerobot_merged_prompt_revised"
BASE_10K="/storage1/sjw_dataset/dataset/neural_traj/open_gr1/2025_12_11_v1_1_lerobot_merged_prompt_revised_10K"
OUT_20K="/storage1/sjw_dataset/dataset/neural_traj/open_gr1/2025_12_11_v1_1_lerobot_merged_prompt_revised_10K_scaleup_20K"
OUT_40K="/storage1/sjw_dataset/dataset/neural_traj/open_gr1/2025_12_11_v1_1_lerobot_merged_prompt_revised_10K_scaleup_40K"
SEED=42
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=== Step 1: 10K -> 20K (diversity 1.5x of 10K) ==="
python "${DIR}/scaleup_dataset.py" \
    --full-dataset "${FULL}" \
    --base-dataset "${BASE_10K}" \
    --target-size 20000 \
    --diversity-increase 0.5 \
    --output "${OUT_20K}" \
    --seed ${SEED}

echo ""
echo "=== Step 2: 20K -> 40K (diversity 2.0x of 10K) ==="
python "${DIR}/scaleup_dataset.py" \
    --full-dataset "${FULL}" \
    --base-dataset "${OUT_20K}" \
    --ref-dataset "${BASE_10K}" \
    --target-size 40000 \
    --diversity-increase 1.0 \
    --output "${OUT_40K}" \
    --seed ${SEED}

echo ""
echo "=== Final Summary ==="
python -c "
import json, sys
sys.path.insert(0, '${DIR}')
from scaleup_dataset import compute_stats, _SKILL_ORDER, read_jsonl

datasets = {
    '10K': '${BASE_10K}',
    '20K': '${OUT_20K}',
    '40K': '${OUT_40K}',
}

results = {}
for name, path in datasets.items():
    eps = read_jsonl(path + '/meta/episodes.jsonl')
    sk, div, _ = compute_stats(eps)
    tot = sum(sk.values())
    results[name] = (tot, sk, div)

# Reference = 10K
ref_div = results['10K'][2]

print()
print('=' * 95)
print(f\"{'Dataset':<8} {'Episodes':>10} {'4-way Combos':>14} {'vs 10K':>10}  Skill Distribution\")
print('-' * 95)
for name in ['10K','20K','40K']:
    tot, sk, div = results[name]
    ratio = div / ref_div if ref_div else 0
    top_skills = ', '.join(f'{s}={sk.get(s,0)/tot*100:.1f}%' for s in _SKILL_ORDER if sk.get(s,0)/tot*100 >= 0.5)
    print(f'{name:<8} {tot:>10,} {div:>14,} {ratio:>9.2f}x  {top_skills}')
print('=' * 95)

vids = {}
for name, path in datasets.items():
    eps = read_jsonl(path + '/meta/episodes.jsonl')
    vids[name] = set(int(e['video_id']) for e in eps if e.get('video_id') is not None)

ok_20 = vids['10K'].issubset(vids['20K'])
ok_40 = vids['20K'].issubset(vids['40K'])
print(f'Inclusion: 10K ⊂ 20K = {ok_20}, 20K ⊂ 40K = {ok_40}')
print()
"
