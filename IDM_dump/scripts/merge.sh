
export PYTHONPATH="/sjw_alinlab3/home/suhyeok/GR00T-Dreams:$PYTHONPATH"
# python IDM_dump/merge_lerobot_datasets.py \
#   --inputs /storage1/oneuniverse_simul/gen*_lerobot_merged \
#   --output /storage1/oneuniverse_simul/gen_all_lerobot_merged

python IDM_dump/validate_merged_dataset.py \
    --merged_dir /storage1/oneuniverse_simul/gen_all_lerobot_merged \
    --inputs /storage1/oneuniverse_simul/gen?_lerobot_merged \

# 1번: 2025_10_28_v1_idm 데이터셋 수합
# python IDM_dump/merge_lerobot_datasets.py \
#   --pattern \
#     "/storage1/sjw_dataset/dataset/neural_traj/open_gr1/2026_0220_i2i_idm_300/dataset_gr1_*" \
#   --output /storage1/sjw_dataset/dataset/neural_traj/open_gr1/2026_0220_i2i_aug_lerobot_merged_prompt_revised \
#   --embodiment gr1_unified \
#   --use_192x320

# python IDM_dump/merge_lerobot_datasets.py \
#   --pattern "/storage1/sjw_dataset/dataset/neural_traj/open_gr1/2026_0220_i2i_idm_300/dataset_gr1_*" \
#   --output /storage1/sjw_dataset/dataset/neural_traj/open_gr1/2026_0220_i2i_aug_lerobot_merged_prompt_revised_192_320 \
#   --embodiment gr1_unified \
#   --use_192x320 \
#   --videos_192x320_dir /storage1/sjw_dataset/dataset/neural_traj/open_gr1/2026_02_11_i2i_i2v_aug_192_320
# read -p "1번 merge 완료. Enter를 눌러 다음 단계로 진행하거나 Ctrl+C로 중단하세요..."


# 2번: 2025_12_01_v1_1 데이터셋 수합
# python IDM_dump/merge_lerobot_datasets.py \
#   --pattern \
#     "/virtual_lab/sjw_alinlab/dataset/neural_traj/open_gr1/dev/lerobot_idm_final_v1_1/dataset_gr1_*" \
#   --output /storage1/sjw_dataset/dataset/neural_traj/open_gr1/2025_12_01_v1_1_lerobot_merged_prompt_revised \
#   --embodiment gr1_unified \
# read -p "2번 merge 완료. Enter를 눌러 다음 단계로 진행하거나 Ctrl+C로 중단하세요..."

# 3번: 2025_12_11_v1_1 데이터셋 수합 (필터링 적용)
# python IDM_dump/merge_lerobot_datasets.py \
#   --pattern \
#     "/virtual_lab/sjw_alinlab/dataset/neural_traj/open_gr1/2025_12_11_v1_1_lerobot_idm_final_32/dataset_gr1_*" \
#   --output /storage1/sjw_dataset/dataset/neural_traj/open_gr1/2025_12_11_v1_1_lerobot_merged_prompt_revised \
#   --embodiment gr1_unified \
#   --filter_videos /virtual_lab/sjw_alinlab/byungjun_alinlab/neural/artifacts/1222_illogical/illogical_videos_list.json \

# use_192x320
# read -p "3번 merge 완료. Enter를 눌러 다음 단계로 진행하거나 Ctrl+C로 중단하세요..."

# 4번: 최종 통합 수합
# python IDM_dump/merge_lerobot_datasets.py \
#   --inputs /sjw_alinlab3/home/suhyeok/GR00T-Dreams/IDM_dump/data/cosmos_pour_can_into_bowl_lerobot_idm_indice_04_best_of_n \
#   --output /storage1/sjw_dataset/dataset/Allex/idm/cosmos_generated_neural_v1_best_of_n_pour \
#   --embodiment allex_ego
# read -p "4번 merge 완료. 모든 작업이 완료되었습니다!"

# Validate with explicit inputs
# python IDM_dump/validate_merged_dataset.py \
#     --merged_dir /storage1/sjw_dataset/dataset/Allex/idm/cosmos_generated_neural_v1_best_of_n_pour \
#     --inputs /sjw_alinlab3/home/suhyeok/GR00T-Dreams/IDM_dump/data/cosmos_pour_can_into_bowl_lerobot_idm_indice_04_best_of_n \
#     --embodiment allex_ego

# (Optional) Recompute stats.json for merged dataset
# This recalculates mean, std, min, max, q01, q99 for observation.state and action
# python IDM_dump/compute_norm_stats.py \
#     --dataset-root /storage1/sjw_dataset/dataset/Allex/real/idm_train_data_teleop_added
