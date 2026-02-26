#!/bin/bash
#SBATCH --job-name="actionnet_preprocess"
#SBATCH --gpus=1
#SBATCH --array=0
#SBATCH --nodes=1
#SBATCH --partition=sjw_alinlab
#SBATCH --output=/virtual_lab/sjw_alinlab/suhyeok/GR00T-Dreams/slurm_results/%x_%A_%a.out
#SBATCH --error=/virtual_lab/sjw_alinlab/suhyeok/GR00T-Dreams/slurm_results/%x_%A_%a.err

# python IDM_dump/split_video_instruction.py \
#     --source_dir gr1_real_video \
#     --output_dir "IDM_dump/data/gr1_real_data" \

# python IDM_dump/preprocess_video.py \
#     --src_dir "IDM_dump/data/gr1_real_data" \
#     --dst_dir "IDM_dump/data/gr1_real_data_split" \
#     --dataset gr1 

# python IDM_dump/preprocess_video.py \
#     --src_dir "/virtual_lab/sjw_alinlab/suhyeok/cosmos-predict2/datasets/opensrc_gr1/actionnet" \
#     --dst_dir "IDM_dump/data/gr1_actionnet_split" \
#     --dataset gr1 \

# python IDM_dump/raw_to_lerobot.py \
#     --input_dir "IDM_dump/data/gr1_real_data_split" \
#     --output_dir "IDM_dump/data/gr1_real_unified_data" \
#     --video_key observation.images.ego_view \
#     --fps 20 -> 16

python IDM_dump/actionnet_to_lerobot.py \
    --raw-dir "IDM_dump/data/gr1_actionnet_split" \
    --hdf5-dir "/virtual_lab/dataset/storage1/ActionNet" \
    --output-dir "/virtual_lab/sjw_alinlab/dataset/ActionNet/gr1_actionnet_lerobot_15fps" \
    --num-workers 16 \
    --stride 2

# export PYTHONPATH="/virtual_lab/sjw_alinlab/suhyeok/GR00T-Dreams:$PYTHONPATH"
# python IDM_dump/dump_idm_actions.py \
#     --checkpoint "seonghyeonye/IDM_gr1" \
#     --dataset "IDM_dump/data/gr1_real_unified_data" \
#     --output_dir "IDM_dump/data/gr1_real_unified_data_idm_test" \
#     --num_gpus 1 \
#     --video_indices "0 8" 

# export PYTHONPATH="/virtual_lab/sjw_alinlab/suhyeok/GR00T-Dreams:$PYTHONPATH"

# source /virtual_lab/sjw_alinlab/suhyeok/miniconda3/etc/profile.d/conda.sh
# conda activate cosmos-predict1

# python IDM_dump/rollout_idm.py \
#     --dataset "/virtual_lab/sjw_alinlab/dataset/neural_traj/open_gr1/lerobot_idm_final/dataset_gr1_1" \
#     --output_dir "/virtual_lab/sjw_alinlab/dataset/neural_traj/open_gr1/lerobot_idm_pickle/dataset_gr1_1"

# Step 2: Rollout actions in simulation and generate videos
# /virtual_lab/sjw_alinlab/john/conda_envs/robocasa/bin/python IDM_dump/idm_gr1_rollout_try_cosmos.py \
#     --pickle_file "/virtual_lab/sjw_alinlab/dataset/neural_traj/open_gr1/lerobot_idm_pickle/dataset_gr1_1/actions_0.pkl" \
#     --video_output_dir "/virtual_lab/sjw_alinlab/dataset/neural_traj/open_gr1/lerobot_idm_replay/dataset_gr1_1"

