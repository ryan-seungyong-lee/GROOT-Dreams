#!/bin/bash
#SBATCH --job-name="dreamgen-idm-allex-ego"
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=sjw_alinlab
#SBATCH --exclude=worker-node1001
#SBATCH --output=slurm_out/%x_%j.out
#SBATCH --error=slurm_out/%x_%j.err


export WANDB_PROJECT=dreamgen-idm
export NO_ALBUMENTATIONS_UPDATE=1
export PYTHONPATH="/sjw_alinlab3/home/suhyeok/GR00T-Dreams:$PYTHONPATH"

CONDA_PATH="/sjw_alinlab3/home/suhyeok/miniconda3"    
source "$CONDA_PATH/etc/profile.d/conda.sh"
conda activate cosmos-predict1

DATASET_PATH=/storage1/sjw_dataset/dataset/Allex/real/idm_train_data_teleop_added
OUTPUT_DIR=/sjw_alinlab3/home/suhyeok/GR00T-Dreams/checkpoints/idm_allex_ego_0124_add_teleop_ae20_bsz64_step20k

torchrun scripts/idm_training.py \
  --dataset-path ${DATASET_PATH} \
  --data-config allex_thetwo_ck40_ego_ae20 \
  --embodiment_tag allex_ego \
  --output-dir ${OUTPUT_DIR} \
  --num-gpus 2 \
  --save-steps 5000 \
  --max-steps 20000 \
  --batch-size 32

