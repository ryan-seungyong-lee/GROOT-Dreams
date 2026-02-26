# python IDM_dump/split_lerobot_dataset.py \
#     --input /virtual_lab/sjw_alinlab/dataset/ActionNet/gr1_actionnet_lerobot_15fps \
#     --output /virtual_lab/sjw_alinlab/dataset/ActionNet/gr1_actionnet_lerobot_15fps_split \
#     --num-splits 2

python IDM_dump/validate_split_dataset.py \
    --original_dir /virtual_lab/sjw_alinlab/dataset/ActionNet/gr1_actionnet_lerobot_15fps \
    --split_base /virtual_lab/sjw_alinlab/dataset/ActionNet/gr1_actionnet_lerobot_15fps_split \
    --num_splits 2