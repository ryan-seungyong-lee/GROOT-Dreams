"""
Dump IDM predicted actions for Allex robot dataset.

IDM Configuration (from checkpoint):
- action_dim: 48
- action_horizon: 40 (at 20fps = 2 seconds of future actions)
- video: camera_ego_left (224x224)

Usage:
    python dump_idm_actions_allex.py \
        --checkpoint /path/to/idm_allex_ego_0121 \
        --dataset /path/to/lerobot_dataset \
        --output_dir /path/to/output \
        --num_gpus 4 \
        --video_indices "0 40"
"""

import argparse
import json
import multiprocessing as mp
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from tianshou.data import Batch

from gr00t.model.idm import IDM
from gr00t.data.dataset import LeRobotSingleDataset, ModalityConfig
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.transform import (
    VideoToTensor,
    VideoCrop,
    VideoResize,
    VideoToNumpy,
    StateActionToTensor,
    StateActionTransform,
    ConcatTransform,
    ComposedModalityTransform,
)
from gr00t.model.transforms_idm import GR00TIDMTransform
from gr00t.utils.video import get_all_frames_and_timestamps


# Allex IDM specific configuration
ALLEX_ACTION_DIM = 48
ALLEX_FPS = 20  # 20 fps
# NOTE: ALLEX_ACTION_HORIZON is now dynamically loaded from checkpoint


# Keys must match training config (allex_thetwo_ck40_ego_config in data_config_idm.py)
ALLEX_VIDEO_KEYS = ["video.camera_ego_left"]
ALLEX_STATE_KEYS = [
    "state.right_arm_joints",
    "state.left_arm_joints",
    "state.right_hand_joints",
    "state.left_hand_joints",
    "state.neck_joints",
    "state.waist_joints",
]
ALLEX_ACTION_KEYS = [
    "action.right_arm_joints",
    "action.left_arm_joints",
    "action.right_hand_joints",
    "action.left_hand_joints",
    "action.neck_joints",
    "action.waist_joints",
]


def create_allex_modality_config(observation_indices, action_horizon):
    """
    Create modality config for Allex IDM inference.
    
    Must match the training config in data_config_idm.py (allex_thetwo_ck40_ego_config).
    
    NOTE: IDM 모델은 실제로 Video만 사용하고 State는 사용하지 않음!
    - Video: observation_indices (e.g., [0, 20]) - 2 프레임
    - State: [0] - 1 프레임 (모델에서 사용 안됨, 데이터 로딩용)
    - Action: list(range(action_horizon)) - action_horizon개 action
    
    Args:
        observation_indices: List of observation frame indices (e.g., [0, 20])
        action_horizon: Number of action steps to predict (loaded from checkpoint)
    """
    
    # Video modality - observation_indices (2 frames for IDM)
    video_modality = ModalityConfig(
        delta_indices=observation_indices,
        modality_keys=ALLEX_VIDEO_KEYS,
    )
    
    # State modality - [0] only (NOT used by IDM model, just for data loading)
    state_modality = ModalityConfig(
        delta_indices=[0],
        modality_keys=ALLEX_STATE_KEYS,
    )
    
    # Action modality - action_indices = list(range(action_horizon))
    action_modality = ModalityConfig(
        delta_indices=list(range(action_horizon)),
        modality_keys=ALLEX_ACTION_KEYS,
    )
    
    return {
        "video": video_modality,
        "state": state_modality,
        "action": action_modality,
    }


def create_allex_transforms(video_keys, state_keys, action_keys, action_horizon, action_dim=48):
    """
    Create transforms for Allex IDM inference.
    
    This should match the training transforms in data_config_idm.py (allex_thetwo_ck40_ego_config),
    but WITHOUT data augmentation (VideoColorJitter).
    
    NOTE: Normalization statistics will be loaded from global metadata (IDM training dataset stats)
    via dataset.set_transforms_metadata() during dataset initialization.
    
    Args:
        video_keys: List of video keys
        state_keys: List of state keys
        action_keys: List of action keys
        action_horizon: Number of action steps (loaded from checkpoint)
        action_dim: Action dimension (default: 48)
    """
    
    transforms_list = [
        # Video transforms (same as training, minus ColorJitter)
        VideoToTensor(apply_to=video_keys),
        VideoCrop(apply_to=video_keys, scale=0.95),  # Same as training
        VideoResize(apply_to=video_keys, height=224, width=224, interpolation="linear"),
        # No VideoColorJitter for inference
        VideoToNumpy(apply_to=video_keys),
        
        # State transforms (q99 normalization, same as training)
        StateActionToTensor(apply_to=state_keys),
        StateActionTransform(
            apply_to=state_keys,
            normalization_modes={key: "q99" for key in state_keys},
        ),
        
        # Action transforms (q99 normalization, same as training)
        # Statistics will be loaded from global metadata via set_transforms_metadata
        StateActionToTensor(apply_to=action_keys),
        StateActionTransform(
            apply_to=action_keys,
            normalization_modes={key: "q99" for key in action_keys},
        ),
        
        # Concat transforms
        ConcatTransform(
            video_concat_order=video_keys,
            state_concat_order=state_keys,
            action_concat_order=action_keys,
        ),
        
        # Model-specific transform
        # NOTE: state_horizon은 GR00TIDMTransform에서 실제로 사용되지 않음
        GR00TIDMTransform(
            state_horizon=1,  # state는 1개만 (실제로 모델에서 사용 안됨)
            action_horizon=action_horizon,  # Dynamically loaded from checkpoint
            max_state_dim=64,
            max_action_dim=action_dim,
        ),
    ]
    
    return ComposedModalityTransform(transforms=transforms_list)


def collate_fn(features_list, device):
    """Collate features into a batch."""
    batch_dict = {}
    keys = features_list[0].keys()
    for key in keys:
        if key in ["images", "view_ids"]:
            vals = [f[key] for f in features_list]
            batch_dict[key] = torch.as_tensor(np.concatenate(vals), device=device)
        else:
            vals = [f[key] for f in features_list]
            batch_dict[key] = torch.as_tensor(np.stack(vals), device=device)
    return batch_dict


def get_step_data(dataset, trajectory_id, base_index):
    """Get state/action data for a single step."""
    data = {}
    dataset.curr_traj_data = dataset.get_trajectory_data(trajectory_id)
    
    for modality in dataset.modality_keys:
        for key in dataset.modality_keys[modality]:
            if modality == "video":
                pass  # Video handled separately
            elif modality == "state" or modality == "action":
                data[key] = dataset.get_state_or_action(trajectory_id, modality, key, base_index)
            elif modality == "language":
                data[key] = dataset.get_language(trajectory_id, key, base_index)
    
    return data


def save_trajectory_data(trajectory_data, dataset, trajectory_id, output_dir):
    """Save trajectory data to parquet file."""
    chunk_index = dataset.get_episode_chunk(trajectory_id)
    chunk_dir = f"chunk-{chunk_index:03d}"
    os.makedirs(os.path.join(output_dir, "data", chunk_dir), exist_ok=True)
    
    episode_id = f"episode_{int(trajectory_id):06d}"
    output_file_path = os.path.join(output_dir, "data", chunk_dir, f"{episode_id}.parquet")
    trajectory_data.to_parquet(output_file_path)


def worker_func(
    gpu_id: int,
    traj_id_list: list,
    checkpoint_path: str,
    dataset_path: str,
    output_dir: str,
    batch_size: int,
    num_workers: int,
    observation_indices: list,
    dataset,
):
    """Worker function that runs on a single GPU."""
    
    print(f"[GPU {gpu_id}] Loading model from: {checkpoint_path}")
    model = IDM.from_pretrained(checkpoint_path)
    model.requires_grad_(False)
    model.eval()
    device = torch.device(f"cuda:{gpu_id}")
    model.to(device)
    
    # Get action horizon from model config
    action_horizon = model.config.action_horizon
    print(f"[GPU {gpu_id}] Action horizon: {action_horizon}")
    
    for tid_idx, tid in enumerate(tqdm(traj_id_list, desc=f"GPU {gpu_id}", position=gpu_id)):
        action_dict = {}
        traj_data = dataset.get_trajectory_data(tid)
        length = len(traj_data)
        
        debug_print = (tid_idx == 0)
        
        if debug_print:
            print(f"\n[GPU {gpu_id}] Processing trajectory {tid} with {length} frames")
            print(f"[GPU {gpu_id}] Observation indices (passed): {observation_indices}")
            print(f"[GPU {gpu_id}] Dataset delta_indices:")
            for key in dataset.modality_keys["video"]:
                print(f"  {key}: {dataset.delta_indices[key]}")
            print(f"[GPU {gpu_id}] Action horizon: {action_horizon} (at {ALLEX_FPS}fps = {action_horizon/ALLEX_FPS:.1f}s)")
        
        # Load video data
        video_data = {}
        for key in dataset.modality_keys["video"]:
            video_path = dataset.get_video_path(tid, key.replace("video.", ""))
            video_backend = dataset.video_backend
            video_backend_kwargs = dataset.video_backend_kwargs
            frames, whole_indices = get_all_frames_and_timestamps(
                video_path.as_posix(), video_backend, video_backend_kwargs
            )
            video_data[key] = (frames, whole_indices)
            if debug_print:
                print(f"[GPU {gpu_id}] Loaded video {key}: {frames.shape}")
        
        def load_and_transform_step(step_idx, video_data):
            """Load and transform data for a single step."""
            step_data = get_step_data(dataset, tid, step_idx)
            timestamp = dataset.curr_traj_data["timestamp"].to_numpy()
            traj_length = len(timestamp)
            
            for key in video_data:
                frames, whole_indices = video_data[key]
                # Use dataset.delta_indices instead of observation_indices directly
                # This ensures we use the same indices as configured in transforms
                step_indices = np.array(dataset.delta_indices[key]) + step_idx
                # Clamp step_indices to valid range BEFORE accessing timestamp
                step_indices = np.maximum(step_indices, 0)
                step_indices = np.minimum(step_indices, traj_length - 1)
                indices = np.array([
                    np.where(np.isclose(whole_indices, val))[0][0] 
                    for val in timestamp[step_indices]
                ])
                step_data[key] = frames[indices]
            
            output = dataset.transforms(step_data)
            return output
        
        # Load all features in parallel
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(load_and_transform_step, step_idx, video_data)
                for step_idx in range(length)
            ]
            all_features = [future.result() for future in futures]
        
        # Process in batches
        for start_idx in range(0, length, batch_size):
            end_idx = min(start_idx + batch_size, length)
            step_ids = list(range(start_idx, end_idx))
            
            batch_features = all_features[start_idx:end_idx]
            batch_dict = collate_fn(batch_features, device)
            
            if debug_print and start_idx == 0:
                print(f"\n[GPU {gpu_id}] === Batch Dict (Model Input) ===")
                print(f"[GPU {gpu_id}] Keys: {list(batch_dict.keys())}")
                for key, val in batch_dict.items():
                    if isinstance(val, torch.Tensor):
                        if val.dtype in [torch.float32, torch.float64, torch.float16]:
                            print(f"  {key}: shape={val.shape}, dtype={val.dtype}, min={val.min():.6f}, max={val.max():.6f}, mean={val.mean():.6f}")
                        else:
                            print(f"  {key}: shape={val.shape}, dtype={val.dtype}, min={val.min().item()}, max={val.max().item()}")
            
            with torch.no_grad():
                out = model.get_action(batch_dict)
            
            pred_actions = out["action_pred"].cpu()
            
            if debug_print and start_idx == 0:
                print(f"\n[GPU {gpu_id}] === Model Output (BEFORE unapply) ===")
                print(f"[GPU {gpu_id}] Model output shape: {pred_actions.shape}")
                print(f"[GPU {gpu_id}] Model output dtype: {pred_actions.dtype}")
                print(f"[GPU {gpu_id}] Model output value range: min={pred_actions.min():.6f}, max={pred_actions.max():.6f}, mean={pred_actions.mean():.6f}, std={pred_actions.std():.6f}")
                print(f"[GPU {gpu_id}] Non-zero values: {torch.count_nonzero(pred_actions).item()} / {pred_actions.numel()}")
                if torch.allclose(pred_actions, torch.zeros_like(pred_actions)):
                    print(f"[GPU {gpu_id}] WARNING: Model output is all zeros!")
                else:
                    # Show sample values
                    non_zero_mask = ~torch.isclose(pred_actions, torch.zeros_like(pred_actions))
                    if torch.any(non_zero_mask):
                        non_zero_values = pred_actions[non_zero_mask]
                        print(f"[GPU {gpu_id}] Sample non-zero values (first 10): {non_zero_values.flatten()[:10].tolist()}")
            
            # Unapply transforms to get original scale
            # ConcatTransform.unapply will split "action" into individual parts (action.right_arm_joints, etc.)
            # The unapply process:
            # 1. GR00TIDMTransform.unapply: passes through (no change)
            # 2. ConcatTransform.unapply: splits "action" into parts (action.right_arm_joints, etc.)
            # 3. StateActionTransform.unapply: denormalizes each part using q99 statistics
            # 4. StateActionToTensor.unapply: converts to numpy
            if debug_print and start_idx == 0:
                print(f"\n[GPU {gpu_id}] === Before unapply ===")
                print(f"  Input shape: {pred_actions.shape}, dtype: {pred_actions.dtype}")
                print(f"  Value range: min={pred_actions.min():.6f}, max={pred_actions.max():.6f}")
            
            pred_actions = dataset.transforms.unapply(Batch(action=pred_actions))
            
            if debug_print and start_idx == 0:
                print(f"[GPU {gpu_id}] Unapplied pred_actions keys: {list(pred_actions.keys())}")
                for key, val in pred_actions.items():
                    if isinstance(val, (torch.Tensor, np.ndarray)):
                        print(f"  {key}: shape={val.shape}, dtype={val.dtype}, min={val.min():.6f}, max={val.max():.6f}")
            
            # Load modality.json for action part indices
            modality_json_path = os.path.join(dataset_path, 'meta', 'modality.json')
            with open(modality_json_path, 'r') as f:
                modality_config = json.load(f)
            
            action_parts = modality_config.get('action', {})
            
            if debug_print and start_idx == 0:
                print(f"[GPU {gpu_id}] Action parts from modality.json: {list(action_parts.keys())}")
                for part, indices in action_parts.items():
                    print(f"  {part}: start={indices.get('start', 0)}, end={indices.get('end', 0)}")
            
            # Calculate total action dimension
            total_dim = max(indices.get('end', 0) for indices in action_parts.values())
            
            if debug_print and start_idx == 0:
                print(f"[GPU {gpu_id}] Total action dimension: {total_dim}")
            
            # Check if we have action horizon dimension
            # ConcatTransform.unapply splits action into parts, so we check the first part's shape
            sample_key = list(pred_actions.keys())[0]
            sample_val = pred_actions[sample_key]
            if isinstance(sample_val, torch.Tensor):
                sample_val = sample_val.cpu().numpy()
            
            has_horizon = len(sample_val.shape) == 3
            if has_horizon:
                current_batch_size, pred_horizon, _ = sample_val.shape
            else:
                current_batch_size = sample_val.shape[0]
                pred_horizon = 1
            
            if debug_print and start_idx == 0:
                print(f"[GPU {gpu_id}] has_horizon: {has_horizon}, pred_horizon: {pred_horizon}, batch_size: {current_batch_size}")
            
            # Reconstruct full action vector from parts
            # ConcatTransform.unapply always splits action into parts, so we always need to reconstruct
            if has_horizon:
                final_actions = np.zeros((current_batch_size, pred_horizon, total_dim), dtype=np.float32)
                
                for part, indices in action_parts.items():
                    action_key = f'action.{part}'
                    if action_key in pred_actions:
                        val = pred_actions[action_key]
                        if isinstance(val, torch.Tensor):
                            val = val.cpu().numpy()
                        start = indices.get('start', 0)
                        end = indices.get('end', 0)
                        final_actions[:, :, start:end] = val
                    else:
                        if debug_print and start_idx == 0:
                            print(f"[GPU {gpu_id}] WARNING: {action_key} not found in pred_actions!")
            else:
                final_actions = np.zeros((current_batch_size, total_dim), dtype=np.float32)
                
                for part, indices in action_parts.items():
                    action_key = f'action.{part}'
                    if action_key in pred_actions:
                        val = pred_actions[action_key]
                        if isinstance(val, torch.Tensor):
                            val = val.cpu().numpy()
                        start = indices.get('start', 0)
                        end = indices.get('end', 0)
                        final_actions[:, start:end] = val
                    else:
                        if debug_print and start_idx == 0:
                            print(f"[GPU {gpu_id}] WARNING: {action_key} not found in pred_actions!")
            
            pred_actions = final_actions
            
            if debug_print and start_idx == 0:
                print(f"[GPU {gpu_id}] Final actions shape: {pred_actions.shape}")
                print(f"[GPU {gpu_id}] has_horizon: {has_horizon}, action_horizon: {action_horizon}")
                print(f"[GPU {gpu_id}] Action value range: min={pred_actions.min():.6f}, max={pred_actions.max():.6f}, mean={pred_actions.mean():.6f}, std={pred_actions.std():.6f}")
                print(f"[GPU {gpu_id}] Non-zero actions: {np.count_nonzero(pred_actions)} / {pred_actions.size} ({100 * np.count_nonzero(pred_actions) / pred_actions.size:.2f}%)")
                if np.allclose(pred_actions, 0):
                    print(f"[GPU {gpu_id}] WARNING: All actions are zero!")
                else:
                    # Show sample of non-zero values
                    non_zero_mask = ~np.isclose(pred_actions, 0)
                    if np.any(non_zero_mask):
                        non_zero_values = pred_actions[non_zero_mask]
                        print(f"[GPU {gpu_id}] Sample non-zero values (first 10): {non_zero_values.flatten()[:10]}")
            
            # Store predictions
            # For each step, we predict action_horizon future actions
            # Multiple predictions for the same step are averaged later
            for i, s in enumerate(step_ids):
                for j in range(action_horizon):
                    if s + j >= length:
                        break
                    if s + j not in action_dict:
                        action_dict[s + j] = []
                    if has_horizon:
                        action_dict[s + j].append(pred_actions[i, j].flatten())
                    else:
                        action_dict[s + j].append(pred_actions[i].flatten())
        
        # Average multiple predictions for each step
        for s in action_dict:
            mean_action = np.mean(action_dict[s], axis=0)
            traj_data.at[s, "action"] = mean_action
        
        if debug_print:
            print(f"[GPU {gpu_id}] Saved trajectory {tid} with {len(action_dict)} actions")
        
        save_trajectory_data(traj_data, dataset, tid, output_dir)
    
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Dump IDM predicted actions for Allex dataset")
    parser.add_argument("--checkpoint", type=str, required=True, 
                        help="Path to IDM checkpoint (e.g., checkpoints/idm_allex_ego_0121)")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to LeRobot dataset with zero actions")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: overwrite input dataset)")
    parser.add_argument("--num_gpus", type=int, default=1,
                        help="Number of GPUs to use")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for inference")
    parser.add_argument("--max_episodes", type=int, default=None,
                        help="Maximum number of episodes to process (for debugging)")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of worker threads for data loading")
    parser.add_argument("--observation_indices", type=str, default=None,
                        help="Observation delta indices for video (space-separated). "
                             "Default: auto-detect from checkpoint action_horizon (e.g., '0 20' for ae20). "
                             "For Neural data with faster actions (5s vs 20s task duration), "
                             "manually specify (e.g., '0 4') to account for different action speed.")
    parser.add_argument("--add_dream_prefix", action="store_true",
                        help="Add <DREAM> prefix to task descriptions")
    
    args = parser.parse_args()
    
    # ============================================================
    # Step 1: Load checkpoint to get action_horizon
    # ============================================================
    print(f"Loading checkpoint config from: {args.checkpoint}")
    temp_model = IDM.from_pretrained(args.checkpoint)
    action_horizon = temp_model.config.action_horizon
    action_dim = temp_model.config.action_dim
    del temp_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print(f"============================================================")
    print(f"Checkpoint config:")
    print(f"  - action_horizon: {action_horizon}")
    print(f"  - action_dim: {action_dim}")
    print(f"  - At {ALLEX_FPS}fps, action_horizon={action_horizon} covers {action_horizon/ALLEX_FPS:.1f}s")
    print(f"============================================================")
    
    # ============================================================
    # Step 2: Set observation_indices (auto-detect or user-specified)
    # ============================================================
    # Load dataset FPS to adjust observation_indices
    dataset_info_path = os.path.join(args.dataset, 'meta', 'info.json')
    if os.path.exists(dataset_info_path):
        with open(dataset_info_path, 'r') as f:
            dataset_info = json.load(f)
        dataset_fps = dataset_info.get('fps', ALLEX_FPS)
    else:
        dataset_fps = ALLEX_FPS
    
    if args.observation_indices is not None:
        observation_indices = [int(x) for x in args.observation_indices.split()]
        print(f"Using user-specified observation_indices: {observation_indices}")
        print(f"  Note: Manual specification overrides auto-detection.")
        print(f"  Use this for Neural data with different action speeds (e.g., '0 4' for 4x faster actions).")
    else:
        # Auto-detect: [0, action_horizon] adjusted for dataset FPS
        # Training was at ALLEX_FPS (20fps), but dataset might be different FPS
        # Convert observation_indices to match the same time span
        if dataset_fps != ALLEX_FPS:
            fps_ratio = dataset_fps / ALLEX_FPS
            adjusted_horizon = int(action_horizon * fps_ratio)
            observation_indices = [0, adjusted_horizon]
            print(f"Auto-detected observation_indices from action_horizon: [0, {action_horizon}] (training @ {ALLEX_FPS}fps)")
            print(f"Adjusted for dataset FPS ({dataset_fps}fps): {observation_indices}")
            print(f"Time span: {max(observation_indices)/dataset_fps:.1f}s (matches {action_horizon/ALLEX_FPS:.1f}s at training FPS)")
        else:
        observation_indices = [0, action_horizon]
        print(f"Auto-detected observation_indices from action_horizon: {observation_indices}")
    print(f"At {ALLEX_FPS}fps, this observes frames spanning {max(observation_indices)/ALLEX_FPS:.1f}s")
    
    # Set output directory
    output_dir = args.output_dir if args.output_dir else args.dataset
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Check GPU availability
    device_count = torch.cuda.device_count()
    print(f"Found {device_count} GPUs available")
    if device_count < args.num_gpus:
        print(f"WARNING: Requested {args.num_gpus} GPUs but only {device_count} available")
        args.num_gpus = device_count
    
    # Load global metadata for normalization statistics
    # This matches the training config which uses global_metadata from IDM training dataset
    script_dir = Path(__file__).parent
    global_metadata_dir = script_dir / "global_metadata" / "allex"
    global_stats_path = global_metadata_dir / "stats.json"
    global_modality_path = global_metadata_dir / "modality.json"
    
    if not global_stats_path.exists():
        raise FileNotFoundError(
            f"Global stats.json not found at {global_stats_path}. "
            f"This should be copied from IDM training dataset stats.json"
        )
    if not global_modality_path.exists():
        raise FileNotFoundError(
            f"Global modality.json not found at {global_modality_path}"
        )
    
    # Copy global metadata to dataset to ensure we use the same statistics as training
    # This is critical: IDM was trained with these statistics, so we must use them for inference
    dataset_modality_path = Path(args.dataset) / "meta" / "modality.json"
    dataset_stats_path = Path(args.dataset) / "meta" / "stats.json"
    
    dataset_modality_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_stats_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Always copy global metadata to ensure consistency with training
    shutil.copy2(global_modality_path, dataset_modality_path)
    shutil.copy2(global_stats_path, dataset_stats_path)
    print(f"Using global metadata from IDM training:")
    print(f"  - modality.json: {global_modality_path} -> {dataset_modality_path}")
    print(f"  - stats.json: {global_stats_path} -> {dataset_stats_path}")
    
    # Create modality config (dynamically based on checkpoint's action_horizon)
    modality_config = create_allex_modality_config(observation_indices, action_horizon)
    
    # Create transforms (dynamically based on checkpoint's action_horizon)
    transforms = create_allex_transforms(
        video_keys=ALLEX_VIDEO_KEYS,
        state_keys=ALLEX_STATE_KEYS,
        action_keys=ALLEX_ACTION_KEYS,
        action_horizon=action_horizon,
        action_dim=action_dim,
    )
    transforms.eval()
    
    # Create dataset
    # The dataset's set_transforms_metadata will load stats from the copied stats.json
    dataset = LeRobotSingleDataset(
        dataset_path=args.dataset,
        modality_configs=modality_config,
        transforms=transforms,
        embodiment_tag=EmbodimentTag.ALLEX_EGO,
    )
    
    # Debug: Check if normalization statistics are set
    print("\n=== Checking Transform Statistics ===")
    for transform in transforms.transforms:
        if isinstance(transform, StateActionTransform):
            print(f"StateActionTransform for: {transform.apply_to}")
            print(f"  Normalization modes: {transform.normalization_modes}")
            print(f"  Normalization statistics keys: {list(transform.normalization_statistics.keys())}")
            for key in transform.apply_to:
                if key in transform.normalization_statistics:
                    stats = transform.normalization_statistics[key]
                    print(f"  {key} statistics: {list(stats.keys())}")
                    if "q01" in stats and "q99" in stats:
                        print(f"    q01 range: [{min(stats['q01']):.6f}, {max(stats['q01']):.6f}]")
                        print(f"    q99 range: [{min(stats['q99']):.6f}, {max(stats['q99']):.6f}]")
                else:
                    print(f"  WARNING: {key} has no normalization statistics!")
    print("=====================================\n")
    
    trajectory_ids = dataset.trajectory_ids
    if args.max_episodes is not None and args.max_episodes > 0:
        trajectory_ids = trajectory_ids[:args.max_episodes]
    
    print(f"Processing {len(trajectory_ids)} trajectories")
    
    # Split trajectories among GPUs
    chunk_size = (len(trajectory_ids) + args.num_gpus - 1) // args.num_gpus
    tasks_by_gpu = {}
    for i in range(args.num_gpus):
        start = i * chunk_size
        end = min(start + chunk_size, len(trajectory_ids))
        if start >= end:
            break
        tasks_by_gpu[i] = trajectory_ids[start:end]
    
    # Run inference
    try:
        if args.num_gpus == 1:
            # Single GPU - run directly
            worker_func(
                0, tasks_by_gpu[0], args.checkpoint, args.dataset,
                output_dir, args.batch_size, args.num_workers,
                observation_indices, dataset
            )
        else:
            # Multi-GPU - use ProcessPoolExecutor
            mp.set_start_method("spawn", force=True)
            with ProcessPoolExecutor(max_workers=args.num_gpus) as executor:
                futures = []
                for gpu_id, gpu_traj_list in tasks_by_gpu.items():
                    future = executor.submit(
                        worker_func,
                        gpu_id, gpu_traj_list, args.checkpoint, args.dataset,
                        output_dir, args.batch_size, args.num_workers,
                        observation_indices, dataset
                    )
                    futures.append(future)
                
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        print(f"Error in worker: {e}")
                        import traceback
                        traceback.print_exc()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Copy metadata and videos if output_dir is different from input
    if output_dir != args.dataset:
        # Copy metadata
        meta_src = os.path.join(args.dataset, "meta")
        meta_dst = os.path.join(output_dir, "meta")
        if os.path.exists(meta_src) and not os.path.exists(meta_dst):
            shutil.copytree(meta_src, meta_dst)
            print(f"Copied metadata to: {meta_dst}")
        
        # Copy videos
        videos_src = os.path.join(args.dataset, "videos")
        videos_dst = os.path.join(output_dir, "videos")
        if os.path.exists(videos_src) and not os.path.exists(videos_dst):
            shutil.copytree(videos_src, videos_dst)
            print(f"Copied videos to: {videos_dst}")
    
    # Add <DREAM> prefix to tasks if requested
    if args.add_dream_prefix:
        tasks_path = os.path.join(output_dir, "meta", "tasks.jsonl")
        if os.path.exists(tasks_path):
            tasks = []
            with open(tasks_path, "r") as f:
                for line in f:
                    tasks.append(json.loads(line))
            
            updated_tasks = []
            for task in tasks:
                if "task" in task and not task["task"].startswith("<DREAM>"):
                    task["task"] = f"<DREAM>{task['task']}"
                updated_tasks.append(task)
            
            with open(tasks_path, "w") as f:
                for task in updated_tasks:
                    f.write(json.dumps(task) + "\n")
            
            print("Added <DREAM> prefix to tasks")
    
    print("\nDone!")


if __name__ == "__main__":
    main()

