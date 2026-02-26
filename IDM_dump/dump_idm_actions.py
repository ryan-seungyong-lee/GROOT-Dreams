import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from tianshou.data import Batch
from huggingface_hub import hf_hub_download
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.model.idm import IDM
from gr00t.utils.video import get_all_frames_and_timestamps
from gr00t.data.embodiment_tags import EmbodimentTag


def load_dataset_and_config(checkpoint_path, validation_dataset_path, video_indices, use_standalone_allex=False):
    """
    Load dataset and config. For Allex, can use standalone mode (no config file needed).
    
    Args:
        checkpoint_path: Path to checkpoint or HuggingFace repo
        validation_dataset_path: Path to validation dataset
        video_indices: Video frame indices (space-separated string)
        use_standalone_allex: If True, use standalone Allex config (no Hydra config needed)
    """
    # For standalone Allex mode, use dump_idm_actions_allex logic
    if use_standalone_allex:
        from gr00t.model.idm import IDM
        from gr00t.data.dataset import LeRobotSingleDataset, ModalityConfig
        from gr00t.data.embodiment_tags import EmbodimentTag
        from gr00t.data.transform import (
            VideoToTensor, VideoCrop, VideoResize, VideoToNumpy,
            StateActionToTensor, StateActionTransform, ConcatTransform, ComposedModalityTransform
        )
        from gr00t.model.transforms_idm import GR00TIDMTransform
        
        # Load checkpoint to get action_horizon
        temp_model = IDM.from_pretrained(checkpoint_path)
        action_horizon = temp_model.config.action_horizon
        action_dim = temp_model.config.action_dim
        del temp_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Auto-detect observation_indices from action_horizon
        if video_indices is not None:
            observation_indices = [int(x) for x in video_indices.split()]
        else:
            observation_indices = [0, action_horizon]
        
        # Allex keys
        ALLEX_VIDEO_KEYS = ["video.camera_ego_left"]
        ALLEX_STATE_KEYS = [
            "state.right_arm_joints", "state.left_arm_joints",
            "state.right_hand_joints", "state.left_hand_joints",
            "state.neck_joints", "state.waist_joints",
        ]
        ALLEX_ACTION_KEYS = [
            "action.right_arm_joints", "action.left_arm_joints",
            "action.right_hand_joints", "action.left_hand_joints",
            "action.neck_joints", "action.waist_joints",
        ]
        
        # Create modality config
        video_modality = ModalityConfig(
            delta_indices=observation_indices,
            modality_keys=ALLEX_VIDEO_KEYS,
        )
        state_modality = ModalityConfig(
            delta_indices=[0],
            modality_keys=ALLEX_STATE_KEYS,
        )
        action_modality = ModalityConfig(
            delta_indices=list(range(action_horizon)),
            modality_keys=ALLEX_ACTION_KEYS,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
        }
        
        # Load stats from checkpoint directory (IDM training dataset stats)
        # Checkpoint directory should contain stats.json from training dataset
        checkpoint_dir = Path(checkpoint_path)
        checkpoint_stats_path = checkpoint_dir / "stats.json"
        checkpoint_modality_path = checkpoint_dir / "modality.json"
        
        # Fallback to IDM training dataset stats, then global metadata
        # idm_train_stats_path = Path("/storage1/sjw_dataset/dataset/Allex/real/idm_train_data/meta/stats.json")
        # idm_train_modality_path = Path("/storage1/sjw_dataset/dataset/Allex/real/idm_train_data/meta/modality.json")
        idm_train_stats_path = Path("/storage1/sjw_dataset/dataset/Allex/real/idm_train_data_teleop_added/meta/stats.json")
        idm_train_modality_path = Path("/storage1/sjw_dataset/dataset/Allex/real/idm_train_data_teleop_added/meta/modality.json")
        
        script_dir = Path(__file__).parent
        global_metadata_dir = script_dir / "global_metadata" / "allex"
        global_stats_path = global_metadata_dir / "stats.json"
        global_modality_path = global_metadata_dir / "modality.json"
        
        # Log all potential stats paths
        print(f"\n{'='*80}")
        print("STATS LOADING - Checking available sources:")
        print(f"{'='*80}")
        print(f"1. Checkpoint stats.json: {checkpoint_stats_path}")
        print(f"   Exists: {checkpoint_stats_path.exists()}")
        print(f"2. IDM training dataset stats.json: {idm_train_stats_path}")
        print(f"   Exists: {idm_train_stats_path.exists()}")
        print(f"3. Global metadata stats.json: {global_stats_path}")
        print(f"   Exists: {global_stats_path.exists()}")
        print(f"{'='*80}\n")
        
        # Use checkpoint stats if available, otherwise use IDM training dataset stats, then global metadata
        if checkpoint_stats_path.exists():
            stats_path = checkpoint_stats_path
            print(f"✓ Using stats from checkpoint: {stats_path}")
        elif idm_train_stats_path.exists():
            stats_path = idm_train_stats_path
            print(f"✓ Using stats from IDM training dataset (checkpoint stats not found): {stats_path}")
        elif global_stats_path.exists():
            stats_path = global_stats_path
            print(f"✓ Using global metadata stats (checkpoint and IDM training stats not found): {stats_path}")
        else:
            raise FileNotFoundError(
                f"Stats.json not found at checkpoint ({checkpoint_stats_path}), IDM training dataset ({idm_train_stats_path}), or global metadata ({global_stats_path})"
            )
        
        # Log all potential modality paths
        print(f"\n{'='*80}")
        print("MODALITY LOADING - Checking available sources:")
        print(f"{'='*80}")
        print(f"1. Checkpoint modality.json: {checkpoint_modality_path}")
        print(f"   Exists: {checkpoint_modality_path.exists()}")
        print(f"2. IDM training dataset modality.json: {idm_train_modality_path}")
        print(f"   Exists: {idm_train_modality_path.exists()}")
        print(f"3. Global metadata modality.json: {global_modality_path}")
        print(f"   Exists: {global_modality_path.exists()}")
        print(f"{'='*80}\n")
        
        if checkpoint_modality_path.exists():
            modality_path = checkpoint_modality_path
            print(f"✓ Using modality from checkpoint: {modality_path}")
        elif idm_train_modality_path.exists():
            modality_path = idm_train_modality_path
            print(f"✓ Using modality from IDM training dataset (checkpoint modality not found): {modality_path}")
        elif global_modality_path.exists():
            modality_path = global_modality_path
            print(f"✓ Using global metadata modality (checkpoint and IDM training modality not found): {modality_path}")
        else:
            raise FileNotFoundError(
                f"Modality.json not found at checkpoint ({checkpoint_modality_path}), IDM training dataset ({idm_train_modality_path}), or global metadata ({global_modality_path})"
            )
        
        # Copy metadata to dataset to ensure we use the same statistics as training
        # This is critical: IDM was trained with these statistics, so we must use them for inference
        dataset_modality_path = Path(validation_dataset_path) / "meta" / "modality.json"
        dataset_stats_path = Path(validation_dataset_path) / "meta" / "stats.json"
        
        import shutil
        dataset_modality_path.parent.mkdir(parents=True, exist_ok=True)
        dataset_stats_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Always copy metadata to ensure consistency
        shutil.copy2(modality_path, dataset_modality_path)
        shutil.copy2(stats_path, dataset_stats_path)
        print(f"Copied metadata to dataset:")
        print(f"  - modality.json: {modality_path} -> {dataset_modality_path}")
        print(f"  - stats.json: {stats_path} -> {dataset_stats_path}")
        
        # Create transforms
        # The dataset's set_transforms_metadata will load stats from the copied stats.json
        transforms_list = [
            VideoToTensor(apply_to=ALLEX_VIDEO_KEYS),
            VideoCrop(apply_to=ALLEX_VIDEO_KEYS, scale=0.95),
            VideoResize(apply_to=ALLEX_VIDEO_KEYS, height=224, width=224, interpolation="linear"),
            VideoToNumpy(apply_to=ALLEX_VIDEO_KEYS),
            StateActionToTensor(apply_to=ALLEX_STATE_KEYS),
            StateActionTransform(
                apply_to=ALLEX_STATE_KEYS,
                normalization_modes={key: "q99" for key in ALLEX_STATE_KEYS},
            ),
            StateActionToTensor(apply_to=ALLEX_ACTION_KEYS),
            StateActionTransform(
                apply_to=ALLEX_ACTION_KEYS,
                normalization_modes={key: "q99" for key in ALLEX_ACTION_KEYS},
            ),
            ConcatTransform(
                video_concat_order=ALLEX_VIDEO_KEYS,
                state_concat_order=ALLEX_STATE_KEYS,
                action_concat_order=ALLEX_ACTION_KEYS,
            ),
            GR00TIDMTransform(
                state_horizon=1,
                action_horizon=action_horizon,
                max_state_dim=64,
                max_action_dim=action_dim,
            ),
        ]
        transform_inst = ComposedModalityTransform(transforms=transforms_list)
        
        dataset = LeRobotSingleDataset(
            dataset_path=validation_dataset_path,
            modality_configs=modality_configs,
            transforms=transform_inst,
            embodiment_tag=EmbodimentTag.ALLEX_EGO,
        )
        
        # Return dummy cfg for compatibility
        class DummyCfg:
            pass
        cfg = DummyCfg()
        
        return cfg, dataset, modality_configs
    
    # Original Hydra config-based loading
    # Check if checkpoint_path is a HuggingFace model repo
    is_hf_repo = not os.path.exists(checkpoint_path) and '/' in checkpoint_path
    
    if is_hf_repo:
        # For HuggingFace repos, we need to download config differently
        try:
            # Download the config file from the repo
            config_file = hf_hub_download(
                repo_id=checkpoint_path,
                filename="experiment_cfg/conf.yaml",
                repo_type="model"
            )
            with open(config_file, "r") as f:
                config = yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading config from HuggingFace repo: {e}")
            raise
    else:
        # Original local path handling
        config_path = os.path.join(checkpoint_path, "experiment_cfg", "conf.yaml")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    
    cfg = OmegaConf.create(config)

    dataset_name = os.path.basename(validation_dataset_path).split(".")[0]
    if "gr1" in dataset_name:
        embodiment = "gr1_unified" #should be hard-coded for predefined IDM config
    else:
        embodiment = dataset_name
    print(f"Dataset name: {dataset_name}")

    modality_configs = cfg["modality_configs"][embodiment]
    if video_indices is not None:
        video_delta_indices = video_indices.split(" ")
        print(f"Using provided video_delta_indices: {video_delta_indices}")
        modality_configs["video"]["delta_indices"] = video_delta_indices
    modality_configs = instantiate(modality_configs)


    if "all_transforms" in cfg:
        transform = cfg["all_transforms"][embodiment]
    else:
        transform = cfg["train_dataset"]["all_transforms"][embodiment]
    
    # Filter out VideoColorJitter transform for inference
    if "transforms" in transform:
        filtered_transforms = []
        for t in transform["transforms"]:
            if t.get("_target_") != "groot.data.transform.VideoColorJitter":
                filtered_transforms.append(t)
            
        transform["transforms"] = filtered_transforms
    transform_inst = instantiate(transform)

    metadata_versions = cfg["metadata_versions"]
    metadata_version = metadata_versions[embodiment]

    if "gr1" in embodiment:
        embodiment_tag = EmbodimentTag.GR1_unified
    elif "franka" in embodiment:
        embodiment_tag = EmbodimentTag.FRANKA
    elif "so100" in embodiment:
        embodiment_tag = EmbodimentTag.SO100
    elif "robocasa" in embodiment:
        embodiment_tag = EmbodimentTag.ROBOCASA
    elif "allex" in embodiment:
        embodiment_tag = EmbodimentTag.ALLEX_EGO
    else:
        raise ValueError(f"Unknown embodiment: {embodiment}")

    dataset = LeRobotSingleDataset(
        dataset_path=validation_dataset_path,
        modality_configs=modality_configs,
        # metadata_version=metadata_version,
        transforms=transform_inst,
        embodiment_tag=embodiment_tag,
    )

    return cfg, dataset, modality_configs


def collate_fn(features_list, device):
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
    data = {}
    dataset.curr_traj_data = dataset.get_trajectory_data(trajectory_id)
    # Get the data for all modalities
    for modality in dataset.modality_keys:
        # Get the data corresponding to each key in the modality
        for key in dataset.modality_keys[modality]:
            if modality == "video":
                pass
            elif modality == "state" or modality == "action":
                data[key] = dataset.get_state_or_action(trajectory_id, modality, key, base_index)
            elif modality == "language":
                data[key] = dataset.get_language(trajectory_id, key, base_index)

    return data

def save_trajectory_data(trajectory_data, dataset, trajectory_id, output_dir):
    chunk_index = dataset.get_episode_chunk(trajectory_id)
    chunk_dir = f"chunk-{chunk_index:03d}"
    os.makedirs(os.path.join(output_dir, "data", chunk_dir), exist_ok=True)

    episode_id = f"episode_{int(trajectory_id):06d}"
    output_file_path = os.path.join(output_dir, "data", chunk_dir, f"{episode_id}.parquet")
    trajectory_data.to_parquet(output_file_path)

##
# Worker function, now handles either validation or writing based on a boolean flag.
##
def worker_func(
    gpu_id: int,
    traj_id_list: list,
    checkpoint_path: str,
    validation_dataset_path: str,
    output_dir: str,
    batch_size: int,
    num_workers: int = 1,
    video_indices=None,
    dataset=None,
    modality_configs=None,
):
    """This function runs in a separate process on GPU `gpu_id` and processes all `traj_id_list`."""
    # Load model on this GPU
    # from_pretrained should handle both local paths and HF model IDs
    print(f"Loading model from: {checkpoint_path}")
    model = IDM.from_pretrained(checkpoint_path)

    model.requires_grad_(False)
    model.eval()
    device = torch.device(f"cuda:{gpu_id}")
    model.to(device)
    

    for tid_idx, tid in enumerate(tqdm(traj_id_list, desc=f"GPU {gpu_id}", position=gpu_id)):
        action_dict = {}
        traj_data = dataset.get_trajectory_data(tid)
        length = len(traj_data)
        
        # Only print debug info for first trajectory and first batch
        debug_print = (tid_idx == 0)


        all_features = []

        # Create a simple prefetching mechanism with a thread pool
        from concurrent.futures import ThreadPoolExecutor

        def load_and_transform_step(step_idx, video_data):
            import torch
            from einops import rearrange

            step_data = get_step_data(dataset, tid, step_idx)
            timestamp = dataset.curr_traj_data["timestamp"].to_numpy()
            for key in video_data: 
                frames, whole_indices = video_data[key]
                step_indices = dataset.delta_indices[key] + step_idx
                step_indices = np.maximum(step_indices, 0)
                step_indices = np.minimum(step_indices, dataset.trajectory_lengths[tid] - 1)
                indices = np.array([np.where(np.isclose(whole_indices, val))[0][0] for val in timestamp[step_indices]])
                step_data[key] = frames[indices]

            output = dataset.transforms(step_data)
            return output
        

        video_data = {}
        for key in dataset.modality_keys["video"]: 
            video_path = dataset.get_video_path(tid, key.replace("video.", ""))
            video_backend = dataset.video_backend
            video_backend_kwargs = dataset.video_backend_kwargs
            frames, whole_indices = get_all_frames_and_timestamps(video_path.as_posix(), video_backend, video_backend_kwargs)
            video_data[key] = (frames, whole_indices)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for step_idx in range(length):
                future = executor.submit(load_and_transform_step, step_idx, video_data)
                futures.append(future)
            
            all_features = []
            for future in as_completed(futures):
                all_features.append(future.result())

        for start_idx in range(0, length, batch_size):
            end_idx = min(start_idx + batch_size, length)
            step_ids = list(range(start_idx, end_idx))

            batch_features = all_features[start_idx:end_idx]
            batch_dict = collate_fn(batch_features, device)

            with torch.no_grad():
                out = model.get_action(batch_dict)

            pred_actions = out["action_pred"].cpu()
            
            # DEBUG: Print model output shape
            if debug_print and start_idx == 0:
                print(f"\n[DEBUG] === Trajectory {tid}, Batch starting at step {start_idx} ===")
                print(f"[DEBUG] Model output (out['action_pred']): {out['action_pred'].shape}")
                print(f"[DEBUG] pred_actions (after .cpu()): {pred_actions.shape}")
            
            pred_actions = dataset.transforms.unapply(Batch(action=pred_actions))

            # DEBUG: Print unapplied pred_actions structure
            if debug_print and start_idx == 0:
                print(f"[DEBUG] pred_actions keys (after unapply): {list(pred_actions.keys())}")
                for key in pred_actions.keys():
                    print(f"[DEBUG]   {key}: shape={pred_actions[key].shape}, dtype={pred_actions[key].dtype}")

            # Load modality.json to get the proper structure
            modality_json_path = os.path.join(validation_dataset_path, 'meta', 'modality.json')

            with open(modality_json_path, 'r') as f:
                modality_config = json.load(f)
            
            # Get action part configurations
            action_parts = modality_config.get('action', {})
            
            # DEBUG: Print action parts from modality.json
            if debug_print and start_idx == 0:
                print(f"[DEBUG] Action parts from modality.json: {list(action_parts.keys())}")
                for part, indices in action_parts.items():
                    print(f"[DEBUG]   {part}: start={indices.get('start', 0)}, end={indices.get('end', 0)}")
            
            # Calculate total action dimension
            total_dim = 0
            for part, indices in action_parts.items():
                total_dim = max(total_dim, indices.get('end', 0))
            
            if debug_print and start_idx == 0:
                print(f"[DEBUG] Total action dimension (total_dim): {total_dim}")
            
            # Check if we have an action horizon dimension
            has_horizon = False
            action_horizon = 1
            sample_key = list(pred_actions.keys())[0]
            if len(pred_actions[sample_key].shape) == 3:
                has_horizon = True
                batch_size, action_horizon, _ = pred_actions[sample_key].shape
            else:
                batch_size = pred_actions[sample_key].shape[0]
            
            if debug_print and start_idx == 0:
                print(f"[DEBUG] has_horizon: {has_horizon}")
                print(f"[DEBUG] action_horizon: {action_horizon}")
                print(f"[DEBUG] batch_size: {batch_size}")
            
            if has_horizon:
                final_actions = np.zeros((batch_size, action_horizon, total_dim))
                
                # Fill in the actions for parts that exist in pred_actions
                for part, indices in action_parts.items():
                    action_key = f'action.{part}'
                    action_start_idx = indices.get('start', 0)
                    action_end_idx = indices.get('end', 0)
                    
                    if action_key in pred_actions:
                        if debug_print and start_idx == 0:
                            print(f"[DEBUG] Filling {action_key}: shape={pred_actions[action_key].shape}, action_start_idx={action_start_idx}, action_end_idx={action_end_idx}")
                        final_actions[:, :, action_start_idx:action_end_idx] = pred_actions[action_key]
                    else:
                        if debug_print and start_idx == 0:
                            print(f"[DEBUG] Warning: {action_key} not found in pred_actions")
            else:
                final_actions = np.zeros((batch_size, total_dim))
                
                # Fill in the actions for parts that exist in pred_actions
                for part, indices in action_parts.items():
                    action_key = f'action.{part}'
                    start_idx_action = indices.get('start', 0)
                    end_idx_action = indices.get('end', 0)
                    
                    if action_key in pred_actions:
                        if debug_print and start_idx == 0:
                            print(f"[DEBUG] Filling {action_key}: shape={pred_actions[action_key].shape}, start_idx={start_idx_action}, end_idx={end_idx_action}")
                        final_actions[:, start_idx_action:end_idx_action] = pred_actions[action_key]
                    else:
                        if debug_print and start_idx == 0:
                            print(f"[DEBUG] Warning: {action_key} not found in pred_actions")
            
            pred_actions = final_actions
            if debug_print and start_idx == 0:
                print(f"[DEBUG] Final actions shape: {pred_actions.shape}")
                print(f"[DEBUG] Step IDs in this batch: {step_ids}")
                print(f"[DEBUG] Trajectory length: {length}")
                
                # Get video delta_indices
                video_key = list(dataset.modality_keys['video'])[0]
                video_delta_indices = dataset.delta_indices[video_key]
                print(f"[DEBUG] Video delta_indices: {video_delta_indices}")
                print(f"\n[DEBUG] === Observation → Action Mapping ===")
                print(f"[DEBUG] Explanation: For each step_idx, we observe states at (step_idx + delta_indices)")
                print(f"[DEBUG] and predict actions for steps [step_idx, step_idx+1, ..., step_idx+action_horizon-1]")
                print()
                for i, s in enumerate(step_ids[:5]):  # Show first 5 steps
                    # Calculate observed step indices
                    observed_steps = [s + delta for delta in video_delta_indices]
                    observed_steps = [max(0, min(step, length-1)) for step in observed_steps]  # Clamp to valid range
                    predicted_action_steps = list(range(s, min(s + action_horizon, length)))
                    print(f"[DEBUG] Step {s}:")
                    print(f"[DEBUG]   Observing states: S_{observed_steps} (step_idx={s} + delta_indices={video_delta_indices})")
                    print(f"[DEBUG]   Predicting actions: a_{predicted_action_steps[0]} to a_{predicted_action_steps[-1]} ({len(predicted_action_steps)} actions)")
                    print(f"[DEBUG]   Action shape per step: {pred_actions[i].shape if has_horizon else pred_actions[i].shape}")
                print()

            # Not validating => we do the usual writing
            # For each step_idx in the batch, we predict action_horizon actions
            # Example: step_idx=0, action_horizon=16
            #   - Observing: S_0, S_8 (if video_indices="0 8")
            #   - Predicting: a_0, a_1, ..., a_15 (16 actions)
            #   - Storing: action_dict[0] = [a_0], action_dict[1] = [a_1], ..., action_dict[15] = [a_15]
            # 
            # When step_idx=1:
            #   - Observing: S_1, S_9
            #   - Predicting: a_1, a_2, ..., a_16
            #   - Storing: action_dict[1] = [a_1 from step0, a_1 from step1] (duplicates averaged later)
            for i, s in enumerate(step_ids):
                for j in range(action_horizon):
                    if s+j >= length:
                        break
                    if s+j not in action_dict:
                        action_dict[s+j] = []
                    if has_horizon:
                        action_dict[s+j].append(pred_actions[i, j].flatten())
                    else:
                        action_dict[s+j].append(pred_actions[i].flatten())
        
        # DEBUG: Print action_dict statistics
        if debug_print:
            print(f"\n[DEBUG] === Final Action Dict Statistics for Trajectory {tid} ===")
            print(f"[DEBUG] Action dict keys: {sorted(action_dict.keys())}")
            print(f"[DEBUG] Total steps with actions: {len(action_dict)}")
            for s in sorted(action_dict.keys())[:10]:  # Print first 10 steps
                print(f"[DEBUG]   Step {s}: {len(action_dict[s])} action(s), shape={np.array(action_dict[s]).shape}")
            if len(action_dict) > 10:
                print(f"[DEBUG]   ... (showing first 10 of {len(action_dict)} steps)")
        
        for s in action_dict:
            mean_action = np.mean(action_dict[s], axis=0)
            traj_data.at[s, "action"] = mean_action
            if debug_print and s < 5:  # Print first 5 actions for debugging
                print(f"[DEBUG] Final action at step {s}: shape={mean_action.shape}, min={mean_action.min():.4f}, max={mean_action.max():.4f}, mean={mean_action.mean():.4f}")

        save_trajectory_data(traj_data, dataset, tid, output_dir)
    
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def validate_checkpoint(
    checkpoint_path,
    validation_dataset_path,
    output_dir=None,
    num_gpus=8,
    batch_size=16,
    max_episodes=None,
    num_workers=1,
    video_indices=None,
    use_standalone_allex=False,
):
    device_count = torch.cuda.device_count()
    print(f"Found {device_count} GPUs available.")
    if device_count < num_gpus:
        print(
            f"WARNING: You requested num_gpus={num_gpus} but only {device_count} GPUs are visible."
        )
        num_gpus = device_count

    # If not validation, create the output directory for saving predicted actions
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Will save predicted actions to: {output_dir}")

    _, dataset, modality_configs = load_dataset_and_config(
        checkpoint_path, validation_dataset_path, video_indices, use_standalone_allex=use_standalone_allex
    )

    dataset.transforms.eval()

    trajectory_ids = dataset.trajectory_ids
    if max_episodes is not None and max_episodes > 0:
        trajectory_ids = trajectory_ids[:max_episodes]

    print(f"Processing {len(trajectory_ids)} trajectories total.")

    # Split trajectories among GPUs
    chunk_size = (len(trajectory_ids) + num_gpus - 1) // num_gpus

    tasks_by_gpu = {}
    for i in range(num_gpus):
        start = i * chunk_size
        end = min(start + chunk_size, len(trajectory_ids))
        if start >= end:
            break
        gpu_traj_list = trajectory_ids[start:end]
        tasks_by_gpu[i] = gpu_traj_list

    try:
        with ProcessPoolExecutor(max_workers=num_gpus) as executor:
            futures = []
            for gpu_id, gpu_traj_list in tasks_by_gpu.items():
                future = executor.submit(
                    worker_func,
                    gpu_id,
                    gpu_traj_list,
                    checkpoint_path,
                    validation_dataset_path,
                    output_dir,
                    batch_size,
                    num_workers,
                    video_indices,
                    dataset,
                    modality_configs,
                )
                futures.append(future)
            
            # Wait for all futures to complete or handle interruption
            for future in as_completed(futures):
                try:
                    # Get the result to catch any exceptions
                    future.result()
                except Exception as e:
                    print(f"Error in worker process: {e}")
                    import traceback
                    traceback.print_exc()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Cleaning up...")
        # The context manager will handle cancellation of pending futures
    except Exception as e:
        print(f"Error in main process: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Force cleanup of CUDA memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # If not validating, optionally copy the meta & videos folders
    if output_dir:
        # Copy the metadata directory
        meta_src = os.path.join(validation_dataset_path, "meta")
        meta_dst = os.path.join(output_dir, "meta")
        if os.path.exists(meta_src):
            import shutil

            if not os.path.exists(meta_dst):
                shutil.copytree(meta_src, meta_dst)
            print(f"Copied metadata to: {meta_dst}")
        
            tasks_path = os.path.join(meta_dst, "tasks.jsonl")
            if os.path.exists(tasks_path):
                # Read the existing tasks
                tasks = []
                with open(tasks_path, "r") as f:
                    for line in f:
                        tasks.append(json.loads(line))

                # Update tasks with <DREAM> prefix if not already present
                updated_tasks = []
                for task in tasks:
                    if "task" in task and not task["task"].startswith("<DREAM>"):
                        task["task"] = f"<DREAM>{task['task']}"
                    updated_tasks.append(task)

                # Write the updated tasks back to the file
                with open(tasks_path, "w") as f:
                    for task in updated_tasks:
                        f.write(json.dumps(task) + "\n")

                print("Updated tasks.jsonl with <DREAM> prefix")


        # Copy the videos directory if it exists
        videos_src = os.path.join(validation_dataset_path, "videos")
        videos_dst = os.path.join(output_dir, "videos")
        if os.path.exists(videos_src):
            import shutil

            if not os.path.exists(videos_dst):
                shutil.copytree(videos_src, videos_dst)
            print(f"Copied videos to: {videos_dst}")

    return {"message": "Dataset with predicted actions written to disk"}


#
# Main
#
if __name__ == "__main__":
    # If you're doing spawn/forkserver across multiple processes:
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(
        description="Validate a checkpoint on a dataset with multi-GPU parallelism"
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the checkpoint")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the validation dataset")
    parser.add_argument(
        "--output_dir", type=str, help="Path to save the output dataset with predicted actions"
    )
    parser.add_argument("--num_gpus", type=int, default=8, help="Number of GPUs to use in parallel")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference steps")
    parser.add_argument(
        "--max_episodes", type=int, default=None, help="Maximum number of episodes to validate"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="Number of worker threads for data loading per GPU",
    )
    parser.add_argument(
        "--video_indices",
        type=str,
        default=None,
        help="Video frame indices to use for inference",
    )
    parser.add_argument(
        "--use_standalone_allex",
        action="store_true",
        help="Use standalone Allex config (no Hydra config file needed). "
             "Automatically detects action_horizon from checkpoint.",
    )

    args = parser.parse_args()

    result = validate_checkpoint(
        checkpoint_path=args.checkpoint,
        validation_dataset_path=args.dataset,
        output_dir=args.output_dir,
        num_gpus=args.num_gpus,
        batch_size=args.batch_size,
        max_episodes=args.max_episodes,
        num_workers=args.num_workers,
        video_indices=args.video_indices,
        use_standalone_allex=args.use_standalone_allex,
    )
