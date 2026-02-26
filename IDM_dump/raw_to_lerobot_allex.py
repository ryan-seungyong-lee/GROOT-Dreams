import json
import shutil
from pathlib import Path
from typing import Dict, List, Any, Tuple
import concurrent.futures
import os

import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse

import subprocess

import multiprocessing
from multiprocessing import Pool
import math


CHUNKS_SIZE = 1000
DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"

# Allex joint names for state and action (48 dimensions)
ALLEX_STATE_NAMES = [
    "R_Shoulder_Pitch_Joint_Pos", "R_Shoulder_Roll_Joint_Pos", "R_Shoulder_Yaw_Joint_Pos",
    "R_Elbow_Joint_Pos", "R_Wrist_Yaw_Joint_Pos", "R_Wrist_Roll_Joint_Pos", "R_Wrist_Pitch_Joint_Pos",
    "L_Shoulder_Pitch_Joint_Pos", "L_Shoulder_Roll_Joint_Pos", "L_Shoulder_Yaw_Joint_Pos",
    "L_Elbow_Joint_Pos", "L_Wrist_Yaw_Joint_Pos", "L_Wrist_Roll_Joint_Pos", "L_Wrist_Pitch_Joint_Pos",
    "R_Thumb_Yaw_Joint_Pos", "R_Thumb_CMC_Joint_Pos", "R_Thumb_MCP_Joint_Pos",
    "R_Index_Roll_Joint_Pos", "R_Index_MCP_Joint_Pos", "R_Index_PIP_Joint_Pos",
    "R_Middle_Roll_Joint_Pos", "R_Middle_MCP_Joint_Pos", "R_Middle_PIP_Joint_Pos",
    "R_Ring_Roll_Joint_Pos", "R_Ring_MCP_Joint_Pos", "R_Ring_PIP_Joint_Pos",
    "R_Little_Roll_Joint_Pos", "R_Little_MCP_Joint_Pos", "R_Little_PIP_Joint_Pos",
    "L_Thumb_Yaw_Joint_Pos", "L_Thumb_CMC_Joint_Pos", "L_Thumb_MCP_Joint_Pos",
    "L_Index_Roll_Joint_Pos", "L_Index_MCP_Joint_Pos", "L_Index_PIP_Joint_Pos",
    "L_Middle_Roll_Joint_Pos", "L_Middle_MCP_Joint_Pos", "L_Middle_PIP_Joint_Pos",
    "L_Ring_Roll_Joint_Pos", "L_Ring_MCP_Joint_Pos", "L_Ring_PIP_Joint_Pos",
    "L_Little_Roll_Joint_Pos", "L_Little_MCP_Joint_Pos", "L_Little_PIP_Joint_Pos",
    "Head_Pan_Joint_Pos", "Head_Tilt_Joint_Pos",
    "Waist_Roll_Joint_Pos", "Waist_Pitch_Joint_Pos"
]

ALLEX_ACTION_NAMES = [
    "R_Shoulder_Pitch_Joint", "R_Shoulder_Roll_Joint", "R_Shoulder_Yaw_Joint",
    "R_Elbow_Joint", "R_Wrist_Yaw_Joint", "R_Wrist_Roll_Joint", "R_Wrist_Pitch_Joint",
    "L_Shoulder_Pitch_Joint", "L_Shoulder_Roll_Joint", "L_Shoulder_Yaw_Joint",
    "L_Elbow_Joint", "L_Wrist_Yaw_Joint", "L_Wrist_Roll_Joint", "L_Wrist_Pitch_Joint",
    "R_Thumb_Yaw_Joint", "R_Thumb_CMC_Joint", "R_Thumb_MCP_Joint",
    "R_Index_Roll_Joint", "R_Index_MCP_Joint", "R_Index_PIP_Joint",
    "R_Middle_Roll_Joint", "R_Middle_MCP_Joint", "R_Middle_PIP_Joint",
    "R_Ring_Roll_Joint", "R_Ring_MCP_Joint", "R_Ring_PIP_Joint",
    "R_Little_Roll_Joint", "R_Little_MCP_Joint", "R_Little_PIP_Joint",
    "L_Thumb_Yaw_Joint", "L_Thumb_CMC_Joint", "L_Thumb_MCP_Joint",
    "L_Index_Roll_Joint", "L_Index_MCP_Joint", "L_Index_PIP_Joint",
    "L_Middle_Roll_Joint", "L_Middle_MCP_Joint", "L_Middle_PIP_Joint",
    "L_Ring_Roll_Joint", "L_Ring_MCP_Joint", "L_Ring_PIP_Joint",
    "L_Little_Roll_Joint", "L_Little_MCP_Joint", "L_Little_PIP_Joint",
    "Head_Pan_Joint", "Head_Tilt_Joint",
    "Waist_Roll_Joint", "Waist_Pitch_Joint"
]

ALLEX_EFFORT_NAMES = [
    "R_Shoulder_Pitch_Joint_Torque", "R_Shoulder_Roll_Joint_Torque", "R_Shoulder_Yaw_Joint_Torque",
    "R_Elbow_Joint_Torque", "R_Wrist_Yaw_Joint_Torque", "R_Wrist_Roll_Joint_Torque", "R_Wrist_Pitch_Joint_Torque",
    "L_Shoulder_Pitch_Joint_Torque", "L_Shoulder_Roll_Joint_Torque", "L_Shoulder_Yaw_Joint_Torque",
    "L_Elbow_Joint_Torque", "L_Wrist_Yaw_Joint_Torque", "L_Wrist_Roll_Joint_Torque", "L_Wrist_Pitch_Joint_Torque",
    "R_Thumb_Yaw_Joint_Torque", "R_Thumb_CMC_Joint_Torque", "R_Thumb_MCP_Joint_Torque",
    "R_Index_Roll_Joint_Torque", "R_Index_MCP_Joint_Torque", "R_Index_PIP_Joint_Torque",
    "R_Middle_Roll_Joint_Torque", "R_Middle_MCP_Joint_Torque", "R_Middle_PIP_Joint_Torque",
    "R_Ring_Roll_Joint_Torque", "R_Ring_MCP_Joint_Torque", "R_Ring_PIP_Joint_Torque",
    "R_Little_Roll_Joint_Torque", "R_Little_MCP_Joint_Torque", "R_Little_PIP_Joint_Torque",
    "L_Thumb_Yaw_Joint_Torque", "L_Thumb_CMC_Joint_Torque", "L_Thumb_MCP_Joint_Torque",
    "L_Index_Roll_Joint_Torque", "L_Index_MCP_Joint_Torque", "L_Index_PIP_Joint_Torque",
    "L_Middle_Roll_Joint_Torque", "L_Middle_MCP_Joint_Torque", "L_Middle_PIP_Joint_Torque",
    "L_Ring_Roll_Joint_Torque", "L_Ring_MCP_Joint_Torque", "L_Ring_PIP_Joint_Torque",
    "L_Little_Roll_Joint_Torque", "L_Little_MCP_Joint_Torque", "L_Little_PIP_Joint_Torque",
    "Head_Pan_Joint_Torque", "Head_Tilt_Joint_Torque",
    "Waist_Roll_Joint_Torque", "Waist_Pitch_Joint_Torque"
]


def get_video_metadata(video_path):
    """Get video metadata using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=height,width,codec_name,pix_fmt,r_frame_rate",
        "-of", "json", video_path,
    ]

    try:
        output = subprocess.check_output(cmd).decode("utf-8")
        probe_data = json.loads(output)
        stream = probe_data["streams"][0]
        
        # Parse frame rate
        num, den = map(int, stream["r_frame_rate"].split("/"))
        fps = num / den

        return {
            "dtype": "video",
            "shape": [stream["height"], stream["width"], 3],
            "names": ["height", "width", "channel"],
            "video_info": {
                "video.fps": fps,
                "video.codec": stream["codec_name"],
                "video.pix_fmt": stream["pix_fmt"],
                "video.is_depth_map": False,
            },
            "info": {
                "video.fps": fps,
                "video.codec": stream["codec_name"],
                "video.pix_fmt": stream["pix_fmt"],
                "video.is_depth_map": False,
                "video.height": stream["height"],
                "video.width": stream["width"],
                "video.channels": 3,
                "has_audio": False
            }
        }
    except Exception as e:
        print(f"Error getting video metadata: {e}")
        return None

def dump_jsonl(data: List[Dict[str, Any]], path: Path) -> None:
    """Dump list of dictionaries as JSONL file."""
    with open(path, 'w') as f:
        for item in data:
            json_str = json.dumps(item)
            f.write(json_str + '\n')

def json_dump(data: Dict[str, Any], path: Path, indent: int = 4) -> None:
    """Dump dictionary as JSON file."""
    with open(path, 'w') as f:
        json.dump(data, f, indent=indent)


def process_video_chunk(args):
    """Process a chunk of videos in parallel."""
    video_files, labels_dir, output_dir, fps, videos_dir = args
    results = []
    
    for video_file in video_files:
        video_id = video_file.stem
        label_file = labels_dir / f"{video_id}.txt"

        # Read annotation
        annotation = ""
        if label_file.exists():
            with open(label_file, "r") as f:
                annotation = f.read().strip()
        
        # Get video frame count
        try:
            # Try to find video in any subfolder
            video_found = False
            for folder in videos_dir.iterdir():
                if folder.is_dir():
                    video_path = folder / f"{video_id}.mp4"
                    if video_path.exists():
                        cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", 
                              "-show_entries", "stream=duration,r_frame_rate", "-of", "json", str(video_path)]
                        output = subprocess.check_output(cmd).decode()
                        probe_data = json.loads(output)
                        
                        if "streams" in probe_data and len(probe_data["streams"]) > 0:
                            stream = probe_data["streams"][0]
                            
                            if "duration" in stream:
                                duration = float(stream["duration"])
                                fps_str = stream.get("r_frame_rate", f"{fps}/1")
                                try:
                                    num, den = map(int, fps_str.split("/"))
                                    video_fps = num / den
                                    frame_count = int(duration * video_fps)
                                except Exception:
                                    frame_count = int(duration * fps)
                            else:
                                frame_count = 93
                        else:
                            frame_count = 93
                        video_found = True
                        break
            
            if not video_found:
                print(f"Warning: Could not find video for {video_id}, using default frame count")
                frame_count = 93
                
        except Exception as e:
            print(f"Error getting frame count for {video_id}: {e}")
            frame_count = 93
        
        # Ensure frame count is at least 1
        frame_count = max(1, frame_count)
        
        results.append((video_id, annotation, frame_count))
    
    return results


def copy_videos_parallel(video_copy_tasks, max_workers=16):
    """Copy multiple videos in parallel using ThreadPoolExecutor."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for source, dest in video_copy_tasks:
            futures.append(executor.submit(shutil.copy2, source, dest))
        
        # Wait for all copy operations to complete
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error copying file: {e}")

def convert_raw_to_lerobot_allex(
    raw_dir: Path,
    output_dir: Path,
    fps: int = 20,
    max_videos: int | None = None,
    num_workers: int = None,
    language_instruction: str = "Real Allex Manipulation",
):
    """Convert raw dataset to LeRobot format for Allex."""

    # Setup directories
    videos_dir = raw_dir / "videos"
    labels_dir = raw_dir / "labels"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create metadata directory
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    # Check for the new folder structure
    video_subfolders = [f for f in videos_dir.iterdir() if f.is_dir()] if videos_dir.exists() else []
    if video_subfolders:
        # New folder structure detected
        print(f"Detected new folder structure with subfolders: {[f.name for f in video_subfolders]}")
        
        # Get all unique video IDs across all subfolders
        all_video_ids = set()
        for folder in video_subfolders:
            video_files = folder.glob("*.mp4")
            all_video_ids.update(video_file.stem for video_file in video_files)
        
        # Convert to sorted list (numeric sort to avoid lexicographic ordering issues)
        # e.g., "1", "10", "100", "2" -> 1, 2, 10, 100
        try:
            # Try numeric sort first
            video_ids = sorted([int(vid) for vid in all_video_ids])
            video_ids = [str(vid) for vid in video_ids]
        except ValueError:
            # Fallback to string sort if video_ids are not numeric
        video_ids = sorted(list(all_video_ids))
        
        # Create dummy video_files list with just the IDs
        video_files = [Path(video_id) for video_id in video_ids]
    else:
        raise ValueError("No video subfolders found in the input directory")
    
    if max_videos is not None:
        video_files = video_files[:max_videos]
        print(f"Processing only first {max_videos} videos for debugging")

    print(f"Processing {len(video_files)} videos in {videos_dir}")

    # Setup multiprocessing
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() // 2)

    # Split videos into chunks for parallel processing
    chunk_size = math.ceil(len(video_files) / num_workers)
    video_chunks = [video_files[i:i + chunk_size] for i in range(0, len(video_files), chunk_size)]
    
    # Prepare arguments for parallel processing
    args_list = [
        (chunk, labels_dir, output_dir, fps, videos_dir)
        for chunk in video_chunks
    ]

    # Process videos in parallel
    total_frames = 0
    annotation_to_index = {}
    episodes_info = []
    
    print(f"Processing {len(video_files)} videos in {raw_dir} using {num_workers} workers")
    with Pool(num_workers) as pool:
        all_results = list(tqdm(
            pool.imap(process_video_chunk, args_list),
            total=len(args_list),
            desc=f"Processing {raw_dir.name}"
        ))

    # Process results and create videos
    print(f"Processing {len(all_results)} chunks of videos...")
    
    # We'll collect all video copy tasks here
    all_video_copy_tasks = []
    
    for chunk_idx, chunk_results in enumerate(all_results):
        for video_id, annotation, frame_count in tqdm(
            chunk_results,
            desc=f"Processing chunk {chunk_idx + 1}/{len(all_results)}",
            leave=False
        ):
            # Use annotation as task, or default to empty string
            if annotation:
                if annotation not in annotation_to_index:
                    annotation_to_index[annotation] = len(annotation_to_index)
                task_index = annotation_to_index[annotation]
            else:
                # Default task
                if "default" not in annotation_to_index:
                    annotation_to_index["default"] = len(annotation_to_index)
                task_index = annotation_to_index["default"]
            
            episode_index = len(episodes_info)
            
            # Create episode data for Allex format
            episode_data = {
                "observation.state": [np.zeros(48, dtype=np.float32)] * frame_count,
                "action": [np.zeros(48, dtype=np.float32)] * frame_count,
                "observation.effort": [np.zeros(48, dtype=np.float32)] * frame_count,
                "timestamp": [i/fps for i in range(frame_count)],
                "frame_index": list(range(frame_count)),
                "episode_index": [episode_index] * frame_count,
                "index": np.arange(total_frames, total_frames + frame_count),
                "task_index": [task_index] * frame_count,
                "language_instruction": [language_instruction] * frame_count,
            }
            
            # Save episode data
            episode_chunk = episode_index // CHUNKS_SIZE
            data_path = DATA_PATH.format(episode_chunk=episode_chunk, episode_index=episode_index)
            save_path = output_dir / data_path
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            df = pd.DataFrame(episode_data)
            df.to_parquet(save_path)

            # Copy videos for each view
            # For Allex, synthetic video is treated as left view, so copy to both left and right
            view_list = []
            source_video_found = None
            
            # Find the source video (could be in any subfolder)
            for folder in video_subfolders:
                source_video_path = folder / f"{video_id}.mp4"
                if source_video_path.exists():
                    source_video_found = source_video_path
                    break
            
            # If video found, copy to both left and right views
            if source_video_found:
                for view_name in ["observation.images.camera_ego_left", "observation.images.camera_ego_right"]:
                    video_save_path = output_dir / VIDEO_PATH.format(
                        episode_chunk=episode_chunk,
                        video_key=view_name,
                        episode_index=episode_index
                    )
                    video_save_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Add to copy tasks instead of copying immediately
                    all_video_copy_tasks.append((source_video_found, video_save_path))
            else:
                print(f"Warning: Video {video_id}.mp4 not found in any subfolder")

            # Update episodes info
            episodes_info.append({
                "episode_index": episode_index,
                "tasks": [task_index],
                "length": frame_count
            })
            
            total_frames += frame_count

    # Now copy all videos in parallel
    print(f"Copying {len(all_video_copy_tasks)} videos in parallel...")
    copy_videos_parallel(all_video_copy_tasks, max_workers=min(32, num_workers))

    # Generate metadata files
    # 1. tasks.jsonl
    tasks_path = meta_dir / "tasks.jsonl"
    tasks = [{"task_index": idx, "task": task} for task, idx in annotation_to_index.items()]
    dump_jsonl(tasks, tasks_path)

    # 2. episodes.jsonl
    episodes_path = meta_dir / "episodes.jsonl"
    dump_jsonl(episodes_info, episodes_path)

    # 3. info.json
    info = {
        "codebase_version": "v2.1",
        "robot_type": "allex",
        "total_episodes": len(video_files),
        "total_frames": total_frames,
        "total_tasks": len(annotation_to_index),
        "total_videos": len(video_files) * 2,  # left and right views (same synthetic video copied to both)
        "total_chunks": (len(video_files) + CHUNKS_SIZE - 1) // CHUNKS_SIZE,
        "chunks_size": CHUNKS_SIZE,
        "fps": fps,
        "splits": {
            "train": f"0:{len(video_files)}"
        },
        "data_path": DATA_PATH,
        "video_path": VIDEO_PATH,
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [48],
                "description": "Robot joint positions",
                "names": ALLEX_STATE_NAMES
            },
            "action": {
                "dtype": "float32",
                "shape": [48],
                "description": "Robot arm and hand joint commands (dual arm)",
                "names": ALLEX_ACTION_NAMES
            },
            "observation.effort": {
                "dtype": "float32",
                "shape": [48],
                "description": "Robot joint torques (dual arm)",
                "names": ALLEX_EFFORT_NAMES
            },
            "language_instruction": {
                "dtype": "string",
                "shape": [1],
                "description": "Language instruction for entire dataset",
                "names": language_instruction
            },
            "timestamp": {
                "dtype": "float32",
                "shape": [1],
                "names": None
            },
            "frame_index": {
                "dtype": "int64",
                "shape": [1],
                "names": None
            },
            "episode_index": {
                "dtype": "int64",
                "shape": [1],
                "names": None
            },
            "index": {
                "dtype": "int64",
                "shape": [1],
                "names": None
            },
            "task_index": {
                "dtype": "int64",
                "shape": [1],
                "names": None
            }
        }
    }

    # Add video features for left and right views
    # Find a sample video from any subfolder (they're all the same synthetic video)
    sample_video_path = None
    for folder in video_subfolders:
        sample_videos = list(folder.glob("*.mp4"))
        if sample_videos:
            sample_video_path = sample_videos[0]
            break
    
    if sample_video_path:
        video_meta = get_video_metadata(sample_video_path)
        if video_meta:
            # Add metadata for both left and right views
            for view_key in ["observation.images.camera_ego_left", "observation.images.camera_ego_right"]:
                info["features"][view_key] = video_meta
    
    info_path = meta_dir / "info.json"
    json_dump(info, info_path, indent=4)

    # 4. modality.json
    modality = {
        "state": {
            "right_arm_joints": {"start": 0, "end": 7},
            "right_hand_joints": {"start": 14, "end": 29},
            "left_arm_joints": {"start": 7, "end": 14},
            "left_hand_joints": {"start": 29, "end": 44},
            "neck_joints": {"start": 44, "end": 46},
            "waist_joints": {"start": 46, "end": 48}
        },
        "action": {
            "right_arm_joints": {"start": 0, "end": 7},
            "right_hand_joints": {"start": 14, "end": 29},
            "left_arm_joints": {"start": 7, "end": 14},
            "left_hand_joints": {"start": 29, "end": 44},
            "neck_joints": {"start": 44, "end": 46},
            "waist_joints": {"start": 46, "end": 48}
        },
        "video": {},
        "annotation": {
            "human.task_description": {
                "original_key": "task_index"
            }
        },
        "torque": {
            "right_arm_effort": {"start": 0, "end": 7, "original_key": "observation.effort"},
            "right_hand_effort": {"start": 14, "end": 29, "original_key": "observation.effort"},
            "left_arm_effort": {"start": 7, "end": 14, "original_key": "observation.effort"},
            "left_hand_effort": {"start": 29, "end": 44, "original_key": "observation.effort"},
            "neck_effort": {"start": 44, "end": 46, "original_key": "observation.effort"},
            "waist_effort": {"start": 46, "end": 48, "original_key": "observation.effort"}
        }
    }
    
    # Add video keys to modality for left and right views
    modality["video"]["camera_ego_left"] = {
        "original_key": "observation.images.camera_ego_left"
    }
    modality["video"]["camera_ego_right"] = {
        "original_key": "observation.images.camera_ego_right"
    }
    
    modality_path = meta_dir / "modality.json"
    json_dump(modality, modality_path, indent=4)

    return output_dir

def main():
    parser = argparse.ArgumentParser(description="Convert raw dataset to LeRobot format for Allex")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory containing videos/ and labels/")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for LeRobot dataset")
    parser.add_argument("--fps", type=int, default=20, help="Video FPS")
    parser.add_argument("--max_videos", type=int, default=None, help="Maximum number of videos to process (for debugging)")
    parser.add_argument("--num_workers", type=int, default=16, help="Total number of worker processes")
    parser.add_argument("--language_instruction", type=str, default="Real Allex Manipulation", help="Language instruction for the dataset")

    args = parser.parse_args()

    convert_raw_to_lerobot_allex(
        raw_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        fps=args.fps,
        max_videos=args.max_videos,
        num_workers=args.num_workers,
        language_instruction=args.language_instruction,
    )

if __name__ == "__main__":
    main()

