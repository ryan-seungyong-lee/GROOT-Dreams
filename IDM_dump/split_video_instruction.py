import os
import shutil
import re
import argparse
import json
from pathlib import Path

def load_failed_videos(filter_stats_file):
    """Load list of failed video paths from filter stats JSON file.
    
    Returns:
        tuple: (failed_paths set, failed_names set)
    """
    if filter_stats_file is None or not os.path.exists(filter_stats_file):
        return set(), set()
    
    try:
        with open(filter_stats_file, 'r') as f:
            stats = json.load(f)
            failed_videos = stats.get("videos_failed", [])
            # Extract video paths from the failed videos list
            failed_paths = {item.get("video_path") for item in failed_videos if "video_path" in item}
            # Also check video_name for matching
            failed_names = {item.get("video_name") for item in failed_videos if "video_name" in item}
            print(f"Loaded {len(failed_paths)} failed video paths and {len(failed_names)} failed video names from filter stats")
            return failed_paths, failed_names
    except Exception as e:
        print(f"Warning: Could not load filter stats from {filter_stats_file}: {e}")
        return set(), set()

def process_mp4_files(source_dir, output_dir, recursive=False, instruction=None, filter_stats_file=None):
    """
    Process MP4 files from source_dir and its subdirectories, extract instructions from filenames,
    and save them to output_dir with appropriate structure.
    
    Args:
        source_dir: Directory containing MP4 files or subdirectories with MP4 files
        output_dir: Directory to save processed data
        recursive: If True, maintain the directory structure from source_dir in output_dir
        instruction: If provided, use this instruction for all videos instead of extracting from filename
    """
    if not recursive:
        # Original behavior: process all MP4 files into a single output directory
        labels_dir = os.path.join(output_dir, "labels")
        videos_dir = os.path.join(output_dir, "videos")
        
        os.makedirs(labels_dir, exist_ok=True)
        os.makedirs(videos_dir, exist_ok=True)
        
        # Load failed videos from filter stats if provided
        failed_paths, failed_names = load_failed_videos(filter_stats_file)
        if failed_paths or failed_names:
            print(f"Excluding {len(failed_paths)} failed videos from filter stats")
        
        # Find all MP4 files in source_dir
        mp4_files = []
        for file in os.listdir(source_dir):
            if file.endswith('.mp4'):
                file_path = os.path.join(source_dir, file)
                # Check if this video should be excluded
                if file_path in failed_paths or file in failed_names:
                    print(f"Skipping failed video: {file}")
                    continue
                mp4_files.append(file_path)
        
        # Sort files to ensure consistent ordering
        mp4_files.sort()

        # Process each MP4 file
        for idx, mp4_path in enumerate(mp4_files, 1):
            mp4_file = os.path.basename(mp4_path)
            
            # Use provided instruction or extract from filename
            if instruction is None:
                video_instruction = re.sub(r'^\d+_', '', mp4_file).replace('.mp4', '').replace('_', ' ')
            else:
                video_instruction = instruction
            
            # Copy video with new name first
            target_video = os.path.join(videos_dir, f"{idx}.mp4")
            shutil.copy2(mp4_path, target_video)
            
            # Save label file with idx (matching the copied video filename {idx}.mp4)
            # raw_to_lerobot_allex.py expects labels/{video_id}.txt where video_id is the stem of the video file
            # Since we copy videos as {idx}.mp4, the video_id will be {idx}
            label_file = os.path.join(labels_dir, f"{idx}.txt")
            with open(label_file, 'w') as f:
                f.write(video_instruction)
            
            print(f"Processed {mp4_path} -> {idx}.mp4, instruction: {video_instruction}")
        
        print(f"Processed {len(mp4_files)} files. Results saved to {output_dir}")
    else:
        # Recursive behavior: maintain directory structure
        print(f"Processing in recursive mode, maintaining directory structure...")
        
        # Get all subdirectories in the source directory
        subdirs = []
        for item in os.listdir(source_dir):
            item_path = os.path.join(source_dir, item)
            if os.path.isdir(item_path):
                subdirs.append(item)
        
        if not subdirs:
            # If no subdirectories, process the source directory directly
            print(f"No subdirectories found in {source_dir}, processing directly...")
            process_mp4_files(source_dir, output_dir, recursive=False, instruction=instruction, filter_stats_file=filter_stats_file)
            return
        
        # Process each subdirectory
        for subdir in subdirs:
            source_subdir = os.path.join(source_dir, subdir)
            output_subdir = os.path.join(output_dir, subdir)
            
            # Create output subdirectory
            os.makedirs(output_subdir, exist_ok=True)
            
            # Process files in this subdirectory
            process_mp4_files(source_subdir, output_subdir, recursive=False, instruction=instruction, filter_stats_file=filter_stats_file)
        
        print(f"Recursive processing complete. Results saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process MP4 files from source directory and save to output directory")
    parser.add_argument("--source_dir", type=str, required=True, help="Path to source directory containing MP4 files")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output directory")
    parser.add_argument("--recursive", action="store_true", help="Process subdirectories recursively, maintaining directory structure")
    parser.add_argument("--instruction", type=str, default=None, help="Instruction to use for all videos (if not provided, extracted from filename)")
    parser.add_argument("--filter_stats_file", type=str, default=None, help="Path to filtered_videos_stats JSON file to exclude failed videos")
    args = parser.parse_args()
    
    process_mp4_files(args.source_dir, args.output_dir, args.recursive, instruction=args.instruction, filter_stats_file=args.filter_stats_file)