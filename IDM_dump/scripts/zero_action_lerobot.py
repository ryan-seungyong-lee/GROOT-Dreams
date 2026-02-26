"""
Copy a LeRobot dataset and zero out all actions in parquet files.
This is useful for preparing datasets for IDM action prediction.
"""

import os
import shutil
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


def zero_action_in_parquet(parquet_path: Path, action_dim: int = 48):
    """Zero out the action column in a parquet file."""
    df = pd.read_parquet(parquet_path)
    
    # Check if 'action' column exists
    if 'action' in df.columns:
        # Get the number of rows
        num_rows = len(df)
        
        # Create zero actions
        zero_actions = [np.zeros(action_dim, dtype=np.float32) for _ in range(num_rows)]
        
        # Replace action column
        df['action'] = zero_actions
        
        # Save back to parquet
        df.to_parquet(parquet_path)
        return True
    else:
        print(f"Warning: 'action' column not found in {parquet_path}")
        return False


def process_parquet_file(args):
    """Process a single parquet file."""
    parquet_path, action_dim = args
    try:
        zero_action_in_parquet(Path(parquet_path), action_dim)
        return parquet_path, True
    except Exception as e:
        print(f"Error processing {parquet_path}: {e}")
        return parquet_path, False


def main():
    parser = argparse.ArgumentParser(description="Copy LeRobot dataset and zero out actions")
    parser.add_argument("--src_dir", type=str, required=True, help="Source LeRobot dataset directory")
    parser.add_argument("--dst_dir", type=str, required=True, help="Destination directory")
    parser.add_argument("--action_dim", type=int, default=48, help="Action dimension (default: 48)")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of parallel workers")
    parser.add_argument("--skip_copy", action="store_true", help="Skip copying (if already copied)")
    
    args = parser.parse_args()
    
    src_dir = Path(args.src_dir)
    dst_dir = Path(args.dst_dir)
    
    # Step 1: Copy the entire dataset
    if not args.skip_copy:
        print(f"Copying dataset from {src_dir} to {dst_dir}...")
        if dst_dir.exists():
            print(f"Destination {dst_dir} already exists. Removing...")
            shutil.rmtree(dst_dir)
        
        shutil.copytree(src_dir, dst_dir)
        print("Copy complete!")
    else:
        print("Skipping copy step...")
    
    # Step 2: Find all parquet files
    data_dir = dst_dir / "data"
    parquet_files = list(data_dir.rglob("*.parquet"))
    print(f"Found {len(parquet_files)} parquet files")
    
    # Step 3: Zero out actions in all parquet files
    print("Zeroing out actions in parquet files...")
    
    args_list = [(str(pf), args.action_dim) for pf in parquet_files]
    
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [executor.submit(process_parquet_file, arg) for arg in args_list]
        
        success_count = 0
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            path, success = future.result()
            if success:
                success_count += 1
    
    print(f"\nDone! Successfully processed {success_count}/{len(parquet_files)} parquet files")
    print(f"Output dataset: {dst_dir}")


if __name__ == "__main__":
    main()

