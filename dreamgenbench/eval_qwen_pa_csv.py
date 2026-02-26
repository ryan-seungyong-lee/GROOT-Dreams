#!/usr/bin/env python3
import os
import sys
import pandas as pd
import torch
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
import argparse

# Add utils.py path to sys.path
sys.path.insert(0, '/sjw_alinlab3/home/suhyeok/cosmos-predict2/datasets')
from utils import load_model, load_processor, set_seed
from collections import defaultdict


def evaluate(input_csv: str, output_csv: str, model_name: str = "Qwen2.5-VL-7B", device: str = None, easy_mode: bool = False):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {model_name} on {device} ...")
    model = load_model(model_name, device)
    processor = load_processor(model_name)

    results = []
    
    if not os.path.exists(input_csv):
        print(f"Error: input_csv {input_csv} not found.")
        return

    df = pd.read_csv(input_csv)
    video_paths = df['videopath'].tolist()
    
    print(f"Loaded {len(video_paths)} videos from {input_csv}.")
    
    for vid_path in tqdm(video_paths, desc="Evaluating videos"):
        if not os.path.exists(vid_path):
            print(f"Warning: video file {vid_path} not found. Skipping.")
            continue

        user_text = (
            f"The video shows a robot arm completing a specific task. "
            f"Does the video show good physics dynamics and showcase a good alignment with the physical world? Please be a strict judge. If it breaks the laws of physics, please answer 0. "
            f"Answer 0 for No or 1 for Yes. Reply only 0 or 1."
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": vid_path},
                    {"type": "text",  "text": user_text},
                ],
            }
        ]

        # prepare inputs
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(device)

        # generate
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=4)
            
        # trim prompt tokens
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        # decode
        output = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        raw_reply = output[0].strip()
        # parse to int prediction (0 or 1)
        pred = 0
        if raw_reply and raw_reply[0] == '1':
            pred = 1
            
        # record
        results.append((vid_path, pred))

    total = len(results)
    sum_preds = sum(r[1] for r in results)
    avg_score = sum_preds / total if total > 0 else 0.0

    # Best-of-N Aggregation Logic
    # Even if input_csv has only one video per prompt, this logic handles it gracefully.
    prompt_stats = defaultdict(list)
    for vid_path, pred in results:
        # Assuming path structure: .../PROMPT_TEXT/bestofn_N/video.mp4
        try:
            parent = os.path.dirname(vid_path)      # bestofn_N
            grandparent = os.path.dirname(parent)   # PROMPT_TEXT
            prompt = os.path.basename(grandparent)
            prompt_stats[prompt].append(pred)
        except:
            continue

    best_of_n_scores = {}
    for prompt, preds in prompt_stats.items():
        # Any success (1) means the prompt is successfully handled in Best-of-N setting
        score = 1 if any(p == 1 for p in preds) else 0
        best_of_n_scores[prompt] = score

    best_of_n_acc = sum(best_of_n_scores.values()) / len(best_of_n_scores) if best_of_n_scores else 0.0
    num_prompts = len(best_of_n_scores)

    with open(output_csv, 'w', encoding='utf-8') as f:
        f.write('video_path,prediction\n')
        for vid, pred in results:
            f.write(f"{vid},{pred}\n")
            
        # Add Prompt-level Best-of-N details
        f.write("\n# Prompt-level Best-of-N Scores\n")
        f.write("prompt,score\n")
        for prompt, score in best_of_n_scores.items():
            # Escape commas in prompt if any
            clean_prompt = prompt.replace(",", " ")
            f.write(f"{clean_prompt},{score}\n")

        # Add summary information at the end
        f.write(f"\n# Summary\n")
        f.write(f"# Average prediction (1=Yes rate): {avg_score:.4f}\n")
        f.write(f"# Correct predictions: {sum_preds}/{total}\n")
        f.write(f"# --------------------------------\n")
        f.write(f"# Best-of-N Accuracy (Any 1 = Success): {best_of_n_acc:.4f}\n")
        f.write(f"# Number of Prompts: {num_prompts}\n")

    print(f"Evaluation done. Results saved to {output_csv}")
    print(f"Average prediction (1=Yes rate): {avg_score:.4f} ({sum_preds}/{total})")
    print(f"Best-of-N Accuracy: {best_of_n_acc:.4f} ({sum(best_of_n_scores.values())}/{num_prompts})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Evaluate whether videos follow given instruction prompts using Qwen models."
    )
    parser.add_argument('--input_csv', type=str, required=True,
                        help='Input CSV file containing videopath and caption columns')
    parser.add_argument('--output_csv', type=str, default='eval_results.csv',
                        help='CSV file to write results')
    parser.add_argument('--model_name', type=str, 
                        choices=["Qwen2.5-VL-7B", "Qwen3-VL-8B"],
                        default="Qwen2.5-VL-7B",
                        help='Model name to use for evaluation')
    parser.add_argument('--device', type=str, default=None,
                        help='Torch device (cuda or cpu)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for reproducibility')
    parser.add_argument('--easy_mode', action='store_true',
                    help='Easy mode for evaluation')
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    evaluate(args.input_csv, args.output_csv, args.model_name, args.device, args.easy_mode)
