from tqdm import tqdm
import argparse
import glob
import torch
import os
import sys
import json

# Add utils.py path to sys.path
sys.path.insert(0, '/virtual_lab/sjw_alinlab/suhyeok/cosmos-predict2/datasets')
from utils import sample_video_frames, set_seed, load_model, load_processor

def has_robot_prefix(prompt: str) -> bool:
    """
    Check if prompt already has the robot prefix.
    """
    prefix = "the robot arm is performing a task"
    normalized = prompt.strip().lower().rstrip(". ")
    return normalized.startswith(prefix)

def evaluate(input_json: str, video_dir: str, output_csv: str, model_name: str = "Qwen2.5-VL-7B", device: str = None, zeroshot: bool = False):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {model_name} on {device} ...")
    model = load_model(model_name, device)
    processor = load_processor(model_name)

    results = []  # list of tuples (vid_path, prompt, prediction_int)
    
    # Load JSON file
    with open(input_json, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    # Find all video files in the specified video directory and create basename mapping
    video_paths = glob.glob(os.path.join(video_dir, "**", "*.mp4"), recursive=True)
    video_basename_to_path = {os.path.basename(vid_path): vid_path for vid_path in video_paths}
    
    # Match JSON output_video with videos in directory
    matched_videos = []
    for item in json_data:
        output_video = item.get('output_video', '')
        prompt = item.get('prompt', '')
        
        if not output_video or not prompt:
            continue
        
        video_basename = os.path.basename(output_video)
        if video_basename in video_basename_to_path:
            matched_videos.append((video_basename_to_path[video_basename], prompt))
    
    if not matched_videos:
        print(f"Error: No videos found that match the JSON file entries.")
        print(f"Searched in: {video_dir}")
        return
    
    for vid_path, prompt in tqdm(matched_videos, desc="Evaluating videos"):
        # Apply prefix logic like video2world_lora.py
        prompt_prefix = "The robot arm is performing a task. "
        full_prompt = prompt if has_robot_prefix(prompt) else prompt_prefix + prompt
        
        # make sure the video path is valid
        if not os.path.exists(vid_path):
            print(f"Video path does not exist: {vid_path}")
            continue

        if zeroshot:
            user_text = (
                f"You are evaluating if a robot arm correctly follows this instruction: '{full_prompt}'\n\n"
                f"CRITICAL EVALUATION PROCESS:\n"
                f"1. FIRST CHECK: If you see HUMAN HANDS instead of robot arms, IMMEDIATELY ANSWER 0.\n"
                f"2. SECOND CHECK: Only if robot arms confirmed, verify if the instruction is followed exactly.\n\n"
                f"Remember: human hands = automatic failure (0). Be extremely strict in your judgment.\n"
                f"Reply ONLY with a single digit: 0 for failure or 1 for success."
            )
        else:
            user_text = (
                f"The video shows a robot arm completing a specific task. "
                f"Does the video follow the instruction to finish the task: '{full_prompt}'? If it fails to follow the instruction (e.g. miss the object, action or do some other actions), please answer 0. "
                f"Answer 0 for No or 1 for Yes. Reply only 0 or 1."
            )
        
        # Sample 49 frames from the video and resize to half the original size
        frame_images = sample_video_frames(vid_path, num_frames=49, scale_factor=0.3)
        
        # Create a message with multiple images (frames) instead of a video
        message_content = [{"type": "text", "text": user_text}]
        
        # Add each frame as an image
        for frame in frame_images:
            message_content.append({"type": "image", "image": frame})
        
        messages = [
            {
                "role": "user",
                "content": message_content
            }
        ]

        # Apply chat template
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Process all frames as images
        image_inputs = frame_images
        
        inputs = processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(device)

        # generate
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
        results.append((vid_path, full_prompt, pred))

    # write to CSV and compute average
    total = len(results)
    sum_preds = sum(r[2] for r in results)
    avg_score = sum_preds / total if total > 0 else 0.0

    with open(output_csv, 'w', encoding='utf-8') as f:
        f.write('video_path,prompt,prediction\n')
        for vid, pr, pred in results:
            # escape commas
            pr_esc = pr.replace(',', ' ')
            f.write(f"{vid},{pr_esc},{pred}\n")
        
        # Add summary information at the end
        f.write(f"\n# Summary\n")
        f.write(f"# Average prediction (1=Yes rate): {avg_score:.4f}\n")
        f.write(f"# Correct predictions: {sum_preds}/{total}\n")

    print(f"Evaluation done. Results saved to {output_csv}")
    print(f"Average prediction (1=Yes rate): {avg_score:.4f} ({sum_preds}/{total})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Evaluate whether videos follow given instruction prompts using Qwen models."
    )
    parser.add_argument('--input_json', type=str, required=True,
                        help='JSON file with input_video, prompt, and output_video fields')
    parser.add_argument('--video_dir', type=str, required=True,
                        help='Directory containing video files to evaluate')
    parser.add_argument('--output_csv', type=str, default='eval_results.csv',
                        help='CSV file to write results')
    parser.add_argument('--model_name', type=str, 
                        choices=["Qwen2.5-VL-7B", "Qwen3-VL-8B"],
                        default="Qwen3-VL-8B",
                        help='Model name to use for evaluation')
    parser.add_argument('--device', type=str, default=None,
                        help='Torch device (cuda or cpu)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for reproducibility')
    parser.add_argument('--zeroshot', action='store_true', help='Use zeroshot mode')

    args = parser.parse_args()
    
    set_seed(args.seed)
    evaluate(args.input_json, args.video_dir, args.output_csv, args.model_name, args.device, args.zeroshot)