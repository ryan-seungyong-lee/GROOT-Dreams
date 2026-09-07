"""
Evaluate the OpenArm IDM on the held-out test split: compare predicted action chunks
against ground-truth actions (radians), per joint group, vs a static baseline.

For each test trajectory and each query step t (strided), the IDM reads 2 frames
(t, t+H) and predicts the H-step action chunk; we compare it to GT action[t:t+H] read
straight from the parquet. Normalization (q99) uses the TRAIN stats (copied into the test
set), exactly as at training time.

Usage:
  python IDM_dump/scripts/eval_idm_openarm.py \
      --checkpoint checkpoints/idm_openarm_ego_ae20_bsz32_step20000 \
      --test-dir   ~/data/openarm-idm/test \
      --train-dir  ~/data/openarm-idm/train
"""
import argparse
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from tianshou.data import Batch
from tqdm import tqdm

from gr00t.data.dataset import LeRobotSingleDataset, ModalityConfig
from gr00t.data.schema import EmbodimentTag
from gr00t.data.transform import (
    ComposedModalityTransform,
    ConcatTransform,
    StateActionToTensor,
    StateActionTransform,
    VideoCrop,
    VideoResize,
    VideoToNumpy,
    VideoToTensor,
)
from gr00t.model.idm import IDM
from gr00t.model.transforms_idm import GR00TIDMTransform
from gr00t.utils.video import get_all_frames_and_timestamps

VIDEO_KEYS = ["video.camera_ego_left"]
STATE_KEYS = [
    "state.neck_joints", "state.left_arm_joints", "state.right_arm_joints",
    "state.left_hand_joints", "state.right_hand_joints",
]
ACTION_KEYS = [
    "action.neck_joints", "action.left_arm_joints", "action.right_arm_joints",
    "action.left_hand_joints", "action.right_hand_joints",
]
GROUPS = ["neck_joints", "left_arm_joints", "right_arm_joints",
          "left_hand_joints", "right_hand_joints"]


def infer_image_preprocess(model):
    """Read the vision tower off a loaded checkpoint and return its preprocessing name.

    Eval builds its transform stack by hand, independent of the training data config, so
    the two can disagree without any error. Deriving this from the checkpoint instead of
    a CLI flag removes the chance of forgetting it -- and the DepthAnythingTower's
    check_input_range() turns the remaining mismatch into a crash rather than a number.
    """
    import json as _json

    blob = _json.dumps(model.config.to_dict(), default=str)
    return "depth_anything" if "DepthAnythingTower" in blob else None


def build_eval_transforms(action_horizon, action_dim, video_keys=None,
                          image_preprocess=None):
    """Training transforms minus VideoColorJitter (deterministic).

    video_keys defaults to the single camera the mono configs read; pass both cameras to
    score a stereo checkpoint. Every video transform is already list-valued, so the only
    thing that changes with two keys is the view axis ConcatTransform adds.

    image_preprocess selects the vision tower's normalisation; None keeps SigLIP, which
    is what every checkpoint before the DepthAnything arm used. Pass
    infer_image_preprocess(model) rather than a literal.
    """
    vk = list(video_keys or VIDEO_KEYS)
    return ComposedModalityTransform(transforms=[
        VideoToTensor(apply_to=vk),
        VideoCrop(apply_to=vk, scale=0.95),
        VideoResize(apply_to=vk, height=224, width=224, interpolation="linear"),
        VideoToNumpy(apply_to=vk),
        StateActionToTensor(apply_to=STATE_KEYS),
        StateActionTransform(apply_to=STATE_KEYS,
                             normalization_modes={k: "q99" for k in STATE_KEYS}),
        StateActionToTensor(apply_to=ACTION_KEYS),
        StateActionTransform(apply_to=ACTION_KEYS,
                             normalization_modes={k: "q99" for k in ACTION_KEYS}),
        ConcatTransform(video_concat_order=vk,
                        state_concat_order=STATE_KEYS,
                        action_concat_order=ACTION_KEYS),
        GR00TIDMTransform(state_horizon=1, action_horizon=action_horizon,
                          max_state_dim=64, max_action_dim=action_dim,
                          image_preprocess=image_preprocess),
    ])


def make_modality_config(action_horizon, video_keys=None):
    return {
        "video": ModalityConfig(delta_indices=[0, action_horizon],
                                modality_keys=list(video_keys or VIDEO_KEYS)),
        "state": ModalityConfig(delta_indices=[0], modality_keys=STATE_KEYS),
        "action": ModalityConfig(delta_indices=list(range(action_horizon)), modality_keys=ACTION_KEYS),
    }


def collate(features, device):
    out = {}
    for k in features[0]:
        vals = [f[k] for f in features]
        if k in ["images", "view_ids"]:
            out[k] = torch.as_tensor(np.concatenate(vals), device=device)
        else:
            out[k] = torch.as_tensor(np.stack(vals), device=device)
    return out


def get_step_data(dataset, tid, base_index):
    data = {}
    dataset.curr_traj_data = dataset.get_trajectory_data(tid)
    for modality in dataset.modality_keys:
        for key in dataset.modality_keys[modality]:
            if modality in ("state", "action"):
                data[key] = dataset.get_state_or_action(tid, modality, key, base_index)
    return data


def reassemble(unapplied, action_parts, horizon):
    """Combine unapplied action.<group> parts into a [horizon, 28] array (raw index order)."""
    total = max(v["end"] for v in action_parts.values())
    arr = np.zeros((horizon, total), dtype=np.float32)
    for group, idx in action_parts.items():
        key = f"action.{group}"
        val = unapplied[key]
        if isinstance(val, torch.Tensor):
            val = val.cpu().numpy()
        val = np.asarray(val)
        if val.ndim == 3:  # [B=1, horizon, dim]
            val = val[0]
        arr[:, idx["start"]:idx["end"]] = val
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--test-dir", default=os.path.expanduser("~/data/openarm-idm/test"))
    ap.add_argument("--train-dir", default=os.path.expanduser("~/data/openarm-idm/train"))
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--stride", type=int, default=10, help="query-step stride within each episode")
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--plot-episodes", type=int, default=3)
    ap.add_argument("--video-keys", nargs="+", default=None,
                    help="cameras the checkpoint reads. Default: the single "
                         "video.camera_ego_left slot. For a stereo checkpoint pass "
                         "'video.camera_ego_left video.camera_ego_right' in the same order "
                         "as its data config's video_keys — the view embedding is indexed "
                         "by position, so the order is part of the model's interface.")
    args = ap.parse_args()

    test_dir = Path(os.path.expanduser(args.test_dir))
    train_dir = Path(os.path.expanduser(args.train_dir))
    out_dir = Path(args.output_dir) if args.output_dir else Path(args.checkpoint) / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use TRAIN normalization stats for the test set (must match training).
    # Only write when the content actually differs: several arms are scored against the
    # same test root, and an unconditional copy makes concurrent evals race each other
    # through a shared file.
    train_stats = train_dir / "meta" / "stats.json"
    assert train_stats.exists(), f"train stats not found: {train_stats} (load the train set once first)"
    dst_stats = test_dir / "meta" / "stats.json"
    if not dst_stats.exists() or dst_stats.read_bytes() != train_stats.read_bytes():
        shutil.copy2(train_stats, dst_stats)
        print(f"Copied train stats -> {dst_stats}")
    else:
        print(f"Train stats already in place at {dst_stats}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IDM.from_pretrained(args.checkpoint)
    model.requires_grad_(False).eval().to(device)
    H = model.config.action_horizon
    A = model.config.action_dim
    ipp = infer_image_preprocess(model)
    print(f"Loaded IDM: action_horizon={H}, action_dim={A}, image_preprocess={ipp}")

    dataset = LeRobotSingleDataset(
        dataset_path=str(test_dir),
        modality_configs=make_modality_config(H, args.video_keys),
        transforms=build_eval_transforms(H, A, args.video_keys, ipp),
        embodiment_tag=EmbodimentTag("new_embodiment"),
        video_backend="decord",
    )
    action_parts = json.load(open(test_dir / "meta" / "modality.json"))["action"]

    tids = list(dataset.trajectory_ids)
    if args.max_episodes:
        tids = tids[:args.max_episodes]

    # accumulators: sum of abs error per dim, and squared error, count
    n_dim = max(v["end"] for v in action_parts.values())
    abs_err = np.zeros(n_dim); sq_err = np.zeros(n_dim); cnt = 0
    base_abs = np.zeros(n_dim); base_sq = np.zeros(n_dim)  # static baseline: repeat action[t]
    per_step_abs = np.zeros((H, n_dim)); per_step_cnt = 0
    # Split the error into the part that is absolute pose and the part that is motion.
    # The IDM is video-only, so reading the arm's absolute joint angles out of pixels is
    # the whole job at step 0; the rest is how the pose evolves over the chunk. Stereo can
    # only attack the pose term (disparity -> depth -> joint angles). If a change moves the
    # motion term instead, something other than stereo geometry is doing the work.
    pose_abs = np.zeros(n_dim); motion_abs = np.zeros(n_dim); motion_cnt = 0
    plot_cache = []

    for ti, tid in enumerate(tqdm(tids, desc="eval episodes")):
        traj = dataset.get_trajectory_data(tid)
        L = len(traj)
        if L <= H:
            continue
        gt_full = np.stack(traj["action"].to_numpy())  # [L, 28]

        # load video frames once
        video_data = {}
        for key in dataset.modality_keys["video"]:
            vpath = dataset.get_video_path(tid, key.replace("video.", ""))
            frames, whole_idx = get_all_frames_and_timestamps(
                vpath.as_posix(), dataset.video_backend, dataset.video_backend_kwargs)
            video_data[key] = (frames, whole_idx)

        query_steps = list(range(0, L - H, args.stride))
        if not query_steps:
            continue

        def transform_step(t):
            sd = get_step_data(dataset, tid, t)
            ts = dataset.curr_traj_data["timestamp"].to_numpy()
            for key in video_data:
                frames, whole_idx = video_data[key]
                si = np.array(dataset.delta_indices[key]) + t
                si = np.clip(si, 0, L - 1)
                idx = np.array([np.where(np.isclose(whole_idx, v))[0][0] for v in ts[si]])
                sd[key] = frames[idx]
            return dataset.transforms(sd)

        with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
            feats = list(ex.map(transform_step, query_steps))

        preds = []
        for s in range(0, len(query_steps), args.batch_size):
            bf = feats[s:s + args.batch_size]
            bd = collate(bf, device)
            with torch.no_grad():
                out = model.get_action(bd)["action_pred"].cpu()  # [b, H, A] normalized
            for b in range(out.shape[0]):
                un = dataset.transforms.unapply(Batch(action=out[b:b+1]))
                preds.append(reassemble(un, action_parts, H))  # [H, 28] radians

        for qi, t in enumerate(query_steps):
            gt = gt_full[t:t + H]                 # [H, 28]
            pr = preds[qi]                         # [H, 28]
            static = np.repeat(gt_full[t:t+1], H, axis=0)  # predict current pose, held
            abs_err += np.abs(pr - gt).sum(0); sq_err += ((pr - gt) ** 2).sum(0)
            base_abs += np.abs(static - gt).sum(0); base_sq += ((static - gt) ** 2).sum(0)
            per_step_abs += np.abs(pr - gt); per_step_cnt += 1
            pose_abs += np.abs(pr[0] - gt[0])
            motion_abs += np.abs((pr[1:] - pr[0]) - (gt[1:] - gt[0])).sum(0)
            motion_cnt += H - 1
            cnt += H

        if ti < args.plot_episodes:
            plot_cache.append((int(tid), gt_full, query_steps, preds))

    # ---- metrics ----
    def grp_reduce(vec_abs, vec_sq):
        res = {}
        for g, idx in action_parts.items():
            sl = slice(idx["start"], idx["end"])
            res[g] = {
                "mae": float(vec_abs[sl].sum() / (cnt * (idx["end"] - idx["start"]))),
                "rmse": float(np.sqrt(vec_sq[sl].sum() / (cnt * (idx["end"] - idx["start"])))),
            }
        res["overall"] = {
            "mae": float(vec_abs.sum() / (cnt * n_dim)),
            "rmse": float(np.sqrt(vec_sq.sum() / (cnt * n_dim))),
        }
        return res

    def grp_mae(vec_abs, n):
        res = {g: float(vec_abs[idx["start"]:idx["end"]].sum() / (n * (idx["end"] - idx["start"])))
               for g, idx in action_parts.items()}
        res["overall"] = float(vec_abs.sum() / (n * n_dim))
        return res

    report = {
        "checkpoint": args.checkpoint,
        "test_dir": str(test_dir),
        "video_keys": list(args.video_keys or VIDEO_KEYS),
        "action_horizon": H, "action_dim": A,
        "n_query_chunks": per_step_cnt, "n_step_samples": cnt,
        "stride": args.stride,
        "idm": grp_reduce(abs_err, sq_err),
        "baseline_static": grp_reduce(base_abs, base_sq),
        "per_horizon_step_mae_overall": (per_step_abs.sum(1) / (per_step_cnt * n_dim)).tolist(),
        # pose: |pred[0] - gt[0]|            — absolute joint angles read from pixels
        # motion: |(pred[h]-pred[0]) - (gt[h]-gt[0])| — how the pose evolves over the chunk
        "pose_mae": grp_mae(pose_abs, per_step_cnt),
        "motion_mae": grp_mae(motion_abs, motion_cnt),
    }
    print("\n===== error decomposition (radians) =====")
    print(f'{"group":20s} {"pose":>9} {"motion":>9}')
    for g in list(action_parts) + ["overall"]:
        print(f'{g:20s} {report["pose_mae"][g]:9.4f} {report["motion_mae"][g]:9.4f}')
    with open(out_dir / "eval_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n===== IDM vs GT (radians) =====")
    print(f"{'group':<18}{'IDM MAE':>10}{'base MAE':>10}{'IDM RMSE':>10}")
    for g in GROUPS + ["overall"]:
        i = report["idm"][g]; b = report["baseline_static"][g]
        print(f"{g:<18}{i['mae']:>10.4f}{b['mae']:>10.4f}{i['rmse']:>10.4f}")
    print(f"\nReport -> {out_dir/'eval_report.json'}")

    # ---- plots ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for tid, gt_full, qsteps, preds in plot_cache:
            # plot a few representative dims: right_arm joint0 (idx 9), left_hand joint0 (idx 16)
            dims = {"right_arm_j1": 9, "left_hand_j1": 16, "left_arm_j1": 2}
            fig, axes = plt.subplots(len(dims), 1, figsize=(10, 7), sharex=True)
            for ax, (name, d) in zip(axes, dims.items()):
                ax.plot(gt_full[:, d], label="GT", color="black", lw=1.5)
                for qi, t in enumerate(qsteps):
                    xs = np.arange(t, t + preds[qi].shape[0])
                    ax.plot(xs, preds[qi][:, d], color="tab:red", alpha=0.5, lw=0.8)
                ax.set_ylabel(name); ax.legend(loc="upper right", fontsize=8)
            axes[-1].set_xlabel("frame")
            fig.suptitle(f"IDM pred (red chunks) vs GT (black) — test episode {tid}")
            fig.tight_layout()
            fig.savefig(out_dir / f"overlay_ep{tid:06d}.png", dpi=110)
            plt.close(fig)
        print(f"Plots -> {out_dir}/overlay_ep*.png")
    except Exception as e:
        print(f"(plotting skipped: {e})")


if __name__ == "__main__":
    main()
