#!/usr/bin/env python
"""Resample the 30Hz teleop set to 20Hz and merge it into the stereo IDM training corpus.

Why 20Hz and not the native 30Hz
--------------------------------
The IDM reads frames t and t+H and predicts H actions, with H=20 fixed by the checkpoint.
At 20Hz that window is 1.000 s; at 30Hz it is 0.667 s. Mixing rates would hand the model two
different physical meanings for one nominal horizon, with the recording era's appearance as
the only cue for which applies -- exactly the kind of shortcut that produced the left-hand
failure. Inference also happens on 20fps generated video, so 20Hz is the deployment rate.

The resampling is exact where it matters. 30:20 is 3:2, so keeping every frame whose index
is not 2 (mod 3) yields output frame k = input frame floor(1.5k). Twenty output steps then
span exactly thirty input frames = exactly 1.000 s, so the (t, t+20) pair the model reads is
bit-for-bit the same interval as in the 20Hz corpus. Only the intermediate steps alternate
33/67 ms, which sits far below the measured servo lag (arms 167 ms, hands 267 ms).

No static filtering
-------------------
The video duplicate rate is 6.7%, but measured on the action signal those duplicates are
isolated single frames: static runs have a median length of 1 frame and runs lasting a second
or more cover 0.0-0.7% of the corpus at any sensible threshold. There is no idle stretch to
cut, and deleting scattered frames would break the fixed time grid for no gain.

Output is one merged stereo dataset plus its mirror overlay (symmetrised stats only), so the
existing training command works unchanged apart from the dataset path.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SRC_FPS, DST_FPS = 30, 20
CHUNK = 1000
L_KEY = "observation.images.camera_ego_left"
R_KEY = "observation.images.camera_ego_right"
VIDEO_KEYS = [L_KEY, R_KEY]
COLUMNS = ["observation.state", "action", "language_instruction", "torque",
           "next.done", "next.success", "timestamp", "frame_index",
           "episode_index", "index", "task_index"]
NEW_TASK = "free bimanual teleoperation for inverse dynamics"
# 28-dim mirror convention, identical to gr00t/data/transform/mirror.py
NECK, L_ARM, R_ARM = slice(0, 2), slice(2, 9), slice(9, 16)
L_HAND, R_HAND = slice(16, 22), slice(22, 28)
NECK_SIGN = np.array([1.0, -1.0], np.float32)
ARM_SIGN = np.array([-1, -1, -1, 1, -1, -1, -1], np.float32)


def keep_indices(n_src: int) -> np.ndarray:
    """30Hz -> 20Hz: drop every third frame. Output k == input floor(1.5k)."""
    idx = np.arange(n_src)
    return idx[idx % 3 != 2]


def mirror_rows(a: np.ndarray) -> np.ndarray:
    o = np.empty_like(a)
    o[:, NECK] = a[:, NECK] * NECK_SIGN
    o[:, L_ARM] = a[:, R_ARM] * ARM_SIGN
    o[:, R_ARM] = a[:, L_ARM] * ARM_SIGN
    o[:, L_HAND] = a[:, R_HAND]
    o[:, R_HAND] = a[:, L_HAND]
    return o


def summarize(a: np.ndarray) -> dict:
    return {"mean": a.mean(0).tolist(), "std": a.std(0).tolist(),
            "min": a.min(0).tolist(), "max": a.max(0).tolist(),
            "q01": np.quantile(a, .01, axis=0).tolist(),
            "q99": np.quantile(a, .99, axis=0).tolist()}


def resample_video(src: Path, dst: Path, keep: np.ndarray, crf: int = 18) -> int:
    """Decode, select the kept frames, re-encode at DST_FPS. Returns frames written."""
    import cv2
    dst.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(src))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    want = set(int(i) for i in keep)
    p = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{w}x{h}", "-r", str(DST_FPS), "-i", "pipe:0",
         "-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p", "-an", str(dst)],
        stdin=subprocess.PIPE)
    n = written = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if n in want:
            p.stdin.write(f.tobytes())
            written += 1
        n += 1
    cap.release()
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {dst}")
    return written


def main() -> None:
    H = Path(os.path.expanduser("~/data"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--teleop", type=Path, default=H / "idm-teleop-lerobot")
    ap.add_argument("--base", type=Path, default=H / "openarm-idm-v3v4-10hz-stereo")
    ap.add_argument("--out", type=Path, default=H / "openarm-idm-v3v4-teleop20-stereo")
    ap.add_argument("--mirror-out", type=Path,
                    default=H / "openarm-idm-v3v4-teleop20-stereo-mirror")
    ap.add_argument("--test-ratio", type=float, default=0.05)
    ap.add_argument("--min-frames", type=int, default=41,
                    help="an episode must hold t and t+20 with margin")
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    teleop_eps = [json.loads(l) for l in open(a.teleop / "meta/episodes.jsonl")]
    print(f"teleop source: {len(teleop_eps)} episodes @ {SRC_FPS}Hz")

    # ---- resample the teleop episodes ------------------------------------
    staged = []
    for e in teleop_eps:
        i = e["episode_index"]
        src_pq = a.teleop / f"data/chunk-{i//CHUNK:03d}/episode_{i:06d}.parquet"
        tbl = pq.read_table(src_pq)
        n_src = tbl.num_rows
        keep = keep_indices(n_src)
        if len(keep) < a.min_frames:
            print(f"  ep{i}: SKIP, {len(keep)} frames after resample")
            continue
        staged.append(dict(src_ep=i, src_pq=src_pq, keep=keep, n=len(keep)))
    print(f"  {len(staged)} episodes survive; "
          f"{sum(s['n'] for s in staged):,} frames @ {DST_FPS}Hz "
          f"({sum(s['n'] for s in staged)/DST_FPS/3600:.2f} h)")

    # ---- split ------------------------------------------------------------
    order = rng.permutation(len(staged))
    n_test = max(1, int(round(len(staged) * a.test_ratio)))
    test_ids = set(order[:n_test].tolist())
    print(f"  split: {len(staged)-n_test} train / {n_test} test")

    base_counts = {}
    for split in ("train", "test"):
        info = json.load(open(a.base / split / "meta/info.json"))
        base_counts[split] = (info["total_episodes"], info["total_frames"])
    print(f"base: train {base_counts['train']}  test {base_counts['test']}")

    # ---- build each split -------------------------------------------------
    for split in ("train", "test"):
        out = a.out / split
        base = a.base / split
        shutil.rmtree(out, ignore_errors=True)
        (out / "meta").mkdir(parents=True, exist_ok=True)

        tasks = [json.loads(l) for l in open(base / "meta/tasks.jsonl")]
        task_uid = {t["task"]: t["task_index"] for t in tasks}
        eps_rows = [json.loads(l) for l in open(base / "meta/episodes.jsonl")]
        stats_rows = [json.loads(l) for l in open(base / "meta/episodes_stats.jsonl")]
        base_ep, base_frames = base_counts[split]
        src_map = json.load(open(base / "meta/split_map.json"))
        if isinstance(src_map, dict):
            src_map = list(src_map.values())

        # base episodes keep their indices; videos are linked, parquet copied verbatim
        for e in eps_rows:
            i = e["episode_index"]
            d = out / f"data/chunk-{i//CHUNK:03d}"
            d.mkdir(parents=True, exist_ok=True)
            shutil.copy2(base / f"data/chunk-{i//CHUNK:03d}/episode_{i:06d}.parquet",
                         d / f"episode_{i:06d}.parquet")
        for key in VIDEO_KEYS:
            src_v = base / f"videos/chunk-000/{key}"
            dst_v = out / f"videos/chunk-000/{key}"
            dst_v.mkdir(parents=True, exist_ok=True)
            for f in sorted(src_v.glob("*.mp4")):
                link = dst_v / f.name
                os.symlink(os.path.relpath(f, dst_v), link)   # relative: tree stays movable

        mine = [s for k, s in enumerate(staged)
                if (k in test_ids) == (split == "test")]
        if NEW_TASK not in task_uid and mine:
            task_uid[NEW_TASK] = len(tasks)
            tasks.append({"task_index": task_uid[NEW_TASK], "task": NEW_TASK})
        ti = task_uid.get(NEW_TASK, 0)

        jobs, offset, new_ep = [], base_frames, base_ep
        added_rows = []
        for s in mine:
            tbl = pq.read_table(s["src_pq"])
            sel = tbl.take(pa.array(s["keep"]))
            n = sel.num_rows
            col = lambda name: np.vstack([np.asarray(x, np.float32) for x in
                                          sel.column(name).to_numpy(zero_copy_only=False)])
            A, S = col("action"), col("observation.state")
            T = col("torque") if sel.column("torque").null_count == 0 else \
                np.full((n, 14), np.nan, np.float32)
            out_t = pa.table({
                "observation.state": pa.array(list(S), type=pa.list_(pa.float32(), 28)),
                "action": pa.array(list(A), type=pa.list_(pa.float32(), 28)),
                "language_instruction": pa.array([NEW_TASK] * n, type=pa.string()),
                "torque": pa.array(list(T), type=pa.list_(pa.float32(), 14)),
                "next.done": pa.array([k == n - 1 for k in range(n)], type=pa.bool_()),
                "next.success": pa.array([k == n - 1 for k in range(n)], type=pa.bool_()),
                "timestamp": pa.array(np.arange(n, dtype=np.float32) / DST_FPS,
                                      type=pa.float32()),
                "frame_index": pa.array(np.arange(n), type=pa.int64()),
                "episode_index": pa.array([new_ep] * n, type=pa.int64()),
                "index": pa.array(np.arange(offset, offset + n), type=pa.int64()),
                "task_index": pa.array([ti] * n, type=pa.int64()),
            })
            d = out / f"data/chunk-{new_ep//CHUNK:03d}"
            d.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.table({c: out_t.column(c) for c in COLUMNS}),
                           d / f"episode_{new_ep:06d}.parquet")

            for key, eye in ((L_KEY, "left"), (R_KEY, "right")):
                src_v = (a.teleop / f"videos/chunk-{s['src_ep']//CHUNK:03d}/"
                                    f"observation.images.camera_ego_{eye}/"
                                    f"episode_{s['src_ep']:06d}.mp4")
                dst_v = out / f"videos/chunk-{new_ep//CHUNK:03d}/{key}/episode_{new_ep:06d}.mp4"
                jobs.append((src_v, dst_v, s["keep"], n))

            eps_rows.append({"episode_index": new_ep, "tasks": [NEW_TASK], "length": n})
            stats_rows.append({"episode_index": new_ep, "stats": {
                "action": {"min": A.min(0).tolist(), "max": A.max(0).tolist(),
                           "mean": A.mean(0).tolist(), "std": A.std(0).tolist(),
                           "count": [n]},
                "observation.state": {"min": S.min(0).tolist(), "max": S.max(0).tolist(),
                                      "mean": S.mean(0).tolist(), "std": S.std(0).tolist(),
                                      "count": [n]}}})
            src_map.append({"new_ep": new_ep, "source": "teleop20", "category": "teleop",
                            "src_ep": s["src_ep"], "view": None, "views": ["left", "right"],
                            "src_start": 0, "src_end": int(s["keep"][-1]) + 1,
                            "clipped": False, "length": n, "src_fps": SRC_FPS})
            added_rows.append((A, S))
            offset += n
            new_ep += 1

        # videos, in parallel -- decode/select/re-encode is the slow part
        bad = []
        with ThreadPoolExecutor(max_workers=a.jobs) as ex:
            for (src_v, dst_v, keep, want), got in zip(
                    jobs, ex.map(lambda j: resample_video(j[0], j[1], j[2]), jobs)):
                if got != want:
                    bad.append((dst_v.name, want, got))
        if bad:
            raise SystemExit(f"frame count mismatch after resample: {bad[:5]}")

        # ---- meta -----------------------------------------------------------
        def wj(rows, p):
            with open(p, "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
        wj(eps_rows, out / "meta/episodes.jsonl")
        wj(stats_rows, out / "meta/episodes_stats.jsonl")
        wj(tasks, out / "meta/tasks.jsonl")
        (out / "meta/split_map.json").write_text(json.dumps(src_map, indent=1))
        shutil.copy2(base / "meta/modality.json", out / "meta/modality.json")

        info = json.load(open(base / "meta/info.json"))
        info["total_episodes"] = new_ep
        info["total_frames"] = offset
        info["total_tasks"] = len(tasks)
        info["total_videos"] = new_ep * 2
        info["total_chunks"] = max(1, (new_ep + CHUNK - 1) // CHUNK)
        info["splits"] = {split: f"0:{new_ep}"}
        (out / "meta/info.json").write_text(json.dumps(info, indent=4))

        # dataset-level stats over base + new, and the mirror-symmetrised variant
        allA, allS = [], []
        for e in eps_rows:
            i = e["episode_index"]
            t = pq.read_table(out / f"data/chunk-{i//CHUNK:03d}/episode_{i:06d}.parquet",
                              columns=["action", "observation.state"])
            g = lambda nm: np.vstack([np.asarray(x, np.float32) for x in
                                      t.column(nm).to_numpy(zero_copy_only=False)])
            allA.append(g("action")); allS.append(g("observation.state"))
        A = np.concatenate(allA); S = np.concatenate(allS)
        (out / "meta/stats.json").write_text(json.dumps(
            {"action": summarize(A), "observation.state": summarize(S)}, indent=4))
        Am = np.concatenate([A, mirror_rows(A)]); Sm = np.concatenate([S, mirror_rows(S)])
        (out / "meta/stats_mirror.json").write_text(json.dumps(
            {"action": summarize(Am), "observation.state": summarize(Sm)}, indent=4))
        print(f"[{split}] {new_ep} episodes ({new_ep-base_ep} new) {offset:,} frames "
              f"({offset/DST_FPS/3600:.2f} h)")

    # ---- mirror overlay: same tree, symmetrised stats.json ----------------
    shutil.rmtree(a.mirror_out, ignore_errors=True)
    for split in ("train", "test"):
        m = a.mirror_out / split
        (m / "meta").mkdir(parents=True, exist_ok=True)
        for sub in ("data", "videos"):
            os.symlink(os.path.relpath(a.out / split / sub, m), m / sub)
        for f in (a.out / split / "meta").glob("*"):
            if f.name == "stats.json":
                continue
            os.symlink(os.path.relpath(f, m / "meta"), m / "meta" / f.name)
        shutil.copy2(a.out / split / "meta/stats_mirror.json", m / "meta/stats.json")
    print(f"\nmirror overlay -> {a.mirror_out}  (stats.json = symmetrised; rest symlinked)")
    print("DONE")


if __name__ == "__main__":
    main()
