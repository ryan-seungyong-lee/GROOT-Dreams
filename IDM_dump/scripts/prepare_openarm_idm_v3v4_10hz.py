#!/usr/bin/env python
"""Prepare a merged OpenArm teleop v3 + v4 dataset for IDM training, with the
v4 frame-drop problem handled by clipping instead of whole-episode exclusion.

Successor to prepare_openarm_idm_v3v4.py. That script dropped any v4 episode whose
cameras had stalled (idm_frame_gap_excludes.json, 30 of 257) and kept the rest whole,
which still left plenty of held frames inside the survivors. Here the source pool is
split three ways instead:

  v3          — whole episodes, video symlinked from the 512x288 re-encode.
                The pixel scan says every v3 episode-view holds >=10Hz end to end,
                so there is nothing to clip.
  v4_new      — the Aug 4-5 sessions (source_dataset prefix '202608'). image_gaps.jsonl
                reports dup=0 / max_gap=0 for all of them, i.e. a clean 20Hz. Whole
                episodes, video symlinked.
  v4_old_10hz — the Jul 23-25 sessions. Only the >=10Hz runs survive, taken per
                (episode, view) from openarm-r2h/out/valid_segments_10hz.json and cut
                into one output episode each. ~19.7% of those frames survive.

Why per-view and not unioned over the two cameras: the IDM emits each camera as an
independent sample, so a stall on the left camera must not shorten the right one's clip.
(v4_build_rldx_dataset.py unions them because RLDX consumes both views of one episode
together.)

The Aug 6 in-place rebuild of ~/data/openarm-teleop-v4 appended the new episodes and
left the old indices untouched — verified by matching every scanned episode's frame
count against meta/episodes.jsonl — so the Aug 1 segment list still addresses the right
episodes and does not need to be regenerated.

Clips are re-encoded rather than symlinked because gr00t's loader seeks video by the
parquet's `timestamp` column (dataset.py get_video -> get_frames_by_timestamps), so a
clip's timestamps must be rebased to start at zero and the video's PTS must agree.

Everything else follows the previous script: unified task table, stratified 95:5 split
done at the source-episode level so an episode's views and clips never straddle
train/test, both cameras emitted into the single camera_ego_left slot the IDM reads,
canonical 11-column parquet order.

Also writes meta/stats_mirror.json alongside meta/stats.json: the same statistics
computed over the union of the original and left-right-mirrored action/state
distributions, for the MirrorLeftRight augmentation ablation. Normalising mirrored
samples with unmirrored q99 is the one way that augmentation breaks silently.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

DATA = Path("/sjw_alinlab/home/seungyong/project/anyh2r/data")
SEGMENTS_JSON = Path("/sjw_alinlab/home/seungyong/project/anyh2r/openarm-r2h/out/valid_segments_10hz.json")

V3_CATS = ["bottle", "cup", "doll", "snack"]
V4_CATS = ["ball", "ball-fail", "banana", "banana-fail", "bottle", "bottle-fail",
           "box", "box-fail", "doll", "doll-fail"]

VIDEO_KEY = "observation.images.camera_ego_left"
SRC_VIEW_KEYS = {"left": "observation.images.camera_ego_left",
                 "right": "observation.images.camera_ego_right"}
COLUMNS = ["observation.state", "action", "language_instruction", "torque",
           "next.done", "next.success", "timestamp", "frame_index",
           "episode_index", "index", "task_index"]
SEED = 42
FPS = 20

# A clip must support at least one full IDM sample: observations at t and t+20 plus
# actions t..t+19, so t <= n-21. n=41 leaves 21 usable start positions.
MIN_CLIP = 41

NEW_SESSION_PREFIX = "202608"

# ---------------------------------------------------------------------------
# Left-right mirror map over the 28-d action/state vector.
#   neck(0:2) | left_arm(2:9) | right_arm(9:16) | left_hand(16:22) | right_hand(22:28)
#
# mirrored[i] = MIRROR_SIGN[i] * x[MIRROR_PERM[i]]
#
# Arm signs [-1,-1,-1,+1,-1,-1,-1] were read off the data, not a URDF: over 100 v3 and
# 100 v4 episodes the frame-0 home pose has |L+R| 5-20x smaller than |L-R| for joints
# 1,2,3,5,6,7 and L~=R~=+1.55 for joint 4. Joint 4 is the elbow, whose axis is
# perpendicular to the sagittal plane and so survives the reflection unchanged.
# Hand joints are non-negative flexion magnitudes (min=0 throughout) on mirror-image
# hardware, so they swap without a sign change. Head yaw negates; head pitch does not.
# ---------------------------------------------------------------------------
ARM_SIGN = [-1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0]
MIRROR_PERM = np.array(
    [0, 1]                       # head pitch, yaw
    + list(range(9, 16))         # left_arm  <- right_arm
    + list(range(2, 9))          # right_arm <- left_arm
    + list(range(22, 28))        # left_hand  <- right_hand
    + list(range(16, 22)),       # right_hand <- left_hand
    dtype=np.int64,
)
MIRROR_SIGN = np.array(
    [1.0, -1.0] + ARM_SIGN + ARM_SIGN + [1.0] * 12,
    dtype=np.float32,
)
# torque is arms only: left_arm_torque(0:7) | right_arm_torque(7:14)
TORQUE_PERM = np.array(list(range(7, 14)) + list(range(0, 7)), dtype=np.int64)
TORQUE_SIGN = np.array(ARM_SIGN + ARM_SIGN, dtype=np.float32)


def mirror_rows(arr: np.ndarray, perm: np.ndarray, sign: np.ndarray) -> np.ndarray:
    """Apply the mirror map to an (N, D) array. Involutive: f(f(x)) == x."""
    return arr[:, perm] * sign


def read_jsonl(p):
    with open(p) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(rows, p):
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


# ---------------------------------------------------------------------------
# Source pool
# ---------------------------------------------------------------------------


def load_segments(path: Path) -> dict:
    """(dataset, category, episode, view) -> list of half-open [start, end) ranges."""
    out = {}
    for r in json.load(open(path)):
        out[(r["dataset"], r["category"], r["episode"], r["view"])] = [
            (int(s), int(e)) for s, e in r["segments"]
        ]
    return out


def v4_new_episodes(v4_root: Path, cat: str) -> set[int]:
    """Episode indices contributed by the Aug 4-5 re-collection."""
    p = v4_root / cat / "meta" / "source_episodes.jsonl"
    if not p.exists():
        return set()
    return {r["episode_index"] for r in read_jsonl(p)
            if str(r.get("source_dataset", "")).startswith(NEW_SESSION_PREFIX)}


def intersect_intervals(a, b):
    """Intersection of two sorted lists of disjoint half-open [s, e) ranges.

    A stereo sample needs BOTH cameras fresh at the same frames, so the per-view >=10Hz
    segments have to be intersected, not unioned. Note this is not
    v4_build_rldx_dataset.py:merge_intervals — that one is a union, the wrong primitive
    here. Touching intervals are deliberately NOT merged: make_valid_segments.py emits
    [i, j+1) where j is the first bad frame and restarts at j+1, so two touching segments
    are separated by exactly one held frame, and merging would quietly relax the >=10Hz
    guarantee the segments exist to provide.
    """
    out, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        s = max(a[i][0], b[j][0])
        e = min(a[i][1], b[j][1])
        if s < e:
            out.append((s, e))
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return out


def collect(v3_root, v4_root, v3_video_root, v4_cats, segments, views, min_clip,
            pair_policy="per-view"):
    """Build the source-episode pool. Each record carries the output units it expands
    into, so the train/test split can be taken before expansion.

    A unit is (view, start, end). `view` is None when the range is shared by both cameras
    (pair_policy intersect/union), which is what a stereo episode needs; under the legacy
    per-view policy each camera keeps its own ranges and `view` names the camera.
    """
    task_uid, tasks = {}, []
    episodes = []
    stats = {"v3": 0, "v4_new": 0, "v4_old_10hz": 0, "dropped_no_segment": 0}

    def register(task_strs):
        for ts in task_strs:
            if ts not in task_uid:
                task_uid[ts] = len(tasks)
                tasks.append(ts)

    for cat in V3_CATS:
        cat_dir = v3_root / cat
        for ep in read_jsonl(cat_dir / "meta" / "episodes.jsonl"):
            register(ep["tasks"])
            episodes.append({
                "kind": "v3", "category": cat, "root": v3_root,
                "video_root": v3_video_root, "src_ep": ep["episode_index"],
                "length": ep["length"], "task_strs": ep["tasks"],
                "primary_task": ep["tasks"][0],
                # v3 is >=10Hz end to end on both cameras, so the whole episode is one
                # shared range regardless of policy.
                "units": ([(v, 0, ep["length"]) for v in views]
                          if pair_policy == "per-view" else [(None, 0, ep["length"])]),
            })
            stats["v3"] += 1

    for cat in v4_cats:
        cat_dir = v4_root / cat
        new_eps = v4_new_episodes(v4_root, cat)
        for ep in read_jsonl(cat_dir / "meta" / "episodes.jsonl"):
            idx, n = ep["episode_index"], ep["length"]
            register(ep["tasks"])
            common = {"category": cat, "root": v4_root, "video_root": None,
                      "src_ep": idx, "length": n, "task_strs": ep["tasks"],
                      "primary_task": ep["tasks"][0]}
            if idx in new_eps:
                # dup=0 / max_gap=0 on BOTH cameras, so the whole episode is shared.
                episodes.append({**common, "kind": "v4_new",
                                 "units": ([(v, 0, n) for v in views]
                                           if pair_policy == "per-view" else [(None, 0, n)])})
                stats["v4_new"] += 1
                continue
            units = []
            if pair_policy == "per-view":
                for v in views:
                    for s, e in segments.get(("v4", cat, idx, v), []):
                        e = min(e, n)  # the scan ran on ogvideo; clamp defensively
                        if e - s >= min_clip:
                            units.append((v, s, e))
            else:
                per_view = [[(s, min(e, n)) for s, e in segments.get(("v4", cat, idx, v), [])]
                            for v in views]
                if pair_policy == "intersect":
                    shared = per_view[0]
                    for other in per_view[1:]:
                        shared = intersect_intervals(shared, other)
                else:  # union — a frame survives if EITHER camera is fresh, so the other
                       # camera's staleness is unbounded. Offered, never the default.
                    merged = sorted(r for pv in per_view for r in pv)
                    shared = []
                    for s, e in merged:
                        if shared and s <= shared[-1][1]:
                            shared[-1] = (shared[-1][0], max(shared[-1][1], e))
                        else:
                            shared.append((s, e))
                units = [(None, s, e) for s, e in shared if e - s >= min_clip]
            if not units:
                stats["dropped_no_segment"] += 1
                continue
            episodes.append({**common, "kind": "v4_old_10hz", "units": units})
            stats["v4_old_10hz"] += 1

    return episodes, tasks, task_uid, stats


def stratified_split(episodes, test_ratio):
    """Unchanged from prepare_openarm_idm_v3v4.py, but grouped by (kind, category) so
    the three source kinds are each represented in the test split."""
    rng = np.random.default_rng(SEED)
    groups = {}
    for i, e in enumerate(episodes):
        groups.setdefault((e["kind"], e["category"]), []).append(i)
    test = set()
    for _, idxs in sorted(groups.items()):
        n_test = int(round(test_ratio * len(idxs)))
        if n_test == 0:
            continue
        by_task = {}
        for i in idxs:
            by_task.setdefault(episodes[i]["primary_task"], []).append(i)
        lists = list(by_task.values())
        for v in lists:
            rng.shuffle(v)
        rng.shuffle(lists)
        picked, ti = [], 0
        while len(picked) < n_test:
            lst = lists[ti % len(lists)]
            if lst:
                picked.append(lst.pop())
            ti += 1
            if all(not v for v in lists):
                break
        test.update(picked)
    return test


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def write_parquet(src, dst, new_ep, offset, task_map, start, end):
    """Slice [start, end) out of the source episode and renumber it as new_ep.

    frame_index / timestamp / index are rebased so the clip stands alone; timestamp in
    particular must start at 0 and step by 1/FPS, because that is what the video decoder
    seeks on and the re-encoded clip's PTS are rebased the same way.
    """
    t = pq.read_table(src)
    if (start, end) != (0, t.num_rows):
        t = t.slice(start, end - start)
    n = t.num_rows
    cols = {name: t.column(name) for name in t.column_names}
    cols["episode_index"] = pa.array([new_ep] * n, type=pa.int64())
    cols["index"] = pa.array(list(range(offset, offset + n)), type=pa.int64())
    cols["frame_index"] = pa.array(list(range(n)), type=pa.int64())
    cols["timestamp"] = pa.array(np.arange(n, dtype=np.float32) / FPS, type=pa.float32())
    cols["task_index"] = pa.array([task_map[v] for v in t.column("task_index").to_pylist()],
                                  type=pa.int64())
    out = pa.table({c: cols[c] for c in COLUMNS})
    pq.write_table(out, dst)
    return out, n


def cut_video(job):
    """ffmpeg parameters follow openarm-r2h/scripts/v4_build_rldx_dataset.py:cut_video,
    where they were tuned against frame-exactness: -g 2 -bf 0 keeps the all-I/P
    structure the sources have so seeking stays exact, and an explicit CFR '-r 20' after
    setpts is a 1:1 frame mapping that the muxer cannot drop or duplicate the tail of.
    """
    src, dst, start, end = job
    cmd = [
        "ffmpeg", "-v", "error", "-y", "-nostdin", "-i", str(src),
        "-vf", f"trim=start_frame={start}:end_frame={end},setpts=N/FRAME_RATE/TB",
        "-r", str(FPS),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-g", "2", "-bf", "0",
        "-an", "-movflags", "+faststart", str(dst),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return str(dst)


def episode_stats(table) -> dict:
    """Recomputed per output episode. Nothing in gr00t reads meta/episodes_stats.jsonl,
    but copying the source episode's stats onto a clip would be actively wrong."""
    out = {}
    for name in table.column_names:
        col = table.column(name)
        if pa.types.is_string(col.type) or pa.types.is_large_string(col.type):
            continue
        try:
            v = col.to_numpy(zero_copy_only=False)
            arr = np.vstack([np.asarray(x, dtype=np.float32) for x in v]) \
                if v.dtype == object else np.asarray(v, dtype=np.float32).reshape(len(v), -1)
        except (ValueError, TypeError):
            continue
        out[name] = {"min": arr.min(0).tolist(), "max": arr.max(0).tolist(),
                     "mean": arr.mean(0).tolist(), "std": arr.std(0).tolist(),
                     "count": [int(arr.shape[0])]}
    return out


def build_split(name, ep_indices, episodes, out_root, tasks, task_uid,
                base_info, base_modality, jobs, layout="mono", views=("left", "right")):
    ds = out_root / name
    chunks_size = int(base_info.get("chunks_size", 1000))
    (ds / "meta").mkdir(parents=True, exist_ok=True)
    # mono flattens every camera into the single slot the 1-view config reads; stereo keeps
    # them apart so the data config can name both.
    out_keys = ([SRC_VIEW_KEYS[v] for v in views] if layout == "stereo" else [VIDEO_KEY])

    def chunk_dirs(ep):
        c = ep // chunks_size
        d = ds / "data" / f"chunk-{c:03d}"
        d.mkdir(parents=True, exist_ok=True)
        vs = {}
        for k in out_keys:
            v = ds / "videos" / f"chunk-{c:03d}" / k
            v.mkdir(parents=True, exist_ok=True)
            vs[k] = v
        return d, vs

    local_to_uid = {}
    for e in episodes:
        key = (e["kind"], e["category"])
        if key in local_to_uid:
            continue
        cat_dir = e["root"] / e["category"]
        local_to_uid[key] = {r["task_index"]: task_uid[r["task"]]
                             for r in read_jsonl(cat_dir / "meta" / "tasks.jsonl")}

    eps_rows, stats_rows, split_map, cut_jobs = [], [], [], []
    low_dim = []            # (28,)-columns kept in memory for the dataset-level stats
    offset = total = new_ep = 0

    for gi in ep_indices:
        e = episodes[gi]
        cat_dir = e["root"] / e["category"]
        src_ep = e["src_ep"]
        # A unit whose view is None is a range both cameras share. In stereo layout it
        # becomes ONE episode carrying both cameras; in mono layout it fans out to one
        # episode per camera. That is what lets both arms come off the same pool and the
        # same train/test split, so they differ only by pairing.
        emissions = []
        for view, start, end in e["units"]:
            if layout == "stereo":
                assert view is None, "stereo layout needs view-less units (--pair-policy intersect/union)"
                emissions.append((list(views), start, end))
            elif view is None:
                emissions.extend(([v], start, end) for v in views)
            else:
                emissions.append(([view], start, end))

        for unit_views, start, end in emissions:
            dchunk, vchunks = chunk_dirs(new_ep)
            table, n = write_parquet(
                cat_dir / "data" / "chunk-000" / f"episode_{src_ep:06d}.parquet",
                dchunk / f"episode_{new_ep:06d}.parquet",
                new_ep, offset, local_to_uid[(e["kind"], e["category"])], start, end)
            offset += n
            total += n
            low_dim.append({
                c: np.vstack([np.asarray(x, dtype=np.float32)
                              for x in table.column(c).to_numpy(zero_copy_only=False)])
                for c in ("observation.state", "action", "torque")
            })

            vroot = (Path(e["video_root"]) / e["category"] if e["video_root"] else cat_dir)
            whole = (start == 0 and end == e["length"])
            for i, view in enumerate(unit_views):
                out_key = out_keys[i] if layout == "stereo" else VIDEO_KEY
                src_vid = (vroot / "videos" / "chunk-000" / SRC_VIEW_KEYS[view]
                           / f"episode_{src_ep:06d}.mp4").resolve()
                dst_vid = vchunks[out_key] / f"episode_{new_ep:06d}.mp4"
                if dst_vid.exists() or dst_vid.is_symlink():
                    dst_vid.unlink()
                if whole:
                    os.symlink(src_vid, dst_vid)
                else:
                    cut_jobs.append((src_vid, dst_vid, start, end))

            eps_rows.append({"episode_index": new_ep, "tasks": e["task_strs"], "length": n})
            stats_rows.append({"episode_index": new_ep, "stats": episode_stats(table)})
            split_map.append({"new_ep": new_ep, "source": e["kind"],
                              "category": e["category"], "src_ep": src_ep,
                              "view": unit_views[0] if len(unit_views) == 1 else None,
                              "views": list(unit_views),
                              "src_start": start, "src_end": end,
                              "clipped": not whole, "length": n})
            new_ep += 1

    if cut_jobs:
        print(f"[{name}] re-encoding {len(cut_jobs)} clips with {jobs} workers...")
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            for i, _ in enumerate(pool.map(cut_video, cut_jobs, chunksize=4), 1):
                if i % 100 == 0 or i == len(cut_jobs):
                    print(f"    {i}/{len(cut_jobs)}")

    write_jsonl(eps_rows, ds / "meta" / "episodes.jsonl")
    write_jsonl(stats_rows, ds / "meta" / "episodes_stats.jsonl")
    write_jsonl([{"task_index": i, "task": t} for i, t in enumerate(tasks)],
                ds / "meta" / "tasks.jsonl")

    info = json.loads(json.dumps(base_info))
    n_chunks = max(1, (new_ep + chunks_size - 1) // chunks_size)
    info.update(total_episodes=new_ep, total_frames=total, total_tasks=len(tasks),
                total_videos=new_ep * len(out_keys), total_chunks=n_chunks,
                splits={"train": f"0:{new_ep}"})
    if layout != "stereo":
        # the base metadata (from v4) declares both cameras; a mono dataset ships one
        info["features"].pop("observation.images.camera_ego_right", None)
    (ds / "meta" / "info.json").write_text(json.dumps(info, indent=4))

    mod = json.loads(json.dumps(base_modality))
    if "video" in mod and layout != "stereo":
        mod["video"] = {"camera_ego_left": mod["video"]["camera_ego_left"]}
    (ds / "meta" / "modality.json").write_text(json.dumps(mod, indent=4))
    (ds / "meta" / "split_map.json").write_text(json.dumps(split_map, indent=2))

    write_dataset_stats(ds, low_dim)

    by_src = {}
    for r in split_map:
        by_src[r["source"]] = by_src.get(r["source"], 0) + 1
    n_clipped = sum(r["clipped"] for r in split_map)
    print(f"[{name}] source_episodes={len(ep_indices)} -> episodes={new_ep} "
          f"frames={total} clipped={n_clipped} by_source={by_src}")


def write_dataset_stats(ds: Path, low_dim: list[dict]):
    """meta/stats.json in the exact shape gr00t's calculate_dataset_statistics emits,
    plus meta/stats_mirror.json over the original-union-mirrored distribution.

    Writing it here means the trainer does not re-scan every parquet on startup, and it
    is the only place the mirrored normalisation constants can come from — under
    MirrorLeftRight(p=0.5) the left and right marginals become identical, and q99 taken
    from the unmirrored data would clip the augmented samples.
    """
    if not low_dim:
        return
    cat = {c: np.concatenate([d[c] for d in low_dim], axis=0)
           for c in ("observation.state", "action", "torque")}

    def summarize(a):
        return {"mean": np.mean(a, 0).tolist(), "std": np.std(a, 0).tolist(),
                "min": np.min(a, 0).tolist(), "max": np.max(a, 0).tolist(),
                "q01": np.quantile(a, 0.01, axis=0).tolist(),
                "q99": np.quantile(a, 0.99, axis=0).tolist()}

    plain = {k: summarize(v) for k, v in cat.items()}
    (ds / "meta" / "stats.json").write_text(json.dumps(plain, indent=4))

    mirrored = {}
    for k, v in cat.items():
        perm, sign = ((TORQUE_PERM, TORQUE_SIGN) if k == "torque"
                      else (MIRROR_PERM, MIRROR_SIGN))
        m = mirror_rows(v, perm, sign)
        assert np.allclose(mirror_rows(m, perm, sign), v, atol=1e-5), \
            f"mirror map is not involutive on {k}"
        mirrored[k] = summarize(np.concatenate([v, m], axis=0))
    (ds / "meta" / "stats_mirror.json").write_text(json.dumps(mirrored, indent=4))
    print(f"    wrote stats.json + stats_mirror.json ({cat['action'].shape[0]} frames)")


# ---------------------------------------------------------------------------


def make_mirror_view(base: Path, out: Path, splits=("train", "test")):
    """A second dataset root that is the base one with stats_mirror.json swapped in.

    data/ and videos/ are symlinked, so this costs no disk. It exists only so the mirror
    run can point --dataset-path somewhere whose meta/stats.json holds the symmetrised
    normalisation constants, without touching the loader.
    """
    for split in splits:
        src, dst = base / split, out / split
        if not src.exists():
            continue
        (dst / "meta").mkdir(parents=True, exist_ok=True)
        for sub in ("data", "videos"):
            link = dst / sub
            if link.is_symlink() or link.exists():
                link.unlink()
            os.symlink((src / sub).resolve(), link)
        for f in sorted((src / "meta").iterdir()):
            link = dst / "meta" / f.name
            if link.is_symlink() or link.exists():
                link.unlink()
            if f.name == "stats.json":
                continue
            os.symlink(f.resolve(), link)
        # the one real file: symmetrised stats standing in for stats.json
        (dst / "meta" / "stats.json").write_text((src / "meta" / "stats_mirror.json").read_text())
    print(f"mirror view -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v3", type=Path, default=DATA / "openarm-teleop-v3")
    ap.add_argument("--v4", type=Path, default=DATA / "openarm-teleop-v4")
    ap.add_argument("--segments", type=Path, default=SEGMENTS_JSON)
    ap.add_argument("--out", type=Path,
                    default=Path(os.path.expanduser("~/data/openarm-idm-v3v4-10hz")))
    ap.add_argument("--mirror-out", type=Path,
                    default=Path(os.path.expanduser("~/data/openarm-idm-v3v4-10hz-mirror")))
    ap.add_argument("--test-ratio", type=float, default=0.05)
    ap.add_argument("--views", nargs="+", default=["left", "right"], choices=["left", "right"])
    ap.add_argument("--v3-video-root", type=Path, default=DATA / "openarm-teleop-v3-512x288",
                    help="tree holding v3 videos re-encoded to v4's 512x288 "
                         "(pass an empty string to use the originals)")
    ap.add_argument("--no-v4-fail", action="store_true",
                    help="drop the *-fail categories (failed demonstrations)")
    ap.add_argument("--min-clip", type=int, default=MIN_CLIP)
    ap.add_argument("--jobs", type=int, default=16, help="parallel ffmpeg workers")
    ap.add_argument("--layout", default="mono", choices=["mono", "stereo"],
                    help="mono: one episode per camera into the single camera_ego_left "
                         "slot. stereo: one episode carrying both camera keys.")
    ap.add_argument("--pair-policy", default="per-view",
                    choices=["per-view", "intersect", "union"],
                    help="how the old-v4 >=10Hz segments become clip ranges. per-view "
                         "reproduces the historical mono dataset. intersect keeps only "
                         "frames where BOTH cameras are fresh (required for a simultaneous "
                         "stereo pair, and the right pool to build a comparable mono arm "
                         "from). union is unbounded in the other camera's staleness.")
    a = ap.parse_args()
    if a.layout == "stereo" and a.pair_policy == "per-view":
        raise SystemExit("--layout stereo needs --pair-policy intersect (or union): a "
                         "per-view pool has no shared ranges to pair.")

    v4_cats = [c for c in V4_CATS if not (a.no_v4_fail and c.endswith("-fail"))]
    v3_vroot = a.v3_video_root if a.v3_video_root and str(a.v3_video_root) else None
    if v3_vroot and not v3_vroot.exists():
        raise SystemExit(f"v3 re-encoded video tree missing: {v3_vroot}")
    if not a.segments.exists():
        raise SystemExit(f"segment list missing: {a.segments}")

    segments = load_segments(a.segments)
    print(f"v3 videos from : {v3_vroot or 'originals (640x480)'}")
    print(f"segment list   : {a.segments} ({len(segments)} episode-view records)")
    print(f"min clip length: {a.min_clip} frames")

    print(f"layout / policy: {a.layout} / {a.pair_policy}")
    episodes, tasks, task_uid, st = collect(
        a.v3, a.v4, v3_vroot, v4_cats, segments, a.views, a.min_clip, a.pair_policy)
    n_units = sum(len(e["units"]) for e in episodes)
    print(f"source episodes: v3={st['v3']} v4_new={st['v4_new']} "
          f"v4_old_10hz={st['v4_old_10hz']} dropped(no >=10Hz clip)={st['dropped_no_segment']}")
    print(f"output units   : {n_units} | unified tasks={len(tasks)}")
    counts = {}
    for e in episodes:
        k = (e["kind"], e["category"])
        counts[k] = counts.get(k, (0, 0))
        counts[k] = (counts[k][0] + 1, counts[k][1] + len(e["units"]))
    for k in sorted(counts):
        print(f"  {k[0]:12s} {k[1]:14s} src_eps={counts[k][0]:4d} units={counts[k][1]:5d}")

    test_ids = sorted(stratified_split(episodes, a.test_ratio))
    train_ids = [i for i in range(len(episodes)) if i not in set(test_ids)]
    print(f"split -> train={len(train_ids)} test={len(test_ids)} "
          f"(ratio={len(test_ids)/len(episodes):.3f})")

    a.out.mkdir(parents=True, exist_ok=True)
    base_info = json.load(open(a.v4 / "ball" / "meta" / "info.json"))
    base_modality = json.load(open(a.v4 / "ball" / "meta" / "modality.json"))
    build_split("train", train_ids, episodes, a.out, tasks, task_uid,
                base_info, base_modality, a.jobs, a.layout, tuple(a.views))
    build_split("test", test_ids, episodes, a.out, tasks, task_uid,
                base_info, base_modality, a.jobs, a.layout, tuple(a.views))
    make_mirror_view(a.out, a.mirror_out)
    print(f"DONE -> {a.out}")


if __name__ == "__main__":
    main()
