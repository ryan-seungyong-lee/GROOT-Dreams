#!/usr/bin/env python
"""Refit the action -> observation.state model on the merged 20Hz corpus.

Why refit
---------
The shipped constants were fitted on openarm-teleop-v4-rldx-v2, where the physical LEFT hand
never closed past 0.2 rad in any of 635 episodes. A gain fitted on a signal that does not
move is unidentifiable, and indeed the shipped left-hand gains all sit near 1.0 (0.79-1.03)
while the right hand's spread 0.30-0.98 -- the real contact behaviour. The merged corpus now
contains 0.86 h of teleop in which both hands grasp equally, so the left-hand parameters can
be identified for the first time.

Second change: one time constant for all 28 joints was a compromise. Cross-correlating action
against state on the new teleop measures the arms lagging 167 ms and the hands 267 ms, so
alpha is fitted per group rather than globally.

Model, unchanged in form:
    s[t]     = (1 - alpha_g) * s[t-1] + alpha_g * action[t-1]     first-order lag, per group
    state[t] = gain_j * s[t] + offset_j                            per joint

Fitted on the train split, reported on the held-out test split. This is a proxy for
generated video, where only actions exist -- for real teleop the measured state is used
directly and none of this applies.
"""
from __future__ import annotations

import argparse, glob, json, os
import numpy as np
import pyarrow.parquet as pq

GROUPS = {"neck": slice(0, 2), "l_arm": slice(2, 9), "r_arm": slice(9, 16),
          "l_hand": slice(16, 22), "r_hand": slice(22, 28)}
GROUP_OF = np.empty(28, dtype=object)
for g, s in GROUPS.items():
    GROUP_OF[s] = g


def load(split_root):
    A, S = [], []
    for f in sorted(glob.glob(f"{split_root}/data/chunk-*/*.parquet")):
        t = pq.read_table(f, columns=["action", "observation.state"])
        col = lambda n: np.vstack([np.asarray(x, np.float32) for x in
                                   t.column(n).to_numpy(zero_copy_only=False)])
        A.append(col("action")); S.append(col("observation.state"))
    return A, S


def lowpass(a, alpha_per_joint):
    s = np.empty_like(a)
    s[0] = a[0]
    one_minus = 1.0 - alpha_per_joint
    for t in range(1, len(a)):
        s[t] = one_minus * s[t - 1] + alpha_per_joint * a[t - 1]
    return s


def resid(A, S, alpha_pj, gain=None, off=None):
    # width comes from the data, not a hardcoded 28: the alpha search calls this with a
    # single group's slice.
    num = np.zeros(A[0].shape[1]); cnt = 0
    for a, st in zip(A, S):
        p = lowpass(a, alpha_pj)
        if gain is not None:
            p = p * gain + off
        num += np.abs(p - st).sum(0); cnt += len(a)
    return num / cnt


def main() -> None:
    H = os.path.expanduser("~/data")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=f"{H}/openarm-idm-v3v4-teleop20-stereo")
    ap.add_argument("--out", default=os.path.expanduser(
        "~/project/anyh2r/GR00T-Dreams/IDM_dump/configs/state_model_teleop20.json"))
    a = ap.parse_args()

    print("loading train ...", flush=True)
    Atr, Str = load(f"{a.dataset}/train")
    print(f"  {len(Atr)} episodes, {sum(len(x) for x in Atr):,} frames")
    Ate, Ste = load(f"{a.dataset}/test")
    print(f"  test: {len(Ate)} episodes, {sum(len(x) for x in Ate):,} frames")

    # ---- 1. alpha per group -------------------------------------------------
    print("\n=== alpha per group (grid search on train) ===")
    grid = np.round(np.arange(0.05, 0.81, 0.025), 4)
    alpha_g = {}
    for g, sl in GROUPS.items():
        best, bestr = None, np.inf
        for al in grid:
            r = resid([x[:, sl] for x in Atr], [x[:, sl] for x in Str],
                      np.full(sl.stop - sl.start, al, np.float32)).mean()
            if r < bestr:
                bestr, best = r, float(al)
        alpha_g[g] = best
        print(f"  {g:8s} alpha {best:.3f}  (tau {50.0/best:.0f} ms)  resid {bestr:.4f}")
    alpha_pj = np.array([alpha_g[GROUP_OF[j]] for j in range(28)], np.float32)

    # ---- 2. gain/offset per joint, least squares ----------------------------
    # Accumulate sufficient statistics so nothing large is held in memory.
    print("\n=== gain/offset per joint (least squares on train) ===")
    n = 0.0
    Sx = np.zeros(28); Sy = np.zeros(28); Sxx = np.zeros(28); Sxy = np.zeros(28)
    for aa, ss in zip(Atr, Str):
        p = lowpass(aa, alpha_pj)
        n += len(aa)
        Sx += p.sum(0); Sy += ss.sum(0)
        Sxx += (p * p).sum(0); Sxy += (p * ss).sum(0)
    den = n * Sxx - Sx * Sx
    gain = np.where(np.abs(den) > 1e-9, (n * Sxy - Sx * Sy) / np.where(den == 0, 1, den), 1.0)
    off = (Sy - gain * Sx) / n
    # A joint that never moves gives a degenerate fit; fall back to identity + mean offset.
    spread = np.sqrt(np.maximum(Sxx / n - (Sx / n) ** 2, 0.0))
    degenerate = spread < 1e-3
    if degenerate.any():
        print(f"  {int(degenerate.sum())} joints have no excursion to fit; identity+offset used")
        gain = np.where(degenerate, 1.0, gain)
        off = np.where(degenerate, (Sy - Sx) / n, off)
    gain = gain.astype(np.float32); off = off.astype(np.float32)
    for g, sl in GROUPS.items():
        print(f"  {g:8s} gain {np.round(gain[sl],3).tolist()}")

    # ---- 3. compare on the held-out test split ------------------------------
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from idm_chunk_fusion import STATE_ALPHA, STATE_GAIN, STATE_OFFSET
    old_pj = np.full(28, STATE_ALPHA, np.float32)
    rows = [
        ("action[t-1] (구)", resid(Ate, Ste, np.ones(28, np.float32))),
        ("shipped fitted", resid(Ate, Ste, old_pj, STATE_GAIN, STATE_OFFSET)),
        ("new: alpha only", resid(Ate, Ste, alpha_pj)),
        ("new: alpha+gain/off", resid(Ate, Ste, alpha_pj, gain, off)),
    ]
    print(f"\n=== held-out test split, |model - measured state| (rad) ===")
    hdr = f"{'model':22s}" + "".join(f"{g:>9}" for g in GROUPS) + f"{'ALL28':>9}"
    print(hdr); print("-" * len(hdr))
    for name, r in rows:
        cells = "".join(f"{r[sl].mean():9.4f}" for sl in GROUPS.values())
        print(f"{name:22s}{cells}{r.mean():9.4f}")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"alpha_per_group": alpha_g,
               "alpha_per_joint": alpha_pj.tolist(),
               "gain": gain.tolist(), "offset": off.tolist(),
               "fitted_on": a.dataset + "/train",
               "train_frames": int(n)}, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
