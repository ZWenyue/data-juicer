# -*- coding: utf-8 -*-
"""Diagnose Stage 1 (sudden_change) long flagged runs on a real episode.

Reuses RobotSuddenChangeFilter's own smoothing/threshold/flag logic (not a
reimplementation) to dump raw vs. smoothed signal + residual + threshold
around the longest consecutive flagged run, so we can visually/numerically
tell whether it is genuine sustained motion mis-flagged, or a real anomaly.

Usage:
    python tests_au/ops/filter/diag_stage1_long_run.py \
        --pointer tests_au/ops/filter/outputs/press_analyze/lerobot_episodes_ptr.jsonl \
        --index 0 \
        --block states
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from data_juicer._au.ops.filter.robot_sudden_change_filter import (  # noqa: E402
    RobotSuddenChangeFilter,
)
from data_juicer._au.utils.lerobot_episode_io import load_episode_arrays  # noqa: E402


def longest_run(mask: np.ndarray):
    best_start = best_len = cur_start = cur_len = 0
    for i, v in enumerate(mask):
        if v:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_len, best_start = cur_len, cur_start
        else:
            cur_len = 0
    return best_start, best_len


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pointer", required=True)
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--block", choices=["states", "actions"], default="states")
    ap.add_argument("--context", type=int, default=5, help="extra frames printed before/after the run")
    args = ap.parse_args()

    lines = [json.loads(l) for l in open(args.pointer) if l.strip()]
    rec = lines[args.index]
    print(f"episode: {rec['id']}")
    print(f"parquet: {rec['parquet_path']}")

    states, actions = load_episode_arrays(rec["parquet_path"])
    raw = states if args.block == "states" else actions
    T = raw.shape[0]
    print(f"T={T}, D={raw.shape[1]}")

    # Same config as analyze_lerobot_press.yaml Stage 1
    op = RobotSuddenChangeFilter(
        signal_source="top_level",
        threshold_mode="mad",
        mad_scale_residual=6.0,
        mad_scale_acc=6.0,
        mad_scale_jerk=6.0,
        max_flagged_ratio=0.3,
        max_run_length=10,
        min_frames=30,
        exclusion_strategy="frame_mask",
    )

    det = op._detect_block(raw)
    ff = det["frame_flags"]
    print(f"num_frames={det['num_frames']}, num_flagged={int(ff.sum())} ({ff.mean():.1%})")
    print(f"bad_dims={det['bad_dims']}")

    start, length = longest_run(ff)
    print(f"\nLongest consecutive flagged run: frames [{start}, {start + length}) length={length}")

    # Recompute the same internal quantities _detect_block used, for the bad dims only.
    dims = op._select_dims(raw.shape[1])
    xs = raw[:, dims].astype(np.float64)
    smooth = op._cascaded_smooth(xs)
    residual = np.abs(xs - smooth)
    acc, jerk = op._finite_diff(xs)
    tr = op._dim_threshold(residual, op.mad_scale_residual, op.residual_threshold)
    ta = op._dim_threshold(np.abs(acc), op.mad_scale_acc, op.acc_threshold)
    tj = op._dim_threshold(np.abs(jerk), op.mad_scale_jerk, op.jerk_threshold)

    lo = max(0, start - args.context)
    hi = min(T, start + length + args.context)

    for local_i, d in enumerate(dims):
        seg_flag = ff[lo:hi]
        if not seg_flag.any():
            continue
        print(f"\n--- dim {d} (thr residual={tr[local_i]:.5f}, acc={ta[local_i]:.5f}, jerk={tj[local_i]:.5f}) ---")
        print("frame | raw       | smooth    | residual  | acc       | jerk      | flagged")
        for t in range(lo, hi):
            print(
                f"{t:5d} | {xs[t, local_i]:9.4f} | {smooth[t, local_i]:9.4f} | "
                f"{residual[t, local_i]:9.5f} | {acc[t, local_i]:9.5f} | {jerk[t, local_i]:9.5f} | "
                f"{'FLAG' if ff[t] else ''}"
            )

    # Global sanity: how much of the trajectory is "near-still" (tiny MAD source)?
    still_frac = {}
    for local_i, d in enumerate(dims):
        v = xs[:, local_i]
        step = np.abs(np.diff(v))
        still_frac[d] = float(np.mean(step < 1e-6)) if len(step) else 0.0
    print("\nFraction of near-zero-motion steps per dim (diagnoses MAD floor collapse):")
    for d, f in sorted(still_frac.items(), key=lambda kv: -kv[1])[:10]:
        print(f"  dim {d}: {f:.1%} of steps are ~0")


if __name__ == "__main__":
    main()
