# -*- coding: utf-8 -*-
"""Diagnose Stage 2 (state_action_alignment) flagged episodes on real data.

Scans real episodes and reuses RobotStateActionAlignmentFilter's own
smoothing/lag/eps/active-frame logic (_smooth_1d, _best_lag, _eps_for_dim,
_check_pair) — not a reimplementation — to find one episode matching a given
flag pattern, then dumps the lag-aligned state/action series for each flagged
dim. This lets us tell apart two very different root causes for a "flagged"
dim:
    - genuine defect: the arm is actually moving with real amplitude, but its
      state and action trend disagree (timestamp/desync/corruption)
    - benign false positive: the arm is basically idle for the whole episode
      and the "disagreement" is just noise-floor jitter clearing the
      min_active_frames gate by chance

Usage:
    python tests_au/ops/filter/diag_stage2_flagged_dim.py \
        --pointer tests_au/ops/filter/outputs/press_analyze/lerobot_episodes_ptr.jsonl \
        --pattern left_only \
        --min-dims 4 \
        --max-scan 300
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from data_juicer._au.ops.filter.robot_state_action_alignment_filter import (  # noqa: E402
    RobotStateActionAlignmentFilter,
)
from data_juicer._au.utils.lerobot_episode_io import load_episode_arrays  # noqa: E402

# Galaxea 16-dim layout: left joints [0:7] (pad@1), left grip[7],
# right joints [8:15] (pad@9), right grip[15].
LEFT_DIMS = {0, 2, 3, 4, 5, 6}
RIGHT_DIMS = {8, 10, 11, 12, 13, 14}


def side_pattern(flagged_dims):
    left = set(flagged_dims) & LEFT_DIMS
    right = set(flagged_dims) & RIGHT_DIMS
    if left and not right:
        return "left_only"
    if right and not left:
        return "right_only"
    if left and right:
        return "mixed"
    return "none"


def build_filter():
    # Same config as analyze_lerobot_press.yaml Stage 2.
    return RobotStateActionAlignmentFilter(
        signal_source="top_level",
        top_level_state_key="states",
        top_level_action_key="actions",
        shared_dims=[0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14],
        action_is_delta=False,
        max_lag=15,
        da_threshold=0.65,
        eps_mode="range_frac",
        eps_frac=0.01,
        min_active_frames=10,
        min_frames=20,
    )


def dump_dim(op, d, state, action, T, dim_da, context):
    xs_raw = state[:, d]
    xa_raw = action[:, d]
    print(
        f"\n--- dim {d}: raw range state=[{xs_raw.min():.4f},{xs_raw.max():.4f}] "
        f"(std={xs_raw.std():.5f})  action=[{xa_raw.min():.4f},{xa_raw.max():.4f}] "
        f"(std={xa_raw.std():.5f}) ---"
    )

    xs = op._smooth_1d(xs_raw.copy())
    xa = xa_raw.copy()
    if op.action_is_delta:
        xa = np.cumsum(xa)
    xa = op._smooth_1d(xa)

    L = op._best_lag(xs, xa, op.max_lag)
    if L >= 0:
        sa, aa = xs[L:], (xa[: T - L] if L > 0 else xa)
    else:
        sa, aa = xs[:L], xa[-L:]
    m = min(len(sa), len(aa))
    sa, aa = sa[:m], aa[:m]

    ds = np.diff(sa)
    da = np.diff(aa)
    eps = op._eps_for_dim(d, xs)
    active = (np.abs(ds) > eps) | (np.abs(da) > eps)
    agree = np.sign(ds) == np.sign(da)

    print(
        f"lag L={L}, eps={eps:.6f}, active_frames={int(active.sum())}/{len(active)}, "
        f"da(reported)={dim_da.get(str(d), dim_da.get(d))}"
    )
    if int(active.sum()) == 0:
        print("(no active frames above eps -> nothing to display; this dim's DA came from < min_active_frames)")
        return

    active_idx = np.where(active)[0]
    center = active_idx[len(active_idx) // 2]
    lo = max(0, center - context)
    hi = min(m - 1, center + context)
    print("idx  | state(smooth)  | action(smooth)   |    ds    |    da    | active | agree")
    for t in range(lo, hi):
        print(
            f"{t:4d} | {sa[t]:14.5f} | {aa[t]:16.5f} | {ds[t]:8.5f} | {da[t]:8.5f} | "
            f"{'  Y' if active[t] else '  N':>6} | {'AGREE' if agree[t] else 'DISAGREE'}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pointer", required=True)
    ap.add_argument("--pattern", choices=["left_only", "right_only", "mixed", "any"], default="left_only")
    ap.add_argument("--min-dims", type=int, default=4, help="min simultaneously-flagged dims to count as a match")
    ap.add_argument("--max-scan", type=int, default=300)
    ap.add_argument("--index", type=int, default=None, help="skip scanning; inspect this pointer index directly")
    ap.add_argument("--context", type=int, default=8)
    args = ap.parse_args()

    lines = [json.loads(l) for l in open(args.pointer) if l.strip()]
    op = build_filter()

    def scan_one(rec):
        states, actions = load_episode_arrays(rec["parquet_path"])
        r = op._check_pair(states, actions)
        flagged = set(r["flagged"].keys())
        return states, actions, r, flagged

    picked = None
    if args.index is not None:
        rec = lines[args.index]
        states, actions, r, flagged = scan_one(rec)
        picked = (rec, states, actions, r, flagged)
    else:
        for i, rec in enumerate(lines[: args.max_scan]):
            try:
                states, actions, r, flagged = scan_one(rec)
            except Exception as e:
                print(f"[{i}] skip {rec.get('id')}: {e}")
                continue
            pat = side_pattern(flagged)
            print(f"[{i}] {rec['id']}: flagged={sorted(flagged)} pattern={pat}")
            if len(flagged) >= args.min_dims and (args.pattern == "any" or pat == args.pattern):
                picked = (rec, states, actions, r, flagged)
                break

    if picked is None:
        print(f"\nNo episode matched pattern={args.pattern} with >= {args.min_dims} dims in first {args.max_scan}.")
        return

    rec, states, actions, r, flagged = picked
    print(f"\n=== Matched episode: {rec['id']} ===")
    print(f"parquet: {rec['parquet_path']}")
    print(f"T={r['num_frames']}, checked_dims={r['checked_dims']}")
    print(f"dim_da={r['dim_da']}")
    print(f"dim_lag={r['dim_lag']}")
    print(f"flagged={r['flagged']}")

    T = r["num_frames"]
    state = np.asarray(states, dtype=np.float64)[:T]
    action = np.asarray(actions, dtype=np.float64)[:T]

    for d in sorted(int(x) for x in flagged):
        dump_dim(op, d, state, action, T, r["dim_da"], args.context)


if __name__ == "__main__":
    main()
