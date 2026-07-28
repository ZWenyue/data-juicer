# -*- coding: utf-8 -*-
"""Scan real episodes and report the ones with the largest Stage 1
(sudden_change) max_run, so the worst one can be inspected in detail with
diag_stage1_long_run.py.

Reuses RobotSuddenChangeFilter.compute_stats_single — the real production
codepath, not a reimplementation — on every episode's (states, actions).

Usage:
    python tests_au/ops/filter/diag_stage1_find_worst.py \
        --pointer tests_au/ops/filter/outputs/press_analyze/lerobot_episodes_ptr.jsonl \
        --top 5

Then inspect the worst one in detail:
    python tests_au/ops/filter/diag_stage1_long_run.py \
        --pointer tests_au/ops/filter/outputs/press_analyze/lerobot_episodes_ptr.jsonl \
        --index <idx>
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from data_juicer._au.ops.filter.robot_sudden_change_filter import (  # noqa: E402
    RobotSuddenChangeFilter,
)
from data_juicer._au.utils.lerobot_episode_io import load_episode_arrays  # noqa: E402
from data_juicer.utils.constant import Fields  # noqa: E402


def build_filter():
    # Same config as analyze_lerobot_press.yaml Stage 1.
    return RobotSuddenChangeFilter(
        signal_source="top_level",
        top_level_state_key="states",
        top_level_action_key="actions",
        threshold_mode="mad",
        mad_scale_residual=6.0,
        mad_scale_acc=6.0,
        mad_scale_jerk=6.0,
        max_flagged_ratio=0.3,
        max_run_length=10,
        min_frames=30,
        exclusion_strategy="frame_mask",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pointer", required=True)
    ap.add_argument("--max-scan", type=int, default=None, help="default: scan all episodes in the pointer file")
    ap.add_argument("--top", type=int, default=5, help="how many worst episodes to report")
    ap.add_argument("--progress-every", type=int, default=200)
    args = ap.parse_args()

    lines = [json.loads(l) for l in open(args.pointer) if l.strip()]
    if args.max_scan is not None:
        lines = lines[: args.max_scan]

    op = build_filter()
    results = []  # (max_run, flagged_ratio, index, rec)

    for i, rec in enumerate(lines):
        try:
            states, actions = load_episode_arrays(rec["parquet_path"])
        except Exception as e:
            print(f"[{i}] skip {rec.get('id')}: {e}")
            continue
        sample = {Fields.stats: {}, "states": states.tolist(), "actions": actions.tolist()}
        sample = op.compute_stats_single(sample)
        stats = sample[Fields.stats]
        results.append((stats["sudden_change_max_run"], stats["sudden_change_flagged_ratio"], i, rec))
        if args.progress_every and (i + 1) % args.progress_every == 0:
            print(f"...scanned {i + 1}/{len(lines)}")

    results.sort(key=lambda r: r[0], reverse=True)
    print(f"\nScanned {len(results)} episodes. Top {args.top} by max_run:")
    print(f"{'rank':>4} | {'index':>6} | {'max_run':>7} | {'flagged_ratio':>13} | id")
    for rank, (max_run, flagged_ratio, idx, rec) in enumerate(results[: args.top], 1):
        print(f"{rank:>4} | {idx:>6} | {max_run:>7} | {flagged_ratio:>13.4f} | {rec['id']}")

    if results:
        _, _, worst_idx, worst_rec = results[0]
        print(f"\nWorst episode: index={worst_idx} id={worst_rec['id']}")
        print(
            "Inspect with:\n"
            f"  python tests_au/ops/filter/diag_stage1_long_run.py "
            f"--pointer {args.pointer} --index {worst_idx} --block states"
        )


if __name__ == "__main__":
    main()
