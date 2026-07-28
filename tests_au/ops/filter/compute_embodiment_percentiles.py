#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Precompute per-embodiment q01/q99 percentiles from LeRobot v2.1 parquet data."""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

# Prefer this workspace over other editable installs (e.g. SRC/Dta/data-juicer).
_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_juicer._au.utils.lerobot_episode_io import (  # noqa: E402
    episode_arrays_from_df,
    iter_task_dirs,
    list_episode_parquets,
    load_episode_arrays,
)


def load_all_frames(dataset_dir, max_files=None):
    """Load frames from one dataset; return (states, actions) np arrays."""
    import pyarrow.parquet as pq

    parquet_files = list_episode_parquets(dataset_dir, max_files=max_files)
    if not parquet_files:
        pattern = os.path.join(dataset_dir, "data", "chunk-*", "episode_*.parquet")
        raise FileNotFoundError(f"No episode parquet files found: {pattern}")

    all_states = []
    all_actions = []
    for pf in parquet_files:
        df = pq.read_table(pf).to_pandas()
        states, actions = episode_arrays_from_df(df)
        all_states.append(states)
        all_actions.append(actions)

    return np.vstack(all_states), np.vstack(all_actions)


def load_all_frames_root(root, max_tasks=None, max_files_per_task=None):
    """Load frames across all LeRobot tasks under root."""
    task_dirs = list(iter_task_dirs(root))
    if max_tasks is not None:
        task_dirs = task_dirs[:max_tasks]
    if not task_dirs:
        raise FileNotFoundError(f"No LeRobot tasks under {root}")

    all_states = []
    all_actions = []
    n_eps = 0
    for i, task_dir in enumerate(task_dirs):
        files = list_episode_parquets(task_dir, max_files=max_files_per_task)
        for pf in files:
            states, actions = load_episode_arrays(pf)
            all_states.append(states.astype(np.float32, copy=False))
            all_actions.append(actions.astype(np.float32, copy=False))
            n_eps += 1
        if (i + 1) % 10 == 0 or (i + 1) == len(task_dirs):
            print(f"  loaded tasks {i + 1}/{len(task_dirs)} ({n_eps} episodes)...", flush=True)

    return np.vstack(all_states), np.vstack(all_actions), n_eps, len(task_dirs)


def compute_percentiles(states, actions):
    return {
        "state": {
            "q01": np.percentile(states, 1, axis=0).tolist(),
            "q99": np.percentile(states, 99, axis=0).tolist(),
        },
        "action": {
            "q01": np.percentile(actions, 1, axis=0).tolist(),
            "q99": np.percentile(actions, 99, axis=0).tolist(),
        },
        "num_frames": int(states.shape[0]),
        "num_episodes": None,
        "state_dim": int(states.shape[1]),
        "action_dim": int(actions.shape[1]),
    }


def main():
    parser = argparse.ArgumentParser(description="Precompute per-embodiment percentiles.")
    parser.add_argument("--dataset_dir", default=None, help="Single LeRobot dataset directory.")
    parser.add_argument("--root", default=None, help="Root of many LeRobot task dirs.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--embodiment", default="default")
    parser.add_argument("--max_tasks", type=int, default=None)
    parser.add_argument("--max_files_per_task", type=int, default=None)
    args = parser.parse_args()

    if args.root:
        print(f"Loading frames from root {args.root} ...")
        states, actions, n_eps, n_tasks = load_all_frames_root(
            args.root, max_tasks=args.max_tasks, max_files_per_task=args.max_files_per_task
        )
        print(
            f"Loaded {states.shape[0]} frames from {n_eps} episodes / {n_tasks} tasks, "
            f"state_dim={states.shape[1]}, action_dim={actions.shape[1]}"
        )
        pct = compute_percentiles(states, actions)
        pct["num_episodes"] = n_eps
        pct["num_tasks"] = n_tasks
    else:
        if not args.dataset_dir:
            raise SystemExit("Provide --dataset_dir or --root")
        print(f"Loading frames from {args.dataset_dir} ...")
        states, actions = load_all_frames(args.dataset_dir)
        print(
            f"Loaded {states.shape[0]} frames, "
            f"state_dim={states.shape[1]}, action_dim={actions.shape[1]}"
        )
        pct = compute_percentiles(states, actions)
        pct["num_episodes"] = len(
            glob.glob(os.path.join(args.dataset_dir, "data", "chunk-*", "episode_*.parquet"))
        )

    result = {args.embodiment: pct}
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Wrote percentiles to {args.output}")
    print(f"  Embodiment: {args.embodiment}")
    print(f"  Frames: {pct['num_frames']}, Episodes: {pct['num_episodes']}")
    print(f"  State dim: {pct['state_dim']}, Action dim: {pct['action_dim']}")


if __name__ == "__main__":
    main()
