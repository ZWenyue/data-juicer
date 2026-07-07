#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Precompute per-embodiment q01/q99 percentiles from LeRobot v2.1 parquet data.

Usage:
    python compute_embodiment_percentiles.py \
        --dataset_dir /path/to/LeRobot/dataset \
        --output percentiles.json \
        --embodiment galaxea_r1_lite
"""

import argparse
import glob
import json
import os

import numpy as np


def load_all_frames(dataset_dir):
    """Load all frames from all episodes, return (states, actions) as np arrays."""
    import pyarrow.parquet as pq

    chunk_dir = os.path.join(dataset_dir, "data", "chunk-000")
    parquet_files = sorted(glob.glob(os.path.join(chunk_dir, "episode_*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"No episode parquet files found in {chunk_dir}")

    all_states = []
    all_actions = []
    for pf in parquet_files:
        table = pq.read_table(pf)
        df = table.to_pandas()
        states = np.stack(df["observation.state"].values)
        actions = np.stack(df["action"].values)
        all_states.append(states)
        all_actions.append(actions)

    return np.vstack(all_states), np.vstack(all_actions)


def compute_percentiles(states, actions):
    """Compute q01 and q99 per dimension."""
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
    parser.add_argument("--dataset_dir", required=True, help="Path to LeRobot dataset directory.")
    parser.add_argument("--output", required=True, help="Output JSON file path.")
    parser.add_argument("--embodiment", default="default", help="Embodiment name (key in output JSON).")
    args = parser.parse_args()

    print(f"Loading frames from {args.dataset_dir} ...")
    states, actions = load_all_frames(args.dataset_dir)
    print(f"Loaded {states.shape[0]} frames, state_dim={states.shape[1]}, action_dim={actions.shape[1]}")

    pct = compute_percentiles(states, actions)
    num_episodes = len(glob.glob(os.path.join(args.dataset_dir, "data", "chunk-000", "episode_*.parquet")))
    pct["num_episodes"] = num_episodes

    result = {args.embodiment: pct}

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Wrote percentiles to {args.output}")
    print(f"  Embodiment: {args.embodiment}")
    print(f"  Frames: {pct['num_frames']}, Episodes: {num_episodes}")
    print(f"  State dim: {pct['state_dim']}, Action dim: {pct['action_dim']}")


if __name__ == "__main__":
    main()
