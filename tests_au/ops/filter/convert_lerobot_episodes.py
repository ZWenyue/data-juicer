#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert LeRobot v2.1 dataset (per-frame parquet) to per-episode JSONL for DJ.

Supports:
  - unified columns: observation.state / action
  - Galaxea decomposed columns packed into 16-dim arms+grippers layout
  - pointer mode: only parquet_path (for large multi-task corpora)
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

# Prefer this workspace over other editable installs (e.g. SRC/Dta/data-juicer).
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pyarrow.parquet as pq

from data_juicer._au.utils.lerobot_episode_io import (  # noqa: E402
    ARM_SLOT_LEFT,
    ARM_SLOT_RIGHT,
    episode_arrays_from_df,
    iter_task_dirs,
    list_episode_parquets,
    pack_decomposed_to_16,
)


def _to_nested_list(arr):
    return np.asarray(arr, dtype=float).tolist()


def convert(dataset_dir: str, output: str):
    """Materialize states/actions into JSONL (legacy / small datasets)."""
    parquet_files = list_episode_parquets(dataset_dir)
    if not parquet_files:
        pattern = os.path.join(dataset_dir, "data", "chunk-*", "episode_*.parquet")
        raise FileNotFoundError(f"No parquet files found: {pattern}")

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    with open(output, "w", encoding="utf-8") as f:
        for pf in parquet_files:
            df = pq.read_table(pf).to_pandas()
            ep_idx = int(df["episode_index"].iloc[0])
            ep_id = f"episode_{ep_idx:06d}"
            states, actions = episode_arrays_from_df(df)
            record = {
                "id": ep_id,
                "episode_index": ep_idx,
                "num_frames": int(states.shape[0]),
                "states": _to_nested_list(states),
                "actions": _to_nested_list(actions),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Converted {len(parquet_files)} episodes -> {output}")


def convert_pointer_root(root: str, output: str, max_tasks: int = None):
    """Scan a root of LeRobot tasks and write pointer-only JSONL."""
    os.makedirs(os.path.dirname(os.path.abspath(output)) or ".", exist_ok=True)
    task_dirs = list(iter_task_dirs(root))
    if max_tasks is not None:
        task_dirs = task_dirs[:max_tasks]
    if not task_dirs:
        raise FileNotFoundError(f"No LeRobot task dirs under {root}")

    n = 0
    with open(output, "w", encoding="utf-8") as f:
        for task_dir in task_dirs:
            task_name = os.path.basename(task_dir.rstrip("/"))
            for pf in list_episode_parquets(task_dir):
                # Prefer filename index; avoid reading parquet for speed.
                base = os.path.splitext(os.path.basename(pf))[0]
                # episode_000012
                try:
                    ep_idx = int(base.split("_")[-1])
                except ValueError:
                    ep_idx = -1
                ep_id = f"{task_name}/{base}"
                record = {
                    "id": ep_id,
                    "text": ep_id,
                    "task": task_name,
                    "episode_index": ep_idx,
                    "parquet_path": os.path.abspath(pf),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                n += 1
    print(f"Wrote {n} pointer episodes from {len(task_dirs)} tasks -> {output}")
    return n


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default=None, help="Single LeRobot dataset dir")
    parser.add_argument("--root", default=None, help="Root containing many LeRobot tasks")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--mode",
        choices=["materialize", "pointer"],
        default="materialize",
        help="materialize=embed states/actions; pointer=only parquet_path",
    )
    parser.add_argument("--max_tasks", type=int, default=None)
    args = parser.parse_args()

    if args.mode == "pointer":
        root = args.root or args.dataset_dir
        if not root:
            raise SystemExit("pointer mode requires --root or --dataset_dir")
        convert_pointer_root(root, args.output, max_tasks=args.max_tasks)
    else:
        if not args.dataset_dir:
            raise SystemExit("materialize mode requires --dataset_dir")
        convert(args.dataset_dir, args.output)
