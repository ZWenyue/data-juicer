#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert LeRobot v2.1 dataset (per-frame parquet) to per-episode JSONL for DJ."""
import argparse
import glob
import json
import os

import numpy as np
import pyarrow.parquet as pq


def _to_nested_list(col):
    """Convert a pandas column of arrays/lists to a plain list-of-list."""
    return [np.asarray(row, dtype=float).tolist() for row in col]


def convert(dataset_dir: str, output: str):
    pattern = os.path.join(dataset_dir, "data", "chunk-*", "episode_*.parquet")
    parquet_files = sorted(glob.glob(pattern))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found: {pattern}")

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    with open(output, "w", encoding="utf-8") as f:
        for pf in parquet_files:
            table = pq.read_table(pf)
            df = table.to_pandas()
            ep_idx = int(df["episode_index"].iloc[0])
            ep_id = f"episode_{ep_idx:06d}"

            record = {
                "id": ep_id,
                "episode_index": ep_idx,
                "num_frames": len(df),
                "states": _to_nested_list(df["observation.state"]),
                "actions": _to_nested_list(df["action"]),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Converted {len(parquet_files)} episodes -> {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    convert(args.dataset_dir, args.output)
