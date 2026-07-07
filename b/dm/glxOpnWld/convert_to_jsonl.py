#!/usr/bin/env python3
"""Convert Galaxea episodes metadata to JSONL for Data-Juicer text analysis.

Creates a JSONL file where each line is a subtask annotation with episode
context, suitable for DJ's text-quality analyzers.

Usage:
    /mnt/r/VENV/dj/bin/python convert_to_jsonl.py [--data-root PATH]
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        default="/mnt/r/DATA/Galaxea-Open-World-Dataset/lerobot_door"
                "/Connect_Router_Cables_20250625_002",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).parent / "dataset_annotations.jsonl"),
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    ep_path = data_root / "meta" / "episodes.jsonl"

    skip_labels = {"qualified", "unqualified", "null", "connect router cables"}
    records = []

    with open(ep_path) as f:
        for line in f:
            ep = json.loads(line)
            ep_idx = ep["episode_index"]
            length = ep["length"]

            for task_str in ep.get("tasks", []):
                if task_str in skip_labels:
                    continue
                parts = task_str.split("@")
                zh = parts[0].strip()
                en = parts[1].strip() if len(parts) > 1 else ""

                records.append({
                    "text": zh,
                    "text_en": en,
                    "episode_index": ep_idx,
                    "episode_length": length,
                })

    with open(args.output, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} subtask annotations to {args.output}")


if __name__ == "__main__":
    main()
