#!/usr/bin/env python3
"""Convert Galaxea LeRobot dataset to per-episode JSONL for Data-Juicer.

Each episode becomes one DJ sample with metadata, video paths, and a
pointer to the parquet file for on-demand trajectory loading by custom ops.

Usage:
    /mnt/r/VENV/dj/bin/python convert_to_jsonl_2.py [--data-root PATH]
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
        default=str(Path(__file__).parent / "episodes_2.jsonl"),
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    ep_path = data_root / "meta" / "episodes.jsonl"
    tasks_path = data_root / "meta" / "tasks.jsonl"

    task_map = {}
    if tasks_path.exists():
        with open(tasks_path) as f:
            for line in f:
                obj = json.loads(line)
                task_map[obj["task_index"]] = obj["task"]

    video_cameras = [
        "observation.images.head_rgb",
        "observation.images.head_right_rgb",
        "observation.images.left_wrist_rgb",
        "observation.images.right_wrist_rgb",
    ]

    records = []
    with open(ep_path) as f:
        for line in f:
            ep = json.loads(line)
            ep_idx = ep["episode_index"]
            length = ep["length"]
            ep_str = f"episode_{ep_idx:06d}"

            parquet_path = str(
                data_root / "data" / "chunk-000" / f"{ep_str}.parquet")

            video_paths = []
            for cam in video_cameras:
                vp = data_root / "videos" / "chunk-000" / cam / f"{ep_str}.mp4"
                if vp.exists():
                    video_paths.append(str(vp))

            skip_labels = {"qualified", "unqualified", "null",
                           "connect router cables"}
            subtasks = [t for t in ep.get("tasks", [])
                        if t not in skip_labels]
            all_tasks = ep.get("tasks", [])

            coarse = "connect router cables"
            for t in all_tasks:
                if t == "connect router cables":
                    coarse = t
                    break

            text_parts = []
            for t in subtasks:
                parts = t.split("@")
                text_parts.append(parts[0].strip())
            text = "; ".join(text_parts) if text_parts else coarse

            records.append({
                "text": text,
                "videos": video_paths,
                "episode_index": ep_idx,
                "episode_length": length,
                "parquet_path": parquet_path,
                "tasks": all_tasks,
                "subtasks": subtasks,
                "coarse_task": coarse,
            })

    with open(args.output, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} episode records to {args.output}")


if __name__ == "__main__":
    main()
