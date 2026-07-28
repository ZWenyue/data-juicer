# -*- coding: utf-8 -*-
"""Build VLA A/B experiment manifest for hand→robot render arms.

Does not run training — prepares paired recipe snippets and dataset roots
for downstream StarVLA / LIBERO-style trainers.

Example::

    python -m data_juicer._au.tools.build_vla_ab_manifest \\
      --output b/d/hand2robot/runs/vla_ab/manifest.json \\
      --dataset-path demos/ego_hand_action_annotation/data/demo-dataset.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_ARMS = [
    {
        "id": "baseline_human",
        "description": "Original ego human-hand frames; world-frame actions.",
        "frame_field": "video_frames",
        "enable_depth_occlusion": False,
        "export_robot_type": "egodex_hand",
        "caption_frame_field": "video_frames",
    },
    {
        "id": "arm_no_depth",
        "description": "Robot render without depth-aware occlusion.",
        "frame_field": "robot_render_frames",
        "enable_depth_occlusion": False,
        "export_robot_type": "r1_lite_ego_retarget",
        "caption_frame_field": "robot_render_frames",
    },
    {
        "id": "arm_depth",
        "description": "Full robot render with P2 depth occlusion.",
        "frame_field": "robot_render_frames",
        "enable_depth_occlusion": True,
        "export_robot_type": "r1_lite_ego_retarget",
        "caption_frame_field": "robot_render_frames",
    },
]


def build_manifest(
    dataset_path: str,
    run_root: str,
    calibration_path: str,
    side: str = "right",
    arms: List[Dict[str, Any]] = None,
) -> dict:
    arms = arms or DEFAULT_ARMS
    run = Path(run_root)
    entries = []
    for arm in arms:
        arm_id = arm["id"]
        out = run / arm_id
        entries.append(
            {
                **arm,
                "dataset_path": dataset_path,
                "run_dir": str(out),
                "lerobot_dir": str(out / "lerobot_dataset"),
                "calibration_path": calibration_path,
                "side": side,
                "process_hints": {
                    "video_hand_to_robot_render_mapper": {
                        "enable_depth_occlusion": arm["enable_depth_occlusion"],
                        "output_frame_field": "robot_render_frames",
                        "hand_type": side,
                        "calibration_path": calibration_path,
                    },
                    "video_hand_action_caption_stub_mapper": {
                        "frame_field": arm["caption_frame_field"],
                        "hand_type": side,
                    },
                    "export_robot_render_lerobot_mapper": {
                        "frame_field": arm["frame_field"],
                        "robot_type": arm["export_robot_type"],
                        "encode_video_from_frames": arm["frame_field"] != "video_frames",
                    },
                },
                "metrics_to_report": [
                    "action_prediction_error",
                    "task_success_rate",
                    "cross_scene_generalization",
                ],
            }
        )
    return {
        "schema_version": 1,
        "stage": "P3_vla_ab",
        "dataset_path": dataset_path,
        "run_root": str(run),
        "side": side,
        "calibration_path": calibration_path,
        "arms": entries,
        "notes": (
            "Train each arm with identical action labels (world-frame hand_action_tags). "
            "Compare hold-out metrics per design §8.5; arm_depth must not regress vs baseline."
        ),
    }


def main(argv: List[str] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dataset-path", type=str, required=True)
    p.add_argument("--run-root", type=str, default="b/d/hand2robot/runs/vla_ab")
    p.add_argument("--calibration-path", type=str, default="b/d/hand2robot/calibration/r1_right_egodex_v1.yaml")
    p.add_argument("--side", default="right", choices=["left", "right"])
    args = p.parse_args(argv)

    manifest = build_manifest(
        dataset_path=args.dataset_path,
        run_root=args.run_root,
        calibration_path=args.calibration_path,
        side=args.side,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
