# -*- coding: utf-8 -*-
"""Acceptance: build assets + run mapper smoke on synthetic / optional real frames.

Outputs a JSON Go/No-Go report under --output-dir.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from data_juicer._au.tools.build_r1_arm_mjcf import DEFAULT_MESH_DIR, DEFAULT_URDF, build_all
from data_juicer._au.ops.mapper.video_hand_to_robot_render_mapper import VideoHandToRobotRenderMapper
from data_juicer.utils.constant import CameraCalibrationKeys, Fields, MetaKeys


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_synthetic_sample(tmp: Path, n_frames: int = 5):
    frames = []
    for i in range(n_frames):
        p = tmp / f"frame_{i:04d}.png"
        img = np.full((240, 320, 3), 30 + i * 5, dtype=np.uint8)
        # Fake hand-ish blob.
        cv2.circle(img, (160 + i * 3, 140), 25, (180, 140, 120), -1)
        cv2.imwrite(str(p), img)
        frames.append(str(p))

    cam_c2w = np.tile(np.eye(4, dtype=np.float64), (n_frames, 1, 1))
    c2w_path = tmp / "cam_c2w.npy"
    np.save(c2w_path, cam_c2w)

    states = []
    joints = []
    for i in range(n_frames):
        x = 0.22 + 0.01 * i
        states.append([x, 0.05, 0.42, 0.0, 0.2, 0.0, 0.0, 1.0 - 0.1 * i])
        joints.append(
            [
                [x - 0.03, 0.0, 0.42],
                [x, 0.02, 0.42],
                [x, -0.02, 0.42],
                [x + 0.03, 0.0, 0.42],
            ]
        )

    return {
        "id": "accept_synth",
        Fields.meta: {
            MetaKeys.video_frames: [frames],
            MetaKeys.hand_reconstruction_hawor_tags: [
                {"right": {"frame_ids": list(range(n_frames)), "joints_cam": joints}}
            ],
            MetaKeys.hand_action_tags: [
                {
                    "right": {
                        "valid_frame_ids": list(range(n_frames)),
                        "states": states,
                        "actions": [[0, 0, 0, 0, 0, 0, 1]] * n_frames,
                    }
                }
            ],
            MetaKeys.camera_calibration_moge_tags: [{}],
            MetaKeys.video_camera_pose_tags: [{CameraCalibrationKeys.cam_c2w: str(c2w_path)}],
        },
    }, frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--side", default="right", choices=["left", "right"])
    parser.add_argument("--n-frames", type=int, default=5)
    parser.add_argument("--gl-backend", default=os.environ.get("MUJOCO_GL", "egl"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MUJOCO_GL"] = args.gl_backend

    report = {"decision": "no_go", "checks": {}, "errors": []}
    try:
        gen_dir = args.output_dir / "generated"
        smoke_dir = args.output_dir / "smoke"
        manifest = build_all(
            urdf_path=DEFAULT_URDF,
            mesh_dir=DEFAULT_MESH_DIR,
            out_dir=gen_dir,
            sides=(args.side,),
            smoke_dir=smoke_dir,
        )
        smoke = manifest["sides"][args.side]["smoke"]
        report["checks"]["asset_load"] = True
        report["checks"]["static_mask_pixels"] = smoke["mask_pixels"]
        report["checks"]["static_mask_ok"] = smoke["mask_pixels"] > 0

        calib = REPO_ROOT / "b" / "d" / "hand2robot" / "calibration" / f"r1_{args.side}_v1.yaml"
        model_path = gen_dir / f"r1_lite_arm_{args.side}.xml"

        synth_dir = args.output_dir / "synth"
        synth_dir.mkdir(exist_ok=True)
        sample, src_frames = _make_synthetic_sample(synth_dir, n_frames=args.n_frames)
        hashes_before = {p: _sha256(Path(p)) for p in src_frames}

        op = VideoHandToRobotRenderMapper(
            robot_model_paths={args.side: str(model_path)},
            calibration_path=str(calib),
            hand_type=args.side,
            ik_solver="jacobian",
            output_root=str(args.output_dir / "robot_render"),
            gl_backend=args.gl_backend,
        )
        out = op.process_single(sample)
        hashes_after = {p: _sha256(Path(p)) for p in src_frames}
        report["checks"]["original_frames_unchanged"] = hashes_before == hashes_after

        quality = out[Fields.meta]["hand_to_robot_render_quality"]
        summary = quality.get("summary", {})
        report["checks"]["ik_success_rate"] = summary.get("ik_success_rate")
        report["checks"]["num_frames"] = summary.get("num_frames")
        report["quality_summary"] = summary

        out_frames = out[Fields.meta]["robot_render_frames"][0]
        report["checks"]["outputs_written"] = all(Path(p).is_file() for p in out_frames)
        report["checks"]["outputs_differ_from_inputs"] = all(a != b for a, b in zip(out_frames, src_frames))

        go = (
            report["checks"]["asset_load"]
            and report["checks"]["static_mask_ok"]
            and report["checks"]["original_frames_unchanged"]
            and report["checks"]["outputs_written"]
            and report["checks"]["outputs_differ_from_inputs"]
            and (summary.get("num_frames") or 0) > 0
        )
        report["decision"] = "go" if go else "no_go"
    except Exception as e:
        report["errors"].append(str(e))
        report["decision"] = "no_go"

    out_json = args.output_dir / "accept_hand_to_robot_render_report.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["decision"] == "go" else 1


if __name__ == "__main__":
    raise SystemExit(main())
