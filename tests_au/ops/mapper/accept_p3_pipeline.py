# -*- coding: utf-8
"""Acceptance: P3 pipeline Render → Caption (stub) → Export with action invariance."""

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

from data_juicer._au.ops.mapper.export_robot_render_lerobot_mapper import ExportRobotRenderLeRobotMapper
from data_juicer._au.ops.mapper.video_hand_action_caption_stub_mapper import VideoHandActionCaptionStubMapper
from data_juicer._au.ops.mapper.video_hand_to_robot_render_mapper import VideoHandToRobotRenderMapper
from data_juicer._au.tools.build_r1_arm_mjcf import DEFAULT_MESH_DIR, DEFAULT_URDF, build_all
from data_juicer._au.utils.hand_to_robot.pipeline_validate import (
    compare_action_invariance,
    read_lerobot_info_json,
    snapshot_hand_actions,
    summarize_render_quality,
)
from data_juicer.utils.constant import CameraCalibrationKeys, Fields, MetaKeys


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_synthetic_sample(tmp: Path, n_frames: int = 8, side: str = "right"):
    frames, depths, states, joints = [], [], [], []
    for i in range(n_frames):
        p = tmp / f"frame_{i:04d}.png"
        img = np.full((240, 320, 3), 30 + i * 5, dtype=np.uint8)
        cv2.circle(img, (160 + i * 3, 140), 25, (180, 140, 120), -1)
        cv2.imwrite(str(p), img)
        frames.append(str(p))
        dp = tmp / f"depth_{i:04d}.npy"
        np.save(dp, np.full((240, 320), 2.5, dtype=np.float32))
        depths.append(str(dp))
        x = 0.22 + 0.01 * i
        states.append([x, 0.05, 0.42, 0.0, 0.2, 0.0, 0.0, 1.0 - 0.1 * i])
        joints.append([[x - 0.03, 0.0, 0.42], [x, 0.02, 0.42], [x, -0.02, 0.42], [x + 0.03, 0.0, 0.42]])
    c2w_path = tmp / "cam_c2w.npy"
    np.save(c2w_path, np.tile(np.eye(4, dtype=np.float64), (n_frames, 1, 1)))
    K = np.array([[500.0, 0, 160.0], [0, 500.0, 120.0], [0, 0, 1.0]], dtype=np.float64)
    return {
        "id": "accept_p3",
        Fields.meta: {
            MetaKeys.video_frames: [frames],
            MetaKeys.hand_reconstruction_hawor_tags: [{side: {"frame_ids": list(range(n_frames)), "joints_cam": joints}}],
            MetaKeys.hand_action_tags: [{side: {"valid_frame_ids": list(range(n_frames)), "states": states, "actions": [[0, 0, 0, 0, 0, 0, 1]] * n_frames}}],
            MetaKeys.camera_calibration_moge_tags: [{CameraCalibrationKeys.depth: depths, CameraCalibrationKeys.intrinsics: [K] * n_frames}],
            MetaKeys.video_camera_pose_tags: [{CameraCalibrationKeys.cam_c2w: str(c2w_path)}],
        },
    }, frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--side", default="right", choices=["left", "right"])
    parser.add_argument("--n-frames", type=int, default=8)
    parser.add_argument("--gl-backend", default=os.environ.get("MUJOCO_GL", "egl"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MUJOCO_GL"] = args.gl_backend

    report = {"decision": "no_go", "checks": {}, "errors": [], "stage": "P3_pipeline"}
    render_op = None
    try:
        gen_dir = args.output_dir / "generated"
        build_all(urdf_path=DEFAULT_URDF, mesh_dir=DEFAULT_MESH_DIR, out_dir=gen_dir, sides=(args.side,))
        calib = REPO_ROOT / "b" / "d" / "hand2robot" / "calibration" / f"r1_{args.side}_v1.yaml"
        model_path = gen_dir / f"r1_lite_arm_{args.side}.xml"
        synth_dir = args.output_dir / "synth"
        synth_dir.mkdir(exist_ok=True)
        sample, src_frames = _make_synthetic_sample(synth_dir, n_frames=args.n_frames, side=args.side)
        hashes_before = {p: _sha256(Path(p)) for p in src_frames}
        actions_before = snapshot_hand_actions(sample)

        render_op = VideoHandToRobotRenderMapper(
            robot_model_paths={args.side: str(model_path)},
            calibration_path=str(calib),
            hand_type=args.side,
            ik_solver="jacobian",
            enable_depth_occlusion=True,
            output_root=str(args.output_dir / "robot_frames"),
            gl_backend=args.gl_backend,
        )
        sample = render_op.process_single(sample)
        report["render_quality"] = summarize_render_quality(sample)

        sample = VideoHandActionCaptionStubMapper(hand_type=args.side, frame_field="robot_render_frames").process_single(sample)
        report["checks"]["caption_text_nonempty"] = bool(sample.get("text"))

        lerobot_dir = args.output_dir / "lerobot_dataset"
        export_op = ExportRobotRenderLeRobotMapper(
            output_dir=str(lerobot_dir),
            hand_action_field="hand_action_tags",
            frame_field="robot_render_frames",
            fps=10,
            robot_type="r1_lite_ego_retarget",
            encode_video_from_frames=True,
        )
        sample = export_op.process_single(sample)
        ExportRobotRenderLeRobotMapper.finalize_dataset(
            str(lerobot_dir), fps=10, robot_type="r1_lite_ego_retarget"
        )

        inv_ok, inv_report = compare_action_invariance(actions_before, snapshot_hand_actions(sample))
        report["action_invariance"] = inv_report
        report["checks"]["action_invariant"] = inv_ok
        report["checks"]["original_frames_unchanged"] = hashes_before == {p: _sha256(Path(p)) for p in src_frames}
        export_meta = sample[Fields.meta].get("lerobot_export", [])
        report["checks"]["lerobot_staged"] = len(export_meta) > 0
        report["checks"]["video_encoded_from_frames"] = any(ep.get("encoded_from_frames") for ep in export_meta)
        info = read_lerobot_info_json(lerobot_dir)
        report["lerobot_info"] = info
        report["checks"]["robot_type_ok"] = info and info.get("robot_type") == "r1_lite_ego_retarget"
        out_frames = sample[Fields.meta].get("robot_render_frames", [[]])[0]
        report["checks"]["robot_outputs_written"] = all(Path(p).is_file() for p in out_frames if p)
        report["checks"]["robot_outputs_differ"] = all(a != b for a, b in zip(out_frames, src_frames) if a and b)
        report["checks"]["pipeline_wiring_ok"] = True
        report["decision"] = "go" if all(report["checks"].get(k) for k in (
            "action_invariant", "original_frames_unchanged", "caption_text_nonempty",
            "lerobot_staged", "video_encoded_from_frames", "robot_type_ok",
            "robot_outputs_written", "pipeline_wiring_ok"
        )) else "no_go"
    except Exception as e:
        report["errors"].append(str(e))
        report["decision"] = "no_go"
    finally:
        if render_op is not None:
            for r in list(render_op._renderers.values()):
                try:
                    r.close()
                except Exception:
                    pass
            render_op._renderers.clear()

    out_json = args.output_dir / "accept_p3_pipeline_report.json"
    out_json.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["decision"] == "go" else 1


if __name__ == "__main__":
    raise SystemExit(main())
