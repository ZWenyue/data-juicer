# -*- coding: utf-8 -*-
"""Acceptance: P4 dual-arm both → state16/action14 LeRobot export."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from data_juicer._au.ops.mapper.export_robot_render_lerobot_mapper import (  # noqa: E402
    ACTION_DIM_DUAL,
    STATE_DIM_DUAL,
    ExportRobotRenderLeRobotMapper,
    merge_dual_arm_trajectories,
)
from data_juicer._au.ops.mapper.video_hand_action_caption_stub_mapper import (  # noqa: E402
    VideoHandActionCaptionStubMapper,
)
from data_juicer._au.ops.mapper.video_hand_to_robot_render_mapper import (  # noqa: E402
    VideoHandToRobotRenderMapper,
)
from data_juicer._au.tools.build_r1_arm_mjcf import DEFAULT_MESH_DIR, DEFAULT_URDF, build_all  # noqa: E402
from data_juicer._au.utils.hand_to_robot.pipeline_validate import (  # noqa: E402
    compare_action_invariance,
    read_lerobot_info_json,
    snapshot_hand_actions,
    summarize_render_quality,
)
from data_juicer.utils.constant import CameraCalibrationKeys, Fields, MetaKeys  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _side_states(n_frames: int, side: str):
    states, joints, actions = [], [], []
    y_sign = 1.0 if side == "right" else -1.0
    for i in range(n_frames):
        x = 0.22 + 0.01 * i
        y = y_sign * (0.05 + 0.002 * i)
        states.append([x, y, 0.42, 0.0, 0.2, 0.0, 0.0, 1.0 - 0.1 * i])
        joints.append(
            [
                [x - 0.03, y, 0.42],
                [x, y + 0.02, 0.42],
                [x, y - 0.02, 0.42],
                [x + 0.03, y, 0.42],
            ]
        )
        actions.append([0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    return states, joints, actions


def _make_synthetic_both_sample(tmp: Path, n_frames: int = 8):
    frames, depths = [], []
    for i in range(n_frames):
        p = tmp / f"frame_{i:04d}.png"
        img = np.full((240, 320, 3), 30 + i * 5, dtype=np.uint8)
        cv2.circle(img, (120 + i * 2, 140), 22, (180, 140, 120), -1)
        cv2.circle(img, (200 + i * 2, 140), 22, (160, 150, 130), -1)
        cv2.imwrite(str(p), img)
        frames.append(str(p))
        dp = tmp / f"depth_{i:04d}.npy"
        np.save(dp, np.full((240, 320), 2.5, dtype=np.float32))
        depths.append(str(dp))

    r_states, r_joints, r_actions = _side_states(n_frames, "right")
    l_states, l_joints, l_actions = _side_states(n_frames, "left")
    c2w_path = tmp / "cam_c2w.npy"
    np.save(c2w_path, np.tile(np.eye(4, dtype=np.float64), (n_frames, 1, 1)))
    K = np.array([[500.0, 0, 160.0], [0, 500.0, 120.0], [0, 0, 1.0]], dtype=np.float64)
    fids = list(range(n_frames))
    return {
        "id": "accept_p4",
        Fields.meta: {
            MetaKeys.video_frames: [frames],
            MetaKeys.hand_reconstruction_hawor_tags: [
                {
                    "right": {"frame_ids": fids, "joints_cam": r_joints},
                    "left": {"frame_ids": fids, "joints_cam": l_joints},
                }
            ],
            MetaKeys.hand_action_tags: [
                {
                    "right": {"valid_frame_ids": fids, "states": r_states, "actions": r_actions},
                    "left": {"valid_frame_ids": fids, "states": l_states, "actions": l_actions},
                }
            ],
            MetaKeys.camera_calibration_moge_tags: [
                {CameraCalibrationKeys.depth: depths, CameraCalibrationKeys.intrinsics: [K] * n_frames}
            ],
            MetaKeys.video_camera_pose_tags: [{CameraCalibrationKeys.cam_c2w: str(c2w_path)}],
        },
    }, frames


def _parquet_state_action_dims(lerobot_dir: Path):
    paths = sorted((lerobot_dir / "data").rglob("episode_*.parquet"))
    if not paths:
        return None, None
    table = pq.read_table(paths[0])
    df = table.to_pandas()
    state0 = df["observation.state"].iloc[0]
    action0 = df["action"].iloc[0]
    return len(list(state0)), len(list(action0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-frames", type=int, default=8)
    parser.add_argument("--gl-backend", default=os.environ.get("MUJOCO_GL", "egl"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MUJOCO_GL"] = args.gl_backend

    report = {"decision": "no_go", "checks": {}, "errors": [], "stage": "P4_dual_arm"}
    render_op = None
    try:
        # Unit-level concat check (no MuJoCo)
        demo_r = {"valid_frame_ids": [0, 1], "states": [[1] * 8, [2] * 8], "actions": [[3] * 7, [4] * 7]}
        demo_l = {"valid_frame_ids": [0, 1], "states": [[5] * 8, [6] * 8], "actions": [[7] * 7, [8] * 7]}
        s, a, fids = merge_dual_arm_trajectories(demo_r, demo_l)
        report["checks"]["merge_helper_16_14"] = (
            len(s) == 2 and len(s[0]) == STATE_DIM_DUAL and len(a[0]) == ACTION_DIM_DUAL and fids == [0, 1]
        )

        gen_dir = args.output_dir / "generated"
        build_all(urdf_path=DEFAULT_URDF, mesh_dir=DEFAULT_MESH_DIR, out_dir=gen_dir, sides=("left", "right"))
        calib = REPO_ROOT / "b" / "d" / "hand2robot" / "calibration" / "r1_both_v1.yaml"
        model_paths = {
            "right": str(gen_dir / "r1_lite_arm_right.xml"),
            "left": str(gen_dir / "r1_lite_arm_left.xml"),
        }
        synth_dir = args.output_dir / "synth"
        synth_dir.mkdir(exist_ok=True)
        sample, src_frames = _make_synthetic_both_sample(synth_dir, n_frames=args.n_frames)
        hashes_before = {p: _sha256(Path(p)) for p in src_frames}
        actions_before = snapshot_hand_actions(sample)

        render_op = VideoHandToRobotRenderMapper(
            robot_model_paths=model_paths,
            calibration_path=str(calib),
            hand_type="both",
            ik_solver="jacobian",
            enable_depth_occlusion=True,
            output_root=str(args.output_dir / "robot_frames"),
            gl_backend=args.gl_backend,
        )
        sample = render_op.process_single(sample)
        quality = summarize_render_quality(sample)
        report["render_quality"] = quality
        report["checks"]["per_side_present"] = isinstance(
            (sample[Fields.meta].get("hand_to_robot_render_quality", {}) or {}).get("summary", {}).get("per_side"),
            dict,
        )

        sample = VideoHandActionCaptionStubMapper(hand_type="both", frame_field="robot_render_frames").process_single(
            sample
        )
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
        report["checks"]["export_marked_dual"] = any(ep.get("dual_arm") for ep in export_meta)
        report["checks"]["export_dims_meta"] = any(
            ep.get("state_dim") == STATE_DIM_DUAL and ep.get("action_dim") == ACTION_DIM_DUAL for ep in export_meta
        )
        report["checks"]["video_encoded_from_frames"] = any(ep.get("encoded_from_frames") for ep in export_meta)

        info = read_lerobot_info_json(lerobot_dir)
        report["lerobot_info"] = info
        feat = (info or {}).get("features") or {}
        report["checks"]["info_state_16"] = feat.get("observation.state", {}).get("shape") == [STATE_DIM_DUAL]
        report["checks"]["info_action_14"] = feat.get("action", {}).get("shape") == [ACTION_DIM_DUAL]
        report["checks"]["info_dual_flag"] = bool((info or {}).get("dual_arm"))

        pq_s, pq_a = _parquet_state_action_dims(lerobot_dir)
        report["parquet_dims"] = {"state": pq_s, "action": pq_a}
        report["checks"]["parquet_state_16"] = pq_s == STATE_DIM_DUAL
        report["checks"]["parquet_action_14"] = pq_a == ACTION_DIM_DUAL

        out_frames = sample[Fields.meta].get("robot_render_frames", [[]])[0]
        report["checks"]["robot_outputs_written"] = all(Path(p).is_file() for p in out_frames if p)
        report["checks"]["robot_outputs_differ"] = all(a != b for a, b in zip(out_frames, src_frames) if a and b)

        required = [
            "merge_helper_16_14",
            "per_side_present",
            "action_invariant",
            "original_frames_unchanged",
            "caption_text_nonempty",
            "lerobot_staged",
            "export_marked_dual",
            "export_dims_meta",
            "video_encoded_from_frames",
            "info_state_16",
            "info_action_14",
            "info_dual_flag",
            "parquet_state_16",
            "parquet_action_14",
            "robot_outputs_written",
        ]
        report["decision"] = "go" if all(report["checks"].get(k) for k in required) else "no_go"
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

    out_json = args.output_dir / "accept_p4_pipeline_report.json"
    out_json.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["decision"] == "go" else 1


if __name__ == "__main__":
    raise SystemExit(main())
