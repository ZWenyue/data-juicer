# -*- coding: utf-8 -*-
"""Acceptance: P2 depth-aware occlusion (synthetic + mapper render path).

Outputs a JSON Go/No-Go report under --output-dir.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from data_juicer._au.ops.mapper.video_hand_to_robot_render_mapper import VideoHandToRobotRenderMapper
from data_juicer._au.tools.build_r1_arm_mjcf import DEFAULT_MESH_DIR, DEFAULT_URDF, build_all
from data_juicer._au.utils.hand_to_robot.calibration import load_calibration
from data_juicer._au.utils.hand_to_robot.composite import DepthAligner, composite_with_depth, fit_depth_aligner


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        x = float(obj)
        return x if np.isfinite(x) else None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _check_visibility_unit() -> dict:
    h, w = 64, 80
    bg = np.full((h, w, 3), 10, dtype=np.uint8)
    robot = np.full((h, w, 3), 200, dtype=np.uint8)
    mask = np.zeros((h, w), dtype=bool)
    mask[20:40, 30:50] = True
    robot_depth = np.full((h, w), 1.0, dtype=np.float32)

    out_front, m_front = composite_with_depth(
        bg, robot, mask, robot_depth, np.full((h, w), 2.0, dtype=np.float32), edge_blur=0
    )
    out_back, m_back = composite_with_depth(
        bg, robot, mask, robot_depth, np.full((h, w), 0.4, dtype=np.float32), edge_blur=0
    )
    scene_nan = np.full((h, w), np.nan, dtype=np.float32)
    _, m_inv = composite_with_depth(
        bg, robot, mask, robot_depth, scene_nan, max_invalid_ratio=0.2, edge_blur=0
    )
    aligner = fit_depth_aligner([1, 2, 3, 4], [2, 4, 6, 8], min_pairs=4)
    return {
        "robot_in_front_visible": bool(
            m_front["ok"] and m_front["visible_pixel_count"] > 0 and np.all(out_front[mask] == 200)
        ),
        "robot_behind_occluded": bool(
            m_back["ok"] and m_back["visible_pixel_count"] == 0 and np.all(out_back[mask] == 10)
        ),
        "depth_invalid_gate": bool(not m_inv["ok"] and m_inv["quality_flag"] == "depth_invalid"),
        "aligner_scale_ok": bool(abs(aligner.scale - 2.0) < 1e-3),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--side", default="right", choices=["left", "right"])
    parser.add_argument("--gl-backend", default=os.environ.get("MUJOCO_GL", "egl"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MUJOCO_GL"] = args.gl_backend

    report = {"decision": "no_go", "checks": {}, "errors": [], "stage": "P2_depth_occlusion"}
    op = None
    try:
        report["checks"]["unit"] = _check_visibility_unit()
        unit_ok = all(report["checks"]["unit"].values())

        gen_dir = args.output_dir / "generated"
        smoke_dir = args.output_dir / "smoke"
        build_all(
            urdf_path=DEFAULT_URDF,
            mesh_dir=DEFAULT_MESH_DIR,
            out_dir=gen_dir,
            sides=(args.side,),
            smoke_dir=smoke_dir,
        )
        calib = REPO_ROOT / "b" / "d" / "hand2robot" / "calibration" / f"r1_{args.side}_v1.yaml"
        egodex = REPO_ROOT / "b" / "d" / "hand2robot" / "calibration" / f"r1_{args.side}_egodex_v1.yaml"
        if egodex.is_file():
            calib = egodex
        model_path = gen_dir / f"r1_lite_arm_{args.side}.xml"

        op = VideoHandToRobotRenderMapper(
            robot_model_paths={args.side: str(model_path)},
            calibration_path=str(calib),
            hand_type=args.side,
            ik_solver="jacobian",
            enable_depth_occlusion=True,
            depth_epsilon_m=0.02,
            max_depth_invalid_ratio=0.2,
            output_root=str(args.output_dir / "robot_render"),
            gl_backend=args.gl_backend,
        )
        op._init_renderer(240, 320)
        side = load_calibration(calib).get_side(args.side)
        frame = np.full((240, 320, 3), 40, dtype=np.uint8)
        hand_mask = np.zeros((240, 320), dtype=bool)
        scene_far = np.full((240, 320), 5.0, dtype=np.float32)
        scene_near = np.full((240, 320), 0.05, dtype=np.float32)

        out_far, meta_far = op._render_and_composite(
            frame,
            side.q_reference,
            0.02,
            side.T_camera_base_ref,
            hand_mask,
            scene_far,
            50.0,
            args.side,
            depth_aligner=DepthAligner(),
        )
        out_near, meta_near = op._render_and_composite(
            frame,
            side.q_reference,
            0.02,
            side.T_camera_base_ref,
            hand_mask,
            scene_near,
            50.0,
            args.side,
            depth_aligner=DepthAligner(),
        )

        report["checks"]["mapper_depth_path"] = {
            "far_used": bool(meta_far.get("depth_occlusion_used")),
            "near_used": bool(meta_near.get("depth_occlusion_used")),
            "far_flag": meta_far.get("quality_flag"),
            "near_flag": meta_near.get("quality_flag"),
            "far_visible": int(meta_far.get("visible_pixel_count") or 0),
            "near_visible": int(meta_near.get("visible_pixel_count") or 0),
            "far_invalid_ratio": float(meta_far.get("depth_invalid_ratio") or 0),
            "near_invalid_ratio": float(meta_near.get("depth_invalid_ratio") or 0),
            "output_shape_ok": list(out_far.shape) == [240, 320, 3] and list(out_near.shape) == [240, 320, 3],
        }
        m = report["checks"]["mapper_depth_path"]
        # Far scene should use depth path; if robot mask exists expect some visible pixels.
        mapper_ok = (
            m["far_used"]
            and m["near_used"]
            and m["output_shape_ok"]
            and m["far_invalid_ratio"] < 0.2
            and m["near_invalid_ratio"] < 0.2
            and m["far_visible"] >= m["near_visible"]
        )
        report["checks"]["mapper_depth_ok"] = bool(mapper_ok)
        report["checks"]["unit_all_pass"] = bool(unit_ok)

        # Persist debug frames.
        import cv2

        cv2.imwrite(str(args.output_dir / "composite_far.png"), out_far)
        cv2.imwrite(str(args.output_dir / "composite_near.png"), out_near)

        go = unit_ok and mapper_ok
        report["decision"] = "go" if go else "no_go"
    except Exception as e:
        report["errors"].append(str(e))
        report["decision"] = "no_go"
    finally:
        if op is not None:
            for r in list(op._renderers.values()):
                try:
                    r.close()
                except Exception:
                    pass
            op._renderers.clear()

    out_json = args.output_dir / "accept_depth_occlusion_report.json"
    out_json.write_text(json.dumps(_json_safe(report), indent=2), encoding="utf-8")
    print(json.dumps(_json_safe(report), indent=2))
    return 0 if report["decision"] == "go" else 1


if __name__ == "__main__":
    raise SystemExit(main())
