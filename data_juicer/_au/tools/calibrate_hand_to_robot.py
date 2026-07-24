# -*- coding: utf-8 -*-
"""CLI: calibrate hand→robot retarget YAML on ego-pipeline clips or Galaxea LeRobot.

Examples
--------
# Synthetic smoke (no real meta required)
python -m data_juicer._au.tools.calibrate_hand_to_robot \\
  --synthetic --side right \\
  --init-calib b/d/hand2robot/calibration/r1_right_v1.yaml \\
  --model b/d/urdf/generated/r1_lite_arm_right.xml \\
  --output-calib b/d/hand2robot/calibration/r1_right_v2.yaml \\
  --report-dir b/d/hand2robot/calib_out

# Galaxea R1 Lite LeRobot (GT joints → FK sites as calibration clip)
python -m data_juicer._au.tools.calibrate_hand_to_robot \\
  --lerobot-root /mnt/r/DATA/pre_train_v1/Galaxea_R1_Lite/Handle_Plates_20250619_001 \\
  --episode 2 --side right --max-frames 100 --stride 3 \\
  --init-calib b/d/hand2robot/calibration/r1_right_v1.yaml \\
  --model b/d/urdf/generated/r1_lite_arm_right.xml \\
  --output-calib b/d/hand2robot/calibration/r1_right_galaxea_v1.yaml \\
  --report-dir b/d/hand2robot/calib_out_galaxea

# Real ego pipeline sample (pkl/json/jsonl/parquet with hand_action_tags)
python -m data_juicer._au.tools.calibrate_hand_to_robot \\
  --data-path path/to/sample.pkl --sample-idx 0 --video-idx 0 \\
  --side right --max-frames 200 --stride 2 \\
  --init-calib b/d/hand2robot/calibration/r1_right_v1.yaml \\
  --model b/d/urdf/generated/r1_lite_arm_right.xml \\
  --output-calib b/d/hand2robot/calibration/r1_right_v2.yaml \\
  --report-dir b/d/hand2robot/calib_out
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np

from data_juicer._au.utils.hand_to_robot.calibrate import (
    CalibWeights,
    clip_from_galaxea_lerobot,
    clip_from_pipeline_sample,
    evaluate_galaxea_fk_ik,
    load_pipeline_sample,
    make_synthetic_clip,
    optimize_side_calibration,
)
from data_juicer._au.utils.hand_to_robot.calibration import (
    load_calibration,
    replace_side,
    save_calibration,
    side_to_dict,
)


def _parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--data-path", type=Path, help="Pipeline sample pkl/json/jsonl/parquet")
    src.add_argument("--synthetic", action="store_true", help="Use synthetic clip for smoke calibration")
    src.add_argument(
        "--lerobot-root",
        type=Path,
        help="Galaxea / LeRobot dataset root (meta/ + data/ + videos/)",
    )

    p.add_argument("--episode", type=int, default=0, help="Episode index for --lerobot-root")
    p.add_argument("--sample-idx", type=int, default=0)
    p.add_argument("--video-idx", type=int, default=0)
    p.add_argument("--side", choices=["left", "right"], default="right")
    p.add_argument("--max-frames", type=int, default=200)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--n-anchors", type=int, default=8)
    p.add_argument("--maxiter", type=int, default=80)

    p.add_argument("--init-calib", type=Path, required=True)
    p.add_argument("--output-calib", type=Path, required=True)
    p.add_argument("--report-dir", type=Path, required=True)
    p.add_argument("--model", type=Path, default=None, help="MJCF for IK-aware loss / Galaxea FK")

    p.add_argument("--optimize-base-orient", action="store_true")
    p.add_argument("--optimize-axis", action="store_true")
    p.add_argument("--gl-backend", default=os.environ.get("MUJOCO_GL", "egl"))
    p.add_argument("--skip-fk-eval", action="store_true", help="Skip Galaxea FK/IK diagnostic block")

    p.add_argument("--w-uv", type=float, default=2e-4)
    p.add_argument("--w-ik", type=float, default=1.0)
    p.add_argument("--w-pos", type=float, default=1.0)
    p.add_argument("--w-rot", type=float, default=0.25)
    return p.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    args = _parse_args(argv)
    os.environ.setdefault("MUJOCO_GL", args.gl_backend)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    init_cal = load_calibration(args.init_calib)
    if args.side not in init_cal.sides:
        raise KeyError(f"init calib {args.init_calib} has no side '{args.side}'")

    galaxea_fk_report = None
    galaxea_meta = None

    if args.synthetic:
        clip = make_synthetic_clip(side=args.side, n_frames=min(args.max_frames, 16))
    elif args.lerobot_root is not None:
        if args.model is None or not Path(args.model).is_file():
            raise FileNotFoundError("--model MJCF is required for --lerobot-root")
        if not args.skip_fk_eval:
            galaxea_fk_report = evaluate_galaxea_fk_ik(
                args.lerobot_root,
                args.model,
                side=args.side,
                episode=args.episode,
                max_frames=args.max_frames,
                stride=max(args.stride, 1),
                gl_backend=args.gl_backend,
            )
        clip, galaxea_meta = clip_from_galaxea_lerobot(
            args.lerobot_root,
            args.model,
            side=args.side,
            episode=args.episode,
            max_frames=args.max_frames,
            stride=max(args.stride, 1),
            gl_backend=args.gl_backend,
        )
        # FK-derived clip: identity scale + refs from first frame.
        # Zero camera_to_base translation so MJ arm-root stays at I (YAML t≠0 breaks GT warm-start).
        side = init_cal.get_side(args.side)
        if clip.wrist_ref_world is not None:
            side.wrist_ref_world = clip.wrist_ref_world.copy()
            side.ee_ref_world = (
                clip.ee_ref_world.copy() if clip.ee_ref_world is not None else clip.wrist_ref_world.copy()
            )
        side.workspace_scale_xyz = np.ones(3, dtype=np.float64)
        side.T_camera_base_ref = side.T_camera_base_ref.copy()
        side.T_camera_base_ref[:3, 3] = 0.0
        if galaxea_meta and galaxea_meta.get("q_reference_median"):
            side.q_reference = np.asarray(galaxea_meta["q_reference_median"], dtype=np.float64)
        init_cal = replace_side(init_cal, args.side, side)
        # UV is not meaningful for FK-proxy clips under the IK-consistent convention.
        args.w_uv = 0.0
    else:
        sample = load_pipeline_sample(args.data_path, sample_idx=args.sample_idx)
        clip = clip_from_pipeline_sample(
            sample,
            side=args.side,
            video_idx=args.video_idx,
            max_frames=args.max_frames,
            stride=max(args.stride, 1),
        )

    weights = CalibWeights(
        w_pos=args.w_pos,
        w_rot=args.w_rot,
        w_uv=args.w_uv,
        w_ik=args.w_ik if args.model else 0.0,
        w_base_reg=5.0 if args.lerobot_root is not None else 0.1,
    )

    result = optimize_side_calibration(
        clip,
        init_cal,
        weights=weights,
        n_anchors=args.n_anchors,
        optimize_base_orient=args.optimize_base_orient,
        optimize_axis=args.optimize_axis,
        model_path=str(args.model) if args.model else None,
        maxiter=args.maxiter,
    )

    result.calibration.version_name = args.output_calib.stem
    save_calibration(result.calibration, args.output_calib)

    report = {
        "decision": "go" if result.success else "no_go",
        "side": result.side,
        "source": clip.source,
        "num_frames": len(clip.frames),
        "init_calib": str(args.init_calib),
        "output_calib": str(args.output_calib),
        "model": str(args.model) if args.model else None,
        "optimizer_message": result.message,
        "metrics_before": result.metrics_before,
        "metrics_after": result.metrics_after,
        "side_params": side_to_dict(result.calibration.get_side(args.side)),
        "galaxea_meta": galaxea_meta,
        "galaxea_fk_ik": galaxea_fk_report,
    }

    before = result.metrics_before
    after = result.metrics_after
    improved_loss = after.get("loss", 1e9) <= before.get("loss", 1e9) + 1e-6
    uv_b, uv_a = before.get("median_reprojection_error_px"), after.get("median_reprojection_error_px")
    improved_uv = uv_a == uv_a and uv_b == uv_b and float(uv_a) <= float(uv_b) + 1e-3
    ik_after = after.get("ik_success_rate", float("nan"))
    ik_ok = ik_after == ik_after and float(ik_after) >= 0.9

    checks = {
        "optimizer_ok": bool(result.success),
        "loss_improved_or_equal": bool(improved_loss),
        "reprojection_improved_or_equal": bool(improved_uv),
        "output_written": args.output_calib.is_file(),
    }
    if galaxea_fk_report is not None:
        checks["galaxea_ik_from_fk_ge_0_95"] = float(galaxea_fk_report.get("ik_from_fk_site_success", 0)) >= 0.95
        checks["galaxea_fk_ori_lt_1deg"] = float(galaxea_fk_report.get("fk_vs_gt_ori_median_deg", 99)) < 1.0
        checks["galaxea_fk_aligned_pos_lt_5cm"] = (
            float(galaxea_fk_report.get("fk_aligned_pos_median_m", 99)) < 0.05
        )
    if args.model is not None:
        checks["calib_ik_success_ge_0_9"] = bool(ik_ok)

    go = checks["output_written"] and (improved_loss or improved_uv or ik_ok)
    if galaxea_fk_report is not None:
        go = (
            checks["output_written"]
            and checks.get("galaxea_ik_from_fk_ge_0_95", False)
            and checks.get("galaxea_fk_ori_lt_1deg", False)
            and checks.get("galaxea_fk_aligned_pos_lt_5cm", False)
            and checks.get("calib_ik_success_ge_0_9", False)
        )
    report["checks"] = checks
    report["decision"] = "go" if go else "no_go"

    report_path = args.report_dir / "calibrate_hand_to_robot_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(json.dumps(report, indent=2, default=float))
    return 0 if report["decision"] == "go" else 1


if __name__ == "__main__":
    raise SystemExit(main())
