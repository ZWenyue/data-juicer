# -*- coding: utf-8 -*-
"""CLI: calibrate hand→robot retarget YAML on ego clips / EgoDex / Galaxea.

Examples
--------
# EgoDex real egocentric hands (P1 human-hand calibration)
python -m data_juicer._au.tools.calibrate_hand_to_robot \\
  --egodex-root /mnt/r/DATA/EgoDex/test_lerobot \\
  --episode 2 --side right --max-frames 120 --stride 4 \\
  --init-calib b/d/hand2robot/calibration/r1_right_v1.yaml \\
  --model b/d/urdf/generated/r1_lite_arm_right.xml \\
  --output-calib b/d/hand2robot/calibration/r1_right_egodex_v1.yaml \\
  --report-dir b/d/hand2robot/calib_out_egodex

# Synthetic smoke
python -m data_juicer._au.tools.calibrate_hand_to_robot \\
  --synthetic --side right \\
  --init-calib b/d/hand2robot/calibration/r1_right_v1.yaml \\
  --model b/d/urdf/generated/r1_lite_arm_right.xml \\
  --output-calib b/d/hand2robot/calib_out/r1_right_calibrated.yaml \\
  --report-dir b/d/hand2robot/calib_out

# Galaxea R1 Lite (FK/IK kinematics check; not human-hand retarget)
python -m data_juicer._au.tools.calibrate_hand_to_robot \\
  --lerobot-root /mnt/r/DATA/pre_train_v1/Galaxea_R1_Lite/Handle_Plates_20250619_001 \\
  --episode 2 --side right \\
  --init-calib b/d/hand2robot/calibration/r1_right_v1.yaml \\
  --model b/d/urdf/generated/r1_lite_arm_right.xml \\
  --output-calib b/d/hand2robot/calib_out_galaxea/r1_right_galaxea.yaml \\
  --report-dir b/d/hand2robot/calib_out_galaxea
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
    clip_from_egodex_lerobot,
    clip_from_galaxea_lerobot,
    clip_from_pipeline_sample,
    evaluate_galaxea_fk_ik,
    load_pipeline_sample,
    make_synthetic_clip,
    optimize_side_calibration,
    p1_metrics_from_eval,
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
    src.add_argument("--synthetic", action="store_true", help="Synthetic smoke clip")
    src.add_argument(
        "--lerobot-root",
        type=Path,
        help="Galaxea robot LeRobot root (GT joints; kinematics check)",
    )
    src.add_argument(
        "--egodex-root",
        type=Path,
        help="EgoDex LeRobot root (real egocentric hand poses; P1 human calib)",
    )

    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--sample-idx", type=int, default=0)
    p.add_argument("--video-idx", type=int, default=0)
    p.add_argument("--side", choices=["left", "right"], default="right")
    p.add_argument("--max-frames", type=int, default=200)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--n-anchors", type=int, default=8)
    p.add_argument("--maxiter", type=int, default=80)
    p.add_argument("--min-wrist-conf", type=float, default=0.5)
    p.add_argument(
        "--egodex-extrinsics-w2c",
        action="store_true",
        help="Treat EgoDex camera.extrinsics as w2c instead of c2w",
    )

    p.add_argument("--init-calib", type=Path, required=True)
    p.add_argument("--output-calib", type=Path, required=True)
    p.add_argument("--report-dir", type=Path, required=True)
    p.add_argument("--model", type=Path, default=None)

    p.add_argument("--optimize-base-orient", action="store_true")
    p.add_argument("--optimize-axis", action="store_true")
    p.add_argument("--gl-backend", default=os.environ.get("MUJOCO_GL", "egl"))
    p.add_argument("--skip-fk-eval", action="store_true")

    p.add_argument("--w-uv", type=float, default=2e-4)
    p.add_argument("--w-ik", type=float, default=1.0)
    p.add_argument("--w-pos", type=float, default=1.0)
    p.add_argument("--w-rot", type=float, default=0.0, help="Identity prior on retarget_R, not a data residual")
    p.add_argument("--w-ik-rot", type=float, default=1.0, help="IK orientation residual: can the arm reach it")
    p.add_argument("--w-grasp", type=float, default=1.0, help="Palm-frame residual: is the gripper held like the hand")
    p.add_argument("--no-seed-retarget-conventions", action="store_true")
    p.add_argument(
        "--anchor-base",
        action="store_true",
        help="Place the arm base so its workspace centre sits on the hand; keeps the robot on task",
    )
    p.add_argument(
        "--fit-workspace",
        action="store_true",
        help="Move targets into the arm's shell instead; solves IK but shifts the robot off task",
    )
    p.add_argument("--workspace-coverage", type=float, default=0.9)
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
    egodex_meta = None
    is_egodex = False
    is_galaxea = False

    if args.synthetic:
        clip = make_synthetic_clip(side=args.side, n_frames=min(args.max_frames, 16))
    elif args.egodex_root is not None:
        is_egodex = True
        if args.model is None or not Path(args.model).is_file():
            raise FileNotFoundError("--model MJCF is required for --egodex-root")
        clip, egodex_meta = clip_from_egodex_lerobot(
            args.egodex_root,
            side=args.side,
            episode=args.episode,
            max_frames=args.max_frames,
            stride=max(args.stride, 1),
            min_wrist_conf=args.min_wrist_conf,
            extrinsics_are_c2w=not args.egodex_extrinsics_w2c,
            camera_as_world=True,
        )
        side = init_cal.get_side(args.side)
        if clip.wrist_ref_world is not None:
            side.wrist_ref_world = clip.wrist_ref_world.copy()
            side.ee_ref_world = (
                clip.ee_ref_world.copy() if clip.ee_ref_world is not None else clip.wrist_ref_world.copy()
            )
        # Seed base translation: FK-align mean camera wrist to q_reference site (MJ=I local).
        side.T_camera_base_ref = side.T_camera_base_ref.copy()
        try:
            from data_juicer._au.utils.hand_to_robot.renderer import RobotArmRenderer

            T_adapt = init_cal.T_mjcam_from_cvcam
            rend = RobotArmRenderer(args.model, width=64, height=48, gl_backend=args.gl_backend)
            try:
                rend.set_base_pose(np.linalg.inv(T_adapt), T_adapt)
                rend.set_arm_qpos(side.q_reference, 0.02)
                rend.mujoco.mj_forward(rend.model, rend.data)
                fk_local, _ = rend.site_pose()
            finally:
                rend.close()
            mean_cam = np.asarray((egodex_meta or {}).get("wrist_cam_mean"), dtype=np.float64)
            if mean_cam.shape == (3,):
                mocap_t = mean_cam - fk_local
                # R_cb=adapter ⇒ t_mj = R_adapt @ base_t ⇒ base_t = R_adapt @ mocap_t
                side.T_camera_base_ref[:3, 3] = T_adapt[:3, :3] @ mocap_t
            else:
                sug = (egodex_meta or {}).get("suggested_camera_to_base_translation_m")
                if sug is not None:
                    side.T_camera_base_ref[:3, 3] = np.asarray(sug, dtype=np.float64)
        except Exception:
            sug = (egodex_meta or {}).get("suggested_camera_to_base_translation_m")
            if sug is not None:
                side.T_camera_base_ref[:3, 3] = np.asarray(sug, dtype=np.float64)
        init_cal = replace_side(init_cal, args.side, side)
        # Prefer IK + UV for real hands; allow base to move.
        args.w_ik = max(args.w_ik, 2.0)
        args.w_uv = max(args.w_uv, 5e-4)
    elif args.lerobot_root is not None:
        is_galaxea = True
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
        w_ik_rot=args.w_ik_rot if args.model else 0.0,
        w_grasp=args.w_grasp,
        # Anchored bases are already where we want them; hold them there.
        w_base_reg=2.0 if args.anchor_base else (0.02 if is_egodex else (5.0 if is_galaxea else 0.1)),
    )

    result = optimize_side_calibration(
        clip,
        init_cal,
        weights=weights,
        n_anchors=args.n_anchors,
        optimize_base_orient=True if is_egodex else args.optimize_base_orient,
        optimize_axis=args.optimize_axis,
        model_path=str(args.model) if args.model else None,
        maxiter=args.maxiter,
        seed_retarget_conventions=not args.no_seed_retarget_conventions,
        fit_workspace=args.fit_workspace,
        workspace_coverage=args.workspace_coverage,
        anchor_base=args.anchor_base,
    )

    result.calibration.version_name = args.output_calib.stem
    save_calibration(result.calibration, args.output_calib)

    p1 = p1_metrics_from_eval(result.metrics_after, len(clip.frames))

    report = {
        "decision": "go" if result.success else "no_go",
        "side": result.side,
        "source": clip.source,
        "num_frames": len(clip.frames),
        "init_calib": str(args.init_calib),
        "output_calib": str(args.output_calib),
        "model": str(args.model) if args.model else None,
        "optimizer_message": result.message,
        "retarget_seed_search": result.seed_search,
        "workspace_fit": result.workspace_fit,
        "metrics_before": result.metrics_before,
        "metrics_after": result.metrics_after,
        "p1_human_hand": p1 if is_egodex else None,
        "side_params": side_to_dict(result.calibration.get_side(args.side)),
        "galaxea_meta": galaxea_meta,
        "galaxea_fk_ik": galaxea_fk_report,
        "egodex_meta": egodex_meta,
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
    if args.model is not None:
        checks["calib_ik_success_ge_0_9"] = bool(ik_ok)
    if galaxea_fk_report is not None:
        checks["galaxea_ik_from_fk_ge_0_95"] = float(galaxea_fk_report.get("ik_from_fk_site_success", 0)) >= 0.95
        checks["galaxea_fk_ori_lt_1deg"] = float(galaxea_fk_report.get("fk_vs_gt_ori_median_deg", 99)) < 1.0
        checks["galaxea_fk_aligned_pos_lt_5cm"] = (
            float(galaxea_fk_report.get("fk_aligned_pos_median_m", 99)) < 0.05
        )
    if is_egodex:
        checks.update(p1["checks"])

    if is_egodex:
        go = checks["output_written"] and bool(p1["checks"]["p1_pass"])
    elif galaxea_fk_report is not None:
        go = (
            checks["output_written"]
            and checks.get("galaxea_ik_from_fk_ge_0_95", False)
            and checks.get("galaxea_fk_ori_lt_1deg", False)
            and checks.get("galaxea_fk_aligned_pos_lt_5cm", False)
            and checks.get("calib_ik_success_ge_0_9", False)
        )
    else:
        go = checks["output_written"] and (improved_loss or improved_uv or ik_ok)

    report["checks"] = checks
    report["decision"] = "go" if go else "no_go"

    report_path = args.report_dir / "calibrate_hand_to_robot_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(json.dumps(report, indent=2, default=float))
    return 0 if report["decision"] == "go" else 1


if __name__ == "__main__":
    raise SystemExit(main())
