# -*- coding: utf-8 -*-
"""Acceptance wrapper for calibrate_hand_to_robot.

Default: Galaxea R1 Lite LeRobot if present; otherwise synthetic smoke.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from data_juicer._au.tools.calibrate_hand_to_robot import main as calib_main

DEFAULT_GALAXEA = Path("/mnt/r/DATA/pre_train_v1/Galaxea_R1_Lite/Handle_Plates_20250619_001")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "b" / "d" / "hand2robot" / "calib_out")
    parser.add_argument("--side", default="right", choices=["left", "right"])
    parser.add_argument("--data-path", type=Path, default=None, help="Ego pipeline sample")
    parser.add_argument("--lerobot-root", type=Path, default=None, help="Galaxea LeRobot dataset root")
    parser.add_argument("--episode", type=int, default=2)
    parser.add_argument("--synthetic", action="store_true", help="Force synthetic smoke")
    parser.add_argument("--gl-backend", default=os.environ.get("MUJOCO_GL", "egl"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    init = REPO_ROOT / "b" / "d" / "hand2robot" / "calibration" / f"r1_{args.side}_v1.yaml"
    out_yaml = args.output_dir / f"r1_{args.side}_calibrated.yaml"
    model = REPO_ROOT / "b" / "d" / "urdf" / "generated" / f"r1_lite_arm_{args.side}.xml"

    argv = [
        "--side",
        args.side,
        "--init-calib",
        str(init),
        "--output-calib",
        str(out_yaml),
        "--report-dir",
        str(args.output_dir),
        "--gl-backend",
        args.gl_backend,
        "--maxiter",
        "40",
        "--n-anchors",
        "6",
    ]
    if model.is_file():
        argv.extend(["--model", str(model)])

    if args.synthetic:
        argv.append("--synthetic")
    elif args.data_path is not None:
        argv.extend(["--data-path", str(args.data_path), "--max-frames", "120", "--stride", "2"])
    else:
        lerobot = args.lerobot_root
        if lerobot is None and DEFAULT_GALAXEA.is_dir():
            lerobot = DEFAULT_GALAXEA
        if lerobot is not None and lerobot.is_dir():
            argv.extend(
                [
                    "--lerobot-root",
                    str(lerobot),
                    "--episode",
                    str(args.episode),
                    "--max-frames",
                    "80",
                    "--stride",
                    "3",
                ]
            )
        else:
            argv.append("--synthetic")

    rc = calib_main(argv)
    report = args.output_dir / "calibrate_hand_to_robot_report.json"
    if report.is_file():
        print(report.read_text())
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
