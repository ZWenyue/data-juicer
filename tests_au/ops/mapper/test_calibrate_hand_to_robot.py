# -*- coding: utf-8 -*-
"""Tests for hand→robot calibration utilities."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from data_juicer._au.utils.hand_to_robot.calibrate import (
    CalibWeights,
    clip_from_galaxea_lerobot,
    evaluate_galaxea_fk_ik,
    make_synthetic_clip,
    optimize_side_calibration,
    select_anchor_indices,
)

GALAXEA_ROOT = Path("/mnt/r/DATA/pre_train_v1/Galaxea_R1_Lite/Handle_Plates_20250619_001")
GALAXEA_AVAILABLE = GALAXEA_ROOT.is_dir() and (GALAXEA_ROOT / "data").is_dir()
from data_juicer._au.utils.hand_to_robot.calibration import load_calibration, save_calibration
from data_juicer._au.utils.hand_to_robot.retarget import project_point_cam, retarget_wrist_to_ee

REPO_ROOT = Path(__file__).resolve().parents[3]
CALIB_RIGHT = REPO_ROOT / "b" / "d" / "hand2robot" / "calibration" / "r1_right_v1.yaml"
MODEL_RIGHT = REPO_ROOT / "b" / "d" / "urdf" / "generated" / "r1_lite_arm_right.xml"

MUJOCO_AVAILABLE = False
try:
    import mujoco  # noqa: F401

    MUJOCO_AVAILABLE = True
except Exception:
    pass


class TestCalibrateHelpers(unittest.TestCase):
    def test_select_anchors(self):
        clip = make_synthetic_clip(n_frames=20)
        ids = select_anchor_indices(clip.frames, 5)
        self.assertEqual(len(ids), 5)
        self.assertEqual(ids[0], 0)
        self.assertEqual(ids[-1], 19)

    def test_retarget_and_project(self):
        cal = load_calibration(CALIB_RIGHT)
        side = cal.get_side("right")
        state = [0.25, 0.05, 0.4, 0.0, 0.1, 0.0, 0.0, 1.0]
        T = retarget_wrist_to_ee(state, side, wrist_ref_world=np.array(state[:3]), ee_ref_world=np.array(state[:3]))
        self.assertEqual(T.shape, (4, 4))
        u, v, ok = project_point_cam(np.array([0.0, 0.0, 0.5]), 500, 500, 160, 120)
        self.assertTrue(ok)
        self.assertAlmostEqual(u, 160.0, places=5)

    def test_optimize_synthetic_without_ik(self):
        cal = load_calibration(CALIB_RIGHT)
        clip = make_synthetic_clip(side="right", n_frames=12)
        # Perturb init so optimizer has room.
        side = cal.get_side("right")
        side.workspace_scale_xyz = np.array([1.2, 1.2, 1.2])
        side.T_camera_base_ref[:3, 3] = np.array([0.05, 0.35, -0.15])
        from data_juicer._au.utils.hand_to_robot.calibration import replace_side

        cal = replace_side(cal, "right", side)
        result = optimize_side_calibration(
            clip,
            cal,
            weights=CalibWeights(w_ik=0.0, w_uv=1e-3),
            n_anchors=6,
            model_path=None,
            maxiter=40,
        )
        self.assertTrue(result.metrics_after["loss"] <= result.metrics_before["loss"] + 1e-5)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "r1_right_test.yaml"
            save_calibration(result.calibration, out)
            reloaded = load_calibration(out)
            self.assertIn("right", reloaded.sides)
            self.assertEqual(reloaded.sides["right"].workspace_scale_xyz.shape, (3,))

    @unittest.skipUnless(MUJOCO_AVAILABLE and MODEL_RIGHT.is_file(), "mujoco/model missing")
    def test_optimize_synthetic_with_ik(self):
        import os

        os.environ.setdefault("MUJOCO_GL", "egl")
        cal = load_calibration(CALIB_RIGHT)
        clip = make_synthetic_clip(side="right", n_frames=10)
        result = optimize_side_calibration(
            clip,
            cal,
            weights=CalibWeights(w_ik=0.5, w_uv=5e-4),
            n_anchors=5,
            model_path=str(MODEL_RIGHT),
            maxiter=25,
        )
        self.assertEqual(result.side, "right")
        self.assertIn("ik_success_rate", result.metrics_after)

    @unittest.skipUnless(
        MUJOCO_AVAILABLE and MODEL_RIGHT.is_file() and GALAXEA_AVAILABLE,
        "mujoco/model/Galaxea missing",
    )
    def test_galaxea_fk_ik_and_clip(self):
        import os

        os.environ.setdefault("MUJOCO_GL", "egl")
        report = evaluate_galaxea_fk_ik(
            GALAXEA_ROOT,
            MODEL_RIGHT,
            side="right",
            episode=2,
            max_frames=40,
            stride=5,
        )
        self.assertGreaterEqual(report["ik_from_fk_site_success"], 0.95)
        self.assertLess(report["fk_vs_gt_ori_median_deg"], 1.0)
        self.assertLess(report["fk_aligned_pos_median_m"], 0.05)

        clip, meta = clip_from_galaxea_lerobot(
            GALAXEA_ROOT,
            MODEL_RIGHT,
            side="right",
            episode=2,
            max_frames=24,
            stride=5,
        )
        self.assertGreaterEqual(len(clip.frames), 8)
        self.assertEqual(clip.side, "right")
        self.assertIn("parquet", meta)


if __name__ == "__main__":
    unittest.main()
