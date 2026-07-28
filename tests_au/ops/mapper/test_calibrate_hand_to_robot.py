# -*- coding: utf-8 -*-
"""Tests for hand→robot calibration utilities."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from data_juicer._au.utils.hand_to_robot.calibrate import (
    CalibClip,
    CalibFrame,
    CalibWeights,
    axis_convention_rotations,
    clip_from_egodex_lerobot,
    clip_from_galaxea_lerobot,
    evaluate_galaxea_fk_ik,
    fit_retarget_R_from_grasp,
    is_egodex_lerobot,
    make_synthetic_clip,
    optimize_side_calibration,
    p1_metrics_from_eval,
    select_anchor_indices,
)

GALAXEA_ROOT = Path("/mnt/r/DATA/pre_train_v1/Galaxea_R1_Lite/Handle_Plates_20250619_001")
GALAXEA_AVAILABLE = GALAXEA_ROOT.is_dir() and (GALAXEA_ROOT / "data").is_dir()
EGODEX_ROOT = Path("/mnt/r/DATA/EgoDex/test_lerobot")
EGODEX_AVAILABLE = EGODEX_ROOT.is_dir() and (EGODEX_ROOT / "data").is_dir()

from data_juicer._au.utils.hand_to_robot.calibration import load_calibration, save_calibration
from data_juicer._au.utils.hand_to_robot.retarget import (
    grasp_orientation_error,
    hand_grasp_frame,
    project_point_cam,
    retarget_wrist_to_ee,
)


def mano21_from_frame(wrist_pos, R_frame):
    """MANO-21 joints whose palm frame is exactly ``R_frame``.

    Knuckles are placed symmetrically about the approach axis so the index→pinky span is
    already perpendicular to it and survives orthogonalization untouched. Fingertips get
    junk on purpose: the palm frame must not depend on them.
    """
    approach, closing = R_frame[:, 0], R_frame[:, 1]
    joints = np.tile(np.asarray(wrist_pos, dtype=np.float64), (21, 1))
    joints[9] = wrist_pos + 0.085 * approach
    joints[5] = wrist_pos + 0.080 * approach + 0.040 * closing
    joints[17] = wrist_pos + 0.080 * approach - 0.040 * closing
    for tip in (4, 8, 12, 16, 20):
        joints[tip] = wrist_pos + 0.13 * approach + 0.05 * R_frame[:, 2]
    return joints

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

    def test_axis_convention_rotations(self):
        mats = axis_convention_rotations()
        self.assertEqual(len(mats), 24)
        self.assertTrue(np.allclose(mats[0], np.eye(3)))
        for R in mats:
            self.assertAlmostEqual(float(np.linalg.det(R)), 1.0, places=9)
            self.assertTrue(np.allclose(R @ R.T, np.eye(3), atol=1e-12))
        flat = {tuple(R.ravel()) for R in mats}
        self.assertEqual(len(flat), 24)

    def test_grasp_frame_axes_and_guard(self):
        R = hand_grasp_frame(mano21_from_frame(np.array([0.1, 0.2, 0.5]), np.eye(3)))
        self.assertTrue(np.allclose(R, np.eye(3), atol=1e-12))
        self.assertAlmostEqual(float(np.linalg.det(R)), 1.0, places=12)
        # Only a full MANO-21 set defines the palm; EgoDex's 3-point sets must opt out.
        self.assertIsNone(hand_grasp_frame(np.zeros((3, 3))))
        self.assertIsNone(hand_grasp_frame(np.zeros((21, 3))))

    def test_grasp_frame_ignores_finger_articulation(self):
        """Why the palm frame uses knuckles: fingertips swing, the gripper cannot."""
        joints = mano21_from_frame(np.array([0.0, 0.0, 0.4]), np.eye(3))
        before = hand_grasp_frame(joints)
        curled = joints.copy()
        for tip in (4, 8, 12, 16, 20):
            curled[tip] += np.array([-0.06, 0.03, 0.02])
        self.assertTrue(np.allclose(before, hand_grasp_frame(curled), atol=1e-12))

    def test_grasp_error_respects_jaw_symmetry(self):
        from scipy.spatial.transform import Rotation

        R = Rotation.from_euler("xyz", [0.3, -0.4, 0.8]).as_matrix()
        # Swapping which jaw is which is a 180° turn about the approach axis: same grasp.
        self.assertAlmostEqual(grasp_orientation_error(R @ np.diag([1.0, -1.0, -1.0]), R), 0.0, places=9)
        self.assertAlmostEqual(grasp_orientation_error(R, R), 0.0, places=9)
        tilted = R @ Rotation.from_euler("y", 0.5).as_matrix()
        self.assertAlmostEqual(grasp_orientation_error(tilted, R), 0.5, places=9)

    def test_fit_retarget_R_recovers_planted_rotation(self):
        """retarget_R is directly observed per frame, so the fit must be exact."""
        from scipy.spatial.transform import Rotation

        from data_juicer._au.utils.hand_to_robot.transforms import state_to_T

        R_true = Rotation.from_euler("xyz", [1.2, -0.6, 2.4]).as_matrix()
        frames = []
        for i in range(9):
            state = np.array([0.2 + 0.01 * i, 0.05, 0.45, 0.1 * i, 0.2 - 0.03 * i, -0.15 * i, 0.0, 0.5])
            R_wrist = state_to_T(state)[:3, :3]
            frames.append(
                CalibFrame(
                    frame_id=i,
                    state=state,
                    T_world_camera=np.eye(4),
                    joints_cam=mano21_from_frame(state[:3], R_wrist @ R_true),
                )
            )
        R_fit, info = fit_retarget_R_from_grasp(CalibClip(side="right", frames=frames))
        self.assertAlmostEqual(grasp_orientation_error(R_fit, R_true), 0.0, places=6)
        self.assertLess(info["dispersion_deg_p90"], 1e-3)
        self.assertEqual(info["n_frames"], 9)

    def test_fit_retarget_R_needs_mano_joints(self):
        self.assertIsNone(fit_retarget_R_from_grasp(make_synthetic_clip(n_frames=6)))

    def test_rot_prior_is_data_independent(self):
        """w_rot scores |rotvec(retarget_R)|, so it cannot observe retarget_R."""
        from data_juicer._au.utils.hand_to_robot.calibrate import evaluate_clip
        from scipy.spatial.transform import Rotation

        cal = load_calibration(CALIB_RIGHT)
        side = cal.get_side("right")
        side.retarget_R = Rotation.from_euler("y", np.pi).as_matrix()
        weights = CalibWeights(w_ik=0.0, w_uv=0.0)
        errs = [
            evaluate_clip(make_synthetic_clip(side="right", n_frames=n), side, weights=weights)[
                "median_cam_rot_err_rad"
            ]
            for n in (6, 14)
        ]
        self.assertAlmostEqual(errs[0], errs[1], places=9)
        self.assertAlmostEqual(errs[0], np.pi, places=6)

    @unittest.skipUnless(MUJOCO_AVAILABLE and MODEL_RIGHT.is_file(), "mujoco/model missing")
    def test_workspace_map_pulls_targets_into_reach(self):
        import os

        os.environ.setdefault("MUJOCO_GL", "egl")
        from data_juicer._au.utils.hand_to_robot.calibrate import fit_workspace_map, sample_reachable_workspace
        from data_juicer._au.utils.hand_to_robot.transforms import invert_T, state_to_T

        cal = load_calibration(CALIB_RIGHT)
        side = cal.get_side("right")
        clip = make_synthetic_clip(side="right", n_frames=12)
        shell = sample_reachable_workspace(str(MODEL_RIGHT), n_samples=800)
        reach_max = float(np.linalg.norm(shell, axis=1).max())

        ws, info = fit_workspace_map(clip, side, str(MODEL_RIGHT), n_samples=800)
        self.assertLessEqual(info["scale"], 1.0)
        side.workspace_map = ws

        for fr in clip.frames:
            T_world_base = fr.T_world_camera @ side.T_camera_base_ref
            T_world_ee = retarget_wrist_to_ee(fr.state, side, T_world_base=T_world_base)
            p_base = (invert_T(T_world_base) @ np.append(T_world_ee[:3, 3], 1.0))[:3]
            self.assertLess(float(np.linalg.norm(p_base)), reach_max)

        # Without T_world_base the anchor path must still apply.
        p_wrist = state_to_T(clip.frames[0].state)[:3, 3]
        T_anchor = retarget_wrist_to_ee(clip.frames[0].state, side, wrist_ref_world=p_wrist, ee_ref_world=p_wrist)
        self.assertTrue(np.allclose(T_anchor[:3, 3], p_wrist))

    @unittest.skipUnless(MUJOCO_AVAILABLE and MODEL_RIGHT.is_file(), "mujoco/model missing")
    def test_anchor_base_keeps_targets_on_hand_and_in_reach(self):
        import os

        os.environ.setdefault("MUJOCO_GL", "egl")
        from data_juicer._au.utils.hand_to_robot.calibrate import anchor_base_to_hand, sample_reachable_workspace
        from data_juicer._au.utils.hand_to_robot.transforms import invert_T, state_to_T

        cal = load_calibration(CALIB_RIGHT)
        side = cal.get_side("right")
        clip = make_synthetic_clip(side="right", n_frames=12)
        reach_max = float(np.linalg.norm(sample_reachable_workspace(str(MODEL_RIGHT), n_samples=800), axis=1).max())

        T_cam_base, info = anchor_base_to_hand(clip, side, str(MODEL_RIGHT), n_samples=800)
        side.T_camera_base_ref = T_cam_base
        self.assertEqual(T_cam_base.shape, (4, 4))

        for fr in clip.frames:
            T_world_base = fr.T_world_camera @ side.T_camera_base_ref
            p_wrist = state_to_T(fr.state)[:3, 3]
            # Anchoring must not displace the target off the hand.
            T_world_ee = retarget_wrist_to_ee(fr.state, side, wrist_ref_world=p_wrist, ee_ref_world=p_wrist)
            self.assertTrue(np.allclose(T_world_ee[:3, 3], p_wrist))
            p_base = (invert_T(T_world_base) @ np.append(p_wrist, 1.0))[:3]
            self.assertLess(float(np.linalg.norm(p_base)), reach_max)
        self.assertGreater(info["robot_shell_radius_m"], 0.0)

    def test_workspace_map_yaml_roundtrip(self):
        from data_juicer._au.utils.hand_to_robot.calibration import WorkspaceMap

        cal = load_calibration(CALIB_RIGHT)
        side = cal.get_side("right")
        side.workspace_map = WorkspaceMap(
            center_human_base=np.array([0.1, -0.2, 0.3]),
            center_robot_base=np.array([0.4, 0.0, -0.1]),
            scale_xyz=np.full(3, 0.62),
        )
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "ws.yaml"
            save_calibration(cal, out)
            back = load_calibration(out).get_side("right").workspace_map
        self.assertIsNotNone(back)
        self.assertTrue(np.allclose(back.center_human_base, [0.1, -0.2, 0.3]))
        self.assertTrue(np.allclose(back.scale_xyz, 0.62))

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

    @unittest.skipUnless(EGODEX_AVAILABLE, "EgoDex test_lerobot missing")
    def test_egodex_clip_and_p1_helper(self):
        self.assertTrue(is_egodex_lerobot(EGODEX_ROOT))
        clip, meta = clip_from_egodex_lerobot(
            EGODEX_ROOT,
            side="right",
            episode=2,
            max_frames=24,
            stride=8,
            min_wrist_conf=0.3,
        )
        self.assertGreaterEqual(len(clip.frames), 4)
        self.assertEqual(clip.frames[0].state.shape, (8,))
        self.assertIsNotNone(clip.frames[0].joints_cam)
        self.assertEqual(meta["side"], "right")
        fake = p1_metrics_from_eval(
            {"ik_success_rate": 0.95, "median_reprojection_error_px": 10.0},
            num_frames=120,
        )
        self.assertTrue(fake["checks"]["p1_pass"])


if __name__ == "__main__":
    unittest.main()
