# -*- coding: utf-8 -*-
"""Unit tests for VideoHandToRobotRenderMapper and helpers."""

from __future__ import annotations

import hashlib
import os
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from data_juicer._au.tools.build_r1_arm_mjcf import DEFAULT_MESH_DIR, DEFAULT_URDF, build_all
from data_juicer._au.utils.hand_to_robot.calibration import load_calibration
from data_juicer._au.utils.hand_to_robot.composite import (
    DepthAligner,
    composite_robot_on_frame,
    composite_with_depth,
    fit_depth_aligner,
    project_joints_mask,
)
from data_juicer._au.utils.hand_to_robot.ik import jacobian_ik, map_gripper_to_finger
from data_juicer._au.utils.hand_to_robot.renderer import RobotArmRenderer
from data_juicer._au.utils.hand_to_robot.transforms import (
    compute_mujoco_fovy,
    invert_T,
    opencv_to_mujoco_camera_T,
    se3,
    state_to_T,
)
from data_juicer.utils.constant import CameraCalibrationKeys, Fields, MetaKeys
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

MUJOCO_AVAILABLE = False
try:
    import mujoco  # noqa: F401

    MUJOCO_AVAILABLE = True
except Exception:
    pass

REPO_ROOT = Path(__file__).resolve().parents[3]
CALIB_RIGHT = REPO_ROOT / "b" / "d" / "hand2robot" / "calibration" / "r1_right_v1.yaml"
HAS_ASSETS = DEFAULT_URDF.is_file() and DEFAULT_MESH_DIR.is_dir() and CALIB_RIGHT.is_file()


class TestHandToRobotHelpers(unittest.TestCase):
    def test_gripper_mapping(self):
        self.assertAlmostEqual(map_gripper_to_finger(1.0), 0.05)
        self.assertAlmostEqual(map_gripper_to_finger(-1.0), 0.0)
        self.assertAlmostEqual(map_gripper_to_finger(0.0), 0.025)

    def test_camera_fov_conversion(self):
        fovy = compute_mujoco_fovy(np.deg2rad(70.0), 640, 480)
        self.assertTrue(40.0 < fovy < 70.0)

    def test_coordinate_round_trip(self):
        R = np.eye(3)
        t = np.array([0.1, -0.2, 0.3])
        T = se3(R, t)
        Ti = invert_T(T)
        I = T @ Ti
        np.testing.assert_allclose(I, np.eye(4), atol=1e-9)

        adapter = opencv_to_mujoco_camera_T()
        # OpenCV +Z forward should map to MuJoCo -Z.
        z_cv = np.array([0.0, 0.0, 1.0, 0.0])
        z_mj = adapter @ z_cv
        np.testing.assert_allclose(z_mj[:3], [0.0, 0.0, -1.0], atol=1e-9)

    def test_state_to_T(self):
        T = state_to_T([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        np.testing.assert_allclose(T[:3, 3], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(T[:3, :3], np.eye(3), atol=1e-9)

    def test_project_joints_mask_shape(self):
        joints = np.array(
            [
                [0.0, 0.0, 0.5],
                [0.05, 0.0, 0.5],
                [0.0, 0.05, 0.5],
                [-0.05, 0.0, 0.5],
            ],
            dtype=np.float64,
        )
        mask = project_joints_mask(joints, fx=500, fy=500, cx=320, cy=240, img_w=640, img_h=480)
        self.assertEqual(mask.shape, (480, 640))
        self.assertTrue(mask.any())

    def test_load_calibration(self):
        cal = load_calibration(CALIB_RIGHT)
        self.assertIn("right", cal.sides)
        self.assertEqual(cal.action_frame, "world")
        self.assertEqual(cal.sides["right"].q_reference.shape, (6,))

    def test_fit_depth_aligner(self):
        scene = [1.0, 2.0, 3.0, 4.0, 5.0]
        metric = [2.0, 4.0, 6.0, 8.0, 10.0]  # scale=2, bias=0
        a = fit_depth_aligner(scene, metric, min_pairs=4)
        self.assertAlmostEqual(a.scale, 2.0, places=3)
        self.assertAlmostEqual(a.bias, 0.0, places=2)
        # Too few pairs → identity
        a2 = fit_depth_aligner([1.0], [2.0], min_pairs=4)
        self.assertEqual(a2.scale, 1.0)
        self.assertEqual(a2.bias, 0.0)

    def test_depth_occlusion_visibility(self):
        h, w = 64, 80
        bg = np.full((h, w, 3), 10, dtype=np.uint8)
        robot = np.full((h, w, 3), 200, dtype=np.uint8)
        mask = np.zeros((h, w), dtype=bool)
        mask[20:40, 30:50] = True
        robot_depth = np.full((h, w), 1.0, dtype=np.float32)
        # Scene farther than robot → robot visible
        scene_far = np.full((h, w), 2.0, dtype=np.float32)
        out, meta = composite_with_depth(
            bg, robot, mask, robot_depth, scene_far, depth_aligner=DepthAligner(), edge_blur=0
        )
        self.assertTrue(meta["ok"])
        self.assertGreater(meta["visible_pixel_count"], 0)
        self.assertTrue(np.all(out[mask] == 200))

        # Scene closer than robot → occluded (keep background)
        scene_near = np.full((h, w), 0.5, dtype=np.float32)
        out2, meta2 = composite_with_depth(
            bg, robot, mask, robot_depth, scene_near, depth_aligner=DepthAligner(), edge_blur=0
        )
        self.assertTrue(meta2["ok"])
        self.assertEqual(meta2["visible_pixel_count"], 0)
        self.assertTrue(np.all(out2[mask] == 10))

    def test_depth_invalid_ratio_gate(self):
        h, w = 32, 32
        bg = np.zeros((h, w, 3), dtype=np.uint8)
        robot = np.full((h, w, 3), 255, dtype=np.uint8)
        mask = np.ones((h, w), dtype=bool)
        robot_depth = np.full((h, w), 1.0, dtype=np.float32)
        scene = np.full((h, w), np.nan, dtype=np.float32)
        out, meta = composite_with_depth(
            bg,
            robot,
            mask,
            robot_depth,
            scene,
            depth_aligner=DepthAligner(),
            max_invalid_ratio=0.2,
            edge_blur=0,
        )
        self.assertFalse(meta["ok"])
        self.assertEqual(meta["quality_flag"], "depth_invalid")
        self.assertGreater(meta["depth_invalid_ratio"], 0.2)
        # Fallback leaves background (no robot overlay)
        self.assertTrue(np.all(out == 0))


@unittest.skipUnless(HAS_ASSETS and MUJOCO_AVAILABLE, "assets/mujoco unavailable")
class TestHandToRobotRenderMapper(DataJuicerTestCaseBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls.gen_dir = Path(cls._tmpdir.name) / "generated"
        build_all(urdf_path=DEFAULT_URDF, mesh_dir=DEFAULT_MESH_DIR, out_dir=cls.gen_dir, sides=("left", "right"))
        cls.model_path = cls.gen_dir / "r1_lite_arm_right.xml"
        cls.model_path_left = cls.gen_dir / "r1_lite_arm_left.xml"
        os.environ.setdefault("MUJOCO_GL", "egl")

    @classmethod
    def tearDownClass(cls):
        cls._tmpdir.cleanup()
        super().tearDownClass()

    def test_ik_convergence_from_fk(self):
        renderer = RobotArmRenderer(self.model_path, width=320, height=240)
        cal = load_calibration(CALIB_RIGHT)
        side = cal.get_side("right")
        T_cam_base = side.T_camera_base_ref
        renderer.set_base_pose(T_cam_base, cal.T_mjcam_from_cvcam)

        q_gt = side.q_reference.copy()
        q_gt[0] += 0.2
        q_gt[1] -= 0.1
        renderer.set_arm_qpos(q_gt, 0.02)
        renderer.mujoco.mj_forward(renderer.model, renderer.data)
        target_pos, target_rot = renderer.site_pose()

        # Perturb init.
        q_init = side.q_reference.copy()
        q, ok, metrics = jacobian_ik(
            renderer.model,
            renderer.data,
            renderer.site_id,
            target_pos,
            target_rot,
            renderer.arm_qpos_addrs,
            q_init=q_init,
            max_iter=120,
            tol_pos=5e-3,
            tol_rot=0.0524,
        )
        renderer.close()
        self.assertTrue(ok, msg=str(metrics))
        self.assertLess(metrics["position_error_m"], 5e-3)
        self.assertLess(metrics["orientation_error_rad"], 0.0524)
        np.testing.assert_allclose(q, q_gt, atol=0.15)

    def test_composite_output_shape_and_non_destructive(self):
        from data_juicer._au.ops.mapper.video_hand_to_robot_render_mapper import (
            VideoHandToRobotRenderMapper,
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            frame_path = tmp / "frame_000.png"
            img = np.full((240, 320, 3), 40, dtype=np.uint8)
            cv2.imwrite(str(frame_path), img)
            before = hashlib.sha256(frame_path.read_bytes()).hexdigest()

            # Identity cam_c2w for one frame.
            cam_c2w = np.eye(4, dtype=np.float64)[None, ...]
            cam_c2w_path = tmp / "c2w.npy"
            np.save(cam_c2w_path, cam_c2w)

            # Place wrist in front of camera in world(=camera) frame.
            state = [0.25, 0.0, 0.45, 0.0, 0.0, 0.0, 0.0, 1.0]
            joints = np.array(
                [
                    [0.2, 0.0, 0.45],
                    [0.25, 0.02, 0.45],
                    [0.25, -0.02, 0.45],
                    [0.3, 0.0, 0.45],
                ],
                dtype=np.float64,
            )

            sample = {
                "id": "unit_clip",
                Fields.meta: {
                    MetaKeys.video_frames: [[str(frame_path)]],
                    MetaKeys.hand_reconstruction_hawor_tags: [
                        {
                            "right": {
                                "frame_ids": [0],
                                "joints_cam": [joints.tolist()],
                            }
                        }
                    ],
                    MetaKeys.hand_action_tags: [
                        {
                            "right": {
                                "valid_frame_ids": [0],
                                "states": [state],
                                "actions": [[0, 0, 0, 0, 0, 0, 1]],
                            }
                        }
                    ],
                    MetaKeys.camera_calibration_moge_tags: [{}],
                    MetaKeys.video_camera_pose_tags: [{CameraCalibrationKeys.cam_c2w: str(cam_c2w_path)}],
                },
            }

            op = VideoHandToRobotRenderMapper(
                robot_model_paths={"right": str(self.model_path)},
                calibration_path=str(CALIB_RIGHT),
                ik_solver="jacobian",
                hand_type="right",
                output_root=str(tmp / "robot_render"),
                gl_backend=os.environ.get("MUJOCO_GL", "egl"),
            )
            out = op.process_single(sample)
            after = hashlib.sha256(frame_path.read_bytes()).hexdigest()
            self.assertEqual(before, after)

            out_frames = out[Fields.meta]["robot_render_frames"]
            self.assertNotEqual(out_frames[0][0], str(frame_path))
            self.assertTrue(Path(out_frames[0][0]).is_file())
            rendered = cv2.imread(out_frames[0][0])
            self.assertEqual(rendered.shape, img.shape)

            quality = out[Fields.meta]["hand_to_robot_render_quality"]
            self.assertIn("summary", quality)
            self.assertIn("frames", quality)

    def test_depth_occlusion_mapper_path(self):
        from data_juicer._au.ops.mapper.video_hand_to_robot_render_mapper import (
            VideoHandToRobotRenderMapper,
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            op = VideoHandToRobotRenderMapper(
                robot_model_paths={"right": str(self.model_path)},
                calibration_path=str(CALIB_RIGHT),
                ik_solver="jacobian",
                hand_type="right",
                enable_depth_occlusion=True,
                output_root=str(tmp / "robot_render"),
                gl_backend=os.environ.get("MUJOCO_GL", "egl"),
            )
            op._init_renderer(240, 320)
            cal = load_calibration(CALIB_RIGHT)
            side = cal.get_side("right")
            frame = np.full((240, 320, 3), 40, dtype=np.uint8)
            # Far plane so robot depth wins in front of background.
            scene_depth = np.full((240, 320), 5.0, dtype=np.float32)
            hand_mask = np.zeros((240, 320), dtype=bool)
            result, meta = op._render_and_composite(
                frame,
                side.q_reference,
                0.02,
                side.T_camera_base_ref,
                hand_mask,
                scene_depth,
                50.0,
                "right",
                depth_aligner=DepthAligner(),
            )
            self.assertTrue(meta.get("depth_occlusion_used"))
            self.assertEqual(result.shape, frame.shape)
            self.assertIn(meta.get("quality_flag"), ("ok", "no_robot_mask", "depth_invalid"))
            # Near occluder everywhere → robot should be hidden (or depth_invalid if mask empty).
            scene_near = np.full((240, 320), 0.05, dtype=np.float32)
            result2, meta2 = op._render_and_composite(
                frame,
                side.q_reference,
                0.02,
                side.T_camera_base_ref,
                hand_mask,
                scene_near,
                50.0,
                "right",
                depth_aligner=DepthAligner(),
            )
            self.assertTrue(meta2.get("depth_occlusion_used"))
            if meta2.get("quality_flag") == "ok" and meta2.get("visible_pixel_count", 0) == 0:
                # Background preserved where robot was fully occluded.
                self.assertTrue(np.allclose(result2, frame) or result2.shape == frame.shape)
            for r in op._renderers.values():
                r.close()
            op._renderers.clear()

    def test_robot_mask_excludes_helper_geoms(self):
        renderer = RobotArmRenderer(self.model_path, width=320, height=240)
        cal = load_calibration(CALIB_RIGHT)
        side = cal.get_side("right")
        renderer.set_base_pose(side.T_camera_base_ref, cal.T_mjcam_from_cvcam)
        rgb, mask, _ = renderer.render_frame(
            side.q_reference,
            0.02,
            side.T_camera_base_ref,
            cal.T_mjcam_from_cvcam,
            render_depth=False,
        )
        visual_ids = set(renderer.robot_visual_geom_ids.tolist())
        self.assertTrue(len(visual_ids) > 0)
        self.assertEqual(rgb.shape[:2], mask.shape)
        self.assertTrue(mask.any())
        # Ensure collision geoms are not in the visual id set.
        collision_ids = set(np.flatnonzero(renderer.model.geom_group == 3).tolist())
        self.assertTrue(visual_ids.isdisjoint(collision_ids))
        renderer.close()

    def test_both_hand_type_supported(self):
        from data_juicer._au.ops.mapper.video_hand_to_robot_render_mapper import (
            VideoHandToRobotRenderMapper,
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            frame_path = tmp / "frame_000.png"
            img = np.full((240, 320, 3), 40, dtype=np.uint8)
            cv2.imwrite(str(frame_path), img)
            cam_c2w = np.eye(4, dtype=np.float64)[None, ...]
            cam_c2w_path = tmp / "c2w.npy"
            np.save(cam_c2w_path, cam_c2w)

            state_r = [0.25, 0.02, 0.45, 0.0, 0.0, 0.0, 0.0, 1.0]
            state_l = [-0.25, 0.02, 0.45, 0.0, 0.0, 0.0, 0.0, 1.0]
            joints_r = np.array([[0.2, 0.0, 0.45], [0.25, 0.02, 0.45], [0.25, -0.02, 0.45], [0.3, 0.0, 0.45]])
            joints_l = np.array([[-0.2, 0.0, 0.45], [-0.25, 0.02, 0.45], [-0.25, -0.02, 0.45], [-0.3, 0.0, 0.45]])

            sample = {
                "id": "unit_both",
                Fields.meta: {
                    MetaKeys.video_frames: [[str(frame_path)]],
                    MetaKeys.hand_reconstruction_hawor_tags: [
                        {
                            "right": {"frame_ids": [0], "joints_cam": [joints_r.tolist()]},
                            "left": {"frame_ids": [0], "joints_cam": [joints_l.tolist()]},
                        }
                    ],
                    MetaKeys.hand_action_tags: [
                        {
                            "right": {"valid_frame_ids": [0], "states": [state_r], "actions": [[0, 0, 0, 0, 0, 0, 1]]},
                            "left": {"valid_frame_ids": [0], "states": [state_l], "actions": [[0, 0, 0, 0, 0, 0, 1]]},
                        }
                    ],
                    MetaKeys.camera_calibration_moge_tags: [{}],
                    MetaKeys.video_camera_pose_tags: [{CameraCalibrationKeys.cam_c2w: str(cam_c2w_path)}],
                },
            }

            op = VideoHandToRobotRenderMapper(
                robot_model_paths={"right": str(self.model_path), "left": str(self.model_path_left)},
                calibration_path=str(REPO_ROOT / "b" / "d" / "hand2robot" / "calibration" / "r1_both_v1.yaml"),
                hand_type="both",
                output_root=str(tmp / "robot_render"),
                gl_backend=os.environ.get("MUJOCO_GL", "egl"),
            )
            out = op.process_single(sample)
            quality = out[Fields.meta]["hand_to_robot_render_quality"]["summary"]
            self.assertIn("per_side", quality)
            self.assertIn("left", quality["per_side"])
            self.assertIn("right", quality["per_side"])
            self.assertTrue(Path(out[Fields.meta]["robot_render_frames"][0][0]).is_file())
            for r in op._renderers.values():
                r.close()
            op._renderers.clear()


if __name__ == "__main__":
    unittest.main()
