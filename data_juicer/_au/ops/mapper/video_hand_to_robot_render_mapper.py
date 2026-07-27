# -*- coding: utf-8 -*-
"""Replace ego hand pixels with a rendered R1 Lite robot arm.

Consumes MotionSmooth world-frame states as the authoritative wrist trajectory,
solves IK in the robot-base frame, renders in the ego camera frame, and writes
non-destructive robot-view frames plus quality metadata.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from loguru import logger

from data_juicer.ops.base_op import OPERATORS, Mapper
from data_juicer.utils.constant import CameraCalibrationKeys, Fields, MetaKeys
from data_juicer.utils.file_utils import load_numpy

from ...utils.hand_to_robot.calibration import HandToRobotCalibration, load_calibration
from ...utils.hand_to_robot.composite import (
    DepthAligner,
    bbox_to_mask,
    composite_robot_on_frame,
    composite_with_depth,
    fit_depth_aligner,
    project_joints_mask,
    resize_depth,
)
from ...utils.hand_to_robot.ik import jacobian_ik, map_gripper_to_finger
from ...utils.hand_to_robot.renderer import RobotArmRenderer
from ...utils.hand_to_robot.retarget import palm_pixel_from_joints, retarget_wrist_to_ee
from ...utils.hand_to_robot.transforms import (
    compute_mujoco_fovy,
    invert_T,
    se3,
    state_to_T,
)

OP_NAME = "video_hand_to_robot_render_mapper"


def _median_or_nan(vals: Sequence[float]) -> float:
    arr = np.asarray([v for v in vals if v is not None and np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.median(arr))


@OPERATORS.register_module(OP_NAME)
class VideoHandToRobotRenderMapper(Mapper):
    """Render R1 Lite arm into ego frames from smoothed world-frame wrist states."""

    def __init__(
        self,
        robot_model_paths: dict,
        calibration_path: str,
        hand_reconstruction_field: str = MetaKeys.hand_reconstruction_hawor_tags,
        hand_action_field: str = MetaKeys.hand_action_tags,
        camera_calibration_field: str = MetaKeys.camera_calibration_moge_tags,
        camera_pose_field: str = MetaKeys.video_camera_pose_tags,
        frame_field: str = MetaKeys.video_frames,
        output_frame_field: str = "robot_render_frames",
        quality_field: str = "hand_to_robot_render_quality",
        output_root: str = None,
        ik_solver: str = "jacobian",
        ik_max_iter: int = 100,
        ik_tol_pos: float = 5e-3,
        ik_tol_rot: float = 0.0524,
        ik_damping: float = 1e-2,
        ik_failure_policy: str = "hold_then_keep_original",
        max_hold_frames: int = 2,
        hand_type: str = "right",
        enable_depth_occlusion: bool = False,
        depth_epsilon_m: float = 0.02,
        max_depth_invalid_ratio: float = 0.2,
        hand_mask_method: str = "joints_bbox_union",
        inpaint_method: str = "telea",
        edge_blur: int = 3,
        gl_backend: str = "egl",
        *args,
        **kwargs,
    ):
        """
        :param robot_model_paths: Mapping ``{"left"|"right": mjcf_path}``.
        :param calibration_path: Versioned retarget/base/camera YAML.
        :param output_root: Directory for rendered frames; defaults next to inputs.
        :param ik_solver: ``jacobian`` (P0) or ``mink`` (not yet implemented).
        :param hand_type: ``left`` or ``right`` (``both`` reserved for P4).
        :param enable_depth_occlusion: P2 depth-aware composite using MoGe scene depth.
        :param depth_epsilon_m: Robot wins if ``D_robot <= D_scene + epsilon``.
        :param max_depth_invalid_ratio: Reject robot overlay when invalid depth fraction
            inside robot mask exceeds this (fallback: hand-inpainted original).
        """
        super().__init__(*args, **kwargs)
        if hand_type == "both":
            raise NotImplementedError("P0-P3 only support one side; dual-arm is a P4 feature")
        if hand_type not in ("left", "right"):
            raise ValueError(f"hand_type must be left/right, got {hand_type}")
        if ik_solver not in ("jacobian", "mink"):
            raise ValueError(f"Unsupported ik_solver: {ik_solver}")
        if ik_solver == "mink":
            raise NotImplementedError("mink solver is planned for P1; use ik_solver='jacobian' for P0")

        self.robot_model_paths = {k: str(v) for k, v in dict(robot_model_paths).items()}
        if hand_type not in self.robot_model_paths:
            raise KeyError(f"robot_model_paths missing entry for hand_type={hand_type}")

        self.calibration_path = str(calibration_path)
        self.calibration: HandToRobotCalibration = load_calibration(self.calibration_path)
        if hand_type not in self.calibration.sides:
            raise KeyError(f"calibration missing side '{hand_type}': {self.calibration_path}")

        self.hand_reconstruction_field = hand_reconstruction_field
        self.hand_action_field = hand_action_field
        self.camera_calibration_field = camera_calibration_field
        self.camera_pose_field = camera_pose_field
        self.frame_field = frame_field
        self.output_frame_field = output_frame_field
        self.quality_field = quality_field
        self.output_root = output_root

        self.ik_solver = ik_solver
        self.ik_max_iter = int(ik_max_iter)
        self.ik_tol_pos = float(ik_tol_pos)
        self.ik_tol_rot = float(ik_tol_rot)
        self.ik_damping = float(ik_damping)
        self.ik_failure_policy = ik_failure_policy
        self.max_hold_frames = int(max_hold_frames)

        self.hand_type = hand_type
        self._hand_sides = [hand_type]
        self.enable_depth_occlusion = bool(enable_depth_occlusion)
        self.depth_epsilon_m = float(depth_epsilon_m)
        self.max_depth_invalid_ratio = float(max_depth_invalid_ratio)
        self.hand_mask_method = hand_mask_method
        self.inpaint_method = inpaint_method
        self.edge_blur = int(edge_blur)
        self.gl_backend = gl_backend

        self._renderers: Dict[str, RobotArmRenderer] = {}
        self._base_T_world: Dict[str, Optional[np.ndarray]] = {hand_type: None}
        self._clip_wrist_refs: Dict[Tuple[int, str], Optional[np.ndarray]] = {}
        self._clip_depth_aligners: Dict[Tuple[int, str], DepthAligner] = {}

    # ------------------------------------------------------------------
    # Lazy init / IO helpers
    # ------------------------------------------------------------------
    def _init_renderer(self, width: int, height: int) -> None:
        side = self.hand_type
        renderer = self._renderers.get(side)
        if renderer is not None and renderer.width == width and renderer.height == height:
            return
        if renderer is not None:
            renderer.close()
        self._renderers[side] = RobotArmRenderer(
            self.robot_model_paths[side],
            width=width,
            height=height,
            gl_backend=self.gl_backend,
        )

    def _atomic_imwrite(self, path: str | Path, image_bgr: np.ndarray) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(suffix=".png", dir=str(path.parent))
        os.close(fd)
        tmp_path = Path(tmp)
        try:
            if not cv2.imwrite(str(tmp_path), image_bgr):
                raise RuntimeError(f"cv2.imwrite failed for {tmp_path}")
            os.replace(tmp_path, path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    def _build_output_path(self, sample: dict, clip_idx: int, frame_id: int, hand_side: str, src_path: str) -> str:
        src = Path(src_path)
        sample_id = str(sample.get("id", sample.get("video_id", "sample")))
        root = Path(self.output_root) if self.output_root else src.parent / "robot_render"
        out = root / sample_id / f"clip{clip_idx:03d}" / hand_side / f"frame_{frame_id:06d}{src.suffix or '.png'}"
        return str(out)

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------
    def _map_gripper(self, gripper_state: float) -> float:
        return map_gripper_to_finger(gripper_state)

    def _retarget_state_to_ee(self, smoothed_state: Sequence[float], hand_side: str, clip_idx: int) -> np.ndarray:
        """Map smoothed world-frame wrist state → T_world_ee."""
        side_cal = self.calibration.get_side(hand_side)
        key = (clip_idx, hand_side)
        wrist_ref = side_cal.wrist_ref_world
        if wrist_ref is None:
            if self._clip_wrist_refs.get(key) is None:
                self._clip_wrist_refs[key] = state_to_T(smoothed_state)[:3, 3].copy()
            wrist_ref = self._clip_wrist_refs[key]
        return retarget_wrist_to_ee(
            smoothed_state,
            side_cal,
            wrist_ref_world=wrist_ref,
            ee_ref_world=side_cal.ee_ref_world if side_cal.ee_ref_world is not None else wrist_ref,
        )

    def _compute_base_poses(
        self,
        cam_c2w: np.ndarray,
        frame_id: int,
        hand_side: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (T_world_base, T_camera_base) with low-pass base tracking."""
        side_cal = self.calibration.get_side(hand_side)
        T_world_camera = np.asarray(cam_c2w[frame_id], dtype=np.float64)
        T_camera_base_ref = side_cal.T_camera_base_ref
        T_world_base_raw = T_world_camera @ T_camera_base_ref

        prev = self._base_T_world.get(hand_side)
        tau = max(float(self.calibration.base_lowpass_tau), 1e-3)
        alpha = 1.0 - np.exp(-1.0 / tau)  # per-frame approx; fps-agnostic default
        if prev is None:
            T_world_base = T_world_base_raw
        else:
            # Translation lerp + rotation slerp via matrix blend renormalized.
            t = (1.0 - alpha) * prev[:3, 3] + alpha * T_world_base_raw[:3, 3]
            R = (1.0 - alpha) * prev[:3, :3] + alpha * T_world_base_raw[:3, :3]
            U, _, Vt = np.linalg.svd(R)
            R = U @ Vt
            if np.linalg.det(R) < 0:
                U[:, -1] *= -1
                R = U @ Vt
            T_world_base = se3(R, t)
        self._base_T_world[hand_side] = T_world_base
        T_camera_base = invert_T(T_world_camera) @ T_world_base
        return T_world_base, T_camera_base

    def _solve_ik(self, T_base_ee: np.ndarray, q_prev: Optional[np.ndarray], hand_side: str) -> Tuple[np.ndarray, bool, dict]:
        renderer = self._renderers[hand_side]
        # Express target in MuJoCo world: with base mocap already set, MuJoCo world == camera-adapted frame.
        # Caller must set base pose before IK so FK matches render.
        # Here T_base_ee is relative to robot base; convert to MuJoCo world via current mocap.
        T_mj_base = np.eye(4, dtype=np.float64)
        T_mj_base[:3, 3] = renderer.data.mocap_pos[renderer.anchor_mocap_id]
        # quat wxyz -> R
        from ...utils.hand_to_robot.transforms import quat_wxyz_to_mat

        T_mj_base[:3, :3] = quat_wxyz_to_mat(renderer.data.mocap_quat[renderer.anchor_mocap_id])
        T_mj_ee = T_mj_base @ T_base_ee
        target_pos = T_mj_ee[:3, 3]
        target_rot = T_mj_ee[:3, :3]

        q, ok, metrics = jacobian_ik(
            renderer.model,
            renderer.data,
            renderer.site_id,
            target_pos,
            target_rot,
            renderer.arm_qpos_addrs,
            q_init=q_prev,
            max_iter=self.ik_max_iter,
            tol_pos=self.ik_tol_pos,
            tol_rot=self.ik_tol_rot,
            damping=self.ik_damping,
        )
        return q, ok, metrics

    def _get_hand_mask(
        self,
        joints_cam: Sequence,
        hand_data: dict,
        frame_shape: Tuple[int, ...],
        intrinsics: Optional[dict],
        hand_idx: int,
    ) -> np.ndarray:
        h, w = frame_shape[:2]
        mask = np.zeros((h, w), dtype=bool)

        if self.hand_mask_method in ("joints_bbox_union", "joints", "mano_projection"):
            fx = fy = None
            cx, cy = w * 0.5, h * 0.5
            if intrinsics is not None:
                fx = intrinsics.get("fx")
                fy = intrinsics.get("fy", fx)
                cx = intrinsics.get("cx", cx)
                cy = intrinsics.get("cy", cy)
            if fx is None:
                # Fallback crude focal from FOV~70deg.
                fx = fy = 0.5 * w / np.tan(np.deg2rad(35.0))
            mask |= project_joints_mask(np.asarray(joints_cam), float(fx), float(fy), float(cx), float(cy), w, h)

        if self.hand_mask_method in ("joints_bbox_union", "bbox"):
            bboxes = hand_data.get("bboxes") or hand_data.get("bbox") or []
            if hand_idx < len(bboxes) and bboxes[hand_idx] is not None:
                mask |= bbox_to_mask(bboxes[hand_idx], frame_shape)
        return mask

    def _load_camera_inputs(self, clip_camera: dict, frame_id: int, frame_shape: Tuple[int, ...]) -> Tuple[Optional[np.ndarray], float, dict]:
        h, w = frame_shape[:2]
        fov_y_deg = 60.0
        intrinsics: dict = {}
        scene_depth = None

        if not clip_camera:
            return scene_depth, fov_y_deg, intrinsics

        # Prefer explicit FOV fields when present.
        hfov = clip_camera.get(CameraCalibrationKeys.hfov) or clip_camera.get("hfov")
        vfov = clip_camera.get(CameraCalibrationKeys.vfov) or clip_camera.get("vfov")
        if isinstance(hfov, (list, tuple)) and frame_id < len(hfov):
            fov_x = float(hfov[frame_id])
            # MoGe may store degrees or radians; treat > π as degrees.
            if fov_x > np.pi:
                fov_x = np.deg2rad(fov_x)
            fov_y_deg = compute_mujoco_fovy(fov_x, w, h)
        elif isinstance(vfov, (list, tuple)) and frame_id < len(vfov):
            fov_y = float(vfov[frame_id])
            fov_y_deg = float(fov_y if fov_y > np.pi else np.degrees(fov_y))

        Ks = clip_camera.get(CameraCalibrationKeys.intrinsics) or clip_camera.get("intrinsics")
        if isinstance(Ks, (list, tuple)) and frame_id < len(Ks):
            K = np.asarray(load_numpy(Ks[frame_id]) if not isinstance(Ks[frame_id], (list, np.ndarray)) else Ks[frame_id], dtype=np.float64)
            if K.shape == (3, 3):
                intrinsics = {"fx": float(K[0, 0]), "fy": float(K[1, 1]), "cx": float(K[0, 2]), "cy": float(K[1, 2])}
                # Recompute fovy from fy when available.
                fov_y_deg = float(np.degrees(2.0 * np.arctan(0.5 * h / max(intrinsics["fy"], 1e-6))))

        depths = clip_camera.get(CameraCalibrationKeys.depth) or clip_camera.get("depth")
        if self.enable_depth_occlusion and isinstance(depths, (list, tuple)) and frame_id < len(depths):
            scene_depth = np.asarray(load_numpy(depths[frame_id]), dtype=np.float32)

        return scene_depth, fov_y_deg, intrinsics

    def _fit_clip_depth_aligner(
        self,
        clip_camera: dict,
        joints_list: Sequence,
        hand_index_by_frame: Dict[int, int],
        frame_ids: Sequence[int],
        frame_shape: Tuple[int, ...],
    ) -> DepthAligner:
        """Fit per-clip affine depth aligner from MANO wrist z vs MoGe depth."""
        if not self.enable_depth_occlusion or not clip_camera:
            return DepthAligner()
        depths = clip_camera.get(CameraCalibrationKeys.depth) or clip_camera.get("depth")
        if not isinstance(depths, (list, tuple)) or not depths:
            return DepthAligner()

        scene_vals: List[float] = []
        metric_vals: List[float] = []
        h, w = frame_shape[:2]
        # Sample up to ~40 frames for speed.
        step = max(1, len(frame_ids) // 40)
        for frame_id in list(frame_ids)[::step]:
            fid = int(frame_id)
            if fid < 0 or fid >= len(depths):
                continue
            hand_idx = hand_index_by_frame.get(fid)
            if hand_idx is None or hand_idx >= len(joints_list):
                continue
            joints = np.asarray(joints_list[hand_idx], dtype=np.float64)
            if joints.ndim != 2 or joints.shape[0] < 1:
                continue
            _, _, intrinsics = self._load_camera_inputs(clip_camera, fid, frame_shape)
            fx = float(intrinsics.get("fx", 0.5 * w / np.tan(np.deg2rad(35.0))))
            fy = float(intrinsics.get("fy", fx))
            cx = float(intrinsics.get("cx", 0.5 * w))
            cy = float(intrinsics.get("cy", 0.5 * h))
            # Use wrist + a few finger tips when available (MANO 0,4,8,12,16,20).
            idxs = [0] + [i for i in (4, 8, 12, 16, 20) if i < joints.shape[0]]
            scene = np.asarray(load_numpy(depths[fid]), dtype=np.float32)
            scene = resize_depth(scene, h, w)
            for ji in idxs:
                u, v, ok = palm_pixel_from_joints(joints[ji : ji + 1], fx, fy, cx, cy)
                if not ok:
                    continue
                ui, vi = int(round(u)), int(round(v))
                if ui < 0 or vi < 0 or ui >= w or vi >= h:
                    continue
                sd = float(scene[vi, ui])
                md = float(joints[ji, 2])
                if np.isfinite(sd) and np.isfinite(md) and sd > 1e-4 and md > 1e-4:
                    scene_vals.append(sd)
                    metric_vals.append(md)
        aligner = fit_depth_aligner(scene_vals, metric_vals)
        logger.debug(
            "Depth aligner fit pairs={} scale={:.4f} bias={:.4f}",
            len(scene_vals),
            aligner.scale,
            aligner.bias,
        )
        return aligner

    def _render_and_composite(
        self,
        frame_bgr: np.ndarray,
        joint_angles: np.ndarray,
        finger_pos: float,
        T_camera_base: np.ndarray,
        hand_mask: np.ndarray,
        scene_depth: Optional[np.ndarray],
        fov_y_deg: float,
        hand_side: str,
        depth_aligner: Optional[DepthAligner] = None,
    ) -> Tuple[np.ndarray, dict]:
        renderer = self._renderers[hand_side]
        renderer.set_camera_fov(fov_y_deg)
        rgb, mask, depth = renderer.render_frame(
            joint_angles,
            finger_pos,
            T_camera_base,
            self.calibration.T_mjcam_from_cvcam,
            render_depth=self.enable_depth_occlusion,
        )
        # MuJoCo RGB is RGB; OpenCV frames are BGR.
        robot_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        composite_meta = {
            "ok": True,
            "quality_flag": "ok",
            "depth_invalid_ratio": 0.0,
            "depth_occlusion_used": False,
        }
        if self.enable_depth_occlusion and depth is not None and scene_depth is not None:
            result, depth_meta = composite_with_depth(
                frame_bgr,
                robot_bgr,
                mask,
                depth,
                scene_depth,
                depth_aligner=depth_aligner or DepthAligner(),
                hand_mask=hand_mask,
                epsilon_m=self.depth_epsilon_m,
                max_invalid_ratio=self.max_depth_invalid_ratio,
                edge_blur=self.edge_blur,
                inpaint_method=self.inpaint_method,
            )
            composite_meta.update(depth_meta)
            composite_meta["depth_occlusion_used"] = True
            return result, composite_meta

        if self.enable_depth_occlusion:
            composite_meta["quality_flag"] = "depth_missing"
            composite_meta["ok"] = False
        result = composite_robot_on_frame(
            frame_bgr,
            robot_bgr,
            mask,
            hand_mask=hand_mask,
            edge_blur=self.edge_blur,
            inpaint_method=self.inpaint_method,
        )
        return result, composite_meta

    def _aggregate_quality(self, records: List[dict], hand_side: str) -> dict:
        if not records:
            return {
                "schema_version": 1,
                "calibration_version": self.calibration.version_name,
                "action_frame": self.calibration.action_frame,
                "render_frame": "camera",
                "hand_side": hand_side,
                "ik_success_rate": 0.0,
                "failed_frame_ids": [],
                "enable_depth_occlusion": self.enable_depth_occlusion,
            }
        ik_success = [bool(r.get("ik_ok", r.get("quality_flag") == "ok")) for r in records]
        pos_errs = [r.get("position_error_m") for r in records]
        rot_errs = [r.get("orientation_error_rad") for r in records]
        failed = [int(r["frame_id"]) for r in records if r.get("quality_flag") == "ik_failed"]
        depth_invalid = [int(r["frame_id"]) for r in records if r.get("quality_flag") == "depth_invalid"]
        depth_ratios = [
            float(r["depth_invalid_ratio"])
            for r in records
            if r.get("depth_invalid_ratio") is not None and np.isfinite(r.get("depth_invalid_ratio"))
        ]
        return {
            "schema_version": 1,
            "calibration_version": self.calibration.version_name,
            "action_frame": self.calibration.action_frame,
            "render_frame": "camera",
            "hand_side": hand_side,
            "ik_success_rate": float(np.mean(ik_success)) if ik_success else 0.0,
            "workspace_projection_rate": float(np.mean([bool(r.get("workspace_projected")) for r in records])),
            "median_position_error_m": _median_or_nan(pos_errs),
            "median_orientation_error_rad": _median_or_nan(rot_errs),
            "failed_frame_ids": failed,
            "num_frames": len(records),
            "enable_depth_occlusion": self.enable_depth_occlusion,
            "median_depth_invalid_ratio": _median_or_nan(depth_ratios),
            "depth_invalid_frame_ids": depth_invalid,
            "depth_occlusion_frame_rate": float(
                np.mean([bool(r.get("depth_occlusion_used")) for r in records])
            ),
        }

    # ------------------------------------------------------------------
    # Main
    # ------------------------------------------------------------------
    def process_single(self, sample, rank=None):
        meta = sample.setdefault(Fields.meta, {})
        frames = meta.get(self.frame_field, []) or []
        hawor = meta.get(self.hand_reconstruction_field, []) or []
        actions = meta.get(self.hand_action_field, []) or []
        cameras = meta.get(self.camera_calibration_field, []) or []
        camera_poses = meta.get(self.camera_pose_field, []) or []

        if not frames:
            logger.warning("No video frames; skip hand-to-robot render.")
            meta[self.output_frame_field] = []
            meta[self.quality_field] = {"frames": [], "summary": {}}
            return sample

        # Probe resolution from first existing frame.
        first_path = None
        for clip in frames:
            for p in clip:
                if p and os.path.isfile(p):
                    first_path = p
                    break
            if first_path:
                break
        if first_path is None:
            logger.warning("Frame paths exist in meta but no readable file found.")
            meta[self.output_frame_field] = [list(c) for c in frames]
            meta[self.quality_field] = {"frames": [], "summary": {}}
            return sample

        probe = cv2.imread(first_path)
        if probe is None:
            raise RuntimeError(f"Failed to read frame: {first_path}")
        h, w = probe.shape[:2]
        self._init_renderer(w, h)

        n_clips = len(frames)
        output_frames = [list(clip) for clip in frames]
        quality_records: List[dict] = []

        for clip_idx in range(n_clips):
            clip_frames = frames[clip_idx]
            clip_hawor = hawor[clip_idx] if clip_idx < len(hawor) else {}
            clip_actions = actions[clip_idx] if clip_idx < len(actions) else {}
            clip_camera = cameras[clip_idx] if clip_idx < len(cameras) else {}
            clip_camera_pose = camera_poses[clip_idx] if clip_idx < len(camera_poses) else {}

            cam_c2w_raw = None
            if isinstance(clip_camera_pose, dict):
                cam_c2w_raw = clip_camera_pose.get(CameraCalibrationKeys.cam_c2w)
            if cam_c2w_raw is None:
                logger.warning(f"clip {clip_idx}: missing cam_c2w; skip render.")
                continue
            cam_c2w = np.asarray(load_numpy(cam_c2w_raw), dtype=np.float64)
            if cam_c2w.ndim != 3 or cam_c2w.shape[-2:] != (4, 4):
                logger.warning(f"clip {clip_idx}: unexpected cam_c2w shape {cam_c2w.shape}")
                continue

            for hand_side in self._hand_sides:
                hand_data = (clip_hawor or {}).get(hand_side, {}) or {}
                action_data = (clip_actions or {}).get(hand_side, {}) or {}
                frame_ids = action_data.get("valid_frame_ids", []) or []
                states = action_data.get("states", []) or []
                if not frame_ids or len(states) != len(frame_ids):
                    continue

                hand_frame_ids = hand_data.get("frame_ids", []) or []
                hand_index_by_frame = {int(fid): i for i, fid in enumerate(hand_frame_ids)}
                joints_list = hand_data.get("joints_cam", []) or []

                self._base_T_world[hand_side] = None
                self._clip_wrist_refs[(clip_idx, hand_side)] = None
                q_prev = None
                hold_count = 0
                side_cal = self.calibration.get_side(hand_side)
                renderer = self._renderers[hand_side]

                # Probe first readable frame for depth-aligner fit resolution.
                probe_shape = None
                for fid0 in frame_ids:
                    fid0 = int(fid0)
                    if 0 <= fid0 < len(clip_frames) and os.path.isfile(clip_frames[fid0]):
                        img0 = cv2.imread(clip_frames[fid0])
                        if img0 is not None:
                            probe_shape = img0.shape
                            break
                depth_aligner = DepthAligner()
                if self.enable_depth_occlusion and probe_shape is not None:
                    depth_aligner = self._fit_clip_depth_aligner(
                        clip_camera,
                        joints_list,
                        hand_index_by_frame,
                        frame_ids,
                        probe_shape,
                    )
                self._clip_depth_aligners[(clip_idx, hand_side)] = depth_aligner

                for t, frame_id in enumerate(frame_ids):
                    frame_id = int(frame_id)
                    if frame_id < 0 or frame_id >= len(clip_frames) or frame_id >= len(cam_c2w):
                        continue
                    hand_idx = hand_index_by_frame.get(frame_id)
                    frame_path = clip_frames[frame_id]
                    frame_img = cv2.imread(frame_path)
                    if frame_img is None:
                        logger.warning(f"Failed to read {frame_path}")
                        continue

                    T_world_ee = self._retarget_state_to_ee(states[t], hand_side, clip_idx)
                    T_world_base, T_camera_base = self._compute_base_poses(cam_c2w, frame_id, hand_side)
                    T_base_ee = invert_T(T_world_base) @ T_world_ee
                    finger_pos = self._map_gripper(float(states[t][7]) if len(states[t]) > 7 else 0.0)

                    # Keep base pose consistent for IK FK and later render.
                    renderer.set_base_pose(T_camera_base, self.calibration.T_mjcam_from_cvcam)
                    if q_prev is None:
                        renderer.data.qpos[renderer.arm_qpos_addrs] = side_cal.q_reference
                        self.mujoco_forward(renderer)

                    q, ok, ik_metrics = self._solve_ik(T_base_ee, q_prev, hand_side)
                    if ok:
                        q_prev = q
                        hold_count = 0
                        render_ok = True
                        quality_flag = "ok"
                    elif (
                        self.ik_failure_policy == "hold_then_keep_original"
                        and q_prev is not None
                        and hold_count < self.max_hold_frames
                    ):
                        q = q_prev
                        hold_count += 1
                        render_ok = True
                        quality_flag = "ik_hold_last"
                    else:
                        render_ok = False
                        quality_flag = "ik_failed"

                    scene_depth, fov_y_deg, intrinsics = self._load_camera_inputs(clip_camera, frame_id, frame_img.shape)
                    if hand_idx is not None and hand_idx < len(joints_list):
                        hand_mask = self._get_hand_mask(joints_list[hand_idx], hand_data, frame_img.shape, intrinsics, hand_idx)
                    else:
                        hand_mask = np.zeros(frame_img.shape[:2], dtype=bool)

                    composite_meta: dict = {}
                    # Capture IK success before depth may rewrite quality_flag.
                    ik_ok = quality_flag == "ok"
                    if render_ok:
                        result, composite_meta = self._render_and_composite(
                            frame_img,
                            q,
                            finger_pos,
                            T_camera_base,
                            hand_mask,
                            scene_depth,
                            fov_y_deg,
                            hand_side,
                            depth_aligner=depth_aligner,
                        )
                        # Prefer depth-gate flag when IK already ok.
                        if quality_flag == "ok" and composite_meta.get("quality_flag") not in (None, "ok"):
                            quality_flag = str(composite_meta["quality_flag"])
                    else:
                        result = frame_img

                    out_path = self._build_output_path(sample, clip_idx, frame_id, hand_side, frame_path)
                    self._atomic_imwrite(out_path, result)
                    output_frames[clip_idx][frame_id] = out_path
                    quality_records.append(
                        {
                            "clip_idx": clip_idx,
                            "frame_id": frame_id,
                            "hand_side": hand_side,
                            "quality_flag": quality_flag,
                            "ik_ok": ik_ok,
                            "output_path": out_path,
                            "depth_invalid_ratio": composite_meta.get("depth_invalid_ratio"),
                            "depth_occlusion_used": bool(composite_meta.get("depth_occlusion_used")),
                            "depth_aligner_scale": float(depth_aligner.scale),
                            "depth_aligner_bias": float(depth_aligner.bias),
                            **ik_metrics,
                        }
                    )

        meta[self.output_frame_field] = output_frames
        meta[self.quality_field] = {
            "frames": quality_records,
            "summary": self._aggregate_quality(quality_records, self.hand_type),
        }
        return sample

    @staticmethod
    def mujoco_forward(renderer: RobotArmRenderer) -> None:
        renderer.mujoco.mj_forward(renderer.model, renderer.data)
