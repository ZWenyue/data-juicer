# -*- coding: utf-8 -*-
"""MuJoCo offscreen renderer for a single R1 Lite arm."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .transforms import mat_to_quat_wxyz

VISUAL_GEOM_GROUP = 1


class RobotArmRenderer:
    """Lazy-friendly wrapper around MjModel/MjData/Renderer for one arm."""

    def __init__(self, model_path: str | Path, width: int = 640, height: int = 480, gl_backend: str = "egl"):
        os.environ.setdefault("MUJOCO_GL", gl_backend)
        import mujoco

        self.mujoco = mujoco
        self.model_path = str(model_path)
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)
        self.width = width
        self.height = height
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)

        self.cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "ego_cam")
        self.site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "gripper_site")
        if self.cam_id < 0 or self.site_id < 0:
            raise RuntimeError(f"ego_cam/gripper_site missing in {self.model_path}")

        anchor_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot_anchor")
        if anchor_body < 0:
            raise RuntimeError(f"robot_anchor missing in {self.model_path}")
        self.anchor_mocap_id = int(self.model.body_mocapid[anchor_body])
        if self.anchor_mocap_id < 0:
            raise RuntimeError("robot_anchor is not a mocap body")

        self.robot_visual_geom_ids = np.flatnonzero(self.model.geom_group == VISUAL_GEOM_GROUP)

        # Arm joints: first 6 hinge qpos addresses in model order.
        arm_addrs = []
        finger_addrs = []
        for j in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
            qadr = int(self.model.jnt_qposadr[j])
            if "finger" in name:
                finger_addrs.append(qadr)
            elif "arm_joint" in name:
                arm_addrs.append(qadr)
        if len(arm_addrs) != 6:
            raise RuntimeError(f"Expected 6 arm joints, found {len(arm_addrs)}")
        self.arm_qpos_addrs = np.asarray(arm_addrs, dtype=np.int32)
        self.finger_qpos_addrs = np.asarray(finger_addrs, dtype=np.int32)

    def set_camera_fov(self, fov_y_deg: float) -> None:
        self.model.cam_fovy[self.cam_id] = float(fov_y_deg)

    def set_base_pose(self, T_camera_base: np.ndarray, T_mjcam_from_cvcam: np.ndarray) -> None:
        T_mj_base = T_mjcam_from_cvcam @ T_camera_base
        self.data.mocap_pos[self.anchor_mocap_id] = T_mj_base[:3, 3]
        self.data.mocap_quat[self.anchor_mocap_id] = mat_to_quat_wxyz(T_mj_base[:3, :3])

    def set_arm_qpos(self, joint_angles: np.ndarray, finger_pos: float) -> None:
        self.data.qpos[self.arm_qpos_addrs] = np.asarray(joint_angles, dtype=np.float64)
        if len(self.finger_qpos_addrs) >= 2:
            self.data.qpos[self.finger_qpos_addrs[0]] = float(finger_pos)
            self.data.qpos[self.finger_qpos_addrs[1]] = -float(finger_pos)

    def render_frame(
        self,
        joint_angles: np.ndarray,
        finger_pos: float,
        T_camera_base: np.ndarray,
        T_mjcam_from_cvcam: np.ndarray,
        render_depth: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        self.set_base_pose(T_camera_base, T_mjcam_from_cvcam)
        self.set_arm_qpos(joint_angles, finger_pos)
        self.mujoco.mj_forward(self.model, self.data)

        self.renderer.update_scene(self.data, camera="ego_cam")
        rgb = self.renderer.render().copy()

        self.renderer.enable_segmentation_rendering()
        self.renderer.update_scene(self.data, camera="ego_cam")
        seg = self.renderer.render()
        self.renderer.disable_segmentation_rendering()
        mask = np.isin(seg[:, :, 0], self.robot_visual_geom_ids)

        depth = None
        if render_depth:
            self.renderer.enable_depth_rendering()
            self.renderer.update_scene(self.data, camera="ego_cam")
            depth = self.renderer.render().copy()
            self.renderer.disable_depth_rendering()
        return rgb, mask, depth

    def site_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        pos = self.data.site_xpos[self.site_id].copy()
        rot = self.data.site_xmat[self.site_id].reshape(3, 3).copy()
        return pos, rot

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
