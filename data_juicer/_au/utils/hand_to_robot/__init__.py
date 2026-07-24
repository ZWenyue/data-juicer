# -*- coding: utf-8 -*-
"""Hand → robot render helpers (transforms, IK, compositing, MuJoCo renderer)."""

from .calibration import HandToRobotCalibration, SideCalibration, load_calibration, save_calibration
from .composite import alpha_blend, bbox_to_mask, composite_robot_on_frame, project_joints_mask
from .ik import jacobian_ik, map_gripper_to_finger
from .renderer import RobotArmRenderer
from .retarget import palm_pixel_from_joints, project_point_cam, retarget_wrist_to_ee, world_to_camera
from .transforms import (
    compute_mujoco_fovy,
    invert_T,
    mat_to_quat_wxyz,
    quat_wxyz_to_mat,
    se3,
    state_to_T,
)

__all__ = [
    "HandToRobotCalibration",
    "SideCalibration",
    "RobotArmRenderer",
    "alpha_blend",
    "bbox_to_mask",
    "composite_robot_on_frame",
    "compute_mujoco_fovy",
    "invert_T",
    "jacobian_ik",
    "load_calibration",
    "map_gripper_to_finger",
    "mat_to_quat_wxyz",
    "palm_pixel_from_joints",
    "project_joints_mask",
    "project_point_cam",
    "quat_wxyz_to_mat",
    "retarget_wrist_to_ee",
    "save_calibration",
    "se3",
    "state_to_T",
    "world_to_camera",
]
