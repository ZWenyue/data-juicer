# -*- coding: utf-8 -*-
"""Pure retarget helpers shared by Mapper and calibration tools."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from .calibration import SideCalibration
from .transforms import invert_T, se3, state_to_T


def retarget_wrist_to_ee(
    smoothed_state: Sequence[float],
    side_cal: SideCalibration,
    wrist_ref_world: Optional[np.ndarray] = None,
    ee_ref_world: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Map smoothed world-frame wrist state → T_world_ee."""
    T_wrist = state_to_T(smoothed_state)
    p_wrist = T_wrist[:3, 3]
    R_wrist = T_wrist[:3, :3]

    if wrist_ref_world is None:
        wrist_ref_world = side_cal.wrist_ref_world
    if wrist_ref_world is None:
        wrist_ref_world = p_wrist

    if ee_ref_world is None:
        ee_ref_world = side_cal.ee_ref_world
    if ee_ref_world is None:
        ee_ref_world = np.asarray(wrist_ref_world, dtype=np.float64)

    S = np.diag(np.asarray(side_cal.workspace_scale_xyz, dtype=np.float64))
    A = np.asarray(side_cal.axis_alignment, dtype=np.float64)
    p_ee = np.asarray(ee_ref_world, dtype=np.float64) + S @ A @ (
        p_wrist - np.asarray(wrist_ref_world, dtype=np.float64)
    )
    R_ee = R_wrist @ np.asarray(side_cal.retarget_R, dtype=np.float64)
    return se3(R_ee, p_ee)


def world_to_camera(T_world_x: np.ndarray, T_world_camera: np.ndarray) -> np.ndarray:
    return invert_T(T_world_camera) @ T_world_x


def project_point_cam(
    p_cam: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> Tuple[float, float, bool]:
    """Project a camera-frame 3D point to pixels. Returns (u, v, valid)."""
    z = float(p_cam[2])
    if not np.isfinite(z) or z <= 1e-6:
        return float("nan"), float("nan"), False
    u = fx * float(p_cam[0]) / z + cx
    v = fy * float(p_cam[1]) / z + cy
    return u, v, True


def palm_pixel_from_joints(
    joints_cam: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> Tuple[float, float, bool]:
    """Use MANO wrist (joint 0) as palm/wrist pixel target."""
    joints = np.asarray(joints_cam, dtype=np.float64)
    if joints.ndim != 2 or joints.shape[0] < 1:
        return float("nan"), float("nan"), False
    return project_point_cam(joints[0], fx, fy, cx, cy)
