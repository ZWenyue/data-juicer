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
    T_world_base: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Map smoothed world-frame wrist state → T_world_ee.

    With ``side_cal.workspace_map`` and ``T_world_base`` set, positions go through the
    base-frame similarity map so targets land inside the arm's reachable shell. Falls
    back to the anchor-based map otherwise.
    """
    T_wrist = state_to_T(smoothed_state)
    p_wrist = T_wrist[:3, 3]
    R_wrist = T_wrist[:3, :3]

    if side_cal.workspace_map is not None and T_world_base is not None:
        ws = side_cal.workspace_map
        T_base_world = invert_T(np.asarray(T_world_base, dtype=np.float64))
        p_base_wrist = (T_base_world @ np.append(p_wrist, 1.0))[:3]
        p_base_ee = ws.center_robot_base + ws.scale_xyz * (p_base_wrist - ws.center_human_base)
        p_ee = (np.asarray(T_world_base, dtype=np.float64) @ np.append(p_base_ee, 1.0))[:3]
        R_ee = R_wrist @ np.asarray(side_cal.retarget_R, dtype=np.float64)
        return se3(R_ee, p_ee)

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


# MANO-21 layout as emitted by HaWoR: joint 0 is the wrist, then thumb, index, middle,
# ring and pinky chains of 4 joints each, ordered base→tip.
MANO_WRIST = 0
MANO_INDEX_MCP = 5
MANO_MIDDLE_MCP = 9
MANO_PINKY_MCP = 17

# The R1 grippers put both fingers at +x and separate them along ±y, so the site frame
# is (approach, jaw closing, palm normal) — the same axis roles as ``hand_grasp_frame``.
# Jaws are interchangeable, making a 180° turn about the approach axis a no-op.
JAW_SYMMETRY = np.diag([1.0, -1.0, -1.0])


def hand_grasp_frame(joints_cam: np.ndarray) -> Optional[np.ndarray]:
    """Palm frame in the joints' own frame: x = approach, y = jaw closing, z = x × y.

    Built from the wrist and knuckles only. Fingertip-based axes describe the grasp
    more literally, but they swing by tens of degrees as the fingers open and close,
    and a gripper bolted to the wrist cannot follow that; the knuckles are rigid with
    respect to the wrist, so the frame they span is the one a wrist-mounted gripper can
    actually track.

    Returns ``None`` unless a full MANO-21 joint set is available.
    """
    joints = np.asarray(joints_cam, dtype=np.float64)
    if joints.ndim != 2 or joints.shape[1] != 3 or joints.shape[0] <= MANO_PINKY_MCP:
        return None

    approach = joints[MANO_MIDDLE_MCP] - joints[MANO_WRIST]
    norm = float(np.linalg.norm(approach))
    if norm < 1e-9:
        return None
    approach = approach / norm

    # Across the palm, pinky knuckle → index knuckle, i.e. pointing to the thumb side
    # for either hand, which is the direction a human closes a grasp along.
    closing = joints[MANO_INDEX_MCP] - joints[MANO_PINKY_MCP]
    closing = closing - approach * float(approach @ closing)
    norm = float(np.linalg.norm(closing))
    if norm < 1e-9:
        return None
    closing = closing / norm

    return np.column_stack([approach, closing, np.cross(approach, closing)])


def grasp_orientation_error(R_ee: np.ndarray, R_grasp: np.ndarray) -> float:
    """Angle in radians between a gripper frame and a palm frame, modulo jaw symmetry."""
    R_ee = np.asarray(R_ee, dtype=np.float64)
    R_grasp = np.asarray(R_grasp, dtype=np.float64)
    best = -1.0
    for R_ref in (R_grasp, R_grasp @ JAW_SYMMETRY):
        best = max(best, float(np.trace(R_ee.T @ R_ref)))
    return float(np.arccos(np.clip(0.5 * (best - 1.0), -1.0, 1.0)))


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
