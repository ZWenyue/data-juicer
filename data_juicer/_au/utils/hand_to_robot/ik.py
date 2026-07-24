# -*- coding: utf-8 -*-
"""Jacobian DLS IK and gripper mapping for R1 Lite arm."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def map_gripper_to_finger(gripper_state: float, max_open: float = 0.05) -> float:
    """Map gripper ∈ [-1, 1] (1=open) to finger1 displacement ∈ [0, max_open]."""
    g = float(np.clip(gripper_state, -1.0, 1.0))
    return (g + 1.0) * 0.5 * max_open


def _orientation_error(R_target: np.ndarray, R_current: np.ndarray) -> np.ndarray:
    R_err = R_target @ R_current.T
    angle = np.arccos(np.clip((np.trace(R_err) - 1.0) * 0.5, -1.0, 1.0))
    if angle < 1e-8:
        return np.zeros(3, dtype=np.float64)
    axis = np.array(
        [
            R_err[2, 1] - R_err[1, 2],
            R_err[0, 2] - R_err[2, 0],
            R_err[1, 0] - R_err[0, 1],
        ],
        dtype=np.float64,
    ) / (2.0 * np.sin(angle))
    return axis * angle


def jacobian_ik(
    model,
    data,
    site_id: int,
    target_pos: np.ndarray,
    target_rot: np.ndarray,
    arm_qpos_addrs: np.ndarray,
    q_init: Optional[np.ndarray] = None,
    max_iter: int = 100,
    tol_pos: float = 5e-3,
    tol_rot: float = 0.0524,
    damping: float = 1e-2,
    step: float = 0.5,
) -> Tuple[np.ndarray, bool, dict]:
    """Damped least-squares IK for the 6-DoF arm using a MuJoCo site.

    Args:
        model/data: MuJoCo model/data with current base mocap already set.
        site_id: gripper site id.
        target_pos/target_rot: EE target in the same world frame as MuJoCo.
        arm_qpos_addrs: length-6 qpos addresses for arm joints.
        q_init: optional warm-start arm angles.

    Returns:
        q_arm (6,), success, metrics dict.
    """
    import mujoco

    if q_init is not None:
        data.qpos[arm_qpos_addrs] = np.asarray(q_init, dtype=np.float64)

    jacp = np.zeros((3, model.nv), dtype=np.float64)
    jacr = np.zeros((3, model.nv), dtype=np.float64)
    pos_err = np.inf
    rot_err = np.inf

    # Map qpos addr -> dof address for Jacobian columns.
    dof_addrs = []
    for qadr in arm_qpos_addrs:
        # Find joint whose qposadr matches.
        jnt_id = None
        for j in range(model.njnt):
            if model.jnt_qposadr[j] == qadr:
                jnt_id = j
                break
        if jnt_id is None:
            raise ValueError(f"No joint found for qpos address {qadr}")
        dof_addrs.append(int(model.jnt_dofadr[jnt_id]))
    dof_addrs = np.asarray(dof_addrs, dtype=np.int32)

    for _ in range(max_iter):
        mujoco.mj_forward(model, data)
        cur_pos = data.site_xpos[site_id].copy()
        cur_rot = data.site_xmat[site_id].reshape(3, 3).copy()
        err_pos = target_pos - cur_pos
        err_rot = _orientation_error(target_rot, cur_rot)
        pos_err = float(np.linalg.norm(err_pos))
        rot_err = float(np.linalg.norm(err_rot))
        if pos_err < tol_pos and rot_err < tol_rot:
            q = data.qpos[arm_qpos_addrs].copy()
            return q, True, {
                "position_error_m": pos_err,
                "orientation_error_rad": rot_err,
                "workspace_projected": False,
            }

        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        J = np.vstack([jacp[:, dof_addrs], jacr[:, dof_addrs]])  # (6, 6)
        err = np.concatenate([err_pos, err_rot])
        JJT = J @ J.T + (damping**2) * np.eye(6)
        dq = J.T @ np.linalg.solve(JJT, err)

        q_new = data.qpos[arm_qpos_addrs] + step * dq
        # Clip by joint limits.
        for i, qadr in enumerate(arm_qpos_addrs):
            jnt_id = None
            for j in range(model.njnt):
                if model.jnt_qposadr[j] == qadr:
                    jnt_id = j
                    break
            if jnt_id is not None and model.jnt_limited[jnt_id]:
                lo, hi = model.jnt_range[jnt_id]
                q_new[i] = np.clip(q_new[i], lo, hi)
        data.qpos[arm_qpos_addrs] = q_new

    q = data.qpos[arm_qpos_addrs].copy()
    return q, False, {
        "position_error_m": float(pos_err),
        "orientation_error_rad": float(rot_err),
        "workspace_projected": False,
    }
