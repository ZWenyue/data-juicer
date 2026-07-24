# -*- coding: utf-8 -*-
"""SE(3) helpers and camera convention adapters."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.spatial.transform import Rotation


def se3(R: np.ndarray, t: Sequence[float]) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64)
    T[:3, 3] = np.asarray(t, dtype=np.float64)
    return T


def invert_T(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def quat_wxyz_to_mat(quat_wxyz: Sequence[float]) -> np.ndarray:
    q = np.asarray(quat_wxyz, dtype=np.float64)
    # scipy uses xyzw
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def mat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    q_xyzw = Rotation.from_matrix(R).as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64)


def state_to_T(state: Sequence[float]) -> np.ndarray:
    """Convert 8-dim state [x,y,z,roll,pitch,yaw,pad,gripper] to SE(3)."""
    s = np.asarray(state, dtype=np.float64)
    R = Rotation.from_euler("xyz", s[3:6], degrees=False).as_matrix()
    return se3(R, s[:3])


def compute_mujoco_fovy(fov_x_rad: float, img_width: int, img_height: int) -> float:
    """Convert horizontal FOV (rad) to MuJoCo vertical FOV (degrees)."""
    f_pixel = 0.5 * img_width / np.tan(0.5 * fov_x_rad)
    fov_y_rad = 2.0 * np.arctan(0.5 * img_height / f_pixel)
    return float(np.degrees(fov_y_rad))


def opencv_to_mujoco_camera_T() -> np.ndarray:
    """Fixed adapter: OpenCV cam (+Z forward, +Y down) → MuJoCo cam (−Z look).

    Equivalent to 180° rotation about X: (x,y,z)_cv → (x,−y,−z)_mj.
    """
    R = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64)
    return se3(R, [0.0, 0.0, 0.0])
