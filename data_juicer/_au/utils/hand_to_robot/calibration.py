# -*- coding: utf-8 -*-
"""Calibration YAML loader for hand→robot retarget."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml

from .transforms import quat_wxyz_to_mat, se3


@dataclass
class SideCalibration:
    q_reference: np.ndarray
    workspace_scale_xyz: np.ndarray
    axis_alignment: np.ndarray
    retarget_R: np.ndarray
    T_camera_base_ref: np.ndarray
    wrist_ref_world: Optional[np.ndarray] = None
    ee_ref_world: Optional[np.ndarray] = None
    velocity_limits: Optional[np.ndarray] = None
    model_sha256: Optional[str] = None


@dataclass
class HandToRobotCalibration:
    schema_version: int
    robot: str
    action_frame: str
    camera_convention: str
    T_mjcam_from_cvcam: np.ndarray
    sides: Dict[str, SideCalibration]
    path: Path
    version_name: str
    base_lowpass_tau: float = 0.3
    raw: Optional[dict] = None

    def get_side(self, side: str) -> SideCalibration:
        if side not in self.sides:
            raise KeyError(f"Calibration has no side '{side}'. Available: {list(self.sides)}")
        return self.sides[side]


def _pose_dict_to_T(pose: dict) -> np.ndarray:
    t = pose.get("translation_m") or pose.get("translation") or [0.0, 0.0, 0.0]
    q = pose.get("quaternion_wxyz") or [1.0, 0.0, 0.0, 0.0]
    return se3(quat_wxyz_to_mat(q), t)


def _parse_side(cfg: dict) -> SideCalibration:
    q_ref = np.asarray(cfg.get("q_reference", [0.0, 1.2, -1.5, 0.0, 0.0, 0.0]), dtype=np.float64)
    scale = np.asarray(cfg.get("workspace_scale_xyz", [1.0, 1.0, 1.0]), dtype=np.float64)
    axis = np.asarray(cfg.get("axis_alignment", np.eye(3).tolist()), dtype=np.float64)
    retarget_q = cfg.get("retarget_quaternion_wxyz", [1.0, 0.0, 0.0, 0.0])
    retarget_R = quat_wxyz_to_mat(retarget_q)
    cam_base = cfg.get("camera_to_base_reference", {})
    T_cam_base = _pose_dict_to_T(cam_base) if cam_base else se3(np.eye(3), [0.0, 0.3, -0.2])

    wrist_ref = cfg.get("wrist_ref_world")
    ee_ref = cfg.get("ee_ref_world")
    vel = cfg.get("velocity_limits")
    return SideCalibration(
        q_reference=q_ref,
        workspace_scale_xyz=scale,
        axis_alignment=axis,
        retarget_R=retarget_R,
        T_camera_base_ref=T_cam_base,
        wrist_ref_world=None if wrist_ref is None else np.asarray(wrist_ref, dtype=np.float64),
        ee_ref_world=None if ee_ref is None else np.asarray(ee_ref, dtype=np.float64),
        velocity_limits=None if vel is None else np.asarray(vel, dtype=np.float64),
        model_sha256=cfg.get("model_sha256"),
    )


def load_calibration(path: str | Path) -> HandToRobotCalibration:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f)

    adapter = raw.get("mujoco_camera_adapter", {})
    if adapter:
        T_adapter = _pose_dict_to_T(
            {
                "translation_m": adapter.get("translation", [0.0, 0.0, 0.0]),
                "quaternion_wxyz": adapter.get("quaternion_wxyz", [0.0, 1.0, 0.0, 0.0]),
            }
        )
    else:
        from .transforms import opencv_to_mujoco_camera_T

        T_adapter = opencv_to_mujoco_camera_T()

    sides = {name: _parse_side(cfg) for name, cfg in (raw.get("sides") or {}).items()}
    return HandToRobotCalibration(
        schema_version=int(raw.get("schema_version", 1)),
        robot=str(raw.get("robot", "r1_lite")),
        action_frame=str(raw.get("action_frame", "world")),
        camera_convention=str(raw.get("camera_convention", "opencv")),
        T_mjcam_from_cvcam=T_adapter,
        sides=sides,
        path=path,
        version_name=path.stem,
        base_lowpass_tau=float(raw.get("base_lowpass_tau", 0.3)),
        raw=raw,
    )


def side_to_dict(side: SideCalibration) -> dict:
    from .transforms import mat_to_quat_wxyz

    out = {
        "q_reference": [float(x) for x in side.q_reference.tolist()],
        "workspace_scale_xyz": [float(x) for x in side.workspace_scale_xyz.tolist()],
        "axis_alignment": [[float(x) for x in row] for row in side.axis_alignment.tolist()],
        "retarget_quaternion_wxyz": [float(x) for x in mat_to_quat_wxyz(side.retarget_R).tolist()],
        "camera_to_base_reference": {
            "translation_m": [float(x) for x in side.T_camera_base_ref[:3, 3].tolist()],
            "quaternion_wxyz": [float(x) for x in mat_to_quat_wxyz(side.T_camera_base_ref[:3, :3]).tolist()],
        },
    }
    if side.velocity_limits is not None:
        out["velocity_limits"] = [float(x) for x in side.velocity_limits.tolist()]
    if side.wrist_ref_world is not None:
        out["wrist_ref_world"] = [float(x) for x in side.wrist_ref_world.tolist()]
    if side.ee_ref_world is not None:
        out["ee_ref_world"] = [float(x) for x in side.ee_ref_world.tolist()]
    if side.model_sha256:
        out["model_sha256"] = side.model_sha256
    return out


def calibration_to_dict(cal: HandToRobotCalibration) -> dict:
    from .transforms import mat_to_quat_wxyz

    adapter_q = mat_to_quat_wxyz(cal.T_mjcam_from_cvcam[:3, :3])
    adapter_t = cal.T_mjcam_from_cvcam[:3, 3]
    return {
        "schema_version": int(cal.schema_version),
        "robot": cal.robot,
        "action_frame": cal.action_frame,
        "camera_convention": cal.camera_convention,
        "base_lowpass_tau": float(cal.base_lowpass_tau),
        "mujoco_camera_adapter": {
            "translation": [float(x) for x in adapter_t.tolist()],
            "quaternion_wxyz": [float(x) for x in adapter_q.tolist()],
        },
        "sides": {name: side_to_dict(side) for name, side in cal.sides.items()},
    }


def save_calibration(cal: HandToRobotCalibration, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = calibration_to_dict(cal)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, default_flow_style=False)
    return path


def replace_side(cal: HandToRobotCalibration, side_name: str, side: SideCalibration) -> HandToRobotCalibration:
    sides = dict(cal.sides)
    sides[side_name] = side
    return HandToRobotCalibration(
        schema_version=cal.schema_version,
        robot=cal.robot,
        action_frame=cal.action_frame,
        camera_convention=cal.camera_convention,
        T_mjcam_from_cvcam=cal.T_mjcam_from_cvcam,
        sides=sides,
        path=cal.path,
        version_name=cal.version_name,
        base_lowpass_tau=cal.base_lowpass_tau,
        raw=cal.raw,
    )
