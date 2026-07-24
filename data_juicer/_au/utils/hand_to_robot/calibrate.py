# -*- coding: utf-8 -*-
"""Offline calibration for hand→robot retarget parameters.

Optimizes workspace scale, retarget rotation, and camera→base translation
against smoothed world-frame wrist states + camera poses from the ego
pipeline. Optionally includes IK residual when a MuJoCo arm model is provided.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from .calibration import (
    HandToRobotCalibration,
    SideCalibration,
    load_calibration,
    replace_side,
    save_calibration,
    side_to_dict,
)
from .retarget import palm_pixel_from_joints, project_point_cam, retarget_wrist_to_ee, world_to_camera
from .transforms import invert_T, mat_to_quat_wxyz, quat_wxyz_to_mat, se3, state_to_T


@dataclass
class CalibFrame:
    frame_id: int
    state: np.ndarray  # (8,)
    T_world_camera: np.ndarray  # (4,4)
    joints_cam: Optional[np.ndarray] = None  # (21,3)
    fx: float = 500.0
    fy: float = 500.0
    cx: float = 320.0
    cy: float = 240.0
    img_w: int = 640
    img_h: int = 480
    q_gt: Optional[np.ndarray] = None  # optional GT arm joints for IK warm-start


@dataclass
class CalibClip:
    side: str
    frames: List[CalibFrame]
    source: str = ""
    wrist_ref_world: Optional[np.ndarray] = None
    ee_ref_world: Optional[np.ndarray] = None


@dataclass
class CalibWeights:
    w_pos: float = 1.0
    w_rot: float = 0.25
    w_uv: float = 2e-4  # px^2 scaled; ~1.0 when err~70px
    w_ik: float = 1.0
    w_scale_reg: float = 0.05
    w_base_reg: float = 0.1


@dataclass
class CalibResult:
    calibration: HandToRobotCalibration
    side: str
    metrics_before: dict
    metrics_after: dict
    success: bool
    message: str = ""
    x_opt: Optional[np.ndarray] = None


def _orthonormalize(R: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(R)
    Rn = U @ Vt
    if np.linalg.det(Rn) < 0:
        U[:, -1] *= -1
        Rn = U @ Vt
    return Rn


def select_anchor_indices(frames: Sequence[CalibFrame], n_anchors: int) -> List[int]:
    """Pick spatially diverse wrist positions (plus endpoints)."""
    n = len(frames)
    if n == 0:
        return []
    if n_anchors >= n:
        return list(range(n))
    positions = np.stack([f.state[:3] for f in frames], axis=0)
    chosen = [0, n - 1]
    while len(chosen) < n_anchors:
        dists = []
        for i in range(n):
            if i in chosen:
                dists.append(-1.0)
                continue
            d = min(np.linalg.norm(positions[i] - positions[j]) for j in chosen)
            dists.append(d)
        chosen.append(int(np.argmax(dists)))
    return sorted(set(chosen))


def _pack_params(
    side: SideCalibration,
    optimize_base_orient: bool = False,
    optimize_axis: bool = False,
) -> Tuple[np.ndarray, dict]:
    meta = {
        "optimize_base_orient": optimize_base_orient,
        "optimize_axis": optimize_axis,
    }
    scale = np.clip(side.workspace_scale_xyz.copy(), 0.15, 2.0)
    retarget_rv = Rotation.from_matrix(side.retarget_R).as_rotvec()
    base_t = side.T_camera_base_ref[:3, 3].copy()
    parts = [np.log(scale), retarget_rv, base_t]
    if optimize_base_orient:
        parts.append(Rotation.from_matrix(side.T_camera_base_ref[:3, :3]).as_rotvec())
    if optimize_axis:
        parts.append(Rotation.from_matrix(side.axis_alignment).as_rotvec())
    return np.concatenate(parts), meta


def _unpack_params(
    x: np.ndarray,
    base_side: SideCalibration,
    meta: dict,
) -> SideCalibration:
    i = 0
    log_s = x[i : i + 3]
    i += 3
    retarget_rv = x[i : i + 3]
    i += 3
    base_t = x[i : i + 3]
    i += 3
    if meta.get("optimize_base_orient"):
        base_R = Rotation.from_rotvec(x[i : i + 3]).as_matrix()
        i += 3
    else:
        base_R = base_side.T_camera_base_ref[:3, :3]
    if meta.get("optimize_axis"):
        axis = _orthonormalize(Rotation.from_rotvec(x[i : i + 3]).as_matrix())
    else:
        axis = base_side.axis_alignment

    scale = np.clip(np.exp(log_s), 0.15, 2.0)
    retarget_R = Rotation.from_rotvec(retarget_rv).as_matrix()
    T_cam_base = se3(base_R, base_t)
    return SideCalibration(
        q_reference=base_side.q_reference.copy(),
        workspace_scale_xyz=scale,
        axis_alignment=np.asarray(axis, dtype=np.float64),
        retarget_R=retarget_R,
        T_camera_base_ref=T_cam_base,
        wrist_ref_world=None if base_side.wrist_ref_world is None else base_side.wrist_ref_world.copy(),
        ee_ref_world=None if base_side.ee_ref_world is None else base_side.ee_ref_world.copy(),
        velocity_limits=None if base_side.velocity_limits is None else base_side.velocity_limits.copy(),
        model_sha256=base_side.model_sha256,
    )


def evaluate_clip(
    clip: CalibClip,
    side_cal: SideCalibration,
    weights: CalibWeights = CalibWeights(),
    renderer=None,
    T_mjcam_from_cvcam: Optional[np.ndarray] = None,
    q_init: Optional[np.ndarray] = None,
) -> dict:
    """Evaluate retarget (+ optional IK) metrics on a clip."""
    from .ik import jacobian_ik

    wrist_ref = clip.wrist_ref_world
    if wrist_ref is None:
        wrist_ref = side_cal.wrist_ref_world
    if wrist_ref is None and clip.frames:
        wrist_ref = clip.frames[0].state[:3].copy()
    ee_ref = clip.ee_ref_world if clip.ee_ref_world is not None else side_cal.ee_ref_world
    if ee_ref is None:
        ee_ref = wrist_ref

    pos_errs = []
    rot_errs = []
    uv_errs = []
    ik_ok = []
    ik_pos = []
    q_prev = q_init if q_init is not None else side_cal.q_reference.copy()

    for fr in clip.frames:
        T_world_ee = retarget_wrist_to_ee(fr.state, side_cal, wrist_ref, ee_ref)
        T_world_camera = fr.T_world_camera
        T_world_base = T_world_camera @ side_cal.T_camera_base_ref
        T_base_ee = invert_T(T_world_base) @ T_world_ee
        T_camera_ee = world_to_camera(T_world_ee, T_world_camera)

        # Reachability proxy in robot-base frame: EE should stay within arm span.
        p_base = T_base_ee[:3, 3]
        reach = float(np.linalg.norm(p_base))
        # Soft hinge around ~0.2–0.75 m workspace shell.
        if reach < 0.15:
            pos_errs.append(0.15 - reach)
        elif reach > 0.80:
            pos_errs.append(reach - 0.80)
        else:
            pos_errs.append(0.0)

        # Orientation: prefer retarget not flipping gripper wildly vs wrist.
        T_camera_wrist = world_to_camera(state_to_T(fr.state), T_world_camera)
        R_err = T_camera_ee[:3, :3] @ T_camera_wrist[:3, :3].T
        rot_errs.append(float(np.linalg.norm(Rotation.from_matrix(_orthonormalize(R_err)).as_rotvec())))

        if fr.joints_cam is not None:
            u_t, v_t, ok_t = palm_pixel_from_joints(fr.joints_cam, fr.fx, fr.fy, fr.cx, fr.cy)
            u_p, v_p, ok_p = project_point_cam(T_camera_ee[:3, 3], fr.fx, fr.fy, fr.cx, fr.cy)
            if ok_t and ok_p:
                uv_errs.append(float(np.hypot(u_p - u_t, v_p - v_t)))

        if renderer is not None and T_mjcam_from_cvcam is not None:
            T_camera_base = invert_T(T_world_camera) @ T_world_base
            renderer.set_base_pose(T_camera_base, T_mjcam_from_cvcam)
            T_mj_base = np.eye(4, dtype=np.float64)
            T_mj_base[:3, 3] = renderer.data.mocap_pos[renderer.anchor_mocap_id]
            T_mj_base[:3, :3] = quat_wxyz_to_mat(renderer.data.mocap_quat[renderer.anchor_mocap_id])
            T_mj_ee = T_mj_base @ T_base_ee
            q_warm = np.asarray(fr.q_gt, dtype=np.float64) if fr.q_gt is not None else q_prev
            q, ok, metrics = jacobian_ik(
                renderer.model,
                renderer.data,
                renderer.site_id,
                T_mj_ee[:3, 3],
                T_mj_ee[:3, :3],
                renderer.arm_qpos_addrs,
                q_init=q_warm,
                max_iter=80,
                tol_pos=5e-3,
                tol_rot=0.0524,
            )
            ik_ok.append(bool(ok))
            ik_pos.append(float(metrics["position_error_m"]))
            if ok:
                q_prev = q
                # If IK ok, also measure FK site reprojection vs palm.
                renderer.set_arm_qpos(q, 0.02)
                renderer.mujoco.mj_forward(renderer.model, renderer.data)
                site_pos_mj, _ = renderer.site_pose()
                # site is in MuJoCo camera world; convert to OpenCV cam for projection.
                p_h = np.array([site_pos_mj[0], site_pos_mj[1], site_pos_mj[2], 1.0])
                p_cv = invert_T(T_mjcam_from_cvcam) @ p_h
                if fr.joints_cam is not None:
                    u_t, v_t, ok_t = palm_pixel_from_joints(fr.joints_cam, fr.fx, fr.fy, fr.cx, fr.cy)
                    u_s, v_s, ok_s = project_point_cam(p_cv[:3], fr.fx, fr.fy, fr.cx, fr.cy)
                    if ok_t and ok_s:
                        uv_errs.append(float(np.hypot(u_s - u_t, v_s - v_t)))

    def _med(xs):
        return float(np.median(xs)) if xs else float("nan")

    out = {
        "num_frames": len(clip.frames),
        "median_reach_violation_m": _med(pos_errs),
        "median_cam_rot_err_rad": _med(rot_errs),
        "median_reprojection_error_px": _med(uv_errs),
        "mean_reprojection_error_px": float(np.mean(uv_errs)) if uv_errs else float("nan"),
        "ik_success_rate": float(np.mean(ik_ok)) if ik_ok else float("nan"),
        "median_ik_position_error_m": _med(ik_pos),
        "loss": 0.0,
    }
    loss = 0.0
    if pos_errs:
        loss += weights.w_pos * float(np.mean(np.square(pos_errs)))
    if rot_errs:
        loss += weights.w_rot * float(np.mean(np.square(rot_errs)))
    if uv_errs:
        loss += weights.w_uv * float(np.mean(np.square(uv_errs)))
    if ik_pos:
        fail = 1.0 - float(np.mean(ik_ok))
        loss += weights.w_ik * (float(np.mean(np.square(ik_pos))) + 0.25 * fail)
    loss += weights.w_scale_reg * float(np.sum((side_cal.workspace_scale_xyz - 1.0) ** 2))
    out["loss"] = float(loss)
    return out


def optimize_side_calibration(
    clip: CalibClip,
    init_cal: HandToRobotCalibration,
    weights: CalibWeights = CalibWeights(),
    n_anchors: int = 8,
    optimize_base_orient: bool = False,
    optimize_axis: bool = False,
    model_path: Optional[str] = None,
    maxiter: int = 80,
) -> CalibResult:
    """Fit side calibration on a clip; returns updated HandToRobotCalibration."""
    if clip.side not in init_cal.sides:
        raise KeyError(f"init calibration missing side {clip.side}")
    if not clip.frames:
        raise ValueError("empty calibration clip")

    base_side = init_cal.get_side(clip.side)
    # Freeze refs from clip first frame unless already set.
    wrist_ref = clip.wrist_ref_world
    if wrist_ref is None:
        wrist_ref = clip.frames[0].state[:3].copy()
    ee_ref = clip.ee_ref_world if clip.ee_ref_world is not None else wrist_ref.copy()
    base_side = SideCalibration(
        q_reference=base_side.q_reference.copy(),
        workspace_scale_xyz=base_side.workspace_scale_xyz.copy(),
        axis_alignment=base_side.axis_alignment.copy(),
        retarget_R=base_side.retarget_R.copy(),
        T_camera_base_ref=base_side.T_camera_base_ref.copy(),
        wrist_ref_world=np.asarray(wrist_ref, dtype=np.float64),
        ee_ref_world=np.asarray(ee_ref, dtype=np.float64),
        velocity_limits=base_side.velocity_limits,
        model_sha256=base_side.model_sha256,
    )
    clip = CalibClip(
        side=clip.side,
        frames=clip.frames,
        source=clip.source,
        wrist_ref_world=base_side.wrist_ref_world,
        ee_ref_world=base_side.ee_ref_world,
    )

    anchor_ids = select_anchor_indices(clip.frames, n_anchors)
    anchor_clip = CalibClip(
        side=clip.side,
        frames=[clip.frames[i] for i in anchor_ids],
        source=clip.source,
        wrist_ref_world=clip.wrist_ref_world,
        ee_ref_world=clip.ee_ref_world,
    )

    renderer = None
    if model_path:
        from .renderer import RobotArmRenderer

        # Use first frame size for renderer.
        fr0 = clip.frames[0]
        renderer = RobotArmRenderer(model_path, width=fr0.img_w, height=fr0.img_h)

    x0, meta = _pack_params(base_side, optimize_base_orient, optimize_axis)
    base_t0 = base_side.T_camera_base_ref[:3, 3].copy()

    metrics_before = evaluate_clip(
        clip,
        base_side,
        weights=weights,
        renderer=renderer,
        T_mjcam_from_cvcam=init_cal.T_mjcam_from_cvcam,
        q_init=base_side.q_reference,
    )

    def objective(x):
        side = _unpack_params(x, base_side, meta)
        # Prior on base translation.
        prior = weights.w_base_reg * float(np.sum((side.T_camera_base_ref[:3, 3] - base_t0) ** 2))
        stats = evaluate_clip(
            anchor_clip,
            side,
            weights=weights,
            renderer=renderer,
            T_mjcam_from_cvcam=init_cal.T_mjcam_from_cvcam,
            q_init=base_side.q_reference,
        )
        return stats["loss"] + prior

    opt = minimize(objective, x0, method="L-BFGS-B", options={"maxiter": maxiter, "ftol": 1e-6})
    side_opt = _unpack_params(opt.x, base_side, meta)
    metrics_after = evaluate_clip(
        clip,
        side_opt,
        weights=weights,
        renderer=renderer,
        T_mjcam_from_cvcam=init_cal.T_mjcam_from_cvcam,
        q_init=side_opt.q_reference,
    )
    if renderer is not None:
        renderer.close()

    new_cal = replace_side(init_cal, clip.side, side_opt)
    improved = metrics_after["loss"] <= metrics_before["loss"] + 1e-9
    return CalibResult(
        calibration=new_cal,
        side=clip.side,
        metrics_before=metrics_before,
        metrics_after=metrics_after,
        success=bool(opt.success or improved),
        message=str(opt.message),
        x_opt=opt.x.copy(),
    )


def load_pipeline_sample(path: str | Path, sample_idx: int = 0) -> dict:
    """Load one sample from json/jsonl/pkl/parquet."""
    import json
    import pickle

    path = Path(path)
    if path.suffix == ".pkl":
        with path.open("rb") as f:
            samples = pickle.load(f)
    elif path.suffix == ".json":
        samples = json.loads(path.read_text())
        if isinstance(samples, dict):
            samples = [samples]
    elif path.suffix == ".jsonl":
        samples = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    elif path.suffix == ".parquet":
        import pyarrow.parquet as pq

        samples = pq.read_table(path).to_pylist()
    else:
        raise ValueError(f"Unsupported data path: {path}")
    if sample_idx < 0 or sample_idx >= len(samples):
        raise IndexError(f"sample_idx {sample_idx} out of range (n={len(samples)})")
    return samples[sample_idx]


def clip_from_pipeline_sample(
    sample: dict,
    side: str = "right",
    video_idx: int = 0,
    max_frames: Optional[int] = None,
    stride: int = 1,
) -> CalibClip:
    """Build a CalibClip from an ego pipeline sample dict."""
    from data_juicer.utils.constant import Fields, MetaKeys
    from data_juicer.utils.file_utils import load_numpy

    meta = sample.get(Fields.meta) or sample.get("__dj__meta__") or {}
    if isinstance(meta, (bytes, bytearray)):
        import pickle

        meta = pickle.loads(meta)

    actions_all = meta.get(MetaKeys.hand_action_tags) or []
    hawor_all = meta.get(MetaKeys.hand_reconstruction_hawor_tags) or []
    cam_all = meta.get(MetaKeys.video_camera_pose_tags) or []
    frames_all = meta.get(MetaKeys.video_frames) or sample.get(MetaKeys.video_frames) or []
    calib_all = meta.get(MetaKeys.camera_calibration_moge_tags) or []

    if video_idx >= len(actions_all) or video_idx >= len(cam_all):
        raise IndexError("video_idx exceeds available clips in sample meta")

    action_tags = actions_all[video_idx]
    if "states" in action_tags:
        action_tags = {action_tags.get("hand_type", side): action_tags}
    hand_action = action_tags.get(side) or {}
    states = hand_action.get("states") or []
    valid_ids = hand_action.get("valid_frame_ids") or []
    joints_list = hand_action.get("joints_cam") or []
    if not states or not valid_ids or len(states) != len(valid_ids):
        raise ValueError(f"No usable {side} states/valid_frame_ids in sample")

    cam_pose = cam_all[video_idx]
    cam_c2w = np.asarray(load_numpy(cam_pose["cam_c2w"]), dtype=np.float64)
    hawor = hawor_all[video_idx] if video_idx < len(hawor_all) else {}
    hand_hawor = (hawor.get(side) or {}) if isinstance(hawor, dict) else {}
    hawor_frame_ids = hand_hawor.get("frame_ids") or []
    hawor_joints = hand_hawor.get("joints_cam") or joints_list
    hawor_index = {int(fid): i for i, fid in enumerate(hawor_frame_ids)}

    # Intrinsics / size
    img_w, img_h = 640, 480
    if video_idx < len(frames_all) and frames_all[video_idx]:
        import cv2

        img = cv2.imread(str(frames_all[video_idx][0]))
        if img is not None:
            img_h, img_w = img.shape[:2]
    fx = fy = 0.5 * img_w / np.tan(np.deg2rad(35.0))
    cx, cy = img_w * 0.5, img_h * 0.5
    if video_idx < len(calib_all) and isinstance(calib_all[video_idx], dict):
        from data_juicer.utils.constant import CameraCalibrationKeys

        Ks = calib_all[video_idx].get(CameraCalibrationKeys.intrinsics)
        if isinstance(Ks, (list, tuple)) and Ks:
            K = np.asarray(load_numpy(Ks[0]), dtype=np.float64)
            if K.shape == (3, 3):
                fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    elif isinstance(hawor, dict) and "fov_x" in hawor:
        fov_x = float(hawor["fov_x"])
        if fov_x > np.pi:
            fov_x = np.deg2rad(fov_x)
        fx = fy = 0.5 * img_w / np.tan(0.5 * fov_x)

    frames: List[CalibFrame] = []
    for t, fid in enumerate(valid_ids):
        if stride > 1 and (t % stride) != 0:
            continue
        fid = int(fid)
        if fid < 0 or fid >= len(cam_c2w):
            continue
        joints = None
        if t < len(joints_list):
            joints = np.asarray(joints_list[t], dtype=np.float64)
        elif fid in hawor_index and hawor_index[fid] < len(hawor_joints):
            joints = np.asarray(hawor_joints[hawor_index[fid]], dtype=np.float64)
        frames.append(
            CalibFrame(
                frame_id=fid,
                state=np.asarray(states[t], dtype=np.float64),
                T_world_camera=np.asarray(cam_c2w[fid], dtype=np.float64),
                joints_cam=joints,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                img_w=img_w,
                img_h=img_h,
            )
        )
        if max_frames is not None and len(frames) >= max_frames:
            break

    if not frames:
        raise ValueError("No calibration frames extracted from sample")
    return CalibClip(side=side, frames=frames, source=str(sample.get("id", "sample")))


def make_synthetic_clip(
    side: str = "right",
    n_frames: int = 12,
    img_w: int = 320,
    img_h: int = 240,
) -> CalibClip:
    """Synthetic reachable-ish clip for unit tests / smoke calibration."""
    fx = fy = 0.5 * img_w / np.tan(np.deg2rad(35.0))
    cx, cy = img_w * 0.5, img_h * 0.5
    frames = []
    for i in range(n_frames):
        # Wrist moves gently in front of camera (OpenCV cam / world-aligned).
        x = 0.20 + 0.01 * i
        y = (0.05 if side == "right" else -0.05) + 0.002 * np.sin(i)
        z = 0.40 + 0.005 * np.cos(i)
        state = np.array([x, y, z, 0.0, 0.15, 0.0, 0.0, 1.0 - 0.05 * i], dtype=np.float64)
        T_c2w = np.eye(4, dtype=np.float64)
        joints = np.array(
            [
                [x, y, z],
                [x + 0.02, y + 0.01, z],
                [x + 0.02, y - 0.01, z],
                [x + 0.04, y, z],
            ],
            dtype=np.float64,
        )
        frames.append(
            CalibFrame(
                frame_id=i,
                state=state,
                T_world_camera=T_c2w,
                joints_cam=joints,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                img_w=img_w,
                img_h=img_h,
            )
        )
    return CalibClip(side=side, frames=frames, source="synthetic")


def _T_to_state8(T: np.ndarray, gripper: float = 0.0) -> np.ndarray:
    """SE(3) → [x,y,z,roll,pitch,yaw,pad,gripper] (xyz Euler)."""
    R = np.asarray(T[:3, :3], dtype=np.float64)
    t = np.asarray(T[:3, 3], dtype=np.float64)
    rpy = Rotation.from_matrix(_orthonormalize(R)).as_euler("xyz", degrees=False)
    return np.array(
        [t[0], t[1], t[2], rpy[0], rpy[1], rpy[2], 0.0, float(gripper)],
        dtype=np.float64,
    )


def _resolve_lerobot_episode_parquet(dataset_dir: str | Path, episode: int) -> Path:
    from data_juicer._au.utils.lerobot_episode_io import list_episode_parquets

    files = list_episode_parquets(str(dataset_dir))
    if not files:
        raise FileNotFoundError(f"No episode parquets under {dataset_dir}")
    needle = f"episode_{int(episode):06d}.parquet"
    for f in files:
        if Path(f).name == needle:
            return Path(f)
    raise FileNotFoundError(f"{needle} not found under {dataset_dir} (have {len(files)} episodes)")


def evaluate_galaxea_fk_ik(
    dataset_dir: str | Path,
    model_path: str | Path,
    side: str = "right",
    episode: int = 0,
    max_frames: int = 120,
    stride: int = 2,
    gl_backend: str = "egl",
) -> Dict[str, Any]:
    """Compare MJCF FK vs GT EE pose and measure IK recoverability on a LeRobot episode.

    EE pose in Galaxea is typically expressed in a body/torso frame that shares the
    arm orientation but not the arm-root origin. We report raw FK error and the
    residual after a constant translation ``t_off = mean(p_gt - p_fk)``.
    """
    import os

    import pyarrow.parquet as pq

    from .ik import jacobian_ik
    from .renderer import RobotArmRenderer
    from .transforms import opencv_to_mujoco_camera_T

    os.environ.setdefault("MUJOCO_GL", gl_backend)
    parquet = _resolve_lerobot_episode_parquet(dataset_dir, episode)
    arm_col = f"observation.state.{side}_arm"
    ee_col = f"observation.state.{side}_ee_pose"
    table = pq.read_table(parquet, columns=[arm_col, ee_col])
    n = table.num_rows
    idxs = list(range(0, n, max(int(stride), 1)))
    if max_frames is not None:
        idxs = idxs[: int(max_frames)]
    q = np.stack([np.asarray(table.column(arm_col)[i].as_py(), dtype=float) for i in idxs])
    ee = np.stack([np.asarray(table.column(ee_col)[i].as_py(), dtype=float) for i in idxs])

    renderer = RobotArmRenderer(model_path, width=64, height=48, gl_backend=gl_backend)
    try:
        T_adapt = opencv_to_mujoco_camera_T()
        # Place MJ arm-root at identity so FK lives in arm-base coordinates.
        renderer.set_base_pose(invert_T(T_adapt), T_adapt)

        fk_pos = []
        fk_ori = []
        for i in range(len(q)):
            renderer.set_arm_qpos(q[i], 0.02)
            renderer.mujoco.mj_forward(renderer.model, renderer.data)
            pos, R = renderer.site_pose()
            gt_p = ee[i, :3]
            qx, qy, qz, qw = ee[i, 3:]
            R_gt = quat_wxyz_to_mat([qw, qx, qy, qz])
            fk_pos.append(float(np.linalg.norm(pos - gt_p)))
            R_err = R_gt.T @ R
            fk_ori.append(float(np.linalg.norm(Rotation.from_matrix(_orthonormalize(R_err)).as_rotvec())))

        # Constant translation alignment (orientation already matches well).
        fk_xyz = []
        for i in range(len(q)):
            renderer.set_arm_qpos(q[i], 0.02)
            renderer.mujoco.mj_forward(renderer.model, renderer.data)
            pos, _ = renderer.site_pose()
            fk_xyz.append(pos)
        fk_xyz = np.asarray(fk_xyz, dtype=np.float64)
        t_off = ee[:, :3].mean(0) - fk_xyz.mean(0)
        aligned = np.linalg.norm(fk_xyz + t_off - ee[:, :3], axis=1)

        ik_fk_ok = []
        ik_fk_qerr = []
        ik_gt_ok = []
        ik_gt_pos = []
        q_prev = q[0].copy()
        for i in range(len(q)):
            renderer.set_arm_qpos(q[i], 0.02)
            renderer.mujoco.mj_forward(renderer.model, renderer.data)
            pos, R = renderer.site_pose()
            qq, ok, _ = jacobian_ik(
                renderer.model,
                renderer.data,
                renderer.site_id,
                pos,
                R,
                renderer.arm_qpos_addrs,
                q_init=q_prev,
                max_iter=80,
            )
            ik_fk_ok.append(bool(ok))
            ik_fk_qerr.append(float(np.linalg.norm(qq - q[i])))
            if ok:
                q_prev = qq

            tgt = ee[i, :3] - t_off
            qx, qy, qz, qw = ee[i, 3:]
            R_gt = quat_wxyz_to_mat([qw, qx, qy, qz])
            qq2, ok2, m2 = jacobian_ik(
                renderer.model,
                renderer.data,
                renderer.site_id,
                tgt,
                R_gt,
                renderer.arm_qpos_addrs,
                q_init=q[i],
                max_iter=100,
                tol_pos=1e-2,
            )
            ik_gt_ok.append(bool(ok2))
            ik_gt_pos.append(float(m2["position_error_m"]))

        def _med(xs):
            return float(np.median(xs)) if len(xs) else float("nan")

        return {
            "dataset_dir": str(dataset_dir),
            "episode": int(episode),
            "parquet": str(parquet),
            "side": side,
            "num_frames": len(q),
            "fk_vs_gt_pos_median_m": _med(fk_pos),
            "fk_vs_gt_ori_median_rad": _med(fk_ori),
            "fk_vs_gt_ori_median_deg": float(np.degrees(_med(fk_ori))),
            "t_off_gt_minus_fk_m": [float(x) for x in t_off.tolist()],
            "fk_aligned_pos_median_m": _med(aligned),
            "fk_aligned_pos_p90_m": float(np.percentile(aligned, 90)) if len(aligned) else float("nan"),
            "ik_from_fk_site_success": float(np.mean(ik_fk_ok)) if ik_fk_ok else float("nan"),
            "ik_from_fk_site_qerr_median_rad": _med(ik_fk_qerr),
            "ik_from_gt_ee_aligned_success": float(np.mean(ik_gt_ok)) if ik_gt_ok else float("nan"),
            "ik_from_gt_ee_aligned_pos_median_m": _med(ik_gt_pos),
        }
    finally:
        renderer.close()


def clip_from_galaxea_lerobot(
    dataset_dir: str | Path,
    model_path: str | Path,
    side: str = "right",
    episode: int = 0,
    max_frames: Optional[int] = 120,
    stride: int = 2,
    img_w: int = 320,
    img_h: int = 180,
    gl_backend: str = "egl",
) -> Tuple[CalibClip, Dict[str, Any]]:
    """Build a CalibClip from Galaxea LeRobot GT joints via MJCF FK.

    World frame is OpenCV-convention with identity ``cam_c2w``. FK site poses in
    the MuJoCo arm-base frame are mapped by the OpenCV↔MuJoCo adapter so that
    the existing IK path (with YAML ``camera_to_base`` quat = adapter) recovers
    the same site. Palm ``joints_cam`` is the projected EE for UV loss.
    """
    import os

    import pyarrow.parquet as pq

    from .renderer import RobotArmRenderer
    from .transforms import opencv_to_mujoco_camera_T

    os.environ.setdefault("MUJOCO_GL", gl_backend)
    parquet = _resolve_lerobot_episode_parquet(dataset_dir, episode)
    arm_col = f"observation.state.{side}_arm"
    grip_col = f"observation.state.{side}_gripper"
    schema_names = set(pq.read_schema(parquet).names)
    cols = [arm_col]
    if grip_col in schema_names:
        cols.append(grip_col)
    table = pq.read_table(parquet, columns=cols)
    n = table.num_rows
    idxs = list(range(0, n, max(int(stride), 1)))
    if max_frames is not None:
        idxs = idxs[: int(max_frames)]

    fx = fy = 0.5 * img_w / np.tan(np.deg2rad(70.0) / 2.0)  # rough head FOV prior
    cx, cy = img_w * 0.5, img_h * 0.5
    T_adapt = opencv_to_mujoco_camera_T()
    # Identity world=OpenCV cam. EE stored as adapter@T_mj so YAML R=adapter + t=0
    # places MJ arm-root at I and recovers FK targets. UV is skipped (z_cam < 0 under
    # this IK-consistent convention); Galaxea acceptance relies on FK/IK metrics.
    T_world_camera = np.eye(4, dtype=np.float64)

    renderer = RobotArmRenderer(model_path, width=64, height=48, gl_backend=gl_backend)
    frames: List[CalibFrame] = []
    q_all = []
    try:
        renderer.set_base_pose(invert_T(T_adapt), T_adapt)
        for fid in idxs:
            q = np.asarray(table.column(arm_col)[fid].as_py(), dtype=np.float64)
            q_all.append(q)
            grip_raw = 0.0
            if grip_col in table.column_names:
                grip_raw = float(table.column(grip_col)[fid].as_py())
            gripper = float(np.clip(grip_raw / 4.0 - 1.0, -1.0, 1.0))
            renderer.set_arm_qpos(q, 0.02)
            renderer.mujoco.mj_forward(renderer.model, renderer.data)
            pos_mj, R_mj = renderer.site_pose()
            T_mj = se3(R_mj, pos_mj)
            T_world = T_adapt @ T_mj
            state = _T_to_state8(T_world, gripper=gripper)
            frames.append(
                CalibFrame(
                    frame_id=int(fid),
                    state=state,
                    T_world_camera=T_world_camera.copy(),
                    joints_cam=None,
                    fx=fx,
                    fy=fy,
                    cx=cx,
                    cy=cy,
                    img_w=img_w,
                    img_h=img_h,
                    q_gt=q.copy(),
                )
            )
    finally:
        renderer.close()

    if not frames:
        raise ValueError(f"No frames extracted from {parquet}")

    wrist_ref = frames[0].state[:3].copy()
    clip = CalibClip(
        side=side,
        frames=frames,
        source=f"galaxea:{Path(dataset_dir).name}:ep{int(episode):06d}",
        wrist_ref_world=wrist_ref,
        ee_ref_world=wrist_ref.copy(),
    )
    meta = {
        "parquet": str(parquet),
        "num_source_rows": int(n),
        "num_frames": len(frames),
        "stride": int(stride),
        "q_reference_median": [float(x) for x in np.median(np.stack(q_all), axis=0).tolist()],
        "force_camera_to_base_translation_zero": True,
    }
    return clip, meta
