# -*- coding: utf-8 -*-
"""Offline calibration for hand→robot retarget parameters.

Optimizes workspace scale, retarget rotation, and camera→base translation
against smoothed world-frame wrist states + camera poses from the ego
pipeline. Optionally includes IK residual when a MuJoCo arm model is provided.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from itertools import permutations, product
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from .calibration import (
    HandToRobotCalibration,
    SideCalibration,
    WorkspaceMap,
    load_calibration,
    replace_side,
    save_calibration,
    side_to_dict,
)
from .retarget import (
    JAW_SYMMETRY,
    grasp_orientation_error,
    hand_grasp_frame,
    palm_pixel_from_joints,
    project_point_cam,
    retarget_wrist_to_ee,
    world_to_camera,
)
from .transforms import invert_T, mat_to_quat_wxyz, quat_wxyz_to_mat, se3, state_to_T


def _jacobian_ik_multistart(
    renderer,
    target_pos: np.ndarray,
    target_rot: np.ndarray,
    q_warm: np.ndarray,
    q_reference: np.ndarray,
    max_iter: int = 100,
    tol_pos: float = 5e-3,
    tol_rot: float = 0.0524,
):
    """Try several IK warm-starts; return best (q, ok, metrics)."""
    from .ik import jacobian_ik

    seeds = [np.asarray(q_warm, dtype=np.float64).reshape(-1)]
    q_ref = np.asarray(q_reference, dtype=np.float64).reshape(-1)
    for cand in (
        q_ref,
        np.zeros(6, dtype=np.float64),
        q_ref * 0.5,
        np.array([0.0, 1.0, -1.2, 0.0, 0.3, 0.0], dtype=np.float64),
        np.array([0.4, 1.0, -1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        np.array([-0.4, 1.0, -1.0, 0.0, 0.0, 0.0], dtype=np.float64),
    ):
        if not any(np.allclose(cand, s) for s in seeds):
            seeds.append(np.asarray(cand, dtype=np.float64).reshape(-1))

    def _rank(m):
        # Trade 1 rad of orientation slack against tol_rot/tol_pos metres of position,
        # so a seed is not preferred purely for closing the positional gap.
        return float(m["position_error_m"]) + (tol_pos / tol_rot) * float(m["orientation_error_rad"])

    best = None
    for q0 in seeds:
        q, ok, metrics = jacobian_ik(
            renderer.model,
            renderer.data,
            renderer.site_id,
            target_pos,
            target_rot,
            renderer.arm_qpos_addrs,
            q_init=q0,
            max_iter=max_iter,
            tol_pos=tol_pos,
            tol_rot=tol_rot,
        )
        if ok:
            return q, True, metrics
        if best is None or _rank(metrics) < _rank(best[2]):
            best = (q, False, metrics)
    return best


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
    # Prior pulling retarget_R toward identity, NOT a data residual: the camera-frame
    # rotation delta it scores is a similarity transform of retarget_R, so its angle
    # equals |rotvec(retarget_R)| regardless of the clip. Keep at 0 unless you really
    # want to bias against reorienting the gripper.
    w_rot: float = 0.0
    w_uv: float = 2e-4  # px^2 scaled; ~1.0 when err~70px
    w_ik: float = 1.0
    # Whether the arm can *reach* the commanded orientation; needs an MJCF model (w_ik>0).
    w_ik_rot: float = 1.0
    # Whether the commanded orientation is the *right* one, i.e. the gripper is held like
    # the hand. Without it nothing pins retarget_R to anatomy and w_ik_rot happily settles
    # on whatever pose the arm reaches most comfortably. Needs MANO-21 joints_cam.
    w_grasp: float = 1.0
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
    seed_search: Optional[dict] = None
    workspace_fit: Optional[dict] = None


def _orthonormalize(R: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(R)
    Rn = U @ Vt
    if np.linalg.det(Rn) < 0:
        U[:, -1] *= -1
        Rn = U @ Vt
    return Rn


def sample_reachable_workspace(
    model_path: str,
    n_samples: int = 4000,
    gl_backend: str = "egl",
    seed: int = 0,
) -> np.ndarray:
    """FK-sample the arm site over its joint limits; returns (N,3) in the base frame."""
    import mujoco

    from .renderer import RobotArmRenderer

    renderer = RobotArmRenderer(model_path, width=64, height=64, gl_backend=gl_backend)
    try:
        model, data = renderer.model, renderer.data
        addrs = renderer.arm_qpos_addrs
        limits = []
        for a in addrs:
            jid = int(np.where(model.jnt_qposadr == a)[0][0])
            lo, hi = model.jnt_range[jid]
            limits.append((float(lo), float(hi)) if hi > lo else (0.0, 0.0))
        rng = np.random.default_rng(seed)
        pts = np.empty((n_samples, 3), dtype=np.float64)
        for i in range(n_samples):
            data.qpos[addrs] = [rng.uniform(lo, hi) for lo, hi in limits]
            mujoco.mj_forward(model, data)
            pts[i] = data.site_xpos[renderer.site_id] - data.mocap_pos[renderer.anchor_mocap_id]
        return pts
    finally:
        renderer.close()


def fit_workspace_map(
    clip: CalibClip,
    side_cal: SideCalibration,
    model_path: str,
    coverage: float = 0.9,
    gl_backend: str = "egl",
    n_samples: int = 4000,
) -> Tuple[WorkspaceMap, dict]:
    """Fit the base-frame similarity map taking the clip's hand motion into the arm shell."""
    robot_pts = sample_reachable_workspace(model_path, n_samples=n_samples, gl_backend=gl_backend)
    center_robot = np.median(robot_pts, axis=0)
    r_robot = float(np.percentile(np.linalg.norm(robot_pts - center_robot, axis=1), coverage * 100.0))

    human_pts = []
    for fr in clip.frames:
        T_world_base = fr.T_world_camera @ side_cal.T_camera_base_ref
        p_world = state_to_T(fr.state)[:3, 3]
        human_pts.append((invert_T(T_world_base) @ np.append(p_world, 1.0))[:3])
    human_pts = np.asarray(human_pts, dtype=np.float64)
    center_human = np.median(human_pts, axis=0)
    r_human = float(np.percentile(np.linalg.norm(human_pts - center_human, axis=1), coverage * 100.0))

    # Isotropic: per-axis factors previously ran into their clip limits and produced
    # skewed, degenerate maps. Never scale up — the arm gains nothing from it.
    scale = 1.0 if r_human <= 1e-6 else min(1.0, r_robot / r_human)
    ws = WorkspaceMap(
        center_human_base=center_human,
        center_robot_base=center_robot,
        scale_xyz=np.full(3, scale, dtype=np.float64),
    )
    return ws, {
        "robot_shell_radius_m": r_robot,
        "human_spread_radius_m": r_human,
        "scale": float(scale),
        "coverage": float(coverage),
        "center_robot_base": [float(v) for v in center_robot],
        "center_human_base": [float(v) for v in center_human],
    }


def anchor_base_to_hand(
    clip: CalibClip,
    side_cal: SideCalibration,
    model_path: str,
    gl_backend: str = "egl",
    n_samples: int = 4000,
) -> Tuple[np.ndarray, dict]:
    """Place the arm base so its reachable shell is centred on the observed hand.

    Preferred over :func:`fit_workspace_map` for ego footage: the map keeps IK happy by
    moving the *targets*, which drags the robot away from where the hand actually
    worked. Here the *base* moves instead, so targets stay on the hand and still sit
    near the middle of the arm's workspace. Returns the new ``T_camera_base_ref``.
    """
    robot_pts = sample_reachable_workspace(model_path, n_samples=n_samples, gl_backend=gl_backend)
    center_robot = np.median(robot_pts, axis=0)

    hand_cam = []
    for fr in clip.frames:
        p_world = state_to_T(fr.state)[:3, 3]
        hand_cam.append((invert_T(fr.T_world_camera) @ np.append(p_world, 1.0))[:3])
    hand_cam = np.asarray(hand_cam, dtype=np.float64)
    center_hand_cam = np.median(hand_cam, axis=0)

    R_cb = side_cal.T_camera_base_ref[:3, :3]
    t_cb = center_hand_cam - R_cb @ center_robot
    return se3(R_cb, t_cb), {
        "center_hand_camera": [float(v) for v in center_hand_cam],
        "center_robot_base": [float(v) for v in center_robot],
        "camera_to_base_translation_m": [float(v) for v in t_cb],
        "hand_spread_radius_m": float(np.percentile(np.linalg.norm(hand_cam - center_hand_cam, axis=1), 90.0)),
        "robot_shell_radius_m": float(np.percentile(np.linalg.norm(robot_pts - center_robot, axis=1), 90.0)),
    }


def axis_convention_rotations() -> List[np.ndarray]:
    """The 24 signed axis permutations with det=+1.

    A MANO-wrist→gripper mismatch is a frame-convention swap, so the true
    retarget_R sits near one of these. Identity is first so a tie keeps it.
    """
    mats = []
    for perm in permutations(range(3)):
        for signs in product((1.0, -1.0), repeat=3):
            R = np.zeros((3, 3), dtype=np.float64)
            for row, col in enumerate(perm):
                R[row, col] = signs[row]
            if np.linalg.det(R) > 0:
                mats.append(R)
    mats.sort(key=lambda R: np.linalg.norm(R - np.eye(3)))
    return mats


def fit_retarget_R_from_grasp(clip: CalibClip) -> Optional[Tuple[np.ndarray, dict]]:
    """Solve retarget_R in closed form from hand anatomy, bypassing any search.

    ``retarget_wrist_to_ee`` defines ``R_ee = R_wrist @ retarget_R``, so retarget_R *is*
    the constant wrist→gripper frame change. Every frame with MANO joints observes it
    directly as ``R_wrist.T @ R_grasp``, and the fit is just a rotation average — no
    reachability involved, which is the point: the arm must not get a vote on which way
    the hand was facing.

    Returns ``(retarget_R, info)``, or ``None`` if the clip carries no MANO-21 joints.
    ``info["dispersion_deg_median"]`` is how rigidly the palm frame tracks the wrist and
    therefore the best orientation accuracy any constant retarget_R can reach here.
    """
    candidates = []
    for fr in clip.frames:
        if fr.joints_cam is None:
            continue
        R_grasp_cam = hand_grasp_frame(fr.joints_cam)
        if R_grasp_cam is None:
            continue
        R_grasp_world = np.asarray(fr.T_world_camera[:3, :3], dtype=np.float64) @ R_grasp_cam
        R_wrist = state_to_T(fr.state)[:3, :3]
        candidates.append(_orthonormalize(R_wrist.T @ R_grasp_world))
    if not candidates:
        return None

    # Averaging is only meaningful once every sample sits on the same side of the jaw
    # symmetry; re-align against the running mean so a bad first frame cannot bias it.
    reference = candidates[0]
    aligned = list(candidates)
    for _ in range(2):
        aligned = []
        for R in candidates:
            R_flipped = R @ JAW_SYMMETRY
            aligned.append(R if np.trace(reference.T @ R) >= np.trace(reference.T @ R_flipped) else R_flipped)
        reference = Rotation.from_matrix(np.stack(aligned)).mean().as_matrix()

    R_fit = _orthonormalize(reference)
    dispersion = [
        float(np.degrees(np.linalg.norm(Rotation.from_matrix(_orthonormalize(R_fit.T @ R)).as_rotvec())))
        for R in aligned
    ]
    info = {
        "mode": "grasp_frame",
        "n_frames": len(aligned),
        "dispersion_deg_median": float(np.median(dispersion)),
        "dispersion_deg_p90": float(np.percentile(dispersion, 90.0)),
        "retarget_quaternion_wxyz": [float(v) for v in mat_to_quat_wxyz(R_fit)],
    }
    return R_fit, info


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
        workspace_map=base_side.workspace_map,
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
    grasp_errs = []
    uv_errs = []
    ik_ok = []
    ik_pos = []
    ik_rot = []
    q_prev = q_init if q_init is not None else side_cal.q_reference.copy()

    for fr in clip.frames:
        T_world_camera = fr.T_world_camera
        T_world_base = T_world_camera @ side_cal.T_camera_base_ref
        T_world_ee = retarget_wrist_to_ee(fr.state, side_cal, wrist_ref, ee_ref, T_world_base=T_world_base)
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

        # How far retarget_R tilts the gripper away from the raw wrist frame. This is
        # a prior only (value is independent of the clip); the data-driven orientation
        # signal is the IK residual below.
        T_camera_wrist = world_to_camera(state_to_T(fr.state), T_world_camera)
        R_err = T_camera_ee[:3, :3] @ T_camera_wrist[:3, :3].T
        rot_errs.append(float(np.linalg.norm(Rotation.from_matrix(_orthonormalize(R_err)).as_rotvec())))

        if fr.joints_cam is not None:
            # Palm frame and T_camera_ee are both camera-frame, so they compare directly.
            R_grasp = hand_grasp_frame(fr.joints_cam)
            if R_grasp is not None:
                grasp_errs.append(grasp_orientation_error(T_camera_ee[:3, :3], R_grasp))

            u_t, v_t, ok_t = palm_pixel_from_joints(fr.joints_cam, fr.fx, fr.fy, fr.cx, fr.cy)
            p_ee_cam = np.asarray(T_camera_ee[:3, 3], dtype=np.float64)
            u_p, v_p, ok_p = project_point_cam(p_ee_cam, fr.fx, fr.fy, fr.cx, fr.cy)
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
            q, ok, metrics = _jacobian_ik_multistart(
                renderer,
                T_mj_ee[:3, 3],
                T_mj_ee[:3, :3],
                q_warm=q_warm,
                q_reference=side_cal.q_reference,
                max_iter=100,
                tol_pos=5e-3,
                tol_rot=0.0524,
            )
            ik_ok.append(bool(ok))
            ik_pos.append(float(metrics["position_error_m"]))
            ik_rot.append(float(metrics["orientation_error_rad"]))
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
        "median_grasp_orientation_error_rad": _med(grasp_errs),
        "median_reprojection_error_px": _med(uv_errs),
        "mean_reprojection_error_px": float(np.mean(uv_errs)) if uv_errs else float("nan"),
        "ik_success_rate": float(np.mean(ik_ok)) if ik_ok else float("nan"),
        "median_ik_position_error_m": _med(ik_pos),
        "median_ik_orientation_error_rad": _med(ik_rot),
        "loss": 0.0,
    }
    loss = 0.0
    if pos_errs:
        loss += weights.w_pos * float(np.mean(np.square(pos_errs)))
    if rot_errs:
        loss += weights.w_rot * float(np.mean(np.square(rot_errs)))
    if grasp_errs:
        loss += weights.w_grasp * float(np.mean(np.square(grasp_errs)))
    if uv_errs:
        loss += weights.w_uv * float(np.mean(np.square(uv_errs)))
    if ik_pos:
        fail = 1.0 - float(np.mean(ik_ok))
        loss += weights.w_ik * (float(np.mean(np.square(ik_pos))) + 0.25 * fail)
    if ik_rot:
        loss += weights.w_ik_rot * float(np.mean(np.square(ik_rot)))
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
    seed_retarget_conventions: bool = True,
    fit_workspace: bool = False,
    workspace_coverage: float = 0.9,
    anchor_base: bool = False,
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
        workspace_map=base_side.workspace_map,
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

    workspace_fit = None
    if anchor_base:
        if not model_path:
            raise ValueError("anchor_base needs model_path to sample the reachable shell")
        T_cam_base, workspace_fit = anchor_base_to_hand(clip, base_side, model_path)
        base_side.T_camera_base_ref = T_cam_base
        workspace_fit["mode"] = "anchor_base"
    if fit_workspace:
        if not model_path:
            raise ValueError("fit_workspace needs model_path to sample the reachable shell")
        ws_map, workspace_fit = fit_workspace_map(
            clip, base_side, model_path, coverage=workspace_coverage, gl_backend="egl"
        )
        workspace_fit["mode"] = "workspace_map"
        base_side.workspace_map = ws_map

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

    # A wrist→gripper convention swap is ~90-180° away from identity, far outside the
    # basin L-BFGS-B can cross on a finite-differenced IK objective. Hand anatomy pins it
    # down exactly when MANO joints are around; otherwise fall back to coarse-searching
    # the 24 axis conventions and refining locally from the winner.
    seed_search = None
    grasp_fit = fit_retarget_R_from_grasp(clip)
    if grasp_fit is not None:
        R_grasp_seed, seed_search = grasp_fit
        x0 = x0.copy()
        x0[3:6] = Rotation.from_matrix(R_grasp_seed).as_rotvec()
    elif seed_retarget_conventions and renderer is not None:
        seeds = axis_convention_rotations()
        losses = []
        for R_seed in seeds:
            x_seed = x0.copy()
            x_seed[3:6] = Rotation.from_matrix(R_seed).as_rotvec()
            losses.append(objective(x_seed))
        best_i = int(np.argmin(losses))
        seed_search = {
            "mode": "axis_conventions",
            "n_seeds": len(seeds),
            "best_index": best_i,
            "best_loss": float(losses[best_i]),
            "identity_loss": float(losses[0]),
            "best_retarget_quaternion_wxyz": [float(v) for v in mat_to_quat_wxyz(seeds[best_i])],
        }
        x0 = x0.copy()
        x0[3:6] = Rotation.from_matrix(seeds[best_i]).as_rotvec()

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
        seed_search=seed_search,
        workspace_fit=workspace_fit,
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


# ---------------------------------------------------------------------------
# EgoDex LeRobot (real egocentric hand poses) — P1 human-hand calibration
# ---------------------------------------------------------------------------

EGODEX_JOINTS_PER_HAND = 25
EGODEX_DIMS_PER_JOINT = 6  # xyz + rpy
EGODEX_HAND_DIM = EGODEX_JOINTS_PER_HAND * EGODEX_DIMS_PER_JOINT  # 150
# Confidence / joint layout: 0=leftHand ... 24=leftThumbTip, 25=rightHand ...
EGODEX_WRIST_CONF_IDX = {"left": 0, "right": 25}
EGODEX_INDEX_TIP_JOINT = 20  # within-hand joint index
EGODEX_THUMB_TIP_JOINT = 24


def _egodex_info(dataset_dir: str | Path) -> dict:
    path = Path(dataset_dir) / "meta" / "info.json"
    if not path.is_file():
        raise FileNotFoundError(f"EgoDex info.json missing: {path}")
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def is_egodex_lerobot(dataset_dir: str | Path) -> bool:
    try:
        info = _egodex_info(dataset_dir)
    except FileNotFoundError:
        return False
    if str(info.get("robot_type", "")).lower().startswith("egodex"):
        return True
    feats = info.get("features") or {}
    return "camera.extrinsics" in feats and "observation.state" in feats


def _egodex_video_wh(info: dict) -> Tuple[int, int]:
    feat = (info.get("features") or {}).get("observation.images.ego") or {}
    shape = feat.get("shape") or [270, 480, 3]
    # LeRobot video shape is [H, W, C]
    return int(shape[1]), int(shape[0])


def _scale_intrinsics_to_video(K: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """Scale K written for a larger sensor to the stored video resolution."""
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    # Infer calibration resolution from principal point (cx≈W/2, cy≈H/2).
    calib_w = max(float(K[0, 2] * 2.0), 1.0)
    calib_h = max(float(K[1, 2] * 2.0), 1.0)
    sx = float(img_w) / calib_w
    sy = float(img_h) / calib_h
    Ks = K.copy()
    Ks[0, :] *= sx
    Ks[1, :] *= sy
    return Ks


def _load_egodex_episode_rows(
    dataset_dir: str | Path,
    episode: int,
    columns: Sequence[str],
):
    """Load one episode from sharded file-*.parquet under data/chunk-*/."""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    root = Path(dataset_dir)
    files = sorted(root.glob("data/chunk-*/file-*.parquet"))
    if not files:
        # fallback to classic episode_*.parquet
        files = sorted(root.glob("data/chunk-*/episode_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet shards under {dataset_dir}/data")

    for pf in files:
        # Cheap probe: episode_index column only
        probe = pq.read_table(pf, columns=["episode_index"])
        if int(episode) not in set(probe.column(0).to_pylist()):
            continue
        table = pq.read_table(pf, columns=list(columns))
        mask = pc.equal(table["episode_index"], int(episode))
        filtered = table.filter(mask)
        if filtered.num_rows == 0:
            continue
        return filtered, Path(pf)
    raise FileNotFoundError(f"episode_index={episode} not found under {dataset_dir}")


def _egodex_gripper_from_state(hand6x25: np.ndarray) -> float:
    """Map thumb–index tip distance to gripper ∈ [-1, 1] (1=open)."""
    idx = hand6x25[EGODEX_INDEX_TIP_JOINT, :3]
    th = hand6x25[EGODEX_THUMB_TIP_JOINT, :3]
    dist = float(np.linalg.norm(idx - th))
    # ~2 cm closed → -1; ~10 cm open → +1
    return float(np.clip(dist / 0.05 - 1.0, -1.0, 1.0))


def clip_from_egodex_lerobot(
    dataset_dir: str | Path,
    side: str = "right",
    episode: int = 0,
    max_frames: Optional[int] = 200,
    stride: int = 2,
    min_wrist_conf: float = 0.5,
    extrinsics_are_c2w: bool = True,
    camera_as_world: bool = True,
) -> Tuple[CalibClip, Dict[str, Any]]:
    """Build a CalibClip from EgoDex LeRobot hand poses.

    EgoDex ``observation.state`` joints are in **camera frame**. For IK-compatible
    calibration we default to ``camera_as_world=True``:

    - ``T_world_camera = I``
    - wrist state encoded as ``T_adapt @ T_cam_wrist`` (same trick as Galaxea FK clips)
    - ``joints_cam`` kept in true OpenCV camera frame for UV (evaluate undoes adapter)

    Set ``camera_as_world=False`` to keep full ``cam_c2w`` world trajectories.
    """
    from .transforms import opencv_to_mujoco_camera_T

    if side not in ("left", "right"):
        raise ValueError(side)
    info = _egodex_info(dataset_dir)
    img_w, img_h = _egodex_video_wh(info)
    T_adapt = opencv_to_mujoco_camera_T()

    cols = [
        "observation.state",
        "observation.state.confidence",
        "camera.intrinsics",
        "camera.extrinsics",
        "frame_index",
        "episode_index",
    ]
    table, parquet = _load_egodex_episode_rows(dataset_dir, episode, cols)
    n = table.num_rows
    hand_offset = 0 if side == "left" else EGODEX_HAND_DIM
    conf_idx = EGODEX_WRIST_CONF_IDX[side]

    frames: List[CalibFrame] = []
    skipped_conf = 0
    wrist_cam_pts = []
    for i in range(0, n, max(int(stride), 1)):
        conf = np.asarray(table.column("observation.state.confidence")[i].as_py(), dtype=np.float64)
        if float(conf[conf_idx]) < float(min_wrist_conf):
            skipped_conf += 1
            continue
        state300 = np.asarray(table.column("observation.state")[i].as_py(), dtype=np.float64)
        hand = state300[hand_offset : hand_offset + EGODEX_HAND_DIM].reshape(
            EGODEX_JOINTS_PER_HAND, EGODEX_DIMS_PER_JOINT
        )
        K_raw = np.asarray(table.column("camera.intrinsics")[i].as_py(), dtype=np.float64)
        K = _scale_intrinsics_to_video(K_raw, img_w, img_h)
        E = np.asarray(table.column("camera.extrinsics")[i].as_py(), dtype=np.float64).reshape(4, 4)
        T_c2w = E if extrinsics_are_c2w else invert_T(E)

        wrist_xyzrpy = hand[0]
        grip = _egodex_gripper_from_state(hand)
        T_cam_wrist = state_to_T(
            [
                wrist_xyzrpy[0],
                wrist_xyzrpy[1],
                wrist_xyzrpy[2],
                wrist_xyzrpy[3],
                wrist_xyzrpy[4],
                wrist_xyzrpy[5],
                0.0,
                grip,
            ]
        )
        wrist_cam_pts.append(wrist_xyzrpy[:3].copy())

        if camera_as_world:
            # Encode so YAML camera_to_base (R=adapter) yields MJ targets with +Z.
            T_world_wrist = T_adapt @ T_cam_wrist
            T_world_camera = np.eye(4, dtype=np.float64)
        else:
            T_world_wrist = T_c2w @ T_cam_wrist
            T_world_camera = np.asarray(T_c2w, dtype=np.float64)

        state8 = _T_to_state8(T_world_wrist, gripper=grip)
        joints_cam = np.stack(
            [hand[0, :3], hand[EGODEX_INDEX_TIP_JOINT, :3], hand[EGODEX_THUMB_TIP_JOINT, :3]],
            axis=0,
        )
        fid = int(table.column("frame_index")[i].as_py())
        frames.append(
            CalibFrame(
                frame_id=fid,
                state=state8,
                T_world_camera=T_world_camera,
                joints_cam=joints_cam,
                fx=float(K[0, 0]),
                fy=float(K[1, 1]),
                cx=float(K[0, 2]),
                cy=float(K[1, 2]),
                img_w=img_w,
                img_h=img_h,
            )
        )
        if max_frames is not None and len(frames) >= int(max_frames):
            break

    if not frames:
        raise ValueError(
            f"No EgoDex frames for episode={episode} side={side} "
            f"(rows={n}, skipped_conf={skipped_conf}, min_wrist_conf={min_wrist_conf})"
        )

    wrist_ref = frames[0].state[:3].copy()
    clip = CalibClip(
        side=side,
        frames=frames,
        source=f"egodex:{Path(dataset_dir).name}:ep{int(episode):06d}",
        wrist_ref_world=wrist_ref,
        ee_ref_world=wrist_ref.copy(),
    )
    wcam = np.stack(wrist_cam_pts, axis=0)
    # Suggested OpenCV-camera base translation: below (+Y) and behind (−Z) the hand cloud.
    suggested_base_t = [
        float(wcam[:, 0].mean()),
        float(wcam[:, 1].mean() + 0.28),
        float(wcam[:, 2].mean() - 0.40),
    ]
    meta = {
        "dataset_dir": str(dataset_dir),
        "parquet": str(parquet),
        "episode": int(episode),
        "side": side,
        "num_source_rows": int(n),
        "num_frames": len(frames),
        "stride": int(stride),
        "skipped_low_confidence": int(skipped_conf),
        "min_wrist_conf": float(min_wrist_conf),
        "img_w": img_w,
        "img_h": img_h,
        "extrinsics_are_c2w": bool(extrinsics_are_c2w),
        "camera_as_world": bool(camera_as_world),
        "robot_type": info.get("robot_type"),
        "suggested_camera_to_base_translation_m": suggested_base_t,
        "wrist_cam_mean": [float(x) for x in wcam.mean(0).tolist()],
    }
    return clip, meta


def p1_metrics_from_eval(metrics: dict, num_frames: int) -> Dict[str, Any]:
    """P1 exit gates: ≥100 frames, IK≥90%, median reprojection <15 px."""
    ik = metrics.get("ik_success_rate", float("nan"))
    uv = metrics.get("median_reprojection_error_px", float("nan"))
    checks = {
        "p1_num_frames_ge_100": int(num_frames) >= 100,
        "p1_ik_success_ge_0_9": ik == ik and float(ik) >= 0.9,
        "p1_reproj_median_lt_15px": uv == uv and float(uv) < 15.0,
    }
    checks["p1_pass"] = all(checks.values())
    return {
        "num_frames": int(num_frames),
        "ik_success_rate": float(ik) if ik == ik else float("nan"),
        "median_reprojection_error_px": float(uv) if uv == uv else float("nan"),
        "median_ik_position_error_m": metrics.get("median_ik_position_error_m"),
        "median_reach_violation_m": metrics.get("median_reach_violation_m"),
        "checks": checks,
    }
