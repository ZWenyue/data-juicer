# -*- coding: utf-8 -*-
"""Canonical 80-dim layout helpers + per-embodiment YAML loading."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import yaml

# Per-arm semantic groups inside the 29-dim block
CANONICAL_JOINT_NAMES = (
    "shoulder_pitch",
    "shoulder_roll",
    "shoulder_yaw",
    "elbow_pitch",
    "forearm_roll",
    "wrist_pitch",
    "wrist_roll",
)

UNIFIED_DIM = 80
ARM_DIM = 29
SHARED_DIM = 22
NUM_JOINTS = 7
NUM_EEF = 9  # xyz(3) + rot6d(6)
NUM_GRIPPER = 1
NUM_HAND = 12

LEFT_BASE = 0
RIGHT_BASE = 29
SHARED_BASE = 58

# Offsets within one arm block
OFF_JOINT = 0
OFF_EEF = 7
OFF_EEF_POS = 7
OFF_EEF_ROT = 10
OFF_GRIPPER = 16
OFF_HAND = 17

_ARM_KEYS = ("left", "right")
_ARM_BASE = {"left": LEFT_BASE, "right": RIGHT_BASE}

_DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "embodiments"


def default_embodiment_config_dir() -> Path:
    return _DEFAULT_CONFIG_DIR


def load_embodiment_config(name_or_path: Union[str, Path]) -> Dict[str, Any]:
    """Load an embodiment YAML by absolute/relative path or by stem name."""
    p = Path(str(name_or_path))
    if not p.is_file():
        cand = _DEFAULT_CONFIG_DIR / f"{name_or_path}.yaml"
        if not cand.is_file():
            cand = _DEFAULT_CONFIG_DIR / f"{name_or_path}.yml"
        if not cand.is_file():
            raise FileNotFoundError(
                f"Embodiment config not found: {name_or_path!r} "
                f"(looked under {_DEFAULT_CONFIG_DIR})"
            )
        p = cand
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Embodiment config must be a mapping, got {type(cfg)}")
    _validate_embodiment_config(cfg)
    cfg["_config_path"] = str(p.resolve())
    return cfg


def _validate_embodiment_config(cfg: Dict[str, Any]) -> None:
    arms = cfg.get("arms")
    if not isinstance(arms, dict) or not arms:
        raise ValueError("embodiment config requires non-empty 'arms'")
    for side, arm in arms.items():
        if side not in _ARM_KEYS:
            raise ValueError(f"Unknown arm key {side!r}; expected left/right")
        jm = arm.get("joint_map")
        if not isinstance(jm, dict):
            raise ValueError(f"arms.{side}.joint_map must be a dict")
        missing = [n for n in CANONICAL_JOINT_NAMES if n not in jm]
        if missing:
            raise ValueError(f"arms.{side}.joint_map missing keys: {missing}")
        for n in CANONICAL_JOINT_NAMES:
            v = jm[n]
            if v is not None and not (isinstance(v, int) and v >= 0):
                raise ValueError(f"arms.{side}.joint_map[{n}] must be non-neg int or null, got {v!r}")


def remap_joints_to_canonical(
    src: np.ndarray,
    joint_map: Dict[str, Optional[int]],
    joint_sign: Optional[Sequence[float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Map source joint vector (T, DoF) into canonical 7 slots.

    Returns
    -------
    joints : (T, 7) float
    occ : (7,) float {0,1} occupancy for the standard slots
    """
    arr = np.asarray(src, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"joint src must be (T,D), got {arr.shape}")
    T = arr.shape[0]
    out = np.zeros((T, NUM_JOINTS), dtype=float)
    occ = np.zeros(NUM_JOINTS, dtype=float)
    signs = list(joint_sign) if joint_sign is not None else [1.0] * NUM_JOINTS
    if len(signs) != NUM_JOINTS:
        raise ValueError(f"joint_sign must have length {NUM_JOINTS}, got {len(signs)}")
    for i, name in enumerate(CANONICAL_JOINT_NAMES):
        idx = joint_map.get(name)
        if idx is None:
            continue
        if idx >= arr.shape[1]:
            raise ValueError(f"joint_map[{name}]={idx} out of range for src dim {arr.shape[1]}")
        out[:, i] = arr[:, idx] * float(signs[i])
        occ[i] = 1.0
    return out, occ


def quat_wxyz_to_rot6d(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion (w,x,y,z) to Zhou et al. 6D rotation (first two columns).

    quat : (..., 4)
    returns : (..., 6)
    """
    q = np.asarray(quat, dtype=float)
    if q.shape[-1] != 4:
        raise ValueError(f"quat last dim must be 4, got {q.shape}")
    # normalize
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    q = q / n
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    # rotation matrix from wxyz
    r00 = 1 - 2 * (y * y + z * z)
    r10 = 2 * (x * y + w * z)
    r20 = 2 * (x * z - w * y)
    r01 = 2 * (x * y - w * z)
    r11 = 1 - 2 * (x * x + z * z)
    r21 = 2 * (y * z + w * x)
    # first two columns flattened column-major as (r00,r10,r20, r01,r11,r21)
    return np.stack([r00, r10, r20, r01, r11, r21], axis=-1)


def ee_pose_to_9d(ee_pose: np.ndarray, rot_repr: str = "quat_wxyz") -> np.ndarray:
    """Pack EE pose into (T, 9) = xyz(3) + rot6d(6)."""
    arr = np.asarray(ee_pose, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[-1] < 7:
        raise ValueError(f"ee_pose expected last dim >= 7 (xyz+quat), got {arr.shape}")
    pos = arr[:, :3]
    if rot_repr == "quat_wxyz":
        rot6 = quat_wxyz_to_rot6d(arr[:, 3:7])
    elif rot_repr == "quat_xyzw":
        q = arr[:, 3:7]
        rot6 = quat_wxyz_to_rot6d(np.concatenate([q[:, 3:4], q[:, :3]], axis=-1))
    elif rot_repr == "rot6d":
        rot6 = arr[:, 3:9]
    else:
        raise ValueError(f"Unsupported rot_repr: {rot_repr}")
    return np.concatenate([pos, rot6], axis=-1)


def _as_TxD(col, expected_last: Optional[int] = None) -> np.ndarray:
    arr = np.stack([np.asarray(v, dtype=float).reshape(-1) for v in col])
    if expected_last is not None and arr.shape[-1] != expected_last:
        raise ValueError(f"Expected last dim {expected_last}, got {arr.shape}")
    return arr


def _as_Tx1(col) -> np.ndarray:
    return np.asarray([float(np.asarray(v).reshape(-1)[0]) for v in col], dtype=float).reshape(-1, 1)


def _column_with_selection(
    df,
    column: Optional[str],
    *,
    indices: Optional[Sequence] = None,
    slice_pair: Optional[Sequence] = None,
) -> Optional[np.ndarray]:
    """Load a dataframe column as (T, D), optionally slicing packed dims.

    ``slice_pair`` is half-open ``[start, end)``. ``indices`` picks arbitrary
    columns. If both are set, ``slice_pair`` wins. Neither → full column.
    """
    if not column or column not in df.columns:
        return None
    arr = _as_TxD(df[column])
    if slice_pair is not None:
        if len(slice_pair) != 2:
            raise ValueError(f"slice must be [start, end), got {slice_pair!r}")
        lo, hi = int(slice_pair[0]), int(slice_pair[1])
        if not (0 <= lo < hi <= arr.shape[1]):
            raise ValueError(f"slice [{lo},{hi}) OOB for column dim {arr.shape[1]}")
        return arr[:, lo:hi]
    if indices is not None:
        idx = [int(i) for i in indices]
        if any(i < 0 or i >= arr.shape[1] for i in idx):
            raise ValueError(f"source indices {idx} OOB for column dim {arr.shape[1]}")
        return arr[:, idx]
    return arr


def build_dim_mask(cfg: Dict[str, Any]) -> np.ndarray:
    """Build static occupancy mask (80,) from embodiment config (no data needed)."""
    mask = np.zeros(UNIFIED_DIM, dtype=float)
    arms = cfg.get("arms") or {}
    for side, base in _ARM_BASE.items():
        arm = arms.get(side)
        if not arm:
            continue
        jm = arm.get("joint_map") or {}
        for i, name in enumerate(CANONICAL_JOINT_NAMES):
            if jm.get(name) is not None:
                mask[base + OFF_JOINT + i] = 1.0
        if arm.get("eef"):
            mask[base + OFF_EEF : base + OFF_EEF + NUM_EEF] = 1.0
        if arm.get("gripper"):
            mask[base + OFF_GRIPPER] = 1.0
        hand = arm.get("hand")
        if hand:
            # hand may declare num_joints or source dim count
            n = int(hand.get("num_joints", NUM_HAND))
            n = min(n, NUM_HAND)
            mask[base + OFF_HAND : base + OFF_HAND + n] = 1.0
    shared = cfg.get("shared") or {}
    for _name, block in shared.items():
        if not block:
            continue
        slots = block.get("slots") or []
        for s in slots:
            si = int(s)
            if not (0 <= si < SHARED_DIM):
                raise ValueError(f"shared slot {si} out of range [0,{SHARED_DIM})")
            mask[SHARED_BASE + si] = 1.0
    return mask


def _fill_arm_block(
    out: np.ndarray,
    mask_frame: np.ndarray,
    base: int,
    arm_cfg: Dict[str, Any],
    df,
    which: str,  # "state" | "action"
) -> None:
    """Fill one arm's 29-dim block for state or action into out (T,80)."""
    T = out.shape[0]
    src_cfg = arm_cfg.get("source") or {}
    col_key = "state_column" if which == "state" else "action_column"
    idx_key = "state_source_indices" if which == "state" else "action_source_indices"
    slice_key = "state_slice" if which == "state" else "action_slice"

    joint_arr = _column_with_selection(
        df,
        src_cfg.get(col_key),
        indices=src_cfg.get(idx_key),
        slice_pair=src_cfg.get(slice_key),
    )
    if joint_arr is not None:
        joints, jocc = remap_joints_to_canonical(
            joint_arr,
            arm_cfg["joint_map"],
            arm_cfg.get("joint_sign"),
        )
        if joints.shape[0] != T:
            raise ValueError(f"T mismatch joints {joints.shape[0]} vs {T}")
        out[:, base + OFF_JOINT : base + OFF_JOINT + NUM_JOINTS] = joints
        mask_frame[base + OFF_JOINT : base + OFF_JOINT + NUM_JOINTS] = jocc

    grip = arm_cfg.get("gripper")
    if grip:
        g_arr = _column_with_selection(
            df,
            grip.get(col_key),
            indices=grip.get(idx_key),
            slice_pair=grip.get(slice_key),
        )
        if g_arr is not None:
            out[:, base + OFF_GRIPPER] = g_arr[:, 0]
            mask_frame[base + OFF_GRIPPER] = 1.0

    eef = arm_cfg.get("eef")
    if eef and which == "state":
        e_arr = _column_with_selection(
            df,
            eef.get("state_column"),
            indices=eef.get("state_source_indices"),
            slice_pair=eef.get("state_slice"),
        )
        if e_arr is not None:
            pose9 = ee_pose_to_9d(e_arr, eef.get("rot_repr", "quat_wxyz"))
            out[:, base + OFF_EEF : base + OFF_EEF + NUM_EEF] = pose9
            mask_frame[base + OFF_EEF : base + OFF_EEF + NUM_EEF] = 1.0
    # action EEF deferred (camera-frame delta)

    hand = arm_cfg.get("hand")
    if hand:
        h_arr = _column_with_selection(
            df,
            hand.get(col_key) or hand.get("state_column" if which == "state" else "action_column"),
            indices=hand.get(idx_key),
            slice_pair=hand.get(slice_key),
        )
        if h_arr is not None:
            n = min(h_arr.shape[1], NUM_HAND)
            out[:, base + OFF_HAND : base + OFF_HAND + n] = h_arr[:, :n]
            mask_frame[base + OFF_HAND : base + OFF_HAND + n] = 1.0


def _resolve_shared_array(block: Dict[str, Any], df, which: str) -> Optional[np.ndarray]:
    """Load shared signal as (T, D). Supports single column or concatenated columns list."""
    singular = "state_column" if which == "state" else "action_column"
    plural = "state_columns" if which == "state" else "action_columns"
    cols = block.get(plural)
    if cols is None:
        col = block.get(singular)
        cols = [col] if col else None
    if not cols:
        return None
    if isinstance(cols, str):
        cols = [cols]
    pieces = []
    for c in cols:
        if c not in df.columns:
            return None
        pieces.append(_as_TxD(df[c]))
    if len(pieces) == 1:
        return pieces[0]
    T0 = pieces[0].shape[0]
    if any(p.shape[0] != T0 for p in pieces):
        raise ValueError(f"shared {plural} length mismatch: {[p.shape for p in pieces]}")
    return np.concatenate(pieces, axis=-1)


def _fill_shared(
    out: np.ndarray,
    mask_frame: np.ndarray,
    shared_cfg: Dict[str, Any],
    df,
    which: str,
) -> None:
    idx_key = "state_source_indices" if which == "state" else "action_source_indices"
    for _name, block in (shared_cfg or {}).items():
        if not block:
            continue
        slots = [int(s) for s in (block.get("slots") or [])]
        if not slots:
            continue
        arr = _resolve_shared_array(block, df, which)
        if arr is None:
            # occupancy still marked via build_dim_mask; values stay 0
            continue
        src_idx = block.get(idx_key)
        if src_idx is None:
            # Default: map consecutive source dims into leading slots.
            n = min(len(slots), arr.shape[1])
            pairs = list(zip(slots[:n], range(n)))
        else:
            if len(src_idx) != len(slots):
                raise ValueError(
                    f"shared {_name}: {idx_key} len {len(src_idx)} != slots len {len(slots)}"
                )
            pairs = []
            for s_slot, s_i in zip(slots, src_idx):
                if s_i is None:
                    continue
                pairs.append((int(s_slot), int(s_i)))
        for s_slot, s_i in pairs:
            if s_i < 0 or s_i >= arr.shape[1]:
                raise ValueError(f"shared {_name}: source index {s_i} OOB for {arr.shape}")
            out[:, SHARED_BASE + s_slot] = arr[:, s_i]
            mask_frame[SHARED_BASE + s_slot] = 1.0


def pack_episode_to_80(df, cfg: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pack one episode dataframe into unified (states, actions, dim_mask).

    Returns
    -------
    states : (T, 80)
    actions : (T, 80)
    dim_mask : (T, 80) broadcast of occupancy (1 = used)
    """
    # Determine T from any present column referenced by the config
    T = None
    for side in _ARM_KEYS:
        arm = (cfg.get("arms") or {}).get(side) or {}
        for key in ("state_column", "action_column"):
            col = ((arm.get("source") or {}).get(key))
            if col and col in df.columns:
                T = len(df[col])
                break
        if T is not None:
            break
    if T is None:
        raise ValueError("Cannot infer episode length T from embodiment columns")

    states = np.zeros((T, UNIFIED_DIM), dtype=float)
    actions = np.zeros((T, UNIFIED_DIM), dtype=float)
    # Occupancy from config (shared by state/action); values may be 0 when a
    # semantic group is declared but action-side fill is deferred (e.g. EEF).
    scratch = np.zeros(UNIFIED_DIM, dtype=float)

    arms = cfg.get("arms") or {}
    for side, base in _ARM_BASE.items():
        arm = arms.get(side)
        if not arm:
            continue
        _fill_arm_block(states, scratch, base, arm, df, "state")
        _fill_arm_block(actions, scratch, base, arm, df, "action")

    shared = cfg.get("shared") or {}
    _fill_shared(states, scratch, shared, df, "state")
    _fill_shared(actions, scratch, shared, df, "action")

    static = build_dim_mask(cfg)
    dim_mask = np.broadcast_to(static.reshape(1, -1), (T, UNIFIED_DIM)).copy()
    return states, actions, dim_mask


def resolve_parquet_path(sample: Dict[str, Any], parquet_field: str = "parquet_path") -> str:
    p = sample.get(parquet_field)
    if not p:
        raise ValueError(f"Sample missing '{parquet_field}'")
    if not os.path.isfile(p):
        raise FileNotFoundError(f"parquet not found: {p}")
    return p
