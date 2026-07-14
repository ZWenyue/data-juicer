# -*- coding: utf-8 -*-
"""Shared LeRobot episode array IO for Galaxea / unified layouts."""

from __future__ import annotations

import glob
import os
from typing import Optional, Tuple

import numpy as np
import pyarrow.parquet as pq

ARM_SLOT_LEFT = [0, 2, 3, 4, 5, 6]
ARM_SLOT_RIGHT = [8, 10, 11, 12, 13, 14]

DECOMP_STATE = {
    "left_arm": "observation.state.left_arm",
    "left_gripper": "observation.state.left_gripper",
    "right_arm": "observation.state.right_arm",
    "right_gripper": "observation.state.right_gripper",
}
DECOMP_ACTION = {
    "left_arm": "action.left_arm",
    "left_gripper": "action.left_gripper",
    "right_arm": "action.right_arm",
    "right_gripper": "action.right_gripper",
}


def _as_TxD(col, expected_last_dim=None):
    """Stack a pandas column of scalars/arrays into (T, D)."""
    arr = np.stack([np.asarray(v, dtype=float).reshape(-1) for v in col])
    if expected_last_dim is not None and arr.shape[-1] != expected_last_dim:
        raise ValueError(f"Expected last dim {expected_last_dim}, got {arr.shape} from column")
    return arr


def pack_decomposed_to_16(left_arm, left_gripper, right_arm, right_gripper):
    """Pack Galaxea R1 Lite arm+gripper channels into (T, 16).

    Arm channels may be 6-DoF (pad slot 1/9) or 7-DoF (fill slots 0..6 / 8..14).
    """
    la = np.asarray(left_arm, dtype=float)
    ra = np.asarray(right_arm, dtype=float)
    if la.ndim != 2 or la.shape[1] not in (6, 7):
        raise ValueError(f"left_arm must be (T,6|7), got {la.shape}")
    if ra.ndim != 2 or ra.shape[1] not in (6, 7):
        raise ValueError(f"right_arm must be (T,6|7), got {ra.shape}")
    lg = np.asarray(left_gripper, dtype=float).reshape(-1)
    rg = np.asarray(right_gripper, dtype=float).reshape(-1)
    T = la.shape[0]
    if not (len(lg) == T and len(rg) == T and ra.shape[0] == T):
        raise ValueError("Mismatched T across arm/gripper arrays")
    out = np.zeros((T, 16), dtype=float)
    if la.shape[1] == 6:
        out[:, ARM_SLOT_LEFT] = la
    else:
        out[:, 0:7] = la
    out[:, 7] = lg
    if ra.shape[1] == 6:
        out[:, ARM_SLOT_RIGHT] = ra
    else:
        out[:, 8:15] = ra
    out[:, 15] = rg
    return out


def _has_unified(df):
    return "observation.state" in df.columns and "action" in df.columns


def _has_decomposed(df):
    needed = list(DECOMP_STATE.values()) + list(DECOMP_ACTION.values())
    return all(c in df.columns for c in needed)


def episode_arrays_from_df(df) -> Tuple[np.ndarray, np.ndarray]:
    """Return (states, actions) as float arrays (T, D)."""
    if _has_unified(df):
        states = np.stack([np.asarray(v, dtype=float) for v in df["observation.state"]])
        actions = np.stack([np.asarray(v, dtype=float) for v in df["action"]])
        return states, actions
    if _has_decomposed(df):
        states = pack_decomposed_to_16(
            _as_TxD(df[DECOMP_STATE["left_arm"]]),
            _as_TxD(df[DECOMP_STATE["left_gripper"]]),
            _as_TxD(df[DECOMP_STATE["right_arm"]]),
            _as_TxD(df[DECOMP_STATE["right_gripper"]]),
        )
        actions = pack_decomposed_to_16(
            _as_TxD(df[DECOMP_ACTION["left_arm"]]),
            _as_TxD(df[DECOMP_ACTION["left_gripper"]]),
            _as_TxD(df[DECOMP_ACTION["right_arm"]]),
            _as_TxD(df[DECOMP_ACTION["right_gripper"]]),
        )
        return states, actions
    missing = []
    for c in ["observation.state", "action"] + list(DECOMP_STATE.values()) + list(DECOMP_ACTION.values()):
        if c not in df.columns:
            missing.append(c)
    raise ValueError(
        "Neither unified nor decomposed state/action columns found. "
        f"Missing examples: {missing[:8]}"
    )


def load_episode_arrays(parquet_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load one episode parquet and return (states, actions)."""
    df = pq.read_table(parquet_path).to_pandas()
    return episode_arrays_from_df(df)


def iter_task_dirs(root: str):
    """Yield task subdirectories that look like LeRobot datasets."""
    root = os.path.abspath(root)
    if os.path.isdir(os.path.join(root, "data")):
        yield root
        return
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if os.path.isdir(path) and os.path.isdir(os.path.join(path, "data")):
            yield path


def list_episode_parquets(dataset_dir: str, max_files: Optional[int] = None):
    pattern = os.path.join(dataset_dir, "data", "chunk-*", "episode_*.parquet")
    files = sorted(glob.glob(pattern))
    if max_files is not None:
        files = files[:max_files]
    return files
