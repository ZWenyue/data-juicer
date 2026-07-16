# -*- coding: utf-8 -*-
"""Shared LeRobot episode array IO for Galaxea / packed / unified layouts."""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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

# Stage 1/2/3 numeric clean layout
CLEAN_SIGNAL_DIM = 16


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


def _is_clean_signal_dim(arr: np.ndarray) -> bool:
    return arr.ndim == 2 and arr.shape[1] == CLEAN_SIGNAL_DIM


def episode_arrays_from_df(df) -> Tuple[np.ndarray, np.ndarray]:
    """Return (states, actions) as float arrays (T, D).

    Preference:
    1. Decomposed Galaxea arm/gripper columns → 16-dim clean layout
    2. ``observation.state`` / ``action`` columns as-is (may be 16 / 80 / packed)
    """
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
    if _has_unified(df):
        states = np.stack([np.asarray(v, dtype=float).reshape(-1) for v in df["observation.state"]])
        actions = np.stack([np.asarray(v, dtype=float).reshape(-1) for v in df["action"]])
        return states, actions
    missing = []
    for c in ["observation.state", "action"] + list(DECOMP_STATE.values()) + list(DECOMP_ACTION.values()):
        if c not in df.columns:
            missing.append(c)
    raise ValueError(
        "Neither unified nor decomposed state/action columns found. "
        f"Missing examples: {missing[:8]}"
    )


def _arm_gripper_from_cfg(df, arm_cfg: Dict[str, Any], which: str) -> Tuple[np.ndarray, np.ndarray]:
    """Extract (T, DoF) arm joints and (T,) gripper using embodiment arm block."""
    from .embodiment_layout import _column_with_selection

    col_key = "state_column" if which == "state" else "action_column"
    idx_key = "state_source_indices" if which == "state" else "action_source_indices"
    slice_key = "state_slice" if which == "state" else "action_slice"

    src = arm_cfg.get("source") or {}
    arm = _column_with_selection(
        df,
        src.get(col_key),
        indices=src.get(idx_key),
        slice_pair=src.get(slice_key),
    )
    if arm is None:
        raise ValueError(f"Cannot resolve {which} arm joints from embodiment source={src}")

    grip_cfg = arm_cfg.get("gripper") or {}
    grip = _column_with_selection(
        df,
        grip_cfg.get(col_key),
        indices=grip_cfg.get(idx_key),
        slice_pair=grip_cfg.get(slice_key),
    )
    if grip is None:
        raise ValueError(f"Cannot resolve {which} gripper from embodiment gripper={grip_cfg}")
    return arm, grip[:, 0]


def episode_arrays_from_embodiment(
    df,
    embodiment: Union[str, Path, Dict[str, Any]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Load Stage1/2/3 clean signals (T, 16) using an embodiment YAML.

    - Decomposed Galaxea columns → ``pack_decomposed_to_16`` (unchanged).
    - Else extract left/right arm+gripper via yaml ``source`` / ``*_slice``.
    - Else if ``observation.state`` is already 16-dim, use as-is.
    """
    from .embodiment_layout import load_embodiment_config

    if _has_decomposed(df):
        return episode_arrays_from_df(df)

    if isinstance(embodiment, dict):
        cfg = embodiment
    else:
        cfg = load_embodiment_config(embodiment)

    arms = cfg.get("arms") or {}
    if "left" in arms and "right" in arms:
        try:
            la, lg = _arm_gripper_from_cfg(df, arms["left"], "state")
            ra, rg = _arm_gripper_from_cfg(df, arms["right"], "state")
            la_a, lg_a = _arm_gripper_from_cfg(df, arms["left"], "action")
            ra_a, rg_a = _arm_gripper_from_cfg(df, arms["right"], "action")
            states = pack_decomposed_to_16(la, lg, ra, rg)
            actions = pack_decomposed_to_16(la_a, lg_a, ra_a, rg_a)
            return states, actions
        except ValueError:
            pass

    if _has_unified(df):
        states, actions = episode_arrays_from_df(df)
        if _is_clean_signal_dim(states) and _is_clean_signal_dim(actions):
            return states, actions
        raise ValueError(
            f"Embodiment {cfg.get('name', embodiment)!r} could not slice arm/gripper, "
            f"and observation.state/action are not 16-dim "
            f"(got state={states.shape}, action={actions.shape}). "
            "Check embodiment YAML source/slice fields."
        )

    raise ValueError(
        f"Embodiment {cfg.get('name', embodiment)!r}: no decomposed columns and "
        "no usable observation.state/action."
    )


def load_episode_arrays(
    parquet_path: str,
    embodiment: Optional[Union[str, Path, Dict[str, Any]]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load one episode parquet and return (states, actions).

    When ``embodiment`` is set, prefer the 16-dim clean layout derived from the
    embodiment YAML (required for packed GR00T-style ``observation.state``).
    """
    df = pq.read_table(parquet_path).to_pandas()
    if embodiment is not None:
        return episode_arrays_from_embodiment(df, embodiment)
    return episode_arrays_from_df(df)


def resolve_video_key(dataset: str, preferred: Optional[str] = None) -> Optional[str]:
    """Pick a camera key under ``videos/chunk-*/{key}/``.

    Order: ``preferred`` if present on disk → info.json video feature whose
    dirname exists (prefer *head*) → first videos subdir.
    """
    root = Path(dataset)
    video_root = root / "videos"
    if not video_root.is_dir():
        return preferred

    def _exists(key: str) -> bool:
        if not key:
            return False
        for chunk in video_root.glob("chunk-*"):
            if (chunk / key).is_dir():
                return True
        return False

    if preferred and _exists(preferred):
        return preferred

    info_path = root / "meta" / "info.json"
    candidates: List[str] = []
    if info_path.is_file():
        try:
            feats = json.loads(info_path.read_text(encoding="utf-8")).get("features") or {}
            for name, meta in feats.items():
                if isinstance(meta, dict) and meta.get("dtype") == "video":
                    candidates.append(name)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass

    # modality.json may map short names → original_key used on disk
    modality_path = root / "meta" / "modality.json"
    if modality_path.is_file():
        try:
            mod = json.loads(modality_path.read_text(encoding="utf-8"))
            for _short, meta in (mod.get("video") or {}).items():
                if isinstance(meta, dict):
                    ok = meta.get("original_key")
                    if ok:
                        candidates.append(ok)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass

    # prefer head camera
    for key in candidates:
        if "head" in key.lower() and _exists(key):
            return key
    for key in candidates:
        if _exists(key):
            return key

    for chunk in sorted(video_root.glob("chunk-*")):
        for sub in sorted(chunk.iterdir()):
            if sub.is_dir():
                return sub.name
    return preferred


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
