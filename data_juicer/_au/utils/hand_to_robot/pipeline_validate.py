# -*- coding: utf-8 -*-
"""P3 pipeline validation helpers (action invariance, quality rollup)."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np

from data_juicer.utils.constant import Fields


def _deep_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _deep_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_deep_json(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def snapshot_hand_actions(sample: dict, hand_action_field: str = "hand_action_tags") -> dict:
    meta = sample.get(Fields.meta, {}) or {}
    return _deep_json(copy.deepcopy(meta.get(hand_action_field, [])))


def _collect_states(hand_action_list: Any) -> List[List[float]]:
    out: List[List[float]] = []
    if not isinstance(hand_action_list, list):
        return out
    for clip in hand_action_list:
        if not isinstance(clip, dict):
            continue
        if "states" in clip:
            for s in clip.get("states", []) or []:
                out.append(list(s))
            continue
        for ht in ("right", "left"):
            hand = clip.get(ht, {}) or {}
            for s in hand.get("states", []) or []:
                out.append(list(s))
    return out


def compare_action_invariance(before: Any, after: Any, atol: float = 1e-9) -> Tuple[bool, dict]:
    report: dict = {"ok": True, "max_abs_diff": 0.0, "issues": []}
    if before == after:
        return True, report
    try:
        b_states, a_states = _collect_states(before), _collect_states(after)
        if len(b_states) != len(a_states):
            report["ok"] = False
            report["issues"].append(f"state_list_len {len(b_states)} != {len(a_states)}")
            return False, report
        max_diff = 0.0
        for bs, as_ in zip(b_states, a_states):
            if len(bs) != len(as_):
                report["ok"] = False
                report["issues"].append(f"state_dim {len(bs)} != {len(as_)}")
                return False, report
            d = float(np.max(np.abs(np.asarray(bs, dtype=np.float64) - np.asarray(as_, dtype=np.float64))))
            max_diff = max(max_diff, d)
        report["max_abs_diff"] = max_diff
        if max_diff > atol:
            report["ok"] = False
            report["issues"].append(f"max_abs_diff {max_diff} > {atol}")
        return report["ok"], report
    except Exception as e:
        report["ok"] = False
        report["issues"].append(f"compare_failed: {e}")
        return False, report


def summarize_render_quality(sample: dict, quality_field: str = "hand_to_robot_render_quality") -> dict:
    meta = sample.get(Fields.meta, {}) or {}
    q = meta.get(quality_field, {}) or {}
    summary = dict(q.get("summary", {}) or {})
    frames = q.get("frames", []) or []
    summary["num_quality_records"] = len(frames)
    flags = [f.get("quality_flag") for f in frames]
    if flags:
        summary["ok_frame_rate"] = float(np.mean([f == "ok" for f in flags]))
    return summary


def read_lerobot_info_json(dataset_dir: Path) -> Optional[dict]:
    p = dataset_dir / "meta" / "info.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))
