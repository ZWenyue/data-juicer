#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Summarize Stage 1/2/3 analyze stats and suggest cleaning thresholds."""

import argparse
import json
import os
from typing import Dict, Iterable, List

import numpy as np

S1_KEYS = ["sudden_change_flagged_ratio", "sudden_change_max_run"]
S2_KEYS = ["state_action_min_da", "state_action_mean_da"]
S3_KEYS = ["extreme_value_flagged_ratio"]
REQUIRED_KEYS = S1_KEYS + S2_KEYS + S3_KEYS
DA_CANDIDATES = (0.60, 0.65, 0.70)
TARGET_KEEP = 0.90


def _stat_dict(row: dict) -> dict:
    stats = {}
    if isinstance(row.get("__dj__stats__"), dict):
        stats.update(row["__dj__stats__"])
    for key in REQUIRED_KEYS:
        if key in row:
            stats.setdefault(key, row[key])
    return stats if stats else row


def load_stats_jsonl_raw(path: str) -> List[dict]:
    """Load JSONL rows without requiring all stage stats keys."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows


def load_stats_jsonl(path: str) -> List[dict]:
    """Load JSONL stats and validate Stage 1/2/3 keys."""
    rows = load_stats_jsonl_raw(path)
    missing = []
    for i, row in enumerate(rows):
        stats = _stat_dict(row)
        for key in REQUIRED_KEYS:
            if key not in stats:
                missing.append(f"row{i}:{key}")
    if missing:
        raise ValueError("Missing stats keys: " + ", ".join(missing[:20]))
    return rows


def merge_stats_jsonl(paths: Iterable[str]) -> List[dict]:
    """Merge per-stage stats JSONL files by id, updating __dj__stats__."""
    by_id = {}
    for path in paths:
        for row in load_stats_jsonl_raw(path):
            eid = row.get("id")
            if eid is None:
                raise ValueError(f"Missing id in {path}")
            by_id.setdefault(eid, {"id": eid, "__dj__stats__": {}})
            by_id[eid]["__dj__stats__"].update(_stat_dict(row))
    return list(by_id.values())


def _col(rows, key):
    return np.array([float(_stat_dict(row)[key]) for row in rows], dtype=float)


def _pcts(x):
    return {
        "min": float(np.min(x)),
        "p50": float(np.percentile(x, 50)),
        "p90": float(np.percentile(x, 90)),
        "p95": float(np.percentile(x, 95)),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
    }


def suggest_thresholds(rows: List[dict], probe_alpha: float = 0.1) -> Dict:
    if not rows:
        raise ValueError("No rows to summarize")

    n = len(rows)
    ratio = _col(rows, "sudden_change_flagged_ratio")
    max_flagged_ratio = float(np.percentile(ratio, TARGET_KEEP * 100))
    max_flagged_ratio = float(np.clip(max_flagged_ratio, 0.05, 0.5))

    min_da = _col(rows, "state_action_min_da")
    best_da = DA_CANDIDATES[0]
    best_score = None
    da_table = []
    for cand in DA_CANDIDATES:
        drop = int(np.sum(min_da < cand))
        drop_frac = drop / n
        da_table.append({"da_threshold": cand, "drop": drop, "drop_frac": drop_frac})
        score = abs(drop_frac - 0.10)
        if best_score is None or score < best_score:
            best_score = score
            best_da = cand

    extreme_value_ratio = _col(rows, "extreme_value_flagged_ratio")
    mean_extreme_value_ratio = float(np.mean(extreme_value_ratio))
    if mean_extreme_value_ratio > 0.15:
        alpha = 0.2 if mean_extreme_value_ratio <= 0.25 else 0.3
    else:
        alpha = probe_alpha

    return {
        "n_episodes": n,
        "max_flagged_ratio": round(max_flagged_ratio, 4),
        "da_threshold": best_da,
        "alpha": alpha,
        "s1_ratio_pcts": _pcts(ratio),
        "s1_max_run_pcts": _pcts(_col(rows, "sudden_change_max_run")),
        "s2_min_da_pcts": _pcts(min_da),
        "s2_mean_da_pcts": _pcts(_col(rows, "state_action_mean_da")),
        "s2_da_table": da_table,
        "s3_ev_pcts": _pcts(extreme_value_ratio),
        "s3_mean_flagged_ratio": mean_extreme_value_ratio,
        "s1_keep_frac_if_discard": float(np.mean(ratio <= max_flagged_ratio)),
        "s2_keep_frac": 1.0 - next(t["drop_frac"] for t in da_table if t["da_threshold"] == best_da),
    }


def _metric_table_row(name, pcts):
    return (
        f"| {name} | {pcts['min']:.4g} | {pcts['p50']:.4g} | {pcts['p90']:.4g} | "
        f"{pcts['p95']:.4g} | {pcts['max']:.4g} | {pcts['mean']:.4g} |\n"
    )


def write_report(rows, sug, md_path, yaml_path, probe_alpha=0.1):
    os.makedirs(os.path.dirname(md_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(yaml_path) or ".", exist_ok=True)

    out_dir = os.path.dirname(os.path.abspath(yaml_path)).replace("\\", "/")
    # Prefer repo-relative paths in suggested recipe when under tests_au/
    marker = "tests_au/ops/filter/outputs/"
    if marker in out_dir:
        rel_out = marker + out_dir.split(marker, 1)[1]
    else:
        rel_out = out_dir
    dataset_path = f"{rel_out}/lerobot_episodes_ptr.jsonl"
    if not os.path.isfile(os.path.join(os.path.dirname(yaml_path), "lerobot_episodes_ptr.jsonl")):
        dataset_path = f"{rel_out}/lerobot_episodes.jsonl"
    percentiles_path = f"{rel_out}/percentiles.json"
    export_path = f"{rel_out}/clean_result.jsonl"

    n = sug["n_episodes"]
    scope_note = (
        f"> 基于本次分析的 **{n}** 条 episode 统计；若尚未覆盖全部任务，生产清洗前请复核。\n"
    )

    lines = []
    lines.append("# Galaxea Threshold Report\n")
    lines.append(scope_note)
    lines.append(f"- Episodes analyzed: **{n}**\n")
    lines.append(f"- Stage 3 probe alpha used in analyze: **{probe_alpha}**\n")

    lines.append("\n## Stage 1 - Sudden change\n")
    lines.append("| metric | min | p50 | p90 | p95 | max | mean |\n|---|---|---|---|---|---|---|\n")
    lines.append(_metric_table_row("flagged_ratio", sug["s1_ratio_pcts"]))
    lines.append(_metric_table_row("max_run", sug["s1_max_run_pcts"]))
    lines.append(
        f"\n**建议 `max_flagged_ratio` = `{sug['max_flagged_ratio']}`** "
        f"（若使用 episode discard，预计保留率约 {sug['s1_keep_frac_if_discard']:.1%}）。\n"
    )
    lines.append("当前分析建议先用 `frame_mask` 观察被标记帧，再决定是否升级为整条 episode 丢弃。\n")

    lines.append("\n## Stage 2 - State-action alignment\n")
    p = sug["s2_min_da_pcts"]
    mean_p = sug["s2_mean_da_pcts"]
    lines.append(
        f"min_da: p50={p['p50']:.4f}, p90={p['p90']:.4f}, mean={p['mean']:.4f}; "
        f"mean_da: p50={mean_p['p50']:.4f}, mean={mean_p['mean']:.4f}\n\n"
    )
    lines.append("| da_threshold | drop eps | drop frac |\n|---|---|---|\n")
    for item in sug["s2_da_table"]:
        lines.append(f"| {item['da_threshold']:.2f} | {item['drop']} | {item['drop_frac']:.1%} |\n")
    lines.append(
        f"\n**建议 `da_threshold` = `{sug['da_threshold']}`** "
        f"（候选阈值中最接近 10% episode drop，预计保留率约 {sug['s2_keep_frac']:.1%}）。\n"
    )

    lines.append("\n## Stage 3 - Extreme value\n")
    p = sug["s3_ev_pcts"]
    lines.append(
        f"extreme_value_flagged_ratio: mean={sug['s3_mean_flagged_ratio']:.4f}, "
        f"p50={p['p50']:.4f}, p90={p['p90']:.4f}, p95={p['p95']:.4f}\n"
    )
    lines.append(
        f"\n**建议 `alpha` = `{sug['alpha']}`** "
        "（目标是把帧标记率稳定在约 5%-15%；夹爪维继续使用 `exempt_dims: [7, 15]`）。\n"
    )

    lines.append("\n## Suggested clean params\n")
    lines.append("```yaml\n")
    lines.append(f"max_flagged_ratio: {sug['max_flagged_ratio']}\n")
    lines.append(f"da_threshold: {sug['da_threshold']}\n")
    lines.append(f"alpha: {sug['alpha']}\n")
    lines.append("```\n")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("".join(lines))

    yaml_text = f"""# Auto-suggested from analyze; REVIEW before production use.
project_name: 'galaxea-clean-suggested'
dataset_path: '{dataset_path}'
export_path: '{export_path}'
np: 4
executor_type: default
keep_stats_in_res_ds: true
text_keys: 'id'

custom_operator_paths:
  - 'data_juicer/_au'

process:
  - robot_lerobot_parquet_loader_mapper:
      parquet_field: 'parquet_path'
      state_key: 'states'
      action_key: 'actions'
  - robot_sudden_change_filter:
      signal_source: 'top_level'
      top_level_state_key: 'states'
      top_level_action_key: 'actions'
      threshold_mode: 'mad'
      mad_scale_residual: 6.0
      mad_scale_acc: 6.0
      mad_scale_jerk: 6.0
      max_flagged_ratio: {sug['max_flagged_ratio']}
      max_run_length: 10
      min_frames: 30
      exclusion_strategy: 'frame_mask'
  - robot_state_action_alignment_filter:
      signal_source: 'top_level'
      top_level_state_key: 'states'
      top_level_action_key: 'actions'
      shared_dims: [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14]
      action_is_delta: false
      max_lag: 15
      da_threshold: {sug['da_threshold']}
      eps_mode: 'range_frac'
      eps_frac: 0.01
      min_active_frames: 10
      min_frames: 20
      exclusion_strategy: 'flag_only'
  - robot_extreme_value_filter:
      signal_source: 'top_level'
      top_level_state_key: 'states'
      top_level_action_key: 'actions'
      percentile_source: 'stats_json'
      percentile_stats_path: '{percentiles_path}'
      embodiment: 'galaxea_r1_lite'
      alpha: {sug['alpha']}
      exempt_dims: [7, 15]
      exclusion_strategy: 'frame_mask'
"""
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_text)


def main():
    parser = argparse.ArgumentParser(description="Summarize pilot analyze stats into report and suggested recipe.")
    parser.add_argument("--stats", required=True, help="JSONL with __dj__stats__ or top-level stats keys")
    parser.add_argument("--report", required=True)
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--probe-alpha", type=float, default=0.1)
    args = parser.parse_args()

    rows = load_stats_jsonl(args.stats)
    suggestions = suggest_thresholds(rows, probe_alpha=args.probe_alpha)
    write_report(rows, suggestions, args.report, args.recipe, probe_alpha=args.probe_alpha)


if __name__ == "__main__":
    main()
