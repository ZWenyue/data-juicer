# -*- coding: utf-8 -*-
"""Build a data-juicer YAML recipe for robot cleaning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import yaml
from loguru import logger

from .config import CleanConfig


def read_source_fps(dataset: str, default: float = 15.0) -> float:
    """Read ``fps`` from ``<dataset>/meta/info.json``; fall back to ``default``."""
    info = Path(dataset) / "meta" / "info.json"
    try:
        with open(info, encoding="utf-8") as f:
            fps = json.load(f).get("fps")
        if fps:
            return float(fps)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        pass
    logger.warning(f"Could not read fps from {info}; using default {default}.")
    return float(default)


def build_process_ops(cfg: CleanConfig) -> List[Dict[str, Any]]:
    """Compose the ``process:`` operator list according to stage toggles."""
    stats_dir = cfg.work_dir / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    ops: List[Dict[str, Any]] = []

    ops.append(
        {
            "robot_lerobot_parquet_loader_mapper": {
                "parquet_field": "parquet_path",
                "state_key": "states",
                "action_key": "actions",
                "skip_if_present": True,
            }
        }
    )

    if cfg.enable_stage1:
        ops.append(
            {
                "robot_sudden_change_filter": {
                    "signal_source": "top_level",
                    "top_level_state_key": "states",
                    "top_level_action_key": "actions",
                    "threshold_mode": "mad",
                    "mad_scale_residual": 6.0,
                    "mad_scale_acc": 6.0,
                    "mad_scale_jerk": 6.0,
                    "max_flagged_ratio": cfg.s1_max_flagged_ratio,
                    "max_run_length": cfg.s1_max_run_length,
                    "min_frames": cfg.s1_min_frames,
                    "exclusion_strategy": cfg.s1_exclusion_strategy,
                    "stats_export_path": str(stats_dir / "s1_stats.jsonl"),
                }
            }
        )

    if cfg.enable_stage2:
        ops.append(
            {
                "robot_state_action_alignment_filter": {
                    "signal_source": "top_level",
                    "top_level_state_key": "states",
                    "top_level_action_key": "actions",
                    "shared_dims": list(cfg.s2_shared_dims),
                    "action_is_delta": False,
                    "max_lag": 15,
                    "da_threshold": cfg.s2_da_threshold,
                    "eps_mode": "range_frac",
                    "eps_frac": 0.01,
                    "min_active_frames": 10,
                    "min_frames": 20,
                    "exclusion_strategy": cfg.s2_exclusion_strategy,
                    "stats_export_path": str(stats_dir / "s2_stats.jsonl"),
                }
            }
        )

    if cfg.enable_stage3:
        ops.append(
            {
                "robot_extreme_value_filter": {
                    "signal_source": "top_level",
                    "top_level_state_key": "states",
                    "top_level_action_key": "actions",
                    "percentile_source": "stats_json",
                    "percentile_stats_path": str(cfg.resolved_percentiles_path),
                    "embodiment": cfg.embodiment,
                    "alpha": cfg.s3_alpha,
                    "exempt_dims": list(cfg.s3_exempt_dims),
                    "exclusion_strategy": cfg.s3_exclusion_strategy,
                    "max_flagged_ratio": cfg.s3_max_flagged_ratio,
                    "stats_export_path": str(stats_dir / "s3_stats.jsonl"),
                }
            }
        )

    if cfg.enable_stage5:
        ops.append(
            {
                "robot_base_frame_alignment_mapper": {
                    "signal_source": "top_level",
                    "top_level_state_key": "states",
                    "top_level_action_key": "actions",
                    "correction_source": "preset",
                    "preset_name": "identity",
                    "report_field": "base_frame_alignment_report",
                }
            }
        )

    if cfg.enable_check3:
        scorer_kwargs: Dict[str, Any] = {
            "blackness_threshold": cfg.check3_blackness_threshold,
            "blur_threshold": cfg.check3_blur_threshold,
            "corrupt_check_enabled": True,
            "video_field_index": 0,
            "decoder": cfg.check3_decoder,
            "report_field": "frame_quality_report",
        }
        if cfg.check3_sampling_fps is not None:
            scorer_kwargs["sampling_fps"] = cfg.check3_sampling_fps
            scorer_kwargs["original_fps"] = read_source_fps(cfg.dataset)
        ops.append({"robot_frame_quality_scorer_mapper": scorer_kwargs})
        ops.append(
            {
                "robot_key_frame_detector_mapper": {
                    "signal_source": "top_level",
                    "top_level_state_key": "states",
                    "top_level_action_key": "actions",
                    "gripper_delta_threshold": 5.0,
                    "gripper_close_direction": "decrease",
                    "state_velocity_percentile": 95.0,
                    "keyframe_window": 3,
                    "report_field": "key_frame_report",
                }
            }
        )
        ops.append(
            {
                "robot_video_quality_episode_filter": {
                    "quality_report_field": "frame_quality_report",
                    "keyframe_report_field": "key_frame_report",
                    "max_bad_ratio": cfg.check3_max_bad_ratio,
                    "min_good_frames": cfg.check3_min_good_frames,
                    "max_keyframe_overlap": cfg.check3_max_keyframe_overlap,
                    "report_field": "video_quality_episode_report",
                }
            }
        )

    if cfg.enable_unified:
        ops.append(
            {
                "robot_unified_state_mapper": {
                    "embodiment": cfg.embodiment,
                    "parquet_field": "parquet_path",
                    "state_key": "unified_states",
                    "action_key": "unified_actions",
                    "mask_key": "unified_dim_mask",
                    "skip_if_present": False,
                    "write_meta_occupancy": True,
                }
            }
        )

    return ops


def write_recipe(cfg: CleanConfig) -> Path:
    """Materialize a YAML recipe under ``cfg.work_dir`` and return its path."""
    recipe = {
        "project_name": "robot-clean",
        "dataset_path": str(cfg.pointer_path.resolve()),
        "export_path": str(cfg.result_path.resolve()),
        "np": int(cfg.np),
        "executor_type": cfg.executor_type,
        "keep_stats_in_res_ds": True,
        "text_keys": "id",
        "custom_operator_paths": ["data_juicer/_au"],
        "process": build_process_ops(cfg),
    }
    path = cfg.recipe_path
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(recipe, f, sort_keys=False, allow_unicode=True)
    return path
