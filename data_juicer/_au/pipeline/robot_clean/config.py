# -*- coding: utf-8 -*-
"""Configuration for the production robot clean pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class CleanConfig:
    """End-to-end LeRobot → cleaned JSONL (+ optional 80-dim parquet) settings."""

    dataset: str
    output_dir: str
    embodiment: str = "galaxea_r1_lite"
    video_key: str = "observation.images.head_rgb"

    # Episode / parallelism
    max_episodes: Optional[int] = None
    np: int = 1
    executor_type: str = "default"

    # Stage toggles (defaults = full clean)
    enable_stage1: bool = True
    enable_stage2: bool = True
    enable_stage3: bool = True
    enable_stage5: bool = True
    enable_check3: bool = True
    enable_unified: bool = True
    export_unified_parquet: bool = False

    # Stage 1
    s1_max_flagged_ratio: float = 0.3
    s1_max_run_length: int = 10
    s1_min_frames: int = 30
    s1_exclusion_strategy: str = "episode_discard"

    # Stage 2
    s2_shared_dims: List[int] = field(
        default_factory=lambda: [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14]
    )
    s2_da_threshold: float = 0.65
    s2_exclusion_strategy: str = "episode_discard"

    # Stage 3
    s3_alpha: float = 0.1
    s3_max_flagged_ratio: float = 0.3
    s3_exempt_dims: List[int] = field(default_factory=lambda: [7, 15])
    s3_exclusion_strategy: str = "episode_discard"
    # If set, reuse this file instead of recomputing percentiles.
    percentiles_path: Optional[str] = None

    # Check 3
    check3_max_bad_ratio: float = 0.1
    check3_min_good_frames: int = 20
    # Keyframe overlap remains observable in reports, but is not a rejection
    # gate by default.  A zero-tolerance gate produced many false positives on
    # otherwise healthy simulation videos.
    check3_max_keyframe_overlap: Optional[int] = None
    check3_gripper_dims: List[int] = field(default_factory=lambda: [7, 15])
    check3_gripper_delta_threshold_frac: float = 0.05
    check3_decoder: str = "auto"
    check3_blackness_threshold: float = 10.0
    # Conservative fallback when no dataset-specific analysis is available.
    check3_blur_threshold: float = 1.0
    # Optional subsample during scoring (None = every frame).
    check3_sampling_fps: Optional[float] = None

    # Work artefacts (filled by runner)
    @property
    def work_dir(self) -> Path:
        return Path(self.output_dir)

    @property
    def pointer_path(self) -> Path:
        return self.work_dir / "pointer.jsonl"

    @property
    def result_path(self) -> Path:
        return self.work_dir / "cleaned.jsonl"

    @property
    def recipe_path(self) -> Path:
        return self.work_dir / "recipe.yaml"

    @property
    def resolved_percentiles_path(self) -> Path:
        if self.percentiles_path:
            return Path(self.percentiles_path)
        return self.work_dir / "percentiles.json"

    @property
    def unified_parquet_dir(self) -> Path:
        return self.work_dir / "unified80_lerobot"
