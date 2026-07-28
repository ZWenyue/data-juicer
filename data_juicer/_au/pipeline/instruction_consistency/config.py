"""Instruction consistency checker configuration."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CheckerConfig:
    """Configuration for the Instruction Consistency Checker."""

    # Stage 2: primary model for structured reasoning
    primary_model: str = "composer-2.5"

    # Stage 3: expert models for multi-expert adjudication
    expert_models: list[str] = field(
        default_factory=lambda: [
            "composer-2.5",
            "composer-2.5",
            "composer-2.5",
        ]
    )

    # voting strategy: "majority" | "weighted" | "unanimous_override"
    voting_strategy: str = "majority"

    # confidence threshold below which Stage 3 is triggered
    confidence_threshold: float = 0.7

    # number of frames to extract per segment
    num_frames: int = 8

    # retry count for API calls
    try_num: int = 3

    # Stage 1 segmentation parameters (defaults tuned for ~15Hz demos)
    speed_smooth_window: int = 11
    min_window: int = 30
    min_segment_frames: int = 45
    max_segment_frames: int = 300
    max_segments: int = 8

    # Episode-level aggregation: consistent unless inconsistent ratio exceeds this.
    # (avoids one idle/camera-shift clip vetoing a long demonstration)
    inconsistent_ratio_threshold: float = 0.35

    # cost / safety limits
    max_samples: int | None = None
    max_cost_usd: float | None = None

    # working directory for local agent execution
    cwd: str = "."

    # use Cursor Cloud (True) or local execution (False)
    use_cloud: bool = False
