"""Main orchestration script for Instruction Consistency Check.

Usage:
    python -m data_juicer._au.pipeline.instruction_consistency.run_instruction_consistency \
        --video /path/to/video.mp4 \
        --instruction "pick up the red cup" \
        --positions /path/to/positions.npy
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os

import numpy as np
from loguru import logger

from .config import CheckerConfig
from .segmentation import segment_episode
from .stage2_evaluator import run_stage2
from .stage3_adjudicator import needs_adjudication, run_stage3


async def check_single_sample(
    video_path: str,
    instruction: str,
    positions: np.ndarray,
    config: CheckerConfig,
) -> dict:
    """Run the full 3-stage pipeline on a single sample."""
    from cursor_sdk import AsyncClient

    api_key = os.environ.get("CURSOR_API_KEY")
    if not api_key:
        raise RuntimeError("CURSOR_API_KEY environment variable not set")

    logger.info("Stage 1: Segmenting episode...")
    segments = segment_episode(
        positions,
        smooth_window=config.speed_smooth_window,
        min_window=config.min_window,
        min_frames=config.min_segment_frames,
        max_frames=config.max_segment_frames,
        max_segments=config.max_segments,
    )
    logger.info(f"  → {len(segments)} segments")

    async with await AsyncClient.launch_bridge(workspace=config.cwd) as client:
        logger.info(f"Stage 2: Evaluating with {config.primary_model}...")
        stage2_result = await run_stage2(
            client,
            video_path,
            segments,
            instruction,
            config.primary_model,
            config.num_frames,
            api_key,
            config.cwd,
            config.try_num,
            inconsistent_ratio_threshold=config.inconsistent_ratio_threshold,
        )
        logger.info(
            f"  → verdict={stage2_result['overall_verdict']}, "
            f"conf={stage2_result['overall_confidence']}, "
            f"inconsistent_ratio={stage2_result.get('inconsistent_ratio')}"
        )

        if needs_adjudication(
            stage2_result, config.confidence_threshold, config.expert_models
        ):
            logger.info(
                f"Stage 3: Multi-expert adjudication with "
                f"{len(config.expert_models)} models..."
            )
            stage3_result = await run_stage3(
                client,
                video_path,
                stage2_result,
                instruction,
                config.expert_models,
                config.voting_strategy,
                config.num_frames,
                api_key,
                config.cwd,
                config.try_num,
                stage1_segments=segments,
            )
            logger.info(
                f"  → final_verdict={stage3_result['final_verdict']}, "
                f"score={stage3_result.get('final_score')}, "
                f"span=[{stage3_result.get('start_frame')},"
                f"{stage3_result.get('end_frame')})"
            )
        else:
            logger.info("Stage 3: Skipped (high-confidence consistent)")
            stage3_result = {
                "final_verdict": stage2_result["overall_verdict"],
                "final_score": stage2_result["overall_confidence"],
                "adjudication_needed": False,
            }

    return {
        "video_path": video_path,
        "instruction": instruction,
        "num_segments": len(segments),
        "segments": segments,
        "stage2": stage2_result,
        "stage3": stage3_result,
        "final_verdict": stage3_result["final_verdict"],
        "is_consistent": stage3_result["final_verdict"] == "consistent",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Instruction Consistency Check (Cursor SDK)"
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument(
        "--positions", required=True, help="Path to .npy file with EEF positions"
    )
    parser.add_argument("--primary-model", default="composer-2.5")
    parser.add_argument(
        "--expert-models",
        nargs="+",
        default=["composer-2.5", "composer-2.5"],
    )
    parser.add_argument(
        "--voting-strategy",
        default="majority",
        choices=["majority", "weighted", "unanimous_override"],
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.7)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--min-window", type=int, default=None)
    parser.add_argument("--min-segment-frames", type=int, default=None)
    parser.add_argument("--max-segments", type=int, default=None)
    parser.add_argument("--inconsistent-ratio-threshold", type=float, default=None)
    parser.add_argument("--output", default="result.json")
    args = parser.parse_args()

    config = CheckerConfig(
        primary_model=args.primary_model,
        expert_models=args.expert_models,
        voting_strategy=args.voting_strategy,
        confidence_threshold=args.confidence_threshold,
        num_frames=args.num_frames,
        cwd=os.getcwd(),
    )
    if args.min_window is not None:
        config.min_window = args.min_window
    if args.min_segment_frames is not None:
        config.min_segment_frames = args.min_segment_frames
    if args.max_segments is not None:
        config.max_segments = args.max_segments
    if args.inconsistent_ratio_threshold is not None:
        config.inconsistent_ratio_threshold = args.inconsistent_ratio_threshold

    positions = np.load(args.positions)
    result = asyncio.run(
        check_single_sample(args.video, args.instruction, positions, config)
    )

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info(f"Result saved to {args.output}")
    logger.info(f"Final verdict: {result['final_verdict']}")


if __name__ == "__main__":
    main()
