"""Stage 3: Multi-expert cross-model adjudication via parallel Cursor Agents."""
from __future__ import annotations

import asyncio

from loguru import logger

from .frame_utils import extract_frames_base64, frames_to_sdk_images
from .parse_utils import parse_vlm_json
from .prompts import SYSTEM_PROMPT, build_user_prompt
from .vote import aggregate_votes


def needs_adjudication(
    stage2_result: dict,
    confidence_threshold: float,
    expert_models: list[str],
) -> bool:
    """Determine if Stage 3 multi-expert adjudication is needed."""
    if not expert_models:
        return False

    verdict = stage2_result.get("overall_verdict", "inconsistent")
    confidence = stage2_result.get("overall_confidence", 0.0)

    if verdict == "consistent" and confidence >= confidence_threshold:
        return False

    return True


def resolve_segment_span(
    worst: dict,
    stage1_segments: list[dict] | None,
) -> tuple[int, int]:
    """Resolve (start_frame, end_frame) for a Stage-2 segment result.

    Stage-2 results may already carry spans; otherwise look up Stage-1
    segments by segment_id (the previous bug used defaults 0,0 → empty frames).
    """
    start = worst.get("start_frame")
    end = worst.get("end_frame")
    if start is not None and end is not None and int(end) > int(start):
        return int(start), int(end)

    sid = worst.get("segment_id")
    if stage1_segments and sid is not None:
        by_id = {s["segment_id"]: s for s in stage1_segments}
        if sid in by_id:
            s = by_id[sid]
            return int(s["start_frame"]), int(s["end_frame"])

    return 0, 0


def pick_worst_segment(stage2_segments: list[dict]) -> dict | None:
    """Prefer inconsistent clips; break ties by lowest confidence."""
    if not stage2_segments:
        return None

    def key(s: dict):
        is_bad = 0 if s.get("verdict") != "consistent" else 1
        conf = float(s.get("confidence", 0.0) or 0.0)
        return (is_bad, conf)

    return min(stage2_segments, key=key)


async def _ask_expert(
    client,
    model_id: str,
    prompt: str,
    images,
    api_key: str,
    cwd: str,
    try_num: int,
) -> dict | None:
    """Query a single expert VLM. Returns parsed result or None on failure."""
    from cursor_sdk import (
        AgentOptions,
        LocalAgentOptions,
        UserMessage,
    )

    for attempt in range(try_num):
        try:
            async with await client.agents.create(
                AgentOptions(
                    api_key=api_key,
                    model=model_id,
                    local=LocalAgentOptions(cwd=cwd),
                )
            ) as agent:
                run = await agent.send(UserMessage(text=prompt, images=images))
                result = await run.wait()
                parsed = parse_vlm_json(result.result)
                parsed["model"] = model_id
                return parsed
        except Exception as e:
            logger.warning(f"Expert {model_id} attempt {attempt + 1}/{try_num}: {e}")
            if attempt < try_num - 1:
                await asyncio.sleep(2**attempt)

    logger.error(f"Expert {model_id} failed after {try_num} attempts")
    return None


async def run_stage3(
    client,
    video_path: str,
    stage2_result: dict,
    instruction: str,
    expert_models: list[str],
    voting_strategy: str,
    num_frames: int,
    api_key: str,
    cwd: str,
    try_num: int = 3,
    stage1_segments: list[dict] | None = None,
) -> dict:
    """Run Stage 3 multi-expert adjudication.

    Parallel-queries each expert VLM and aggregates votes.
    """
    stage2_segments = stage2_result.get("segments", [])
    worst = pick_worst_segment(stage2_segments)
    if worst is None:
        return {
            "final_verdict": "inconsistent",
            "final_score": 0.0,
            "adjudication_needed": True,
            "error": "no_segments",
        }

    start_frame, end_frame = resolve_segment_span(worst, stage1_segments)
    if end_frame <= start_frame:
        logger.error(
            f"Stage 3 could not resolve frames for segment_id="
            f"{worst.get('segment_id')} (start={start_frame}, end={end_frame})"
        )
        return {
            "final_verdict": "ambiguous",
            "final_score": 0.0,
            "adjudication_needed": True,
            "error": "unresolved_segment_span",
            "worst_segment_id": worst.get("segment_id"),
        }

    frames_b64 = extract_frames_base64(video_path, start_frame, end_frame, num_frames)
    if not frames_b64:
        logger.error(
            f"Stage 3 extracted 0 frames for segment_id={worst.get('segment_id')} "
            f"span=[{start_frame},{end_frame})"
        )
        return {
            "final_verdict": "ambiguous",
            "final_score": 0.0,
            "adjudication_needed": True,
            "error": "no_frames",
            "worst_segment_id": worst.get("segment_id"),
            "start_frame": start_frame,
            "end_frame": end_frame,
        }

    images = frames_to_sdk_images(frames_b64)
    prompt = f"{SYSTEM_PROMPT}\n\n{build_user_prompt(instruction, len(frames_b64))}"

    votes = await asyncio.gather(
        *[
            _ask_expert(client, m, prompt, images, api_key, cwd, try_num)
            for m in expert_models
        ]
    )
    valid_votes = [v for v in votes if v is not None]

    if len(valid_votes) < 2:
        logger.warning(
            f"Only {len(valid_votes)} experts responded (need >= 2). Marking as ambiguous."
        )
        return {
            "final_verdict": "ambiguous",
            "final_score": 0.0,
            "adjudication_needed": True,
            "expert_votes": valid_votes,
            "num_experts_responded": len(valid_votes),
            "worst_segment_id": worst.get("segment_id"),
            "start_frame": start_frame,
            "end_frame": end_frame,
        }

    result = aggregate_votes(valid_votes, voting_strategy)
    result["adjudication_needed"] = True
    result["worst_segment_id"] = worst.get("segment_id")
    result["start_frame"] = start_frame
    result["end_frame"] = end_frame
    result["num_frames_sent"] = len(frames_b64)
    return result
