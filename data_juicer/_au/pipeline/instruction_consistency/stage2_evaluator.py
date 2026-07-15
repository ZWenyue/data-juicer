"""Stage 2: Structured reasoning VLM evaluation via Cursor Agent."""
from __future__ import annotations

import asyncio

from loguru import logger

from .frame_utils import extract_frames_base64, frames_to_sdk_images
from .parse_utils import parse_vlm_json
from .prompts import SYSTEM_PROMPT, build_user_prompt


async def evaluate_segment(
    client,
    video_path: str,
    segment: dict,
    instruction: str,
    model_id: str,
    num_frames: int,
    api_key: str,
    cwd: str,
    try_num: int = 3,
) -> dict:
    """Evaluate a single segment's instruction consistency via VLM.

    Returns parsed 4-dimension analysis dict with verdict + confidence,
    plus start_frame / end_frame for Stage 3 re-evaluation.
    """
    from cursor_sdk import (
        AgentOptions,
        LocalAgentOptions,
        UserMessage,
    )

    meta = {
        "segment_id": segment["segment_id"],
        "start_frame": int(segment["start_frame"]),
        "end_frame": int(segment["end_frame"]),
        "model": model_id,
    }

    frames_b64 = extract_frames_base64(
        video_path, segment["start_frame"], segment["end_frame"], num_frames
    )
    if not frames_b64:
        return {
            "verdict": "inconsistent",
            "confidence": 0.0,
            "error": "no_frames",
            **meta,
        }

    images = frames_to_sdk_images(frames_b64)
    user_text = f"{SYSTEM_PROMPT}\n\n{build_user_prompt(instruction, len(frames_b64))}"

    for attempt in range(try_num):
        try:
            async with await client.agents.create(
                AgentOptions(
                    api_key=api_key,
                    model=model_id,
                    local=LocalAgentOptions(cwd=cwd),
                )
            ) as agent:
                run = await agent.send(UserMessage(text=user_text, images=images))
                result = await run.wait()
                parsed = parse_vlm_json(result.result)
                parsed.update(meta)
                return parsed
        except Exception as e:
            logger.warning(f"Stage 2 attempt {attempt + 1}/{try_num} failed: {e}")
            if attempt < try_num - 1:
                await asyncio.sleep(2**attempt)

    return {
        "verdict": "inconsistent",
        "confidence": 0.0,
        "error": "all_retries_failed",
        **meta,
    }


def aggregate_stage2(
    results: list[dict],
    inconsistent_ratio_threshold: float = 0.35,
) -> dict:
    """Aggregate per-segment verdicts into an episode-level decision."""
    n = len(results)
    if n == 0:
        return {
            "segments": [],
            "overall_verdict": "inconsistent",
            "overall_confidence": 0.0,
            "inconsistent_ratio": 1.0,
            "num_inconsistent": 0,
        }

    verdicts = [r.get("verdict", "inconsistent") for r in results]
    confidences = [float(r.get("confidence", 0.0) or 0.0) for r in results]
    num_bad = sum(1 for v in verdicts if v != "consistent")
    ratio = num_bad / n

    if ratio > inconsistent_ratio_threshold:
        overall = "inconsistent"
        # confidence among inconsistent segments (how sure we are they are bad)
        bad_conf = [c for v, c in zip(verdicts, confidences) if v != "consistent"]
        overall_conf = float(min(bad_conf)) if bad_conf else 0.0
    else:
        overall = "consistent"
        good_conf = [c for v, c in zip(verdicts, confidences) if v == "consistent"]
        overall_conf = float(min(good_conf)) if good_conf else 0.0

    return {
        "segments": results,
        "overall_verdict": overall,
        "overall_confidence": round(overall_conf, 4),
        "inconsistent_ratio": round(ratio, 4),
        "num_inconsistent": int(num_bad),
    }


async def run_stage2(
    client,
    video_path: str,
    segments: list[dict],
    instruction: str,
    model_id: str,
    num_frames: int,
    api_key: str,
    cwd: str,
    try_num: int = 3,
    inconsistent_ratio_threshold: float = 0.35,
) -> dict:
    """Run Stage 2 on all segments and aggregate results."""
    results = []
    for seg in segments:
        r = await evaluate_segment(
            client,
            video_path,
            seg,
            instruction,
            model_id,
            num_frames,
            api_key,
            cwd,
            try_num,
        )
        results.append(r)

    out = aggregate_stage2(results, inconsistent_ratio_threshold)
    out["model"] = model_id
    return out
