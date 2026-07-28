"""Robust JSON parsing for VLM responses."""
from __future__ import annotations

import json
import re

from loguru import logger

REQUIRED_DIMENSIONS = [
    "objects", "action_semantics",
    "temporal_ordering", "agent_environment_interaction",
]


def parse_vlm_json(response: str) -> dict:
    """Parse structured JSON from a VLM response string.

    Handles: raw JSON, markdown code blocks, JSON embedded in text.
    Falls back to a default "inconsistent" result on parse failure.
    """
    if not response:
        return _default_result("empty response")

    # 1. Direct JSON parse
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    # 2. Extract from markdown code block
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```",
                      response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # 3. Find first { ... last }
    start = response.find("{")
    end = response.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(response[start:end + 1])
        except json.JSONDecodeError:
            pass

    return _default_result(f"unparseable: {response[:200]}")


def _default_result(reason: str) -> dict:
    logger.warning(f"VLM parse failed: {reason}")
    return {
        "verdict": "inconsistent",
        "confidence": 0.0,
        "parse_error": True,
        "error_reason": reason,
    }


def validate_dimensions(parsed: dict) -> bool:
    """Check that all 4 analysis dimensions are present."""
    return all(d in parsed for d in REQUIRED_DIMENSIONS)
