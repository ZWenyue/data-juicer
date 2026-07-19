# -*- coding: utf-8 -*-
"""Resolve structured embodiment-prompt fields from LeRobot meta + embodiment YAML.

Writes raw fields plus discrete ``speed`` (ceil(length/500)*500). Prompt dropout
still belongs in the trainer.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

_QUALITY_TASK_LABELS = frozenset({"qualified", "unqualified", "null", "none", ""})
DEFAULT_SPEED_BIN = 500


def discretize_speed(length: int, bin_size: int = DEFAULT_SPEED_BIN) -> int:
    """Episode length → speed bin (multiples of ``bin_size``), paper-style.

    ``ceil(length / bin_size) * bin_size``; e.g. 8278 → 8500 with bin 500.
    """
    n = int(length)
    b = int(bin_size)
    if b <= 0:
        raise ValueError(f"bin_size must be positive, got {bin_size}")
    if n <= 0:
        raise ValueError(f"length must be positive to discretize speed, got {length}")
    return int(math.ceil(n / b) * b)


def task_root_from_parquet(parquet_path: Union[str, Path]) -> Path:
    """``.../task/data/chunk-XXX/episode_YYYYYY.parquet`` → ``.../task``."""
    p = Path(parquet_path).resolve()
    # episode → chunk → data → task
    if p.parent.parent.name == "data":
        return p.parent.parent.parent
    # fallback: walk up looking for meta/
    for anc in p.parents:
        if (anc / "meta").is_dir() and (anc / "data").is_dir():
            return anc
    raise FileNotFoundError(f"Cannot locate LeRobot task root for {parquet_path}")


def load_jsonl(path: Union[str, Path]) -> List[dict]:
    path = Path(path)
    if not path.is_file():
        return []
    rows: List[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_tasks_map(meta_dir: Union[str, Path]) -> Dict[int, str]:
    """``task_index → task`` from ``meta/tasks.jsonl``."""
    out: Dict[int, str] = {}
    for row in load_jsonl(Path(meta_dir) / "tasks.jsonl"):
        if "task_index" not in row:
            continue
        out[int(row["task_index"])] = str(row.get("task", "") or "")
    return out


def load_episodes_by_index(meta_dir: Union[str, Path]) -> Dict[int, dict]:
    out: Dict[int, dict] = {}
    for row in load_jsonl(Path(meta_dir) / "episodes.jsonl"):
        if "episode_index" not in row:
            continue
        out[int(row["episode_index"])] = row
    return out


def load_info_fps(meta_dir: Union[str, Path], default: float = 15.0) -> float:
    info_path = Path(meta_dir) / "info.json"
    if not info_path.is_file():
        return float(default)
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return float(default)
    fps = info.get("fps", default)
    try:
        return float(fps)
    except (TypeError, ValueError):
        return float(default)


def pick_instruction_lang(task: str, lang: str = "en") -> str:
    """Split ``中文@English``; ``lang=en`` keeps the side after ``@`` when present."""
    text = (task or "").strip()
    if "@" not in text:
        return text
    left, right = text.split("@", 1)
    left, right = left.strip(), right.strip()
    if lang.lower() in ("en", "english"):
        return right or left
    if lang.lower() in ("zh", "cn", "chinese"):
        return left or right
    return right or left


def _is_quality_label(text: str) -> bool:
    return text.strip().lower() in _QUALITY_TASK_LABELS


def resolve_instruction(
    tasks: Optional[Sequence[str]] = None,
    tasks_map: Optional[Dict[int, str]] = None,
    task_index: Optional[int] = None,
    *,
    lang: str = "en",
) -> str:
    """Pick a usable instruction string for one episode.

    Preference:
    1. First non-empty, non-quality label in ``tasks``
    2. ``tasks_map[task_index]``
    3. Empty string
    """
    if tasks:
        for t in tasks:
            if t is None:
                continue
            s = str(t).strip()
            if not s or _is_quality_label(s):
                continue
            return pick_instruction_lang(s, lang=lang)

    if tasks_map is not None and task_index is not None:
        raw = tasks_map.get(int(task_index))
        if raw and not _is_quality_label(str(raw)):
            return pick_instruction_lang(str(raw), lang=lang)

    return ""


def resolve_camera_view_direction(
    cfg: Dict[str, Any],
    video_key: Optional[str] = None,
) -> str:
    """Map a camera key → ``arm side`` / ``opposite side`` via embodiment YAML ``prompt``."""
    prompt = cfg.get("prompt") or {}
    views = prompt.get("camera_views") or {}
    if video_key:
        # exact match, then suffix match (basename)
        if video_key in views:
            return str(views[video_key])
        short = video_key.split("/")[-1]
        if short in views:
            return str(views[short])
        for k, v in views.items():
            if k.endswith(short) or short.endswith(k.split(".")[-1]):
                return str(v)
    default = prompt.get("default_camera_view_direction")
    if default:
        return str(default)
    return "opposite side"


def embodiment_id_from_cfg(cfg: Dict[str, Any]) -> str:
    prompt = cfg.get("prompt") or {}
    eid = prompt.get("embodiment_id")
    if eid:
        return str(eid)
    return str(cfg.get("name") or "unknown")


def resolve_prompt_fields(
    *,
    embodiment: str,
    instruction: str,
    length: Optional[int],
    fps: Optional[float],
    camera_view_direction: str,
    speed_bin_size: int = DEFAULT_SPEED_BIN,
) -> Dict[str, Any]:
    """Assemble prompt fields. Includes discrete ``speed`` from ``length``; no dropout."""
    fields: Dict[str, Any] = {
        "embodiment": str(embodiment),
        "instruction": str(instruction or ""),
        "camera_view_direction": str(camera_view_direction),
    }
    if length is not None:
        fields["length"] = int(length)
        if int(length) > 0:
            fields["speed"] = discretize_speed(int(length), bin_size=speed_bin_size)
    if fps is not None:
        fields["fps"] = float(fps) if float(fps) != int(fps) else int(fps)
    return fields


def format_prompt_fields_for_debug(fields: Dict[str, Any]) -> str:
    """Human-readable dump for logs/acceptance."""
    order = (
        "embodiment",
        "instruction",
        "speed",
        "length",
        "fps",
        "camera_view_direction",
    )
    lines = []
    for k in order:
        if k in fields:
            lines.append(f"{k}: {fields[k]}")
    for k, v in fields.items():
        if k not in order:
            lines.append(f"{k}: {v}")
    return "\n".join(lines)


def build_prompt_fields_for_episode(
    cfg: Dict[str, Any],
    episode_row: Optional[dict] = None,
    *,
    tasks_map: Optional[Dict[int, str]] = None,
    fps: Optional[float] = None,
    length: Optional[int] = None,
    task_index: Optional[int] = None,
    video_key: Optional[str] = None,
    instruction_lang: str = "en",
    speed_bin_size: int = DEFAULT_SPEED_BIN,
) -> Dict[str, Any]:
    """High-level helper: episode meta + cfg → ``prompt_fields`` dict."""
    episode_row = episode_row or {}
    tasks = episode_row.get("tasks")
    if tasks is not None and not isinstance(tasks, (list, tuple)):
        tasks = [tasks]

    ti = task_index
    if ti is None and "task_index" in episode_row:
        try:
            ti = int(episode_row["task_index"])
        except (TypeError, ValueError):
            ti = None

    instruction = resolve_instruction(
        tasks=tasks,
        tasks_map=tasks_map,
        task_index=ti,
        lang=instruction_lang,
    )

    # Prefer episode meta length; caller length (e.g. packed T) is fallback.
    ep_len = None
    if episode_row.get("length") is not None:
        try:
            ep_len = int(episode_row["length"])
        except (TypeError, ValueError):
            ep_len = None
    if ep_len is None:
        ep_len = int(length) if length is not None else None

    ep_fps = None
    if episode_row.get("fps") is not None:
        try:
            ep_fps = float(episode_row["fps"])
        except (TypeError, ValueError):
            ep_fps = None
    if ep_fps is None:
        ep_fps = fps

    view = resolve_camera_view_direction(cfg, video_key=video_key)
    return resolve_prompt_fields(
        embodiment=embodiment_id_from_cfg(cfg),
        instruction=instruction,
        length=ep_len,
        fps=ep_fps,
        camera_view_direction=view,
        speed_bin_size=speed_bin_size,
    )


def write_episodes_jsonl_with_prompt_fields(
    src_episodes: Union[str, Path],
    dst_episodes: Union[str, Path],
    *,
    cfg: Dict[str, Any],
    fps: float,
    tasks_map: Dict[int, str],
    keep_episode_indices: Optional[Iterable[int]] = None,
    length_by_episode: Optional[Dict[int, int]] = None,
    video_key: Optional[str] = None,
    instruction_lang: str = "en",
) -> int:
    """Copy/filter ``episodes.jsonl`` and attach ``prompt_fields`` on each kept row."""
    src = Path(src_episodes)
    dst = Path(dst_episodes)
    dst.parent.mkdir(parents=True, exist_ok=True)
    keep = set(int(i) for i in keep_episode_indices) if keep_episode_indices is not None else None
    length_by_episode = length_by_episode or {}

    n = 0
    with open(dst, "w", encoding="utf-8") as fout:
        for row in load_jsonl(src) if src.is_file() else []:
            ep_idx = row.get("episode_index")
            if ep_idx is None:
                continue
            ep_idx = int(ep_idx)
            if keep is not None and ep_idx not in keep:
                continue
            fields = build_prompt_fields_for_episode(
                cfg,
                row,
                tasks_map=tasks_map,
                fps=fps,
                length=length_by_episode.get(ep_idx),
                video_key=video_key,
                instruction_lang=instruction_lang,
            )
            out_row = dict(row)
            out_row["prompt_fields"] = fields
            fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            n += 1
    return n
