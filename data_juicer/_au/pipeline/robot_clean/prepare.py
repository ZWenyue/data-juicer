# -*- coding: utf-8 -*-
"""Prepare pointer JSONL and embodiment percentiles for cleaning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

from ...utils.lerobot_episode_io import list_episode_parquets, load_episode_arrays, resolve_video_key
from .config import CleanConfig


def build_pointer_jsonl(
    dataset: str,
    output: str,
    video_key: str = "observation.images.head_rgb",
    max_episodes: Optional[int] = None,
) -> dict:
    """Write per-episode pointer records with ``parquet_path`` (+ ``videos`` if present)."""
    root = Path(dataset).resolve()
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)

    resolved_key = resolve_video_key(str(root), preferred=video_key) or video_key
    if resolved_key != video_key:
        logger.info(f"Resolved video_key {video_key!r} → {resolved_key!r}")

    files = list_episode_parquets(str(root))
    if not files:
        pattern = str(root / "data" / "chunk-*" / "episode_*.parquet")
        raise FileNotFoundError(f"No episode parquet under {root} (glob {pattern})")
    if max_episodes is not None:
        files = files[: int(max_episodes)]

    n_vid = 0
    with open(out, "w", encoding="utf-8") as f:
        for pf in files:
            p = Path(pf)
            stem = p.stem
            chunk = p.parent.name
            rec = {
                "id": stem,
                "text": stem,
                "parquet_path": str(p.resolve()),
            }
            vid = root / "videos" / chunk / resolved_key / f"{stem}.mp4"
            if vid.is_file():
                rec["videos"] = [str(vid.resolve())]
                n_vid += 1
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary = {
        "num_episodes": len(files),
        "num_with_videos": n_vid,
        "video_key": resolved_key,
        "pointer": str(out),
        "dataset": str(root),
    }
    logger.info(
        f"Pointer JSONL: {summary['num_episodes']} episodes, "
        f"{n_vid} with videos[{resolved_key}] -> {out}"
    )
    if n_vid == 0:
        logger.warning(
            "No videos attached; Check3 frame scorer will skip (fail-open keep)."
        )
    return summary


def compute_embodiment_percentiles(
    dataset: str,
    output: str,
    embodiment: str,
    max_files: Optional[int] = None,
) -> dict:
    """Compute per-dim q01/q99 for states/actions and write ``{embodiment: ...}`` JSON.

    Arrays are loaded via the embodiment YAML into the Stage1/2/3 clean layout
    (typically 16-dim arm+gripper), not the raw packed ``observation.state``.
    """
    root = Path(dataset).resolve()
    files = list_episode_parquets(str(root), max_files=max_files)
    if not files:
        raise FileNotFoundError(f"No parquet under {root}")

    all_states = []
    all_actions = []
    for i, pf in enumerate(files):
        states, actions = load_episode_arrays(pf, embodiment=embodiment)
        all_states.append(states.astype(np.float32, copy=False))
        all_actions.append(actions.astype(np.float32, copy=False))
        if (i + 1) % 50 == 0 or (i + 1) == len(files):
            logger.info(f"  percentile scan {i + 1}/{len(files)} episodes...")

    states = np.vstack(all_states)
    actions = np.vstack(all_actions)
    pct = {
        "state": {
            "q01": np.percentile(states, 1, axis=0).tolist(),
            "q99": np.percentile(states, 99, axis=0).tolist(),
        },
        "action": {
            "q01": np.percentile(actions, 1, axis=0).tolist(),
            "q99": np.percentile(actions, 99, axis=0).tolist(),
        },
        "num_frames": int(states.shape[0]),
        "num_episodes": len(files),
        "state_dim": int(states.shape[1]),
        "action_dim": int(actions.shape[1]),
    }
    result = {embodiment: pct}
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    logger.info(
        f"Percentiles -> {out} ({pct['num_frames']} frames, "
        f"{pct['num_episodes']} eps, dim={pct['state_dim']})"
    )
    return result


def prepare(cfg: CleanConfig) -> None:
    """Build work dir artefacts required before ``dj-process``."""
    cfg.work_dir.mkdir(parents=True, exist_ok=True)
    build_pointer_jsonl(
        cfg.dataset,
        str(cfg.pointer_path),
        video_key=cfg.video_key,
        max_episodes=cfg.max_episodes,
    )
    if cfg.enable_stage3:
        if cfg.percentiles_path and Path(cfg.percentiles_path).is_file():
            logger.info(f"Reusing percentiles: {cfg.percentiles_path}")
        else:
            compute_embodiment_percentiles(
                cfg.dataset,
                str(cfg.resolved_percentiles_path),
                cfg.embodiment,
                max_files=cfg.max_episodes,
            )
