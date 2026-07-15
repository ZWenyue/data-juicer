# -*- coding: utf-8 -*-
"""Export kept cleaned episodes to LeRobot-layout 80-dim parquets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Union

import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger

from ...utils.embodiment_layout import UNIFIED_DIM, load_embodiment_config, pack_episode_to_80


def _jsonl_files_under(path: Path) -> List[Path]:
    """Return jsonl-like files under a directory (used for sharded exports)."""
    return sorted(
        p
        for p in path.rglob("*")
        if p.is_file() and ("json" in p.suffix.lower() or "json" in p.name.lower())
    )


def _iter_result_rows(result_path: Union[str, Path]) -> List[dict]:
    path = Path(result_path)
    if path.is_file():
        paths = [path]
    elif path.is_dir():
        # Ray / multi-shard export writes a directory of shards.
        paths = _jsonl_files_under(path)
    else:
        paths = []
        for cand in sorted(path.parent.glob(path.name + "*")):
            if cand.is_file():
                paths.append(cand)
            elif cand.is_dir():
                paths.extend(_jsonl_files_under(cand))
    if not paths:
        raise FileNotFoundError(f"No cleaned result at {result_path}")

    rows: List[dict] = []
    for p in paths:
        # Skip obvious stats sidecars if naming differs; main export is cleaned.jsonl
        if "stats" in p.name and p.name != Path(result_path).name:
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def export_kept_unified_parquets(
    result_jsonl: str,
    dataset: str,
    output_dir: str,
    embodiment: str = "galaxea_r1_lite",
    link_videos: bool = True,
) -> dict:
    """Write ``OUT/<task>/data/chunk-*/episode_*.parquet`` for kept episodes."""
    rows = _iter_result_rows(result_jsonl)
    if not rows:
        raise RuntimeError(f"Empty cleaned result: {result_jsonl}")

    cfg = load_embodiment_config(embodiment)
    src_root = Path(dataset).resolve()
    task_name = src_root.name
    out_root = Path(output_dir)
    task_out = out_root / task_name
    n = 0

    for r in rows:
        src = Path(r["parquet_path"]).resolve()
        chunk = src.parent.name
        stem = src.name
        dst = task_out / "data" / chunk / stem
        dst.parent.mkdir(parents=True, exist_ok=True)

        df = pq.read_table(str(src)).to_pandas()
        states, actions, dim_mask = pack_episode_to_80(df, cfg)
        if states.shape[1] != UNIFIED_DIM:
            raise ValueError(f"Unexpected unified dim {states.shape} for {src}")

        keep_cols = {}
        for col in (
            "timestamp",
            "frame_index",
            "episode_index",
            "index",
            "coarse_task_index",
            "task_index",
            "coarse_quality_index",
            "quality_index",
        ):
            if col in df.columns:
                keep_cols[col] = df[col].tolist()

        table = pa.table(
            {
                **keep_cols,
                "observation.state": [row.tolist() for row in states],
                "action": [row.tolist() for row in actions],
                "observation.state_dim_mask": [row.tolist() for row in dim_mask],
            }
        )
        pq.write_table(table, dst)
        n += 1

    meta_dir = task_out / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    src_info = src_root / "meta" / "info.json"
    info = json.loads(src_info.read_text(encoding="utf-8")) if src_info.is_file() else {}
    info.update(
        {
            "total_episodes": n,
            "unified_embodiment": embodiment,
            "unified_dim": UNIFIED_DIM,
            "source_clean_jsonl": str(result_jsonl),
        }
    )
    (meta_dir / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    if link_videos:
        videos_link = task_out / "videos"
        src_videos = src_root / "videos"
        if src_videos.is_dir() and not videos_link.exists():
            videos_link.symlink_to(src_videos, target_is_directory=True)

    summary = {
        "kept_episodes": n,
        "out": str(task_out),
        "result_jsonl": str(result_jsonl),
        "embodiment": embodiment,
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "export_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    logger.info(f"Exported {n} kept episodes -> {task_out}")
    return summary
