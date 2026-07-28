# -*- coding: utf-8 -*-
"""Export episodes to full LeRobot v2.1 80-dim layout.

Two modes:

1. **cleaned-kept** (pretrain): export episodes listed in ``cleaned.jsonl``.
2. **keep-all / pad-only** (post-train): export every episode under the source
   task — same 80-dim packing, no Stage1/2/3/5 or Check3 filtering.

Layout:

  OUT/<task>/
    data/chunk-XXX/episode_YYYYYY.parquet
    meta/
      info.json
      episodes.jsonl
      tasks.jsonl
      episodes_stats.jsonl
    videos -> <source videos> # symlink

Each parquet frame has:
  observation.state[80], action[80], action_dim_mask[80]
  + timestamp / frame_index / episode_index / ... when present

``action_dim_mask`` marks supervised action dims for training ``loss × mask``
(state-only channels such as EEF pose are 0).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger

from ...utils.embodiment_layout import UNIFIED_DIM, load_embodiment_config, pack_episode_to_80
from ...utils.embodiment_prompt import (
    load_tasks_map,
    write_episodes_jsonl_with_prompt_fields,
)
from ...utils.lerobot_episode_io import list_episode_parquets, resolve_video_key

_INDEX_COLS = (
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "coarse_task_index",
    "task_index",
    "coarse_quality_index",
    "quality_index",
)


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
        if "stats" in p.name and p.name != Path(result_path).name:
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def _list_col(arr: np.ndarray) -> pa.Array:
    return pa.array(
        [row.tolist() for row in np.asarray(arr, dtype=np.float64)],
        type=pa.list_(pa.float64()),
    )


def _vector_stats(arr: np.ndarray) -> dict:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return {
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0).tolist(),
        "count": [int(arr.shape[0])],
    }


def _scalar_stats(col) -> dict:
    a = np.asarray(col, dtype=np.float64).reshape(-1)
    return {
        "min": [float(a.min())] if len(a) else [0.0],
        "max": [float(a.max())] if len(a) else [0.0],
        "mean": [float(a.mean())] if len(a) else [0.0],
        "std": [float(a.std())] if len(a) else [0.0],
        "count": [int(len(a))],
    }


def _symlink_videos(task_out: Path, src_task: Path) -> Optional[str]:
    videos = src_task / "videos"
    if videos.is_symlink():
        target = Path(os.path.realpath(videos))
        if not target.is_dir():
            return None
    elif videos.is_dir():
        target = videos.resolve()
    else:
        return None

    link = task_out / "videos"
    if link.is_symlink() or link.is_file():
        link.unlink()
    elif link.is_dir():
        if any(link.iterdir()):
            raise RuntimeError(f"Refusing to overwrite non-empty videos path: {link}")
        link.rmdir()
    link.symlink_to(target, target_is_directory=True)
    return str(target)


def _write_task_meta(
    task_out: Path,
    src_task: Path,
    embodiment: str,
    episode_results: List[Dict[str, Any]],
    link_videos: bool = True,
    source_clean_jsonl: Optional[str] = None,
    export_mode: str = "cleaned_kept",
) -> dict:
    meta_dir = task_out / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    src_meta = src_task / "meta"
    src_info = src_meta / "info.json"
    base = json.loads(src_info.read_text(encoding="utf-8")) if src_info.is_file() else {}

    episode_results = sorted(episode_results, key=lambda r: r["episode_index"])
    n_eps = len(episode_results)
    n_frames = sum(r["T"] for r in episode_results)
    keep_idx = {r["episode_index"] for r in episode_results}
    length_by_ep = {int(r["episode_index"]): int(r["T"]) for r in episode_results}

    cfg = load_embodiment_config(embodiment)
    fps = float(base.get("fps", 15) or 15)
    tasks_map = load_tasks_map(src_meta)
    video_key = resolve_video_key(str(src_task))
    n_ep_meta = write_episodes_jsonl_with_prompt_fields(
        src_meta / "episodes.jsonl",
        meta_dir / "episodes.jsonl",
        cfg=cfg,
        fps=fps,
        tasks_map=tasks_map,
        keep_episode_indices=keep_idx,
        length_by_episode=length_by_ep,
        video_key=video_key,
    )
    if (src_meta / "tasks.jsonl").is_file():
        shutil.copy2(src_meta / "tasks.jsonl", meta_dir / "tasks.jsonl")
    else:
        (meta_dir / "tasks.jsonl").write_text("", encoding="utf-8")

    with open(meta_dir / "episodes_stats.jsonl", "w", encoding="utf-8") as f:
        for r in episode_results:
            f.write(
                json.dumps(
                    {"episode_index": r["episode_index"], "stats": r["stats"]},
                    ensure_ascii=False,
                )
                + "\n"
            )

    features = {
        "observation.state": {
            "dtype": "float64",
            "shape": [UNIFIED_DIM],
            "names": [f"unified_{i}" for i in range(UNIFIED_DIM)],
        },
        "action": {
            "dtype": "float64",
            "shape": [UNIFIED_DIM],
            "names": [f"unified_{i}" for i in range(UNIFIED_DIM)],
        },
        "action_dim_mask": {
            "dtype": "float64",
            "shape": [UNIFIED_DIM],
            "names": [f"mask_{i}" for i in range(UNIFIED_DIM)],
        },
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "coarse_task_index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
        "coarse_quality_index": {"dtype": "int64", "shape": [1], "names": None},
        "quality_index": {"dtype": "int64", "shape": [1], "names": None},
    }

    video_target = None
    if link_videos:
        video_target = _symlink_videos(task_out, src_task)
        src_feats = base.get("features") or {}
        for k, v in src_feats.items():
            if isinstance(v, dict) and v.get("dtype") == "video":
                features[k] = v

    chunks_size = int(base.get("chunks_size", 1000))
    info = {
        "codebase_version": base.get("codebase_version", "v2.1"),
        "robot_type": base.get("robot_type", "r1lite"),
        "total_episodes": n_eps,
        "total_frames": n_frames,
        "total_tasks": base.get("total_tasks", 1),
        "total_videos": base.get("total_videos", 0) if video_target else 0,
        "total_chunks": max(1, (n_eps + chunks_size - 1) // chunks_size),
        "chunks_size": chunks_size,
        "fps": base.get("fps", 15),
        "splits": {"train": f"0:{n_eps}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "features": features,
        "unified_embodiment": embodiment,
        "unified_dim": UNIFIED_DIM,
        "unified_export_mode": export_mode,
    }
    if source_clean_jsonl:
        info["source_clean_jsonl"] = source_clean_jsonl
    if video_target:
        info["video_path"] = base.get(
            "video_path",
            "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        )
        info["videos_symlink_target"] = video_target

    (meta_dir / "info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {
        "episodes_jsonl": n_ep_meta,
        "episodes": n_eps,
        "frames": n_frames,
        "videos_link": video_target,
    }


def _export_one_episode(src_parquet: str, dst_parquet: Path, cfg: dict) -> dict:
    df = pq.read_table(src_parquet).to_pandas()
    states, actions, dim_mask = pack_episode_to_80(df, cfg)
    if states.shape[1] != UNIFIED_DIM:
        raise ValueError(f"Unexpected unified dim {states.shape} for {src_parquet}")

    cols = {
        "observation.state": _list_col(states),
        "action": _list_col(actions),
        "action_dim_mask": _list_col(dim_mask),
    }
    for name in _INDEX_COLS:
        if name in df.columns:
            cols[name] = pa.array(df[name].to_numpy())

    dst_parquet.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(cols), dst_parquet, compression="zstd")

    if "episode_index" in df.columns:
        ep_idx = int(df["episode_index"].iloc[0])
    else:
        ep_idx = int(dst_parquet.stem.split("_")[-1])

    stats = {
        "observation.state": _vector_stats(states),
        "action": _vector_stats(actions),
        "action_dim_mask": _vector_stats(dim_mask),
    }
    for name in _INDEX_COLS:
        if name in df.columns:
            stats[name] = _scalar_stats(df[name])

    return {
        "src": src_parquet,
        "out": str(dst_parquet),
        "T": int(states.shape[0]),
        "active": int(dim_mask[0].sum()) if len(dim_mask) else 0,
        "episode_index": ep_idx,
        "stats": stats,
    }


def _resolve_task_out(dataset: str, output_dir: str) -> tuple[Path, Path]:
    """Return ``(src_root, task_out)``.

    ``output_dir`` may be the task output root itself, or a parent that will
    hold ``<task_name>/``.
    """
    src_root = Path(dataset).resolve()
    task_name = src_root.name
    out_root = Path(output_dir)
    if out_root.name == task_name:
        task_out = out_root
    else:
        task_out = out_root / task_name
    return src_root, task_out


def _export_parquet_paths(
    parquet_paths: Sequence[str],
    dataset: str,
    output_dir: str,
    embodiment: str,
    link_videos: bool,
    export_mode: str,
    source_clean_jsonl: Optional[str] = None,
    extra_summary: Optional[Dict[str, Any]] = None,
) -> dict:
    """Pack a list of source episode parquets into a unified80 LeRobot task."""
    if not parquet_paths:
        raise RuntimeError(f"No episode parquets to export for dataset={dataset}")

    cfg = load_embodiment_config(embodiment)
    src_root, task_out = _resolve_task_out(dataset, output_dir)

    episode_results: List[Dict[str, Any]] = []
    for pf in parquet_paths:
        src = Path(pf).resolve()
        chunk = src.parent.name
        dst = task_out / "data" / chunk / src.name
        if not dst.name.endswith(".parquet"):
            dst = dst.with_suffix(".parquet")
        episode_results.append(_export_one_episode(str(src), dst, cfg))

    meta_info = _write_task_meta(
        task_out,
        src_root,
        embodiment,
        episode_results,
        link_videos=link_videos,
        source_clean_jsonl=source_clean_jsonl,
        export_mode=export_mode,
    )

    summary: Dict[str, Any] = {
        "kept_episodes": len(episode_results),
        "total_frames": meta_info["frames"],
        "out": str(task_out),
        "embodiment": embodiment,
        "export_mode": export_mode,
        "mask_active": episode_results[0]["active"] if episode_results else 0,
        "layout": "LeRobot v2.1 (data/ + meta/ + videos symlink)",
        "videos_link": meta_info.get("videos_link"),
        "episodes_jsonl": meta_info.get("episodes_jsonl"),
    }
    if source_clean_jsonl is not None:
        summary["result_jsonl"] = source_clean_jsonl
    if extra_summary:
        summary.update(extra_summary)

    task_out.parent.mkdir(parents=True, exist_ok=True)
    (task_out.parent / "export_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info(
        f"Exported {summary['kept_episodes']} episodes [{export_mode}] "
        f"({summary['total_frames']} frames, mask_active={summary['mask_active']}/{UNIFIED_DIM}) "
        f"-> {task_out}"
    )
    return summary


def export_kept_unified_parquets(
    result_jsonl: str,
    dataset: str,
    output_dir: str,
    embodiment: str = "galaxea_r1_lite",
    link_videos: bool = True,
) -> dict:
    """Write full LeRobot v2.1 80-dim layout for episodes kept in cleaned JSONL.

    ``output_dir`` is the task output root (contains ``data/`` ``meta/``),
    OR a parent directory that will hold ``<task_name>/``. If ``output_dir``
    already ends with the task name (or equals it), files go directly under it.
    """
    rows = _iter_result_rows(result_jsonl)
    if not rows:
        raise RuntimeError(f"Empty cleaned result: {result_jsonl}")

    parquet_paths: List[str] = []
    for r in rows:
        if "parquet_path" not in r:
            raise KeyError(f"cleaned row missing parquet_path: keys={sorted(r)}")
        parquet_paths.append(str(Path(r["parquet_path"]).resolve()))

    return _export_parquet_paths(
        parquet_paths,
        dataset=dataset,
        output_dir=output_dir,
        embodiment=embodiment,
        link_videos=link_videos,
        export_mode="cleaned_kept",
        source_clean_jsonl=str(result_jsonl),
    )


def export_all_unified_parquets(
    dataset: str,
    output_dir: str,
    embodiment: str = "galaxea_r1_lite",
    link_videos: bool = True,
    max_episodes: Optional[int] = None,
) -> dict:
    """Pad-only export: every source episode → unified80 LeRobot layout.

    No Stage1/2/3/5 or Check3 filtering. Intended for post-training data prep
    that needs the same 80-dim layout as the pretrain clean pipeline.
    """
    files = list_episode_parquets(dataset, max_files=max_episodes)
    if not files:
        pattern = str(Path(dataset) / "data" / "chunk-*" / "episode_*.parquet")
        raise FileNotFoundError(f"No episode parquet under {dataset} (glob {pattern})")

    return _export_parquet_paths(
        files,
        dataset=dataset,
        output_dir=output_dir,
        embodiment=embodiment,
        link_videos=link_videos,
        export_mode="keep_all",
        extra_summary={"dataset": str(Path(dataset).resolve()), "max_episodes": max_episodes},
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Export LeRobot episodes to unified 80-dim layout. "
            "Use --cleaned for kept-after-clean export, or --keep-all for pad-only."
        )
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--cleaned",
        default=None,
        help="Path to cleaned.jsonl (export only kept episodes).",
    )
    src.add_argument(
        "--keep-all",
        action="store_true",
        help="Pad-only: export every episode under --dataset (no cleaning).",
    )
    p.add_argument("--dataset", required=True, help="Source LeRobot task dir")
    p.add_argument("--output", required=True, help="Output task dir or parent dir")
    p.add_argument("--embodiment", default="galaxea_r1_lite")
    p.add_argument("--no-link-videos", action="store_true")
    p.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Only with --keep-all: cap number of episodes (smoke tests).",
    )
    args = p.parse_args(argv)

    if args.keep_all:
        summary = export_all_unified_parquets(
            args.dataset,
            args.output,
            embodiment=args.embodiment,
            link_videos=not args.no_link_videos,
            max_episodes=args.max_episodes,
        )
    else:
        if args.max_episodes is not None:
            logger.warning("--max-episodes is ignored with --cleaned (use cleaned.jsonl filtering).")
        summary = export_kept_unified_parquets(
            args.cleaned,
            args.dataset,
            args.output,
            embodiment=args.embodiment,
            link_videos=not args.no_link_videos,
        )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
