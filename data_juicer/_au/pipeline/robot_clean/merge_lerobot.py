# -*- coding: utf-8 -*-
"""Merge multiple LeRobot v2.1 task roots into one dataset.

Typical input: unified80 exports where each *task* is already a LeRobot root
(``data/`` + ``meta/`` + ``videos``). This tool concatenates episodes across
tasks/roots into a single LeRobot layout with remapped ``episode_index`` /
global ``index``.

Heterogeneous sources (different ``robot_type``, fps, camera keys) are allowed:
per-episode provenance is stored in ``meta/episodes.jsonl``; video feature keys
in ``info.json`` are the **union** across sources (not every episode has every
video key — loaders should use ``video_keys`` on each episode row).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger

from ...utils.embodiment_layout import UNIFIED_DIM

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

_NUMERIC_FEATURES = {
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


@dataclass
class TaskSource:
    path: Path
    info: Dict[str, Any]
    episodes: List[Dict[str, Any]]
    tasks: List[Dict[str, Any]]
    stats_by_ep: Dict[int, dict] = field(default_factory=dict)


def _read_jsonl(path: Path) -> List[dict]:
    if not path.is_file():
        return []
    rows: List[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def _is_lerobot_task(path: Path) -> bool:
    return (path / "meta" / "info.json").is_file() and (path / "data").is_dir()


def discover_task_dirs(
    roots: Sequence[Union[str, Path]],
    max_tasks_per_root: Optional[int] = None,
) -> List[Path]:
    """Expand roots into LeRobot task directories.

    A root may itself be a task dir, or a parent containing task subdirs.
    """
    out: List[Path] = []
    for root in roots:
        root = Path(root).resolve()
        if not root.exists():
            raise FileNotFoundError(f"Merge root not found: {root}")
        if _is_lerobot_task(root):
            candidates = [root]
        else:
            candidates = sorted(
                p for p in root.iterdir() if p.is_dir() and _is_lerobot_task(p)
            )
            if not candidates:
                raise FileNotFoundError(
                    f"No LeRobot task dirs under {root} "
                    f"(expected meta/info.json + data/)"
                )
        if max_tasks_per_root is not None:
            candidates = candidates[: max(0, int(max_tasks_per_root))]
        out.extend(candidates)
    return out


def load_task_source(task_dir: Path) -> TaskSource:
    meta = task_dir / "meta"
    info = json.loads((meta / "info.json").read_text(encoding="utf-8"))
    episodes = _read_jsonl(meta / "episodes.jsonl")
    tasks = _read_jsonl(meta / "tasks.jsonl")
    stats_by_ep: Dict[int, dict] = {}
    for row in _read_jsonl(meta / "episodes_stats.jsonl"):
        if "episode_index" in row:
            stats_by_ep[int(row["episode_index"])] = row.get("stats") or {}
    if not episodes:
        # Fall back to parquet stems when episodes.jsonl is empty.
        for pq_path in sorted((task_dir / "data").rglob("episode_*.parquet")):
            try:
                ep_idx = int(pq_path.stem.split("_")[-1])
            except ValueError:
                continue
            episodes.append({"episode_index": ep_idx, "length": None})
    return TaskSource(
        path=task_dir,
        info=info,
        episodes=sorted(episodes, key=lambda r: int(r["episode_index"])),
        tasks=tasks,
        stats_by_ep=stats_by_ep,
    )


def _find_episode_parquet(task_dir: Path, episode_index: int) -> Path:
    name6 = f"episode_{episode_index:06d}.parquet"
    name8 = f"episode_{episode_index:08d}.parquet"
    matches = [
        p
        for p in (task_dir / "data").rglob("episode_*.parquet")
        if p.name in (name6, name8)
        or p.stem.endswith(f"_{episode_index:06d}")
        or p.stem.endswith(f"_{episode_index:08d}")
    ]
    if not matches:
        # Direct chunk layout probe.
        for pad in (6, 8):
            for chunk in sorted((task_dir / "data").glob("chunk-*")):
                cand = chunk / f"episode_{episode_index:0{pad}d}.parquet"
                if cand.is_file():
                    matches.append(cand)
    if not matches:
        raise FileNotFoundError(
            f"Missing parquet for episode {episode_index} under {task_dir}/data"
        )
    return sorted(matches)[0]


def _video_keys_from_info(info: dict) -> List[str]:
    feats = info.get("features") or {}
    keys = []
    for k, v in feats.items():
        if isinstance(v, dict) and v.get("dtype") == "video":
            keys.append(k)
    return keys


def _resolve_video_file(
    task_dir: Path,
    video_key: str,
    episode_index: int,
    video_path_tmpl: Optional[str] = None,
) -> Optional[Path]:
    """Locate an episode mp4; tolerate 6/8-digit padding and chunk folders."""
    videos = task_dir / "videos"
    if videos.is_symlink() or videos.is_dir():
        root = Path(os.path.realpath(videos))
    else:
        return None

    chunk = episode_index // 1000
    chunk_names = (f"chunk-{chunk:03d}", "chunk-000")

    # Prefer template-derived relative paths when present.
    if video_path_tmpl and "{video_key}" in video_path_tmpl:
        for episode_chunk in (chunk, 0):
            try:
                rel = video_path_tmpl.format(
                    episode_chunk=episode_chunk,
                    video_key=video_key,
                    episode_index=episode_index,
                )
            except (KeyError, ValueError):
                continue
            # info.json templates usually start with "videos/"; root is already videos/.
            rel_path = Path(rel)
            if rel_path.parts and rel_path.parts[0] == "videos":
                rel_path = Path(*rel_path.parts[1:])
            cand = root / rel_path
            if cand.is_file():
                return cand
            parent = cand.parent
            for p in (6, 8):
                alt = parent / f"episode_{episode_index:0{p}d}.mp4"
                if alt.is_file():
                    return alt

    for pad in (6, 8):
        for chunk_name in chunk_names:
            cand = root / chunk_name / video_key / f"episode_{episode_index:0{pad}d}.mp4"
            if cand.is_file():
                return cand

    for pad in (6, 8):
        for cand in root.glob(f"**/episode_{episode_index:0{pad}d}.mp4"):
            if video_key in cand.parts:
                return cand
    return None


def _link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "symlink":
        dst.symlink_to(src.resolve())
    elif mode == "hardlink":
        os.link(src, dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unknown link mode: {mode}")


def _remap_task_indices(
    sources: Sequence[TaskSource],
) -> Tuple[List[dict], Dict[Tuple[str, int], int]]:
    """Build global tasks.jsonl and (task_dir, old_task_index) → new index."""
    global_tasks: List[dict] = []
    text_to_new: Dict[str, int] = {}
    old_to_new: Dict[Tuple[str, int], int] = {}

    for src in sources:
        task_dir_key = str(src.path)
        for row in src.tasks:
            old_idx = int(row.get("task_index", len(global_tasks)))
            text = str(row.get("task", ""))
            if text not in text_to_new:
                new_idx = len(global_tasks)
                text_to_new[text] = new_idx
                global_tasks.append({"task_index": new_idx, "task": text})
            old_to_new[(task_dir_key, old_idx)] = text_to_new[text]
    if not global_tasks:
        global_tasks = [{"task_index": 0, "task": ""}]
    return global_tasks, old_to_new


def _rewrite_parquet(
    src_parquet: Path,
    dst_parquet: Path,
    new_episode_index: int,
    global_index_start: int,
    task_index_map: Dict[int, int],
) -> Tuple[int, int]:
    """Rewrite one episode parquet with new episode/global indices.

    Returns (n_frames, next_global_index).
    """
    table = pq.read_table(src_parquet)
    n = table.num_rows
    names = table.column_names
    arrays: Dict[str, pa.Array] = {}

    for name in names:
        if name == "episode_index":
            arrays[name] = pa.array(np.full(n, new_episode_index, dtype=np.int64))
        elif name == "index":
            arrays[name] = pa.array(
                np.arange(global_index_start, global_index_start + n, dtype=np.int64)
            )
        elif name == "frame_index":
            arrays[name] = pa.array(np.arange(n, dtype=np.int64))
        elif name == "task_index" and task_index_map:
            old = table.column(name).to_numpy()
            mapped = np.array(
                [task_index_map.get(int(v), int(v)) for v in old], dtype=np.int64
            )
            arrays[name] = pa.array(mapped)
        elif name == "coarse_task_index" and task_index_map:
            old = table.column(name).to_numpy()
            mapped = np.array(
                [task_index_map.get(int(v), int(v)) for v in old], dtype=np.int64
            )
            arrays[name] = pa.array(mapped)
        else:
            arrays[name] = table.column(name)

    # Ensure required index columns exist.
    if "episode_index" not in arrays:
        arrays["episode_index"] = pa.array(np.full(n, new_episode_index, dtype=np.int64))
    if "index" not in arrays:
        arrays["index"] = pa.array(
            np.arange(global_index_start, global_index_start + n, dtype=np.int64)
        )
    if "frame_index" not in arrays:
        arrays["frame_index"] = pa.array(np.arange(n, dtype=np.int64))

    # Preserve column order: original first, then any added.
    ordered = [c for c in names if c in arrays]
    for c in ("observation.state", "action", "action_dim_mask", *_INDEX_COLS):
        if c in arrays and c not in ordered:
            ordered.append(c)
    for c in arrays:
        if c not in ordered:
            ordered.append(c)

    dst_parquet.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({c: arrays[c] for c in ordered}), dst_parquet, compression="zstd")
    return n, global_index_start + n


def merge_lerobot_datasets(
    roots: Sequence[Union[str, Path]],
    output_dir: Union[str, Path],
    *,
    chunks_size: int = 1000,
    video_policy: str = "keep",
    link_mode: str = "symlink",
    max_tasks_per_root: Optional[int] = None,
    max_episodes_per_task: Optional[int] = None,
    dry_run: bool = False,
) -> dict:
    """Merge LeRobot task dirs under ``roots`` into ``output_dir``.

    Parameters
    ----------
    roots:
        Task dirs and/or parents of task dirs (e.g. ``unified80`` folders).
    output_dir:
        Destination LeRobot root (will contain ``data/``, ``meta/``, ``videos/``).
    video_policy:
        ``keep`` — symlink/copy videos under original feature keys;
        ``none`` — skip videos (numeric-only merge).
    link_mode:
        ``symlink`` | ``hardlink`` | ``copy`` for video files.
    """
    if video_policy not in ("keep", "none"):
        raise ValueError("video_policy must be 'keep' or 'none'")
    if link_mode not in ("symlink", "hardlink", "copy"):
        raise ValueError("link_mode must be symlink|hardlink|copy")

    task_dirs = discover_task_dirs(roots, max_tasks_per_root=max_tasks_per_root)
    sources = [load_task_source(p) for p in task_dirs]
    global_tasks, old_task_map = _remap_task_indices(sources)

    out = Path(output_dir).resolve()
    if not dry_run:
        if out.exists() and any(out.iterdir()):
            raise FileExistsError(
                f"Output dir not empty: {out}. Choose a fresh path or clear it."
            )
        (out / "data").mkdir(parents=True, exist_ok=True)
        (out / "meta").mkdir(parents=True, exist_ok=True)
        if video_policy == "keep":
            (out / "videos").mkdir(parents=True, exist_ok=True)

    video_features: Dict[str, dict] = {}
    robot_types: List[str] = []
    embodiments: List[str] = []
    fps_values: List[float] = []

    episode_rows: List[dict] = []
    stats_rows: List[dict] = []
    source_rows: List[dict] = []

    new_ep = 0
    global_index = 0
    total_videos = 0
    missing_videos = 0

    for src in sources:
        info = src.info
        rt = str(info.get("robot_type") or "")
        emb = str(info.get("unified_embodiment") or "")
        fps = info.get("fps")
        if rt and rt not in robot_types:
            robot_types.append(rt)
        if emb and emb not in embodiments:
            embodiments.append(emb)
        if fps is not None:
            fps_values.append(float(fps))

        vkeys = _video_keys_from_info(info)
        if video_policy == "keep":
            feats = info.get("features") or {}
            for vk in vkeys:
                if vk not in video_features and isinstance(feats.get(vk), dict):
                    video_features[vk] = feats[vk]

        # Per-source task_index remapping for this task dir.
        task_dir_key = str(src.path)
        local_task_map = {
            old: new
            for (td, old), new in old_task_map.items()
            if td == task_dir_key
        }

        ep_list = list(src.episodes)
        if max_episodes_per_task is not None:
            ep_list = ep_list[: max(0, int(max_episodes_per_task))]

        video_tmpl = info.get("video_path")
        for ep_meta in ep_list:
            src_ep = int(ep_meta["episode_index"])
            src_pq = _find_episode_parquet(src.path, src_ep)
            chunk = new_ep // int(chunks_size)
            dst_pq = out / "data" / f"chunk-{chunk:03d}" / f"episode_{new_ep:06d}.parquet"

            if dry_run:
                # Estimate frames from meta length when possible.
                n_frames = int(ep_meta.get("length") or 0)
                if n_frames <= 0:
                    n_frames = pq.read_metadata(src_pq).num_rows
            else:
                n_frames, global_index = _rewrite_parquet(
                    src_pq,
                    dst_pq,
                    new_episode_index=new_ep,
                    global_index_start=global_index,
                    task_index_map=local_task_map,
                )

            linked_keys: List[str] = []
            if video_policy == "keep" and not dry_run:
                for vk in vkeys:
                    src_vid = _resolve_video_file(src.path, vk, src_ep, video_tmpl)
                    if src_vid is None:
                        missing_videos += 1
                        logger.warning(
                            f"Missing video {vk} ep={src_ep} in {src.path.name}"
                        )
                        continue
                    dst_vid = (
                        out
                        / "videos"
                        / f"chunk-{chunk:03d}"
                        / vk
                        / f"episode_{new_ep:06d}.mp4"
                    )
                    _link_or_copy(src_vid, dst_vid, link_mode)
                    linked_keys.append(vk)
                    total_videos += 1
            elif video_policy == "keep" and dry_run:
                for vk in vkeys:
                    if _resolve_video_file(src.path, vk, src_ep, video_tmpl):
                        linked_keys.append(vk)
                        total_videos += 1
                    else:
                        missing_videos += 1

            ep_out = {
                "episode_index": new_ep,
                "tasks": ep_meta.get("tasks"),
                "length": n_frames,
                "fps": fps,
                "robot_type": rt,
                "unified_embodiment": emb,
                "source_root": str(src.path.parent),
                "source_task": src.path.name,
                "source_episode_index": src_ep,
                "source_parquet": str(src_pq),
                "video_keys": linked_keys if video_policy == "keep" else [],
            }
            # Preserve useful extra fields when present.
            for k in ("raw_file_name", "discarded_trajectory"):
                if k in ep_meta:
                    ep_out[k] = ep_meta[k]
            episode_rows.append(ep_out)

            stats = src.stats_by_ep.get(src_ep)
            if stats is not None:
                stats_rows.append({"episode_index": new_ep, "stats": stats})

            source_rows.append(
                {
                    "episode_index": new_ep,
                    "source_task": src.path.name,
                    "source_episode_index": src_ep,
                    "source_path": str(src.path),
                }
            )
            new_ep += 1
            if dry_run:
                global_index += n_frames

    n_eps = new_ep
    n_frames = global_index
    features = dict(_NUMERIC_FEATURES)
    if video_policy == "keep":
        features.update(video_features)

    # Pick a representative fps for info.json (loaders often require a number).
    fps_info: Any
    if not fps_values:
        fps_info = None
    elif len(set(fps_values)) == 1:
        fps_info = fps_values[0]
    else:
        # majority
        fps_info = max(set(fps_values), key=fps_values.count)

    info_out = {
        "codebase_version": "v2.1",
        "robot_type": robot_types[0] if len(robot_types) == 1 else "mixed",
        "robot_types": robot_types,
        "unified_embodiments": embodiments,
        "unified_dim": UNIFIED_DIM,
        "total_episodes": n_eps,
        "total_frames": n_frames,
        "total_tasks": len(global_tasks),
        "total_videos": total_videos if video_policy == "keep" else 0,
        "total_chunks": max(1, (n_eps + chunks_size - 1) // chunks_size) if n_eps else 0,
        "chunks_size": int(chunks_size),
        "fps": fps_info,
        "fps_heterogeneous": len(set(fps_values)) > 1,
        "splits": {"train": f"0:{n_eps}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "features": features,
        "merged_from": [str(Path(r).resolve()) for r in roots],
        "merged_tasks": [p.name for p in task_dirs],
        "video_policy": video_policy,
        "video_keys_union": sorted(video_features.keys()),
        "notes": (
            "Merged multi-embodiment unified80 LeRobot dataset. "
            "Video feature keys are a union; use episodes.jsonl video_keys per episode. "
            "Per-episode fps/robot_type/unified_embodiment are in episodes.jsonl."
        ),
    }
    if video_policy == "keep":
        info_out["video_path"] = (
            "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
        )

    summary = {
        "output": str(out),
        "total_episodes": n_eps,
        "total_frames": n_frames,
        "total_tasks_meta": len(global_tasks),
        "source_task_dirs": len(task_dirs),
        "robot_types": robot_types,
        "unified_embodiments": embodiments,
        "fps_values": sorted(set(fps_values)),
        "total_videos_linked": total_videos,
        "missing_videos": missing_videos,
        "video_policy": video_policy,
        "dry_run": dry_run,
    }

    if dry_run:
        logger.info(f"[dry-run] would merge -> {summary}")
        return summary

    _write_jsonl(out / "meta" / "episodes.jsonl", episode_rows)
    _write_jsonl(out / "meta" / "tasks.jsonl", global_tasks)
    _write_jsonl(out / "meta" / "episodes_stats.jsonl", stats_rows)
    _write_jsonl(out / "meta" / "sources.jsonl", source_rows)
    (out / "meta" / "info.json").write_text(
        json.dumps(info_out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out / "merge_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info(
        f"Merged {n_eps} episodes / {n_frames} frames from {len(task_dirs)} tasks -> {out}"
    )
    return summary


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Merge multiple LeRobot v2.1 (unified80) task roots into one dataset "
            "with remapped episode_index / global index."
        )
    )
    p.add_argument(
        "--root",
        action="append",
        required=True,
        dest="roots",
        help="LeRobot task dir or parent of tasks (repeatable)",
    )
    p.add_argument("--output", required=True, help="Destination LeRobot root (must be empty/new)")
    p.add_argument("--chunks-size", type=int, default=1000)
    p.add_argument(
        "--video-policy",
        choices=["keep", "none"],
        default="keep",
        help="keep: symlink/copy videos under original keys; none: numeric only",
    )
    p.add_argument(
        "--link-mode",
        choices=["symlink", "hardlink", "copy"],
        default="symlink",
    )
    p.add_argument("--max-tasks-per-root", type=int, default=None)
    p.add_argument("--max-episodes-per-task", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    summary = merge_lerobot_datasets(
        args.roots,
        args.output,
        chunks_size=args.chunks_size,
        video_policy=args.video_policy,
        link_mode=args.link_mode,
        max_tasks_per_root=args.max_tasks_per_root,
        max_episodes_per_task=args.max_episodes_per_task,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
