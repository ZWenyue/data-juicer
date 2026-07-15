#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export unified 80-dim state/action/mask as LeRobot v2.1 per-episode parquet.

Output mirrors source task layout (NO videos/):

  OUT/<task_name>/
    data/chunk-XXX/episode_YYYYYY.parquet
    meta/
      info.json              # features = observation.state/action/mask[80], total_videos=0
      episodes.jsonl         # copied/filtered from source
      tasks.jsonl            # copied from source
      episodes_stats.jsonl   # recomputed for unified columns

Each parquet is frame-wise with:
  observation.state[80], action[80], observation.state_dim_mask[80]
  + index fields when present in source
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data_juicer._au.utils.embodiment_layout import (  # noqa: E402
    UNIFIED_DIM,
    load_embodiment_config,
    pack_episode_to_80,
)
from data_juicer._au.utils.lerobot_episode_io import (  # noqa: E402
    iter_task_dirs,
    list_episode_parquets,
)

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


def _list_col(arr: np.ndarray) -> pa.Array:
    """(T, D) float -> list<float64> column."""
    return pa.array([row.tolist() for row in np.asarray(arr, dtype=np.float64)], type=pa.list_(pa.float64()))


def _out_parquet_path(src_parquet: str, root: str, out_root: str) -> Path:
    """Map source parquet to OUT/<relpath from task root or root>."""
    src = Path(src_parquet).resolve()
    root_p = Path(root).resolve()
    # Prefer: .../<task>/data/chunk-*/episode_*.parquet -> OUT/<task>/data/...
    parts = src.parts
    if "data" in parts:
        i = parts.index("data")
        # task dir is parent of data
        task_name = parts[i - 1] if i >= 1 else src.parent.parent.name
        rel = Path(*parts[i:])  # data/chunk-xxx/episode_yyy.parquet
        return Path(out_root) / task_name / rel
    try:
        rel = src.relative_to(root_p)
        return Path(out_root) / rel
    except ValueError:
        return Path(out_root) / src.parent.name / "data" / "chunk-000" / src.name


def _vector_stats(arr: np.ndarray) -> dict:
    """Per-dim stats matching LeRobot episodes_stats style."""
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


def _export_one(args) -> dict:
    src_parquet, out_parquet, cfg_path, embodiment = args
    cfg = load_embodiment_config(cfg_path if cfg_path else embodiment)
    table = pq.read_table(src_parquet)
    df = table.to_pandas()
    states, actions, dim_mask = pack_episode_to_80(df, cfg)
    T = states.shape[0]
    if states.shape[1] != UNIFIED_DIM:
        raise ValueError(f"bad dim {states.shape} from {src_parquet}")

    cols = {
        "observation.state": _list_col(states),
        "action": _list_col(actions),
        "observation.state_dim_mask": _list_col(dim_mask),
    }
    for name in _INDEX_COLS:
        if name in df.columns:
            cols[name] = pa.array(df[name].to_numpy())

    out_path = Path(out_parquet)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(cols), out_path, compression="zstd")

    # episode_index from data or filename
    if "episode_index" in df.columns:
        ep_idx = int(df["episode_index"].iloc[0])
    else:
        stem = out_path.stem  # episode_000012
        ep_idx = int(stem.split("_")[-1])

    stats = {
        "observation.state": _vector_stats(states),
        "action": _vector_stats(actions),
        "observation.state_dim_mask": _vector_stats(dim_mask),
    }
    for name in _INDEX_COLS:
        if name in df.columns:
            stats[name] = _scalar_stats(df[name])

    active = int(dim_mask[0].sum()) if T else 0
    return {
        "src": src_parquet,
        "out": str(out_path),
        "T": T,
        "active": active,
        "episode_index": ep_idx,
        "stats": stats,
        "task": out_path.parts[-4] if len(out_path.parts) >= 4 else "",
    }


def _copy_jsonl_filtered(src: Path, dst: Path, keep_episode_indices: set = None):
    """Copy JSONL; optionally keep only rows whose episode_index is in the set."""
    if not src.is_file():
        return 0
    n = 0
    with open(src, encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            if keep_episode_indices is not None:
                row = json.loads(line)
                if row.get("episode_index") not in keep_episode_indices:
                    continue
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            else:
                fout.write(line + "\n")
            n += 1
    return n


def _resolve_videos_target(src_task: Path) -> Path | None:
    """Return absolute videos directory for src_task, following one-level symlink."""
    videos = src_task / "videos"
    if videos.is_symlink():
        target = Path(os.path.realpath(videos))
        return target if target.is_dir() else None
    if videos.is_dir():
        return videos.resolve()
    return None


def _symlink_videos(task_out: Path, src_task: Path) -> str | None:
    """Create task_out/videos -> real source videos dir (like lerobot_press layout).

    Does not copy files; only a symlink. Returns the link target path or None.
    """
    target = _resolve_videos_target(src_task)
    if target is None:
        return None

    link = task_out / "videos"
    # Replace broken/old link; never delete a real directory full of videos.
    if link.is_symlink() or link.exists():
        if link.is_symlink() or link.is_file():
            link.unlink()
        elif link.is_dir() and not any(link.iterdir()):
            link.rmdir()
        else:
            raise RuntimeError(
                f"Refusing to overwrite existing non-empty path for videos link: {link}"
            )
    link.symlink_to(target, target_is_directory=True)
    return str(target)


def _write_task_meta(
    task_out: Path,
    src_task: Path,
    embodiment: str,
    episode_results: list,
    link_videos: bool = True,
):
    """Write LeRobot v2.1 meta/ (+ optional videos/ symlink to source)."""
    meta_dir = task_out / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    src_meta = src_task / "meta"
    src_info = src_meta / "info.json"
    base = {}
    if src_info.is_file():
        with open(src_info, encoding="utf-8") as f:
            base = json.load(f)

    episode_results = sorted(episode_results, key=lambda r: r["episode_index"])
    n_eps = len(episode_results)
    n_frames = sum(r["T"] for r in episode_results)
    keep_idx = {r["episode_index"] for r in episode_results}

    n_ep_meta = _copy_jsonl_filtered(
        src_meta / "episodes.jsonl", meta_dir / "episodes.jsonl", keep_idx
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
        "observation.state_dim_mask": {
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
        # Keep camera/video feature decls so loaders can resolve mp4 via video_path.
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
    }
    if video_target:
        info["video_path"] = base.get(
            "video_path",
            "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        )
        info["videos_symlink_target"] = video_target

    with open(meta_dir / "info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    return {
        "episodes_jsonl": n_ep_meta,
        "episodes": n_eps,
        "frames": n_frames,
        "videos_link": video_target,
    }


def collect_jobs(root: str, out: str, max_tasks: int = None):
    root_p = Path(root).resolve()
    jobs = []  # (src, out_parquet)
    if (root_p / "data").is_dir():
        task_dirs = [root_p]
    else:
        task_dirs = list(iter_task_dirs(str(root_p)))
        if max_tasks is not None:
            task_dirs = task_dirs[:max_tasks]
    for task_dir in task_dirs:
        for pf in list_episode_parquets(str(task_dir)):
            out_pf = _out_parquet_path(pf, str(root_p), out)
            jobs.append((pf, str(out_pf), str(task_dir)))
    return jobs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, help="LeRobot task dir or multi-task root")
    ap.add_argument("--out", required=True, help="Output root (mirrors task/data/chunk/...)")
    ap.add_argument("--embodiment", default="galaxea_r1_lite")
    ap.add_argument("--embodiment_config", default=None)
    ap.add_argument("--max_tasks", type=int, default=None)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument(
        "--link_videos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Symlink OUT/<task>/videos -> source videos (default: on)",
    )
    args = ap.parse_args()

    cfg_ref = args.embodiment_config or args.embodiment
    # validate config early in parent
    load_embodiment_config(cfg_ref)

    os.makedirs(args.out, exist_ok=True)
    collected = collect_jobs(args.root, args.out, args.max_tasks)
    if not collected:
        raise SystemExit(f"No episodes found under {args.root}")

    work = [(src, dst, args.embodiment_config, args.embodiment) for src, dst, _task in collected]
    results = []
    n_workers = max(1, min(args.num_workers, len(work)))
    print(f"Exporting {len(work)} episodes with {n_workers} workers -> {args.out}")

    if n_workers == 1:
        for w in work:
            results.append(_export_one(w))
            if len(results) % 20 == 0:
                print(f"  ... {len(results)}/{len(work)}")
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = {ex.submit(_export_one, w): w for w in work}
            for i, fut in enumerate(as_completed(futs), 1):
                results.append(fut.result())
                if i % 20 == 0 or i == len(work):
                    print(f"  ... {i}/{len(work)}")

    # out_parquet -> source task dir
    src_task_by_out = {dst: Path(task_dir) for (_src, dst, task_dir) in collected}

    task_groups = {}
    for r in results:
        out_p = Path(r["out"])
        src_task = src_task_by_out.get(str(out_p))
        if src_task is None:
            task_name = out_p.parts[-4] if len(out_p.parts) >= 4 else "unknown"
            src_task = Path(args.root) / task_name
            if not src_task.is_dir():
                src_task = Path(args.root)
        else:
            task_name = src_task.name
        g = task_groups.setdefault(
            task_name,
            {"src": src_task, "out": Path(args.out) / task_name, "results": []},
        )
        g["results"].append(r)

    for task_name, g in sorted(task_groups.items()):
        meta_info = _write_task_meta(
            g["out"],
            g["src"],
            args.embodiment,
            g["results"],
            link_videos=args.link_videos,
        )
        print(
            f"  meta[{task_name}]: episodes={meta_info['episodes']} "
            f"frames={meta_info['frames']} episodes.jsonl={meta_info['episodes_jsonl']} "
            f"videos -> {meta_info.get('videos_link')}"
        )

    summary = {
        "root": args.root,
        "out": args.out,
        "embodiment": args.embodiment,
        "num_episodes": len(results),
        "num_frames": sum(r["T"] for r in results),
        "mask_active": results[0]["active"] if results else 0,
        "layout": "LeRobot v2.1 (data/ + meta/ + videos symlink)",
        "link_videos": args.link_videos,
        "tasks": {
            k: {
                "episodes": len(v["results"]),
                "frames": sum(r["T"] for r in v["results"]),
            }
            for k, v in task_groups.items()
        },
    }
    with open(Path(args.out) / "export_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(
        f"Done: {summary['num_episodes']} episodes, {summary['num_frames']} frames, "
        f"mask_active={summary['mask_active']}/{UNIFIED_DIM}"
    )
    print(f"summary -> {Path(args.out) / 'export_summary.json'}")


if __name__ == "__main__":
    main()
