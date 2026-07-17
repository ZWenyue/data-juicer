#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Acceptance checks for merge_lerobot (LeRobot v2.1 concat)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq


def _fail(msg: str) -> None:
    print(f"[FAIL] {msg}", file=sys.stderr)
    raise SystemExit(1)


def _ok(msg: str) -> None:
    print(f"[OK] {msg}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--merged", required=True, help="Merged LeRobot root")
    p.add_argument("--expect-episodes", type=int, required=True)
    p.add_argument("--expect-min-frames", type=int, default=1)
    p.add_argument("--require-videos", action="store_true")
    args = p.parse_args()

    root = Path(args.merged)
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        _fail(f"missing {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))

    eps_path = root / "meta" / "episodes.jsonl"
    episodes = []
    with open(eps_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))

    if info.get("total_episodes") != args.expect_episodes:
        _fail(
            f"info.total_episodes={info.get('total_episodes')} "
            f"!= expect {args.expect_episodes}"
        )
    if len(episodes) != args.expect_episodes:
        _fail(f"episodes.jsonl rows={len(episodes)} != {args.expect_episodes}")
    _ok(f"episodes={args.expect_episodes}")

    indices = [int(e["episode_index"]) for e in episodes]
    if indices != list(range(args.expect_episodes)):
        _fail(f"episode_index not contiguous 0..N-1: {indices[:10]}...")
    _ok("episode_index contiguous")

    total_frames = int(info.get("total_frames") or 0)
    if total_frames < args.expect_min_frames:
        _fail(f"total_frames={total_frames} < {args.expect_min_frames}")
    _ok(f"total_frames={total_frames}")

    # Spot-check first / last parquet remapping.
    for ep in (0, args.expect_episodes - 1):
        chunk = ep // int(info.get("chunks_size") or 1000)
        pq_path = root / "data" / f"chunk-{chunk:03d}" / f"episode_{ep:06d}.parquet"
        if not pq_path.is_file():
            _fail(f"missing parquet {pq_path}")
        table = pq.read_table(pq_path, columns=["episode_index", "index"])
        ep_col = table.column("episode_index").to_pylist()
        idx_col = table.column("index").to_pylist()
        if any(int(v) != ep for v in ep_col):
            _fail(f"{pq_path.name}: episode_index not all {ep}")
        if idx_col != list(range(idx_col[0], idx_col[0] + len(idx_col))):
            _fail(f"{pq_path.name}: index not contiguous")
        # state dim
        st = pq.read_table(pq_path, columns=["observation.state"])
        row0 = st.column("observation.state")[0].as_py()
        if len(row0) != 80:
            _fail(f"{pq_path.name}: observation.state dim={len(row0)} != 80")
    _ok("parquet remapping + unified80 dims")

    if info.get("unified_dim") != 80:
        _fail(f"unified_dim={info.get('unified_dim')}")
    if "mixed" not in str(info.get("robot_type")) and len(info.get("robot_types") or []) > 1:
        _fail("expected robot_type=mixed for multi-robot merge")
    _ok(f"robot_types={info.get('robot_types')}")

    if args.require_videos:
        if not (root / "videos").is_dir():
            _fail("videos/ missing")
        # At least one linked mp4 for ep0
        ep0 = episodes[0]
        vkeys = ep0.get("video_keys") or []
        if not vkeys:
            _fail("episode 0 has empty video_keys")
        vk = vkeys[0]
        vid = root / "videos" / "chunk-000" / vk / "episode_000000.mp4"
        if not (vid.is_file() or vid.is_symlink()):
            _fail(f"missing video link {vid}")
        _ok(f"videos linked (ep0 key={vk})")

    sources = root / "meta" / "sources.jsonl"
    if not sources.is_file():
        _fail("missing sources.jsonl")
    _ok("sources.jsonl present")

    print(json.dumps({"status": "pass", "merged": str(root), **{k: info.get(k) for k in (
        "total_episodes", "total_frames", "robot_type", "fps", "video_policy"
    )}}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
