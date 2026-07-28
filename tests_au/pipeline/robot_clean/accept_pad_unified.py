#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Acceptance checks for pad-only (keep-all) unified80 export."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

from data_juicer._au.utils.embodiment_layout import UNIFIED_DIM
from data_juicer._au.utils.lerobot_episode_io import list_episode_parquets


def _fail(msg: str) -> None:
    print(f"[FAIL] {msg}", file=sys.stderr)
    raise SystemExit(1)


def _ok(msg: str) -> None:
    print(f"[OK] {msg}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, help="Exported task root (data/ meta/)")
    p.add_argument("--src", required=True, help="Source LeRobot task root")
    p.add_argument("--expect-episodes", type=int, default=None)
    p.add_argument("--require-videos", action="store_true")
    args = p.parse_args()

    out = Path(args.out)
    src = Path(args.src)

    info_path = out / "meta" / "info.json"
    if not info_path.is_file():
        _fail(f"missing {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))

    if info.get("unified_export_mode") != "keep_all":
        _fail(f"unified_export_mode={info.get('unified_export_mode')!r} != 'keep_all'")
    _ok("export_mode=keep_all")

    if int(info.get("unified_dim") or 0) != UNIFIED_DIM:
        _fail(f"unified_dim={info.get('unified_dim')} != {UNIFIED_DIM}")
    _ok(f"unified_dim={UNIFIED_DIM}")

    src_n = len(list_episode_parquets(str(src)))
    expect = args.expect_episodes if args.expect_episodes is not None else src_n
    if int(info.get("total_episodes") or 0) != expect:
        _fail(f"info.total_episodes={info.get('total_episodes')} != expect {expect}")
    _ok(f"episodes={expect} (source had {src_n})")

    for name in ("episodes.jsonl", "tasks.jsonl", "episodes_stats.jsonl"):
        if not (out / "meta" / name).is_file():
            _fail(f"missing meta/{name}")
    _ok("meta files present")

    if args.require_videos:
        videos = out / "videos"
        if not (videos.is_symlink() or videos.is_dir()):
            _fail("videos/ missing")
        if not videos.exists():
            _fail("videos link broken")
        _ok("videos present")

    # Spot-check first parquet dims.
    pq_files = sorted((out / "data").glob("chunk-*/episode_*.parquet"))
    if not pq_files:
        _fail("no parquet under data/")
    table = pq.read_table(pq_files[0], columns=["observation.state", "action", "action_dim_mask"])
    for col in ("observation.state", "action", "action_dim_mask"):
        row0 = table.column(col)[0].as_py()
        if len(row0) != UNIFIED_DIM:
            _fail(f"{pq_files[0].name}: {col} dim={len(row0)} != {UNIFIED_DIM}")
    _ok(f"parquet dims OK ({pq_files[0].name})")

    if len(pq_files) != expect:
        _fail(f"exported parquet count={len(pq_files)} != expect {expect}")
    _ok(f"parquet count={len(pq_files)}")

    print("ACCEPT PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
