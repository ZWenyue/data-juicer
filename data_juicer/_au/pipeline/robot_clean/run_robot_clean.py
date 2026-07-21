#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Production entry: clean LeRobot robot demos with data_juicer/_au ops.

Pipeline (default):
  pointer JSONL → Stage1/2/3 → Stage5 → Check3 (episode gate) → unified 80-dim
  optional: LeRobot-layout 80-dim parquet for kept episodes

Usage:
  python -m data_juicer._au.pipeline.robot_clean.run_robot_clean \\
      --dataset /path/to/LeRobot_task \\
      --output /path/to/out_dir

  # small smoke
  python -m data_juicer._au.pipeline.robot_clean.run_robot_clean \\
      --dataset ... --output ... --max-episodes 8

  # skip visual Check3 / skip JSONL 80-dim / enable parquet export
  python -m data_juicer._au.pipeline.robot_clean.run_robot_clean \\
      --dataset ... --output ... \\
      --no-check3 --export-unified-parquet
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from loguru import logger

from ...utils.lerobot_episode_io import resolve_video_key
from .config import CleanConfig
from .export_unified import export_kept_unified_parquets
from .prepare import prepare
from .recipe import write_recipe


def _repo_root() -> Path:
    # data_juicer/_au/pipeline/robot_clean/run_robot_clean.py → repo root
    return Path(__file__).resolve().parents[4]


def _run_dj_process(recipe_path: Path, np_override: int | None = None) -> None:
    """Invoke process_data with *this* interpreter (avoid PATH dj-process / wrong env)."""
    env = os.environ.copy()
    root = str(_repo_root())
    env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    tools_main = _repo_root() / "tools" / "process_data.py"
    if tools_main.is_file():
        cmd = [sys.executable, str(tools_main), "--config", str(recipe_path)]
    else:
        # Fallback: console script next to the current python
        dj = Path(sys.executable).parent / "dj-process"
        if dj.is_file():
            cmd = [str(dj), "--config", str(recipe_path)]
        else:
            cmd = [sys.executable, "-m", "data_juicer.tools.process_data", "--config", str(recipe_path)]

    logger.info(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(_repo_root()), env=env)


def _summarize(cfg: CleanConfig) -> dict:
    path = cfg.result_path
    if not path.is_file():
        cands = sorted(cfg.work_dir.glob(path.name + "*"))
        cands = [p for p in cands if p.is_file()]
        if not cands:
            return {"kept_episodes": 0, "result": None}
        path = cands[0]

    n = 0
    with_video_gate = 0
    with_unified = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            n += 1
            stats = row.get("__dj__stats__") or {}
            if stats.get("video_quality_episode_keep") is True:
                with_video_gate += 1
            if "unified_states" in row:
                with_unified += 1
    return {
        "kept_episodes": n,
        "result": str(path),
        "with_check3_keep_flag": with_video_gate,
        "with_unified_fields": with_unified,
    }


def run(cfg: CleanConfig) -> dict:
    """Execute prepare → recipe → dj-process → optional parquet export."""
    logger.info(f"Robot clean: dataset={cfg.dataset}")
    logger.info(f"Output dir: {cfg.output_dir}")
    cfg.work_dir.mkdir(parents=True, exist_ok=True)

    # Clear previous export shards
    for p in cfg.work_dir.glob(cfg.result_path.name + "*"):
        if p.is_file():
            p.unlink()

    prepare(cfg)
    recipe = write_recipe(cfg)
    logger.info(f"Wrote recipe: {recipe}")

    _run_dj_process(recipe)

    summary = _summarize(cfg)
    summary["recipe"] = str(recipe)
    summary["pointer"] = str(cfg.pointer_path)
    summary["stages"] = {
        "stage1": cfg.enable_stage1,
        "stage2": cfg.enable_stage2,
        "stage3": cfg.enable_stage3,
        "stage5": cfg.enable_stage5,
        "check3": cfg.enable_check3,
        "unified": cfg.enable_unified,
        "export_unified_parquet": cfg.export_unified_parquet,
    }

    if cfg.export_unified_parquet:
        if summary.get("kept_episodes", 0) == 0:
            logger.warning(
                "No episodes survived cleaning; skipping unified parquet export. "
                "Check the video quality gate (e.g. --check3-blur-threshold) or stage filters."
            )
        else:
            if not cfg.enable_unified:
                logger.warning(
                    "--export-unified-parquet requested but --no-unified was set; "
                    "export will pack from source parquet for kept ids."
                )
            export_summary = export_kept_unified_parquets(
                str(cfg.result_path),
                cfg.dataset,
                str(cfg.unified_parquet_dir),
                embodiment=cfg.embodiment,
            )
            summary["unified_parquet"] = export_summary

    summary_path = cfg.work_dir / "run_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"Summary -> {summary_path}")
    logger.info(
        f"Done. kept={summary.get('kept_episodes')} result={summary.get('result')}"
    )
    return summary


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Production robot data cleaning with data_juicer/_au "
            "(Stage1/2/3/5 + Check3 episode gate + unified 80-dim)."
        )
    )
    p.add_argument(
        "--dataset",
        required=True,
        help="LeRobot v2.1 task directory (contains data/ meta/ videos/).",
    )
    p.add_argument(
        "--output",
        required=True,
        help="Output work directory (pointer, recipe, cleaned.jsonl, stats).",
    )
    p.add_argument("--embodiment", default="galaxea_r1_lite")
    p.add_argument(
        "--video-key",
        default="observation.images.head_rgb",
        help="Camera key under videos/chunk-*/{video_key}/",
    )
    p.add_argument("--max-episodes", type=int, default=None)
    p.add_argument("--np", type=int, default=1, help="data-juicer num_proc")
    p.add_argument(
        "--executor-type",
        default="default",
        choices=["default", "ray", "ray_partitioned"],
    )

    g = p.add_argument_group("stage toggles")
    g.add_argument("--no-stage1", action="store_true")
    g.add_argument("--no-stage2", action="store_true")
    g.add_argument("--no-stage3", action="store_true")
    g.add_argument("--no-stage5", action="store_true")
    g.add_argument("--no-check3", action="store_true", help="Skip video quality gate")
    g.add_argument(
        "--no-unified",
        action="store_true",
        help="Skip writing unified_states/actions/mask into cleaned JSONL",
    )
    g.add_argument(
        "--export-unified-parquet",
        action="store_true",
        help="Also export LeRobot-layout 80-dim parquet for kept episodes",
    )

    gn = p.add_argument_group("numeric stage thresholds (override CleanConfig defaults)")
    gn.add_argument(
        "--s1-max-flagged-ratio",
        type=float,
        default=None,
        help="Stage1 episode-level max flagged frame ratio (default 0.3)",
    )
    gn.add_argument(
        "--s1-max-run-length",
        type=int,
        default=None,
        help="Stage1 max consecutive flagged frames before episode discard (default 10)",
    )
    gn.add_argument(
        "--s2-da-threshold",
        type=float,
        default=None,
        help="Stage2 direction-agreement threshold (default 0.65)",
    )
    gn.add_argument(
        "--s3-alpha",
        type=float,
        default=None,
        help="Stage3 IQR bandwidth multiplier alpha (default 0.1)",
    )

    g3 = p.add_argument_group("check3 / percentiles")
    g3.add_argument("--percentiles", default=None, help="Reuse existing percentiles JSON")
    g3.add_argument("--check3-max-bad-ratio", type=float, default=0.1)
    g3.add_argument("--check3-min-good-frames", type=int, default=20)
    g3.add_argument(
        "--check3-max-keyframe-overlap",
        type=int,
        default=None,
        help=(
            "Max allowed bad/keyframe overlap before discard "
            "(default: disabled; overlap is still reported)"
        ),
    )
    g3.add_argument(
        "--check3-blackness-threshold",
        type=float,
        default=None,
        help="Mean-intensity threshold for black frames (default 10.0)",
    )
    g3.add_argument(
        "--check3-blur-threshold",
        type=float,
        default=None,
        help="Laplacian-variance threshold for blurred frames (fallback default 1.0)",
    )
    g3.add_argument("--check3-decoder", default="auto", choices=["auto", "pyav", "opencv", "ffmpeg"])
    g3.add_argument(
        "--check3-sampling-fps",
        type=float,
        default=None,
        help="Subsample frames when scoring (faster; None = all frames)",
    )
    return p


def main(argv=None) -> int:
    args = build_argparser().parse_args(argv)
    dataset = Path(args.dataset)
    if not (dataset / "data").is_dir():
        logger.error(f"--dataset must be a LeRobot task dir with data/: {dataset}")
        return 2

    video_key = resolve_video_key(str(dataset), preferred=args.video_key) or args.video_key
    if video_key != args.video_key:
        logger.info(f"Resolved video_key {args.video_key!r} → {video_key!r}")

    cfg_kwargs = dict(
        dataset=str(dataset.resolve()),
        output_dir=str(Path(args.output).resolve()),
        embodiment=args.embodiment,
        video_key=video_key,
        max_episodes=args.max_episodes,
        np=args.np,
        executor_type=args.executor_type,
        enable_stage1=not args.no_stage1,
        enable_stage2=not args.no_stage2,
        enable_stage3=not args.no_stage3,
        enable_stage5=not args.no_stage5,
        enable_check3=not args.no_check3,
        enable_unified=not args.no_unified,
        export_unified_parquet=args.export_unified_parquet,
        percentiles_path=args.percentiles,
        check3_max_bad_ratio=args.check3_max_bad_ratio,
        check3_min_good_frames=args.check3_min_good_frames,
        check3_decoder=args.check3_decoder,
        check3_sampling_fps=args.check3_sampling_fps,
    )
    # Optional threshold overrides: only apply when explicitly given so that
    # CleanConfig defaults remain the single source of truth otherwise.
    optional_overrides = {
        "s1_max_flagged_ratio": args.s1_max_flagged_ratio,
        "s1_max_run_length": args.s1_max_run_length,
        "s2_da_threshold": args.s2_da_threshold,
        "s3_alpha": args.s3_alpha,
        "check3_max_keyframe_overlap": args.check3_max_keyframe_overlap,
        "check3_blackness_threshold": args.check3_blackness_threshold,
        "check3_blur_threshold": args.check3_blur_threshold,
    }
    for key, val in optional_overrides.items():
        if val is not None:
            cfg_kwargs[key] = val

    cfg = CleanConfig(**cfg_kwargs)
    run(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
