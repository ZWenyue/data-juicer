# -*- coding: utf-8 -*-
"""Pre-clean analyzer: probe a single LeRobot task and suggest cleaning thresholds.

Mirrors :mod:`run_robot_clean` but is *read-only* / non-destructive: it runs the
Stage1/2/3 filters in stats-only mode and samples the Check3 video scorer, then
emits suggested CLI flags for ``run_robot_clean`` (numeric + video quality).

Numeric side reuses the real operators' ``compute_stats_single`` so suggestions
match true cleaning behaviour; video side reuses the Check3 scorer's frame
metrics (blackness + blur Laplacian variance).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger

from data_juicer import _au  # noqa: F401  (triggers operator registration)
from data_juicer.ops.base_op import OPERATORS
from data_juicer.utils.constant import Fields

from ...utils.lerobot_episode_io import list_episode_parquets, load_episode_arrays
from .config import CleanConfig
from .prepare import compute_embodiment_percentiles
from .recipe import read_source_fps

DA_CANDIDATES = (0.60, 0.65, 0.70)
TARGET_KEEP = 0.90


def _pcts(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=float)
    return {
        "min": float(np.min(x)),
        "p01": float(np.percentile(x, 1)),
        "p05": float(np.percentile(x, 5)),
        "p10": float(np.percentile(x, 10)),
        "p50": float(np.percentile(x, 50)),
        "p90": float(np.percentile(x, 90)),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
    }


# --------------------------------------------------------------------------- #
# Numeric side (Stage 1 / 2 / 3)
# --------------------------------------------------------------------------- #
def collect_numeric_stats(
    files: List[str],
    cfg: CleanConfig,
    percentiles_path: str,
    probe_alpha: float,
) -> List[Dict[str, Any]]:
    """Run S1/S2/S3 ``compute_stats_single`` per episode; return stats rows."""
    s1 = OPERATORS.modules["robot_sudden_change_filter"](
        signal_source="top_level",
        top_level_state_key="states",
        top_level_action_key="actions",
        threshold_mode="mad",
        mad_scale_residual=6.0,
        mad_scale_acc=6.0,
        mad_scale_jerk=6.0,
        max_flagged_ratio=cfg.s1_max_flagged_ratio,
        max_run_length=cfg.s1_max_run_length,
        min_frames=cfg.s1_min_frames,
        exclusion_strategy="frame_mask",
    )
    s2 = OPERATORS.modules["robot_state_action_alignment_filter"](
        signal_source="top_level",
        top_level_state_key="states",
        top_level_action_key="actions",
        shared_dims=list(cfg.s2_shared_dims),
        action_is_delta=False,
        max_lag=15,
        da_threshold=cfg.s2_da_threshold,
        eps_mode="range_frac",
        eps_frac=0.01,
        min_active_frames=10,
        min_frames=20,
        exclusion_strategy="flag_only",
    )
    s3 = OPERATORS.modules["robot_extreme_value_filter"](
        signal_source="top_level",
        top_level_state_key="states",
        top_level_action_key="actions",
        percentile_source="stats_json",
        percentile_stats_path=percentiles_path,
        embodiment=cfg.embodiment,
        alpha=probe_alpha,
        exempt_dims=list(cfg.s3_exempt_dims),
        exclusion_strategy="frame_mask",
    )

    rows: List[Dict[str, Any]] = []
    for pf in files:
        states, actions = load_episode_arrays(pf)
        sample = {
            "states": states.tolist(),
            "actions": actions.tolist(),
            Fields.stats: {},
            Fields.meta: {},
        }
        for op in (s1, s2, s3):
            op.compute_stats_single(sample)
        rows.append({"id": Path(pf).stem, "stats": dict(sample[Fields.stats])})
    return rows


def suggest_numeric(
    rows: List[Dict[str, Any]],
    probe_alpha: float,
    s1_max_run_length: int = 10,
    s3_max_flagged_ratio: float = 0.3,
) -> Dict[str, Any]:
    """Suggest S1/S2/S3 thresholds from collected per-episode stats."""
    n = len(rows)
    ratio = np.array([r["stats"]["sudden_change_flagged_ratio"] for r in rows], dtype=float)
    max_run = np.array([r["stats"]["sudden_change_max_run"] for r in rows], dtype=float)
    min_da = np.array([r["stats"]["state_action_min_da"] for r in rows], dtype=float)
    mean_da = np.array([r["stats"]["state_action_mean_da"] for r in rows], dtype=float)
    ev = np.array([r["stats"]["extreme_value_flagged_ratio"] for r in rows], dtype=float)

    max_flagged_ratio = float(np.clip(np.percentile(ratio, TARGET_KEEP * 100), 0.05, 0.5))

    da_table = []
    best_da, best_score = DA_CANDIDATES[0], None
    for cand in DA_CANDIDATES:
        drop_frac = float(np.mean(min_da < cand))
        da_table.append({"da_threshold": cand, "drop_frac": drop_frac})
        score = abs(drop_frac - 0.10)
        if best_score is None or score < best_score:
            best_score, best_da = score, cand

    mean_ev = float(np.mean(ev))
    if mean_ev > 0.15:
        alpha = 0.2 if mean_ev <= 0.25 else 0.3
    else:
        alpha = probe_alpha

    # Pathologically high flagging usually means near-constant dims (q01≈q99 →
    # zero-width band). With episode_discard this would wipe most of the set.
    s3_warning = None
    if mean_ev > 0.5:
        s3_warning = (
            f"Stage3 极值标记率极高 (mean={mean_ev:.2f})：很可能存在近常数维 "
            f"(q01≈q99，带宽≈0)。当前为 episode_discard，会大批丢弃 episode；"
            f"请核查体态分位数 / 扩充 exempt_dims，或临时改回 frame_mask。"
        )

    # Per-episode reject under suggested thresholds + episode_discard.
    s1_rej = (ratio > max_flagged_ratio) | (max_run > s1_max_run_length)
    s2_rej = min_da < best_da
    s3_rej = ev > float(s3_max_flagged_ratio)
    numeric_union = s1_rej | s2_rej | s3_rej

    return {
        "n_episodes": n,
        "max_flagged_ratio": round(max_flagged_ratio, 4),
        "da_threshold": best_da,
        "alpha": alpha,
        "s3_max_flagged_ratio": float(s3_max_flagged_ratio),
        "s3_warning": s3_warning,
        "s1_ratio_pcts": _pcts(ratio),
        "s1_max_run_pcts": _pcts(max_run),
        "s2_min_da_pcts": _pcts(min_da),
        "s2_mean_da_pcts": _pcts(mean_da),
        "s2_da_table": da_table,
        "s3_ev_pcts": _pcts(ev),
        "s3_mean_flagged_ratio": mean_ev,
        "wash": {
            "s1_mean_flagged_frame_frac": float(np.mean(ratio)),
            "s1_episode_drop_frac": float(np.mean(s1_rej)),
            "s2_episode_drop_frac": float(np.mean(s2_rej)),
            "s3_mean_flagged_frame_frac": mean_ev,
            "s3_episode_drop_frac": float(np.mean(s3_rej)),
            "numeric_union_episode_drop_frac": float(np.mean(numeric_union)),
        },
    }


# --------------------------------------------------------------------------- #
# Video side (Check3 blur / blackness)
# --------------------------------------------------------------------------- #
def collect_video_stats(
    video_files: List[str],
    decoder: str,
    sampling_fps: Optional[float],
    original_fps: float,
    blur_threshold: float,
    blackness_threshold: float,
) -> Dict[str, Any]:
    """Score sampled frames with the real Check3 scorer; aggregate distributions."""
    scorer = OPERATORS.modules["robot_frame_quality_scorer_mapper"](
        blackness_threshold=blackness_threshold,
        blur_threshold=blur_threshold,
        decoder=decoder,
        sampling_fps=sampling_fps,
        original_fps=original_fps,
    )
    blur_all: List[float] = []
    black_all: List[float] = []
    per_ep: List[Dict[str, Any]] = []
    n_decoded = 0
    for vf in video_files:
        try:
            frames = scorer._load_frames_from_video(vf)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"decode failed for {vf}: {e}")
            continue
        if not frames:
            continue
        n_decoded += 1
        b_list, bl_list = [], []
        for fr in frames:
            sc = scorer._score_frame(fr)
            b_list.append(sc["blackness"])
            bl_list.append(sc["blur_laplacian_var"])
        blur_all.extend(bl_list)
        black_all.extend(b_list)
        per_ep.append({"blur": bl_list, "black": b_list, "n_sampled": len(frames)})

    if n_decoded == 0:
        return {"n_videos_decoded": 0}

    return {
        "n_videos_decoded": n_decoded,
        "blur_pcts": _pcts(np.array(blur_all)),
        "blackness_pcts": _pcts(np.array(black_all)),
        "probe_sampling_fps": sampling_fps,
        "original_fps": original_fps,
        "_blur": np.array(blur_all),
        "_black": np.array(black_all),
        "_per_ep": per_ep,
    }


def suggest_video(
    vstats: Dict[str, Any],
    max_bad_ratio: float = 0.1,
    min_good_frames: int = 20,
) -> Dict[str, Any]:
    """Suggest blur/blackness thresholds and estimate Check3 episode drop."""
    blur = vstats["_blur"]
    black = vstats["_black"]

    # Flag frames noticeably blurrier than typical: sit just below the low tail.
    blur_thr = float(round(max(1.0, np.percentile(blur, 2)), 1))
    # Blackness: normal frames are bright; keep default 10 unless data is very dark.
    p01_black = float(np.percentile(black, 1))
    black_thr = 10.0 if p01_black >= 20.0 else float(round(max(1.0, p01_black * 0.5), 1))

    def _flag_frac_below(vals: np.ndarray, thr: float) -> float:
        return float(np.mean(np.asarray(vals) < thr))

    blur_candidates = sorted({round(float(np.percentile(blur, q)), 1) for q in (1, 2, 5, 10)})
    blur_table = [{"blur_threshold": t, "flag_frac": _flag_frac_below(blur, t)} for t in blur_candidates]

    # Estimate Check3 episode reject under suggested thresholds.
    # When probe used sampling, scale good-frame count back to full-length estimate
    # (default clean scores all frames; min_good_frames applies to full T).
    sampling_fps = vstats.get("probe_sampling_fps")
    original_fps = float(vstats.get("original_fps") or 15.0)
    scale = (
        float(original_fps) / float(sampling_fps)
        if sampling_fps is not None and float(sampling_fps) > 0
        else 1.0
    )

    reject = []
    bad_ratios = []
    for ep in vstats.get("_per_ep") or []:
        bl = np.asarray(ep["blur"], dtype=float)
        bk = np.asarray(ep["black"], dtype=float)
        n_s = max(1, int(ep["n_sampled"]))
        bad_mask = (bl < blur_thr) | (bk < black_thr)
        bad_ratio = float(bad_mask.mean())
        # Estimate full-length good frames for the min_good_frames gate.
        n_full_est = max(n_s, int(round(n_s * scale)))
        good_full_est = int(round((1.0 - bad_ratio) * n_full_est))
        drop = (bad_ratio > max_bad_ratio) or (good_full_est < min_good_frames)
        reject.append(drop)
        bad_ratios.append(bad_ratio)

    episode_drop_frac = float(np.mean(reject)) if reject else 0.0
    mean_bad_ratio = float(np.mean(bad_ratios)) if bad_ratios else 0.0

    return {
        "suggested_blur_threshold": blur_thr,
        "suggested_blackness_threshold": black_thr,
        "blur_flag_table": blur_table,
        "blur_flag_frac_at_suggested": _flag_frac_below(blur, blur_thr),
        "wash": {
            "check3_max_bad_ratio": max_bad_ratio,
            "check3_min_good_frames": min_good_frames,
            "mean_bad_ratio_at_suggested": mean_bad_ratio,
            "episode_drop_frac": episode_drop_frac,
            "n_videos_probed": len(reject),
            "note": (
                "基于探针视频+抽帧的近似估计（未含关键帧污染门控）；"
                "默认清洗全帧评分时以实际 Check3 结果为准。"
            ),
        },
    }


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _list_videos(dataset: str, files: List[str], video_key: str) -> List[str]:
    root = Path(dataset).resolve()
    out = []
    for pf in files:
        p = Path(pf)
        vid = root / "videos" / p.parent.name / video_key / f"{p.stem}.mp4"
        if vid.is_file():
            out.append(str(vid))
    return out


def analyze_task(
    dataset: str,
    output: str,
    embodiment: str = "galaxea_r1_lite",
    video_key: str = "observation.images.head_rgb",
    max_episodes: Optional[int] = None,
    probe_video_episodes: int = 8,
    probe_sampling_fps: float = 2.0,
    probe_alpha: float = 0.1,
    analyze_video: bool = True,
) -> Dict[str, Any]:
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = CleanConfig(dataset=dataset, output_dir=output, embodiment=embodiment, video_key=video_key)

    files = list_episode_parquets(dataset)
    if not files:
        raise FileNotFoundError(f"No episode parquet under {dataset}")
    if max_episodes is not None:
        files = files[: int(max_episodes)]

    logger.info(f"[numeric] scanning {len(files)} episodes...")
    pct_path = str(out_dir / "percentiles.json")
    compute_embodiment_percentiles(dataset, pct_path, embodiment, max_files=max_episodes)
    rows = collect_numeric_stats(files, cfg, pct_path, probe_alpha)
    numeric = suggest_numeric(
        rows,
        probe_alpha,
        s1_max_run_length=cfg.s1_max_run_length,
        s3_max_flagged_ratio=cfg.s3_max_flagged_ratio,
    )

    result: Dict[str, Any] = {
        "dataset": str(Path(dataset).resolve()),
        "embodiment": embodiment,
        "num_episodes": len(files),
        "numeric": numeric,
        "suggested_clean_flags": {
            "--s1-max-flagged-ratio": numeric["max_flagged_ratio"],
            "--s2-da-threshold": numeric["da_threshold"],
            "--s3-alpha": numeric["alpha"],
        },
    }

    check3_drop = None
    if analyze_video:
        vfiles = _list_videos(dataset, files, video_key)
        if not vfiles:
            logger.warning(f"[video] no videos for key {video_key}; skipping video probe")
            result["video"] = {"n_videos_decoded": 0, "reason": "no_video_key"}
        else:
            vfiles = vfiles[: int(probe_video_episodes)]
            fps = read_source_fps(dataset)
            logger.info(f"[video] scoring {len(vfiles)} videos @ sampling_fps={probe_sampling_fps} (src {fps})...")
            vstats = collect_video_stats(
                vfiles,
                decoder=cfg.check3_decoder,
                sampling_fps=probe_sampling_fps,
                original_fps=fps,
                blur_threshold=cfg.check3_blur_threshold,
                blackness_threshold=cfg.check3_blackness_threshold,
            )
            if vstats.get("n_videos_decoded", 0) > 0:
                vsug = suggest_video(
                    vstats,
                    max_bad_ratio=cfg.check3_max_bad_ratio,
                    min_good_frames=cfg.check3_min_good_frames,
                )
                # strip raw arrays before serialising
                vstats_clean = {k: v for k, v in vstats.items() if not k.startswith("_")}
                result["video"] = {**vstats_clean, "suggestion": vsug}
                result["suggested_clean_flags"]["--check3-blur-threshold"] = vsug["suggested_blur_threshold"]
                result["suggested_clean_flags"]["--check3-blackness-threshold"] = vsug["suggested_blackness_threshold"]
                check3_drop = vsug["wash"]["episode_drop_frac"]
            else:
                result["video"] = {"n_videos_decoded": 0, "reason": "decode_failed"}

    # Default clean: S1/S2/S3 + Check3 all use episode-level discard.
    nw = numeric["wash"]
    numeric_union = float(nw["numeric_union_episode_drop_frac"])
    if check3_drop is None:
        default_drop = numeric_union
        note = (
            "Stage1/2/3 均为 episode_discard；合计为三者并集。"
            "未跑 Check3 视频探针，合计未含视频门控。"
        )
    else:
        # Independence approx between numeric union and Check3 probe rate.
        default_drop = 1.0 - (1.0 - numeric_union) * (1.0 - float(check3_drop))
        note = (
            "Stage1/2/3 均为 episode_discard（数值侧为逐 episode 并集）；"
            "再与 Check3 探针丢弃率按独立近似合成。"
            "Check3 未含关键帧污染；全量清洗以实际结果为准。"
        )
    n_eps = len(files)
    result["estimated_wash"] = {
        "default_pipeline_episode_drop_frac": default_drop,
        "default_pipeline_episodes_dropped_est": int(round(default_drop * n_eps)),
        "default_pipeline_episodes_kept_est": int(round((1.0 - default_drop) * n_eps)),
        "numeric_union_episode_drop_frac": numeric_union,
        "note": note,
        "stages": {
            "stage1": {
                "strategy": cfg.s1_exclusion_strategy,
                "episode_drop_frac": nw["s1_episode_drop_frac"],
                "mean_flagged_frame_frac": nw["s1_mean_flagged_frame_frac"],
            },
            "stage2": {
                "strategy": cfg.s2_exclusion_strategy,
                "episode_drop_frac": nw["s2_episode_drop_frac"],
            },
            "stage3": {
                "strategy": cfg.s3_exclusion_strategy,
                "episode_drop_frac": nw["s3_episode_drop_frac"],
                "mean_flagged_frame_frac": nw["s3_mean_flagged_frame_frac"],
                "max_flagged_ratio": cfg.s3_max_flagged_ratio,
            },
            "check3": {
                "episode_drop_frac": check3_drop,
                "probed": check3_drop is not None,
            },
        },
    }

    (out_dir / "analysis.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_markdown(result, out_dir / "threshold_report.md")
    logger.info(f"Analysis -> {out_dir / 'analysis.json'}")
    wash = result["estimated_wash"]
    logger.info(
        f"Estimated wash (default pipeline): drop {wash['default_pipeline_episode_drop_frac']:.1%} "
        f"({wash['default_pipeline_episodes_dropped_est']}/{n_eps} episodes)"
    )
    return result


def _write_markdown(result: Dict[str, Any], path: Path) -> None:
    num = result["numeric"]
    wash = result.get("estimated_wash") or {}
    n = result["num_episodes"]
    lines = [f"# 阈值分析报告：{Path(result['dataset']).name}\n"]
    lines.append(f"- Episodes: **{n}**  embodiment: `{result['embodiment']}`\n")

    # ---- wash summary (most actionable) ----
    lines.append("\n## 预计洗掉比例（默认清洗流水线）\n")
    drop = wash.get("default_pipeline_episode_drop_frac")
    if drop is None:
        lines.append("未能估计（缺少视频探针）。\n")
    else:
        dropped = wash.get("default_pipeline_episodes_dropped_est", 0)
        kept = wash.get("default_pipeline_episodes_kept_est", n)
        lines.append(
            f"**预计丢弃 episode：`{drop:.1%}`** "
            f"（约 **{dropped}/{n}** 条被洗掉，保留约 **{kept}/{n}**）\n\n"
        )
        lines.append(f"> {wash.get('note', '')}\n\n")
        st = wash.get("stages") or {}
        lines.append("| 阶段 | 策略 | 预计丢 episode | 备注 |\n|---|---|---|---|\n")
        s1 = st.get("stage1") or {}
        s2 = st.get("stage2") or {}
        s3 = st.get("stage3") or {}
        c3 = st.get("check3") or {}
        lines.append(
            f"| Stage1 突变 | `{s1.get('strategy')}` | "
            f"{s1.get('episode_drop_frac', 0):.1%} | "
            f"标记帧均值 {s1.get('mean_flagged_frame_frac', 0):.1%} |\n"
        )
        lines.append(
            f"| Stage2 对齐 | `{s2.get('strategy')}` | "
            f"{s2.get('episode_drop_frac', 0):.1%} | "
            f"min_da < 建议阈值 |\n"
        )
        lines.append(
            f"| Stage3 极值 | `{s3.get('strategy')}` | "
            f"{s3.get('episode_drop_frac', 0):.1%} | "
            f"标记帧均值 {s3.get('mean_flagged_frame_frac', 0):.1%}；"
            f"丢弃当 flagged_ratio > {s3.get('max_flagged_ratio', 0.3)} |\n"
        )
        c3_drop = c3.get("episode_drop_frac")
        c3_txt = "未探测" if c3_drop is None else f"{c3_drop:.1%}"
        lines.append(
            f"| Check3 视频 | episode 门控 | {c3_txt} | 探针近似 |\n"
        )
        nu = wash.get("numeric_union_episode_drop_frac")
        if nu is not None:
            lines.append(
                f"\n数值侧并集（S1∪S2∪S3）预计丢 **{nu:.1%}**；"
                f"再叠加 Check3 后合计约 **{drop:.1%}**。\n"
            )

    lines.append("\n## Stage1 突变\n")
    p = num["s1_ratio_pcts"]
    lines.append(f"flagged_ratio: p50={p['p50']:.3f} p90={p['p90']:.3f} max={p['max']:.3f}\n\n")
    lines.append(f"**建议 `--s1-max-flagged-ratio {num['max_flagged_ratio']}`**\n")

    lines.append("\n## Stage2 状态-动作对齐\n")
    lines.append("| da_threshold | drop_frac |\n|---|---|\n")
    for t in num["s2_da_table"]:
        lines.append(f"| {t['da_threshold']:.2f} | {t['drop_frac']:.1%} |\n")
    lines.append(f"\n**建议 `--s2-da-threshold {num['da_threshold']}`**\n")

    lines.append("\n## Stage3 极值\n")
    p = num["s3_ev_pcts"]
    lines.append(f"flagged_ratio(alpha probe): mean={num['s3_mean_flagged_ratio']:.3f} p90={p['p90']:.3f}\n\n")
    lines.append(f"**建议 `--s3-alpha {num['alpha']}`**（夹爪维 exempt_dims 保持 [7,15]）\n")
    if num.get("s3_warning"):
        lines.append(f"\n> ⚠️ {num['s3_warning']}\n")

    video = result.get("video", {})
    lines.append("\n## Check3 视频质量\n")
    if video.get("n_videos_decoded", 0) > 0:
        b = video["blur_pcts"]
        k = video["blackness_pcts"]
        sug = video["suggestion"]
        vw = sug.get("wash") or {}
        lines.append(f"解码视频数: {video['n_videos_decoded']}\n\n")
        lines.append(f"blur_laplacian_var: min={b['min']:.1f} p01={b['p01']:.1f} p05={b['p05']:.1f} p50={b['p50']:.1f}\n\n")
        lines.append(f"blackness(mean intensity): min={k['min']:.1f} p01={k['p01']:.1f} p50={k['p50']:.1f}\n\n")
        lines.append("| blur_threshold | flag_frac |\n|---|---|\n")
        for t in sug["blur_flag_table"]:
            lines.append(f"| {t['blur_threshold']} | {t['flag_frac']:.1%} |\n")
        lines.append(
            f"\n**建议 `--check3-blur-threshold {sug['suggested_blur_threshold']}` "
            f"`--check3-blackness-threshold {sug['suggested_blackness_threshold']}`** "
            f"（建议阈值下约 {sug['blur_flag_frac_at_suggested']:.1%} 帧被判模糊；"
            f"平均坏帧比 {vw.get('mean_bad_ratio_at_suggested', 0):.1%}；"
            f"预计丢 episode **{vw.get('episode_drop_frac', 0):.1%}**）\n"
        )
    else:
        lines.append(f"未评分（{video.get('reason', 'n/a')}）。\n")

    flags = " ".join(f"{kk} {vv}" for kk, vv in result["suggested_clean_flags"].items())
    lines.append("\n## 建议清洗命令（本任务）\n")
    lines.append("```bash\n")
    lines.append(
        "python -m data_juicer._au.pipeline.robot_clean.run_robot_clean \\\n"
        f"  --dataset {result['dataset']} \\\n"
        "  --output <OUT> \\\n"
        f"  --embodiment {result['embodiment']} \\\n"
        f"  {flags}\n"
    )
    lines.append("```\n")
    path.write_text("".join(lines), encoding="utf-8")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Analyze one LeRobot task and suggest robot_clean thresholds."
    )
    p.add_argument("--dataset", required=True, help="LeRobot task dir (contains data/).")
    p.add_argument("--output", required=True, help="Analysis output dir.")
    p.add_argument("--embodiment", default="galaxea_r1_lite")
    p.add_argument("--video-key", default="observation.images.head_rgb")
    p.add_argument("--max-episodes", type=int, default=None, help="Limit numeric scan episodes")
    p.add_argument("--probe-video-episodes", type=int, default=8, help="Videos to score for Check3 probe")
    p.add_argument("--probe-sampling-fps", type=float, default=2.0, help="Sampling fps for video probe")
    p.add_argument("--probe-alpha", type=float, default=0.1, help="Stage3 alpha used while probing")
    p.add_argument("--no-video", action="store_true", help="Skip Check3 video probe")
    return p


def main(argv=None) -> int:
    args = build_argparser().parse_args(argv)
    dataset = Path(args.dataset)
    if not (dataset / "data").is_dir():
        logger.error(f"--dataset must be a LeRobot task dir with data/: {dataset}")
        return 2
    result = analyze_task(
        dataset=str(dataset.resolve()),
        output=str(Path(args.output).resolve()),
        embodiment=args.embodiment,
        video_key=args.video_key,
        max_episodes=args.max_episodes,
        probe_video_episodes=args.probe_video_episodes,
        probe_sampling_fps=args.probe_sampling_fps,
        probe_alpha=args.probe_alpha,
        analyze_video=not args.no_video,
    )
    flags = " ".join(f"{k} {v}" for k, v in result["suggested_clean_flags"].items())
    wash = result.get("estimated_wash") or {}
    drop = wash.get("default_pipeline_episode_drop_frac")
    if drop is not None:
        logger.info(
            f"Estimated wash: drop {drop:.1%} "
            f"({wash.get('default_pipeline_episodes_dropped_est')}/"
            f"{result['num_episodes']} episodes)"
        )
    logger.info(f"Suggested clean flags: {flags}")
    print(flags)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
