#!/usr/bin/env python3
"""Galaxea Open-World Dataset quality analysis — pure Data-Juicer version.

Uses DJ's NestedDataset.process() API with custom operators defined in
custom_ops_2.py to compute episode-level quality metrics and emit a
JSON report identical in structure to the original analyze_glx.py output.

Usage:
    /mnt/r/VENV/dj/bin/python analyze_glx_2.py [--data-root PATH] [--output-dir PATH]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import custom_ops_2  # noqa: E402  — triggers DJ operator registration

from data_juicer.core.data import NestedDataset
from data_juicer.utils.constant import Fields

FPS = 15


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def ensure_jsonl(data_root, output_dir):
    """Generate episodes_2.jsonl if not present."""
    jsonl_path = output_dir / "episodes_2.jsonl"
    if not jsonl_path.exists():
        from convert_to_jsonl_2 import main as convert_main
        sys.argv = ["convert_to_jsonl_2.py",
                     "--data-root", str(data_root),
                     "--output", str(jsonl_path)]
        convert_main()
    return jsonl_path


def run_analysis(jsonl_path):
    """Load dataset and run all DJ operators in analysis (stats-only) mode.

    Uses dataset.map() with compute_stats to collect metrics on ALL
    episodes without filtering any out.  The Mapper runs via op.run().
    """
    ds = NestedDataset.from_json(str(jsonl_path))
    print(f"  Loaded {len(ds)} episodes from {jsonl_path.name}")

    if Fields.stats not in ds.column_names:
        ds = ds.add_column(Fields.stats, [{}] * len(ds))
    if Fields.meta not in ds.column_names:
        ds = ds.add_column(Fields.meta, [{}] * len(ds))

    filters = [
        custom_ops_2.EpisodeOffTaskFilter(),
        custom_ops_2.TrajectorySALFilter(),
        custom_ops_2.TrajectoryDAFilter(),
        custom_ops_2.TrajectorySuddenChangeFilter(),
        custom_ops_2.QualityLabelFilter(
            qualified_task_index=25, unqualified_task_index=9),
    ]

    for filt in filters:
        op_name = filt._name or type(filt).__name__
        print(f"  Computing stats: {op_name}...")
        ds = ds.map(filt.compute_stats, num_proc=1)

    mapper = custom_ops_2.TrajectoryStatsMapper()
    print(f"  Running {mapper._name}...")
    ds = mapper.run(ds)

    return ds


def collect_basic_stats(ds):
    """Extract basic statistics from processed dataset."""
    lengths = [s["episode_length"] for s in ds]
    durations = [l / FPS for l in lengths]

    stats_list = [s.get(Fields.stats, {}) for s in ds]
    total_qual = sum(st.get("qualified_frames", 0) for st in stats_list)
    total_unqual = sum(st.get("unqualified_frames", 0) for st in stats_list)
    total_frames = sum(st.get("total_frames", s.get("episode_length", 0))
                       for st, s in zip(stats_list, ds))

    return {
        "num_episodes": len(ds),
        "total_frames": total_frames,
        "total_duration_sec": round(sum(durations), 1),
        "episode_length": {
            "min": int(min(lengths)),
            "max": int(max(lengths)),
            "mean": round(float(np.mean(lengths)), 1),
            "median": round(float(np.median(lengths)), 1),
            "std": round(float(np.std(lengths)), 1),
        },
        "episode_duration_sec": {
            "min": round(min(durations), 1),
            "max": round(max(durations), 1),
            "mean": round(float(np.mean(durations)), 1),
            "median": round(float(np.median(durations)), 1),
        },
        "quality_frames": {
            "qualified": total_qual,
            "unqualified": total_unqual,
            "qualified_ratio": round(
                total_qual / total_frames if total_frames else 0, 4),
        },
    }


def collect_sal_results(ds):
    """Aggregate SAL results from Fields.stats."""
    per_episode = {}
    sal_left_all, sal_right_all = [], []

    for s in ds:
        ep = s["episode_index"]
        st = s.get(Fields.stats, {})
        sal_l = st.get("sal_left_ee", 0.0)
        sal_r = st.get("sal_right_ee", 0.0)
        sal_c = st.get("sal_combined", 0.0)

        per_episode[str(ep)] = {
            "sal_left_mean": st.get("sal_left_mean", 0.0),
            "sal_right_mean": st.get("sal_right_mean", 0.0),
            "sal_left_ee": sal_l,
            "sal_right_ee": sal_r,
            "num_frames": s["episode_length"],
        }
        sal_left_all.append(sal_l)
        sal_right_all.append(sal_r)

    def _summary(vals):
        return {
            "min": round(min(vals), 2),
            "max": round(max(vals), 2),
            "mean": round(float(np.mean(vals)), 2),
            "median": round(float(np.median(vals)), 2),
            "std": round(float(np.std(vals)), 2),
        }

    return {
        "per_episode": per_episode,
        "summary": {
            "sal_left_ee": _summary(sal_left_all),
            "sal_right_ee": _summary(sal_right_all),
        },
    }


def collect_da_results(ds):
    """Aggregate directional alignment results."""
    per_episode = {}
    misaligned_eps = []

    for s in ds:
        ep = s["episode_index"]
        st = s.get(Fields.stats, {})
        n_mis = st.get("da_misaligned_dims", 0)
        total = st.get("da_total_checked", 0)
        details = st.get("da_details", [])

        per_episode[str(ep)] = {
            "misaligned_dims": n_mis,
            "total_checked": total,
            "misaligned_details": details,
            "episode_aligned": n_mis == 0,
        }
        if n_mis > 0:
            misaligned_eps.append(ep)

    return {
        "per_episode": per_episode,
        "num_misaligned_episodes": len(misaligned_eps),
        "misaligned_episode_indices": misaligned_eps,
    }


def collect_sudden_change_results(ds):
    """Aggregate sudden change detection results."""
    per_episode = {}
    flagged_eps = []

    for s in ds:
        ep = s["episode_index"]
        st = s.get(Fields.stats, {})
        total = st.get("sudden_change_flags", 0)
        dims = st.get("sudden_change_dims", [])

        per_episode[str(ep)] = {
            "total_flags": total,
            "flagged_dims": dims,
            "flagged": total > 0,
        }
        if total > 0:
            flagged_eps.append(ep)

    return {
        "per_episode": per_episode,
        "num_flagged_episodes": len(flagged_eps),
        "flagged_episode_indices": flagged_eps,
    }


def collect_anomalies(ds):
    """Collect off-task and length anomalies."""
    anomalies = []
    for s in ds:
        ep = s["episode_index"]
        st = s.get(Fields.stats, {})
        length = s["episode_length"]

        if st.get("is_off_task", False):
            anomalies.append({
                "episode_index": ep,
                "type": "off_task",
                "detail": f"Episode contains non-router-cable subtasks "
                          f"({len(s.get('subtasks', []))} subtasks, "
                          f"{length} frames)",
                "severity": "high",
            })
        if length > 5000:
            anomalies.append({
                "episode_index": ep,
                "type": "abnormal_length",
                "detail": f"Episode has {length} frames "
                          f"({length/FPS:.0f}s), far above typical",
                "severity": "medium",
            })
    return anomalies


def collect_length_distribution(ds):
    """Compute episode length histogram."""
    lengths = [s["episode_length"] for s in ds]
    bins = [0, 500, 1000, 1500, 2000, 3000, 5000, 10000]
    hist, _ = np.histogram(lengths, bins=bins)
    return {
        "bins": bins,
        "counts": hist.tolist(),
        "bin_labels": [f"{bins[i]}-{bins[i+1]}" for i in range(len(hist))],
    }


def collect_gripper_analysis(ds):
    """Extract gripper stats from __dj__meta__."""
    results = {}
    for s in ds:
        ep = s["episode_index"]
        meta = s.get(Fields.meta, {}) or {}
        gs = meta.get("gripper_stats", {})
        if gs:
            results[str(ep)] = gs
    return results


def collect_action_state_stats(ds):
    """Aggregate per-dim stats across all episodes."""
    import pandas as pd
    all_states, all_actions = [], []
    pad_s = None

    for s in ds:
        ppath = s.get("parquet_path", "")
        if not ppath:
            continue
        states, actions, df = custom_ops_2._load_episode_arrays(ppath)
        all_states.append(states)
        all_actions.append(actions)
        if pad_s is None:
            pad_s = np.array(df["observation.state_dim_is_pad"].iloc[0])

    if not all_states:
        return {}

    states = np.concatenate(all_states, axis=0)
    actions = np.concatenate(all_actions, axis=0)
    pad_a_row = np.array(
        pd.read_parquet(ds[0]["parquet_path"],
                        columns=["action_dim_is_pad"]
                        )["action_dim_is_pad"].iloc[0])

    valid_s = np.where(~pad_s)[0]
    valid_a = np.where(~pad_a_row)[0]

    state_names = {v: k for k, v in custom_ops_2.STATE_SLOT_NAMES.items()}
    action_names = {
        0: "left_arm_j0", 2: "left_arm_j2", 3: "left_arm_j3",
        4: "left_arm_j4", 5: "left_arm_j5", 6: "left_arm_j6",
        7: "left_gripper",
        8: "right_arm_j0", 10: "right_arm_j2", 11: "right_arm_j3",
        12: "right_arm_j4", 13: "right_arm_j5", 14: "right_arm_j6",
        15: "right_gripper",
        16: "torso_0", 17: "torso_1", 18: "torso_2",
        19: "torso_3", 20: "torso_4",
        24: "base_vx", 25: "base_vy", 26: "base_wz",
    }

    def dim_stats(data, valid_dims, name_map):
        results = {}
        for d in valid_dims:
            col = data[:, d]
            name = name_map.get(int(d), f"dim_{d}")
            results[name] = {
                "dim": int(d),
                "min": round(float(col.min()), 6),
                "max": round(float(col.max()), 6),
                "mean": round(float(col.mean()), 6),
                "std": round(float(col.std()), 6),
                "q01": round(float(np.percentile(col, 1)), 6),
                "q99": round(float(np.percentile(col, 99)), 6),
                "range": round(float(col.max() - col.min()), 6),
            }
        return results

    return {
        "state": {
            "total_dims": 56,
            "valid_dims": len(valid_s),
            "valid_indices": valid_s.tolist(),
            "per_dim": dim_stats(states, valid_s, state_names),
        },
        "action": {
            "total_dims": 50,
            "valid_dims": len(valid_a),
            "valid_indices": valid_a.tolist(),
            "per_dim": dim_stats(actions, valid_a, action_names),
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Galaxea dataset (Data-Juicer version)")
    parser.add_argument(
        "--data-root",
        default="/mnt/r/DATA/Galaxea-Open-World-Dataset/lerobot_door"
                "/Connect_Router_Cables_20250625_002",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).parent),
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)

    print("=== Galaxea Connect_Router_Cables Analysis (DJ version) ===")
    print(f"Data root: {data_root}\n")

    print("[1/4] Preparing per-episode JSONL...")
    jsonl_path = ensure_jsonl(data_root, output_dir)

    print("[2/4] Running DJ operators (analysis mode)...")
    ds = run_analysis(jsonl_path)

    print("[3/4] Collecting results...")
    bs = collect_basic_stats(ds)
    sal = collect_sal_results(ds)
    da = collect_da_results(ds)
    sc = collect_sudden_change_results(ds)
    anomalies = collect_anomalies(ds)
    len_dist = collect_length_distribution(ds)
    grip = collect_gripper_analysis(ds)

    print("[4/4] Computing global action/state statistics...")
    as_stats = collect_action_state_stats(ds)

    report = {
        "dataset": "Galaxea-Open-World / Connect_Router_Cables_20250625_002",
        "format": "LeRobot v2.1 unified (lerobot_door)",
        "pipeline": "Data-Juicer custom operators",
        "basic_stats": bs,
        "action_state_stats": as_stats,
        "smoothness_sal": sal,
        "sudden_changes": sc,
        "directional_alignment": da,
        "anomalies": anomalies,
        "length_distribution": len_dist,
        "gripper_analysis": grip,
    }

    out_path = output_dir / "analysis_results_2.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    print(f"\nReport saved to {out_path}")

    # ── summary ──
    print("\n" + "=" * 60)
    print("QUALITY SUMMARY")
    print("=" * 60)
    print(f"  Episodes: {bs['num_episodes']}")
    print(f"  Frames: {bs['total_frames']}")
    print(f"  Duration: {bs['total_duration_sec']}s "
          f"({bs['total_duration_sec']/60:.1f} min)")
    print(f"  Qualified frames: {bs['quality_frames']['qualified']} "
          f"({bs['quality_frames']['qualified_ratio']*100:.1f}%)")
    print(f"  Unqualified frames: {bs['quality_frames']['unqualified']}")

    if anomalies:
        print(f"\n  Anomalies ({len(anomalies)}):")
        for a in anomalies:
            print(f"    [{a['severity'].upper()}] Ep {a['episode_index']}: "
                  f"{a['type']} — {a['detail']}")

    print(f"\n  SAL (smoothness):")
    print(f"    Left arm:  mean={sal['summary']['sal_left_ee']['mean']}, "
          f"std={sal['summary']['sal_left_ee']['std']}")
    print(f"    Right arm: mean={sal['summary']['sal_right_ee']['mean']}, "
          f"std={sal['summary']['sal_right_ee']['std']}")

    print(f"  Sudden changes: {sc['num_flagged_episodes']}/{bs['num_episodes']}"
          f" episodes flagged")
    print(f"  DA alignment: {da['num_misaligned_episodes']}/{bs['num_episodes']}"
          f" episodes misaligned")

    sal_combined = {
        int(k): (v["sal_left_ee"] + v["sal_right_ee"]) / 2
        for k, v in sal["per_episode"].items()
    }
    sorted_sal = sorted(sal_combined.items(), key=lambda x: x[1], reverse=True)
    print("\n  Top 5 smoothest (SAL ↑):")
    for ep, s in sorted_sal[:5]:
        print(f"    Episode {ep:3d}: SAL = {s:.1f}")
    print("  Top 5 roughest (SAL ↓):")
    for ep, s in sorted_sal[-5:]:
        print(f"    Episode {ep:3d}: SAL = {s:.1f}")


if __name__ == "__main__":
    main()
