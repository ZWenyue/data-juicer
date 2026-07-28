#!/usr/bin/env python3
"""Galaxea Open-World Dataset — Connect_Router_Cables quality analysis.

Loads the LeRobot v2.1 unified-format dataset, computes episode-level and
frame-level quality metrics (Spectral Arc Length, sudden-change detection,
action-state directional alignment), and emits a JSON report.

Usage:
    /mnt/r/VENV/dj/bin/python analyze_glx.py [--data-root PATH] [--output-dir PATH]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Unified slot layout (from process_to_unified50.py)
# ---------------------------------------------------------------------------
# Shared semantic slots (action & state identical for [0..23]):
#   left_arm:  [0..6]  (slot 1 = shoulder_roll, padded for R1 Lite 6-DoF)
#   left_hand: [7]
#   right_arm: [8..14] (slot 9 = shoulder_roll, padded)
#   right_hand:[15]
#   torso:     [16..20] (state: 4 valid + 1 pad; action: 5 valid)
#   head:      [21..23] (all padded for R1 Lite)
# Divergent:
#   action  [24..26] = base (vx,vy,wz), [27..49] = reserved
#   state   [24..32] = chassis (pos3+vel3+pad3), [33..55] = reserved

STATE_SLOT_NAMES = {
    "left_arm_j0": 0, "left_arm_j1_pad": 1, "left_arm_j2": 2,
    "left_arm_j3": 3, "left_arm_j4": 4, "left_arm_j5": 5, "left_arm_j6": 6,
    "left_gripper": 7,
    "right_arm_j0": 8, "right_arm_j1_pad": 9, "right_arm_j2": 10,
    "right_arm_j3": 11, "right_arm_j4": 12, "right_arm_j5": 13, "right_arm_j6": 14,
    "right_gripper": 15,
    "torso_0": 16, "torso_1": 17, "torso_2": 18, "torso_3": 19,
    "chassis_x": 24, "chassis_y": 25, "chassis_z": 26,
    "chassis_vx": 27, "chassis_vy": 28, "chassis_vz": 29,
}

ACTION_SLOT_NAMES = {
    "left_arm_j0": 0, "left_arm_j2": 2, "left_arm_j3": 3,
    "left_arm_j4": 4, "left_arm_j5": 5, "left_arm_j6": 6,
    "left_gripper": 7,
    "right_arm_j0": 8, "right_arm_j2": 10, "right_arm_j3": 11,
    "right_arm_j4": 12, "right_arm_j5": 13, "right_arm_j6": 14,
    "right_gripper": 15,
    "torso_0": 16, "torso_1": 17, "torso_2": 18, "torso_3": 19, "torso_4": 20,
    "base_vx": 24, "base_vy": 25, "base_wz": 26,
}

LEFT_ARM_VALID = [0, 2, 3, 4, 5, 6]
RIGHT_ARM_VALID = [8, 10, 11, 12, 13, 14]
FPS = 15
DT = 1.0 / FPS


# ── data loading ──────────────────────────────────────────────────────────

def load_episodes_meta(data_root: Path):
    """Load per-episode metadata from episodes.jsonl."""
    ep_path = data_root / "meta" / "episodes.jsonl"
    episodes = []
    with open(ep_path) as f:
        for line in f:
            episodes.append(json.loads(line))
    return episodes


def load_tasks_meta(data_root: Path):
    """Load task_index -> task text mapping from tasks.jsonl."""
    t_path = data_root / "meta" / "tasks.jsonl"
    if not t_path.exists():
        return {}
    tasks = {}
    with open(t_path) as f:
        for line in f:
            obj = json.loads(line)
            tasks[obj["task_index"]] = obj["task"]
    return tasks


def load_all_parquets(data_root: Path, max_episodes=None):
    """Load all episode parquet files into a dict {ep_idx: DataFrame}."""
    data_dir = data_root / "data"
    files = sorted(data_dir.rglob("*.parquet"))
    if max_episodes:
        files = files[:max_episodes]

    episodes = {}
    for f in files:
        df = pd.read_parquet(f)
        ep_idx = df["episode_index"].iloc[0]
        episodes[ep_idx] = df
    return episodes


def extract_array_column(df, col, dim):
    """Convert a column of numpy arrays into a (N, dim) float32 matrix."""
    return np.stack(df[col].values).astype(np.float32).reshape(-1, dim)


# ── basic statistics ──────────────────────────────────────────────────────

def basic_stats(episodes_data, episodes_meta):
    """Compute basic dataset statistics."""
    lengths = [len(df) for df in episodes_data.values()]
    durations = [l / FPS for l in lengths]
    total_frames = sum(lengths)

    meta_lengths = [e["length"] for e in episodes_meta]
    quality_labels = {}
    for e in episodes_meta:
        for t in e.get("tasks", []):
            if t in ("qualified", "unqualified"):
                key = f"ep_{e['episode_index']}_quality"
                quality_labels[e["episode_index"]] = t

    return {
        "num_episodes": len(episodes_data),
        "total_frames": total_frames,
        "total_duration_sec": sum(durations),
        "episode_length": {
            "min": int(min(lengths)),
            "max": int(max(lengths)),
            "mean": float(np.mean(lengths)),
            "median": float(np.median(lengths)),
            "std": float(np.std(lengths)),
        },
        "episode_duration_sec": {
            "min": round(min(durations), 1),
            "max": round(max(durations), 1),
            "mean": round(float(np.mean(durations)), 1),
            "median": round(float(np.median(durations)), 1),
        },
        "quality_labels": quality_labels,
        "num_qualified": sum(1 for v in quality_labels.values() if v == "qualified"),
        "num_unqualified": sum(1 for v in quality_labels.values() if v == "unqualified"),
    }


# ── action / state statistics ────────────────────────────────────────────

def action_state_stats(episodes_data):
    """Per-dimension min/max/mean/std for action and state (valid dims only)."""
    all_states, all_actions = [], []
    for df in episodes_data.values():
        s = extract_array_column(df, "observation.state", 56)
        a = extract_array_column(df, "action", 50)
        all_states.append(s)
        all_actions.append(a)

    states = np.concatenate(all_states, axis=0)
    actions = np.concatenate(all_actions, axis=0)

    pad_s = np.stack(list(episodes_data.values())[0]["observation.state_dim_is_pad"].values[:1])[0]
    pad_a = np.stack(list(episodes_data.values())[0]["action_dim_is_pad"].values[:1])[0]

    valid_s = np.where(~pad_s)[0]
    valid_a = np.where(~pad_a)[0]

    def dim_stats(data, valid_dims, slot_names):
        name_lookup = {v: k for k, v in slot_names.items()}
        results = {}
        for d in valid_dims:
            col = data[:, d]
            name = name_lookup.get(d, f"dim_{d}")
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
            "per_dim": dim_stats(states, valid_s, STATE_SLOT_NAMES),
        },
        "action": {
            "total_dims": 50,
            "valid_dims": len(valid_a),
            "valid_indices": valid_a.tolist(),
            "per_dim": dim_stats(actions, valid_a, ACTION_SLOT_NAMES),
        },
    }


# ── Spectral Arc Length (SAL) ─────────────────────────────────────────────

def spectral_arc_length(speed_signal, dt=DT, eps=1e-5):
    """Compute SAL for a 1-D speed signal.

    SAL(v) = -sum_k sqrt( (f_k - f_{k-1})^2 + (L_k - L_{k-1})^2 )
    where L_k = log(|FFT(v)_k| + eps).
    A less-negative SAL indicates smoother motion.
    """
    T = len(speed_signal)
    if T < 4:
        return 0.0

    V = np.fft.rfft(speed_signal)
    freqs = np.fft.rfftfreq(T, d=dt)
    L = np.log(np.abs(V) + eps)

    df = np.diff(freqs)
    dL = np.diff(L)
    sal = -np.sum(np.sqrt(df ** 2 + dL ** 2))
    return float(sal)


def compute_sal_per_episode(episodes_data):
    """Compute SAL for left and right arm joint velocities per episode."""
    results = {}
    for ep_idx, df in sorted(episodes_data.items()):
        states = extract_array_column(df, "observation.state", 56)
        n = states.shape[0]
        if n < 10:
            continue

        sal_left, sal_right = [], []
        for jdim in LEFT_ARM_VALID:
            vel = np.diff(states[:, jdim]) / DT
            speed = np.abs(vel)
            sal_left.append(spectral_arc_length(speed))
        for jdim in RIGHT_ARM_VALID:
            vel = np.diff(states[:, jdim]) / DT
            speed = np.abs(vel)
            sal_right.append(spectral_arc_length(speed))

        ee_left = states[:, LEFT_ARM_VALID]
        ee_right = states[:, RIGHT_ARM_VALID]
        vel_left = np.linalg.norm(np.diff(ee_left, axis=0) / DT, axis=1)
        vel_right = np.linalg.norm(np.diff(ee_right, axis=0) / DT, axis=1)

        results[ep_idx] = {
            "sal_left_mean": round(float(np.mean(sal_left)), 2),
            "sal_right_mean": round(float(np.mean(sal_right)), 2),
            "sal_left_ee": round(spectral_arc_length(vel_left), 2),
            "sal_right_ee": round(spectral_arc_length(vel_right), 2),
            "num_frames": n,
        }

    sal_left_all = [v["sal_left_ee"] for v in results.values()]
    sal_right_all = [v["sal_right_ee"] for v in results.values()]

    return {
        "per_episode": {str(k): v for k, v in results.items()},
        "summary": {
            "sal_left_ee": {
                "min": round(min(sal_left_all), 2),
                "max": round(max(sal_left_all), 2),
                "mean": round(float(np.mean(sal_left_all)), 2),
                "median": round(float(np.median(sal_left_all)), 2),
                "std": round(float(np.std(sal_left_all)), 2),
            },
            "sal_right_ee": {
                "min": round(min(sal_right_all), 2),
                "max": round(max(sal_right_all), 2),
                "mean": round(float(np.mean(sal_right_all)), 2),
                "median": round(float(np.median(sal_right_all)), 2),
                "std": round(float(np.std(sal_right_all)), 2),
            },
        },
    }


# ── sudden-change detection ──────────────────────────────────────────────

def detect_sudden_changes(episodes_data, c_threshold=8.0, floor_frac=0.02):
    """Detect episodes with sudden jumps in joint positions (Stage 1 logic).

    Flags episodes where acceleration exceeds c_threshold * sigma.
    """
    results = {}
    arm_dims = LEFT_ARM_VALID + RIGHT_ARM_VALID

    for ep_idx, df in sorted(episodes_data.items()):
        states = extract_array_column(df, "observation.state", 56)
        n = states.shape[0]
        if n < 15:
            continue

        flagged_dims = []
        total_flags = 0

        for dim in arm_dims:
            signal = states[:, dim]
            q01, q99 = np.percentile(signal, [1, 99])
            joint_range = q99 - q01

            accel = np.diff(signal, n=2)
            sigma = max(np.std(accel), floor_frac * joint_range, 0.01)
            flags = np.abs(accel) > c_threshold * sigma

            n_flags = int(flags.sum())
            if n_flags > 0:
                flagged_dims.append({"dim": dim, "n_flags": n_flags})
                total_flags += n_flags

        results[ep_idx] = {
            "total_flags": total_flags,
            "flagged_dims": flagged_dims,
            "flagged": total_flags > 0,
        }

    flagged_eps = [k for k, v in results.items() if v["flagged"]]
    return {
        "per_episode": {str(k): v for k, v in results.items()},
        "num_flagged_episodes": len(flagged_eps),
        "flagged_episode_indices": flagged_eps,
    }


# ── state-action directional alignment ───────────────────────────────────

def directional_alignment(episodes_data, da_thresh=0.65, eps_frac=0.01):
    """Check state-action trend consistency (Stage 2 logic).

    For each shared joint dimension, compute directional alignment between
    state velocity and action velocity.
    """
    shared_dims = LEFT_ARM_VALID + RIGHT_ARM_VALID

    results = {}
    for ep_idx, df in sorted(episodes_data.items()):
        states = extract_array_column(df, "observation.state", 56)
        actions = extract_array_column(df, "action", 50)
        n = states.shape[0]
        if n < 20:
            continue

        dim_results = []
        for dim in shared_dims:
            s = states[:, dim]
            a = actions[:, dim]
            q01, q99 = np.percentile(s, [1, 99])
            joint_range = q99 - q01
            eps = max(eps_frac * joint_range, 1e-6)

            ds = np.diff(s)
            da = np.diff(a)

            active = (np.abs(ds) > eps) | (np.abs(da) > eps)
            if active.sum() < 10:
                continue

            ds_a = ds[active]
            da_a = da[active]
            agree = np.sign(ds_a) == np.sign(da_a)
            da_score = float(agree.mean())

            dim_results.append({
                "dim": dim,
                "da_score": round(da_score, 3),
                "n_active": int(active.sum()),
                "aligned": da_score >= da_thresh,
            })

        misaligned = [d for d in dim_results if not d["aligned"]]
        results[ep_idx] = {
            "misaligned_dims": len(misaligned),
            "total_checked": len(dim_results),
            "misaligned_details": misaligned,
            "episode_aligned": len(misaligned) == 0,
        }

    misaligned_eps = [k for k, v in results.items() if not v["episode_aligned"]]
    return {
        "per_episode": {str(k): v for k, v in results.items()},
        "num_misaligned_episodes": len(misaligned_eps),
        "misaligned_episode_indices": misaligned_eps,
    }


# ── anomaly detection ─────────────────────────────────────────────────────

def detect_anomalies(episodes_meta, tasks_meta):
    """Detect episode-level anomalies from metadata."""
    anomalies = []

    for e in episodes_meta:
        ep_idx = e["episode_index"]
        tasks = e.get("tasks", [])
        length = e.get("length", 0)

        subtasks = [t for t in tasks
                    if t not in ("qualified", "unqualified", "null",
                                 "connect router cables")]

        is_off_task = False
        off_task_keywords = ["冰箱", "refrigerator", "mushroom", "蘑菇",
                             "西葫芦", "zucchini", "冰红茶", "iced tea",
                             "钙奶", "calcium milk"]
        for st in subtasks:
            if any(kw in st.lower() for kw in off_task_keywords):
                is_off_task = True
                break

        if is_off_task:
            anomalies.append({
                "episode_index": ep_idx,
                "type": "off_task",
                "detail": f"Episode contains non-router-cable subtasks "
                          f"({len(subtasks)} subtasks, {length} frames)",
                "severity": "high",
            })

        if length > 5000:
            anomalies.append({
                "episode_index": ep_idx,
                "type": "abnormal_length",
                "detail": f"Episode has {length} frames "
                          f"({length/FPS:.0f}s), far above typical",
                "severity": "medium",
            })

    return anomalies


# ── episode length distribution ───────────────────────────────────────────

def length_distribution(episodes_meta):
    """Compute episode length histogram buckets."""
    lengths = [e["length"] for e in episodes_meta]
    bins = [0, 500, 1000, 1500, 2000, 3000, 5000, 10000]
    hist, _ = np.histogram(lengths, bins=bins)
    return {
        "bins": bins,
        "counts": hist.tolist(),
        "bin_labels": [f"{bins[i]}-{bins[i+1]}" for i in range(len(hist))],
    }


# ── gripper analysis ──────────────────────────────────────────────────────

def gripper_analysis(episodes_data):
    """Analyze gripper open/close patterns."""
    left_grip_dim = 7
    right_grip_dim = 15
    results = {}

    for ep_idx, df in sorted(episodes_data.items()):
        states = extract_array_column(df, "observation.state", 56)
        lg = states[:, left_grip_dim]
        rg = states[:, right_grip_dim]

        lg_changes = int(np.sum(np.abs(np.diff(lg)) > 1.0))
        rg_changes = int(np.sum(np.abs(np.diff(rg)) > 1.0))

        results[ep_idx] = {
            "left_gripper": {
                "min": round(float(lg.min()), 2),
                "max": round(float(lg.max()), 2),
                "mean": round(float(lg.mean()), 2),
                "n_state_changes": lg_changes,
            },
            "right_gripper": {
                "min": round(float(rg.min()), 2),
                "max": round(float(rg.max()), 2),
                "mean": round(float(rg.mean()), 2),
                "n_state_changes": rg_changes,
            },
        }

    return results


# ── main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyze Galaxea dataset")
    parser.add_argument(
        "--data-root",
        default="/mnt/r/DATA/Galaxea-Open-World-Dataset/lerobot_door"
                "/Connect_Router_Cables_20250625_002",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).parent),
    )
    parser.add_argument("--max-episodes", type=int, default=None)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)

    print(f"=== Galaxea Connect_Router_Cables Analysis ===")
    print(f"Data root: {data_root}")
    print()

    # Load metadata
    print("[1/7] Loading metadata...")
    episodes_meta = load_episodes_meta(data_root)
    tasks_meta = load_tasks_meta(data_root)
    print(f"  {len(episodes_meta)} episodes, {len(tasks_meta)} task labels")

    # Load parquet data
    print("[2/7] Loading parquet data...")
    episodes_data = load_all_parquets(data_root, max_episodes=args.max_episodes)
    print(f"  Loaded {len(episodes_data)} episodes, "
          f"{sum(len(df) for df in episodes_data.values())} total frames")

    # Basic stats
    print("[3/7] Computing basic statistics...")
    bs = basic_stats(episodes_data, episodes_meta)
    print(f"  Episodes: {bs['num_episodes']}, Frames: {bs['total_frames']}")
    print(f"  Duration: {bs['total_duration_sec']:.0f}s "
          f"({bs['total_duration_sec']/60:.1f} min)")
    print(f"  Episode length: {bs['episode_length']['min']}-"
          f"{bs['episode_length']['max']} frames "
          f"(median {bs['episode_length']['median']:.0f})")
    print(f"  Quality: {bs['num_qualified']} qualified, "
          f"{bs['num_unqualified']} unqualified")

    # Action/state stats
    print("[4/7] Computing action/state statistics...")
    as_stats = action_state_stats(episodes_data)
    print(f"  State: {as_stats['state']['valid_dims']}/56 valid dims")
    print(f"  Action: {as_stats['action']['valid_dims']}/50 valid dims")

    # SAL
    print("[5/7] Computing Spectral Arc Length (smoothness)...")
    sal = compute_sal_per_episode(episodes_data)
    print(f"  Left arm SAL:  mean={sal['summary']['sal_left_ee']['mean']}, "
          f"std={sal['summary']['sal_left_ee']['std']}")
    print(f"  Right arm SAL: mean={sal['summary']['sal_right_ee']['mean']}, "
          f"std={sal['summary']['sal_right_ee']['std']}")

    # Sudden changes
    print("[6/7] Detecting sudden changes...")
    sc = detect_sudden_changes(episodes_data)
    print(f"  Flagged episodes: {sc['num_flagged_episodes']}/{len(episodes_data)}")

    # Directional alignment
    print("[7/7] Checking state-action alignment...")
    da = directional_alignment(episodes_data)
    print(f"  Misaligned episodes: {da['num_misaligned_episodes']}/{len(episodes_data)}")

    # Additional analyses
    anomalies = detect_anomalies(episodes_meta, tasks_meta)
    len_dist = length_distribution(episodes_meta)
    grip = gripper_analysis(episodes_data)

    # Assemble report
    report = {
        "dataset": "Galaxea-Open-World / Connect_Router_Cables_20250625_002",
        "format": "LeRobot v2.1 unified (lerobot_door)",
        "basic_stats": bs,
        "action_state_stats": as_stats,
        "smoothness_sal": sal,
        "sudden_changes": sc,
        "directional_alignment": da,
        "anomalies": anomalies,
        "length_distribution": len_dist,
        "gripper_analysis": {str(k): v for k, v in grip.items()},
    }

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    out_path = output_dir / "analysis_results.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    print(f"\nReport saved to {out_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("QUALITY SUMMARY")
    print("=" * 60)

    issues = []
    if anomalies:
        for a in anomalies:
            issues.append(f"[{a['severity'].upper()}] Episode {a['episode_index']}: "
                          f"{a['type']} — {a['detail']}")
    if sc["num_flagged_episodes"] > 0:
        issues.append(f"[MEDIUM] {sc['num_flagged_episodes']} episodes with "
                      f"sudden joint changes")
    if da["num_misaligned_episodes"] > 0:
        issues.append(f"[MEDIUM] {da['num_misaligned_episodes']} episodes with "
                      f"state-action misalignment")

    if issues:
        print(f"\n{len(issues)} issue(s) found:")
        for iss in issues:
            print(f"  {iss}")
    else:
        print("\nNo major issues detected.")

    kept = bs["num_episodes"] - len([a for a in anomalies if a["severity"] == "high"])
    kept -= sc["num_flagged_episodes"]
    print(f"\nEstimated clean episodes: ~{max(kept, 0)}/{bs['num_episodes']}")

    # SAL ranking (top 5 smoothest and roughest)
    sal_eps = sal["per_episode"]
    sal_combined = {int(k): (v["sal_left_ee"] + v["sal_right_ee"]) / 2
                    for k, v in sal_eps.items()}
    sorted_sal = sorted(sal_combined.items(), key=lambda x: x[1], reverse=True)

    print("\nTop 5 smoothest episodes (highest SAL, closer to 0):")
    for ep, s in sorted_sal[:5]:
        print(f"  Episode {ep:3d}: SAL = {s:.1f}")

    print("\nTop 5 roughest episodes (lowest SAL):")
    for ep, s in sorted_sal[-5:]:
        print(f"  Episode {ep:3d}: SAL = {s:.1f}")


if __name__ == "__main__":
    main()
