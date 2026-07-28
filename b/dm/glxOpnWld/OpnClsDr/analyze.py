#!/usr/bin/env python3
"""
数据分析脚本: Open_And_Close_The_Door_20250802_012
Galaxea Open-World Dataset — LeRobot v2.1 格式
"""

import json
import glob
import os
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch
import matplotlib.ticker as mticker

# ── Paths ──────────────────────────────────────────────────────────────
DATASET = '/mnt/r/DATA/Galaxea-Open-World-Dataset/lerobot/Open_And_Close_The_Door_20250802_012'
OUT_DIR = Path(__file__).parent
FPS = 15

# ── Dataviz palette (light mode) ──────────────────────────────────────
SURFACE   = '#fcfcfb'
INK_PRI   = '#0b0b0b'
INK_SEC   = '#52514e'
INK_MUTED = '#898781'
GRIDLINE  = '#e1e0d9'
BASELINE  = '#c3c2b7'

CAT = ['#2a78d6', '#1baf7a', '#eda100', '#008300',
       '#4a3aa7', '#e34948', '#e87ba4', '#eb6834']
SEQ_BLUE = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5',
            '#2a78d6', '#256abf', '#1c5cab', '#184f95']
STATUS_GOOD     = '#0ca30c'
STATUS_CRITICAL = '#d03b3b'


def setup_mpl():
    plt.rcParams.update({
        'figure.facecolor': SURFACE,
        'axes.facecolor':   SURFACE,
        'axes.edgecolor':   BASELINE,
        'axes.labelcolor':  INK_SEC,
        'axes.grid':        True,
        'grid.color':       GRIDLINE,
        'grid.linewidth':   0.5,
        'text.color':       INK_PRI,
        'xtick.color':      INK_MUTED,
        'ytick.color':      INK_MUTED,
        'xtick.labelsize':  9,
        'ytick.labelsize':  9,
        'axes.titlesize':   13,
        'axes.labelsize':   10,
        'legend.fontsize':  9,
        'legend.frameon':   False,
        'font.family':      'sans-serif',
        'font.sans-serif':  ['DejaVu Sans', 'Liberation Sans', 'sans-serif'],
        'figure.dpi':       150,
        'savefig.dpi':      150,
        'savefig.bbox':     'tight',
        'savefig.pad_inches': 0.3,
    })


# ── Load data ─────────────────────────────────────────────────────────

def _en_label(task_str):
    """Extract English part from bilingual 'CN@EN' label, or return as-is."""
    if '@' in task_str:
        return task_str.split('@', 1)[1]
    return task_str


def _cn_label(task_str):
    """Extract Chinese part from bilingual 'CN@EN' label, or return as-is."""
    if '@' in task_str:
        return task_str.split('@', 1)[0]
    return task_str


def load_metadata():
    with open(f'{DATASET}/meta/info.json') as f:
        info = json.load(f)

    tasks = {}
    with open(f'{DATASET}/meta/tasks.jsonl') as f:
        for line in f:
            t = json.loads(line)
            tasks[t['task_index']] = t['task']

    episodes = []
    with open(f'{DATASET}/meta/episodes.jsonl') as f:
        for line in f:
            episodes.append(json.loads(line))

    return info, tasks, episodes


def load_all_parquet():
    files = sorted(glob.glob(f'{DATASET}/data/chunk-000/episode_*.parquet'))
    dfs = [pd.read_parquet(f) for f in files]
    return pd.concat(dfs, ignore_index=True)


# ── Chart 1: Episode length distribution ──────────────────────────────

def plot_episode_length_dist(episodes):
    lengths = [e['length'] for e in episodes]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    n, bins, patches = ax.hist(lengths, bins=25, color=CAT[0], edgecolor=SURFACE,
                                linewidth=1.5, rwidth=0.92)
    for p in patches:
        p.set_linewidth(0.5)
        p.set_edgecolor(SURFACE)

    ax.axvline(np.median(lengths), color=CAT[5], linewidth=1.5, linestyle='--',
               label=f'Median = {np.median(lengths):.0f}')
    ax.axvline(np.mean(lengths), color=CAT[2], linewidth=1.5, linestyle=':',
               label=f'Mean = {np.mean(lengths):.0f}')

    ax.set_xlabel('Episode Length (frames)')
    ax.set_ylabel('Count')
    ax.set_title('Episode Length Distribution')
    ax.legend(loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.savefig(OUT_DIR / 'episode_length_dist.png')
    plt.close(fig)
    print(f'  ✓ episode_length_dist.png')
    return lengths


# ── Chart 2: Task frequency ──────────────────────────────────────────

def _cn_to_en(cn_key, tasks):
    """Given a Chinese semantic key, find its best English translation."""
    for task in tasks.values():
        if _cn_label(task) == cn_key and '@' in task:
            return _en_label(task)
    return cn_key


def plot_task_frequency(tasks, df_all):
    cn_groups = defaultdict(list)
    for idx, task in tasks.items():
        cn = _cn_label(task)
        cn_groups[cn].append(idx)

    group_frames = {}
    task_frame_counts = df_all['task_index'].value_counts()
    for cn, indices in cn_groups.items():
        total = sum(task_frame_counts.get(i, 0) for i in indices)
        en = _cn_to_en(cn, tasks)
        group_frames[en] = total

    sorted_groups = sorted(group_frames.items(), key=lambda x: -x[1])[:20]
    labels = [g[0][:50] for g in sorted_groups]
    values = [g[1] for g in sorted_groups]

    fig, ax = plt.subplots(figsize=(9, 6))
    y_pos = range(len(labels))
    bars = ax.barh(y_pos, values, color=CAT[0], height=0.7, edgecolor=SURFACE,
                   linewidth=0.5)

    for bar, val in zip(bars, values):
        ax.text(val + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                f'{val:,}', va='center', ha='left', fontsize=8, color=INK_SEC)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Frame Count')
    ax.set_title('Task Frequency (Semantic Groups)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', visible=False)

    fig.savefig(OUT_DIR / 'task_frequency.png')
    plt.close(fig)
    print(f'  ✓ task_frequency.png')


# ── Chart 3: Task duration ────────────────────────────────────────────

def plot_task_duration(tasks, df_all):
    task_durations = defaultdict(list)
    for ep_idx in df_all['episode_index'].unique():
        ep = df_all[df_all['episode_index'] == ep_idx].sort_values('frame_index')
        prev_t = None
        seg_start = 0
        for _, row in ep.iterrows():
            if row['task_index'] != prev_t:
                if prev_t is not None:
                    cn = _cn_label(tasks.get(prev_t, '?'))
                    en = _cn_to_en(cn, tasks)
                    task_durations[en].append((row['frame_index'] - seg_start) / FPS)
                prev_t = row['task_index']
                seg_start = row['frame_index']
        if prev_t is not None:
            cn = _cn_label(tasks.get(prev_t, '?'))
            en = _cn_to_en(cn, tasks)
            task_durations[en].append((int(ep['frame_index'].max()) - seg_start + 1) / FPS)

    filtered = {k: v for k, v in task_durations.items()
                if k not in ('qualified', 'unqualified', 'null', 'open and close the door')
                and len(v) >= 3}

    sorted_tasks = sorted(filtered.items(), key=lambda x: -np.mean(x[1]))[:15]
    labels = [t[0][:50] for t in sorted_tasks]
    means = [np.mean(t[1]) for t in sorted_tasks]
    stds = [np.std(t[1]) for t in sorted_tasks]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    y_pos = range(len(labels))
    bars = ax.barh(y_pos, means, xerr=stds, color=CAT[1], height=0.7,
                   edgecolor=SURFACE, linewidth=0.5,
                   error_kw={'elinewidth': 1, 'capsize': 3, 'color': INK_MUTED})

    for bar, val in zip(bars, means):
        ax.text(val + max(means) * 0.02, bar.get_y() + bar.get_height() / 2,
                f'{val:.1f}s', va='center', ha='left', fontsize=8, color=INK_SEC)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Duration (seconds)')
    ax.set_title('Average Task Duration (Top-15, ≥3 segments)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', visible=False)

    fig.savefig(OUT_DIR / 'task_duration.png')
    plt.close(fig)
    print(f'  ✓ task_duration.png')


# ── Chart 4: Joint ranges ────────────────────────────────────────────

def plot_joint_ranges(df_all):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, side, color in zip(axes, ['left', 'right'], [CAT[0], CAT[5]]):
        col = f'observation.state.{side}_arm'
        arr = np.array(df_all[col].tolist())

        joint_labels = [f'J{i}' for i in range(arr.shape[1])]
        mins = arr.min(axis=0)
        maxs = arr.max(axis=0)
        means = arr.mean(axis=0)
        stds = arr.std(axis=0)

        n_joints = arr.shape[1]
        x = np.arange(n_joints)
        ax.bar(x - 0.15, maxs - mins, bottom=mins, width=0.3,
               color=color, alpha=0.25, edgecolor=color, linewidth=0.8,
               label='Range (min–max)')
        ax.errorbar(x + 0.15, means, yerr=stds, fmt='o', color=color,
                    markersize=5, capsize=4, linewidth=1.2,
                    label='Mean ± Std')

        ax.set_xticks(x)
        ax.set_xticklabels(joint_labels)
        ax.set_title(f'{side.capitalize()} Arm Joint Space')
        ax.set_ylabel('Radians' if ax == axes[0] else '')
        ax.legend(fontsize=8, loc='upper right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle('Arm Joint Position Ranges', fontsize=14, y=1.02)
    fig.savefig(OUT_DIR / 'joint_ranges.png')
    plt.close(fig)
    print(f'  ✓ joint_ranges.png')


# ── Chart 5: End-effector workspace ───────────────────────────────────

def plot_ee_workspace(df_all):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    left_ee = np.array(df_all['observation.state.left_ee_pose'].tolist())
    right_ee = np.array(df_all['observation.state.right_ee_pose'].tolist())

    step = max(1, len(left_ee) // 5000)

    projections = [('X', 'Y', 0, 1), ('X', 'Z', 0, 2), ('Y', 'Z', 1, 2)]
    for ax, (xlabel, ylabel, xi, yi) in zip(axes, projections):
        ax.scatter(left_ee[::step, xi], left_ee[::step, yi],
                   s=2, alpha=0.3, color=CAT[0], label='Left EE', rasterized=True)
        ax.scatter(right_ee[::step, xi], right_ee[::step, yi],
                   s=2, alpha=0.3, color=CAT[5], label='Right EE', rasterized=True)

        ax.set_xlabel(f'{xlabel} (m)')
        ax.set_ylabel(f'{ylabel} (m)')
        ax.set_aspect('equal')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if ax == axes[0]:
            ax.legend(fontsize=8, markerscale=4)

    fig.suptitle('End-Effector Workspace (Position Projections)', fontsize=14, y=1.02)
    fig.savefig(OUT_DIR / 'ee_workspace.png')
    plt.close(fig)
    print(f'  ✓ ee_workspace.png')


# ── Chart 6: Gripper distribution ─────────────────────────────────────

def plot_gripper_dist(df_all):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, side, color in zip(axes, ['left', 'right'], [CAT[0], CAT[5]]):
        obs_col = f'observation.state.{side}_gripper'
        act_col = f'action.{side}_gripper'

        obs_vals = df_all[obs_col].values.astype(float)
        act_vals = df_all[act_col].values.astype(float)

        ax.hist(obs_vals, bins=50, color=color, alpha=0.6,
                edgecolor=SURFACE, linewidth=0.5, label='Observation', density=True)
        ax.hist(act_vals, bins=50, color=CAT[2], alpha=0.5,
                edgecolor=SURFACE, linewidth=0.5, label='Action', density=True)

        ax.set_xlabel('Gripper Value')
        ax.set_ylabel('Density')
        ax.set_title(f'{side.capitalize()} Gripper')
        ax.legend(fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle('Gripper State & Action Distribution', fontsize=14, y=1.02)
    fig.savefig(OUT_DIR / 'gripper_dist.png')
    plt.close(fig)
    print(f'  ✓ gripper_dist.png')


# ── Chart 7: Quality distribution across episodes ────────────────────

def plot_quality_dist(episodes, df_all):
    fig, ax = plt.subplots(figsize=(12, 5))

    ep_indices = sorted(df_all['episode_index'].unique())
    for i, ep_idx in enumerate(ep_indices):
        ep = df_all[df_all['episode_index'] == ep_idx].sort_values('frame_index')
        qual = ep['quality_index'].values
        frames = ep['frame_index'].values

        segments_qual = []
        seg_start = 0
        prev_q = qual[0]
        for j in range(1, len(qual)):
            if qual[j] != prev_q:
                segments_qual.append((frames[seg_start], frames[j-1], prev_q))
                seg_start = j
                prev_q = qual[j]
        segments_qual.append((frames[seg_start], frames[-1], prev_q))

        for start, end, q in segments_qual:
            color = STATUS_GOOD if q == 1 else STATUS_CRITICAL
            ax.barh(i, (end - start) / FPS, left=start / FPS, height=0.8,
                    color=color, edgecolor=SURFACE, linewidth=0.3)

    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Episode Index')
    ax.set_title('Quality Label Distribution Across Episodes')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    legend_elements = [Patch(facecolor=STATUS_GOOD, label='Qualified (1)'),
                       Patch(facecolor=STATUS_CRITICAL, label='Unqualified (5)')]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

    fig.savefig(OUT_DIR / 'quality_dist.png')
    plt.close(fig)
    print(f'  ✓ quality_dist.png')


# ── Chart 8: Task flow for sample episodes ────────────────────────────

def plot_task_flow(tasks, df_all):
    available = sorted(df_all['episode_index'].unique())
    sample_eps = [ep for ep in [0, 5, 15, 30, 50] if ep in available]
    if len(sample_eps) < 3:
        sample_eps = available[:5]

    fig, axes = plt.subplots(len(sample_eps), 1, figsize=(14, 8), sharex=False)
    if len(sample_eps) == 1:
        axes = [axes]

    all_en_tasks = set()
    for ep_idx in sample_eps:
        ep = df_all[df_all['episode_index'] == ep_idx]
        for tidx in ep['task_index'].unique():
            t = tasks.get(tidx, '?')
            en = _en_label(t)
            all_en_tasks.add(en)

    task_color_map = {}
    operational_tasks = [t for t in all_en_tasks
                         if t not in ('qualified', 'unqualified', 'null',
                                      'open and close the door')]
    for i, t in enumerate(sorted(operational_tasks)):
        task_color_map[t] = CAT[i % len(CAT)]
    task_color_map['qualified'] = STATUS_GOOD
    task_color_map['unqualified'] = STATUS_CRITICAL
    task_color_map['null'] = INK_MUTED
    task_color_map['open and close the door'] = BASELINE

    for ax, ep_idx in zip(axes, sample_eps):
        ep = df_all[df_all['episode_index'] == ep_idx].sort_values('frame_index')

        prev_t = None
        seg_start = 0
        segments = []
        for _, row in ep.iterrows():
            if row['task_index'] != prev_t:
                if prev_t is not None:
                    en = _en_label(tasks.get(prev_t, '?'))
                    segments.append((seg_start, row['frame_index'], en))
                prev_t = row['task_index']
                seg_start = row['frame_index']
        if prev_t is not None:
            en = _en_label(tasks.get(prev_t, '?'))
            segments.append((seg_start, int(ep['frame_index'].max()) + 1, en))

        for start, end, en in segments:
            color = task_color_map.get(en, INK_MUTED)
            ax.barh(0, (end - start) / FPS, left=start / FPS, height=0.6,
                    color=color, edgecolor=SURFACE, linewidth=0.3)
            if (end - start) / FPS > 3:
                ax.text((start + end) / 2 / FPS, 0, en[:20],
                        ha='center', va='center', fontsize=5.5, color=INK_PRI,
                        rotation=45)

        ax.set_yticks([])
        ax.set_ylabel(f'Ep.{ep_idx}', fontsize=9, rotation=0, labelpad=30)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

    axes[-1].set_xlabel('Time (seconds)')
    fig.suptitle('Task Flow Timeline (Sample Episodes)', fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'task_flow.png')
    plt.close(fig)
    print(f'  ✓ task_flow.png')


# ── Chart 9: Chassis & IMU overview ──────────────────────────────────

def plot_chassis_imu(df_all):
    available = sorted(df_all['episode_index'].unique())
    ep_idx = available[len(available) // 2]
    ep = df_all[df_all['episode_index'] == ep_idx].sort_values('frame_index')
    t = ep['frame_index'].values / FPS

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    chassis = np.array(ep['observation.state.chassis'].tolist())
    ax = axes[0, 0]
    for i, lbl in enumerate(['x', 'y', 'yaw']):
        ax.plot(t, chassis[:, i], color=CAT[i], linewidth=0.8, label=lbl)
    ax.set_title(f'Chassis Position (Ep.{ep_idx})')
    ax.set_ylabel('Value')
    ax.legend(fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    chassis_vel = np.array(ep['observation.state.chassis.velocities'].tolist())
    ax = axes[0, 1]
    for i, lbl in enumerate(['vx', 'vy', 'vyaw']):
        ax.plot(t, chassis_vel[:, i], color=CAT[i], linewidth=0.8, label=lbl)
    ax.set_title(f'Chassis Velocity (Ep.{ep_idx})')
    ax.set_ylabel('Value')
    ax.legend(fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    imu = np.array(ep['observation.state.chassis.imu'].tolist())
    ax = axes[1, 0]
    for i, lbl in enumerate(['quat_w', 'quat_x', 'quat_y', 'quat_z']):
        ax.plot(t, imu[:, i], color=CAT[i], linewidth=0.8, label=lbl)
    ax.set_title(f'IMU Orientation (Ep.{ep_idx})')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Quaternion')
    ax.legend(fontsize=7, ncol=2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax = axes[1, 1]
    for i, lbl in enumerate(['ax', 'ay', 'az']):
        ax.plot(t, imu[:, 4 + i], color=CAT[i], linewidth=0.8, label=lbl)
    for i, lbl in enumerate(['gx', 'gy', 'gz']):
        ax.plot(t, imu[:, 7 + i], color=CAT[3 + i], linewidth=0.8,
                label=lbl, linestyle='--')
    ax.set_title(f'IMU Accel & Gyro (Ep.{ep_idx})')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Value')
    ax.legend(fontsize=7, ncol=2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.suptitle('Chassis & IMU Signals', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'chassis_imu.png')
    plt.close(fig)
    print(f'  ✓ chassis_imu.png')


# ── Collect statistics → JSON ─────────────────────────────────────────

def collect_stats(info, tasks, episodes, df_all):
    lengths = [e['length'] for e in episodes]

    left_arm = np.array(df_all['observation.state.left_arm'].tolist())
    right_arm = np.array(df_all['observation.state.right_arm'].tolist())
    left_ee = np.array(df_all['observation.state.left_ee_pose'].tolist())
    right_ee = np.array(df_all['observation.state.right_ee_pose'].tolist())

    left_grip_obs = df_all['observation.state.left_gripper'].values.astype(float)
    right_grip_obs = df_all['observation.state.right_gripper'].values.astype(float)
    left_grip_act = df_all['action.left_gripper'].values.astype(float)
    right_grip_act = df_all['action.right_gripper'].values.astype(float)

    task_counts = df_all['task_index'].value_counts().to_dict()
    qual_counts = df_all['quality_index'].value_counts().to_dict()

    def arm_stats(arr, name):
        return {
            f'{name}_min': arr.min(axis=0).tolist(),
            f'{name}_max': arr.max(axis=0).tolist(),
            f'{name}_mean': arr.mean(axis=0).tolist(),
            f'{name}_std': arr.std(axis=0).tolist(),
        }

    def ee_stats(arr, name):
        pos = arr[:, :3]
        return {
            f'{name}_pos_min': pos.min(axis=0).tolist(),
            f'{name}_pos_max': pos.max(axis=0).tolist(),
            f'{name}_pos_mean': pos.mean(axis=0).tolist(),
            f'{name}_pos_std': pos.std(axis=0).tolist(),
        }

    stats = {
        'dataset': info.get('repo_id', 'Open_And_Close_The_Door_20250802_012'),
        'robot_type': info.get('robot_type', 'unknown'),
        'fps': info.get('fps', FPS),
        'total_episodes': len(episodes),
        'total_frames': len(df_all),
        'total_duration_s': round(len(df_all) / FPS, 1),
        'episode_length': {
            'min': int(min(lengths)),
            'max': int(max(lengths)),
            'mean': round(float(np.mean(lengths)), 1),
            'median': round(float(np.median(lengths)), 1),
            'std': round(float(np.std(lengths)), 1),
        },
        'tasks': {str(k): _en_label(v) for k, v in tasks.items()},
        'task_frame_counts': {str(k): int(v) for k, v in task_counts.items()},
        'quality_counts': {str(k): int(v) for k, v in qual_counts.items()},
        'joint_space': {
            **arm_stats(left_arm, 'left_arm'),
            **arm_stats(right_arm, 'right_arm'),
        },
        'ee_workspace': {
            **ee_stats(left_ee, 'left_ee'),
            **ee_stats(right_ee, 'right_ee'),
        },
        'gripper': {
            'left_obs_min': round(float(left_grip_obs.min()), 4),
            'left_obs_max': round(float(left_grip_obs.max()), 4),
            'left_obs_mean': round(float(left_grip_obs.mean()), 4),
            'right_obs_min': round(float(right_grip_obs.min()), 4),
            'right_obs_max': round(float(right_grip_obs.max()), 4),
            'right_obs_mean': round(float(right_grip_obs.mean()), 4),
            'left_act_min': round(float(left_grip_act.min()), 4),
            'left_act_max': round(float(left_grip_act.max()), 4),
            'right_act_min': round(float(right_grip_act.min()), 4),
            'right_act_max': round(float(right_grip_act.max()), 4),
        },
    }

    with open(OUT_DIR / 'analysis_results.json', 'w') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f'  ✓ analysis_results.json')
    return stats


# ── Main ──────────────────────────────────────────────────────────────

def main():
    setup_mpl()
    print('Loading metadata...')
    info, tasks, episodes = load_metadata()

    print('Loading all parquet files...')
    df_all = load_all_parquet()
    print(f'  Loaded {len(df_all)} frames from {len(episodes)} episodes')

    print('\nGenerating charts:')
    plot_episode_length_dist(episodes)
    plot_task_frequency(tasks, df_all)
    plot_task_duration(tasks, df_all)
    plot_joint_ranges(df_all)
    plot_ee_workspace(df_all)
    plot_gripper_dist(df_all)
    plot_quality_dist(episodes, df_all)
    plot_task_flow(tasks, df_all)
    plot_chassis_imu(df_all)

    print('\nCollecting statistics:')
    stats = collect_stats(info, tasks, episodes, df_all)

    print(f'\nAll outputs saved to {OUT_DIR}/')


if __name__ == '__main__':
    main()
