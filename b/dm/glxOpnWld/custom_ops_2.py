"""Custom Data-Juicer operators for Galaxea trajectory quality analysis.

Registers Filter/Mapper operators with DJ's OPERATORS registry for:
- Episode-level off-task detection
- Spectral Arc Length (SAL) smoothness filtering
- State-action directional alignment (DA)
- Sudden change (acceleration spike) detection
- Quality label (qualified/unqualified frame ratio) filtering
- Per-episode trajectory statistics computation

Usage:
    # As DJ custom operators (YAML config):
    dj-analyze --config dj_analyze_2.yaml --custom-operator-paths custom_ops_2.py

    # As Python imports (triggers registration):
    import custom_ops_2
"""
import sys

import numpy as np
import pandas as pd

from data_juicer.ops.base_op import OPERATORS, Filter, Mapper
from data_juicer.utils.constant import Fields

# ── shared constants ────────────────────────────────────────────────────

LEFT_ARM_VALID = [0, 2, 3, 4, 5, 6]
RIGHT_ARM_VALID = [8, 10, 11, 12, 13, 14]
FPS = 15
DT = 1.0 / FPS

STATE_DIM = 56
ACTION_DIM = 50


def _load_episode_arrays(parquet_path):
    """Read one episode parquet and return (states, actions) as numpy arrays."""
    df = pd.read_parquet(parquet_path)
    states = np.stack(df["observation.state"].values).astype(np.float32)
    actions = np.stack(df["action"].values).astype(np.float32)
    return states, actions, df


def _spectral_arc_length(speed_signal, dt=DT, eps=1e-5):
    T = len(speed_signal)
    if T < 4:
        return 0.0
    V = np.fft.rfft(speed_signal)
    freqs = np.fft.rfftfreq(T, d=dt)
    L = np.log(np.abs(V) + eps)
    df = np.diff(freqs)
    dL = np.diff(L)
    return float(-np.sum(np.sqrt(df ** 2 + dL ** 2)))


# ═══════════════════════════════════════════════════════════════════════
# Filter 1: Episode Off-Task Filter
# ═══════════════════════════════════════════════════════════════════════

@OPERATORS.register_module("episode_off_task_filter")
class EpisodeOffTaskFilter(Filter):
    """Filter out episodes whose subtask annotations contain off-task keywords."""

    def __init__(self,
                 off_task_keywords=None,
                 task_field="tasks",
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.off_task_keywords = off_task_keywords or [
            "冰箱", "refrigerator", "mushroom", "蘑菇",
            "西葫芦", "zucchini", "冰红茶", "iced tea",
            "钙奶", "calcium milk",
        ]
        self.task_field = task_field

    def compute_stats_single(self, sample, context=False):
        tasks = sample.get(self.task_field, [])
        is_off = False
        for t in (tasks or []):
            t_lower = t.lower()
            if any(kw in t_lower for kw in self.off_task_keywords):
                is_off = True
                break
        sample[Fields.stats]["is_off_task"] = is_off
        return sample

    def process_single(self, sample):
        return not sample[Fields.stats].get("is_off_task", False)


# ═══════════════════════════════════════════════════════════════════════
# Filter 2: Trajectory SAL (Spectral Arc Length) Filter
# ═══════════════════════════════════════════════════════════════════════

@OPERATORS.register_module("trajectory_sal_filter")
class TrajectorySALFilter(Filter):
    """Compute SAL smoothness metric and filter rough episodes.

    SAL is computed on joint velocity magnitude for left/right arms.
    Episodes with combined SAL below min_sal are filtered out.
    """

    def __init__(self,
                 min_sal=-1500.0,
                 parquet_field="parquet_path",
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_sal = min_sal
        self.parquet_field = parquet_field

    def compute_stats_single(self, sample, context=False):
        ppath = sample.get(self.parquet_field, "")
        if not ppath:
            sample[Fields.stats]["sal_combined"] = 0.0
            return sample

        states, _, _ = _load_episode_arrays(ppath)
        n = states.shape[0]
        if n < 10:
            sample[Fields.stats]["sal_combined"] = 0.0
            return sample

        def _arm_ee_sal(dims):
            ee = states[:, dims]
            vel = np.linalg.norm(np.diff(ee, axis=0) / DT, axis=1)
            return _spectral_arc_length(vel)

        sal_left = _arm_ee_sal(LEFT_ARM_VALID)
        sal_right = _arm_ee_sal(RIGHT_ARM_VALID)

        sal_left_per_joint = [
            _spectral_arc_length(np.abs(np.diff(states[:, d]) / DT))
            for d in LEFT_ARM_VALID
        ]
        sal_right_per_joint = [
            _spectral_arc_length(np.abs(np.diff(states[:, d]) / DT))
            for d in RIGHT_ARM_VALID
        ]

        sample[Fields.stats]["sal_left_ee"] = round(sal_left, 2)
        sample[Fields.stats]["sal_right_ee"] = round(sal_right, 2)
        sample[Fields.stats]["sal_combined"] = round(
            (sal_left + sal_right) / 2, 2)
        sample[Fields.stats]["sal_left_mean"] = round(
            float(np.mean(sal_left_per_joint)), 2)
        sample[Fields.stats]["sal_right_mean"] = round(
            float(np.mean(sal_right_per_joint)), 2)
        return sample

    def process_single(self, sample):
        val = sample[Fields.stats].get("sal_combined", 0.0)
        return val >= self.min_sal


# ═══════════════════════════════════════════════════════════════════════
# Filter 3: Trajectory DA (Directional Alignment) Filter
# ═══════════════════════════════════════════════════════════════════════

@OPERATORS.register_module("trajectory_da_filter")
class TrajectoryDAFilter(Filter):
    """Check state-action directional alignment per joint dimension.

    For each arm joint, computes the fraction of active frames where
    state and action move in the same direction.  Episodes with too many
    misaligned dimensions are filtered.
    """

    def __init__(self,
                 da_thresh=0.65,
                 max_misaligned_dims=0,
                 eps_frac=0.01,
                 parquet_field="parquet_path",
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.da_thresh = da_thresh
        self.max_misaligned_dims = max_misaligned_dims
        self.eps_frac = eps_frac
        self.parquet_field = parquet_field

    def compute_stats_single(self, sample, context=False):
        ppath = sample.get(self.parquet_field, "")
        if not ppath:
            sample[Fields.stats]["da_misaligned_dims"] = 0
            return sample

        states, actions, _ = _load_episode_arrays(ppath)
        n = states.shape[0]
        if n < 20:
            sample[Fields.stats]["da_misaligned_dims"] = 0
            sample[Fields.stats]["da_details"] = []
            return sample

        shared_dims = LEFT_ARM_VALID + RIGHT_ARM_VALID
        dim_results = []

        for dim in shared_dims:
            s = states[:, dim]
            a = actions[:, dim]
            q01, q99 = np.percentile(s, [1, 99])
            joint_range = q99 - q01
            eps = max(self.eps_frac * joint_range, 1e-6)

            ds = np.diff(s)
            da = np.diff(a)
            active = (np.abs(ds) > eps) | (np.abs(da) > eps)
            if active.sum() < 10:
                continue

            agree = np.sign(ds[active]) == np.sign(da[active])
            da_score = float(agree.mean())
            aligned = da_score >= self.da_thresh

            dim_results.append({
                "dim": int(dim),
                "da_score": round(da_score, 3),
                "n_active": int(active.sum()),
                "aligned": aligned,
            })

        misaligned = [d for d in dim_results if not d["aligned"]]
        sample[Fields.stats]["da_misaligned_dims"] = len(misaligned)
        sample[Fields.stats]["da_total_checked"] = len(dim_results)
        sample[Fields.stats]["da_aligned"] = len(misaligned) == 0
        sample[Fields.stats]["da_details"] = misaligned
        return sample

    def process_single(self, sample):
        n_mis = sample[Fields.stats].get("da_misaligned_dims", 0)
        return n_mis <= self.max_misaligned_dims


# ═══════════════════════════════════════════════════════════════════════
# Filter 4: Trajectory Sudden Change Filter
# ═══════════════════════════════════════════════════════════════════════

@OPERATORS.register_module("trajectory_sudden_change_filter")
class TrajectorySuddenChangeFilter(Filter):
    """Detect sudden jumps in joint positions via acceleration thresholding.

    Flags episodes where |accel| > c_threshold * sigma for any arm joint.
    """

    def __init__(self,
                 c_threshold=8.0,
                 floor_frac=0.02,
                 parquet_field="parquet_path",
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.c_threshold = c_threshold
        self.floor_frac = floor_frac
        self.parquet_field = parquet_field

    def compute_stats_single(self, sample, context=False):
        ppath = sample.get(self.parquet_field, "")
        if not ppath:
            sample[Fields.stats]["sudden_change_flags"] = 0
            return sample

        states, _, _ = _load_episode_arrays(ppath)
        n = states.shape[0]
        if n < 15:
            sample[Fields.stats]["sudden_change_flags"] = 0
            return sample

        arm_dims = LEFT_ARM_VALID + RIGHT_ARM_VALID
        total_flags = 0
        flagged_dims = []

        for dim in arm_dims:
            signal = states[:, dim]
            q01, q99 = np.percentile(signal, [1, 99])
            joint_range = q99 - q01
            accel = np.diff(signal, n=2)
            sigma = max(np.std(accel), self.floor_frac * joint_range, 0.01)
            n_flags = int(np.sum(np.abs(accel) > self.c_threshold * sigma))
            if n_flags > 0:
                flagged_dims.append({"dim": int(dim), "n_flags": n_flags})
                total_flags += n_flags

        sample[Fields.stats]["sudden_change_flags"] = total_flags
        sample[Fields.stats]["sudden_change_flagged"] = total_flags > 0
        sample[Fields.stats]["sudden_change_dims"] = flagged_dims
        return sample

    def process_single(self, sample):
        return not sample[Fields.stats].get("sudden_change_flagged", False)


# ═══════════════════════════════════════════════════════════════════════
# Filter 5: Quality Label Filter
# ═══════════════════════════════════════════════════════════════════════

@OPERATORS.register_module("quality_label_filter")
class QualityLabelFilter(Filter):
    """Compute qualified/unqualified frame ratio from parquet quality_index.

    Optionally filter episodes where qualified ratio is below a threshold.
    """

    def __init__(self,
                 qualified_task_index=25,
                 unqualified_task_index=9,
                 min_qualified_ratio=0.0,
                 parquet_field="parquet_path",
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.qualified_idx = qualified_task_index
        self.unqualified_idx = unqualified_task_index
        self.min_qualified_ratio = min_qualified_ratio
        self.parquet_field = parquet_field

    def compute_stats_single(self, sample, context=False):
        ppath = sample.get(self.parquet_field, "")
        if not ppath:
            sample[Fields.stats]["qualified_ratio"] = 0.0
            return sample

        df = pd.read_parquet(ppath, columns=["quality_index"])
        qi = df["quality_index"].values
        n_total = len(qi)
        n_qualified = int(np.sum(qi == self.qualified_idx))
        n_unqualified = int(np.sum(qi == self.unqualified_idx))

        ratio = n_qualified / n_total if n_total > 0 else 0.0
        sample[Fields.stats]["qualified_frames"] = n_qualified
        sample[Fields.stats]["unqualified_frames"] = n_unqualified
        sample[Fields.stats]["total_frames"] = n_total
        sample[Fields.stats]["qualified_ratio"] = round(ratio, 4)
        return sample

    def process_single(self, sample):
        ratio = sample[Fields.stats].get("qualified_ratio", 0.0)
        return ratio >= self.min_qualified_ratio


# ═══════════════════════════════════════════════════════════════════════
# Mapper: Trajectory Statistics
# ═══════════════════════════════════════════════════════════════════════

STATE_SLOT_NAMES = {
    "left_arm_j0": 0, "left_arm_j2": 2,
    "left_arm_j3": 3, "left_arm_j4": 4, "left_arm_j5": 5, "left_arm_j6": 6,
    "left_gripper": 7,
    "right_arm_j0": 8, "right_arm_j2": 10,
    "right_arm_j3": 11, "right_arm_j4": 12, "right_arm_j5": 13,
    "right_arm_j6": 14, "right_gripper": 15,
    "torso_0": 16, "torso_1": 17, "torso_2": 18, "torso_3": 19,
    "chassis_x": 24, "chassis_y": 25, "chassis_z": 26,
    "chassis_vx": 27, "chassis_vy": 28, "chassis_vz": 29,
}


@OPERATORS.register_module("trajectory_stats_mapper")
class TrajectoryStatsMapper(Mapper):
    """Compute per-episode trajectory statistics and store in __dj__meta__."""

    def __init__(self, parquet_field="parquet_path", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parquet_field = parquet_field

    def process_single(self, sample):
        ppath = sample.get(self.parquet_field, "")
        if not ppath:
            return sample

        states, actions, df = _load_episode_arrays(ppath)
        pad_s = np.array(df["observation.state_dim_is_pad"].iloc[0])
        valid_s = np.where(~pad_s)[0]
        name_lookup = {v: k for k, v in STATE_SLOT_NAMES.items()}

        per_dim = {}
        for d in valid_s:
            col = states[:, d]
            name = name_lookup.get(int(d), f"dim_{d}")
            per_dim[name] = {
                "dim": int(d),
                "min": round(float(col.min()), 6),
                "max": round(float(col.max()), 6),
                "mean": round(float(col.mean()), 6),
                "std": round(float(col.std()), 6),
            }

        lg = states[:, 7]
        rg = states[:, 15]
        gripper_stats = {
            "left": {
                "min": round(float(lg.min()), 2),
                "max": round(float(lg.max()), 2),
                "mean": round(float(lg.mean()), 2),
                "n_changes": int(np.sum(np.abs(np.diff(lg)) > 1.0)),
            },
            "right": {
                "min": round(float(rg.min()), 2),
                "max": round(float(rg.max()), 2),
                "mean": round(float(rg.mean()), 2),
                "n_changes": int(np.sum(np.abs(np.diff(rg)) > 1.0)),
            },
        }

        if Fields.meta not in sample or sample[Fields.meta] is None:
            sample[Fields.meta] = {}
        sample[Fields.meta]["state_per_dim"] = per_dim
        sample[Fields.meta]["gripper_stats"] = gripper_stats
        sample[Fields.meta]["num_frames"] = int(states.shape[0])
        sample[Fields.meta]["valid_state_dims"] = len(valid_s)
        return sample
