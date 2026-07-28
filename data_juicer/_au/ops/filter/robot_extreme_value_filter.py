# -*- coding: utf-8 -*-
"""Qwen-RobotManip Stage 3: Extreme Value Filtering — data-juicer 自定义 Filter。

对机器人轨迹的 state/action 逐维检测帧级极值：
    加载 per-embodiment 全局分位数 → 计算 alpha-扩展带 → 逐帧逐维判定
    → 标记/裁剪/丢弃。

夹爪维度豁免（双峰分布不适合分位数过滤）。
"""

import json

import numpy as np
from loguru import logger

from data_juicer.ops.base_op import OPERATORS, Filter
from data_juicer.utils.constant import Fields, MetaKeys

OP_NAME = "robot_extreme_value_filter"


@OPERATORS.register_module(OP_NAME)
class RobotExtremeValueFilter(Filter):
    """Filter out frames (or episodes) with extreme state/action values.

    判定公式（逐维 d）：
        flag_t = OR_d( x_{t,d} < lo_d  OR  x_{t,d} > hi_d )
    其中 lo_d = q01_d - alpha * IQR_d, hi_d = q99_d + alpha * IQR_d,
    IQR_d = q99_d - q01_d。夹爪维度跳过（exempt_dims）。
    """

    def __init__(
        self,
        # ---- 信号来源 ----
        signal_source: str = "top_level",
        hand_action_field: str = MetaKeys.hand_action_tags,
        states_key: str = "states",
        actions_key: str = "actions",
        top_level_state_key: str = "states",
        top_level_action_key: str = "actions",
        meta_signal_field: str = None,
        # ---- 分位数来源 ----
        percentile_source: str = "param",
        percentile_stats_path: str = None,
        embodiment: str = None,
        embodiment_field: str = "robot_type",
        state_q01: list = None,
        state_q99: list = None,
        action_q01: list = None,
        action_q99: list = None,
        # ---- 过滤参数 ----
        alpha: float = 0.1,
        check_dims: dict = None,
        exempt_dims: list = None,
        min_frames: int = 4,
        # ---- 排除策略 ----
        exclusion_strategy: str = "frame_mask",
        max_flagged_ratio: float = 0.3,
        mask_field: str = "valid_frame_mask",
        report_field: str = "extreme_value_report",
        recompute_actions_on_remove: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if signal_source not in ("hand_action_tags", "top_level", "meta_field"):
            raise ValueError(f"Invalid signal_source: {signal_source}")
        if percentile_source not in ("stats_json", "param", "self"):
            raise ValueError(f"Invalid percentile_source: {percentile_source}")
        if exclusion_strategy not in ("episode_discard", "frame_mask", "frame_remove"):
            raise ValueError(f"Invalid exclusion_strategy: {exclusion_strategy}")
        if percentile_source == "stats_json" and percentile_stats_path is None:
            raise ValueError("stats_json mode requires percentile_stats_path.")
        if percentile_source == "param" and (
            state_q01 is None or state_q99 is None or action_q01 is None or action_q99 is None
        ):
            raise ValueError("param mode requires state_q01/q99 and action_q01/q99.")

        self.signal_source = signal_source
        self.hand_action_field = hand_action_field
        self.states_key = states_key
        self.actions_key = actions_key
        self.top_level_state_key = top_level_state_key
        self.top_level_action_key = top_level_action_key
        self.meta_signal_field = meta_signal_field

        self.percentile_source = percentile_source
        self.percentile_stats_path = percentile_stats_path
        self.embodiment = embodiment
        self.embodiment_field = embodiment_field
        self.state_q01 = state_q01
        self.state_q99 = state_q99
        self.action_q01 = action_q01
        self.action_q99 = action_q99

        self.alpha = float(alpha)
        self.check_dims = check_dims
        self.exempt_dims = set(exempt_dims) if exempt_dims else set()
        self.min_frames = int(min_frames)

        self.exclusion_strategy = exclusion_strategy
        self.max_flagged_ratio = float(max_flagged_ratio)
        self.mask_field = mask_field
        self.report_field = report_field
        self.recompute_actions_on_remove = recompute_actions_on_remove

        self._percentile_cache = {}

    def _get_check_dims(self, D):
        """Return sorted list of dimension indices to check."""
        if self.check_dims and "include" in self.check_dims:
            dims = set(self.check_dims["include"])
        elif self.check_dims and "exclude" in self.check_dims:
            dims = set(range(D)) - set(self.check_dims["exclude"])
        else:
            dims = set(range(D))
        dims -= self.exempt_dims
        return sorted(d for d in dims if 0 <= d < D)

    def _iter_signal_blocks(self, sample):
        """Extract (name, states_array, actions_array) from sample."""
        blocks = []
        if self.signal_source == "top_level":
            s = sample.get(self.top_level_state_key)
            a = sample.get(self.top_level_action_key)
            if s is not None and a is not None:
                blocks.append(("top_level", s, a))
        elif self.signal_source == "hand_action_tags":
            meta = sample.get(Fields.meta, {}) or {}
            tags = meta.get(self.hand_action_field)
            if tags:
                if isinstance(tags, str):
                    tags = json.loads(tags)
                for i, hand in enumerate(tags):
                    s = hand.get(self.states_key)
                    a = hand.get(self.actions_key)
                    if s is not None and a is not None:
                        blocks.append((f"hand_{i}", s, a))
        elif self.signal_source == "meta_field" and self.meta_signal_field:
            meta = sample.get(Fields.meta, {}) or {}
            sig = meta.get(self.meta_signal_field)
            if sig is not None:
                mat = np.array(sig, dtype=np.float64)
                half = mat.shape[1] // 2
                blocks.append(("meta", mat[:, :half], mat[:, half:]))
        return blocks

    def _resolve_bands(self, sample, D_state, D_action):
        """Load/compute percentiles and return (state_lo, state_hi, action_lo, action_hi)."""
        if self.percentile_source == "stats_json":
            emb = self.embodiment
            if emb is None:
                emb = sample.get(self.embodiment_field, "default")
            if emb in self._percentile_cache:
                return self._percentile_cache[emb]
            with open(self.percentile_stats_path) as f:
                all_stats = json.load(f)
            if emb not in all_stats:
                logger.warning(f"Embodiment '{emb}' not in {self.percentile_stats_path}, using first available.")
                emb = next(iter(all_stats))
            stats = all_stats[emb]
            sq01 = np.array(stats["state"]["q01"], dtype=np.float64)
            sq99 = np.array(stats["state"]["q99"], dtype=np.float64)
            aq01 = np.array(stats["action"]["q01"], dtype=np.float64)
            aq99 = np.array(stats["action"]["q99"], dtype=np.float64)

        elif self.percentile_source == "param":
            if "__param__" in self._percentile_cache:
                return self._percentile_cache["__param__"]
            sq01 = np.array(self.state_q01, dtype=np.float64)
            sq99 = np.array(self.state_q99, dtype=np.float64)
            aq01 = np.array(self.action_q01, dtype=np.float64)
            aq99 = np.array(self.action_q99, dtype=np.float64)

        elif self.percentile_source == "self":
            states = np.array(sample.get(self.top_level_state_key, []), dtype=np.float64)
            actions = np.array(sample.get(self.top_level_action_key, []), dtype=np.float64)
            if states.ndim < 2 or actions.ndim < 2:
                return (np.zeros(D_state), np.zeros(D_state), np.zeros(D_action), np.zeros(D_action))
            sq01 = np.percentile(states, 1, axis=0)
            sq99 = np.percentile(states, 99, axis=0)
            aq01 = np.percentile(actions, 1, axis=0)
            aq99 = np.percentile(actions, 99, axis=0)

        s_iqr = sq99 - sq01
        a_iqr = aq99 - aq01
        bands = (
            sq01 - self.alpha * s_iqr,
            sq99 + self.alpha * s_iqr,
            aq01 - self.alpha * a_iqr,
            aq99 + self.alpha * a_iqr,
        )

        if self.percentile_source == "stats_json":
            self._percentile_cache[emb] = bands
        elif self.percentile_source == "param":
            self._percentile_cache["__param__"] = bands

        return bands

    def _flag_frames(self, states, actions, bands):
        """Vectorized frame-level extreme value detection. Returns bool mask (T,)."""
        state_lo, state_hi, action_lo, action_hi = bands
        T = states.shape[0]
        flag_mask = np.zeros(T, dtype=bool)

        s_check = self._get_check_dims(states.shape[1])
        if s_check:
            s = states[:, s_check]
            lo = state_lo[s_check]
            hi = state_hi[s_check]
            flag_mask |= ((s < lo) | (s > hi)).any(axis=1)

        a_check = self._get_check_dims(actions.shape[1])
        if a_check:
            a = actions[:, a_check]
            lo = action_lo[a_check]
            hi = action_hi[a_check]
            flag_mask |= ((a < lo) | (a > hi)).any(axis=1)

        return flag_mask

    def _apply_frame_remove(self, sample, keep_mask):
        """Remove flagged frames from states/actions arrays."""
        for key in (self.top_level_state_key, self.top_level_action_key):
            val = sample.get(key)
            if val is not None:
                arr = np.array(val, dtype=np.float64)
                sample[key] = arr[keep_mask].tolist()
        return sample

    def _write_default_stats(self, sample, keep=True):
        sample[Fields.stats]["extreme_value_keep"] = keep
        sample[Fields.stats]["extreme_value_flagged_frames"] = 0
        sample[Fields.stats]["extreme_value_flagged_ratio"] = 0.0
        sample[Fields.stats]["extreme_value_num_check_dims"] = 0
        sample[Fields.stats]["extreme_value_num_frames"] = 0
        return sample

    def compute_stats_single(self, sample, context=False):
        if Fields.stats not in sample or sample[Fields.stats] is None:
            sample[Fields.stats] = {}
        if Fields.meta not in sample or sample[Fields.meta] is None:
            sample[Fields.meta] = {}

        blocks = self._iter_signal_blocks(sample)
        if not blocks:
            return self._write_default_stats(sample)

        name, states_raw, actions_raw = blocks[0]
        states = np.array(states_raw, dtype=np.float64)
        actions = np.array(actions_raw, dtype=np.float64)

        if states.ndim < 2 or actions.ndim < 2:
            return self._write_default_stats(sample)

        T = states.shape[0]
        if T < self.min_frames:
            return self._write_default_stats(sample)

        bands = self._resolve_bands(sample, states.shape[1], actions.shape[1])
        flag_mask = self._flag_frames(states, actions, bands)

        flagged_frames = int(flag_mask.sum())
        flagged_ratio = flagged_frames / T
        num_check_dims = len(self._get_check_dims(states.shape[1])) + len(self._get_check_dims(actions.shape[1]))

        sample[Fields.stats]["extreme_value_flagged_frames"] = flagged_frames
        sample[Fields.stats]["extreme_value_flagged_ratio"] = flagged_ratio
        sample[Fields.stats]["extreme_value_num_check_dims"] = num_check_dims
        sample[Fields.stats]["extreme_value_num_frames"] = T

        report = {
            "embodiment": sample.get(self.embodiment_field, "unknown"),
            "strategy": self.exclusion_strategy,
            "alpha": self.alpha,
            "flagged_frames": flagged_frames,
            "flagged_ratio": round(flagged_ratio, 6),
            "total_frames": T,
            "num_check_dims": num_check_dims,
        }
        sample[Fields.meta][self.report_field] = json.dumps(report)

        if self.exclusion_strategy == "frame_mask":
            keep_mask = ~flag_mask
            existing = sample[Fields.meta].get(self.mask_field)
            if existing is not None:
                if isinstance(existing, str):
                    prev = np.array(json.loads(existing), dtype=bool)
                elif isinstance(existing, (list, np.ndarray)):
                    prev = np.asarray(existing, dtype=bool)
                else:
                    prev = None
                if prev is not None and prev.ndim == 1 and len(prev) == T:
                    keep_mask = keep_mask & prev
            sample[Fields.meta][self.mask_field] = json.dumps(keep_mask.tolist())
            sample[Fields.stats]["extreme_value_keep"] = True

        elif self.exclusion_strategy == "frame_remove":
            keep_mask = ~flag_mask
            if int(keep_mask.sum()) >= self.min_frames:
                sample = self._apply_frame_remove(sample, keep_mask)
                sample[Fields.stats]["extreme_value_keep"] = True
            else:
                sample[Fields.stats]["extreme_value_keep"] = False

        elif self.exclusion_strategy == "episode_discard":
            sample[Fields.stats]["extreme_value_keep"] = flagged_ratio <= self.max_flagged_ratio

        return sample

    def process_single(self, sample):
        return sample[Fields.stats].get("extreme_value_keep", True)
