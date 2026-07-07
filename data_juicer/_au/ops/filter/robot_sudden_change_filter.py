# b/d/QwenRobotmanip/dj_custom_ops/robot_sudden_change_filter.py
# -*- coding: utf-8 -*-
"""Qwen-RobotManip Stage 1: Sudden Change Detection —— data-juicer 自定义 Filter。

对机器人轨迹信号（states/actions 或统一向量）逐维做：
    级联中值 + Savitzky-Golay 平滑 -> 残差 / 二阶差分(acc) / 三阶差分(jerk)
    -> 联合阈值判定异常帧 -> episode 级判据 -> 按策略删除。
"""

import json

import numpy as np
from loguru import logger

from data_juicer.ops.base_op import OPERATORS, Filter
from data_juicer.utils.constant import Fields, MetaKeys

OP_NAME = "robot_sudden_change_filter"


@OPERATORS.register_module(OP_NAME)
class RobotSuddenChangeFilter(Filter):
    """Filter out episodes (or mark frames) with sudden changes in trajectories.

    判定公式（逐维）：
        flag = (residual > tau_r) AND (|acc| > tau_a OR |jerk| > tau_j)
    帧级为各维 OR；episode 级按异常帧比例 / 最长连续异常段判定是否丢弃。
    """

    def __init__(
        self,
        # ---- 信号来源 ----
        signal_source: str = "hand_action_tags",
        hand_action_field: str = MetaKeys.hand_action_tags,
        states_key: str = "states",
        actions_key: str = "actions",
        top_level_state_key: str = "states",
        top_level_action_key: str = "actions",
        meta_signal_field: str = None,
        # ---- 平滑参数 ----
        median_windows: tuple = (3, 5),
        savgol_window: int = 11,
        savgol_polyorder: int = 3,
        # ---- 阈值参数 ----
        threshold_mode: str = "mad",
        mad_scale_residual: float = 6.0,
        mad_scale_acc: float = 6.0,
        mad_scale_jerk: float = 6.0,
        residual_threshold: float = None,
        acc_threshold: float = None,
        jerk_threshold: float = None,
        # ---- 维度选择 ----
        check_dims: dict = None,   # 例：{"include":[0,1,2]} 或 {"exclude":[6,7]}
        exempt_dims: list = None,  # 直接跳过的维度（如夹爪/padding）
        angular_dims: list = None,  # 需要先 np.unwrap 的角度维（欧拉角）
        # ---- episode 级判据 ----
        max_flagged_ratio: float = 0.0,
        max_run_length: int = 0,
        min_frames: int = 4,
        # ---- 删除策略 ----
        exclusion_strategy: str = "episode_discard",
        mask_field: str = "valid_frame_mask",
        report_field: str = "sudden_change_report",
        recompute_actions_on_remove: bool = False,
        *args,
        **kwargs,
    ):
        """
        :param signal_source: 信号来源，{"hand_action_tags","top_level","meta_field"}。
        :param hand_action_field: signal_source=hand_action_tags 时，meta 中的字段名。
        :param states_key/actions_key: hand_action_tags 每个 hand dict 内的键名。
        :param top_level_state_key/top_level_action_key: top_level 时样本顶层键名。
        :param meta_signal_field: meta_field 时，meta 下的 (T,D) 数组字段名。
        :param median_windows: 级联中值滤波窗口序列（会自动取奇数并裁剪到不超过 T）。
        :param savgol_window: SG 平滑窗口（自动取奇数，且需 >= polyorder+2 才启用）。
        :param savgol_polyorder: SG 多项式阶数。
        :param threshold_mode: {"mad","manual"}。
        :param mad_scale_*: mad 模式下残差/acc/jerk 的缩放系数 lambda。
        :param *_threshold: manual 模式下的固定阈值。
        :param check_dims: {"include":[...]} 与/或 {"exclude":[...]}，控制参与检测的维度。
        :param exempt_dims: 一律跳过的维度索引（离散通道，如夹爪开合、padding）。
        :param angular_dims: 需先做 np.unwrap 的角度维索引（避免 pi/-pi 跳变误判）。
        :param max_flagged_ratio: episode 允许的异常帧最大占比（>此值则丢弃）。
        :param max_run_length: episode 允许的最长连续异常段（>此值则丢弃）。
        :param min_frames: 少于该帧数的轨迹视为过短，安全保留、不检测。
        :param exclusion_strategy: {"episode_discard","frame_mask","frame_remove"}。
        :param mask_field: frame_mask 策略下写入 meta 的有效帧掩码字段名。
        :param report_field: 明细报告写入 meta 的字段名。
        :param recompute_actions_on_remove: frame_remove 且 8 维 states 时重算 delta。
        """
        super().__init__(*args, **kwargs)

        if signal_source not in ("hand_action_tags", "top_level", "meta_field"):
            raise ValueError(f"Invalid signal_source: {signal_source}")
        if threshold_mode not in ("mad", "manual"):
            raise ValueError(f"Invalid threshold_mode: {threshold_mode}")
        if exclusion_strategy not in ("episode_discard", "frame_mask", "frame_remove"):
            raise ValueError(f"Invalid exclusion_strategy: {exclusion_strategy}")
        if threshold_mode == "manual" and (
            residual_threshold is None or acc_threshold is None or jerk_threshold is None
        ):
            raise ValueError("manual mode requires residual/acc/jerk_threshold all set.")

        self.signal_source = signal_source
        self.hand_action_field = hand_action_field
        self.states_key = states_key
        self.actions_key = actions_key
        self.top_level_state_key = top_level_state_key
        self.top_level_action_key = top_level_action_key
        self.meta_signal_field = meta_signal_field

        self.median_windows = tuple(int(w) for w in median_windows)
        self.savgol_window = int(savgol_window)
        self.savgol_polyorder = int(savgol_polyorder)

        self.threshold_mode = threshold_mode
        self.mad_scale_residual = float(mad_scale_residual)
        self.mad_scale_acc = float(mad_scale_acc)
        self.mad_scale_jerk = float(mad_scale_jerk)
        self.residual_threshold = residual_threshold
        self.acc_threshold = acc_threshold
        self.jerk_threshold = jerk_threshold

        self.check_dims = check_dims
        self.exempt_dims = set(exempt_dims) if exempt_dims else set()
        self.angular_dims = set(angular_dims) if angular_dims else set()

        self.max_flagged_ratio = float(max_flagged_ratio)
        self.max_run_length = int(max_run_length)
        self.min_frames = int(min_frames)

        self.exclusion_strategy = exclusion_strategy
        self.mask_field = mask_field
        self.report_field = report_field
        self.recompute_actions_on_remove = bool(recompute_actions_on_remove)

    # ------------------------------------------------------------------ #
    # 基础工具
    # ------------------------------------------------------------------ #
    @staticmethod
    def _as_2d(x) -> np.ndarray:
        """把任意轨迹归一为 (T, D) 的 float64 数组；(T,) -> (T, 1)。"""
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[:, None]
        return arr

    def _select_dims(self, D: int) -> list:
        """按 check_dims / exempt_dims 计算真正参与检测的维度索引。"""
        dims = list(range(D))
        if self.check_dims:
            if "include" in self.check_dims:
                inc = set(self.check_dims["include"])
                dims = [d for d in dims if d in inc]
            if "exclude" in self.check_dims:
                exc = set(self.check_dims["exclude"])
                dims = [d for d in dims if d not in exc]
        if self.exempt_dims:
            dims = [d for d in dims if d not in self.exempt_dims]
        return dims

    def _cascaded_smooth(self, x: np.ndarray) -> np.ndarray:
        """级联中值 + Savitzky-Golay 平滑，输入/输出均为 (T, D)。"""
        from scipy.ndimage import median_filter
        from scipy.signal import savgol_filter

        y = x.copy()
        T = y.shape[0]
        for w in self.median_windows:
            w = int(w)
            if w % 2 == 0:
                w -= 1
            if 3 <= w <= T:
                y = median_filter(y, size=(w, 1), mode="nearest")

        win = min(self.savgol_window, T)
        if win % 2 == 0:
            win -= 1
        if win >= self.savgol_polyorder + 2:
            y = savgol_filter(y, win, self.savgol_polyorder, axis=0, mode="interp")
        return y

    @staticmethod
    def _finite_diff(x: np.ndarray):
        """二阶差分(acc)与三阶差分(jerk)，对齐到长度 T（边界补 0）。"""
        T = x.shape[0]
        acc = np.zeros_like(x)
        jerk = np.zeros_like(x)
        if T >= 3:
            acc[1 : T - 1] = x[2:] - 2.0 * x[1:-1] + x[:-2]
        if T >= 4:
            jerk[1 : T - 2] = x[3:] - 3.0 * x[2:-1] + 3.0 * x[1:-2] - x[:-3]
        return acc, jerk

    def _dim_threshold(self, values: np.ndarray, scale: float, manual):
        """逐维阈值：manual 返回常数向量；mad 返回 median + scale*1.4826*MAD。"""
        D = values.shape[1]
        if self.threshold_mode == "manual":
            return np.full(D, float(manual))
        med = np.nanmedian(values, axis=0)
        mad = np.nanmedian(np.abs(values - med), axis=0)
        return med + scale * 1.4826 * np.clip(mad, 1e-8, None)

    @staticmethod
    def _max_run_length(mask: np.ndarray) -> int:
        """连续 True 的最长游程。"""
        best = cur = 0
        for v in mask:
            if v:
                cur += 1
                best = max(best, cur)
            else:
                cur = 0
        return int(best)

    # ------------------------------------------------------------------ #
    # 信号抽取
    # ------------------------------------------------------------------ #
    def _iter_signal_blocks(self, sample: dict) -> list:
        """返回 [(block_name, raw_block), ...]，raw_block 可被 _as_2d 处理。"""
        blocks = []
        if self.signal_source == "hand_action_tags":
            meta = sample.get(Fields.meta, {}) or {}
            clips = meta.get(self.hand_action_field) or []
            for ci, clip in enumerate(clips):
                if not isinstance(clip, dict):
                    continue
                for hand_type, hand in clip.items():
                    if not isinstance(hand, dict):
                        continue
                    for key in (self.states_key, self.actions_key):
                        blk = hand.get(key)
                        if blk is not None and len(blk) > 0:
                            blocks.append((f"clip{ci}.{hand_type}.{key}", blk))
        elif self.signal_source == "meta_field":
            meta = sample.get(Fields.meta, {}) or {}
            blk = meta.get(self.meta_signal_field)
            if blk is not None and len(blk) > 0:
                blocks.append((self.meta_signal_field, blk))
        else:  # top_level
            for key in (self.top_level_state_key, self.top_level_action_key):
                blk = sample.get(key)
                if blk is not None and len(blk) > 0:
                    blocks.append((key, blk))
        return blocks

    # ------------------------------------------------------------------ #
    # 单块检测
    # ------------------------------------------------------------------ #
    def _detect_block(self, block) -> dict:
        x = self._as_2d(block)
        T, D = x.shape
        result = {
            "num_frames": T,
            "frame_flags": np.zeros(T, dtype=bool),
            "max_residual": 0.0,
            "max_acc": 0.0,
            "max_jerk": 0.0,
            "bad_dims": [],
            "insufficient_length": bool(T < self.min_frames),
        }
        if T < self.min_frames:
            return result

        dims = self._select_dims(D)
        if not dims:
            return result

        xs = x[:, dims].copy()
        # 角度维先解卷绕，避免 pi/-pi 跳变误判
        if self.angular_dims:
            for local_i, d in enumerate(dims):
                if d in self.angular_dims:
                    xs[:, local_i] = np.unwrap(xs[:, local_i])

        smooth = self._cascaded_smooth(xs)
        residual = np.abs(xs - smooth)
        acc, jerk = self._finite_diff(xs)
        abs_acc, abs_jerk = np.abs(acc), np.abs(jerk)

        tr = self._dim_threshold(residual, self.mad_scale_residual, self.residual_threshold)
        ta = self._dim_threshold(abs_acc, self.mad_scale_acc, self.acc_threshold)
        tj = self._dim_threshold(abs_jerk, self.mad_scale_jerk, self.jerk_threshold)

        dim_flags = (residual > tr) & ((abs_acc > ta) | (abs_jerk > tj))
        frame_flags = np.any(dim_flags, axis=1)

        result["frame_flags"] = frame_flags
        result["max_residual"] = float(residual.max()) if residual.size else 0.0
        result["max_acc"] = float(abs_acc.max()) if abs_acc.size else 0.0
        result["max_jerk"] = float(abs_jerk.max()) if abs_jerk.size else 0.0
        bad_local = np.where(np.any(dim_flags, axis=0))[0].tolist()
        result["bad_dims"] = [dims[i] for i in bad_local]
        return result

    # ------------------------------------------------------------------ #
    # frame_remove 辅助（仅 top_level 8 维 states + 7 维 actions 支持自动重算）
    # ------------------------------------------------------------------ #
    @staticmethod
    def _recompute_actions(states: np.ndarray) -> np.ndarray:
        """由连续 8 维 states 重算 7 维 delta actions（与官方 mapper 一致）。"""
        from scipy.spatial.transform import Rotation

        T = len(states)
        actions = np.zeros((T, 7), dtype=np.float64)
        for t in range(T - 1):
            actions[t, 0:3] = states[t + 1, 0:3] - states[t, 0:3]
            r_prev = Rotation.from_euler("xyz", states[t, 3:6], degrees=False)
            r_next = Rotation.from_euler("xyz", states[t + 1, 3:6], degrees=False)
            actions[t, 3:6] = (r_next * r_prev.inv()).as_euler("xyz", degrees=False)
            actions[t, 6] = states[t + 1, 7]
        if T > 0:
            actions[T - 1, 6] = states[T - 1, 7]
        return actions

    def _apply_frame_remove(self, sample: dict, keep_mask: np.ndarray):
        """按 keep_mask 裁剪 top_level states/actions 并（可选）重算 actions。"""
        st_key, ac_key = self.top_level_state_key, self.top_level_action_key
        states = sample.get(st_key)
        if states is not None and len(states) == len(keep_mask):
            arr = np.asarray(states, dtype=np.float64)[keep_mask]
            if self.recompute_actions_on_remove and arr.ndim == 2 and arr.shape[1] == 8:
                sample[st_key] = arr.tolist()
                sample[ac_key] = self._recompute_actions(arr).tolist()
                return
            sample[st_key] = arr.tolist()
        actions = sample.get(ac_key)
        if actions is not None and len(actions) == len(keep_mask):
            sample[ac_key] = np.asarray(actions, dtype=np.float64)[keep_mask].tolist()

    # ------------------------------------------------------------------ #
    # 两阶段接口
    # ------------------------------------------------------------------ #
    def compute_stats_single(self, sample, context=False):
        stats = sample[Fields.stats]
        blocks = self._iter_signal_blocks(sample)

        total_frames = 0
        total_flagged = 0
        max_run = 0
        max_res = max_acc = max_jerk = 0.0
        bad_dims = set()
        per_block_reports = []
        combined_masks = {}

        for name, blk in blocks:
            r = self._detect_block(blk)
            ff = r["frame_flags"]
            total_frames += r["num_frames"]
            total_flagged += int(ff.sum())
            max_run = max(max_run, self._max_run_length(ff))
            max_res = max(max_res, r["max_residual"])
            max_acc = max(max_acc, r["max_acc"])
            max_jerk = max(max_jerk, r["max_jerk"])
            for d in r["bad_dims"]:
                bad_dims.add(f"{name}.dim{d}")
            per_block_reports.append(
                {
                    "name": name,
                    "num_frames": r["num_frames"],
                    "num_flagged": int(ff.sum()),
                    "flagged_frame_ids": np.where(ff)[0].tolist(),
                    "insufficient_length": r["insufficient_length"],
                }
            )
            combined_masks[name] = (~ff).tolist()  # True = 有效帧

        flagged_ratio = (total_flagged / total_frames) if total_frames else 0.0
        if total_frames == 0:
            keep = True  # 无可检测信号 -> 安全保留
        else:
            keep = not (
                (flagged_ratio > self.max_flagged_ratio) or (max_run > self.max_run_length)
            )

        # --- 标量写 stats（Analyzer 友好）---
        stats["sudden_change_keep"] = bool(keep)
        stats["sudden_change_flagged_ratio"] = float(flagged_ratio)
        stats["sudden_change_num_flagged"] = int(total_flagged)
        stats["sudden_change_max_run"] = int(max_run)
        stats["sudden_change_max_residual"] = float(max_res)
        stats["sudden_change_max_acc"] = float(max_acc)
        stats["sudden_change_max_jerk"] = float(max_jerk)

        # --- 明细写 meta（JSON 字符串，避免 Arrow 嵌套 schema 冲突）---
        meta = sample.setdefault(Fields.meta, {})
        meta[self.report_field] = json.dumps({
            "keep": bool(keep),
            "strategy": self.exclusion_strategy,
            "num_frames": total_frames,
            "num_flagged": total_flagged,
            "flagged_ratio": float(flagged_ratio),
            "max_run_length": int(max_run),
            "bad_dimensions": sorted(bad_dims),
            "blocks": per_block_reports,
        }, ensure_ascii=False)
        if self.exclusion_strategy == "frame_mask":
            meta[self.mask_field] = json.dumps(combined_masks, ensure_ascii=False)
        elif self.exclusion_strategy == "frame_remove":
            if self.signal_source == "top_level" and len(combined_masks) > 0:
                # 各块共享 T，取按名合并的整体保留掩码（任一块异常即删该帧）
                keep_mask = None
                for m in combined_masks.values():
                    mm = np.asarray(m, dtype=bool)
                    keep_mask = mm if keep_mask is None else (keep_mask & mm)
                if keep_mask is not None and keep_mask.any():
                    self._apply_frame_remove(sample, keep_mask)
            else:
                logger.warning(
                    "frame_remove 目前仅对 signal_source=top_level 自动裁剪；"
                    "其他来源请改用 frame_mask 由下游按 mask 处理。"
                )
        return sample

    def process_single(self, sample):
        stats = sample.get(Fields.stats, {})
        # frame_mask / frame_remove 都保留样本（帧级处理已在 compute_stats 完成）
        if self.exclusion_strategy in ("frame_mask", "frame_remove"):
            return True
        # episode_discard：按 episode 级判据决定保留
        return bool(stats.get("sudden_change_keep", True))
