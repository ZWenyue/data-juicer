# data_juicer/_au/ops/filter/robot_state_action_alignment_filter.py
# -*- coding: utf-8 -*-
"""Qwen-RobotManip Stage 2: State-Action Trend Alignment —— data-juicer 自定义 Filter。

校验一段 episode 内 state 与 action 轨迹的「趋势一致性」这一因果不变量：
正确录制时，action 指令应领先或同步于其引起的 state 变化；时间戳错位或丢包
会破坏该不变量。逐「共享关节维」执行：

    平滑 state / (可选积分)平滑 action
        -> 互相关估计最优时延 L
        -> 时延对齐后取一阶差分
        -> 在「显著变化帧」(|Δ|>eps) 上计算方向一致性 DA = mean(sign(Δs)==sign(Δa))
        -> 任一维 DA < 阈值 则整段 episode 判定为不一致。

参考实现：Galaxea 数据集 clean_stage123.py 中的 `_best_lag` 与 `stage2_check`。
"""

import json

import numpy as np

from data_juicer.ops.base_op import OPERATORS, Filter
from data_juicer.utils.constant import Fields, MetaKeys

OP_NAME = "robot_state_action_alignment_filter"


@OPERATORS.register_module(OP_NAME)
class RobotStateActionAlignmentFilter(Filter):
    """Filter episodes whose state/action trajectories are trend-misaligned.

    逐共享关节维计算方向一致性 DA；任一维 DA 低于阈值即认为 state 与 action
    不一致（时间戳错位 / 丢包 / delta 语义错误等），默认整段 episode 丢弃。
    """

    def __init__(
        self,
        # ---- 信号来源 ----
        signal_source: str = "top_level",
        top_level_state_key: str = "states",
        top_level_action_key: str = "actions",
        hand_action_field: str = MetaKeys.hand_action_tags,
        states_key: str = "states",
        actions_key: str = "actions",
        meta_state_field: str = None,
        meta_action_field: str = None,
        # ---- delta 动作 ----
        action_is_delta: bool = False,
        # ---- 平滑参数 ----
        median_windows: tuple = (3, 5),
        savgol_window: int = 11,
        savgol_polyorder: int = 3,
        # ---- 时延估计 ----
        max_lag: int = 15,
        # ---- 方向一致性阈值 ----
        da_threshold: float = 0.65,
        eps_mode: str = "range_frac",
        eps_frac: float = 0.01,
        eps_abs: float = 1e-6,
        dim_range: list = None,
        min_active_frames: int = 10,
        # ---- 维度选择 ----
        shared_dims: list = None,
        check_dims: dict = None,
        exempt_dims: list = None,
        angular_dims: list = None,
        # ---- episode 级判据 ----
        min_frames: int = 20,
        # ---- 处理策略 ----
        exclusion_strategy: str = "episode_discard",
        report_field: str = "state_action_alignment_report",
        *args,
        **kwargs,
    ):
        """
        :param signal_source: 信号来源，{"top_level","hand_action_tags","meta_field"}。
        :param top_level_state_key/top_level_action_key: top_level 时样本顶层键名。
        :param hand_action_field: hand_action_tags 时 meta 中的字段名。
        :param states_key/actions_key: hand_action_tags 每个 hand dict 内的键名。
        :param meta_state_field/meta_action_field: meta_field 时 meta 下的 (T,D) 字段名。
        :param action_is_delta: 若 action 为增量参数化，比较前先对其做 cumsum 积分还原绝对量。
        :param median_windows: 级联中值滤波窗口序列（自动取奇数并裁剪到不超过 T）。
        :param savgol_window: Savitzky-Golay 窗口（自动取奇数，需 >= polyorder+2 才启用）。
        :param savgol_polyorder: SG 多项式阶数。
        :param max_lag: 互相关搜索的最大时延（帧）。
        :param da_threshold: 方向一致性阈值，论文建议 0.6~0.7；低于则该维被 flag。
        :param eps_mode: 最小变化阈值模式，{"range_frac","abs"}。
        :param eps_frac: range_frac 模式下 eps = max(eps_frac*量程, eps_abs)。
        :param eps_abs: abs 模式下的固定 eps；range_frac 模式下作为下限。
        :param dim_range: 逐维物理量程 (q99-q01) 向量；缺省时用逐 episode 的 1/99 分位跨度估计。
        :param min_active_frames: 少于该活跃帧数则跳过该维（不判失败），避免近零噪声误判。
        :param shared_dims: 参与检测的共享维索引（state 与 action 同索引）；None 时自动推断。
        :param check_dims: {"include":[...]} 与/或 {"exclude":[...]}，进一步约束参与维度。
        :param exempt_dims: 一律跳过的维度索引（夹爪等离散/双峰通道、padding）。
        :param angular_dims: 需先 np.unwrap 的角度维索引（欧拉角），避免 pi/-pi 跳变。
        :param min_frames: 少于该帧数的轨迹视为过短，安全保留、不检测。
        :param exclusion_strategy: {"episode_discard","flag_only"}。flag_only 只标注不丢弃。
        :param report_field: 明细报告写入 meta 的字段名。
        """
        super().__init__(*args, **kwargs)

        if signal_source not in ("top_level", "hand_action_tags", "meta_field"):
            raise ValueError(f"Invalid signal_source: {signal_source}")
        if eps_mode not in ("range_frac", "abs"):
            raise ValueError(f"Invalid eps_mode: {eps_mode}")
        if exclusion_strategy not in ("episode_discard", "flag_only"):
            raise ValueError(f"Invalid exclusion_strategy: {exclusion_strategy}")
        if signal_source == "meta_field" and (meta_state_field is None or meta_action_field is None):
            raise ValueError("meta_field source requires meta_state_field and meta_action_field.")

        self.signal_source = signal_source
        self.top_level_state_key = top_level_state_key
        self.top_level_action_key = top_level_action_key
        self.hand_action_field = hand_action_field
        self.states_key = states_key
        self.actions_key = actions_key
        self.meta_state_field = meta_state_field
        self.meta_action_field = meta_action_field

        self.action_is_delta = bool(action_is_delta)

        self.median_windows = tuple(int(w) for w in median_windows)
        self.savgol_window = int(savgol_window)
        self.savgol_polyorder = int(savgol_polyorder)

        self.max_lag = int(max_lag)

        self.da_threshold = float(da_threshold)
        self.eps_mode = eps_mode
        self.eps_frac = float(eps_frac)
        self.eps_abs = float(eps_abs)
        self.dim_range = np.asarray(dim_range, dtype=np.float64) if dim_range is not None else None
        self.min_active_frames = int(min_active_frames)

        self.shared_dims = list(shared_dims) if shared_dims is not None else None
        self.check_dims = check_dims
        self.exempt_dims = set(exempt_dims) if exempt_dims else set()
        self.angular_dims = set(angular_dims) if angular_dims else set()

        self.min_frames = int(min_frames)

        self.exclusion_strategy = exclusion_strategy
        self.report_field = report_field

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

    def _resolve_dims(self, d_state: int, d_action: int) -> list:
        """计算真正参与检测的共享维索引（state 与 action 同索引）。"""
        d_shared = min(d_state, d_action)
        if self.shared_dims is not None:
            dims = [d for d in self.shared_dims if 0 <= d < d_shared]
        else:
            dims = list(range(d_shared))
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

    def _smooth_1d(self, x: np.ndarray) -> np.ndarray:
        """级联中值 + Savitzky-Golay 平滑（1D），对短信号做保护。"""
        from scipy.ndimage import median_filter
        from scipy.signal import savgol_filter

        y = np.asarray(x, dtype=np.float64).copy()
        T = y.shape[0]
        if T < 5:
            return y
        for w in self.median_windows:
            w = int(w)
            if w % 2 == 0:
                w -= 1
            if 3 <= w <= T:
                y = median_filter(y, size=w, mode="nearest")
        win = min(self.savgol_window, T)
        if win % 2 == 0:
            win -= 1
        if win >= self.savgol_polyorder + 2:
            try:
                y = savgol_filter(y, win, self.savgol_polyorder, mode="interp")
            except Exception:
                pass
        return y

    @staticmethod
    def _best_lag(a: np.ndarray, b: np.ndarray, max_lag: int) -> int:
        """互相关求最优时延 L（帧）：使 b 对齐到 a 的归一化互相关最大。

        L>=0 表示 b 落后于 a（把 b 左移 L 对齐）。与参考实现一致。
        """
        a = a - a.mean()
        b = b - b.mean()
        if np.allclose(a, 0) or np.allclose(b, 0):
            return 0
        best_L, best_c = 0, -np.inf
        for L in range(-max_lag, max_lag + 1):
            if L >= 0:
                aa, bb = a[L:], (b[: len(b) - L] if L > 0 else b)
            else:
                aa, bb = a[:L], b[-L:]
            m = min(len(aa), len(bb))
            if m < 5:
                continue
            c = float(np.dot(aa[:m], bb[:m]))
            denom = np.linalg.norm(aa[:m]) * np.linalg.norm(bb[:m]) + 1e-12
            c /= denom
            if c > best_c:
                best_c, best_L = c, L
        return best_L

    def _eps_for_dim(self, d: int, xs: np.ndarray) -> float:
        """逐维最小变化阈值 eps。"""
        if self.eps_mode == "abs":
            return self.eps_abs
        if self.dim_range is not None and d < len(self.dim_range):
            rng = float(self.dim_range[d])
        else:
            # 缺省用逐 episode 的 1/99 分位跨度作为量程的稳健估计
            rng = float(np.percentile(xs, 99) - np.percentile(xs, 1))
        return max(self.eps_frac * rng, self.eps_abs)

    # ------------------------------------------------------------------ #
    # 信号抽取（配对 state / action）
    # ------------------------------------------------------------------ #
    def _iter_state_action_pairs(self, sample: dict) -> list:
        """返回 [(pair_name, state_raw, action_raw), ...]。"""
        pairs = []
        if self.signal_source == "top_level":
            st = sample.get(self.top_level_state_key)
            ac = sample.get(self.top_level_action_key)
            if st is not None and ac is not None and len(st) > 0 and len(ac) > 0:
                pairs.append(("top_level", st, ac))
        elif self.signal_source == "meta_field":
            meta = sample.get(Fields.meta, {}) or {}
            st = meta.get(self.meta_state_field)
            ac = meta.get(self.meta_action_field)
            if st is not None and ac is not None and len(st) > 0 and len(ac) > 0:
                pairs.append((self.meta_state_field, st, ac))
        else:  # hand_action_tags
            meta = sample.get(Fields.meta, {}) or {}
            clips = meta.get(self.hand_action_field) or []
            for ci, clip in enumerate(clips):
                if not isinstance(clip, dict):
                    continue
                for hand_type, hand in clip.items():
                    if not isinstance(hand, dict):
                        continue
                    st = hand.get(self.states_key)
                    ac = hand.get(self.actions_key)
                    if st is not None and ac is not None and len(st) > 0 and len(ac) > 0:
                        pairs.append((f"clip{ci}.{hand_type}", st, ac))
        return pairs

    # ------------------------------------------------------------------ #
    # 单对 (state, action) 检测
    # ------------------------------------------------------------------ #
    def _check_pair(self, state_raw, action_raw) -> dict:
        state = self._as_2d(state_raw)
        action = self._as_2d(action_raw)
        T = min(state.shape[0], action.shape[0])
        result = {
            "num_frames": int(T),
            "checked_dims": [],
            "dim_da": {},  # {dim: DA}
            "dim_lag": {},  # {dim: L}
            "flagged": {},  # {dim: DA(<thresh)}
            "insufficient_length": bool(T < self.min_frames),
        }
        if T < self.min_frames:
            return result
        state = state[:T]
        action = action[:T]

        dims = self._resolve_dims(state.shape[1], action.shape[1])
        for d in dims:
            xs = state[:, d].astype(np.float64)
            xa = action[:, d].astype(np.float64)
            if d in self.angular_dims:
                xs = np.unwrap(xs)
            xs = self._smooth_1d(xs)
            if self.action_is_delta:
                xa = np.cumsum(xa)
            if d in self.angular_dims and not self.action_is_delta:
                xa = np.unwrap(xa)
            xa = self._smooth_1d(xa)

            L = self._best_lag(xs, xa, self.max_lag)
            if L >= 0:
                sa, aa = xs[L:], (xa[: T - L] if L > 0 else xa)
            else:
                sa, aa = xs[:L], xa[-L:]
            m = min(len(sa), len(aa))
            if m < 5:
                continue

            ds = np.diff(sa[:m])
            da = np.diff(aa[:m])
            eps = self._eps_for_dim(d, xs)
            active = (np.abs(ds) > eps) | (np.abs(da) > eps)
            if int(active.sum()) < self.min_active_frames:
                continue

            agree = np.sign(ds[active]) == np.sign(da[active])
            da_value = float(agree.mean())
            result["checked_dims"].append(int(d))
            result["dim_da"][int(d)] = round(da_value, 4)
            result["dim_lag"][int(d)] = int(L)
            if da_value < self.da_threshold:
                result["flagged"][int(d)] = round(da_value, 4)
        return result

    # ------------------------------------------------------------------ #
    # 两阶段接口
    # ------------------------------------------------------------------ #
    def compute_stats_single(self, sample, context=False):
        stats = sample[Fields.stats]
        pairs = self._iter_state_action_pairs(sample)

        all_da = []
        all_lags = []
        num_flagged = 0
        num_checked = 0
        per_pair_reports = []
        flagged_all = {}

        for name, st, ac in pairs:
            r = self._check_pair(st, ac)
            num_checked += len(r["checked_dims"])
            num_flagged += len(r["flagged"])
            for d, v in r["dim_da"].items():
                all_da.append(v)
            for d, L in r["dim_lag"].items():
                all_lags.append(abs(int(L)))
            for d, v in r["flagged"].items():
                flagged_all[f"{name}.dim{d}"] = v
            per_pair_reports.append(
                {
                    "name": name,
                    "num_frames": r["num_frames"],
                    "checked_dims": r["checked_dims"],
                    "dim_da": r["dim_da"],
                    "dim_lag": r["dim_lag"],
                    "flagged": r["flagged"],
                    "insufficient_length": r["insufficient_length"],
                }
            )

        min_da = float(min(all_da)) if all_da else 1.0
        mean_da = float(np.mean(all_da)) if all_da else 1.0
        max_abs_lag = int(max(all_lags)) if all_lags else 0
        keep = num_flagged == 0

        # --- 标量写 stats（Analyzer 友好）---
        stats["state_action_alignment_keep"] = bool(keep)
        stats["state_action_min_da"] = float(min_da)
        stats["state_action_mean_da"] = float(mean_da)
        stats["state_action_num_flagged_dims"] = int(num_flagged)
        stats["state_action_num_checked_dims"] = int(num_checked)
        stats["state_action_max_abs_lag"] = int(max_abs_lag)

        # --- 明细写 meta（JSON 字符串，规避 Arrow 嵌套 schema 冲突）---
        meta = sample.setdefault(Fields.meta, {})
        meta[self.report_field] = json.dumps(
            {
                "keep": bool(keep),
                "strategy": self.exclusion_strategy,
                "da_threshold": self.da_threshold,
                "num_checked_dims": int(num_checked),
                "num_flagged_dims": int(num_flagged),
                "min_da": float(min_da),
                "mean_da": float(mean_da),
                "max_abs_lag": int(max_abs_lag),
                "flagged": flagged_all,
                "pairs": per_pair_reports,
            },
            ensure_ascii=False,
        )
        return sample

    def process_single(self, sample):
        stats = sample.get(Fields.stats, {})
        if self.exclusion_strategy == "flag_only":
            return True
        # episode_discard：按 episode 级 DA 判据决定保留
        return bool(stats.get("state_action_alignment_keep", True))
