# data_cur1_2.md：Stage 1 突变检测自定义 Filter 可落地实现方案

> 本文是 [`data_cur1_1.md`](data_cur1_1.md) 第 5 节「推荐方案 A：新增自定义 Filter」的**落地强化版**。目标是把「思路级骨架」升级为「可照抄落地、可测试、可验收、可复用」的完整实现方案。
>
> 对应论文方法：`Stage 1: Sudden Change Detection`，见 [`b/d/QwenRobotmanip/TeX_Source/chapter/data.tex`](TeX_Source/chapter/data.tex) 第 206-211 行。
>
> 交付物边界：本文档内嵌全部落地所需代码（算子源码 / 单元测试 / 验收脚本 / 演示数据 / Recipe / 使用说明），供后续照抄创建实体文件。**本文档本身不创建 `.py`/`.yaml`/`.jsonl` 实体文件，不修改 data-juicer 源码，不运行流水线。**

---

## 目录

- [第 1 章 勘误：`data_cur1_1.md` 第 5 节的问题清单](#第-1-章-勘误data_cur1_1md-第-5-节的问题清单)
- [第 2 章 设计定稿（降低落地风险的关键决策）](#第-2-章-设计定稿降低落地风险的关键决策)
- [第 3 章 算子完整源码 `robot_sudden_change_filter.py`](#第-3-章-算子完整源码-robot_sudden_change_filterpy)
- [第 4 章 单元测试完整代码](#第-4-章-单元测试完整代码)
- [第 5 章 验收脚本、命令与演示数据](#第-5-章-验收脚本命令与演示数据)
- [第 6 章 使用说明与示范](#第-6-章-使用说明与示范)
- [第 7 章 自检清单](#第-7-章-自检清单)

---

## 第 1 章 勘误：`data_cur1_1.md` 第 5 节的问题清单

`data_cur1_1.md` 第 5 节给出了正确的**方向**（继承 `Filter`、两阶段过滤、写 stats、按策略删除），但作为落地文档存在以下问题。下表逐条列出「问题 → 影响 → 本文修正」。

| # | 问题 | 影响 | 本文修正 |
|---|---|---|---|
| 1 | `compute_stats_single` 是空壳：定义了 `_detect_matrix` 却从未调用，没有信号抽取、没有报告组装 | 直接复制无法运行，`process_single` 永远读到空 report → 全部保留 | 第 3 章给出 `compute_stats_single` 完整串联：抽取 → 检测 → 写 stats/meta |
| 2 | 声明的参数（`check_dims`、`residual_thresholds`、`acc_thresholds`、`jerk_thresholds`、`threshold_mode="manual"`、`dataset_policy_key`、`frame_mask`/`frame_remove`）在骨架里全部未实现 | 参数表与实现严重脱节，使用者按参数表配置会静默失效 | 第 3 章「参数 = 实现 = Recipe」三处一一对齐，未实现的参数不出现在参数表 |
| 3 | 通配符路径矛盾：9.1 的 `signal_fields` 用了 `__dj__meta__.hand_action_tags.*.*.states`，但 10.1 的 `get_by_path` 按 `.` 逐级字典索引，遇到 `*` 必抛 `KeyError` | 推荐 Recipe 与解析器互相打架，跑不通 | 改为**显式 `signal_source` 枚举**（`hand_action_tags` / `top_level` / `meta_field`），在算子内部按真实结构遍历，杜绝通配符字符串解析 |
| 4 | stats 存嵌套 dict：5.1 把 `{"sudden_change": {...}}` 写进 `Fields.stats`，同时 5.4 又主张用 `stats_export_path` | `Analyzer`/`OverallAnalysis` 用 pandas `describe` 处理 stats，嵌套 dict 会导致分位数分析报错或无意义 | **stats 只存标量键**（bool/float/int），详细报告与 mask 写入 `Fields.meta`，两者职责分离 |
| 5 | 形状坑：`median_filter(y, size=(win, 1))` 要求 2D 输入，若信号是一维 `(T,)` 会直接报错 | 单维信号（如单个夹爪通道）崩溃 | 第 3 章加 `_as_2d`，统一把 `(T,)` 归一为 `(T, 1)` |
| 6 | 旋转 wrapping 未处理：骨架对所有维度一视同仁做差分 | 欧拉角 π/−π 跳变被误判为突变（假阳性） | 第 3 章加 `angular_dims` 参数，对角度维先 `np.unwrap` 再检测 |
| 7 | 执行器坑：9.1 用 `executor_type: "ray"` + 单文件 `custom_operator_paths` | ray 下自定义算子要走 `py_modules` 打包（见 [`data_juicer/utils/ray_utils.py`](../../../data_juicer/utils/ray_utils.py)），单文件路径易失败 | 首次落地用 `executor_type: default`（本地），ray 仅在附注中说明打包注意点 |
| 8 | 注册方式坑：把算子放进 `dj_custom_ops/` 目录但未说明 `__init__.py` | `load_custom_operators` 对「文件」和「目录」处理不同，目录必须含 `__init__.py`，否则报错 | 第 2 章明确「单文件方式不需要 `__init__.py`」，并给出目录方式的补充说明 |
| 9 | 缺少可运行的完整源码、单元测试、验收脚本、演示数据、使用说明 | 无法验收，无法判断实现是否正确 | 第 3-6 章补齐全部 |

> 结论：`data_cur1_1.md` 第 5 节可作为「设计意图」参考，但**不可直接落地**。本文在其基础上重写为可运行、可测试、可验收的版本。

---

## 第 2 章 设计定稿（降低落地风险的关键决策）

### 2.1 为什么用 `Filter` 两阶段

data-juicer 的 `Filter` 基类把过滤拆成 `compute_stats`（map 阶段，算指标）+ `process`（filter 阶段，判定保留）两步，正好匹配 Stage 1「先算突变指标 → 再决定丢弃」的语义：

```832:864:data_juicer/ops/base_op.py
    def run(self, dataset, *, exporter=None, tracer=None, reduce=True):
        dataset = super(Filter, self).run(dataset)
        new_dataset = dataset.map(
            self.compute_stats,
            num_proc=self.runtime_np(),
            with_rank=self.use_cuda(),
            batch_size=self.batch_size,
            desc=self._name + "_compute_stats",
        )
        if exporter and self.stats_export_path is not None:
            exporter.export_compute_stats(new_dataset, self.stats_export_path)
        if reduce:
            ...
            new_dataset = new_dataset.filter(
                self.process, num_proc=self.runtime_np(), batch_size=self.batch_size, desc=self._name + "_process"
            )
        free_models()
        return new_dataset
```

关键点：`run()` 会自动给数据集补 `Fields.stats` 列（对非 `NON_STATS_FILTERS` 的 Filter），并支持 `stats_export_path` 导出指标——这是我们做阈值审计的基础。

### 2.2 决策 1：stats 标量化，报告进 meta

论文的检测结果既包含「适合做分位数分析的标量」（异常帧比例、最大残差等），也包含「适合审计的明细」（异常帧 id 列表、逐块报告）。为兼容 `Analyzer`（pandas `describe`），**明确分离**：

- 写入 `Fields.stats`（`__dj__stats__`）的只有标量：

| stats 键 | 类型 | 含义 |
|---|---|---|
| `sudden_change_keep` | bool | episode 是否保留 |
| `sudden_change_flagged_ratio` | float | 异常帧占比 |
| `sudden_change_num_flagged` | int | 异常帧总数 |
| `sudden_change_max_run` | int | 最长连续异常帧长度 |
| `sudden_change_max_residual` | float | 最大绝对残差 |
| `sudden_change_max_acc` | float | 最大 \|加速度\| |
| `sudden_change_max_jerk` | float | 最大 \|jerk\| |

- 写入 `Fields.meta`（`__dj__meta__`）的是明细报告 `sudden_change_report` 与可选的 `valid_frame_mask`。

data-juicer 允许 stats 用任意字符串标量键（参考 `specified_numeric_field_filter` 直接用 `sample[Fields.stats][self.field_key] = field_value`）：

```60:69:data_juicer/ops/filter/specified_numeric_field_filter.py
    def compute_stats_single(self, sample):
        # get the value from the original field
        field_value = sample
        for key in self.field_key.split("."):
            assert key in field_value.keys(), "'{}' not in {}".format(key, field_value.keys())
            field_value = field_value[key]
        # copy it into the stats field
        if self.field_key not in sample[Fields.stats]:
            sample[Fields.stats][self.field_key] = field_value
        return sample
```

### 2.3 决策 2：信号抽取用显式枚举，不用通配符

不解析 `*.*` 字符串路径，改为 `signal_source` 枚举，覆盖三种真实情况：

- `hand_action_tags`：H2R 管道产物，真实结构为 `list[clip] → {hand_type: {states, actions, valid_frame_ids, joints_world}}`（见 [`video_hand_motion_smooth_mapper.py`](../../../data_juicer/ops/mapper/video_hand_motion_smooth_mapper.py) 的 `process_single` 遍历方式）。
- `top_level`：数据集顶层就有 `states` / `actions` 二维数组。
- `meta_field`：`__dj__meta__` 下某个自定义字段（如统一 80 维向量）存 `(T, D)` 数组。

```mermaid
flowchart TB
    sample["Sample"] --> src{"signal_source"}
    src -->|hand_action_tags| hat["meta.hand_action_tags<br>list of clip dicts"]
    src -->|top_level| tl["sample.states / sample.actions"]
    src -->|meta_field| mf["meta.custom_field"]
    hat --> blocks["signal blocks: name and T-by-D array"]
    tl --> blocks
    mf --> blocks
    blocks --> detect["per-block detection"]
```

### 2.4 决策 3：执行器与注册方式

- **首次落地固定用本地执行器** `executor_type: default`。自定义算子通过 `custom_operator_paths` 注册，其加载逻辑：

```53:74:data_juicer/config/config.py
def load_custom_operators(paths):
    """Dynamically load custom operator modules or packages in the specified path."""
    for path in paths:
        abs_path = os.path.abspath(path)
        if os.path.isfile(abs_path):
            module_name = _generate_module_name(abs_path)
            if module_name in sys.modules:
                existing_path = sys.modules[module_name].__file__
                raise RuntimeError(
                    f"Module '{module_name}' already loaded from '{existing_path}'. "
                    f"Conflict detected while loading '{abs_path}'."
                )
            try:
                spec = importlib.util.spec_from_file_location(module_name, abs_path)
                ...
                spec.loader.exec_module(module)
```

- **单文件方式**（推荐）：`custom_operator_paths: ["b/d/QwenRobotmanip/dj_custom_ops/robot_sudden_change_filter.py"]`，**不需要 `__init__.py`**。
- **目录方式**：若把多个自定义算子放同一目录注册，则该目录**必须含 `__init__.py`**（否则 `load_custom_operators` 抛 `ValueError`）。
- **ray 附注**：`executor_type: ray` 时自定义算子改由 `runtime_env.py_modules` 分发（见 [`data_juicer/utils/ray_utils.py`](../../../data_juicer/utils/ray_utils.py) L39-45），需要把算子打包成可导入模块；本文不作为首选。

### 2.5 判定逻辑（论文 Stage 1 形式化）

对每个信号块 $\mathbf{X}\in\mathbb{R}^{T\times D}$，逐维计算：

- 平滑趋势（级联中值 + Savitzky-Golay）：$\hat{x}_{:,d}=\mathrm{SG}_{w_s,p}\big(\mathrm{Median}_{w_2}(\mathrm{Median}_{w_1}(x_{:,d}))\big)$
- 残差：$r_{t,d}=|x_{t,d}-\hat{x}_{t,d}|$
- 加速度（二阶差分）：$a_{t,d}=x_{t+1,d}-2x_{t,d}+x_{t-1,d}$
- jerk（三阶差分）：$j_{t,d}=x_{t+2,d}-3x_{t+1,d}+3x_{t,d}-x_{t-1,d}$

联合判定（论文核心：残差超阈值**且**加速度或 jerk 超阈值）：

$$
\mathrm{flag}_{t,d}=\big(r_{t,d}>\tau^{(r)}_{d}\big)\ \wedge\ \big(|a_{t,d}|>\tau^{(a)}_{d}\ \vee\ |j_{t,d}|>\tau^{(j)}_{d}\big)
$$

帧级异常与 episode 级判据：

$$
\mathrm{frameFlag}_{t}=\bigvee_{d}\mathrm{flag}_{t,d},\qquad
\mathrm{drop}=\Big(\tfrac{1}{T}\textstyle\sum_t \mathrm{frameFlag}_t>\rho_{\max}\Big)\ \vee\ \big(\mathrm{maxRun}(\mathrm{frameFlag})>L_{\max}\big)
$$

阈值两种模式：

- `mad`（自动）：$\tau_d=\mathrm{median}(v_{:,d})+\lambda\cdot 1.4826\cdot\mathrm{MAD}(v_{:,d})$，其中 $v$ 为对应的 $r$/$|a|$/$|j|$。
- `manual`（固定）：$\tau^{(r)}=$`residual_threshold`，$\tau^{(a)}=$`acc_threshold`，$\tau^{(j)}=$`jerk_threshold`。

> `1.4826` 是把 MAD 缩放为正态标准差的稳健常数，与 data-juicer 现有实现一致（`limit = median_speed + threshold_mad * mad * 1.4826`，见 `video_hand_motion_smooth_mapper.py` L129）。

---

## 第 3 章 算子完整源码 `robot_sudden_change_filter.py`

### 3.1 类结构（UML）

```mermaid
classDiagram
    class Filter {
        +compute_stats(sample)
        +process(sample)
        +run(dataset)
    }
    class RobotSuddenChangeFilter {
        +signal_source: str
        +threshold_mode: str
        +exclusion_strategy: str
        +compute_stats_single(sample, context)
        +process_single(sample)
        -_iter_signal_blocks(sample)
        -_detect_block(block)
        -_cascaded_smooth(x)
        -_finite_diff(x)
        -_dim_threshold(values, scale, manual)
        -_select_dims(D)
        -_as_2d(x)
        -_max_run_length(mask)
    }
    Filter <|-- RobotSuddenChangeFilter
```

### 3.2 `compute_stats_single` 调用时序

```mermaid
sequenceDiagram
    participant Run as "Filter.run()"
    participant CS as "compute_stats_single"
    participant Iter as "_iter_signal_blocks"
    participant Det as "_detect_block"
    participant PS as "process_single"

    Run->>CS: map over samples
    CS->>Iter: 抽取 (name, T×D array) 列表
    loop 每个信号块
        CS->>Det: 检测突变
        Det->>Det: _as_2d → _cascaded_smooth → _finite_diff → 阈值 → flag
        Det-->>CS: frame_flags, max_res/acc/jerk, bad_dims
    end
    CS->>CS: 汇总标量写 stats, 明细写 meta
    CS-->>Run: sample (含 stats/meta)
    Run->>PS: filter over samples
    PS-->>Run: keep(bool)
```

### 3.3 完整源码

> 落地路径建议：`b/d/QwenRobotmanip/dj_custom_ops/robot_sudden_change_filter.py`（单文件方式，无需 `__init__.py`）。
> 依赖：`numpy`、`scipy`（data-juicer 环境已含）。`scipy` 采用惰性导入，避免无谓的导入开销。

```python
# b/d/QwenRobotmanip/dj_custom_ops/robot_sudden_change_filter.py
# -*- coding: utf-8 -*-
"""Qwen-RobotManip Stage 1: Sudden Change Detection —— data-juicer 自定义 Filter。

对机器人轨迹信号（states/actions 或统一向量）逐维做：
    级联中值 + Savitzky-Golay 平滑 -> 残差 / 二阶差分(acc) / 三阶差分(jerk)
    -> 联合阈值判定异常帧 -> episode 级判据 -> 按策略删除。
"""

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
        :param recompute_actions_on_remove: frame_remove 且 8 维 states/7 维 actions 时重算 delta。
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

        # --- 明细写 meta ---
        meta = sample.setdefault(Fields.meta, {})
        meta[self.report_field] = {
            "keep": bool(keep),
            "strategy": self.exclusion_strategy,
            "num_frames": total_frames,
            "num_flagged": total_flagged,
            "flagged_ratio": float(flagged_ratio),
            "max_run_length": int(max_run),
            "bad_dimensions": sorted(bad_dims),
            "blocks": per_block_reports,
        }
        if self.exclusion_strategy == "frame_mask":
            meta[self.mask_field] = combined_masks
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
```

### 3.4 三种删除策略语义

```mermaid
flowchart LR
    cs["compute_stats:<br>逐块检测 + 写标量 stats + 写 meta 报告"] --> strat{"exclusion_strategy"}
    strat -->|episode_discard| ed["process 返回 keep<br>丢弃整条 episode"]
    strat -->|frame_mask| fm["写 meta.valid_frame_mask<br>process 返回 True 保留样本"]
    strat -->|frame_remove| fr["裁剪 top_level states/actions<br>可选重算 delta；process 返回 True"]
```

> 设计取舍：`frame_remove` 会改变轨迹长度，若与 actions（帧间 delta）配套，必须重算 delta 才能保持一致性（`recompute_actions_on_remove=True` 且为 8 维 states 时自动重算）。由于其对下游影响最大，**默认关闭**、仅 `top_level` 自动支持；`hand_action_tags` 的嵌套结构建议用 `frame_mask` 交下游处理，避免破坏 `valid_frame_ids` 等关联字段。

---

## 第 4 章 单元测试完整代码

### 4.1 测试矩阵

| 场景 | 构造 | 期望 |
|---|---|---|
| 平滑正弦 | `0.1*sin(2πt/20)` | `keep=True`，`num_flagged=0` |
| 缓慢漂移 | `0.01*t` 线性 | `keep=True`，`num_flagged=0` |
| 单帧尖峰 | 平滑基线 + `x[20]+=1` | `keep=False`，`20∈flagged` |
| 阶跃跳变 | `t>=20` 加 `1.0` 持续 | `keep=False`，`num_flagged>=1` |
| 角度 wrapping | 绕 π/−π 卷绕 | `angular_dims` 开 → `keep=True`；关 → `keep=False` |
| 短轨迹 T<4 | 长度 3 | `keep=True`，`insufficient_length=True` |
| frame_mask 策略 | 单帧尖峰 | 样本保留，`mask[20]=False` |
| episode_discard 管道 | good/spike/short 三条 | 剩余 `['good','short']` |

### 4.2 完整测试源码

> 落地路径建议：`b/d/QwenRobotmanip/dj_custom_ops/test_robot_sudden_change_filter.py`（与算子同目录，直接 `import` 同级模块）。
> 运行：`python -m pytest b/d/QwenRobotmanip/dj_custom_ops/test_robot_sudden_change_filter.py -v`

```python
# b/d/QwenRobotmanip/dj_custom_ops/test_robot_sudden_change_filter.py
# -*- coding: utf-8 -*-
import os
import sys
import unittest

import numpy as np

# 让测试能直接 import 同目录下的算子模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from robot_sudden_change_filter import RobotSuddenChangeFilter  # noqa: E402

from data_juicer.core.data import NestedDataset as Dataset  # noqa: E402
from data_juicer.utils.constant import Fields  # noqa: E402
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase  # noqa: E402


def _col(vec):
    """把一维序列变成 (T, 1) 的 list-of-list，模拟单维 states 列。"""
    return np.asarray(vec, dtype=float).reshape(-1, 1).tolist()


# 统一用 manual 阈值，保证断言确定性
MANUAL = dict(
    signal_source="top_level",
    threshold_mode="manual",
    residual_threshold=0.3,
    acc_threshold=0.3,
    jerk_threshold=0.3,
    max_flagged_ratio=0.0,
    max_run_length=0,
    min_frames=4,
)


class RobotSuddenChangeFilterTest(DataJuicerTestCaseBase):

    def _stats(self, op, states):
        sample = {Fields.stats: {}, "states": states}
        sample = op.compute_stats_single(sample)
        return sample

    # ---- 平滑正弦：保留 ----
    def test_smooth_sine_keep(self):
        t = np.arange(40)
        x = 0.1 * np.sin(2 * np.pi * t / 20)
        op = RobotSuddenChangeFilter(**MANUAL)
        s = self._stats(op, _col(x))
        self.assertTrue(s[Fields.stats]["sudden_change_keep"])
        self.assertEqual(s[Fields.stats]["sudden_change_num_flagged"], 0)
        self.assertTrue(op.process_single(s))

    # ---- 缓慢漂移：保留 ----
    def test_slow_drift_keep(self):
        t = np.arange(40)
        x = 0.01 * t
        op = RobotSuddenChangeFilter(**MANUAL)
        s = self._stats(op, _col(x))
        self.assertTrue(s[Fields.stats]["sudden_change_keep"])
        self.assertEqual(s[Fields.stats]["sudden_change_num_flagged"], 0)

    # ---- 单帧尖峰：命中并丢弃 ----
    def test_single_spike_drop(self):
        t = np.arange(40)
        x = 0.1 * np.sin(2 * np.pi * t / 20)
        x[20] += 1.0
        op = RobotSuddenChangeFilter(**MANUAL)
        s = self._stats(op, _col(x))
        self.assertFalse(s[Fields.stats]["sudden_change_keep"])
        flagged = s[Fields.meta]["sudden_change_report"]["blocks"][0]["flagged_frame_ids"]
        self.assertIn(20, flagged)
        self.assertFalse(op.process_single(s))

    # ---- 阶跃跳变：命中边界 ----
    def test_step_change_drop(self):
        t = np.arange(40)
        x = 0.1 * np.sin(2 * np.pi * t / 20) + (t >= 20).astype(float) * 1.0
        op = RobotSuddenChangeFilter(**MANUAL)
        s = self._stats(op, _col(x))
        self.assertFalse(s[Fields.stats]["sudden_change_keep"])
        self.assertGreaterEqual(s[Fields.stats]["sudden_change_num_flagged"], 1)

    # ---- 角度 wrapping：开启 angular_dims 不误杀，关闭则误杀 ----
    def test_angular_wrapping(self):
        t = np.arange(40)
        true = 0.3 * t
        wrapped = ((true + np.pi) % (2 * np.pi)) - np.pi

        op_on = RobotSuddenChangeFilter(angular_dims=[0], **MANUAL)
        s_on = self._stats(op_on, _col(wrapped))
        self.assertTrue(s_on[Fields.stats]["sudden_change_keep"])
        self.assertEqual(s_on[Fields.stats]["sudden_change_num_flagged"], 0)

        op_off = RobotSuddenChangeFilter(**MANUAL)
        s_off = self._stats(op_off, _col(wrapped))
        self.assertFalse(s_off[Fields.stats]["sudden_change_keep"])

    # ---- 短轨迹：安全保留 ----
    def test_short_trajectory_keep(self):
        op = RobotSuddenChangeFilter(**MANUAL)
        s = self._stats(op, _col([0.0, 1.0, 0.0]))
        self.assertTrue(s[Fields.stats]["sudden_change_keep"])
        blk = s[Fields.meta]["sudden_change_report"]["blocks"][0]
        self.assertTrue(blk["insufficient_length"])
        self.assertEqual(s[Fields.stats]["sudden_change_num_flagged"], 0)

    # ---- frame_mask 策略：保留样本 + mask 打点 ----
    def test_frame_mask_strategy(self):
        t = np.arange(40)
        x = 0.1 * np.sin(2 * np.pi * t / 20)
        x[20] += 1.0
        cfg = dict(MANUAL)
        cfg["exclusion_strategy"] = "frame_mask"
        op = RobotSuddenChangeFilter(**cfg)
        s = self._stats(op, _col(x))
        # frame_mask 一律保留样本
        self.assertTrue(op.process_single(s))
        mask = s[Fields.meta]["valid_frame_mask"]["states"]
        self.assertEqual(len(mask), 40)
        self.assertFalse(mask[20])  # 尖峰帧被标记为无效

    # ---- episode_discard 全流程管道：good/spike/short ----
    def test_pipeline_episode_discard(self):
        t = np.arange(40)
        good = 0.1 * np.sin(2 * np.pi * t / 20)
        spike = good.copy()
        spike[20] += 1.0
        ds_list = [
            {"id": "good", "states": _col(good)},
            {"id": "spike", "states": _col(spike)},
            {"id": "short", "states": _col([0.0, 1.0, 0.0])},
        ]
        dataset = Dataset.from_list(ds_list)
        if Fields.stats not in dataset.features:
            dataset = dataset.add_column(name=Fields.stats, column=[{}] * dataset.num_rows)
        op = RobotSuddenChangeFilter(**MANUAL)
        dataset = dataset.map(op.compute_stats)
        dataset = dataset.filter(op.process)
        kept_ids = sorted(dataset.select_columns(["id"]).to_list(), key=lambda r: r["id"])
        self.assertEqual(kept_ids, [{"id": "good"}, {"id": "short"}])


if __name__ == "__main__":
    unittest.main()
```

### 4.3 断言设计说明

- **确定性**：全部用 `threshold_mode="manual"` 固定阈值，避免 MAD 依赖数据分布导致的浮动，保证 CI 可复现。
- **AND 逻辑验证**：`test_single_spike_drop` 证明「残差 ∧（acc ∨ jerk）」能命中瞬时尖峰；`test_slow_drift_keep` 证明缓慢漂移（acc≈0）不会误杀——这正是论文用联合阈值而非单一阈值的目的。
- **鲁棒性验证**：`test_angular_wrapping` 用开/关 `angular_dims` 的对照，证明角度维需要 `np.unwrap`；`test_short_trajectory_keep` 证明短轨迹走安全分支不崩溃。
- **策略验证**：`test_frame_mask_strategy` 证明 `frame_mask` 保留样本、mask 长度 = T 且尖峰帧为 `False`；`test_pipeline_episode_discard` 走 `map→filter` 全链路证明整条 episode 被丢弃。

---

## 第 5 章 验收脚本、命令与演示数据

### 5.1 落地文件清单

```text
b/d/QwenRobotmanip/dj_custom_ops/
├── robot_sudden_change_filter.py         # 第 3 章算子源码
├── test_robot_sudden_change_filter.py    # 第 4 章单元测试
├── demo_stage1.jsonl                     # 本节演示数据（4 条）
├── stage1_accept.yaml                    # 本节验收 Recipe
└── outputs/                              # 运行产物（自动生成）
```

### 5.2 演示数据 `demo_stage1.jsonl`

> 单维 `states`，每帧写作 `[v]`（即 `(T, 1)`）。刻意包含 4 类：平滑、尖峰、缓慢漂移、短轨迹。

```json
{"id": "good", "states": [[0.0],[0.05],[0.1],[0.12],[0.14],[0.15],[0.15],[0.15]]}
{"id": "spike", "states": [[0.0],[0.05],[0.1],[0.15],[1.2],[0.2],[0.22],[0.24]]}
{"id": "drift", "states": [[0.0],[0.02],[0.04],[0.06],[0.08],[0.1]]}
{"id": "short", "states": [[0.0],[0.5],[0.0]]}
```

> 若要用更接近真实的 40 帧正弦/尖峰数据，可用下面脚本生成（可选）：

```python
# gen_demo.py（可选，生成更长的演示数据）
import json, numpy as np
t = np.arange(40)
good = 0.1 * np.sin(2 * np.pi * t / 20)
spike = good.copy(); spike[20] += 1.0
rows = [
    {"id": "good", "states": good.reshape(-1, 1).tolist()},
    {"id": "spike", "states": spike.reshape(-1, 1).tolist()},
    {"id": "short", "states": [[0.0], [0.5], [0.0]]},
]
with open("demo_stage1.jsonl", "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
```

### 5.3 验收 Recipe `stage1_accept.yaml`

```yaml
# b/d/QwenRobotmanip/dj_custom_ops/stage1_accept.yaml
project_name: 'stage1-accept'
dataset_path: 'b/d/QwenRobotmanip/dj_custom_ops/demo_stage1.jsonl'
export_path: 'b/d/QwenRobotmanip/dj_custom_ops/outputs/stage1_result.jsonl'
np: 1
executor_type: default
keep_stats_in_res_ds: true

# 单文件方式注册自定义算子（无需 __init__.py）
custom_operator_paths:
  - 'b/d/QwenRobotmanip/dj_custom_ops/robot_sudden_change_filter.py'

process:
  - robot_sudden_change_filter:
      signal_source: 'top_level'
      top_level_state_key: 'states'
      threshold_mode: 'manual'      # 验收用固定阈值，结果可复现
      residual_threshold: 0.3
      acc_threshold: 0.3
      jerk_threshold: 0.3
      max_flagged_ratio: 0.0
      max_run_length: 0
      min_frames: 4
      exclusion_strategy: 'episode_discard'
      stats_export_path: 'b/d/QwenRobotmanip/dj_custom_ops/outputs/stage1_stats.jsonl'
```

> 说明：`stats_export_path` 是 `Filter` 基类支持的每算子参数（见 [`data_juicer/ops/base_op.py`](../../../data_juicer/ops/base_op.py) L745、L841-842），会把每条样本的标量 stats 导出，便于阈值审计。

### 5.4 验收命令

```bash
# 1) 单元测试（逻辑验收）
python -m pytest b/d/QwenRobotmanip/dj_custom_ops/test_robot_sudden_change_filter.py -v

# 2) 端到端处理（管道验收）
dj-process --config b/d/QwenRobotmanip/dj_custom_ops/stage1_accept.yaml

# 3) 阈值校准分析（mad 模式调参用，可选）
#    把 recipe 的 threshold_mode 改为 mad 后，用 dj-analyze 查看分位数分布
dj-analyze --config b/d/QwenRobotmanip/dj_custom_ops/stage1_accept.yaml
```

### 5.5 通过标准

| 检查项 | 期望结果 |
|---|---|
| 单元测试 | 8 个用例全部 PASS |
| 输入条数 | `demo_stage1.jsonl` 共 4 条 |
| 输出 `stage1_result.jsonl` 条数 | **3 条**（剔除 `spike`） |
| 保留的 id | `good`、`drift`、`short` |
| `stage1_stats.jsonl` 字段 | 每行含 7 个标量：`sudden_change_keep/flagged_ratio/num_flagged/max_run/max_residual/max_acc/max_jerk` |
| `spike` 行标量 | `sudden_change_keep=false` 且 `sudden_change_num_flagged>=1` |
| `good`/`drift`/`short` 行标量 | `sudden_change_keep=true` 且 `sudden_change_num_flagged=0` |

### 5.6 frame_mask 变体验收（不丢样本，只打掩码）

把 Recipe 中 `exclusion_strategy` 改为 `frame_mask`，重跑后：

| 检查项 | 期望结果 |
|---|---|
| 输出条数 | **4 条**（全部保留） |
| `spike` 行 meta | `valid_frame_mask.states` 长度 = T，且异常帧位为 `false` |
| 其余行 meta | 掩码全为 `true` |

> 提示：查看 meta 需保留 meta 列。可在 result jsonl 中检查 `__dj__meta__.valid_frame_mask` 与 `__dj__meta__.sudden_change_report`。

---

## 第 6 章 使用说明与示范

### 6.1 参数速查表

> 下表与第 3 章 `__init__` 签名**逐一对应**（参数=实现=Recipe 三处一致）。

| 参数 | 类型 | 默认 | 说明 / 取值建议 |
|---|---|---|---|
| `signal_source` | str | `hand_action_tags` | `hand_action_tags`(H2R 产物) / `top_level`(顶层 states/actions) / `meta_field`(meta 下自定义向量) |
| `hand_action_field` | str | `MetaKeys.hand_action_tags` | `hand_action_tags` 来源时的 meta 字段名 |
| `states_key` / `actions_key` | str | `states` / `actions` | 每个 hand dict 内的键名 |
| `top_level_state_key` / `top_level_action_key` | str | `states` / `actions` | `top_level` 来源时顶层键名 |
| `meta_signal_field` | str | `None` | `meta_field` 来源时的 meta 键名 |
| `median_windows` | tuple | `(3, 5)` | 级联中值窗口，自动取奇数并裁剪到 ≤T |
| `savgol_window` | int | `11` | SG 窗口；自动取奇数，需 ≥`polyorder+2` 才启用 |
| `savgol_polyorder` | int | `3` | SG 多项式阶数 |
| `threshold_mode` | str | `mad` | `mad`(自适应，推荐生产) / `manual`(固定，推荐验收) |
| `mad_scale_residual/acc/jerk` | float | `6.0` | MAD 缩放 λ；越大越宽松（越少命中） |
| `residual_threshold/acc_threshold/jerk_threshold` | float | `None` | manual 模式必填 |
| `check_dims` | dict | `None` | `{"include":[...]}` 且/或 `{"exclude":[...]}` |
| `exempt_dims` | list | `None` | 一律跳过的维度（夹爪/padding 等离散通道） |
| `angular_dims` | list | `None` | 需先 `np.unwrap` 的角度维（欧拉角） |
| `max_flagged_ratio` | float | `0.0` | episode 允许异常帧占比上限；>此值丢弃 |
| `max_run_length` | int | `0` | episode 允许最长连续异常段；>此值丢弃 |
| `min_frames` | int | `4` | 少于该帧数安全保留、不检测 |
| `exclusion_strategy` | str | `episode_discard` | `episode_discard` / `frame_mask` / `frame_remove` |
| `mask_field` | str | `valid_frame_mask` | frame_mask 掩码写入 meta 的字段名 |
| `report_field` | str | `sudden_change_report` | 明细报告写入 meta 的字段名 |
| `recompute_actions_on_remove` | bool | `False` | frame_remove 且 8 维 states 时重算 7 维 delta |

> 常用组合：`min_frames=4`、`savgol_window=11`、`savgol_polyorder=3`；生产用 `threshold_mode=mad`、λ 从 6.0 起调；验收/回归用 `threshold_mode=manual` 保证复现。

### 6.2 三种接法示范

#### 接法 A：H2R 平滑之后（推荐主链路）

Stage 1 位于人手→机器人（H2R）重定向与平滑之后，作用于 `hand_action_tags`：

```yaml
process:
  - video_hand_motion_smooth_mapper: {}        # 已有：SG + MAD 平滑
  - robot_sudden_change_filter:
      signal_source: 'hand_action_tags'
      states_key: 'states'
      actions_key: 'actions'
      threshold_mode: 'mad'
      mad_scale_residual: 6.0
      mad_scale_acc: 6.0
      mad_scale_jerk: 6.0
      angular_dims: [3, 4, 5]                   # roll/pitch/yaw
      exempt_dims: [6, 7]                       # padding / gripper
      exclusion_strategy: 'frame_mask'         # 嵌套结构建议只打掩码
```

#### 接法 B：统一向量之后

若已把多源信号拼成统一 `(T, D)` 向量并存于 meta：

```yaml
process:
  - robot_sudden_change_filter:
      signal_source: 'meta_field'
      meta_signal_field: 'unified_traj'
      threshold_mode: 'mad'
      check_dims: {"exclude": [78, 79]}         # 排除非连续维
      exclusion_strategy: 'episode_discard'
```

#### 接法 C：快速验证（不写算子，先看效果）

用现有算子快速摸底：`python_file_mapper` 跑一个只写标量到字段的函数，再用 `general_field_filter` 按布尔条件过滤。适合上算子前的可行性验证：

```yaml
process:
  - python_file_mapper:
      file_path: 'b/d/QwenRobotmanip/dj_custom_ops/quick_probe.py'
      function_name: 'flag_sudden_change'      # 返回 sample，写 sample['sc_keep']=bool
  - general_field_filter:
      filter_condition: 'sc_keep == True'
```

> 对照关系：接法 C 是「探针」，逻辑简单、无 stats/meta 规范；接法 A/B 是「正式算子」，标量进 stats（可 `dj-analyze`）、明细进 meta（可审计）。验证通过后应切换到正式算子。

### 6.3 调参与人工审计闭环

```mermaid
flowchart TD
    start["准备样本子集"] --> analyze["dj-analyze:<br>看 residual/acc/jerk 分位数分布"]
    analyze --> setlam["设 threshold_mode=mad, 选 λ (从 6.0 起)"]
    setlam --> run["dj-process 跑子集"]
    run --> audit["抽查 meta.sudden_change_report<br>看 flagged_frame_ids 是否合理"]
    audit --> ok{"命中是否符合预期?"}
    ok -->|漏检| lower["调小 λ / 阈值"]
    ok -->|误杀| raise["调大 λ / 或用 angular_dims、exempt_dims 豁免"]
    lower --> run
    raise --> run
    ok -->|合适| freeze["固化参数, 全量处理"]
```

### 6.4 常见报错排查

| 现象 | 原因 | 解决 |
|---|---|---|
| `KeyError: '__dj__stats__'` | 直接调 `compute_stats_single` 未给 stats 键 | 传入 `{Fields.stats: {}, ...}`；走 `dj-process` 会自动补列 |
| 算子未注册 / `Unknown OP` | `custom_operator_paths` 未配或路径错 | 用绝对/相对正确路径指向 `.py`；单文件无需 `__init__.py` |
| `Package directory ... must contain __init__.py` | 用了目录方式注册但缺 `__init__.py` | 目录方式补 `__init__.py`，或改用单文件方式 |
| ray 下自定义算子找不到 | ray 走 `py_modules` 分发 | 首选 `executor_type: default`；确需 ray 时把算子打包为模块并配 `py_modules`（见 `ray_utils.py` L39-45） |
| `dj-analyze` 报错/分位数无意义 | stats 存了嵌套 dict | 本算子已只存标量；勿把 dict 塞进 `Fields.stats` |
| 角度维大量误杀 | 未处理 π/−π 卷绕 | 配 `angular_dims=[...]` 对角度维 `np.unwrap` |
| 短轨迹被判异常 | `min_frames` 过小 | 保持默认 `min_frames=4`（差分至少需 4 帧）或调大 |
| `median_filter` 维度报错 | 一维信号未升维 | 本算子 `_as_2d` 已修正；自定义扩展时注意保持 `(T, D)` |

---

## 第 7 章 自检清单

### 7.1 API 与行号引用一致性（已逐条核对真实源码）

| 引用 | 位置 | 核对结论 |
|---|---|---|
| `Filter.run()` 自动 map(compute_stats)+filter(process)，支持 `stats_export_path` | `data_juicer/ops/base_op.py` L832-864 | ✓ 一致 |
| `self.stats_export_path = kwargs.get(...)` | `data_juicer/ops/base_op.py` L745 | ✓ 一致 |
| stats 可用任意字符串标量键 | `data_juicer/ops/filter/specified_numeric_field_filter.py` L60-69 | ✓ 一致 |
| `load_custom_operators`：文件 vs 目录（目录需 `__init__.py`） | `data_juicer/config/config.py` L53-78 | ✓ 一致 |
| ray 下 `custom_operator_paths → py_modules` | `data_juicer/utils/ray_utils.py` L39-45 | ✓ 一致 |
| MAD→σ 缩放常数 `1.4826` | `data_juicer/ops/mapper/video_hand_motion_smooth_mapper.py` L129 | ✓ 一致 |
| `_recompute_actions`（8 维 states→7 维 delta） | `data_juicer/ops/mapper/video_hand_motion_smooth_mapper.py` L14-39 | ✓ 逻辑对齐 |
| `MetaKeys.hand_action_tags` | `data_juicer/utils/constant.py` L104 | ✓ 一致 |
| `python_file_mapper(file_path, function_name)` | `data_juicer/ops/mapper/python_file_mapper.py` L24 | ✓ 一致 |
| `general_field_filter(filter_condition=...)` | `tests/ops/filter/test_general_field_filter.py` L29 | ✓ 一致 |
| 测试范式 `add_column(Fields.stats)`→`map(compute_stats)`→`filter(process)` | `tests/ops/filter/test_general_field_filter.py` L12-19 | ✓ 一致 |
| `executor_type` / `custom_operator_paths` 全局键 | `data_juicer/config/config_all.yaml` L82、L101 | ✓ 一致 |

### 7.2 参数 = 实现 = Recipe 三处一致性

- 第 6.1 参数表的每个参数都能在第 3 章 `__init__` 签名中找到，无「声明未实现」项（修正了 `data_cur1_1.md` 问题 #2）。
- 第 5.3 / 6.2 的 Recipe 只使用参数表中存在的键，无通配符路径（修正了问题 #3）。
- `stats` 只写 7 个标量键，`meta` 写报告/掩码（修正了问题 #4）。

### 7.3 数值逻辑自检

- 判定式 $\mathrm{flag}=(r>\tau_r)\wedge(|a|>\tau_a\vee|j|>\tau_j)$ 在 `_detect_block` 中实现为 `(residual > tr) & ((abs_acc > ta) | (abs_jerk > tj))`，与第 2.5 节 LaTeX 一致。
- 单帧尖峰经级联中值被移除 → 残差大且 acc/jerk 大 → 命中；缓慢漂移 acc≈0 → 不命中：与第 4.3 断言解释自洽。
- `_finite_diff` 对齐到长度 T（边界补 0），`_max_run_length` 与 `flagged_ratio` 双判据与第 2.5 episode 级公式一致。

### 7.4 渲染自检

- 全文 mermaid 图节点均用 camelCase 标识符、含特殊字符的标签加引号，`flowchart`/`sequenceDiagram`/`classDiagram` 语法合规。
- 数学用 `$...$` 与 `$$...$$`；参数表中的绝对值以 `\|` 转义，避免破坏表格。
- 现有代码用 `起始行:结束行:文件路径` 引用块，新代码用带语言标签围栏块。

### 7.5 落地检查清单（照此即可上线）

- [ ] 按第 3 章创建 `robot_sudden_change_filter.py`
- [ ] 按第 4 章创建 `test_robot_sudden_change_filter.py`，`pytest` 全绿
- [ ] 按第 5.2 创建 `demo_stage1.jsonl`
- [ ] 按第 5.3 创建 `stage1_accept.yaml`
- [ ] 跑 `dj-process`，核对第 5.5 通过标准（4 进 3 出，剔除 spike）
- [ ] （生产）切 `threshold_mode: mad`，用第 6.3 闭环调 λ 并人工抽查 `meta.sudden_change_report`

---

## 附：与 `data_cur1_1.md` 的关系

`data_cur1_1.md` 提供方向与背景（论文形式化、能力映射、方案选型）；本文（`data_cur1_2.md`）是其第 5 节的**可落地重写版**，修正了 9 类问题并补齐源码、测试、验收、演示、使用说明。两者配合阅读：先读 `data_cur1_1.md` 建立全局认知，再照 `data_cur1_2.md` 落地实现。
