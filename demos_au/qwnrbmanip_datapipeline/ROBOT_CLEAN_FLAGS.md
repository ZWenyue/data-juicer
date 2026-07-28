# `suggested_clean_flags` 参数说明

`analysis.json` 中的 `suggested_clean_flags` 会由 `robot_data_clean.sh` 自动读入，并传给 `run_robot_clean`。它们覆盖对应 `CleanConfig` 字段，再经 `recipe.py` 写入各清洗算子。

脚本用法见 [ROBOT_DATA_SCRIPTS_USAGE.md](./ROBOT_DATA_SCRIPTS_USAGE.md)。

实现入口：`data_juicer/_au/pipeline/robot_clean/analyze.py`（`suggest_numeric` / `suggest_video`）。

## 示例

```json
"suggested_clean_flags": {
  "--s1-max-flagged-ratio": 0.0513,
  "--s1-max-run-length": 10,
  "--s2-da-threshold": 0.6,
  "--s3-alpha": 0.1,
  "--check3-blur-threshold": 1045.1,
  "--check3-blackness-threshold": 10.0
}
```

## 参数如何进入流水线

```text
analysis.json suggested_clean_flags
  → robot_data_clean.sh
  → run_robot_clean CLI
  → CleanConfig
  → recipe.yaml 中各 OP 构造参数
```

## 推荐总体流程

analyze **只读探针**，不改原始数据：

1. 列出全部（或 `--max-episodes` 截断后的）episode parquet。
2. 计算 embodiment 全局分位数 → `percentiles.json`（供 Stage3）。
3. 对每个 episode 跑 S1/S2/S3 的 `compute_stats_single`，收集标量统计。
4. `suggest_numeric` → 推荐 S1/S2/S3 旗标。
5. 从全任务均匀分层抽样视频（默认 32 条、`PROBE_FPS=2`）跑帧质量打分，`suggest_video` → 推荐 Check3 旗标。
6. 写入 `analysis.json` / `threshold_report.md`。

分析器采用保守的 Tukey outer fence，不再预设固定淘汰比例。
`DA_CANDIDATES = (0.60, 0.65, 0.70)` 仅用于报告不同阈值下的敏感度。

---

## Stage1：`--s1-max-flagged-ratio` / `--s1-max-run-length`

对应算子：`robot_sudden_change_filter`。

### 清洗时含义

对 state/action 做残差、加速度、jerk 检测，标出突变帧，再算：

- `flagged_ratio` = 异常帧数 / 总帧数
- `max_run` = 最长连续异常段长度

丢弃判定（`exclusion_strategy=episode_discard`）：

```text
keep = not (flagged_ratio > max_flagged_ratio or max_run > max_run_length)
```

| 参数 | 含义 | 默认 | 调大效果 |
|---|---|---|---|
| `--s1-max-flagged-ratio` | 允许的突变帧最大占比；超过则整集丢弃 | `0.3` | 更松 |
| `--s1-max-run-length` | 允许的最长连续突变段；超过则整集丢弃 | `10` | 更松 |

### 推荐原理

探针收集每个 episode 的 `sudden_change_flagged_ratio`、`sudden_change_max_run`，记为数组 `ratio`、`max_run`。

```text
max_flagged_ratio = max(默认0.3, q75(ratio) + 3·IQR(ratio), max(ratio))
max_run_length    = max(默认10, ceil(q75(max_run) + 3·IQR(max_run)), max(max_run))
```

含义：以当前分析集的完整观测范围作为保守基线；报告仍展示离群分布，但自动建议不会在没有质量标签时删除当前数据。后续超出该基线的新 episode 才会触发门控。

---

## Stage2：`--s2-da-threshold`

对应算子：`robot_state_action_alignment_filter`。

### 清洗时含义

对共享维度估计状态–动作方向一致性（direction agreement, DA）。任一维 `da < da_threshold` 即 flag；`num_flagged_dims > 0` 时整集丢弃。

| 参数 | 含义 | 默认 | 调大效果 |
|---|---|---|---|
| `--s2-da-threshold` | 方向一致性下限；低于则该维不合格 | `0.65` | 更严 |

### 推荐原理

探针收集每个 episode 的 `state_action_min_da`（最差维 DA）。候选阈值用于报告敏感度；推荐阈值采用低侧 outer fence，并限制在 `[0.5, 默认0.65]`：

```text
da_threshold = min(默认0.65, q25(min_da) - 3·IQR(min_da), min(min_da))
```

该策略只会放宽默认阈值，并以当前观测到的最低 DA 为基线，不会为了达到预设淘汰率主动收紧。

---

## Stage3：`--s3-alpha`

对应算子：`robot_extreme_value_filter`。

### 清洗时含义

先用全局分位数构造允许带（夹爪维 `exempt_dims=[7,15]` 豁免）：

\[
[q_{01} - \alpha \cdot IQR,\quad q_{99} + \alpha \cdot IQR],\quad IQR = q_{99}-q_{01}
\]

出界帧记为极值帧。`alpha` 只决定带宽；真正丢 episode 还看 `s3_max_flagged_ratio`（默认 `0.3`，旗标里一般不单独建议）：

```text
extreme_value_keep = (flagged_ratio <= max_flagged_ratio)
```

| 参数 | 含义 | 默认 | 调大效果 |
|---|---|---|---|
| `--s3-alpha` | IQR 带宽倍数；越大允许带越宽 | `0.1` | 更松 |

### 推荐原理

1. 用固定探针带宽 `probe_alpha`（默认 `0.1`，CLI `--probe-alpha`）跑 Stage3，得到每集 `extreme_value_flagged_ratio`。
2. 算全任务平均标记率 `mean_ev = mean(ev)`。
3. 按门槛选离散档：

| 探针下的 `mean_ev` | 推荐 `--s3-alpha` | 含义 |
|---|---|---|
| ≤ 0.15 | `probe_alpha`（默认 0.1） | 标记不多，保持默认窄带 |
| (0.15, 0.25] | `0.2` | 偏高，放宽带宽 |
| > 0.25 | `0.3` | 很高，再放宽 |

若 `mean_ev > 0.5`，额外写 warning（可能有近常数维导致 `q01≈q99`、带宽≈0），推荐值仍为 `0.3`。

注意：

- **不**重估丢弃占比门控，清洗仍用默认 `s3_max_flagged_ratio=0.3`
- 报告里 Stage3 预计丢弃率用的是探针阶段（`probe_alpha` 下）的 `ev`，**不是**用推荐后的新 alpha 再扫一遍

---

## Check3：`--check3-blur-threshold` / `--check3-blackness-threshold`

这两个参数先进 `robot_frame_quality_scorer_mapper`，只决定哪些帧算坏帧，不直接丢 episode：

```text
blackness = mean(gray)
blur     = Laplacian(gray).var()
is_bad   = corrupt or blackness < blackness_threshold or blur < blur_threshold
```

之后由 `robot_video_quality_episode_filter` 按坏帧报告做整集门控（这些门控阈值通常不在 `suggested_clean_flags` 里，用 CleanConfig 默认值）：

| 门控 | 默认 | 含义 |
|---|---|---|
| `check3_max_bad_ratio` | `0.1` | 坏帧占比上限 |
| `check3_min_good_frames` | `20` | 好帧数下限 |
| `check3_max_keyframe_overlap` | 禁用 | 默认仅报告重叠；显式设置后才作为删除门控 |

| 参数 | 含义 | 默认 | 调大效果 |
|---|---|---|---|
| `--check3-blur-threshold` | Laplacian 方差低于此值 → 判模糊 | `1.0` | 更严 |
| `--check3-blackness-threshold` | 灰度均值低于此值 → 判过黑 | `10.0` | 更严 |

### 推荐原理

默认从完整 episode 序列均匀抽取 `PROBE_EPS`（默认 32）条视频、以 `PROBE_FPS`（默认 2）抽帧打分，避免只看数据集前缀。

**模糊阈值**（贴分布低尾，标出“明显比常态糊”的帧）：

```text
blur_thr = max(1.0, q25(blur) - 3·IQR(blur))   # 再 round 到 1 位小数
```

**黑帧阈值**（正常画面通常很亮；只有整体偏暗才下调）：

```text
p01_black = percentile(black, 1)
black_thr = 10.0 if p01_black >= 20.0 else max(1.0, p01_black * 0.5)
```

该阈值只捕获与主体清晰度分布明显分离的低侧离群帧，不再按 p2 固定把一部分健康帧标坏。

报告里还会用建议阈值估算 Check3 丢 episode 比例（抽帧结果按 `original_fps/sampling_fps` 外推全长好帧数；默认关键帧重叠只报告，不参与删除）。

---

## 速查表

| 旗标 | 对应算子 | 清洗时作用 | 推荐怎么算 |
|---|---|---|---|
| `--s1-max-flagged-ratio` | `robot_sudden_change_filter` | `flagged_ratio > thr` → 丢 | `max(0.3, q75+3·IQR, observed_max)` |
| `--s1-max-run-length` | `robot_sudden_change_filter` | `max_run > thr` → 丢 | `max(10, ceil(q75+3·IQR), observed_max)` |
| `--s2-da-threshold` | `robot_state_action_alignment_filter` | 任一维 `da < thr` → 丢 | `min(0.65, q25-3·IQR, observed_min)` |
| `--s3-alpha` | `robot_extreme_value_filter` | 带宽 = `α·IQR` | 按探针 `mean_ev` 选 `{0.1,0.2,0.3}` |
| `--check3-blur-threshold` | `robot_frame_quality_scorer_mapper` | `laplacian_var < thr` → 坏帧 | `max(1, q25-3·IQR)` |
| `--check3-blackness-threshold` | `robot_frame_quality_scorer_mapper` | `mean_gray < thr` → 坏帧 | 默认 10；画面很暗时用 `0.5·p01` |

前四个管数值轨迹是否保留整条 episode；后两个管视频帧如何标坏，再交给 Check3 episode 门控决定去留。
