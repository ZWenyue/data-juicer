# Galaxea Open-World 全量数据质量分析报告

> **分析运行**: `20260713_075405_fbbca6`
> **数据规模**: 22,626 条 episode（Galaxea R1 Lite 机体，16 维关节空间）
> **分析工具**: data-juicer `dj-analyze`，配置文件 `analyze_qwenrobomanip_full.yaml`
> **分析日期**: 2026-07-13

---

## 1. 分析流水线概述

本次分析对 Galaxea Open-World 数据集的全部 22,626 条 episode 执行三级级联质量检测。三级 Filter 均以**非丢弃策略**运行（`frame_mask` 或 `flag_only`），只计算统计量和标记帧，不实际过滤任何 episode，目的是全量摸底数据质量分布，为后续构建 clean recipe 提供阈值选择依据。

```mermaid
flowchart LR
    A[原始 Parquet] --> B[robot_lerobot_parquet_loader_mapper<br/>加载 states/actions]
    B --> S1[Stage 1: 突变检测<br/>robot_sudden_change_filter<br/>策略: frame_mask]
    S1 --> S2[Stage 2: State-Action 对齐<br/>robot_state_action_alignment_filter<br/>策略: flag_only]
    S2 --> S3[Stage 3: 极值过滤<br/>robot_extreme_value_filter<br/>策略: frame_mask]
    S3 --> OUT[overall.md / overall.csv<br/>+ 逐列直方图 & 箱线图<br/>+ Pearson 相关矩阵]
```

代码链路：`dj-analyze` → `Analyzer.run()` → 逐 OP `compute_stats_single()` 写入 `Fields.stats` → `OverallAnalysis.analyze()` 对全列 `pandas.describe()` → 输出 `overall.md` + `overall.csv`，`ColumnWiseAnalysis` 输出逐列 hist/box PNG，`CorrelationAnalysis` 输出 Pearson 热力图。

---

## 2. 运行超参汇总

### 2.1 Stage 1 — `robot_sudden_change_filter`（突变检测）

| 参数 | 值 | 说明 |
|------|-----|------|
| `threshold_mode` | `mad` | 基于中位绝对偏差(MAD)自适应计算阈值 |
| `mad_scale_residual` | 6.0 | 残差阈值 = median + 6.0 × 1.4826 × MAD |
| `mad_scale_acc` | 6.0 | 加速度阈值同上 |
| `mad_scale_jerk` | 6.0 | Jerk 阈值同上 |
| `max_flagged_ratio` | 0.3 | 异常帧占比 > 30% 则标记 episode |
| `max_run_length` | 10 | 最长连续异常段 > 10 帧则标记 |
| `min_frames` | 30 | 少于 30 帧的轨迹跳过检测 |
| `exclusion_strategy` | `frame_mask` | 只标记帧掩码，不丢弃 episode |

### 2.2 Stage 2 — `robot_state_action_alignment_filter`（趋势一致性）

| 参数 | 值 | 说明 |
|------|-----|------|
| `shared_dims` | [0,1,2,3,4,5,6,8,9,10,11,12,13,14] | 参与检测的 14 个共享维度（跳过 dim 7,15 即夹爪） |
| `action_is_delta` | false | action 为绝对量，不做 cumsum 积分 |
| `max_lag` | 15 | 互相关搜索最大时延 ±15 帧 |
| `da_threshold` | 0.65 | 方向一致性阈值 |
| `eps_mode` | `range_frac` | 最小变化阈值 = eps_frac × 量程 |
| `eps_frac` | 0.01 | 量程 1% 以下视为静止 |
| `min_active_frames` | 10 | 活跃帧不足的维度跳过 |
| `min_frames` | 20 | 少于 20 帧的轨迹跳过 |
| `exclusion_strategy` | `flag_only` | 只标注不丢弃 |

### 2.3 Stage 3 — `robot_extreme_value_filter`（极值过滤）

| 参数 | 值 | 说明 |
|------|-----|------|
| `percentile_source` | `stats_json` | 从外部 JSON 加载全局分位数 |
| `embodiment` | `galaxea_r1_lite` | 机体型号 |
| `alpha` | 0.1 | 边界扩展系数：lo = q01 − 0.1×IQR, hi = q99 + 0.1×IQR |
| `exempt_dims` | [7, 15] | 夹爪维度豁免（双峰分布不适合分位数方法） |
| `exclusion_strategy` | `frame_mask` | 只标记帧掩码，不丢弃 episode |

---

## 3. 总体统计表（overall.md）

| 指标 | count | mean | std | min | 25% | 50% | 75% | max |
|------|-------|------|-----|-----|-----|-----|-----|-----|
| **sudden_change_flagged_ratio** | 22626 | 0.633 | 0.176 | 0 | 0.495 | 0.627 | 0.780 | 0.994 |
| **sudden_change_num_flagged** | 22626 | 1760 | 1412 | 0 | 686 | 1459 | 2375 | 18203 |
| **sudden_change_max_run** | 22626 | 204 | 212 | 0 | 49 | 123 | 302 | 2382 |
| **sudden_change_max_residual** | 22626 | 38.0 | 9.5 | 0 | 39.6 | 39.6 | 39.6 | 108.4 |
| **sudden_change_max_acc** | 22626 | 95.3 | 22.1 | 0 | 100 | 100 | 100 | 201.4 |
| **sudden_change_max_jerk** | 22626 | 190.3 | 43.6 | 0 | 200 | 200 | 200 | 379.7 |
| **state_action_min_da** | 22626 | 0.507 | 0.469 | 0 | 0 | 0.852 | 0.973 | 1.0 |
| **state_action_mean_da** | 22626 | 0.798 | 0.216 | 0.001 | 0.612 | 0.962 | 0.994 | 1.0 |
| **state_action_num_flagged_dims** | 22626 | 2.56 | 3.15 | 0 | 0 | 0 | 5 | 13 |
| **state_action_num_checked_dims** | 22626 | 11.5 | 0.9 | 0 | 11 | 12 | 12 | 14 |
| **state_action_max_abs_lag** | 22626 | 6.57 | 3.83 | 0 | 4 | 5 | 7 | 15 |
| **extreme_value_flagged_frames** | 22626 | 96.8 | 349.2 | 0 | 0 | 0 | 83 | 6132 |
| **extreme_value_flagged_ratio** | 22626 | 0.056 | 0.122 | 0 | 0 | 0 | 0.061 | 1.0 |
| **extreme_value_num_check_dims** | 22626 | 28 | 0 | 28 | 28 | 28 | 28 | 28 |
| **extreme_value_num_frames** | 22626 | 1291 | 878 | 12 | 678 | 1112 | 1664 | 11363 |

布尔判定列：

| 指标 | keep=True (通过) | keep=False (不通过) |
|------|-----------------|-------------------|
| **sudden_change_keep** | 27 (0.12%) | 22,599 (99.88%) |
| **state_action_alignment_keep** | 11,662 (51.5%) | 10,964 (48.5%) |
| **extreme_value_keep** | 22,626 (100%) | 0 (0%) |

---

## 4. 逐 Stage 分布分析

### 4.1 Stage 1: 突变检测

#### 4.1.1 flagged_ratio 分布

**直方图特征**：呈单峰偏右分布，峰值在 0.45–0.55 附近，整体分布偏高。

- 中位数 0.627 意味着**超过一半的 episode 有 62.7% 以上的帧被标记为突变异常**
- 25% 分位数也高达 0.495，说明即使是相对"干净"的 episode，也有近一半帧被标记
- 极少数 episode 的 flagged_ratio 接近 0（约占 < 1%），这些是真正平滑的轨迹

**箱线图特征**：箱体集中在 0.50–0.78，中位数线约 0.63。下方有极少量离群点（接近 0 的 episode）。

**解读**：当前 MAD scale=6.0 的阈值设定下，突变检测对这批数据**非常敏感**——绝大多数 episode 的大部分帧都触发了判定公式 `(residual > τ_r) AND (|acc| > τ_a OR |jerk| > τ_j)`。这暗示以下可能：

1. 数据本身的关节空间轨迹不够平滑（采样频率相对运动速度偏低，产生大量"锯齿"）
2. MAD scale=6.0 对这种数据特性可能仍然偏紧
3. 需要结合具体任务（如 push door vs. connect cable）分任务分析，不同操作难度的轨迹平滑度差异可能很大

#### 4.1.2 max_run 分布

**直方图特征**：典型的右偏分布（power-law 型长尾），绝大多数 episode 的 max_run 集中在 0–200 帧，但有显著的长尾延伸到 2000+ 帧。

- 中位数 123 帧表示一半的 episode 存在至少 123 帧的连续异常段
- 配置中 `max_run_length=10` 意味着绝大多数 episode 都超过了该阈值

**箱线图特征**：箱体 49–302，大量离群点散布在 700–2400 范围，呈密集的上方异常值带。

#### 4.1.3 max_residual / max_acc / max_jerk 分布

三个指标的直方图呈现**极其相似的形态**：一个极高的尖峰 + 一个微小的近零峰。

- **max_residual**：绝大多数 episode 集中在 ≈39.6（约 21000+ 个），近零的约 1000 个，少数离群到 60–108。p25=p50=p75=39.627 表明超过 75% 的 episode 的 max_residual 几乎完全相同
- **max_acc**：类似，峰值在 ≈100（p25=p50=p75=100）
- **max_jerk**：类似，峰值在 ≈200（p25=p50=p75=200）

**解读**：这三个极值指标的分布极度集中意味着它们主要由**某个固定的维度上的固定信号特征**主导——很可能是某个关节维度（如机械臂末端或某个旋转关节）存在量化台阶或固定幅值的控制指令跳变。这种"固定天花板"效应使得这些 max 指标的区分度很低——无论 episode 好坏，max 值都差不多。建议后续增加**逐维分解分析**，找出是哪些维度在主导这些峰值。

#### 4.1.4 keep 分布

直方图只有一根条：99.88% 的 episode 被标记为 keep=False。在当前 `max_flagged_ratio=0.3, max_run_length=10` 的判据下，几乎所有 episode 都不合格。这并非意味着数据全部不可用——实际排除策略是 `frame_mask`，episode 级 keep 标志仅供参考，后续 clean 时将以**帧级掩码**为准。

---

### 4.2 Stage 2: State-Action 趋势一致性

#### 4.2.1 min_da 分布

**直方图特征**：非常典型的**双峰分布（bimodal）**。

- **左峰**（DA ≈ 0）：约 8000–9000 个 episode，min_da 接近 0。这些 episode 中至少有一个关节维度的 state 与 action 趋势**完全不一致**（方向一致性接近随机）
- **右峰**（DA ≈ 0.9–1.0）：约 2000–3000 个 episode，所有维度的 DA 都很高

**箱线图特征**：箱体从 0 延伸到 0.97，中位数 ≈ 0.85。巨大的箱体跨度反映了双峰分布的特性。

**解读**：min_da 的双峰分布是本次分析中**最重要的发现之一**。它表明数据集在 state-action 一致性方面有**清晰的二分**：约一半 episode 的 state 与 action 在所有维度上高度一致（min_da > 0.65），另一半至少有一个维度严重不一致。这种双峰结构说明设置 `da_threshold` 在 0.6–0.7 之间可以有效地切分好/坏 episode，而且阈值在该范围内的具体取值对结果影响很小（threshold_report 显示 0.60/0.65/0.70 的丢弃率分别为 48.3%/48.5%/48.6%，差异 < 0.3%）。

#### 4.2.2 mean_da 分布

**直方图特征**：同样右偏，但右峰更为突出。大量 episode 集中在 0.95–1.0（约 6000 个），在 0.55–0.75 有一个次峰（约 200–500 个/bin），长左尾延伸到接近 0。

**箱线图特征**：中位数 ≈ 0.96，Q1 ≈ 0.61，下须延伸到约 0.04，底部有一簇离群点聚集在 0 附近。

**解读**：mean_da 的分布比 min_da 更乐观——大部分 episode 的**平均**方向一致性很高（p50=0.962），说明不一致的通常只是个别维度。但次峰在 0.55–0.75 提示有一部分 episode 存在**多维度中等程度的不一致**，不是简单的单维度故障。

#### 4.2.3 num_flagged_dims 分布

**直方图特征**：离散分布，呈现显著的双峰结构：

- **主峰**在 0（≈ 11,600 个 episode，无维度不通过）
- **次峰群**在 4–6 维（各 1,200–3,500 个），表明不一致通常涉及 4–6 个关节
- 在 12 维处有一个小峰（≈ 900 个），这些是几乎所有维度都不一致的 episode

这个分布形态与 min_da 的双峰一致：要么全好（0 flagged），要么大面积坏（4–6 维或 12 维 flagged）。

#### 4.2.4 max_abs_lag 分布

**直方图特征**：离散整数值分布，有两个显著峰：

- **主峰**在 5 帧（≈ 7,800 个 episode）
- **次峰**在 4 帧（≈ 5,000 个）
- 在 15 帧处有明显的**边界堆积**（≈ 3,000 个），这是 `max_lag=15` 的搜索上界

**解读**：4–5 帧的典型时延反映了数据采集系统中 state 反馈相对 action 指令的**固有延迟**——以常见的 30Hz 采样率估算，4–5 帧对应约 130–170ms 的控制延迟，这是合理的。但 15 帧处的边界堆积（约 13%）值得关注——这些 episode 的实际时延可能超过了搜索窗口，建议后续尝试增大 `max_lag` 验证。

---

### 4.3 Stage 3: 极值过滤

#### 4.3.1 flagged_ratio 分布

**直方图特征**：极度右偏的零膨胀分布。约 12,000+ 个 episode 的 flagged_ratio 为 0（完全无极值帧），然后迅速衰减，少数 episode 的 ratio 达到 0.2–0.4，极个别达到 1.0。

**箱线图特征**：箱体完全压缩在 0–0.06 范围，中位数在 0。上须延伸到约 0.15，之上密布大量离群点直到 1.0。

**解读**：在 α=0.1 的宽松设定下，大部分 episode 的关节值都在分位数边界内（p50=0, p75=0.061）。这说明 Galaxea R1 Lite 的关节空间数据在值域层面总体正常。但长尾中的离群 episode（flagged_ratio > 0.3，约 5% 左右）可能存在传感器漂移、标定偏差或异常操作，值得进一步逐维排查。

#### 4.3.2 flagged_frames 分布

**直方图特征**：类似的零膨胀右偏分布。约 15,000 个 episode 无极值帧，然后逐渐衰减，长尾延伸到 6,000+ 帧。

#### 4.3.3 num_frames（episode 长度）分布

**直方图特征**：右偏分布，双峰结构——在 300–500 帧和 1200–1400 帧各有一个峰。中位数 1,112 帧，跨度从 12 帧到 11,363 帧。

**解读**：episode 长度的双峰可能反映了不同采集任务的特性差异（短任务 vs. 长任务）。最短的 12 帧 episode 接近检测下限（min_frames=4/20/30），可能是中断录制或错误标注。

#### 4.3.4 num_check_dims 分布

全部为 28（16 维 states 去掉 dim7 = 15 维 + 16 维 actions 去掉 dim15 = 15 维 → 但实际 14+14=28），标准差 = 0，表明所有 episode 的维度布局完全一致。

#### 4.3.5 keep 分布

全部为 True（22,626/22,626）。因为策略为 `frame_mask`，极值过滤只标记帧掩码，episode 级始终保留。

---

## 5. 跨 Stage 相关性分析（Pearson 相关矩阵）

从相关矩阵热力图中可以读出以下关键关联：

### 5.1 Stage 1 内部

| 指标对 | 相关系数 | 解读 |
|--------|---------|------|
| max_residual ↔ max_acc | **0.96** | 残差和加速度高度共线——大残差的维度通常也有大加速度 |
| max_residual ↔ max_jerk | **0.94** | 同上，三阶导与残差也高度相关 |
| max_acc ↔ max_jerk | **0.99** | 加速度和 jerk 几乎完全共线 |
| flagged_ratio ↔ max_run | **0.73** | 中等正相关——异常帧多的 episode 通常也有长连续异常段 |
| flagged_ratio ↔ num_flagged | **0.62** | 中等正相关（受 episode 长度影响） |
| num_flagged ↔ num_frames | **0.95** | 异常帧的绝对数量几乎完全由 episode 长度决定——说明 flagged_ratio 才是更好的归一化指标 |

**发现**：max_residual/max_acc/max_jerk 三者高度共线（r > 0.94），信息高度冗余。在构建筛选条件时只需使用其中之一即可。flagged_ratio 和 max_run 的相关性较弱（0.73），说明它们提供了互补信息——有些 episode 异常帧比例高但分散（ratio 高、run 低），有些则集中在一段（run 高但 ratio 不一定高）。

### 5.2 Stage 2 内部

| 指标对 | 相关系数 | 解读 |
|--------|---------|------|
| min_da ↔ mean_da | **0.90** | 高度正相关——min 差的 episode，mean 也差 |
| mean_da ↔ num_flagged_dims | **-0.96** | 强负相关——flagged 维度越多，mean_da 越低 |
| min_da ↔ num_flagged_dims | **-0.81** | 强负相关 |
| max_abs_lag ↔ mean_da | **-0.48** | 中等负相关——时延越大，一致性越差 |
| max_abs_lag ↔ num_flagged_dims | **0.55** | 中等正相关——时延大的 episode 倾向于有更多 flagged 维 |

**发现**：mean_da 与 num_flagged_dims 的 -0.96 表明这两个指标几乎提供相同的信息。max_abs_lag 与一致性指标的中等相关性（0.48–0.55）提示时延是影响 DA 的因素之一，但不是唯一因素——部分 DA 低的 episode 可能是因为数据错误（如通道映射错乱）而非简单的时延偏移。

### 5.3 跨 Stage 关联

| 指标对 | 相关系数 | 解读 |
|--------|---------|------|
| flagged_ratio (S1) ↔ min_da (S2) | **0.63** | 中等正相关——突变多的 episode，state-action 一致性**反而较高** |
| flagged_ratio (S1) ↔ num_checked_dims (S2) | **0.60** | 中等正相关——突变多的 episode 反而有更多维度被检测到 |
| num_flagged (S1) ↔ num_frames (S3) | **0.95** | 强正相关——纯粹由 episode 长度驱动 |
| extreme_value_flagged_frames ↔ extreme_value_flagged_ratio | **0.78** | 高正相关 |

**关键发现**：Stage 1（突变检测）与 Stage 2（趋势一致性）之间存在一个**反直觉的正相关**（r=0.63）：突变帧比例高的 episode，min_da 反而较高。这可能是因为：

1. 长 episode 同时有更多突变帧（绝对数量）和更稳定的趋势一致性（更长的信号提供更可靠的 DA 估计）
2. 突变检测和趋势一致性检测的侧重点确实不同——前者关注帧级信号跳变，后者关注信号间的因果趋势

Stage 3（极值）与其他两个 Stage 的相关性普遍较低（r < 0.2），说明极值检测确实捕获了不同维度的质量问题。

---

## 6. 数据质量画像与建议

### 6.1 三级检测综合通过率

假设三级独立应用 episode 级判据：

| 阶段 | 通过标准 | 通过数 | 通过率 |
|------|---------|-------|--------|
| Stage 1（以当前 max_flagged_ratio=0.3, max_run_length=10） | keep=True | 27 | 0.12% |
| Stage 2（da_threshold=0.65） | keep=True | 11,662 | 51.5% |
| Stage 3（frame_mask，不丢弃） | keep=True | 22,626 | 100% |

### 6.2 threshold_report 建议的阈值

| 参数 | 建议值 | 预期 episode 保留率 |
|------|--------|-------------------|
| `max_flagged_ratio` | 0.5 | ≈ 26% |
| `da_threshold` | 0.6 | ≈ 51.7% |
| `alpha` | 0.1 | 帧标记率 5%–15% |

### 6.3 分析建议

1. **Stage 1 阈值过紧**：当前 `max_flagged_ratio=0.3, max_run_length=10` 导致 99.88% 的 episode 不通过。建议：
   - 将 `max_flagged_ratio` 放宽到 0.5（保留率提升到 ≈ 26%）或更高
   - 将 `max_run_length` 放宽到 50–100（当前 p25=49）
   - 或直接使用 `frame_mask` 策略做帧级清洗，而非 episode 级丢弃
   - **增加逐维分析**：max_residual/max_acc/max_jerk 的分布极度集中暗示某个固定维度在主导，需找出并考虑将其加入 `exempt_dims`

2. **Stage 2 双峰清晰可切**：min_da 的双峰结构使得 `da_threshold=0.6` 是一个自然的切分点，几乎不受阈值微调影响。建议：
   - 采用 `da_threshold=0.6`，此时保留率 ≈ 51.7%
   - 对 max_abs_lag=15（搜索上界堆积的 ≈13% episode）进行专项排查——可能需增大 `max_lag` 到 20–25 以获得更准确的 DA 估计

3. **Stage 3 数据总体正常**：α=0.1 下大部分 episode 无极值（p50=0），说明关节值域分布合理。建议：
   - 保持 `alpha=0.1` 作为生产配置
   - 对 flagged_ratio > 0.3 的尾部 episode（约 5%）做抽样人工检查

4. **综合清洗策略建议**：
   - 考虑 Stage 1 使用 `frame_mask` 做帧级过滤 + Stage 2 使用 `episode_discard` 做 episode 级过滤 + Stage 3 使用 `frame_mask` 做帧级过滤
   - 预期最终 episode 保留率 ≈ 50%（主要由 Stage 2 决定），帧级保留率需结合 Stage 1 和 Stage 3 的掩码交叉进一步评估

---

## 附录 A: 统计图索引

| 图文件 | 类型 | 所属阶段 |
|--------|------|---------|
| `sudden_change_flagged_ratio-hist.png` | 直方图 | Stage 1 |
| `sudden_change_flagged_ratio-box.png` | 箱线图 | Stage 1 |
| `sudden_change_max_run-hist.png` | 直方图 | Stage 1 |
| `sudden_change_max_run-box.png` | 箱线图 | Stage 1 |
| `sudden_change_max_residual-hist.png` | 直方图 | Stage 1 |
| `sudden_change_max_residual-box.png` | 箱线图 | Stage 1 |
| `sudden_change_max_acc-hist.png` | 直方图 | Stage 1 |
| `sudden_change_max_jerk-hist.png` | 直方图 | Stage 1 |
| `sudden_change_num_flagged-hist.png` | 直方图 | Stage 1 |
| `sudden_change_keep-hist.png` | 直方图 | Stage 1 |
| `state_action_min_da-hist.png` | 直方图 | Stage 2 |
| `state_action_min_da-box.png` | 箱线图 | Stage 2 |
| `state_action_mean_da-hist.png` | 直方图 | Stage 2 |
| `state_action_mean_da-box.png` | 箱线图 | Stage 2 |
| `state_action_num_flagged_dims-hist.png` | 直方图 | Stage 2 |
| `state_action_max_abs_lag-hist.png` | 直方图 | Stage 2 |
| `extreme_value_flagged_ratio-hist.png` | 直方图 | Stage 3 |
| `extreme_value_flagged_ratio-box.png` | 箱线图 | Stage 3 |
| `extreme_value_flagged_frames-hist.png` | 直方图 | Stage 3 |
| `extreme_value_num_frames-hist.png` | 直方图 | Stage 3 |
| `stats-corr-pearson.png` | 相关矩阵热力图 | 全局 |
