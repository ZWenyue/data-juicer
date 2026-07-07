# Offline Policy Diagnostics for VLA / WAM

> v2: 在 v1 (Tracking / Smoothness / Stability / Task) 基础上，补齐**闭环部署**最关键的三块——
> Chunk Decay、Safety Spike、Group-aware Aggregation，并区分 open-loop 与 closed-loop 两个评估视角。

---

## 1. Motivation

当前 VLA/WAM 的评估方式主要依赖：

- GT vs Pred 曲线可视化
- 人眼判断轨迹是否"像"
- 主观判断是否平滑

这些方式不可量化、不可复现、无法横向比较，**且严重低估闭环风险**。

### v1 框架的盲点（必须补齐）

| 盲点 | 后果 |
|---|---|
| 把所有 `chunk_offset` 平均掉 | 看不出"chunk 末端衰减"——闭环最重要的指标 |
| Smoothness 只看均值，不看尖峰 | 一个 0.9 rad 的关节抽动平均后看不出来，但真机会急停 |
| 用 P95 当尖峰 | 95% 分位掩盖 100% 分位，真机怕的是那 1 次 spike |
| 16 个 signal 直接求均值 | gripper 的离散 jerk 量级压死 arm 的连续 jerk |
| 一个总分 PQS | 假设你的部署场景与训练权重一致；现实是 open-loop / closed-loop 完全不同 |

---

## 2. 两个评估视角（必须区分）

| 视角 | 推理输入 | 关注指标 | 适用场景 |
|---|---|---|---|
| **Open-loop** | 永远用 GT 作为下一步输入 | RMSE / NRMSE / DTW / 模仿精度 | 训练验收、checkpoint 选优 |
| **Closed-loop** | 用上一帧 pred 作为下一步输入 | **Chunk Decay**、Spike、Stability | **真机部署决策** |

两个视角的结论**可能完全相反**：
- open-loop 表现最好的模型，可能 chunk 末端衰减最快 → 闭环最差
- 离线 P95 误差最低的模型，可能有 1 个 0.9 rad 的瞬时 spike → 真机直接 estop

> **建议：两个视角各算一份 PQS，分别命名 `PQS_open` 和 `PQS_closed`，不要再用单一总分掩盖差异。**

---

## 3. Overall Score

### 3.1 Open-loop PQS（模仿验收）

$$\text{PQS}_{open} = w_1 S_{track} + w_2 S_{smooth} + w_3 S_{stable} + w_4 S_{task}$$

### 3.2 Closed-loop PQS（真机部署）

$$\text{PQS}_{closed} = w_1 S_{track}^{(0)} + w_2 S_{decay} + w_3 S_{safety} + w_4 S_{smooth} + w_5 S_{task}$$

其中 $S_{track}^{(0)}$ 表示 chunk offset=0 时的 tracking，$S_{decay}$ 量化 offset 0 → H 的误差增长率。

---

## 4. Tracking Score（v1 保留）

衡量预测动作与 GT 动作的接近程度。

$$S_{track} = \alpha_1 (1-\text{NRMSE}) + \alpha_2 (1-\text{NDTW}) + \alpha_3 (1-\text{P95Error})$$

### 4.1 RMSE

$$\text{RMSE} = \sqrt{\frac{1}{T}\sum_t (a_t-\hat a_t)^2}, \quad \text{NRMSE} = \frac{\text{RMSE}}{\text{range}(a)}$$

### 4.2 DTW

$$\text{NDTW} = \frac{\text{DTW}(a, \hat a)}{T \cdot \text{range}(a)}$$

允许轻微时间偏移，适合 reaction delay / phase mismatch。

### 4.3 P95 Error

$$\text{P95} = \text{percentile}_{95}(|a-\hat a|)$$

检测局部大误差（**注意**：P95 不替代 max，见 §6.2）。

---

## 5. Smoothness Score（v1 保留 + 修正聚合）

$$S_{smooth} = \beta_1 (1-\text{MeanJerk}) + \beta_2 (1-\text{PeakJerk}) + \beta_3 (1-\text{TV})$$

### 5.1 Mean Jerk

$$j_t = \frac{a_t - 2 a_{t-1} + a_{t-2}}{\Delta t^2}, \quad \text{MeanJerk} = \frac{1}{T}\sum_t |j_t|$$

### 5.2 Peak Jerk

$$\text{PeakJerk} = \max_t |j_t|$$

### 5.3 Total Variation

$$\text{TV} = \sum_t |a_t - a_{t-1}|$$

### 5.4 ⚠️ 聚合规则（v2 新增，关键）

**不能**把 14 维 arm + 2 维 gripper + 4 维 torso + 3 维 chassis_velocity 直接求均值——
gripper 的离散开/合会让 jerk 量级比 arm 高 100× 以上，**单一均值会被它压死**。

正确做法：

```
S_smooth = mean(
  α_arm   * S_smooth(arm 14 dim,        normalized per-signal),
  α_grip  * S_smooth(gripper 2 dim,     normalized per-signal),
  α_torso * S_smooth(torso/chassis dim, normalized per-signal),
)
```

建议默认权重：`α_arm = 0.6, α_grip = 0.2, α_torso = 0.2`（按"真机执行风险占比"配）。

---

## 6. Stability Score（v1 保留 + Spike 新增）

$$S_{stable} = \gamma_1 (1-\text{FlipRate}) + \gamma_2 (1-\text{HFEnergy}) + \gamma_3 (1-\text{SpikeRisk})$$

### 6.1 Flip Rate

$$\text{FlipRate} = \frac{1}{T-1}\sum_t \mathbb{1}[\text{sign}(a_t) \ne \text{sign}(a_{t-1})]$$

⚠️ **重要解释**：高 flip_rate **未必是坏事**——
- 如果 GT 本身就在反复试探（如抓取试探阶段），匹配 GT 的 flip 是正确的
- 真正的问题是 `FlipRate(pred) − FlipRate(gt)`，记为 **ExcessFlipRate**

建议默认上报 `ExcessFlipRate` 而非原始 `flip_rate`。

### 6.2 High Frequency Energy

对 pred 做 FFT：

$$E_{HF} = \frac{\sum_{f>f_c}|A(f)|^2}{\sum_f |A(f)|^2}$$

`f_c` 取控制频率的 1/4（如 30 Hz 控制环则 `f_c = 7.5 Hz`）。

### 6.3 ⭐ Spike Risk（v2 新增）

P95 / mean 这些都看不出**单点抽动**。真机最怕的就是那一个 0.9 rad 的瞬时跳。

$$\text{SpikeRisk} = \frac{\max_t |a_t - \hat a_t|}{\text{joint\_max\_step\_limit}}$$

其中 `joint_max_step_limit` = 该关节单步允许的最大位移（由 hardware spec 决定，例如 30 Hz 控制下取 `v_max / 30`）。

- `SpikeRisk > 1`：必触发限速/急停
- `SpikeRisk ∈ (0.5, 1)`：高风险，可能触发 jerk 限位
- `SpikeRisk < 0.3`：安全

可选：定义 `SpikeCount = #{t : SpikeRisk_t > 0.5}` 作为辅助监控。

---

## 7. ⭐ Chunk Decay Score（v2 新增主模块）

这是 v1 完全没有，但**对闭环部署最关键**的一块。

### 7.1 背景

VLA 通常一次预测一个长度为 H 的 action chunk（典型 H=16）。
Open-loop 评估时，每个 step 的输入都是 GT，所以 offset 0~H 的误差差距被掩盖。
**Closed-loop 时，上一个 chunk 的 offset=H 的预测会变成下一个 chunk 的 offset=0 的输入**。
→ chunk 末端衰减得越快，闭环就越容易飘。

### 7.2 定义

每个 chunk offset `k ∈ [0, H)` 单独计算 RMSE：

$$\text{RMSE}(k) = \sqrt{\frac{1}{N_k}\sum_{(t,k)} (a_t - \hat a_{t,k})^2}$$

定义衰减率：

$$\text{DecayRatio} = \frac{\text{RMSE}(H-1)}{\text{RMSE}(0)}$$

定义平均衰减斜率：

$$\text{DecaySlope} = \text{linear\_fit\_slope}(\text{RMSE}(0..H-1)) \big/ \text{RMSE}(0)$$

### 7.3 Score 归一化

$$S_{decay} = 100 \cdot \max(0,\ 1 - (\text{DecayRatio} - 1) / R_{ref})$$

其中 `R_ref` 是经验上界（建议 3.0，即 ratio 4× 时 score 归零）。

### 7.4 真实示例（本仓库三个模型）

| Model | RMSE(0) | RMSE(15) | DecayRatio | $S_{decay}$ |
|---|---|---|---|---|
| **rldx_1** | 0.0285 | **0.0636** | **2.23 ★** | ~ 59 |
| pi0_5     | 0.0247 | 0.0748 | 3.03 | ~ 32 |
| fastwam   | 0.0253 | 0.0785 | 3.10 | ~ 30 |

→ rldx_1 的 chunk decay 显著最慢，可以容忍**更长 chunk + 更低推理频率** = GPU 预算多 30%+。
→ open-loop offset=0 时 pi0_5 反而最准，但 closed-loop 跑不过 rldx_1，这正是 v1 框架抓不到的关键差异。

### 7.5 Inter-chunk Discontinuity（可选，闭环专属）

闭环时，每个 chunk 切换瞬间会有一个"跳跃"：

$$\text{ChunkJump} = \text{median}_{chunk}\big(\|\hat a_{c, 0} - \hat a_{c-1, H-1}\|\big)$$

值越大表示新 chunk 与旧 chunk 的衔接越不平滑，真机会感受为"顿挫"。

---

## 8. Task Score（v1 保留）

$$S_{task} = \delta_1 \text{SuccessRate} + \delta_2 (1-\text{FinalError})$$

### 8.1 Success Rate

任务定义的成功（grasp / place / goal reach）。这是最重要也最难自动判定的指标，建议至少：

- final_error_normalized < 0.05 算 success（粗糙代理）
- 关键关节 final_error < 5° 算 success（更严）

### 8.2 Final State Error

$$\|x_T - \hat x_T\|, \quad \text{Normalized} = \frac{\|x_T - \hat x_T\|}{\text{range}(x)}$$

---

## 9. ⭐ Per-phase 分解（v2 新增）

将 trajectory 按时间四分位拆开，单独看各阶段的 tracking：

| 阶段 | 时间段 | 典型任务含义 |
|---|---|---|
| Q1 | 0% - 25% | 起步加速 |
| Q2 | 25% - 50% | 接近目标 |
| Q3 | 50% - 75% | **关键操作**（抓取、对位） |
| Q4 | 75% - 100% | 收尾稳态 |

为什么重要：
- **Q1 / Q3 是真机最容易失败的两段**（起步惯性 + 抓取接触切换）
- 一个 Q4 表现极好但 Q3 表现差的模型，离线总分可能与对手持平，**但真机抓取成功率会差 20%+**

输出建议：每个 model × 每个 group × 每个 phase 的 RMSE 表（不是单一总分）。

---

## 10. Group-aware Aggregation 规则（v2 强制）

**禁止**对异质 signal 直接 `mean()`：

| ❌ 错误 | ✅ 正确 |
|---|---|
| `mean(all 16 signals)` | `weighted_mean({arm: ..., gripper: ..., torso: ...})` |
| 单一 `peak_jerk` 平均值 | 各 group 分开报告，并标注哪个 group 是 bottleneck |
| flip_rate 直接取均值 | `flip_rate(pred) − flip_rate(gt)` 即 ExcessFlipRate |

**强制规范**：summary CSV **必须**同时输出
- `pqs_open` / `pqs_closed`
- 每个 group 各自的 PQS
- 各 group 的 bottleneck 维度名称（哪个 signal 在哪个指标上拖了后腿）

---

## 11. Recommended Weights

### 11.1 Open-loop / Imitation Learning Oriented

| Module | Weight |
|---|---|
| Tracking | 0.45 |
| Smoothness | 0.25 |
| Stability | 0.15 |
| Task | 0.15 |

### 11.2 Closed-loop / Real Robot Deployment Oriented ⭐

| Module | Weight |
|---|---|
| Tracking(offset=0) | 0.15 |
| **Chunk Decay** | **0.25** |
| Safety (Spike + ExcessFlip) | 0.20 |
| Smoothness | 0.20 |
| Task | 0.20 |

> 原因：在真机闭环中，Chunk Decay 决定推理频率预算，Safety 决定会不会急停，
> Smoothness 决定关节寿命，Tracking 反而只在 offset=0 起作用。
> 模仿精度（v1 占 45%）应降到 15%。

---

## 12. Recommended Minimal Version（v2）

建议最小可用版本：

### Tracking

- RMSE（**按 chunk_offset 分组**，不要直接平均）
- NRMSE

### Chunk Decay ⭐ 新增

- DecayRatio = RMSE(H-1) / RMSE(0)

### Smoothness

- Mean Jerk（按 group 分开）
- TV

### Stability / Safety

- ExcessFlipRate（pred − gt）
- **SpikeRisk = max|err| / joint_step_limit** ⭐ 新增

### Task

- Success Rate
- Final Error (normalized)

输出 3 张表：
1. `summary.csv`：每 model 一行，含 `pqs_open` / `pqs_closed` / 各 group 的 PQS
2. `by_offset.csv`：每 model × 每 offset 的 RMSE（用于画 chunk decay 曲线）
3. `by_signal.csv`：每 model × 每 signal 的所有原子指标 + bottleneck 标记

---

## 13. Implementation Pitfalls（v2 新增）

### 13.1 ⚠️ load_csv 不能按 step 直接平均

当前 `offline_policy_diagnostics.py` 的 `load_csv()` 函数会把相同 `step` 的多行**直接求均值**
（line 117-122）。这会把 chunk_offset 维度抹平，导致 Chunk Decay 完全算不出来。

修复：

```python
# 错误：按 step 聚合时直接 mean()
sums[step][name] += value
counts[step][name] += 1

# 正确：保留 (step, chunk_offset) 二维索引
records[(step, chunk_offset)][name] = value
```

并允许下游按 `chunk_offset` group-by 计算。

### 13.2 ⚠️ 不要混用 `chunk_offset` / `horizon_index` / `chunk_index` 列名

不同模型的 CSV schema 不同：
- `rldx_1`: `chunk_offset`
- `pi0_5`:  `horizon_index`
- `fastwam`: `chunk_offset` + `chunk_index`

建议加一层 schema 适配层，统一映射到内部字段 `offset` / `chunk_id`。

### 13.3 ⚠️ Normalization 必须用 GT 的 range，不能用 pred 的 range

如果 pred 过度保守（输出几乎不动），用 pred range 会让所有归一化指标看起来都很好。
**永远用 GT 的 `max - min` 作为分母**，并对接近 0 的 range 加 epsilon 保护。

### 13.4 ⚠️ Success Rate 不是 final_error 阈值

`final_error < threshold` 只是一个**粗糙代理**。真正的成功率应该来自：
- 仿真器或真机的 task-specific success signal
- 由人工标注的 bad case 集合
- 或者关键关节同时满足多个阈值的复合判定

不要被 "success_rate = 1.0" 这种数字误导——v1 当前所有模型都是 1.0，根本没区分度。

---

## 14. Conclusion

最终评估体系应当不仅关注：

> "预测是否接近 GT"（open-loop tracking）

更要关注：

- **chunk 末端是否快速衰减**（闭环可推理性）
- **是否有单步抽动**（真机安全）
- **每个 group 各自是否健康**（不被均值掩盖）
- **任务关键阶段是否稳**（Q1 / Q3 而非平均）
- **是否可执行 + 真正完成任务**

因此采用 **Offline Policy Diagnostics v2** 作为统一评估框架：
- 拆 open-loop 与 closed-loop 两个视角
- 强制 group-aware 聚合
- 把 Chunk Decay 与 Spike Safety 纳入主指标

---

## Appendix A: v1 → v2 变更速查

| 变更 | v1 | v2 |
|---|---|---|
| 视角 | 单一 PQS | open / closed 两套 |
| Chunk 维度 | 平均掉 | **作为主模块** Chunk Decay |
| Peak | 用 P95 | P95 + **max + SpikeRisk** |
| Flip | 原始 flip_rate | **ExcessFlipRate = pred − gt** |
| 聚合 | 16 signal 直接 mean | **group-aware weighted mean** |
| 阶段 | 全 trajectory 平均 | **per-phase (Q1/Q2/Q3/Q4)** |
| 部署权重 | 模仿主导 (Track 0.45) | **Chunk Decay 0.25 + Safety 0.20 主导** |
| 已知 bug | `load_csv` 抹平 chunk 维度 | 显式修复 |
