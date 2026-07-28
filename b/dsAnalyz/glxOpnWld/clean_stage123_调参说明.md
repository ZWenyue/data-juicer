# `clean_stage123.py` 调参说明

Stage 1–3 状态/动作信号清洗脚本的参数调整指南。脚本只读 `lerobot_processed/` 下的
unified parquet，**不修改源数据**，输出报告 `cleaning_stage123.json` / `.csv`。

---

## 1. 三个 Stage 在做什么（先理解再调参）

| Stage | 作用 | 命中后果 | 作用信号 |
|-------|------|----------|----------|
| **Stage 1 突变检测** | 抓单帧/瞬时跳变（碰撞、丢包毛刺） | **整 episode 丢弃** | 位置类：双臂关节（state+action）、torso 位置（state） |
| **Stage 2 状态-动作趋势对齐** | 抓 state 与 action 时序方向不一致（时钟不同步/丢包） | **整 episode 丢弃** | 双臂关节（state vs action 同名维） |
| **Stage 3 极值过滤** | 标记落在正常分布之外的极端帧 | **帧级 flag**（保留 episode） | 所有有效维（夹爪除外） |

> **不使用末端 EE**：Stage 4/5（FK 一致性、基座系对齐）已完全去掉，三个 Stage 只跑
> 关节/夹爪/torso/底盘信号。
>
> **速度/指令通道被有意排除出 Stage 1**：torso、base 是阶跃式速度指令，每个速度脉冲
> 都像"突变"，会误报。Stage 1 只作用于平滑的位置信号。

---

## 2. 调参总流程（推荐迭代方式）

```bash
# 1) 先跑全量，拿到基线报告
python3 clean_stage123.py --jobs 16

# 2) 看整体保留率和坏数据集（读 cleaning_stage123.json 的 summary / per_dataset）
#    - kept_frac 太低 → 放宽阈值；太高（怀疑没清干净）→ 收紧
#    - 找出 kept 比例异常低的数据集，锁定是 s1 还是 s2 在丢

# 3) 锁定可疑数据集单独调，快速迭代
python3 clean_stage123.py --only <数据集名> --jobs 4 --out /tmp/probe

# 4) 满意后再跑全量固化
```

**判断标准（经验值）**：
- 单一 embodiment 的真实遥操作数据，健康的整体保留率通常在 **85%–95%**。
- 若某数据集 Stage 2 丢 > 50%，多半是该数据集**整体** state-action 错位（类似论文里
  RoboMIND-UR 81% 的情况），值得人工看几条再决定。
- Stage 3 帧级标记通常在 **5%–15%** 之间；远高于此说明 `--alpha` 太小或该 embodiment
  的全局分位数不适配该任务子集。

---

## 3. 参数逐项说明

### Stage 1（突变检测 → 整 episode 丢）

| 参数 | 默认 | 含义 | 怎么调 |
|------|------|------|--------|
| `--c-res` | 8.0 | 残差阈值 = `c_res × sigma` | **调大→更宽松**（少丢）。误伤多时先动这个 |
| `--c-acc` | 8.0 | 加速度（二阶差分）阈值倍数 | 同上 |
| `--c-jerk` | 8.0 | jerk（三阶差分）阈值倍数 | 同上 |
| `--floor-frac` | 0.02 | sigma 下限 = 关节量程(q99−q01) 的比例 | 决定"多大的跳变才算突变"。`c_res × floor_frac` ≈ 触发所需的量程占比（默认 8×0.02 ≈ **16% 量程**）。调大→更宽松 |
| `--abs-floor` | 0.01 | sigma 绝对下限（rad/m） | 保护量程极小的关节，避免噪声误报。一般不用动 |
| `--s1-min-frames` | 1 | episode 内 flagged 帧数 ≥ 此值即丢 | **你当前设成 1（命中即丢）**。想容忍零星毛刺→设 3~5 |
| `--s1-min-len` | 15 | 短于此帧数的 episode 跳过 Stage 1 | 一般不动 |
| `--med-k` | 7 | 中值滤波核 | 抑制脉冲用；一般不动 |
| `--sg-win` | 15 | Savitzky–Golay 窗口 | 平滑趋势窗，一般不动 |
| `--sg-poly` | 3 | Savitzky–Golay 阶数 | 一般不动 |

**关键直觉**：Stage 1 的有效触发门槛 ≈ `c_res × floor_frac × 关节量程`。
- 想"只抓大跳变、少丢"：调大 `--c-res`（如 10~12）或 `--floor-frac`（如 0.03~0.05）。
- 想"更敏感、多抓"：调小它们，或把 `--s1-min-frames` 保持 1。

### Stage 2（状态-动作趋势对齐 → 整 episode 丢）

| 参数 | 默认 | 含义 | 怎么调 |
|------|------|------|--------|
| `--da-thresh` | 0.65 | 方向一致性(DA)阈值，低于则该维 flag | **论文区间 0.6–0.7**。调低→更宽松（少丢）；调高→更严 |
| `--eps-frac` | 0.01 | "有效运动"阈值 = 关节量程的比例 | 只在真有运动的帧上算 DA。**太小会被噪声污染 DA**（务必保持 ≥0.005）。数据抖动大可调到 0.02 |
| `--max-lag` | 15 | 互相关搜索的最大时移（帧，≈1s@15fps） | 若怀疑 state/action 有较大延迟未对齐，可调大到 30 |
| `--s2-min-active` | 10 | 少于此数量的有效运动帧则跳过该维 | 一般不动 |
| `--s2-min-len` | 20 | 短于此帧数的 episode 跳过 Stage 2 | 一般不动 |
| `--action-is-delta` | 关 | action 为增量时先积分再比较 | Galaxea 是绝对关节指令，**保持关闭** |

**关键直觉**：Stage 2 最敏感的两个旋钮是 `--da-thresh` 和 `--eps-frac`。
- 一个数据集被大面积丢：先 `--da-thresh 0.6` 看是否只是卡在边界；仍大量失败则多为真错位。
- 明明轨迹对得很好却大量失败：多半是 `--eps-frac` 太小（噪声帧拖低 DA），调大到 0.015~0.02。

### Stage 3（极值过滤 → 帧级 flag）

| 参数 | 默认 | 含义 | 怎么调 |
|------|------|------|--------|
| `--alpha` | 0.1 | 容差带宽 = `[q1−α·(q99−q1), q99+α·(q99−q1)]` | **调大→带更宽、标记更少**（如 0.2、0.5）；调小→更严 |

> 夹爪维（bimodal）已自动豁免 Stage 3；分位数取自 `dataset_stats.json` 的
> per-embodiment `global_q01/q99`。

### 通用/运行参数

| 参数 | 默认 | 含义 |
|------|------|------|
| `--root` | `lerobot_processed` | 输入根目录 |
| `--only <名...>` | 全部 | 只处理指定数据集目录 |
| `--limit N` | 全部 | 只处理前 N 个数据集（快速试跑） |
| `--jobs N` | 8 | 并行进程数（227 数据集用 16 约 45s） |
| `--out <前缀>` | `<root>/cleaning_stage123` | 报告输出路径前缀 |

---

## 4. 常见场景 → 具体调法

**a) 整体丢太多**
```bash
python3 clean_stage123.py --da-thresh 0.6 --c-res 10 --c-acc 10 --c-jerk 10 --s1-min-frames 3
```

**b) 怀疑没清干净、想更严**
```bash
python3 clean_stage123.py --da-thresh 0.7 --c-res 6 --alpha 0.05
```

**c) 某数据集 Stage 2 大面积丢，先确认是不是真错位**
```bash
python3 clean_stage123.py --only Arrange_Fruits_20250819_011 --jobs 4 --out /tmp/probe
# 再读 /tmp/probe.json 里该 episode 的 stage2_flagged_dims，看 DA 值和命中的是哪些关节维
```

**d) Stage 1 允许零星毛刺、只丢严重的**
```bash
python3 clean_stage123.py --s1-min-frames 5 --c-res 10
```

**e) Stage 3 标记率过高（>20%）**
```bash
python3 clean_stage123.py --alpha 0.3
```

---

## 5. 怎么读报告来指导调参

`cleaning_stage123.json` 结构：
- `summary`：全局统计 + 本次运行的完整 `config`（复现用）。
  - `kept_frac`、`stage1_dropped_episodes`、`stage2_dropped_episodes`、`stage3_frac_in_kept`
- `per_dataset`：每个数据集的 `eps / kept / s1 / s2` 计数 → 快速定位坏数据集、判断是 s1 还是 s2 主导。
- `episodes`：逐 episode 明细
  - `stage1_flagged_frames`、`stage1_dims`：命中帧数 + 命中的维（定位是哪个关节在毛刺）
  - `stage2_flagged_dims`：`{维: DA值}` → **DA 离 0.65 多远**决定该不该调 `--da-thresh`
  - `stage3_frac`：该 episode 被标记的帧比例

`.csv` 是同样信息的扁平表，方便直接排序/透视找异常数据集。

**调参口诀**：先看 `per_dataset` 找异常 → 再看该数据集 `episodes` 里的 `stage1_dims` /
`stage2_flagged_dims` 定位到具体维和 DA 值 → 据此决定是放宽阈值还是判定为真坏数据。
