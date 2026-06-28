# WAM 预训练数据处理方案（FastWAM & AHA-WAM）

> 面向**多个开源、异构轮式人形全身数据集（不同自由度 / 频率 / 相机 / 动作语义，~20+ 维）**的大规模预训练。
> 目标表征：**连续统一动作空间**（不做 RVQ 离散化），采用 **GR00T / OpenVLA 路线 —— 原生表征 + 本体标签**：不强行把关节角/EEF/速度统一成一种物理量，保留各数据集原生动作，靠「语义槽位搬运 + per-embodiment 归一化 + embodiment 条件」让模型消歧。直接喂给两个模型现有的 `ActionDiT`。
> 本方案把 [`action_tokenizer_data_pipeline.md`](action_tokenizer_data_pipeline.md) 的多本体对齐思想，落到 FastWAM / AHA-WAM 真实的 LeRobot + 视频 + 文本数据栈上。

---

## 0. 结论速览（TL;DR）

1. **这是真正的大规模异构对齐**（同 Open X-Embodiment / GR00T / OpenVLA）。难点**不在自由度数对不齐**（pad + mask 已解决），而在**动作语义/表征一致性**：同一「左臂」槽位在不同数据集可能是关节角 / EEF delta / 速度，约定（夹爪开合、坐标系、绝对 vs 增量）也各不相同。**对齐工作量的 80% 在每个数据集的 adapter 上。**

2. **表征策略已定：原生 + 本体标签**（见 §4.0）。不做 IK/FK 不统一物理量 → adapter 变轻（只需语义槽位映射 + 频率重采样 + 可选约定统一）；代价是**模型必须吃 embodiment 标签**才能区分同槽位的不同语义。

3. **两个模型已经有多本体脚手架，但没接线。** `embodiment_datasets` / `embodiment_processors` 配置 schema、`max_action_dim` Hydra resolver、`ConcatLeftAlign` 的「pad 到统一维 + 生成 `action_dim_is_pad` 有效性 mask」**都已存在**。缺的是：① 真正的多本体 Dataset（带 embodiment 标签）、② per-embodiment 统计量、③ 跨频率重采样、④ **loss 里用上维度 mask**、⑤ `action_dim = max_action_dim(...)` 接到 model config、⑥ **ActionDiT 注入 embodiment 条件**（全新，原方案没有）。

4. **开 doc 的五步，在 WAM 里这样落位**（见 §3 映射表）：
   - 频率重采样 → **新建**（base dataset 现在断言所有 dir 同 fps）
   - per-embodiment 归一化 → 扩展现有 `LinearNormalizer` / `dataset_stats.json`
   - 统一空间 + mask → **复用** `ConcatLeftAlign` + `action_dim_is_pad`
   - 语义槽位对齐 → 实现为 per-embodiment 的 `action_state_transforms`（每个数据集一个 adapter）
   - patchify / RVQ → **不做**（连续路线），仅留作可选未来分支

5. **统一动作空间按全身轮式人形 pad 到 32 维**（左右臂 7+7、左右手 1+1、腰 5、头 3、底盘 3 = 27 语义维 + 5 预留维），由数据驱动、可配置；不足维度在语义槽位内右侧 pad、mask 置 0。

6. **两个硬模型改动**：① 动作 flow-matching loss 目前只屏蔽时间维 pad（`action_is_pad`），**必须补上维度 pad（`action_dim_is_pad`）**，否则把占位 0 当真实动作学；② **embodiment 条件注入**（per-embodiment learnable soft-prompt / embedding，参照 ActionCodec），否则「原生 + 标签」里的标签是空的。世界模型的视频像素隐含部分本体信息 → ② 是「强烈建议」而非「绝对必须」，但便宜且稳。

---

## 1. 背景与目标

### 1.1 两个模型现状（来自代码核查）

| | FastWAM | AHA-WAM |
|---|---|---|
| 定位 | World-Action Model（视频专家 + 动作专家 MoT 混合注意力） | 异步、horizon 自适应 WAM（慢视频规划器 + 快动作执行器，OVCR 路由） |
| 视频骨干 | Wan2.2-TI2V-5B（30 层, 3072 hid, 24 heads, head_dim 128） | 同上 |
| 动作专家 | `ActionDiT`（1024 hid，heads/head_dim/层数与视频对齐以做 MoT） | 同上，额外有 chunk / offset / history |
| 动作表征 | **连续**，`action_dim` 由 data config 决定（LIBERO 7 / RoboTwin 14） | **连续**，14 维；`action_chunk_size=16`, `action_horizon=64` |
| 归一化 | `LinearNormalizer`，模式 `min/max`·`z-score`·`q01/q99`·常数；全局/分步 | 同上（默认 `z-score`，clamp 到 ±5） |
| 数据格式 | LeRobot v2.1（`data/` parquet + `meta/` + `videos/` mp4） | 同上 |
| 视频采样 | `num_frames=33`, `action_video_freq_ratio=4` → 9 视频帧 + 32 动作 | `num_frames=65`, `freq_ratio=8` → 9 视频帧 + 32 动作；offset 配置 `num_frames=97` |
| 文本 | T5（UMT5-XXL）预算缓存 `{context[128,4096], mask[128]}` | 同上 |
| 训练前置 | `preprocess_action_dit_backbone.py`（Wan22→ActionDiT 插值）、`precompute_text_embeds.py` | 同上 |

### 1.2 预训练目标

- 在**多个轮式人形全身数据源**上联合预训练，统一动作 32 维连续空间。
- 各源控制频率/相机配置/具体自由度可能不同 → 必须做**频率重采样 + 语义槽位对齐 + per-embodiment 归一化 + 有效性 mask**。
- 产物可直接被两个模型的 dataloader 消费，无需改动模型主干（仅补 loss 维度 mask + 少量接线）。

---

## 2. 与 `action_tokenizer_data_pipeline.md` 的关系

| 开 doc 的设计 | 本方案是否沿用 | 落地方式 |
|---|---|---|
| 频率重采样到 `TARGET_FREQ`（夹爪等用最近邻） | ✅ 沿用 | 新建 `resample_actions`，放在 LeRobot 化阶段（离线）或 dataset transform（在线） |
| per-embodiment 归一化（各源独立统计、不全局混合） | ✅ 沿用，**这是核心** | 扩展 `dataset_stats.json` 为 per-embodiment；归一化模式沿用现有 `LinearNormalizer` |
| 统一动作空间 + 有效性 mask（占位 0 不当真实动作） | ✅ 沿用 | 复用 `ConcatLeftAlign` 的 pad + `action_dim_is_pad`，**并在 loss 用上 mask** |
| 物理语义分组（左右臂/夹爪/腰/底盘…） | ✅ 沿用，扩展到全身 | per-embodiment `action_state_transforms` 做「原始维 → 统一语义槽位」映射 |
| Patchify 成 `(m·n, h·d)` | ❌ 不做 | 连续 ActionDiT 不需要 patch；仅作为可选 RVQ 分支保留（见 §11） |
| RVQ tokenizer 训练 | ❌ 不做 | 超出本方案 |

> 一句话：**开 doc 的「重采样 → per-embodiment 归一化 → 统一空间 + mask」三步全部保留并落到真实代码 hook 上；patchify/RVQ 砍掉，因为我们走连续路线。**

---

## 3. 关键发现：codebase 已有的多本体脚手架（务必复用）

下面是已存在、可直接利用的 hook（FastWAM / AHA-WAM 两边基本一致）：

### 3.1 配置 schema 已支持多本体（仅缺实现）
- `search_dataset_stats_cache_json` 已读取 `dataset.embodiment_datasets`（每本体 `dataset_groups → dataset_dirs`）与 `processor.embodiment_processors`（每本体 `action_state_transforms`）。
  - `FastWAM/src/fastwam/datasets/lerobot/utils/normalizer.py:227-264`
- Hydra resolver `max_action_dim` / `max_state_dim`：对所有 embodiment 的 `shape_meta` 取最大维度。
  - `FastWAM/src/fastwam/utils/config_resolvers.py:34-70`（AHA-WAM 同路径）
  - 用法：`action_dim: ${max_action_dim:${data.train.dataset.embodiment_datasets}}`

### 3.2 统一维 pad + 有效性 mask 已实现（直接用）
- `ConcatLeftAlign`：拼接各 action/state key → pad 到 `target_dim` → 返回 `action_dim_is_pad` / `state_dim_is_pad`（`True` = 占位维）。
  - `FastWAM/src/fastwam/datasets/lerobot/transforms/action_state_merger.py:20-53`
- 处理器把它们透出到 batch：`action_dim_is_pad`（维度 mask, `[dim]`）与 `action_is_pad`（时间 mask, `[T]`）。
  - `FastWAM/src/fastwam/datasets/lerobot/processors/base_processor.py:225-235`

### 3.3 缺口（需要新建/修改）
| 缺口 | 现状 | 位置 |
|---|---|---|
| 多本体 Dataset（拼接 + 打 embodiment 标签） | `BaseLerobotDataset` 只接 `dataset_dirs` 列表，且**断言所有 dir 同 fps**，无 embodiment id | `base_lerobot_dataset.py:18-124` |
| per-embodiment 统计量 | `get_dataset_stats` 对所有 dir **全局**算 stats | `base_lerobot_dataset.py:251-373` |
| 跨频率重采样 | **无**（同 fps 断言把问题挡在门外） | 同上 |
| **loss 用维度 mask** | 动作 loss 只用 `action_is_pad`（时间），**没用 `action_dim_is_pad`（维度）** | FastWAM `fastwam.py:550-561`；AHA-WAM `base_wam.py:1210-1249` |
| **embodiment 条件注入** | ActionDiT 完全不吃 embodiment 信号，「原生 + 标签」无处落地 | `action_dit.py`（新增 `nn.Embedding`） |
| 每数据集 adapter（语义槽位映射） | 仅有 `LiberoTransform` 单例，缺通用/声明式映射 | `transforms/*.py` + `embodiment_processors` 配置 |
| `action_dim` 接线 | model config 的 `action_dim` 需指向 `max_action_dim(embodiment_datasets)` | `configs/model/*.yaml` |

---

## 4. 统一动作空间设计（轮式人形全身）

### 4.0 表征策略：canonical 关节表示 + 本体标签（参照 Qwen-RobotManip，仅取关节轴）

参照 **Qwen-RobotManip《Alignment Unlocks Scale》**——**对齐是数据扩展的前提**，不对齐则"加数据 = 加干扰"。它把对齐拆成三轴；**本方案只用关节角做预训练**，因此取其中两轴、显式跳过第三轴：

1. ✅ **Representation 表示对齐**（§4.1/§4.2）：canonical 模板（每臂块 = 关节 + 夹爪，外加腰/头/底盘 + reserved）+ **per-dim binary mask**。不同本体只填自己拥有的子集，pad 维 mask=0 不进 loss。
2. ❌ **Motion 运动对齐（camera-frame delta EE）—— 本方案不做**。Qwen 用末端 camera-frame delta 让"视觉相同的动作数值也接近"，是它跨本体迁移的最强机制；但那要求末端位姿 + 相机标定，且会把动作空间改成 EE。**我们只预训练关节角**，故不引入 EE / 相机系 delta。
3. ✅ **Behavioral 行为对齐**（§6 P0-b）：embodiment 条件——结构化 embodiment prompt（含 speed/fps，§S0）/ learnable embedding + 可选执行历史 in-context。

> **代价与缓解（重要）**：跳过 Motion 轴 = 放弃 Qwen 最强的数值对齐手段。纯关节 + pad/mask 下，"同一物理动作在不同本体数值不一致"只能靠三件事压住：① **§4.3 的臂内关节语义对齐（尤其 R1 Pro↔G2 这对 7-DoF）**、② per-embodiment 归一化、③ embodiment 条件。这三者必须做扎实，否则跨本体共享收益很弱。好在训的是**世界模型**，视频像素隐含本体信息，能进一步兜底。
>
> **FK 的用途**：仍要用各本体自身 URDF 的 FK，但**只用于** §4.3 的 G2 关节符号/零偏标定 + §S1.5 的关节-末端一致性校验（数据清洗），**不产出 EE 动作通道**。

**adapter 工作量（每个数据集一个）**：
- ✅ 必做：关节语义槽位映射（reorder，§4.3）、频率重采样、声明 `embodiment_id` 与拓扑/mask。
- 🟡 建议：统一明显约定（夹爪开合方向、底盘速度正方向）——能省样本。
- ❌ 不做：EE / 相机系 delta、跨本体关节 retarget、统一成单一物理量。

### 4.1 语义槽位（32 维 = 27 语义 + 5 预留，**语义维度由数据驱动**）

按物理语义分组、固定槽位顺序（半开区间）；语义维度由实际数据集 `shape_meta` 经 `max_action_dim` 求并集决定，下表为典型轮式人形布局（腰/头按最大本体 G2 的 5/3 DoF 取并集）：

```python
UNIFIED_ACTION_SPACE = {
    'left_arm':      (0, 7),    # 7  (≤7 DoF 臂，右侧 pad)
    'left_hand':     (7, 8),    # 1  (夹爪/灵巧手 1 维标量；多指手另议)
    'right_arm':     (8, 15),   # 7
    'right_hand':    (15, 16),  # 1
    'torso':         (16, 21),  # 5  (腰: 最多 5 DoF，如 G2 body_joint1~5)
    'head':          (21, 24),  # 3  (yaw/roll/pitch；仅 pan/tilt 的本体右侧 pad)
    'base':          (24, 27),  # 3  (轮式底盘: vx, vy, wz；差速则 vy pad)
    'reserved':      (27, 32),  # 5  预留 / pad（恒为占位，mask=0）
}  # UNIFIED_DIM = 32 (27 语义 + 5 预留)
```

设计要点：
- **同语义对齐**：所有本体的「右臂」都落在 `(8,15)`，跨本体可共享同一组 ActionDiT 权重。
- **槽内右侧 pad**：6-DoF 臂映射到 7 维槽，最后 1 维 pad、mask=0；差速底盘 2 维映射到 3 维槽，`vy` pad。
- **缺失部件整槽 pad**：无头/无腰的本体，对应整段槽位全 0、mask=0。
- **连杆/欠驱动夹爪只取驱动 DoF**：如 G2 的 OmniPicker（T1 `G2_90D.urdf`）是多连杆机构（inner `mimic` outer + 闭环 `loop_joint`），URDF 里有 6~8 个 finger 关节但**只有 1 个驱动 DoF**。`left_hand`/`right_hand` 槽取那 1 维开合标量，**被动连杆关节绝不当独立动作维**喂入。
- **27 语义 + 5 预留 = 32**：`reserved (27,32)` 恒为占位（mask=0），给未来本体/新部件留扩展位，使 `UNIFIED_DIM` 固定为 32（便于跨实验对齐、维度对齐到整数块）。
- **与 Qwen-RobotManip 的关系**：本布局相当于 Qwen 的 "per-arm block + 共享 reserved" 去掉末端位姿/灵巧手——Qwen 每臂块 = 关节7 + 末端位姿9 + 夹爪1 + 灵巧手12；我们**只取关节 + 夹爪**（`left_arm`+`left_hand` 即一条臂块），EE 那 9 维**有意不要**（§4.0 仅关节预训练）。日后若想补 EE 做数值对齐，从 `reserved` 切 9 维即可。
- **维度可调**：若引入多指灵巧手或更多腰部自由度，优先吃掉 reserved 槽；不够再扩大对应语义槽位并重算 `UNIFIED_DIM`；模型 `action_dim` 经 resolver 自动跟随。

### 4.2 有效性 mask 语义
- `action_dim_is_pad[i] = True` 表示第 `i` 维是占位（该本体不存在此自由度）。
- mask **只依赖 embodiment，不随时间变化** → 每个本体一个 `[UNIFIED_DIM]` 向量，加载时按 embodiment 生成即可，无需逐帧存储。

### 4.3 不同机械臂的关节重映射（per-embodiment）

§4.1 只规定了「左臂落在 `(0,7)`」这种**部件级**搬运；但三台真实机械臂的**臂内关节序、轴约定、自由度数都不同**，部件级搬运不足以让 `(0,7)` 这 7 维在不同本体间一一对应。这里规定臂内的逐关节重映射。

**先看三臂的实际差异（来自 URDF）**：

| | R1 Pro（目标机） | G2 | R1 Lite |
|---|---|---|---|
| 单臂 DoF | 7 | 7 | 6 |
| 轴约定 | 每关节显式世界轴 `Y,X,Z,Y,Z,Y,X` | **仅臂** DH 风格（全 local-Z + rpy 重定向）；腰/头为显式轴 | 每关节显式世界轴 `Z,Y,Y,Y,Z,X` |
| 结构 | S-R-S 拟人臂 | S-R-S 拟人臂 | 少一个肩部自由度 |

**统一臂槽语义模板**（左臂 `(0,7)` 内的固定角色顺序，右臂 `(8,15)` 同序）：

```python
ARM_SLOTS = ['shoulder_pitch', 'shoulder_roll', 'shoulder_yaw',
             'elbow_pitch', 'forearm_roll', 'wrist_pitch', 'wrist_roll']  # 索引 0..6
```

**左臂逐关节映射表**（`原始关节 idx → 槽位`；右臂同序、索引偏到各自原始组）：

| 槽位（角色） | R1 Pro | G2 | R1 Lite |
|---|---|---|---|
| 0 shoulder_pitch | j1 (Y) | j1* | j2 (Y) |
| 1 shoulder_roll | j2 (X) | j2* | — **pad** |
| 2 shoulder_yaw | j3 (Z) | j3* | j1 (Z) |
| 3 elbow_pitch | j4 (Y) | j4* | j3 (Y) |
| 4 forearm_roll | j5 (Z) | j5* | j5 (Z) |
| 5 wrist_pitch | j6 (Y) | j6* | j4 (Y) |
| 6 wrist_roll | j7 (X) | j7* | j6 (X) |

> `*` **仅手臂如此**：G2 的 arm_l/arm_r 用 DH 写法（7 个关节 `axis` 全是 `0 0 1`，靠 origin 的 ±90° rpy 重定向），**不能读 `axis` 字面**——结构上是 S-R-S 7-DoF、槽位 identity 对应 R1 Pro，但每关节 local frame 不同，符号/零偏须用 FK 在统一基座标系里标定。
> G2 的**腰/头/底盘则是显式轴、可直接读**（origin `rpy=0`），只是个别轴带负号（T1 版腰 `Y,Y,−X,Y,−Z`、头 `Z,−X,Y`）——这些不需要 FK，按该版 URDF 字面取符号即可。
> R1 Lite / R1 Pro 的角色是按显式轴推断的，**接数据前用 FK 可视化逐关节验证一次**（单独转一个关节看末端往哪动）。
> 以上 G2 数值以 **T1 版 `G2_90D.urdf`** 为准（腰/头的 limit、轴符号与旧 `G2.urdf` 不同）。

**两种处理强度，按 §4.0「原生 + 标签」的取舍选**：

- **A. 位置式（默认，最省）**：原始关节按序填入臂槽、6-DoF 末位 pad，不做角色对齐。同槽位跨本体语义不同，完全靠 **embodiment 条件（§6 P0-b）+ per-emb 归一化** 消歧。符合 §4.0 的「不做精确对齐」。
- **B. 语义对齐（推荐用于 R1 Pro↔G2）**：臂内也按上表角色对齐 + 统一旋转正方向（sign 翻转）。**因为你的目标机是 R1 Pro，而 G2 与它同为 7-DoF S-R-S**，让这一对在 `(0,7)/(8,15)` 上真正同语义，跨本体权重共享才有效；R1 Lite 缺的 `shoulder_roll` **按语义位（slot 1）pad，而不是末位 pad**。这是 §4.0 标 🟡「建议统一约定」的那一类，不是被禁止的 IK/FK retarget。

**声明式映射 + 通用 transform**（落到 `embodiment_processors.{emb}.action_state_transforms`，在 §S2 reorder 阶段执行；scale 交给 §S3 归一化）：

```yaml
# configs/data/arm_remap.yaml（节选）—— 每本体每臂一份；值为 [src_idx, sign, offset]，null=该槽缺失
r1_pro:   { left_arm: {shoulder_pitch: [0,1,0], shoulder_roll: [1,1,0], shoulder_yaw: [2,1,0],
                       elbow_pitch: [3,1,0], forearm_roll: [4,1,0], wrist_pitch: [5,1,0], wrist_roll: [6,1,0]} }
r1_lite:  { left_arm: {shoulder_pitch: [1,1,0], shoulder_roll: null,   shoulder_yaw: [0,1,0],
                       elbow_pitch: [2,1,0], forearm_roll: [4,1,0], wrist_pitch: [3,1,0], wrist_roll: [5,1,0]} }
g2:       { left_arm: {shoulder_pitch: [0,1,0], shoulder_roll: [1,1,0], shoulder_yaw: [2,1,0],   # sign/offset 待 FK 标定
                       elbow_pitch: [3,1,0], forearm_roll: [4,1,0], wrist_pitch: [5,1,0], wrist_roll: [6,1,0]} }
```

```python
import numpy as np

ARM_SLOTS = ['shoulder_pitch', 'shoulder_roll', 'shoulder_yaw',
             'elbow_pitch', 'forearm_roll', 'wrist_pitch', 'wrist_roll']

class ArmRemap:
    """把某本体某臂的原始关节向量 reorder 到统一 7 槽语义顺序。
       缺失槽位输出 0（由 ConcatLeftAlign 标占位 mask=1）。
       策略 A（位置式）：传入 identity 映射 + 不填角色即可；
       策略 B（语义对齐）：传入上表的 slot_map（含 sign/offset）。"""

    def __init__(self, slot_map):           # slot_map: {slot: [src_idx, sign, offset] | None}
        self.slot_map = slot_map

    def __call__(self, raw_arm):            # raw_arm: (..., n_src)
        out = np.zeros(raw_arm.shape[:-1] + (len(ARM_SLOTS),), dtype=raw_arm.dtype)
        for k, slot in enumerate(ARM_SLOTS):
            spec = self.slot_map.get(slot)
            if spec is None:                # 缺失自由度 -> 0，对应槽位由 mask 标占位
                continue
            src, sign, offset = spec
            out[..., k] = sign * raw_arm[..., src] + offset
        return out
```

**右臂镜像**：右臂用同一 `ARM_SLOTS` 顺序，只是 `src_idx` 偏到右臂原始组；左右臂的 `shoulder_roll`/`forearm_roll`/`wrist_roll` 等绕"长轴/侧向"的关节符号常相反，**右臂 sign 需相对左臂取镜像**（同样靠 FK 确认，不要假设）。

**sign / offset 怎么定**：加载每个 URDF，把机器人摆到统一中性姿态、逐关节单独转动，用观测到的**末端运动方向**反推每个槽位的 `sign`（让 +pitch=屈曲、+yaw 同向等），用中性姿态的读数定 `offset`。**不要用 XML 的 `axis` 字面判断**（尤其 G2 的 DH 表示、以及 R1 Pro `torso_joint3` 这类 `axis=0 -1 0` 的反号）。

**验证**（并入 §10）：
- round-trip：`ArmRemap` 的逆映射 + denormalize 能恢复原始关节向量。
- 语义一致性：三台机器人在同一中性姿态下，各臂槽位归一化前的符号/零位方向一致（可视化抽查）。
- 缺失维：R1 Lite 的 `shoulder_roll`（左臂 slot 1 / 右臂 slot 9）`action_dim_is_pad` 恒为 True 且值恒 0。

---

## 5. 端到端数据流水线

```
多源轮式人形原始数据（不同 fps / 相机 / 自由度）
   │
   ▼  S0  LeRobot 化 + Embodiment 登记（每源一个 embodiment_id + 拓扑）
   ▼  S1  频率重采样 → 统一 TARGET_FREQ（连续量线性插值，夹爪/离散最近邻）
   ▼  S2  语义槽位映射（per-embodiment action_state_transforms：原始维→统一语义顺序）
   ▼  S3  per-embodiment 归一化（各源独立统计，写 per-embodiment dataset_stats.json）
   ▼  S4  统一维 pad + 有效性 mask（ConcatLeftAlign → action_dim_is_pad）
   ▼  S5  视频/文本对齐（多相机 canvas、action_video_freq_ratio、T5 缓存）
   ▼  S6  模型侧 windowing（FastWAM 连续 horizon / AHA-WAM chunk+offset+history）
   ▼
两个模型的 dataloader 直接消费（统一 32 维连续动作 + mask）
```

### S0 — LeRobot 化 + Embodiment 登记

- 每个数据源转换为 LeRobot v2.1（`data/` parquet、`meta/`、`videos/`），期望 key：
  `observation.images.{cam}`、`observation.state.{...}`、`action.{...}`、`task`。
- 维护一张 **embodiment 注册表**（建议 `configs/data/embodiments.yaml`），每条目记录：
  - `control_freq`（源控制频率，Hz）
  - 原始动作 `groups`（部件名、维度、语义类型，如 `position` / `velocity` / `binary`）
  - **原始维 → 统一语义槽位**映射（供 S2）
  - 相机 → 统一画布槽位映射（供 S5）
  - `binary_dims`（夹爪等需最近邻重采样的维度）
  - **embodiment prompt 字段**（参照 Qwen-RobotManip §3.4，供 §6 P0-b 行为条件）：`embodiment`（平台名）、`instruction`、`speed`（episode 步数分箱，如每 500 一档）、`fps`、`camera view direction`（arm side / opposite）。训练时按 ~15% 概率随机丢 embodiment/speed/fps 字段，提升缺字段鲁棒性。**speed/fps 入 prompt 与 §S1 重采样互补**：重采样统一物理时长，prompt 再显式告知采样率/速度。

### S1 — 频率重采样（新建）

直接沿用开 doc 的 `resample_actions`（连续量线性插值、`binary_dims` 最近邻）。两种落地：

- **离线（推荐）**：在 S0 LeRobot 化阶段把每条轨迹重采样到 `TARGET_FREQ`（如 10Hz），写回 parquet。好处：绕开 `BaseLerobotDataset` 的「同 fps 断言」，dataloader 零改动。
- **在线**：作为 dataset transform；需放宽 `base_lerobot_dataset.py:18-124` 的 fps 断言并按源 fps 重采样。

> 为什么必须做：不同源同样 16 步在不同 fps 下代表不同物理时长，horizon 语义会错位（开 doc §数据对齐策略）。

### S1.5 — 信号 curation（参照 Qwen-RobotManip §2.4）

多源异构数据的 state/action 信号噪声很杂（碰撞突变、state-action 时序错位、极值、同型号不同关节约定）。Qwen 的五阶段过滤 + 三项跨模态检查可直接借用；**纯关节预训练（跳过 §4.0 的 Motion 轴）下，关节信号质量更要干净，否则脏 lag / 错符号直接污染 loss**。

**五阶段 state-action 过滤**
1. **突变检测**：每维求平滑趋势（中值 + Savitzky–Golay），再看残差 / 二阶差分（加速度）/ 三阶差分（jerk）；阈值按本体类型、real/sim、底盘是否移动分别设。**sim（如 R1 Pro）里碰撞引起的突变 → 整段 episode 丢弃**。
2. **state-action 时序对齐**：action 应在因果上领先/同步于 state 变化；按维互相关估 lag，算方向一致性 DA，DA<0.6~0.7 的 episode 剔除（delta action 先积分再比）。Qwen 靠这条发现 RoboMIND UR 子集 81% episode 不合格。
3. **极值过滤**：在分位归一化 `[q01,q99]→[-1,1]` 前剔除超出 `[q01−α·IQR, q99+α·IQR]` 的帧；**夹爪因双峰分布豁免**（与 §S3 夹爪不走 quantile 一致）。
4. **关节-末端 FK 一致性**：用 Pinocchio 从各本体 URDF 做 FK 对比记录的末端位姿。**这正是 §4.3 的 G2 关节符号/零偏标定要做的事**；Qwen 由此发现"同型号机器人在不同数据集关节约定不同"——以**校正**为主（调 TCP 定义、肩相对→世界系），而非一律丢弃。
5. **基座/朝向统一**：per-dataset 旋转校正，保证 +x 一致朝机器人正前方。

**三项跨模态检查**：① 指令一致性（VLM 多评委投票）；② video-state 一致性（用 URDF + 关节状态投影渲染机器人 mask，与分割 mask 比 IoU，低重叠剔除）；③ 视频质量过滤（黑/糊/长静止帧，保留夹爪开合等关键帧）。

> 对你这套（关节 + 世界模型），Stage 1/2/4 收益最大：1/2 保证关节信号本身干净、时序对齐；4 顺带把 G2 的符号标定一起做了。

### S2 — 语义槽位映射（per-embodiment transform）

- 为每个 embodiment 写一个 `action_state_transform`（类比现有 `transforms/libero.py` 的 `LiberoTransform`），把原始动作维**重排/搬运到统一语义顺序**（左臂→左手→右臂→右手→腰→头→底盘）。
- **臂内逐关节**的角色对齐（shoulder_pitch↔shoulder_pitch…）、6↔7 DoF 的 pad 位置、sign/offset 标定见 **§4.3**；`ArmRemap` 即在此阶段执行。
- 通过 `processor.embodiment_processors.{emb}.action_state_transforms` 配置注入（schema 已支持，见 §3.1）。
- **为什么不能只靠 ConcatLeftAlign**：`ConcatLeftAlign` 只按 key 顺序拼接 + 末尾 pad，不做跨本体语义对齐；语义对齐必须在它之前由 transform 完成。

### S3 — per-embodiment 归一化

- 沿用 `LinearNormalizer`（模式按部件选：臂位 `q01/q99` 或 `z-score`、底盘速度 `z-score`、夹爪 `min/max` 或常数 `-1/1`）。
- **关键改动：stats 按 embodiment 分别计算**（开 doc 的 per-embodiment 原则），扩展 `dataset_stats.json` 为：

```jsonc
{
  "embodiments": {
    "wheeled_humanoid_A": { "action": { "default": { "global_mean": [...], "global_std": [...], "global_q01": [...], "global_q99": [...] } },
                            "state":  { "default": { ... } } },
    "wheeled_humanoid_B": { "action": { "default": { ... } } }
  },
  "unified_dim": 32
}
```

- 归一化在**语义映射后、pad 前**进行，padding 维保持 0；mask=0 的维不参与统计。
- `pretrained_norm_stats` 指向该文件；首训设 `null` 自动生成，后续复用（README 流程一致）。

### S4 — 统一维 pad + 有效性 mask（复用 `ConcatLeftAlign`）

- 设 `action_target_dim = UNIFIED_DIM = 32`、`state_target_dim` 同理。
- `ConcatLeftAlign` 自动 pad 到 32 并产出 `action_dim_is_pad` / `state_dim_is_pad`。
- 处理器透出 `action`/`proprio`（`[T,32]`）、`action_is_pad`（`[T]` 时间）、`action_dim_is_pad`（`[32]` 维度）。
- 设 `processor.action_output_dim = proprio_output_dim = 32`（断言会校验形状）。

### S5 — 视频 / 文本对齐

- **多相机画布**：不同本体相机数/布局不同 → 为每本体定义相机→统一画布槽位映射，统一到固定 `concat_multi_camera` 布局与 `video_size`（如 RoboTwin 模式 3 相机）。缺失相机槽位填零帧并在文档记录（避免误当真实观测）。
- **时间对齐**：`action_video_freq_ratio` 必须满足 `(num_frames-1) % ratio == 0` 且 `((num_frames-1)//ratio) % 4 == 0`（tokenization 对齐约束，见 FastWAM 数据校验）。重采样后按统一 `TARGET_FREQ` 选择 `num_frames` / `ratio`。
- **文本**：沿用 `precompute_text_embeds.py`，按本体各自 `meta/tasks.jsonl` 去重哈希缓存；多本体可共享同一 `text_embedding_cache_dir`（哈希天然去重）。

### S6 — 模型侧 windowing 差异

| | FastWAM | AHA-WAM |
|---|---|---|
| 动作切窗 | 连续 horizon（如 32 步 @10Hz），与 9 视频帧对齐 | **chunk 化**：`action_chunk_size=16`，`action_horizon=64`（4 chunk） |
| 额外 | — | **offset 训练**：`max_action_offset=15`，需 `num_frames=97`；`num_history_frames=6`（OVCR 历史路由）；`history_aware_batching=true` |
| mask 透传 | `action_dim_is_pad` 需进入 loss | 同左；chunk/offset 维度上需对齐 mask 广播 |

上述差异**只在 windowing/采样层**，统一动作空间 + mask 的 S0–S4 完全共享。

---

## 6. 必须的代码改动（gap 清单）

按优先级：

### 🔴 P0-a — loss 屏蔽占位维度（否则统一空间不成立）
当前动作 loss 只屏蔽时间维。需把 `action_dim_is_pad` 广播进来：

```python
# FastWAM: src/fastwam/models/wan22/fastwam.py:550 附近
# AHA-WAM: src/ahawam/models/wan22/base_wam.py:1218 附近 (_compute_weighted_action_loss)
per_elem = F.mse_loss(pred_action.float(), target_action.float(), reduction="none")  # [B,T,D]
if action_dim_is_pad is not None:                       # [B,D] or [D]
    dim_valid = (~action_dim_is_pad).to(per_elem.dtype) # [.,D]
    per_elem = per_elem * dim_valid.unsqueeze(-2)       # 广播到 [B,T,D]
    denom = dim_valid.sum(-1).clamp(min=1.0)            # 每样本有效维数
    action_loss_token = per_elem.sum(-1) / denom        # [B,T]，按有效维归一
else:
    action_loss_token = per_elem.mean(dim=2)            # 原行为
# 后续保持原有 action_is_pad 时间 mask + training_weight 不变
```
> 注意 batch collate 需把 `action_dim_is_pad` 一并带到模型 `forward` / `training_step`。

### 🔴 P0-b — ActionDiT 注入 embodiment 条件（「原生 + 标签」的命根子）
不加这个，同槽位跨本体的矛盾语义会把模型搞乱。最小实现（参照 ActionCodec soft-prompt）：

```python
# action_dit.py：新增 per-embodiment 可学习嵌入
self.embodiment_embed = nn.Embedding(num_embodiments, hidden_dim)
# forward：把 embodiment 嵌入加进条件（与 timestep/text 同路）
cond = cond + self.embodiment_embed(embodiment_id)        # [B, hidden_dim] 广播到 token
```
- `embodiment_id` 由 P0 的多本体 Dataset 提供，经 collate 进 `forward`。
- MoT 里视频专家可同样注入（可选），让世界预测也条件于本体。
- 备选：把 embodiment 名拼进 text prompt（零模型改动，但条件信号弱、不推荐做主手段）。

### 🔴 P0-c — 多本体 Dataset（拼接 + embodiment 标签）
新建 `MultiEmbodimentVideoDataset`（包在 `BaseLerobotDataset` 外）：
- 读取 `embodiment_datasets`（schema 已存在），逐本体实例化子 dataset；
- 每条样本注入 `embodiment_id`（同时供 P0-b 模型条件 + P1 normalizer 选择 + mask 生成）；
- 支持**采样配比**（按本体或按数据量加权，避免长尾本体被淹没——开 doc DataValidator 的 imbalance 检查可复用）。
- 每个数据集对应一个 §4.0 的 adapter（语义槽位映射 transform），经 `embodiment_processors.{emb}.action_state_transforms` 注入。

### 🟠 P1 — per-embodiment 统计量
扩展 `base_lerobot_dataset.py:251-373` 的 `get_dataset_stats`：按 embodiment 分桶计算，写入 §S3 的扩展 `dataset_stats.json`；`LinearNormalizer` 按样本 `embodiment_id` 取对应 stats。

### 🟠 P1 — 频率重采样
S1 离线方案优先（dataloader 零改）；若在线，放宽 `base_lerobot_dataset.py` 的同 fps 断言并加重采样 transform。

### 🟢 P2 — config 接线
`configs/model/{fastwam,ahawam}.yaml` 的 `action_dim` / `state_dim` 改为：
```yaml
action_dim: ${max_action_dim:${data.train.dataset.embodiment_datasets}}
```
ActionDiT encoder/decoder 维度随之自动对齐（`action_dit.py` 的 `action_encoder`/`head`）。

---

## 7. 配置样例（节选）

```yaml
# configs/data/wheeled_humanoid_pretrain.yaml
train:
  _target_: fastwam.datasets.lerobot.robot_video_dataset.MultiEmbodimentVideoDataset  # 新建
  dataset:
    action_size: 32            # = UNIFIED_DIM
    embodiment_datasets:
      wheeled_humanoid_A:
        control_freq: 20       # 源频率（S1 重采样到 TARGET_FREQ）
        shape_meta:
          action: [{key: default, raw_shape: 22, shape: 32}]   # 原始22 → 统一32
          state:  [{key: default, raw_shape: 22, shape: 32}]
        dataset_groups:
          - dataset_dirs: [./data/wh_A/lerobot]
      wheeled_humanoid_B:
        control_freq: 30
        shape_meta:
          action: [{key: default, raw_shape: 20, shape: 32}]
          state:  [{key: default, raw_shape: 20, shape: 32}]
        dataset_groups:
          - dataset_dirs: [./data/wh_B/lerobot]
  num_frames: 33
  action_video_freq_ratio: 4   # 重采样后 @TARGET_FREQ 选定，满足对齐约束
  pretrained_norm_stats: ./data/pretrain/dataset_stats.json   # per-embodiment 扩展版
  processor:
    action_output_dim: 32
    proprio_output_dim: 32
    norm_default_mode: "z-score"
    action_state_merger:
      _target_: fastwam.datasets.lerobot.transforms.action_state_merger.ConcatLeftAlign
    embodiment_processors:                # schema 已支持
      wheeled_humanoid_A:
        action_state_transforms:
          - _target_: fastwam.datasets.lerobot.transforms.wh_a.WheeledHumanoidATransform   # 语义槽位映射
      wheeled_humanoid_B:
        action_state_transforms:
          - _target_: fastwam.datasets.lerobot.transforms.wh_b.WheeledHumanoidBTransform
  text_embedding_cache_dir: ./data/text_embeds_cache/pretrain
  context_len: 128
```

```yaml
# configs/model/fastwam.yaml（关键一行）
action_dit_config:
  action_dim: ${max_action_dim:${data.train.dataset.embodiment_datasets}}   # → 32
```

---

## 8. 产物清单（喂给训练）

| 产物 | 形状 / 内容 | 生成阶段 |
|---|---|---|
| LeRobot 数据（已重采样到 `TARGET_FREQ`） | `data/`+`meta/`+`videos/`，含 `embodiment_id` | S0–S1 |
| `dataset_stats.json`（per-embodiment 扩展版） | 见 §S3 结构 | S3，首训自动生成 |
| 统一动作张量 | `action`：`[T, 32]`，连续，归一化后落 ±范围 | S2–S4（在线） |
| 有效性 mask | `action_dim_is_pad`：`[32]`；`action_is_pad`：`[T]` | S4 |
| 文本嵌入缓存 | `{hash}.t5_len128.*.pt` → `{context[128,4096], mask[128]}` | S5 |
| ActionDiT backbone | `ActionDiT_linear_interp_Wan22_*.pt`（`action_dim=32`） | `preprocess_action_dit_backbone.py` |

---

## 9. 落地步骤与命令

```bash
# 1) S0–S1：各源转 LeRobot 并离线重采样到 TARGET_FREQ（自写脚本，复用 resample_actions）
python tools/to_lerobot_and_resample.py --src ... --target-freq 10 --out ./data/wh_A/lerobot

# 2) 写 embodiment registry + 语义槽位 transform（configs/data/embodiments.yaml + transforms/wh_*.py）

# 3) 实现 P0/P1 代码改动（多本体 Dataset、per-embodiment stats、loss 维度 mask）

# 4) 预生成 ActionDiT backbone（action_dim 经 resolver = 32）
python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/fastwam.yaml \
  --output checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim_32dim.pt --dtype bfloat16

# 5) 预算文本嵌入缓存
torchrun --standalone --nproc_per_node=8 scripts/precompute_text_embeds.py task=wheeled_humanoid_pretrain

# 6) 首训（pretrained_norm_stats=null 自动产 per-embodiment stats），再复用
bash scripts/train_zero1.sh 8 task=wheeled_humanoid_pretrain
```

AHA-WAM 同流程，task 换成对应的 `*_ahawam` / `*_ahawam_offset`（注意 offset 配置需 `num_frames=97`、`num_history_frames=6`）。

---

## 10. 验证与检查清单

- [ ] **频率**：所有源轨迹已重采样到 `TARGET_FREQ`（抽查物理时长一致）。
- [ ] **语义对齐**：随机抽本体，确认「右臂」永远落在统一槽 `(8,15)`（可视化/断言）。
- [ ] **mask 正确**：`action_dim_is_pad` 与本体拓扑一致；padding 维数值恒为 0。
- [ ] **loss 用了维度 mask**：占位维梯度为 0（单测：人为污染 padding 维，loss 不变）。
- [ ] **embodiment 条件生效**：`embodiment_id` 从 Dataset 一路传到 ActionDiT；不同本体嵌入不同（单测：换 id 输出应变）。
- [ ] **adapter round-trip**：每个数据集 adapter 抽样可逆（反槽位映射 + denormalize 恢复原生动作）。
- [ ] **per-embodiment stats**：`dataset_stats.json` 每本体独立；归一化后真实维落入合理范围（沿用开 doc DataValidator 的 `[-1,1]`/finite/imbalance 检查，阈值按 `z-score` 的 ±5 调整）。
- [ ] **per-channel mask 覆盖率**：逐维统计 `action_dim_is_pad==False` 的样本占比（按 embodiment + 全局）。`reserved (27,32)` 必须恒为 0；头/腰等高维槽位若某 channel 全局有效占比过低（阈值如 <5%）需告警——否则该维只学到单一本体或几乎全占位，码本/权重无意义。详见下方 `per_channel_coverage`。
- [ ] **配比**：本体间样本量比 ≤ 10×（否则加采样权重）。
- [ ] **视频对齐约束**：`(num_frames-1) % ratio == 0` 且 `((num_frames-1)//ratio) % 4 == 0`。
- [ ] **可逆**：`denormalize` + 反槽位映射能恢复原始本体动作（端到端 round-trip 单测）。

### per-channel mask 覆盖率（实现）

`pad_ratio` 是全局标量，会掩盖「某些维几乎全占位」的问题；下面按**统一维**逐列统计有效占比，并按 embodiment 拆分，定位只有单一本体供数或基本恒占位的槽位。

```python
import numpy as np

# 语义槽位边界（与 §4.1 UNIFIED_ACTION_SPACE 对齐），用于把逐维结果归到部件
UNIFIED_ACTION_SPACE = {
    'left_arm': (0, 7), 'left_hand': (7, 8), 'right_arm': (8, 15), 'right_hand': (15, 16),
    'torso': (16, 21), 'head': (21, 24), 'base': (24, 27), 'reserved': (27, 32),
}

def per_channel_coverage(dim_is_pad, embodiments, unified_dim=32,
                         low_thresh=0.05, slots=UNIFIED_ACTION_SPACE):
    """
    Args:
        dim_is_pad:   (N, unified_dim) bool/0-1，True=占位（取每样本的 action_dim_is_pad，
                      与时间无关 -> 每个样本一行即可）
        embodiments:  (N,) 每个样本的 embodiment_id
        low_thresh:   全局有效占比低于此值的「非 reserved」维度触发告警
    Returns:
        dict: 全局逐维有效占比、按本体逐维有效占比、告警列表
    """
    dim_is_pad = np.asarray(dim_is_pad, dtype=bool)
    valid = ~dim_is_pad                                  # True=真实维
    embs = np.asarray(embodiments)

    global_cov = valid.mean(axis=0)                      # (D,) 每维全局有效占比
    per_emb_cov = {e: valid[embs == e].mean(axis=0)      # 每本体每维有效占比
                   for e in np.unique(embs)}

    # 每维由几个本体供数（该本体在该维有效占比 > 0）
    n_emb_supplying = sum((c > 0).astype(int) for c in per_emb_cov.values())

    reserved = slots.get('reserved', (unified_dim, unified_dim))
    warnings = []
    for d in range(unified_dim):
        in_reserved = reserved[0] <= d < reserved[1]
        if in_reserved:
            if global_cov[d] > 0:                        # 预留维不该有有效样本
                warnings.append(f"dim {d} (reserved) 有效占比 {global_cov[d]:.3f} > 0，疑似映射越界")
        else:
            if global_cov[d] < low_thresh:
                warnings.append(f"dim {d} 全局有效占比 {global_cov[d]:.3f} < {low_thresh}")
            if n_emb_supplying[d] <= 1:
                warnings.append(f"dim {d} 仅 {int(n_emb_supplying[d])} 个本体供数，跨本体共享无意义")

    # 按语义部件聚合，便于人读
    by_part = {part: float(global_cov[s:e].mean()) for part, (s, e) in slots.items()}

    return {
        'global_coverage': global_cov,        # (D,)
        'per_embodiment_coverage': per_emb_cov,
        'coverage_by_part': by_part,
        'n_embodiments_supplying': n_emb_supplying,
        'warnings': warnings,
    }
```

> `action_dim_is_pad` 只依赖 embodiment、不随时间变（§4.2），所以每个样本取一行即可；可在 `create_dataset` 末尾或独立 audit 脚本里跑。reserved(27,32) 有效占比应恒为 0——若非 0，说明某本体的语义映射越界写进了预留区。

1. **语义映射人工成本**：每新增本体要写一个 `action_state_transform` + registry 条目。建议做成「声明式映射表 + 通用 transform」，避免逐本体写类。
2. **底盘速度 vs 关节位置混在一个 loss**：量纲差异大。per-embodiment 归一化 + 按部件选归一化模式可缓解；必要时对 base/torso 单独加 loss 权重。
3. **占位维占比**：pad 到 32 后语义只占 27、且多数本体缺头/腰，统一 32 维里 mask=0 占比会偏高（尤其 reserved 5 维恒占位），注意 §10 的 `pad_ratio` 统计；若过高且无扩展计划，可收缩 reserved / `UNIFIED_DIM`。
4. **AHA-WAM offset/history 与 mask 广播**：chunk + offset 重排时务必让 `action_dim_is_pad` 跟随同样的 reshape，否则维度 mask 与动作错位。
5. **（可选）未来 RVQ 分支**：若后续要离散动作 token，可在 S4 之后接开 doc 的 `ActionPatchifier`（`(m·n, h·d)`）产出 patch 训练 RVQ tokenizer——但那需要改 ActionDiT 为离散输入，属另一条 roadmap，不在本连续方案内。

---

## 附：开 doc 五步 → WAM hook 速查

| 开 doc 步骤 | WAM 落点（文件/字段） | 改动 |
|---|---|---|
| 频率重采样 | S0/S1，离线写回 parquet | 新建脚本 |
| per-embodiment 归一化 | `LinearNormalizer` + `dataset_stats.json` | 扩展为 per-embodiment |
| 语义分组/对齐 | `embodiment_processors.{emb}.action_state_transforms` | 每本体一个 transform |
| 统一维 + mask | `ConcatLeftAlign` → `action_dim_is_pad` | 复用（设 target_dim=32） |
| mask 入训练 | `fastwam.py` / `base_wam.py` 动作 loss | **P0 必改** |
| `action_dim` 接线 | `${max_action_dim:${...embodiment_datasets}}` | config 一行 |
| patchify/RVQ | — | 不做（可选未来分支） |
