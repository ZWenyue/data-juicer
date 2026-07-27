# Hand→Robot 脚本（对外 4 个入口）

把「人手操作数据集 → R1 Lite 机器人视角 LeRobot」收成固定流程。设计说明：

- `b/d/hand2robot/hand_to_robot_code_design_summary.md`
- `b/d/hand2robot/hand_to_robot_render_design.md`

旧的编号脚本已移到 `internal/`，日常只用下面 4 个。

## 四个入口

| 脚本 | 何时用 |
|------|--------|
| `setup.sh` | 第一次 / 换机器：生成左右臂 MJCF |
| `calibrate.sh` | 新人手数据集需要标定 YAML |
| `process.sh` | **主入口**：数据集 → 机器人画面 + LeRobot（含 finalize） |
| `check.sh` | 工程验收（p3/p4/depth/smoke） |

## 典型流程（EgoDex 举例）

```bash
cd /mnt/r/share/zwy/Projects/data-juicer

# 0) 一次性
bash b/scripts/hand2robot/setup.sh

# 1) 标定（已有 r1_*_egodex_v1.yaml 可跳过）
DATASET=/mnt/r/DATA/EgoDex/test_lerobot SIDE=right EPISODE=2 \
  bash b/scripts/hand2robot/calibrate.sh

# 2) 处理一批（先小规模）
DATASET=/mnt/r/DATA/EgoDex/test_lerobot SIDE=right MAX_VIDEOS=2 \
  bash b/scripts/hand2robot/process.sh

# 3) 可选：工程自检
bash b/scripts/hand2robot/check.sh
```

双臂：

```bash
DATASET=/mnt/r/DATA/EgoDex/test_lerobot SIDE=both MAX_VIDEOS=2 \
  CALIBRATION_PATH=b/d/hand2robot/calibration/r1_both_egodex_v1.yaml \
  bash b/scripts/hand2robot/process.sh
```

## `process.sh` 输入

`DATASET` 可以是：

- EgoDex LeRobot 根目录（自动从 `videos/observation.images.ego` 收 mp4 → jsonl）
- 普通视频目录 / 单个 mp4
- 已有 jsonl（每行 `{"videos":["..."], "text":"", ...}`）

常用变量：

| 变量 | 默认 | 含义 |
|------|------|------|
| `DATASET` | （必填） | 输入数据 |
| `SIDE` | `right` | `left` / `right` / `both` |
| `CALIBRATION_PATH` | `calibration/r1_${SIDE}_egodex_v1.yaml` | 标定 YAML |
| `MAX_VIDEOS` | 全部 | 只处理前 N 条视频 |
| `OFFSET` | 0 | 跳过前 N 条 |
| `RUN_DIR` | `runs/process_${SIDE}_时间戳` | 本次输出目录 |
| `SKIP_FINALIZE` | 0 | 设为 1 则只写 staging |
| `BUILD_VLA_MANIFEST` | 0 | 设为 1 额外写 VLA A/B manifest |
| `MUJOCO_GL` | `egl` | MuJoCo 后端 |

产物（在 `RUN_DIR/`）：

- `input_dataset.jsonl` — 本次实际喂给 pipeline 的样本
- `frames/` / `robot_frames/` — 原帧 / 机器人合成帧
- `lerobot_dataset/` — 最终 LeRobot（finalize 后有 `meta/info.json`）

单臂导出 state **8** / action **7**；`SIDE=both` 为 state **16** / action **14**（right\|\|left）。

## `calibrate.sh` 模式

| `SOURCE` | 说明 |
|----------|------|
| `egodex`（默认） | `DATASET` / `EGODEX_ROOT` 指向 EgoDex LeRobot |
| `ego` | 已有 pipeline sample：`DATA_PATH=...pkl` |
| `galaxea` | 真机 FK/IK 代理标定 |
| `synthetic` | 无数据 smoke |

## `check.sh`

```bash
bash b/scripts/hand2robot/check.sh        # p3 + p4
bash b/scripts/hand2robot/check.sh all
bash b/scripts/hand2robot/check.sh depth
bash b/scripts/hand2robot/check.sh smoke
```

## 注意

1. 全链路仍走 recipe：抽帧 → MoGe → HaWoR → MegaSaM → action → smooth → render → caption(stub) → export。需相应权重与环境。
2. EgoDex 的 300-D 手势真值目前用于**标定**；`process.sh` 从视频重新重建手，不直接读 parquet state。
3. 生产 caption 请把 recipe 里的 stub 换成真实 VLM caption mapper。
4. 低层脚本在 `internal/`，一般无需直接调用。
