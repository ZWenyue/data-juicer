# P1 真人手标定指标（EgoDex）

> 数据集：`/mnt/r/DATA/EgoDex/test_lerobot`  
> 脚本：`b/scripts/hand2robot/04_calibrate_egodex.sh`  
> Episode：2 · `MAX_FRAMES=120` · `STRIDE=4`

## 退出条件（设计文档 P1）

| 指标 | 门槛 | 右臂 | 左臂 |
|------|------|------|------|
| 标定帧数 | ≥ 100 | 120 | 120 |
| IK 成功率 | ≥ 90% | **90%** | **100%** |
| 重投影中位误差 | &lt; 15 px | **0.04 px** | **0.02 px** |
| IK 位置误差中位 | （参考） | 3.6 mm | 3.5 mm |
| decision | go | **go** | **go** |

### 右臂

- 报告：`b/d/hand2robot/runs/calib_egodex_right_ep000002/calibrate_hand_to_robot_report.json`
- YAML：`b/d/hand2robot/calibration/r1_right_egodex_v1.yaml`
- `p1_pass: true`

### 左臂

- 报告：`b/d/hand2robot/runs/calib_egodex_left_ep000002/calibrate_hand_to_robot_report.json`
- YAML：`b/d/hand2robot/calibration/r1_left_egodex_v1.yaml`
- `p1_pass: true`
- 关键：EgoDex 用 FK@`q_reference` 对齐 mean wrist 初始化 `camera_to_base`；评估 IK 用 multistart

## 数据与约定

- EgoDex `observation.state`（300-D）为左右手各 25 关节 × (xyz+rpy)，位姿在 **相机系**
- 标定默认 `camera_as_world=True`：手腕状态编码为 $T_{\mathrm{adapt}} T_{\mathrm{cam},wrist}$，与 MJCF / YAML adapter 对齐
- `camera.intrinsics` 按视频分辨率 480×270 从全分辨率主点缩放
- 低置信度腕关节帧（`min_wrist_conf`）丢弃

## 复现

```bash
# 右臂
EGODEX_ROOT=/mnt/r/DATA/EgoDex/test_lerobot \
EPISODE=2 SIDE=right MAX_FRAMES=120 STRIDE=4 \
  bash b/scripts/hand2robot/04_calibrate_egodex.sh

# 左臂
EGODEX_ROOT=/mnt/r/DATA/EgoDex/test_lerobot \
EPISODE=2 SIDE=left MAX_FRAMES=120 STRIDE=4 \
  bash b/scripts/hand2robot/04_calibrate_egodex.sh
```
