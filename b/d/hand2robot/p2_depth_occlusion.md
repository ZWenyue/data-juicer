# P2 Depth-aware 遮挡

## 退出条件

| 检查 | 门槛 | 本次 |
|------|------|------|
| 机器人在前可见 | robot_depth ≤ scene + ε 时绘制 | pass |
| 场景在前遮挡 | scene 更近时保留背景 | pass |
| 无效 depth 门控 | mask 内无效占比 >20% → `depth_invalid` | pass |
| Aligner 仿射拟合 | `metric ≈ scale·scene + bias` | pass |
| Mapper 渲染路径 | far 可见 ≥ near 可见；invalid_ratio <0.2 | pass |

**decision: go**（报告：`b/d/hand2robot/runs/accept_depth/accept_depth_occlusion_report.json`）

## 实现要点

- `composite_with_depth`：对齐 scene depth 后按 $D_{robot} \le D_{scene}+\epsilon$ 生成可见 mask
- `DepthAligner`：每 clip 用 MANO 腕/指尖 z 对 MoGe depth 做仿射拟合；不足样本时退回 identity
- Mapper：`enable_depth_occlusion=true` 时渲染 robot metric depth；缺 depth 时降级 mask blend 并标 `depth_missing`
- Recipe：`b/scripts/hand2robot/configs/ego_to_robot_recipe.yaml` 已打开深度遮挡

## 复现

```bash
bash b/scripts/hand2robot/05_accept_depth.sh
```
