# P3 Pipeline 与 VLA A/B 准备

## 范围

- **Pipeline**：MotionSmooth（前置）→ Render → Caption → Export
- **动作不变性**：`hand_action_tags` 在 Render/Caption/Export 后数值不变
- **观测视频**：LeRobot episode 由 `robot_render_frames` 编码，而非原始 ego mp4
- **VLA A/B**：manifest 定义 baseline / arm-no-depth / arm-depth 三路配置

## 验收

```bash
bash b/scripts/hand2robot/05_accept_p3.sh
```

报告：`b/d/hand2robot/runs/accept_p3/accept_p3_pipeline_report.json`

## Recipe

`b/scripts/hand2robot/configs/ego_to_robot_recipe.yaml` 已包含：

1. `video_hand_motion_smooth_mapper`
2. `video_hand_to_robot_render_mapper`（P2 depth on）
3. `video_hand_action_caption_stub_mapper`（smoke；生产换 VLM）
4. `export_robot_render_lerobot_mapper`

全链路：`bash b/scripts/hand2robot/07_process_ego_to_robot.sh`

## VLA A/B manifest

```bash
bash b/scripts/hand2robot/08_build_vla_ab_manifest.sh
```

产出 `manifest.json`，含三路 `process_hints` 与 `lerobot_dir` 规划路径。
实际训练与 hold-out 指标需在 StarVLA / 自有 trainer 上按设计 §8.5 执行。

## 新增算子（`_au`）

| 算子 | 作用 |
|------|------|
| `video_hand_action_caption_stub_mapper` | 无 VLM 的任务描述占位 |
| `export_robot_render_lerobot_mapper` | 从渲染帧路径编码 episode 视频 |
