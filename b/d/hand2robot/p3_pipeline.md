# P3 Pipeline 与 VLA A/B 准备

## 范围

- MotionSmooth（前置）→ Render → Caption → Export
- `hand_action_tags` 在 Render / Caption / Export 后保持不变
- LeRobot 视频来自 `robot_render_frames`
- A/B manifest 定义 `baseline / arm-no-depth / arm-depth`

## 验收

```bash
bash b/scripts/hand2robot/check.sh p3
```

报告：

- `b/d/hand2robot/runs/accept_p3/accept_p3_pipeline_report.json`

## Recipe

`b/scripts/hand2robot/configs/ego_to_robot_recipe.yaml` 已串起：

1. `video_hand_motion_smooth_mapper`
2. `video_hand_to_robot_render_mapper`
3. `video_hand_action_caption_stub_mapper`
4. `export_robot_render_lerobot_mapper`

全链路：

```bash
bash b/scripts/hand2robot/process.sh
```

## VLA A/B manifest

```bash
DATASET=/path/to/data.jsonl BUILD_VLA_MANIFEST=1 \
  bash b/scripts/hand2robot/process.sh
# 或单独：
bash b/scripts/hand2robot/internal/08_build_vla_ab_manifest.sh
```

当前只生成 manifest，不直接跑训练。
