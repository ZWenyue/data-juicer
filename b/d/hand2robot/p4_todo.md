# P4 双臂数据工程代办

## 目标

先完成 **P4-engineering**：

- `hand_type=both` 可跑通
- 可稳定产出双臂 `robot_render_frames`
- 通过双臂工程验收（`check.sh p4`）
- 双臂 LeRobot 导出：`observation.state` **16** = right(8)||left(8)，`action` **14** = right(7)||left(7)

后续有训练算力后，再推进 **P4-validated**：

- 跑双臂 VLA A/B
- 验证相对 baseline / 单臂不退化

## 待办

1. ~~开放 `video_hand_to_robot_render_mapper.hand_type=both`~~
2. ~~双臂状态分别维护：`q_prev` / `hold_count` / `base_T_world`~~
3. ~~双臂渲染结果按深度融合（`merge_render_layers`）~~
4. ~~双臂手部 mask 与 inpaint 联合~~
5. ~~双臂 quality summary，含 `per_side`~~
6. ~~增加双臂 acceptance（`accept_p4_pipeline`）~~
7. ~~双臂导出 schema 完整化（16/14 + modality/info）~~
8. 双臂 caption / VLA 验证后补（需 GPU）

## 当前进度

- P4-engineering：**完成**（验收：`bash b/scripts/hand2robot/check.sh p4`）
- 导出约定：帧取左右 `valid_frame_ids` **交集**；缺臂时回退单臂 8/7
- finalize：由 `process.sh` 自动调用 `ExportRobotRenderLeRobotMapper.finalize_dataset`
- 日常入口：`setup.sh` / `calibrate.sh` / `process.sh` / `check.sh`（见 `b/scripts/hand2robot/README.md`）
