# Galaxea Full Corpus Threshold Analyze (Pointer JSONL)

**日期**：2026-07-13  
**状态**：实现中 / 全量作业已启动

## 目标

对 227 个 Galaxea LeRobot 任务做 Stage 1/2/3 分析并给出清洗阈值建议；JSONL **只存 `parquet_path`**，运行时由 Loader Mapper 读 parquet。

## 架构

```
pointer JSONL (~7.6MB / 22626 eps)
  → robot_lerobot_parquet_loader_mapper (TAGGING_OPS)
  → Stage1/2/3 Filters
  → threshold_report.md + clean_recipe_suggested.yaml
```

全局 `percentiles.json` 直接扫全部任务 parquet（不经 JSONL）。

## 关键文件

- `data_juicer/_au/utils/lerobot_episode_io.py`
- `data_juicer/_au/ops/mapper/robot_lerobot_parquet_loader_mapper.py`
- `tests_au/ops/filter/convert_lerobot_episodes.py` (`--mode pointer --root`)
- `tests_au/ops/filter/compute_embodiment_percentiles.py` (`--root`)
- `tests_au/ops/filter/analyze_qwenrobomanip_full.yaml`
- `tests_au/ops/filter/accept_analyze_qwenrobomanip_full.sh`
- 输出：`tests_au/ops/filter/outputs/full_analyze/`

## 运行

```bash
bash tests_au/ops/filter/accept_analyze_qwenrobomanip_full.sh
# 日志
tail -f tests_au/ops/filter/outputs/full_analyze/run.log
```

## 备注

- 手臂维支持 6-DoF（pad slot 1/9）与 7-DoF（填满 0..6 / 8..14）。
- `dj-analyze` 需 Loader 注册为 `TAGGING_OPS` 才会执行。
