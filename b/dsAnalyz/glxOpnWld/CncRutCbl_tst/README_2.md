# Galaxea Open-World Dataset 分析 — Connect_Router_Cables (Data-Juicer 版)

> 使用 Data-Juicer 自定义算子对 [Galaxea Open-World Dataset](https://arxiv.org/abs/2509.00576) 的 Connect_Router_Cables 任务进行数据质量分析与处理管道设计。**所有分析和清洗功能均通过 DJ 实现**，无外部脚本依赖。

## 数据集

- **位置**: `/mnt/r/DATA/Galaxea-Open-World-Dataset/lerobot_door/Connect_Router_Cables_20250625_002/`
- **格式**: LeRobot v2.1 统一格式 (56-dim state / 50-dim action)
- **规模**: 125 episodes, 209K frames, 4 cameras × 720p@15fps

## 文件

| 文件 | 说明 |
|------|------|
| [analysis_2.md](analysis_2.md) | 完整分析报告（数据需求、格式解析、质量评估、DJ 管道设计） |
| [custom_ops_2.py](custom_ops_2.py) | DJ 自定义算子（SAL、DA、突变检测、质量标签、统计） |
| [analyze_glx_2.py](analyze_glx_2.py) | DJ Python API 分析脚本（生成 JSON 报告） |
| [analysis_results_2.json](analysis_results_2.json) | 分析脚本输出的完整 JSON 报告 |
| [dj_analyze_2.yaml](dj_analyze_2.yaml) | Data-Juicer YAML 配置（分析/处理模式） |
| [convert_to_jsonl_2.py](convert_to_jsonl_2.py) | LeRobot 数据 → DJ per-episode JSONL 转换 |
| [episodes_2.jsonl](episodes_2.jsonl) | 转换后的 per-episode 数据 |

## 快速开始

```bash
# 1. 转换数据为 DJ 格式
/mnt/r/VENV/dj/bin/python b/dm/glxOpnWld/convert_to_jsonl_2.py

# 2. 运行完整质量分析（DJ Python API）
/mnt/r/VENV/dj/bin/python b/dm/glxOpnWld/analyze_glx_2.py

# 3. 或通过 DJ 命令行运行分析/处理
/mnt/r/VENV/dj/bin/dj-analyze --config b/dm/glxOpnWld/dj_analyze_2.yaml
/mnt/r/VENV/dj/bin/dj-process --config b/dm/glxOpnWld/dj_analyze_2.yaml
```

## 自定义算子

| 算子名 | 类型 | 功能 |
|--------|------|------|
| `episode_off_task_filter` | Filter | 关键词匹配过滤无关任务 episode |
| `trajectory_sal_filter` | Filter | SAL 频域平滑度分析与过滤 |
| `trajectory_da_filter` | Filter | State-action 方向一致性检测 |
| `trajectory_sudden_change_filter` | Filter | 加速度突变检测 |
| `quality_label_filter` | Filter | Qualified/unqualified 帧比例统计 |
| `trajectory_stats_mapper` | Mapper | 关节范围、夹爪分析等统计 |

## 关键发现

- Episode 0 混入无关任务（冰箱操作），需移除
- 44.9% 帧标记为 unqualified
- 32/125 episodes 存在 state-action 方向不一致
- SAL 中位数 −420（中等偏好平滑度）
