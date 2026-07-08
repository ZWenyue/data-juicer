# Galaxea Open-World Dataset 分析 — Connect_Router_Cables

> 使用 Data-Juicer 对 [Galaxea Open-World Dataset](https://arxiv.org/abs/2509.00576) 的 Connect_Router_Cables 任务进行数据质量分析与处理管道设计。

## 数据集

- **位置**: `/mnt/r/DATA/Galaxea-Open-World-Dataset/lerobot_door/Connect_Router_Cables_20250625_002/`
- **格式**: LeRobot v2.1 统一格式 (56-dim state / 50-dim action)
- **规模**: 125 episodes, 209K frames, 4 cameras × 720p@15fps

## 文件

| 文件 | 说明 |
|------|------|
| [analysis.md](analysis.md) | 完整分析报告（数据需求、格式解析、质量评估、提升建议） |
| [analyze_glx.py](analyze_glx.py) | 质量分析脚本（SAL 平滑度、DA 对齐、异常检测） |
| [analysis_results.json](analysis_results.json) | 分析脚本输出的 JSON 报告 |
| [dj_analyze.yaml](dj_analyze.yaml) | Data-Juicer 分析器配置 |
| [convert_to_jsonl.py](convert_to_jsonl.py) | Episode 元数据 → DJ JSONL 转换 |

## 快速开始

```bash
# 运行质量分析
/mnt/r/VENV/dj/bin/python b/dm/glxOpnWld/analyze_glx.py

# 运行 DJ 文本标注分析
/mnt/r/VENV/dj/bin/python b/dm/glxOpnWld/convert_to_jsonl.py
/mnt/r/VENV/dj/bin/dj-analyze --config b/dm/glxOpnWld/dj_analyze.yaml
```

## 关键发现

- Episode 0 混入无关任务（冰箱操作），需移除
- 44.9% 帧标记为 unqualified
- 32/125 episodes 存在 state-action 方向不一致
- SAL 中位数 −420（中等偏好平滑度）
