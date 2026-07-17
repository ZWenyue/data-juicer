#!/usr/bin/env bash
# 合并多个 unified80 LeRobot 根目录 → 单一 LeRobot v2.1 数据集。
#
# 默认合并:
#   PhysicalAI .../process_clean/unified80
#   Galaxea      .../process_clean/260711_unified80
#
# 输出:
#   OUT/
#     data/chunk-XXX/episode_YYYYYY.parquet   # episode_index / index 重编号
#     meta/{info.json,episodes.jsonl,tasks.jsonl,episodes_stats.jsonl,sources.jsonl}
#     videos/chunk-XXX/<video_key>/episode_YYYYYY.mp4  # 默认 symlink
#     merge_summary.json
#
# 用法:
#   bash robot_data_merge_lerobot.sh
#   OUT=/path/to/merged bash robot_data_merge_lerobot.sh
#   VIDEO_POLICY=none bash robot_data_merge_lerobot.sh          # 只合并 parquet
#   DRY_RUN=1 bash robot_data_merge_lerobot.sh
#   MAX_TASKS=1 MAX_EPS=2 bash robot_data_merge_lerobot.sh      # 小规模试跑
set -euo pipefail

ROOT_SIM="${ROOT_SIM:-/mnt/r/DATA/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/process_clean/unified80/use}"
ROOT_GLX="${ROOT_GLX:-/mnt/r/DATA/Galaxea-Open-World-Dataset/process_clean/260711_unified80/use}"
OUT="${OUT_ROOT:-/mnt/r/DATA/merged_unified80}"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${DJ_VENV:-$REPO_ROOT/.venv}/bin/python"
VIDEO_POLICY="${VIDEO_POLICY:-keep}"
LINK_MODE="${LINK_MODE:-symlink}"
MAX_TASKS="${MAX_TASKS:-}"
MAX_EPS="${MAX_EPS:-}"
DRY_RUN="${DRY_RUN:-0}"

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

ARGS=(
  --root "$ROOT_SIM"
  --root "$ROOT_GLX"
  --output "$OUT"
  --video-policy "$VIDEO_POLICY"
  --link-mode "$LINK_MODE"
)
if [[ -n "$MAX_TASKS" ]]; then
  ARGS+=(--max-tasks-per-root "$MAX_TASKS")
fi
if [[ -n "$MAX_EPS" ]]; then
  ARGS+=(--max-episodes-per-task "$MAX_EPS")
fi
if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" ]]; then
  ARGS+=(--dry-run)
fi

echo "=== Merge LeRobot unified80 ==="
echo "ROOT_SIM=$ROOT_SIM"
echo "ROOT_GLX=$ROOT_GLX"
echo "OUT=$OUT  video_policy=$VIDEO_POLICY  link_mode=$LINK_MODE"
echo "MAX_TASKS=${MAX_TASKS:-all}  MAX_EPS=${MAX_EPS:-all}  DRY_RUN=$DRY_RUN"

"$PY" -m data_juicer._au.pipeline.robot_clean.merge_lerobot "${ARGS[@]}"

echo "==================================================================="
echo "完成。输出: $OUT"
