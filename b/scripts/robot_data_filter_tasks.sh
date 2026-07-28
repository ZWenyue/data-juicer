#!/usr/bin/env bash
# 按 task 目录名和 meta/tasks.jsonl 标签筛选完整 LeRobot 任务。
#
# 默认筛选开门任务并以软链接组成子集；不读取、切分或改写 episode/帧数据。
#
# 用法：
#   DATASET_ROOT=/path/to/unified80 OUT_ROOT=/path/to/open_door \
#     bash robot_data_filter_tasks.sh
#   SKILL="" KEYWORDS="插网线,connect router cable" MODE=copy \
#     DATASET_ROOT=... OUT_ROOT=... bash robot_data_filter_tasks.sh
set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-/mnt/r/DATA/Galaxea-Open-World-Dataset/lerobot}"
OUT="${OUT_ROOT:-/mnt/r/DATA/Galaxea-Open-World-Dataset/task_subsets/open_door}"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${DJ_VENV:-$REPO_ROOT/.venv}/bin/python"

SKILL="${SKILL:-open_door}"
KEYWORDS="${KEYWORDS:-}"
EXCLUDE_KEYWORDS="${EXCLUDE_KEYWORDS:-}"
MODE="${MODE:-symlink}"
ON_EXISTING="${ON_EXISTING:-error}"
DRY_RUN="${DRY_RUN:-0}"
CASE_SENSITIVE="${CASE_SENSITIVE:-0}"
INCLUDE_TASK_LABELS="${INCLUDE_TASK_LABELS:-1}"

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

args=(
  --dataset-root "$DATASET_ROOT"
  --output "$OUT"
  --mode "$MODE"
  --on-existing "$ON_EXISTING"
)
if [[ -n "$SKILL" ]]; then
  args+=(--skill "$SKILL")
fi
if [[ -n "$KEYWORDS" ]]; then
  args+=(--keyword "$KEYWORDS")
fi
if [[ -n "$EXCLUDE_KEYWORDS" ]]; then
  args+=(--exclude-keyword "$EXCLUDE_KEYWORDS")
fi
if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" ]]; then
  args+=(--dry-run)
fi
if [[ "$CASE_SENSITIVE" == "1" || "$CASE_SENSITIVE" == "true" ]]; then
  args+=(--case-sensitive)
fi
if [[ "$INCLUDE_TASK_LABELS" == "1" || "$INCLUDE_TASK_LABELS" == "true" ]]; then
  args+=(--include-task-labels)
else
  args+=(--task-name-only)
fi

echo "=== Filter LeRobot tasks ==="
echo "DATASET_ROOT=$DATASET_ROOT"
echo "OUT=$OUT"
echo "SKILL=${SKILL:-custom}  KEYWORDS=${KEYWORDS:-none}"
echo "MODE=$MODE  ON_EXISTING=$ON_EXISTING  INCLUDE_TASK_LABELS=$INCLUDE_TASK_LABELS  DRY_RUN=$DRY_RUN"

"$PY" -m data_juicer._au.pipeline.filter_robot_tasks "${args[@]}"

echo "==================================================================="
if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" ]]; then
  echo "Dry-run 完成，未创建输出目录或清单。"
else
  echo "完成。筛选清单: $OUT/task_filter_results.jsonl"
  echo "筛选摘要: $OUT/task_filter_summary.json"
  echo "可继续: DATASET_ROOT=$OUT bash b/scripts/robot_data_analyze.sh"
fi
