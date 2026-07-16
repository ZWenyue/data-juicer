#!/usr/bin/env bash
# 把 robot_data_clean.sh 产出的 cleaned.jsonl（保留 episode）转成完整 LeRobot v2.1 80 维布局。
#
# 输出（与 tests_au/ops/mapper/export_unified_to_process_test.sh 一致）：
#   OUT/<task>/
#     data/chunk-XXX/episode_YYYYYY.parquet   # observation.state/action/mask 各 80 维
#     meta/{info.json,episodes.jsonl,tasks.jsonl,episodes_stats.jsonl}
#     videos -> 源 videos（symlink）
#
# 用法：
#   bash robot_data_export_unified.sh
#   CLEAN_ROOT=... OUT=... bash robot_data_export_unified.sh
set -euo pipefail

CLEAN_ROOT="${CLEAN_ROOT:-/mnt/r/DATA/Galaxea-Open-World-Dataset/process_clean/260711}"
SRC_ROOT="${DATASET_ROOT:-/mnt/r/DATA/Galaxea-Open-World-Dataset/260711}"
OUT="${OUT_ROOT:-/mnt/r/DATA/Galaxea-Open-World-Dataset/process_clean/260711_unified80}"
PY="${DJ_VENV:-/mnt/r/VENV/dj}/bin/python"

cd "$(dirname "$0")"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUT"

# robot_type (info.json) → embodiment YAML stem under configs/embodiments/
embodiment_from_robot_type() {
  case "$1" in
    r1lite|r1_lite) echo "galaxea_r1_lite" ;;
    r1pro|r1_pro)   echo "galaxea_r1_pro" ;;
    *)              echo "" ;;
  esac
}

n_ok=0; n_skip=0
for d in "$CLEAN_ROOT"/*/; do
  d="${d%/}"; task="$(basename "$d")"
  cleaned="$d/cleaned.jsonl"
  src="$SRC_ROOT/$task"
  [[ -s "$cleaned" ]] || { echo "[SKIP] $task (无 cleaned.jsonl)"; n_skip=$((n_skip+1)); continue; }
  [[ -d "$src/data" ]] || { echo "[SKIP] $task (源任务不存在: $src)"; n_skip=$((n_skip+1)); continue; }

  rt="$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1])).get('robot_type',''))" "$src/meta/info.json" 2>/dev/null || echo "")"
  emb="$(embodiment_from_robot_type "$rt")"
  if [[ -z "$emb" ]]; then
    echo "[SKIP] $task (robot_type=$rt 无 embodiment 布局)"
    n_skip=$((n_skip+1))
    continue
  fi

  echo "=== $task → $OUT/$task (robot_type=$rt, embodiment=$emb)"
  "$PY" -m data_juicer._au.pipeline.robot_clean.export_unified \
    --cleaned "$cleaned" \
    --dataset "$src" \
    --output "$OUT/$task" \
    --embodiment "$emb"
  n_ok=$((n_ok+1))
done

echo "==================================================================="
echo "完成。exported=$n_ok  skipped=$n_skip"
echo "最终目录: $OUT/<task>/{data,meta,videos}"
