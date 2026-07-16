#!/usr/bin/env bash
# 批量分析 Galaxea 多任务 root 下每个 LeRobot 任务，给出建议的清洗阈值。
# 逐任务输出 analysis.json / threshold_report.md，并汇总建议的 clean 命令行旗标。
# 先跑这个定参，再用同样的旗标跑 robot_data_clean.sh。
set -euo pipefail

ROOT="${DATASET_ROOT:-/mnt/r/DATA/Galaxea-Open-World-Dataset/260711}"
OUT="${OUT_ROOT:-/mnt/r/DATA/Galaxea-Open-World-Dataset/process_clean/260711_analyze}"
PY="${DJ_VENV:-/mnt/r/VENV/dj}/bin/python"
PROBE_EPS="${PROBE_EPS:-8}"           # Check3 视频探针每任务评分的 episode 数
PROBE_FPS="${PROBE_FPS:-2}"           # 视频探针抽帧 fps

cd "$(dirname "$0")"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUT"
SUMMARY="$OUT/suggested_flags.txt"; : > "$SUMMARY"

# robot_type (info.json) → embodiment YAML stem under configs/embodiments/
embodiment_from_robot_type() {
  case "$1" in
    r1lite|r1_lite) echo "galaxea_r1_lite" ;;
    r1pro|r1_pro)   echo "galaxea_r1_pro" ;;
    *)              echo "" ;;
  esac
}

for d in "$ROOT"/*/; do
  d="${d%/}"; task="$(basename "$d")"
  [[ -d "$d/data" ]] || continue

  rt="$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1])).get('robot_type',''))" "$d/meta/info.json")"
  emb="$(embodiment_from_robot_type "$rt")"
  if [[ -z "$emb" ]]; then
    echo "[SKIP] $task (robot_type=$rt 无 embodiment 布局)"
    continue
  fi

  echo "=== $task (robot_type=$rt, embodiment=$emb)"
  flags="$("$PY" -m data_juicer._au.pipeline.robot_clean.analyze \
    --dataset "$d" --output "$OUT/$task" --embodiment "$emb" \
    --probe-video-episodes "$PROBE_EPS" --probe-sampling-fps "$PROBE_FPS" "$@" | tail -1)"
  printf '%s\t%s\n' "$task" "$flags" >> "$SUMMARY"
  echo "  建议旗标: $flags"
done

echo "==================================================================="
echo "分析完成。逐任务报告: $OUT/<task>/threshold_report.md"
echo "建议旗标汇总: $SUMMARY"
