#!/usr/bin/env bash
# 批量清洗 Galaxea 多任务 root 下的每个 LeRobot 任务。
# r1lite → 完整流程(含 unified80 + parquet)；其它体态 → 仅数值+Check3。
#
# 若 ANALYZE_ROOT/<task>/analysis.json 存在，自动使用分析给出的建议旗标；
# 否则回退到默认 BLUR_TH（及命令行透传 "$@"）。
#
# 典型流程：
#   bash robot_data_analyze.sh
#   bash robot_data_clean.sh
set -euo pipefail

ROOT="${DATASET_ROOT:-/mnt/r/DATA/Galaxea-Open-World-Dataset/260711}"
OUT="${OUT_ROOT:-/mnt/r/DATA/Galaxea-Open-World-Dataset/process_clean/260711}"
ANALYZE_ROOT="${ANALYZE_ROOT:-/mnt/r/DATA/Galaxea-Open-World-Dataset/process_clean/260711_analyze}"
PY="${DJ_VENV:-/mnt/r/VENV/dj}/bin/python"
NP="${NP:-16}"
# 仅当某任务没有 analysis.json 时使用
BLUR_TH="${BLUR_TH:-20}"

cd "$(dirname "$0")"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

# 从 analysis.json 解析 suggested_clean_flags → bash 数组元素（空格分隔）
flags_from_analysis() {
  local aj="$1"
  "$PY" -c "
import json, sys
d = json.load(open(sys.argv[1]))
flags = d.get('suggested_clean_flags') or {}
parts = []
for k, v in flags.items():
    parts.append(str(k))
    parts.append(str(v))
print(' '.join(parts))
" "$aj"
}

for d in "$ROOT"/*/; do
  d="${d%/}"; task="$(basename "$d")"
  [[ -d "$d/data" ]] || continue

  rt="$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1])).get('robot_type',''))" "$d/meta/info.json")"
  if [[ "$rt" == r1lite || "$rt" == r1_lite ]]; then
    emb=(--embodiment galaxea_r1_lite --export-unified-parquet)
  else
    echo "[WARN] $task: robot_type=$rt 无布局配置 → 跳过 unified"
    emb=(--embodiment "galaxea_$rt" --no-unified)
  fi

  aj="$ANALYZE_ROOT/$task/analysis.json"
  if [[ -f "$aj" ]]; then
    # shellcheck disable=SC2206
    thresh=( $(flags_from_analysis "$aj") )
    echo "=== $task (robot_type=$rt, 使用分析旗标: ${thresh[*]})"
  else
    thresh=(--check3-blur-threshold "$BLUR_TH")
    echo "=== $task (robot_type=$rt, 无 analysis.json → blur=$BLUR_TH)"
  fi

  "$PY" -m data_juicer._au.pipeline.robot_clean.run_robot_clean \
    --dataset "$d" --output "$OUT/$task" --np "$NP" \
    "${emb[@]}" "${thresh[@]}" "$@"
done
