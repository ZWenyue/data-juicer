#!/usr/bin/env bash
# 批量清洗 LeRobot 任务（Galaxea / GR00T sim 等）。
# 已支持布局的体态 → 完整流程(含 unified80 + parquet)；
# 其它体态 → 仅数值+Check3。
#
# 若 ANALYZE_ROOT/<task>/analysis.json 存在，自动使用分析给出的建议旗标；
# 否则回退到默认 BLUR_TH（及命令行透传 "$@"）。
#
# 典型流程：
#   bash robot_data_analyze.sh
#   bash robot_data_clean.sh
set -euo pipefail

ROOT="${DATASET_ROOT:-/mnt/r/DATA/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/press}"
OUT="${OUT_ROOT:-/mnt/r/DATA/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/process_clean}"
ANALYZE_ROOT="${ANALYZE_ROOT:-/mnt/r/DATA/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/process_clean/analyze}"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${DJ_VENV:-$REPO_ROOT/.venv}/bin/python"
NP="${NP:-16}"
# 仅当某任务没有 analysis.json 时使用
BLUR_TH="${BLUR_TH:-1}"

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Prefer embodiment.json tag when present (GR00T R1Pro vs Galaxea r1pro).
embodiment_from_meta() {
  local info="$1"
  local emb_json="$(dirname "$info")/embodiment.json"
  local tag=""
  if [[ -f "$emb_json" ]]; then
    tag="$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1])).get('embodiment_tag') or '')" "$emb_json" 2>/dev/null || true)"
  fi
  case "$tag" in
    sim_behavior_r1_pro|behavior_r1_pro) echo "sim_behavior_r1_pro"; return ;;
    agilex_cobot_magic|agilex_cobot_decoupled_magic) echo "agilex_cobot_magic"; return ;;
    aloha) echo "aloha"; return ;;
  esac
  local rt="$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1])).get('robot_type',''))" "$info")"
  case "$rt" in
    r1lite|r1_lite) echo "galaxea_r1_lite" ;;
    r1pro|r1_pro)   echo "galaxea_r1_pro" ;;
    R1Pro)          echo "sim_behavior_r1_pro" ;;
    agilex_cobot_decoupled_magic|agilex_cobot_magic) echo "agilex_cobot_magic" ;;
    aloha)          echo "aloha" ;;
    *)              echo "" ;;
  esac
}

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
  emb_name="$(embodiment_from_meta "$d/meta/info.json")"
  if [[ -n "$emb_name" ]]; then
    emb=(--embodiment "$emb_name" --export-unified-parquet)
  else
    echo "[WARN] $task: robot_type=$rt 无布局配置 → 跳过 unified"
    emb=(--embodiment "galaxea_${rt:-unknown}" --no-unified)
  fi

  aj="$ANALYZE_ROOT/$task/analysis.json"
  if [[ -f "$aj" ]]; then
    analysis_version="$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1])).get('analysis_version', 0))" "$aj")"
    if [[ "$analysis_version" -lt 3 ]]; then
      echo "[ERROR] $task: analysis.json 由旧版阈值策略生成，拒绝复用以避免漏删或误删。"
      echo "        请先重新运行 b/scripts/robot_data_analyze.sh（analysis: $aj）"
      exit 2
    fi
    # shellcheck disable=SC2206
    thresh=( $(flags_from_analysis "$aj") )
    echo "=== $task (robot_type=$rt, embodiment=${emb_name:-none}, 使用分析旗标: ${thresh[*]})"
  else
    thresh=(--check3-blur-threshold "$BLUR_TH")
    echo "=== $task (robot_type=$rt, embodiment=${emb_name:-none}, 无 analysis.json → blur=$BLUR_TH)"
  fi

  "$PY" -m data_juicer._au.pipeline.robot_clean.run_robot_clean \
    --dataset "$d" --output "$OUT/$task" --np "$NP" \
    "${emb[@]}" "${thresh[@]}" "$@"
done
