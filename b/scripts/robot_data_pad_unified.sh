#!/usr/bin/env bash
# 后训练用：对原始 LeRobot 任务做 80 维 padding，不做任何数据清洗。
#
# 与 robot_data_clean.sh 的区别：
#   - 不跑 Stage1/2/3/5 / Check3
#   - 保留全部 episode
#   - 仍写出与预训练相同的 unified80 LeRobot v2.1 布局
#
# 用法：
#   bash robot_data_pad_unified.sh
#   DATASET_ROOT=... OUT_ROOT=... NP=8 bash robot_data_pad_unified.sh
#   MAX_EPS=2 bash robot_data_pad_unified.sh   # smoke
set -euo pipefail

ROOT="${DATASET_ROOT:-/mnt/r/DATA/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/press}"
OUT="${OUT_ROOT:-/mnt/r/DATA/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/process_pad_unified80}"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${DJ_VENV:-$REPO_ROOT/.venv}/bin/python"
MAX_EPS="${MAX_EPS:-}"

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUT"

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
  local rt="$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1])).get('robot_type',''))" "$info" 2>/dev/null || echo "")"
  case "$rt" in
    r1lite|r1_lite) echo "galaxea_r1_lite" ;;
    r1pro|r1_pro)   echo "galaxea_r1_pro" ;;
    R1Pro)          echo "sim_behavior_r1_pro" ;;
    agilex_cobot_decoupled_magic|agilex_cobot_magic) echo "agilex_cobot_magic" ;;
    aloha)          echo "aloha" ;;
    *)              echo "" ;;
  esac
}

n_ok=0; n_skip=0
for d in "$ROOT"/*/; do
  d="${d%/}"; task="$(basename "$d")"
  [[ -d "$d/data" ]] || continue
  [[ -f "$d/meta/info.json" ]] || {
    echo "[SKIP] $task (无 meta/info.json)"
    n_skip=$((n_skip+1))
    continue
  }

  rt="$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1])).get('robot_type',''))" "$d/meta/info.json" 2>/dev/null || echo "")"
  emb="$(embodiment_from_meta "$d/meta/info.json")"
  if [[ -z "$emb" ]]; then
    echo "[SKIP] $task (robot_type=$rt 无 embodiment 布局)"
    n_skip=$((n_skip+1))
    continue
  fi

  echo "=== $task → $OUT/$task (robot_type=$rt, embodiment=$emb, keep-all)"
  args=(
    --keep-all
    --dataset "$d"
    --output "$OUT/$task"
    --embodiment "$emb"
  )
  if [[ -n "$MAX_EPS" ]]; then
    args+=(--max-episodes "$MAX_EPS")
  fi

  "$PY" -m data_juicer._au.pipeline.robot_clean.export_unified "${args[@]}"
  n_ok=$((n_ok+1))
done

echo "==================================================================="
echo "完成。padded=$n_ok  skipped=$n_skip"
echo "最终目录: $OUT/<task>/{data,meta,videos}"
echo "可继续: ROOT_SIM=$OUT ROOT_GLX=... bash robot_data_merge_lerobot.sh"
