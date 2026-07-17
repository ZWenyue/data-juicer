#!/usr/bin/env bash
# 把 robot_data_clean.sh 产出的 cleaned.jsonl（保留 episode）转成完整 LeRobot v2.1 80 维布局。
#
# 输出（与 tests_au/ops/mapper/export_unified_to_process_test.sh 一致）：
#   OUT/<task>/
#     data/chunk-XXX/episode_YYYYYY.parquet   # observation.state/action/action_dim_mask 各 80 维
#     meta/{info.json,episodes.jsonl,tasks.jsonl,episodes_stats.jsonl}
#     videos -> 源 videos（symlink）
#
# 用法：
#   bash robot_data_export_unified.sh
#   CLEAN_ROOT=... OUT=... bash robot_data_export_unified.sh
set -euo pipefail

CLEAN_ROOT="${CLEAN_ROOT:-/mnt/r/DATA/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/process_clean}"
SRC_ROOT="${DATASET_ROOT:-/mnt/r/DATA/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/press}"
OUT="${OUT_ROOT:-/mnt/r/DATA/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/process_clean/unified80}"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${DJ_VENV:-$REPO_ROOT/.venv}/bin/python"

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUT"

# robot_type / embodiment_tag → YAML stem under configs/embodiments/
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
  esac
  local rt="$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1])).get('robot_type',''))" "$info" 2>/dev/null || echo "")"
  case "$rt" in
    r1lite|r1_lite) echo "galaxea_r1_lite" ;;
    r1pro|r1_pro)   echo "galaxea_r1_pro" ;;
    R1Pro)          echo "sim_behavior_r1_pro" ;;
    agilex_cobot_decoupled_magic|agilex_cobot_magic) echo "agilex_cobot_magic" ;;
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
  emb="$(embodiment_from_meta "$src/meta/info.json")"
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
