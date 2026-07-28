#!/usr/bin/env bash
# Engineering acceptance checks (render / depth / p3 / p4 / smoke).
#
# Usage:
#   bash b/scripts/hand2robot/check.sh              # default: p3 + p4
#   bash b/scripts/hand2robot/check.sh all
#   bash b/scripts/hand2robot/check.sh p3
#   bash b/scripts/hand2robot/check.sh p4
#   bash b/scripts/hand2robot/check.sh depth
#   bash b/scripts/hand2robot/check.sh render
#   bash b/scripts/hand2robot/check.sh smoke
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_env.sh"
INTERNAL="$SCRIPT_DIR/internal"

TARGET="${1:-default}"

run_one() {
  local name="$1"
  echo
  echo "########## check: $name ##########"
  case "$name" in
    render)   OUT_DIR="${OUT_DIR:-$OUT_ROOT/accept_render}" bash "$INTERNAL/05_accept_render.sh" ;;
    calibrate) OUT_DIR="${OUT_DIR:-$OUT_ROOT/accept_calibrate}" bash "$INTERNAL/05_accept_calibrate.sh" synthetic ;;
    depth)    OUT_DIR="${OUT_DIR:-$OUT_ROOT/accept_depth}" bash "$INTERNAL/05_accept_depth.sh" ;;
    p3)       OUT_DIR="${OUT_DIR:-$OUT_ROOT/accept_p3}" bash "$INTERNAL/05_accept_p3.sh" ;;
    p4)       OUT_DIR="${OUT_DIR:-$OUT_ROOT/accept_p4}" bash "$INTERNAL/05_accept_p4.sh" ;;
    smoke)    bash "$INTERNAL/06_run_smoke.sh" ;;
    *)
      echo "unknown check: $name" >&2
      exit 2
      ;;
  esac
}

case "$TARGET" in
  default)
    run_one p3
    run_one p4
    ;;
  all)
    run_one render
    run_one calibrate
    run_one depth
    run_one p3
    run_one p4
    ;;
  *)
    run_one "$TARGET"
    ;;
esac

echo
echo "Check finished: $TARGET"
