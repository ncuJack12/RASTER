#!/usr/bin/env bash
set -euo pipefail

REPO=${A100_REPO:-/wjy/cuvs}
SUITE=${A100_SUITE:-results/range_cagra/paper_full_suite/a100_fast90_full_v3_20260608_complete}
LOG_DIR=$REPO/$SUITE/logs
LOG=$LOG_DIR/complete_lxc_guarded.log
PIDFILE=$LOG_DIR/complete_lxc_guarded.pid

mkdir -p "$LOG_DIR"

if [[ -s "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "already running: pid=$(cat "$PIDFILE")"
  echo "log=$LOG"
  exit 0
fi

BIN_DIR=$REPO/cpp/build-a100-sm80-cuda129/gtests
if [[ -x "$BIN_DIR/NEIGHBORS_RANGE_CAGRA_SEGMENT_TREE_TEST_codex_20260607" ]]; then
  ln -sfn NEIGHBORS_RANGE_CAGRA_SEGMENT_TREE_TEST_codex_20260607 \
    "$BIN_DIR/NEIGHBORS_RANGE_CAGRA_SEGMENT_TREE_TEST"
fi

cd "$REPO"
nohup python3 results/range_cagra/run_a100_full_suite_guarded.py \
  --suite-dir "$SUITE" \
  --gpu-id "${A100_GPU_ID:-0}" \
  >"$LOG" 2>&1 &
echo $! >"$PIDFILE"
echo "started pid=$(cat "$PIDFILE")"
echo "log=$LOG"
