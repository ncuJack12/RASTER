#!/usr/bin/env bash
set -euo pipefail

REPO=${A100_REPO:-/hdZFS/subvol-125-disk-1/cuvs}
SUITE=${A100_SUITE:-results/range_cagra/paper_full_suite/a100_fast90_full_v3_20260608_complete}
CHROOT_EXEC=${A100_CHROOT_EXEC:-$REPO/$SUITE/a100_host_chroot_exec.sh}
LOG_DIR=$REPO/$SUITE/logs
LOG=$LOG_DIR/complete_host_chroot_guarded.log
PIDFILE=$LOG_DIR/complete_host_chroot_guarded.pid

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

nohup "$CHROOT_EXEC" python3 results/range_cagra/run_a100_full_suite_guarded.py \
  --suite-dir "$SUITE" \
  --gpu-id "${A100_GPU_ID:-0}" \
  >"$LOG" 2>&1 &
echo $! >"$PIDFILE"
echo "started pid=$(cat "$PIDFILE")"
echo "log=$LOG"
