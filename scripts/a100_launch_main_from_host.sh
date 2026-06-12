#!/usr/bin/env bash
set -euo pipefail

REPO=${A100_REPO:-/hdZFS/subvol-125-disk-1/cuvs}
SUITE=${A100_SUITE:-results/range_cagra/paper_full_suite/a100_fast90_full_v2_20260607}
CHROOT_EXEC=${A100_CHROOT_EXEC:-$REPO/$SUITE/a100_host_chroot_exec.sh}
RESUME_SCRIPT=${A100_RESUME_SCRIPT:-$SUITE/commands_gpu0_main_expanded_search_resume_from_arxiv_20260608.sh}
LOG_DIR=$REPO/$SUITE/logs
LOG=$LOG_DIR/host_chroot_main_expanded_resume_from_arxiv_20260608.log
PIDFILE=$LOG_DIR/host_chroot_main_expanded_resume_from_arxiv_20260608.pid

mkdir -p "$LOG_DIR"

if [[ -s "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "already running: pid=$(cat "$PIDFILE")"
  exit 0
fi

nohup "$CHROOT_EXEC" bash "$RESUME_SCRIPT" >"$LOG" 2>&1 &
echo $! >"$PIDFILE"
echo "started pid=$(cat "$PIDFILE")"
echo "log=$LOG"
