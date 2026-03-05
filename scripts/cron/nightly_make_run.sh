#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="$REPO_DIR/logs/cron"
mkdir -p "$LOG_DIR"

TS="$(date +"%Y%m%d_%H%M%S")"
LOG_FILE="$LOG_DIR/make_run_$TS.log"

export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
cd "$REPO_DIR"

if command -v flock >/dev/null 2>&1; then
  LOCK_FILE="$REPO_DIR/.cron_make_run.lock"
  exec 9>"$LOCK_FILE"
  if ! flock -n 9; then
    echo "[$(date -Is)] Another run is active; exiting." >> "$LOG_FILE"
    exit 0
  fi
fi

exec > >(tee -a "$LOG_FILE") 2>&1

echo "[$(date -Is)] Starting nightly make run"
make run
EXIT_CODE=$?
echo "[$(date -Is)] Finished nightly make run with exit code $EXIT_CODE"
exit $EXIT_CODE
