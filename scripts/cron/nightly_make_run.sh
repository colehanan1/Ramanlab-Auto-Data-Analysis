#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="$REPO_DIR/logs/cron"
CONDA_ROOT="/home/ramanlab/anaconda3"
CONDA_SH="$CONDA_ROOT/etc/profile.d/conda.sh"
CONDA_ENV_NAME="yolo-env"
mkdir -p "$LOG_DIR"

TS="$(date +"%Y%m%d_%H%M%S")"
LOG_FILE="$LOG_DIR/make_run_$TS.log"

export PATH="$CONDA_ROOT/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
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

if [[ ! -f "$CONDA_SH" ]]; then
  echo "[$(date -Is)] Missing conda activation script: $CONDA_SH"
  exit 1
fi

source "$CONDA_SH"
conda activate "$CONDA_ENV_NAME"

NTFY_TOPIC="ramanlab-pipeline"

echo "[$(date -Is)] Starting nightly make run"
echo "[$(date -Is)] Conda env: ${CONDA_DEFAULT_ENV:-unknown}"
echo "[$(date -Is)] Python: $(command -v python3)"
if OUTPUT=$(make run 2>&1); then
  EXIT_CODE=0
  echo "$OUTPUT"
  echo "[$(date -Is)] Finished nightly make run with exit code $EXIT_CODE"
  curl -s \
    -H "Title: Pipeline completed" \
    -H "Priority: default" \
    -H "Tags: white_check_mark" \
    -d "Nightly make run finished successfully at $(date -Is)" \
    "https://ntfy.sh/$NTFY_TOPIC" > /dev/null 2>&1
else
  EXIT_CODE=$?
  echo "$OUTPUT"
  echo "[$(date -Is)] Finished nightly make run with exit code $EXIT_CODE"
  # Get last 10 lines of output for error context
  ERROR_TAIL=$(echo "$OUTPUT" | tail -n 10)
  curl -s \
    -H "Title: Pipeline FAILED (exit code $EXIT_CODE)" \
    -H "Priority: high" \
    -H "Tags: x" \
    -d "Nightly make run failed at $(date -Is)

Error output:
$ERROR_TAIL" \
    "https://ntfy.sh/$NTFY_TOPIC" > /dev/null 2>&1
fi
exit $EXIT_CODE
