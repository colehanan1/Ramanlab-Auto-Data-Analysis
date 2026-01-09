#!/bin/bash
# Run pseudolabel in video batches to avoid crashes
# Usage: bash scripts/train/pseudolabel_batch.sh [batch_num]

BATCH_NUM=${1:-1}
TOTAL_BATCHES=5

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/config/config.yaml}"

DATASET_OUT="${DATASET_OUT:-}"
if [ -z "${DATASET_OUT}" ] && [ -f "${CONFIG_PATH}" ]; then
    DATASET_OUT="$(python - <<'PY' "${CONFIG_PATH}"
from pathlib import Path
import sys
try:
    import yaml
except ImportError:
    print("")
    raise SystemExit(0)

path = Path(sys.argv[1])
data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
value = data.get("pseudolabel", {}).get("dataset_out") or ""
print(value)
PY
)"
fi
if [ -z "${DATASET_OUT}" ]; then
    DATASET_OUT="${REPO_ROOT}/data/pseudolabel_dataset"
fi
TARGET_PER_BATCH=10000  # 10k per batch = 50k total
BATCH_SIZE=2
STRIDE=15
MIN_CONF=0.85

# Calculate video stride to process different videos each batch
VIDEO_STRIDE=$((TOTAL_BATCHES))
VIDEO_OFFSET=$((BATCH_NUM - 1))

LOG_FILE="${REPO_ROOT}/logs/pseudolabel_batch${BATCH_NUM}_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "${REPO_ROOT}/logs"

echo "=========================================="
echo "Pseudolabel Batch $BATCH_NUM/$TOTAL_BATCHES"
echo "=========================================="
echo "This batch will process every ${VIDEO_STRIDE}th video, starting at offset $VIDEO_OFFSET"
echo "Target: $TARGET_PER_BATCH frames"
echo "Output: $DATASET_OUT"
echo "Log: $LOG_FILE"
echo ""

# Run with random sampling to get different frames each batch
RANDOM_SEED=$((1337 + BATCH_NUM))

cd "${REPO_ROOT}"

nohup nice -n 10 \
  env PYTHONPATH=src python -m fbpipe.steps.pseudolabel_export \
  --config config/config.yaml \
  --dataset-out "$DATASET_OUT" \
  --target-total "$TARGET_PER_BATCH" \
  --batch-size "$BATCH_SIZE" \
  --random-sample-per-video 50 \
  --min-conf "$MIN_CONF" \
  --val-frac 0.1 \
  --label-format obb \
  --seed "$RANDOM_SEED" \
  > "$LOG_FILE" 2>&1 &

PID=$!
echo "✓ Batch $BATCH_NUM started (PID: $PID)"
echo ""
echo "Monitor: tail -f $LOG_FILE"
echo "Stop: kill $PID"
echo ""
echo "After this batch completes, run:"
echo "  bash scripts/train/pseudolabel_batch.sh $((BATCH_NUM + 1))"
