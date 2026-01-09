#!/bin/bash
# Safe pseudolabel export that won't crash your display
# Run with: bash scripts/train/safe_pseudolabel.sh

set -e

# Configuration
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
TARGET_TOTAL=50000
BATCH_SIZE=2  # Small batch to prevent GPU overload
STRIDE=15     # Sample every 15th frame
MIN_CONF=0.85

# Ensure we're in the right directory
cd "${REPO_ROOT}"

# Set up logging
LOG_FILE="${REPO_ROOT}/logs/pseudolabel_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "${REPO_ROOT}/logs"

echo "=========================================="
echo "Safe Pseudolabel Export"
echo "=========================================="
echo "Output: $DATASET_OUT"
echo "Target: $TARGET_TOTAL frames"
echo "Batch size: $BATCH_SIZE (low to prevent GPU crash)"
echo "Log file: $LOG_FILE"
echo "=========================================="
echo ""
echo "This will run in the background."
echo "Monitor progress with: tail -f $LOG_FILE"
echo ""

# Run with nice priority to not starve display
nohup nice -n 10 \
  env PYTHONPATH=src python -m fbpipe.steps.pseudolabel_export \
  --config config/config.yaml \
  --dataset-out "$DATASET_OUT" \
  --target-total "$TARGET_TOTAL" \
  --batch-size "$BATCH_SIZE" \
  --stride "$STRIDE" \
  --min-conf "$MIN_CONF" \
  --val-frac 0.1 \
  --label-format obb \
  --seed 1337 \
  > "$LOG_FILE" 2>&1 &

PID=$!

echo "✓ Started in background (PID: $PID)"
echo ""
echo "Commands:"
echo "  Monitor: tail -f $LOG_FILE"
echo "  Stop:    kill $PID"
echo "  GPU:     watch -n 2 nvidia-smi"
echo ""
echo "The script will automatically stop at 50k frames."
echo "Your display should remain stable."
