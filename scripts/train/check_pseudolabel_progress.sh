#!/bin/bash
# Check pseudolabel dataset progress

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
TARGET=50000

echo "=========================================="
echo "Pseudolabel Progress Check"
echo "=========================================="
echo ""

if [ ! -d "$DATASET_OUT" ]; then
    echo "Dataset directory not found: $DATASET_OUT"
    echo "No frames collected yet."
    exit 0
fi

# Count images
TRAIN_COUNT=$(find "$DATASET_OUT/images/train" -name "*.jpg" -o -name "*.png" 2>/dev/null | wc -l)
VAL_COUNT=$(find "$DATASET_OUT/images/val" -name "*.jpg" -o -name "*.png" 2>/dev/null | wc -l)
TOTAL=$((TRAIN_COUNT + VAL_COUNT))

echo "Images collected:"
echo "  Train: $TRAIN_COUNT"
echo "  Val:   $VAL_COUNT"
echo "  Total: $TOTAL / $TARGET"
echo ""

# Calculate percentage
PERCENT=$((TOTAL * 100 / TARGET))
echo "Progress: $PERCENT% complete"
echo ""

# Show progress bar
BAR_WIDTH=50
FILLED=$((TOTAL * BAR_WIDTH / TARGET))
if [ $FILLED -gt $BAR_WIDTH ]; then
    FILLED=$BAR_WIDTH
fi
printf "["
printf "%${FILLED}s" | tr ' ' '='
printf "%$((BAR_WIDTH - FILLED))s" | tr ' ' '-'
printf "]\n"
echo ""

# Check manifest
if [ -f "$DATASET_OUT/manifest.csv" ]; then
    MANIFEST_ROWS=$(wc -l < "$DATASET_OUT/manifest.csv")
    echo "Manifest entries: $((MANIFEST_ROWS - 1))"  # -1 for header
    echo ""
fi

# Estimate remaining
if [ $TOTAL -gt 0 ]; then
    REMAINING=$((TARGET - TOTAL))
    echo "Remaining: $REMAINING frames"

    if [ $REMAINING -gt 0 ]; then
        echo ""
        echo "Run again to continue:"
        echo "  bash scripts/train/safe_pseudolabel.sh"
    else
        echo ""
        echo "✓ TARGET REACHED! Dataset complete."
    fi
fi
