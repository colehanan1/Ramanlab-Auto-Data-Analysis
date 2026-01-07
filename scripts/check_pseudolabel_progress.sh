#!/bin/bash
# Check pseudolabel dataset progress

DATASET_OUT="/home/ramanlab/Documents/cole/model/pseudolabel_dataset_50k"
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
        echo "  bash scripts/safe_pseudolabel.sh"
    else
        echo ""
        echo "✓ TARGET REACHED! Dataset complete."
    fi
fi
