#!/bin/bash
# Run pseudolabel in video batches to avoid crashes
# Usage: bash scripts/pseudolabel_batch.sh [batch_num]

BATCH_NUM=${1:-1}
TOTAL_BATCHES=5

DATASET_OUT="/home/ramanlab/Documents/cole/model/pseudolabel_dataset_50k"
TARGET_PER_BATCH=10000  # 10k per batch = 50k total
BATCH_SIZE=2
STRIDE=15
MIN_CONF=0.85

# Calculate video stride to process different videos each batch
VIDEO_STRIDE=$((TOTAL_BATCHES))
VIDEO_OFFSET=$((BATCH_NUM - 1))

LOG_FILE="logs/pseudolabel_batch${BATCH_NUM}_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs

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

nohup nice -n 10 \
  env PYTHONPATH=src python -m fbpipe.steps.pseudolabel_export \
  --config config.yaml \
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
echo "  bash scripts/pseudolabel_batch.sh $((BATCH_NUM + 1))"
