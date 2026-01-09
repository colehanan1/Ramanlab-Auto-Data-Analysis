#!/bin/bash
# Safely stop pseudolabel export

echo "Looking for running pseudolabel processes..."

# Find all pseudolabel processes
PIDS=$(ps aux | grep "pseudolabel_export" | grep -v grep | awk '{print $2}')

if [ -z "$PIDS" ]; then
    echo "No pseudolabel processes found running."
    exit 0
fi

echo "Found processes: $PIDS"
echo ""

for PID in $PIDS; do
    echo "Stopping PID $PID..."
    kill -15 "$PID"  # Graceful shutdown first
done

sleep 2

# Check if still running
REMAINING=$(ps aux | grep "pseudolabel_export" | grep -v grep | awk '{print $2}')
if [ ! -z "$REMAINING" ]; then
    echo "Some processes still running, force killing..."
    for PID in $REMAINING; do
        kill -9 "$PID"
    done
fi

echo "✓ All pseudolabel processes stopped."
