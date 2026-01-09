#!/bin/bash
# Monitor pseudolabel progress and GPU health

echo "=========================================="
echo "Pseudolabel Monitor"
echo "=========================================="
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOG_DIR="${REPO_ROOT}/logs"

# Find the latest log file
LATEST_LOG=$(ls -t "${LOG_DIR}"/pseudolabel_*.log 2>/dev/null | head -1)

if [ -z "$LATEST_LOG" ]; then
    echo "No pseudolabel log files found in ${LOG_DIR}"
    exit 1
fi

echo "Monitoring: $LATEST_LOG"
echo ""
echo "GPU Status:"
nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
echo ""
echo "Last 20 lines of log:"
echo "=========================================="
tail -20 "$LATEST_LOG"
echo "=========================================="
echo ""
echo "Live tail (Ctrl+C to exit):"
tail -f "$LATEST_LOG"
