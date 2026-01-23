#!/bin/bash
# Backup script to run after pipeline completes
# Usage: ./run_backup.sh [--csvs-only] [--dry-run]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$SCRIPT_DIR/backup.log"

echo "========================================" | tee -a "$LOG_FILE"
echo "Starting Box backup from pipeline completion" | tee -a "$LOG_FILE"
echo "Timestamp: $(date)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# Run the backup script with passed arguments
cd "$SCRIPT_DIR"
python3 backup_to_box.py "$@"

BACKUP_STATUS=$?

echo "========================================" | tee -a "$LOG_FILE"
if [ $BACKUP_STATUS -eq 0 ]; then
    echo "✓ Backup completed successfully" | tee -a "$LOG_FILE"
else
    echo "✗ Backup failed with status $BACKUP_STATUS" | tee -a "$LOG_FILE"
fi
echo "========================================" | tee -a "$LOG_FILE"

exit $BACKUP_STATUS
