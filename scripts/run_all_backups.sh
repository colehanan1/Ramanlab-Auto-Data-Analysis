#!/bin/bash
# Master backup script - runs all backup operations
# Usage: ./scripts/run_all_backups.sh [--dry-run] [--csv-only] [--compressed-only] [--full]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$PROJECT_DIR/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/backups_${TIMESTAMP}.log"

# Create logs directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Parse arguments
DRY_RUN=false
CSV_ONLY=false
COMPRESSED_ONLY=false
FULL_BACKUP=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --csv-only)
            CSV_ONLY=true
            shift
            ;;
        --compressed-only)
            COMPRESSED_ONLY=true
            shift
            ;;
        --full)
            FULL_BACKUP=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Determine what to run
if [ "$FULL_BACKUP" = true ]; then
    RUN_INCREMENTAL=true
    RUN_COMPRESSED=true
elif [ "$COMPRESSED_ONLY" = true ]; then
    RUN_INCREMENTAL=false
    RUN_COMPRESSED=true
elif [ "$CSV_ONLY" = true ]; then
    RUN_INCREMENTAL=true
    RUN_COMPRESSED=false
else
    # Default: run incremental only
    RUN_INCREMENTAL=true
    RUN_COMPRESSED=false
fi

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

log_info "=========================================="
log_info "Master Backup Runner"
log_info "=========================================="
log_info "Timestamp: $(date)"
log_info "Dry-run: $DRY_RUN"
log_info "Incremental: $RUN_INCREMENTAL"
log_info "Compressed: $RUN_COMPRESSED"
log_info "Log file: $LOG_FILE"
log_info "=========================================="

cd "$PROJECT_DIR"

# Run incremental backup if enabled
if [ "$RUN_INCREMENTAL" = true ]; then
    log_info "Starting incremental backup to Box..."

    if [ "$CSV_ONLY" = true ]; then
        BACKUP_ARGS="--csvs-only"
    else
        BACKUP_ARGS=""
    fi

    if [ "$DRY_RUN" = true ]; then
        BACKUP_ARGS="$BACKUP_ARGS --dry-run"
    fi

    if python "$SCRIPT_DIR/backup_to_box.py" $BACKUP_ARGS 2>&1 | tee -a "$LOG_FILE"; then
        log_info "✓ Incremental backup completed successfully"
    else
        log_error "✗ Incremental backup failed"
        exit 1
    fi
fi

# Run compressed backup if enabled
if [ "$RUN_COMPRESSED" = true ]; then
    log_info "Starting compressed backup..."

    COMPRESS_ARGS=""
    if [ "$DRY_RUN" = true ]; then
        COMPRESS_ARGS="--dry-run"
    fi

    if python "$SCRIPT_DIR/compress_and_backup.py" $COMPRESS_ARGS 2>&1 | tee -a "$LOG_FILE"; then
        log_info "✓ Compressed backup completed successfully"
    else
        log_error "✗ Compressed backup failed"
        exit 1
    fi
fi

log_info "=========================================="
log_info "✓ All backups completed successfully!"
log_info "=========================================="
log_info "Log saved to: $LOG_FILE"
