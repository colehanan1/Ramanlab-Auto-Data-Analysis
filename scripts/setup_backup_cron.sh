#!/bin/bash
# Setup automatic backup schedules in cron

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_PATH="$PROJECT_DIR/scripts/run_all_backups.sh"

echo "=========================================="
echo "Backup Cron Setup"
echo "=========================================="
echo ""
echo "This script will set up automatic backups:"
echo "  - Daily incremental CSV backup at 2 AM"
echo "  - Weekly compressed archive on Sunday at 3 AM"
echo ""

# Check if script exists
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "ERROR: Script not found at $SCRIPT_PATH"
    exit 1
fi

# Create temporary cron entries file
TEMP_CRON=$(mktemp)

# Get current crontab (or empty if doesn't exist)
crontab -l 2>/dev/null > "$TEMP_CRON" || true

# Remove old backup entries if they exist
sed -i '/run_all_backups.sh/d' "$TEMP_CRON" || true
sed -i '/backup_to_box.py/d' "$TEMP_CRON" || true
sed -i '/compress_and_backup.py/d' "$TEMP_CRON" || true

# Add new cron entries
echo "" >> "$TEMP_CRON"
echo "# Auto-generated backup schedules (DO NOT EDIT MANUALLY)" >> "$TEMP_CRON"
echo "# Daily CSV backup at 2 AM" >> "$TEMP_CRON"
echo "0 2 * * * cd $PROJECT_DIR && python scripts/backup_to_box.py --csvs-only >> logs/backup_daily_\$(date +\\%Y\\%m\\%d).log 2>&1" >> "$TEMP_CRON"
echo "" >> "$TEMP_CRON"
echo "# Weekly compressed archive on Sunday at 3 AM" >> "$TEMP_CRON"
echo "0 3 * * 0 cd $PROJECT_DIR && python scripts/compress_and_backup.py >> logs/backup_weekly_\$(date +\\%Y\\%m\\%d).log 2>&1" >> "$TEMP_CRON"

# Install new crontab
crontab "$TEMP_CRON"

# Clean up
rm "$TEMP_CRON"

echo ""
echo "✓ Cron schedules installed!"
echo ""
echo "Scheduled backups:"
echo "  ✓ Daily at 2:00 AM - Incremental CSV backup to Box"
echo "  ✓ Weekly (Sundays) at 3:00 AM - Compressed archive backup"
echo ""
echo "To view current crontab:"
echo "  crontab -l"
echo ""
echo "To remove backup schedules:"
echo "  crontab -e  (and manually delete the backup entries)"
echo ""
