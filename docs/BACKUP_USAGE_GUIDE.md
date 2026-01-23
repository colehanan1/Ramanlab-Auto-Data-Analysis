# Backup Usage Guide

Quick reference for all backup commands and operations.

## Quick Commands

### Via Make (Recommended)

```bash
# Backup only CSVs (fast, after pipeline)
make backup-csvs

# Full incremental backup
make backup

# Compressed emergency archive
make backup-compressed

# Clean old archives (>30 days)
make clean-backups
```

### Via Direct Scripts

```bash
# Incremental backup
python scripts/backup_to_box.py

# Compressed backup
python scripts/compress_and_backup.py

# Master backup runner
./scripts/run_all_backups.sh
```

---

## Automatic Scheduling

### Install Automatic Backups

```bash
./scripts/setup_backup_cron.sh
```

This sets up:
- **Daily at 2:00 AM** - Incremental CSV backup to Box
- **Weekly (Sundays) at 3:00 AM** - Compressed archive backup

### View Scheduled Backups

```bash
crontab -l
```

### Remove Scheduled Backups

```bash
crontab -e
# Delete the backup entries
```

---

## Backup Integration with Pipeline

The pipeline now **automatically runs CSV backups**:

```bash
make run   # Runs: backup-csvs → pipeline → backup-csvs
```

This ensures your analysis results are backed up before and after processing.

---

## Detailed Command Reference

### Incremental Backup to Box

```bash
# Full backup (all files, incremental)
python scripts/backup_to_box.py

# CSV files only
python scripts/backup_to_box.py --csvs-only

# Results only
python scripts/backup_to_box.py --results-only

# Data folders only
python scripts/backup_to_box.py --data-only

# Model files only
python scripts/backup_to_box.py --models-only

# Preview without uploading (dry-run)
python scripts/backup_to_box.py --dry-run
```

### Compressed Emergency Archive

```bash
# Full compressed backup (CSVs + Results)
python scripts/compress_and_backup.py

# CSVs only (smaller file)
python scripts/compress_and_backup.py --csvs-only

# Results only
python scripts/compress_and_backup.py --results-only

# Preview without compressing (dry-run)
python scripts/compress_and_backup.py --dry-run

# Clean old archives only
python scripts/compress_and_backup.py --cleanup-only

# Don't auto-delete old archives
python scripts/compress_and_backup.py --no-cleanup

# Backup to SMB only (skip Box)
python scripts/compress_and_backup.py --smb-only

# Backup to Box only (skip SMB)
python scripts/compress_and_backup.py --box-only
```

### Master Backup Runner

```bash
# Incremental backup only (default)
./scripts/run_all_backups.sh

# Compressed backup only
./scripts/run_all_backups.sh --compressed-only

# CSV-only backup
./scripts/run_all_backups.sh --csv-only

# Full backup (incremental + compressed)
./scripts/run_all_backups.sh --full

# Dry-run (preview)
./scripts/run_all_backups.sh --dry-run

# Combined options
./scripts/run_all_backups.sh --full --dry-run
```

---

## Backup Types

### Type 1: Incremental (backup_to_box.py)

**When to use:** After analysis, during pipeline runs
**Frequency:** Every pipeline run
**Where:** Box Cloud
**Size:** Full dataset (no compression)
**Time:** Fast (only changed files)
**Typical time:** 1-5 minutes

```bash
make backup-csvs  # Recommended default
```

### Type 2: Compressed Archive (compress_and_backup.py)

**When to use:** Weekly or before major changes
**Frequency:** Once weekly
**Where:** SMB + Box Cloud
**Size:** 60-96% smaller (compressed)
**Time:** Slower (compression overhead)
**Typical time:** 5-20 minutes
**Benefit:** Emergency-ready, space-efficient

```bash
make backup-compressed
```

---

## Storage Locations

### Box Cloud
```
Box-Folder:Ramanlab-Backups/
├── CSVs/
│   └── [CSV files and compressed archives]
├── Results/
│   └── [Result/Figure files and compressed archives]
├── Data/
│   └── [Data folder backups]
└── Models/
    └── [Model weight backups]
```

### SMB Network Share
```
smb://ramanfile.local/ramanfiles/cole/
├── flyTrackingData/
│   └── [CSV and compressed archives]
└── Figures/
    └── [Results/Figures]
```

### Local Compressed Archives
```
/home/ramanlab/Documents/cole/backups_compressed/
├── CSVs_archive_20260123_150532.zip
├── Results_archive_20260123_150532.zip
└── [auto-deleted after 30 days]
```

---

## Monitoring Backups

### Check Logs

```bash
# View latest backup log
tail -f logs/backup_*.log

# View compressed backup log
tail -f compress_backup.log

# View incremental backup log
tail -f backup.log
```

### Verify Backups

```bash
# Check Box contents
rclone ls Box-Folder:Ramanlab-Backups/

# Check local archives
ls -lh /home/ramanlab/Documents/cole/backups_compressed/

# Check Box storage usage
rclone about Box-Folder:
```

### Test Restore

```bash
# Extract from local archive
unzip /home/ramanlab/Documents/cole/backups_compressed/CSVs_archive_*.zip -d /tmp/test_restore/

# Verify files
ls -lh /tmp/test_restore/
```

---

## Troubleshooting

### Box Upload Fails

```bash
# Check Box configuration
rclone listremotes

# Test Box access
rclone lsd Box-Folder:Ramanlab-Backups/

# Reconfigure if needed
rclone config
```

### SMB Not Accessible

```bash
# Check mount status
mount | grep ramanfiles

# Verify path exists
ls -l /path/to/smb/ramanfiles/

# Test write permission
touch /path/to/smb/ramanfiles/cole/test.txt && rm /path/to/smb/ramanfiles/cole/test.txt
```

### Slow Backups

- Check network speed: `iperf3`
- Reduce compression level in script (default is 9, max)
- Run during off-peak hours
- Split large backups: use `--csvs-only` first

### Low Disk Space

```bash
# Clean old archives
make clean-backups

# Check disk usage
df -h

# Check archive folder size
du -sh /home/ramanlab/Documents/cole/backups_compressed/
```

---

## Best Practices

✅ **Do:**
- Run `make backup-csvs` after each pipeline execution
- Schedule weekly compressed backups (already done with cron setup)
- Monitor backup logs for errors
- Test restore occasionally
- Keep both SMB + Box backups

❌ **Don't:**
- Run full data backup during peak hours
- Delete local archives unless space is critical
- Change SMB paths without updating config
- Trust single backup location

---

## Recovery Procedures

### Restore from Box

```bash
# List available backups
rclone ls Box-Folder:Ramanlab-Backups/

# Download backup
rclone copy Box-Folder:Ramanlab-Backups/CSVs/CSVs_archive_*.zip ./

# Extract
unzip CSVs_archive_*.zip -d /home/ramanlab/Documents/cole/Data/Opto/Combined/
```

### Restore from SMB

```bash
# Copy from SMB
cp /path/to/smb/ramanfiles/cole/flyTrackingData/CSVs_archive_*.zip ./

# Extract
unzip CSVs_archive_*.zip
```

### Restore from Local Archive

```bash
# List local archives
ls -lh /home/ramanlab/Documents/cole/backups_compressed/

# Extract
unzip /home/ramanlab/Documents/cole/backups_compressed/CSVs_archive_*.zip
```

---

## File Organization

All backup-related files are organized as follows:

```
Ramanlab-Auto-Data-Analysis/
├── docs/
│   ├── BACKUP_SETUP.md ........................... Initial setup guide
│   ├── BACKUP_USAGE_GUIDE.md ..................... This file
│   ├── COMPRESSION_BACKUP_GUIDE.md .............. Compression details
│   └── DATA_SAFETY_SUMMARY.md ................... Complete overview
├── scripts/
│   ├── backup_to_box.py ......................... Incremental backup
│   ├── compress_and_backup.py ................... Compressed backup
│   ├── run_all_backups.sh ....................... Master runner
│   ├── run_backup.sh ............................ Legacy wrapper
│   └── setup_backup_cron.sh ..................... Auto-schedule setup
├── logs/
│   ├── backup_daily_*.log ....................... Daily backup logs
│   ├── backup_weekly_*.log ...................... Weekly backup logs
│   └── [backup log files]
└── Makefile
    ├── make backup ............................ Full backup
    ├── make backup-csvs ....................... CSV backup
    ├── make backup-compressed ................ Compressed backup
    └── make clean-backups .................... Cleanup old archives
```

---

## Summary

Your backup system has **three layers**:

1. **Incremental Backups** → Box Cloud, after each pipeline
2. **Compressed Archives** → SMB + Box, weekly
3. **Pipeline Integration** → Automatic CSV backups

Start with: `make backup-csvs` or `make run`

For automation: `./scripts/setup_backup_cron.sh`

For emergency: `make backup-compressed` or `./scripts/run_all_backups.sh --full`
