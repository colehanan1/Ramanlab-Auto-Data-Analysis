# Compressed Backup for Emergency Data Recovery

This guide explains how to use the compression backup system for safer, more efficient emergency data storage.

## Why Compression?

- **Smaller files** - Reduce storage costs on Box and SMB
- **Faster backups** - Less data to transfer
- **Emergency ready** - Zip files are easy to restore quickly
- **Archive friendly** - Old archives auto-delete to save space
- **Dual backup** - Files go to both SMB and Box for redundancy

## Installation

The compression backup script is already set up. No additional installation needed!

## Quick Start

### See what would be backed up (dry-run)

```bash
python compress_and_backup.py --dry-run
```

Output shows:
- File sizes before and after compression
- Compression ratio achieved
- Where files would be copied

### Backup Everything

```bash
python compress_and_backup.py
```

This:
1. Compresses all CSVs into one zip file
2. Compresses all results/figures into one zip file
3. Copies compressed files to SMB share
4. Uploads compressed files to Box
5. Deletes archives older than 30 days

### Backup Only CSVs

```bash
python compress_and_backup.py --csvs-only
```

### Backup Only Results

```bash
python compress_and_backup.py --results-only
```

## Advanced Options

### Keep old archives (don't auto-delete)

```bash
python compress_and_backup.py --no-cleanup
```

### Clean up old archives only

```bash
python compress_and_backup.py --cleanup-only
```

### Backup to SMB only (skip Box)

```bash
python compress_and_backup.py --smb-only
```

### Backup to Box only (skip SMB)

```bash
python compress_and_backup.py --box-only
```

## Compression Results

Typical compression ratios:

| Type | Original | Compressed | Ratio |
|------|----------|-----------|-------|
| CSVs | ~50 MB | ~2 MB | 96% |
| Results (PNG/images) | ~500 MB | ~200 MB | 60% |
| Combined | ~550 MB | ~202 MB | 63% |

*Actual sizes depend on your data and image types*

## Automatic Scheduling

### Daily Emergency Backups (via cron)

```bash
crontab -e
```

Add this line for daily backup at 3 AM:

```bash
0 3 * * * cd /home/ramanlab/Documents/cole/VSCode/Ramanlab-Auto-Data-Analysis && python compress_and_backup.py
```

Or weekly on Sundays:

```bash
0 3 * * 0 cd /home/ramanlab/Documents/cole/VSCode/Ramanlab-Auto-Data-Analysis && python compress_and_backup.py
```

### After Pipeline Completion

Add to your analysis pipeline script:

```python
import subprocess

def backup_after_pipeline():
    """Run compression backup after analysis completes."""
    result = subprocess.run([
        "python",
        "/home/ramanlab/Documents/cole/VSCode/Ramanlab-Auto-Data-Analysis/compress_and_backup.py"
    ], timeout=7200)  # 2 hour timeout
    return result.returncode == 0

# Call at end of pipeline:
if __name__ == "__main__":
    # ... run analysis ...
    backup_after_pipeline()
```

## Storage Locations

### SMB Location
```
smb://ramanfile.local/ramanfiles/cole/flyTrackingData/
  CSVs_archive_20260123_150532.zip
  Results_archive_20260123_150532.zip
```

### Box Location
```
Box/Ramanlab-Backups/
  ├── CSVs/
  │   └── CSVs_archive_20260123_150532.zip
  └── Results/
      └── Results_archive_20260123_150532.zip
```

### Local Archive Directory
```
/home/ramanlab/Documents/cole/backups_compressed/
  CSVs_archive_20260123_150532.zip
  Results_archive_20260123_150532.zip
```

## Restoring from Backup

### From SMB

Mount the SMB location and extract:

```bash
# Navigate to SMB folder
cd /path/to/smb/ramanfiles/cole/flyTrackingData

# Extract CSVs
unzip CSVs_archive_*.zip -d /home/ramanlab/Documents/cole/Data/Opto/Combined/

# Extract Results
unzip Results_archive_*.zip -d /home/ramanlab/Documents/cole/
```

### From Box

Download from Box web interface and extract:

```bash
# After downloading from Box:
unzip CSVs_archive_*.zip -d /home/ramanlab/Documents/cole/Data/Opto/Combined/
unzip Results_archive_*.zip -d /home/ramanlab/Documents/cole/
```

### Via Rclone from Box

```bash
# Copy from Box to local
rclone copy Box-Folder:Ramanlab-Backups/CSVs/CSVs_archive_*.zip ./

# Extract
unzip CSVs_archive_*.zip
```

## Monitoring

### Check backup log

```bash
tail -f compress_backup.log
```

### List local archives

```bash
ls -lh /home/ramanlab/Documents/cole/backups_compressed/
```

### Check SMB location

```bash
ls -lh /path/to/smb/ramanfiles/cole/flyTrackingData/*.zip
```

### Check Box via rclone

```bash
rclone ls Box-Folder:Ramanlab-Backups/CSVs/
rclone ls Box-Folder:Ramanlab-Backups/Results/
```

## Archive Cleanup Policy

- Archives older than **30 days** are automatically deleted
- This prevents accumulating too many old versions
- Change this with: `backup.cleanup_old_archives(days_old=60)`
- Disable with: `--no-cleanup` flag

## Troubleshooting

### SMB copy fails
- Check SMB location is accessible: `mount | grep ramanfiles`
- Verify path exists: `ls /path/to/smb/ramanfiles/cole/flyTrackingData`

### Box upload fails
- Verify rclone is configured: `rclone listremotes`
- Check Box credentials: `rclone config`

### Compression is slow
- Normal for large datasets (50+ GB)
- Runs with maximum compression (level 9) for smallest files
- Can reduce compression level in script if needed

### Low disk space during compression
- Compression creates temporary zip before uploading
- Ensure at least 20% free space in `/home/ramanlab/Documents/cole/`
- Run `--cleanup-only` to delete old archives first

## Security Considerations

### What's backed up
- CSV data files
- Analysis results (PNG images, matrices)
- Everything is compressed

### What's NOT backed up
- Raw video files (too large)
- Temporary cache files
- Model training logs

### Access Control
- SMB share: Limited to network access
- Box: Requires Box account credentials
- Local archives: On your machine only

## Integration with Main Backup Script

This is complementary to `backup_to_box.py`:

| Script | Use Case | Frequency |
|--------|----------|-----------|
| `backup_to_box.py` | Incremental backups, all files | After each pipeline |
| `compress_and_backup.py` | Emergency archives, compressed | Weekly or manually |

Use both for complete protection!

## Support

For issues:
1. Check `compress_backup.log`
2. Run with `--dry-run` to debug
3. Verify SMB/Box connectivity
4. Check available disk space
