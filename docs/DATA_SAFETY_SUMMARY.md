# Data Safety & Backup Summary

Your data is now protected with a **3-layer backup system**:

## Layer 1: Incremental Backups (Regular)
**Script:** `backup_to_box.py`

- Backs up ALL files (incremental - only new/modified)
- Destinations: Box cloud
- Frequency: After each pipeline run (optional)
- Best for: Continuous protection, all data types

**How to use:**
```bash
python backup_to_box.py
```

## Layer 2: SMB Network Share (Fast Access)
**Config:** Added to `config.yaml`

- Two SMB locations for fly data and results
- Accessible from network immediately
- Best for: Quick access, collaboration

**Locations:**
- Data: `smb://ramanfile.local/ramanfiles/cole/flyTrackingData/`
- Results: `smb://ramanfile.local/ramanfiles/cole/Figures/`

## Layer 3: Compressed Backups (Emergency Archive)
**Script:** `compress_and_backup.py`

- Compresses CSVs + Results into zip files
- ~60-96% size reduction
- Backs up to BOTH SMB + Box
- Auto-deletes old archives (30 days)
- Best for: Emergency recovery, storage efficiency

**How to use:**
```bash
python compress_and_backup.py
```

---

## Quick Reference

### Regular Daily Backups
```bash
# CSVs only (fast)
python backup_to_box.py --csvs-only

# Everything
python backup_to_box.py
```

### Emergency Archives (Compressed)
```bash
# Create compressed backups (once weekly or monthly)
python compress_and_backup.py

# Dry-run to see what would be backed up
python compress_and_backup.py --dry-run
```

### Integration with Pipeline
Add this at the end of your analysis script:

```python
import subprocess

# After analysis completes:
subprocess.run(["python", "backup_to_box.py", "--csvs-only"])
subprocess.run(["python", "compress_and_backup.py"])
```

---

## Storage Breakdown

| Location | Data | Size | Use Case |
|----------|------|------|----------|
| Local | Original files | Full | Working/Processing |
| SMB Share | Original files | Full | Quick network access |
| Box Cloud | Original files | Full | Complete backup |
| Compressed Archive | ZIP files | 60-96% reduction | Emergency only |

---

## Automated Scheduling

### Daily Backups to Box (cron)
```bash
crontab -e
```

Add:
```bash
# Daily incremental backup at 2 AM
0 2 * * * python /home/ramanlab/Documents/cole/VSCode/Ramanlab-Auto-Data-Analysis/backup_to_box.py

# Weekly compressed backup every Sunday at 3 AM
0 3 * * 0 python /home/ramanlab/Documents/cole/VSCode/Ramanlab-Auto-Data-Analysis/compress_and_backup.py
```

---

## Recovery Procedures

### If you need files from backup:

**From Box (via web):**
1. Go to Box.com
2. Navigate to `Ramanlab-Backups/`
3. Download files

**From Box (via command line):**
```bash
# List files
rclone ls Box-Folder:Ramanlab-Backups/

# Download
rclone copy Box-Folder:Ramanlab-Backups/CSVs/CSVs_archive_*.zip ./
```

**From SMB:**
```bash
# Mount SMB (if not already mounted)
mount //ramanfile.local/ramanfiles /mnt/ramanfiles

# Copy compressed archive
cp /mnt/ramanfiles/cole/flyTrackingData/CSVs_archive_*.zip ./

# Extract
unzip CSVs_archive_*.zip
```

**From Local Compressed Archives:**
```bash
# List archives
ls -lh /home/ramanlab/Documents/cole/backups_compressed/

# Extract
unzip /home/ramanlab/Documents/cole/backups_compressed/CSVs_archive_*.zip
```

---

## Data Redundancy

Your critical data has multiple copies:

```
Local Computer
    ↓
SMB Network Share (fast access)
    ↓
Box Cloud (full backup)
    ↓
Compressed Archives (emergency)
```

**All three locations exist simultaneously** - if one fails, you have backups.

---

## What Gets Backed Up

### Backed Up
- ✓ CSV data files
- ✓ Analysis results (images, matrices)
- ✓ Model weights
- ✓ Configuration files

### NOT Backed Up (too large)
- ✗ Raw video files (back up separately if needed)
- ✗ Temporary cache files
- ✗ Build artifacts

---

## File Sizes

Typical compressed sizes after analysis:

```
CSVs:                    50 MB → 2 MB (96% reduction)
Results/Figures:         500 MB → 200 MB (60% reduction)
Combined with data:      1.5 GB → 400 MB (73% reduction)
```

---

## Monitoring

Check backup logs:
```bash
# Incremental backups
tail -f backup.log

# Compressed backups
tail -f compress_backup.log
```

Check storage usage:
```bash
# Local archives
du -sh /home/ramanlab/Documents/cole/backups_compressed/

# Box storage
rclone about Box-Folder:

# SMB usage
df -h /path/to/smb/ramanfiles/
```

---

## Best Practices

1. **Run after each pipeline:** `python backup_to_box.py --csvs-only`
2. **Weekly compression:** Run `compress_and_backup.py` once weekly
3. **Monitor logs:** Check backup.log for errors
4. **Test restores:** Occasionally extract from backup to verify
5. **Keep both:** Use both incremental + compressed for redundancy

---

## Emergency Checklist

If disaster strikes:

- [ ] All data in Box Cloud: `rclone ls Box-Folder:Ramanlab-Backups/`
- [ ] Compressed archives available: `ls backups_compressed/`
- [ ] SMB share accessible: Check network connection
- [ ] Restore test: Verify archives extract correctly
- [ ] Box credentials: `rclone config` works
- [ ] SMB mount: Can access `smb://ramanfile.local/`

---

## Getting Help

**Backup Script Issues:**
- Check: `backup.log` and `compress_backup.log`
- Run dry-run: `python script.py --dry-run`

**SMB Issues:**
- Test connection: `ls /path/to/smb/`
- Remount if needed: `mount //ramanfile.local/ramanfiles /mnt/`

**Box Issues:**
- Verify config: `rclone listremotes`
- Check access: `rclone ls Box-Folder:Ramanlab-Backups/`

---

## Summary

You now have:

✅ **Incremental cloud backup** (all files, Box)
✅ **Network shared backup** (fast SMB access)
✅ **Compressed emergency archives** (60-96% smaller)
✅ **Automatic cleanup** (old archives deleted)
✅ **Scheduling support** (cron integration)
✅ **Dual-destination backup** (SMB + Box)

**Your data is safe!**
