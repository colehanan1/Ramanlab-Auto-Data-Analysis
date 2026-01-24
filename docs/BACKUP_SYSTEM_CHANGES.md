# Backup System Redesign - Complete Changes

## Summary

The backup system has been completely redesigned to give you full control with these key improvements:

### ✅ What Changed

**1. Compression: DEFAULT OFF**
- **Old:** Automatic compression enabled
- **New:** Direct file copy (no compression) by default - much faster!
- **Option:** Enable compression in config if needed for space savings

**2. rsync for SMB**
- **Old:** Used `cp` command
- **New:** Uses `rsync --update` - more efficient, only syncs changed files
- **Benefit:** Faster subsequent backups, checksum verification

**3. No Automatic Schedules**
- **Old:** Cron jobs ran nightly (2 AM) and weekly (Sundays 3 AM)
- **New:** NO automatic cron schedules whatsoever
- **Control:** Backups only run when YOU explicitly run commands
- **Action taken:** All cron schedules have been removed from your crontab

**4. Unified Backup System**
- **Old:** Multiple scripts (backup_to_box.py, compress_and_backup.py, run_all_backups.sh, setup_backup_cron.sh)
- **New:** Single unified script: `backup_system.py`
- **Benefit:** Simpler, easier to maintain, consistent behavior

**5. Config-Driven**
- **Old:** Backup behavior hardcoded in scripts
- **New:** All settings in `config/config.yaml` under `backups` section
- **Benefit:** Change behavior without editing code

---

## Files Changed/Created

### New Files

| File | Purpose |
|------|---------|
| `scripts/backup_system.py` | Unified backup system (replaces old scripts) |
| `docs/BACKUP_SYSTEM_GUIDE.md` | Comprehensive backup documentation |
| `BACKUP_QUICKSTART.txt` | Quick reference card |
| `BACKUP_SYSTEM_CHANGES.md` | This file - what changed |

### Modified Files

| File | Changes |
|------|---------|
| `config/config.yaml` | Added `backups` section with all settings |
| `Makefile` | Updated to use `backup_system.py`, removed cron targets |
| `crontab` | **Removed all automatic backup schedules** |

### Files Kept (but not used by new system)

These old scripts are still present but not used:
- `scripts/backup_to_box.py`
- `scripts/compress_and_backup.py`
- `scripts/run_all_backups.sh`
- `scripts/setup_backup_cron.sh`
- `scripts/run_backup.sh`

You can delete them if you want, or keep for reference.

### Old Documentation (Still Valid)

The following docs are still relevant but superseded by BACKUP_SYSTEM_GUIDE.md:
- `docs/BACKUP_SETUP.md`
- `docs/BACKUP_USAGE_GUIDE.md`
- `docs/COMPRESSION_BACKUP_GUIDE.md`
- `docs/DATA_SAFETY_SUMMARY.md`
- `docs/JUPYTER_PIPELINE_GUIDE.md`

---

## Configuration

All backup settings are now in `config/config.yaml`:

```yaml
backups:
  # Enable/disable all backups
  enabled: true

  # Compression settings (DEFAULT: disabled)
  compression:
    enabled: false  # Set to true to compress CSVs and Results

  # Backup destinations (all enabled by default)
  destinations:
    smb:
      enabled: true
      base_path: "smb://ramanfile.local/ramanfiles/cole"
      csvs_path: "smb://ramanfile.local/ramanfiles/cole/flyTrackingData"
      results_path: "smb://ramanfile.local/ramanfiles/cole/Figures"
      use_compression: false  # Override global setting

    box:
      enabled: true
      remote: "Box-Folder"
      folder: "Ramanlab-Backups"
      use_compression: false  # Override global setting

    secured:
      enabled: true
      base_path: "/securedstorage/DATAsec/cole/Data-secured"
      use_compression: false  # Override global setting
```

---

## Usage

### Run Pipeline with Backups (Default)

```bash
make run
```

Executes:
1. Backup to SMB/Box/Secured (BEFORE)
2. Full analysis pipeline
3. Backup to SMB/Box/Secured (AFTER)

### Manual Backup Only

```bash
make backup
```

Or:

```bash
python scripts/backup_system.py
```

### Dry-Run (See What Would Happen)

```bash
python scripts/backup_system.py --dry-run
```

### Backup to Specific Destination

```bash
python scripts/backup_system.py --smb-only
python scripts/backup_system.py --box-only
python scripts/backup_system.py --secured-only
```

---

## Enable Compression (Optional)

Edit `config/config.yaml`:

**Option 1: Global Compression**

```yaml
backups:
  compression:
    enabled: true  # Compress all destinations
```

**Option 2: Per-Destination Compression**

```yaml
backups:
  compression:
    enabled: false  # Global default: no compression

  destinations:
    smb:
      use_compression: true   # But compress for SMB
    box:
      use_compression: false  # Not for Box
    secured:
      use_compression: true   # Compress for secured
```

### Compression Results

- **CSVs:** 50-200 MB → 2-10 MB (96% reduction)
- **Results:** 100-500 MB → 40-200 MB (60% reduction)
- **Time:** 5-15 minutes for full compression
- **Output:** Stored in `backups_compressed/` as `.zip` files

---

## Backup Methods

| Destination | Method | Behavior | Default |
|------------|--------|----------|---------|
| SMB | rsync | `--update` (only changed files) | Enabled |
| Box | rclone | `--update` (only changed files) | Enabled |
| Secured | rsync | `--update` (only changed files) | Enabled |

All methods use efficient incremental syncing - only new/modified files are transferred.

---

## Disable Backups

### Disable Specific Destination

In `config/config.yaml`:

```yaml
destinations:
  smb:
    enabled: false  # Skip SMB
  box:
    enabled: true   # Still backup to Box
```

### Disable All Backups

```yaml
backups:
  enabled: false  # No backups at all
```

---

## Cron Schedules - REMOVED

**All automatic cron schedules have been removed.**

Your crontab no longer has:
- ~~Daily CSV backup at 2 AM~~
- ~~Weekly compressed archive at 3 AM Sundays~~

Backups now **only run when you execute commands**:
- `make run` - includes backups
- `make backup` - backup only
- During Jupyter notebook execution

**This gives you complete control over when backups happen.**

---

## Key Differences from Old System

| Feature | Old | New |
|---------|-----|-----|
| **Default Compression** | Enabled | Disabled ✅ |
| **SMB Method** | cp | rsync |
| **Automatic Schedule** | Cron (nightly/weekly) | None (manual only) |
| **Scripts** | 5 different scripts | 1 unified script |
| **Configuration** | Hardcoded | config.yaml |
| **Control** | Limited | Full |

---

## Pipeline Integration

The pipeline (`make run`) now includes backup targets:

```makefile
run: backup
    export MPLBACKEND=Agg && ... python scripts/pipeline/run_workflows.py ...
    $(MAKE) backup
```

This means:
1. When you run `make run`, backups happen automatically (before and after)
2. Backup behavior is controlled by `config/config.yaml`
3. No separate cron jobs needed
4. You see all output in real-time

---

## Migration Notes

### For Users Coming from Old System

**No action required.** The new system is:
- Backward compatible with all your data
- Uses same backup destinations (SMB, Box, Secured)
- Automatic during `make run`
- But much faster (no compression by default)

### If You Want Compression Back

Edit `config/config.yaml`:

```yaml
backups:
  compression:
    enabled: true
```

Then run: `make run`

### If You Want to Use Old Scripts

You can still use the old scripts directly:

```bash
python scripts/backup_to_box.py
python scripts/compress_and_backup.py
```

But the new `backup_system.py` is recommended.

---

## Troubleshooting

### SMB backups failing

```bash
# Check SMB is mounted
mount | grep smb

# Test rsync
rsync -av /path/to/csv smb://ramanfile.local/ramanfiles/cole/flyTrackingData/
```

### Box backups failing

```bash
# Check rclone configuration
rclone listremotes

# Test rclone
rclone ls Box-Folder:Ramanlab-Backups/
```

### Monitor backups

```bash
# Watch in real-time
tail -f backup.log
```

---

## Performance Impact

### Speed (No Compression)

- **Direct copy:** Very fast, uses network speed
- **rsync:** Fast for first run (copies all), subsequent runs only sync changes
- **rclone:** Fast to Box, network limited

### Speed (With Compression)

- **Compression:** 5-15 minutes depending on data size
- **Still better than uploading large uncompressed files**
- **Good for emergency archives**

---

## Complete Control

You now have:

✅ **Control over compression** - Enable/disable in config
✅ **Control over timing** - Only backup when you run commands
✅ **Control over destinations** - Enable/disable each one
✅ **Control over performance** - rsync is efficient by default
✅ **No surprises** - No hidden automatic backups
✅ **Simple configuration** - All in config.yaml
✅ **Easy to understand** - Single unified script

---

## Next Steps

1. **Review backup config** in `config/config.yaml`
2. **Run a test** with `make run` or `make backup --dry-run`
3. **Enable compression** (optional) if you need space savings
4. **Monitor backups** with `tail -f backup.log`

For more details, see:
- `BACKUP_QUICKSTART.txt` - Quick reference
- `docs/BACKUP_SYSTEM_GUIDE.md` - Comprehensive guide

---

## Questions?

Check the log file:
```bash
tail -f backup.log
```

Test a backup:
```bash
python scripts/backup_system.py --dry-run
```

View configuration:
```bash
cat config/config.yaml | grep -A 30 "^backups:"
```

---

**Your backup system is now under your complete control!**
