# Unified Backup System

Complete guide for the new backup system with rsync for SMB and configurable compression.

## Overview

The new backup system:
- **Default: NO COMPRESSION** - Files are copied directly (fast, no space overhead)
- **Uses rsync for SMB** - Efficient syncing with checksums
- **Uses rclone for Box** - Cloud-optimized sync
- **Uses rsync for Secured storage** - Secure, efficient syncing
- **NO automatic schedules** - Backups only run when explicitly triggered
- **Configurable compression** - Enable in config file if needed

## Quick Start

### Automatic Backups (During Pipeline Runs)

Backups run automatically when you run the pipeline:

```bash
# Full pipeline with automatic backups before and after
make run
```

This executes:
1. Backup to SMB/Box/Secured (before pipeline)
2. Full analysis pipeline
3. Backup to SMB/Box/Secured (after pipeline)

### Manual Backups (Anytime)

```bash
# Backup manually without running pipeline
make backup
```

## Configuration

Edit `config/config.yaml` in the `backups` section:

```yaml
backups:
  # Enable/disable all backups
  enabled: true

  # Compression settings (DEFAULT: false = no compression, direct copy)
  compression:
    enabled: false  # Set to true to compress files before backing up

  destinations:
    # SMB network share - uses rsync
    smb:
      enabled: true
      base_path: "smb://ramanfile.local/ramanfiles/cole"
      csvs_path: "smb://ramanfile.local/ramanfiles/cole/flyTrackingData"
      results_path: "smb://ramanfile.local/ramanfiles/cole/Figures"
      use_compression: false  # Override global compression setting

    # Box cloud storage - uses rclone
    box:
      enabled: true
      remote: "Box-Folder"  # rclone remote name
      folder: "Ramanlab-Backups"
      use_compression: false  # Override global compression setting

    # Secured storage - uses rsync
    secured:
      enabled: true
      base_path: "/securedstorage/DATAsec/cole/Data-secured"
      use_compression: false  # Override global compression setting
```

## Backup Methods

### SMB Network Share (rsync)

Files are synced to SMB via rsync with `--update` flag:

**Without compression (default):**
```
CSVs → smb://ramanfile.local/ramanfiles/cole/flyTrackingData/
Results → smb://ramanfile.local/ramanfiles/cole/Figures/
```

**With compression:**
```
CSVs_archive_TIMESTAMP.zip → smb://...
Results_archive_TIMESTAMP.zip → smb://...
```

### Box Cloud Storage (rclone)

Files are synced to Box via rclone with `--update` flag:

**Without compression (default):**
```
CSVs → Box-Folder:Ramanlab-Backups/CSVs/
Results → Box-Folder:Ramanlab-Backups/Results/
```

**With compression:**
```
CSVs_archive_TIMESTAMP.zip → Box-Folder:Ramanlab-Backups/CSVs/
Results_archive_TIMESTAMP.zip → Box-Folder:Ramanlab-Backups/Results/
```

### Secured Storage (rsync)

Files are synced to secured storage via rsync:

**Without compression (default):**
```
CSVs → /securedstorage/DATAsec/cole/Data-secured/CSVs/
```

**With compression:**
```
*.zip → /securedstorage/DATAsec/cole/Data-secured/backups/
```

## Enabling Compression

### Option 1: Global Compression

Enable compression for all destinations in `config/config.yaml`:

```yaml
backups:
  compression:
    enabled: true  # Compress CSVs and Results before backing up
```

### Option 2: Per-Destination Compression

Override global setting for specific destinations:

```yaml
backups:
  compression:
    enabled: false  # Global default: no compression

  destinations:
    smb:
      use_compression: true  # But compress for SMB only
    box:
      use_compression: false  # No compression for Box
    secured:
      use_compression: true   # Compress for secured storage
```

## Compression Benefits

When enabled, files are compressed with ZIP (level 9):

**CSVs compression:**
- Original: ~50-200 MB
- Compressed: ~2-10 MB (96% reduction)
- Files: `CSVs_archive_TIMESTAMP.zip`

**Results compression:**
- Original: ~100-500 MB
- Compressed: ~40-200 MB (60% reduction)
- Files: `Results_archive_TIMESTAMP.zip`

Compressed archives are stored in: `backups_compressed/`

## Disabling Specific Destinations

To disable a backup destination, set `enabled: false`:

```yaml
destinations:
  smb:
    enabled: false  # Skip SMB backups
  box:
    enabled: true   # Still backup to Box
  secured:
    enabled: true   # Still backup to secured
```

## Manual Backup Commands

```bash
# Full backup to all enabled destinations
make backup

# Backup only (no pipeline run)
python scripts/backup_system.py

# Dry-run (see what would be backed up)
python scripts/backup_system.py --dry-run

# SMB only
python scripts/backup_system.py --smb-only

# Box only
python scripts/backup_system.py --box-only

# Secured storage only
python scripts/backup_system.py --secured-only
```

## Pipeline Backups

During `make run`:

1. **Before pipeline:**
   ```bash
   python scripts/backup_system.py
   ```

2. **Run analysis:**
   ```bash
   python scripts/pipeline/run_workflows.py --config config/config.yaml
   ```

3. **After pipeline:**
   ```bash
   python scripts/backup_system.py
   ```

This ensures data is backed up before processing and results are backed up immediately after.

## Monitoring Backups

Watch backup progress:

```bash
# Monitor main backup log
tail -f backup.log

# Check specific destination
# SMB: Check rsync output in backup.log
# Box: Check rclone output in backup.log
# Secured: Check rsync output in backup.log
```

## Performance Notes

### rsync (SMB & Secured)
- Fast for direct copies (no compression)
- Only syncs changed files with `--update` flag
- Checksum-based verification
- Network speed dependent

### Compression (when enabled)
- Takes 1-5 minutes for CSVs
- Takes 5-15 minutes for Results (depends on size)
- 60-96% size reduction
- Good for emergency archives

### rclone (Box)
- Fast cloud sync
- Incremental with `--update` flag
- Network speed dependent
- Can be slower than SMB on local network

## Troubleshooting

### SMB backups failing
```bash
# Check SMB is mounted
mount | grep smb

# Verify path is accessible
ls smb://ramanfile.local/ramanfiles/cole/

# Test rsync
rsync -av /home/ramanlab/Documents/cole/Data/Opto/Combined/*.csv smb://ramanfile.local/ramanfiles/cole/flyTrackingData/
```

### Box backups failing
```bash
# Verify rclone is configured
rclone listremotes

# Check Box connection
rclone ls Box-Folder:Ramanlab-Backups/

# Test rclone
rclone copy /path/to/file Box-Folder:Ramanlab-Backups/ -v
```

### Secured storage not syncing
```bash
# Check path exists and is writable
ls -l /securedstorage/DATAsec/cole/Data-secured/

# Verify permissions
touch /securedstorage/DATAsec/cole/Data-secured/test.txt
```

## What Gets Backed Up

### CSVs
- `all_envelope_rows_wide.csv`
- `all_envelope_rows_wide_training.csv`
- `all_envelope_rows_wide_combined_base.csv`
- `all_envelope_rows_wide_combined_base_training.csv`
- `model_predictions.csv`
- `all_eye_prob_coords_wide.csv`

### Results/Figures
- `Results/Opto/Reaction_Matrices/`
- `Results/Opto/Reaction_Matrices/Overlay/`
- `Results/Opto/PER-Envelopes/`
- `Results/Opto/Training-PER-Envelopes/`
- `Results/Opto/Reaction_Predictions(Strictest)/`
- `Results/Opto/CombinedBase-PER-Envelopes/`
- `Results/Opto/CombinedBase-Training-PER-Envelopes/`
- `Results/Opto/Weekly-Training-Envelopes/`

## Backup Locations

**SMB Network Share:**
- CSVs: `smb://ramanfile.local/ramanfiles/cole/flyTrackingData/`
- Results: `smb://ramanfile.local/ramanfiles/cole/Figures/`

**Box Cloud:**
- CSVs: `Box-Folder:Ramanlab-Backups/CSVs/`
- Results: `Box-Folder:Ramanlab-Backups/Results/`

**Secured Storage:**
- CSVs: `/securedstorage/DATAsec/cole/Data-secured/CSVs/`
- Backups: `/securedstorage/DATAsec/cole/Data-secured/backups/`

## Summary

✓ **Default:** Direct file copy (fast, no compression)
✓ **rsync for SMB:** Efficient local network sync
✓ **rclone for Box:** Cloud-optimized sync
✓ **rsync for Secured:** Secure storage sync
✓ **Manual control:** Only backup when you run commands
✓ **Optional compression:** Enable in config if needed
✓ **Pipeline integration:** Automatic backup before/after runs

Your backup system is under your complete control!
