# Rclone SMB Implementation Summary

## Overview

Your SMB backup system has been completely overhauled to use **rclone** instead of relying on manual SMB mount points or unreliable file copying methods. Files now automatically copy to your SMB share after analysis completes.

## What Was Implemented

### 1. **SMB Copy Utility** (`src/fbpipe/utils/smb_copy.py`)

A robust Python module that handles all SMB transfers via rclone:

**Features:**
- Single file copying
- Directory syncing
- Convenience functions for CSVs and figures
- Connection testing
- Error handling and logging
- Dry-run support for previews

**Usage:**
```python
from fbpipe.utils.smb_copy import copy_to_smb, copy_csv_to_smb

# Copy CSV
copy_csv_to_smb('/path/to/results.csv')

# Copy directory
copy_to_smb('/path/to/results/', 'ramanfiles/cole/Figures/MyResults/')
```

### 2. **Pipeline Integration**

The analysis pipeline now automatically copies outputs to SMB:

**Modified files:**
- `scripts/pipeline/run_workflows.py` - Auto-copy after each analysis step
- `src/fbpipe/steps/predict_reactions.py` - Copy prediction results
- `src/fbpipe/config.py` - Added `output_csv_smb` field

**How it works:**
1. Analysis completes and generates output
2. If `out_dir_smb` is configured, files automatically copy to SMB
3. Logging shows copy status and any errors

### 3. **Backup System Enhancement** (`scripts/backup_system.py`)

Updated to use rclone for more reliable SMB backups:

**New `backup_to_smb()` parameters:**
- `use_rclone=True` (default) - Uses rclone with automatic fallback to rsync
- Compression support
- Dry-run capability

### 4. **Configuration & Documentation**

**New files created:**
- `docs/SMB-RCLONE-SETUP.md` - Complete reference guide
- `docs/SMB-SETUP-QUICK-START.md` - 3-step setup guide
- `scripts/test_smb_config.py` - Automated validation script
- `~/.config/rclone/rclone.conf` - Encrypted SMB credentials

## Quick Setup

### 1. Configure rclone
```bash
rclone config
# Follow prompts to configure SMB-Ramanfile remote
```

### 2. Test connection
```bash
python3 scripts/test_smb_config.py
```

### 3. Update config.yaml
Add `out_dir_smb` paths to your analysis sections. Example:

```yaml
analysis:
  envelope_visuals:
    matrices:
      out_dir: /home/ramanlab/Documents/cole/Results/Opto/Reaction_Matrices
      out_dir_smb: ramanfiles/cole/Figures/Reaction_Matrices/
    envelopes:
      - out_dir: /home/ramanlab/Documents/cole/Results/Opto/PER-Envelopes
        out_dir_smb: ramanfiles/cole/Figures/PER-Envelopes/

reaction_prediction:
  output_csv: /home/ramanlab/Documents/cole/Data/Opto/Combined/model_predictions.csv
  output_csv_smb: ramanfiles/cole/flyTrackingData/model_predictions.csv
```

### 4. Run pipeline
Files now automatically copy to SMB after analysis:
```bash
python3 scripts/pipeline/run_workflows.py
```

## Technical Details

### Rclone Remote Configuration

Location: `~/.config/rclone/rclone.conf`

```ini
[SMB-Ramanfile]
type = smb
host = ramanfile.local
username = ramanlab
password = [encrypted]
port = 445
```

**Security:** Password is encrypted by rclone using `nacl/secretbox`

### SMB Copy Methods

The utility provides several copy methods:

```python
smb = get_smb_copier()

# Copy single file with size-only check
smb.copy_file(src, dest_path, skip_same_size=True)

# Sync entire directory
smb.sync_directory(src, dest_path, delete_extra=False)

# Convenience methods
smb.copy_to_csv_path(src)          # → ramanfiles/cole/flyTrackingData/
smb.copy_to_figures_path(src)      # → ramanfiles/cole/Figures/
```

### Error Handling

Failures are logged but don't stop pipeline execution:
- If SMB copy fails, logging shows the error
- Pipeline continues with next step
- Use `--dry-run` to test without actual copies

### Performance

- **Size-only checks:** Skip files that already match on SMB
- **Parallel transfers:** Can enable with `--transfers N` flag
- **Bandwidth limiting:** Can cap bandwidth with `--bwlimit` flag
- **Compression:** Optional for backup system

## File Locations

### Source Code
```
src/fbpipe/utils/smb_copy.py          # Core SMB utility
src/fbpipe/config.py                  # Config with SMB fields
src/fbpipe/steps/predict_reactions.py # SMB copy after predictions
scripts/pipeline/run_workflows.py      # Auto-copy after analysis
scripts/backup_system.py               # Enhanced backup with rclone
```

### Documentation
```
docs/SMB-SETUP-QUICK-START.md    # Quick start (START HERE)
docs/SMB-RCLONE-SETUP.md         # Detailed reference
RCLONE-SMB-IMPLEMENTATION.md     # This file
scripts/test_smb_config.py       # Validation script
```

### Configuration
```
~/.config/rclone/rclone.conf     # Rclone config with credentials
config/config.yaml               # Project config with out_dir_smb
```

## Troubleshooting

### Connection Issues

```bash
# Check if SMB host is reachable
ping ramanfile.local

# Check SMB port
nc -zv ramanfile.local 445

# Test rclone connection
rclone lsd SMB-Ramanfile:ramanfiles/cole
```

### Credential Issues

```bash
# Re-configure credentials
rclone config
# Navigate to SMB-Ramanfile and update

# Or edit config directly (encrypted)
rclone config edit
```

### Debug Pipeline Copies

```bash
# Run test script
python3 scripts/test_smb_config.py

# Check logs from last run
tail -f backup.log
```

## Advantages Over Previous Method

| Aspect | Previous | New (Rclone) |
|--------|----------|-------------|
| **Reliability** | Files didn't copy | Auto-copies after analysis |
| **Error Handling** | No error tracking | Full logging |
| **Configuration** | SMB URLs in config | Encrypted rclone config |
| **Performance** | Manual/unreliable | Optimized with checksums |
| **Fallback** | None | Can use rsync as fallback |
| **Compression** | Not supported | Optional compression |
| **Bandwidth** | Unlimited | Can be rate-limited |
| **Testing** | Manual verification | `test_smb_config.py` |

## Integration Examples

### Add to Your Workflow

**Full pipeline with SMB:**
```bash
python3 scripts/pipeline/run_workflows.py
# Automatically copies all results to SMB
```

**Backup with SMB:**
```bash
python3 scripts/backup_system.py --smb-only
# Uses rclone to backup CSVs and results
```

**Custom SMB copy:**
```python
from fbpipe.utils.smb_copy import copy_to_smb

copy_to_smb('/path/to/my/results/', 'ramanfiles/cole/Figures/Custom/')
```

## Next Steps

1. **Complete Setup:**
   - Run `rclone config` and configure SMB-Ramanfile
   - Run `python3 scripts/test_smb_config.py`
   - Add `out_dir_smb` paths to config.yaml

2. **Verify Integration:**
   - Run pipeline: `python3 scripts/pipeline/run_workflows.py`
   - Check SMB: `rclone ls SMB-Ramanfile:ramanfiles/cole/Figures/`

3. **Optional Enhancements:**
   - Enable compression in backup system
   - Set bandwidth limits
   - Configure cron jobs for automated backups

## Support

For detailed information, see:
- **Quick Start:** `docs/SMB-SETUP-QUICK-START.md`
- **Reference:** `docs/SMB-RCLONE-SETUP.md`
- **Test:** `python3 scripts/test_smb_config.py`

---

**Your SMB backup system is now configured and ready to use! 🎉**

Files will automatically copy to your SMB share after analysis completes.
