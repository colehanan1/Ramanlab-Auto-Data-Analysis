# SMB Rclone Quick Reference Card

## Setup (One-Time)

```bash
# 1. Configure SMB
rclone config
# → n (new remote)
# → SMB-Ramanfile
# → smb
# → ramanfile.local
# → ramanlab
# → [your password]

# 2. Verify
python3 scripts/test_smb_config.py
```

## Common Commands

### List SMB Contents
```bash
# List all folders
rclone lsd SMB-Ramanfile:ramanfiles/cole

# List specific folder
rclone ls SMB-Ramanfile:ramanfiles/cole/Figures/ -h

# Check specific file
rclone ls SMB-Ramanfile:ramanfiles/cole/flyTrackingData/
```

### Copy Files
```bash
# Single file
rclone copy /local/file.csv SMB-Ramanfile:ramanfiles/cole/flyTrackingData/

# Entire directory
rclone sync /local/results/ SMB-Ramanfile:ramanfiles/cole/Figures/MyResults/

# Dry-run (preview)
rclone copy /local/file SMB-Ramanfile:dest --dry-run -v

# With progress
rclone copy /local/file SMB-Ramanfile:dest -v --progress
```

### Backup
```bash
# Full backup (all destinations)
python3 scripts/backup_system.py

# SMB only
python3 scripts/backup_system.py --smb-only

# Dry-run
python3 scripts/backup_system.py --dry-run

# Verbose
python3 scripts/backup_system.py -v
```

## Config.yaml Examples

### Envelope Visuals
```yaml
analysis:
  envelope_visuals:
    envelopes:
      - after_show_sec: 30.0
        out_dir: /home/ramanlab/Documents/cole/Results/Opto/PER-Envelopes
        out_dir_smb: ramanfiles/cole/Figures/PER-Envelopes/
        # ... rest of config
```

### Reaction Predictions
```yaml
reaction_prediction:
  output_csv: /home/ramanlab/Documents/cole/Data/Opto/Combined/model_predictions.csv
  output_csv_smb: ramanfiles/cole/flyTrackingData/model_predictions.csv
```

### Matrices
```yaml
analysis:
  combined:
    matrices:
      out_dir: /home/ramanlab/Documents/cole/Results/Opto/Reaction_Matrices
      out_dir_smb: ramanfiles/cole/Figures/Reaction_Matrices/
```

## Python Code Examples

### Test Connection
```python
from fbpipe.utils.smb_copy import get_smb_copier
copier = get_smb_copier()
copier.test_connection()
```

### Copy Files
```python
from fbpipe.utils.smb_copy import copy_to_smb, copy_csv_to_smb

# CSV to standard location
copy_csv_to_smb('/path/to/file.csv')

# Custom path
copy_to_smb('/path/to/results/', 'ramanfiles/cole/Figures/Custom/')

# Dry-run
copy_to_smb('/path/to/src', 'ramanfiles/cole/dest/', dry_run=True)
```

### Sync Directory
```python
from fbpipe.utils.smb_copy import get_smb_copier

copier = get_smb_copier()
copier.sync_directory(
    '/path/to/results/',
    'ramanfiles/cole/Figures/MyResults/',
    skip_same_size=True,
    verbose=True
)
```

## SMB Paths

| Purpose | Path |
|---------|------|
| Results/Figures | `ramanfiles/cole/Figures/[Name]/` |
| CSV Data | `ramanfiles/cole/flyTrackingData/` |
| Backups | `ramanfiles/cole/backups/` |
| Custom | `ramanfiles/cole/[YourPath]/` |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Can't connect | `rclone config` → update credentials |
| Host not found | `ping ramanfile.local` → check network |
| Permission denied | Check SMB user permissions |
| Files not copying | Check `out_dir_smb` in config.yaml |
| Slow transfers | Add `--bwlimit 10M` for bandwidth cap |

## Files

| File | Purpose |
|------|---------|
| `~/.config/rclone/rclone.conf` | Encrypted SMB credentials |
| `config/config.yaml` | Project config with SMB paths |
| `src/fbpipe/utils/smb_copy.py` | Core SMB utility |
| `scripts/test_smb_config.py` | Validation script |

## Documentation

| Document | Purpose |
|----------|---------|
| `SMB-SETUP-QUICK-START.md` | 3-step setup guide |
| `SMB-RCLONE-SETUP.md` | Detailed reference |
| `RCLONE-SMB-IMPLEMENTATION.md` | Technical details |
| `SMB-QUICK-REFERENCE.md` | This file |

## Most Used Commands

```bash
# Verify setup works
python3 scripts/test_smb_config.py

# Run pipeline (auto-copies to SMB)
python3 scripts/pipeline/run_workflows.py

# Check what's on SMB
rclone ls SMB-Ramanfile:ramanfiles/cole/Figures/ -h

# Full backup to all destinations
python3 scripts/backup_system.py
```

---

**Keep this handy! Copy these commands as needed.** 📋
