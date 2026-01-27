# SMB Backup Setup - Quick Start Guide

Your SMB file copying is now configured to work with **rclone**, which is more reliable and robust than the previous approach.

## What Changed?

✅ **Old approach**: Files were configured but not actually copied
✅ **New approach**: Files are automatically copied to SMB using rclone after analysis completes

## Setup (3 Steps)

### Step 1: Configure rclone for SMB

Run the interactive configuration:
```bash
rclone config
```

**Follow the prompts:**
1. Press `n` for **New remote**
2. Name: `SMB-Ramanfile`
3. Type: `smb`
4. Host: `ramanfile.local`
5. Username: `ramanlab`
6. Password: *(enter your password)*
7. Port: `445` (or press Enter)
8. Domain: *(leave empty, press Enter)*
9. Confirm: `y`

### Step 2: Test Connection

```bash
# Verify remote is configured
rclone listremotes
# Output should include: SMB-Ramanfile

# Test SMB connection
rclone lsd SMB-Ramanfile:ramanfiles/cole
# Should list your SMB folders
```

Or run the automated test:
```bash
python3 scripts/test_smb_config.py
```

### Step 3: Enable in Config

Edit `config/config.yaml` and add `out_dir_smb` paths to your analysis sections:

**Example for envelope visuals:**
```yaml
analysis:
  envelope_visuals:
    envelopes:
      - after_show_sec: 30.0
        out_dir: /home/ramanlab/Documents/cole/Results/Opto/PER-Envelopes
        out_dir_smb: ramanfiles/cole/Figures/PER-Envelopes/
        # ... rest of config
```

**Example for reaction predictions:**
```yaml
reaction_prediction:
  output_csv: /home/ramanlab/Documents/cole/Data/Opto/Combined/model_predictions.csv
  output_csv_smb: ramanfiles/cole/flyTrackingData/model_predictions.csv
```

That's it! Files will now copy automatically after analysis completes.

## Common SMB Paths

Use these standard paths in your config:

| Type | SMB Path |
|------|----------|
| Figures/Results | `ramanfiles/cole/Figures/[ResultName]/` |
| CSV Data | `ramanfiles/cole/flyTrackingData/` |
| Custom | `ramanfiles/cole/[YourPath]/` |

## Verify It's Working

After running the pipeline:

```bash
# List what's on the SMB
rclone ls SMB-Ramanfile:ramanfiles/cole/Figures -h

# Or check specific folder
rclone ls SMB-Ramanfile:ramanfiles/cole/flyTrackingData
```

## Troubleshooting

### "Could not create file system for SMB-Ramanfile"

**Solution**: Re-run `rclone config` to verify credentials and host:
```bash
rclone config
# Navigate to SMB-Ramanfile and update password if needed
```

### "Connection refused" on SMB

**Check:**
```bash
ping ramanfile.local
# Should respond with an IP address

# Or test SMB port
nc -zv ramanfile.local 445
```

### Files not copying automatically

**Check config.yaml**: Make sure `out_dir_smb` is set for your analysis sections
**Check logs**: Run pipeline with verbose output to see copy status

### Want to test without running full pipeline?

```bash
python3 -c "
from fbpipe.utils.smb_copy import get_smb_copier
copier = get_smb_copier()
copier.test_connection()
"
```

## Backup System Integration

The automated backup system now also uses rclone for SMB:

```bash
# Run backups with rclone
python3 scripts/backup_system.py

# Backup with dry-run to preview
python3 scripts/backup_system.py --dry-run

# SMB-only backup
python3 scripts/backup_system.py --smb-only
```

## Advanced Options

### Copy with progress
```bash
rclone copy /local/path SMB-Ramanfile:ramanfiles/cole/ -v --progress
```

### Check what would be copied (dry-run)
```bash
rclone sync /local/path SMB-Ramanfile:ramanfiles/cole/ --dry-run -v
```

### Limit bandwidth
Add to analysis config or backup call:
```bash
rclone sync /src SMB-Ramanfile:dest --bwlimit 10M
```

## File Structure

- **Config**: `~/.config/rclone/rclone.conf` (encrypted credentials)
- **Utility**: `src/fbpipe/utils/smb_copy.py` (core copying functionality)
- **Docs**: `docs/SMB-RCLONE-SETUP.md` (detailed reference)
- **Test**: `scripts/test_smb_config.py` (validation script)

## Next Steps

1. ✅ Run `rclone config` to set up SMB-Ramanfile
2. ✅ Run `python3 scripts/test_smb_config.py` to verify
3. ✅ Add `out_dir_smb` paths to your `config/config.yaml`
4. ✅ Run pipeline - SMB copies happen automatically!

## Questions or Issues?

- See detailed docs: [SMB-RCLONE-SETUP.md](SMB-RCLONE-SETUP.md)
- Test config: `python3 scripts/test_smb_config.py`
- Check logs: `backup.log` (for backup system)

---

**Your SMB share is ready to receive files! 🎉**
