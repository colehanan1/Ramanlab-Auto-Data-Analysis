# 🚀 SMB Backup System - START HERE

Your SMB file copying system using rclone has been fully implemented!

## What You Need to Do (5 minutes)

### 1️⃣ Configure Rclone
```bash
rclone config
```
Follow the prompts to create `SMB-Ramanfile` remote:
- Name: `SMB-Ramanfile`
- Type: `smb`
- Host: `ramanfile.local`
- Username: `ramanlab`
- Password: *(enter your password)*
- Port: `445` (press Enter)
- Domain: *(leave empty, press Enter)*
- Confirm: `y`

### 2️⃣ Test It Works
```bash
python3 scripts/test_smb_config.py
```
Should show all ✓ passing tests.

### 3️⃣ Update Your Config
Edit `config/config.yaml` and add `out_dir_smb` paths:

**Example:**
```yaml
analysis:
  envelope_visuals:
    envelopes:
      - after_show_sec: 30.0
        out_dir: /home/ramanlab/Documents/cole/Results/Opto/PER-Envelopes
        out_dir_smb: ramanfiles/cole/Figures/PER-Envelopes/

reaction_prediction:
  output_csv: /path/to/output.csv
  output_csv_smb: ramanfiles/cole/flyTrackingData/output.csv
```

### 4️⃣ Run Pipeline
```bash
python3 scripts/pipeline/run_workflows.py
```
Files now auto-copy to SMB! ✨

## Documentation

| Document | When to Read |
|----------|------------|
| **[SMB-SETUP-QUICK-START.md](docs/SMB-SETUP-QUICK-START.md)** | Setup help & troubleshooting |
| **[SMB-QUICK-REFERENCE.md](docs/SMB-QUICK-REFERENCE.md)** | Common commands & examples |
| **[SMB-RCLONE-SETUP.md](docs/SMB-RCLONE-SETUP.md)** | Complete detailed reference |
| **[RCLONE-SMB-IMPLEMENTATION.md](RCLONE-SMB-IMPLEMENTATION.md)** | Technical details & architecture |
| **[SMB-IMPLEMENTATION-CHECKLIST.md](SMB-IMPLEMENTATION-CHECKLIST.md)** | Setup checklist & verification |
| **[FILES-CHANGED.md](FILES-CHANGED.md)** | What was implemented |

## Quick Commands

```bash
# Test configuration
python3 scripts/test_smb_config.py

# Run pipeline with auto-SMB copying
python3 scripts/pipeline/run_workflows.py

# Check what's on SMB
rclone ls SMB-Ramanfile:ramanfiles/cole/Figures/ -h

# Full backup to all destinations
python3 scripts/backup_system.py

# SMB-only backup
python3 scripts/backup_system.py --smb-only
```

## What's Been Implemented

✅ **Core SMB Utility** - `src/fbpipe/utils/smb_copy.py`
- File copying with smart checksums
- Directory syncing
- Error handling & logging

✅ **Pipeline Integration** - Auto-copy after each analysis step
- Envelope visuals
- Training analysis
- Combined analysis
- Reaction predictions

✅ **Configuration Support** - New `out_dir_smb` and `output_csv_smb` fields
- Works with all analysis sections
- Backward compatible

✅ **Backup System** - Enhanced with rclone
- Reliable SMB backups
- Fallback to rsync if needed

✅ **Comprehensive Documentation** - 4 guides + checklist
- Setup instructions
- Command reference
- Troubleshooting
- Examples

✅ **Automated Testing** - `scripts/test_smb_config.py`
- Validates configuration
- Tests connection
- Provides guidance

## Common SMB Paths

Use these in your config.yaml:

```
ramanfiles/cole/Figures/[YourResultName]/     ← Results/figures
ramanfiles/cole/flyTrackingData/              ← CSV data
ramanfiles/cole/Backups/                      ← Backups
ramanfiles/cole/[CustomPath]/                 ← Custom locations
```

## Troubleshooting

### Can't connect to SMB?
```bash
# Check host
ping ramanfile.local

# Re-configure
rclone config
```

### Files not copying?
- Check `out_dir_smb` is set in config.yaml
- Check logs for error messages
- Run test script: `python3 scripts/test_smb_config.py`

### Need bandwidth limiting?
Add to rclone commands: `--bwlimit 10M`

## Next Steps

1. ✅ Run `rclone config`
2. ✅ Run `python3 scripts/test_smb_config.py`
3. ✅ Update `config/config.yaml`
4. ✅ Run pipeline
5. ✅ Verify files on SMB

---

**Everything is ready! Your files will now automatically copy to SMB after each analysis completes.** 🎉

For details, see: **[SMB-SETUP-QUICK-START.md](docs/SMB-SETUP-QUICK-START.md)**
