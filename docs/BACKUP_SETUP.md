# Box Cloud Backup Setup Guide

This guide helps you set up automatic backups of your Ramanlab data to Box cloud storage.

## Prerequisites

1. **Box Account** - You need access to a Box account
2. **Rclone** - A command-line cloud sync tool

## Step 1: Install Rclone

```bash
# Using package manager (Linux/Mac)
curl https://rclone.org/install.sh | sudo bash

# Or using Homebrew (Mac)
brew install rclone

# Or using Windows installer
# Download from https://rclone.org/downloads/
```

Verify installation:
```bash
rclone version
```

## Step 2: Configure Box Remote

Run the interactive setup:

```bash
python backup_to_box.py --setup
```

Or manually:
```bash
rclone config
```

Follow these steps:
1. Choose `n` for new remote
2. Enter name: `box`
3. Choose storage type: `box`
4. Follow Box authentication prompts
5. Select standard Box for business (unless you use Box Developer)
6. Authorize when prompted in your browser
7. Type `y` to confirm and save

## Step 3: Test the Backup

Do a dry-run to see what would be backed up:

```bash
python backup_to_box.py --dry-run
```

Or test only CSVs:
```bash
python backup_to_box.py --csvs-only --dry-run
```

## Step 4: Run First Full Backup

Once confirmed, run the actual backup:

```bash
python backup_to_box.py
```

## How Incremental Backups Work

The backup script only uploads **new and modified files**, not the entire dataset each time:

- **First run**: All files are uploaded to Box
- **Subsequent runs**: Only files that are newer than what's already on Box are uploaded
- **Deleted files**: Kept on Box (safe archive approach)

This means:
- ✓ Fast backups after the first run
- ✓ Minimal bandwidth usage
- ✓ Safe (deleted local files aren't removed from Box)
- ✓ Perfect for scheduling after each pipeline run

### Checking What Will Be Backed Up

Before running a backup, use dry-run to see exactly which files will be synced:

```bash
python backup_to_box.py --dry-run
```

This shows only the files that will be transferred, not existing ones.

## Backup Options

### Manual Backups

Run specific backup types:

```bash
# Backup only CSV files
python backup_to_box.py --csvs-only

# Backup only results/figures
python backup_to_box.py --results-only

# Backup only data folders
python backup_to_box.py --data-only

# Backup only model files
python backup_to_box.py --models-only
```

### Automatic Scheduled Backups

#### Linux/Mac - Using Cron

Edit crontab:
```bash
crontab -e
```

Add one of these entries:

```bash
# Daily backup at 2 AM
0 2 * * * cd /home/ramanlab/Documents/cole/VSCode/Ramanlab-Auto-Data-Analysis && python backup_to_box.py

# Weekly backup every Sunday at 3 AM
0 3 * * 0 cd /home/ramanlab/Documents/cole/VSCode/Ramanlab-Auto-Data-Analysis && python backup_to_box.py

# Daily backup of CSVs only at midnight
0 0 * * * cd /home/ramanlab/Documents/cole/VSCode/Ramanlab-Auto-Data-Analysis && python backup_to_box.py --csvs-only
```

#### After Pipeline Completion

If you want backups to run after your analysis pipeline completes, add this to your pipeline script:

```python
import subprocess
import logging

logger = logging.getLogger(__name__)

def run_backup():
    """Backup results to Box after pipeline completes."""
    try:
        result = subprocess.run(
            ["python", "/home/ramanlab/Documents/cole/VSCode/Ramanlab-Auto-Data-Analysis/backup_to_box.py"],
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        if result.returncode == 0:
            logger.info("✓ Backup to Box completed successfully")
        else:
            logger.error(f"✗ Backup to Box failed: {result.stderr}")
    except Exception as e:
        logger.error(f"Error running backup: {e}")

# Call this at the end of your pipeline:
if __name__ == "__main__":
    # ... run your analysis pipeline ...

    # After pipeline completes:
    run_backup()
```

## Backup Location

All backups are stored in your Box account under:
```
Ramanlab-Backups/
├── CSVs/
│   ├── all_envelope_rows_wide.csv
│   ├── all_envelope_rows_wide_training.csv
│   └── ...
├── Results/
│   ├── Reaction_Matrices/
│   ├── PER-Envelopes/
│   ├── Training-PER-Envelopes/
│   └── ...
├── Data/
│   ├── flys/
│   └── Opto/
└── Models/
    └── weights/
```

## Monitoring Backups

Check the backup log:
```bash
tail -f backup.log
```

## Troubleshooting

### Rclone not found
```bash
pip install rclone
```

### Box remote not configured
```bash
python backup_to_box.py --setup
```

### Authentication errors
Delete the config and reconfigure:
```bash
rm ~/.config/rclone/rclone.conf
python backup_to_box.py --setup
```

### Slow backups
- Use `--csvs-only` for quick backups of just data files
- Schedule large data backups during off-hours
- Exclude certain file types if needed

### Check current backups in Box
```bash
rclone ls box:Ramanlab-Backups
rclone lsd box:Ramanlab-Backups
```

## Storage Limits

Box offers:
- **Free tier**: 10 GB
- **Paid tier**: Typically 100 GB+ depending on plan

Monitor your usage:
```bash
rclone about box:
```

## Restore from Backup

To restore files from Box:

```bash
# Restore CSVs
rclone sync box:Ramanlab-Backups/CSVs /home/ramanlab/Documents/cole/Data/Opto/Combined/

# Restore all results
rclone sync box:Ramanlab-Backups/Results /home/ramanlab/Documents/cole/Results/

# Restore a specific folder
rclone sync box:Ramanlab-Backups/Results/Reaction_Matrices /home/ramanlab/Documents/cole/Results/Opto/Reaction_Matrices
```

## Support

For issues with:
- **Rclone**: https://rclone.org/
- **Box API**: https://developer.box.com/
- **Script bugs**: Check backup.log for detailed error messages
