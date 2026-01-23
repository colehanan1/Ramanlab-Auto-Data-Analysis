# Backup System - Complete Setup

All backup files are now organized and integrated with the pipeline!

## 📁 File Organization

```
scripts/
├── backup_to_box.py ................... Incremental backup script
├── compress_and_backup.py ............ Compressed backup script
├── run_all_backups.sh ................ Master backup runner
├── run_backup.sh ..................... Legacy wrapper
└── setup_backup_cron.sh .............. Automatic scheduling setup

docs/
├── BACKUP_SETUP.md ................... Initial setup (read first!)
├── BACKUP_USAGE_GUIDE.md ............ Complete command reference
├── COMPRESSION_BACKUP_GUIDE.md ...... Compression details
├── DATA_SAFETY_SUMMARY.md ........... System overview
└── BACKUPS_README.md ................ This file
```

## 🚀 Quick Start

### Step 1: View Available Commands

```bash
make help
```

You'll see:
```
  backup    - run full incremental backup to Box
  backup-csvs - backup only CSVs to Box
  backup-compressed - create compressed emergency archives
  clean-backups - delete old compressed archives
```

### Step 2: Run a Test Backup

```bash
# Preview (dry-run) - no files actually uploaded
python scripts/backup_to_box.py --dry-run

# OR via Make
make backup --dry-run
```

### Step 3: Enable Automatic Backups (Optional)

```bash
./scripts/setup_backup_cron.sh
```

This schedules:
- ✅ Daily at 2:00 AM - CSV backups
- ✅ Weekly (Sundays) 3:00 AM - Compressed archives

---

## 📊 Three-Layer Backup System

### Layer 1: Incremental Backups (Automatic)

**What:** All files, only new/modified uploaded
**Where:** Box Cloud
**When:** After each pipeline run
**Size:** Full dataset (no compression)
**Time:** 1-5 minutes
**Command:**
```bash
make run              # Pipeline + auto backup
make backup-csvs      # CSV backup only (fastest)
make backup           # Full backup
```

### Layer 2: SMB Network Share (Fast Access)

**What:** Original files
**Where:** `smb://ramanfile.local/ramanfiles/cole/`
**When:** Configured in your config.yaml
**Time:** Immediate (network access)
**Locations:**
- `flyTrackingData/` - CSV files
- `Figures/` - Results

### Layer 3: Compressed Archives (Emergency)

**What:** ZIP files, 60-96% compressed
**Where:** SMB + Box Cloud + Local folder
**When:** Weekly or manual
**Time:** 5-20 minutes
**Command:**
```bash
make backup-compressed        # Create compressed archive
make clean-backups            # Delete old ones (>30 days)
./scripts/run_all_backups.sh --full  # Everything together
```

---

## 💻 Most Common Commands

### During Development
```bash
# After running analysis
make run           # Runs pipeline + auto backup CSVs
```

### Weekly Maintenance
```bash
# Create emergency archive
make backup-compressed

# View status
crontab -l
```

### Manual Backups
```bash
# Quick CSV backup
make backup-csvs

# Full backup
make backup

# Compressed backup
make backup-compressed

# See what would be backed up (no upload)
python scripts/backup_to_box.py --dry-run
```

---

## 🔄 Pipeline Integration

The backup system is **automatically integrated** with the pipeline:

```bash
make run
# Runs:
# 1. Backup CSVs (before processing)
# 2. Run full pipeline
# 3. Backup CSVs (after processing)
```

No additional steps needed!

---

## ⏰ Automatic Scheduling

### Install Auto-Backups

```bash
./scripts/setup_backup_cron.sh
```

### View Scheduled Backups

```bash
crontab -l
```

### Remove Scheduled Backups

```bash
crontab -e
# Delete backup entries manually
```

---

## 📚 Documentation

| File | Purpose |
|------|---------|
| [BACKUP_SETUP.md](BACKUP_SETUP.md) | Initial setup & Box configuration |
| [BACKUP_USAGE_GUIDE.md](BACKUP_USAGE_GUIDE.md) | Complete command reference |
| [COMPRESSION_BACKUP_GUIDE.md](COMPRESSION_BACKUP_GUIDE.md) | Compression details & recovery |
| [DATA_SAFETY_SUMMARY.md](DATA_SAFETY_SUMMARY.md) | System overview & best practices |

**Start with:** [BACKUP_SETUP.md](BACKUP_SETUP.md) if this is your first time

**For all commands:** [BACKUP_USAGE_GUIDE.md](BACKUP_USAGE_GUIDE.md)

---

## ✅ Verification Checklist

- [ ] Box remote configured: `rclone listremotes`
- [ ] SMB location accessible: `ls /path/to/smb/ramanfiles/`
- [ ] Backups folder created: `ls /home/ramanlab/Documents/cole/backups_compressed/`
- [ ] Make commands work: `make help | grep backup`
- [ ] Test backup runs: `python scripts/backup_to_box.py --dry-run`
- [ ] Cron setup (optional): `./scripts/setup_backup_cron.sh`

---

## 🆘 Troubleshooting

### Box not working?
```bash
rclone config         # Reconfigure Box
rclone lsd Box-Folder:Ramanlab-Backups/  # Test access
```

### SMB not accessible?
```bash
mount | grep ramanfiles  # Check if mounted
ls -l /path/to/smb/ramanfiles/  # Verify path
```

### Backups running slow?
```bash
# Check logs
tail -f logs/backup_*.log

# Run smaller backup
make backup-csvs      # Just CSVs (faster)
```

### Low disk space?
```bash
make clean-backups    # Delete old compressed archives
du -sh backups_compressed/  # Check size
```

---

## 📋 File Manifest

### Scripts (5 files)
- `backup_to_box.py` - Incremental backup to Box (11K)
- `compress_and_backup.py` - Compressed backup (14K)
- `run_all_backups.sh` - Master runner (3.3K)
- `run_backup.sh` - Legacy wrapper (929B)
- `setup_backup_cron.sh` - Cron setup (2.0K)

### Documentation (4 files)
- `BACKUP_SETUP.md` - Setup guide (5.7K)
- `BACKUP_USAGE_GUIDE.md` - Commands reference (8.4K)
- `COMPRESSION_BACKUP_GUIDE.md` - Compression guide (6.2K)
- `DATA_SAFETY_SUMMARY.md` - Overview (5.7K)

**Total:** ~65 KB of code + documentation

---

## 🎯 Next Steps

1. **Read:** [BACKUP_SETUP.md](BACKUP_SETUP.md) (5 minutes)
2. **Test:** `python scripts/backup_to_box.py --dry-run` (2 minutes)
3. **Use:** `make run` for normal operations (automatic)
4. **Schedule:** `./scripts/setup_backup_cron.sh` for auto-backups (optional)

---

## 🔐 Your Data is Protected

✅ **3-layer redundancy:**
- Incremental backups to Box
- SMB network share access
- Compressed emergency archives

✅ **Automatic:** Integrated with pipeline

✅ **Efficient:** Only changed files uploaded

✅ **Organized:** Everything in proper folders

✅ **Documented:** Complete guides included

---

## 💡 Quick Commands Cheat Sheet

```bash
# Most common
make run                    # Pipeline + auto backup
make backup-csvs            # Quick CSV backup
make backup-compressed      # Emergency archive

# Setup
./scripts/setup_backup_cron.sh  # Auto-schedule

# Maintenance
make clean-backups          # Delete old archives
crontab -l                  # View scheduled backups

# Troubleshooting
python scripts/backup_to_box.py --dry-run  # Preview
rclone listremotes          # Check Box config
ls backups_compressed/      # View local archives
```

---

## 📞 Getting Help

1. Check logs: `tail -f logs/backup_*.log`
2. Run dry-run: `python scripts/backup_to_box.py --dry-run`
3. Read guide: [BACKUP_USAGE_GUIDE.md](BACKUP_USAGE_GUIDE.md)
4. Check status: `crontab -l`

---

**Your backup system is ready! Start with `make run` 🚀**
