# Jupyter Pipeline Runner Guide

Run your entire analysis pipeline from a Jupyter notebook, perfect for auto-running after YOLO training.

## Quick Start

1. **Open the notebook:**
   ```bash
   jupyter notebook Auto_Pipeline_Runner.ipynb
   ```

2. **Run cells in order:**
   - Setup
   - Find Latest YOLO Model
   - Update Config (Optional A or B)
   - Execute: Run Full Pipeline

3. **Monitor progress** - Output appears in real-time

4. **Check results** - Automatically generated results summary

## What It Does

### Cell 1: Setup
- Initializes paths
- Creates logs directory
- Verifies configuration

### Cell 2: Find Latest YOLO Model
- Scans `model/YOLOProjectProboscisLegs/runs/obb/` for new models
- Shows the newest trained model
- Displays model info (size, date modified)

### Cell 3-5: Update Config
**Option A (Recommended):** Auto-update with latest model
```python
update_config_with_model(latest_model)
```

**Option B:** Manual model path
```python
manual_model_path = "/path/to/your/model.pt"
update_config_with_model(manual_model_path)
```

### Cell 6-7: Run Pipeline
Executes: `make run`

This includes:
1. ✅ CSV backup (before)
2. 🔄 Full analysis pipeline
3. ✅ CSV backup (after)

**Real-time output** appears as it runs.

### Cell 8: Check Results
Shows:
- Generated result files and folders
- CSV output files
- Compressed backups created

### Cell 9: View Logs
Displays last 50 lines of pipeline log for debugging.

## Workflow Examples

### After Training New YOLO Model

```
1. Train YOLO model (in separate notebook/terminal)
2. Open Auto_Pipeline_Runner.ipynb
3. Run Setup cell
4. Run Find Latest Model cell
5. Run Update Config cell (auto-detects new model)
6. Run Execute Pipeline cell (monitors progress)
7. Check Results cell (view outputs)
```

### Scheduled Daily Runs

If you want fully automatic runs without notebook interaction, use cron instead:
```bash
./scripts/setup_backup_cron.sh
```

Then the pipeline runs automatically at scheduled times.

### Manual Restart with Different Model

```python
# Manually specify model path
model_path = "/path/to/specific/model.pt"
update_config_with_model(model_path)

# Then run pipeline cell
run_pipeline()
```

## What Gets Backed Up

After pipeline completes, automatically backs up:
- ✅ All CSV analysis files
- ✅ All results/figures
- ✅ Sent to Box Cloud (incremental)
- ✅ Sent to SMB share (network access)

## Monitoring Long Runs

The pipeline can take 15-60+ minutes depending on data size.

**Monitor from other terminal:**
```bash
# Watch backup logs
tail -f logs/backup.log

# Watch main logs
tail -f logs/pipeline_run_*.log

# Check job status
ps aux | grep python
```

## Tips

✅ **Run from Jupyter:** Perfect for interactive notebooks

✅ **One notebook per training:** Create copies for different experiments

✅ **Check logs:** All output saved to `logs/pipeline_run_TIMESTAMP.log`

✅ **Manual commands:** Can run `make run` directly in terminal

✅ **Combine with cron:** Use notebook for manual runs, cron for scheduled

## Troubleshooting

### Model Not Auto-Detected
- Check YOLO training output folder
- Verify `model/YOLOProjectProboscisLegs/runs/obb/` exists
- Use Option B for manual path

### Pipeline Fails
- Check error output in notebook
- View full log file
- Run `make backup-csvs --dry-run` to test backup

### Slow Performance
- Close other applications
- Run during off-peak hours
- Monitor with `htop`

### Disk Space Issues
- Clean old archives: `make clean-backups`
- Check space: `df -h`
- Reduce archive generation frequency

## Full Example: From Training to Results

```python
# 1. Setup
# Run Setup cell to initialize

# 2. Train YOLO model elsewhere, then...

# 3. Auto-detect new model
latest_model = find_latest_yolo_model()

# 4. Update config
update_config_with_model(latest_model)

# 5. Run pipeline (15-60+ minutes)
success = run_pipeline()

# 6. Check results
if success:
    check_pipeline_results()
    print("Analysis complete and backed up!")

# 7. Optional: Create compressed archive
os.chdir(PROJECT_ROOT)
subprocess.run(['make', 'backup-compressed'])
```

## Integration with YOLO Training

If you train YOLO in a separate notebook, you can:

1. **Train notebook:**
   ```python
   # Train YOLO
   results = model.train(...)
   print(f"Model saved to: {results.save_dir}")
   ```

2. **Pipeline notebook:**
   ```python
   # Immediately run pipeline
   latest_model = find_latest_yolo_model()
   update_config_with_model(latest_model)
   run_pipeline()
   ```

Or use **JupyterHub's multi-notebook support** to run both in parallel.

## Advanced: Custom Pipeline Options

For more control, pass custom arguments:

```python
# Use custom config
os.chdir(PROJECT_ROOT)
subprocess.run(['python', 'scripts/pipeline/run_workflows.py', '--config', 'config/custom.yaml'])

# Run specific steps only
subprocess.run(['make', 'backup-csvs'])  # Just backup CSVs
subprocess.run(['make', 'yolo'])          # Just YOLO inference
```

## Related Documentation

- [BACKUP_USAGE_GUIDE.md](BACKUP_USAGE_GUIDE.md) - All backup commands
- [BACKUPS_README.md](BACKUPS_README.md) - Backup system overview
- [QUICK_BACKUP_REFERENCE.txt](../QUICK_BACKUP_REFERENCE.txt) - Quick commands

## Summary

**Auto Pipeline Runner Notebook** provides:
- ✅ Auto-detects new YOLO models
- ✅ Runs full pipeline from Jupyter
- ✅ Real-time progress monitoring
- ✅ Automatic backup integration
- ✅ Results verification
- ✅ Log viewing and debugging

Perfect for running analysis after model training! 🚀
