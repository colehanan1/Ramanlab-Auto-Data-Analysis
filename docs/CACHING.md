# Smart Caching System Documentation

## Overview

The Ramanlab Auto Data Analysis pipeline includes a **smart, manifest-based caching system** that dramatically reduces processing time by skipping datasets that haven't changed since the last run.

### Key Benefits

- ✅ **Intelligent File Tracking**: Monitors CSV files for changes using modification time and file size
- ✅ **Automatic Cache Invalidation**: Detects new, modified, or deleted files and reprocesses only when needed
- ✅ **Transparent Logging**: Clear messages explain why each dataset is cached or reprocessed
- ✅ **Force Flags**: Manual control to bypass cache when needed
- ✅ **Fast Performance**: ~0.2ms per file (180ms for 900 files)

---

## How It Works

### 1. First Run (No Cache)

When you run `make run` for the first time:

```bash
$ make run

[analysis] pipeline → processing /path/to/.../opto_EB/
2025-12-11 15:30:00 [INFO] Built file manifest for opto_EB: 247 files in 0.05s
[YOLO] Processing opto_EB...                    # ← Full processing
[DISTANCE] Normalizing distances...
[RMS] Calculating envelopes...
...
✓ Pipeline complete: 5 minutes
```

**What happens:**
- Pipeline processes all datasets completely
- Builds a **file manifest** for each dataset (tracks mtime + size of all CSV files)
- Saves manifest to cache directory: `~/.cache/ramanlab_auto_data_analysis/pipeline/<dataset>/state.json`
- Saves configuration values (non_reactive_span_px, etc.) to same state file

### 2. Second Run (Cached, No Changes)

When you run `make run` again without any file changes:

```bash
$ make run

[CACHE HIT] No file changes in opto_EB (247 files checked)
[CACHE HIT] No file changes in opto_hex (312 files checked)
[CACHE HIT] No file changes in opto_ACV (189 files checked)
...
[analysis] pipeline skipped for all datasets (cached).
✓ Pipeline complete: 2 seconds                   # ← 150x faster!
```

**What happens:**
- Pipeline checks cached manifests for each dataset
- Compares current file mtime/size against cached values
- All files unchanged → **CACHE HIT** → Skip processing ✅

### 3. After Adding New Data

When you add new CSV files to a dataset:

```bash
$ make run

[CACHE HIT] No file changes in opto_EB (247 files checked)
[CACHE MISS] File changes detected: 2 new, 0 modified, 0 deleted
[analysis] pipeline → processing /path/to/.../opto_hex/
[YOLO] Processing opto_hex...                    # ← Only this dataset
...
[CACHE HIT] No file changes in opto_ACV (189 files checked)
✓ Pipeline complete: 1 minute                    # ← Only reprocesses opto_hex
```

**What happens:**
- Pipeline detects 2 new CSV files in `opto_hex/`
- **CACHE MISS** → Reprocess only `opto_hex` 🔄
- Other datasets remain cached ✅

---

## Cache Decision Tree

The pipeline follows this logic for each dataset:

```
┌─────────────────────────────────────┐
│ Is force.pipeline = true?           │
├─────────────────────────────────────┤
│ YES → Skip cache, reprocess         │ ← Manual override
│ NO  → Continue checking...          │
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│ Does cached state.json exist?       │
├─────────────────────────────────────┤
│ NO  → CACHE MISS, reprocess         │ ← First run
│ YES → Continue checking...          │
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│ Does state version match?           │
├─────────────────────────────────────┤
│ NO  → CACHE MISS, reprocess         │ ← Pipeline upgrade
│ YES → Continue checking...          │
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│ Do config values match?              │
│ (non_reactive_span_px, etc.)         │
├─────────────────────────────────────┤
│ NO  → CACHE MISS, reprocess         │ ← Config changed
│ YES → Continue checking...          │
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│ Does file manifest match?            │
│ (mtime + size of all CSV files)      │
├─────────────────────────────────────┤
│ NO  → CACHE MISS, reprocess         │ ← Files changed
│ YES → CACHE HIT, skip! ✅            │ ← Use cached results
└─────────────────────────────────────┘
```

---

## File Manifest System

### What Files Are Tracked?

The manifest tracks **CSV files in nested directories** (fly/trial structure):

✅ **Tracked:**
- `dataset/fly1/trial1/coords.csv`
- `dataset/fly2/trial3/updated_fly2_trial3_distances.csv`
- `dataset/batch_1/RMS_calculations/updated_batch_1_testing_1_fly1_distances.csv`

❌ **NOT Tracked:**
- `dataset/metadata.csv` (root level)
- `dataset/fly1/sensors_temp.csv` (sensors_ prefix)
- `dataset/fly1/.hidden.csv` (hidden files)
- `dataset/fly1/data.txt` (non-CSV)

### Manifest Structure

Each dataset's manifest is stored in `state.json`:

```json
{
  "version": 1,
  "non_reactive_span_px": 7.5,
  "file_manifest": {
    "/path/to/dataset/fly1/trial1/coords.csv": {
      "mtime": 1765417790.8924167,
      "size": 1412238
    },
    "/path/to/dataset/fly1/trial2/coords.csv": {
      "mtime": 1765417791.5234310,
      "size": 1408297
    }
  }
}
```

### Change Detection

The system detects three types of changes:

1. **New Files**: CSV file added to dataset
   ```
   [CACHE MISS] File changes detected: 1 new, 0 modified, 0 deleted
     + fly3/trial1/coords.csv
   ```

2. **Modified Files**: CSV file's mtime or size changed
   ```
   [CACHE MISS] File changes detected: 0 new, 1 modified, 0 deleted
     ~ fly1/trial1/coords.csv (mtime: 1765417790 → 1765417800)
   ```

3. **Deleted Files**: CSV file removed from dataset
   ```
   [CACHE MISS] File changes detected: 0 new, 0 modified, 1 deleted
     - fly2/trial3/bad_trial.csv
   ```

---

## Managing the Cache

### View Cache Status

```bash
scripts/dev/cache_manager.sh status
```

Output:
```
=== Cache Status ===

Cache directory: /path/to/cache
Total size: 2.0M

Pipeline caches: 10 datasets
Combined analysis cache: Present
  └─ Covers 10 datasets
Reaction prediction cache: Present
Reaction matrix cache: Present

Next make run will:
  ✓ Use cached pipelines (force.pipeline: false)
  ✓ Use cached combined analysis (force.combined: false)
```

### Force Reprocessing

#### Method 1: Edit `config/config.yaml`

```yaml
force:
  pipeline: true              # Reprocess ALL datasets (ignore cache)
  yolo: true                  # Reprocess YOLO step only
  combined: true              # Reprocess combined analysis
  reaction_prediction: true   # Rerun prediction model
  reaction_matrix: true       # Regenerate reaction matrices
```

After changing force flags, run:
```bash
make run                      # Will bypass cache for enabled flags
```

**Remember to set flags back to `false` after running!**

#### Method 2: Clear Cache Manually

```bash
# Clear ALL caches (forces full reprocessing)
scripts/dev/cache_manager.sh clear-all

# Clear only pipeline caches
scripts/dev/cache_manager.sh clear-pipeline

# Clear specific dataset
scripts/dev/cache_manager.sh clear-dataset /path/to/Data/flys/opto_EB

# Clear combined analysis only
scripts/dev/cache_manager.sh clear-combined

# Clear reaction prediction/matrix
scripts/dev/cache_manager.sh clear-reactions
```

### List Cached Datasets

```bash
scripts/dev/cache_manager.sh list-datasets
```

Output:
```
=== Cached Datasets ===

1. /path/to/Data/flys/opto_EB
2. /path/to/Data/flys/opto_hex
3. /path/to/Data/flys/opto_ACV
...

Total: 10 datasets
```

---

## Performance Characteristics

### Manifest Building Speed

- **~0.2ms per file** (based on mtime + size, no hashing)
- **180ms for 900 files** (typical dataset)
- **Scales linearly** with file count

### Cache Hit vs Miss Timing

| Scenario | Datasets Processed | Time | Speedup |
|----------|-------------------|------|---------|
| First run (no cache) | 10 datasets | ~10 min | 1x (baseline) |
| Second run (cached) | 0 datasets | ~2 sec | **300x faster** |
| After adding 1 new video | 1 dataset | ~1 min | **10x faster** |
| After changing config | 10 datasets | ~10 min | 1x (cache invalidated) |

---

## Common Scenarios

### Scenario 1: Daily Analysis Workflow

**Situation**: You run analysis every morning to check for new data.

**Workflow**:
```bash
# Morning: Check if new data arrived overnight
make run

# Output:
[CACHE HIT] No file changes in opto_EB (247 files checked)
[CACHE MISS] File changes detected: 3 new, 0 modified, 0 deleted
  + fly4/trial1/coords.csv
  + fly4/trial2/coords.csv
  + fly4/trial3/coords.csv
[analysis] pipeline → processing opto_EB...
✓ Pipeline complete: 1 minute
```

**Result**: Only new data is processed! 🎉

### Scenario 2: Config Tuning

**Situation**: You're experimenting with `non_reactive_span_px` threshold.

**Workflow**:
```yaml
# config/config.yaml - try new threshold
non_reactive_span_px: 10.0    # was 7.5
```

```bash
make run

# Output:
[CACHE MISS] Config changed: non_reactive_span_px (7.5 → 10.0)
[analysis] pipeline → processing all datasets...
✓ Pipeline complete: 10 minutes
```

**Result**: All datasets reprocessed with new threshold. Cache will be updated.

### Scenario 3: Force Regeneration for Presentation

**Situation**: You want fresh outputs for a presentation, even if nothing changed.

**Workflow**:
```yaml
# config/config.yaml - force reprocessing
force:
  combined: true              # Regenerate combined analysis
  reaction_matrix: true       # Regenerate plots
```

```bash
make run

# Output:
[analysis] combined analysis → reprocessing (force.combined: true)
[analysis] reaction matrix → regenerating (force.reaction_matrix: true)
✓ Pipeline complete: 3 minutes
```

**Remember to set back to `false` afterward!**

### Scenario 4: Troubleshooting Bad Cache

**Situation**: Results look wrong, you suspect cache corruption.

**Workflow**:
```bash
# Nuclear option: clear everything
scripts/dev/cache_manager.sh clear-all

# Or targeted clearing
scripts/dev/cache_manager.sh clear-dataset /path/to/suspicious/dataset

# Then rerun
make run
```

**Result**: Fresh processing with no cached state.

---

## Technical Details

### State File Location

Caches are stored in:
```
~/.cache/ramanlab_auto_data_analysis/
├── pipeline/
│   ├── home_ramanlab_Documents_cole_Data_flys_opto_EB/
│   │   └── state.json                    ← Dataset cache
│   ├── home_ramanlab_Documents_cole_Data_flys_opto_hex/
│   │   └── state.json
│   └── ...
├── combined/
│   └── analysis/
│       └── state.json                    ← Combined analysis cache
├── reaction_prediction/
│   └── predict/
│       └── state.json                    ← Prediction cache
└── reaction_matrix/
    └── reaction_prediction/
        └── state.json                    ← Matrix cache
```

### Cache Categories

The system uses separate cache categories:

| Category | Purpose | Force Flag | Manifest Tracking |
|----------|---------|------------|-------------------|
| `pipeline` | Per-dataset processing (YOLO, distance, RMS) | `force.pipeline` | ✅ Yes (per dataset) |
| `combined` | Cross-dataset combined analysis | `force.combined` | ✅ Yes (all datasets) |
| `reaction_prediction` | ML model predictions | `force.reaction_prediction` | ❌ No (mtime only) |
| `reaction_matrix` | Reaction matrix generation | `force.reaction_matrix` | ❌ No (config only) |

### Logged Messages Reference

#### Cache Hit Messages

```
[CACHE HIT] No file changes in opto_EB (247 files checked)
```
- **Meaning**: All 247 CSV files unchanged, using cached results
- **Action**: Pipeline skips this dataset entirely

#### Cache Miss Messages

```
[CACHE MISS] File changes detected: 2 new, 1 modified, 0 deleted
  + fly3/trial1/coords.csv
  + fly3/trial2/coords.csv
  ~ fly1/trial1/coords.csv
```
- **Meaning**: Dataset has file changes, must reprocess
- **Action**: Pipeline processes this dataset from scratch

```
[CACHE MISS] Config changed: non_reactive_span_px (7.5 → 10.0)
```
- **Meaning**: Configuration value changed, must reprocess
- **Action**: Pipeline processes this dataset with new config

```
[CACHE MISS] No cached state found
```
- **Meaning**: First time processing this dataset
- **Action**: Pipeline processes from scratch and creates cache

#### Force Flag Messages

```
[pipeline] force flag set, bypassing cache
```
- **Meaning**: `force.pipeline: true` in config
- **Action**: Pipeline ignores cache and reprocesses

---

## Best Practices

### ✅ DO

- **Let the cache work**: Set all `force` flags to `false` for daily workflows
- **Use targeted clearing**: Clear specific datasets instead of `clear-all`
- **Check status first**: Run `scripts/dev/cache_manager.sh status` before clearing
- **Trust the cache**: File manifest tracking is reliable (mtime + size)

### ❌ DON'T

- **Don't leave force flags enabled**: Always set back to `false` after forced reprocessing
- **Don't manually edit state.json**: Use scripts/dev/cache_manager.sh instead
- **Don't delete cache directory**: Use `clear-all` command instead
- **Don't worry about hash collisions**: mtime + size is sufficient for CSV files

---

## Troubleshooting

### Problem: Cache not working (always reprocessing)

**Check**:
```yaml
# config/config.yaml
force:
  pipeline: false    # Must be false!
  combined: false    # Must be false!
```

**Solution**: Set force flags to `false`

### Problem: Results seem stale despite new data

**Check**:
```bash
scripts/dev/cache_manager.sh status
# Look for "Pipeline caches: X datasets"
```

**Solution**:
```bash
scripts/dev/cache_manager.sh clear-pipeline
make run
```

### Problem: "Large manifest" warning

**Message**:
```
[WARNING] Large manifest: 5200 files in opto_mega_dataset. Consider dataset splitting.
```

**Solution**: Large datasets (>5000 files) work fine but are slower. Consider splitting into subdirectories if manifest building takes >1 second.

### Problem: Cache taking too much disk space

**Check**:
```bash
du -sh ~/.cache/ramanlab_auto_data_analysis
```

**Solution**: Cache size is minimal (~2MB for 10 datasets). If large:
```bash
scripts/dev/cache_manager.sh clear-all
```

---

## Testing

Comprehensive tests for the caching system are available:

```bash
# Run standalone tests
python test_manifest_standalone.py

# Expected output:
# 🎉 All tests passed! Caching system is working correctly.
```

Tests cover:
- ✅ File tracking filters
- ✅ Manifest building
- ✅ Manifest comparison (new, modified, deleted files)
- ✅ End-to-end change detection
- ✅ Cache stability (no false invalidations)

---

## Summary

The smart caching system provides:

1. **Automatic Optimization**: No manual intervention needed for daily workflows
2. **Transparent Operation**: Clear logging of cache hits/misses and reasons
3. **Safe Invalidation**: Detects all file changes reliably
4. **Manual Control**: Force flags and cache clearing when needed
5. **Fast Performance**: Minimal overhead for manifest checking

**Typical speedup**:
- 2 seconds (cached) vs 10 minutes (uncached) = **300x faster** ⚡

For questions or issues, check the logs for `[CACHE HIT]` and `[CACHE MISS]` messages to understand the system's decisions.
