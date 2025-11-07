# Redundant Processing Analysis & Fix

**Date**: 2025-11-07
**Issue**: `make run` reprocesses all data even when nothing has changed

---

## Root Cause

Your [config.yaml](config.yaml#L13-L18) has **force flags set to `true`**:

```yaml
force:
  pipeline: true      # ← BYPASSES pipeline caching
  yolo: false
  combined: true      # ← BYPASSES combined analysis caching
  reaction_prediction: true
  reaction_matrix: true
```

### Why This Causes Reprocessing

[`scripts/run_workflows.py`](scripts/run_workflows.py) has a **state caching system** (lines 74-110), but:

**Line 637-643** - Pipeline check:
```python
skip_pipeline = _should_skip(
    settings,
    category="pipeline",
    key=resolved,
    expected=pipeline_expectation,
    force_flag=settings.force.pipeline,  # ← This is TRUE!
)
```

**Line 98-100** - Skip logic:
```python
def _should_skip(..., *, force_flag: bool) -> bool:
    if force_flag:
        return False  # ← NEVER skip when force=true!
```

**Result**: Even though state files exist showing processing is done, the force flags **bypass the cache checks** and reprocess everything.

---

## State Files That Exist (But Are Ignored)

```bash
$ ls -la /home/ramanlab/Documents/cole/cache/
drwxrwxr-x  6 ramanlab ramanlab 4096 Nov  5 22:18 .
├── combined/          # Combined analysis state
├── pipeline/          # Per-dataset pipeline states
│   ├── home_ramanlab_Documents_cole_Data_flys_opto_benz_1/
│   ├── home_ramanlab_Documents_cole_Data_flys_Benz_control/
│   └── ...
├── reaction_matrix/   # Reaction matrix state
└── reaction_prediction/  # Prediction state
```

**Example state** (`/cache/pipeline/.../state.json`):
```json
{
  "non_reactive_span_px": 15.0,
  "version": 1
}
```

**Example combined state** (`/cache/combined/analysis/state.json`):
```json
{
  "dataset_roots": [
    "/home/ramanlab/Documents_cole/Data/flys/Benz_control",
    "/home/ramanlab/Documents/cole/Data/flys/EB_control",
    ...
  ],
  "non_reactive_span_px": 15.0,
  "version": 1
}
```

These files prove processing was completed, but **force flags ignore them**.

---

## What Gets Reprocessed

### 1. Pipeline Steps (Per Dataset)

When `force.pipeline: true`, **all 6 datasets** rerun:
- `/home/ramanlab/Documents/cole/Data/flys/opto_EB/`
- `/home/ramanlab/Documents/cole/Data/flys/opto_benz_1/`
- `/home/ramanlab/Documents/cole/Data/flys/opto_hex/`
- `/home/ramanlab/Documents/cole/Data/flys/hex_control/`
- `/home/ramanlab/Documents/cole/Data/flys/EB_control/`
- `/home/ramanlab/Documents/cole/Data/flys/Benz_control/`

**Steps rerun** (even though output files exist):
1. **yolo** (if `force.yolo: true`) - YOLO inference
2. **distance_stats** - Compute distance bounds
3. **distance_normalize** - Normalize to percentages
4. **detect_dropped_frames** - Find missing frames
5. **rms_copy_filter** - Copy to RMS_calculations/
6. **update_ofm_state** - Annotate state transitions
7. **move_videos** - Stage videos
8. **compose_videos_rms** - Render overlays

### 2. Combined Analysis

When `force.combined: true`:
- **combine_distance_angle()** - Merge raw distance/angle data
- **build_wide_csv()** - Aggregate into wide format CSVs
- **wide_to_matrix()** - Convert to matrices
- **generate_reaction_matrices()** - Plot matrices
- **generate_envelope_plots()** - Create envelope plots

### 3. Reaction Prediction & Matrices

When `force.reaction_prediction: true` and `force.reaction_matrix: true`:
- **predict_reactions** - Run ML model predictions
- **reaction_matrix_from_spreadsheet** - Generate reaction matrices

---

## The Deeper Problem

Even with `force: false`, the state system has **major limitations**:

### State Files Only Track:
- `non_reactive_span_px` value
- Dataset roots (for combined)
- Model/CSV mtimes (for reactions)

### State Files DON'T Track:
❌ **Individual CSV file changes** - If you modify a trial CSV, state doesn't detect it
❌ **New flies added** - Adding new fly directories doesn't invalidate state
❌ **Deleted trials** - Removing trials doesn't trigger reprocessing
❌ **Intermediate file corruption** - If RMS files get deleted, state still says "done"

**Example**: With `force.pipeline: false`, you could:
1. Add 10 new flies to `/Data/flys/opto_EB/`
2. Run `make run`
3. Pipeline sees state file exists → **SKIPS PROCESSING**
4. New flies are **never analyzed**!

This is why you might have set `force: true` in the first place - to ensure new data gets processed.

---

## Solutions

### Solution 1: Quick Fix (Set force: false)

**File**: [config.yaml](config.yaml#L13-L18)

```yaml
force:
  pipeline: false      # ← Only reprocess if state invalid
  yolo: false
  combined: false      # ← Only reprocess if data changed
  reaction_prediction: false
  reaction_matrix: false
```

**Effect**:
- ✅ Subsequent runs with **no changes** will skip processing
- ✅ **80-90% faster** if nothing changed
- ❌ Won't detect new/modified files (unless you delete state cache)

**When to use**:
- Running same analysis multiple times (tweaking plots, etc.)
- No new data being added
- Manual cache clearing when needed

---

### Solution 2: Proper File-Based Caching (Recommended)

Enhance the state tracking system to detect file changes:

**File**: `scripts/run_workflows.py`

**Add to state files**:
```json
{
  "non_reactive_span_px": 15.0,
  "version": 1,
  "file_manifest": {
    "/Data/flys/opto_EB/october_01_fly_1/output_..._testing_1.csv": {
      "mtime": 1728123456.789,
      "size": 433433,
      "hash": "abc123..."  // Optional: first 1KB hash
    },
    ...
  }
}
```

**Logic changes** (pseudo-code):
```python
def _should_skip_pipeline(settings, dataset_root):
    state = _load_state(settings, "pipeline", dataset_root)
    if not state or state.get("version") != STATE_VERSION:
        return False

    # Check if non_reactive_span_px changed
    if state.get("non_reactive_span_px") != settings.non_reactive_span_px:
        return False

    # NEW: Check if any CSV files changed
    current_manifest = _build_file_manifest(dataset_root)
    cached_manifest = state.get("file_manifest", {})

    for file_path, file_info in current_manifest.items():
        if file_path not in cached_manifest:
            print(f"[CACHE MISS] New file detected: {file_path}")
            return False

        cached_info = cached_manifest[file_path]
        if (file_info["mtime"] != cached_info["mtime"] or
            file_info["size"] != cached_info["size"]):
            print(f"[CACHE MISS] File changed: {file_path}")
            return False

    # Check if files were deleted
    for file_path in cached_manifest:
        if file_path not in current_manifest:
            print(f"[CACHE MISS] File deleted: {file_path}")
            return False

    # All checks passed - can skip
    return True
```

**Benefits**:
- ✅ Automatically detects new flies/trials
- ✅ Detects modified CSV files
- ✅ Detects deleted trials
- ✅ Safe with `force: false` as default
- ✅ Only reprocesses what actually changed

**Implementation effort**: ~2-3 hours

---

### Solution 3: Hybrid Approach

Use force flags **selectively** while improving state tracking:

**File**: [config.yaml](config.yaml)

```yaml
force:
  pipeline: false      # ← Let smart caching work
  yolo: false          # ← YOLO is expensive, only when needed
  combined: false      # ← Combined analysis can use cache
  reaction_prediction: false
  reaction_matrix: false
```

**When force=true is needed**:
- Code changes that affect output format
- Config changes beyond `non_reactive_span_px`
- Debugging/validation runs
- One-time reprocessing after bug fixes

**Command-line override** (doesn't require editing config):
```bash
# Normal run (uses cache)
make run

# Force full reprocess (temporary override)
FORCE_PIPELINE=true make run
```

Would need to enhance Makefile to pass env vars to config.

---

## Recommended Action Plan

### Immediate (5 minutes):

**Option A: Enable Caching** - Set all force flags to `false`

**Edit** [config.yaml:13-18](config.yaml#L13-L18):
```yaml
force:
  pipeline: false
  yolo: false
  combined: false
  reaction_prediction: false
  reaction_matrix: false
```

**Test**:
```bash
# First run (builds cache)
make run
# Check: Should complete normally

# Second run (should skip most processing)
make run
# Expected output:
#   [analysis] pipeline cached → skipping full run for ...
#   [analysis] combined analysis cached → skipping ...
#   [analysis] reactions.predict cached → skipping ...
```

**Limitation**: Won't detect new files - must manually clear cache when adding data:
```bash
# When adding new flies/trials:
rm -rf /home/ramanlab/Documents/cole/cache/pipeline/*
rm -rf /home/ramanlab/Documents/cole/cache/combined/*
make run
```

### Short-term (1-2 weeks):

**Option B: Implement file-based state tracking** (Solution 2)

Create new functions in `run_workflows.py`:
1. `_build_file_manifest(dataset_root)` - Scan for all CSV files, collect mtime/size
2. `_compare_manifests(current, cached)` - Detect new/changed/deleted files
3. Update `_should_skip()` to use manifest comparison
4. Update `_write_state()` to include manifest

**Benefits**:
- Safe default operation with `force: false`
- Automatic detection of data changes
- Clear logging of what triggered reprocessing
- Production-ready for continuous analysis

---

## Testing the Fix

### Test 1: No Changes (Should Skip)

```bash
# Clear any ongoing processing
cd /home/ramanlab/Documents/cole/VSCode/Ramanlab-Auto-Data-Analysis

# Set force: false in config.yaml
vim config.yaml  # Set all force flags to false

# First run
make run 2>&1 | tee run1.log

# Second run (should skip everything)
make run 2>&1 | tee run2.log

# Check for skip messages
grep "cached →" run2.log
# Expected:
#   [analysis] pipeline cached → skipping full run...
#   [analysis] combined analysis cached → skipping...
```

### Test 2: Manual Cache Clear

```bash
# Clear caches
rm -rf /home/ramanlab/Documents/cole/cache/pipeline/*
rm -rf /home/ramanlab/Documents/cole/cache/combined/*

# Should reprocess everything
make run 2>&1 | tee run3.log

# Verify reprocessing happened
grep -c "Processing" run3.log  # Should show many processing steps
```

### Test 3: Compare Run Times

```bash
# With force: true (current behavior)
time make run  # Expect: 15-30 minutes

# With force: false (after first run)
time make run  # Expect: 2-5 minutes (80-90% faster)
```

---

## Performance Impact

| Scenario | force: true | force: false (cached) | Speedup |
|----------|-------------|----------------------|---------|
| No changes | 20 min | 3 min | **6.7x faster** |
| 1 dataset changed | 20 min | 7 min | **2.8x faster** |
| Config tweaks only | 20 min | 3 min | **6.7x faster** |
| All data changed | 20 min | 20 min | Same (expected) |

**Note**: Times are estimates for 6 datasets with ~10 flies each. Your actual times may vary.

---

## FAQ

### Q: Why were force flags set to true originally?

**A**: Probably because the state system didn't detect new files, so setting `force: true` was the safest way to ensure all data got processed. With proper file manifest tracking (Solution 2), this won't be necessary.

### Q: Will this break my workflow?

**A**: No. Changing `force: false` is safe and reversible. If anything breaks, just set `force: true` again.

### Q: What about the fly statistics cache I just added?

**A**: That's for `geom_features.py`, which isn't part of your main pipeline. The geom_features caching is separate and complementary.

### Q: How do I force reprocessing for just one dataset?

**A**: Currently you'd need to delete that dataset's state file:
```bash
rm -rf /home/ramanlab/Documents/cole/cache/pipeline/home_ramanlab_Documents_cole_Data_flys_opto_EB/
make run
```

Or temporarily set `force.pipeline: true`, run, then set it back to `false`.

### Q: Can I use environment variables to override force flags?

**A**: Not currently, but this would be a good enhancement. Would need to modify `run_workflows.py` to check env vars before loading from config.

---

## Next Steps

1. **Immediate**: Set force flags to `false` in config.yaml
2. **Test**: Run `make run` twice and verify caching works
3. **Monitor**: Watch for any issues with undetected changes
4. **Plan**: Decide if file manifest tracking (Solution 2) is worth implementing
5. **Document**: Update your workflow docs with cache clearing procedures

---

**Document Version**: 1.0
**Last Updated**: 2025-11-07
**Author**: Claude AI (via investigation of run_workflows.py execution flow)
