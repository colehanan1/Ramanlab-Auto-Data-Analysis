# Threshold Baseline Configuration - Implementation Complete ✅

## What Was Implemented

A fully configurable system to control how baseline values are calculated for threshold computation in envelope analysis. You can now choose between two modes:

1. **Fly-Level Mean Mode** (`use_per_trial_baseline: false`) - DEFAULT
   - Uses average of all before periods for a fly (same trial type)
   - Current behavior (unchanged)
   - Better for consistent reference baselines

2. **Per-Trial Baseline Mode** (`use_per_trial_baseline: true`) - NEW
   - Uses only that trial's before period for its threshold
   - Each trial gets its own baseline
   - Better for capturing trial-to-trial variations

## Files Modified

### Code Changes

1. **`scripts/analysis/envelope_exports.py`**
   - Added `use_per_trial_baseline` parameter to `CollectConfig` dataclass
   - Updated `_compute_trial_metrics()` function with baseline selection logic
   - Added `--use-per-trial-baseline` CLI flag
   - Updated `main()` to read from config and pass parameter

2. **`scripts/analysis/envelope_combined.py`**
   - Added `use_per_trial_baseline` parameter to `build_wide_csv()` function
   - Updated `_compute_trial_metrics()` function with baseline selection logic
   - Added `--use-per-trial-baseline` CLI flag to `wide` subcommand
   - Updated CLI argument parsing and main function

3. **`scripts/pipeline/run_workflows.py`** ← NOW INTEGRATED!
   - Line 703: Reads `use_per_trial_baseline` from config
   - Line 743: Passes to main combined.wide workflow
   - Line 843: Passes to combined_base.wide workflow
   - Line 971: Passes to pair_groups.wide workflow

4. **`config/config.yaml`**
   - Added `use_per_trial_baseline: false` to `analysis.combined.wide`
   - Added `use_per_trial_baseline: false` to `analysis.combined_base.wide`

### Documentation Created

1. **`THRESHOLD_BASELINE_CONFIG.md`** - Technical reference
   - Configuration options
   - How baseline calculation works
   - Files modified
   - Backward compatibility notes

2. **`BASELINE_EXAMPLES.md`** - Practical examples
   - Real-world comparison with numbers
   - When to use each mode
   - Example configurations
   - Command reference

3. **`IMPLEMENTATION_COMPLETE.md`** - This file
   - Summary of what was implemented
   - Integration points
   - How to use the feature

## Integration with Main Pipeline

✅ **YES - FULLY INTEGRATED**

The feature is now part of the main pipeline (`scripts/pipeline/run_workflows.py`):

```
run_workflows.py
    ↓
    Reads config.yaml (use_per_trial_baseline)
    ↓
    Passes to build_wide_csv() calls:
        • combined.wide
        • combined_base.wide
        • pair_groups.wide
    ↓
    Uses in _compute_trial_metrics()
    ↓
    Threshold calculation uses your chosen baseline mode
```

## How to Use

### Method 1: Config File (Recommended)

Edit `config/config.yaml`:

```yaml
analysis:
  combined:
    wide:
      use_per_trial_baseline: false  # Change to true for per-trial mode
```

Then run normally:
```bash
python scripts/pipeline/run_workflows.py
```

### Method 2: Command Line (Direct Script Usage)

```bash
# Using envelope_exports.py
python scripts/analysis/envelope_exports.py collect \
  --use-per-trial-baseline \
  --roots /path/to/data \
  --out-csv output.csv

# Using envelope_combined.py
python scripts/analysis/envelope_combined.py wide \
  --use-per-trial-baseline \
  --root /path/to/data \
  --output-csv output.csv
```

## Backward Compatibility

✅ **Fully backward compatible**

- Default: `use_per_trial_baseline: false` (current behavior unchanged)
- All existing configurations work as before
- No breaking changes
- Can switch between modes at any time

## Testing

The baseline selection logic has been verified to work correctly:

```
✓ Per-trial mode: Uses trial's own before mean
✓ Fly-level mode: Uses fly average across same trial type
✓ Fallback: When fly mean unavailable, falls back to trial's before mean
```

## What's Not Changed

- `envelope_visuals.py` - Already uses per-trial baseline (no changes needed)
- `envelope_training.py` - Already uses per-trial baseline (no changes needed)
- Raw envelope data and metadata - Unaffected
- All other analysis parameters - Unchanged

## Configuration Locations

The `use_per_trial_baseline` option appears in two places in `config.yaml`:

1. **For main combined analysis:**
   ```yaml
   analysis:
     combined:
       wide:
         use_per_trial_baseline: false
   ```

2. **For combined_base analysis:**
   ```yaml
   analysis:
     combined_base:
       wide:
         use_per_trial_baseline: false
   ```

Both default to `false` to maintain current behavior.

## Summary

| Aspect | Status |
|--------|--------|
| Feature Implementation | ✅ Complete |
| Pipeline Integration | ✅ Complete |
| Config File Support | ✅ Complete |
| CLI Flag Support | ✅ Complete |
| Documentation | ✅ Complete |
| Backward Compatibility | ✅ Maintained |
| Testing | ✅ Verified |

## Next Steps

1. ✅ Implementation is complete and ready to use
2. Edit your `config.yaml` to set `use_per_trial_baseline` to your desired value
3. Run your analysis pipeline normally
4. The feature will be automatically applied to all relevant workflows

## Questions?

Refer to:
- `THRESHOLD_BASELINE_CONFIG.md` - Technical details
- `BASELINE_EXAMPLES.md` - Practical examples with numbers
- Source files mentioned above - Implementation details
