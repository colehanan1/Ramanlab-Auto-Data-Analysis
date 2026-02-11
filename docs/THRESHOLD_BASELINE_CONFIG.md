# Threshold Baseline Configuration

## Overview

You can now configure how the baseline (used to calculate response thresholds) is computed. Two modes are available:

1. **Fly-level mean** (default, `use_per_trial_baseline: false`): Uses the average of the before period across **all trials of the same type** (training or testing) for each fly
2. **Per-trial baseline** (`use_per_trial_baseline: true`): Uses only the **before period from each individual trial** to compute its threshold

## Configuration

### In Config File

Add the `use_per_trial_baseline` option to your configuration:

```yaml
analysis:
  combined:
    wide:
      use_per_trial_baseline: false  # Set to true for per-trial baseline mode
```

Or for combined_base:

```yaml
analysis:
  combined_base:
    wide:
      use_per_trial_baseline: false  # Set to true for per-trial baseline mode
```

### Via Command Line

For `envelope_exports.py`:
```bash
python scripts/analysis/envelope_exports.py collect \
  --roots /path/to/data \
  --use-per-trial-baseline \
  --out-csv output.csv
```

For `envelope_combined.py`:
```bash
python scripts/analysis/envelope_combined.py wide \
  --root /path/to/data \
  --output-csv output.csv \
  --use-per-trial-baseline
```

## How It Works

### Fly-Level Mean Mode (Current Default)

```
For each fly:
  1. Calculate mean of before periods across all testing trials: fly_before_mean
  2. For each trial:
     - If fly_before_mean is available: baseline = fly_before_mean
     - Else: baseline = that trial's before period mean
  3. threshold = baseline + k * std_dev (of that trial's before period)
```

### Per-Trial Baseline Mode

```
For each trial:
  1. baseline = that trial's before period mean only
  2. threshold = baseline + k * std_dev (of that trial's before period)
```

## Files Modified

### Direct Changes

1. **`scripts/analysis/envelope_exports.py`**
   - Added `use_per_trial_baseline` parameter to `CollectConfig` dataclass
   - Updated `_compute_trial_metrics()` function to accept and use `use_per_trial_baseline`
   - Added `--use-per-trial-baseline` CLI flag
   - Updated `main()` to read config and pass parameter through

2. **`scripts/analysis/envelope_combined.py`**
   - Added `use_per_trial_baseline` parameter to `build_wide_csv()` function
   - Updated `_compute_trial_metrics()` function to accept and use `use_per_trial_baseline`
   - Added `--use-per-trial-baseline` CLI flag to `wide` subcommand
   - Updated argument parsing and main function

3. **`config/config.yaml`**
   - Added `use_per_trial_baseline: false` to `analysis.combined.wide`
   - Added `use_per_trial_baseline: false` to `analysis.combined_base.wide`

### No Changes Required (Already Per-Trial)

- **`scripts/analysis/envelope_visuals.py`**: Already computes threshold using only the current trial's before period
- **`scripts/analysis/envelope_training.py`**: Already computes threshold using only the current trial's before period

## Backward Compatibility

The default value is `false`, which maintains the current behavior using fly-level means. Existing workflows will continue to work unchanged.

## Quick Start

To switch to per-trial baseline mode:

1. **Edit config file**: Change `use_per_trial_baseline: false` to `use_per_trial_baseline: true`
2. **Or use CLI flag**: Add `--use-per-trial-baseline` when running the script
3. Re-run your envelope analysis pipeline

## Example Workflow

### Current (fly-level) calculation:
- Training trials 1-4 for Fly A have before means: [10.2, 10.5, 9.8, 10.1]
- Fly-level mean: 10.15
- Trial 5 uses baseline = 10.15

### New (per-trial) calculation:
- Trial 5 for Fly A has before mean: 10.7
- Trial 5 uses baseline = 10.7 (from that trial alone)

This can lead to more trial-specific thresholds that account for individual trial variations in baseline activity.
