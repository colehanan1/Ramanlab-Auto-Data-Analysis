# Baseline Calculation Examples

## Real-World Comparison

Let's walk through a concrete example to illustrate the difference between the two modes.

### Scenario: Fly A with 5 Testing Trials

Imagine you have the following before-period statistics for Fly A's testing trials:

| Trial | Before Mean | Before Std | Trial Type |
|-------|-------------|------------|-----------|
| Test 1 | 12.5 | 2.1 | testing |
| Test 2 | 11.8 | 1.9 | testing |
| Test 3 | 13.2 | 2.3 | testing |
| Test 4 | 12.1 | 2.0 | testing |
| Test 5 | 14.7 | 2.2 | testing |

### Mode 1: Fly-Level Mean (Current Default)

**Configuration:**
```yaml
use_per_trial_baseline: false
```

**Calculation:**

1. Calculate fly-level mean from all testing trials:
   ```
   fly_before_mean = (12.5 + 11.8 + 13.2 + 12.1 + 14.7) / 5 = 12.86
   ```

2. For each trial, use the fly mean as baseline:
   ```
   Test 1: baseline = 12.86, threshold = 12.86 + 4 × 2.1 = 21.26
   Test 2: baseline = 12.86, threshold = 12.86 + 4 × 1.9 = 20.46
   Test 3: baseline = 12.86, threshold = 12.86 + 4 × 2.3 = 21.06
   Test 4: baseline = 12.86, threshold = 12.86 + 4 × 2.0 = 20.86
   Test 5: baseline = 12.86, threshold = 12.86 + 4 × 2.2 = 21.66
   ```

**Observations:**
- All trials use the same baseline (12.86)
- Thresholds vary only due to differences in std_dev
- Provides consistency across a fly's testing phase
- Good for comparing response patterns across trials

### Mode 2: Per-Trial Baseline (New Option)

**Configuration:**
```yaml
use_per_trial_baseline: true
```

**Calculation:**

For each trial, use only that trial's before mean:
```
Test 1: baseline = 12.5,  threshold = 12.5 + 4 × 2.1 = 20.9
Test 2: baseline = 11.8,  threshold = 11.8 + 4 × 1.9 = 19.4
Test 3: baseline = 13.2,  threshold = 13.2 + 4 × 2.3 = 22.4
Test 4: baseline = 12.1,  threshold = 12.1 + 4 × 2.0 = 20.1
Test 5: baseline = 14.7,  threshold = 14.7 + 4 × 2.2 = 23.5
```

**Observations:**
- Each trial has its own baseline (different for each)
- Test 5 has higher baseline (14.7 vs 12.86 in mode 1)
  - This results in higher threshold (23.5 vs 21.66)
  - Requires stronger signal to be "over threshold"
- Test 2 has lower baseline (11.8 vs 12.86)
  - Lower threshold (19.4 vs 20.46)
  - Easier to cross threshold
- Accounts for trial-to-trial baseline variations
- Better for detecting within-trial dynamics

## When to Use Each Mode

### Use Fly-Level Mean (`false`) When:
- ✓ You want each fly's testing or training phase to have a consistent reference baseline
- ✓ You're comparing responsiveness across trials and want baseline to be controlled
- ✓ You're looking at consistent thresholds across a learning period
- ✓ You have some trials with unstable baselines and want to smooth them out

**Example use case:** Tracking how flies learn over training trials 1-4, expecting each trial to use the same reference baseline.

### Use Per-Trial Baseline (`true`) When:
- ✓ Individual trial baselines vary significantly and you want to capture that
- ✓ You're interested in trial-specific responses relative to that trial's own state
- ✓ You want sensitivity to how a fly's baseline changes from trial to trial
- ✓ You're analyzing flies with variable baseline activity between trials

**Example use case:** Detecting responses in testing trials where baseline activity varies (some trials more active, others less), and you want thresholds appropriate to each trial's actual state.

## Configuration Examples

### Example 1: Config File Setup

```yaml
# config/config.yaml

analysis:
  combined:
    combine:
      roots:
        - /home/user/data/training_flies
        - /home/user/data/testing_flies

    wide:
      # Use per-trial baseline for your analysis
      use_per_trial_baseline: true

      measure_cols:
        - envelope_of_rms
      fps_fallback: 40.0
      output_csv: /path/to/output.csv
```

Then run:
```bash
python scripts/analysis/envelope_exports.py collect
```

The script will automatically use `use_per_trial_baseline: true` from your config.

### Example 2: Override via Command Line

```bash
# Override config file with command line flag
python scripts/analysis/envelope_exports.py collect \
  --roots /home/user/data/training \
  --use-per-trial-baseline \
  --out-csv output.csv
```

This will use per-trial baseline regardless of what's in the config file.

### Example 3: Comparing Both Modes

You can generate results with both modes to compare:

```bash
# Generate with fly-level mean
python scripts/analysis/envelope_exports.py collect \
  --roots /data \
  --out-csv results_fly_level.csv
  # (no --use-per-trial-baseline flag)

# Generate with per-trial baseline
python scripts/analysis/envelope_exports.py collect \
  --roots /data \
  --out-csv results_per_trial.csv \
  --use-per-trial-baseline
```

Then compare the AUC and threshold columns between the two CSV files to see the impact.

## Impact on Analysis Outputs

### Affected Metrics:

When you switch between modes, these metrics in the output CSV will change:

- `AUC-Before`: May change if baseline is different
- `AUC-During`: May change (depends on comparison to threshold)
- `AUC-After`: May change
- `AUC-During-Before-Ratio`: Likely to change
- `AUC-After-Before-Ratio`: Likely to change
- `Peak-Value`: Unchanged (raw data)
- `TimeToPeak-During`: Unchanged (raw data)

### Unchanged:

- Raw envelope values (`env_0`, `env_1`, etc.)
- Metadata (dataset, fly, trial_type, fps)
- Geometric features (if using envelope_combined.py)

## Troubleshooting

### "My thresholds are different after changing the mode!"
This is expected. The thresholds are calculated differently in each mode. Re-run your analysis with the new setting.

### "Should I re-process all my old data?"
It depends on your research needs:
- If you're comparing to previous results: keep using the old mode
- If you're starting fresh analysis: consider which mode fits your hypothesis better
- If comparing two cohorts: use the same mode for both

### "Some trials have very unstable baselines"
This is where fly-level mean mode (`false`) can help - it smooths out individual trial variations by using the fly average. If this is an issue, stick with fly-level mean or investigate why those trials are unstable.

## Command Reference

```bash
# Help text for the new option
python scripts/analysis/envelope_exports.py collect --help
# Look for: --use-per-trial-baseline

# Or for envelope_combined.py
python scripts/analysis/envelope_combined.py wide --help
# Look for: --use-per-trial-baseline
```

## Further Reading

- See `THRESHOLD_BASELINE_CONFIG.md` for technical details
- Check the modified files for implementation details:
  - `scripts/analysis/envelope_exports.py` - `_compute_trial_metrics()` function
  - `scripts/analysis/envelope_combined.py` - `_compute_trial_metrics()` function
