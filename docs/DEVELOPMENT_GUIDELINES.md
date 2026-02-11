# Development Guidelines

## Convention: Auto-Pipeline Integration

**When implementing changes to analysis scripts, automatically integrate them into the main pipeline** (`scripts/pipeline/run_workflows.py`) **unless explicitly told not to.**

### Why

- **Consistency**: All analysis features are immediately available via the main workflow
- **Convenience**: Don't require manual pipeline updates after feature development
- **Discoverability**: Features configured in config.yaml work automatically
- **Testing**: Pipeline integration happens immediately, catching integration issues early

### How

When adding a new configurable feature:

1. ✅ Implement in the analysis script (e.g., `envelope_exports.py`, `envelope_combined.py`)
2. ✅ Add config options to `config/config.yaml`
3. ✅ Add CLI flags where applicable
4. ✅ **Automatically integrate into `scripts/pipeline/run_workflows.py`**
   - Read config parameter (usually from `wide_cfg.get()`)
   - Pass to function calls
   - Document in comments

### Example Pattern

```python
# In run_workflows.py (in the function that sets up analysis)

# Read new parameter from config
my_new_param = wide_cfg.get("my_new_param", default_value)

# Pass to all relevant function calls
build_wide_csv(
    ...,
    my_new_param=my_new_param,  # ← Auto-integrated!
)
```

### Exceptions

Explicitly request **NO pipeline integration** if:
- Feature is experimental/testing-only
- Feature should only be used via CLI/direct script invocation
- Feature has incompatibilities with the pipeline
- Feature is for a different analysis script outside the pipeline

Example:
```
"Add a debug mode to envelope_visuals.py - don't add to pipeline"
```

## Documentation Pattern

When implementing a new feature, create/update:

1. **Code comments** - Explain what the feature does
2. **Config examples** - Show how to configure it
3. **CLI help text** - Describe the flag via `add_argument(..., help="...")`
4. **Documentation file** - Create/update `.md` file explaining the feature
5. **This file** - Reference the feature if it's a major new addition

## Files That Handle Pipeline Integration

- **`scripts/pipeline/run_workflows.py`** - Main pipeline orchestrator
  - Reads from config.yaml
  - Calls envelope_combined.build_wide_csv()
  - Sets up all analysis workflows

- **`config/config.yaml`** - Configuration source
  - `analysis.combined.wide.*` - Settings for combined analysis
  - `analysis.combined_base.wide.*` - Settings for combined_base
  - Add new parameters here for pipeline accessibility

## Testing Pipeline Integration

After adding a new parameter to the pipeline:

```bash
# Verify it reads from config correctly
grep "my_new_param" scripts/pipeline/run_workflows.py

# Verify it's passed to functions
grep "my_new_param=" scripts/pipeline/run_workflows.py

# Quick test run (dry-run or test data)
python scripts/pipeline/run_workflows.py --help
```

## Recent Features Following This Pattern

1. **Threshold Baseline Configuration** (`use_per_trial_baseline`)
   - Implemented in: envelope_exports.py, envelope_combined.py
   - Integrated into: run_workflows.py (3 call sites)
   - Config: analysis.combined.wide.use_per_trial_baseline
   - Auto-applied to all workflows ✅

## Future Reference

This guideline applies to:
- New analysis features
- Configurable parameters
- Command-line options
- Anything in the analysis scripts that affects threshold/envelope calculation

**Default assumption**: Features go into the pipeline automatically.
**Alternative**: Explicitly request exclusion if needed.
