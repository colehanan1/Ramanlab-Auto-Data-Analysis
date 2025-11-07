# Quick Start Guide: Fly Statistics Caching

This guide shows you how to use the new caching system to speed up your analysis pipeline.

## TL;DR

```bash
# Enable caching (recommended for all workflows)
python geom_features.py --roots /path/to/data --outdir results \
    --cache-dir /home/ramanlab/Documents/cole/cache

# First run: Normal speed (builds cache)
# Second run: 80%+ faster (uses cache)
```

## Basic Usage

### 1. Enable Caching (Recommended)

Add the `--cache-dir` flag to your existing `geom_features.py` commands:

```bash
python geom_features.py \
    --roots /data/flys/opto_EB/ /data/flys/opto_benz_1/ \
    --outdir /results/ \
    --cache-dir /home/ramanlab/Documents/cole/cache
```

**What happens:**
- First run: Processes all flies and saves statistics to cache
- Subsequent runs: Loads statistics from cache for unchanged flies
- Only recomputes flies whose trial data has changed

### 2. Disable Caching (For Validation)

Use `--no-cache` to force full recomputation:

```bash
python geom_features.py \
    --roots /data/flys/opto_EB/ \
    --outdir /results/ \
    --no-cache
```

**When to use:**
- Validating results against cached version
- Debugging suspicious output
- Testing code changes

### 3. Check Cache Performance

Look for this line in the output:

```
INFO Cache statistics: 45 hits, 5 misses (90.0% hit rate) out of 50 flies
```

**Interpretation:**
- **High hit rate (>80%)**: Cache working well, significant speedup
- **Low hit rate (<50%)**: Data files changing frequently
- **0% hit rate**: Cache not working (check `--cache-dir` path)

## Common Workflows

### Workflow 1: Re-running Analysis on Same Data

```bash
# Day 1: Initial analysis
python geom_features.py --roots /data --outdir results --cache-dir /cache
# Time: 15 minutes

# Day 2: Re-run with different output options
python geom_features.py --roots /data --outdir results2 --cache-dir /cache
# Time: 3 minutes (80% faster!)
# Cache hit rate: 100%
```

**Why it's faster**: Fly statistics loaded from cache, skipping expensive concatenation and numpy calculations.

### Workflow 2: Adding New Flies Incrementally

```bash
# Week 1: Analyze first batch (50 flies)
python geom_features.py --roots /data --outdir results --cache-dir /cache
# Time: 15 minutes
# Cache: 0 hits, 50 misses (0% - building cache)

# Week 2: Add 10 new flies, re-analyze
python geom_features.py --roots /data --outdir results --cache-dir /cache
# Time: 4 minutes (instead of 18 minutes!)
# Cache: 50 hits, 10 misses (83% hit rate)
```

**Why it's faster**: Only the 10 new flies are processed from scratch; 50 existing flies use cached statistics.

### Workflow 3: Modifying Subset of Data

```bash
# Initial run
python geom_features.py --roots /data --outdir results --cache-dir /cache

# Correct tracking errors for 5 flies (re-run YOLO, regenerate CSVs)
# ... modify CSVs ...

# Re-analyze
python geom_features.py --roots /data --outdir results --cache-dir /cache
# Cache: 45 hits, 5 misses (90% hit rate)
# Only the 5 modified flies are recomputed
```

**Why it's faster**: Cache automatically detects changed CSV files and invalidates only affected flies.

## Understanding Cache Behavior

### What Gets Cached?

Per-fly aggregate statistics computed from all trials for that fly:
- Geometric bounds: `W_est_fly`, `H_est_fly`, `diag_est_fly`
- Distance stats: `r_min_fly`, `r_max_fly`, `r_p01_fly`, `r_p99_fly`
- Normalization: `r_mean_fly`, `r_std_fly`

**NOT cached** (always recomputed):
- Per-trial enriched data
- Per-frame features
- Output CSV files

### When Does Cache Invalidate?

Cache is automatically invalidated (fly recomputed) when:
- Any trial CSV for that fly is modified (size, timestamp, or content changes)
- New trials added for that fly
- Trials removed or renamed
- Cache files manually deleted

Cache remains valid when:
- Output files change (cache is input-based)
- Code changes (unless FlyStats structure changes - then clear cache manually)
- Different `--outdir` specified

### Cache Storage

**Location**: Specified by `--cache-dir` (e.g., `/home/ramanlab/Documents/cole/cache`)

**Structure**:
```
/cache/
├── fly_stats_manifest.json    # Metadata and hashes
└── fly_stats/
    ├── opto_EB__Fly1.json     # Cached stats for dataset/fly
    ├── opto_EB__Fly2.json
    └── ...
```

**Size**: ~10-50KB for 50 flies (negligible disk space)

## Troubleshooting

### Problem: Cache Not Being Used (0% Hit Rate)

**Symptoms**: Second run is just as slow as first run

**Checks**:
1. Verify cache directory exists and is writable:
   ```bash
   ls -ld /home/ramanlab/Documents/cole/cache
   ```

2. Check if cache was created:
   ```bash
   ls /home/ramanlab/Documents/cole/cache/fly_stats/
   ```

3. Enable debug logging:
   ```bash
   python geom_features.py ... --log-level DEBUG 2>&1 | grep CACHE
   ```

**Common Causes**:
- Wrong `--cache-dir` path (typo)
- CSV files modified between runs (check timestamps)
- `--no-cache` flag accidentally set

### Problem: "Failed to Load Cached Stats" Warnings

**Symptoms**: Warnings in output about corrupted cache

**Solution**: Clear cache and rebuild:
```bash
rm -rf /home/ramanlab/Documents/cole/cache/fly_stats*
python geom_features.py ... --cache-dir /cache
```

**Prevention**: Don't manually edit cache JSON files

### Problem: Stale Cache (Need to Force Refresh)

**Symptoms**: Results seem outdated despite data changes

**Solution**: Run once with `--no-cache` to force full recomputation:
```bash
python geom_features.py ... --no-cache
```

Then resume using cache:
```bash
python geom_features.py ... --cache-dir /cache
```

## Advanced Tips

### Tip 1: Monitor Cache Effectiveness

Add this to your analysis scripts to track cache performance:

```bash
python geom_features.py ... --cache-dir /cache 2>&1 | \
    grep "Cache statistics" | \
    tee -a cache_performance.log
```

### Tip 2: Separate Caches for Different Projects

Use different cache directories for different experiments:

```bash
# Opto experiments
python geom_features.py --roots /data/opto --cache-dir /cache/opto

# Control experiments
python geom_features.py --roots /data/control --cache-dir /cache/control
```

### Tip 3: Inspect Cache Contents

View cached stats for a specific fly:

```bash
cat /cache/fly_stats/opto_EB__Fly1.json | jq .
```

View entire manifest:

```bash
cat /cache/fly_stats_manifest.json | jq .
```

### Tip 4: Automate Cache Management

Add cache clearing to cleanup scripts:

```bash
#!/bin/bash
# clear_old_caches.sh

# Clear caches older than 30 days
find /home/ramanlab/Documents/cole/cache/fly_stats/ \
    -type f -name "*.json" -mtime +30 -delete

echo "Old caches cleared"
```

## Integration with Existing Workflows

### Script Integration

Add caching to existing analysis scripts:

```bash
# OLD
python geom_features.py --roots $DATA_DIR --outdir $RESULTS_DIR

# NEW (with caching)
python geom_features.py --roots $DATA_DIR --outdir $RESULTS_DIR \
    --cache-dir /home/ramanlab/Documents/cole/cache
```

### Pipeline Integration

For `src/fbpipe/pipeline.py` integration (future enhancement):

```yaml
# config.yaml
cache_dir: "/home/ramanlab/Documents/cole/cache"
force:
  pipeline: false  # Set to true to bypass cache
```

## Performance Expectations

| Dataset Size | First Run | Cached Run | Speedup |
|--------------|-----------|------------|---------|
| 10 flies | 3 min | 1 min | 3x |
| 50 flies | 15 min | 3 min | 5x |
| 100 flies | 30 min | 5 min | 6x |

**Note**: Speedup increases with dataset size because the fixed overhead (output writing) becomes smaller relative to cached statistics loading.

## Best Practices

1. **Always use `--cache-dir` in production workflows** unless you have a specific reason not to
2. **Monitor cache hit rates** - low rates may indicate data quality issues
3. **Clear cache after major code changes** that affect FlyStats computation
4. **Use separate cache directories** for different projects/experiments
5. **Back up cache manifest** periodically if re-runs are expensive
6. **Include cache directory in .gitignore** - don't version control cached data

## Getting Help

- **Documentation**: See [OPTIMIZATION_REPORT.md](OPTIMIZATION_REPORT.md) for detailed technical info
- **Code**: Search for "cache" in [geom_features.py](geom_features.py)
- **Issues**: Check git history or contact the maintainer

---

**Last Updated**: 2025-11-07
