# Pipeline performance optimization (CSV → Parquet, content-hash cache, parallelism)

This branch (`perf/optimization`) makes the pipeline faster and stops it repeating
work, **without changing any numerical output**. Revert point: branch
`backup/pre-optimization-2026-06-24` + tag `pre-optimization-2026-06-24` (`main` is
untouched).

## What changed

1. **Parquet I/O (`src/fbpipe/utils/tables.py`).** All pipeline-produced tables are
   now Parquet instead of CSV: ~**4× faster reads, ~10× faster writes, ~4× smaller**,
   float-exact. `read_table` / `write_table` / `read_schema_columns` / `resolve_existing`
   are the single chokepoint; `iter_fly_distance_csvs` discovers `.parquet` or `.csv`
   (parquet preferred). `write_table` removes a superseded same-stem `.csv` (single
   source of truth).
   - **External files stay CSV** (read transparently, never converted): recording-rig
     `output_*.csv` (ActiveOFM), YOLO timestamp inputs, the temp CSV handed to the
     `flybehavior-response` scorer, and **`model_predictions.csv`** (human-/tool-facing).

2. **Content-hash cross-run cache (`scripts/pipeline/run_workflows.py`).** The manifest
   tracks Parquet + CSV and compares by **content hash** (blake2b) + size, so a file
   that is merely *touched* (re-sync, chmod) no longer triggers reprocessing. Unchanged
   data is never reprocessed across runs. `STATE_VERSION` is 2 (old caches auto-invalidate).

3. **Opt-in CPU parallelism (`src/fbpipe/utils/parallel.py`, `cfg.parallel`).** The 6 CPU
   per-fly stages can fan out across cores. **Disabled by default** (identical to serial).

4. **Analysis (`src/fbpipe/analysis/traces.py`).** `read_wide_table` reads the big
   `all_envelope_rows_wide*.csv` tables Parquet-first; `baseline_correct` is now shared.

## How to use it

### Convert existing data to Parquet (one-time, per dataset / wide-CSV dir)
```bash
# Keep the CSVs (safe, dual format) — downstream auto-prefers the .parquet:
python scripts/migrate_csv_to_parquet.py --root /path/to/dataset
# …or reclaim disk by removing the CSVs once you trust the migration:
python scripts/migrate_csv_to_parquet.py --root /path/to/dataset --delete-csv --yes
```
Fresh YOLO runs already write Parquet directly — no migration needed for new data.
Convert the big analysis tables too, e.g.:
```bash
python scripts/migrate_csv_to_parquet.py --root /home/ramanlab/Documents/cole/Data/CSVs-ALL-Opto-Flys
```

### Enable parallelism (large batches)
In `config/config.yaml`:
```yaml
parallel:
  enabled: true
  n_jobs: 0     # 0 = auto (cores − 2)
```
or per-run: `PARALLEL_ENABLED=true PARALLEL_N_JOBS=8 python -m fbpipe.pipeline ...`.
Note: parallelism helps on **large** datasets; on small per-fly files the process
startup overhead cancels the gain (hence opt-in). The big wins are Parquet + the cache.

### Get a CSV back for Excel / external tools
```bash
python scripts/export_csv.py --path /path/to/file_or_dir
```

### Benchmark
```bash
python scripts/bench_pipeline.py
```

## Verification & safety
- The test suite was **already red** before this work (16 pre-existing failures from
  in-progress analysis refactors + 1 needing the secured-storage mount). That set was
  frozen as a baseline; **every phase added zero new failures**. ~140 new tests were added.
- Numerical identity rests on: Parquet round-trips float64 exactly (tested), each step
  only swapped the serialization (in-memory frame untouched), parallel output is asserted
  identical to serial, and an end-to-end stats→normalize + 3-fly-sanitization integration
  test.

## Known follow-ups (optional, low priority)
- Dead helpers `utils/gpu_accelerated.py` / `utils/gpu_batch_optimizer.py` still call
  `pd.read_csv` but are unused by any step/test — delete or route through `read_table`.
- `scripts/run_workflows.py` remains a thin back-compat shim for
  `scripts/pipeline/run_workflows.py` (canonical).
