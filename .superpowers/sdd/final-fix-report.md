# Final fix report — freeze fingerprint `trial_type_filter` gap

Branch `feature/per-fly-score-matrix`, base HEAD `548e4a9`, fix commit
`1c2fe2fa704751da02ccd1503a27e1bc577db838`.

## The bug

`freeze.build_fingerprint` recorded everything that determines a frozen
dataset's cached rows EXCEPT `trial_type_filter`, which `build_wide_csv`
(`scripts/analysis/envelope_combined.py:2661-2673`, `:2703`) uses to gate
which trials become rows at all. All three freeze call sites in
`run_workflows.py` passed `trial_type_filter` to `build_wide_csv` but never
recorded it in the fingerprint — a config change narrowing the filter would
fingerprint-match a stale cache, splice it in pre-filtered, then re-cache the
narrowed result, permanently evicting the dropped rows.

## What changed (file:line references are to the post-fix files)

### `src/fbpipe/freeze.py`
- `build_fingerprint` (`:67-78`): added `trial_type_filter` as a **required**
  keyword-only parameter (no default — same as `tracking`), positioned right
  after `use_per_trial_baseline`.
- `:90-115`: normalization block, matching `build_wide_csv`'s own
  `trial_type_allow` logic exactly (`envelope_combined.py:2661-2673`):
  - `None` → `None`
  - `str`/`bytes` → `sorted({stripped-lowercased value})` (a 1-element list)
  - other iterable → `sorted({stripped-lowercased non-empty values})`, or
    `None` if that set ends up empty
- `:129-130`: `"trial_type_filter": trial_type_norm` added to the returned
  dict, next to `use_per_trial_baseline`, with a one-line pointer comment to
  the block above.

### `scripts/pipeline/run_workflows.py`
- `_freeze_fingerprint` (`:417-450`): added required `trial_type_filter`
  param, threaded to `_freeze.build_fingerprint(trial_type_filter=...)`.
- `_resolve_frozen_slices` (`:452-...`): added required `trial_type_filter`
  param, threaded to its internal `_freeze_fingerprint(...)` call.
- Three `_fp_kw` dict sites, each adding `trial_type_filter=trial_type_filter`
  using the SAME local variable already passed to that site's `build_wide_csv`
  call, with a comment pointing at the line of that `build_wide_csv` call:
  - `combined.wide` site: `_fp_kw` at `:1329-1333`, comment points to `:1343`
    (the `build_wide_csv(..., trial_type_filter=trial_type_filter, ...)` call).
  - `combined_base`/`distance_base` site: `_fp_kw` at `:1483-1487`, comment
    points to `:1497`.
  - `pair_groups` site: `_fp_kw` at `:1661-1665`, comment points to `:1675`.
- `_write_freeze_cache` (`:513-...`): wrapped the `pd.concat(frames, ...)` /
  `"dataset" not in allrows.columns` / `allrows.groupby(...)` block — which
  built `allrows` OUTSIDE the per-dataset `try/except` — in its own
  `try/except Exception`, printing `[FREEZE] Could not prepare rows to cache
  for {wide_block}: {exc}` and returning without caching. The per-dataset
  `try/except` around `_freeze.save_slice` is unchanged. Docstring updated to
  state the guard now covers the whole function.

### Tests
- `tests/test_freeze_cache.py`: `_fp(**kw)` base dict gets
  `trial_type_filter=None`. Added three tests:
  - `test_trial_type_filter_drift_returns_none` — save with `None`, load with
    `"testing"` → `None` (invalidated).
  - `test_trial_type_filter_unchanged_still_loads` — save and load both with
    `"testing"` → not `None` (positive control, proves drift detection isn't
    blanket invalidation).
  - `test_trial_type_filter_order_insensitive` — save with
    `["testing","training"]`, load with `["training","testing"]` → not `None`.
- `tests/test_freeze_pipeline.py`: `_fp_kw()` gets `trial_type_filter=None`
  (feeds both direct `build_fingerprint` calls and, via `**_fp_kw`,
  `_resolve_frozen_slices` / `_freeze_fingerprint`, both of which now require
  it).
- `tests/test_freeze_e2e.py`: the `fp_kw` dict and the second inline
  `build_fingerprint(...)` call both get `trial_type_filter=None`.

## Mutation check (mandatory)

Removed `"trial_type_filter": trial_type_norm` from `build_fingerprint`'s
returned dict (commented out), then ran the three new tests:

```
$ python -m pytest tests/test_freeze_cache.py -k "trial_type_filter" -v
...
tests/test_freeze_cache.py::test_trial_type_filter_drift_returns_none FAILED
tests/test_freeze_cache.py::test_trial_type_filter_unchanged_still_loads PASSED
tests/test_freeze_cache.py::test_trial_type_filter_order_insensitive PASSED
1 failed, 2 passed, 20 deselected, 14 warnings in 0.08s
```

Failure detail:

```
    def test_trial_type_filter_drift_returns_none(tmp_path):
        freeze.save_slice(tmp_path, "wide", "DS", _rows([5], 5), _fp(trial_type_filter=None))
        drifted = _fp(trial_type_filter="testing")
>       assert freeze.load_slice(tmp_path, "wide", "DS", drifted) is None
E       AssertionError: assert FrozenSlice(rows=  dataset trial_type  trace_len ...
E       [1 rows x 8 columns], own_max_len=5) is None
```

i.e. with the field removed, a cache saved under `trial_type_filter=None`
was incorrectly loaded under `trial_type_filter="testing"` — exactly the
stale-splice/permanent-eviction scenario the fix prevents. (The other two
tests still passed because their own assertions don't depend on this field
being tracked — they assert a *positive* match, which holds trivially when
the field is absent from both fingerprints.)

Restored the field, re-ran the same selection:

```
$ python -m pytest tests/test_freeze_cache.py -k "trial_type_filter" -v
tests/test_freeze_cache.py::test_trial_type_filter_drift_returns_none PASSED
tests/test_freeze_cache.py::test_trial_type_filter_unchanged_still_loads PASSED
tests/test_freeze_cache.py::test_trial_type_filter_order_insensitive PASSED
3 passed, 20 deselected, 14 warnings in ...
```

Mutation check: **fail → pass**, as required.

## Order-insensitivity assertion

`test_trial_type_filter_order_insensitive` (`tests/test_freeze_cache.py`):
saves a slice fingerprinted with `trial_type_filter=["testing", "training"]`,
then loads it with a fingerprint built from `["training", "testing"]` (same
values, reversed order) — asserts the load succeeds (`is not None`), proving
`sorted()` in the normalization makes the two fingerprint-equal.

## Full test output

### Target files (`test_freeze_cache.py`, `test_freeze_pipeline.py`, `test_freeze_e2e.py`)

```
$ python -m pytest tests/test_freeze_cache.py tests/test_freeze_pipeline.py tests/test_freeze_e2e.py -v
...
tests/test_freeze_cache.py::test_own_max_len_is_max_trace_len PASSED
tests/test_freeze_cache.py::test_own_max_len_clamped_to_present_columns PASSED
tests/test_freeze_cache.py::test_save_load_round_trip PASSED
tests/test_freeze_cache.py::test_load_miss_returns_none PASSED
tests/test_freeze_cache.py::test_fingerprint_drift_returns_none PASSED
tests/test_freeze_cache.py::test_protocol_drift_returns_none PASSED
tests/test_freeze_cache.py::test_odor_remap_drift_returns_none PASSED
tests/test_freeze_cache.py::test_trial_type_filter_drift_returns_none PASSED
tests/test_freeze_cache.py::test_trial_type_filter_unchanged_still_loads PASSED
tests/test_freeze_cache.py::test_trial_type_filter_order_insensitive PASSED
tests/test_freeze_cache.py::test_tracking_max_missing_frames_per_trial_drift_returns_none PASSED
tests/test_freeze_cache.py::test_tracking_max_missing_frames_pct_per_trial_drift_returns_none PASSED
tests/test_freeze_cache.py::test_tracking_apply_missing_frame_check_toggle_returns_none PASSED
tests/test_freeze_cache.py::test_tracking_unchanged_still_loads PASSED
tests/test_freeze_cache.py::test_figure_output_subdir_does_not_invalidate PASSED
tests/test_freeze_cache.py::test_blocks_do_not_collide PASSED
tests/test_freeze_cache.py::test_corrupt_cache_returns_none_not_raise PASSED
tests/test_freeze_cache.py::test_corrupt_parquet_returns_none_not_raise PASSED
tests/test_freeze_cache.py::test_freeze_flags_reads_override PASSED
tests/test_freeze_cache.py::test_freeze_flags_unknown_dataset_is_unfrozen PASSED
tests/test_freeze_cache.py::test_thaw_overrides_named_dataset PASSED
tests/test_freeze_cache.py::test_thaw_does_not_affect_other_datasets PASSED
tests/test_freeze_cache.py::test_thaw_all_overrides_everything PASSED
tests/test_freeze_pipeline.py::test_frozen_dataset_with_cache_resolves PASSED
tests/test_freeze_pipeline.py::test_unfrozen_dataset_is_not_resolved_even_with_a_cache PASSED
tests/test_freeze_pipeline.py::test_frozen_but_cache_miss_auto_rebuilds PASSED
tests/test_freeze_pipeline.py::test_frozen_but_config_drift_auto_rebuilds PASSED
tests/test_freeze_pipeline.py::test_thaw_ignores_freeze PASSED
tests/test_freeze_pipeline.py::test_write_freeze_cache_stores_each_dataset_slice PASSED
tests/test_freeze_pipeline.py::test_write_freeze_cache_includes_extra_trial_exports PASSED
tests/test_freeze_pipeline.py::test_freeze_round_trip_write_then_resolve PASSED
tests/test_freeze_pipeline.py::test_freeze_round_trip_misses_when_resolve_kw_diverges PASSED
tests/test_freeze_pipeline.py::test_run_combined_threads_config_path_to_build_wide_csv PASSED
tests/test_freeze_e2e.py::test_freeze_then_rerun_is_byte_identical PASSED
tests/test_freeze_e2e.py::test_deleting_the_cache_is_safe PASSED
======================= 35 passed, 14 warnings in 1.86s ========================
```

(Baseline before this fix: 32 passed. +3 = the new `trial_type_filter` tests.)

### Full suite

```
$ python -m pytest tests/ -q
...
621 passed, 1 xfailed, 367 warnings in 46.31s
```

No failures, no errors. (Task baseline: 618 passed, 1 xfailed at HEAD; +3 new
tests = 621. The 3 pre-existing `HomeAssistant/test_influxdb_enclosure.py`
errors noted in the task brief live outside `tests/` and are not collected by
`pytest tests/ -q`.)

## Scope

Only the five files named in scope were modified:
`src/fbpipe/freeze.py`, `scripts/pipeline/run_workflows.py`,
`tests/test_freeze_cache.py`, `tests/test_freeze_pipeline.py`,
`tests/test_freeze_e2e.py`. `git status --porcelain` after the commit shows a
clean tree with no other changes. `_freeze_fingerprint` and
`_resolve_frozen_slices` in `run_workflows.py` both needed a new required
`trial_type_filter` parameter (not just the three `_fp_kw` dicts) because the
dicts are expanded into both functions via `**_fp_kw` — this was a necessary
consequence of making the fingerprint field required, entirely within the
allowed `run_workflows.py` file, not a scope expansion.

## Commit

`1c2fe2fa704751da02ccd1503a27e1bc577db838` —
`fix(freeze): fingerprint trial_type_filter; guard _write_freeze_cache concat`
