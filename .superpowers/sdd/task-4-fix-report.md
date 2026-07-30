# Task 4 fix report

Branch `feature/per-fly-score-matrix`, commit **ca326c3b819186d1a8fc11ff00e91c5381608234**.

Scope respected: only `scripts/pipeline/run_workflows.py` and
`tests/test_freeze_pipeline.py` touched. `git diff --stat` at commit time:

```
 scripts/pipeline/run_workflows.py |   6 ++
 tests/test_freeze_pipeline.py     | 127 +++++++++++++++++++++++++++++++++++++-
 2 files changed, 131 insertions(+), 2 deletions(-)
```

## FIX 1 — thread `--config` through to `build_wide_csv`

**Root cause (verified).** `build_wide_csv`
(`scripts/analysis/envelope_combined.py:2633`) does
`settings = load_settings(config_path or DEFAULT_CONFIG_PATH)` and derives
`tracking_missing_frames` / `tracking_pct_missing` / `tracking_flagged` from
`settings.tracking`. None of `_run_combined`'s three `build_wide_csv` call
sites passed `config_path`, so a run started with `--config
config_new.yaml` silently computed those three columns from
`config/config.yaml`'s thresholds (`max_missing_frames_per_trial: 5000`)
instead of the caller's config (`1800`).

**Change — `scripts/pipeline/run_workflows.py`:**
- `:1087` — `_run_combined` gains a keyword-only `config_path: str | Path |
  None = None` parameter (default `None`, so any other caller/test is
  unaffected).
- `:1298` — first `build_wide_csv(...)` call (plain `combined.wide` block)
  now passes `config_path=config_path`.
- `:1449` — second `build_wide_csv(...)` call (`_process_base_block`, used
  for `combined_base`/`distance_base`) now passes `config_path=config_path`.
- `:1624` — third `build_wide_csv(...)` call (`pair_groups`) now passes
  `config_path=config_path`.
- `:2430` and `:2448` — the two `_run_combined(...)` call sites inside
  `main()` now pass `config_path=config_path` (the `Path` already resolved
  at `:2163` via `resolve_config_path(args.config)`).

Blast radius confirmed unchanged from the brief: inside `build_wide_csv`,
`config_path` is also read by `_resolve_distance_limits` (no-op here — every
call site already passes an explicit `distance_limits=limits`) and by the
`exclude_roots` load (no-op — neither config defines `analysis.combined.wide
.exclude_roots`). The only observable behavior change is the tracking
columns now honoring `--config`. Under the default `config.yaml`,
`config_path` resolves to `DEFAULT_CONFIG_PATH`, so that path is unchanged.

**Test — `tests/test_freeze_pipeline.py::test_run_combined_threads_config_path_to_build_wide_csv`.**
Drives the real `_run_combined` (not a narrower seam) with a minimal `cfg`
containing a `wide` block, a `combined_base` block, and a `pair_groups`
entry — enough to reach all three `build_wide_csv` call sites in one call —
and a stub `_Settings`. `rw.build_wide_csv` is monkeypatched with a spy that
records `kwargs.get("config_path")`; `rw.wide_to_matrix` is stubbed to a
no-op because the `pair_groups` branch calls it unconditionally on the
(nonexistent, since `build_wide_csv` is spied) wide CSV. Asserts all 3
captured call sites received the sentinel `config_path`.

**Failing-before / passing-after evidence.**

Before the code change (test written first, run against unmodified
`run_workflows.py`):
```
FAILED tests/test_freeze_pipeline.py::test_run_combined_threads_config_path_to_build_wide_csv
E       TypeError: _run_combined() got an unexpected keyword argument 'config_path'
1 failed, 9 passed, 14 warnings in 2.71s
```
Note: the failure mode is a `TypeError` (the kwarg doesn't exist yet on
`_run_combined`), not an assertion of `None != sentinel` — `config_path` is
a brand-new parameter, so pre-fix the call itself cannot be made. This is a
stronger failure signal than an assertion mismatch and still satisfies
"write the test first, watch it fail."

After the code change:
```
tests/test_freeze_pipeline.py::test_run_combined_threads_config_path_to_build_wide_csv PASSED [100%]
10 passed, 14 warnings in 1.80s
```

## FIX 2 — real round-trip test through `_freeze_fingerprint`

Added two tests that exercise the exact pattern `_run_combined` relies on:
`_write_freeze_cache(..., fingerprint_for=lambda ds: _freeze_fingerprint(settings, ds, **kw))`
followed by `_resolve_frozen_slices(settings, roots, block, **kw)` with the
same `kw`.

- `test_freeze_round_trip_write_then_resolve` — writes dataset `"DS"`'s
  slice via `_write_freeze_cache` using `_freeze_fingerprint(s, ds, **kw)`,
  then calls `_resolve_frozen_slices(s, [...], "combined_base", **kw)` with
  the identical `kw`. Asserts `"DS" in got` and `got["DS"].own_max_len == 4`.
  **Observed: PASS** — `"DS"` resolves, `own_max_len == 4`, proving the
  save-path and load-path fingerprints agree today.
- `test_freeze_round_trip_misses_when_resolve_kw_diverges` — same write,
  but the resolve-side `kw["measure_cols"]` is changed to
  `["distance_percentage"]` (mimicking a `base_measure_cols` /
  `wide_measure_cols` mixup) before calling `_resolve_frozen_slices`.
  Asserts `got == {}`. **Observed: PASS** — the mutated kw is *not* found,
  proving the round-trip test above is not vacuous and genuinely depends on
  fingerprint agreement between the write and read paths.

Both tests passed on first run (no pre-existing bug), which is expected —
this is coverage-gap closure per the Task 4 review finding, not a bugfix; no
production code changed for FIX 2.

## FIX 3 — unused imports

Removed `import numpy as np` and `import pytest` from
`tests/test_freeze_pipeline.py` (grep confirmed no `np.` / `pytest.` usage
anywhere in the file, including the new tests — `monkeypatch` is
auto-injected by pytest and needs no import).

## Test contract — actual output

`python -m pytest tests/test_freeze_pipeline.py -v`:
```
tests/test_freeze_pipeline.py::test_frozen_dataset_with_cache_resolves PASSED [ 10%]
tests/test_freeze_pipeline.py::test_unfrozen_dataset_is_not_resolved_even_with_a_cache PASSED [ 20%]
tests/test_freeze_pipeline.py::test_frozen_but_cache_miss_auto_rebuilds PASSED [ 30%]
tests/test_freeze_pipeline.py::test_frozen_but_config_drift_auto_rebuilds PASSED [ 40%]
tests/test_freeze_pipeline.py::test_thaw_ignores_freeze PASSED           [ 50%]
tests/test_freeze_pipeline.py::test_write_freeze_cache_stores_each_dataset_slice PASSED [ 60%]
tests/test_freeze_pipeline.py::test_write_freeze_cache_includes_extra_trial_exports PASSED [ 70%]
tests/test_freeze_pipeline.py::test_freeze_round_trip_write_then_resolve PASSED [ 80%]
tests/test_freeze_pipeline.py::test_freeze_round_trip_misses_when_resolve_kw_diverges PASSED [ 90%]
tests/test_freeze_pipeline.py::test_run_combined_threads_config_path_to_build_wide_csv PASSED [100%]
10 passed, 14 warnings in 1.80s
```

`python -m pytest tests/test_protocol_legacy_regression.py tests/test_protocol_v2_golden.py -q`:
```
7 passed, 28 warnings in 10.30s
```
(Both legacy regression tests and v2 golden tests call `build_wide_csv`
directly with no `config_path`, unaffected by the fix — confirmed green.)

`python -m pytest tests/ -q`:
```
581 passed, 1 xfailed, 355 warnings in 47.19s
```
Baseline was 578 passed, 1 xfailed; 581 = 578 + 3 new tests added in this
task (`test_run_combined_threads_config_path_to_build_wide_csv`,
`test_freeze_round_trip_write_then_resolve`,
`test_freeze_round_trip_misses_when_resolve_kw_diverges`). No regressions.
`HomeAssistant/test_influxdb_enclosure.py` is outside `tests/` and not
collected by this invocation, consistent with the pre-existing-errors note
in the brief.

## Commit

`ca326c3b819186d1a8fc11ff00e91c5381608234` —
"fix(run_workflows): thread --config through to build_wide_csv's tracking columns"
