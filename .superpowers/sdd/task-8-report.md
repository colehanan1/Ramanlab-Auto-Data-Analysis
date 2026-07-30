# Task 8 report: end-to-end verification

**HEAD commit:** `bb60c00723563741f39508192d97231d14bf8d47` (branch `feature/per-fly-score-matrix`)
**New file:** `tests/test_freeze_e2e.py` (uncommitted at time of writing — not yet added per scope guard; ready for the caller/user to commit)

## 1. `tests/test_freeze_e2e.py`

Authored per the brief's exact test code, with one strengthening required by the
task instructions: `test_freeze_then_rerun_is_byte_identical` now also asserts
`first.read_bytes() == second.read_bytes()` (raw byte identity of the two
wide-CSV files) in addition to the brief's `assert_frame_equal` after sorting,
since a sort-then-compare alone would not catch a row-order/formatting
regression.

Signatures were verified against the current tree before writing:
- `freeze.build_fingerprint(..., tracking=...)` — required kwarg, confirmed at
  `src/fbpipe/freeze.py:67-78`. The brief's calls already pass `tracking=_Tracking()`
  correctly; no change needed.
- `build_wide_csv(..., frozen_slices={dataset: (rows_df, own_max_len)})` —
  confirmed at `scripts/analysis/envelope_combined.py:2614` and the fold site
  `:2761-2766`.
- `ev.set_protocol("v2")` autouse fixture — matches `tests/test_freeze_wide.py`.
- Dataset layout (`<root>/<fly>/angle_distance_rms_envelope/<fly>_testing_1_angle_distance_rms_envelope.csv`
  with `envelope_of_rms` column) — copied verbatim from `tests/test_freeze_wide.py::_make_dataset`.

**Result:** `python -m pytest tests/test_freeze_e2e.py -v` → **2 passed**
(`test_freeze_then_rerun_is_byte_identical`, `test_deleting_the_cache_is_safe`).

**Non-vacuousness check (mutation testing, not part of the deliverable file):**
Temporarily deleted the `own_max_len` fold at
`scripts/analysis/envelope_combined.py:2765-2766` (replaced
`max_len = max(max_len, int(_own))` with `pass`), re-ran
`tests/test_freeze_e2e.py tests/test_freeze_wide.py`: **4 failed** including
`test_freeze_then_rerun_is_byte_identical` — confirming the byte-identity
assertion genuinely depends on the fold and is not vacuous. Reverted the
mutation immediately afterward; `git status`/`git diff --stat` confirmed the
source tree returned to a clean, unmodified state (only
`tests/test_freeze_e2e.py` remains untracked). Re-ran the e2e test afterward:
2 passed again.

## 2. Legacy / v2-golden regression

`python -m pytest tests/test_protocol_legacy_regression.py tests/test_protocol_v2_golden.py -v`

**Result: 7 passed** (4 legacy: columns/values/code-maps/matrix match v1;
3 v2-golden: wide-csv/matrix/code-maps match golden). No failures — freeze did
not disturb legacy or v2 golden output. (Only Pandas4Warning/matplotlib
DeprecationWarning noise, pre-existing and unrelated.)

## 3. Full freeze suite together

`python -m pytest tests/test_freeze_config.py tests/test_freeze_cache.py tests/test_freeze_wide.py tests/test_freeze_pipeline.py tests/test_freeze_cli.py tests/test_freeze_figures.py tests/test_freeze_missing_folder.py tests/test_freeze_e2e.py -v`

**Result: 80 passed**, 0 failed.

## 4. Whole suite

`python -m pytest tests/ -q`

**Result: 618 passed, 1 xfailed** in 48.26s, no errors, no failures.

Note on the "3 pre-existing HomeAssistant errors" mentioned in the task
instructions: `HomeAssistant/test_influxdb_enclosure.py` lives outside the
`tests/` directory and is therefore not collected by `pytest tests/ -q` at all
(confirmed via `--collect-only -q | grep -i influxdb` → no matches). No
errors were observed or suppressed; this is consistent with (not a deviation
from) the expected baseline, since the invocation only ever targeted `tests/`.
618 = the baseline (~616) + the 2 new e2e tests, matching expectations.

## 5. Real pipeline `--figures-only` smoke check

**Not run.** Inspected `config/config_new.yaml` first: it points at real
experiment data (`data_root: /home/ramanlab/Documents/cole/Data/flys_New`,
`source_root: /securedstorage/DATAsec/cole/Data-secured-New`) and writes to
real output locations (`output_dir: /home/ramanlab/Documents/cole/Results/New-Opto-Fly-Figures/Weekly-Training-Envelopes`,
plus an SMB mirror path). Per the task's explicit guardrail — do not run if it
needs data that may not be fully accessible, could be slow, or would write to
real Results directories — this was skipped. The automated test suites above
(2 + 80 + 618 passed) are the verification deliverable in lieu of this manual
smoke check; a human with access to the live rig/data should run it
separately if the config-driven `[FROZEN] Not walking root ...` log line needs
manual confirmation against real data.

## Findings

None. No source-file changes were needed or made; only `tests/test_freeze_e2e.py`
was added, exactly as scoped. The mutation check above was reverted and left
no trace on any source file.

## Global constraints check

- Legacy protocol output byte-for-byte identical: confirmed by §2 (7/7 passed).
- Default off: not independently re-verified in this task (covered by
  `test_freeze_config.py`/`test_freeze_pipeline.py` in the full freeze suite,
  §3, all passing); no config in this repo sets `freeze.data: true` by default.
- No regression: confirmed by §3 and §4 (80 and 618 passed respectively).
