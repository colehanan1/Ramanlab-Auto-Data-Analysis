# Task 7 Report: Frozen dataset with a missing folder is a hard error

Note: this file previously held a report for an earlier, differently-scoped
"Task 7" (control-left/training-right guard against a column swap, commit
`47d6fe3`). That work is already committed and unrelated to this brief
(`.superpowers/sdd/task-7-brief.md`, dated after the old report). This report
replaces it per the report contract in the current task instructions.

## What changed

- `src/fbpipe/config.py:148-162` (`_expand_datasets`): inserted a check between
  computing `skipped` and the existing "Skipping datasets not yet on disk" print.
  Reads `data.get("dataset_overrides")` (the RAW yaml dict, since `_expand_datasets`
  runs inside `load_raw_config` before `Settings`/`DatasetOverride` exist), filters
  `skipped` down to datasets whose raw override is a dict with
  `(raw_overrides[ds].get("freeze") or {}).get("data", False)` truthy, and raises
  `RuntimeError` naming the frozen-missing dataset(s) if any. The pre-existing
  silent-skip print is unchanged and still runs for whatever remains in `skipped`
  (unfrozen missing datasets); the `raise` short-circuits before that print when
  there's a frozen-missing dataset. No other line in `_expand_datasets` or
  `load_raw_config` touched. The early return at `_expand_datasets:133-135`
  (`if not datasets: return data`) — the path legacy configs with no top-level
  `datasets:` take — is untouched, so legacy protocol configs never reach this
  new code.

  Diff:
  ```diff
       skipped = [ds for ds in datasets
                  if not Path(data_base, ds).is_dir() and not Path(secured_base, ds).is_dir()]
  +    # A frozen dataset that is not on disk cannot be silently dropped: freeze
  +    # assumes the raw data stays put, and we cannot auto-rebuild what is absent.
  +    # Dropping it would quietly delete its rows from the wide CSV.
  +    raw_overrides = data.get("dataset_overrides") or {}
  +    frozen_missing = [
  +        ds for ds in skipped
  +        if isinstance(raw_overrides.get(ds), dict)
  +        and (raw_overrides[ds].get("freeze") or {}).get("data", False)
  +    ]
  +    if frozen_missing:
  +        raise RuntimeError(
  +            f"[config] Frozen dataset(s) not found on disk: {', '.join(sorted(frozen_missing))}\n"
  +            f"         freeze.data assumes the raw data stays on disk. Restore the "
  +            f"folder(s), or remove freeze.data from the dataset_overrides block."
  +        )
       if skipped:
           print(f"[config] Skipping datasets not yet on disk: {', '.join(skipped)}")
  ```

- `tests/test_freeze_missing_folder.py` (new): the brief's 3 tests verbatim
  (`test_frozen_dataset_missing_from_disk_raises`,
  `test_unfrozen_dataset_missing_from_disk_still_skips_silently`,
  `test_frozen_dataset_present_on_disk_is_fine`), plus one extra test
  `test_frozen_dataset_but_present_on_disk_does_not_raise` that explicitly
  asserts the missing-case fixture's `GONE-24-1` folder is genuinely absent
  under both `data/` and `secured/` before asserting a *different*, present
  frozen dataset (`HERE-24-2`) does not raise — confirming presence, not the
  `freeze.data` flag, is what avoids the error (per the task's "one extra
  assertion" requirement).

## Adaptation from the brief

None needed. The brief's target line range (`:146-149` for the `skipped`
block in `_expand_datasets`) matched the current file exactly — no drift
since the brief was written.

## Exact test commands and actual output

```
$ python -m pytest tests/test_freeze_missing_folder.py -v
```
Before the fix (Step 2, TDD red):
```
FAILED tests/test_freeze_missing_folder.py::test_frozen_dataset_missing_from_disk_raises
  Failed: DID NOT RAISE <class 'RuntimeError'>
Captured stdout: [config] Skipping datasets not yet on disk: GONE-24-1
1 failed, 3 passed, 14 warnings in 0.06s
```

After the fix (Step 4, TDD green):
```
======================== 4 passed, 14 warnings in 0.03s ========================
```
(4 passed, not 3, because of the extra presence-vs-flag assertion test added
per the task instructions.)

Freeze-suite cross-check:
```
$ python -m pytest tests/test_freeze_config.py tests/test_freeze_cache.py \
    tests/test_freeze_cli.py tests/test_freeze_figures.py \
    tests/test_freeze_pipeline.py tests/test_freeze_wide.py \
    tests/test_freeze_missing_folder.py -q
74 passed, 192 warnings in 4.66s
```

Full suite:
```
$ python -m pytest tests/ -q
612 passed, 1 xfailed, 367 warnings in 49.39s
```
612 passed here vs. the brief's/task instructions' baseline of 608 passed,
1 xfailed at HEAD — the delta of 4 is exactly the 4 new tests in
`test_freeze_missing_folder.py`. No regressions. `HomeAssistant/test_influxdb_enclosure.py`
lives outside `tests/` and is not collected by `pytest tests/ -q`, consistent
with the instructions to ignore its 3 pre-existing errors.

## Self-review

- Scope: `git status` after the commit shows only
  `src/fbpipe/config.py` (modified) and `tests/test_freeze_missing_folder.py`
  (new, tracked) as the commit's contents — no other file touched, moved, or
  deleted. `git show --stat` on the commit confirms exactly these 2 files.
- Behavior matches spec precisely:
  - missing + `freeze.data: true` -> `RuntimeError` naming the dataset(s).
  - missing + no freeze (or `freeze.figures` only, or `freeze.data: false`) ->
    unchanged silent skip (`.get("data", False)` defaults falsy in all of
    these cases; `freeze.figures` alone never sets `.get("data")` truthy).
  - present + frozen -> no error, dataset included in `main_directories` as
    before (handled entirely by the pre-existing `data_roots`/`secured_roots`
    list comprehensions; a present dataset never enters `skipped` at all).
  - Non-dict override values (e.g. `dataset_overrides: {GONE-24-1: null}`)
    are guarded by `isinstance(raw_overrides.get(ds), dict)`, so a malformed
    or empty override doesn't crash the check.
  - `freeze: None` / `freeze: {}` in yaml -> `(raw_overrides[ds].get("freeze") or {})`
    coerces `None`/missing to `{}`, `.get("data", False)` then defaults to
    `False` -> no raise, matching "only freeze.data triggers it."
- Legacy protocol path: `_expand_datasets`'s early return
  (`if not datasets: return data`, lines 133-135) is unmodified and sits
  before all new code, so legacy configs with no top-level `datasets:` never
  execute the new branch. Confirmed by reading the diff — the new block is
  inserted strictly after the existing `skipped = [...]` line, well past the
  early return.
- Default-off: a config with a `datasets:` list but no frozen datasets
  produces `frozen_missing == []` always (list comprehension over `skipped`,
  guarded by the `freeze.data` check), so `if frozen_missing:` is always
  false and behavior is byte-for-byte identical to before.
- Working tree left clean after commit (verified via `git status`).

## Commit

`86c5d289e926aeb318eadc73cef6015cf69a804d`
`feat(config): frozen dataset missing from disk is a hard error, not a silent drop`
