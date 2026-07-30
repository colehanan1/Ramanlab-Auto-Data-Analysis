# Task 5 Report: Recompute rig_3 angles — invalidation tool

**Scope executed: Steps 1-5 ONLY.** Steps 6-10 (apply invalidation, rerun
pipeline, verify, commit) were explicitly NOT executed per the hard-stop
instruction in this task's brief. No file under
`/home/ramanlab/Documents/cole/Data/` or `/securedstorage/` was written,
modified, or deleted. `scripts/pipeline/run_workflows.py` was never invoked.
The recompute script was only ever run with `--dry-run`.

## Files created

- `scripts/pipeline/recompute_rig3_angles.py` (verbatim from the brief)
- `tests/test_recompute_rig3_angles.py` (verbatim from the brief)

No other file was modified. Both are currently **uncommitted** (untracked) —
see "Commit" section below for why, and the concurrent-edit note for context
on the rest of the working tree.

## Step 1: Write the failing test

Created `tests/test_recompute_rig3_angles.py` exactly as specified in the
brief (4 tests: `test_angle_columns_cover_both_producers`,
`test_invalidate_drops_only_angle_columns`,
`test_invalidate_is_safe_when_columns_absent`,
`test_find_rig3_tables_selects_only_rig3`).

## Step 2: Run test to verify it fails (RED)

Command:
```
conda run -n yolo-env python -m pytest tests/test_recompute_rig3_angles.py -v
```

Output (relevant excerpt):
```
ERROR collecting tests/test_recompute_rig3_angles.py
ImportError while importing test module '.../tests/test_recompute_rig3_angles.py'.
tests/test_recompute_rig3_angles.py:10: in <module>
    from scripts.pipeline.recompute_rig3_angles import (
E   ModuleNotFoundError: No module named 'scripts.pipeline.recompute_rig3_angles'
...
======================== 14 warnings, 1 error in 0.07s =========================
```

RED confirmed — matches the brief's expected failure exactly.

## Step 3: Write the implementation

Created `scripts/pipeline/recompute_rig3_angles.py` exactly as specified in
the brief. Verified prerequisite interfaces already exist on disk (Task 1):
`src/fbpipe/utils/rig_anchor.py` defines `MIRRORED_RIGS = frozenset({"rig_3"})`
and `rig_token()` with the underscore-required regex (`rig_(\d+)`), matching
the brief's description.

`scripts/__init__.py` and `scripts/pipeline/__init__.py` already existed, so
no package-init changes were needed for the `scripts.pipeline.*` import path.

## Step 4: Run test to verify it passes (GREEN)

Command:
```
conda run -n yolo-env python -m pytest tests/test_recompute_rig3_angles.py -v
```

Output:
```
tests/test_recompute_rig3_angles.py::test_angle_columns_cover_both_producers PASSED [ 25%]
tests/test_recompute_rig3_angles.py::test_invalidate_drops_only_angle_columns PASSED [ 50%]
tests/test_recompute_rig3_angles.py::test_invalidate_is_safe_when_columns_absent PASSED [ 75%]
tests/test_recompute_rig3_angles.py::test_find_rig3_tables_selects_only_rig3 PASSED [100%]
======================== 4 passed, 14 warnings in 0.02s ========================
```

GREEN confirmed, 4/4 passed.

## Mutation check (required by brief's testing note)

Per the brief's instruction: "Before reporting done, verify your tests bite —
at minimum make `find_rig3_tables` return ALL tables regardless of rig and
confirm the rig_2-exclusion test FAILS. Revert."

1. Backed up the original implementation file.
2. Mutated `find_rig3_tables` so the `rig_token(path) in MIRRORED_RIGS` guard
   was removed — every `*_distances.parquet` path found by `rglob` was
   appended unconditionally (i.e. it would return rig_2 tables too).
3. Reran the suite:
   ```
   tests/test_recompute_rig3_angles.py::test_angle_columns_cover_both_producers PASSED [ 25%]
   tests/test_recompute_rig3_angles.py::test_invalidate_drops_only_angle_columns PASSED [ 50%]
   tests/test_recompute_rig3_angles.py::test_invalidate_is_safe_when_columns_absent PASSED [ 75%]
   tests/test_recompute_rig3_angles.py::test_find_rig3_tables_selects_only_rig3 FAILED [100%]
   =================== 1 failed, 3 passed, 14 warnings in 0.05s ===================
   ```
   `test_find_rig3_tables_selects_only_rig3` correctly FAILED (it asserts
   `len(found) == 1`; with the rig_2 guard removed it found 2). The test
   bites — it is not vacuous.
4. Reverted the mutation. Diffed the restored file byte-for-byte against the
   pre-mutation backup: **IDENTICAL**. Reran the suite: 4/4 PASSED again
   (confirmed above is the same post-revert run).

Note on the `tmp_path`/rig-token trap the brief warned about: this test's
node id is `test_find_rig3_tables_selects_only_rig3`, which contains no
`rig_2`/`rig_3` substring itself, so pytest's `tmp_path` (which bakes the test
name into the directory) does not accidentally make the bare `tmp_path` root
resolve as a rig. The test additionally builds its own explicit
`..._rig_3/...` and `..._rig_2/...` subdirectories under `tmp_path` and reads
`rig_token` from the full constructed path, not from `tmp_path` itself, so the
trap does not apply here regardless.

## Step 5: Dry-run against the real trees

Command:
```
conda run -n yolo-env python -m scripts.pipeline.recompute_rig3_angles --dry-run \
  /home/ramanlab/Documents/cole/Data/flys_New \
  /securedstorage/DATAsec/cole/Data-secured-New
```

This is **read-only** — `--dry-run` was passed, so no `.to_parquet()` write
path was ever entered.

### Summary of output

```
found 948 rig_3 tables
...(946 "would drop [...] from <path>" lines)...
would update 948 of 948 tables
```

### Validation against ground truth

| Check | Result |
| --- | --- |
| Total rig_3 tables found | 948 |
| Tables with angle columns present (`changed`) | 948 of 948 |
| Unreadable/SKIP tables | 0 |
| Datasets represented | `EB-Control-24-1` (168 tables), `EB-Training-24-1` (780 tables) — **only these two** |
| `rig_2` substring in any reported path | **0 occurrences** |
| `3Oct` substring in any reported path | **0 occurrences** (consistent with the two `-0.11` datasets holding zero `*_distances.parquet`) |
| Distinct `*_rig_3` directory names represented | 6: `july_17_batch_1_rig_3`, `july_17_batch_2_rig_3`, `july_18_batch_1_rig_3`, `july_18_batch_2_rig_3`, `july_19_batch_1_rig_3`, `july_19_batch_2_rig_3` |
| Distinct rig_3 dirs per dataset | `EB-Control-24-1` → 1 (`july_17_batch_1_rig_3`); `EB-Training-24-1` → 5 (the other five) |

This is an **exact match** to the ground truth stated in the task brief:
- EB-Training-24-1 → 5 rig_3 dirs, HAS data ✓ (5 distinct dirs, 780 tables)
- EB-Control-24-1 → 1 rig_3 dir, HAS data ✓ (1 distinct dir, 168 tables)
- 3Oct-Training-24-0.11 / 3Oct-Control-24-0.11 → absent from output ✓ (zero
  parquets, as expected)
- No rig_2 path anywhere in the output ✓

**No STOP-THE-LINE bug found in `find_rig3_tables`.** The full 952-line raw
output is appended verbatim below for the record.

One incidental observation (not a bug, just noted for context): every "would
drop" line lists exactly `['angle_ARB_deg', 'angle_centered_deg',
'angle_centered_pct']` — `angle_multiplier` never appears in any drop list,
meaning that column is not currently present in any of these 948 on-disk
tables (only the other three angle columns are cached so far). The script's
drop set only ever includes columns actually present, so this is expected,
correct behavior — nothing to fix.

## Explicitly NOT executed (per hard stop)

- Step 6 (hash rig_2 baseline) — NOT run.
- Step 7 (apply invalidation for real, then `run_workflows.py` for
  EB-Training-24-1 / EB-Control-24-1) — NOT run. The recompute script was
  never invoked without `--dry-run`.
- Step 8 (verify rig_2 byte-identical) — NOT run (nothing was changed to
  verify against).
- Step 9 (verify rig_3 corrected via correlation sign) — NOT run.
- Step 10 (git commit of the two new files, per the brief's exact message) —
  NOT run, per this task's own instructions overriding the brief's Step 10.
  The two files exist on disk, untracked, ready for the user/orchestrator to
  commit when appropriate.

No file under `/home/ramanlab/Documents/cole/Data/` or `/securedstorage/` was
touched. `git status` inside the repo confirms only two new untracked files
attributable to this task; everything else in `git status` (staged and
unstaged) predates this task and was left untouched (see below).

## Concurrent-edit note

The working tree has substantial staged and unstaged changes from the user's
concurrent work session, exactly as flagged in this task's instructions
("USER IS EDITING THIS REPO CONCURRENTLY"). At the time this task finished,
`git status --porcelain` additionally showed a newly-appeared pair of files
that were not present in the git status snapshot given at task start:
`scripts/analysis/remake_reaction_matrix_split.py` and
`tests/test_remake_reaction_matrix_split.py` (both untracked). These were not
touched, created, or referenced by this task — they appeared from the
concurrent session while this task was running. Flagging per the scope-guard
instruction to mention unfamiliar changes rather than silently ignore or
touch them.

## Commit

**No commit was made.** The brief's Step 10 instructs committing the two new
files, but this task's hard-stop instructions restrict execution to Steps
1-5 only ("STOP after Step 5 (the dry-run) and report back... You must NOT
execute Steps 6, 7, 8, 9, or 10 under any circumstances"). Step 10 is
explicitly in that excluded range, so it was skipped. The two files
(`scripts/pipeline/recompute_rig3_angles.py`,
`tests/test_recompute_rig3_angles.py`) remain uncommitted/untracked on
`feature/rig3-mirrored-anchor`, matching every byte of the brief's Step 3
code block.

Commit SHA: **N/A — not committed** (by design, per hard-stop scope).

## Appendix: full Step 5 dry-run output (verbatim, 952 lines)

```
found 948 rig_3 tables
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Control-24-1/july_17_batch_1_rig_3/RMS_calculations/updated_july_17_batch_1_testing_1_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Control-24-1/july_17_batch_1_rig_3/RMS_calculations/updated_july_17_batch_1_testing_1_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Control-24-1/july_17_batch_1_rig_3/RMS_calculations/updated_july_17_batch_1_testing_1_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Control-24-1/july_17_batch_1_rig_3/RMS_calculations/updated_july_17_batch_1_testing_2_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Control-24-1/july_17_batch_1_rig_3/RMS_calculations/updated_july_17_batch_1_testing_2_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Control-24-1/july_17_batch_1_rig_3/RMS_calculations/updated_july_17_batch_1_testing_2_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Control-24-1/july_17_batch_1_rig_3/RMS_calculations/updated_july_17_batch_1_testing_3_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Control-24-1/july_17_batch_1_rig_3/RMS_calculations/updated_july_17_batch_1_testing_3_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Control-24-1/july_17_batch_1_rig_3/RMS_calculations/updated_july_17_batch_1_testing_3_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Control-24-1/july_17_batch_1_rig_3/RMS_calculations/updated_july_17_batch_1_testing_4_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Control-24-1/july_17_batch_1_rig_3/RMS_calculations/updated_july_17_batch_1_testing_4_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Control-24-1/july_17_batch_1_rig_3/RMS_calculations/updated_july_17_batch_1_testing_4_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Control-24-1/july_17_batch_1_rig_3/RMS_calculations/updated_july_17_batch_1_testing_5_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Control-24-1/july_17_batch_1_rig_3/RMS_calculations/updated_july_17_batch_1_testing_5_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Control-24-1/july_17_batch_1_rig_3/RMS_calculations/updated_july_17_batch_1_testing_5_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Control-24-1/july_17_batch_1_rig_3/RMS_calculations/updated_july_17_batch_1_testing_6_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Control-24-1/july_17_batch_1_rig_3/RMS_calculations/updated_july_17_batch_1_testing_6_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Control-24-1/july_17_batch_1_rig_3/RMS_calculations/updated_july_17_batch_1_testing_6_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Control-24-1/july_17_batch_1_rig_3/RMS_calculations/updated_july_17_batch_1_testing_7_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Control-24-1/july_17_batch_1_rig_3/RMS_calculations/updated_july_17_batch_1_testing_7_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Control-24-1/july_17_batch_1_rig_3/RMS_calculations/updated_july_17_batch_1_testing_7_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Control-24-1/july_17_batch_1_rig_3/RMS_calculations/updated_july_17_batch_1_testing_8_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Control-24-1/july_17_batch_1_rig_3/RMS_calculations/updated_july_17_batch_1_testing_8_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Control-24-1/july_17_batch_1_rig_3/RMS_calculations/updated_july_17_batch_1_testing_8_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Control-24-1/july_17_batch_1_rig_3/RMS_calculations/updated_july_17_batch_1_testing_9_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Control-24-1/july_17_batch_1_rig_3/RMS_calculations/updated_july_17_batch_1_testing_9_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Control-24-1/july_17_batch_1_rig_3/RMS_calculations/updated_july_17_batch_1_testing_9_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Control-24-1/july_17_batch_1_rig_3/RMS_calculations/updated_july_17_batch_1_training_2_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Control-24-1/july_17_batch_1_rig_3/RMS_calculations/updated_july_17_batch_1_training_2_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Control-24-1/july_17_batch_1_rig_3/RMS_calculations/updated_july_17_batch_1_training_2_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Control-24-1/july_17_batch_1_rig_3/RMS_calculations/updated_july_17_batch_1_training_3_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Control-24-1/july_17_batch_1_rig_3/RMS_calculations/updated_july_17_batch_1_training_3_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Control-24-1/july_17_batch_1_rig_3/RMS_calculations/updated_july_17_batch_1_training_3_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Control-24-1/july_17_batch_1_rig_3/RMS_calculations/updated_july_17_batch_1_training_4_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Control-24-1/july_17_batch_1_rig_3/RMS_calculations/updated_july_17_batch_1_training_4_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Control-24-1/july_17_batch_1_rig_3/RMS_calculations/updated_july_17_batch_1_training_4_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Control-24-1/july_17_batch_1_rig_3/RMS_calculations/updated_july_17_batch_1_training_5_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Control-24-1/july_17_batch_1_rig_3/RMS_calculations/updated_july_17_batch_1_training_5_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Control-24-1/july_17_batch_1_rig_3/RMS_calculations/updated_july_17_batch_1_training_5_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Control-24-1/july_17_batch_1_rig_3/RMS_calculations/updated_july_17_batch_1_training_6_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Control-24-1/july_17_batch_1_rig_3/RMS_calculations/updated_july_17_batch_1_training_6_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Control-24-1/july_17_batch_1_rig_3/RMS_calculations/updated_july_17_batch_1_training_6_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Control-24-1/july_17_batch_1_rig_3/july_17_batch_1_testing_1/july_17_batch_1_testing_1_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Control-24-1/july_17_batch_1_rig_3/july_17_batch_1_testing_1/july_17_batch_1_testing_1_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Control-24-1/july_17_batch_1_rig_3/july_17_batch_1_testing_1/july_17_batch_1_testing_1_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Control-24-1/july_17_batch_1_rig_3/july_17_batch_1_testing_2/july_17_batch_1_testing_2_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Control-24-1/july_17_batch_1_rig_3/july_17_batch_1_testing_2/july_17_batch_1_testing_2_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Control-24-1/july_17_batch_1_rig_3/july_17_batch_1_testing_2/july_17_batch_1_testing_2_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Control-24-1/july_17_batch_1_rig_3/july_17_batch_1_testing_3/july_17_batch_1_testing_3_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Control-24-1/july_17_batch_1_rig_3/july_17_batch_1_testing_3/july_17_batch_1_testing_3_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Control-24-1/july_17_batch_1_rig_3/july_17_batch_1_testing_3/july_17_batch_1_testing_3_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Control-24-1/july_17_batch_1_rig_3/july_17_batch_1_testing_4/july_17_batch_1_testing_4_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Control-24-1/july_17_batch_1_rig_3/july_17_batch_1_testing_4/july_17_batch_1_testing_4_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Control-24-1/july_17_batch_1_rig_3/july_17_batch_1_testing_4/july_17_batch_1_testing_4_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Control-24-1/july_17_batch_1_rig_3/july_17_batch_1_testing_5/july_17_batch_1_testing_5_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Control-24-1/july_17_batch_1_rig_3/july_17_batch_1_testing_5/july_17_batch_1_testing_5_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Control-24-1/july_17_batch_1_rig_3/july_17_batch_1_testing_5/july_17_batch_1_testing_5_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Control-24-1/july_17_batch_1_rig_3/july_17_batch_1_testing_6/july_17_batch_1_testing_6_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Control-24-1/july_17_batch_1_rig_3/july_17_batch_1_testing_6/july_17_batch_1_testing_6_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Control-24-1/july_17_batch_1_rig_3/july_17_batch_1_testing_6/july_17_batch_1_testing_6_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Control-24-1/july_17_batch_1_rig_3/july_17_batch_1_testing_7/july_17_batch_1_testing_7_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Control-24-1/july_17_batch_1_rig_3/july_17_batch_1_testing_7/july_17_batch_1_testing_7_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Control-24-1/july_17_batch_1_rig_3/july_17_batch_1_testing_7/july_17_batch_1_testing_7_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Control-24-1/july_17_batch_1_rig_3/july_17_batch_1_testing_8/july_17_batch_1_testing_8_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Control-24-1/july_17_batch_1_rig_3/july_17_batch_1_testing_8/july_17_batch_1_testing_8_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Control-24-1/july_17_batch_1_rig_3/july_17_batch_1_testing_8/july_17_batch_1_testing_8_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Control-24-1/july_17_batch_1_rig_3/july_17_batch_1_testing_9/july_17_batch_1_testing_9_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Control-24-1/july_17_batch_1_rig_3/july_17_batch_1_testing_9/july_17_batch_1_testing_9_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Control-24-1/july_17_batch_1_rig_3/july_17_batch_1_testing_9/july_17_batch_1_testing_9_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Control-24-1/july_17_batch_1_rig_3/july_17_batch_1_training_2/july_17_batch_1_training_2_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Control-24-1/july_17_batch_1_rig_3/july_17_batch_1_training_2/july_17_batch_1_training_2_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Control-24-1/july_17_batch_1_rig_3/july_17_batch_1_training_2/july_17_batch_1_training_2_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Control-24-1/july_17_batch_1_rig_3/july_17_batch_1_training_3/july_17_batch_1_training_3_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Control-24-1/july_17_batch_1_rig_3/july_17_batch_1_training_3/july_17_batch_1_training_3_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Control-24-1/july_17_batch_1_rig_3/july_17_batch_1_training_3/july_17_batch_1_training_3_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Control-24-1/july_17_batch_1_rig_3/july_17_batch_1_training_4/july_17_batch_1_training_4_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Control-24-1/july_17_batch_1_rig_3/july_17_batch_1_training_4/july_17_batch_1_training_4_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Control-24-1/july_17_batch_1_rig_3/july_17_batch_1_training_4/july_17_batch_1_training_4_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Control-24-1/july_17_batch_1_rig_3/july_17_batch_1_training_5/july_17_batch_1_training_5_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Control-24-1/july_17_batch_1_rig_3/july_17_batch_1_training_5/july_17_batch_1_training_5_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Control-24-1/july_17_batch_1_rig_3/july_17_batch_1_training_5/july_17_batch_1_training_5_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Control-24-1/july_17_batch_1_rig_3/july_17_batch_1_training_6/july_17_batch_1_training_6_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Control-24-1/july_17_batch_1_rig_3/july_17_batch_1_training_6/july_17_batch_1_training_6_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Control-24-1/july_17_batch_1_rig_3/july_17_batch_1_training_6/july_17_batch_1_training_6_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_17_batch_2_rig_3/RMS_calculations/updated_july_17_batch_2_testing_1_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_17_batch_2_rig_3/RMS_calculations/updated_july_17_batch_2_testing_2_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_17_batch_2_rig_3/RMS_calculations/updated_july_17_batch_2_testing_3_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_17_batch_2_rig_3/RMS_calculations/updated_july_17_batch_2_testing_4_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_17_batch_2_rig_3/RMS_calculations/updated_july_17_batch_2_testing_5_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_17_batch_2_rig_3/RMS_calculations/updated_july_17_batch_2_testing_6_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_17_batch_2_rig_3/RMS_calculations/updated_july_17_batch_2_testing_7_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_17_batch_2_rig_3/RMS_calculations/updated_july_17_batch_2_testing_8_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_17_batch_2_rig_3/RMS_calculations/updated_july_17_batch_2_testing_9_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_17_batch_2_rig_3/RMS_calculations/updated_july_17_batch_2_training_1_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_17_batch_2_rig_3/RMS_calculations/updated_july_17_batch_2_training_2_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_17_batch_2_rig_3/RMS_calculations/updated_july_17_batch_2_training_3_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_17_batch_2_rig_3/RMS_calculations/updated_july_17_batch_2_training_4_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_17_batch_2_rig_3/RMS_calculations/updated_july_17_batch_2_training_5_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_17_batch_2_rig_3/RMS_calculations/updated_july_17_batch_2_training_6_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_17_batch_2_rig_3/july_17_batch_2_testing_1/july_17_batch_2_testing_1_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_17_batch_2_rig_3/july_17_batch_2_testing_2/july_17_batch_2_testing_2_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_17_batch_2_rig_3/july_17_batch_2_testing_3/july_17_batch_2_testing_3_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_17_batch_2_rig_3/july_17_batch_2_testing_4/july_17_batch_2_testing_4_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_17_batch_2_rig_3/july_17_batch_2_testing_5/july_17_batch_2_testing_5_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_17_batch_2_rig_3/july_17_batch_2_testing_6/july_17_batch_2_testing_6_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_17_batch_2_rig_3/july_17_batch_2_testing_7/july_17_batch_2_testing_7_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_17_batch_2_rig_3/july_17_batch_2_testing_8/july_17_batch_2_testing_8_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_17_batch_2_rig_3/july_17_batch_2_testing_9/july_17_batch_2_testing_9_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_17_batch_2_rig_3/july_17_batch_2_training_1/july_17_batch_2_training_1_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_17_batch_2_rig_3/july_17_batch_2_training_2/july_17_batch_2_training_2_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_17_batch_2_rig_3/july_17_batch_2_training_3/july_17_batch_2_training_3_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_17_batch_2_rig_3/july_17_batch_2_training_4/july_17_batch_2_training_4_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_17_batch_2_rig_3/july_17_batch_2_training_5/july_17_batch_2_training_5_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_17_batch_2_rig_3/july_17_batch_2_training_6/july_17_batch_2_training_6_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_testing_1_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_testing_1_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_testing_1_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_testing_1_fly4_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_testing_2_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_testing_2_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_testing_2_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_testing_2_fly4_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_testing_3_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_testing_3_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_testing_3_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_testing_3_fly4_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_testing_4_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_testing_4_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_testing_4_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_testing_4_fly4_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_testing_5_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_testing_5_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_testing_5_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_testing_5_fly4_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_testing_6_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_testing_6_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_testing_6_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_testing_6_fly4_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_testing_7_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_testing_7_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_testing_7_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_testing_7_fly4_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_testing_8_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_testing_8_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_testing_8_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_testing_8_fly4_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_testing_9_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_testing_9_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_testing_9_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_testing_9_fly4_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_training_1_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_training_1_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_training_1_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_training_1_fly4_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_training_2_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_training_2_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_training_2_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_training_2_fly4_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_training_3_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_training_3_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_training_3_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_training_3_fly4_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_training_4_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_training_4_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_training_4_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_training_4_fly4_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_training_5_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_training_5_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_training_5_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_training_5_fly4_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_training_6_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_training_6_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_training_6_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_training_6_fly4_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_testing_1/july_18_batch_1_testing_1_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_testing_1/july_18_batch_1_testing_1_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_testing_1/july_18_batch_1_testing_1_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_testing_1/july_18_batch_1_testing_1_fly4_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_testing_2/july_18_batch_1_testing_2_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_testing_2/july_18_batch_1_testing_2_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_testing_2/july_18_batch_1_testing_2_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_testing_2/july_18_batch_1_testing_2_fly4_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_testing_3/july_18_batch_1_testing_3_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_testing_3/july_18_batch_1_testing_3_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_testing_3/july_18_batch_1_testing_3_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_testing_3/july_18_batch_1_testing_3_fly4_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_testing_4/july_18_batch_1_testing_4_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_testing_4/july_18_batch_1_testing_4_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_testing_4/july_18_batch_1_testing_4_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_testing_4/july_18_batch_1_testing_4_fly4_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_testing_5/july_18_batch_1_testing_5_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_testing_5/july_18_batch_1_testing_5_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_testing_5/july_18_batch_1_testing_5_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_testing_5/july_18_batch_1_testing_5_fly4_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_testing_6/july_18_batch_1_testing_6_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_testing_6/july_18_batch_1_testing_6_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_testing_6/july_18_batch_1_testing_6_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_testing_6/july_18_batch_1_testing_6_fly4_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_testing_7/july_18_batch_1_testing_7_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_testing_7/july_18_batch_1_testing_7_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_testing_7/july_18_batch_1_testing_7_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_testing_7/july_18_batch_1_testing_7_fly4_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_testing_8/july_18_batch_1_testing_8_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_testing_8/july_18_batch_1_testing_8_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_testing_8/july_18_batch_1_testing_8_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_testing_8/july_18_batch_1_testing_8_fly4_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_testing_9/july_18_batch_1_testing_9_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_testing_9/july_18_batch_1_testing_9_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_testing_9/july_18_batch_1_testing_9_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_testing_9/july_18_batch_1_testing_9_fly4_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_training_1/july_18_batch_1_training_1_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_training_1/july_18_batch_1_training_1_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_training_1/july_18_batch_1_training_1_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_training_1/july_18_batch_1_training_1_fly4_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_training_2/july_18_batch_1_training_2_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_training_2/july_18_batch_1_training_2_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_training_2/july_18_batch_1_training_2_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_training_2/july_18_batch_1_training_2_fly4_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_training_3/july_18_batch_1_training_3_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_training_3/july_18_batch_1_training_3_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_training_3/july_18_batch_1_training_3_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_training_3/july_18_batch_1_training_3_fly4_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_training_4/july_18_batch_1_training_4_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_training_4/july_18_batch_1_training_4_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_training_4/july_18_batch_1_training_4_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_training_4/july_18_batch_1_training_4_fly4_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_training_5/july_18_batch_1_training_5_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_training_5/july_18_batch_1_training_5_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_training_5/july_18_batch_1_training_5_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_training_5/july_18_batch_1_training_5_fly4_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_training_6/july_18_batch_1_training_6_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_training_6/july_18_batch_1_training_6_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_training_6/july_18_batch_1_training_6_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_training_6/july_18_batch_1_training_6_fly4_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_2_rig_3/RMS_calculations/updated_july_18_batch_2_testing_1_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_2_rig_3/RMS_calculations/updated_july_18_batch_2_testing_1_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_2_rig_3/RMS_calculations/updated_july_18_batch_2_testing_1_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_2_rig_3/RMS_calculations/updated_july_18_batch_2_testing_2_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_2_rig_3/RMS_calculations/updated_july_18_batch_2_testing_2_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_2_rig_3/RMS_calculations/updated_july_18_batch_2_testing_2_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_2_rig_3/RMS_calculations/updated_july_18_batch_2_testing_3_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_2_rig_3/RMS_calculations/updated_july_18_batch_2_testing_3_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_2_rig_3/RMS_calculations/updated_july_18_batch_2_testing_3_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_2_rig_3/RMS_calculations/updated_july_18_batch_2_testing_4_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_2_rig_3/RMS_calculations/updated_july_18_batch_2_testing_4_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_2_rig_3/RMS_calculations/updated_july_18_batch_2_testing_4_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_2_rig_3/RMS_calculations/updated_july_18_batch_2_testing_5_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_2_rig_3/RMS_calculations/updated_july_18_batch_2_testing_5_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_2_rig_3/RMS_calculations/updated_july_18_batch_2_testing_5_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_2_rig_3/RMS_calculations/updated_july_18_batch_2_testing_6_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_2_rig_3/RMS_calculations/updated_july_18_batch_2_testing_6_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_2_rig_3/RMS_calculations/updated_july_18_batch_2_testing_6_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_2_rig_3/RMS_calculations/updated_july_18_batch_2_testing_7_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_2_rig_3/RMS_calculations/updated_july_18_batch_2_testing_7_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_2_rig_3/RMS_calculations/updated_july_18_batch_2_testing_7_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_2_rig_3/RMS_calculations/updated_july_18_batch_2_testing_8_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_2_rig_3/RMS_calculations/updated_july_18_batch_2_testing_8_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_2_rig_3/RMS_calculations/updated_july_18_batch_2_testing_8_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_2_rig_3/RMS_calculations/updated_july_18_batch_2_testing_9_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_2_rig_3/RMS_calculations/updated_july_18_batch_2_testing_9_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_2_rig_3/RMS_calculations/updated_july_18_batch_2_testing_9_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_2_rig_3/RMS_calculations/updated_july_18_batch_2_training_1_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_2_rig_3/RMS_calculations/updated_july_18_batch_2_training_1_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_2_rig_3/RMS_calculations/updated_july_18_batch_2_training_1_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_2_rig_3/RMS_calculations/updated_july_18_batch_2_training_2_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_2_rig_3/RMS_calculations/updated_july_18_batch_2_training_2_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_2_rig_3/RMS_calculations/updated_july_18_batch_2_training_2_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_2_rig_3/RMS_calculations/updated_july_18_batch_2_training_3_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_2_rig_3/RMS_calculations/updated_july_18_batch_2_training_3_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_2_rig_3/RMS_calculations/updated_july_18_batch_2_training_3_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_2_rig_3/RMS_calculations/updated_july_18_batch_2_training_4_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_2_rig_3/RMS_calculations/updated_july_18_batch_2_training_4_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_2_rig_3/RMS_calculations/updated_july_18_batch_2_training_4_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_2_rig_3/RMS_calculations/updated_july_18_batch_2_training_5_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_2_rig_3/RMS_calculations/updated_july_18_batch_2_training_5_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_2_rig_3/RMS_calculations/updated_july_18_batch_2_training_5_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_2_rig_3/RMS_calculations/updated_july_18_batch_2_training_6_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_2_rig_3/RMS_calculations/updated_july_18_batch_2_training_6_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_2_rig_3/RMS_calculations/updated_july_18_batch_2_training_6_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_2_rig_3/july_18_batch_2_testing_1/july_18_batch_2_testing_1_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_2_rig_3/july_18_batch_2_testing_1/july_18_batch_2_testing_1_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_2_rig_3/july_18_batch_2_testing_1/july_18_batch_2_testing_1_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_2_rig_3/july_18_batch_2_testing_2/july_18_batch_2_testing_2_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_2_rig_3/july_18_batch_2_testing_2/july_18_batch_2_testing_2_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_2_rig_3/july_18_batch_2_testing_2/july_18_batch_2_testing_2_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_2_rig_3/july_18_batch_2_testing_3/july_18_batch_2_testing_3_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_2_rig_3/july_18_batch_2_testing_3/july_18_batch_2_testing_3_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_2_rig_3/july_18_batch_2_testing_3/july_18_batch_2_testing_3_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_2_rig_3/july_18_batch_2_testing_4/july_18_batch_2_testing_4_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_2_rig_3/july_18_batch_2_testing_4/july_18_batch_2_testing_4_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_2_rig_3/july_18_batch_2_testing_4/july_18_batch_2_testing_4_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_2_rig_3/july_18_batch_2_testing_5/july_18_batch_2_testing_5_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_2_rig_3/july_18_batch_2_testing_5/july_18_batch_2_testing_5_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_2_rig_3/july_18_batch_2_testing_5/july_18_batch_2_testing_5_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_2_rig_3/july_18_batch_2_testing_6/july_18_batch_2_testing_6_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_2_rig_3/july_18_batch_2_testing_6/july_18_batch_2_testing_6_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_2_rig_3/july_18_batch_2_testing_6/july_18_batch_2_testing_6_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_2_rig_3/july_18_batch_2_testing_7/july_18_batch_2_testing_7_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_2_rig_3/july_18_batch_2_testing_7/july_18_batch_2_testing_7_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_2_rig_3/july_18_batch_2_testing_7/july_18_batch_2_testing_7_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_2_rig_3/july_18_batch_2_testing_8/july_18_batch_2_testing_8_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_2_rig_3/july_18_batch_2_testing_8/july_18_batch_2_testing_8_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_2_rig_3/july_18_batch_2_testing_8/july_18_batch_2_testing_8_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_2_rig_3/july_18_batch_2_testing_9/july_18_batch_2_testing_9_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_2_rig_3/july_18_batch_2_testing_9/july_18_batch_2_testing_9_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_2_rig_3/july_18_batch_2_testing_9/july_18_batch_2_testing_9_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_2_rig_3/july_18_batch_2_training_1/july_18_batch_2_training_1_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_2_rig_3/july_18_batch_2_training_1/july_18_batch_2_training_1_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_2_rig_3/july_18_batch_2_training_1/july_18_batch_2_training_1_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_2_rig_3/july_18_batch_2_training_2/july_18_batch_2_training_2_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_2_rig_3/july_18_batch_2_training_2/july_18_batch_2_training_2_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_2_rig_3/july_18_batch_2_training_2/july_18_batch_2_training_2_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_2_rig_3/july_18_batch_2_training_3/july_18_batch_2_training_3_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_2_rig_3/july_18_batch_2_training_3/july_18_batch_2_training_3_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_2_rig_3/july_18_batch_2_training_3/july_18_batch_2_training_3_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_2_rig_3/july_18_batch_2_training_4/july_18_batch_2_training_4_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_2_rig_3/july_18_batch_2_training_4/july_18_batch_2_training_4_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_2_rig_3/july_18_batch_2_training_4/july_18_batch_2_training_4_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_2_rig_3/july_18_batch_2_training_5/july_18_batch_2_training_5_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_2_rig_3/july_18_batch_2_training_5/july_18_batch_2_training_5_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_2_rig_3/july_18_batch_2_training_5/july_18_batch_2_training_5_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_2_rig_3/july_18_batch_2_training_6/july_18_batch_2_training_6_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_2_rig_3/july_18_batch_2_training_6/july_18_batch_2_training_6_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_18_batch_2_rig_3/july_18_batch_2_training_6/july_18_batch_2_training_6_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_1_rig_3/RMS_calculations/updated_july_19_batch_1_testing_1_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_1_rig_3/RMS_calculations/updated_july_19_batch_1_testing_1_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_1_rig_3/RMS_calculations/updated_july_19_batch_1_testing_2_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_1_rig_3/RMS_calculations/updated_july_19_batch_1_testing_2_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_1_rig_3/RMS_calculations/updated_july_19_batch_1_testing_3_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_1_rig_3/RMS_calculations/updated_july_19_batch_1_testing_3_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_1_rig_3/RMS_calculations/updated_july_19_batch_1_testing_4_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_1_rig_3/RMS_calculations/updated_july_19_batch_1_testing_4_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_1_rig_3/RMS_calculations/updated_july_19_batch_1_testing_5_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_1_rig_3/RMS_calculations/updated_july_19_batch_1_testing_5_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_1_rig_3/RMS_calculations/updated_july_19_batch_1_testing_6_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_1_rig_3/RMS_calculations/updated_july_19_batch_1_testing_6_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_1_rig_3/RMS_calculations/updated_july_19_batch_1_testing_7_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_1_rig_3/RMS_calculations/updated_july_19_batch_1_testing_7_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_1_rig_3/RMS_calculations/updated_july_19_batch_1_testing_8_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_1_rig_3/RMS_calculations/updated_july_19_batch_1_testing_8_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_1_rig_3/RMS_calculations/updated_july_19_batch_1_testing_9_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_1_rig_3/RMS_calculations/updated_july_19_batch_1_testing_9_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_1_rig_3/RMS_calculations/updated_july_19_batch_1_training_1_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_1_rig_3/RMS_calculations/updated_july_19_batch_1_training_1_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_1_rig_3/RMS_calculations/updated_july_19_batch_1_training_2_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_1_rig_3/RMS_calculations/updated_july_19_batch_1_training_2_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_1_rig_3/RMS_calculations/updated_july_19_batch_1_training_3_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_1_rig_3/RMS_calculations/updated_july_19_batch_1_training_3_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_1_rig_3/RMS_calculations/updated_july_19_batch_1_training_4_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_1_rig_3/RMS_calculations/updated_july_19_batch_1_training_4_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_1_rig_3/RMS_calculations/updated_july_19_batch_1_training_5_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_1_rig_3/RMS_calculations/updated_july_19_batch_1_training_5_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_1_rig_3/RMS_calculations/updated_july_19_batch_1_training_6_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_1_rig_3/RMS_calculations/updated_july_19_batch_1_training_6_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_1_rig_3/july_19_batch_1_testing_1/july_19_batch_1_testing_1_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_1_rig_3/july_19_batch_1_testing_1/july_19_batch_1_testing_1_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_1_rig_3/july_19_batch_1_testing_2/july_19_batch_1_testing_2_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_1_rig_3/july_19_batch_1_testing_2/july_19_batch_1_testing_2_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_1_rig_3/july_19_batch_1_testing_3/july_19_batch_1_testing_3_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_1_rig_3/july_19_batch_1_testing_3/july_19_batch_1_testing_3_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_1_rig_3/july_19_batch_1_testing_4/july_19_batch_1_testing_4_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_1_rig_3/july_19_batch_1_testing_4/july_19_batch_1_testing_4_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_1_rig_3/july_19_batch_1_testing_5/july_19_batch_1_testing_5_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_1_rig_3/july_19_batch_1_testing_5/july_19_batch_1_testing_5_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_1_rig_3/july_19_batch_1_testing_6/july_19_batch_1_testing_6_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_1_rig_3/july_19_batch_1_testing_6/july_19_batch_1_testing_6_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_1_rig_3/july_19_batch_1_testing_7/july_19_batch_1_testing_7_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_1_rig_3/july_19_batch_1_testing_7/july_19_batch_1_testing_7_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_1_rig_3/july_19_batch_1_testing_8/july_19_batch_1_testing_8_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_1_rig_3/july_19_batch_1_testing_8/july_19_batch_1_testing_8_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_1_rig_3/july_19_batch_1_testing_9/july_19_batch_1_testing_9_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_1_rig_3/july_19_batch_1_testing_9/july_19_batch_1_testing_9_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_1_rig_3/july_19_batch_1_training_1/july_19_batch_1_training_1_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_1_rig_3/july_19_batch_1_training_1/july_19_batch_1_training_1_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_1_rig_3/july_19_batch_1_training_2/july_19_batch_1_training_2_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_1_rig_3/july_19_batch_1_training_2/july_19_batch_1_training_2_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_1_rig_3/july_19_batch_1_training_3/july_19_batch_1_training_3_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_1_rig_3/july_19_batch_1_training_3/july_19_batch_1_training_3_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_1_rig_3/july_19_batch_1_training_4/july_19_batch_1_training_4_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_1_rig_3/july_19_batch_1_training_4/july_19_batch_1_training_4_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_1_rig_3/july_19_batch_1_training_5/july_19_batch_1_training_5_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_1_rig_3/july_19_batch_1_training_5/july_19_batch_1_training_5_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_1_rig_3/july_19_batch_1_training_6/july_19_batch_1_training_6_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_1_rig_3/july_19_batch_1_training_6/july_19_batch_1_training_6_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_2_rig_3/RMS_calculations/updated_july_19_batch_2_testing_1_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_2_rig_3/RMS_calculations/updated_july_19_batch_2_testing_1_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_2_rig_3/RMS_calculations/updated_july_19_batch_2_testing_1_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_2_rig_3/RMS_calculations/updated_july_19_batch_2_testing_2_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_2_rig_3/RMS_calculations/updated_july_19_batch_2_testing_2_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_2_rig_3/RMS_calculations/updated_july_19_batch_2_testing_2_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_2_rig_3/RMS_calculations/updated_july_19_batch_2_testing_3_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_2_rig_3/RMS_calculations/updated_july_19_batch_2_testing_3_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_2_rig_3/RMS_calculations/updated_july_19_batch_2_testing_3_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_2_rig_3/RMS_calculations/updated_july_19_batch_2_testing_4_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_2_rig_3/RMS_calculations/updated_july_19_batch_2_testing_4_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_2_rig_3/RMS_calculations/updated_july_19_batch_2_testing_4_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_2_rig_3/RMS_calculations/updated_july_19_batch_2_testing_5_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_2_rig_3/RMS_calculations/updated_july_19_batch_2_testing_5_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_2_rig_3/RMS_calculations/updated_july_19_batch_2_testing_5_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_2_rig_3/RMS_calculations/updated_july_19_batch_2_testing_6_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_2_rig_3/RMS_calculations/updated_july_19_batch_2_testing_6_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_2_rig_3/RMS_calculations/updated_july_19_batch_2_testing_6_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_2_rig_3/RMS_calculations/updated_july_19_batch_2_testing_7_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_2_rig_3/RMS_calculations/updated_july_19_batch_2_testing_7_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_2_rig_3/RMS_calculations/updated_july_19_batch_2_testing_7_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_2_rig_3/RMS_calculations/updated_july_19_batch_2_testing_8_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_2_rig_3/RMS_calculations/updated_july_19_batch_2_testing_8_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_2_rig_3/RMS_calculations/updated_july_19_batch_2_testing_8_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_2_rig_3/RMS_calculations/updated_july_19_batch_2_testing_9_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_2_rig_3/RMS_calculations/updated_july_19_batch_2_testing_9_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_2_rig_3/RMS_calculations/updated_july_19_batch_2_testing_9_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_2_rig_3/RMS_calculations/updated_july_19_batch_2_training_1_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_2_rig_3/RMS_calculations/updated_july_19_batch_2_training_1_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_2_rig_3/RMS_calculations/updated_july_19_batch_2_training_1_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_2_rig_3/RMS_calculations/updated_july_19_batch_2_training_2_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_2_rig_3/RMS_calculations/updated_july_19_batch_2_training_2_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_2_rig_3/RMS_calculations/updated_july_19_batch_2_training_2_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_2_rig_3/RMS_calculations/updated_july_19_batch_2_training_3_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_2_rig_3/RMS_calculations/updated_july_19_batch_2_training_3_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_2_rig_3/RMS_calculations/updated_july_19_batch_2_training_3_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_2_rig_3/RMS_calculations/updated_july_19_batch_2_training_4_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_2_rig_3/RMS_calculations/updated_july_19_batch_2_training_4_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_2_rig_3/RMS_calculations/updated_july_19_batch_2_training_4_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_2_rig_3/RMS_calculations/updated_july_19_batch_2_training_5_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_2_rig_3/RMS_calculations/updated_july_19_batch_2_training_5_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_2_rig_3/RMS_calculations/updated_july_19_batch_2_training_5_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_2_rig_3/RMS_calculations/updated_july_19_batch_2_training_6_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_2_rig_3/RMS_calculations/updated_july_19_batch_2_training_6_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_2_rig_3/RMS_calculations/updated_july_19_batch_2_training_6_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_2_rig_3/july_19_batch_2_testing_1/july_19_batch_2_testing_1_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_2_rig_3/july_19_batch_2_testing_1/july_19_batch_2_testing_1_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_2_rig_3/july_19_batch_2_testing_1/july_19_batch_2_testing_1_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_2_rig_3/july_19_batch_2_testing_2/july_19_batch_2_testing_2_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_2_rig_3/july_19_batch_2_testing_2/july_19_batch_2_testing_2_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_2_rig_3/july_19_batch_2_testing_2/july_19_batch_2_testing_2_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_2_rig_3/july_19_batch_2_testing_3/july_19_batch_2_testing_3_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_2_rig_3/july_19_batch_2_testing_3/july_19_batch_2_testing_3_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_2_rig_3/july_19_batch_2_testing_3/july_19_batch_2_testing_3_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_2_rig_3/july_19_batch_2_testing_4/july_19_batch_2_testing_4_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_2_rig_3/july_19_batch_2_testing_4/july_19_batch_2_testing_4_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_2_rig_3/july_19_batch_2_testing_4/july_19_batch_2_testing_4_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_2_rig_3/july_19_batch_2_testing_5/july_19_batch_2_testing_5_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_2_rig_3/july_19_batch_2_testing_5/july_19_batch_2_testing_5_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_2_rig_3/july_19_batch_2_testing_5/july_19_batch_2_testing_5_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_2_rig_3/july_19_batch_2_testing_6/july_19_batch_2_testing_6_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_2_rig_3/july_19_batch_2_testing_6/july_19_batch_2_testing_6_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_2_rig_3/july_19_batch_2_testing_6/july_19_batch_2_testing_6_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_2_rig_3/july_19_batch_2_testing_7/july_19_batch_2_testing_7_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_2_rig_3/july_19_batch_2_testing_7/july_19_batch_2_testing_7_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_2_rig_3/july_19_batch_2_testing_7/july_19_batch_2_testing_7_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_2_rig_3/july_19_batch_2_testing_8/july_19_batch_2_testing_8_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_2_rig_3/july_19_batch_2_testing_8/july_19_batch_2_testing_8_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_2_rig_3/july_19_batch_2_testing_8/july_19_batch_2_testing_8_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_2_rig_3/july_19_batch_2_testing_9/july_19_batch_2_testing_9_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_2_rig_3/july_19_batch_2_testing_9/july_19_batch_2_testing_9_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_2_rig_3/july_19_batch_2_testing_9/july_19_batch_2_testing_9_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_2_rig_3/july_19_batch_2_training_1/july_19_batch_2_training_1_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_2_rig_3/july_19_batch_2_training_1/july_19_batch_2_training_1_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_2_rig_3/july_19_batch_2_training_1/july_19_batch_2_training_1_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_2_rig_3/july_19_batch_2_training_2/july_19_batch_2_training_2_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_2_rig_3/july_19_batch_2_training_2/july_19_batch_2_training_2_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_2_rig_3/july_19_batch_2_training_2/july_19_batch_2_training_2_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_2_rig_3/july_19_batch_2_training_3/july_19_batch_2_training_3_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_2_rig_3/july_19_batch_2_training_3/july_19_batch_2_training_3_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_2_rig_3/july_19_batch_2_training_3/july_19_batch_2_training_3_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_2_rig_3/july_19_batch_2_training_4/july_19_batch_2_training_4_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_2_rig_3/july_19_batch_2_training_4/july_19_batch_2_training_4_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_2_rig_3/july_19_batch_2_training_4/july_19_batch_2_training_4_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_2_rig_3/july_19_batch_2_training_5/july_19_batch_2_training_5_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_2_rig_3/july_19_batch_2_training_5/july_19_batch_2_training_5_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_2_rig_3/july_19_batch_2_training_5/july_19_batch_2_training_5_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_2_rig_3/july_19_batch_2_training_6/july_19_batch_2_training_6_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_2_rig_3/july_19_batch_2_training_6/july_19_batch_2_training_6_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1/july_19_batch_2_rig_3/july_19_batch_2_training_6/july_19_batch_2_training_6_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Control-24-1/july_17_batch_1_rig_3/RMS_calculations/updated_july_17_batch_1_testing_1_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Control-24-1/july_17_batch_1_rig_3/RMS_calculations/updated_july_17_batch_1_testing_1_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Control-24-1/july_17_batch_1_rig_3/RMS_calculations/updated_july_17_batch_1_testing_1_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Control-24-1/july_17_batch_1_rig_3/RMS_calculations/updated_july_17_batch_1_testing_2_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Control-24-1/july_17_batch_1_rig_3/RMS_calculations/updated_july_17_batch_1_testing_2_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Control-24-1/july_17_batch_1_rig_3/RMS_calculations/updated_july_17_batch_1_testing_2_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Control-24-1/july_17_batch_1_rig_3/RMS_calculations/updated_july_17_batch_1_testing_3_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Control-24-1/july_17_batch_1_rig_3/RMS_calculations/updated_july_17_batch_1_testing_3_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Control-24-1/july_17_batch_1_rig_3/RMS_calculations/updated_july_17_batch_1_testing_3_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Control-24-1/july_17_batch_1_rig_3/RMS_calculations/updated_july_17_batch_1_testing_4_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Control-24-1/july_17_batch_1_rig_3/RMS_calculations/updated_july_17_batch_1_testing_4_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Control-24-1/july_17_batch_1_rig_3/RMS_calculations/updated_july_17_batch_1_testing_4_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Control-24-1/july_17_batch_1_rig_3/RMS_calculations/updated_july_17_batch_1_testing_5_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Control-24-1/july_17_batch_1_rig_3/RMS_calculations/updated_july_17_batch_1_testing_5_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Control-24-1/july_17_batch_1_rig_3/RMS_calculations/updated_july_17_batch_1_testing_5_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Control-24-1/july_17_batch_1_rig_3/RMS_calculations/updated_july_17_batch_1_testing_6_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Control-24-1/july_17_batch_1_rig_3/RMS_calculations/updated_july_17_batch_1_testing_6_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Control-24-1/july_17_batch_1_rig_3/RMS_calculations/updated_july_17_batch_1_testing_6_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Control-24-1/july_17_batch_1_rig_3/RMS_calculations/updated_july_17_batch_1_testing_7_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Control-24-1/july_17_batch_1_rig_3/RMS_calculations/updated_july_17_batch_1_testing_7_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Control-24-1/july_17_batch_1_rig_3/RMS_calculations/updated_july_17_batch_1_testing_7_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Control-24-1/july_17_batch_1_rig_3/RMS_calculations/updated_july_17_batch_1_testing_8_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Control-24-1/july_17_batch_1_rig_3/RMS_calculations/updated_july_17_batch_1_testing_8_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Control-24-1/july_17_batch_1_rig_3/RMS_calculations/updated_july_17_batch_1_testing_8_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Control-24-1/july_17_batch_1_rig_3/RMS_calculations/updated_july_17_batch_1_testing_9_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Control-24-1/july_17_batch_1_rig_3/RMS_calculations/updated_july_17_batch_1_testing_9_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Control-24-1/july_17_batch_1_rig_3/RMS_calculations/updated_july_17_batch_1_testing_9_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Control-24-1/july_17_batch_1_rig_3/RMS_calculations/updated_july_17_batch_1_training_2_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Control-24-1/july_17_batch_1_rig_3/RMS_calculations/updated_july_17_batch_1_training_2_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Control-24-1/july_17_batch_1_rig_3/RMS_calculations/updated_july_17_batch_1_training_2_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Control-24-1/july_17_batch_1_rig_3/RMS_calculations/updated_july_17_batch_1_training_3_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Control-24-1/july_17_batch_1_rig_3/RMS_calculations/updated_july_17_batch_1_training_3_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Control-24-1/july_17_batch_1_rig_3/RMS_calculations/updated_july_17_batch_1_training_3_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Control-24-1/july_17_batch_1_rig_3/RMS_calculations/updated_july_17_batch_1_training_4_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Control-24-1/july_17_batch_1_rig_3/RMS_calculations/updated_july_17_batch_1_training_4_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Control-24-1/july_17_batch_1_rig_3/RMS_calculations/updated_july_17_batch_1_training_4_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Control-24-1/july_17_batch_1_rig_3/RMS_calculations/updated_july_17_batch_1_training_5_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Control-24-1/july_17_batch_1_rig_3/RMS_calculations/updated_july_17_batch_1_training_5_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Control-24-1/july_17_batch_1_rig_3/RMS_calculations/updated_july_17_batch_1_training_5_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Control-24-1/july_17_batch_1_rig_3/RMS_calculations/updated_july_17_batch_1_training_6_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Control-24-1/july_17_batch_1_rig_3/RMS_calculations/updated_july_17_batch_1_training_6_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Control-24-1/july_17_batch_1_rig_3/RMS_calculations/updated_july_17_batch_1_training_6_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Control-24-1/july_17_batch_1_rig_3/july_17_batch_1_testing_1/july_17_batch_1_testing_1_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Control-24-1/july_17_batch_1_rig_3/july_17_batch_1_testing_1/july_17_batch_1_testing_1_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Control-24-1/july_17_batch_1_rig_3/july_17_batch_1_testing_1/july_17_batch_1_testing_1_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Control-24-1/july_17_batch_1_rig_3/july_17_batch_1_testing_2/july_17_batch_1_testing_2_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Control-24-1/july_17_batch_1_rig_3/july_17_batch_1_testing_2/july_17_batch_1_testing_2_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Control-24-1/july_17_batch_1_rig_3/july_17_batch_1_testing_2/july_17_batch_1_testing_2_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Control-24-1/july_17_batch_1_rig_3/july_17_batch_1_testing_3/july_17_batch_1_testing_3_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Control-24-1/july_17_batch_1_rig_3/july_17_batch_1_testing_3/july_17_batch_1_testing_3_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Control-24-1/july_17_batch_1_rig_3/july_17_batch_1_testing_3/july_17_batch_1_testing_3_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Control-24-1/july_17_batch_1_rig_3/july_17_batch_1_testing_4/july_17_batch_1_testing_4_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Control-24-1/july_17_batch_1_rig_3/july_17_batch_1_testing_4/july_17_batch_1_testing_4_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Control-24-1/july_17_batch_1_rig_3/july_17_batch_1_testing_4/july_17_batch_1_testing_4_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Control-24-1/july_17_batch_1_rig_3/july_17_batch_1_testing_5/july_17_batch_1_testing_5_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Control-24-1/july_17_batch_1_rig_3/july_17_batch_1_testing_5/july_17_batch_1_testing_5_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Control-24-1/july_17_batch_1_rig_3/july_17_batch_1_testing_5/july_17_batch_1_testing_5_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Control-24-1/july_17_batch_1_rig_3/july_17_batch_1_testing_6/july_17_batch_1_testing_6_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Control-24-1/july_17_batch_1_rig_3/july_17_batch_1_testing_6/july_17_batch_1_testing_6_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Control-24-1/july_17_batch_1_rig_3/july_17_batch_1_testing_6/july_17_batch_1_testing_6_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Control-24-1/july_17_batch_1_rig_3/july_17_batch_1_testing_7/july_17_batch_1_testing_7_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Control-24-1/july_17_batch_1_rig_3/july_17_batch_1_testing_7/july_17_batch_1_testing_7_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Control-24-1/july_17_batch_1_rig_3/july_17_batch_1_testing_7/july_17_batch_1_testing_7_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Control-24-1/july_17_batch_1_rig_3/july_17_batch_1_testing_8/july_17_batch_1_testing_8_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Control-24-1/july_17_batch_1_rig_3/july_17_batch_1_testing_8/july_17_batch_1_testing_8_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Control-24-1/july_17_batch_1_rig_3/july_17_batch_1_testing_8/july_17_batch_1_testing_8_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Control-24-1/july_17_batch_1_rig_3/july_17_batch_1_testing_9/july_17_batch_1_testing_9_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Control-24-1/july_17_batch_1_rig_3/july_17_batch_1_testing_9/july_17_batch_1_testing_9_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Control-24-1/july_17_batch_1_rig_3/july_17_batch_1_testing_9/july_17_batch_1_testing_9_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Control-24-1/july_17_batch_1_rig_3/july_17_batch_1_training_2/july_17_batch_1_training_2_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Control-24-1/july_17_batch_1_rig_3/july_17_batch_1_training_2/july_17_batch_1_training_2_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Control-24-1/july_17_batch_1_rig_3/july_17_batch_1_training_2/july_17_batch_1_training_2_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Control-24-1/july_17_batch_1_rig_3/july_17_batch_1_training_3/july_17_batch_1_training_3_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Control-24-1/july_17_batch_1_rig_3/july_17_batch_1_training_3/july_17_batch_1_training_3_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Control-24-1/july_17_batch_1_rig_3/july_17_batch_1_training_3/july_17_batch_1_training_3_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Control-24-1/july_17_batch_1_rig_3/july_17_batch_1_training_4/july_17_batch_1_training_4_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Control-24-1/july_17_batch_1_rig_3/july_17_batch_1_training_4/july_17_batch_1_training_4_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Control-24-1/july_17_batch_1_rig_3/july_17_batch_1_training_4/july_17_batch_1_training_4_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Control-24-1/july_17_batch_1_rig_3/july_17_batch_1_training_5/july_17_batch_1_training_5_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Control-24-1/july_17_batch_1_rig_3/july_17_batch_1_training_5/july_17_batch_1_training_5_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Control-24-1/july_17_batch_1_rig_3/july_17_batch_1_training_5/july_17_batch_1_training_5_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Control-24-1/july_17_batch_1_rig_3/july_17_batch_1_training_6/july_17_batch_1_training_6_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Control-24-1/july_17_batch_1_rig_3/july_17_batch_1_training_6/july_17_batch_1_training_6_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Control-24-1/july_17_batch_1_rig_3/july_17_batch_1_training_6/july_17_batch_1_training_6_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_17_batch_2_rig_3/RMS_calculations/updated_july_17_batch_2_testing_1_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_17_batch_2_rig_3/RMS_calculations/updated_july_17_batch_2_testing_2_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_17_batch_2_rig_3/RMS_calculations/updated_july_17_batch_2_testing_3_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_17_batch_2_rig_3/RMS_calculations/updated_july_17_batch_2_testing_4_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_17_batch_2_rig_3/RMS_calculations/updated_july_17_batch_2_testing_5_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_17_batch_2_rig_3/RMS_calculations/updated_july_17_batch_2_testing_6_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_17_batch_2_rig_3/RMS_calculations/updated_july_17_batch_2_testing_7_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_17_batch_2_rig_3/RMS_calculations/updated_july_17_batch_2_testing_8_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_17_batch_2_rig_3/RMS_calculations/updated_july_17_batch_2_testing_9_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_17_batch_2_rig_3/RMS_calculations/updated_july_17_batch_2_training_1_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_17_batch_2_rig_3/RMS_calculations/updated_july_17_batch_2_training_2_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_17_batch_2_rig_3/RMS_calculations/updated_july_17_batch_2_training_3_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_17_batch_2_rig_3/RMS_calculations/updated_july_17_batch_2_training_4_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_17_batch_2_rig_3/RMS_calculations/updated_july_17_batch_2_training_5_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_17_batch_2_rig_3/RMS_calculations/updated_july_17_batch_2_training_6_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_17_batch_2_rig_3/july_17_batch_2_testing_1/july_17_batch_2_testing_1_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_17_batch_2_rig_3/july_17_batch_2_testing_2/july_17_batch_2_testing_2_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_17_batch_2_rig_3/july_17_batch_2_testing_3/july_17_batch_2_testing_3_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_17_batch_2_rig_3/july_17_batch_2_testing_4/july_17_batch_2_testing_4_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_17_batch_2_rig_3/july_17_batch_2_testing_5/july_17_batch_2_testing_5_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_17_batch_2_rig_3/july_17_batch_2_testing_6/july_17_batch_2_testing_6_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_17_batch_2_rig_3/july_17_batch_2_testing_7/july_17_batch_2_testing_7_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_17_batch_2_rig_3/july_17_batch_2_testing_8/july_17_batch_2_testing_8_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_17_batch_2_rig_3/july_17_batch_2_testing_9/july_17_batch_2_testing_9_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_17_batch_2_rig_3/july_17_batch_2_training_1/july_17_batch_2_training_1_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_17_batch_2_rig_3/july_17_batch_2_training_2/july_17_batch_2_training_2_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_17_batch_2_rig_3/july_17_batch_2_training_3/july_17_batch_2_training_3_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_17_batch_2_rig_3/july_17_batch_2_training_4/july_17_batch_2_training_4_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_17_batch_2_rig_3/july_17_batch_2_training_5/july_17_batch_2_training_5_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_17_batch_2_rig_3/july_17_batch_2_training_6/july_17_batch_2_training_6_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_testing_1_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_testing_1_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_testing_1_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_testing_1_fly4_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_testing_2_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_testing_2_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_testing_2_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_testing_2_fly4_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_testing_3_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_testing_3_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_testing_3_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_testing_3_fly4_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_testing_4_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_testing_4_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_testing_4_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_testing_4_fly4_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_testing_5_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_testing_5_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_testing_5_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_testing_5_fly4_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_testing_6_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_testing_6_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_testing_6_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_testing_6_fly4_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_testing_7_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_testing_7_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_testing_7_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_testing_7_fly4_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_testing_8_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_testing_8_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_testing_8_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_testing_8_fly4_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_testing_9_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_testing_9_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_testing_9_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_testing_9_fly4_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_training_1_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_training_1_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_training_1_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_training_1_fly4_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_training_2_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_training_2_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_training_2_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_training_2_fly4_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_training_3_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_training_3_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_training_3_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_training_3_fly4_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_training_4_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_training_4_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_training_4_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_training_4_fly4_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_training_5_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_training_5_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_training_5_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_training_5_fly4_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_training_6_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_training_6_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_training_6_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/RMS_calculations/updated_july_18_batch_1_training_6_fly4_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_testing_1/july_18_batch_1_testing_1_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_testing_1/july_18_batch_1_testing_1_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_testing_1/july_18_batch_1_testing_1_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_testing_1/july_18_batch_1_testing_1_fly4_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_testing_2/july_18_batch_1_testing_2_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_testing_2/july_18_batch_1_testing_2_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_testing_2/july_18_batch_1_testing_2_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_testing_2/july_18_batch_1_testing_2_fly4_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_testing_3/july_18_batch_1_testing_3_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_testing_3/july_18_batch_1_testing_3_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_testing_3/july_18_batch_1_testing_3_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_testing_3/july_18_batch_1_testing_3_fly4_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_testing_4/july_18_batch_1_testing_4_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_testing_4/july_18_batch_1_testing_4_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_testing_4/july_18_batch_1_testing_4_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_testing_4/july_18_batch_1_testing_4_fly4_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_testing_5/july_18_batch_1_testing_5_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_testing_5/july_18_batch_1_testing_5_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_testing_5/july_18_batch_1_testing_5_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_testing_5/july_18_batch_1_testing_5_fly4_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_testing_6/july_18_batch_1_testing_6_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_testing_6/july_18_batch_1_testing_6_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_testing_6/july_18_batch_1_testing_6_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_testing_6/july_18_batch_1_testing_6_fly4_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_testing_7/july_18_batch_1_testing_7_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_testing_7/july_18_batch_1_testing_7_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_testing_7/july_18_batch_1_testing_7_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_testing_7/july_18_batch_1_testing_7_fly4_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_testing_8/july_18_batch_1_testing_8_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_testing_8/july_18_batch_1_testing_8_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_testing_8/july_18_batch_1_testing_8_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_testing_8/july_18_batch_1_testing_8_fly4_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_testing_9/july_18_batch_1_testing_9_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_testing_9/july_18_batch_1_testing_9_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_testing_9/july_18_batch_1_testing_9_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_testing_9/july_18_batch_1_testing_9_fly4_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_training_1/july_18_batch_1_training_1_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_training_1/july_18_batch_1_training_1_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_training_1/july_18_batch_1_training_1_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_training_1/july_18_batch_1_training_1_fly4_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_training_2/july_18_batch_1_training_2_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_training_2/july_18_batch_1_training_2_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_training_2/july_18_batch_1_training_2_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_training_2/july_18_batch_1_training_2_fly4_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_training_3/july_18_batch_1_training_3_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_training_3/july_18_batch_1_training_3_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_training_3/july_18_batch_1_training_3_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_training_3/july_18_batch_1_training_3_fly4_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_training_4/july_18_batch_1_training_4_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_training_4/july_18_batch_1_training_4_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_training_4/july_18_batch_1_training_4_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_training_4/july_18_batch_1_training_4_fly4_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_training_5/july_18_batch_1_training_5_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_training_5/july_18_batch_1_training_5_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_training_5/july_18_batch_1_training_5_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_training_5/july_18_batch_1_training_5_fly4_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_training_6/july_18_batch_1_training_6_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_training_6/july_18_batch_1_training_6_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_training_6/july_18_batch_1_training_6_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_1_rig_3/july_18_batch_1_training_6/july_18_batch_1_training_6_fly4_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_2_rig_3/RMS_calculations/updated_july_18_batch_2_testing_1_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_2_rig_3/RMS_calculations/updated_july_18_batch_2_testing_1_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_2_rig_3/RMS_calculations/updated_july_18_batch_2_testing_1_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_2_rig_3/RMS_calculations/updated_july_18_batch_2_testing_2_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_2_rig_3/RMS_calculations/updated_july_18_batch_2_testing_2_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_2_rig_3/RMS_calculations/updated_july_18_batch_2_testing_2_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_2_rig_3/RMS_calculations/updated_july_18_batch_2_testing_3_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_2_rig_3/RMS_calculations/updated_july_18_batch_2_testing_3_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_2_rig_3/RMS_calculations/updated_july_18_batch_2_testing_3_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_2_rig_3/RMS_calculations/updated_july_18_batch_2_testing_4_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_2_rig_3/RMS_calculations/updated_july_18_batch_2_testing_4_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_2_rig_3/RMS_calculations/updated_july_18_batch_2_testing_4_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_2_rig_3/RMS_calculations/updated_july_18_batch_2_testing_5_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_2_rig_3/RMS_calculations/updated_july_18_batch_2_testing_5_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_2_rig_3/RMS_calculations/updated_july_18_batch_2_testing_5_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_2_rig_3/RMS_calculations/updated_july_18_batch_2_testing_6_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_2_rig_3/RMS_calculations/updated_july_18_batch_2_testing_6_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_2_rig_3/RMS_calculations/updated_july_18_batch_2_testing_6_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_2_rig_3/RMS_calculations/updated_july_18_batch_2_testing_7_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_2_rig_3/RMS_calculations/updated_july_18_batch_2_testing_7_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_2_rig_3/RMS_calculations/updated_july_18_batch_2_testing_7_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_2_rig_3/RMS_calculations/updated_july_18_batch_2_testing_8_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_2_rig_3/RMS_calculations/updated_july_18_batch_2_testing_8_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_2_rig_3/RMS_calculations/updated_july_18_batch_2_testing_8_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_2_rig_3/RMS_calculations/updated_july_18_batch_2_testing_9_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_2_rig_3/RMS_calculations/updated_july_18_batch_2_testing_9_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_2_rig_3/RMS_calculations/updated_july_18_batch_2_testing_9_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_2_rig_3/RMS_calculations/updated_july_18_batch_2_training_1_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_2_rig_3/RMS_calculations/updated_july_18_batch_2_training_1_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_2_rig_3/RMS_calculations/updated_july_18_batch_2_training_1_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_2_rig_3/RMS_calculations/updated_july_18_batch_2_training_2_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_2_rig_3/RMS_calculations/updated_july_18_batch_2_training_2_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_2_rig_3/RMS_calculations/updated_july_18_batch_2_training_2_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_2_rig_3/RMS_calculations/updated_july_18_batch_2_training_3_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_2_rig_3/RMS_calculations/updated_july_18_batch_2_training_3_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_2_rig_3/RMS_calculations/updated_july_18_batch_2_training_3_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_2_rig_3/RMS_calculations/updated_july_18_batch_2_training_4_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_2_rig_3/RMS_calculations/updated_july_18_batch_2_training_4_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_2_rig_3/RMS_calculations/updated_july_18_batch_2_training_4_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_2_rig_3/RMS_calculations/updated_july_18_batch_2_training_5_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_2_rig_3/RMS_calculations/updated_july_18_batch_2_training_5_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_2_rig_3/RMS_calculations/updated_july_18_batch_2_training_5_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_2_rig_3/RMS_calculations/updated_july_18_batch_2_training_6_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_2_rig_3/RMS_calculations/updated_july_18_batch_2_training_6_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_2_rig_3/RMS_calculations/updated_july_18_batch_2_training_6_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_2_rig_3/july_18_batch_2_testing_1/july_18_batch_2_testing_1_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_2_rig_3/july_18_batch_2_testing_1/july_18_batch_2_testing_1_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_2_rig_3/july_18_batch_2_testing_1/july_18_batch_2_testing_1_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_2_rig_3/july_18_batch_2_testing_2/july_18_batch_2_testing_2_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_2_rig_3/july_18_batch_2_testing_2/july_18_batch_2_testing_2_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_2_rig_3/july_18_batch_2_testing_2/july_18_batch_2_testing_2_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_2_rig_3/july_18_batch_2_testing_3/july_18_batch_2_testing_3_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_2_rig_3/july_18_batch_2_testing_3/july_18_batch_2_testing_3_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_2_rig_3/july_18_batch_2_testing_3/july_18_batch_2_testing_3_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_2_rig_3/july_18_batch_2_testing_4/july_18_batch_2_testing_4_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_2_rig_3/july_18_batch_2_testing_4/july_18_batch_2_testing_4_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_2_rig_3/july_18_batch_2_testing_4/july_18_batch_2_testing_4_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_2_rig_3/july_18_batch_2_testing_5/july_18_batch_2_testing_5_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_2_rig_3/july_18_batch_2_testing_5/july_18_batch_2_testing_5_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_2_rig_3/july_18_batch_2_testing_5/july_18_batch_2_testing_5_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_2_rig_3/july_18_batch_2_testing_6/july_18_batch_2_testing_6_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_2_rig_3/july_18_batch_2_testing_6/july_18_batch_2_testing_6_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_2_rig_3/july_18_batch_2_testing_6/july_18_batch_2_testing_6_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_2_rig_3/july_18_batch_2_testing_7/july_18_batch_2_testing_7_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_2_rig_3/july_18_batch_2_testing_7/july_18_batch_2_testing_7_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_2_rig_3/july_18_batch_2_testing_7/july_18_batch_2_testing_7_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_2_rig_3/july_18_batch_2_testing_8/july_18_batch_2_testing_8_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_2_rig_3/july_18_batch_2_testing_8/july_18_batch_2_testing_8_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_2_rig_3/july_18_batch_2_testing_8/july_18_batch_2_testing_8_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_2_rig_3/july_18_batch_2_testing_9/july_18_batch_2_testing_9_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_2_rig_3/july_18_batch_2_testing_9/july_18_batch_2_testing_9_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_2_rig_3/july_18_batch_2_testing_9/july_18_batch_2_testing_9_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_2_rig_3/july_18_batch_2_training_1/july_18_batch_2_training_1_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_2_rig_3/july_18_batch_2_training_1/july_18_batch_2_training_1_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_2_rig_3/july_18_batch_2_training_1/july_18_batch_2_training_1_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_2_rig_3/july_18_batch_2_training_2/july_18_batch_2_training_2_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_2_rig_3/july_18_batch_2_training_2/july_18_batch_2_training_2_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_2_rig_3/july_18_batch_2_training_2/july_18_batch_2_training_2_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_2_rig_3/july_18_batch_2_training_3/july_18_batch_2_training_3_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_2_rig_3/july_18_batch_2_training_3/july_18_batch_2_training_3_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_2_rig_3/july_18_batch_2_training_3/july_18_batch_2_training_3_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_2_rig_3/july_18_batch_2_training_4/july_18_batch_2_training_4_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_2_rig_3/july_18_batch_2_training_4/july_18_batch_2_training_4_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_2_rig_3/july_18_batch_2_training_4/july_18_batch_2_training_4_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_2_rig_3/july_18_batch_2_training_5/july_18_batch_2_training_5_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_2_rig_3/july_18_batch_2_training_5/july_18_batch_2_training_5_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_2_rig_3/july_18_batch_2_training_5/july_18_batch_2_training_5_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_2_rig_3/july_18_batch_2_training_6/july_18_batch_2_training_6_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_2_rig_3/july_18_batch_2_training_6/july_18_batch_2_training_6_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_18_batch_2_rig_3/july_18_batch_2_training_6/july_18_batch_2_training_6_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_1_rig_3/RMS_calculations/updated_july_19_batch_1_testing_1_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_1_rig_3/RMS_calculations/updated_july_19_batch_1_testing_1_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_1_rig_3/RMS_calculations/updated_july_19_batch_1_testing_2_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_1_rig_3/RMS_calculations/updated_july_19_batch_1_testing_2_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_1_rig_3/RMS_calculations/updated_july_19_batch_1_testing_3_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_1_rig_3/RMS_calculations/updated_july_19_batch_1_testing_3_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_1_rig_3/RMS_calculations/updated_july_19_batch_1_testing_4_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_1_rig_3/RMS_calculations/updated_july_19_batch_1_testing_4_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_1_rig_3/RMS_calculations/updated_july_19_batch_1_testing_5_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_1_rig_3/RMS_calculations/updated_july_19_batch_1_testing_5_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_1_rig_3/RMS_calculations/updated_july_19_batch_1_testing_6_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_1_rig_3/RMS_calculations/updated_july_19_batch_1_testing_6_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_1_rig_3/RMS_calculations/updated_july_19_batch_1_testing_7_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_1_rig_3/RMS_calculations/updated_july_19_batch_1_testing_7_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_1_rig_3/RMS_calculations/updated_july_19_batch_1_testing_8_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_1_rig_3/RMS_calculations/updated_july_19_batch_1_testing_8_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_1_rig_3/RMS_calculations/updated_july_19_batch_1_testing_9_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_1_rig_3/RMS_calculations/updated_july_19_batch_1_testing_9_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_1_rig_3/RMS_calculations/updated_july_19_batch_1_training_1_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_1_rig_3/RMS_calculations/updated_july_19_batch_1_training_1_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_1_rig_3/RMS_calculations/updated_july_19_batch_1_training_2_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_1_rig_3/RMS_calculations/updated_july_19_batch_1_training_2_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_1_rig_3/RMS_calculations/updated_july_19_batch_1_training_3_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_1_rig_3/RMS_calculations/updated_july_19_batch_1_training_3_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_1_rig_3/RMS_calculations/updated_july_19_batch_1_training_4_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_1_rig_3/RMS_calculations/updated_july_19_batch_1_training_4_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_1_rig_3/RMS_calculations/updated_july_19_batch_1_training_5_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_1_rig_3/RMS_calculations/updated_july_19_batch_1_training_5_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_1_rig_3/RMS_calculations/updated_july_19_batch_1_training_6_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_1_rig_3/RMS_calculations/updated_july_19_batch_1_training_6_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_1_rig_3/july_19_batch_1_testing_1/july_19_batch_1_testing_1_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_1_rig_3/july_19_batch_1_testing_1/july_19_batch_1_testing_1_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_1_rig_3/july_19_batch_1_testing_2/july_19_batch_1_testing_2_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_1_rig_3/july_19_batch_1_testing_2/july_19_batch_1_testing_2_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_1_rig_3/july_19_batch_1_testing_3/july_19_batch_1_testing_3_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_1_rig_3/july_19_batch_1_testing_3/july_19_batch_1_testing_3_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_1_rig_3/july_19_batch_1_testing_4/july_19_batch_1_testing_4_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_1_rig_3/july_19_batch_1_testing_4/july_19_batch_1_testing_4_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_1_rig_3/july_19_batch_1_testing_5/july_19_batch_1_testing_5_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_1_rig_3/july_19_batch_1_testing_5/july_19_batch_1_testing_5_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_1_rig_3/july_19_batch_1_testing_6/july_19_batch_1_testing_6_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_1_rig_3/july_19_batch_1_testing_6/july_19_batch_1_testing_6_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_1_rig_3/july_19_batch_1_testing_7/july_19_batch_1_testing_7_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_1_rig_3/july_19_batch_1_testing_7/july_19_batch_1_testing_7_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_1_rig_3/july_19_batch_1_testing_8/july_19_batch_1_testing_8_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_1_rig_3/july_19_batch_1_testing_8/july_19_batch_1_testing_8_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_1_rig_3/july_19_batch_1_testing_9/july_19_batch_1_testing_9_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_1_rig_3/july_19_batch_1_testing_9/july_19_batch_1_testing_9_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_1_rig_3/july_19_batch_1_training_1/july_19_batch_1_training_1_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_1_rig_3/july_19_batch_1_training_1/july_19_batch_1_training_1_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_1_rig_3/july_19_batch_1_training_2/july_19_batch_1_training_2_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_1_rig_3/july_19_batch_1_training_2/july_19_batch_1_training_2_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_1_rig_3/july_19_batch_1_training_3/july_19_batch_1_training_3_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_1_rig_3/july_19_batch_1_training_3/july_19_batch_1_training_3_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_1_rig_3/july_19_batch_1_training_4/july_19_batch_1_training_4_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_1_rig_3/july_19_batch_1_training_4/july_19_batch_1_training_4_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_1_rig_3/july_19_batch_1_training_5/july_19_batch_1_training_5_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_1_rig_3/july_19_batch_1_training_5/july_19_batch_1_training_5_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_1_rig_3/july_19_batch_1_training_6/july_19_batch_1_training_6_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_1_rig_3/july_19_batch_1_training_6/july_19_batch_1_training_6_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_2_rig_3/RMS_calculations/updated_july_19_batch_2_testing_1_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_2_rig_3/RMS_calculations/updated_july_19_batch_2_testing_1_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_2_rig_3/RMS_calculations/updated_july_19_batch_2_testing_1_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_2_rig_3/RMS_calculations/updated_july_19_batch_2_testing_2_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_2_rig_3/RMS_calculations/updated_july_19_batch_2_testing_2_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_2_rig_3/RMS_calculations/updated_july_19_batch_2_testing_2_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_2_rig_3/RMS_calculations/updated_july_19_batch_2_testing_3_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_2_rig_3/RMS_calculations/updated_july_19_batch_2_testing_3_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_2_rig_3/RMS_calculations/updated_july_19_batch_2_testing_3_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_2_rig_3/RMS_calculations/updated_july_19_batch_2_testing_4_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_2_rig_3/RMS_calculations/updated_july_19_batch_2_testing_4_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_2_rig_3/RMS_calculations/updated_july_19_batch_2_testing_4_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_2_rig_3/RMS_calculations/updated_july_19_batch_2_testing_5_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_2_rig_3/RMS_calculations/updated_july_19_batch_2_testing_5_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_2_rig_3/RMS_calculations/updated_july_19_batch_2_testing_5_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_2_rig_3/RMS_calculations/updated_july_19_batch_2_testing_6_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_2_rig_3/RMS_calculations/updated_july_19_batch_2_testing_6_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_2_rig_3/RMS_calculations/updated_july_19_batch_2_testing_6_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_2_rig_3/RMS_calculations/updated_july_19_batch_2_testing_7_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_2_rig_3/RMS_calculations/updated_july_19_batch_2_testing_7_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_2_rig_3/RMS_calculations/updated_july_19_batch_2_testing_7_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_2_rig_3/RMS_calculations/updated_july_19_batch_2_testing_8_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_2_rig_3/RMS_calculations/updated_july_19_batch_2_testing_8_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_2_rig_3/RMS_calculations/updated_july_19_batch_2_testing_8_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_2_rig_3/RMS_calculations/updated_july_19_batch_2_testing_9_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_2_rig_3/RMS_calculations/updated_july_19_batch_2_testing_9_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_2_rig_3/RMS_calculations/updated_july_19_batch_2_testing_9_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_2_rig_3/RMS_calculations/updated_july_19_batch_2_training_1_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_2_rig_3/RMS_calculations/updated_july_19_batch_2_training_1_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_2_rig_3/RMS_calculations/updated_july_19_batch_2_training_1_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_2_rig_3/RMS_calculations/updated_july_19_batch_2_training_2_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_2_rig_3/RMS_calculations/updated_july_19_batch_2_training_2_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_2_rig_3/RMS_calculations/updated_july_19_batch_2_training_2_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_2_rig_3/RMS_calculations/updated_july_19_batch_2_training_3_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_2_rig_3/RMS_calculations/updated_july_19_batch_2_training_3_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_2_rig_3/RMS_calculations/updated_july_19_batch_2_training_3_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_2_rig_3/RMS_calculations/updated_july_19_batch_2_training_4_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_2_rig_3/RMS_calculations/updated_july_19_batch_2_training_4_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_2_rig_3/RMS_calculations/updated_july_19_batch_2_training_4_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_2_rig_3/RMS_calculations/updated_july_19_batch_2_training_5_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_2_rig_3/RMS_calculations/updated_july_19_batch_2_training_5_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_2_rig_3/RMS_calculations/updated_july_19_batch_2_training_5_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_2_rig_3/RMS_calculations/updated_july_19_batch_2_training_6_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_2_rig_3/RMS_calculations/updated_july_19_batch_2_training_6_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_2_rig_3/RMS_calculations/updated_july_19_batch_2_training_6_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_2_rig_3/july_19_batch_2_testing_1/july_19_batch_2_testing_1_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_2_rig_3/july_19_batch_2_testing_1/july_19_batch_2_testing_1_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_2_rig_3/july_19_batch_2_testing_1/july_19_batch_2_testing_1_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_2_rig_3/july_19_batch_2_testing_2/july_19_batch_2_testing_2_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_2_rig_3/july_19_batch_2_testing_2/july_19_batch_2_testing_2_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_2_rig_3/july_19_batch_2_testing_2/july_19_batch_2_testing_2_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_2_rig_3/july_19_batch_2_testing_3/july_19_batch_2_testing_3_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_2_rig_3/july_19_batch_2_testing_3/july_19_batch_2_testing_3_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_2_rig_3/july_19_batch_2_testing_3/july_19_batch_2_testing_3_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_2_rig_3/july_19_batch_2_testing_4/july_19_batch_2_testing_4_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_2_rig_3/july_19_batch_2_testing_4/july_19_batch_2_testing_4_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_2_rig_3/july_19_batch_2_testing_4/july_19_batch_2_testing_4_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_2_rig_3/july_19_batch_2_testing_5/july_19_batch_2_testing_5_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_2_rig_3/july_19_batch_2_testing_5/july_19_batch_2_testing_5_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_2_rig_3/july_19_batch_2_testing_5/july_19_batch_2_testing_5_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_2_rig_3/july_19_batch_2_testing_6/july_19_batch_2_testing_6_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_2_rig_3/july_19_batch_2_testing_6/july_19_batch_2_testing_6_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_2_rig_3/july_19_batch_2_testing_6/july_19_batch_2_testing_6_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_2_rig_3/july_19_batch_2_testing_7/july_19_batch_2_testing_7_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_2_rig_3/july_19_batch_2_testing_7/july_19_batch_2_testing_7_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_2_rig_3/july_19_batch_2_testing_7/july_19_batch_2_testing_7_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_2_rig_3/july_19_batch_2_testing_8/july_19_batch_2_testing_8_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_2_rig_3/july_19_batch_2_testing_8/july_19_batch_2_testing_8_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_2_rig_3/july_19_batch_2_testing_8/july_19_batch_2_testing_8_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_2_rig_3/july_19_batch_2_testing_9/july_19_batch_2_testing_9_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_2_rig_3/july_19_batch_2_testing_9/july_19_batch_2_testing_9_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_2_rig_3/july_19_batch_2_testing_9/july_19_batch_2_testing_9_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_2_rig_3/july_19_batch_2_training_1/july_19_batch_2_training_1_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_2_rig_3/july_19_batch_2_training_1/july_19_batch_2_training_1_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_2_rig_3/july_19_batch_2_training_1/july_19_batch_2_training_1_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_2_rig_3/july_19_batch_2_training_2/july_19_batch_2_training_2_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_2_rig_3/july_19_batch_2_training_2/july_19_batch_2_training_2_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_2_rig_3/july_19_batch_2_training_2/july_19_batch_2_training_2_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_2_rig_3/july_19_batch_2_training_3/july_19_batch_2_training_3_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_2_rig_3/july_19_batch_2_training_3/july_19_batch_2_training_3_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_2_rig_3/july_19_batch_2_training_3/july_19_batch_2_training_3_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_2_rig_3/july_19_batch_2_training_4/july_19_batch_2_training_4_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_2_rig_3/july_19_batch_2_training_4/july_19_batch_2_training_4_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_2_rig_3/july_19_batch_2_training_4/july_19_batch_2_training_4_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_2_rig_3/july_19_batch_2_training_5/july_19_batch_2_training_5_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_2_rig_3/july_19_batch_2_training_5/july_19_batch_2_training_5_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_2_rig_3/july_19_batch_2_training_5/july_19_batch_2_training_5_fly3_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_2_rig_3/july_19_batch_2_training_6/july_19_batch_2_training_6_fly1_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_2_rig_3/july_19_batch_2_training_6/july_19_batch_2_training_6_fly2_distances.parquet
  would drop ['angle_ARB_deg', 'angle_centered_deg', 'angle_centered_pct'] from /securedstorage/DATAsec/cole/Data-secured-New/EB-Training-24-1/july_19_batch_2_rig_3/july_19_batch_2_training_6/july_19_batch_2_training_6_fly3_distances.parquet
would update 948 of 948 tables

```

---

## Pre-merge review fixes (docs/config accuracy, no logic changes) — 2026-07-21

Applied the final whole-branch review's 4 pre-merge fixes on
`feature/rig3-mirrored-anchor`.

**Status:** all 4 fixes landed, no production logic changed.

**Commit:** `7c9bf15` — "docs(rig3): correct pre-merge documentation and drop
dead ANCHOR_X/Y knob" (5 files changed, 72 insertions, 33 deletions).

**Fixes landed:**
1. `config/example.env` — deleted `ANCHOR_X`/`ANCHOR_Y` lines (dead knob,
   `Settings.anchor_x/anchor_y` already removed). Not gitignored (negated
   by `!config/example.env`), committed normally.
2. `scripts/pipeline/recompute_rig3_angles.py` — rewrote module docstring:
   now correctly states this is a diagnostic/belt-and-braces tool, since
   `_ensure_angle_percentages` recomputes unconditionally, `_process_fly_angles`
   is unwired from the live pipeline, and 0/474 real rig_3 parquets carry
   `angle_multiplier`. `--dry-run` framed as primary use.
3. `src/fbpipe/utils/rig_anchor.py` + `tests/test_rig_anchor.py` — reworded
   the underscore-requirement rationale from "silently mirrored correct
   rig_2 data" (implying an observed bug) to defensive/latent, since
   `resolve_anchor` is never called on `Results/Figures`. Regex unchanged.
4. `docs/superpowers/plans/2026-07-21-rig3-mirrored-anchor.md` Task 5 —
   corrected Step 7's rig_2 safety rationale (was: `_process_fly_angles`
   short-circuit, which doesn't apply and in fact rewrote 60 files + changed
   `angle_centered_deg` when tested manually; now: `resolve_anchor` returns
   the identical `DEFAULT_ANCHOR` for rig_2). Added a top-of-Task-5 note that
   the invalidation pass is not required for today's data. `docs/` is
   gitignored but this file was already tracked, so `git check-ignore`
   reports it as not ignored; committed with `git add -f` per instructions
   anyway.

**Suite:** `conda run -n yolo-env python -m pytest tests/ -q` →
697 passed, 1 xfailed (matches expected baseline, unchanged).

**Not committed / gitignored:** nothing — both `config/example.env` and the
docs plan file were already git-tracked, so neither was actually blocked by
`.gitignore` in practice.

**Out of scope (recorded, not fixed):**
`scripts/analysis/abdomen_per_tracking.py:72` hardcodes a third,
unthreaded copy of `ANCHOR_X, ANCHOR_Y = 1079.0, 540.0`. Standalone CLI,
not wired into the pipeline (no data corruption), but reports inverted
angles for rig_3 and disagrees with `DEFAULT_ANCHOR` by 1px.

**Concurrent user work observed, left untouched:** the working tree had
substantial staged/unstaged changes from concurrent user work (e.g.
`.gitignore`, `src/fbpipe/odor_constants.py`, several new scripts/tests
like `remake_reaction_matrix_split.py`, `test_3oct_odor_remap.py`,
`test_model_score_lookup.py`) — none of these were touched; only the 5
files named in the review scope were modified and committed via explicit
pathspec.
