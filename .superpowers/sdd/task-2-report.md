# Task 2 Report: Thread the anchor through `envelope_combined`

Note: this report file previously held the report for a different plan's
"Task 2" (freeze cache module, commit `88b1df0`). This overwrites it with the
report for the CURRENT plan's Task 2 (rig_3 mirrored-anchor angle threading),
per the fresh `task-2-brief.md` on branch `feature/rig3-mirrored-anchor`.

Status: DONE
Commit: `94cc2a9a8a51c189a2b144f5640c145b789069d9`

## Step 1-2: RED

Created `tests/test_rig3_anchor_angles.py` with the three tests from the brief
(`test_default_anchor_matches_module_constant_behaviour`,
`test_mirrored_anchor_is_supplement_of_default`,
`test_extension_toward_anchor_reads_as_zero_degrees`).

Command:
```
conda run -n yolo-env python -m pytest tests/test_rig3_anchor_angles.py -v
```

Output (all 3 failed, exactly as predicted by the brief):
```
tests/test_rig3_anchor_angles.py::test_default_anchor_matches_module_constant_behaviour FAILED
tests/test_rig3_anchor_angles.py::test_mirrored_anchor_is_supplement_of_default FAILED
tests/test_rig3_anchor_angles.py::test_extension_toward_anchor_reads_as_zero_degrees FAILED
...
E       TypeError: _compute_angle_deg() takes 1 positional argument but 2 were given
=========================== 3 failed, 14 warnings in 0.29s ===========================
```

## Step 3-4: GREEN (anchor param added to `_compute_angle_deg`)

Changed the signature in `scripts/analysis/envelope_combined.py`:

```python
def _compute_angle_deg(
    df: pd.DataFrame, anchor: tuple[float, float] | None = None
) -> pd.Series:
```

and the vector setup:

```python
    ax, ay = (ANCHOR_X, ANCHOR_Y) if anchor is None else anchor
    ux = ax - p2x
    uy = ay - p2y
```

Re-ran:
```
conda run -n yolo-env python -m pytest tests/test_rig3_anchor_angles.py -v
```
Result: `3 passed, 14 warnings in 0.24s`

## Step 5: Threaded anchor through the three callers

- `_find_reference_angle(csv_paths, anchor=None)` — passes `anchor` into its
  `_compute_angle_deg` call.
- `_fly_max_centered(csv_paths, reference_angle, anchor=None)` — same.
- `_ensure_angle_percentages(fly_dir, suffix_globs)` — signature unchanged;
  resolves `anchor = resolve_anchor(fly_dir)` once and passes it to
  `_find_reference_angle`, `_fly_max_centered`, and the per-file
  `_compute_angle_deg` call in its loop.
- Added `from fbpipe.utils.rig_anchor import resolve_anchor` next to the other
  `fbpipe.utils` imports (after the `fly_type` import, before `tables`).

Verified via `grep` that every call site of `_compute_angle_deg`,
`_find_reference_angle`, and `_fly_max_centered` in the file now passes
`anchor` explicitly — no stray un-threaded call sites left, and no other
callers of these three functions exist elsewhere in the file or repo tests.

## Step 6-7: Baseline-consistency test + full run

Appended `test_reference_and_measurement_share_one_anchor` (the spy-based test
from the brief) to `tests/test_rig3_anchor_angles.py`.

```
conda run -n yolo-env python -m pytest tests/test_rig3_anchor_angles.py -v
```
Result: `4 passed, 14 warnings in 0.28s`

## Mutation testing (tests must bite)

**Mutation A** — made `_compute_angle_deg` ignore its `anchor` argument
entirely:
```python
    ax, ay = (ANCHOR_X, ANCHOR_Y)  # MUTATION: anchor arg ignored
```
Result:
```
test_default_anchor_matches_module_constant_behaviour PASSED
test_mirrored_anchor_is_supplement_of_default FAILED
test_extension_toward_anchor_reads_as_zero_degrees FAILED
test_reference_and_measurement_share_one_anchor PASSED
2 failed, 2 passed, 14 warnings in 0.31s
```
The two mirrored-anchor tests correctly caught this (they're the only ones
that actually exercise a non-default anchor and check its numeric effect).
The default-anchor test and the spy test cannot catch this mutation by
design — the spy only checks *which* anchor value was passed through the call
chain, not whether `_compute_angle_deg` honored it. Reverted immediately after
observing the failure.

**Mutation B** — broke anchor agreement in `_ensure_angle_percentages` by
passing `None` to `_find_reference_angle` while still passing the resolved
`anchor` to `_fly_max_centered` and the measurement:
```python
    reference = _find_reference_angle(csv_paths, None)  # MUTATION: anchor mismatch
```
Result:
```
test_default_anchor_matches_module_constant_behaviour PASSED
test_mirrored_anchor_is_supplement_of_default PASSED
test_extension_toward_anchor_reads_as_zero_degrees PASSED
test_reference_and_measurement_share_one_anchor FAILED
1 failed, 3 passed, 14 warnings in 0.27s
```
Failure detail:
```
AssertionError: anchors disagreed: {None, (0.0, 540.0)}
assert {None, (0.0, 540.0)} == {(0.0, 540.0)}
```
This is exactly the failure mode the brief called "the single most dangerous
failure mode in this task" (reference and measurement computed against
different anchors) — caught cleanly by the spy-based test. Reverted
immediately after observing the failure.

Confirmed no mutation artifacts remain (`grep -n "MUTATION" scripts/analysis/envelope_combined.py`
returns nothing) and re-ran the full test file post-revert: `4 passed, 14 warnings in 0.28s`.

## Regression check (read-only, no files touched)

Ran the pre-existing `tests/test_envelope_combined.py` suite (28 tests) to
confirm the added optional parameters don't disturb existing callers/behavior:
```
28 passed, 208 warnings in 4.75s
```

## Final diff

```
 scripts/analysis/envelope_combined.py | 31 +++++++++++++------
 tests/test_rig3_anchor_angles.py      | 88 +++++++++++++++++++++++++++++++
 2 files changed, 109 insertions(+), 10 deletions(-)
```

## Commit

```
git add scripts/analysis/envelope_combined.py tests/test_rig3_anchor_angles.py
git commit -m "feat(envelope_combined): resolve the angle anchor per rig

_ensure_angle_percentages resolves the anchor from fly_dir and threads it
through the reference angle, the fly-max scale, and the measurement so all
three agree. Default argument preserves existing behaviour." \
  -- scripts/analysis/envelope_combined.py tests/test_rig3_anchor_angles.py
```
SHA: `94cc2a9a8a51c189a2b144f5640c145b789069d9`

Verified before and after the commit (via `git status`) that the pre-existing
staged files (`scripts/check_all_light_stimulus.py`,
`scripts/check_light_stimulus.py`, `src/fbpipe/pipeline.py`,
`src/fbpipe/steps/check_light_stimulus.py`,
`src/fbpipe/utils/light_stimulus.py`, `src/fbpipe/utils/trial_metadata.py`)
and all unstaged/untracked files were untouched by this commit — the
pathspec-scoped `git commit -- <paths>` left the rest of the index and working
tree exactly where it was found at the start of this task. Did not use
`git add -A`/`git add .`, `git clean`, `git stash`, or `git checkout .` at any
point.

## Surprising notes

- The brief's line numbers (`:1493`, `:1588`, `:1629`, `:1670`) matched the
  file exactly before editing; they drift by +1 for everything after the new
  `resolve_anchor` import line once that import is added. Cosmetic only — all
  edits were matched by content (via `Read`/`grep`), not by line number, so
  nothing was misapplied.
- No other call sites of `_compute_angle_deg`, `_find_reference_angle`, or
  `_fly_max_centered` exist anywhere else in `envelope_combined.py` or in any
  test file, so there was nothing else in this module needing an anchor
  update.
- `test_envelope_combined.py`'s 28 pre-existing tests all pass unmodified,
  confirming the default-argument change is fully backward-compatible for
  every existing (non-rig_3) caller.
- The repo's `git status` at the start of this task showed substantial
  unrelated staged and unstaged work (light-stimulus feature files, config/
  video_writer changes, several untracked analysis scripts and test files).
  All of it was left completely untouched — confirmed identical in `git
  status` output before and after this task's commit.

---

# Task 2 Review-Fix Report: Test coverage gaps

Status: DONE
Commit: `7385012` (`test(rig3-anchor): cover wrong-rig hardcode and unsigned-angle regressions`)
Scope: only `tests/test_rig3_anchor_angles.py` modified. `scripts/analysis/envelope_combined.py`
had two temporary mutations applied and exactly reverted (`git diff` confirmed empty both times).

## Finding 1 (Important): no test pinned the anchor to the CORRECT rig

The pre-existing `test_reference_and_measurement_share_one_anchor` only asserted
that the three call sites inside `_ensure_angle_percentages` agree with each
other (`set(seen) == {MIRRORED_ANCHOR}` for a rig_3 dir) -- it never checked
that a *non*-rig_3 fly resolves to `DEFAULT_ANCHOR`. A reviewer's mutation
hardcoding `anchor = (0.0, 540.0)` for every fly (silently inverting every
non-rig_3 angle) passed all 32 existing tests.

Added `test_non_rig3_fly_resolves_to_default_anchor`, parametrized over a
rig_2 directory (`july_17_batch_2_rig_2`) and a directory with no rig token at
all (`july_18_batch_1`), using the same spy technique (monkeypatch
`_compute_angle_deg` to record its `anchor` arg, monkeypatch
`_trial_csv_candidates` to return one fixture parquet) -- both must resolve to
`DEFAULT_ANCHOR`.

Note: initially wrote this test using pytest's `tmp_path` fixture. It failed
for the no-rig-token case even against correct code, because pytest derives
`tmp_path`'s directory name from the test's own node id, and the function
name `test_non_rig3_fly_resolves_to_default_anchor` contains the literal
substring "rig3" (inside "non_rig3_fly"). `rig_token()` scans every ancestor
path component (not just the leaf), so for the no-rig-token leaf directory it
walked up and matched "rig3" in the *fixture's own scaffolding directory*,
misreading it as a rig_3 trial. Fixed by building an isolated
`tempfile.TemporaryDirectory(prefix="anchor_test_")` for this test instead of
relying on `tmp_path`, keeping the fixture's naming out of the path under
test. The rig_2 parametrized case never showed this because its own leaf
directory name matches first (`rig_token` returns on first match scanning
from the leaf upward), so the ancestor is never consulted -- but the
no-rig-token case falls through to ancestors, which is exactly where the
pollution lived.

### Mutation experiment 1

Applied (temporary edit to `scripts/analysis/envelope_combined.py`,
`_ensure_angle_percentages`):
```python
anchor = (0.0, 540.0)   # was: anchor = resolve_anchor(fly_dir)
```

Command:
```
conda run -n yolo-env python -m pytest tests/test_rig3_anchor_angles.py tests/test_rig_anchor.py -v
```

Output (relevant lines):
```
tests/test_rig3_anchor_angles.py::test_non_rig3_fly_resolves_to_default_anchor[july_17_batch_2_rig_2] FAILED
tests/test_rig3_anchor_angles.py::test_non_rig3_fly_resolves_to_default_anchor[july_18_batch_1] FAILED
...
AssertionError: anchors disagreed: {(0.0, 540.0)}
assert {(0.0, 540.0)} == {(1080.0, 540.0)}
================== 2 failed, 16 passed, 14 warnings in 0.33s ===================
```
New test dies as designed. All other 16 tests still pass (confirming the
mutation is narrow and the new test is what catches it).

Reverted the mutation exactly:
```python
anchor = resolve_anchor(fly_dir)
```
`git diff scripts/analysis/envelope_combined.py` -> empty.

Green run after revert:
```
======================= 18 passed, 14 warnings in 0.29s ========================
```

## Finding 2 (Minor): "angle must stay unsigned" had no dedicated guard

The shared fixture (`_frame()`) is collinear (eye and proboscis share
y=540), so `cross` is exactly `-0.0` and a signed-vs-unsigned `arctan2` gives
the same result there -- no existing test could distinguish them.

Added `test_off_axis_angle_stays_unsigned_for_both_anchors`: proboscis at
`(550, 480)` vs eye at `(500, 540)` (`y_class1 != y_class0`, genuinely
off-axis), asserting `0.0 <= angle <= 180.0` for both `DEFAULT_ANCHOR` and
`MIRRORED_ANCHOR`.

### Mutation experiment 2

Applied (temporary edit to `scripts/analysis/envelope_combined.py`,
`_compute_angle_deg`):
```python
ang = np.arctan2(cross[valid], dot[valid])   # was: np.arctan2(np.abs(cross[valid]), dot[valid])
```

Command:
```
conda run -n yolo-env python -m pytest tests/test_rig3_anchor_angles.py tests/test_rig_anchor.py -v
```

Output (relevant lines):
```
tests/test_rig3_anchor_angles.py::test_mirrored_anchor_is_supplement_of_default FAILED
tests/test_rig3_anchor_angles.py::test_extension_toward_anchor_reads_as_zero_degrees FAILED
tests/test_rig3_anchor_angles.py::test_off_axis_angle_stays_unsigned_for_both_anchors FAILED
================== 3 failed, 15 passed, 14 warnings in 0.34s ===================
```
New test dies as designed (along with two pre-existing tests, confirming the
mutation is a genuine, broad-reaching regression).

Reverted the mutation exactly:
```python
ang = np.arctan2(np.abs(cross[valid]), dot[valid])
```
`git diff scripts/analysis/envelope_combined.py` -> empty.

Green run after revert:
```
======================= 18 passed, 14 warnings in 0.28s ========================
```

## Final verification

Command:
```
conda run -n yolo-env python -m pytest tests/test_rig3_anchor_angles.py tests/test_rig_anchor.py -v
```
Result: `18 passed, 14 warnings in 0.31s` -- 15 pre-existing + 3 new
(`test_non_rig3_fly_resolves_to_default_anchor[july_17_batch_2_rig_2]`,
`test_non_rig3_fly_resolves_to_default_anchor[july_18_batch_1]`,
`test_off_axis_angle_stays_unsigned_for_both_anchors`).

## Scope compliance

- Only `tests/test_rig3_anchor_angles.py` was modified and committed
  (`git commit ... -- tests/test_rig3_anchor_angles.py`, explicit pathspec,
  no `-A`/`.`).
- `scripts/analysis/envelope_combined.py` carries zero net diff (`git diff`
  empty) after both mutation experiments were reverted.
- No `git clean`, `git stash`, or `git checkout .` was run.
- Pre-existing unrelated uncommitted/untracked work (e.g.
  `scripts/analysis/envelope_visuals.py`, `src/fbpipe/config.py`,
  `src/fbpipe/odor_constants.py`, `debug_yolo_july17/`,
  `tests/test_3oct_odor_remap.py`, `tests/test_3oct_trained_label.py`, and
  several other modified/untracked files) was present in `git status` before
  this task started and left completely untouched. Some of these files
  (`odor_constants.py`, `debug_yolo_july17/`, `test_3oct_odor_remap.py`,
  `test_3oct_trained_label.py`) were not present in the very first `git
  status` snapshot at the start of this session, indicating concurrent user
  activity in the repo during this task -- consistent with the brief's note
  that the user may be editing files concurrently. They were not touched.
