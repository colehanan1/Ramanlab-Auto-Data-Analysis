# Task 3 Report: Thread the anchor through `compose_videos_rms`

Note: this report file previously held the report for an unrelated Task 3
from a different plan. This overwrites it with the report for the CURRENT
plan's Task 3 (rig_3 mirrored-anchor: `src/fbpipe/steps/compose_videos_rms.py`),
per `.superpowers/sdd/task-3-brief.md`.

Branch: `feature/rig3-mirrored-anchor`
Commit: `07e74f8` — "feat(compose_videos_rms): resolve the angle anchor per rig"
Status: **DONE** (with one honestly-reported, then closed, coverage gap — see Mutation 2 below)

## Files touched (scope guard honored)

- Modified: `src/fbpipe/steps/compose_videos_rms.py`
- Created: `tests/test_rig3_anchor_compose.py`
- Nothing else was staged into the commit; `git commit ... -- <paths>` used an
  explicit pathspec so the unrelated already-staged/unstaged user work in the
  repo was never touched (see "Concurrent user work" below).

## Steps 1-2: RED

Wrote `tests/test_rig3_anchor_compose.py` using the brief's Step 1 code
verbatim (3 tests).

```
conda run -n yolo-env python -m pytest tests/test_rig3_anchor_compose.py -v
```

All 3 FAILED, exactly as predicted:
```
E       TypeError: compute_angle_deg_at_point2() takes 1 positional argument but 2 were given
FAILED tests/test_rig3_anchor_compose.py::test_default_argument_preserves_current_values
FAILED tests/test_rig3_anchor_compose.py::test_mirrored_anchor_is_supplement_of_default
FAILED tests/test_rig3_anchor_compose.py::test_agrees_with_envelope_combined_implementation
3 failed, 14 warnings in 0.57s
```

## Steps 3-5: implementation (verbatim per brief)

In `src/fbpipe/steps/compose_videos_rms.py`:

1. Added `from ..utils.rig_anchor import resolve_anchor` import.
2. `compute_angle_deg_at_point2(df, anchor=None)` — added the parameter;
   replaced `ux, uy = (ANCHOR_X - p2x), (ANCHOR_Y - p2y)` with
   `ax, ay = (ANCHOR_X, ANCHOR_Y) if anchor is None else anchor` then
   `ux, uy = (ax - p2x), (ay - p2y)`.
3. `find_fly_reference_angle(csvs_raw, trimmed_min=None, anchor=None)` —
   added the parameter, threaded to its internal
   `compute_angle_deg_at_point2(df, anchor)` call.
4. `compute_fly_max_abs_centered(csvs_raw, ref_angle, anchor=None)` — same
   treatment. Confirmed dead code (`grep -rn "compute_fly_max_abs_centered"`
   → only its own definition, no callers anywhere in the repo), updated only
   for internal consistency per the brief's note.
5. `_process_fly_angles(fly_dir)` — added `anchor = resolve_anchor(fly_dir)`
   once, before computing the reference angle; passed `anchor=anchor` into
   `find_fly_reference_angle`; passed `anchor` into the per-file-loop call to
   `compute_angle_deg_at_point2`.

## Step 4/6: GREEN

```
conda run -n yolo-env python -m pytest tests/test_rig3_anchor_compose.py -v
```
`3 passed, 14 warnings in 0.50s`

```
conda run -n yolo-env python -m pytest tests/test_compose_videos_rms.py tests/test_compose_videos_rms_parquet.py tests/test_rig3_anchor_compose.py -v
```
`25 passed` (18 in `test_compose_videos_rms.py` + 4 in
`test_compose_videos_rms_parquet.py` + 3 new) — no pre-existing test
regressed.

## Mutation testing (per task instructions)

### Mutation 1 — `compute_angle_deg_at_point2` ignores its `anchor` argument

Hardcoded `ax, ay = (ANCHOR_X, ANCHOR_Y)` regardless of the argument. Ran
`tests/test_rig3_anchor_compose.py`:
```
test_default_argument_preserves_current_values PASSED
test_mirrored_anchor_is_supplement_of_default FAILED
test_agrees_with_envelope_combined_implementation FAILED
2 failed, 1 passed
```
CAUGHT, as required. Reverted; confirmed `git diff` returned to the intended
state before continuing.

### Mutation 2 — resolved anchor to the measurement, DEFAULT to `find_fly_reference_angle`

Changed the `_process_fly_angles` call site so
`find_fly_reference_angle(csv_paths, trimmed_min=trimmed_min)` omitted
`anchor=` (falling back to `None` → `DEFAULT_ANCHOR` inside that function),
while the per-file loop still passed the resolved `anchor` to
`compute_angle_deg_at_point2`. Ran the full regression set at that point
(brief's original 3 tests + the 22 pre-existing compose tests, 25 total):

```
25 passed, 14 warnings in 0.58s
```

**Not caught.** Nothing in the brief's prescribed test file, nor in the
pre-existing `test_compose_videos_rms.py` / `test_compose_videos_rms_parquet.py`
suites, exercises `_process_fly_angles` end-to-end against a rig-tokened
directory, so a reference-vs-measurement anchor mismatch inside that
function was invisible. Reported this honestly per the task's instruction
rather than papering over it.

Because Mutation 3 (below) is explicitly the same *class* of bug that
survived Task 2's initial test pass (envelope_combined's commit 7385012 had
to add a spy test after review found the gap), I added two more tests to
`tests/test_rig3_anchor_compose.py`, ported directly from
`tests/test_rig3_anchor_angles.py`'s
`test_reference_and_measurement_share_one_anchor` and
`test_non_rig3_fly_resolves_to_default_anchor`:

- `test_reference_and_measurement_share_one_anchor` — spies on
  `compute_angle_deg_at_point2` inside the `compose_videos_rms` module,
  runs `_process_fly_angles` on a `.../rig_3` fly_dir fixture, and asserts
  every call recorded during the run (the one inside
  `find_fly_reference_angle` *and* the one in the per-file loop) used
  `MIRRORED_ANCHOR` — i.e. `set(seen) == {MIRRORED_ANCHOR}`, which fails if
  the two call sites ever disagree.
- `test_non_rig3_fly_resolves_to_default_anchor` (parametrized over a
  `rig_2`-named dir and a no-rig-token dir) — same spy, asserts
  `set(seen) == {DEFAULT_ANCHOR}`.

Re-verified this closes the Mutation 2 gap: reapplied the Mutation 2 edit
and reran the (now 6-test) file. `test_reference_and_measurement_share_one_anchor`
FAILED with `anchors disagreed: {(1080.0, 540.0), (0.0, 540.0)}` — the rig_3
fixture's reference call fell back to `DEFAULT_ANCHOR` while the measurement
call still used `MIRRORED_ANCHOR`, exactly the mismatch Mutation 2
introduces. Reverted immediately after confirming.

### Mutation 3 — hardcode `anchor = MIRRORED_ANCHOR` for every fly in `_process_fly_angles`

Replaced `anchor = resolve_anchor(fly_dir)` with a hardcoded
`anchor = MIRRORED_ANCHOR`. Ran `tests/test_rig3_anchor_compose.py` (now 6
tests, after adding the two spy tests above):

```
test_default_argument_preserves_current_values PASSED
test_mirrored_anchor_is_supplement_of_default PASSED
test_agrees_with_envelope_combined_implementation PASSED
test_reference_and_measurement_share_one_anchor PASSED   (rig_3 fixture — correctly MIRRORED_ANCHOR both places)
test_non_rig3_fly_resolves_to_default_anchor[july_17_batch_2_rig_2] FAILED
test_non_rig3_fly_resolves_to_default_anchor[july_18_batch_1] FAILED
2 failed, 4 passed, 14 warnings in 0.58s
```

CAUGHT, as required ("something must FAIL; do not repeat it here"). Note
this mutation would **not** have been caught by the brief's original 3-test
file alone — none of those 3 tests call `_process_fly_angles` or
`resolve_anchor`. The two spy tests added above were necessary to close this
gap. Reverted immediately after confirming.

### Aside: `tmp_path`-name substring trap (recurrence of a Task 2 gotcha)

While writing `test_non_rig3_fly_resolves_to_default_anchor`, the
`july_18_batch_1` (no-rig-token) parametrization initially FAILED with
*correct* code, because pytest's `tmp_path` fixture derives its directory
name from the test node id, and the node id
`...resolves_to_default_anchor[july_18_batch_1]` contains the substring
`rig3` (from "non_**rig3**_fly"). `rig_token`'s regex `rig_?(\d+)` matched
that substring in an ancestor path component, so the fly_dir was
misdetected as rig_3 and resolved to `MIRRORED_ANCHOR`. This is the exact
gotcha Task 2 documented and fixed in `tests/test_rig3_anchor_angles.py`
(using its own `tempfile.TemporaryDirectory` instead of `tmp_path`). Applied
the identical fix here. This was a real RED encountered and diagnosed during
development, not a hypothetical — confirms the test suite isn't vacuous.

## Final verification

`git diff -- src/fbpipe/steps/compose_videos_rms.py` after all mutation
experiments were reverted showed exactly the 5 intended edits (import,
`compute_angle_deg_at_point2` signature+body, `find_fly_reference_angle`
signature+call, `compute_fly_max_abs_centered` signature+call,
`_process_fly_angles` anchor resolution + two call sites) — no leftover
mutation code, confirmed by re-reading the diff.

Final full run:
```
conda run -n yolo-env python -m pytest tests/test_compose_videos_rms.py tests/test_compose_videos_rms_parquet.py tests/test_rig3_anchor_compose.py -q
28 passed, 14 warnings in 0.56s
```
(18 in `test_compose_videos_rms.py` + 4 in `test_compose_videos_rms_parquet.py`
+ 6 in the new/expanded `test_rig3_anchor_compose.py` = 28, confirmed against
`--collect-only` as well). All pre-existing tests still pass; zero
regressions.

## Commit

```
07e74f8 feat(compose_videos_rms): resolve the angle anchor per rig
 2 files changed, 163 insertions(+), 8 deletions(-)
 create mode 100644 tests/test_rig3_anchor_compose.py
```
Staged and committed via explicit pathspec:
`git commit -m "..." -- src/fbpipe/steps/compose_videos_rms.py tests/test_rig3_anchor_compose.py`

## Concurrent user work observed (left untouched)

Compared to the git status snapshot given at task start, by the time this
task finished the following additional changes had appeared in the working
tree — none related to this task, none read or modified:

- `src/fbpipe/odor_constants.py` — newly modified (not in the original
  snapshot).
- `debug_yolo_july17/` — new untracked directory.
- `tests/test_3oct_odor_remap.py`, `tests/test_3oct_trained_label.py`,
  `tests/test_model_score_lookup.py` — new untracked test files.

The pre-existing staged changes from the original snapshot (e.g.
`scripts/check_all_light_stimulus.py`, `src/fbpipe/pipeline.py`,
`src/fbpipe/steps/check_light_stimulus.py`,
`src/fbpipe/utils/light_stimulus.py`, `src/fbpipe/utils/trial_metadata.py`)
remained staged in the index exactly as found and were excluded from this
task's commit via the explicit pathspec — never added with `-A`/`.`.

## Summary

The brief's implementation was applied verbatim. The brief's prescribed
3-test file passes, but by itself would not have caught two of the three
required mutations (2 and 3) — both involve `_process_fly_angles`, which the
brief's Step-1 test never exercises. I added two spy tests (ported from
Task 2's already-established pattern in `tests/test_rig3_anchor_angles.py`)
that close this gap: `_process_fly_angles` now has direct coverage proving
it resolves the anchor from `fly_dir` (not a hardcoded value) and uses the
*same* anchor for both the reference angle and the measurement. Final
suite: 28/28 passed, 0 regressions.

---

# Review-fix report: Findings 1 & 2 (rig_token ancestor false-positive; vacuous drift guard)

Branch: `feature/rig3-mirrored-anchor`

## Finding 1 — `rig_token` ancestor false-positive (PRODUCTION BUG)

**Fix**: `src/fbpipe/utils/rig_anchor.py` — changed `_RIG_RE` from
`re.compile(r"rig_?(\d+)", re.IGNORECASE)` to
`re.compile(r"rig_(\d+)", re.IGNORECASE)` (underscore now required between
"rig" and the digit). Updated the comment above the regex and the module
docstring to explain why (an ancestor named `..._excl_rig3` must not mirror;
verified every real rig directory on disk under
`/home/ramanlab/Documents/cole/Data/flys_New` uses the underscore form).

**Tests** (`tests/test_rig_anchor.py`):
- Added two `rig_token` cases proving `rig3`/`Rig3` (no underscore) now
  return `None` (documented, intentional behaviour change from Task 1).
- Added `test_resolve_anchor_ignores_no_underscore_ancestor_mentions`,
  parametrized on the real poisoner names from the finding:
  - `EB-Training-24-1_excl_rig3/july_18_batch_1` → `DEFAULT_ANCHOR`
  - `EB-Training-24-1_excl_rig3_and_july17b2rig2/july_18_batch_1` → `DEFAULT_ANCHOR`
  - `EB-Training-24-1_excl_rig3/july_18_batch_1_rig_3` → `MIRRORED_ANCHOR` (deepest real token wins)
  - `july_17_batch_2_rig_3` → `MIRRORED_ANCHOR` (unchanged)

**Mutation experiment** (revert `_RIG_RE` to old `rig_?(\d+)`):
```
conda run -n yolo-env python -m pytest tests/test_rig_anchor.py -v
```
Result: `4 failed, 13 passed` — both no-underscore `rig_token` cases and both
`DEFAULT_ANCHOR`-poisoner cases failed (the two `MIRRORED_ANCHOR` cases
still passed, as expected since they contain genuine underscored tokens).
Reverted mutation; reran → `17 passed`.

## Finding 2 — vacuous cross-implementation drift guard

**Fix**: `tests/test_rig3_anchor_compose.py` — added `_non_collinear_frame()`
(eye `(500,540)`, proboscis `(550,500)`, i.e. `y_class1 != y_class0`) as a
fixture used ONLY by `test_agrees_with_envelope_combined_implementation`.
Left `_frame()` (collinear, y=540 for everything) untouched and still used
by `test_mirrored_anchor_is_supplement_of_default`, which genuinely requires
`p2y == anchor_y` for the `180 - angle` supplement identity.

**Mutation experiment** (in `src/fbpipe/steps/compose_videos_rms.py`,
`compute_angle_deg_at_point2`): changed
`vx, vy = (p3x - p2x), (p3y - p2y)` to
`vx, vy = (p3x - p2x), (p3y - p2y) * 3.0`.
```
conda run -n yolo-env python -m pytest tests/test_rig3_anchor_compose.py -v
```
Result: `1 failed, 5 passed` — only `test_agrees_with_envelope_combined_implementation`
FAILED; `test_mirrored_anchor_is_supplement_of_default` correctly stayed
green (collinear fixture, vy=0, unaffected by the ×3.0 scale — confirms the
caveat in the finding). Reverted the mutation; `git diff -- src/fbpipe/steps/compose_videos_rms.py`
confirmed byte-identical to the pre-mutation state (empty diff).

## Full suite after both fixes

```
conda run -n yolo-env python -m pytest tests/test_rig_anchor.py tests/test_rig3_anchor_compose.py tests/test_rig3_anchor_angles.py -v
```
`30 passed, 14 warnings in 0.58s` — zero regressions across all three files.

## Scope guard honored

Only `src/fbpipe/utils/rig_anchor.py`, `tests/test_rig_anchor.py`, and
`tests/test_rig3_anchor_compose.py` were modified and committed (explicit
pathspec, no `-A`/`.`). `src/fbpipe/steps/compose_videos_rms.py` was mutated
only transiently for the Finding 2 verification and restored exactly
(confirmed clean `git diff`).

## Concurrent user work observed (left untouched)

At task start, `git status` already showed substantial unrelated
staged/unstaged/untracked changes (`.gitignore`, `scripts/analysis/*`,
`src/fbpipe/config.py`, `src/fbpipe/odor_constants.py`,
`src/fbpipe/pipeline.py`, `src/fbpipe/steps/check_light_stimulus.py`,
`src/fbpipe/utils/light_stimulus.py`, `src/fbpipe/utils/trial_metadata.py`,
`src/fbpipe/utils/video_writer.py`, several new/modified test files under
`tests/`, and new scripts under `scripts/`). None of it was read, staged, or
modified by this task.
