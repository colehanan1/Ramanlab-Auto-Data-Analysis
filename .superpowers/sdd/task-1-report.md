# Task 1 Report: Shared per-rig anchor helper

## Status: DONE

## Files created

- `src/fbpipe/utils/rig_anchor.py`
- `tests/test_rig_anchor.py`

Both created verbatim from the brief (`.superpowers/sdd/task-1-brief.md`), no
deviations.

## Pre-flight check

Before touching anything, confirmed the working tree matched the described
state (substantial pre-existing staged/unstaged/untracked unrelated work) and
that neither target file existed yet:

```
$ git status
On branch feature/rig3-mirrored-anchor
Changes to be committed:
	new file:   scripts/check_all_light_stimulus.py
	new file:   scripts/check_light_stimulus.py
	modified:   src/fbpipe/pipeline.py
	new file:   src/fbpipe/steps/check_light_stimulus.py
	new file:   src/fbpipe/utils/light_stimulus.py
	modified:   src/fbpipe/utils/trial_metadata.py
Changes not staged for commit:
	modified:   scripts/analysis/envelope_visuals.py
	modified:   scripts/analysis/reaction_matrix_training_vs_control.py
	modified:   scripts/pipeline/run_workflows.py
	modified:   src/fbpipe/config.py
	modified:   src/fbpipe/utils/video_writer.py
	modified:   tests/test_freeze_pipeline.py
	modified:   tests/test_reaction_matrix_presentations.py
Untracked files:
	debug_yolo_july17/
	scripts/analysis/randompanel_trial_position.py
	scripts/analysis/reaction_matrix_specific_flies_vs_control.py
	tests/test_randompanel_trial_position.py
	tests/test_reaction_matrix_exclude_flies.py
	tests/test_video_writer.py

$ ls src/fbpipe/utils/rig_anchor.py tests/test_rig_anchor.py
ls: cannot access 'src/fbpipe/utils/rig_anchor.py': No such file or directory
ls: cannot access 'tests/test_rig_anchor.py': No such file or directory
```

Confirmed: neither target file existed, and the unrelated tree state matched
what the scope guard described. Nothing was moved, deleted, or "tidied."

(Note: this report file itself, `.superpowers/sdd/task-1-report.md`, already
existed on disk with stale content from an unrelated earlier plan's "Task 1"
— a config freeze.data/freeze.figures task. That content has been replaced
below with this task's report; the earlier content is preserved in git
history if needed.)

## Step 1: Write the failing test

Created `tests/test_rig_anchor.py` with the exact content from the brief
(6 top-level test functions, one of them parametrized with 5 cases — 11
collected test items total).

## Step 2: Run test to verify it fails (RED)

Command:

```
conda run -n yolo-env python -m pytest tests/test_rig_anchor.py -v
```

Result: collection error, exactly as predicted by the brief.

```
ERROR collecting tests/test_rig_anchor.py
ImportError while importing test module '.../tests/test_rig_anchor.py'.
Traceback:
  ...
  tests/test_rig_anchor.py:12: in <module>
    from fbpipe.utils.rig_anchor import (
E   ModuleNotFoundError: No module named 'fbpipe.utils.rig_anchor'
...
=========================== 14 warnings, 1 error in 0.08s =========================
```

RED confirmed for the exact stated reason: `ModuleNotFoundError: No module
named 'fbpipe.utils.rig_anchor'`.

## Step 3: Minimal implementation

Created `src/fbpipe/utils/rig_anchor.py` with the exact content from the
brief:

- `DEFAULT_ANCHOR: Tuple[float, float] = (1080.0, 540.0)`
- `MIRRORED_ANCHOR: Tuple[float, float] = (0.0, 540.0)`
- `MIRRORED_RIGS = frozenset({"rig_3"})`
- `rig_token(path)` — regex `rig_?(\d+)` scanned over `Path(path).parts`
  reversed (deepest-first), returns normalized `"rig_{n}"` or `None`.
- `resolve_anchor(path)` — `MIRRORED_ANCHOR` if `rig_token(path) in
  MIRRORED_RIGS` else `DEFAULT_ANCHOR`.

## Step 4: Run test to verify it passes (GREEN)

Command:

```
conda run -n yolo-env python -m pytest tests/test_rig_anchor.py -v
```

Result:

```
tests/test_rig_anchor.py::test_default_and_mirrored_anchor_values PASSED [  9%]
tests/test_rig_anchor.py::test_rig_token[/data/EB-Training-24-1/july_17_batch_2_rig_3/trial_1-rig_3] PASSED [ 18%]
tests/test_rig_anchor.py::test_rig_token[/data/EB-Training-24-1/july_17_batch_2_rig_2/trial_1-rig_2] PASSED [ 27%]
tests/test_rig_anchor.py::test_rig_token[/data/3Oct-Training-24-0.1/july_20_batch_1_rig_3-rig_3] PASSED [ 36%]
tests/test_rig_anchor.py::test_rig_token[relative/july_18_batch_1_rig_3/x.parquet-rig_3] PASSED [ 45%]
tests/test_rig_anchor.py::test_rig_token[/data/no_rig_here/trial_1-None] PASSED [ 54%]
tests/test_rig_anchor.py::test_rig_3_resolves_to_mirrored_anchor PASSED  [ 63%]
tests/test_rig_anchor.py::test_rig_2_resolves_to_default_anchor PASSED   [ 72%]
tests/test_rig_anchor.py::test_unknown_path_falls_back_to_default PASSED [ 81%]
tests/test_rig_anchor.py::test_accepts_path_objects PASSED               [ 90%]
tests/test_rig_anchor.py::test_deepest_rig_token_wins PASSED             [100%]
======================= 11 passed, 14 warnings in 0.02s ========================
```

GREEN: 11 passed (not 9 — see "surprises" below).

## Vacuity check (mutation test)

Per the task's testing note, I temporarily flipped `MIRRORED_RIGS` to an
empty `frozenset()` in `src/fbpipe/utils/rig_anchor.py` (via a small Python
one-liner, then re-ran pytest) to confirm the mirrored-anchor tests actually
bite rather than passing vacuously:

```
$ python3 -c "... replace MIRRORED_RIGS = frozenset({'rig_3'}) with frozenset() ..."
$ conda run -n yolo-env python -m pytest tests/test_rig_anchor.py -v
tests/test_rig_anchor.py::test_default_and_mirrored_anchor_values PASSED [  9%]
tests/test_rig_anchor.py::test_rig_token[...rig_3] PASSED [ 18%]
tests/test_rig_anchor.py::test_rig_token[...rig_2] PASSED [ 27%]
tests/test_rig_anchor.py::test_rig_token[...rig_3] PASSED [ 36%]
tests/test_rig_anchor.py::test_rig_token[...rig_3] PASSED [ 45%]
tests/test_rig_anchor.py::test_rig_token[...None] PASSED [ 54%]
tests/test_rig_anchor.py::test_rig_3_resolves_to_mirrored_anchor FAILED  [ 63%]
tests/test_rig_anchor.py::test_rig_2_resolves_to_default_anchor PASSED   [ 72%]
tests/test_rig_anchor.py::test_unknown_path_falls_back_to_default PASSED [ 81%]
tests/test_rig_anchor.py::test_accepts_path_objects FAILED               [ 90%]
tests/test_rig_anchor.py::test_deepest_rig_token_wins FAILED             [100%]
=================== 3 failed, 8 passed, 14 warnings in 0.05s ===================
```

Observation: exactly the 3 tests that exercise `resolve_anchor` on a rig_3
path (`test_rig_3_resolves_to_mirrored_anchor`, `test_accepts_path_objects`,
`test_deepest_rig_token_wins`) FAILED when the mirrored-rig set was emptied,
while `rig_token`-only tests and the default-anchor/unknown-path tests still
passed (correctly — they don't depend on `MIRRORED_RIGS` membership, or in
the unknown-path case, correctly fall back regardless). This is exactly the
expected fault signature: the tests genuinely assert on mirrored-vs-default
behavior and are not vacuous.

Reverted the experiment immediately via `Edit` (restoring
`MIRRORED_RIGS = frozenset({"rig_3"})`), confirmed the file was byte-identical
to the original via `cat -n`, and re-ran the suite to confirm all 11 pass
again:

```
======================= 11 passed, 14 warnings in 0.02s ========================
```

Note: `git checkout -- src/fbpipe/utils/rig_anchor.py` could not be used to
revert (the file was untracked at that point, so git had no committed
baseline to restore from — `error: pathspec ... did not match any file(s)
known to git`). Reverted manually via `Edit` instead, then verified the
resulting file against the original text.

## Commit

Two commit attempts:

1. **First attempt failed harmlessly.** I ran
   `git commit -- <path1> <path2> -m "<message>"` — putting `-m` *after* the
   `--` pathspec separator. Git treated `-m` and the message text themselves
   as pathspecs (`error: pathspec '-m' did not match any file(s) known to
   git`), so the commit did not happen. No damage: `git log -1` afterward
   showed the branch's pre-existing HEAD (`41f33f9`, a "docs: implementation
   plan" commit already on this branch from earlier planning work, not
   authored by this task).
2. **Second attempt succeeded**, with `-m` before `--`:

```
$ git commit -m "$(cat <<'EOF'
feat(rig_anchor): resolve the geometric anchor per rig

rig_3 is physically mirrored, so its anchor sits on the left edge. Unknown
and rig-less paths fall back to the existing right-edge anchor.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)" -- src/fbpipe/utils/rig_anchor.py tests/test_rig_anchor.py

[feature/rig3-mirrored-anchor c339b95] feat(rig_anchor): resolve the geometric anchor per rig
 2 files changed, 111 insertions(+)
 create mode 100644 src/fbpipe/utils/rig_anchor.py
 create mode 100644 tests/test_rig_anchor.py
```

**Commit SHA: `c339b95e9c77f5295ab6ae7a733abacf11add6a2`**

Verified via `git show --stat HEAD` that the commit contains exactly these
two files and nothing else (`2 files changed, 111 insertions(+)`).

I deliberately used the `git commit -m "..." -- <path1> <path2>` form (commit
restricted to an explicit pathspec) rather than a bare `git commit`, because
the repo's index already had six unrelated files staged
(`scripts/check_all_light_stimulus.py`, `scripts/check_light_stimulus.py`,
`src/fbpipe/pipeline.py`, `src/fbpipe/steps/check_light_stimulus.py`,
`src/fbpipe/utils/light_stimulus.py`, `src/fbpipe/utils/trial_metadata.py`)
from before this task started. A bare `git commit` after `git add
src/fbpipe/utils/rig_anchor.py tests/test_rig_anchor.py` would have swept all
of those pre-existing staged changes into my commit — exactly what the scope
guard forbids. Restricting the commit to an explicit pathspec commits only
the named files' changes regardless of what else is sitting in the index.

## Post-commit verification

`git status` after the commit is byte-for-byte identical to the pre-task
snapshot except that `src/fbpipe/utils/rig_anchor.py` and
`tests/test_rig_anchor.py` have moved from "untracked" to "committed" (no
longer appear anywhere in status output). All six pre-existing staged files
remain staged, untouched. All seven pre-existing unstaged modifications
remain unstaged, untouched. All remaining untracked files
(`debug_yolo_july17/`, `scripts/analysis/randompanel_trial_position.py`,
`scripts/analysis/reaction_matrix_specific_flies_vs_control.py`,
`tests/test_randompanel_trial_position.py`,
`tests/test_reaction_matrix_exclude_flies.py`, `tests/test_video_writer.py`)
remain untracked, untouched.

Final sanity re-run of the test suite post-commit: `11 passed`.

## What surprised me

1. **Test count mismatch in the brief.** Step 4 of the brief says "Expected:
   PASS (9 tests)" but the test file as written (verbatim from the brief
   itself) actually collects 11 test cases once pytest expands the 5-case
   `@pytest.mark.parametrize` on `test_rig_token` (6 standalone functions + 5
   parametrized cases = 11, minus the 1 counted twice = 11 total). This is a
   harmless inaccuracy in the brief's count, not a functional issue — every
   test passes and the vacuity check confirms they're meaningful.
2. **`git commit -- <paths> -m "..."` argument order matters and fails.**
   Putting `-m "message"` after the `--` pathspec separator causes git to
   interpret the flag and message as literal pathspecs, which don't match
   any file and abort the commit with a `pathspec did not match` error. Easy
   to get backwards; caught it because I verified `git show --stat HEAD`
   immediately after, which showed an unrelated pre-existing commit still at
   HEAD rather than my new commit.
3. **`git checkout -- <file>` cannot revert an untracked file.** During the
   vacuity check I needed to undo my `MIRRORED_RIGS = frozenset()` mutation
   before the file was ever committed, so there was no git-tracked baseline
   to check out from. Had to revert manually with `Edit` and verify by eye
   instead of relying on git.
4. **This report file already existed with unrelated content.** Before
   writing this report, `.superpowers/sdd/task-1-report.md` was found to
   already contain a full report for a different "Task 1" (a
   freeze.data/freeze.figures config feature) from an earlier, unrelated
   plan run in this same `.superpowers/sdd/` directory. Read it first, then
   overwrote it with this task's report as instructed.

No other deviations from the brief. The implementation, both anchor tuples,
`MIRRORED_RIGS`, `rig_token`, and `resolve_anchor` are exactly as specified.
