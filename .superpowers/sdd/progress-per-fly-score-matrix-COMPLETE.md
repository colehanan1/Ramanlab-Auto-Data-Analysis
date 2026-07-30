# Per-Fly Score Matrix — Progress Ledger

Plan: docs/superpowers/plans/2026-07-14-per-fly-score-matrix.md
Spec: docs/superpowers/specs/2026-07-14-per-fly-score-matrix-design.md
Branch: feature/per-fly-score-matrix (off feature/randompanel-manual-labels)
Skill: superpowers:subagent-driven-development
BASE (branch point): 4d67e1b

## Acceptance gate (user's words)
"i need it to work when i run config_new" → `python3 scripts/pipeline/run_workflows.py
--config config/config_new.yaml --figures-only` must emit the matrix.
config_new.yaml: protocol: v2; datasets RandomPanel-Training-24-10,
RandomPanel-24-0.1, RandomPanel-24-1, EB-Control-24-1, EB-Training-24-1.

## Key decisions (user-approved, do not re-litigate)
- Palette: PRGn purple→green, 7 fixed hexes. Red→green REJECTED (CVD ΔE 4.1 —
  a protanope cannot tell -1 from 5). Chosen ramp measures ΔE 19.5.
- Quiet baseline: score 0 (71% of cells) recedes to pale lavender so reactions pop.
- v2 ONLY. Legacy mean_score_*.png must stay byte-for-byte identical.
- No in-cell numbers, no fly row labels (user removed both).
- Odor labels BELOW the matrix; bar y-axis −1..5; key = "Odor Response Score".

## Pre-flight
- Old ledger was an unrelated COMPLETE effort (yolo-frame-batching) → archived to
  .superpowers/sdd/progress-yolo-frame-batching-COMPLETE.md. Not resumed.
- PLAN BUG FOUND + FIXED: Task 4 Step 4 used `git stash` for the "before" legacy
  figures. stash only reverts uncommitted work → once Tasks 1–3 commit it is a
  no-op and the diff is after-vs-after (vacuous green). Fixed: legacy baseline
  captured at /tmp/legacy_baseline from 4d67e1b BEFORE implementation (11 figures).
- Commit 4d67e1b: "Reaction Boundary" legend relabel (3 sites). 17 tests passed.

## Baselines
- tests/test_score_summary.py → 2 passed (legacy protocol; _ACTIVE_PROTOCOL
  defaults to "legacy" at envelope_visuals.py:67 — these tests never call
  set_protocol, so the v2-only change cannot disturb them).
- tests/conftest.py has autouse _isolate_protocol → new v2 tests cannot leak.
- test_protocol_legacy_regression.py guards envelope_combined ONLY, NOT
  mean_score_*.png → "legacy unchanged" needs its own test (Task 3) + the
  /tmp/legacy_baseline diff (Task 4).
- Legacy figure baseline: /tmp/legacy_baseline (11 PNGs @ 4d67e1b).

## Tasks
- Task 1: complete (commits 4d67e1b..09fc065, review clean — Spec ✅, Approved).
  Palette + _score_cmap(). Reviewer caught an Important: the plan's
  `test_score_cmap_is_a_fixed_map_not_rank_based` was VACUOUS (_score_cmap takes
  no data, so it reduced to "a nullary fn is deterministic"). Replaced with
  `test_each_score_maps_to_its_own_designated_hex`, proven falsifiable by
  mutating ListedColormap construction (the naive "reverse SCORE_COLORS" mutant
  survives — it moves both sides of the comparison together).
  → The real rank-invariance guard is DEFERRED TO TASK 3 (added to plan:
  `test_score_colour_does_not_depend_on_which_scores_are_present`). Task 3's
  reviewer MUST hold us to it — the "never rank-based" constraint lands there.
- Task 2: complete (commit 1676bee, review clean — Spec ✅, Approved, all Minor).
  _per_fly_score_matrix(). Reviewer independently confirmed the implementer's
  honest self-flag: the square-2x2 shape test is weak but its sibling kills the
  transpose mutant, so the suite is sound. NOT fixed (not warranted).
  → CARRIED TO TASK 3: an odor in `columns` absent from the data yields an
    all-NaN column (correct by inspection, UNTESTED). Task 3 passes an
    independently-derived column list, so that path is real. Test added to T3.
- Task 3: complete (commits 1676bee..dce3b7a, opus review — Spec ✅, Approved).
  Implementer found TWO MORE vacuous tests in the plan and strengthened them:
  (i) test_bar_labels_survive_matrix_labelling checked the WRONG AXIS —
      _draw_score_matrix labels the matrix BEFORE the bars label themselves,
      so sharex's shared formatter clobbers the MATRIX's labels, not the bars'.
  (ii) test_score_colour_does_not_depend_on_which_scores_are_present re-called
      _score_cmap() (dataset-invariant by construction) instead of reading the
      rendered artist — never touched the render call site. Now reads
      ax_m.images[0].cmap/.norm live. THE DEFERRED RANK-INVARIANCE GUARD LANDED.
  Also fixed an order-dependent flake (fake plt.close leaked figures).
  Reviewer then found an Important the implementer missed: the trap-2 guard was
  VACUOUS-BY-EMPTINESS — labelbottom=False makes get_xticklabels() return [],
  so the negative assertion passed over an empty list. Mutants B (label ax_m
  directly) and C (delete the whole secondary_xaxis block) survived all 20 tests
  => the 'odor labels uppercased+bold+#1a3a6b' spec line was UNGUARDED.
  Fixed (dce3b7a) with a positive assertion on ax_m.child_axes[0] (NOT fig.axes —
  the secondary axis isn't there). Reviewer re-ran B and C itself + added mutant F
  (drop .upper()); all three die. 21 tests, order-independent.
  Reviewer independently verified legacy byte-for-byte via worktree.
- Task 4: complete (commit de9c9da, controller-run). ALL GATES GREEN:
  * Full suite 511 passed, 1 xfailed, 0 failures.
  * Legacy BYTE-FOR-BYTE IDENTICAL (11 figures) vs /tmp/legacy_baseline captured
    at 4d67e1b before implementation. Real green, not the vacuous stash version.
  * ACCEPTANCE GATE PASSED: run_workflows.py --config config/config_new.yaml
    --figures-only ran clean (0 tracebacks, 214 log lines), --protocol v2 threaded
    through, matrix emitted for all v2 datasets incl. testing-only RandomPanel
    (no NoTargetTrialsError). Figures land in
    Results/New-Opto-Fly-Figures/Matrix-PER-Reactions-Model/score_summary/.
  * Figure matches the approved mockup. NOTE: pipeline output shows "Isoamyl
    Acetate" where the standalone run showed "Apple Cider Vinegar" — that is
    config_new.yaml:90-93 odor_remap (OFM_A delivered isoamyl acetate in this
    cohort), NOT a defect. Verified against the config, same value (2.14).

## Minor findings roll-up (for final review)
- T1: unused numpy/pandas/pytest imports in tests/test_score_matrix.py (inherited
  from brief; Tasks 2–3 use them, so likely self-resolving — recheck at final).
- T1: `zip(SCORES, SCORE_COLORS)` truncates silently on length mismatch (covered
  by test_score_palette_has_one_colour_per_score).
- T2 (implementer self-flagged): test_per_fly_matrix_shape_and_column_alignment is
  weak — fixture is a square 2x2 so a transpose bug survives it. Sibling test
  values_land_in_the_right_cell DOES kill that mutant, so the suite is sound.

## FINAL whole-branch review (opus, 4d67e1b..HEAD)
- Verdict: READY TO MERGE = YES. No Critical, no Important remaining.
- Found a 5th vacuous test (test_score_cmap_missing_is_grey...): claimed "is grey"
  but never checked grey; MISSING_COLOR appeared NOWHERE in the test file. Deleting
  set_bad() survived all 23 tests -> a gap renders TRANSPARENT/white, visually
  identical to score 1's near-white #f2ebf5 => a data gap would read as "no
  reaction". Silent scientific misread. Fixed (aa773d5).
- 6th catch: my own fix prompt was ALSO tautological (read its expectation from
  module.MISSING_COLOR, so the "#ff0000" mutant moved both sides together and
  survived). Fixer self-caught it and added an independent grey assertion.
- 7th: reviewer then found r==g==b still admits "1.0" (white) -> the exact original
  misread. Closed by pinning the literal (controller, verified mutant dies).
- SIX vacuous tests total on this branch; FOUR were authored by me in the plan.
  Recurring shape: the test reads its expectation from the same constant it
  validates, so mutating that constant shifts both sides together.
- Deferred Minors (none block): height_ratios floor/cap untested (impl verified by
  render: 1 fly -> 0.350, 20 -> 1.400); "no in-cell numbers" satisfied by omission,
  unpinned; leftover vacuous assertion at test_score_matrix.py:288 (harmless, the
  sibling positive guard does the real work); plt.close("all") cleans before not
  after; n_fly derived two ways; square-2x2 shape fixture.
STATUS: ALL 4 TASKS COMPLETE + final review clean. Branch NOT merged/pushed.
