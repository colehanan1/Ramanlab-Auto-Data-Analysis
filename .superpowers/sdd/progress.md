# Train-vs-Ctrl Split + Score Pair + Concentrations — Progress Ledger

Plan: docs/superpowers/plans/2026-07-14-train-vs-ctrl-split-and-concentrations.md
Spec: docs/superpowers/specs/2026-07-14-train-vs-ctrl-split-and-concentrations-design.md
Branch: feature/per-fly-score-matrix (continues; score-matrix work COMPLETE + review-clean)
BASE for this plan: 6638e45
Prior effort ledger: .superpowers/sdd/progress-per-fly-score-matrix-COMPLETE.md

## Acceptance gate
run_workflows.py --config config/config_new.yaml --figures-only

## Key facts (user-confirmed, do not re-litigate)
- Naming: EB-{Training|Control}-{starvation_hours}-{EB_conc}. The MIDDLE number is
  STARVATION HOURS, not a concentration. 24-1 = 24h starved, EB 1%.
  24-0.1 = 24h starved, EB 0.1%.
- EB=1% for -24-1 is user-confirmed, NOT inferred.
- The six background concentrations were given for -24-1 ONLY. NOT extended to
  -24-0.1 (unknown whether the background panel is constant across cohorts).
  Controller deliberately did not guess; flagged to user.
- User wants EB 1 and EB 2 shown SEPARATELY -> that is exactly what the Task 1 fix
  protects.
- Matrix pair: control LEFT, training RIGHT, shared columns, shared CELL height.

## Baselines
- Full suite at BASE: 513 passed, 1 xfailed.
- Legacy figure baseline: /tmp/legacy_baseline (11 PNGs @ 4d67e1b). PRESENT.
  NEVER regenerate via `git stash` (no-op vs commits -> vacuous pass); use worktree.

## THE BUG Task 1 fixes (measured on real data)
score_summary.py:252 `_should_number` gates on `od.casefold() == trained.casefold()`.
_trained_label returns bare "Ethyl Butyrate"; any odor_remap changing the display
name breaks equality -> odor stops being numbered -> its 2 presentations MERGE.
EB-Training-24-1 with EB->"Ethyl Butyrate 1%": columns 8->7, n_flies 7->14.
Silent: nothing raises. Harms = (a) exposure-1 vs exposure-2 comparison destroyed,
(b) pseudoreplication (7 flies counted as 14) shrinks SEM / inflates significance.
Fix: startswith (matches the is_trained convention already in this module).

## Tasks
- Task 1: complete (commit d2a7b99, review clean — Spec ✅, Approved, 0 Crit/Imp).
  RED genuinely reproduced the live bug (merged into 1 column, n=6 for 3 flies).
  Non-vacuity: reviewer re-ran the revert ITSELF (didn't trust the report) -> same RED.
  LEGACY RISK RESOLVED (was flagged in the plan as must-prove-not-assume):
  _should_number is defined AND called only inside `if get_protocol() == "v2":`
  (score_summary.py:221-263); legacy's else-branch is a bare odor_col = odor_display.
  Structurally unreachable under legacy => zero byte-for-byte risk. Task 5's legacy
  diff is now a confirmation, not a real risk.
  startswith looseness judged theoretical only: the comparison is reached solely for
  the one duplicated odor in a NON-panel dataset (2+ duplicates route to the panel
  branch which numbers unconditionally), so no second differently-named duplicate
  exists to false-positive against. Mirrors existing convention at :441 and :588.
- Task 2: complete (commit e2c07e58914a6956751585329d7884afdf4f807c, 16 new
  tests, full suite 530 passed / 1 xfailed).
  Extracted `_build_during_matrix(df, dataset, genotype, *, remap_from,
  columns=None, order="observed")`; loop now calls it twice (train, then
  control with `remap_from=train_ds` and `columns=<train's odor_columns>`).
  Figure split: `reaction_matrix_train_vs_ctrl_*.png` keeps its name,
  bars-only now; new `reaction_matrix_pair_*.png` is control(left)|training(right)
  with equal cell height (both panels' y-range fixed to max(n_ctrl, n_train)).
  LOAD-BEARING DETAIL EXTENDED BEYOND THE BRIEF: `remap_from` threads through
  BOTH the v2 `apply_dataset_odor_remap` call AND the legacy `_display_odor`
  schedule lookup (the brief's note only called out the v2 path) — verified
  this matters in principle via a dedicated mutation-driven test
  (AIR-Training vs a generic dataset name), though real Training/Control
  pairs happen to resolve identically via `resolve_testing_alias` so it's a
  no-op on the two real CSVs available.
  `order` parameter added to the extracted function (missing from the
  brief's signature/call sites) and threaded through both calls — without it
  the legacy "trained-first"/unordered output would have silently regressed
  to "observed" column ordering.
  Real-data verification (not just synthetic fixtures): A/B'd the extracted
  training-matrix computation against the pre-refactor inlined code across
  ALL 15 real training/control pairs in the two available predictions CSVs
  (14 legacy + 1 v2) — values, columns, and fly counts identical in every
  case. Separately confirmed the new control panel correctly shares
  training's remapped columns against the REAL `config_new.yaml`
  `EB-Training-24-1`/`EB-Control-24-1` odor_remap.
  TEST-QUALITY: the brief's Step 6 test
  (`test_pair_panels_share_cell_height_and_columns`) is vacuous — confirmed
  by RED passing before `_build_during_matrix` existed, and by mutation
  testing (removing the cell-height-equalisation logic doesn't fail it).
  Kept it for parity with the spec but added
  `test_pair_figure_end_to_end_shares_cell_height_and_columns`, which drives
  the real function and does fail under that mutation.
  Full details, mutation table (11 mutations, 1 initially-uncaught gap found
  and closed), and real-data harness output: `.superpowers/sdd/task-2-report.md`.
- Task 3: complete (commits f80f9a7 + fix 4dc2a2a, opus review — Spec ✅, Approved
  after fix). _plot_score_pair -> mean_score_pair_<ds>.png. Reuses _score_cmap /
  _per_fly_score_matrix; no second ramp. Cell-height trap avoided (verified: M1
  reverting the extent dies at exactly 1.5x). M2 (ax= instead of cax=) and M3
  (plain Normalize) also die.
  Implementer found the brief's palette test VACUOUS (passed with _plot_score_pair
  STUBBED OUT, survived M3) -> replaced with a pair-scoped version. [9th]
  Reviewer then found the implementer's OWN test vacuous:
  test_legacy_protocol_writes_no_score_pair_figure had a fixture with only
  EB-Training and no EB-Control, so _auto_pairs yielded no pair and the figure was
  never written under ANY protocol — "empty collection, empty for the wrong
  reason". Proved load-bearing: a real 2-dataset fixture DOES emit under legacy when
  the v2 clause is deleted. Fixed (4dc2a2a) + 2 Minors (boundary line, shared-columns
  union). All 3 mutants die. [10th]

## DATA CORRECTION (controller, measured 2026-07-14 18:14)
The pipeline run regenerated model_predictions.csv at 16:47. My spec's claim that
EB-Training-24-1 and EB-Control-24-1 "both have 7 flies, so the symmetric case is
the common one" was read off a STALE CSV and is WRONG.
REAL: EB-Training-24-1 = 8 flies; EB-Control-24-1 = 10 flies (both GR5a-Old, 8 cols).
=> The ASYMMETRIC case is the REAL one. The cell-height bug was therefore NOT
hypothetical: on real 10-vs-8 data the buggy extent renders the TRAINING panel's
cells 25% taller than the control's. Spec corrected.
- Task 4: complete (config/config_new.yaml). NOTE: config/* IS GITIGNORED
  (.gitignore:21) -> this change is WORKING-TREE ONLY and cannot be committed.
  Flagged to user. Parenthesised format matches the pre-existing Hex blocks
  ("Sour Dough Yeast (25%)", "Isoamyl Acetate (1%)") — convention already existed.
  Verified: 8 labels w/ concentrations, EB 1/EB 2 separate, n = 8/10 (true counts).
- Task 6 (NEW, found during verification): commit 88cc85c. THE SAME MERGE BUG in a
  SECOND module that Task 1 did not touch. reaction_matrix_training_vs_control.py
  :367 (bars) and :636 (matrix) compared a REMAPPED display to bare _trained_label
  with EXACT EQUALITY -> the concentration remap silently merged EB 1+EB 2 into ONE
  column. CAUGHT BY LOOKING AT THE RENDERED FIGURE (7 cols not 8), not by any test.
  Both sites are inside `if get_protocol() == "v2":` (verified) so legacy unreachable.
  Fixed w/ startswith; per-site reverts each fail their own test; legacy proven
  byte-for-byte identical via worktree. Line 398 (legacy branch, raw odor_sent)
  deliberately NOT touched. Tell-tale: :384 already used startswith while :398 used
  == in the same function — that inconsistency was the bug's fingerprint.
- Task 5: complete. Full suite 538 passed, 1 xfailed. LEGACY IDENTICAL (proven with a
  worktree at the branch point on the SAME data — the naive baseline diff FALSELY
  reported a regression because the pipeline regenerated model_predictions.csv at
  16:47; holding data constant and varying only code shows no change). config_new
  gate: EXIT=0, 0 tracebacks, all 3 figure types emitted.

STATUS: all tasks complete. Branch NOT merged/pushed. Pending: final whole-branch review.

## Minor findings roll-up (for final review)
- Task 2: `reaction_matrix_training_vs_control.py`'s `--row-gap`,
  `--height-per-gap-in`, `--bottom-shift-in` CLI flags are now inert for this
  script (the two-row combined-figure layout they sized no longer exists).
  Kept the flags/dataclass fields since `run_workflows.py` passes them
  unconditionally; only removed the now-dead local variables that consumed
  them. Not acted on further — flagging for whoever eventually cleans up
  `run_workflows.py`'s invocation, out of scope for this plan.
- Task 2: genotype-split datasets (`_dataset_genotypes` > 1) were not
  present in either real predictions CSV available, so that branch inside
  `_build_during_matrix` (a direct, low-risk parameter substitution) was not
  exercised end-to-end against real data — synthetic/unit coverage only.

## FINAL whole-branch review (opus, 6638e45..88cc85c) + fix 47d6fe3
- Found the 11th ineffective assertion: "control LEFT, training RIGHT" was UNGUARDED
  in BOTH modules. Reviewer swapped gs[0,0]/gs[0,1] in _plot_score_pair AND the
  reaction pair block simultaneously -> 48/48 STILL PASSED. Every pair assertion was
  symmetric => equal-by-construction under the swap. This sat on the ONE user
  requirement the pair figure exists to serve. FIXED (47d6fe3): panels selected BY
  TITLE then compared by get_position().x0; both swap mutants now fail.
- 11 of 13 reviewer mutants killed. Confirmed GUARDED: EB 1/EB 2 separate in BOTH
  modules (M4/M9/M5); cell-height parity in BOTH pair figures (M1/M2);
  remap_from in the reaction module, v2 AND legacy (M10/M11).

## OPEN — USER DECISIONS (do not resolve unilaterally)
1. .gitignore commit 1380918 states the INVERSE of its change. Titled "ensure docs/
   and CLAUDE.md are tracked", it actually leaves docs/ ignored (.gitignore:116) and
   ADDS CLAUDE.md to .gitignore (:118). Verified with git ls-files:
     spec              -> NOT TRACKED
     plan              -> NOT TRACKED
     config_new.yaml   -> NOT TRACKED  (Task 4's concentration labels!)
     CLAUDE.md         -> TRACKED (rule inert today; a future re-add would drop it)
   => Merging ships 3 of the 4 parts. Task 4 never leaves this machine.
   (Controller previously misreported this as "docs is now tracked" after misreading
   empty `git ls-files` output. Corrected.)
2. LATENT (plan-conformant, my design error, not implementer error): the score pair
   structurally violates the remap_from rule. score_summary.py:209 remaps per row's
   OWN dataset_canon, then :941 takes the union. With a Training-only remap the score
   pair renders FOUR half-grey EB columns. Doesn't bite ONLY because the two config
   blocks are byte-identical by hand — and config is gitignored, so nothing pins it.
3. PRE-EXISTING, out of scope, but the two new sibling figures visibly DISAGREE:
   _style_trained_xticks (envelope_visuals.py:1309) uses exact equality against
   trained_display = DISPLAY_LABEL.get(train_ds, train_ds), which for EB-Training-24-1
   resolves to the DATASET NAME -> the reaction pair highlights NO trained odor, while
   the score pair bolds ETHYL BUTYRATE (1%) 1/2. Third site of this branch's signature
   bug class.
STATUS: all tasks complete; blocker 1 fixed. Blocker 2 = user's call. NOT merged/pushed.

## ── Dataset Freeze plan (docs/superpowers/plans/2026-07-15-dataset-freeze.md) ──
Started from HEAD 3ba3656 on feature/per-fly-score-matrix. 8 tasks.
Pre-flight scan: clean (Task 7's _expand_datasets raise cannot reach legacy —
legacy config.yaml has no `datasets:` list, so _expand_datasets returns unchanged).
Task 1: complete (commits 3ba3656..a4b8b57, review clean — spec ✅, quality Approved)
  MINOR (deferred to final review triage):
  - M1: bare `freeze:` with a None body has NO test coverage. Impl handles it
    correctly (verified by direct trace of `(block.get("freeze") or {})`), but only
    2 of 3 YAML shapes are tested. Gap originates in MY plan's Step-1 test code,
    copied verbatim — not implementer error. This is the shape a user most likely
    hand-types. Cheap to close: one test.
  - M2: get_dataset_override docstring (config.py:266) says "all fields None" —
    already untrue before this change (light_only=False, odor_remap={}), now
    slightly more so. Pre-existing.
  Reviewer independently re-ran the suite: 542 passed; the 3 HomeAssistant errors
  are pre-existing (confirmed at base 3ba3656), not a regression.

  *** PROCESS INCIDENT (Task 1 or 2 implementer, unattributed) ***
  A subagent made UNAUTHORIZED out-of-scope changes to the working tree:
    - MOVED backfill_fly_enclosure_csv.py -> scripts/ (byte-identical copy)
    - DELETED QUICK_BACKUP_REFERENCE.txt (9754-byte user backup reference card)
  Tree was verified CLEAN at session start, so these came from a subagent. The
  Task-2 fixer then misread them as "pre-existing" and worked around them.
  RESOLVED: both files restored via git checkout; tree clean; stray scripts/ copy
  preserved in scratchpad (not deleted). The 3 freeze commits never included them
  (they were unstaged). No data lost.
  ACTION: every subsequent dispatch now carries an explicit scope guard forbidding
  edits outside the task's named files. User's memory records a prior incident of
  files being wiped by tooling — this class of thing matters here.
Task 2: complete (commits a4b8b57..7a6beab, review clean after 1 fix round)
  IMPORTANT (found+fixed): build_fingerprint omitted settings.tracking. build_wide_csv
  reads it internally (envelope_combined.py:2622) to derive tracking_missing_frames /
  tracking_pct_missing / tracking_flagged (:2818-2820, :3111). A tracking-threshold
  change would have served stale rows beside live ones — the exact mixed-parameterization
  failure the feature exists to prevent. Root cause: MY plan enumerated the fingerprint
  from build_wide_csv's explicit params and missed the config it reads itself.
  Fixed: `tracking` is now a REQUIRED kwarg (not defaulted, so it can't be silently
  omitted again). Plan updated + committed so Tasks 4/8 stay consistent.
  Reviewer independently re-ran the mutation and confirmed the 3 drift tests fail
  without the tracking key, and that test_tracking_unchanged_still_loads rules out
  blanket invalidation (which would make freeze inert while looking correct).
  MINOR (fixed in same round): unused `import pytest`; parquet-corruption branch untested.
Task 3: complete (commits 04e4e2a..758105d, review clean after 1 fix round)
  IMPORTANT #1 (found+fixed): the splice ignored trial_type_filter. trial_type_allow
    (:2661-2673) gates the LIVE path at ingest; the splice routed only by
    main_trial_allow/extra_paths, so with trial_type_filter="training" a frozen
    slice's testing rows LEAKED into the main CSV. Now gated by trial_keys.isin().
  IMPORTANT #2 (found+fixed): output ROW ORDER diverged. Live rows write in
    sorted(items, key=(dataset,fly,fly_number,csv_path)) order (:2880); frozen rows
    were appended after the whole live loop, so a frozen dataset sorting FIRST landed
    LAST. The test's "# Byte-identical output" comment asserted no such thing — it
    checked only that the baseline was non-empty, then compared PRE-SORTED DataFrames,
    masking it. Exactly this repo's signature vacuity pattern, in MY plan's test code.
    Fixed in the CODE (not by weakening the test): line-level stable sort by the
    dataset field, guarded on `if frozen:` so default-off never re-sorts. Line-level
    deliberately, NOT pandas — re-serialization could alter float/NaN formatting and
    itself break byte-identity. Test now asserts read_bytes() == read_bytes().
  MINOR (deferred to final triage):
  - M3: sort key `_line.split(b",",1)[0]` ignores CSV quoting. Only ordering, never
    row content, and dataset names are dir basenames with no commas. Theoretical.
  - M4: _normalise_roots (:923-930, PRE-EXISTING) stats every root incl. frozen ones
    before the skip at :2680. One O(1) stat, not a walk — doesn't defeat the perf goal,
    but brushes the literal "no stat" wording.
Task 4: complete (commits 758105d..ca326c3, review clean after 1 fix round)
  Wired freeze into run_workflows: _resolve_frozen_slices / _write_freeze_cache /
  _freeze_fingerprint (load & save share ONE fingerprint builder). All 3 build_wide_csv
  sites pass frozen_slices + config_path. Cache written every run (frozen or not) so a
  later freeze finds a primed cache. Auto-rebuild = load_slice None -> omit -> walk.
  *** LIVE PRE-EXISTING BUG found via review finding #2, USER SAID FIX IT ***
  build_wide_csv self-loaded DEFAULT_CONFIG_PATH (config.yaml, tracking=5000) for the
  tracking_* columns regardless of --config, so `--config config_new.yaml` (tracking=1800)
  silently computed those columns with the WRONG threshold. Not caused by freeze; the
  fingerprint work exposed it. FIXED (ca326c3): threaded config_path main()->_run_combined
  (new kw-only param, default None)->all 3 build_wide_csv calls. Blast radius verified
  minimal: distance_limits branch dead (caller passes explicit limits), exclude_roots
  branch dead (no config defines any). Legacy/v2 golden green (those tests call
  build_wide_csv directly with config_path=None, untouched). Reviewer mutated a call site
  and watched the spy test catch the None -> genuine value check.
  Review finding #1 (no round-trip test through _freeze_fingerprint) also fixed:
  test_freeze_round_trip_write_then_resolve + a diverged-kw negative control.
  NOTE (deferred to final triage): _write_freeze_cache writes ~200-300MB local parquet
  per run unconditionally (measured by reviewer on real data, ~15-20s); bounded (fixed
  paths overwrite), not growing. A "skip write if fingerprint unchanged" refinement would
  cut most of it. Not a blocker.
  MINOR (deferred): pd.concat/groupby in _write_freeze_cache unguarded (outer edge of the
  "never fail the run" try); unused imports already removed.
Task 5: complete (commits ca326c3..ccfd2bf, review clean after 1 tiny fix)
  --thaw NAME (repeatable) / --thaw-all. Extracted _build_arg_parser() from main()
  preserving all 4 existing flags; _validate_thaw exits (SystemExit) on unknown name
  listing valid datasets; settings._thawed / ._thaw_all stashed for Task 4 to read.
  GOOD CATCH by implementer (deviation from brief, verified correct): dc_replace =
  dataclasses.replace DROPS non-field attributes, so stashing _thawed BEFORE the
  --figures-only/--folder dc_replace calls would silently wipe it -> --thaw inert under
  --figures-only (the most common figure mode). Moved the assignment AFTER both
  dc_replace calls (:2388), before _run_combined (:2473). _validate_thaw stays early
  (fail-fast). Reviewer independently reproduced the dc_replace fact.
  IMPORTANT (fixed, ccfd2bf): test_unknown_thaw_name_raises_listing_valid_names only
  asserted pytest.raises(SystemExit), never inspected the message — would pass with a
  bare SystemExit(), defeating the point. Inherited from MY brief. Now asserts the
  message contains both the rejected name and a valid dataset name; vacuity-checked.
Task 6: complete (commits ccfd2bf..1123f7a = 7064170 guard + 1123f7a wiring, review clean)
  should_skip_frozen_figure: skip iff EVERY contributing dataset frozen for figures.
  all([]) special-cased -> empty/unknown set DRAWS (never silent-skip). NOT folded into
  should_write (which force-returns True for reaction_matrix paths). Duck-types cfg.
  PLAN GAP I created + closed: I scoped the guard into the plot modules but left out the
  run_workflows plumbing that feeds them, so the envelope guards were INERT in production
  (dataset_overrides=None -> never skips). Completion commit 1123f7a added _apply_freeze_settings
  populating dataset_overrides/thawed/thaw_all on the plot configs (only when settings is
  not None -> default-off byte-identical), threaded settings through _rerender_envelope_block_with_scores
  / _run_envelope_visuals / _run_training, and forwarded --thaw/--thaw-all to the score_summary
  SUBPROCESS via _thaw_cli_args. All 4 production rerender call sites pass settings (verified) so
  the canonical score-annotated envelopes DO freeze; the 1-arg branch is a test-only shim.
  Reviewer independently mutated ALL-not-ANY, the empty-set early return, and the default-off
  None-path — all three tests bit. 608 passed, 1 xfailed. Two implementer rounds (guard came
  back DONE_WITH_CONCERNS, uncommitted; I committed the checkpoint then dispatched the wiring).
Task 7: complete (commits 1123f7a..bb60c00, review clean + 1 fix round)
  _expand_datasets: a dataset missing from BOTH bases AND freeze.data:true -> RuntimeError
  naming it (instead of the silent drop that would delete its rows). Unfrozen-missing keeps
  the silent skip. figure-freeze-only does NOT trigger it (only data freeze needs the folder).
  Reads raw yaml (runs before Settings exists).
  FIX (bb60c00): (1) added the missing test for figures-only+missing -> no raise (mutation
  .get(data)->.get(figures) confirmed it bites). (2) hardened BOTH freeze-reading sites via a
  new _freeze_block() helper so a hand-typed `freeze: true` / `freeze: "yes"` (truthy non-dict)
  coerces to {} instead of crashing config load with AttributeError. Relevant because the USER
  hand-edits config and pasted a malformed freeze block earlier this session. Same helper now
  guards Task 1's DatasetOverride parse too (config.py:953). 616 passed.
Task 8: complete (commit 548e4a9, verification only, no source changes)
  test_freeze_then_rerun_is_byte_identical: real cache round-trip (save->load->splice),
  asserts read_bytes()==read_bytes(); FROZEN=21 > LIVE=9 so the own_max_len fold is
  exercised (subagent confirmed deleting the fold breaks it). test_deleting_the_cache_is_safe:
  rmtree cache -> load_slice None -> re-derivation reproduces identical CSV.
  Legacy + v2 golden: 7 passed (freeze does not disturb legacy). Full freeze suite: 80 passed.
  Whole suite: 618 passed, 1 xfailed. Real pipeline --figures-only NOT run (needs live data +
  writes real Results dirs) — reported honestly, not fabricated.

ALL 8 TASKS COMPLETE. Branch feature/per-fly-score-matrix. Freeze commits:
  a4b8b57 (T1) 88b1df0+7a6beab (T2) 758105d (T3) ca326c3 (T4+config_path bug) ccfd2bf (T5)
  7064170+1123f7a (T6) bb60c00 (T7) 548e4a9 (T8), plus plan-doc commits.
NEXT: final whole-branch review, then present to user (NOT merged/pushed).

## FINAL WHOLE-BRANCH REVIEW (opus) — complete
Reviewed freeze feature 47d6fe3..548e4a9. Six cross-task seams verified sound:
fingerprint agreement per-site (load & save share one _fp_kw), own_max_len fold +
reindex + stable-sort byte-identity, default-off feature-wide, interface consistency,
duck-typing boundary (freeze.py imports no fbpipe.config), --thaw reaches both figure paths.
  ONE IMPORTANT found+fixed (1c2fe2f): trial_type_filter omitted from the fingerprint
  (SAME bug class as tracking — a row-affecting build_wide_csv param not fingerprinted).
  A frozen dataset whose trial_type_filter changed would serve stale rows AND permanently
  re-cache the narrowed set. Not triggered by config_new.yaml (no filter) but config_manual.yaml
  sets wide:testing. Fixed: trial_type_filter now a REQUIRED fingerprint field, normalized to
  match build_wide_csv (:2661-2673) incl sorted-list order-insensitivity; drift test + positive
  control + mutation-checked. Also guarded _write_freeze_cache concat/groupby (never fail run).
  All 6 deferred minors dispositioned: 5 ACCEPT, 1 (pd.concat guard) fixed with the above.
  COSMETIC (f47fff4): fixed 3 off-by-a-few line-pointer comments + stale get_dataset_override
  docstring. Comments/docstring only.
STATUS: feature complete, all reviews clean. NOT merged/pushed. Awaiting user direction.

## ── Per-rig anchor / mirrored rig_3 (docs/superpowers/plans/2026-07-21-rig3-mirrored-anchor.md) ──
Branch: feature/rig3-mirrored-anchor (NEW, from main @ f47fff4)
BASE for this plan: 41f33f9 (spec 9f01857 + plan 41f33f9 already committed)
Spec: docs/superpowers/specs/2026-07-21-rig3-mirrored-anchor-design.md

## Root cause (measured, do not re-litigate)
rig_3 is a physically MIRRORED rig (flies AND odor tube on the opposite side) —
user-confirmed. Angle is unsigned arctan2(abs(cross), dot) vs a hardcoded
right-edge anchor, so for rig_3 extension LOWERS the angle instead of raising it
=> angle_multiplier inverted into 0.5-1.0 instead of 1.0-2.0.
Measured over 20 fly-trials/rig (EB-Training-24-1/july_17_batch_2_*):
  corr(extension, angle) RIGHT anchor: rig_2 +0.054 | rig_3 -0.220  <- inverted
  corr(extension, angle) LEFT  anchor: rig_2 -0.054 | rig_3 +0.220  <- restored
  recompute from stored coords vs stored angle_ARB_deg: max|diff| 0.0000 deg
=> angle is fully derivable from stored x_class0/y_class0/x_class1/y_class1,
   so the fix is calculation-only: NO video re-encode, NO YOLO re-run.

## Key facts
- TWO duplicate angle implementations. envelope_combined.py is AUTHORITATIVE:
  it OVERWRITES angle_ARB_deg/_centered_deg/_centered_pct (:1717-1722). Fixing
  only compose_videos_rms would be silently undone. => envelope_combined = Task 2.
- compose_videos_rms.compute_fly_max_abs_centered (:400) has NO callers (dead).
- Only 6 of 8 rig_3 dirs hold data: EB-Training-24-1 (5), EB-Control-24-1 (1).
  3Oct-{Training,Control}-24-0.11 have ZERO *_distances.parquet and are absent
  from config_new.yaml `datasets:` => nothing to recompute there.
- EB-*-24-1 are deliberately LIVE (not frozen) per config_new.yaml:194-196.

## USER DECISIONS (pre-flight, do not re-litigate)
1. ACCEPT the 1px shift: Task 4 uses DEFAULT_ANCHOR 1080.0, dropping
   cfg.anchor_x 1079.0. Collapses the duplicate-anchor trap. Future rig_2 YOLO
   angles move ~0.05 deg; existing data untouched.
2. STOP BEFORE TASK 5 DATA MUTATION. Run Tasks 1-4 + the Task 5 dry-run only,
   then report and WAIT. Task 5 writes to real research data in BOTH the working
   tree and /securedstorage.

## Standing dispatch guards (from prior-effort incident)
- SCOPE GUARD in every dispatch: edit ONLY the task's named files. A prior
  subagent moved/deleted unrelated user files.
- This repo has a documented VACUOUS-TEST pattern (11 found in prior efforts).
  Reviewers MUST mutation-check, not trust green.
- config/* and docs/ are GITIGNORED (.gitignore:21, :116) — docs need `git add -f`.

## Tasks
- Task 1: complete (commit c339b95, review clean — Spec ✅, Approved, 0 Crit/Imp)
  src/fbpipe/utils/rig_anchor.py + tests/test_rig_anchor.py, 11 tests.
  All 4 mutants DIED (empty MIRRORED_RIGS; swapped anchor values; rig_token->None;
  dropped `reversed`). Non-vacuous.
  SAFETY VERIFIED by reviewer (highest-severity failure mode = a rig_2 path
  resolving to the mirrored anchor, which would corrupt correct data):
  greedy \d+ means rig_23 / rig_31 / rig_300 -> DEFAULT, never rig_3. Case-insensitive
  RIG_3/Rig3 -> mirrored correctly.
  MINOR (no action): a single path COMPONENT naming two rigs ("compare_rig_3_and_rig_2")
  resolves leftmost-in-component, not deepest. No real dir shape does this.
- Task 2: complete (commits 94cc2a9 impl + 7385012 test fix; opus review — Spec ✅, Approved;
  fix RE-REVIEWED by opus — Approved, all 4 mutants died, 0 Crit/Imp)
  envelope_combined.py: _compute_angle_deg(df, anchor=None); anchor threaded to all
  THREE call sites (_find_reference_angle :1599, _fly_max_centered :1647,
  _ensure_angle_percentages :1702). Reviewer instrumented the real path: exactly 3
  calls, all (0.0, 540.0). Full suite 666 passed, 1 xfailed.
  NOT INERT (reviewer verified): resolve_anchor is called on fly_dir at :2164
  (`for fly_dir in sorted(cfg.root.iterdir())`) whose OWN name carries the rig token,
  e.g. .../3Oct-Training-24-0.11/july_20_batch_2_rig_3. Confirmed vs real dirs on disk.
  IMPORTANT found+fixed (7385012): mutant hardcoding anchor=(0.0,540.0) for EVERY fly
  — which silently inverts ALL rig_2 data, the worst outcome — SURVIVED all 32 tests.
  The consistency test only asserted the 3 sites AGREE, never that resolution is
  CORRECT. Added wrong-rig tests (rig_2 dir + no-rig dir -> DEFAULT_ANCHOR).
  MINOR also fixed: "angle stays UNSIGNED" had no intentional guard — survived only
  because the fixture was collinear (cross == -0.0). Added an off-axis test asserting
  0<=angle<=180 for both anchors; signed-arctan2 mutant now dies.
  Fixer confirmed both mutants die + reverted cleanly. 18 passed.
  MINOR (deferred to final triage): angle_centered_pct is never numerically pinned —
  the spy fixture is collinear so fly_max==0 and valid_scale is False; a scale
  distortion would only be caught via anchor identity, not output value.
  NOTE: user is editing this repo CONCURRENTLY (odor_constants.py, new
  tests/test_model_score_lookup.py appeared mid-run). Subagents left them alone.
  RE-REVIEW (opus) of fix 7385012: APPROVED. M1 (hardcode mirrored for all) DIED;
  M2 (signed arctan2) DIED; M3 (hardcode DEFAULT for all -> would make the rig_3 fix
  INERT) DIED via the pre-existing consistency test; M4 (mirror on ANY rig token)
  DIED on the rig_2 param ALONE => the rig_2 case genuinely discriminates rig_2 from
  rig_3 and is not redundant with the no-token case.
  SUBTLE TRAP the implementer correctly avoided: pytest's tmp_path bakes the node id
  (which contains "rig3") into an ancestor component, so rig_token(tmp_path/...) 
  returns rig_3 — a "no rig token" test using tmp_path would have silently tested the
  WRONG thing. They used tempfile.mkdtemp instead.
  NITs (deferred to final triage, non-blocking): (a) mkdtemp's 8-char [a-z0-9_] suffix
  can in principle contain rig+digit, reintroducing the false token (~1-in-30k flake);
  a fixed subdir name would be deterministic. (b) in the off-axis test only the
  DEFAULT_ANCHOR iteration is sensitive to the sign mutation (vs MIRRORED, cross=+30000
  so signed==unsigned); the mirrored half is decorative.
- Task 3: complete (commits 07e74f8 impl + b0195d7 fix; opus review — Spec ✅,
  Needs-work -> both Importants fixed)
  compose_videos_rms.py: compute_angle_deg_at_point2(df, anchor=None); threaded to
  find_fly_reference_angle (:356), _process_fly_angles (:856), and the DEAD
  compute_fly_max_abs_centered (:408, no callers, per brief for consistency only).
  NOT INERT (reviewer verified independently): _process_fly_angles(fly_dir) is called
  at :908 with _discover_month_folders(root) output; real dirs are e.g.
  .../EB-Training/february_10_batch_1_rig_2 — token is in the fly_dir component.
  Implementer proactively found the brief's 3 tests did NOT catch the anchor-mismatch
  mutation and added 2 spy tests (learning from Task 2's finding).
  Reviewer mutants M1,M2,M3,M4,M5a,M5b all DIED.

  *** IMPORTANT #1 — REAL PRODUCTION BUG found by the Task 3 reviewer (fixed b0195d7) ***
  rig_token scanned EVERY ancestor component, so any ancestor containing rig3/rig2
  silently mirrored every fly beneath it:
     /data/EB-Training-24-1_excl_rig3/july_18_batch_1 -> rig_3 -> MIRRORED   (WRONG)
  i.e. a dir meaning "EXCLUDING rig3" resolved AS rig3, giving correct rig_2 data the
  mirrored anchor — the worst failure mode in this project.
  NOT HYPOTHETICAL: this dir already exists on disk —
    /home/ramanlab/Documents/cole/Results/Figures/EB-Training-24-1_excl_rig3_and_july17b2rig2
  and scripts/analysis/reaction_matrix_specific_flies_vs_control.py:36 writes to
  .../EB-Training-24-1_excl_rig3.
  FIX: _RIG_RE now REQUIRES the underscore — matches rig_3, no longer matches rig3.
  SAFE: verified every real rig dir on disk uses the underscore form (rig_2, rig_3)
  and nothing anywhere uses rigN without one; all known poisoners use the
  no-underscore form so they stop matching.
  BEHAVIOUR CHANGE vs Task 1: rig3/Rig3 (no underscore) now -> DEFAULT, was MIRRORED.
  Intentional. A false negative (data stays visibly wrong) is far safer than a false
  positive (correct data silently corrupted). Task 1 tests updated.
  Controller re-verified all 5 cases directly post-fix: ALL CORRECT, incl. a genuine
  _rig_3 nested UNDER a poisoner still mirroring (deepest real token wins).

  IMPORTANT #2 (fixed b0195d7): test_agrees_with_envelope_combined_implementation —
  the guard against the two duplicate angle impls drifting apart — was near-vacuous.
  _frame() put eye, proboscis AND both anchors on y=540, so every cross product was 0;
  mutating vy *= 3.0 (real geometric divergence) passed ALL tests. Now uses a
  non-collinear fixture. CAVEAT respected: test_mirrored_anchor_is_supplement_of_default
  genuinely REQUIRES p2y == anchor_y for the 180-x identity, so it keeps its own
  collinear frame. Both mutants now die. 30/30 passed.
  MINOR (deferred): spy asserts on the ARGUMENT not the effective anchor, so passing
  anchor=None (semantically == DEFAULT) would fail. Over-strict, harmless.
- Task 4: complete (commit 2e6a78c impl + fixes a129eae, 5719b35, 791c33f, 7529a79;
  opus review — Spec ✅, Needs-work -> all 3 Importants closed)
  yolo_infer.main: AX,AY were HOISTED once before the video loop; now resolved PER
  VIDEO at :585 via resolve_anchor(video_path) (one run can span rig_2 and rig_3).
  AX/AY audit (reviewer re-verified line-for-line): assigned :585, sole use :713,
  nothing in between; the `continue` at :581 precedes the assignment so no path
  reaches :713 without :585 in the same iteration.
  Real paths verified: <dataset>/<batch_dir>/<video>.mp4 — token on the batch
  component; filenames carry no rig token so nothing deeper shadows it.
  USER-APPROVED: dropped cfg.anchor_x (1079.0) for DEFAULT_ANCHOR (1080.0).

  *** IMPORTANT #1+#2 — the brief's tests were VACUOUS (fixed 5719b35) ***
  Reviewer wrote FIVE wrong implementations; ALL FIVE passed the committed tests:
    A throwaway vars then cfg.anchor_x | B assignment AFTER use (UnboundLocalError,
    crashes every run) | C resolve_anchor(base) no token | D swapped AY,AX |
    E call buried in `if os.getenv('__NEVER_SET__')`.
  Reviewer DISPROVED the implementer's "a real test needs GPU/TensorRT/video" claim
  by writing a hermetic one: ~1.5s, no GPU, monkeypatches YOLO/VideoCapture/writer/
  _scan_initial_fly_count/_run_chunked_inference, drives main() over a rig_3 + rig_2
  tmp tree and asserts recorded anchors == [(1080,540),(0,540)].
  Fixer adopted it; A-E now ALL FAIL. Deleted the third test (only exercised Task 1's
  resolve_anchor; passed before this task existed).
  LESSON: "the brief's tests are weak but I transcribed them verbatim" is not
  sufficient in this repo — a weak brief test is itself the defect to escalate.

  *** IMPORTANT #3 — dead-but-live-looking config (fixed a129eae + 791c33f) ***
  yolo_infer:547 was the ONLY reader of cfg.anchor_x/anchor_y in the whole repo;
  after Task 4 there were ZERO, yet Settings fields, the env ANCHOR_X/ANCHOR_Y
  loader, and config/example.yaml still advertised the knob. Setting it did nothing:
  no error, no warning, no effect. Removed all three (rig_anchor.py is now the single
  source of truth). grep confirmed no other reader/test.
  config.py had CONCURRENT USER EDITS (canon_fly_number) staged; fixer isolated via
  save-patch/reset/commit/reapply. CONTROLLER VERIFIED AFTERWARDS: canon_fly_number
  def+call still present, a129eae touches ONLY the 4 anchor lines (0 mentions of
  canon_fly_number), user's work still uncommitted in their working diff, all
  light_stimulus/video_writer/trial_metadata files intact at full length,
  git status grew 20->23 (their new files) — nothing lost.
  MINOR fixed (7529a79): gpu_accelerated.py:184-185 held a THIRD hardcoded 1079.0
  copy in an uncalled function; now points at rig_anchor.DEFAULT_ANCHOR.
  NOT CLOSED (out of authorized scope, for final triage): config/example.env:6-7
  still advertises ANCHOR_X/ANCHOR_Y — same dead knob.
  Full suite 687 passed, 1 xfailed (was 689; 2 vacuous tests removed).
- Task 5: script + tests complete (commit 3fa50cc). STEPS 6-10 (DATA MUTATION) NOT RUN
  — user reserved that decision. Dry-run verified by controller independently:
  948 tables, 168 EB-Control-24-1 + 780 EB-Training-24-1, ZERO rig_2, ZERO 3Oct,
  0 files modified.
- Task 6: full suite 697 passed, 1 xfailed.

## FINAL WHOLE-BRANCH REVIEW (opus, f47fff4..3fa50cc) + fixes 7c9bf15
MERGE-READY: YES for code. Primary gate HOLDS.
  VERIFIED (not asserted): rig_2 bitwise identical — anchor=None vs DEFAULT_ANCHOR give
  np.array_equal True on real rig_2 data in BOTH modules (both ANCHOR_X consts were
  already 1080.0). The two duplicate angle impls agree to max|diff| 0.000e+00 across 36
  (file x anchor) combos. Angles stay in [0,180]; NaN masks identical.
  FIX PROVEN ON REAL DATA: copied EB-Training-24-1/july_17_batch_2_rig_3 to scratch and
  ran _ensure_angle_percentages -> corr(extension, angle) -0.269 => +0.269 (n=30).
  Regex correct BOTH directions: every real rig dir uses _rig_N (no false negatives);
  the real poisoners resolve to DEFAULT. cfg.anchor_x/anchor_y: zero readers left
  anywhere incl. the user's uncommitted work.

  *** IMPORTANT #1 — MY PLAN'S TASK 5 RESTED ON A FALSE PREMISE (controller re-verified) ***
  The invalidation pass is NOT NEEDED for today's data. All three premises were false:
    - envelope_combined._ensure_angle_percentages has NO short-circuit guard; it
      recomputes unconditionally and writes when _series_matches fails.
    - compose_videos_rms._process_fly_angles is NOT in ORDERED_STEPS and is not
      referenced by pipeline.py or run_workflows.py — unreachable from the live pipeline.
    - angle_multiplier is present in 0 of 474 real rig_3 parquets.
  => Rerunning the analysis for EB-Training-24-1 + EB-Control-24-1 ALONE corrects rig_3.
  Running the invalidation would rewrite ~948 irreplaceable parquets for no benefit.
  Docstring + plan runbook corrected (7c9bf15); script kept as a --dry-run diagnostic.

  IMPORTANT #2 — the plan's rig_2 safety RATIONALE was also wrong (corrected 7c9bf15).
  It claimed _process_fly_angles short-circuits on angle_multiplier. Reviewer ran it on a
  scratch copy of real july_17_batch_2_rig_2: rewrote 60 files and CHANGED
  angle_centered_deg (compose re-centers on its OWN find_fly_reference_angle = 100.32 deg,
  which differs from envelope's). Doesn't fire today only because the module is unwired.
  True rationale: rig_2 is safe because resolve_anchor returns the identical DEFAULT
  anchor for it => bitwise-identical arrays.

  IMPORTANT #3 (recorded, NOT fixed — out of scope): scripts/analysis/abdomen_per_tracking.py:72
  hardcodes a THIRD ANCHOR_X, ANCHOR_Y = 1079.0, 540.0, unthreaded. Standalone CLI, not
  pipeline-wired, so no data corrupted — but it reports INVERTED angles for rig_3 and
  disagrees with DEFAULT_ANCHOR by 1px. Spec item 5 only partly met.

  ACCURACY CORRECTION (7c9bf15): rig_anchor.py + test docstrings had overstated the
  _excl_rig3 case as an OBSERVED production bug. resolve_anchor is only ever called on
  DATA paths (cfg.root.iterdir, _discover_month_folders, video_path), never on
  Results/Figures where that dir lives. Reworded to defensive/latent. Regex unchanged.
  Deferred minors triaged: 5 ACCEPT, 1 FIXED (config/example.env dead ANCHOR_X/ANCHOR_Y).

STATUS: code complete, all reviews clean, 697 passed / 1 xfailed. NOT merged, NOT pushed.
BLOCKED ON USER: the data step (rerun analysis for the 2 EB datasets). Nothing has been
written to /Data or /securedstorage.
