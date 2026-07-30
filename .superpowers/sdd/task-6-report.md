# Task 6 Report: Figure freeze

## What changed

### `scripts/analysis/envelope_visuals.py`

- **`should_skip_frozen_figure(cfg, datasets, *, thawed=(), thaw_all=False) -> bool`**
  (new, `:353-380`, right after `should_write`): verbatim from the brief.
  Empty/unknown contributor set -> `False` (draw). Otherwise skips iff every
  named dataset's `freeze_flags(cfg, name, thawed=thawed, thaw_all=thaw_all)`
  returns `freeze_figures=True`.
- **`MatrixPlotConfig`** (`:1077-1101`) and **`EnvelopePlotConfig`**
  (`:1860-1971`): each gained three new *optional, defaulted* fields —
  `dataset_overrides: Mapping[str, Any] | None = None`, `thawed: Sequence[str]
  = ()`, `thaw_all: bool = False`. No existing caller (run_workflows.py,
  envelope_combined.py, envelope_training.py, envelope_visuals.py's own CLI
  `main()`) sets these, so every existing call keeps today's defaults —
  `should_skip_frozen_figure` always returns `False` for them, unchanged
  behaviour. These dataclasses are passed *directly* as the `cfg` argument to
  `should_skip_frozen_figure` (duck-typed via `getattr` in
  `fbpipe.freeze.freeze_flags`) — no new local variable needed at the call
  sites.
- **`generate_reaction_matrices`** (`:1447`), guard at `:1564-1573`, inside
  `for order in cfg.trial_orders: for odor in ordered_present + extras:`,
  right after `if subset.empty: continue`. The loop variable `odor` **is**
  the canonical dataset for that iteration (`scores_df["dataset"]` is built
  from `dataset_canon`, not the odor name — the name is historical). Dataset
  set: `[odor]` — a reaction-matrix figure has exactly one dataset. Skipping
  here also prevents the `plot_reaction_rate_bars` call further down in the
  same iteration (`:~1735`) from running — see "plot_reaction_rate_bars"
  below for why that function itself was not touched.
- **`generate_envelope_plots`** (`:2081`), guard at `:2259-2261`, in the
  per-fly loop, right after the existing `[DEBUG] envelope_plots: generating`
  print and before the `out_path.exists() and not cfg.overwrite` check.
  Dataset set: `dataset_candidates` (`:2237`,
  `[dataset_lookup[idx] for idx in indices if dataset_lookup[idx]]`) — the
  exact set the function already builds to resolve that fly's output folder.
  This is "that fly's dataset," per the brief.
- **`plot_reaction_rate_bars`** (`:1279`): **left untouched.** It draws into
  a caller-supplied `ax` and has no path/`cfg` parameter and no independent
  save call — it is a sub-panel of whatever figure the caller is building. It
  has three callers: `generate_reaction_matrices` (in-scope, guarded
  transitively as above), and `reaction_matrix_training_vs_control.py:898`
  and `reaction_matrix_from_spreadsheet.py:485` (both **out of scope**, not
  modified). Giving it its own `cfg`/`datasets` parameters would force a
  signature change onto those two out-of-scope call sites or leave new
  parameters unreachable dead code for the one in-scope caller, which already
  gets the correct behaviour for free from the `generate_reaction_matrices`
  guard. See "Known gaps" below for what this means for the two other
  scripts.

### `scripts/analysis/score_summary.py`

- Imports `should_skip_frozen_figure` from `envelope_visuals` (`:49`).
- **`_plot_bar_charts`** (`:566`), guard at `:594-596`: one dataset per
  figure (`odor` == `dataset_canon` here too). Set: `[odor]`.
- **`_plot_training_vs_control_bars`** (`:739`), guard at `:760-766`: THE
  canonical mixed case from the brief. Set:
  `{train_ds, *sub["control_dataset"].dropna().unique().tolist()}` — both
  halves of the pair this figure actually draws.
- **`_plot_heatmap`** (`:883`), guard at `:892-899`: one figure pooling every
  dataset in `summary`. Set: `set(summary["dataset_canon"].unique())`.
- **`_plot_score_pair`** (`:965`), guard at `:985-988`: same pairing as
  `_plot_training_vs_control_bars`. Set: `[train_ds, ctrl_ds]`.
- All four gained `cfg: Any = None, thawed: Sequence[str] = (), thaw_all:
  bool = False` keyword params, threaded through `_summarise_and_plot`
  (`:1082`) and `generate_score_summary` (`:1116`).
- **`main()`** (`:1195`): `settings` is now initialised to `None` before the
  `--config` branch (was previously scoped only inside the `try:`, so a
  missing `--config` left it unbound). Passed to `generate_score_summary` as
  `cfg=settings`. On `load_settings` failure, `settings` is explicitly reset
  to `None` in the `except` — a broken config must read as "no freeze info"
  (never skip), not silently retain a partially-built value.
- **`_parse_args`** (`:1136`): added `--thaw` (repeatable) and `--thaw-all`,
  mirroring `run_workflows.py`'s Task-5 flags, threaded into
  `generate_score_summary(..., thawed=tuple(args.thaw or ()), thaw_all=bool(args.thaw_all))`.

### `tests/test_freeze_figures.py` (new, 15 tests)

- The 9 tests from brief Step 1 + `test_figures_only_does_not_bypass_freeze`
  from Step 6, verbatim.
- 5 additional end-to-end emit-site verification tests (not in the brief,
  added per its own instruction to "verify at least one emit site's guard by
  constructing the contributing-dataset set it would pass and confirming a
  live dataset in that set forces a draw"):
  - `test_generate_envelope_plots_skips_frozen_dataset_draws_live` — real
    `wide_to_matrix` + `generate_envelope_plots` run, two datasets
    (Hex-Training frozen, Hex-Control live). Frozen dataset's folder has no
    PNG; live dataset's folder does.
  - `test_generate_reaction_matrices_skips_frozen_dataset_draws_live` — same
    idea through `generate_reaction_matrices`.
  - `test_score_summary_mixed_frozen_live_pair_still_draws` — real
    `generate_score_summary` run, EB-Training frozen / EB-Control live.
    Asserts EB-Training's solo bar chart is skipped, EB-Control's is drawn,
    the pooled heatmap is drawn (EB-Control is live), and — the load-bearing
    assertion — `mean_score_train_vs_ctrl_EB-Training.png` (frozen Training
    paired with live Control) is still drawn.
  - `test_score_summary_all_frozen_pair_is_skipped` — same pair, both frozen:
    every figure skips; non-figure CSV output is untouched (freeze.figures
    only gates figures).
  - `test_plot_score_pair_mixed_frozen_live_still_draws` — v2-protocol-only
    `_plot_score_pair`, mixed then all-frozen.

## Design decisions the brief left to be resolved

**Where does `cfg` come from at each `envelope_visuals.py` site?** The
functions listed in the brief (`generate_envelope_plots`,
`generate_reaction_matrices`) already take a parameter named `cfg`, but it is
`EnvelopePlotConfig` / `MatrixPlotConfig` — a plotting-options dataclass with
no `dataset_overrides` field, not the pipeline `Settings` object the brief's
pseudocode implies. Rather than invent a second parameter or a module-level
global, I added `dataset_overrides` / `thawed` / `thaw_all` as new optional
fields directly on those two dataclasses and pass the *same* `cfg` the
function already has straight into `should_skip_frozen_figure`. This keeps
every existing caller's behaviour byte-identical (new fields default to "no
freeze") and makes the guard fully unit- and integration-testable within this
task's three allowed files.

## Known gap: production wiring for `envelope_visuals.py` sites (report per brief's escape valve)

The new `dataset_overrides` / `thawed` / `thaw_all` fields on
`MatrixPlotConfig` / `EnvelopePlotConfig` are **not yet populated by
`run_workflows.py`**, so `generate_envelope_plots` / `generate_reaction_matrices`
guards are inert in a real pipeline run today (safe — they default to "always
draw," never to "always skip" — but the feature does not yet take effect for
per-fly RMS or reaction-matrix figures end-to-end).

Concretely: `run_workflows.py:674` (`_matrix_plot_config`) and `:713`
(`_envelope_plot_config`) build these dataclasses from a `data: Mapping`
that is never given `dataset_overrides`/`thawed`/`thaw_all` keys, across all
nine `generate_envelope_plots(...)`/`generate_reaction_matrices(...)` call
sites (`:876, 900, 911, 955, 1050, 1077, 1506, 1660, 1680`). Activating the
guard there needs `run_workflows.py` to pass, e.g.,
`opts["dataset_overrides"] = settings.dataset_overrides`,
`opts["thawed"] = getattr(settings, "_thawed", ())`,
`opts["thaw_all"] = getattr(settings, "_thaw_all", False)` into those two
builder functions (both already execute with `settings` in scope at every
call site, per the existing `_resolve_frozen_slices(..., thawed=getattr(settings,
"_thawed", ()), ...)` pattern Task 4/5 already established at
`run_workflows.py:1286-1287, 1437-1438, 1612-1613`).

**This is `run_workflows.py` — outside this task's scope guard — so I did not
touch it.** Per the brief: "If you think another file must change, STOP and
report instead."

**`score_summary.py` does NOT have this gap.** `run_workflows.py`'s
score-summary subprocess invocation (`:2033-2048`) already forwards
`--config` when available, and `score_summary.py`'s own `main()` already
loads that config for `odor_remap` — I extended the same load to also supply
`cfg=settings` to `generate_score_summary`. **The `freeze.figures` guard for
all four `score_summary.py` sites is fully active in production today**,
with no further wiring needed. The one thing NOT forwarded there is
`--thaw`/`--thaw-all` — `run_workflows.py`'s `score_cmd` list does not pass
them to the subprocess, so `--thaw` on the top-level CLI will not currently
un-freeze a dataset's `score_summary.py` figures specifically (it does work
for `envelope_combined`/`build_wide_csv`'s own freeze.data per Task 5). This
is a narrower gap than the `envelope_visuals.py` one — the default-off /
freeze-on-request behaviour is unaffected, only the escape hatch is
incomplete for this one script — and fixing it also requires editing
`run_workflows.py`'s `score_cmd` construction, out of scope here.

**`plot_reaction_rate_bars`'s other two callers are fully unguarded.**
`reaction_matrix_training_vs_control.py:898` and
`reaction_matrix_from_spreadsheet.py:485` call it directly and are not in
this task's scope; their figures (and the reaction-rate panel specifically)
will always redraw regardless of any dataset's `freeze.figures` setting.

## TDD: failing test first

```
$ python -m pytest tests/test_freeze_figures.py -v
...
FAILED tests/test_freeze_figures.py::test_single_frozen_dataset_figure_is_skipped - AttributeError: module 'scripts.analysis.envelope_visuals' has no attribute 'should_skip_frozen_figure'
FAILED tests/test_freeze_figures.py::test_single_live_dataset_figure_is_drawn - AttributeError: ...
FAILED tests/test_freeze_figures.py::test_all_frozen_aggregate_is_skipped - AttributeError: ...
FAILED tests/test_freeze_figures.py::test_mixed_frozen_and_live_is_DRAWN - AttributeError: ...
FAILED tests/test_freeze_figures.py::test_unknown_dataset_counts_as_live - AttributeError: ...
FAILED tests/test_freeze_figures.py::test_empty_dataset_set_is_drawn - AttributeError: ...
FAILED tests/test_freeze_figures.py::test_freeze_data_alone_does_not_skip_figures - AttributeError: ...
FAILED tests/test_freeze_figures.py::test_thaw_all_draws_everything - AttributeError: ...
FAILED tests/test_freeze_figures.py::test_thaw_named_dataset_draws - AttributeError: ...
FAILED tests/test_freeze_figures.py::test_figures_only_does_not_bypass_freeze - AttributeError: ...
10 failed, 14 warnings in 0.13s
```

Failed for the expected reason. (The 5 emit-site verification tests were
added after implementation, since they exercise the guard sites, not just
the core function — TDD was applied to `should_skip_frozen_figure` itself,
which is the piece with brief-mandated pre-written failing tests.)

## After implementation

```
$ python -m pytest tests/test_freeze_figures.py -v
...
15 passed, 192 warnings in 2.47s
```

`-s` on the emit-site tests confirms the `[FROZEN]` log lines fire with the
correct path, e.g.:

```
[FROZEN] Skipping figure (all contributors frozen): .../plots/Hex-Training/frozen_fly_fly1_testing_envelope_trials_by_odor_30_shifted.png
[FROZEN] Skipping figure (all contributors frozen): .../out/mean_score_EB-Training.png
[FROZEN] Skipping figure (all contributors frozen): .../out/mean_score_heatmap.png   # (all-frozen case only)
[FROZEN] Skipping figure (all contributors frozen): .../out/mean_score_train_vs_ctrl_EB-Training.png  # (all-frozen case only)
```

...and, from the same `-s` run, the mixed-pair test's `mean_score_heatmap.png`
and `mean_score_train_vs_ctrl_EB-Training.png` do **not** print a `[FROZEN]`
line — i.e. they draw, as required.

Full suite:

```
$ python -m pytest -q
...
ERROR HomeAssistant/test_influxdb_enclosure.py::test_connectivity
ERROR HomeAssistant/test_influxdb_enclosure.py::test_basic_query
ERROR HomeAssistant/test_influxdb_enclosure.py::test_period_queries
602 passed, 1 xfailed, 367 warnings, 3 errors in 48.26s
```

587 (baseline HEAD) + 15 new = 602 passed, 1 xfailed — matches. The 3 errors
are the pre-existing `HomeAssistant/test_influxdb_enclosure.py` errors,
present at base already (network-dependent, unrelated).

CLI sanity:

```
$ python scripts/analysis/score_summary.py --help   # exit 0, shows --thaw/--thaw-all
$ python scripts/analysis/envelope_visuals.py --help            # exit 0, unchanged
$ python scripts/analysis/envelope_visuals.py envelopes --help  # exit 0, unchanged
$ python -c "import scripts.pipeline.run_workflows as rw"       # imports cleanly, unmodified
```

## Self-review findings

- **Vacuity check on the mixed-case assertions**: ran
  `test_score_summary_mixed_frozen_live_pair_still_draws` and
  `test_plot_score_pair_mixed_frozen_live_still_draws` with `-s` and manually
  confirmed no `[FROZEN]` line appears for the four assertions that require a
  draw (`mean_score_EB-Control.png`, `mean_score_heatmap.png`,
  `mean_score_train_vs_ctrl_EB-Training.png`, `mean_score_pair_EB-Training.png`)
  while a `[FROZEN]` line does appear for `mean_score_EB-Training.png` — not
  just checking file existence, which could pass vacuously if the function
  silently no-op'd for an unrelated reason.
- **Confirmed `all([])` trap is actually exercised**: temporarily reverted
  the `if not names: return False` early-return locally and re-ran
  `test_empty_dataset_set_is_drawn` — it failed (`assert True is False`),
  confirming the test is not vacuous and the guard clause is load-bearing.
  Restored before finishing.
- **Default-off verified two ways**: (a) the full suite (602/602, no
  regressions) exercises every existing `EnvelopePlotConfig`/`MatrixPlotConfig`
  construction in the repo's test suite with the new fields absent/default,
  and none of their output assertions changed; (b) `should_skip_frozen_figure`
  called with `cfg=None` (score_summary.py's default) returns `False`
  unconditionally, traced through `freeze_flags(None, name)` ->
  `getattr(None, "dataset_overrides", None)` -> `None` -> `override is None`
  -> `(False, False)`.
- **`should_write` was not reused or modified** — confirmed by grep: no new
  call to `should_write` was added, and the freeze guard is a distinct
  `if should_skip_frozen_figure(...): ...; continue/return` block at each
  site, always placed as a *separate* check (in 5 of 6 sites, immediately
  after an existing `should_write`/overwrite check; never inside it).
- **Legacy protocol byte-for-byte**: no rendering code path was touched; the
  new fields and guards only add early-exit branches before any drawing
  begins. `ev.set_protocol("legacy")` is exercised by
  `test_score_summary_mixed_frozen_live_pair_still_draws` and
  `test_score_summary_all_frozen_pair_is_skipped` without incident.
- **Scope**: `git status --porcelain` shows only `M
  scripts/analysis/envelope_visuals.py`, `M scripts/analysis/score_summary.py`,
  `?? tests/test_freeze_figures.py`. No other file moved, renamed, or edited.
- **Not implemented / reported instead of guessed**: the `run_workflows.py`
  wiring needed to activate the `envelope_visuals.py` guards in production,
  and the `--thaw`/`--thaw-all` forwarding to the `score_summary.py`
  subprocess — both detailed above, both require editing a file outside this
  task's scope guard.

## Commit

Not yet committed — leaving that to the caller per this session's practice of
committing only when explicitly asked. Working tree is clean apart from the
three intended files; ready for:

```
git add scripts/analysis/envelope_visuals.py scripts/analysis/score_summary.py tests/test_freeze_figures.py
git commit -m "feat(figures): skip a figure only when every contributing dataset is frozen"
```

Base commit this work sits on: `ccfd2bf` (test: guard control-left/training-right against a column swap).
