# Dataset Freeze — Design

**Date:** 2026-07-15
**Status:** Approved, pending implementation plan
**Branch:** `feature/per-fly-score-matrix` (freeze work to branch from here)

## Problem

A dataset that is scientifically finished still costs a full re-derivation on every
run. `build_wide_csv` rebuilds the wide CSV from scratch each time: it walks every
dataset root, recomputes per-trial statistics (AUC, local extrema, distance stats),
writes a header, then appends rows one at a time
(`scripts/analysis/envelope_combined.py:2600-3338`). A dataset's rows exist in that
file only because its raw data was just re-read off disk. Its figures are likewise
redrawn every run.

There is no way to say "this dataset is done — keep its rows, stop re-deriving them,
stop redrawing its figures."

### Why the existing cache does not solve this

The pipeline already has a content-hash manifest cache
(`scripts/pipeline/run_workflows.py:284-414`), but its `combined` category is keyed on
the single string `"analysis"` over a *union* manifest of every root
(`:497-518`). One changed file in one dataset invalidates the whole gate and re-runs
`build_wide_csv` over **all** roots. There is no per-dataset granularity below the
`combined` stage.

`--figures-only` already works around this sensitivity by hard-setting
`skip_combined = True` regardless of the manifest (`:2177-2188`), with the comment that
the manifest check "is too sensitive (it re-runs combined on any source file mtime
change, even if no new flies were added) — defeats the purpose of the mode." That
workaround is global and all-or-nothing. Freeze is the per-dataset version of the same
intent.

## Decisions

Settled during brainstorming; recorded here because each one closes off a design branch.

| Question | Decision | Consequence |
| --- | --- | --- |
| Raw data on disk when frozen? | **Stays on disk, just skipped** | The cache is a derived artifact, not primary data. Deleting it must always be safe. Auto-rebuild is always possible. |
| Frozen dataset's files change on disk? | **Nothing — pure trust** | `freeze.data` means never walk the folder. Zero I/O. Staleness is the user's promise to keep. |
| Config value affecting numbers changes? | **Auto-rebuild that dataset once** | Config is already in memory, so checking is free. Prevents one CSV mixing two parameterizations. Self-heals. |
| Figure mixing frozen + live dataset? | **Redraw; skip only if ALL contributors frozen** | Live changes always surface. Frozen half comes from cache, so still cheap. |
| Freeze granularity | **Two independent flags** (`data`, `figures`) | Either may be set alone; four valid combinations. |

**Scope note on "pure trust":** it governs *raw data* only. We never hash or stat the
frozen dataset's folder. It does **not** extend to config, which is free to check.

## Config surface

Freeze lives on the existing per-dataset override map, keyed by dataset name
(`config/config_new.yaml:17-113`, dataclass `src/fbpipe/config.py:456-483`):

```yaml
dataset_overrides:
  EB-Control-24-1:
    freeze:
      data: true       # don't re-derive wide rows; splice from cache
      figures: true    # don't regenerate this dataset's own figures
```

Two new fields on `DatasetOverride`, both defaulting to `false`:

- `freeze_data: bool = False`
- `freeze_figures: bool = False`

Parsed from a nested `freeze:` mapping at `src/fbpipe/config.py:881-895`. An absent
`freeze:` block, or an empty one, means both `false` — identical to today's behavior.

The flag lives in config; the **rows do not**. A single dataset's slice is thousands of
rows by hundreds of `dir_val_*` columns. Storing that in YAML would be unreadable,
unmergeable, and would defeat the "cache is deletable" property.

### Flag combinations

All four are valid and must be tested:

| `data` | `figures` | Meaning |
| --- | --- | --- |
| `false` | `false` | Normal — today's behavior, unchanged. |
| `true` | `false` | Rows from cache, figures redraw. Fast figure-styling iteration. |
| `false` | `true` | Rows recomputed, figures not redrawn. Refresh CSV without figure cost. |
| `true` | `true` | Fully frozen. |

### Escape hatch

New CLI flags on `scripts/pipeline/run_workflows.py:1908-1934`:

- `--thaw NAME` — repeatable (`action="append"`); ignore freeze for the named dataset
  this run only.
- `--thaw-all` — `store_true`; ignore every freeze this run.

Neither edits YAML. Thawing a dataset re-derives it and **re-caches** it, so a thawed
run also refreshes the frozen slice.

Unknown name passed to `--thaw` → hard error listing valid dataset names. A silently
ignored typo would look exactly like a successful thaw.

## Data freeze

### Cache location and layout

Under the configured `cache_dir` (`src/fbpipe/config.py:603-610`; defaults to
`~/.cache/ramanlab_auto_data_analysis`, already managed by `cache_manager.sh` and
outside the repo):

```text
<cache_dir>/frozen/<wide_block>/<dataset>/
    rows.parquet     # the dataset's slice of that block's wide CSV
    meta.json        # fingerprint + own_max_len
```

**Keyed per wide block, not per dataset alone.** `build_wide_csv` is called up to four
times per run with different `measure_cols`, each producing different numbers:

| Block key | Call site | `measure_cols` |
| --- | --- | --- |
| `wide` | `run_workflows.py:1135` | `wide_cfg.measure_cols` or `["envelope_of_rms"]` |
| `combined_base` | `run_workflows.py:1257` | `["combined_pct", "combined_base"]` |
| `distance_base` | `run_workflows.py:1257` | `["distance_percentage"]` |
| `pair_groups` | `run_workflows.py:1403` | outer `wide_measure_cols` |

A single per-dataset cache would collide across these and serve `distance_percentage`
rows to a `combined_base` build.

**The block name is a partitioning label, not the correctness mechanism.** A dataset's
rows are derived independently of which *other* datasets share the run, so the slice for
a given dataset is fully determined by its raw data plus the fingerprint (which includes
`measure_cols`). Keying by block name is therefore a safe **over**-partition: it can
duplicate an identical slice across two blocks that happen to share parameters
(e.g. `pair_groups` and `wide`), but it can never serve the wrong rows. This is why
`pair_groups` needs no per-pair key despite running once per pair — the slice for
dataset X is the same in every pair X belongs to. Correctness rests on the fingerprint;
the block name only keeps the directory layout legible.

### `meta.json`

```json
{
  "schema_version": 1,
  "dataset": "EB-Control-24-1",
  "wide_block": "combined_base",
  "own_max_len": 1487,
  "row_count": 312,
  "columns": ["dataset", "fly", "..."],
  "fingerprint": { "...": "see below" }
}
```

`own_max_len` is the dataset's **own** maximum trace length, recorded at cache time.
It is the load-bearing field — see below.

### The `max_len` problem

`build_wide_csv` computes a single global `max_len` across all roots
(`envelope_combined.py:2637`, `:2736-2743`), builds the header
`dir_val_0..dir_val_{max_len-1}` from it (`:2839-2841`), and then pads **or truncates**
every row to it:

```python
if len(values) < max_len:
    row.extend([np.nan] * (max_len - len(values)))     # :3281-3282
elif len(values) > max_len:
    row = row[: len(metadata) + len(AUC_COLUMNS) + max_len]   # :3283-3284
```

That truncation is why cached rows cannot simply be concatenated. If a frozen dataset
has longer traces than any live dataset, and its length is not folded into the global
max, its cached rows are silently **chopped** — data loss with no error.

So the splice is:

1. For each frozen dataset, read `meta.json` and fold `own_max_len` into the global
   `max_len` computation, **without walking the root**.
2. Compute the final global `max_len = max(live_max_len, *frozen_own_max_lens)`.
3. Write the header from that final `max_len`.
4. Re-pad (or truncate) cached rows to it exactly as live rows are, then append.

Cached rows are stored padded to their own `own_max_len`, not the global max at cache
time — the global max is a property of the *run*, not the dataset, so storing it would
make the cache depend on which other datasets happened to be present.

### Skipping the walk

`build_wide_csv` walks roots at `envelope_combined.py:2664-2687`. Frozen roots are
excluded from that iteration entirely — no `rglob`, no `stat`, no hashing. This is the
entire performance win, and it is what "pure trust" buys.

### Fingerprint

`meta.json.fingerprint` records everything that determines the rows *other than* the raw
data:

- `protocol` (`legacy` / `v2`) — these have **different column sets**. `_is_legacy()`
  gates `fly_type` (`:2800-2801`) and the `trial_*_s` columns (`:2828-2836`). A legacy
  cache spliced into a v2 run is a schema mismatch, not a value drift.
- `measure_cols`
- `fps_fallback`
- `distance_limits`
- `non_reactive_threshold`
- `low_max_threshold_px`
- `use_per_trial_baseline`
- the dataset's own resolved `DatasetOverride` (including `odor_remap`,
  `trial_type_override`, `odor_on_s`, `odor_off_s`, `light_only`, `light_start_s`,
  `light_duration_s`)

On mismatch: **auto-rebuild that dataset once**, then re-cache under the new
fingerprint. A cache miss (no `rows.parquet`) takes the same path. Both self-heal
without user intervention, and log one line naming the dataset and which fingerprint
field drifted.

Note `figure_output_subdir` is deliberately excluded — it affects figure routing, not
row values, and a change to it must not invalidate a data cache.

### Cache writes

After any run that derives a dataset live, write its slice for each wide block it
participated in. This happens whether or not the dataset is currently frozen — freezing
a dataset later then finds a cache already waiting, rather than requiring one
"priming" run.

Extract by `dataset == <name>`. The `dataset` column is `column_order[0]` and its value
is the root's directory **basename** (`:2668`, emitted `:3254`).

**Basename collision hazard:** because `dataset` is a basename, two roots with the same
basename under different bases (e.g. `flys_New/EB-Control-24-1` and
`Data-secured-New/EB-Control-24-1`) collapse to the same `dataset` value. The freeze
cache key inherits this. This matches existing pipeline behavior — `_auto_sync_wide_roots`
(`run_workflows.py:861-891`) and the flagged-dir dedup (`:1031-1066`) both already match
on lowercase basename — so freeze does not make it worse, but the cache key must use the
same basename normalization as the `dataset` column or the two will disagree.

### Frozen but folder missing

`_expand_datasets` (`src/fbpipe/config.py:141-152`) filters the `datasets:` list down to
directories that exist on disk and prints "Skipping datasets not yet on disk". Today,
archiving a dataset's raw data therefore silently removes it from the wide CSV.

For a dataset with `freeze.data: true`, a missing folder is a **hard error** naming the
dataset — not a silent drop. We cannot auto-rebuild what is not there, and the decision
above ("stays on disk") means a missing folder is a mistake, not a workflow.

This is a deliberate, narrow change to `_expand_datasets` behavior: it applies only to
frozen datasets. Unfrozen datasets keep the existing silent-skip semantics.

## Figure freeze

**Rule: skip a figure iff every dataset contributing to it has `freeze.figures: true`.**

This satisfies "don't make any figures per fly or whole dataset again" for a frozen
dataset's own figures, while keeping mixed figures correct.

### Why figures need the rule rather than a flat skip

Figures come in two shapes:

- **Per-dataset** — per-fly envelope traces, routed to `base/<Dataset-Name>/` by
  `resolve_dataset_output_dir` (`scripts/analysis/envelope_visuals.py:303-331`).
- **Aggregate** — the score heatmap, `mean_score_pair_*`, reaction matrices,
  `dataset_means`. Built from all datasets at once.

A frozen dataset's rows are *ingredients* in aggregate figures. If EB-Control-24-1 is
frozen but flies are added to a live EB-Training-24-1, `mean_score_pair_EB-Training-24-1`
must redraw — it draws both side by side. Freezing it would mean live changes silently
fail to appear.

### Where the guard lands

`resolve_dataset_output_dir` (`envelope_visuals.py:303-331`) already computes the
contributing dataset set — it joins multiple names into one folder as
`"_".join(sorted_names)` (`:328-329`). That set is exactly what the rule needs.

The existing `should_write` (`envelope_visuals.py:334-343`) **cannot** be reused: it
force-returns `True` for any path containing `reaction_matrix` / `reaction_prediction`
regardless of `overwrite`. The freeze guard is a separate check that runs before it.

Per-fly figures are guarded by their own fly's dataset. Aggregate figures are guarded by
the union of contributing datasets.

`--figures-only` currently force-sets every figure step to `True`
(`run_workflows.py:2024-2046`). `freeze.figures` is honored under `--figures-only` —
otherwise the most common way to run figures would bypass the feature entirely. `--thaw`
remains the way to force a redraw.

## Non-goals

- **Freezing raw data away.** Data stays on disk. Freeze is a speed optimization, not an
  archival mechanism.
- **Hashing frozen roots.** Explicitly rejected; it is the cost freeze exists to avoid.
- **Replacing the manifest cache.** Freeze sits beside it. Unfrozen datasets keep using
  it unchanged.
- **Per-fly freeze.** Freeze is per-dataset. Per-fly exclusion already exists via
  `flagged_flies_csv` (`config.py:491-497`).
- **Changing legacy protocol output.** Legacy must stay byte-for-byte identical, per this
  branch's existing regression guarantee.

## Testing

Per `CLAUDE.md`: tests are written and run **before** the implementation, to confirm
current behavior and validate assumptions.

### The load-bearing test: round-trip equivalence

Run two datasets live → snapshot the wide CSV → set `freeze.data: true` on one → re-run
→ assert the output is **byte-identical**.

**This test must be built to bite.** The frozen dataset is given a *deliberately longer*
trace length than the live one. With equal lengths, the test would pass even if the
`own_max_len` fold were entirely missing — the global max would be correct by accident.
That is precisely the vacuous-assertion shape this branch has repeatedly caught
(asserting a quantity that is equal by construction).

### Required cases

| Case | Asserts |
| --- | --- |
| Round-trip, frozen traces **longer** than live | Frozen rows not truncated; global `max_len` grew to frozen's length. |
| Round-trip, frozen traces **shorter** than live | Frozen rows padded with NaN to global max. |
| Frozen root never walked | Zero filesystem reads under the frozen root (spy/instrument the walk). This is the actual perf claim. |
| Config drift (`non_reactive_threshold` changed) | Dataset auto-rebuilds, re-caches, new fingerprint persisted. |
| Protocol drift (legacy cache, v2 run) | Auto-rebuild, not a schema crash or silent column mismatch. |
| Cache miss | Same path as drift; run succeeds. |
| Frozen + folder missing | Hard error naming the dataset; **not** a silent drop. |
| Per-block isolation | `combined_base` and `distance_base` caches do not collide. |
| `data: true, figures: false` | Rows cached, figures still drawn. |
| `data: false, figures: true` | Rows recomputed, figures skipped. |
| All-frozen aggregate figure | Skipped. |
| Mixed frozen+live aggregate figure | **Redrawn**, and contains the frozen dataset's rows. |
| `--thaw NAME` | Overrides freeze; re-caches. |
| `--thaw` unknown name | Hard error listing valid names. |
| Legacy protocol regression | Byte-for-byte identical, freeze absent. |

### Verification discipline

Drawn from this branch's recorded failures:

- Place mutants **inside the unit under test** and re-run, rather than trusting a report
  that a test passes.
- Never verify "legacy unchanged" with `git stash` (a no-op diff produces a vacuous
  pass) and never against a stale baseline — the pipeline regenerates
  `model_predictions.csv`. Use a worktree at the branch point with the **same** data.
- An assertion that reads its expectation from the constant it validates proves nothing.
- Confirm figure-level behavior by **inspecting a rendered figure**, not only by
  asserting on intermediate data. The reaction-module merge bug on this branch was caught
  only by looking at the rendered output (7 columns where 8 were expected).

## Open item (not blocking)

`docs/` is gitignored at `.gitignore:116`, yet tracked files already exist under it
(gitignore does not affect already-tracked files). This spec is force-added to match.
Commit `1380918`, titled "ensure docs/ and CLAUDE.md are tracked", in fact left `docs/`
ignored and added `CLAUDE.md` at `:118`. `config/*` is likewise untracked, which is why
`config/config_new.yaml` — and therefore any `freeze:` block added to it — is not pinned
by version control. Worth a decision before freeze config is relied upon, since an
untracked config means the freeze declarations live only in the working tree.
