# Random-Split Null Distribution of Control Reaction-Rate Gaps (EB-24-1)

**Date:** 2026-07-23
**Status:** Approved (design)
**Author:** cole (with Claude)

## Problem

The reference figure `reaction_matrix_ctrl_batch_1_vs_batch_2_30_latency_2.150s.png`
compares per-odor PER reaction rates between control **batch 1** (9 flies) and
**batch 2** (18 flies). We want to know how much of that gap is just an artifact of
splitting 27 control flies into two subgroups — i.e. the chance/noise floor.

To measure it, repeatedly split the **EB-Control-24-1** pool into two random groups
and look at the per-odor reaction-rate gap across many splits. The average of that
gap is the "typical difference two random control groups show by luck."

## Goal

Produce a single averaged figure (two variants) showing, **per odor**, the mean
absolute reaction-rate gap `|rate_A − rate_B|` over **100** random 10-vs-10 splits of
the control pool, with a spread band — and overlay the real batch-1-vs-batch-2 gap for
context.

## Requirements

### Sampling
- Pool = the **27 unique `(fly, fly_number)` control flies** in `EB-Control-24-1`
  (after flagged-fly exclusion, `testing` trials only).
- Each of **100 iterations** independently:
  - Reshuffle **all 27** flies (fresh RNG draw, without replacement).
  - Group **A = first 10**, Group **B = next 10**, **7 held out**.
  - The held-out 7 are **different on every iteration** — they rotate, not fixed.
- Reproducible via `--seed` (default fixed, e.g. 0). Uses `numpy.random.default_rng`.
- Parameters exposed: `--n-splits` (default 100), `--group-size` (default 10),
  `--seed`, plus dataset/CSV/flagged/config/out paths.

### Metric
- Fix the **8 odor columns once** from the full control pool via `_build_during_matrix`
  (`order="observed"`), so every split shares identical axes. Columns include the
  trained odor Ethyl Butyrate split into **EB1 / EB2** (its two presentations).
- Per split: build the during-matrix for A and for B on the **fixed columns**
  (`columns=cols`), compute per-odor reaction rate via `_rates_from_matrix`
  (`rate = reactions / trials * 100`), record `gap = |rate_A − rate_B|` per odor.
- Aggregate per odor across the 100 splits:
  - `mean |A−B|` (the headline value),
  - **5th–95th percentile** spread band,
  - std (stashed in sidecar).

### Real overlay
- Compute observed **batch-1 (9) vs batch-2 (18)** `|gap|` per odor using the same
  helpers (folder-substring `batch_1` / `batch_2` selection, fixed columns).
- Approximate empirical p per odor = fraction of the 100 chance splits with
  `chance_gap ≥ real_gap`.
- **Caveat (must be labeled on figure + sidecar):** the chance band uses **10/10**
  groups while the real split is **9/18**. Smaller groups swing more, so the overlay
  is an *approximate* reference, **not** a size-matched permutation test. A future
  `--match-observed-sizes` flag could size-match (9/18) for a strict test; out of
  scope for v1.

### Outputs
- New subfolder: `Results/Figures/EB-24-1_random_split_null/`.
- Figure 1 `...chance_gap_only.png` — per odor: mean `|A−B|` bar + 5–95% whisker.
- Figure 2 `...chance_gap_with_real_overlay.png` — same, plus red marker at the real
  batch-1-vs-batch-2 `|gap|` per odor and its approximate p annotation.
- JSON sidecar with per-odor: mean, p05, p95, std, real_gap, empirical_p, n_splits,
  group_size, seed, pool size, and the size-mismatch caveat string.
- Both PNGs at 300 dpi. Trained odor (EB) bolded on x-axis via `_style_trained_xticks`.

## Architecture

New script: `scripts/analysis/reaction_rate_random_split_null.py` (permanent, argparse).

Reuses tested helpers (no reimplementation of reaction logic):
- Loading/cleaning: mirror `prep()` from the reference driver — `read_table`,
  `_filter_trial_types(allowed=("testing",))`, `_normalise_fly_columns`,
  `_resolve_non_reactive_mask`, `_canon_dataset`, `_normalise_trial_label`,
  `_trial_num`, `set_protocol("v2")`, config `odor_remap` via `set_dataset_odor_remap`.
- Matrix/rates: `_build_during_matrix`, `_rates_from_matrix`.
- Styling: `_RC_CONTEXT`, `_style_trained_xticks`, `_trained_label`, `DISPLAY_LABEL`,
  `_matrix_title`.

New code (small, focused):
- `sample_split(pool, group_size, rng)` → (A_pairs, B_pairs).
- `gaps_for_split(df_ctrl, A, B, cols, ...)` → per-odor `|rate_A − rate_B|` array,
  selecting flies by `(fly, fly_number)` membership (not folder substring).
- `real_batch_gap(df_ctrl, cols, ...)` → per-odor real `|gap|`.
- `aggregate(gaps_matrix)` → mean, p05, p95, std per odor.
- `plot_chance_gap(..., overlay=None)` → one figure; called twice.

## Testing (test-first, per project rule)

Before wiring the figure, a standalone check validates the reused helpers on this data:
1. Cleaned pool has **27** unique `(fly, fly_number)` control flies.
2. Fixed columns = the expected **8** odors including **EB1** and **EB2**.
3. One random 10/10 split yields per-odor rates within **[0, 100]** and gaps within
   `[0, 100]`.
4. Batch-1 = 9 and batch-2 = 18 fly counts reproduce (matches sidecar of the
   reference figure).

Only after these pass do we build the plotting + aggregation.

## Non-Goals (YAGNI)
- No score-variant (ordinal `score`) figure — reaction rate (binary `prediction`) only.
- No per-fly heatmap (`reaction_matrix_pair_*`) — single averaged summary only.
- No size-matched (9/18) strict permutation test in v1 (flag noted for future).
- No signed-difference or grouped-A-vs-B-bars figure (rejected during brainstorming;
  signed mean ≈ 0 by symmetry, grouped bars ≈ identical).
