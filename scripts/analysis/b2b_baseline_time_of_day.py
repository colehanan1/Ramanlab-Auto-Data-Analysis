#!/usr/bin/env python3
"""Pre-exposure baseline and time-of-day sets for the b2b (IAA odor-pair) folders.

The b2b counterparts of ``Results/Sensitiviy-Results(11-15)``'s
``Pre-Exposure-Baseline/`` and ``Time-of-Day_2h-bins_whole-dataset/`` (user 2026-11-17),
drawn by ``b2b_pair_figures.py`` into each clock-window folder:

* ``Pre-Exposure-Baseline/``: the naive pre-test (every odor alone, before the pair
  block) of the folder's flies by group (IAA-first / IAA-second), by rig and group x
  rig, as % responding (``pre_exposure_baseline.png``) and mean score
  (``pre_exposure_baseline_score.png``), both drawn here.
* ``Time-of-Day_<h>h-bins_whole-dataset/<dataset>/``: EVERY fly of the dataset, all
  clock windows together, binned by the start of its first pre-test trial:

  - ``time_of_day_pretest.png`` (+ ``pretest_pooled_by_bin.csv``, ``pretest_stats.csv``,
    ``time_of_day_table.csv``): time_of_day_trend's naive pre-test figure and tests;
  - ``time_of_day_pretest_by_odor.png`` (+ ``pretest_by_odor_by_bin.csv``): the same
    naive pre-test one panel per odor (user 2026-11-20), with each odor's p from
    ``pretest_stats.csv``;
  - ``time_of_day_test_block.png`` (+ ``test_block_by_bin.csv``,
    ``test_block_stats.csv``): the scored odor naive vs in the pair block (odor 2, both
    presentations pooled: the b2b ``*_scores_pooled`` comparison), per bin.

  time_of_day_trend's change / post-training / trained-odor figures read one pre and
  one post-training trial per fly and odor, which a pair block does not have; they are
  not drawn. Starvation (the strip under each figure) is each batch's target from its
  ``session_metadata.txt``.

Group, rig and start time are BATCH properties (stats audit 2026-11-17, user
2026-11-18): a batch is one batch folder (the ``fly`` column, ``november_13_batch_1_rig_3``),
whose flies share a day, a rig and a vial. So every group / rig / start-time test here
permutes whole batch folders (:mod:`b2b_stats`): group and rig within the session (the
folder name without ``_rig_N``; a session's rigs start together), start bins and the
start-time trend across all of the dataset's batches. Each figure prints only
Holm-adjusted p and says "exploratory · Holm-adjusted over N tests"; the old fly-level
tests stay as secondary ``*fly_level*`` CSV columns.
"""
from __future__ import annotations

import itertools
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import scripts.analysis.starvation_pre_post_comparison as spc  # noqa: E402
from fbpipe.analysis import fly_stats as fs  # noqa: E402
from scripts.analysis import b2b_stats as bs  # noqa: E402
from scripts.analysis import sensitivity_window_results as swres  # noqa: E402
from scripts.analysis import time_of_day_trend as tod  # noqa: E402
from scripts.analysis.experiment_start_times import target_hours  # noqa: E402

BASELINE_DIR = "Pre-Exposure-Baseline"
_RIG = re.compile(r"_rig_(\d+)$", re.IGNORECASE)
#: What every p on these figures is (with the family size appended).
EXPLORATORY = "exploratory · Holm-adjusted over"


def time_of_day_dir(width: float) -> str:
    return f"Time-of-Day_{width:g}h-bins_whole-dataset"


def rig_of(folder: str) -> int:
    """``..._rig_3`` -> 3; a batch folder without a rig tail is rig 1."""
    m = _RIG.search(str(folder))
    return int(m[1]) if m else 1


def _scored(df: pd.DataFrame) -> pd.DataFrame:
    s = pd.to_numeric(df["score"], errors="coerce")
    return df.loc[s.notna()].assign(score=s[s.notna()].astype(float))


# ── batch-level label permutation ────────────────────────────────────────────
#: Strata up to this many batches go to bs.batch_label_perm as they are (it lists every
#: ordering, n!: 9! = 0.36 M, instant); a larger one is enumerated here.
BS_MAX_STRATUM = 9


def _distinct_arrangements(codes: Sequence[int]) -> np.ndarray:
    """Every distinct ordering of a multiset of label codes, one row each (multinomial
    many), built level by level from the positions each level can take -- never the n!
    orderings."""
    counts = sorted(Counter(codes).items())
    rows: list[np.ndarray] = []
    row = np.empty(len(codes), dtype=int)

    def place(free: tuple, k: int) -> None:
        if k == len(counts):
            rows.append(row.copy())
            return
        level, c = counts[k]
        for pick in itertools.combinations(free, c):
            row[list(pick)] = level
            taken = set(pick)
            place(tuple(i for i in free if i not in taken), k + 1)

    place(tuple(range(len(codes))), 0)
    return np.array(rows, dtype=int).reshape(len(rows), len(codes))


def label_perm(values, batches, labels: Mapping, strata: Mapping | None = None) -> dict:
    """:func:`b2b_stats.batch_label_perm`, safe for a large stratum.

    That function lists every ordering of a stratum's labels (``itertools.permutations``)
    before it keeps the distinct ones, so its cost grows as n!: 12 batches in one stratum
    take ~18 s, 14 about an hour. The start-bin test has a single stratum holding all of
    a dataset's batches. So: every stratum of at most :data:`BS_MAX_STRATUM` batches ->
    ``bs.batch_label_perm`` itself; otherwise the distinct arrangements are counted
    (multinomial per stratum) and, up to ``bs.MAX_EXACT``, enumerated directly (the same
    exact p, hits / arrangements); beyond it the same statistic (between-group sum of
    squares of the fly-pooled group means) over ``bs.N_MC`` seeded draws, labels
    shuffled within each stratum, p = (hits + 1) / (draws + 1). Same keys returned."""
    s, n, ids = bs._by_batch(values, batches)
    lab = np.array([labels[b] for b in ids], dtype=object)
    strat = np.array([(strata or {}).get(b, "_all") if strata else "_all" for b in ids],
                     dtype=object)
    members = [np.flatnonzero(strat == st) for st in sorted(set(strat.tolist()), key=str)]
    if all(len(idx) <= BS_MAX_STRATUM for idx in members):
        return bs.batch_label_perm(values, batches, labels, strata)
    levels = sorted(set(lab.tolist()), key=str)
    code = {lv: i for i, lv in enumerate(levels)}
    base = np.array([code[x] for x in lab])
    total = 1
    for idx in members:
        c = Counter(base[idx].tolist()).values()
        total *= math.factorial(sum(c)) // math.prod(math.factorial(k) for k in c)
    if total <= bs.MAX_EXACT:
        arr = [_distinct_arrangements(base[idx].tolist()) for idx in members]
        grid = np.meshgrid(*[np.arange(len(a)) for a in arr], indexing="ij")
        rows = np.empty((total, len(ids)), dtype=int)
        for idx, a, g in zip(members, arr, grid):
            rows[:, idx] = a[g.ravel()]
        exact = True
    else:
        rng = np.random.default_rng(bs.SEED)
        rows = np.tile(base, (bs.N_MC, 1))
        for idx in members:
            rows[:, idx] = rng.permuted(rows[:, idx], axis=1)
        exact = False
    obs = bs._between_ss(s, n, base[None, :], range(len(levels)))[0]
    null = bs._between_ss(s, n, rows, range(len(levels)))
    hits = int(np.sum(null >= obs - 1e-9))
    p = hits / len(null) if exact else (hits + 1) / (len(null) + 1)
    return {"n_flies": int(n.sum()), "n_batches": len(ids),
            "means": {lv: float(s[lab == lv].sum() / n[lab == lv].sum()) for lv in levels},
            "p": float(min(1.0, p)), "exact": exact, "n_perm": int(len(null))}


def _holm_n(p) -> int:
    """How many tests a Holm family holds (NaN = no test, not counted)."""
    return int(np.isfinite(np.asarray(p, dtype=float)).sum())


def _over(n: int) -> str:
    return f"{EXPLORATORY} {n} test{'s' if n != 1 else ''}"


# ── Pre-Exposure-Baseline ────────────────────────────────────────────────────
def baseline_long(pre: pd.DataFrame, names: Mapping[str, str], threshold: float) -> pd.DataFrame:
    """b2b ``pretest_records`` -> the fly x odor table the baseline functions read
    (cohort, odor, rig, batch, fly_id, score_pre, responded_pre). ``batch`` is the batch
    folder: the unit of the group and rig tests."""
    p = _scored(pre)
    return pd.DataFrame({
        "cohort": p["dataset"].map(names).to_numpy(),
        "odor": p["odor"].to_numpy(),
        "rig": [rig_of(f) for f in p["fly"]],
        "batch": p["fly"].astype(str).to_numpy(),
        "fly_id": (p["dataset"].astype(str) + "|" + p["fly"].astype(str) + "|"
                   + p["fly_number"].astype(str)).to_numpy(),
        "score_pre": p["score"].to_numpy(dtype=float),
        "responded_pre": (p["score"] >= threshold).to_numpy(),
    })


BASELINE_PHASE = "Naive pre-test"
#: metric -> (value, interval low, interval high, drawn p, its Holm family size,
#: y label, whiskers)
BASELINE_METRICS = {
    "pct": ("pct_responding", "ci95_lo", "ci95_hi", "p_rate_batch_holm", "holm_n_rate",
            "% responding", "95% Wilson interval, flies as the unit (All odors: at the "
            "fly-cluster effective n)"),
    "score": ("mean_score_pre", "score_ci95_lo", "score_ci95_hi", "p_score_batch_holm",
              "holm_n_score", "mean score", "95% t interval, flies as the unit (All odors: "
              "fly-cluster robust)"),
}
BASELINE_STATS_COLS = [
    "compared", "odor", "n_groups", "n_flies", "n_batches", "n_trials",
    "p_rate_batch", "p_rate_batch_holm", "holm_n_rate",
    "p_score_batch", "p_score_batch_holm", "holm_n_score", "perm_exact", "n_perm",
    "fly_level_p_rate_ffh", "fly_level_p_score_kw"]


def baseline_table(long: pd.DataFrame, by: str | list[str]) -> pd.DataFrame:
    """% responding and mean score per group x odor, plus an all-odor row per group
    (the columns of sensitivity_window_results.baseline_table).

    Point estimates pool trials (responses or scores / trials), never a mean of the
    per-odor values. Intervals take the fly as the unit (fs.fly_cluster_rate_ci,
    fs.fly_cluster_mean_ci): a single odor, one trial per fly, gets the plain Wilson / t
    interval; the all-odor row, where every fly gave one trial per odor, the
    fly-cluster one, so a fly's trials are not counted as that many flies."""
    by = [by] if isinstance(by, str) else list(by)
    rows = []
    for key, g in long.groupby(by, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        for odor, go in [*g.groupby("odor", sort=True), (swres.ALL_ODORS, g)]:
            pct, lo, hi = fs.fly_cluster_rate_ci(go["responded_pre"].to_numpy(dtype=bool),
                                                 go["fly_id"].to_numpy())
            mean, slo, shi = fs.fly_cluster_mean_ci(go["score_pre"].to_numpy(dtype=float),
                                                    go["fly_id"].to_numpy())
            rows.append({**dict(zip(by, key)), "odor": odor,
                         "n_flies": int(go["fly_id"].nunique()), "n_trials": len(go),
                         "n_responding": int(go["responded_pre"].sum()),
                         "pct_responding": round(pct, 2), "ci95_lo": round(lo, 2),
                         "ci95_hi": round(hi, 2), "mean_score_pre": round(mean, 4),
                         "score_ci95_lo": round(slo, 4), "score_ci95_hi": round(shi, 4)})
    return pd.DataFrame(rows)


def _batch_labels(long: pd.DataFrame, by: str) -> dict:
    """batch folder -> its group (``by``); a folder in two groups is an error."""
    pairs = long[["batch", by]].drop_duplicates()
    split = sorted(set(pairs.loc[pairs["batch"].duplicated(keep=False), "batch"]))
    if split:
        raise ValueError(f"batch folders in more than one {by}: {', '.join(split)}")
    return dict(zip(pairs["batch"].astype(str), pairs[by]))


def baseline_stats(long: pd.DataFrame, by: str) -> pd.DataFrame:
    """Do the groups differ before the pair block? ``by``: "cohort" (IAA-first vs
    IAA-second) or "rig". One row per odor, all odors first.

    ``p_rate_batch`` / ``p_score_batch``: whole batch folders permuted within their
    session (:func:`label_perm`, strata = session): the group test keeps each session's
    number of folders per group, the rig test shuffles the rig labels among a session's
    folders. Statistic: the between-group sum of squares of the trial-pooled group
    values the figure plots (responded 0/1; score); the all-odor row takes every trial.
    ``*_holm``: Holm over this panel's rows (the odors and all odors; ``holm_n_*``
    tests). ``fly_level_p_rate_ffh`` / ``fly_level_p_score_kw``: the former fly-as-unit
    tests (Fisher-Freeman-Halton, Kruskal-Wallis; all odors: one value per fly), kept
    as secondary columns only: flies of a batch share a day, a rig and a vial."""
    labels = _batch_labels(long, by)
    strata = {b: bs.session_of(b) for b in labels}
    groups = sorted(long[by].dropna().unique())
    flies = (long.groupby([by, "fly_id"])
             .agg(any_resp=("responded_pre", "any"), score=("score_pre", "mean"))
             .reset_index())
    rows = []
    for odor, go in [(swres.ALL_ODORS, long), *long.groupby("odor", sort=True)]:
        present = [g for g in groups if (go[by] == g).any()]
        fl, rate_col, score_col = ((flies, "any_resp", "score") if odor == swres.ALL_ODORS
                                   else (go, "responded_pre", "score_pre"))
        tested = len(present) > 1
        rate = label_perm(go["responded_pre"].to_numpy(dtype=float), go["batch"], labels, strata)
        score = label_perm(go["score_pre"].to_numpy(dtype=float), go["batch"], labels, strata)
        rows.append({
            "compared": by, "odor": odor, "n_groups": len(present),
            "n_flies": int(go["fly_id"].nunique()), "n_batches": int(go["batch"].nunique()),
            "n_trials": len(go), "p_rate_batch": rate["p"], "p_score_batch": score["p"],
            "perm_exact": bool(score["exact"]), "n_perm": int(score["n_perm"]),
            "fly_level_p_rate_ffh": spc._ffh([fl.loc[fl[by] == g, rate_col].to_numpy()
                                              for g in present]) if tested else None,
            "fly_level_p_score_kw": spc._kw([fl.loc[fl[by] == g, score_col].to_numpy(dtype=float)
                                             for g in present]) if tested else None})
    out = pd.DataFrame(rows)
    for m in ("rate", "score"):
        out[f"p_{m}_batch_holm"] = fs.holm_adjust(out[f"p_{m}_batch"].to_numpy(dtype=float))
        out[f"holm_n_{m}"] = _holm_n(out[f"p_{m}_batch"])
    return out[BASELINE_STATS_COLS]


def baseline_tables(long: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The CSVs of ``Pre-Exposure-Baseline/`` by file name."""
    return {
        "pre_exposure_by_cohort.csv": baseline_table(long, "cohort"),
        "pre_exposure_by_rig.csv": baseline_table(long, "rig"),
        "pre_exposure_by_cohort_x_rig.csv": baseline_table(long, ["cohort", "rig"]),
        "pre_exposure_stats.csv": pd.concat([baseline_stats(long, "cohort"),
                                             baseline_stats(long, "rig")], ignore_index=True),
    }


def score_grid(long: pd.DataFrame, cohorts: Sequence[str], metric: str = "score"):
    """(values, cell notes, column names) of the group x rig grid, all odors pooled over
    trials, with an "All rigs" column. ``metric="score"``: mean score (sum of trial
    scores / trials); ``"pct"``: % of trials responding."""
    rigs = sorted(int(r) for r in long["rig"].dropna().unique())
    cols = [f"Rig {r}" for r in rigs] + ["All rigs"]
    vals = np.full((len(cohorts), len(cols)), np.nan)
    notes = [["no flies"] * len(cols) for _ in cohorts]
    for i, c in enumerate(cohorts):
        for j, r in enumerate([*rigs, None]):
            g = long[(long["cohort"] == c) & ((long["rig"] == r) if r is not None else True)]
            if len(g):
                n = g["fly_id"].nunique()
                if metric == "pct":
                    vals[i, j] = 100.0 * float(g["responded_pre"].sum()) / len(g)
                    notes[i][j] = f"{vals[i, j]:.0f}%\n{n} flies"
                else:
                    vals[i, j] = float(g["score_pre"].sum() / len(g))
                    notes[i][j] = f"{vals[i, j]:.2f}\n{n} flies"
    return vals, notes, cols


def p_label(p) -> str:
    """A Holm-adjusted p as drawn above an odor: "" for no test, never a star."""
    try:
        p = float(p)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(p):
        return ""
    return "p<0.001" if p < 0.001 else f"p={p:.3f}" if p < 0.1 else f"p={p:.2f}"


def _baseline_panel(ax, table: pd.DataFrame, key: str, groups: Sequence, odors: Sequence[str],
                    style: Mapping, pvals: Mapping, overall: pd.DataFrame, title: str,
                    metric: str) -> None:
    """One baseline panel (sensitivity_window_results._baseline_panel's layout): per odor
    each group's value and interval, the all-flies value dashed, ``pvals[odor]`` (the
    Holm-adjusted batch-level p) above, bold below 0.05."""
    value, lo_col, hi_col, *_ = BASELINE_METRICS[metric]
    ylabel = BASELINE_METRICS[metric][5]
    k = len(groups)
    step = 0.78 / max(k, 1)
    by = table.set_index([key, "odor"])
    ov = overall.set_index("odor")
    hi_all = float(pd.concat([table[hi_col], table[value]]).max())
    lo_all = float(pd.concat([table[lo_col], table[value]]).min())
    if metric == "pct":
        bottom, top, pad = 0.0, hi_all + 14, 3.0
    else:
        bottom = min(0.0, lo_all - 0.1)
        span = max(hi_all - bottom, 1.0)
        top, pad = hi_all + 0.2 * span, 0.05 * span
    for i, odor in enumerate(odors):
        if i % 2:
            ax.axvspan(i - 0.5, i + 0.5, color="#f4f3f0", zorder=0, lw=0)
        if odor in ov.index:
            ref = float(ov.loc[odor, value])
            ax.plot([i - 0.42, i + 0.42], [ref, ref], color=swres.INK_2, lw=1.2,
                    ls=(0, (3, 2)), zorder=1)
        text = p_label(pvals.get(odor))
        ax.text(i, top - pad, text, ha="center", va="top", fontsize=9, color=swres.INK,
                fontweight="bold" if text and float(pvals[odor]) < 0.05 else "normal")
        for j, g in enumerate(groups):
            if (g, odor) not in by.index:
                continue
            r = by.loc[(g, odor)]
            x = i + (j - (k - 1) / 2) * step
            colour, marker = style[g]
            if np.isfinite(r[lo_col]) and np.isfinite(r[hi_col]):
                ax.plot([x, x], [r[lo_col], r[hi_col]], color=colour, lw=1.4, alpha=0.6,
                        zorder=2, solid_capstyle="round")
            ax.plot(x, r[value], marker=marker, ms=8, color=colour, mec="white", mew=1.0,
                    zorder=3)
    if bottom < 0:
        ax.axhline(0, color=swres.INK_2, lw=0.8, zorder=1)
    ax.axvline(len(odors) - 1.5, color=swres.INK_2, lw=0.8)
    ax.set_xlim(-0.5, len(odors) - 0.5)
    ax.set_ylim(bottom, top)
    ax.set_xticks(range(len(odors)))
    ax.set_xticklabels([f"{o}\n(pooled trials)" if o == swres.ALL_ODORS else o for o in odors],
                       fontsize=10, color=swres.INK)
    ax.set_ylabel(f"{BASELINE_PHASE}\n{ylabel}", fontsize=10, color=swres.INK)
    ax.set_title(title, loc="left", fontsize=11.5, color=swres.INK, pad=26)
    ax.yaxis.grid(True, color=swres.GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(swres.INK_2)
    ax.tick_params(colors=swres.INK_2)


def _family_note(holm_n: Mapping[str, int]) -> str:
    """How many tests each panel's Holm family holds, as the caption says it."""
    na, nb = holm_n["cohort"], holm_n["rig"]
    if na and nb:
        return (f"{_over(na)} per panel" if na == nb
                else f"{EXPLORATORY} {na} tests in A and {nb} in B")
    if na or nb:
        letter, other = ("A", "B") if na else ("B", "A")
        return f"{_over(na or nb)} in {letter} ({other}: one group, no test)"
    return "no test (one group per panel)"


def baseline_figure(long: pd.DataFrame, cohorts: Sequence[str], tables: Mapping, title: str,
                    *, metric: str = "pct"):
    """(A) by group, (B) by rig: % responding (``metric="pct"``) or mean score
    (``"score"``) per odor and all odors pooled, the batch-level Holm-adjusted p above
    each odor; (C) group x rig, all odors pooled. Returns the figure (not saved)."""
    from matplotlib.lines import Line2D

    whiskers = BASELINE_METRICS[metric][6]
    p_col, n_col = BASELINE_METRICS[metric][3:5]
    present = [c for c in cohorts if c in set(long["cohort"])]
    rigs = sorted(int(r) for r in long["rig"].dropna().unique())
    odors = sorted(set(long["odor"])) + [swres.ALL_ODORS]
    overall = baseline_table(long.assign(_all=1), "_all")
    colours = swres.cohort_colours(list(cohorts))
    cstyle = {c: (colours[c], swres.COHORT_MARKERS[list(cohorts).index(c)
                                                   % len(swres.COHORT_MARKERS)])
              for c in present}
    rstyle = {r: (swres.RIG_PALETTE.get(r, swres.INK_2), swres.RIG_MARKERS.get(r, "o"))
              for r in rigs}
    stats = tables["pre_exposure_stats.csv"]
    fig = plt.figure(figsize=(max(11.0, 1.5 * len(odors) + 2.5), 13.5))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 0.62], hspace=0.5)
    ax1, ax2, ax3 = (fig.add_subplot(gs[i]) for i in range(3))
    holm_n = {}
    for ax, key, groups, style, letter, n, label in (
            (ax1, "cohort", present, cstyle, "A  By group",
             long.groupby("cohort")["fly_id"].nunique(), lambda g: g),
            (ax2, "rig", rigs, rstyle, "B  By rig",
             long.groupby("rig")["fly_id"].nunique(), lambda g: f"Rig {g}")):
        st = stats[stats["compared"] == key]
        holm_n[key] = int(st[n_col].max()) if len(st) else 0
        _baseline_panel(ax, tables[f"pre_exposure_by_{key}.csv"], key, groups, odors, style,
                        dict(zip(st["odor"], st[p_col])), overall, letter, metric)
        ax.legend(handles=[Line2D([], [], ls="none", marker=style[g][1], color=style[g][0],
                                  mec="white", ms=8, label=f"{label(g)} (n={n[g]})")
                           for g in groups]
                  + [Line2D([], [], color=swres.INK_2, lw=1.2, ls=(0, (3, 2)),
                            label="all flies")],
                  loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(groups) + 1,
                  frameon=False, fontsize=9, labelcolor=swres.INK, borderaxespad=0.1)

    vals, notes, cols = score_grid(long, present, metric)
    finite = vals[np.isfinite(vals)]
    if metric == "pct":
        vmin, vmax = 0.0, max(50.0, float(finite.max()) if finite.size else 50.0)
    else:
        vmin = min(0.0, float(finite.min())) if finite.size else 0.0
        vmax = max(1.0, float(finite.max())) if finite.size else 1.0
    ax3.imshow(np.ma.masked_invalid(vals), cmap=plt.get_cmap("Blues"), vmin=vmin,
               vmax=vmax * 1.25, aspect="auto")
    for i in range(len(present)):
        for j in range(len(cols)):
            v = vals[i, j]
            dark = np.isfinite(v) and (v - vmin) / (vmax * 1.25 - vmin) > 0.55
            ax3.text(j, i, notes[i][j], ha="center", va="center", fontsize=9,
                     color="white" if dark else (swres.INK if np.isfinite(v) else swres.INK_2))
    ax3.axvline(len(cols) - 1.5, color="white", lw=3)
    ax3.set_xticks(range(len(cols)))
    ax3.set_xticklabels(cols, fontsize=10, color=swres.INK)
    ax3.set_yticks(range(len(present)))
    ax3.set_yticklabels(present, fontsize=10, color=swres.INK)
    ax3.tick_params(length=0)
    for side in ax3.spines.values():
        side.set_visible(False)
    what = ("% of the fly x odor trials responding" if metric == "pct"
            else "mean score of the fly x odor trials")
    ax3.set_title(f"C  Group x rig, all odors pooled ({what})", loc="left", fontsize=11.5,
                  color=swres.INK, pad=6)
    fig.suptitle(title, fontsize=13, color=swres.INK, y=0.995)
    fig.text(0.5, 0.975,
             "Before the pair block every fly is naive and gets the same panel, so these "
             f"should match. Whiskers: {whiskers}. Dashed line: all flies together.\n"
             f"p above each odor: batch level, batches permuted within session · "
             f"{_family_note(holm_n)}\n"
             "(A: group labels shuffled among each session's folders, each session keeping "
             "its count per group; B: rig labels shuffled among each session's folders. "
             "Fly-level tests: pre_exposure_stats.csv)",
             ha="center", va="top", fontsize=9, color=swres.INK_2, linespacing=1.4)
    fig.subplots_adjust(top=0.9, bottom=0.04, left=0.17, right=0.98)
    return fig


def plot_baseline(long: pd.DataFrame, cohorts: Sequence[str], tables: Mapping, title: str,
                  path: Path, metric: str = "pct") -> Path:
    """:func:`baseline_figure`, saved to ``path``."""
    fig = baseline_figure(long, cohorts, tables, title, metric=metric)
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    return path


def plot_baseline_score(long: pd.DataFrame, cohorts: Sequence[str], tables: Mapping,
                        title: str, path: Path, metric: str = "score") -> Path:
    """The baseline figure as mean score (``metric="pct"``: % responding)."""
    return plot_baseline(long, cohorts, tables, title, path, metric=metric)


def write_baseline(out: Path, pre: pd.DataFrame, names: Mapping[str, str], *,
                   threshold: float, window: str) -> list[Path]:
    """``Pre-Exposure-Baseline/``: the naive pre-test by group, rig and group x rig, as %
    responding (``pre_exposure_baseline.png``) and as mean score (``..._score.png``), with
    the batch-level tests (``pre_exposure_stats.csv``)."""
    long = baseline_long(pre, names, threshold)
    if long.empty:
        return []
    d = out / BASELINE_DIR
    d.mkdir(parents=True, exist_ok=True)
    tables = baseline_tables(long)
    written = []
    for name, df in tables.items():
        df.to_csv(d / name, index=False)
        written.append(d / name)
    cohorts = [names[k] for k in names if names[k] in set(long["cohort"])]
    started = f": flies started {window}" if window else ""
    head = ("Naive pre-test (before the pair block), IAA-first and IAA-second flies by group "
            f"and rig{started}")
    png = plot_baseline(long, cohorts, tables, f"{head}: % responding",
                        d / "pre_exposure_baseline.png", metric="pct")
    score_png = plot_baseline(long, cohorts, tables, f"{head}: mean score",
                              d / "pre_exposure_baseline_score.png", metric="score")
    return [png, score_png, *written]


# ── Time of day ──────────────────────────────────────────────────────────────
def starvation_of(roots: Sequence[Path], dataset: str, folder: str) -> float | None:
    """The batch's target starvation (h) from the first root holding its metadata."""
    for root in roots:
        h = target_hours(Path(root) / dataset / folder)
        if h is not None:
            return float(h)
    return None


def _starts_map(starts: pd.DataFrame) -> dict:
    return {(str(d), str(f)): t for d, f, t in
            zip(starts["dataset"], starts["folder"], starts["first_pretest"])
            if t is not None and not pd.isna(t)}


def pretest_frame(pre: pd.DataFrame, start_of: Mapping, starvation: Mapping) -> pd.DataFrame:
    """b2b ``pretest_records`` -> the per fly x odor frame time_of_day_trend bins.

    ``score_post`` repeats ``score_pre``: time_of_day_trend's omnibus helper reads it,
    and no table written from here keeps a post column."""
    p = _scored(pre)
    keys = list(zip(p["dataset"].astype(str), p["fly"].astype(str)))
    out = pd.DataFrame({
        "dataset_canon": p["dataset"].to_numpy(), "fly": p["fly"].to_numpy(),
        "fly_number": p["fly_number"].to_numpy(), "odor": p["odor"].to_numpy(),
        "score_pre": p["score"].to_numpy(dtype=float),
        "score_post": p["score"].to_numpy(dtype=float),
        "start_time": [start_of.get(k) for k in keys],
        "starvation_h": [starvation.get(k) for k in keys],
    })
    return out.dropna(subset=["start_time"]).reset_index(drop=True)


def _with_bins(df: pd.DataFrame, start_of: Mapping, width: float) -> pd.DataFrame:
    keys = list(zip(df["dataset"].astype(str), df["fly"].astype(str)))
    df = df.assign(start_time=[start_of.get(k) for k in keys]).dropna(subset=["start_time"])
    bins = [tod.time_bin(pd.Timestamp(t).to_pydatetime(), width) for t in df["start_time"]]
    return df.assign(bin_start=[b[0] for b in bins], bin_label=[b[1] for b in bins],
                     x=[b[0] + width / 2 for b in bins])


def _sem(v) -> float:
    v = np.asarray(v, dtype=float)
    return float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else np.nan


ROLES = (("pre", "naive"), ("post", "pair"))


def test_block_table(recs: pd.DataFrame, start_of: Mapping, *, width: float,
                     threshold: float) -> pd.DataFrame:
    """Per start bin and role ("naive": the scored odor alone in the pre-test, "pair": it
    in the pair block, both presentations): pooled trials (sum / n trials, never a mean
    of fly means), SEM over flies of each fly's own mean, % of trials responding."""
    r = _with_bins(_scored(recs), start_of, width)
    rows = []
    for (start, role), g in r.groupby(["bin_start", "role"]):
        name = dict(ROLES).get(role)
        if name is None:
            continue
        per_fly = g.groupby(["dataset", "fly", "fly_number"])["score"].mean()
        s = g["score"].to_numpy(dtype=float)
        rows.append({"bin_start": start, "bin_label": g["bin_label"].iloc[0],
                     "x": g["x"].iloc[0], "role": name, "n_flies": len(per_fly),
                     "n_trials": len(s), "mean_score": float(s.sum() / len(s)),
                     "sem_score_flies": _sem(per_fly.to_numpy()),
                     "pct_responding": 100.0 * float(np.mean(s >= threshold))})
    cols = ["bin_start", "bin_label", "x", "role", "n_flies", "n_trials", "mean_score",
            "sem_score_flies", "pct_responding"]
    return pd.DataFrame(rows, columns=cols).sort_values(["role", "bin_start"]).reset_index(drop=True)


def fly_changes(recs: pd.DataFrame, start_of: Mapping, width: float) -> pd.DataFrame:
    """One row per fly: its pair-block mean minus its naive mean (the change), its bin."""
    r = _with_bins(_scored(recs), start_of, width)
    m = r.pivot_table(index=["dataset", "fly", "fly_number", "bin_start", "bin_label",
                             "start_time"], columns="role", values="score", aggfunc="mean")
    if not {"pre", "post"} <= set(m.columns):
        return pd.DataFrame(columns=["dataset", "fly", "fly_number", "bin_start", "bin_label",
                                     "start_time", "naive", "pair", "change"])
    m = m.dropna(subset=["pre", "post"]).rename(columns={"pre": "naive", "post": "pair"})
    m["change"] = m["pair"] - m["naive"]
    return m.reset_index()[["dataset", "fly", "fly_number", "bin_start", "bin_label",
                            "start_time", "naive", "pair", "change"]]


TB_PAIRED = "naive vs pair within bin (paired, per fly)"
TB_ACROSS = "change across bins (whole batches permuted)"
TB_TREND = "start clock time vs change (Spearman, one value per batch)"
TB_COLS = ["test", "family", "bin_label", "n_flies", "n_batches", "mean_change", "rho", "p",
           "p_holm", "holm_n", "perm_exact", "rho_fly_level", "p_fly_level"]


def test_block_stats(changes: pd.DataFrame, paired_p: Callable) -> pd.DataFrame:
    """Tests on the per-fly change (pair-block mean minus naive mean), two Holm families.

    ``per-bin paired``: per start bin, ``paired_p`` on the flies' changes; Holm across
    the bins. ``batch level`` (what the figure draws), batches as the unit -- a batch
    folder's flies share its start, so start time is a batch property: across bins,
    :func:`label_perm` with each batch labelled by its start bin (no strata), and the
    trend, :func:`b2b_stats.batch_trend` (Spearman of the batch's start clock time vs
    its fly-pooled change, batches permuted); Holm over those 2. ``rho_fly_level`` /
    ``p_fly_level``: the former fly-as-unit Kruskal-Wallis / Spearman, secondary only."""
    rows = []
    for (start, label), g in changes.groupby(["bin_start", "bin_label"]):
        rows.append({"test": TB_PAIRED, "family": "per-bin paired", "bin_label": label,
                     "n_flies": len(g), "n_batches": int(g["fly"].nunique()),
                     "mean_change": float(g["change"].mean()),
                     "p": paired_p(g["change"].to_numpy(dtype=float))})
    change = changes["change"].to_numpy(dtype=float)
    batch = changes["fly"].astype(str).to_numpy()
    clock = pd.to_datetime(changes["start_time"]).map(tod.clock_hours).to_numpy(dtype=float)
    groups = [changes.loc[changes["bin_start"] == b, "change"].to_numpy(dtype=float)
              for b in sorted(changes["bin_start"].unique())]
    across = label_perm(change, batch, dict(zip(batch, changes["bin_start"])))
    trend = bs.batch_trend(change, batch, dict(zip(batch, clock)))
    rho_f, p_f = tod._spearman(clock, change)
    every = {"family": "batch level", "bin_label": "all", "n_flies": len(changes),
             "n_batches": int(pd.unique(batch).size),
             "mean_change": float(change.mean()) if len(change) else np.nan}
    rows.append({**every, "test": TB_ACROSS, "p": across["p"],
                 "perm_exact": bool(across["exact"]),
                 "p_fly_level": spc._kw(groups) if len(groups) > 1 else np.nan})
    rows.append({**every, "test": TB_TREND, "rho": trend["rho"], "p": trend["p"],
                 "perm_exact": bool(trend["exact"]), "rho_fly_level": rho_f,
                 "p_fly_level": p_f})
    out = pd.DataFrame(rows).reindex(columns=TB_COLS)
    for fam in ("per-bin paired", "batch level"):
        sel = out["family"] == fam
        out.loc[sel, "p_holm"] = fs.holm_adjust(out.loc[sel, "p"].to_numpy(dtype=float))
        out.loc[sel, "holm_n"] = _holm_n(out.loc[sel, "p"])
    return out


def _p(p) -> str:
    p = float(p) if p is not None else np.nan
    return "n/a" if not np.isfinite(p) else ("p<0.001" if p < 0.001 else f"p={p:.3f}")


def test_block_figure(table: pd.DataFrame, comp: pd.DataFrame, stats: pd.DataFrame, *,
                      title: str, width: float, threshold: float, colors: Mapping,
                      labels: Mapping):
    """Mean score (± SEM over flies) and % responding trials per start bin, naive vs in
    the pair block, over the flies-per-bin strip. The stats text: the batch-level tests
    of :func:`test_block_stats` (across bins, trend), Holm-adjusted together."""
    starts = sorted(table["bin_start"].unique())
    with plt.rc_context(tod._RC_CONTEXT):
        fig, (a1, a2, a3) = plt.subplots(
            3, 1, figsize=(max(6.4, 0.55 * len(starts) * width + 3.6), 7.8), sharex=True,
            gridspec_kw={"height_ratios": [3, 3, 1.25]})
        for role, marker in (("naive", "o"), ("pair", "s")):
            t = table[table["role"] == role].sort_values("bin_start")
            if t.empty:
                continue
            c = colors[role]
            a1.errorbar(t["x"], t["mean_score"], yerr=t["sem_score_flies"], fmt="none",
                        ecolor=c, elinewidth=1.0, capsize=3, zorder=3)
            for ax, key in ((a1, "mean_score"), (a2, "pct_responding")):
                tod._line(ax, t, key, width, color=c, linewidth=tod.LINE_W, zorder=4)
                ax.plot(t["x"], t[key], marker, color=c, zorder=4,
                        label=labels[role] if ax is a1 else None, **tod.MARKER)
        for ax in (a1, a2):
            tod._recessive_grid(ax)
        a1.axhline(0.0, **tod.ZERO_LINE)
        lo, hi = a1.get_ylim()
        a1.set_ylim(min(lo, 0.0), hi + 0.4 * (hi - min(lo, 0.0)))    # room for the stats text
        a1.set_ylabel("Mean PER score\n(pooled trials, ± SEM over flies)")
        a2.set_ylabel(f"Responding trials %\n(score ≥ {threshold:g})")
        a2.set_ylim(0, 100)
        a2.set_yticks([0, 25, 50, 75, 100])
        st = stats.set_index("test")
        across, sp = st.loc[TB_ACROSS], st.loc[TB_TREND]
        rho = float(sp["rho"])
        trend = (f" · vs start time Spearman ρ={rho:.2f} {_p(sp['p_holm'])}"
                 if np.isfinite(rho) else " · vs start time: n/a")
        fam = int(np.nan_to_num(float(across["holm_n"])))
        a1.text(0.02, 0.97, f"Change (pair − naive), batch level ({int(across['n_batches'])} "
                "batches; whole batches permuted):\n"
                f"across bins {_p(across['p_holm'])}{trend}\n{_over(fam)}",
                transform=a1.transAxes, ha="left", va="top", fontsize=7.5, color=tod.INK)
        a1.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=8)
        tod._composition_strip(a3, comp, width, legend=True)
        n = table[table["role"] == "pair"].set_index("bin_start")["n_flies"]
        tod._clock_axis(a3, starts, width, n=n if len(n) else None)
        a3.set_xlabel(f"Start of the first pre-test trial (time of day, {width:g} h bins; "
                      "n = flies)")
        paired = stats[stats["test"] == TB_PAIRED]
        k = _holm_n(paired["p"])
        fig.suptitle(f"{title}\nin test_block_stats.csv: the per-bin paired tests (flies, "
                     f"Holm over {k} bins) and the fly-level tests", x=0.01, ha="left",
                     fontsize=10, color=tod.INK)
        fig.tight_layout()
    return fig


def pretest_caption(stats: pd.DataFrame) -> str:
    """The b2b caption of time_of_day_trend's pre-test figure (its own names a figure b2b
    does not draw): what its p are, from the ``pretest_stats`` columns it drew them
    from -- per test, Holm across the single odors; the all-odor row, over its own
    tests."""
    tests = getattr(tod, "PRETEST_TESTS", ())
    cols = [f"{t}_batch" for t in tests]
    if not tests or not all(c in stats and f"{c}_holm" in stats for c in cols):
        return "per-odor tests: pretest_stats.csv · fly level, uncorrected"
    every = stats["odor"] == getattr(tod, "ALL_ODORS", "")
    n_single = max(_holm_n(stats.loc[~every, c]) for c in cols)
    n_all = _holm_n(stats.loc[every, cols].to_numpy(dtype=float).ravel())
    note = getattr(tod, "P_NOTE", "")
    return (f"per-odor tests: pretest_stats.csv · {_over(n_single)} (single odors, per "
            f"test; all odors: over its {n_all})" + (f"\n{note}" if note else ""))


# ── Time of day: the naive pre-test odor by odor ─────────────────────────────
#: Panel order: the scored odor, the six odor partners, then anything else; the click
#: (the Paraffin Oil partner, a master-valve click since 2026-11-10) always last.
ODOR_ORDER = ("IAA", "Hexanol", "Linalool", "Benzaldehyde", "Ethyl Butyrate", "Citral",
              "3-Octanol")
CLICK_ODOR = "Paraffin Oil"
BY_ODOR_COLS = ["bin_start", "bin_label", "x", "odor", "n_flies", "n_trials", "mean_pre",
                "sem_pre_flies", "pct_pre", "pct_ci95_lo", "pct_ci95_hi"]


def odor_order(odors) -> list[str]:
    """:data:`ODOR_ORDER` first, other odors alphabetically, the click last."""
    odors = list(dict.fromkeys(str(o) for o in odors))
    rest = sorted((o for o in odors if o not in ODOR_ORDER and o != CLICK_ODOR),
                  key=str.casefold)
    return ([o for o in ODOR_ORDER if o in odors] + rest
            + ([CLICK_ODOR] if CLICK_ODOR in odors else []))


def odor_title(odor: str) -> str:
    return f"{odor} (valve click, no odor)" if odor == CLICK_ODOR else str(odor)


def pretest_odor_table(binned: pd.DataFrame, *, threshold: float) -> pd.DataFrame:
    """One row per start bin x odor of the naive pre-test (time_of_day_trend's binned
    frame). Each fly smells each odor once, so a cell is that bin's flies; a fly with a
    repeated trial pools its trials and still counts once. Mean and % pool trials
    (never a mean of fly means); SEM over flies of each fly's own mean; the % interval
    is fs.fly_cluster_rate_ci (plain Wilson when every fly gave one trial)."""
    rows = []
    for (start, odor), g in binned.groupby(["bin_start", "odor"], sort=True):
        s = g["score_pre"].to_numpy(dtype=float)
        fly = (g["dataset_canon"].astype(str) + "|" + g["fly"].astype(str) + "|"
               + g["fly_number"].astype(str)).to_numpy()
        per_fly = pd.Series(s).groupby(fly).mean().to_numpy()
        pct, lo, hi = fs.fly_cluster_rate_ci(s >= threshold, fly)
        rows.append({"bin_start": start, "bin_label": g["bin_label"].iloc[0],
                     "x": g["x"].iloc[0], "odor": odor, "n_flies": len(per_fly),
                     "n_trials": len(s), "mean_pre": float(s.sum() / len(s)),
                     "sem_pre_flies": _sem(per_fly), "pct_pre": pct, "pct_ci95_lo": lo,
                     "pct_ci95_hi": hi})
    return pd.DataFrame(rows, columns=BY_ODOR_COLS)


def odor_stat_line(row: pd.Series) -> tuple[str, bool]:
    """One odor's pretest_stats row as drawn: the batch-level, Holm-adjusted p of each
    test (score KW, response FFH, Spearman trend); significant if any is < 0.05."""
    def shown(test):
        v = row.get(f"{test}_batch_holm", np.nan)
        return float(v) if tod._finite(v) else np.nan

    kw, ffh, sp = (shown(t) for t in tod.PRETEST_TESTS)
    rho = row.get("rho_pre_batch", np.nan)
    trend = (f"trend ρ={float(rho):+.2f} {p_label(sp) or 'n/a'}" if tod._finite(rho)
             else f"trend {p_label(sp) or 'n/a'}")
    sig = any(np.isfinite(p) and p < 0.05 for p in (kw, ffh, sp))
    return (f"score {p_label(kw) or 'n/a'} · resp {p_label(ffh) or 'n/a'} · {trend}", sig)


def pretest_odor_figure(table: pd.DataFrame, pooled: pd.DataFrame, stats: pd.DataFrame, *,
                        title: str, width: float, threshold: float, ncols: int = 4):
    """Small multiples, one odor per cell: the naive pre-test mean score (± SEM over flies,
    top) and % responding (95 % interval, flies as the unit, bottom) per start bin, the
    all-odors pooled line dashed behind, and under the odor's name the batch-level,
    Holm-adjusted p already in ``pretest_stats.csv`` (no new tests). Axes are not shared
    (sharex shares the tick formatter): every cell gets the same limits instead."""
    from matplotlib.lines import Line2D

    odors = odor_order(table["odor"].unique())
    starts = sorted(table["bin_start"].unique())
    st = stats.set_index("odor")
    single = stats[stats["odor"] != tod.ALL_ODORS]
    fam = max((_holm_n(single[f"{t}_batch_holm"]) for t in tod.PRETEST_TESTS
               if f"{t}_batch_holm" in single), default=0)
    ncols = max(1, min(ncols, len(odors)))
    nrows = math.ceil(len(odors) / ncols)
    # Scores run -1..5: the axis reaches the lowest mean - SEM (0.25 steps), and a small
    # margin under 0 keeps a 0-score or 0 % marker clear of the frame.
    sem = table["sem_pre_flies"].fillna(0)
    top, low = (table["mean_pre"] + sem).max(), (table["mean_pre"] - sem).min()
    smax = max(2.0, math.ceil(2 * float(top)) / 2) if np.isfinite(top) else 2.0
    smin = min(0.0, math.floor(4 * float(low)) / 4) if np.isfinite(low) else 0.0
    spad = 0.04 * (smax - smin)
    ref = pooled.sort_values("bin_start")
    ink, ctx = tod.EMPHASIS_COLOR, tod.CONTEXT_COLOR
    height = 4.2 * nrows + 1.25
    with plt.rc_context(tod._RC_CONTEXT):
        fig = plt.figure(figsize=(3.4 * ncols + 1.2, height))
        outer = fig.add_gridspec(nrows, ncols, hspace=0.42, wspace=0.14, left=0.08,
                                 right=0.99, top=1 - 1.15 / height, bottom=0.75 / height)
        for i, odor in enumerate(odors):
            r, c = divmod(i, ncols)
            inner = outer[r, c].subgridspec(2, 1, height_ratios=[3, 2], hspace=0.1)
            a_s, a_p = fig.add_subplot(inner[0]), fig.add_subplot(inner[1])
            t = table[table["odor"] == odor].sort_values("bin_start")
            for ax, key in ((a_s, "mean_pre"), (a_p, "pct_pre")):
                x, y = tod._gap_break(ref, key, width)
                ax.plot(x, y, color=ctx, lw=1.3, ls=(0, (3, 2)), zorder=2)
            a_s.errorbar(t["x"], t["mean_pre"], yerr=t["sem_pre_flies"], fmt="none",
                         ecolor=ink, elinewidth=1.0, capsize=2.5, zorder=3)
            a_p.vlines(t["x"], t["pct_ci95_lo"], t["pct_ci95_hi"], color=ink, lw=1.0,
                       alpha=0.4, zorder=3)
            for ax, key in ((a_s, "mean_pre"), (a_p, "pct_pre")):
                tod._line(ax, t, key, width, color=ink, linewidth=tod.LINE_W, zorder=4)
                ax.plot(t["x"], t[key], "o", color=ink, zorder=5, **tod.MARKER)
                tod._recessive_grid(ax)
            a_s.set_ylim(smin - spad, smax)
            if smin < 0:
                a_s.axhline(0.0, **tod.ZERO_LINE)
            a_p.set_ylim(-4, 100)
            a_p.set_yticks([0, 25, 50, 75, 100])
            tod._clock_axis(a_s, starts, width)
            tod._clock_axis(a_p, starts, width, n=t.set_index("bin_start")["n_flies"])
            a_s.tick_params(axis="x", labelbottom=False)
            for ax in (a_s, a_p):
                ax.tick_params(axis="both", labelsize=7.5)
                if c:
                    ax.tick_params(axis="y", labelleft=False)
            line, sig = odor_stat_line(st.loc[odor]) if odor in st.index else ("no test", False)
            a_s.set_title(odor_title(odor), loc="left", fontsize=10.5, color=tod.INK, pad=15,
                          fontweight="bold" if sig else "normal")
            a_s.text(0.0, 1.02, line, transform=a_s.transAxes, ha="left", va="bottom",
                     fontsize=7.3, color=tod.INK, fontweight="bold" if sig else "normal")
            if c == 0:
                a_s.set_ylabel("Naive pre-test\nmean PER score", fontsize=8.5)
                a_p.set_ylabel(f"% responding\n(score ≥ {threshold:g})", fontsize=8.5)
        fig.supxlabel(f"Start of the first pre-test trial (time of day, {width:g} h bins; "
                      "n = flies under each odor)", fontsize=9.5, color=tod.INK, y=0.005)
        fig.legend(handles=[
            Line2D([], [], color=ink, lw=tod.LINE_W, marker="o", **tod.MARKER,
                   label="this odor: mean ± SEM over flies; % with its 95 % interval"),
            Line2D([], [], color=ctx, lw=1.3, ls=(0, (3, 2)),
                   label="all odors pooled (time_of_day_pretest.png)")],
            loc="upper right", bbox_to_anchor=(0.995, 0.995), frameon=False, fontsize=8)
        fig.suptitle(
            f"{title}: naive pre-test odor by odor, by start time\n"
            "each fly smells each odor once before the pair block, so a point is that "
            "bin's flies\n"
            "p under each odor (pretest_stats.csv): whole batches permuted between bins "
            "(score KW, response FFH); trend = Spearman on one value per batch\n"
            f"{_over(fam)} (per test, across the {len(single)} odors)",
            x=0.01, y=0.998, ha="left", va="top", fontsize=9.5, color=tod.INK,
            linespacing=1.35)
    return fig


def write_time_of_day(out: Path, pre: pd.DataFrame, recs: pd.DataFrame, starts: pd.DataFrame,
                      roots: Sequence[Path], names: Mapping[str, str], heads: Mapping[str, str],
                      *, threshold: float, paired_p: Callable, colors: Mapping,
                      labels: Mapping, width: float = 2.0) -> list[Path]:
    """``Time-of-Day_<h>h-bins_whole-dataset/<dataset>/`` from every fly given."""
    start_of = _starts_map(starts)
    folders = {(str(d), str(f)) for d, f in zip(starts["dataset"], starts["folder"])}
    starvation, unknown = {}, []
    for d, f in sorted(folders):
        h = starvation_of(roots, d, f)
        if h is None:
            unknown.append(f"{d}/{f}")
        starvation[(d, f)] = 0.0 if h is None else h
    if unknown:
        print(f"[b2b][WARN] no session_metadata.txt starvation target, shown as 0 h: "
              f"{', '.join(unknown)}")
    written = []
    for ds, short in names.items():
        frame = pretest_frame(pre[pre["dataset"] == ds], start_of, starvation)
        if frame.empty:
            continue
        d = out / time_of_day_dir(width) / ds
        d.mkdir(parents=True, exist_ok=True)
        binned = tod.assign_bins(frame, width)
        table = tod.bin_table(binned, binary_threshold=threshold)
        comp = tod.composition(binned)
        pooled = tod.pooled_pretest(binned, binary_threshold=threshold)
        stats = tod.pretest_stats(binned, binary_threshold=threshold)
        title = f"{short} ({ds}): every fly, by start time"
        fig = tod.build_pretest(table, pooled, comp, stats=stats, title=title, width=width)
        fig.axes[0].set_ylabel("Naive pre-test PER score\n(before the pair block)")
        fig._suptitle.set_text(f"{title} — is the naive pre-test different by time of "
                               f"day?\n{pretest_caption(stats)}")
        written.append(tod._save(fig, d / "time_of_day_pretest.png"))
        for name, df in (("time_of_day_table.csv",
                          table[["bin_start", "bin_label", "odor", "n_flies",
                                 "starvation_groups_h", "mean_pre", "pct_pre"]]),
                         ("pretest_pooled_by_bin.csv", pooled), ("pretest_stats.csv", stats),
                         ("flies_per_bin_by_starvation.csv", comp.reset_index())):
            df.to_csv(d / name, index=False)
            written.append(d / name)
        by_odor = pretest_odor_table(binned, threshold=threshold)
        fig = pretest_odor_figure(by_odor, pooled, stats, title=f"{short} ({ds})",
                                  width=width, threshold=threshold)
        written.append(tod._save(fig, d / "time_of_day_pretest_by_odor.png"))
        by_odor.to_csv(d / "pretest_by_odor_by_bin.csv", index=False)
        written.append(d / "pretest_by_odor_by_bin.csv")

        r = recs[recs["dataset"] == ds]
        tb = test_block_table(r, start_of, width=width, threshold=threshold)
        if tb.empty:
            continue
        ch = fly_changes(r, start_of, width)
        st = test_block_stats(ch, paired_p)
        fig = test_block_figure(tb, comp, st, title=f"{short}: {heads[ds]}, by start time "
                                f"(every fly)", width=width, threshold=threshold,
                                colors=colors, labels=labels)
        written.append(tod._save(fig, d / "time_of_day_test_block.png"))
        for name, df in (("test_block_by_bin.csv", tb), ("test_block_stats.csv", st),
                         ("test_block_per_fly.csv", ch)):
            df.to_csv(d / name, index=False)
            written.append(d / name)
    return written
