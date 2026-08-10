"""Compare RandomPanel odorant responses across concentrations (10, 1, 0.1).

For each odorant presented in the RandomPanel testing protocol this script
compares the response at the three delivered concentrations (10, 1, 0.1) and
tests whether the response differs across concentration with Fisher's exact
test:

* **Mean score** — grouped bar chart (mean ordinal score +/- SEM), with an
  omnibus Fisher-Freeman-Halton exact test on the (concentration x score-level)
  contingency table per odorant.
* **% reaction** — grouped bar chart (fraction of trials with score >= 2, the
  reaction boundary, +/- Wilson 95% CI), with an omnibus Fisher-Freeman-Halton
  exact test on the (concentration x reacted/not) contingency table per odorant.

All pairwise concentration comparisons (0.1 vs 1, 1 vs 10, 0.1 vs 10) are
written to ``randompanel_conc_comparison_stats.csv`` alongside the omnibus
results, but **only significant pairs (p < 0.05) are bracketed on the figure** —
non-significant comparisons are omitted rather than labelled "ns", and the
surviving brackets pack downward so no empty rows are left behind.

Figures are written publication-ready: 300-dpi PNG plus PDF and SVG with live
(editable) text.

Both exposures of each odorant ("Name 1"/"Name 2") are pooled, so each
odorant/concentration cell aggregates 2 presentations x N flies trials.

Usage::

    python scripts/analysis/randompanel_conc_comparison.py \
        --csv-path /path/to/model_predictions.csv \
        --out-dir  /path/to/score_summary/GR5a-Old \
        --config   config/config_new.yaml
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from itertools import combinations
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.stats import fisher_exact, kruskal, mannwhitneyu

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.score_summary import _load_scores
from scripts.analysis.envelope_visuals import set_protocol, should_write

# Map the three RandomPanel testing datasets to their delivered concentration.
CONC_BY_DATASET = {
    "RandomPanel-24-0.1": 0.1,
    "RandomPanel-24-1": 1.0,
    "RandomPanel-Training-24-10": 10.0,
}
CONC_ORDER = [0.1, 1.0, 10.0]  # ascending — used for stat coding / pairwise labels
CONC_PLOT_ORDER = [10.0, 1.0, 0.1]  # high -> low, left to right on the plot

# Sequential single-hue ramp (ColorBrewer Blues, colourblind-safe) for the
# ORDERED concentration variable: light (low conc) -> dark (high conc).
CONC_COLOR = {
    0.1: "#9ecae1",
    1.0: "#4292c6",
    10.0: "#08306b",
}

REACTION_BOUNDARY = 2  # score >= 2 counts as a reaction (matches score_summary)

ALPHA = 0.05  # brackets are drawn only for pairs below this

# Publication styling: single-column-plus width, 7-9 pt type, hairline spines,
# and vector output whose text stays editable in Illustrator/Inkscape.
_RC_CONTEXT = {
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.fontsize": 8,
    "font.size": 8,
    "font.family": "Arial",
    "font.sans-serif": ["Arial"],
    "pdf.fonttype": 42,   # TrueType -> editable text in the PDF
    "ps.fonttype": 42,
    "svg.fonttype": "none",  # keep <text> elements instead of outlining glyphs
}


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def _sig_stars(p: float) -> str:
    if p is None or np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _fmt_p(p: float) -> str:
    """Bracket label: stars + p-value (only significant pairs are labelled)."""
    if p is None or np.isnan(p):
        return ""
    ptxt = "p<0.001" if p < 0.001 else f"p={p:.3f}"
    stars = _sig_stars(p)
    return f"{stars} {ptxt}" if stars and stars != "ns" else ptxt


# Pairwise concentration brackets, ordered narrow -> wide so that the outer
# (10 vs 0.1) span always stacks above the adjacent ones. ``tag`` matches the
# column naming built by ``_compute_stats`` (ascending "loVhi", float repr).
_PAIR_BRACKETS = [
    (10.0, 1.0, "1.0v10.0"),   # adjacent left  (10 vs 1)
    (1.0, 0.1, "0.1v1.0"),     # adjacent right (1 vs 0.1)
    (10.0, 0.1, "0.1v10.0"),   # outer          (10 vs 0.1)
]


def significant_brackets(
    row, measure: str, alpha: float = ALPHA
) -> list[tuple[float, float, float, int]]:
    """Pairs worth bracketing for one odorant: ``(conc_a, conc_b, q, level)``.

    Reads the Holm-adjusted post-hoc columns, which are NaN whenever the omnibus
    did not survive correction — so an unprotected odorant is never bracketed.
    Non-significant (and missing/NaN) comparisons are dropped outright; their
    p-values stay in the stats CSV. Levels are assigned consecutively over the
    survivors so the stack never carries a gap where an "ns" bracket used to be.
    """
    out: list[tuple[float, float, float, int]] = []
    for c_a, c_b, tag in _PAIR_BRACKETS:
        p = row.get(f"q_{measure}_{tag}", np.nan)
        if p is None or not np.isfinite(p) or p >= alpha:
            continue
        out.append((c_a, c_b, float(p), len(out)))
    return out


def _table_logp_const(table: np.ndarray, gln: np.ndarray) -> float:
    """The count-dependent part of the multiple-hypergeometric log-probability.

    For a contingency table with FIXED margins, the log-probability is::

        sum(lgamma(rowsum+1)) + sum(lgamma(colsum+1)) - lgamma(N+1)
            - sum(lgamma(cell+1))

    Only ``-sum(lgamma(cell+1))`` varies under a margin-preserving permutation,
    so ``sum(lgamma(cell+1))`` alone orders tables by probability (larger sum ->
    smaller probability). We return that sum so callers can compare directly.
    """
    return float(gln[table].sum())


def fisher_freeman_halton_mc(
    cats: np.ndarray,
    groups: np.ndarray,
    *,
    n_iter: int = 50_000,
    seed: int = 0,
) -> float:
    """Monte-Carlo Fisher-Freeman-Halton exact test for an R x C table.

    ``cats`` and ``groups`` are per-observation integer-coded category and group
    labels. Permuting ``cats`` across observations keeps both margins fixed
    (group sizes and pooled category counts), so this is exactly R's
    ``fisher.test(..., simulate.p.value=TRUE)``. Returns a two-sided p-value:
    the fraction of permuted tables at least as improbable as the observed one.

    Falls back to :func:`scipy.stats.fisher_exact` for the 2x2 case (exact).
    """
    cats = np.asarray(cats, dtype=int)
    groups = np.asarray(groups, dtype=int)
    n = cats.size
    if n == 0:
        return float("nan")

    k = int(cats.max()) + 1
    g = int(groups.max()) + 1

    obs = np.zeros((g, k), dtype=int)
    np.add.at(obs, (groups, cats), 1)

    # Degenerate: a group or category with no spread -> nothing to test.
    if g < 2 or (obs.sum(axis=0) > 0).sum() < 2:
        return 1.0

    if g == 2 and k == 2:
        _, p = fisher_exact(obs, alternative="two-sided")
        return float(p)

    gln = gammaln(np.arange(n + 2) + 1.0)
    obs_stat = _table_logp_const(obs, gln)

    sizes = np.bincount(groups, minlength=g)
    edges = np.cumsum(sizes)[:-1]
    rng = np.random.default_rng(seed)

    # Larger sum(lgamma(cell+1)) <-> smaller probability. Count permuted tables
    # at least as improbable as observed (>= observed statistic).
    at_least_as_extreme = 0
    eps = 1e-9
    for _ in range(n_iter):
        perm = rng.permutation(cats)
        stat = 0.0
        for block in np.split(perm, edges):
            counts = np.bincount(block, minlength=k)
            stat += gln[counts].sum()
        if stat >= obs_stat - eps:
            at_least_as_extreme += 1

    return at_least_as_extreme / n_iter


def _wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score 95% CI for a binomial proportion k/n.

    Not used by this figure any more (both panels report mean +/- SEM over
    flies), but ``randompanel_trial_position`` imports it from here.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    phat = k / n
    denom = 1.0 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = (z / denom) * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


def holm_adjust(pvals: Sequence[float]) -> np.ndarray:
    """Holm-Bonferroni step-down adjusted p-values (NaNs pass through)."""
    p = np.asarray(pvals, dtype=float)
    out = np.full(p.shape, np.nan)
    finite = np.flatnonzero(np.isfinite(p))
    m = finite.size
    running = 0.0
    for rank, idx in enumerate(finite[np.argsort(p[finite])]):
        running = max(running, (m - rank) * p[idx])
        out[idx] = min(1.0, running)
    return out


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def _load_panel(
    csv_path: Path, *, fly_type: str, config: str
) -> pd.DataFrame:
    """Load the three RandomPanel concentrations for one genotype, pooled."""
    set_protocol("v2")
    if config:
        try:
            from fbpipe.config import load_settings
            from scripts.analysis.envelope_visuals import set_dataset_odor_remap

            settings = load_settings(config)
            remap = {
                str(ds): dict(ov.odor_remap)
                for ds, ov in settings.dataset_overrides.items()
                if getattr(ov, "odor_remap", None)
            }
            if remap:
                set_dataset_odor_remap(remap)
        except Exception as exc:  # noqa: BLE001 — defensive
            print(f"[WARN] Failed to load odor_remap from {config}: {exc}")

    df = _load_scores(csv_path)
    if "fly_type" in df.columns and fly_type:
        df = df[df["fly_type"].astype(str).str.strip() == fly_type].copy()
    df = df[df["dataset_canon"].isin(CONC_BY_DATASET)].copy()
    if df.empty:
        return df  # caller skips cleanly (not every dataset is a RandomPanel run)
    df["conc"] = df["dataset_canon"].map(CONC_BY_DATASET)
    df["odor"] = df["odor_display"]

    # Collapse each fly's two exposures ("Name 1"/"Name 2") to ONE per-fly value
    # so the unit of analysis is the fly (n = 20 per odorant x concentration),
    # not the presentation (avoids pseudoreplication). A fly's score is the mean
    # of its exposures; its response rate is the fraction of those exposures that
    # cleared the reaction boundary (0, 0.5 or 1). Averaging that fraction over
    # flies reproduces the pooled per-trial response rate reported in the paper.
    df["_reacted_trial"] = (df["score"] >= REACTION_BOUNDARY).astype(float)
    fly = df.groupby(["odor", "conc", "fly", "fly_number"], as_index=False).agg(
        score=("score", "mean"),
        reacted=("_reacted_trial", "mean"),
        n_trials=("score", "size"),
    )
    return fly


_MEASURE_COL = {"score": "score", "reaction": "reacted"}


def _compute_stats(
    df: pd.DataFrame, *, n_iter: int, seed: int, alpha: float = ALPHA
) -> pd.DataFrame:
    """Per-odorant statistics, in the framework reported in the paper.

    For each measure (mean ordinal score, response rate) and each odorant:

    * **Omnibus** — Kruskal-Wallis across the three concentrations on the per-fly
      value (``H(2)``), then Holm-Bonferroni corrected across the odorant family.
    * **Post-hoc** — pairwise Mann-Whitney U, Holm-corrected over the three
      concentration pairs, and only *run at all* for odorants whose omnibus
      survives correction (protected post-hoc). Unprotected odorants get NaN, so
      the figure never brackets a pair under a non-significant omnibus.

    The Fisher-Freeman-Halton exact omnibus/pairwise p-values are also retained
    (``p_score_*`` / ``p_reaction_*``) as a distribution-free cross-check.
    """
    rows: list[dict] = []
    odors = sorted(df["odor"].unique())

    for odor in odors:
        sub = df[df["odor"] == odor]

        # Encode (per-fly, possibly fractional) values to contiguous categories
        # for the FFH test.
        score_vals = sub["score"].to_numpy(float)
        react_vals = sub["reacted"].to_numpy(float)
        score_code = np.searchsorted(np.unique(score_vals), score_vals)
        react_code = np.searchsorted(np.unique(react_vals), react_vals)

        conc_vals = sub["conc"].to_numpy(float)
        conc_code_all = np.array([CONC_ORDER.index(c) for c in conc_vals])

        row: dict = {"odor": odor}

        # --- Kruskal-Wallis omnibus (the test reported in the paper) ---------
        for measure, col in _MEASURE_COL.items():
            groups = [
                sub.loc[sub["conc"] == c, col].to_numpy(float) for c in CONC_ORDER
            ]
            if min(len(g) for g in groups) == 0 or len(np.unique(np.concatenate(groups))) < 2:
                h, p = float("nan"), float("nan")
            else:
                res = kruskal(*groups)
                h, p = float(res.statistic), float(res.pvalue)
            row[f"H_{measure}"] = h
            row[f"p_{measure}_kw"] = p

        # --- Fisher-Freeman-Halton omnibus (cross-check) ---------------------
        row["p_score_omnibus"] = fisher_freeman_halton_mc(
            score_code, conc_code_all, n_iter=n_iter, seed=seed
        )
        row["p_reaction_omnibus"] = fisher_freeman_halton_mc(
            react_code, conc_code_all, n_iter=n_iter, seed=seed
        )

        # --- Pairwise FFH (cross-check; unprotected, uncorrected) ------------
        for c_lo, c_hi in combinations(CONC_ORDER, 2):
            mask = np.isin(conc_vals, [c_lo, c_hi])
            grp = (conc_vals[mask] == c_hi).astype(int)  # 0=lo, 1=hi
            tag = f"{c_lo}v{c_hi}"
            row[f"p_score_{tag}"] = fisher_freeman_halton_mc(
                np.searchsorted(np.unique(score_vals[mask]), score_vals[mask]),
                grp, n_iter=n_iter, seed=seed,
            )
            row[f"p_reaction_{tag}"] = fisher_freeman_halton_mc(
                np.searchsorted(np.unique(react_vals[mask]), react_vals[mask]),
                grp, n_iter=n_iter, seed=seed,
            )

        rows.append(row)

    stats = pd.DataFrame(rows)

    # Holm across the odorant family, per measure — then the protected post-hoc.
    for measure, col in _MEASURE_COL.items():
        stats[f"q_{measure}_kw"] = holm_adjust(stats[f"p_{measure}_kw"])
        stats[f"sig_{measure}_kw"] = [_sig_stars(q) for q in stats[f"q_{measure}_kw"]]

        posthoc: dict[str, list[float]] = {
            f"q_{measure}_{c_lo}v{c_hi}": []
            for c_lo, c_hi in combinations(CONC_ORDER, 2)
        }
        for _, srow in stats.iterrows():
            sub = df[df["odor"] == srow["odor"]]
            protected = np.isfinite(srow[f"q_{measure}_kw"]) and (
                srow[f"q_{measure}_kw"] < alpha
            )
            raw, tags = [], []
            for c_lo, c_hi in combinations(CONC_ORDER, 2):
                tags.append(f"q_{measure}_{c_lo}v{c_hi}")
                if not protected:
                    raw.append(np.nan)
                    continue
                a = sub.loc[sub["conc"] == c_lo, col].to_numpy(float)
                b = sub.loc[sub["conc"] == c_hi, col].to_numpy(float)
                if a.size == 0 or b.size == 0:
                    raw.append(np.nan)
                else:
                    raw.append(float(mannwhitneyu(a, b).pvalue))
            for tag, q in zip(tags, holm_adjust(raw)):
                posthoc[tag].append(q)
        for tag, vals in posthoc.items():
            stats[tag] = vals

    return stats


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _grouped_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per (odor, conc): mean +/- SEM of the per-fly score and response rate.

    Both panels summarise the same unit (the fly) with the same estimator, so
    the error bars match the tests run on those values.
    """
    recs = []
    for (odor, conc), grp in df.groupby(["odor", "conc"]):
        n = len(grp)
        recs.append(
            {
                "odor": odor,
                "conc": conc,
                "n": n,
                "n_trials": int(grp["n_trials"].sum()) if "n_trials" in grp else n,
                "mean_score": grp["score"].mean(),
                "sem_score": grp["score"].sem(ddof=1) if n > 1 else 0.0,
                "pct_react": grp["reacted"].mean(),
                "sem_react": grp["reacted"].sem(ddof=1) if n > 1 else 0.0,
            }
        )
    out = pd.DataFrame(recs)
    out["sem_score"] = out["sem_score"].fillna(0.0)
    out["sem_react"] = out["sem_react"].fillna(0.0)
    return out


def _draw_significance_brackets(
    ax,
    *,
    odors: Sequence[str],
    stats_by_odor: dict,
    group_tops: dict[str, float],
    bar_x: dict[str, dict[float, float]],
    measure: str,
    ref: float,
    alpha: float = ALPHA,
) -> float:
    """Draw the significant pairwise brackets; return the highest y they reach.

    ``ref`` is the data extent (tallest bar + error bar) and sets the vertical
    rhythm, so spacing stays consistent whichever measure is plotted. Nothing is
    drawn for an odorant whose comparisons are all non-significant.
    """
    pad = ref * 0.06     # clearance between the tallest bar and the first bracket
    step = ref * 0.11    # gap between stacked bracket levels
    tick = ref * 0.015   # downward ticks at the bracket ends
    label_h = ref * 0.06  # room for the label sitting on the bracket

    top = max(group_tops.values()) if group_tops else 0.0
    for odor in odors:
        row = stats_by_odor.get(odor)
        if row is None:
            continue
        base = group_tops[odor] + pad
        for c_a, c_b, p, level in significant_brackets(row, measure, alpha):
            xa, xb = bar_x[odor].get(c_a), bar_x[odor].get(c_b)
            if xa is None or xb is None:
                continue
            xl, xr = sorted((xa, xb))
            yb = base + level * step
            ax.plot(
                [xl, xl, xr, xr], [yb - tick, yb, yb, yb - tick],
                color="#333333", linewidth=0.7, solid_capstyle="butt",
            )
            ax.text(
                (xl + xr) / 2, yb + ref * 0.008, _fmt_p(p),
                ha="center", va="bottom", fontsize=6.5, color="#222222",
            )
            top = max(top, yb + label_h)
    return top


def _fmt_omnibus(row, measure: str) -> str:
    """One-line omnibus label: ``H(2) = 13.8, p = 0.007`` (Holm-adjusted p)."""
    h = row.get(f"H_{measure}", np.nan)
    q = row.get(f"q_{measure}_kw", np.nan)
    if h is None or q is None or not np.isfinite(h) or not np.isfinite(q):
        return ""
    qtxt = "p<0.001" if q < 0.001 else f"p={q:.3f}"
    return f"H(2)={h:.1f}, {qtxt}"


def _draw_omnibus_labels(
    ax,
    *,
    odors: Sequence[str],
    stats_by_odor: dict,
    x: np.ndarray,
    measure: str,
    y: float = -0.155,
    alpha: float = ALPHA,
) -> None:
    """Print the per-odorant Kruskal-Wallis omnibus under each group.

    Significant omnibus tests are inked dark; non-significant ones stay muted so
    the eye lands on the odorants that carry a concentration effect.
    """
    for odor, xi in zip(odors, x):
        row = stats_by_odor.get(odor)
        if row is None:
            continue
        label = _fmt_omnibus(row, measure)
        if not label:
            continue
        q = row.get(f"q_{measure}_kw", np.nan)
        sig = bool(np.isfinite(q) and q < alpha)
        ax.annotate(
            label, xy=(xi, y), xycoords=("data", "axes fraction"),
            ha="center", va="top", fontsize=6,
            color="#1a1a1a" if sig else "#8a8a8a",
            fontweight="bold" if sig else "normal",
            annotation_clip=False,
        )


def _xtick_labels(summary: pd.DataFrame, odors: Sequence[str]) -> list[str]:
    """Odorant names wrapped to two lines; per-odorant n only when it varies."""
    ns = {o: int(summary[summary["odor"] == o]["n"].iloc[0]) for o in odors}
    uniform = len(set(ns.values())) <= 1
    labels = []
    for o in odors:
        name = "\n".join(textwrap.wrap(o, width=13)) or o
        labels.append(name if uniform else f"{name}\n(n={ns[o]})")
    return labels


def _plot_grouped(
    summary: pd.DataFrame,
    stats: pd.DataFrame,
    *,
    value: str,
    err: tuple[str, str] | str,
    measure: str,
    ylabel: str,
    title: str,
    png_path: Path,
    as_pct: bool = False,
    footnote: bool = True,
    value_labels: bool = False,
    omnibus_labels: bool = True,
    return_fig: bool = False,
):
    odors = sorted(summary["odor"].unique())
    x = np.arange(len(odors))
    n_conc = len(CONC_PLOT_ORDER)
    bar_w = 0.78 / n_conc

    stats_by_odor = {row["odor"]: row for _, row in stats.iterrows()}

    with plt.rc_context(_RC_CONTEXT):
        # Journal-width figure (~180 mm) rather than a poster-sized canvas.
        fig, ax = plt.subplots(figsize=(min(7.2, 1.6 + len(odors) * 0.85), 4.0))
        ax.set_axisbelow(True)

        group_tops: dict[str, float] = {o: 0.0 for o in odors}
        # bar_x[odor][conc] -> x position of that bar (for the brackets).
        bar_x: dict[str, dict[float, float]] = {o: {} for o in odors}
        for j, conc in enumerate(CONC_PLOT_ORDER):
            offs = (j - (n_conc - 1) / 2) * bar_w
            vals, errs_lo, errs_hi = [], [], []
            for odor in odors:
                cell = summary[(summary["odor"] == odor) & (summary["conc"] == conc)]
                if cell.empty:
                    vals.append(0.0); errs_lo.append(0.0); errs_hi.append(0.0)
                    continue
                v = float(cell[value].iloc[0])
                vals.append(v)
                if isinstance(err, tuple):
                    lo = v - float(cell[err[0]].iloc[0])
                    hi = float(cell[err[1]].iloc[0]) - v
                    errs_lo.append(max(0.0, lo)); errs_hi.append(max(0.0, hi))
                else:
                    e = float(cell[err].iloc[0])
                    errs_lo.append(e); errs_hi.append(e)
            vals = np.array(vals)
            yerr = np.vstack([errs_lo, errs_hi])
            ax.bar(
                x + offs, vals, width=bar_w * 0.94, yerr=yerr,
                color=CONC_COLOR[conc], linewidth=0,
                label=f"{conc:g}%",
                error_kw={"elinewidth": 0.8, "capsize": 2, "capthick": 0.8,
                          "ecolor": "#333333"},
            )
            if value_labels:
                for xi, v, ehi in zip(x + offs, vals, errs_hi):
                    txt = f"{v*100:.0f}%" if as_pct else f"{v:.2f}"
                    ax.text(
                        xi, v + ehi + (0.015 if as_pct else 0.05), txt,
                        ha="center", va="bottom", fontsize=6, rotation=90,
                        color="#444444",
                    )
            for odor, xi, v, ehi in zip(odors, x + offs, vals, errs_hi):
                group_tops[odor] = max(group_tops[odor], v + ehi)
                bar_x[odor][conc] = xi

        data_top = max(group_tops.values()) if group_tops else 1.0
        ref = max(data_top, 1e-6)
        bracket_top = _draw_significance_brackets(
            ax, odors=odors, stats_by_odor=stats_by_odor, group_tops=group_tops,
            bar_x=bar_x, measure=measure, ref=ref,
        )

        # Headroom is now only what the surviving brackets actually need.
        if as_pct:
            ax.set_ylim(0, max(1.04, bracket_top * 1.04))
            ax.set_yticks(np.arange(0, 1.01, 0.25))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
        else:
            ax.set_ylim(0, max(REACTION_BOUNDARY * 1.3, bracket_top * 1.05))

        ax.set_xticks(x)
        ax.set_xticklabels(_xtick_labels(summary, odors))
        ax.tick_params(axis="x", length=0, pad=4)
        if omnibus_labels:
            _draw_omnibus_labels(
                ax, odors=odors, stats_by_odor=stats_by_odor, x=x, measure=measure,
            )
        ax.set_ylabel(ylabel)
        ax.set_xlabel("")
        ax.margins(x=0.02)
        ax.set_title(title, fontsize=10, pad=10, color="#111111")
        ax.grid(axis="y", linestyle="-", linewidth=0.5, color="#e6e6e6")

        # Legend lives inside the plot (upper left is always the emptiest
        # corner here), sized up so the concentrations read at a glance.
        leg = ax.legend(
            title="Concentration", ncol=1, frameon=False, loc="upper left",
            fontsize=10, handlelength=1.3, handleheight=1.3,
            labelspacing=0.45, handletextpad=0.6, borderpad=0.2,
        )
        leg.get_title().set_fontsize(10)
        leg._legend_box.align = "left"

        if footnote:
            fig.text(
                0.0, -0.03,
                f"Mean ± SEM over flies (n = {int(summary['n'].iloc[0])} per bar; "
                "each fly's two exposures averaged). Under each odorant: "
                "Kruskal–Wallis across concentrations, Holm-corrected across the "
                "7 odorants. Brackets: Mann–Whitney post-hoc, Holm-corrected "
                "within odorant, drawn only where the omnibus survives correction "
                "and the pair reaches p < 0.05 (*** p<0.001, ** p<0.01, * p<0.05). "
                "All p-values in randompanel_conc_comparison_stats.csv.",
                fontsize=6, color="#666666", ha="left", va="top", wrap=True,
            )

        fig.tight_layout()
        for path in (png_path, png_path.with_suffix(".pdf"),
                     png_path.with_suffix(".svg")):
            fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"[conc_compare] Wrote {png_path} (+ .pdf/.svg)")

    if return_fig:
        return fig
    plt.close(fig)
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _resolve_out_dir(
    out_dir: Path, csv_path: Path, fly_type: str, *, genotype_subdir: bool
) -> Path:
    """Mirror score_summary foldering: a per-genotype subfolder is used only
    when the predictions CSV holds more than one genotype."""
    if not (genotype_subdir and fly_type):
        return out_dir
    try:
        genos = pd.read_csv(csv_path, usecols=["fly_type"])["fly_type"]
        n_geno = len({str(g).strip() for g in genos if str(g).strip()})
    except Exception:  # noqa: BLE001 — fly_type column may be absent
        n_geno = 1
    if n_geno <= 1:
        return out_dir
    from scripts.analysis.envelope_visuals import _safe_dirname

    return out_dir / _safe_dirname(fly_type)


def generate_conc_comparison(
    csv_path: Path,
    out_dir: Path,
    *,
    fly_type: str = "GR5a-Old",
    config: str = "",
    n_iter: int = 50_000,
    seed: int = 0,
    overwrite: bool = True,
    genotype_subdir: bool = False,
) -> None:
    df = _load_panel(csv_path, fly_type=fly_type, config=config)
    if df.empty:
        print(
            f"[conc_compare] No RandomPanel concentrations for fly_type={fly_type!r}"
            f" in {csv_path.name}; skipping."
        )
        return
    # Require all three concentrations before drawing the comparison.
    present = set(df["conc"].unique())
    if not set(CONC_ORDER).issubset(present):
        print(
            f"[conc_compare] Only concentrations {sorted(present)} present for "
            f"fly_type={fly_type!r}; need {CONC_ORDER}. Skipping."
        )
        return

    out_dir = _resolve_out_dir(
        out_dir, csv_path, fly_type, genotype_subdir=genotype_subdir
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = _grouped_summary(df)
    stats = _compute_stats(df, n_iter=n_iter, seed=seed)

    # Merge omnibus results into the summary CSV for convenience.
    stats_csv = out_dir / "randompanel_conc_comparison_stats.csv"
    if should_write(stats_csv, overwrite):
        stats.to_csv(stats_csv, index=False, float_format="%.5f")
        print(f"[conc_compare] Wrote {stats_csv}")

    summary_csv = out_dir / "randompanel_conc_comparison_summary.csv"
    if should_write(summary_csv, overwrite):
        summary.sort_values(["odor", "conc"]).to_csv(
            summary_csv, index=False, float_format="%.4f"
        )
        print(f"[conc_compare] Wrote {summary_csv}")

    geno = f" ({fly_type})" if fly_type else ""
    score_png = out_dir / "randompanel_conc_score_comparison.png"
    if should_write(score_png, overwrite):
        _plot_grouped(
            summary, stats,
            value="mean_score", err="sem_score",
            measure="score",
            ylabel="Mean Ordinal Score",
            title=f"RandomPanel: mean PER score by odorant × concentration{geno}",
            png_path=score_png,
        )

    react_png = out_dir / "randompanel_conc_reaction_comparison.png"
    if should_write(react_png, overwrite):
        _plot_grouped(
            summary, stats,
            value="pct_react", err="sem_react",
            measure="reaction",
            ylabel="Response rate (score ≥ 2)",
            title=f"RandomPanel: reaction rate by odorant × concentration{geno}",
            png_path=react_png,
            as_pct=True,
        )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-path", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--fly-type", type=str, default="GR5a-Old")
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--n-iter", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true", default=True)
    parser.add_argument(
        "--genotype-subdir", action="store_true", default=False,
        help="Write into <out-dir>/<fly_type>/ when the CSV holds >1 genotype "
             "(matches score_summary foldering). Pass the score_summary parent "
             "as --out-dir with this flag.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    generate_conc_comparison(
        csv_path=args.csv_path,
        out_dir=args.out_dir,
        fly_type=args.fly_type,
        config=args.config,
        n_iter=args.n_iter,
        seed=args.seed,
        overwrite=args.overwrite,
        genotype_subdir=args.genotype_subdir,
    )


if __name__ == "__main__":
    main()
