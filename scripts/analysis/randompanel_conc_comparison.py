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

The omnibus p-value (does the response vary across *any* concentration) is
marked with significance stars above each odorant group. All pairwise
concentration comparisons (0.1 vs 1, 1 vs 10, 0.1 vs 10) are written to
``randompanel_conc_comparison_stats.csv`` alongside the omnibus results.

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
from itertools import combinations
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.stats import fisher_exact

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

_RC_CONTEXT = {
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "Arial",
    "font.sans-serif": ["Arial"],
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
    """Bracket label: p-value always, prefixed with stars when significant."""
    if p is None or np.isnan(p):
        return ""
    ptxt = "p<0.001" if p < 0.001 else f"p={p:.3f}"
    stars = _sig_stars(p)
    return f"{stars} {ptxt}" if stars and stars != "ns" else ptxt


# Pairwise concentration brackets, drawn low->high. ``tag`` matches the column
# naming built by ``_compute_stats`` (ascending "loVhi" using float repr).
_PAIR_BRACKETS = [
    (10.0, 1.0, "1.0v10.0", 0),   # adjacent left  (10 vs 1)
    (1.0, 0.1, "0.1v1.0", 1),     # adjacent right (1 vs 0.1)
    (10.0, 0.1, "0.1v10.0", 2),   # outer          (10 vs 0.1)
]


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
    """Wilson score 95% CI for a binomial proportion k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    phat = k / n
    denom = 1.0 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = (z / denom) * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


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
    # of its exposures; it "reacted" if that per-fly mean clears the boundary.
    fly = (
        df.groupby(["odor", "conc", "fly", "fly_number"], as_index=False)["score"]
        .mean()
    )
    fly["reacted"] = (fly["score"] >= REACTION_BOUNDARY).astype(int)
    return fly


def _compute_stats(df: pd.DataFrame, *, n_iter: int, seed: int) -> pd.DataFrame:
    """Per-odorant omnibus + pairwise Fisher tests for score and % reaction."""
    rows: list[dict] = []
    odors = sorted(df["odor"].unique())

    for odor in odors:
        sub = df[df["odor"] == odor]

        # Encode (per-fly, possibly fractional) score levels to contiguous
        # categories for the FFH test.
        score_vals = sub["score"].to_numpy(float)
        uniq = np.unique(score_vals)
        score_code = np.searchsorted(uniq, score_vals)
        react = sub["reacted"].to_numpy(int)

        conc_vals = sub["conc"].to_numpy(float)
        conc_code_all = np.array([CONC_ORDER.index(c) for c in conc_vals])

        # Omnibus across all three concentrations.
        p_score_omni = fisher_freeman_halton_mc(
            score_code, conc_code_all, n_iter=n_iter, seed=seed
        )
        p_react_omni = fisher_freeman_halton_mc(
            react, conc_code_all, n_iter=n_iter, seed=seed
        )

        row: dict = {
            "odor": odor,
            "p_score_omnibus": p_score_omni,
            "sig_score_omnibus": _sig_stars(p_score_omni),
            "p_reaction_omnibus": p_react_omni,
            "sig_reaction_omnibus": _sig_stars(p_react_omni),
        }

        # Pairwise concentration comparisons.
        for c_lo, c_hi in combinations(CONC_ORDER, 2):
            mask = np.isin(conc_vals, [c_lo, c_hi])
            grp = (conc_vals[mask] == c_hi).astype(int)  # 0=lo, 1=hi
            sc = np.searchsorted(
                np.unique(score_vals[mask]), score_vals[mask]
            )
            rc = react[mask]
            tag = f"{c_lo}v{c_hi}"
            row[f"p_score_{tag}"] = fisher_freeman_halton_mc(
                sc, grp, n_iter=n_iter, seed=seed
            )
            row[f"p_reaction_{tag}"] = fisher_freeman_halton_mc(
                rc, grp, n_iter=n_iter, seed=seed
            )

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _grouped_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per (odor, conc): mean/SEM score, reaction count/rate/Wilson CI, n."""
    recs = []
    for (odor, conc), grp in df.groupby(["odor", "conc"]):
        n = len(grp)
        k = int(grp["reacted"].sum())
        lo, hi = _wilson_ci(k, n)
        recs.append(
            {
                "odor": odor,
                "conc": conc,
                "n": n,
                "mean_score": grp["score"].mean(),
                "sem_score": grp["score"].sem(ddof=1) if n > 1 else 0.0,
                "pct_react": k / n if n else np.nan,
                "ci_lo": lo,
                "ci_hi": hi,
            }
        )
    out = pd.DataFrame(recs)
    out["sem_score"] = out["sem_score"].fillna(0.0)
    return out


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
    reaction_line: bool = False,
    footnote: bool = True,
) -> None:
    odors = sorted(summary["odor"].unique())
    x = np.arange(len(odors))
    n_conc = len(CONC_PLOT_ORDER)
    bar_w = 0.8 / n_conc

    stats_by_odor = {row["odor"]: row for _, row in stats.iterrows()}

    with plt.rc_context(_RC_CONTEXT):
        fig, ax = plt.subplots(figsize=(max(10, len(odors) * 1.7), 6.6))

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
                x + offs, vals, width=bar_w, yerr=yerr, capsize=3,
                color=CONC_COLOR[conc], edgecolor="white", linewidth=0.6,
                label=f"{conc:g}",
            )
            # Value label on each bar (above its error bar).
            for xi, v, ehi in zip(x + offs, vals, errs_hi):
                txt = f"{v*100:.0f}%" if as_pct else f"{v:.2f}"
                ax.text(
                    xi, v + ehi + (0.015 if as_pct else 0.05), txt,
                    ha="center", va="bottom", fontsize=7, rotation=90,
                )
            for odor, xi, v, ehi in zip(odors, x + offs, vals, errs_hi):
                group_tops[odor] = max(group_tops[odor], v + ehi)
                bar_x[odor][conc] = xi

        # Axis limits first, so bracket geometry can be sized off the y-range.
        if as_pct:
            ax.set_ylim(0, 1.55)
            ax.set_yticks(np.arange(0, 1.01, 0.2))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
        else:
            top_needed = max(group_tops.values()) if group_tops else 5.0
            ax.set_ylim(-0.2, max(6.8, top_needed + 3.0))
        y0, y1 = ax.get_ylim()
        span = y1 - y0

        # Pairwise significance brackets between the three concentrations,
        # stacked per odorant. Each bracket is labelled with its p-value
        # (stars added when significant).
        val_pad = span * 0.11   # clearance above rotated value labels
        step = span * 0.085     # vertical gap between bracket levels
        tick = span * 0.012     # little downward ticks at bracket ends
        for odor in odors:
            row = stats_by_odor.get(odor)
            if row is None:
                continue
            base = group_tops[odor] + val_pad
            for c_a, c_b, tag, level in _PAIR_BRACKETS:
                p = row.get(f"p_{measure}_{tag}", np.nan)
                xa, xb = bar_x[odor].get(c_a), bar_x[odor].get(c_b)
                if xa is None or xb is None:
                    continue
                xl, xr = sorted((xa, xb))
                yb = base + level * step
                ax.plot(
                    [xl, xl, xr, xr], [yb - tick, yb, yb, yb - tick],
                    color="black", linewidth=0.8, clip_on=False,
                )
                stars = _sig_stars(p)
                weight = "bold" if stars and stars != "ns" else "normal"
                ax.text(
                    (xl + xr) / 2, yb + span * 0.004, _fmt_p(p),
                    ha="center", va="bottom", fontsize=6.5, fontweight=weight,
                )

        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{o}\n(n={int(summary[summary['odor']==o]['n'].iloc[0])}/conc)" for o in odors],
            fontsize=9,
        )
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Odorant")
        ax.set_title(title, fontsize=13, weight="bold")
        if reaction_line:
            ax.axhline(
                y=REACTION_BOUNDARY, color="red", linewidth=0.8, linestyle=":",
                alpha=0.6, label="Reaction Boundary",
            )
        ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)

        leg = ax.legend(
            title="Concentration", fontsize=8, title_fontsize=9,
            loc="upper left", framealpha=0.9, ncol=1,
        )
        leg._legend_box.align = "left"

        # Footnote: how to read the brackets.
        if footnote:
            ax.text(
                0.0, -0.22,
                "Unit = fly (both exposures averaged per fly, n=20/concentration). "
                "Brackets: pairwise Fisher's exact test between concentrations "
                "(2x2 for % reaction; Fisher-Freeman-Halton on the score distribution). "
                "p-value shown for every pair; *** p<0.001, ** p<0.01, * p<0.05.",
                transform=ax.transAxes, fontsize=7, color="#444444",
            )

        plt.tight_layout()
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[conc_compare] Wrote {png_path}")


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
            title=f"RandomPanel: Mean Score by Odorant x Concentration{geno}",
            png_path=score_png,
            reaction_line=True,
            footnote=False,
        )

    react_png = out_dir / "randompanel_conc_reaction_comparison.png"
    if should_write(react_png, overwrite):
        _plot_grouped(
            summary, stats,
            value="pct_react", err=("ci_lo", "ci_hi"),
            measure="reaction",
            ylabel="% Flies Reacting (mean score >= 2)",
            title=f"RandomPanel: % Reaction by Odorant x Concentration{geno}",
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
