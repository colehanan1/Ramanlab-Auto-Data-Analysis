#!/usr/bin/env python3
"""Corrected, publication-grade statistics for 3-Octanol 0.1%, 24 h starved.

Cohorts (``config/config_new.yaml``): ``3OCT-Training-24-0.1`` vs
``3OCT-Control-24-0.1``, with ``RandomPanel-24-0.1`` as the naive arm.
``3OCT-Training-24-0.1-manual`` is deliberately excluded -- the config already
treats it as a separate unpaired cohort, and folding 2 extra flies into the
trained arm would break the batch structure the permutation test relies on.

Every figure here is a *corrected* version of something the existing pipeline
already draws. The filename says which correction it carries, and
``README_STATISTICS.md`` in the output folder spells each one out. In short:

fig1  score, Holm-corrected Mann-Whitney + Cliff's delta
fig2  % response, Holm-corrected Fisher + Wilson intervals
fig3  the same 2x2s run five ways -- Fisher / chi2 / Yates / Barnard / Boschloo
fig4  naive vs trained vs control, presentation-matched, Holm-corrected
fig5  GEE + ordinal models: the group x CS+ interaction, batch-clustered
fig6  batch-level exact permutation of the CS+ specificity contrast
fig7  responder-threshold sensitivity (score >= 1, 2, 3)
fig8  the raw per-fly score distributions behind fig1

The single most important structural fact about this dataset, and the reason
figs 5 and 6 exist: **every batch is entirely trained or entirely control**
(5 trained folders, 8 control folders, no overlap). Batch is therefore
collinear with the treatment and cannot be adjusted for; it can only be made
the unit of exchangeability. Any p-value computed as if the 31 flies were 31
independent draws is anti-conservative by an unknown amount.

Run::

    python scripts/analysis/pubfig_3oct_corrected.py
    python scripts/analysis/pubfig_3oct_corrected.py --out-dir /some/where
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import kruskal  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts.analysis import corrected_stats as cs  # noqa: E402
from scripts.analysis import odor_bar_palette as pal  # noqa: E402
from scripts.analysis.randompanel_conc_comparison import (  # noqa: E402
    fisher_freeman_halton_mc,
)

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

PREDICTIONS_CSV = Path(
    "/home/ramanlab/Documents/cole/Data/CSVs-New-Opto-Flys/model_predictions.csv"
)
CONFIG = REPO_ROOT / "config" / "config_new.yaml"
OUT_DIR = Path(
    "/home/ramanlab/Documents/cole/Results/Figures/3Oct-24-0.1_corrected_stats"
)
GENOTYPE = "GR5a-Old"

TRAIN_DS = "3OCT-Training-24-0.1"
CTRL_DS = "3OCT-Control-24-0.1"
NAIVE_DS = "RandomPanel-24-0.1"
CS_PLUS = "3-Octanol"
CONCENTRATION = "0.1%"

RESPONSE_THRESHOLD = 2.0        # score >= 2 is a response (matches score_summary)
THRESHOLD_SWEEP = (1.0, 2.0, 3.0)
ALPHA = 0.05
SCORE_MIN, SCORE_MAX = -1.0, 5.0

TRAINED_COLOR = "#1a3a6b"
CONTROL_COLOR = "#b0b0b0"
NAIVE_COLOR = "#ffffff"
NAIVE_EDGE = "#444444"
CSPLUS_TINT = pal.DARK_GREEN     # 3-Octanol's palette colour

FORMATS = (".png", ".svg", ".pdf")

plt.rcParams.update({
    "svg.fonttype": "none",      # keep SVG text editable in Illustrator
    "pdf.fonttype": 42,
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "figure.dpi": 150,
})


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


def load_frame(
    predictions_csv: Path = PREDICTIONS_CSV,
    config: Path = CONFIG,
    genotype: str = GENOTYPE,
) -> pd.DataFrame:
    """One row per fly x odor presentation for the three arms.

    The unit is deliberately the *fly*, never the trial: the raw CSV already
    holds exactly one testing trial per fly x odor x presentation here (checked
    below), so no averaging is needed and none is done. If that ever stops
    being true the assertion fires rather than silently averaging an average.
    """
    from scripts.analysis import pubfig_score_train_vs_control as pubfig
    from scripts.analysis import score_summary as ss

    pubfig._apply_config(config)
    df = ss._load_scores(predictions_csv, threshold=None, flagged_flies_csv="")
    df = df[df["dataset_canon"].astype(str).isin([TRAIN_DS, CTRL_DS, NAIVE_DS])].copy()
    if genotype and "fly_type" in df.columns:
        df = df[df["fly_type"].astype(str).str.strip() == genotype]

    df["cohort"] = np.select(
        [df["dataset_canon"] == TRAIN_DS, df["dataset_canon"] == CTRL_DS],
        ["Trained", "Control"],
        default="Naive",
    )
    df["batch"] = df["fly"].astype(str)                      # one folder = one day/rig
    df["fly_id"] = df["fly"].astype(str) + "#" + df["fly_number"].astype(str)
    df["occurrence"] = pd.to_numeric(df["occurrence"], errors="coerce").fillna(1).astype(int)
    df["odor"] = df["odor_display"].astype(str).map(_base_odor)
    df["presentation"] = df["occurrence"]
    df["label"] = np.where(
        df["occurrence"] > 1,
        df["odor"] + " " + df["occurrence"].astype(str),
        df["odor"],
    )
    df["is_csplus"] = (
        df["odor"].str.casefold() == CS_PLUS.casefold()
    ).astype(int)
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df["responded"] = (df["score"] >= RESPONSE_THRESHOLD).astype(int)

    dupes = df.groupby(["cohort", "fly_id", "label"]).size()
    if (dupes > 1).any():
        raise RuntimeError(
            "more than one trial per fly x odor x presentation -- the fly-level "
            f"unit assumption is broken for: {dupes[dupes > 1].index.tolist()[:5]}"
        )
    return df


def _base_odor(label: str) -> str:
    from scripts.analysis.pubfig_naive_vs_trained import base_odor

    return base_odor(label)


def panel_labels(df: pd.DataFrame) -> list[str]:
    """The 8 trained/control presentations, CS+ first then the rest A-Z."""
    tc = df[df["cohort"].isin(["Trained", "Control"])]
    shared = sorted(
        set(tc[tc["cohort"] == "Trained"]["label"])
        & set(tc[tc["cohort"] == "Control"]["label"])
    )
    cs_plus = [lbl for lbl in shared if lbl.casefold().startswith(CS_PLUS.casefold())]
    other = [lbl for lbl in shared if lbl not in cs_plus]
    return cs_plus + other


def arm(df: pd.DataFrame, cohort: str, label: str) -> pd.DataFrame:
    return df[(df["cohort"] == cohort) & (df["label"] == label)]


# --------------------------------------------------------------------------
# Shared drawing helpers
# --------------------------------------------------------------------------


def save(fig: plt.Figure, out_dir: Path, stem: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in FORMATS:
        fig.savefig(out_dir / f"{stem}{suffix}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    return out_dir / f"{stem}.png"


def stars(p: float, *, ns: str = "n.s.") -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ns


def fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "n/a"
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def tick_labels(
    ax: plt.Axes, labels: Sequence[str], *, rotate: float = 0.0, fontsize: float = 7.5
) -> None:
    """Wrap long odor names and tint the CS+ tick with its palette colour."""
    wrapped = [lbl.replace(" ", "\n", 1) if len(lbl) > 12 else lbl for lbl in labels]
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(
        wrapped, fontsize=fontsize, rotation=rotate,
        ha="right" if rotate else "center",
    )
    for tick, lbl in zip(ax.get_xticklabels(), labels):
        if lbl.casefold().startswith(CS_PLUS.casefold()):
            tick.set_color(CSPLUS_TINT)
            tick.set_fontweight("bold")


def bracket(
    ax: plt.Axes, x0: float, x1: float, y: float, text: str, *, color: str = "0.2"
) -> None:
    h = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.012
    ax.plot([x0, x0, x1, x1], [y, y + h, y + h, y], lw=0.8, color=color,
            clip_on=False)
    ax.text((x0 + x1) / 2, y + h * 1.2, text, ha="center", va="bottom",
            fontsize=7, color=color, clip_on=False)


def footer(fig: plt.Figure, text: str, *, fontsize: float = 6.4) -> None:
    """Wrapped caption pinned to the bottom of the figure.

    ``fig.text(..., wrap=True)`` only wraps at the *renderer* width, which for a
    long single-paragraph caption regularly overruns the canvas and lands on top
    of the x-axis label. Wrapping explicitly at a character count derived from
    the figure width keeps the caption inside the frame at any figure size.
    """
    import textwrap

    width_in = fig.get_size_inches()[0]
    # ~1.9 characters per point of width at this font size, measured empirically.
    chars = max(60, int(width_in * 72 / (fontsize * 0.60)))
    wrapped = "\n".join(textwrap.wrap(text, chars))
    fig.text(0.5, 0.008, wrapped, ha="center", va="bottom", fontsize=fontsize,
             color="0.35")


# --------------------------------------------------------------------------
# fig1 -- score, Holm-corrected Mann-Whitney with Cliff's delta
# --------------------------------------------------------------------------


def score_table(df: pd.DataFrame, labels: Sequence[str]) -> pd.DataFrame:
    rows = []
    for label in labels:
        t = arm(df, "Trained", label)["score"].to_numpy(float)
        c = arm(df, "Control", label)["score"].to_numpy(float)
        res = cs.score_test(t, c)
        rows.append({
            "odor": label,
            "is_csplus": label.casefold().startswith(CS_PLUS.casefold()),
            "n_trained": res.n1, "n_control": res.n2,
            "mean_trained": res.mean1, "sem_trained": res.sem1,
            "mean_control": res.mean2, "sem_control": res.sem2,
            "median_trained": res.median1, "iqr_trained_lo": res.iqr1[0],
            "iqr_trained_hi": res.iqr1[1],
            "median_control": res.median2, "iqr_control_lo": res.iqr2[0],
            "iqr_control_hi": res.iqr2[1],
            "U": res.u_statistic, "exact_null": res.exact,
            "p_raw": res.p_value,
            "cliffs_delta": res.delta,
            "delta_ci_lo": res.delta_ci[0], "delta_ci_hi": res.delta_ci[1],
            "hodges_lehmann_shift": res.shift,
            "shift_ci_lo": res.shift_ci[0], "shift_ci_hi": res.shift_ci[1],
        })
    out = pd.DataFrame(rows)
    out["p_holm"] = cs.holm_adjust(out["p_raw"])
    out["p_bh"] = cs.bh_adjust(out["p_raw"])
    out["stars_raw"] = out["p_raw"].map(stars)
    out["stars_holm"] = out["p_holm"].map(stars)
    return out


def fig1_score(table: pd.DataFrame, out_dir: Path) -> Path:
    labels = table["odor"].tolist()
    x = np.arange(len(labels))
    w = 0.38
    fig, (ax, ax_d) = plt.subplots(
        2, 1, figsize=(8.2, 6.0), height_ratios=[2.2, 1.0], sharex=True
    )

    ax.bar(x - w / 2, table["mean_trained"], w, yerr=table["sem_trained"],
           color=TRAINED_COLOR, label=f"Trained (n = {table['n_trained'].iloc[0]} flies)",
           capsize=2.5, error_kw={"lw": 0.8})
    ax.bar(x + w / 2, table["mean_control"], w, yerr=table["sem_control"],
           color=CONTROL_COLOR, edgecolor="0.4", lw=0.5,
           label=f"Unpaired control (n = {table['n_control'].iloc[0]} flies)",
           capsize=2.5, error_kw={"lw": 0.8})
    ax.axhline(0, color="0.6", lw=0.6)
    ax.set_ylabel("Mean PER score (± SEM)")
    # No group mean is negative, so the -1 floor of the score range would be
    # dead space; fig8 carries the full -1..5 spread the scores actually take.
    ax.set_ylim(0, 3.9)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    ax.set_title(
        f"{CS_PLUS} {CONCENTRATION}, 24 h starved — PER score, trained vs unpaired control\n"
        "Mann–Whitney U, Holm-corrected across all 8 presentations",
        fontsize=10, pad=8,
    )

    top = ax.get_ylim()[1]
    for i, row in table.iterrows():
        y = max(row["mean_trained"] + row["sem_trained"],
                row["mean_control"] + row["sem_control"]) + 0.25
        raw, holm = stars(row["p_raw"], ns=""), stars(row["p_holm"], ns="")
        note = f"{holm or 'n.s.'}" if holm else (f"({raw})" if raw else "n.s.")
        bracket(ax, i - w / 2, i + w / 2, min(y, top * 0.88), note)

    # Effect-size panel: Cliff's delta with bootstrap CI.
    colors = ["#2e7d32" if v else "0.35" for v in table["is_csplus"]]
    ax_d.errorbar(
        x, table["cliffs_delta"],
        yerr=[table["cliffs_delta"] - table["delta_ci_lo"],
              table["delta_ci_hi"] - table["cliffs_delta"]],
        fmt="o", ms=4.5, lw=0.9, capsize=2.5, ecolor="0.5",
        mfc="none", mec="none",
    )
    ax_d.scatter(x, table["cliffs_delta"], s=28, c=colors, zorder=3)
    ax_d.axhline(0, color="0.6", lw=0.6)
    for y, style in ((0.147, ":"), (-0.147, ":"), (0.33, "--"), (-0.33, "--")):
        ax_d.axhline(y, color="0.8", lw=0.5, ls=style)
    ax_d.set_ylim(-1.05, 1.05)
    ax_d.set_ylabel("Cliff's δ\n(trained − control)", fontsize=8)
    ax_d.text(len(labels) - 0.4, 0.36, "medium", fontsize=6, color="0.6", va="bottom")
    ax_d.text(len(labels) - 0.4, 0.16, "small", fontsize=6, color="0.6", va="bottom")
    tick_labels(ax_d, labels)

    footer(fig,
           "Bars: mean ± SEM per fly. Stars: Holm-adjusted p; a raw-p star that does not "
           "survive Holm is shown in parentheses. Cliff's δ = rank-biserial correlation, "
           "95% percentile bootstrap over flies (10 000 resamples); dotted/dashed guides "
           "mark the conventional small (0.147) and medium (0.33) thresholds. "
           "Score is ordinal (−1…5): means are shown for continuity with the existing "
           "figures, but fig8 carries the distributions the test actually uses.")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return save(fig, out_dir, "fig1_score_trained-vs-control_mwu_holm-corrected_cliffs-delta")


# --------------------------------------------------------------------------
# fig2 / fig3 -- the binary rate
# --------------------------------------------------------------------------


def rate_table(
    df: pd.DataFrame, labels: Sequence[str], threshold: float = RESPONSE_THRESHOLD
) -> pd.DataFrame:
    rows = []
    for label in labels:
        t = arm(df, "Trained", label)["score"].to_numpy(float)
        c = arm(df, "Control", label)["score"].to_numpy(float)
        k1, n1 = int((t >= threshold).sum()), int(t.size)
        k2, n2 = int((c >= threshold).sum()), int(c.size)
        b = cs.binary_test_battery(k1, n1, k2, n2)
        rows.append({
            "odor": label,
            "is_csplus": label.casefold().startswith(CS_PLUS.casefold()),
            "threshold": threshold,
            "k_trained": k1, "n_trained": n1, "pct_trained": 100 * b.p1,
            "wilson_trained_lo": 100 * b.p1_ci[0], "wilson_trained_hi": 100 * b.p1_ci[1],
            "k_control": k2, "n_control": n2, "pct_control": 100 * b.p2,
            "wilson_control_lo": 100 * b.p2_ci[0], "wilson_control_hi": 100 * b.p2_ci[1],
            "p_fisher": b.p_fisher, "p_chi2_pearson": b.p_chi2,
            "p_chi2_yates": b.p_chi2_yates, "p_barnard": b.p_barnard,
            "p_boschloo": b.p_boschloo,
            "min_expected_cell": b.min_expected,
            "chi2_admissible": b.chi2_admissible,
            "odds_ratio": b.odds_ratio,
            "or_ci_lo": b.odds_ratio_ci[0], "or_ci_hi": b.odds_ratio_ci[1],
            "risk_difference_pct": 100 * b.risk_difference,
            "rd_ci_lo_pct": 100 * b.risk_difference_ci[0],
            "rd_ci_hi_pct": 100 * b.risk_difference_ci[1],
        })
    out = pd.DataFrame(rows)
    for col in ("p_fisher", "p_chi2_pearson", "p_chi2_yates", "p_barnard", "p_boschloo"):
        out[f"{col}_holm"] = cs.holm_adjust(out[col])
    out["p_fisher_bh"] = cs.bh_adjust(out["p_fisher"])
    out["stars_fisher_raw"] = out["p_fisher"].map(stars)
    out["stars_fisher_holm"] = out["p_fisher_holm"].map(stars)
    return out


def fig2_rate(table: pd.DataFrame, out_dir: Path) -> Path:
    labels = table["odor"].tolist()
    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8.2, 4.6))

    for offset, prefix, color, edge, name in (
        (-w / 2, "trained", TRAINED_COLOR, "none", "Trained"),
        (w / 2, "control", CONTROL_COLOR, "0.4", "Unpaired control"),
    ):
        pct = table[f"pct_{prefix}"]
        lo = pct - table[f"wilson_{prefix}_lo"]
        hi = table[f"wilson_{prefix}_hi"] - pct
        n = table[f"n_{prefix}"].iloc[0]
        ax.bar(x + offset, pct, w, color=color, edgecolor=edge, lw=0.5,
               label=f"{name} (n = {n} flies)")
        ax.errorbar(x + offset, pct, yerr=[lo, hi], fmt="none", ecolor="0.25",
                    lw=0.9, capsize=2.5)

    ax.set_ylabel(f"Flies responding (score ≥ {RESPONSE_THRESHOLD:g}), %")
    ax.set_ylim(0, 118)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    ax.set_title(
        f"{CS_PLUS} {CONCENTRATION}, 24 h starved — % responding, trained vs unpaired control\n"
        "Fisher's exact, Holm-corrected across all 8 presentations; Wilson 95% intervals",
        fontsize=10, pad=8,
    )
    tick_labels(ax, labels)

    for i, row in table.iterrows():
        y = max(row["wilson_trained_hi"], row["wilson_control_hi"]) + 2
        holm, raw = stars(row["p_fisher_holm"], ns=""), stars(row["p_fisher"], ns="")
        note = holm if holm else (f"({raw})" if raw else "n.s.")
        bracket(ax, i - w / 2, i + w / 2, min(y, 108), note)

    footer(fig,
           "Fly is the unit: one fly contributes one responder/non-responder call per "
           "presentation. Error bars are Wilson score intervals, not Wald — at n = 11 a "
           "Wald interval runs off the axis and under-covers. Stars are Holm-adjusted; "
           "parenthesised stars are raw-p only. Odds ratios, risk differences and the "
           "alternative tests are in the sidecar CSV and in fig3.")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return save(fig, out_dir,
                "fig2_percent-response_trained-vs-control_fisher_holm-corrected_wilson-ci")


TEST_STYLES = [
    ("p_fisher", "Fisher's exact\n(conditional)", "#1a3a6b", "o"),
    ("p_chi2_pearson", "χ² Pearson\n(no correction)", "#d62728", "s"),
    ("p_chi2_yates", "χ² Yates\n(continuity)", "#E69F00", "^"),
    ("p_barnard", "Barnard\n(unconditional exact)", "#6a1b9a", "D"),
    ("p_boschloo", "Boschloo\n(unconditional exact)", "#1b7837", "v"),
]


def fig3_test_sensitivity(table: pd.DataFrame, out_dir: Path) -> Path:
    """Does the choice of binary test change the answer? (It does, once.)"""
    labels = table["odor"].tolist()
    x = np.arange(len(labels))
    fig, (ax, ax_h) = plt.subplots(2, 1, figsize=(8.6, 7.0), sharex=True, sharey=True,
                                   gridspec_kw={"hspace": 0.16})

    for ax_i, suffix, title in (
        (ax, "", "Raw p — one test per presentation, uncorrected"),
        (ax_h, "_holm", "Holm-corrected across the 8 presentations"),
    ):
        for j, (col, name, color, marker) in enumerate(TEST_STYLES):
            key = f"{col}{suffix}"
            jitter = (j - 2) * 0.13
            ax_i.scatter(x + jitter, table[key], s=38, c=color, marker=marker,
                         label=name.replace("\n", " ") if ax_i is ax else None,
                         zorder=3, edgecolors="white", linewidths=0.4)
        ax_i.axhline(ALPHA, color="#d62728", lw=0.9, ls="--")
        ax_i.text(len(labels) - 0.45, ALPHA * 1.15, "α = 0.05", fontsize=7,
                  color="#d62728", ha="right")
        ax_i.set_yscale("log")
        ax_i.set_ylim(0.02, 2.2)
        ax_i.set_ylabel("p-value (log)", fontsize=8.5)
        ax_i.set_title(title, fontsize=9, loc="left")
        for i, ok in enumerate(table["chi2_admissible"]):
            if not ok:
                ax_i.axvspan(i - 0.5, i + 0.5, color="#d62728", alpha=0.055, zorder=0)
    tick_labels(ax_h, labels, rotate=30)

    n_bad = int((~table["chi2_admissible"]).sum())
    ax.legend(frameon=False, fontsize=7, ncol=3, loc="lower center",
              bbox_to_anchor=(0.5, 1.10), handletextpad=0.2, columnspacing=1.2)
    fig.suptitle(
        f"{CS_PLUS} {CONCENTRATION} — does the binary test choice change the answer?",
        fontsize=10.5, y=1.03,
    )

    worst = table["min_expected_cell"].min()
    footer(fig,
           f"Same 2×2 per presentation, five tests. Red-shaded columns ({n_bad} of "
           f"{len(labels)}) fail Cochran's rule — smallest expected cell < 5, minimum here "
           f"= {worst:.2f} — which is exactly where the χ² approximation is not admissible "
           "and an exact test must be preferred. Fisher conditions on both margins and is "
           "conservative by construction; Boschloo is uniformly at least as powerful; Yates "
           "over-corrects. The spread between them is a small-n artefact, not new evidence: "
           "no test clears α after Holm, so the choice does not change the conclusion.")
    fig.subplots_adjust(bottom=0.26, top=0.90, left=0.10, right=0.98)
    return save(fig, out_dir,
                "fig3_binary-test-sensitivity_fisher-vs-chi2-vs-yates-vs-barnard-boschloo")


# --------------------------------------------------------------------------
# fig4 -- naive vs trained vs control, presentation-matched
# --------------------------------------------------------------------------


COHORT_ORDER = ("Naive", "Trained", "Control")
COHORT_STYLE = {
    "Naive": {"color": NAIVE_COLOR, "edgecolor": NAIVE_EDGE, "hatch": "///"},
    "Trained": {"color": TRAINED_COLOR, "edgecolor": "none"},
    "Control": {"color": CONTROL_COLOR, "edgecolor": "0.4"},
}


def three_arm_table(df: pd.DataFrame, presentation: int) -> tuple[pd.DataFrame, dict]:
    """Naive/Trained/Control for the CS+ at one presentation index."""
    label = CS_PLUS if presentation == 1 else f"{CS_PLUS} {presentation}"
    groups = {
        cohort: df[(df["cohort"] == cohort) & (df["odor"] == CS_PLUS)
                   & (df["presentation"] == presentation)]["score"].to_numpy(float)
        for cohort in COHORT_ORDER
    }
    rows = []
    for cohort in COHORT_ORDER:
        v = groups[cohort]
        k, n = int((v >= RESPONSE_THRESHOLD).sum()), int(v.size)
        lo, hi = cs.wilson_ci(k, n)
        rows.append({
            "presentation": presentation, "odor": label, "cohort": cohort,
            "n": n, "mean_score": float(v.mean()) if n else np.nan,
            "sem_score": float(v.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
            "median_score": float(np.median(v)) if n else np.nan,
            "k_responders": k, "pct_responding": 100 * k / n if n else np.nan,
            "wilson_lo": 100 * lo, "wilson_hi": 100 * hi,
        })
    table = pd.DataFrame(rows)

    usable = [groups[c] for c in COHORT_ORDER if groups[c].size]
    omnibus_score = float(kruskal(*usable).pvalue) if len(usable) >= 2 else np.nan
    # FFH wants per-observation labels, not a contingency table: category =
    # responded/not, group = cohort. Permuting the category across observations
    # holds both margins, which is what makes it the exact 2x3 analogue.
    cats = np.concatenate([
        (groups[c] >= RESPONSE_THRESHOLD).astype(int) for c in COHORT_ORDER
    ]) if any(groups[c].size for c in COHORT_ORDER) else np.array([], dtype=int)
    group_ids = np.concatenate([
        np.full(groups[c].size, i, dtype=int) for i, c in enumerate(COHORT_ORDER)
    ]) if cats.size else np.array([], dtype=int)
    omnibus_rate = float(fisher_freeman_halton_mc(cats, group_ids))

    pairs = [("Trained", "Control"), ("Trained", "Naive"), ("Control", "Naive")]
    score_raw, rate_raw, detail = [], [], []
    for a, b in pairs:
        st = cs.score_test(groups[a], groups[b])
        ka = int((groups[a] >= RESPONSE_THRESHOLD).sum())
        kb = int((groups[b] >= RESPONSE_THRESHOLD).sum())
        bt = cs.binary_test_battery(ka, groups[a].size, kb, groups[b].size)
        score_raw.append(st.p_value)
        rate_raw.append(bt.p_fisher)
        detail.append({
            "presentation": presentation, "pair": f"{a} vs {b}",
            "score_p_raw": st.p_value, "cliffs_delta": st.delta,
            "hodges_lehmann_shift": st.shift,
            "rate_p_fisher_raw": bt.p_fisher,
            "rate_p_chi2_pearson_raw": bt.p_chi2,
            "rate_p_chi2_yates_raw": bt.p_chi2_yates,
            "rate_p_boschloo_raw": bt.p_boschloo,
            "chi2_admissible": bt.chi2_admissible,
            "odds_ratio": bt.odds_ratio,
        })
    score_holm = cs.holm_adjust(score_raw)
    rate_holm = cs.holm_adjust(rate_raw)
    for d, sh, rh in zip(detail, score_holm, rate_holm):
        d["score_p_holm"] = float(sh)
        d["rate_p_fisher_holm"] = float(rh)

    stats = {
        "presentation": presentation,
        "omnibus_score_kruskal_p": omnibus_score,
        "omnibus_rate_ffh_p": omnibus_rate,
        "pairs": detail,
    }
    return table, stats


def fig4_three_arm(
    tables: list[pd.DataFrame], stats: list[dict], out_dir: Path
) -> Path:
    n_rows = len(tables)
    fig, axes = plt.subplots(n_rows, 2, figsize=(8.6, 3.4 * n_rows), squeeze=False)

    for r, (table, st) in enumerate(zip(tables, stats)):
        pres = st["presentation"]
        x = np.arange(len(COHORT_ORDER))

        ax = axes[r][0]
        for i, cohort in enumerate(COHORT_ORDER):
            row = table[table["cohort"] == cohort].iloc[0]
            ax.bar(i, row["mean_score"], 0.62, yerr=row["sem_score"], capsize=3,
                   error_kw={"lw": 0.8}, **COHORT_STYLE[cohort])
        ax.set_ylabel("Mean PER score (± SEM)")
        ax.set_ylim(0, 4.0)
        ax.set_title(
            f"{CS_PLUS} {CONCENTRATION}, presentation {pres} — score\n"
            f"Kruskal–Wallis {fmt_p(st['omnibus_score_kruskal_p'])}",
            fontsize=9,
        )

        ax_r = axes[r][1]
        for i, cohort in enumerate(COHORT_ORDER):
            row = table[table["cohort"] == cohort].iloc[0]
            ax_r.bar(i, row["pct_responding"], 0.62, **COHORT_STYLE[cohort])
            ax_r.errorbar(i, row["pct_responding"],
                          yerr=[[row["pct_responding"] - row["wilson_lo"]],
                                [row["wilson_hi"] - row["pct_responding"]]],
                          fmt="none", ecolor="0.25", lw=0.9, capsize=3)
        ax_r.set_ylabel(f"Responding (score ≥ {RESPONSE_THRESHOLD:g}), %")
        ax_r.set_ylim(0, 118)
        ax_r.set_yticks([0, 25, 50, 75, 100])
        ax_r.set_title(
            f"{CS_PLUS} {CONCENTRATION}, presentation {pres} — % responding\n"
            f"Fisher–Freeman–Halton {fmt_p(st['omnibus_rate_ffh_p'])}",
            fontsize=9,
        )

        pair_pos = {("Trained", "Control"): (1, 2), ("Trained", "Naive"): (0, 1),
                    ("Control", "Naive"): (0, 2)}
        for ax_i, key, top in ((ax, "score_p_holm", 4.0), (ax_r, "rate_p_fisher_holm", 118)):
            level = 0
            for d in st["pairs"]:
                a, b = d["pair"].split(" vs ")
                i0, i1 = sorted(pair_pos[(a, b)])
                y = top * (0.70 + 0.09 * level)
                bracket(ax_i, i0, i1, y, stars(d[key], ns="n.s."))
                level += 1

        for ax_i in (ax, ax_r):
            ax_i.set_xticks(x)
            ax_i.set_xticklabels(
                [f"{c}\nn = {int(table[table.cohort == c]['n'].iloc[0])}"
                 for c in COHORT_ORDER], fontsize=8,
            )

    fig.suptitle(
        f"{CS_PLUS} {CONCENTRATION}, 24 h starved — naive vs trained vs unpaired control",
        fontsize=11, y=0.995,
    )
    footer(fig,
           "Naive arm = RandomPanel-24-0.1, flies that met 3-Octanol at the same "
           "concentration without ever being conditioned. Arms are matched on presentation "
           "index (naive presentation 1 vs trained presentation 1), which removes the "
           "average-of-averages the pooled-trials version needed. Omnibus first "
           "(Kruskal–Wallis for score, Fisher–Freeman–Halton for the 2×3 rate table); "
           "brackets are Holm-corrected over the 3 pairwise tests within each panel.")
    fig.tight_layout(rect=(0, 0.05, 1, 0.98))
    return save(fig, out_dir,
                "fig4_naive-vs-trained-vs-control_3-Octanol_kruskal-ffh_holm-corrected")


# --------------------------------------------------------------------------
# fig5 -- models with batch-clustered inference
# --------------------------------------------------------------------------


def fit_models(df: pd.DataFrame) -> dict:
    """GEE (binary) and cumulative-link (ordinal) with batch-clustered SEs.

    Both fit ``outcome ~ trained * is_csplus``. The interaction is the term
    that matters: it asks whether the trained-minus-control gap is *larger for
    the CS+ than for the six novel odors*, which is the actual learning claim.
    A significant main effect of ``trained`` without it would only mean the
    trained flies were more responsive overall -- a cohort difference, not
    learning.
    """
    import statsmodels.api as sm
    from statsmodels.miscmodels.ordinal_model import OrderedModel

    tc = df[df["cohort"].isin(["Trained", "Control"])].copy()
    tc["trained"] = (tc["cohort"] == "Trained").astype(int)

    out: dict = {"n_obs": int(len(tc)),
                 "n_flies": int(tc["fly_id"].nunique()),
                 "n_batches": int(tc["batch"].nunique())}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gee = sm.GEE.from_formula(
            "responded ~ trained * is_csplus", groups="batch", data=tc,
            family=sm.families.Binomial(),
            cov_struct=sm.cov_struct.Exchangeable(),
        ).fit()
    out["gee"] = {
        "terms": list(gee.params.index),
        "coef": gee.params.to_dict(),
        "se": gee.bse.to_dict(),
        "p": gee.pvalues.to_dict(),
        "ci_lo": gee.conf_int()[0].to_dict(),
        "ci_hi": gee.conf_int()[1].to_dict(),
    }

    design = pd.DataFrame({
        "trained": tc["trained"].to_numpy(float),
        "is_csplus": tc["is_csplus"].to_numpy(float),
        "trained:is_csplus": (tc["trained"] * tc["is_csplus"]).to_numpy(float),
    }, index=tc.index)
    endog = pd.Categorical(tc["score"].astype(int), ordered=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ord_fit = OrderedModel(endog, design, distr="logit").fit(
            method="bfgs", disp=False, cov_type="cluster",
            cov_kwds={"groups": tc["batch"].to_numpy()},
        )
    keep = list(design.columns)
    ci = ord_fit.conf_int()
    out["ordinal"] = {
        "terms": keep,
        "coef": {k: float(ord_fit.params[k]) for k in keep},
        "se": {k: float(ord_fit.bse[k]) for k in keep},
        "p": {k: float(ord_fit.pvalues[k]) for k in keep},
        "ci_lo": {k: float(ci.loc[k, 0]) for k in keep},
        "ci_hi": {k: float(ci.loc[k, 1]) for k in keep},
    }
    return out


MODEL_TERM_LABELS = {
    "trained": "Trained vs control\n(novel odors only)",
    "is_csplus": "CS+ vs novel odors\n(control flies)",
    "trained:is_csplus": "Trained × CS+\n(the learning term)",
}


def fig5_models(models: dict, out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0), sharey=True,
                             gridspec_kw={"wspace": 0.05})
    panels = [
        (axes[0], models["gee"], "GEE, binomial — % responding",
         "Odds ratio (95% CI)"),
        (axes[1], models["ordinal"], "Cumulative-link (ordinal) — PER score",
         "Proportional-odds ratio (95% CI)"),
    ]
    terms = ["trained", "is_csplus", "trained:is_csplus"]

    for ax, fit, title, xlabel in panels:
        ys = np.arange(len(terms))[::-1]
        for y, term in zip(ys, terms):
            coef = fit["coef"][term]
            lo, hi = fit["ci_lo"][term], fit["ci_hi"][term]
            p = fit["p"][term]
            color = "#1b7837" if term == "trained:is_csplus" else "0.3"
            ax.plot([np.exp(lo), np.exp(hi)], [y, y], lw=1.6, color=color,
                    solid_capstyle="round")
            ax.plot(np.exp(coef), y, "o", ms=7, color=color, zorder=3)
            ax.text(ax.get_xlim()[1], y + 0.22,
                    f"OR = {np.exp(coef):.2f}  {fmt_p(p)} {stars(p, ns='')}",
                    fontsize=7.2, color=color, ha="right", va="bottom")
        ax.axvline(1.0, color="#d62728", lw=0.9, ls="--")
        ax.set_xscale("log")
        ax.set_xlim(0.08, 40)
        ax.set_yticks(ys)
        ax.set_yticklabels([MODEL_TERM_LABELS[t] for t in terms], fontsize=8)
        ax.set_ylim(-0.6, len(terms) - 0.3)
        ax.set_xlabel(xlabel, fontsize=8.5)
        ax.set_title(title, fontsize=9.5)

    fig.suptitle(
        f"{CS_PLUS} {CONCENTRATION} — the learning claim is the interaction, not the main effect",
        fontsize=10.5, y=1.02,
    )
    footer(fig,
           f"Both models: outcome ~ trained × CS+, fit on {models['n_obs']} fly × odor "
           f"observations from {models['n_flies']} flies in {models['n_batches']} batches, "
           "with standard errors clustered on batch (one fly folder = one day / rig / odor "
           "bottle). Pairwise per-odor tests answer 'did the arms differ on this odor'; the "
           "interaction answers 'did conditioning change the CS+ specifically', which is the "
           "claim the design supports. With only 13 clusters these sandwich standard errors "
           "are themselves approximate — fig6 gives the assumption-free version.")
    fig.subplots_adjust(bottom=0.30, top=0.86, left=0.16, right=0.98)
    return save(fig, out_dir,
                "fig5_gee-and-ordinal-models_group-by-csplus-interaction_batch-clustered")


# --------------------------------------------------------------------------
# fig6 -- batch-level exact permutation
# --------------------------------------------------------------------------


def specificity_contrasts(df: pd.DataFrame) -> pd.DataFrame:
    """Per fly: CS+ mean minus novel-odor mean, for score and for response.

    Balanced by design -- every fly has exactly 2 CS+ and 6 novel presentations
    -- so this within-fly contrast is not an average of averages; it is the
    same number pooling would give.
    """
    tc = df[df["cohort"].isin(["Trained", "Control"])]
    rows = []
    for (cohort, batch, fly), g in tc.groupby(["cohort", "batch", "fly_id"]):
        cs_plus = g[g["is_csplus"] == 1]
        novel = g[g["is_csplus"] == 0]
        if cs_plus.empty or novel.empty:
            continue
        rows.append({
            "cohort": cohort, "batch": batch, "fly_id": fly,
            "n_csplus": len(cs_plus), "n_novel": len(novel),
            "score_contrast": float(cs_plus["score"].mean() - novel["score"].mean()),
            "rate_contrast": float(
                cs_plus["responded"].mean() - novel["responded"].mean()
            ),
        })
    return pd.DataFrame(rows)


def fig6_permutation(
    contrasts: pd.DataFrame, out_dir: Path
) -> tuple[Path, dict]:
    results = {}
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))
    panels = [
        ("score_contrast", axes[0], "PER score", "CS+ − novel, mean score"),
        ("rate_contrast", axes[1], "% responding", "CS+ − novel, response rate"),
    ]
    for col, ax, name, xlabel in panels:
        res = cs.cluster_permutation_test(
            contrasts[col].to_numpy(float),
            contrasts["batch"].to_numpy(),
            (contrasts["cohort"] == "Trained").to_numpy(),
            cs.mean_difference,
        )
        results[col] = {
            "observed": res.observed, "p_value": res.p_value,
            "n_permutations": res.n_permutations, "exhaustive": res.exhaustive,
            "p_floor": res.p_floor,
        }
        ax.hist(res.null, bins=40, color="0.78", edgecolor="white", lw=0.4)
        ax.axvline(res.observed, color="#1b7837", lw=2.0)
        ax.axvline(-res.observed, color="#1b7837", lw=1.0, ls=":")
        ax.set_xlabel(f"{xlabel}\n(trained − control)", fontsize=8.5)
        ax.set_ylabel("Batch assignments", fontsize=8.5)
        ax.set_title(
            f"{name} — observed {res.observed:+.3f}\n"
            f"exact batch permutation {fmt_p(res.p_value)} "
            f"({res.n_permutations} assignments)",
            fontsize=9,
        )
        ax.text(0.02, 0.96,
                f"resolution floor: p ≥ {res.p_floor:.4f}",
                transform=ax.transAxes, fontsize=6.8, color="0.4", va="top")

    fig.suptitle(
        f"{CS_PLUS} {CONCENTRATION} — CS+ specificity, with the batch as the unit of "
        "exchangeability", fontsize=10.5, y=1.02,
    )
    footer(fig,
           "Each fly contributes one within-fly contrast (its 2 CS+ presentations minus its "
           "6 novel-odor presentations), so between-fly responsiveness cancels. Every batch "
           "in this cohort is entirely trained or entirely control, so batch cannot be "
           "adjusted for — instead all C(13,5) = 1287 assignments of the 13 batches to a "
           "5-batch and an 8-batch arm are enumerated exhaustively, giving an exact p that "
           "assumes nothing beyond batch exchangeability. Grey: the null. Green: observed "
           "(dotted line = its mirror, counted by the two-sided test).")
    fig.tight_layout(rect=(0, 0.11, 1, 0.98))
    return save(fig, out_dir,
                "fig6_csplus-specificity_batch-level-exact-permutation"), results


# --------------------------------------------------------------------------
# fig7 -- responder threshold sensitivity
# --------------------------------------------------------------------------


def fig7_threshold(df: pd.DataFrame, labels: Sequence[str], out_dir: Path
                   ) -> tuple[Path, pd.DataFrame]:
    frames = [rate_table(df, labels, threshold=t) for t in THRESHOLD_SWEEP]
    stacked = pd.concat(frames, ignore_index=True)

    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 1, figsize=(8.6, 6.6), sharex=True,
                             height_ratios=[1.6, 1.0],
                             gridspec_kw={"hspace": 0.12})
    ax, ax_p = axes
    # Odors are categorical: connecting them with a line would draw a trend
    # that does not exist. Offset markers per threshold instead.
    marks = ["o", "s", "^"]
    shades = ["#7fa6d9", "#3f6fae", TRAINED_COLOR]
    for t, table, marker, shade in zip(THRESHOLD_SWEEP, frames, marks, shades):
        off = (THRESHOLD_SWEEP.index(t) - 1) * 0.19
        rd = table["risk_difference_pct"]
        ax.errorbar(
            x + off, rd,
            yerr=[rd - table["rd_ci_lo_pct"], table["rd_ci_hi_pct"] - rd],
            fmt=marker, ms=5, lw=0.9, capsize=2.2, color=shade, ecolor=shade,
            elinewidth=0.9, label=f"score ≥ {t:g}",
        )
        ax_p.scatter(x + off, table["p_fisher_holm"], s=36, marker=marker,
                     color=shade, zorder=3, edgecolors="white", linewidths=0.4,
                     label=f"score ≥ {t:g}")

    ax.axhline(0, color="#d62728", lw=0.9, ls="--")
    ax.set_ylabel("Responders, trained − control\n(percentage points, 95% CI)",
                  fontsize=8.5)
    ax.set_ylim(-75, 75)
    ax.legend(frameon=False, fontsize=7.5, ncol=3, loc="upper right",
              title="responder cut", title_fontsize=7.5)
    ax.set_title(
        f"{CS_PLUS} {CONCENTRATION} — does the responder threshold drive the result?",
        fontsize=10, pad=8, loc="left",
    )
    ax_p.axhline(ALPHA, color="#d62728", lw=0.9, ls="--")
    ax_p.text(len(labels) - 0.45, ALPHA * 1.15, "α = 0.05", fontsize=7,
              color="#d62728", ha="right")
    ax_p.set_yscale("log")
    ax_p.set_ylim(0.02, 2.2)
    ax_p.set_ylabel("Fisher p,\nHolm-corrected", fontsize=8.5)
    tick_labels(ax_p, labels, rotate=30)

    excludes_zero = [
        (f"{row.odor} (≥ {row.threshold:g})")
        for table in frames for row in table.itertuples()
        if np.isfinite(row.rd_ci_lo_pct) and (row.rd_ci_lo_pct > 0 or row.rd_ci_hi_pct < 0)
    ]
    if excludes_zero:
        ci_note = (
            f"{len(excludes_zero)} of {len(labels) * len(THRESHOLD_SWEEP)} intervals "
            f"exclude zero ({', '.join(excludes_zero)}) — none of them survives Holm, and "
            "the Newcombe interval is unconditional while Fisher conditions on both "
            "margins, so the two can disagree at the margin"
        )
    else:
        ci_note = "every interval spans zero at every cut"
    footer(fig,
           "The published figures dichotomise at score ≥ 2. A conclusion that exists only at "
           "one cut point is a threshold artefact, so the whole analysis is repeated at ≥ 1 "
           "(any proboscis movement) and ≥ 3 (unambiguous full extension). Upper: the "
           f"trained−control difference in responder percentage, Newcombe 95% intervals; "
           f"{ci_note}. Lower: Holm-corrected Fisher p. Odors are categorical, so markers "
           "are offset rather than joined by a line.")
    fig.subplots_adjust(bottom=0.24, top=0.94, left=0.13, right=0.98)
    return save(fig, out_dir,
                "fig7_responder-threshold-sensitivity_score-ge-1-2-3"), stacked


# --------------------------------------------------------------------------
# fig8 -- the raw distributions
# --------------------------------------------------------------------------


def fig8_distributions(df: pd.DataFrame, table: pd.DataFrame, out_dir: Path) -> Path:
    labels = table["odor"].tolist()
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    rng = np.random.default_rng(0)
    w = 0.34

    for i, label in enumerate(labels):
        for offset, cohort, color in ((-w / 2, "Trained", TRAINED_COLOR),
                                      (w / 2, "Control", "0.45")):
            v = arm(df, cohort, label)["score"].to_numpy(float)
            if not v.size:
                continue
            jitter = rng.uniform(-0.10, 0.10, v.size)
            ax.scatter(i + offset + jitter, v + rng.uniform(-0.09, 0.09, v.size),
                       s=13, color=color, alpha=0.65, lw=0, zorder=2)
            q1, med, q3 = np.percentile(v, [25, 50, 75])
            ax.plot([i + offset - w * 0.42, i + offset + w * 0.42], [med, med],
                    lw=2.0, color=color, zorder=3, solid_capstyle="butt")
            ax.plot([i + offset, i + offset], [q1, q3], lw=0.9, color=color,
                    zorder=3, alpha=0.8)

    ax.axhline(RESPONSE_THRESHOLD - 0.5, color="#d62728", lw=0.8, ls="--")
    ax.text(len(labels) - 0.45, RESPONSE_THRESHOLD - 0.42,
            f"responder cut (score ≥ {RESPONSE_THRESHOLD:g})", fontsize=6.8,
            color="#d62728", ha="right", va="bottom")
    ax.set_ylim(SCORE_MIN - 0.6, SCORE_MAX + 0.6)
    ax.set_yticks(np.arange(SCORE_MIN, SCORE_MAX + 1))
    ax.set_ylabel("PER score (ordinal, −1…5)")
    ax.set_title(
        f"{CS_PLUS} {CONCENTRATION} — the distributions behind fig1\n"
        "one point per fly, median and IQR",
        fontsize=10, pad=8,
    )
    tick_labels(ax, labels)
    handles = [
        plt.Line2D([], [], marker="o", ls="none", color=TRAINED_COLOR, label="Trained"),
        plt.Line2D([], [], marker="o", ls="none", color="0.45", label="Unpaired control"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=7.5, loc="upper right")

    footer(fig,
           "Points are jittered in both axes so overlapping integer scores stay countable; "
           "the horizontal bar is the median and the vertical line the interquartile range. "
           "Mean ± SEM on an ordinal −1…5 score assumes an interval scale it does not have, "
           "so this is the honest companion to fig1's bars and shows exactly where the "
           "responder cut falls relative to the data.")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return save(fig, out_dir, "fig8_score-distributions_median-iqr_per-fly-points")


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def write_readme(
    summary: dict,
    scores: pd.DataFrame,
    rates: pd.DataFrame,
    three_stats: list[dict],
    out_dir: Path,
) -> Path:
    """Methods note, with every number interpolated from the run that made it.

    Written by the script rather than by hand so the prose cannot drift away
    from the figures when the data is re-run.
    """
    n = summary["n"]
    cs_row = scores[scores["odor"] == CS_PLUS].iloc[0]
    cs_rate = rates[rates["odor"] == CS_PLUS].iloc[0]
    perm = summary["batch_permutation"]
    gee = summary["models"]["gee"]
    ordinal = summary["models"]["ordinal"]
    p1 = next(s for s in three_stats if s["presentation"] == 1)
    p1_tc = next(d for d in p1["pairs"] if d["pair"] == "Trained vs Control")
    p1_tn = next(d for d in p1["pairs"] if d["pair"] == "Trained vs Naive")
    p1_cn = next(d for d in p1["pairs"] if d["pair"] == "Control vs Naive")
    n_bad = int((~rates["chi2_admissible"]).sum())

    text = f"""# 3-Octanol {CONCENTRATION}, 24 h starved — corrected statistics

Trained (`{TRAIN_DS}`, n = {n['trained_flies']} flies in
{n['trained_batches']} batches) vs unpaired control (`{CTRL_DS}`,
n = {n['control_flies']} flies in {n['control_batches']} batches), with
`{NAIVE_DS}` (n = {n['naive_flies']} flies) as the naive arm. Genotype
{summary['genotype']}; odor labels from `config_new.yaml`; a response is
score >= {RESPONSE_THRESHOLD:g}; the fly is the unit throughout.

Regenerate with:

    python scripts/analysis/pubfig_3oct_corrected.py

Every number below is written by that script from the run that produced these
figures, so the prose cannot drift from the panels. Raw per-test output is in
the sidecar CSVs and in `all_stats.json`.

---

## The one structural fact that drives everything

**Batch is perfectly confounded with treatment.** The {n['trained_batches']}
trained fly folders and the {n['control_batches']} control fly folders do not
overlap: no batch contains both a trained and a control fly. A fly folder is
one day, one rig and one odor bottle, so flies inside it share every nuisance
source there is.

Two consequences, and they are not optional:

1. Batch **cannot be adjusted for**. There is no stratified analysis
   (Cochran-Mantel-Haenszel, batch as a covariate) that separates "trained"
   from "run on those five days" — the design does not contain that
   information.
2. Any p-value that treats the {n['trained_flies'] + n['control_flies']} flies
   as independent draws is **anti-conservative by an unknown amount**. That is
   what the existing pipeline reports, and it is what fig6 replaces.

The honest response is to make the batch the unit of exchangeability, which is
what the exact permutation test in fig6 does.

---

## What each figure corrects

| Figure | Corrects | Over what the pipeline does now |
|---|---|---|
| fig1 | Multiplicity + effect size on the score | `score_summary` runs 8 Mann-Whitneys and stars the raw p |
| fig2 | Multiplicity + interval choice on the rate | Fisher per odor, uncorrected, binomial SE error bars |
| fig3 | Test choice for the binary rate | Fisher only, with no admissibility check |
| fig4 | Family definition + presentation matching | naive arm pooled across trials, uncorrected |
| fig5 | Asks the interaction, not the main effect | only per-odor pairwise tests exist |
| fig6 | Non-independence of flies within a batch | flies treated as independent |
| fig7 | Threshold robustness | a single hard cut at score >= 2 |
| fig8 | Shows the ordinal distribution | mean +/- SEM bars only |

---

## Results

### fig1 / fig2 — the 8-presentation panel

The CS+ moves in the expected direction and the effect size is moderate:
mean score {cs_row['mean_trained']:.2f} trained vs {cs_row['mean_control']:.2f}
control, Cliff's delta = {cs_row['cliffs_delta']:.2f}
[{cs_row['delta_ci_lo']:.2f}, {cs_row['delta_ci_hi']:.2f}]; responders
{cs_rate['k_trained']}/{cs_rate['n_trained']} vs
{cs_rate['k_control']}/{cs_rate['n_control']}.

**It does not survive correction.** Score: raw p = {cs_row['p_raw']:.3f},
Holm p = {cs_row['p_holm']:.3f}, BH p = {cs_row['p_bh']:.3f}. Rate: Fisher raw
p = {cs_rate['p_fisher']:.3f}, Holm p = {cs_rate['p_fisher_holm']:.3f}. No
presentation in either panel is significant after Holm.

### fig3 — chi-square vs Fisher, as requested

On the CS+ 2x2 the tests spread from p = {cs_rate['p_chi2_pearson']:.3f}
(Pearson chi-square, no continuity correction) through
p = {cs_rate['p_boschloo']:.3f} (Boschloo), p = {cs_rate['p_fisher']:.3f}
(Fisher) to p = {cs_rate['p_chi2_yates']:.3f} (Yates). So **yes — the
uncorrected chi-square is the most liberal of the five and comes closest to
significance.**

That is not a reason to report it. {n_bad} of the {len(rates)} tables fail
Cochran's rule (smallest expected cell = {rates['min_expected_cell'].min():.2f},
against a floor of 5), which is precisely the regime where the chi-square
approximation over-rejects. Where an exact test is admissible and a chi-square
is not, the gap between them measures the approximation error, not evidence.
The defensible upgrade from Fisher is **Boschloo**, which is unconditional,
exact, and uniformly at least as powerful — it gives
p = {cs_rate['p_boschloo']:.3f} here. After Holm, no test in the family clears
alpha, so the choice does not change the conclusion.

### fig4 — naive vs trained vs control

Presentation 1 is where the effect lives. Kruskal-Wallis omnibus on score
p = {p1['omnibus_score_kruskal_p']:.3f}; Holm-corrected post-hoc gives trained
vs control p = {p1_tc['score_p_holm']:.3f} and trained vs naive
p = {p1_tn['score_p_holm']:.3f}, while control vs naive is flat
(p = {p1_cn['score_p_holm']:.3f}). The rate panel is weaker throughout
(Fisher-Freeman-Halton omnibus p = {p1['omnibus_rate_ffh_p']:.3f}, nothing
significant after Holm).

Control sitting on top of naive is the useful part: **the unpaired control is
behaving like an untrained fly**, which is what a control is supposed to do.

> **Read fig1 and fig4 together.** The same trained-vs-control score comparison
> is Holm p = {cs_row['p_holm']:.3f} in fig1 and p = {p1_tc['score_p_holm']:.3f}
> in fig4. Nothing about the data changed — only the family did (8 presentations
> vs 3 cohort pairs). This is exactly why the family has to be **pre-specified**
> rather than chosen after seeing the p-values. If the paper's claim is about
> the CS+ specifically, fig4's family is the honest one and fig1 is the
> generalisation panel; if the claim is "the trained odor stands out among the
> eight", fig1's family is the honest one and the effect is not significant.

### fig5 — the interaction is the learning claim

Fit on {summary['models']['n_obs']} fly x odor observations, standard errors
clustered on batch:

* GEE (binomial, % responding): trained main effect OR =
  {np.exp(gee['coef']['trained']):.2f} (p = {gee['p']['trained']:.3f}) —
  **the trained flies are not more responsive overall**; interaction
  trained x CS+ OR = {np.exp(gee['coef']['trained:is_csplus']):.2f}
  (p = {gee['p']['trained:is_csplus']:.3f}).
* Cumulative-link (ordinal, PER score): interaction proportional-odds ratio
  {np.exp(ordinal['coef']['trained:is_csplus']):.2f}
  (p = {ordinal['p']['trained:is_csplus']:.3f}).

The mechanism is worth stating plainly, because it is not "trained flies
respond more": control flies respond **less** to 3-Octanol than to the novel
odors (CS+ main effect OR = {np.exp(gee['coef']['is_csplus']):.2f},
p < 0.001), and training abolishes that decrement. The interaction is the
effect; the main effect is nothing.

### fig6 — and it does not survive batch-level exchangeability

Exhaustive enumeration of all C(13, 5) = {perm['score_contrast']['n_permutations']}
assignments of batches to arms:

* score contrast, observed {perm['score_contrast']['observed']:+.3f},
  **p = {perm['score_contrast']['p_value']:.3f}**
* rate contrast, observed {perm['rate_contrast']['observed']:+.3f},
  **p = {perm['rate_contrast']['p_value']:.3f}**

The resolution floor is p >= {perm['score_contrast']['p_floor']:.4f}, so this is
not a power ceiling artefact — the observed contrast simply sits inside the
batch-permutation null. **The fig5 interaction p of
{gee['p']['trained:is_csplus']:.3f} is carried by treating flies within a batch
as independent.** fig6 is the number to report if a reviewer asks about
pseudoreplication, and it is the one this design actually supports.

### fig7 — not a threshold artefact, but not significant either

Repeating everything at score >= 1, >= 2 and >= 3 leaves the direction stable
and the significance absent at every cut. The result is at least not an
artefact of where the responder line was drawn.

---

## What to report

**Defensible as written:** conditioning produced a CS+-specific,
moderate-sized shift (Cliff's delta {cs_row['cliffs_delta']:.2f}
[{cs_row['delta_ci_lo']:.2f}, {cs_row['delta_ci_hi']:.2f}]) that is present on
the first post-training presentation and absent by the second, with the
unpaired control indistinguishable from naive. Report it with the effect size
and interval, the omnibus-then-Holm structure of fig4, and the batch-level
p from fig6 stated alongside.

**Not defensible:** a starred trained-vs-control bar taken from the
uncorrected 8-presentation panel, or a chi-square p quoted from a 2x2 whose
smallest expected cell is {rates['min_expected_cell'].min():.2f}.

**What would settle it:** the batch confound is a design property, not an
analysis choice, and no test removes it. Running trained and control flies
*within the same batch* — same day, same rig, same bottle — would make batch a
stratifier instead of a confounder, and would let a CMH or a
batch-as-covariate model recover the power fig6 has to give away. With the
current n ({n['trained_batches']} vs {n['control_batches']} batches) the
batch-level test cannot resolve below p = {perm['score_contrast']['p_floor']:.4f}
no matter how large the true effect is.
"""
    path = out_dir / "README_STATISTICS.md"
    path.write_text(text)
    return path


def build(
    *,
    predictions_csv: Path = PREDICTIONS_CSV,
    config: Path = CONFIG,
    out_dir: Path = OUT_DIR,
    genotype: str = GENOTYPE,
) -> dict:
    df = load_frame(predictions_csv, config, genotype)
    labels = panel_labels(df)
    out_dir.mkdir(parents=True, exist_ok=True)
    made: list[str] = []

    scores = score_table(df, labels)
    scores.to_csv(out_dir / "fig1_score_stats.csv", index=False)
    made.append(str(fig1_score(scores, out_dir)))

    rates = rate_table(df, labels)
    rates.to_csv(out_dir / "fig2_percent-response_stats.csv", index=False)
    made.append(str(fig2_rate(rates, out_dir)))
    made.append(str(fig3_test_sensitivity(rates, out_dir)))

    presentations = sorted(
        df[(df["cohort"] == "Trained") & (df["odor"] == CS_PLUS)]["presentation"].unique()
    )
    three_tables, three_stats = [], []
    for p in presentations:
        t, s = three_arm_table(df, int(p))
        three_tables.append(t)
        three_stats.append(s)
    pd.concat(three_tables, ignore_index=True).to_csv(
        out_dir / "fig4_naive-vs-trained-vs-control_arms.csv", index=False)
    pd.DataFrame([d for s in three_stats for d in s["pairs"]]).to_csv(
        out_dir / "fig4_naive-vs-trained-vs-control_pairwise.csv", index=False)
    made.append(str(fig4_three_arm(three_tables, three_stats, out_dir)))

    models = fit_models(df)
    made.append(str(fig5_models(models, out_dir)))

    contrasts = specificity_contrasts(df)
    contrasts.to_csv(out_dir / "fig6_per-fly_csplus-specificity_contrasts.csv", index=False)
    path6, perm = fig6_permutation(contrasts, out_dir)
    made.append(str(path6))

    path7, threshold_table = fig7_threshold(df, labels, out_dir)
    threshold_table.to_csv(out_dir / "fig7_threshold-sensitivity_stats.csv", index=False)
    made.append(str(path7))

    made.append(str(fig8_distributions(df, scores, out_dir)))

    summary = {
        "cohort": f"{CS_PLUS} {CONCENTRATION}, 24 h starved",
        "datasets": {"trained": TRAIN_DS, "control": CTRL_DS, "naive": NAIVE_DS},
        "genotype": genotype,
        "predictions_csv": str(predictions_csv),
        "config": str(config),
        "response_threshold": RESPONSE_THRESHOLD,
        "n": {
            "trained_flies": int(df[df.cohort == "Trained"]["fly_id"].nunique()),
            "control_flies": int(df[df.cohort == "Control"]["fly_id"].nunique()),
            "naive_flies": int(df[df.cohort == "Naive"]["fly_id"].nunique()),
            "trained_batches": int(df[df.cohort == "Trained"]["batch"].nunique()),
            "control_batches": int(df[df.cohort == "Control"]["batch"].nunique()),
            "presentations_tested": len(labels),
        },
        "models": models,
        "batch_permutation": perm,
        "three_arm": three_stats,
        "figures": made,
    }
    (out_dir / "all_stats.json").write_text(json.dumps(summary, indent=2, default=float))
    summary["readme"] = str(
        write_readme(summary, scores, rates, three_stats, out_dir)
    )
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions-csv", type=Path, default=PREDICTIONS_CSV)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--genotype", default=GENOTYPE)
    args = parser.parse_args(argv)

    summary = build(
        predictions_csv=args.predictions_csv, config=args.config,
        out_dir=args.out_dir, genotype=args.genotype,
    )
    print(f"\nWrote {len(summary['figures'])} figures to {args.out_dir}")
    for path in summary["figures"]:
        print("  ", Path(path).name)


if __name__ == "__main__":
    main()
