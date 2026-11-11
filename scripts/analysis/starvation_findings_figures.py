#!/usr/bin/env python3
"""Plain-language findings figures for a starvation comparison.

Each figure answers ONE question. The title is the question, and the line
under it is the answer, computed from the data, so it stays true when the
data change. Reads the folder written by ``starvation_pre_post_comparison.py``
(``paired_scores_per_fly.csv``) and, optionally, the
``odor_holdout_regression.py`` output folder.

    01_did_training_raise_PER.png        % of flies whose PER went up / same / down
    02_was_it_specific_to_trained_odor.png  trained-odor change minus other-odor change
    03_where_did_PER_change.png          odor x starvation map of the mean change
    04_starvation_before_training.png    pre-test (naive) PER by starvation group
    05_general_responsiveness.png        responding to other odors predicts this one
    06_time_of_day_before_training.png   pre-test PER vs clock time (needs raw data)
    FINDINGS.txt                         every answer line, in order

Colour: green = PER went up, purple = went down (the repo's PRGn score
semantics: green = more PER), gray = no change; validated (poles pass every
categorical check, CVD ΔE 15.5). Blue = pre-test, as everywhere in this repo.
All p-values uncorrected; n counts flies unless a panel says fly-odor pairs.

Usage
-----
    python3 scripts/analysis/starvation_findings_figures.py \\
        --comparison-dir <root>/1h_vs_6h_vs_12h_vs_18h_vs_24h_vs_30h \\
        --regression-dir <root>/odor_holdout_regression
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy.stats import kruskal, spearmanr, wilcoxon  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT, REPO_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from scripts.analysis import phase_terms  # noqa: E402
from scripts.analysis.pretest_vs_test_comparison import (  # noqa: E402
    BINARY_THRESHOLD,
    INK,
    INK_MUTED,
    PRE_COLOR,
    _RC_CONTEXT,
    _recessive_grid,
    wilcoxon_p,
)
from scripts.analysis.starvation_pre_post_comparison import (  # noqa: E402
    PREDICTIONS_CSV,
    _phase_cohort,
    odor_concentrations,
    odor_text,
)
from scripts.analysis.starvation_trend_figure import hours_from_label  # noqa: E402

UP, SAME, DOWN = "#2f8a44", "#b9b8b2", "#8a4f9e"
DOT = "#a8a7a1"
EMPH = "#1a1a19"
DIVERGING = LinearSegmentedColormap.from_list("down_up", [DOWN, "#f4f3f0", UP])
SCORE_HINT = "0 = no extension, 5 = strongest"


# ── helpers ──────────────────────────────────────────────────────────────


def group_order(groups) -> list[str]:
    return sorted(set(groups), key=hours_from_label)


def _fmt_p(p) -> str:
    if p is None or pd.isna(p):
        return "n/a"
    return "p<0.001" if p < 0.001 else f"p={p:.3f}"


def boot_ci(values: np.ndarray, *, n_boot: int = 5000, seed: int = 0) -> tuple[float, float]:
    """95% percentile bootstrap CI of the mean (resampling the given units)."""
    v = np.asarray(values, dtype=float)
    if v.size < 2:
        return (float(v.mean()), float(v.mean())) if v.size else (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, v.size, size=(n_boot, v.size))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _with_delta(paired: pd.DataFrame) -> pd.DataFrame:
    out = paired.copy()
    out["delta"] = out["score_post"] - out["score_pre"]
    return out


def _fly_key(df):
    return [c for c in ("group", "dataset_canon", "fly", "fly_number") if c in df]


def _cohort(paired: pd.DataFrame) -> str | None:
    """The cohort the wording follows (None unless every dataset drawn is a sensitivity one)."""
    return _phase_cohort(paired.get("dataset_canon", ()))


def _pre_name(cohort: str | None, *, lower: bool = False) -> str:
    """The first panel: "Pre-test", or "Pre-exposure" for a sensitivity cohort."""
    if lower:
        return phase_terms.pre_label(cohort, "pre-test", case="lower")
    return phase_terms.pre_label(cohort, "Pre-test")


def _block_name(cohort: str | None) -> str:
    """What the post panel comes after: "training", or "exposure" for a sensitivity cohort."""
    return phase_terms.block_label(cohort)


# ── tables ───────────────────────────────────────────────────────────────


def direction_shares(paired: pd.DataFrame, *, odors: Sequence[str]) -> pd.DataFrame:
    """Per group: % of fly-odor pairs whose score went up / stayed / went down."""
    d = _with_delta(paired[paired["odor"].isin(odors)])
    rows = {}
    for g in group_order(d["group"]):
        s = d.loc[d["group"] == g, "delta"]
        rows[g] = {"up": 100.0 * (s > 0).mean(), "same": 100.0 * (s == 0).mean(),
                   "down": 100.0 * (s < 0).mean(), "n": int(s.size)}
    return pd.DataFrame.from_dict(rows, orient="index")


def specificity(paired: pd.DataFrame, *, trained: str) -> pd.DataFrame:
    """Per fly: trained-odor change minus the mean change of its other odors."""
    d = _with_delta(paired)
    key = _fly_key(d)
    wide = d.pivot_table(index=key, columns="odor", values="delta")
    others = wide.drop(columns=[trained])
    out = (wide[trained] - others.mean(axis=1)).rename("index").reset_index()
    return out.dropna(subset=["index"])


def change_matrix(paired: pd.DataFrame, *, trained: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mean change (post - pre) and paired Wilcoxon p, odors x groups, trained odor first."""
    d = _with_delta(paired)
    groups = group_order(d["group"])
    odors = [trained] + sorted((o for o in d["odor"].unique() if o != trained), key=str.casefold)
    mean = pd.DataFrame(index=odors, columns=groups, dtype=float)
    p = pd.DataFrame(index=odors, columns=groups, dtype=float)
    for o in odors:
        for g in groups:
            s = d[(d["odor"] == o) & (d["group"] == g)]
            if len(s):
                mean.loc[o, g] = s["delta"].mean()
                pv = wilcoxon_p(s["score_pre"].to_numpy(float), s["score_post"].to_numpy(float))
                p.loc[o, g] = np.nan if pv is None else pv
    return mean, p


def baseline(paired: pd.DataFrame) -> pd.DataFrame:
    """Per group: pre-test score pooled over every trial (sum / n trials); CI over flies."""
    rows = []
    for g in group_order(paired["group"]):
        s = paired[paired["group"] == g]
        per_fly = s.groupby(_fly_key(s))["score_pre"].mean().to_numpy(float)
        lo, hi = boot_ci(per_fly)
        rows.append({"group": g, "hours": hours_from_label(g), "n_flies": per_fly.size,
                     "n_trials": len(s), "mean_pre": float(s["score_pre"].mean()),
                     "ci_lo": lo, "ci_hi": hi,
                     "pct_responding": 100.0 * float((s["score_pre"] >= BINARY_THRESHOLD).mean())})
    return pd.DataFrame(rows)


_RESP_BINS = [-np.inf, 0.5, 1.0, 1.5, 2.0, np.inf]
_RESP_LABELS = ["under 0.5", "0.5–1", "1–1.5", "1.5–2", "2 or more"]


def responsiveness(paired: pd.DataFrame, *,
                   binary_threshold: int = BINARY_THRESHOLD) -> pd.DataFrame:
    """Pre-test: each fly x odor, the fly's mean score to its OTHER odors vs whether
    it responded (score >= threshold) to this one. Binned by the other-odor mean."""
    key = _fly_key(paired)
    wide = paired.pivot_table(index=key, columns="odor", values="score_pre")
    rows = []
    for odor in wide.columns:
        other = wide.drop(columns=[odor]).mean(axis=1)
        ok = wide[odor].notna() & other.notna()
        rows.append(pd.DataFrame({"other_mean": other[ok].to_numpy(),
                                  "responded": (wide.loc[ok, odor] >= binary_threshold).to_numpy()}))
    pairs = pd.concat(rows, ignore_index=True)
    pairs["bin"] = pd.cut(pairs["other_mean"], _RESP_BINS, labels=_RESP_LABELS, right=False)
    out = (pairs.groupby("bin", observed=False)
           .agg(n=("responded", "size"), pct_responding=("responded", "mean")).reset_index())
    out["pct_responding"] *= 100.0
    out["bin"] = out["bin"].astype(str)
    return out


def regression_table(reg_dir: Path) -> pd.DataFrame | None:
    """Ridge, leave-one-batch-out, score targets: per odor/phase the Q2 of each predictor set."""
    reg_dir = Path(reg_dir)
    l2, l3 = reg_dir / "level2_individual_pooled.csv", reg_dir / "level3_population_batches.csv"
    if not (l2.exists() and l3.exists()):
        return None
    a, b = pd.read_csv(l2), pd.read_csv(l3)
    sel = lambda d, t: d[(d["model"] == "Ridge") & (d["target"] == t)  # noqa: E731
                         & (d["scheme"] == "leave-one-batch-out")]
    fly = sel(a, "score")
    bat = sel(b, "mean score")
    out = pd.concat([
        fly[fly["variant"] == "odors"].assign(series="single fly, from its other odors"),
        bat[bat["variant"] == "odors"].assign(series="batch average, from its other odors"),
        fly[fly["variant"] == "starvation only"].assign(series="single fly, from starvation time only"),
    ])
    return out[["series", "odor", "phase", "n", "q2", "p_perm"]].reset_index(drop=True)


def time_blocks(clock: pd.Series, *, gap_h: float = 1.5) -> pd.Series:
    """Block number per start time: a new block wherever no experiment started for > gap_h.

    Sessions cluster (morning / early afternoon / evening), and fixed clock bins
    would cut one session in half.
    """
    order = clock.sort_values()
    block = (order.diff() > gap_h).cumsum()
    return block.reindex(clock.index)


# ── figures ──────────────────────────────────────────────────────────────


def _finish(fig, question: str, answer: str, *, bottom: float = 0.0, tight: bool = True) -> None:
    """Question as the bold title, the answer wrapped underneath, plot laid out below both.

    The header height is measured in inches (title + wrapped answer lines) so it
    never collides with panel titles, whatever the figure height.
    """
    w, h = fig.get_size_inches()
    width_chars = max(40, int(w * 12.5))
    lines = [ln for para in answer.split("\n") for ln in textwrap.wrap(para, width_chars)]
    header_in = 0.42 + 0.19 * len(lines) + 0.08
    top = 1 - header_in / h
    if tight:
        fig.tight_layout(rect=(0, bottom, 1, top))
    else:
        fig.subplots_adjust(top=top)
    fig.suptitle(question, x=0.01, y=1 - 0.06 / h, ha="left", va="top", fontsize=12.5,
                 color=INK, fontweight="bold")
    fig.text(0.01, 1 - 0.42 / h, "\n".join(lines), ha="left", va="top", fontsize=9.5,
             color=INK, linespacing=1.35)
    fig.answer = f"{question}\n  {answer}"


def _diverging_rows(ax, shares: pd.DataFrame, *, n_word: str) -> None:
    y = np.arange(len(shares))[::-1]
    for yi, (g, r) in zip(y, shares.iterrows()):
        half = r["same"] / 2
        segs = ((-half - r["down"], r["down"], DOWN, "white"), (-half, r["same"], SAME, INK),
                (half, r["up"], UP, "white"))
        for left, width, color, tc in segs:
            if width <= 0:
                continue
            ax.barh(yi, width, left=left, height=0.62, color=color, edgecolor="white",
                    linewidth=2, zorder=2)
            if width >= 12:  # only where "NN%" fits inside the segment with padding
                ax.text(left + width / 2, yi, f"{width:.0f}%", ha="center", va="center",
                        fontsize=8, color=tc, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{g}  (n={int(r['n'])} {n_word})" for g, r in shares.iterrows()])
    ax.axvline(0, color=INK_MUTED, linewidth=0.8, zorder=1)
    lim = 100
    ax.set_xlim(-lim, lim)
    ax.set_xticks([-100, -50, 0, 50, 100])
    ax.set_xticklabels(["100%", "50%", "0", "50%", "100%"])
    ax.tick_params(axis="y", length=0)
    for side in ("left", "top", "right"):
        ax.spines[side].set_visible(False)


def fig_direction(paired: pd.DataFrame, *, trained: str, concentrations=None):
    """Q1: did training raise PER to the trained odor? Up / same / down per group.

    ``concentrations`` (``odor_concentrations``): the percent in each drawn odor name.
    """
    cohort = _cohort(paired)
    role = phase_terms.odor_tag(cohort, "trained odor")
    name = odor_text(trained, cohort, concentrations)
    others = sorted(o for o in paired["odor"].unique() if o != trained)
    cs = direction_shares(paired, odors=[trained])
    ot = direction_shares(paired, odors=others)
    ps = {}
    for g in cs.index:
        s = paired[(paired["odor"] == trained) & (paired["group"] == g)]
        ps[g] = wilcoxon_p(s["score_pre"].to_numpy(float), s["score_post"].to_numpy(float))
    sig = [g for g in cs.index if ps[g] is not None and ps[g] < 0.05 and cs.loc[g, "up"] > cs.loc[g, "down"]]
    if sig:
        best = sig[0]
        answer = (f"Clearly only at {', '.join(sig)}: {cs.loc[best, 'up']:.0f}% of flies went up, "
                  f"{cs.loc[best, 'down']:.0f}% went down ({_fmt_p(ps[best])}). "
                  "Other starvation times: no significant rise.")
    else:
        answer = "No starvation group shows a significant rise (paired Wilcoxon, p<0.05)."
    with plt.rc_context(_RC_CONTEXT):
        fig, (a, b) = plt.subplots(1, 2, figsize=(12.0, 0.55 * len(cs) + 2.6), sharey=False)
        _diverging_rows(a, cs, n_word="flies")
        for yi, g in zip(np.arange(len(cs))[::-1], cs.index):
            a.text(103, yi, _fmt_p(ps[g]) if ps[g] is not None else "too few changed",
                   va="center", ha="left", fontsize=8.5, color=INK,
                   fontweight="bold" if ps[g] is not None and ps[g] < 0.05 else "normal",
                   clip_on=False)
        _diverging_rows(b, ot, n_word="fly-odor pairs")
        a.set_title(f"{name} — the {role}", loc="left", fontsize=10.5, color=INK,
                    fontweight="bold")
        b.set_title(f"The other {len(others)} odors (not trained), every fly × odor",
                    loc="left", fontsize=10.5, color=INK)
        a.set_xlabel("← % of flies whose PER went down   |   went up →")
        b.set_xlabel("← % went down   |   went up →")
        after = f"after {_block_name(cohort)}"
        handles = [Patch(color=DOWN, label=f"PER went down {after}"),
                   Patch(color=SAME, label="no change"),
                   Patch(color=UP, label=f"PER went up {after}")]
        fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
                   bbox_to_anchor=(0.5, -0.01))
        _finish(fig, f"Did {_block_name(cohort)} raise the PER to the {role} ({name})?", answer,
                bottom=0.06)
    return fig


def fig_specificity(paired: pd.DataFrame, *, trained: str, concentrations=None):
    """Q2: did PER rise MORE for the trained odor than for the others, in the same fly?"""
    cohort = _cohort(paired)
    role = phase_terms.odor_tag(cohort, "trained odor")
    name = odor_text(trained, cohort, concentrations)
    s = specificity(paired, trained=trained)
    groups = group_order(s["group"])
    rng = np.random.default_rng(0)
    stats = []
    with plt.rc_context(_RC_CONTEXT):
        fig, ax = plt.subplots(figsize=(8.6, 5.0))
        ax.axhline(0, color=INK_MUTED, linewidth=1.0, zorder=1)
        for i, g in enumerate(groups):
            v = s.loc[s["group"] == g, "index"].to_numpy(float)
            ax.scatter(i + rng.uniform(-0.18, 0.18, v.size), v, s=22, color=DOT,
                       edgecolors="white", linewidths=0.8, zorder=2)
            lo, hi = boot_ci(v)
            ax.plot([i, i], [lo, hi], color=EMPH, linewidth=2, zorder=3, solid_capstyle="round")
            ax.plot(i, v.mean(), "o", color=EMPH, markersize=9, markeredgecolor="white",
                    markeredgewidth=2, zorder=4)
            p = None
            if (v != 0).sum() >= 1:
                try:
                    p = float(wilcoxon(v).pvalue)
                except ValueError:
                    p = None
            stats.append((g, v.mean(), p, v.size))
            ax.text(i, 0.985, _fmt_p(p), transform=ax.get_xaxis_transform(), ha="center",
                    va="top", fontsize=8.5, color=INK,
                    fontweight="bold" if p is not None and p < 0.05 else "normal")
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([f"{g}\nn={n}" for g, _m, _p, n in stats])
        ax.set_xlabel("Starvation before the experiment")
        ax.set_ylabel(f"{name} change − average change\nof the other odors (score points)")
        lo_all, hi_all = float(s["index"].min()), float(s["index"].max())
        ax.set_ylim(lo_all - 0.4, hi_all + 1.1)  # headroom for the p-values
        side = ax.get_yaxis_transform()  # x in axes fraction, y in data
        ax.text(1.01, 0.25, f"↑ rose more for the {role}", transform=side, ha="left",
                va="bottom", fontsize=8, color=INK_MUTED, clip_on=False)
        ax.text(1.01, -0.25, "↓ rose more for the other odors", transform=side, ha="left",
                va="top", fontsize=8, color=INK_MUTED, clip_on=False)
        ax.set_xlim(-0.6, len(groups) - 0.4)
        _recessive_grid(ax)
        legend = [plt.Line2D([], [], marker="o", linestyle="", color=DOT, label="one fly"),
                  plt.Line2D([], [], marker="o", color=EMPH, label="average, 95% CI")]
        ax.legend(handles=legend, loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False)
        sig = [g for g, m, p, _n in stats if p is not None and p < 0.05 and m > 0]
        answer = (f"Only at {', '.join(sig)} did PER rise more for {name} than for the "
                  f"other odors in the same fly (one-sample Wilcoxon vs 0)." if sig else
                  f"In no group did PER rise significantly more for the {role} than for the others.")
        _finish(fig, f"Was the rise specific to the {role}?", answer)
    return fig


def fig_change_map(paired: pd.DataFrame, *, trained: str, concentrations=None):
    """Q3: odor x starvation map of the mean change; * = paired Wilcoxon p<0.05."""
    cohort = _cohort(paired)
    mean, p = change_matrix(paired, trained=trained)
    names = {o: odor_text(o, cohort, concentrations) for o in mean.index}
    n = paired.drop_duplicates(_fly_key(paired)).groupby("group").size()
    lim = max(1.0, float(np.ceil(np.nanmax(np.abs(mean.to_numpy(float))) * 2) / 2))
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
    with plt.rc_context(_RC_CONTEXT):
        fig = plt.figure(figsize=(1.15 * mean.shape[1] + 4.2, 0.55 * mean.shape[0] + 2.6))
        gs = fig.add_gridspec(1, 2, width_ratios=[mean.shape[1], 0.25], wspace=0.08)
        ax, cax = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
        im = ax.imshow(mean.to_numpy(float), cmap=DIVERGING, norm=norm, aspect="auto")
        for i, o in enumerate(mean.index):
            for j, g in enumerate(mean.columns):
                v = mean.loc[o, g]
                if pd.isna(v):
                    continue
                star = "*" if pd.notna(p.loc[o, g]) and p.loc[o, g] < 0.05 else ""
                dark = abs(v) > 0.55 * lim
                ax.text(j, i, f"{v:+.1f}{star}", ha="center", va="center", fontsize=9.5,
                        color="white" if dark else INK,
                        fontweight="bold" if star else "normal")
        ax.set_xticks(range(mean.shape[1]))
        ax.set_xticklabels([f"{g}\nn={int(n.get(g, 0))}" for g in mean.columns])
        ax.set_yticks(range(mean.shape[0]))
        tag = phase_terms.odor_tag(cohort, "trained")
        ax.set_yticklabels([f"{names[o]}  ({tag})" if o == trained else names[o]
                            for o in mean.index])
        ax.get_yticklabels()[0].set_fontweight("bold")
        ax.set_xlabel("Starvation before the experiment")
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xticks(np.arange(-0.5, mean.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, mean.shape[0], 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=2)
        ax.tick_params(which="minor", length=0)
        cb = fig.colorbar(im, cax=cax)
        cb.set_label("Mean change in PER score per fly (post − pre)")
        cb.outline.set_visible(False)
        fig.subplots_adjust(left=0.2, right=0.9, bottom=0.14)
        sig = [(o, g) for o in mean.index for g in mean.columns
               if pd.notna(p.loc[o, g]) and p.loc[o, g] < 0.05]
        ups = [f"{names[o]} at {g}" for o, g in sig if mean.loc[o, g] > 0]
        downs = [f"{names[o]} at {g}" for o, g in sig if mean.loc[o, g] < 0]
        after = f"after {_block_name(cohort)}"
        answer = (f"Each cell: average change per fly (green = PER went up {after}, "
                  "purple = went down). * = p<0.05, paired Wilcoxon, uncorrected.\n"
                  f"Significant rises: {', '.join(ups) or 'none'}. "
                  f"Significant drops: {', '.join(downs) or 'none'}.")
        _finish(fig, f"Where did PER change {after}?", answer, tight=False)
    return fig


def fig_baseline(paired: pd.DataFrame):
    """Q4: before any training, does starvation change PER? Pre-test, all odors pooled."""
    cohort = _cohort(paired)
    b = baseline(paired)
    key = _fly_key(paired)
    per_fly = paired.groupby(key)["score_pre"].mean().reset_index()
    samples = [per_fly.loc[per_fly["group"] == g, "score_pre"].to_numpy(float) for g in b["group"]]
    p = float(kruskal(*samples).pvalue) if len(samples) > 1 and all(len(x) for x in samples) else None
    rng = np.random.default_rng(0)
    with plt.rc_context(_RC_CONTEXT):
        fig, ax = plt.subplots(figsize=(8.6, 5.0))
        for i, (g, v) in enumerate(zip(b["group"], samples)):
            ax.scatter(i + rng.uniform(-0.18, 0.18, v.size), v, s=20, color=DOT,
                       edgecolors="white", linewidths=0.8, zorder=2)
        x = np.arange(len(b))
        ax.vlines(x, b["ci_lo"], b["ci_hi"], color=PRE_COLOR, linewidth=2.2, zorder=3)
        ax.plot(x, b["mean_pre"], "o", color=PRE_COLOR, markersize=9, markeredgecolor="white",
                markeredgewidth=2, zorder=4)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{g}\nn={n}" for g, n in zip(b["group"], b["n_flies"])])
        ax.set_xlabel("Starvation before the experiment")
        ax.set_ylabel(f"{_pre_name(cohort)} PER score, all odors\n({SCORE_HINT})")
        ax.set_ylim(-0.3, max(3.0, float(per_fly["score_pre"].max()) + 0.3))
        _recessive_grid(ax)
        legend = [plt.Line2D([], [], marker="o", linestyle="", color=DOT,
                             label="one fly (its average over the odors)"),
                  plt.Line2D([], [], marker="o", color=PRE_COLOR,
                             label="group average, 95% CI")]
        ax.legend(handles=legend, loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False)
        lo, hi = b["mean_pre"].min(), b["mean_pre"].max()
        verdict = "No" if p is None or p >= 0.05 else "Yes"
        answer = (f"{verdict}: the average {_pre_name(cohort, lower=True)} score is "
                  f"{lo:.2f}–{hi:.2f} in every group (Kruskal-Wallis across groups {_fmt_p(p)}).")
        _finish(fig, f"Before {_block_name(cohort)}, does starvation change how much flies respond?",
                answer)
    return fig


def fig_responsiveness(paired: pd.DataFrame, *, regression: pd.DataFrame | None,
                       concentrations=None):
    """Q5: does responding to some odors predict responding to others?"""
    cohort = _cohort(paired)
    r = responsiveness(paired)
    r = r[r["n"] > 0]
    with plt.rc_context(_RC_CONTEXT):
        if regression is not None and len(regression):
            fig, (a, b) = plt.subplots(1, 2, figsize=(13.0, 5.2),
                                       gridspec_kw={"width_ratios": [1, 1.25]})
        else:
            fig, a = plt.subplots(figsize=(7.0, 5.0))
            b = None
        x = np.arange(len(r))
        a.bar(x, r["pct_responding"], 0.6, color=PRE_COLOR, zorder=2)
        for xi, v, n in zip(x, r["pct_responding"], r["n"]):
            a.text(xi, v + 1.5, f"{v:.0f}%", ha="center", va="bottom", fontsize=9, color=INK)
        a.set_xticks(x)
        a.set_xticklabels([f"{b_}\nn={n}" for b_, n in zip(r["bin"], r["n"])])
        a.set_ylim(0, 100)
        a.set_xlabel(f"Fly's average {_pre_name(cohort, lower=True)} score to the OTHER six odors")
        a.set_ylabel("% responding to this odor (score ≥ 2)")
        a.set_title(f"{_pre_name(cohort)}, every fly × odor", loc="left", fontsize=10, color=INK)
        _recessive_grid(a)
        lo_pct, hi_pct = r["pct_responding"].iloc[0], r["pct_responding"].iloc[-1]
        answer = (f"Yes. A fly that responds to the other odors is far more likely to respond to "
                  f"this one ({lo_pct:.0f}% → {hi_pct:.0f}%): flies differ in overall responsiveness.")
        if b is not None:
            pre = regression[regression["phase"] == "pre"]
            odors = sorted(pre["odor"].unique(), key=str.casefold)[::-1]
            styles = {"single fly, from its other odors": dict(marker="o", color=EMPH, ms=8),
                      "batch average, from its other odors": dict(marker="s", color=PRE_COLOR, ms=8),
                      "single fly, from starvation time only": dict(marker="x", color=INK_MUTED, ms=8)}
            for series, st in styles.items():
                s = pre[pre["series"] == series].set_index("odor")
                ys = [i for i, o in enumerate(odors) if o in s.index]
                xs = [float(np.clip(100 * s.loc[o, "q2"], -50, 100)) for o in odors if o in s.index]
                b.plot(xs, ys, linestyle="", marker=st["marker"], color=st["color"],
                       markersize=st["ms"], markeredgecolor="white" if st["marker"] != "x" else None,
                       markeredgewidth=1.5 if st["marker"] != "x" else 2, label=series, zorder=3)
            b.axvline(0, color=INK_MUTED, linewidth=1.0, zorder=1)
            b.set_yticks(range(len(odors)))
            b.set_yticklabels([odor_text(o, cohort, concentrations) for o in odors])
            top = float(np.nanmax(100 * pre["q2"].to_numpy(float)))
            b.set_xlim(-55, max(40.0, np.ceil((top + 10) / 10) * 10))
            b.set_xlabel("% of the differences predicted, beyond simply guessing the average\n"
                         "(0 = no better than the average; values below −50 drawn at −50)")
            n_fly = int(pre.loc[pre["series"] == "single fly, from its other odors", "n"].max())
            n_bat = int(pre.loc[pre["series"] == "batch average, from its other odors", "n"].max())
            b.set_title(f"Predicting the {_pre_name(cohort, lower=True)} score to each odor "
                        f"({n_fly} flies, {n_bat} batches;\n"
                        "Ridge, each batch predicted by a model that never saw it)",
                        loc="left", fontsize=10, color=INK)
            b.legend(loc="lower left", frameon=False, fontsize=8)  # the left side is empty (Q2 < 0 is rare)
            _recessive_grid(b)
            fly = pre[pre["series"] == "single fly, from its other odors"]["q2"].median()
            bat = pre[pre["series"] == "batch average, from its other odors"]["q2"].median()
            stv = pre[pre["series"] == "single fly, from starvation time only"]["q2"].median()
            pct = lambda q: f"{round(100 * q) + 0:.0f}%"  # noqa: E731  (+0 turns -0 into 0)
            answer += (f"\n{_pre_name(cohort)}: the other odors predict about {pct(fly)} of a single fly's differences "
                       f"and {pct(bat)} of a batch's (median over odors); starvation time alone "
                       f"predicts {pct(stv)}.")
        _finish(fig, "Does responding to some odors predict responding to others?", answer)
    return fig


def fig_time_of_day(paired: pd.DataFrame, *, data_root: Path):
    """Q6: before training, does the time of day change PER?"""
    from scripts.analysis.time_of_day_trend import attach_start_times, clock_hours

    p, _missing = attach_start_times(paired, data_root)
    if p.empty:
        return None
    cohort = _cohort(paired)
    per_fly = (p.groupby(_fly_key(p)).agg(start=("start_time", "first"),
                                          pre=("score_pre", "mean")).reset_index())
    per_fly["clock"] = pd.to_datetime(per_fly["start"]).map(clock_hours)
    res = spearmanr(per_fly["clock"], per_fly["pre"])
    rng = np.random.default_rng(0)
    with plt.rc_context(_RC_CONTEXT):
        fig, ax = plt.subplots(figsize=(9.0, 5.0))
        ax.scatter(per_fly["clock"] + rng.uniform(-0.05, 0.05, len(per_fly)), per_fly["pre"],
                   s=22, color=DOT, edgecolors="white", linewidths=0.8, zorder=2,
                   label="one fly (its average over the odors)")
        per_fly["bin"] = time_blocks(per_fly["clock"])
        for _block, g in per_fly.groupby("bin"):
            # Drawn at the flies' own mean start time, so the average sits among its dots.
            xm = float(g["clock"].mean())
            lo, hi = boot_ci(g["pre"].to_numpy(float))
            ax.plot([xm, xm], [lo, hi], color=PRE_COLOR, linewidth=2.2, zorder=3)
            ax.plot(xm, g["pre"].mean(), "o", color=PRE_COLOR, markersize=9,
                    markeredgecolor="white", markeredgewidth=2, zorder=4)
            ax.text(xm, -0.25, f"n={len(g)}", ha="center", va="top", fontsize=8,
                    color=INK_MUTED)
        ax.plot([], [], "o", color=PRE_COLOR, label="average of each block of\nexperiments, 95% CI")
        lo_h, hi_h = int(np.floor(per_fly["clock"].min())), int(np.ceil(per_fly["clock"].max()))
        ax.set_xticks(range(lo_h, hi_h + 1))
        ax.set_xticklabels([f"{h}:00" for h in range(lo_h, hi_h + 1)])
        ax.set_xlim(lo_h - 0.3, hi_h + 0.3)
        ax.set_ylim(-0.5, max(3.0, float(per_fly["pre"].max()) + 0.3))
        ax.set_xlabel(f"Time of day of the first {_pre_name(cohort, lower=True)} trial")
        ax.set_ylabel(f"{_pre_name(cohort)} PER score, all odors\n({SCORE_HINT})")
        ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False)
        _recessive_grid(ax)
        verdict = "No" if res.pvalue >= 0.05 else "Possibly"
        answer = (f"{verdict}: no trend through the day (Spearman rho={res.statistic:.2f}, "
                  f"{_fmt_p(res.pvalue)}). Caveat: each starvation group was run at its own time of day.")
        _finish(fig, f"Before {_block_name(cohort)}, does the time of day change how much flies "
                "respond?", answer)
    return fig


# ── driver ───────────────────────────────────────────────────────────────


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--comparison-dir", type=Path, required=True,
                    help="folder with paired_scores_per_fly.csv")
    ap.add_argument("--regression-dir", type=Path, default=None,
                    help="odor_holdout_regression output folder (adds the prediction panel)")
    ap.add_argument("--data-root", type=Path,
                    default=Path("/home/ramanlab/Documents/cole/Data/flys_New"),
                    help="raw batch folders, for the time-of-day figure")
    ap.add_argument("--trained-odor", default="Isoamyl Acetate")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default: <comparison-dir>/../findings")
    ap.add_argument("--labels-csv", type=Path, default=PREDICTIONS_CSV,
                    help="predictions table with the trial labels as recorded: the concentrations "
                         "in the odor names of a sensitivity figure (default: the live table; "
                         "'' draws the names without one)")
    args = ap.parse_args(argv)

    paired = pd.read_csv(args.comparison_dir / "paired_scores_per_fly.csv")
    out = args.out_dir or args.comparison_dir.parent / "findings"
    out.mkdir(parents=True, exist_ok=True)
    reg = regression_table(args.regression_dir) if args.regression_dir else None
    t = args.trained_odor
    # The folder's tables carry no concentration: the drawn flies' recorded labels do.
    c = odor_concentrations(args.labels_csv, flies=paired)
    figs = [("01_did_training_raise_PER.png", fig_direction(paired, trained=t, concentrations=c)),
            ("02_was_it_specific_to_trained_odor.png",
             fig_specificity(paired, trained=t, concentrations=c)),
            ("03_where_did_PER_change.png", fig_change_map(paired, trained=t, concentrations=c)),
            ("04_starvation_before_training.png", fig_baseline(paired)),
            ("05_general_responsiveness.png",
             fig_responsiveness(paired, regression=reg, concentrations=c)),
            ("06_time_of_day_before_training.png", fig_time_of_day(paired, data_root=args.data_root))]
    answers = []
    for name, fig in figs:
        if fig is None:
            continue
        with plt.rc_context(_RC_CONTEXT):
            fig.savefig(out / name, bbox_inches="tight")
        answers.append(f"{name}\n{fig.answer}\n")
        plt.close(fig)
        print(f"[OK] {out / name}")
    (out / "FINDINGS.txt").write_text("\n".join(answers))
    print(f"[OK] {out / 'FINDINGS.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
