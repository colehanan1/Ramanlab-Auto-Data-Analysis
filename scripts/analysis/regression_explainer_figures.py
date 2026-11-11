#!/usr/bin/env python3
"""Explainer figures for the hold-one-odor-out Ridge / Lasso results.

``odor_holdout_regression.py`` reports Q² and permutation p for hundreds of
models, which is hard to read. These figures explain it from the ground up,
one question per figure (title = the question, the line under it = the
answer, computed from the data):

    01_how_it_works.png          one real worked example: hide an odor, guess it, score the guess
    02_real_or_luck.png          the shuffle test behind every p-value
    03_how_predictable.png       prediction skill per odor, single fly vs batch average
    04_within_one_group.png      why it fails inside a single starvation group
    05_ridge_vs_lasso.png        what the two methods do differently, and whether they agree
    06_odors_go_together.png     which odors a fly responds to together
    07_starvation_time.png       does knowing the starvation time help?
    EXPLAINED.txt                glossary + every answer line

The worked example re-runs the regression's own code (same model, same
leave-one-batch-out splits), so its number matches the results tables.

Usage
-----
    python3 scripts/analysis/regression_explainer_figures.py \\
        --regression-dir <root>/odor_holdout_regression
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import warnings
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT, REPO_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import scripts.analysis.odor_holdout_regression as ohr  # noqa: E402
from scripts.analysis import phase_terms  # noqa: E402
from scripts.analysis.pretest_vs_test_comparison import (  # noqa: E402
    INK,
    INK_MUTED,
    POST_COLOR,
    PRE_COLOR,
    _RC_CONTEXT,
    _recessive_grid,
)
from scripts.analysis.score_summary import _score_cmap  # noqa: E402
from scripts.analysis.starvation_findings_figures import _finish, _fmt_p  # noqa: E402
from scripts.analysis.starvation_trend_figure import hours_from_label  # noqa: E402

DOT = "#a8a7a1"
SHADE = "#f1f0ec"     # "worse than guessing the average" region
#: Sequential (one hue, light -> dark) for correlations, all of which are >= 0:
#: blue, because every value in it is a pre-test score.
BLUES = LinearSegmentedColormap.from_list("pre_blues", ["#f4f7fb", "#86b6ef", "#1c5299"])
PHASE_STYLE = {"pre": dict(color=PRE_COLOR, marker="o", label="pre-test (before training)"),
               "post": dict(color=POST_COLOR, marker="s", label="post-training")}
CLIP = -100.0  # Q2 below -100 % is drawn at -100 % (the scale is otherwise unreadable)


def pct(q: float) -> str:
    return f"{round(100 * q) + 0:.0f}%"


def _phase_names(cohort: str | None) -> tuple[str, str]:
    """("pre-test", "post-training"), or pre-/post-exposure for a sensitivity cohort."""
    return (phase_terms.pre_label(cohort, "pre-test", case="lower"),
            phase_terms.post_label(cohort, "post-training", case="lower"))


def _example_phase(phase: str, cohort: str | None) -> str:
    """The worked example's phase as its text names it: "pre-test" / "post-test"."""
    label = phase_terms.pre_label if phase == "pre" else phase_terms.post_label
    return label(cohort, f"{phase}-test", case="lower")


# ── data ─────────────────────────────────────────────────────────────────


def load(reg_dir: Path):
    reg_dir = Path(reg_dir)
    m = pd.read_csv(reg_dir / "input_fly_matrix.csv")
    l1 = pd.read_csv(reg_dir / "level1_individual_within_starvation.csv")
    l2 = pd.read_csv(reg_dir / "level2_individual_pooled.csv")
    l3 = pd.read_csv(reg_dir / "level3_population_batches.csv")
    return m, l1, l2, l3


def regression_cohorts(reg_dir: Path) -> list[str]:
    """The datasets behind a regression folder. Its tables don't name them, so they
    come from the comparison it was fitted on: odor_holdout_regression writes to
    <comparison-dir>/../odor_holdout_regression, a sibling. [] when none is found."""
    parent = Path(reg_dir).resolve().parent
    return sorted({c for p in parent.glob("*/paired_scores_per_fly.csv")
                   for c in ohr.comparison_cohorts(p.parent)})


def _sel(d: pd.DataFrame, *, model="Ridge", variant="odors", scheme="leave-one-batch-out"):
    out = d[(d["model"] == model) & (d["variant"] == variant)]
    if scheme is not None and "scheme" in d:
        out = out[out["scheme"] == scheme]
    return out[out["target"].isin(["score", "mean score"])]


def pick_example(l2: pd.DataFrame) -> tuple[str, str]:
    """The best-predicted pre-test odor for a single fly (Ridge, new batch held out)."""
    s = _sel(l2)
    s = s[s["phase"] == "pre"].sort_values("q2", ascending=False)
    return str(s["odor"].iloc[0]), "pre"


@dataclasses.dataclass
class Example:
    odor: str
    phase: str
    table: pd.DataFrame        # per fly: actual, predicted, average_guess (+ ids)
    q2: float
    rmse_model: float
    rmse_average: float


def _xy(m: pd.DataFrame, odor: str, phase: str):
    X, y, feats = ohr.feature_target(m, odor, phase)
    return X, y, feats, ohr.group_splits(m["batch"])


def worked_example(m: pd.DataFrame, odor: str, phase: str) -> Example:
    """Ridge, leave-one-batch-out: exactly the regression's level-2 'odors' fit."""
    X, y, _f, splits = _xy(m, odor, phase)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pred, base = ohr.cv_predict(X, y, splits, make=lambda n, mi=None: ohr.make_model("ridge", n))
    t = m[["group", "fly", "fly_number", "batch"]].copy()
    t["actual"], t["predicted"], t["average_guess"] = y, pred, base
    return Example(odor, phase, t, ohr.q2(y, pred, base),
                   float(np.sqrt(np.mean((y - pred) ** 2))),
                   float(np.sqrt(np.mean((y - base) ** 2))))


def null_q2(m: pd.DataFrame, odor: str, phase: str, *, n_perm: int = 200, seed: int = 0) -> np.ndarray:
    """Q2 after shuffling the target across flies, n_perm times (the permutation test)."""
    X, y, _f, splits = _xy(m, odor, phase)
    rng = np.random.default_rng(seed)
    make = lambda n, mi=None: ohr.make_model("ridge", n)  # noqa: E731
    out = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(n_perm):
            yp = ohr._permute(y, None, rng)
            out.append(ohr.q2(yp, *ohr.cv_predict(X, yp, splits, make=make)))
    return np.array(out)


def coefficients(m: pd.DataFrame, odor: str, phase: str) -> pd.DataFrame:
    """Standardised weights each model gives the other odors, fitted on ALL flies (descriptive)."""
    X, y, feats, _s = _xy(m, odor, phase)
    out = {"feature": feats}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for kind, name in (("ridge", "Ridge"), ("lasso", "Lasso")):
            out[name] = ohr.make_model(kind, len(y)).fit(X, y)[-1].coef_
    return pd.DataFrame(out)


def odor_correlations(m: pd.DataFrame, *, phase: str = "pre") -> pd.DataFrame:
    cols = sorted((c for c in ohr.score_columns(m) if c.endswith(f"|{phase}")), key=str.casefold)
    r = m[cols].corr(method="spearman")
    r.index = r.columns = [c.split("|")[0] for c in cols]
    return r


def skill_table(l2: pd.DataFrame, l3: pd.DataFrame) -> pd.DataFrame:
    a = _sel(l2).assign(level="single fly")
    b = _sel(l3).assign(level="batch average")
    return pd.concat([a, b])[["level", "odor", "phase", "n", "q2", "p_perm"]].reset_index(drop=True)


# ── figures ──────────────────────────────────────────────────────────────


def _hollow_or_filled(ax, x, y, phase, significant, *, size=8, z=3):
    st = PHASE_STYLE[phase]
    ax.plot(x, y, linestyle="", marker=st["marker"], markersize=size, zorder=z,
            markerfacecolor=st["color"] if significant else "white",
            markeredgecolor=st["color"], markeredgewidth=1.8)


def _worse_zone(ax, lo):
    ax.axvspan(lo, 0, color=SHADE, zorder=0)
    ax.axvline(0, color=INK_MUTED, linewidth=1.0, zorder=1)


def fig_how_it_works(m: pd.DataFrame, ex: Example, *, cohorts: Sequence[str] = ()):
    """Q1: one real target, start to finish."""
    phase = _example_phase(ex.phase, ohr.wording_cohort(cohorts))
    cols = ohr.score_columns(m)
    order = np.argsort(ex.table["actual"].to_numpy(), kind="stable")
    pick = order[np.linspace(0, len(order) - 1, min(8, len(order))).round().astype(int)]
    sub = m.iloc[pick]
    vals = sub[cols].to_numpy(float).copy()
    target_col = cols.index(f"{ex.odor}|{ex.phase}")
    other_col = cols.index(f"{ex.odor}|{'post' if ex.phase == 'pre' else 'pre'}")
    vals[:, [target_col, other_col]] = np.nan
    cmap, norm = _score_cmap()
    cmap = cmap.copy()
    cmap.set_bad("white")
    with plt.rc_context(_RC_CONTEXT):
        fig = plt.figure(figsize=(13.0, 10.0))
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], width_ratios=[1.35, 1])
        a = fig.add_subplot(gs[0, :])
        a.imshow(vals, cmap=cmap, norm=norm, aspect="auto")
        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                if j == target_col:
                    a.text(j, i, "?", ha="center", va="center", fontsize=13, fontweight="bold",
                           color=INK)
                elif j == other_col:
                    a.text(j, i, "hidden", ha="center", va="center", fontsize=7, color=INK_MUTED)
                else:
                    v = vals[i, j]
                    a.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=9,
                           color="white" if v >= 3 or v <= -1 else INK)
        a.add_patch(Rectangle((target_col - 0.5, -0.5), 1, vals.shape[0], fill=False,
                              edgecolor=INK, linewidth=2.5, zorder=5))
        a.set_xticks(range(len(cols)))
        a.set_xticklabels([c.replace("|", " ") for c in cols], fontsize=8, rotation=35,
                          ha="left", rotation_mode="anchor")
        a.xaxis.tick_top()
        a.set_yticks(range(len(sub)))
        a.set_yticklabels([f"fly {i + 1}  ({g})" for i, g in enumerate(sub["group"])], fontsize=8.5)
        a.tick_params(length=0)
        for sp in a.spines.values():
            sp.set_visible(False)
        a.set_title(f"Step 1 — hide one odor. Each row is a fly (8 of {len(m)} shown); each number is a PER score.\n"
                    f"The model sees the 12 coloured columns and must guess the '?' ({ex.odor}, {phase}). "
                    f"Both {ex.odor} columns are hidden.",
                    loc="left", fontsize=9.5, color=INK, pad=62)

        b = fig.add_subplot(gs[1, 0])
        t = ex.table
        rng = np.random.default_rng(0)
        b.scatter(t["predicted"], t["actual"] + rng.uniform(-0.15, 0.15, len(t)), s=20, color=DOT,
                  edgecolors="white", linewidths=0.8, zorder=2, label="one fly")
        lo = float(min(t["predicted"].min(), t["actual"].min())) - 0.3
        hi = float(max(t["predicted"].max(), t["actual"].max())) + 0.3
        b.plot([lo, hi], [lo, hi], color=INK, linewidth=1.5, zorder=3, label="a perfect guess")
        b.axhline(t["average_guess"].mean(), color=INK_MUTED, linewidth=1.5, linestyle="-",
                  zorder=1, label="'guess the average' for every fly")
        b.set_xlim(lo, hi)
        b.set_ylim(lo, hi)
        b.set_xlabel(f"Guessed {ex.odor} score (from the fly's other odors)")
        b.set_ylabel(f"Real {ex.odor} score")
        b.set_title("Step 2 — guess every fly. The guess for each fly comes from a\n"
                    "model trained on the OTHER batches only (it never saw this fly).",
                    loc="left", fontsize=9.5, color=INK)
        b.legend(loc="lower right", frameon=False, fontsize=8)  # the corner the data leave empty
        _recessive_grid(b)

        c = fig.add_subplot(gs[1, 1])
        labels = ["Guess the average\nfor every fly", "Guess from the\nfly's other odors"]
        errs = [ex.rmse_average, ex.rmse_model]
        c.barh([1, 0], errs, 0.5, color=[DOT, INK], zorder=2)
        for yv, e in zip([1, 0], errs):
            c.text(e + 0.02, yv, f"{e:.2f}", va="center", ha="left", fontsize=9.5, color=INK)
        c.set_yticks([1, 0])
        c.set_yticklabels(labels)
        c.set_xlim(0, max(errs) * 1.35)
        c.set_xlabel("Typical miss (score points, root-mean-square)")
        c.set_title(f"Step 3 — score the guesses. Prediction skill (Q²)\n"
                    f"= 1 − ({ex.rmse_model:.2f} / {ex.rmse_average:.2f})² = {pct(ex.q2)}.  "
                    "0% = no better than the\naverage, 100% = perfect, below 0 = worse than the average.",
                    loc="left", fontsize=9.5, color=INK)
        for sp in ("top", "right"):
            c.spines[sp].set_visible(False)
        c.tick_params(axis="y", length=0)
        answer = (f"Hide one odor, guess it from the fly's other odors, compare with just guessing the average. "
                  f"For {ex.odor} ({phase}) the guess removes {pct(ex.q2)} of the squared error — "
                  "that percentage is the 'Q²' (prediction skill) in every other figure. "
                  "The same thing is repeated for every odor, pre and post.")
        _finish(fig, "How does the hold-one-odor-out regression work?", answer)
    return fig


def fig_is_it_luck(ex: Example, null: np.ndarray):
    """Q2: the permutation test, drawn."""
    null = null[np.isfinite(null)]
    p = (1 + int(np.sum(null >= ex.q2))) / (1 + null.size)
    with plt.rc_context(_RC_CONTEXT):
        fig, ax = plt.subplots(figsize=(9.0, 4.8))
        ax.hist(100 * null, bins=30, color=DOT, edgecolor="white", linewidth=1, zorder=2,
                label=f"{null.size} shuffled datasets")
        ax.axvline(100 * ex.q2, color=INK, linewidth=2.5, zorder=3)
        ax.text(100 * ex.q2, ax.get_ylim()[1] * 0.95, f"  real data: {pct(ex.q2)}", ha="left",
                va="top", fontsize=10, color=INK, fontweight="bold")
        ax.set_xlabel("Prediction skill (Q², %)")
        ax.set_ylabel("Number of shuffles")
        ax.legend(loc="upper left", frameon=False)
        _recessive_grid(ax)
        beat = int(np.sum(null < ex.q2))
        answer = (f"Shuffle which fly had which {ex.odor} score (destroying any real link) and re-run everything. "
                  f"Shuffled data score {pct(float(np.median(null)))} on average; the real data beat "
                  f"{beat} of {null.size} shuffles, so p = {p:.3f}"
                  f"{' (the smallest p this many shuffles can give)' if beat == null.size else ''}. "
                  "p = the chance a skill this high happens with no real link.")
        _finish(fig, "Is the prediction real, or could it be luck?", answer)
    return fig


def fig_skill_by_odor(skill: pd.DataFrame, *, cohorts: Sequence[str] = ()):
    """Q3: how predictable is each odor, for a single fly and for a batch average?"""
    cohort = ohr.wording_cohort(cohorts)
    odors = sorted(skill["odor"].unique(), key=str.casefold)[::-1]
    lo = max(CLIP, float(np.floor(100 * skill["q2"].min() / 10) * 10) - 5)
    hi = max(40.0, float(np.ceil(100 * skill["q2"].max() / 10) * 10) + 5)
    with plt.rc_context(_RC_CONTEXT):
        fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2), sharey=True)
        for ax, level in zip(axes, ("single fly", "batch average")):
            s = skill[skill["level"] == level]
            _worse_zone(ax, lo)
            for i, odor in enumerate(odors):
                for ph, dy in (("pre", 0.15), ("post", -0.15)):
                    r = s[(s["odor"] == odor) & (s["phase"] == ph)]
                    if len(r):
                        q = float(np.clip(100 * r["q2"].iloc[0], lo, None))
                        _hollow_or_filled(ax, q, i + dy, ph, r["p_perm"].iloc[0] < 0.05)
            n = int(s["n"].max()) if len(s) else 0
            unit = "flies" if level == "single fly" else "batches"
            ax.set_title(f"{'A single fly' if level == 'single fly' else 'A batch average'}"
                         f"  (n = {n} {unit})", loc="left", fontsize=10.5, color=INK)
            ax.set_xlim(lo, hi)
            ax.set_xlabel("Prediction skill (Q², %)")
            ax.text(lo + 1, -0.45, "worse than\nguessing the average", fontsize=7.5,
                    color=INK_MUTED, va="bottom")
            _recessive_grid(ax)
        axes[0].set_yticks(range(len(odors)))
        axes[0].set_yticklabels(odors)
        handles = [plt.Line2D([], [], linestyle="", marker="o", color=PRE_COLOR, markersize=8,
                              label=phase_terms.pre_label(cohort, PHASE_STYLE["pre"]["label"],
                                                          case="lower")),
                   plt.Line2D([], [], linestyle="", marker="s", color=POST_COLOR, markersize=8,
                              label=phase_terms.post_label(cohort, PHASE_STYLE["post"]["label"],
                                                           case="lower")),
                   plt.Line2D([], [], linestyle="", marker="o", color=INK, markersize=8,
                              label="filled = p<0.05 (not luck)"),
                   plt.Line2D([], [], linestyle="", marker="o", markerfacecolor="white",
                              markeredgecolor=INK, markersize=8, label="hollow = could be luck")]
        fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
                   bbox_to_anchor=(0.5, -0.01))
        fly = skill[skill["level"] == "single fly"]
        bat = skill[skill["level"] == "batch average"]
        answer = (f"Single fly: about {pct(fly['q2'].median())} (median), and "
                  f"{int((fly['p_perm'] < 0.05).sum())} of {len(fly)} odor/phase targets beat luck. "
                  f"Batch average: {pct(bat['q2'].median())} (median), "
                  f"{int((bat['p_perm'] < 0.05).sum())} of {len(bat)} beat luck. "
                  "Modest but real: a fly's other odors say something about this one. Ridge model, each "
                  "batch predicted by a model that never saw it.")
        _finish(fig, "How well can each odor be predicted from the others?", answer, bottom=0.07)
    return fig


def fig_within_group(l1: pd.DataFrame, *, cohorts: Sequence[str] = ()):
    """Q4: inside one starvation group the model is starved of data."""
    pre, post = _phase_names(ohr.wording_cohort(cohorts))
    s = _sel(l1, scheme=None)
    groups = sorted(s["group"].unique(), key=hours_from_label)
    rng = np.random.default_rng(0)
    with plt.rc_context(_RC_CONTEXT):
        fig, ax = plt.subplots(figsize=(9.6, 5.0))
        ax.axhspan(CLIP - 10, 0, color=SHADE, zorder=0)
        ax.axhline(0, color=INK_MUTED, linewidth=1.0, zorder=1)
        meds = []
        for i, g in enumerate(groups):
            r = s[s["group"] == g]
            for ph in ("pre", "post"):
                q = np.clip(100 * r.loc[r["phase"] == ph, "q2"].to_numpy(float), CLIP, None)
                q = q[np.isfinite(q)]
                ax.plot(i + rng.uniform(-0.2, 0.2, q.size), q, linestyle="",
                        marker=PHASE_STYLE[ph]["marker"], color=PHASE_STYLE[ph]["color"],
                        markersize=6, markeredgecolor="white", markeredgewidth=0.8, zorder=2)
            med = float(np.nanmedian(r["q2"])) * 100
            meds.append(med)
            ax.plot([i - 0.3, i + 0.3], [max(med, CLIP)] * 2, color=INK, linewidth=2.5, zorder=3)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([f"{g}\nn={int(s.loc[s['group'] == g, 'n'].max())} flies" for g in groups])
        ax.set_ylim(CLIP - 5, max(60.0, float(np.nanmax(100 * s["q2"])) + 10))
        ax.set_ylabel("Prediction skill (Q², %)")
        ax.set_xlabel("Model trained and tested inside ONE starvation group")
        ax.text(len(groups) - 0.5, CLIP + 3, "worse than guessing the group's average "
                f"(values below {CLIP:.0f}% drawn at {CLIP:.0f}%)", ha="right", va="bottom",
                fontsize=7.5, color=INK_MUTED)
        handles = [plt.Line2D([], [], linestyle="", marker="o", color=PRE_COLOR, label=f"{pre} odor"),
                   plt.Line2D([], [], linestyle="", marker="s", color=POST_COLOR, label=f"{post} odor"),
                   plt.Line2D([], [], color=INK, linewidth=2.5, label="median")]
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False)
        _recessive_grid(ax)
        n_lo, n_hi = int(s.groupby("group")["n"].max().min()), int(s.groupby("group")["n"].max().max())
        answer = (f"Mostly not: the median skill is {min(meds):.0f}% to {max(meds):.0f}% across groups "
                  f"({n_lo}–{n_hi} flies each). Inside one group the model rarely beats that group's own "
                  "average; pooling every group (fig 03) gives it more flies and more spread to learn from.")
        _finish(fig, "Does it work inside a single starvation group?", answer)
    return fig


def fig_ridge_vs_lasso(coefs: pd.DataFrame, l1, l2, l3, ex: Example, *, cohorts: Sequence[str] = ()):
    """Q5: the two methods, side by side."""
    phase = _example_phase(ex.phase, ohr.wording_cohort(cohorts))
    c = coefs.sort_values("Ridge")
    labels = [f.replace("|", " ") for f in c["feature"]]
    key = ["level", "group", "odor", "phase", "target", "scheme"]
    ridge = pd.concat([_sel(l1, model="Ridge", scheme=None), _sel(l2, model="Ridge"),
                       _sel(l3, model="Ridge")])
    lasso = pd.concat([_sel(l1, model="Lasso", scheme=None), _sel(l2, model="Lasso"),
                       _sel(l3, model="Lasso")])
    both = ridge.merge(lasso, on=key, suffixes=("_r", "_l")).dropna(subset=["q2_r", "q2_l"])
    rho = both[["q2_r", "q2_l"]].corr(method="spearman").iloc[0, 1] if len(both) > 2 else np.nan
    with plt.rc_context(_RC_CONTEXT):
        fig = plt.figure(figsize=(14.0, 5.8))
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.25])
        a = fig.add_subplot(gs[0, 0])
        b = fig.add_subplot(gs[0, 1], sharey=a)
        y = np.arange(len(c))
        lim = float(np.abs(c[["Ridge", "Lasso"]].to_numpy()).max()) * 1.25 or 1.0
        for ax, name in ((a, "Ridge"), (b, "Lasso")):
            ax.barh(y, c[name], 0.62, color=INK, zorder=2)
            ax.axvline(0, color=INK_MUTED, linewidth=1.0)
            if name == "Lasso":
                for yi, v in zip(y, c[name]):
                    if v == 0:
                        ax.text(0.01 * lim, yi, " 0 (dropped)", va="center", ha="left", fontsize=7.5,
                                color=INK_MUTED)
            ax.set_xlim(-lim, lim)
            ax.set_xlabel("Weight (standardised)")
            ax.set_title(f"{name}: weight on each other odor\n(predicting {ex.odor} {phase}, all flies)",
                         loc="left", fontsize=10, color=INK)
            _recessive_grid(ax)
        a.set_yticks(y)
        a.set_yticklabels(labels, fontsize=8.5)
        plt.setp(b.get_yticklabels(), visible=False)

        s = fig.add_subplot(gs[0, 2])
        xr = np.clip(100 * both["q2_r"], CLIP, None)
        xl = np.clip(100 * both["q2_l"], CLIP, None)
        s.plot([CLIP, 100], [CLIP, 100], color=INK_MUTED, linewidth=1.0, zorder=1)
        s.scatter(xr, xl, s=18, color=DOT, edgecolors="white", linewidths=0.6, zorder=2)
        top = float(max(xr.max(), xl.max())) + 10 if len(both) else 50
        s.set_xlim(CLIP - 5, top)
        s.set_ylim(CLIP - 5, top)
        s.set_xlabel("Ridge prediction skill (%)")
        s.set_ylabel("Lasso prediction skill (%)")
        s.set_title(f"Every target, every level ({len(both)} pairs)", loc="left", fontsize=10, color=INK)
        s.text(0.97, 0.03, "on the line = they agree", transform=s.transAxes, ha="right",
               fontsize=8, color=INK_MUTED)
        _recessive_grid(s)
        dropped = int((c["Lasso"] == 0).sum())
        answer = (f"Both guess the hidden odor from the others; they differ in how. Ridge gives every other odor a "
                  f"little weight. Lasso keeps only the most useful and sets the rest to exactly 0 "
                  f"(here it dropped {dropped} of {len(c)}). "
                  f"Their scores broadly agree (rank correlation {rho:.2f}); Ridge scores higher in "
                  f"{100 * float((both['q2_r'] > both['q2_l']).mean()):.0f}% of the {len(both)} comparisons. "
                  "So the information is spread across many odors, not carried by one.")
        _finish(fig, "What is the difference between Ridge and Lasso, and do they agree?", answer)
    return fig


def fig_odors_together(corr: pd.DataFrame, *, cohorts: Sequence[str] = ()):
    """Q6: which odors does a fly respond to together (pre-test)?"""
    pre = phase_terms.pre_label(ohr.wording_cohort(cohorts), "Pre-test")
    v = corr.to_numpy(float).copy()
    np.fill_diagonal(v, np.nan)
    vmax = max(0.5, float(np.nanmax(v)))
    cmap = BLUES.copy()
    cmap.set_bad("white")
    with plt.rc_context(_RC_CONTEXT):
        fig = plt.figure(figsize=(9.0, 7.2))
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.05], wspace=0.05)
        ax, cax = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
        im = ax.imshow(v, cmap=cmap, vmin=0, vmax=vmax)
        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                if np.isfinite(v[i, j]):
                    ax.text(j, i, f"{v[i, j]:.2f}", ha="center", va="center", fontsize=9,
                            color="white" if v[i, j] > 0.6 * vmax else INK)
        names = list(corr.columns)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=35, ha="right")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xticks(np.arange(-0.5, len(names), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(names), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=2)
        ax.tick_params(which="minor", length=0)
        cb = fig.colorbar(im, cax=cax)
        cb.set_label("Correlation across flies (Spearman)\n0 = unrelated · 1 = always together")
        cb.outline.set_visible(False)
        fig.subplots_adjust(left=0.2, right=0.86, bottom=0.18)
        iu = np.triu_indices_from(v, 1)
        flat = v[iu]
        k = int(np.nanargmax(flat))
        a_, b_ = names[iu[0][k]], names[iu[1][k]]
        sign = "positive" if np.nanmin(flat) > 0 else "mostly positive"
        answer = (f"Every pair is {sign} ({np.nanmin(flat):.2f} to {np.nanmax(flat):.2f}): a fly that responds "
                  f"to one odor tends to respond to the others. Strongest pair: {a_} and {b_}. "
                  f"This shared 'responsiveness' is what the regression picks up. {pre} scores, every fly.")
        _finish(fig, "Which odors does a fly respond to together?", answer, tight=False)
    return fig


def fig_starvation(l2: pd.DataFrame, l3: pd.DataFrame, *, cohorts: Sequence[str] = ()):
    """Q7: does knowing the starvation time help the prediction?"""
    pre, post = _phase_names(ohr.wording_cohort(cohorts))
    variants = [("starvation only", "starvation time\nonly"), ("odors", "the other odors"),
                ("odors + starvation", "other odors +\nstarvation time")]
    rng = np.random.default_rng(0)
    meds = {}
    with plt.rc_context(_RC_CONTEXT):
        fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0), sharey=True)
        for ax, (d, level) in zip(axes, ((l2, "A single fly"), (l3, "A batch average"))):
            ax.axhspan(CLIP - 10, 0, color=SHADE, zorder=0)
            ax.axhline(0, color=INK_MUTED, linewidth=1.0, zorder=1)
            for i, (var, _lab) in enumerate(variants):
                s = _sel(d, variant=var)
                for ph in ("pre", "post"):
                    q = np.clip(100 * s.loc[s["phase"] == ph, "q2"].to_numpy(float), CLIP, None)
                    ax.plot(i + rng.uniform(-0.18, 0.18, q.size), q, linestyle="",
                            marker=PHASE_STYLE[ph]["marker"], color=PHASE_STYLE[ph]["color"],
                            markersize=6, markeredgecolor="white", markeredgewidth=0.8, zorder=2)
                med = 100 * float(s["q2"].median())
                meds[(level, var)] = med
                ax.plot([i - 0.3, i + 0.3], [med, med], color=INK, linewidth=2.5, zorder=3)
                ax.text(i + 0.32, med, f"{round(med) + 0:.0f}%", va="center", fontsize=8.5, color=INK)
            ax.set_xticks(range(len(variants)))
            ax.set_xticklabels([lab for _v, lab in variants])
            ax.set_xlabel("What the model is allowed to use")
            ax.set_title(level, loc="left", fontsize=10.5, color=INK)
            _recessive_grid(ax)
        lo = min(-30.0, float(np.nanmin([100 * _sel(d, variant=v)["q2"].min()
                                          for d in (l2, l3) for v, _l in variants])) - 5)
        axes[0].set_ylim(max(CLIP - 5, lo), max(60.0, axes[0].get_ylim()[1]))
        axes[0].set_ylabel("Prediction skill (Q², %)")
        handles = [plt.Line2D([], [], linestyle="", marker="o", color=PRE_COLOR, label=f"{pre} odor"),
                   plt.Line2D([], [], linestyle="", marker="s", color=POST_COLOR, label=f"{post} odor"),
                   plt.Line2D([], [], color=INK, linewidth=2.5, label="median")]
        axes[1].legend(handles=handles, loc="upper right", frameon=False)
        f = "A single fly"
        p0 = lambda v: f"{round(v) + 0:.0f}%"  # noqa: E731  (+0 turns -0 into 0)
        answer = (f"No. Starvation time alone predicts {p0(meds[(f, 'starvation only')])} (median, single fly): "
                  f"nothing. Adding it to the odors changes {p0(meds[(f, 'odors')])} to "
                  f"{p0(meds[(f, 'odors + starvation')])}. How a fly responds depends on the fly, not on "
                  "how long it was starved.")
        _finish(fig, "Does knowing the starvation time help predict a fly's PER?", answer)
    return fig


GLOSSARY = """\
How to read the regression (plain language)
===========================================
The question: if you hide a fly's response to one odor, can you guess it from how
the same fly responded to the other six odors (pre-test and post-training)?

Ridge, Lasso   Two ways of drawing a "best-fit" rule from the other odors to the hidden
               one. Both shrink the rule to avoid over-fitting a small dataset.
               Ridge spreads weight over every odor; Lasso keeps only a few and sets the
               rest to exactly zero. If they agree, the answer doesn't depend on the method.
Q² (skill)     How much better the guess is than guessing the average for everyone.
               0% = no better, 100% = perfect, below 0 = worse than the average.
               Always measured on flies/batches the model never saw.
Leave-one-batch-out
               Each batch (one rig's flies from one session) is guessed by a model fitted
               on all the OTHER batches, so the model can't memorise that day or rig.
p (permutation)  Shuffle which fly had which score, redo everything many times; p is the
               fraction of shuffles that scored as well as the real data. p<0.05 = unlikely
               to be luck. Uncorrected: with hundreds of models, ~5% pass by chance.
Levels         single fly (each fly guessed), batch average (each batch's mean guessed),
               within one starvation group (only that group's flies used).
"""


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--regression-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=None, help="default: <regression-dir>/explained")
    ap.add_argument("--example", default=None,
                    help='"ODOR|pre" or "ODOR|post" for the worked example (default: best pre-test odor)')
    ap.add_argument("--n-perm", type=int, default=200, help="shuffles for the luck figure")
    args = ap.parse_args(argv)

    m, l1, l2, l3 = load(args.regression_dir)
    out = args.out_dir or args.regression_dir / "explained"
    out.mkdir(parents=True, exist_ok=True)
    odor, phase = args.example.split("|") if args.example else pick_example(l2)
    ex = worked_example(m, odor, phase)
    print(f"worked example: {odor} {phase}-test, Q2={ex.q2:.3f}")
    cohorts = regression_cohorts(args.regression_dir)
    figs = [("01_how_it_works.png", fig_how_it_works(m, ex, cohorts=cohorts)),
            ("02_real_or_luck.png", fig_is_it_luck(ex, null_q2(m, odor, phase, n_perm=args.n_perm))),
            ("03_how_predictable.png", fig_skill_by_odor(skill_table(l2, l3), cohorts=cohorts)),
            ("04_within_one_group.png", fig_within_group(l1, cohorts=cohorts)),
            ("05_ridge_vs_lasso.png", fig_ridge_vs_lasso(coefficients(m, odor, phase), l1, l2, l3, ex,
                                                         cohorts=cohorts)),
            ("06_odors_go_together.png", fig_odors_together(odor_correlations(m), cohorts=cohorts)),
            ("07_starvation_time.png", fig_starvation(l2, l3, cohorts=cohorts))]
    answers = [GLOSSARY]
    for name, fig in figs:
        with plt.rc_context(_RC_CONTEXT):
            fig.savefig(out / name, bbox_inches="tight")
        answers.append(f"{name}\n{fig.answer}\n")
        plt.close(fig)
        print(f"[OK] {out / name}")
    (out / "EXPLAINED.txt").write_text("\n".join(answers))
    print(f"[OK] {out / 'EXPLAINED.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
