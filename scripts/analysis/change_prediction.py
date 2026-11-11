#!/usr/bin/env python3
"""Is sensitization predictable and consistent -- within a fly, and across a dataset?

"Change" is a fly's PER score after training minus before (post-training minus
pre-test), per odor. One figure per question; the title is the question and the
line under it the answer, computed from the data:

Across the dataset
    01_dataset_average_vs_individual.png    does the dataset's average change for an
                                            odor predict what each fly does?
    02_batch_to_batch.png                   is the change pattern the same from batch
                                            to batch?
Within a fly
    03_changes_go_together.png              do a fly's changes to different odors go
                                            together?
    04_predict_change.png                   can a fly's change to one odor be
                                            predicted from its changes to the other
                                            odors, or from its pre-test?
    05_trained_odor_general_vs_specific.png how much of the trained-odor change is the
                                            fly's general change?

Models (04, 05) are Ridge and Lasso, built by ``odor_holdout_regression`` (same
scaling, same alpha search inside each training fold). Every rig folder is
predicted by a model trained on the rest of the dataset only. Skill is
Q2 = 1 - SSE(model) / SSE(the training flies' average change): how much better
than just using the dataset's average change. p comes from shuffling the target
across flies and re-running everything (default 200 times). A subset of flies
(``--subset-hours 13 14`` = first pre-test started 13:00-14:00) is scored on its
own and highlighted in every figure; its predictions come from the same
leave-one-folder-out models. All p-values uncorrected.

Usage
-----
    python3 scripts/analysis/change_prediction.py --dataset IAA-Sensitivity-1-1 \\
        --subset-hours 13 14 --subset-label "1–2 pm"
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import friedmanchisquare, kruskal, pearsonr  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT, REPO_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import scripts.analysis.odor_holdout_regression as ohr  # noqa: E402
from scripts.analysis import phase_terms  # noqa: E402
from scripts.analysis.day_vs_day_report import (  # noqa: E402
    folder_batch,
    folder_date,
    starvation_hours,
    trained_odor,
)
from scripts.analysis.pretest_vs_test_comparison import (  # noqa: E402
    INK,
    INK_MUTED,
    PRE_COLOR,
    _RC_CONTEXT,
    _recessive_grid,
)
from scripts.analysis.starvation_findings_figures import (  # noqa: E402
    DIVERGING,
    DOWN,
    EMPH,
    SAME,
    UP,
    _finish,
    _fmt_p,
    boot_ci,
)

CSV = Path("/home/ramanlab/Documents/cole/Data/CSVs-New-Opto-Flys")
DATA_ROOT = Path("/home/ramanlab/Documents/cole/Data/flys_New")
OUT_ROOT = Path("/home/ramanlab/Documents/cole/Results/Figures")
#: What each predictor set uses to guess a fly's change for one odor.
PREDICTOR_SETS = {
    "delta_others": "its changes to the other odors",
    "pre_others": "its pre-test to the other odors",
    "pre_own": "its own starting score for this odor",
    "pre_all": "its whole pre-test (all odors)",
}
CLIP = -100.0
SHADE = "#f1f0ec"
LIGHT_BLUE = "#86b6ef"


# ── data ─────────────────────────────────────────────────────────────────


def _odors(m: pd.DataFrame) -> list[str]:
    return sorted({c[:-4] for c in m.columns if isinstance(c, str) and c.endswith("|pre")},
                  key=str.casefold)


def order_odors(odors: Sequence[str], trained: str) -> list[str]:
    """The trained odor first, the rest alphabetical."""
    rest = sorted((o for o in odors if o != trained), key=str.casefold)
    return ([trained] if trained in odors else []) + rest


def change_matrix(paired: pd.DataFrame) -> pd.DataFrame:
    """One row per complete fly: pre, post and change (post - pre) per odor, plus its
    rig folder (``batch``) and session (the batch across rigs)."""
    p = paired.copy()
    if "group" not in p:
        p["group"] = "1 h"
    m = ohr.fly_matrix(p)
    for o in _odors(m):
        m[f"{o}|delta"] = m[f"{o}|post"] - m[f"{o}|pre"]
    m["session"] = m["fly"].astype(str).str.replace(r"_rig_\d+$", "", regex=True)
    return m.reset_index(drop=True)


def predictors(m: pd.DataFrame, odor: str, which: str) -> np.ndarray:
    odors = _odors(m)
    cols = {"pre_all": [f"{o}|pre" for o in odors],
            "pre_others": [f"{o}|pre" for o in odors if o != odor],
            "pre_own": [f"{odor}|pre"],
            "delta_others": [f"{o}|delta" for o in odors if o != odor]}[which]
    return m[cols].to_numpy(dtype=float)


# ── models ───────────────────────────────────────────────────────────────


def subset_q2(y, pred, base, mask) -> float:
    """Q2 over the flies in ``mask`` only (their predictions, their fold baselines)."""
    mask = np.asarray(mask, dtype=bool)
    if mask.sum() < 2:
        return float("nan")
    return ohr.q2(np.asarray(y)[mask], np.asarray(pred)[mask], np.asarray(base)[mask])


def evaluate(X, y, splits, subset, *, kind: str, n_perm: int, seed: int) -> dict:
    """Out-of-folder Q2 for all flies and for the subset, each with a permutation p."""
    def make(n, minority=None):
        return ohr.make_model(kind, n)

    splits = list(splits)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pred, base = ohr.cv_predict(X, y, splits, make=make)
        q_all, q_sub = ohr.q2(y, pred, base), subset_q2(y, pred, base, subset)
        rng = np.random.default_rng(seed)
        null_all, null_sub = [], []
        for _ in range(n_perm):
            yp = rng.permutation(y)
            pp, bb = ohr.cv_predict(X, yp, splits, make=make)
            null_all.append(ohr.q2(yp, pp, bb))
            null_sub.append(subset_q2(yp, pp, bb, subset))
    return {"n_all": len(y), "n_subset": int(np.asarray(subset, bool).sum()),
            "q2_all": q_all, "p_all": ohr.permutation_p(q_all, np.array(null_all)) if n_perm else np.nan,
            "q2_subset": q_sub,
            "p_subset": ohr.permutation_p(q_sub, np.array(null_sub)) if n_perm else np.nan,
            "pred": pred, "base": base}


def run(m: pd.DataFrame, subset, *, n_perm: int = 200, n_jobs: int = -1,
        models: Sequence[str] = ("ridge", "lasso")) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every odor x predictor set x model; returns (skill table, per-fly predictions)."""
    subset = np.asarray(subset, dtype=bool)
    splits = ohr.group_splits(m["batch"])
    tasks = [(o, w, k) for o in _odors(m) for w in PREDICTOR_SETS for k in models]

    def one(i, task):
        o, w, k = task
        y = m[f"{o}|delta"].to_numpy(dtype=float)
        return task, evaluate(predictors(m, o, w), y, splits, subset, kind=k, n_perm=n_perm,
                              seed=i)

    if n_jobs == 1:
        out = [one(i, t) for i, t in enumerate(tasks)]
    else:
        from joblib import Parallel, delayed
        out = Parallel(n_jobs=n_jobs)(delayed(one)(i, t) for i, t in enumerate(tasks))
    rows, preds = [], []
    for (o, w, k), r in out:
        rows.append({"odor": o, "predictors": w, "uses": PREDICTOR_SETS[w],
                     "model": ohr.MODEL_NAME[k],
                     **{key: r[key] for key in ("n_all", "q2_all", "p_all", "n_subset",
                                                 "q2_subset", "p_subset")}})
        preds.append(pd.DataFrame({
            "fly": m["fly"], "fly_number": m["fly_number"], "batch": m["batch"], "odor": o,
            "predictors": w, "model": ohr.MODEL_NAME[k], "actual": m[f"{o}|delta"],
            "predicted": r["pred"], "dataset_average": r["base"], "in_subset": subset}))
    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True)


# ── dataset-level consistency ────────────────────────────────────────────


def odor_average(m: pd.DataFrame, subset=None) -> dict:
    """How well the dataset's per-odor average change predicts each fly's change.

    Each rig folder is predicted from the OTHER folders' per-odor means; the
    baseline is their overall mean change (all odors). Pooled over every fly x
    odor. Friedman: do odors differ consistently in how much flies change?
    """
    odors = _odors(m)
    D = m[[f"{o}|delta" for o in odors]].to_numpy(dtype=float)
    pred, base = np.full_like(D, np.nan), np.full_like(D, np.nan)
    for train, test in ohr.group_splits(m["batch"]):
        pred[test] = np.nanmean(D[train], axis=0)
        base[test] = np.nanmean(D[train])
    out = {"q2_all": ohr.q2(D.ravel(), pred.ravel(), base.ravel()), "q2_subset": float("nan")}
    if subset is not None and np.asarray(subset, bool).sum() >= 2:
        s = np.asarray(subset, bool)
        out["q2_subset"] = ohr.q2(D[s].ravel(), pred[s].ravel(), base[s].ravel())
    out["p_friedman"] = (float(friedmanchisquare(*[D[:, j] for j in range(len(odors))]).pvalue)
                         if len(odors) >= 3 and len(D) >= 2 else float("nan"))
    rows = []
    for j, o in enumerate(odors):
        d = D[:, j]
        rows.append({"odor": o, "n_flies": int(np.isfinite(d).sum()), "mean_change": float(np.nanmean(d)),
                     "pct_up": 100 * float(np.mean(d > 0)), "pct_same": 100 * float(np.mean(d == 0)),
                     "pct_down": 100 * float(np.mean(d < 0))})
    out["per_odor"] = pd.DataFrame(rows)
    return out


def _session_key(s: str):
    try:
        return folder_date(s), folder_batch(s)
    except ValueError:
        return (pd.Timestamp.max.date(), 0)


def batch_consistency(m: pd.DataFrame, *, trained: str = "Isoamyl Acetate", min_flies: int = 3):
    """Per session (a batch across its rigs): mean change per odor, and how well that
    profile matches the rest of the dataset (Pearson r over odors)."""
    odors = order_odors(_odors(m), trained)
    cols = [f"{o}|delta" for o in odors]
    n = m.groupby("session").size()
    prof = m.groupby("session")[cols].mean()
    prof.columns = odors
    prof = prof.loc[sorted((s for s in prof.index if n[s] >= min_flies), key=_session_key)]
    rows = []
    for s in prof.index:
        rest = m.loc[m["session"] != s, cols].mean().to_numpy(dtype=float)
        mine = prof.loc[s].to_numpy(dtype=float)
        ok = np.ptp(rest) > 0 and np.ptp(mine) > 0
        rows.append({"session": s, "n_flies": int(n[s]),
                     "r_with_rest": float(pearsonr(mine, rest).statistic) if ok else np.nan})
    return prof, pd.DataFrame(rows)


# ── figures ──────────────────────────────────────────────────────────────


def skill_words(q: float) -> str:
    """A prediction skill in words; 2% or less counts as no skill (it is noise at this n)."""
    if not np.isfinite(q):
        return "not testable"
    pct = f"{100 * q:+.0f}%".replace("+", "")
    return f"no better than the dataset average ({pct})" if q <= 0.02 else f"{pct} better than the dataset average"


def _session_label(s: str) -> str:
    try:
        d = folder_date(s)
        return f"{d.strftime('%b')} {d.day} b{folder_batch(s)}"
    except ValueError:
        return s


def fig_dataset_consistency(m: pd.DataFrame, subset, *, trained: str, subset_label: str,
                            dataset: str | None = None):
    """Q1: does the dataset's average change for an odor predict each fly?"""
    tag = phase_terms.odor_tag(dataset, "trained")
    # "(sensitized odor)" on its own line: on the odor's line it runs into the next tick
    sep = "\n" if phase_terms.is_sensitivity(dataset) else " "
    block = phase_terms.block_label(dataset)
    subset = np.asarray(subset, dtype=bool)
    odors = order_odors(_odors(m), trained)
    avg = odor_average(m, subset)
    t = avg["per_odor"].set_index("odor")
    rng = np.random.default_rng(0)
    with plt.rc_context(_RC_CONTEXT):
        fig, ax = plt.subplots(figsize=(11.5, 5.4))
        ax.axhline(0, color=INK_MUTED, linewidth=1.0, zorder=1)
        for j, o in enumerate(odors):
            d = m[f"{o}|delta"].to_numpy(dtype=float)
            x = j + rng.uniform(-0.22, 0.22, d.size)
            yj = d + rng.uniform(-0.12, 0.12, d.size)
            col = np.where(d > 0, UP, np.where(d < 0, DOWN, SAME))
            ax.scatter(x[~subset], yj[~subset], s=20, c=list(col[~subset]), edgecolors="white",
                       linewidths=0.7, zorder=2)
            ax.scatter(x[subset], yj[subset], s=30, c=list(col[subset]), edgecolors=INK,
                       linewidths=1.3, zorder=3)
            lo, hi = boot_ci(d[np.isfinite(d)])
            ax.plot([j, j], [lo, hi], color=EMPH, linewidth=2.4, zorder=4)
            ax.plot(j, np.nanmean(d), "o", color=EMPH, markersize=10, markeredgecolor="white",
                    markeredgewidth=2, zorder=5)
        def label(o):
            r = t.loc[o]
            name = f"{o}{sep}({tag})" if o == trained else o
            return f"{name}\n↑{r['pct_up']:.0f}%  ={r['pct_same']:.0f}%  ↓{r['pct_down']:.0f}%"

        ax.set_xticks(range(len(odors)))
        ax.set_xticklabels([label(o) for o in odors], fontsize=8.5)
        if odors and odors[0] == trained:
            ax.get_xticklabels()[0].set_fontweight("bold")
        ax.set_ylabel(f"Change in PER score after {block}\n(post − pre, one dot per fly)")
        _recessive_grid(ax)
        handles = [plt.Line2D([], [], marker="o", linestyle="", color=UP, label="went up"),
                   plt.Line2D([], [], marker="o", linestyle="", color=SAME, label="no change"),
                   plt.Line2D([], [], marker="o", linestyle="", color=DOWN, label="went down"),
                   plt.Line2D([], [], marker="o", linestyle="", markerfacecolor="white",
                              markeredgecolor=INK, markeredgewidth=1.3,
                              label=f"{subset_label} flies (ringed)"),
                   plt.Line2D([], [], marker="o", color=EMPH, label="dataset average, 95% CI")]
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False)
        q = avg["q2_all"]
        tr = t.loc[trained] if trained in t.index else None
        qs = avg["q2_subset"]
        pct = lambda v: f"{100 * v:.0f}%"  # noqa: E731
        sub = f"; {subset_label} flies: {pct(qs)}" if np.isfinite(qs) else ""
        if q <= 0.02:
            answer = (f"No: knowing which odor it is doesn't help predict how a fly will change ({pct(q)} "
                      f"vs one overall average{sub}). How much a fly changes depends on the fly, not on "
                      f"the odor. Odors differ consistently? Friedman {_fmt_p(avg['p_friedman'])}.")
        else:
            answer = (f"{'A little' if q < 0.2 else 'Yes'}: each odor's average change predicts individual "
                      f"flies {pct(q)} better than one overall average{sub}. Odors differ consistently: "
                      f"Friedman {_fmt_p(avg['p_friedman'])}.")
        if tr is not None:
            answer += (f"\n{trained}: {tr['pct_up']:.0f}% of flies went up, {tr['pct_same']:.0f}% "
                       f"stayed, {tr['pct_down']:.0f}% went down (average {tr['mean_change']:+.2f}).")
        _finish(fig, "Does the dataset's average change predict what an individual fly does?",
                answer, bottom=0.03)
        fig.answer_q2 = q
    return fig


def fig_batch_consistency(m: pd.DataFrame, *, trained: str, dataset: str | None = None):
    """Q2: is the pattern of changes the same from batch to batch?"""
    tag = phase_terms.odor_tag(dataset, "trained")
    prof, r = batch_consistency(m, trained=trained)
    odors = list(prof.columns)
    kw = {}
    for o in odors:
        groups = [g[f"{o}|delta"].to_numpy(dtype=float) for _s, g in m.groupby("session")
                  if _s in prof.index]
        try:
            kw[o] = float(kruskal(*groups).pvalue) if len(groups) > 1 else np.nan
        except ValueError:
            kw[o] = np.nan
    v = prof.to_numpy(dtype=float)
    lim = max(1.0, float(np.ceil(np.nanmax(np.abs(v)) * 2) / 2)) if v.size else 1.0
    from matplotlib.colors import TwoSlopeNorm
    with plt.rc_context(_RC_CONTEXT):
        fig = plt.figure(figsize=(1.05 * len(odors) + 5.0, 0.42 * len(prof) + 2.8))
        gs = fig.add_gridspec(1, 3, width_ratios=[len(odors), 1.0, 0.22])
        ax, rx, cax = (fig.add_subplot(gs[0, i]) for i in range(3))
        im = ax.imshow(v, cmap=DIVERGING, norm=TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim),
                       aspect="auto")
        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                ax.text(j, i, f"{v[i, j]:+.1f}", ha="center", va="center", fontsize=8.5,
                        color="white" if abs(v[i, j]) > 0.55 * lim else INK)
        ax.set_xticks(range(len(odors)))
        ax.set_xticklabels([f"{o}\n({tag})" if o == trained else o for o in odors], fontsize=8.5)
        ax.get_xticklabels()[0].set_fontweight("bold")
        ax.set_yticks(range(len(prof)))
        ax.set_yticklabels([f"{_session_label(s)}  (n={int(n)})"
                            for s, n in zip(r["session"], r["n_flies"])])
        ax.set_xticks(np.arange(-0.5, len(odors), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(prof), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=2)
        ax.tick_params(which="both", length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xlabel("Kruskal-Wallis across batches:  "
                      + "   ".join(f"{o} {_fmt_p(kw[o])}" for o in odors), fontsize=7.5)
        rx.barh(range(len(r)), r["r_with_rest"].fillna(0), 0.6, color=INK_MUTED, zorder=2)
        rx.set_ylim(len(r) - 0.5, -0.5)
        rx.set_xlim(-1, 1)
        rx.axvline(0, color=INK_MUTED, linewidth=0.8)
        rx.set_yticks([])
        rx.set_title("matches rest\nof dataset (r)", fontsize=8.5, color=INK)
        for i, val in enumerate(r["r_with_rest"]):
            if pd.notna(val):
                rx.text(val + (0.05 if val >= 0 else -0.05), i, f"{val:.2f}", va="center",
                        ha="left" if val >= 0 else "right", fontsize=8, color=INK)
        for sp in ("top", "right", "left"):
            rx.spines[sp].set_visible(False)
        cb = fig.colorbar(im, cax=cax)
        cb.set_label("Average change per fly (post − pre)")
        cb.outline.set_visible(False)
        med = float(np.nanmedian(r["r_with_rest"])) if len(r) else float("nan")
        n_hi = int((r["r_with_rest"] > 0.5).sum())
        answer = (f"Partly: each batch's pattern of changes across odors matches the rest of the dataset with "
                  f"median r = {med:.2f} ({n_hi} of {len(r)} batches above 0.5). "
                  f"{trained}: batches differ in how much it changes (Kruskal-Wallis {_fmt_p(kw.get(trained))}).")
        if med >= 0.7:
            answer = answer.replace("Partly", "Mostly yes")
        elif med < 0.3:
            answer = answer.replace("Partly", "No")
        _finish(fig, "Is the pattern of changes the same from batch to batch?", answer)
    return fig


def fig_change_correlations(m: pd.DataFrame, *, trained: str):
    """Q3: do a fly's changes to different odors go together?"""
    odors = order_odors(_odors(m), trained)
    c = m[[f"{o}|delta" for o in odors]].corr(method="spearman").to_numpy(dtype=float).copy()
    np.fill_diagonal(c, np.nan)
    lim = max(0.5, float(np.nanmax(np.abs(c))))
    iu = np.triu_indices_from(c, 1)
    flat = c[iu]
    with plt.rc_context(_RC_CONTEXT):
        fig = plt.figure(figsize=(9.0, 7.2))
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.05])
        ax, cax = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
        cmap = ohr._diverging_cmap().copy()
        cmap.set_bad("white")
        im = ax.imshow(c, cmap=cmap, vmin=-lim, vmax=lim)
        for i in range(c.shape[0]):
            for j in range(c.shape[1]):
                if np.isfinite(c[i, j]):
                    ax.text(j, i, f"{c[i, j]:.2f}", ha="center", va="center", fontsize=9,
                            color="white" if abs(c[i, j]) > 0.6 * lim else INK)
        ax.set_xticks(range(len(odors)))
        ax.set_xticklabels(odors, rotation=35, ha="right")
        ax.set_yticks(range(len(odors)))
        ax.set_yticklabels(odors)
        for lab in (ax.get_xticklabels()[0], ax.get_yticklabels()[0]):
            lab.set_fontweight("bold")
        ax.set_xticks(np.arange(-0.5, len(odors), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(odors), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=2)
        ax.tick_params(which="both", length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        cb = fig.colorbar(im, cax=cax)
        cb.set_label("Correlation of the change across flies (Spearman)\n"
                     "blue = change together · red = opposite")
        cb.outline.set_visible(False)
        cs = c[0, 1:] if trained in odors else np.array([])
        partner = odors[1 + int(np.nanargmax(cs))] if cs.size and np.isfinite(cs).any() else None
        mean_r = float(np.nanmean(flat))
        if mean_r >= 0.15:
            how = "Yes, mostly" if mean_r >= 0.3 else "Somewhat"
            meaning = ("A fly whose PER rises to one odor tends to rise to the others too: part of "
                       "sensitization is general, not odor-specific.")
        else:
            how = "No"
            meaning = ("A fly that rises to one odor is no more likely to rise to another: each odor's "
                       "change is its own (or noise), not a general shift of the whole fly.")
        answer = (f"{how}: across flies the changes correlate at r = {mean_r:.2f} on average "
                  f"({np.nanmin(flat):.2f} to {np.nanmax(flat):.2f}). {meaning}")
        if partner is not None:
            answer += (f"\n{trained}'s change goes with the others at r = {np.nanmean(cs):.2f} on average; "
                       f"closest partner {partner} (r = {np.nanmax(cs):.2f}).")
        _finish(fig, "Do a fly's changes to different odors go together?", answer)
    return fig


SKILL_STYLE = {
    "delta_others": dict(marker="D", color=INK, label="from its changes to the other odors"),
    "pre_others": dict(marker="o", color=PRE_COLOR, label="from its pre-test to the other odors"),
    "pre_own": dict(marker="s", color=LIGHT_BLUE, label="from its own starting score (this odor)"),
}


def fig_skill(res: pd.DataFrame, *, trained: str, subset_label: str, dataset: str | None = None):
    """Q4: can a fly's change be predicted from its other changes or its pre-test?"""
    pre_name = phase_terms.pre_label(dataset, "pre-test", case="lower")
    tag = phase_terms.odor_tag(dataset, "trained")
    r = res[res["model"] == "Ridge"]
    odors = order_odors(sorted(r["odor"].unique(), key=str.casefold), trained)[::-1]
    lo = max(CLIP, float(np.floor(100 * np.nanmin(r[["q2_all", "q2_subset"]].to_numpy()) / 10) * 10) - 5)
    hi = max(40.0, float(np.ceil(100 * np.nanmax(r[["q2_all", "q2_subset"]].to_numpy()) / 10) * 10) + 5)
    offsets = {"delta_others": 0.22, "pre_others": 0.0, "pre_own": -0.22}
    with plt.rc_context(_RC_CONTEXT):
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6), sharey=True)
        for ax, (qk, pk, title) in zip(axes, (("q2_all", "p_all", "All flies in the dataset"),
                                             ("q2_subset", "p_subset", f"Only the {subset_label} flies"))):
            ax.axvspan(lo, 0, color=SHADE, zorder=0)
            ax.axvline(0, color=INK_MUTED, linewidth=1.0, zorder=1)
            for i, o in enumerate(odors):
                for w, st in SKILL_STYLE.items():
                    row = r[(r["odor"] == o) & (r["predictors"] == w)]
                    if not len(row) or pd.isna(row[qk].iloc[0]):
                        continue
                    sig = pd.notna(row[pk].iloc[0]) and row[pk].iloc[0] < 0.05
                    ax.plot(float(np.clip(100 * row[qk].iloc[0], lo, None)), i + offsets[w],
                            linestyle="", marker=st["marker"], markersize=8,
                            markerfacecolor=st["color"] if sig else "white",
                            markeredgecolor=st["color"], markeredgewidth=1.8, zorder=3)
            n = int(r["n_all" if qk == "q2_all" else "n_subset"].max())
            ax.set_title(f"{title}  (n = {n} flies)", loc="left", fontsize=10.5, color=INK)
            ax.set_xlim(lo, hi)
            ax.set_xlabel("Prediction skill (% better than the dataset's average change)")
            ax.text(lo + 1, -0.6, "worse than the\ndataset average", fontsize=7.5, color=INK_MUTED,
                    va="bottom")
            _recessive_grid(ax)
        axes[0].set_yticks(range(len(odors)))
        axes[0].set_yticklabels([f"{o} ({tag})" if o == trained else o for o in odors])
        axes[0].get_yticklabels()[-1].set_fontweight("bold")
        handles = [plt.Line2D([], [], linestyle="", marker=st["marker"], color=st["color"],
                              markersize=8, label=st["label"].replace("pre-test", pre_name))
                   for st in SKILL_STYLE.values()]
        handles += [plt.Line2D([], [], linestyle="", marker="o", color=INK, markersize=8,
                               label="filled = p<0.05 (not luck)"),
                    plt.Line2D([], [], linestyle="", marker="o", markerfacecolor="white",
                               markeredgecolor=INK, markersize=8, label="hollow = could be luck")]
        fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
                   bbox_to_anchor=(0.5, -0.02))

        def med(w, key="q2_all"):
            return float(r.loc[r["predictors"] == w, key].median())

        def nsig(w):
            s = r[r["predictors"] == w]
            return f"{int((s['p_all'] < 0.05).sum())} of {len(s)}"

        answer = (f"From its changes to the other odors: {skill_words(med('delta_others'))}; "
                  f"{nsig('delta_others')} odors beat luck. From its {pre_name} to the other odors: "
                  f"{skill_words(med('pre_others'))}; {nsig('pre_others')}. From its own starting score: "
                  f"{skill_words(med('pre_own'))}; {nsig('pre_own')} -- a fly that starts high has room only "
                  f"to go down.\n{subset_label} flies (models trained on the rest): "
                  f"{skill_words(med('delta_others', 'q2_subset'))} from the other changes. Medians over odors; "
                  "Ridge; each rig folder predicted by a model that never saw it.")
        _finish(fig, "Can a fly's change to one odor be predicted from the rest of the fly?", answer,
                bottom=0.09)
    return fig


def fig_trained_example(preds: pd.DataFrame, res: pd.DataFrame, m: pd.DataFrame, subset, *,
                        trained: str, subset_label: str):
    """Q5: how much of the trained-odor change is the fly's general change?"""
    p = preds[(preds["odor"] == trained) & (preds["predictors"] == "delta_others")
              & (preds["model"] == "Ridge")]
    row = res[(res["odor"] == trained) & (res["predictors"] == "delta_others") & (res["model"] == "Ridge")]
    q, pa = float(row["q2_all"].iloc[0]), float(row["p_all"].iloc[0])
    qs, ps = float(row["q2_subset"].iloc[0]), float(row["p_subset"].iloc[0])
    s = p["in_subset"].to_numpy(dtype=bool)
    rng = np.random.default_rng(0)
    with plt.rc_context(_RC_CONTEXT):
        fig, ax = plt.subplots(figsize=(8.2, 6.2))
        x = p["predicted"].to_numpy(dtype=float)
        y = p["actual"].to_numpy(dtype=float) + rng.uniform(-0.15, 0.15, len(p))
        ax.scatter(x[~s], y[~s], s=22, color="#a8a7a1", edgecolors="white", linewidths=0.8,
                   zorder=2, label="other flies")
        ax.scatter(x[s], y[s], s=34, color=INK, edgecolors="white", linewidths=1.0, zorder=3,
                   label=f"{subset_label} flies")
        lo = float(min(np.nanmin(x), np.nanmin(y))) - 0.5
        hi = float(max(np.nanmax(x), np.nanmax(y))) + 0.5
        ax.plot([lo, hi], [lo, hi], color=INK, linewidth=1.3, zorder=1, label="perfect prediction")
        ax.axhline(float(np.nanmean(p["dataset_average"])), color=INK_MUTED, linewidth=1.3,
                   label="dataset average change")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(f"Predicted {trained} change\n(from the fly's changes to the other odors)")
        ax.set_ylabel(f"Actual {trained} change (post − pre)")
        ax.legend(loc="upper left", frameon=False, fontsize=8)
        _recessive_grid(ax)
        if q > 0.02:
            answer = (f"{100 * q:.0f}% of the fly-to-fly differences in {trained} change follow the fly's changes "
                      f"to the other odors ({_fmt_p(pa)}): that part is general sensitization; the other "
                      f"{100 - 100 * q:.0f}% is specific to {trained}, or noise.")
        else:
            answer = (f"None of it: a fly's {trained} change can't be predicted from its changes to the other "
                      f"odors -- {skill_words(q)}, {_fmt_p(pa)}. Whatever drives it is specific to {trained} "
                      "(or noise), not a general sensitization of the fly.")
        answer += f" {subset_label} flies: {skill_words(qs)}, {_fmt_p(ps)}."
        _finish(fig, f"How much of a fly's {trained} change is its general change?", answer)
    return fig


# ── driver ───────────────────────────────────────────────────────────────


def subset_by_start(m: pd.DataFrame, dataset: str, lo: float, hi: float,
                    data_root: Path = DATA_ROOT) -> np.ndarray:
    """Flies whose rig folder's first pre-test started at clock hour [lo, hi)."""
    from scripts.analysis.time_of_day_trend import clock_hours, first_pretest_time

    clock = {}
    for f in m["fly"].unique():
        t = first_pretest_time(Path(data_root) / dataset / f)
        clock[f] = clock_hours(t) if t is not None else np.nan
    c = m["fly"].map(clock).to_numpy(dtype=float)
    return (c >= lo) & (c < hi)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", default="IAA-Sensitivity-1-1")
    ap.add_argument("--after", type=pd.Timestamp, default=None,
                    help="only batch folders recorded after this date (YYYY-MM-DD)")
    ap.add_argument("--subset-hours", type=float, nargs=2, default=None, metavar=("FROM", "TO"),
                    help="flies whose first pre-test started in [FROM, TO) clock hours, e.g. 13 14")
    ap.add_argument("--subset-label", default=None)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--predictions-csv", type=Path, default=CSV / "model_predictions.csv")
    ap.add_argument("--flagged-flies-csv", type=Path, default=CSV / "flagged-flys-truth.csv")
    ap.add_argument("--data-root", type=Path, default=DATA_ROOT)
    ap.add_argument("--figures-only", action="store_true",
                    help="redraw the figures from the tables a previous run saved (no refitting)")
    args = ap.parse_args(argv)

    from scripts.analysis.time_of_day_trend import load_paired

    out = args.out_dir or OUT_ROOT / f"{args.dataset}_sensitization_consistency"
    out.mkdir(parents=True, exist_ok=True)
    if args.figures_only:
        m = pd.read_csv(out / "change_matrix_per_fly.csv")
        subset = m.pop("in_subset").to_numpy(dtype=bool)
        res, preds = pd.read_csv(out / "prediction_skill.csv"), pd.read_csv(out / "predictions_per_fly.csv")
        return _render(m, subset, res, preds, out=out, dataset=args.dataset,
                       trained=trained_odor(args.dataset), label=args.subset_label or "subset")
    paired = load_paired([args.dataset], predictions_csv=args.predictions_csv,
                         flagged_flies_csv=args.flagged_flies_csv, tmp_dir=out / "inputs",
                         after=args.after.date() if args.after is not None else None)
    paired["group"] = f"{starvation_hours(args.dataset)[0]:g} h"
    m = change_matrix(paired)
    if args.subset_hours:
        lo, hi = args.subset_hours
        subset = subset_by_start(m, args.dataset, lo, hi, args.data_root)
        label = args.subset_label or f"{lo:g}:00–{hi:g}:00"
    else:
        subset, label = np.zeros(len(m), dtype=bool), args.subset_label or "subset"
    trained = trained_odor(args.dataset)
    print(f"{len(m)} flies, {m['batch'].nunique()} rig folders, {m['session'].nunique()} batches; "
          f"{int(subset.sum())} in '{label}'. Trained odor: {trained}")

    res, preds = run(m, subset, n_perm=args.n_perm, n_jobs=args.n_jobs)
    avg = odor_average(m, subset)
    prof, cons = batch_consistency(m, trained=trained)
    m.assign(in_subset=subset).to_csv(out / "change_matrix_per_fly.csv", index=False)
    res.to_csv(out / "prediction_skill.csv", index=False)
    preds.to_csv(out / "predictions_per_fly.csv", index=False)
    avg["per_odor"].to_csv(out / "dataset_average_per_odor.csv", index=False)
    prof.join(cons.set_index("session")).to_csv(out / "batch_profiles.csv")
    return _render(m, subset, res, preds, out=out, dataset=args.dataset, trained=trained, label=label)


def _render(m, subset, res, preds, *, out: Path, dataset: str, trained: str, label: str) -> int:
    figs = [("01_dataset_average_vs_individual.png",
             fig_dataset_consistency(m, subset, trained=trained, subset_label=label, dataset=dataset)),
            ("02_batch_to_batch.png", fig_batch_consistency(m, trained=trained, dataset=dataset)),
            ("03_changes_go_together.png", fig_change_correlations(m, trained=trained)),
            ("04_predict_change.png", fig_skill(res, trained=trained, subset_label=label,
                                                dataset=dataset)),
            ("05_trained_odor_general_vs_specific.png",
             fig_trained_example(preds, res, m, subset, trained=trained, subset_label=label))]
    answers = [f"{dataset}: {len(m)} flies ({int(subset.sum())} in '{label}'), "
               f"{m['batch'].nunique()} rig folders. Change = "
               f"{phase_terms.post_label(dataset, 'post-training', case='lower')} minus "
               f"{phase_terms.pre_label(dataset, 'pre-test', case='lower')} PER score.\n"]
    for name, fig in figs:
        with plt.rc_context(_RC_CONTEXT):
            fig.savefig(out / name, bbox_inches="tight")
        answers.append(f"{name}\n{fig.answer}\n")
        plt.close(fig)
        print(f"[OK] {out / name}")
    (out / "ANSWERS.txt").write_text("\n".join(answers))
    print(f"[OK] {out / 'ANSWERS.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
