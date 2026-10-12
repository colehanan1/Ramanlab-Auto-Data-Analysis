#!/usr/bin/env python3
"""Can a fly's (or a batch's) response to one odor be predicted from the others?

Hold-one-odor-out Lasso and Ridge regression on the sensitivity cohorts, at
three levels:

1. **Individual, within one starvation group.** Hold out odor k for fly f.
   Predict f's score on k (pre-test or post-training) from f's own scores on
   the other six odors, both phases (12 features), with a model fitted only on
   the OTHER flies of the same starvation group (leave-one-fly-out). The
   baseline is the population answer: the other flies' mean on k.
2. **Individual, pooled across starvation.** All flies together; feature sets
   "odors", "odors + starvation" (hours, hours²) and "starvation only". Two
   schemes: leave-one-batch-out (a batch's flies share a day and rig, so they
   leave together) and leave-one-starvation-group-out (train on the other
   starvation times, predict this one).
3. **Population, across starvation.** One row per batch (its flies' mean score
   and % responding per odor-phase); same feature sets and schemes, with
   leave-one-batch-out in place of leave-one-fly-out.

Scores are regressed with Ridge (``RidgeCV``) and Lasso (``LassoCV``);
responded/not (score >= 2) with L2 and L1 logistic regression. The
regularisation strength is chosen INSIDE each training fold, and features are
standardised inside it too, so the held-out unit never informs its own
prediction.

Metric: out-of-sample Q² = 1 - SSE(model) / SSE(baseline), the baseline being
each fold's training mean (prevalence, for responded/not, where Q² is the Brier
skill score). Q² > 0 means the other odors predict better than the population
average; Q² <= 0 means they add nothing. Every Q² carries a permutation p: the
target is shuffled across units (features untouched), the whole cross-validation
re-run, and p = (1 + #null >= observed) / (1 + n_perm). Uncorrected.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.starvation_trend_figure import hours_from_label  # noqa: E402

PHASES = ("pre", "post")
BINARY_THRESHOLD = 2
RIDGE_ALPHAS = np.logspace(-2, 4, 25)
LASSO_ALPHAS = np.logspace(-3, 1, 15)
LOGIT_CS = np.logspace(-2, 2, 9)
REGRESSORS = ("ridge", "lasso")
CLASSIFIERS = ("logit_l2", "logit_l1")
MODEL_NAME = {"ridge": "Ridge", "lasso": "Lasso",
              "logit_l2": "Ridge", "logit_l1": "Lasso"}  # L2 / L1 logistic
STARVATION = ("hours", "hours2")
VARIANTS = ("odors", "odors + starvation", "starvation only")
ID_COLS = ["group", "hours", "fly", "fly_number", "batch"]


# ── data ──────────────────────────────────────────────────────────────────


def fly_matrix(paired: pd.DataFrame) -> pd.DataFrame:
    """One row per fly: group, hours, batch, and ``<odor>|pre`` / ``<odor>|post`` scores."""
    odors = sorted(paired["odor"].unique(), key=str.casefold)
    cols = [f"{o}|{ph}" for o in odors for ph in PHASES]
    rows = []
    for (group, fly, number), sub in paired.groupby(["group", "fly", "fly_number"], sort=True):
        row = {"group": group, "hours": hours_from_label(group), "fly": fly,
               "fly_number": number, "batch": f"{group}/{fly}"}
        for _, r in sub.iterrows():
            row[f"{r['odor']}|pre"] = float(r["score_pre"])
            row[f"{r['odor']}|post"] = float(r["score_post"])
        rows.append(row)
    df = pd.DataFrame(rows).dropna(subset=cols)  # complete flies only
    return df[ID_COLS + cols].sort_values(["hours", "batch", "fly_number"]).reset_index(drop=True)


def score_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if isinstance(c, str) and c.count("|") == 1]


def feature_target(df: pd.DataFrame, odor: str, phase: str, *, suffix: str = "",
                   extra: Sequence[str] = ()):
    """X = every OTHER odor (both phases), y = ``<odor>|<phase>``. Both phases of
    the held-out odor are excluded, so its pre score never predicts its post."""
    feats = [f"{c}{suffix}" for c in score_columns(df) if c.split("|")[0] != odor]
    feats += list(extra)
    return (df[feats].to_numpy(dtype=float), df[f"{odor}|{phase}{suffix}"].to_numpy(dtype=float),
            feats)


def batch_table(m: pd.DataFrame, *, binary_threshold: int = BINARY_THRESHOLD) -> pd.DataFrame:
    """One row per batch: mean score and % responding per odor-phase, plus n flies."""
    cols = score_columns(m)
    rows = []
    for (group, hours, batch), sub in m.groupby(["group", "hours", "batch"], sort=True):
        row = {"group": group, "hours": hours, "batch": batch, "n_flies": len(sub)}
        for c in cols:
            row[c] = float(sub[c].mean())
            row[f"{c}|pct"] = 100.0 * float((sub[c] >= binary_threshold).mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["hours", "batch"]).reset_index(drop=True)


# ── cross-validation ─────────────────────────────────────────────────────


def loo_splits(n: int) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    idx = np.arange(n)
    for i in range(n):
        yield np.delete(idx, i), np.array([i])


def group_splits(labels: Sequence) -> list[tuple[np.ndarray, np.ndarray]]:
    labels = np.asarray(labels)
    idx = np.arange(len(labels))
    return [(idx[labels != u], idx[labels == u]) for u in pd.unique(labels)]


def make_model(kind: str, n_train: int, minority: int | None = None):
    from sklearn.linear_model import LassoCV, LogisticRegressionCV, RidgeCV
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if kind == "ridge":
        est = RidgeCV(alphas=RIDGE_ALPHAS)
    elif kind == "lasso":
        est = LassoCV(alphas=LASSO_ALPHAS, cv=max(2, min(5, n_train)), max_iter=50_000)
    elif kind in CLASSIFIERS:
        k = max(2, min(3, minority or 3))
        est = LogisticRegressionCV(
            Cs=LOGIT_CS, cv=StratifiedKFold(k, shuffle=True, random_state=0),
            penalty="l2" if kind == "logit_l2" else "l1", solver="liblinear",
            scoring="neg_brier_score", max_iter=5000)
    else:
        raise ValueError(kind)
    return make_pipeline(StandardScaler(), est)


def cv_predict(X, y, splits, *, make: Callable, binary: bool = False):
    """Out-of-fold predictions and the per-fold training-mean baseline."""
    pred = np.full(len(y), np.nan)
    base = np.full(len(y), np.nan)
    for train, test in splits:
        ytr = y[train]
        b = float(np.mean(ytr))
        base[test] = b
        if np.ptp(ytr) == 0:
            pred[test] = b
            continue
        if binary:
            pos = int(ytr.sum())
            minority = min(pos, len(ytr) - pos)
            if minority < 2:  # too few of one class to fit or tune
                pred[test] = b
                continue
            model = make(len(train), minority).fit(X[train], ytr.astype(int))
            pred[test] = model.predict_proba(X[test])[:, 1]
        else:
            pred[test] = make(len(train)).fit(X[train], ytr).predict(X[test])
    return pred, base


def q2(y, pred, base) -> float:
    den = float(np.sum((y - base) ** 2))
    if den == 0:
        return float("nan")
    return 1.0 - float(np.sum((y - pred) ** 2)) / den


def permutation_p(observed: float, null: np.ndarray) -> float:
    null = np.asarray(null, dtype=float)
    null = null[np.isfinite(null)]
    if not np.isfinite(observed) or null.size == 0:
        return float("nan")
    return (1 + int(np.sum(null >= observed))) / (1 + null.size)


def _permute(y, strata, rng):
    if strata is None:
        return rng.permutation(y)
    out = y.copy()
    for s in pd.unique(strata):
        idx = np.flatnonzero(strata == s)
        out[idx] = rng.permutation(y[idx])
    return out


def evaluate(X, y, splits, *, kind: str, n_perm: int, seed: int, strata=None) -> dict:
    binary = kind in CLASSIFIERS
    splits = list(splits)

    def make(n, minority=None):
        return make_model(kind, n, minority)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # convergence chatter on tiny folds
        pred, base = cv_predict(X, y, splits, make=make, binary=binary)
        observed = q2(y, pred, base)
        rng = np.random.default_rng(seed)
        null = np.array([q2(yp, *cv_predict(X, yp, splits, make=make, binary=binary))
                         for yp in (_permute(y, strata, rng) for _ in range(n_perm))])
    out = {"n": len(y), "q2": observed, "p_perm": permutation_p(observed, null) if n_perm else np.nan,
           "null_q2_mean": float(np.nanmean(null)) if n_perm else np.nan,
           "spearman": np.nan, "auc": np.nan, "mean_target": float(np.mean(y))}
    if np.ptp(pred) > 0 and np.ptp(y) > 0:
        out["spearman"] = float(spearmanr(pred, y).statistic)
    if binary and 0 < y.sum() < len(y):
        from sklearn.metrics import roc_auc_score
        out["auc"] = float(roc_auc_score(y, pred))
    return out


# ── the three levels ─────────────────────────────────────────────────────


def _odors(m: pd.DataFrame) -> list[str]:
    return sorted({c.split("|")[0] for c in score_columns(m)}, key=str.casefold)


def _run(tasks: list[dict], n_jobs: int) -> pd.DataFrame:
    from joblib import Parallel, delayed

    def one(t):
        meta = {k: v for k, v in t.items() if k not in ("X", "y", "splits", "strata", "n_perm", "seed")}
        return {**meta, **evaluate(t["X"], t["y"], t["splits"], kind=t["kind"],
                                   n_perm=t["n_perm"], seed=t["seed"], strata=t.get("strata"))}

    rows = Parallel(n_jobs=n_jobs)(delayed(one)(t) for t in tasks) if n_jobs != 1 else [one(t) for t in tasks]
    df = pd.DataFrame(rows)
    df["model"] = df["kind"].map(MODEL_NAME)
    return df


def level1_tasks(m, *, n_perm, binary_threshold=BINARY_THRESHOLD):
    tasks = []
    for group, sub in m.groupby("group", sort=False):
        sub = sub.reset_index(drop=True)
        splits = list(loo_splits(len(sub)))
        for odor in _odors(sub):
            for phase in PHASES:
                X, y, _ = feature_target(sub, odor, phase)
                for target, yy, kinds in (("score", y, REGRESSORS),
                                          ("responded", (y >= binary_threshold).astype(float),
                                           CLASSIFIERS)):
                    for kind in kinds:
                        tasks.append(dict(level="1 individual, within starvation group",
                                          group=group, hours=float(sub["hours"].iloc[0]),
                                          odor=odor, phase=phase, target=target, kind=kind,
                                          scheme="leave-one-fly-out", variant="odors",
                                          X=X, y=yy, splits=splits, n_perm=n_perm,
                                          seed=len(tasks)))
    return tasks


def _variant_X(df, odor, phase, variant, *, suffix=""):
    if variant == "starvation only":
        _, y, _ = feature_target(df, odor, phase, suffix=suffix)
        return df[list(STARVATION)].to_numpy(dtype=float), y
    extra = STARVATION if variant == "odors + starvation" else ()
    X, y, _ = feature_target(df, odor, phase, suffix=suffix, extra=extra)
    return X, y


def _with_hours2(df):
    df = df.copy()
    df["hours2"] = df["hours"] ** 2
    return df


def level2_tasks(m, *, n_perm, binary_threshold=BINARY_THRESHOLD):
    df = _with_hours2(m).reset_index(drop=True)
    schemes = {"leave-one-batch-out": group_splits(df["batch"]),
               "leave-one-starvation-group-out": group_splits(df["group"])}
    tasks = []
    for odor in _odors(df):
        for phase in PHASES:
            for variant in VARIANTS:
                X, y = _variant_X(df, odor, phase, variant)
                for target, yy, kinds in (("score", y, REGRESSORS),
                                          ("responded", (y >= binary_threshold).astype(float),
                                           CLASSIFIERS)):
                    for scheme, splits in schemes.items():
                        for kind in kinds:
                            tasks.append(dict(level="2 individual, pooled across starvation",
                                              group="all", odor=odor, phase=phase, target=target,
                                              kind=kind, scheme=scheme, variant=variant, X=X, y=yy,
                                              splits=splits, n_perm=n_perm, seed=len(tasks)))
    return tasks


def level3_tasks(m, *, n_perm, binary_threshold=BINARY_THRESHOLD):
    b = _with_hours2(batch_table(m, binary_threshold=binary_threshold)).reset_index(drop=True)
    schemes = {"leave-one-batch-out": list(loo_splits(len(b))),
               "leave-one-starvation-group-out": group_splits(b["group"])}
    tasks = []
    for odor in _odors(b):
        for phase in PHASES:
            for variant in VARIANTS:
                for target, suffix in (("mean score", ""), ("% responding", "|pct")):
                    X, y = _variant_X(b, odor, phase, variant, suffix=suffix)
                    for scheme, splits in schemes.items():
                        for kind in REGRESSORS:
                            tasks.append(dict(level="3 population (batch), across starvation",
                                              group="all", odor=odor, phase=phase, target=target,
                                              kind=kind, scheme=scheme, variant=variant, X=X, y=y,
                                              splits=splits, n_perm=n_perm, seed=len(tasks)))
    return tasks


def ridge_coefficients(m: pd.DataFrame) -> pd.DataFrame:
    """Descriptive: standardised Ridge coefficients fitted on ALL flies (odors only).

    In-sample, so it shows which odors co-vary, not predictive skill."""
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for odor in _odors(m):
            for phase in PHASES:
                X, y, feats = feature_target(m, odor, phase)
                coef = make_model("ridge", len(y)).fit(X, y)[-1].coef_
                for f, c in zip(feats, coef):
                    rows.append({"target": f"{odor}|{phase}", "feature": f, "coef": float(c)})
    return pd.DataFrame(rows)


# ── figures ──────────────────────────────────────────────────────────────

#: Ridge vs Lasso: two categorical slots that are NOT this repo's pre/post blue
#: and orange (validated with the dataviz validator, --pairs all).
MODEL_COLORS = {"Ridge": "#1baf7a", "Lasso": "#4a3aa7"}
DIVERGING = ("#c43d3c", "#f0efec", "#256abf")  # worse than average ← 0 → better


def _diverging_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("q2", DIVERGING)


def _target_label(odor: str, phase: str) -> str:
    return f"{odor} — {'pre-test' if phase == 'pre' else 'post-training'}"


def figure_overview(l1, l2, l3, out: Path, trained_odor: str) -> Path:
    """Every held-out target as one dot per setting: is any of it above zero?"""
    import matplotlib.pyplot as plt

    from scripts.analysis.pretest_vs_test_comparison import INK, INK_MUTED, _RC_CONTEXT

    settings = []
    for g, sub in l1.sort_values("hours").groupby("group", sort=False):
        settings.append((f"Individual · within {g}", sub))
    for scheme, short in (("leave-one-batch-out", "new batch"),
                          ("leave-one-starvation-group-out", "new starvation time")):
        for v in VARIANTS:
            settings.append((f"Individual · pooled · {v} · predict {short}",
                             l2[(l2.scheme == scheme) & (l2.variant == v)]))
    for scheme, short in (("leave-one-batch-out", "new batch"),
                          ("leave-one-starvation-group-out", "new starvation time")):
        for v in VARIANTS:
            settings.append((f"Population (batch) · {v} · predict {short}",
                             l3[(l3.scheme == scheme) & (l3.variant == v)]))

    panels = (("Score (0–5 PER score / batch mean score)", ("score", "mean score")),
              ("Responded (score ≥ 2) / batch % responding", ("responded", "% responding")))
    with plt.rc_context({**_RC_CONTEXT, "savefig.dpi": 200}):
        fig, axes = plt.subplots(1, 2, figsize=(14, 0.36 * len(settings) + 2.2), sharey=True)
        rng = np.random.default_rng(0)
        for ax, (title, targets) in zip(axes, panels):
            ax.axvline(0, color="#9a9994", linewidth=0.9, zorder=1)
            for yi, (_label, sub) in enumerate(settings):
                sub = sub[sub.target.isin(targets)]
                for mi, model in enumerate(("Ridge", "Lasso")):
                    s = sub[sub.model == model]
                    q = s["q2"].clip(-1.5, 1.0).to_numpy()
                    sig = (s["p_perm"] < 0.05).to_numpy()
                    yy = yi + (mi - 0.5) * 0.3 + rng.uniform(-0.06, 0.06, len(q))
                    color = MODEL_COLORS[model]
                    ax.scatter(q[~sig], yy[~sig], s=22, facecolors="white", edgecolors=color,
                               linewidths=1.1, zorder=3)
                    ax.scatter(q[sig], yy[sig], s=34, color=color, edgecolors="white",
                               linewidths=0.8, zorder=4)
            ax.set_xlim(-1.55, 1.05)
            ax.set_xlabel("Out-of-sample Q²  (0 = no better than the population average)")
            ax.set_title(title, loc="left", color=INK, fontsize=10)
            ax.xaxis.grid(True, color="#e4e4e0", linewidth=0.6)
            ax.set_axisbelow(True)
        axes[0].set_yticks(range(len(settings)))
        axes[0].set_yticklabels([s[0] for s in settings], fontsize=8)
        axes[0].invert_yaxis()
        from matplotlib.lines import Line2D
        handles = [Line2D([], [], marker="o", ls="", color=MODEL_COLORS[m], label=m)
                   for m in ("Ridge", "Lasso")]
        handles += [Line2D([], [], marker="o", ls="", markerfacecolor="white",
                           markeredgecolor=INK_MUTED, label="p ≥ 0.05"),
                    Line2D([], [], marker="o", ls="", color=INK_MUTED, label="p < 0.05 (filled)")]
        # Above the plots: a legend inside the panel would sit on the dots.
        fig.legend(handles=handles, loc="upper right", ncol=4, frameon=False, fontsize=8.5,
                   bbox_to_anchor=(0.99, 0.995))
        fig.suptitle("Hold-one-odor-out prediction: one dot per held-out odor × phase (14 per row)\n"
                     "Q² below −1.5 drawn at −1.5 · permutation p, uncorrected · "
                     "Ridge/Lasso = L2/L1 logistic for responded",
                     x=0.01, ha="left", fontsize=10.5, color=INK)
        fig.tight_layout()
        path = out / "overview_q2_all_levels.png"
        fig.savefig(path)
        plt.close(fig)
    return path


def figure_level1_heatmap(l1, out: Path, trained_odor: str) -> Path:
    import matplotlib.pyplot as plt

    from scripts.analysis.pretest_vs_test_comparison import INK, _RC_CONTEXT

    groups = list(l1.sort_values("hours")["group"].unique())
    odors = sorted(l1["odor"].unique(), key=str.casefold)
    rows = [(o, p) for o in odors for p in PHASES]
    with plt.rc_context(_RC_CONTEXT):
        fig, axes = plt.subplots(1, 4, figsize=(15, 6.2), sharey=True,
                                 gridspec_kw={"width_ratios": [1, 1, 1, 1]})
        cmap = _diverging_cmap()
        im = None
        for ax, (target, model) in zip(axes, [("score", "Ridge"), ("score", "Lasso"),
                                              ("responded", "Ridge"), ("responded", "Lasso")]):
            s = l1[(l1.target == target) & (l1.model == model)]
            mat = np.full((len(rows), len(groups)), np.nan)
            pmat = np.full_like(mat, np.nan)
            for i, (o, p) in enumerate(rows):
                for j, g in enumerate(groups):
                    r = s[(s.odor == o) & (s.phase == p) & (s.group == g)]
                    if len(r):
                        mat[i, j], pmat[i, j] = r["q2"].iloc[0], r["p_perm"].iloc[0]
            im = ax.imshow(np.clip(mat, -1, 1), cmap=cmap, vmin=-1, vmax=1, aspect="auto")
            for i in range(len(rows)):
                for j in range(len(groups)):
                    if np.isfinite(pmat[i, j]) and pmat[i, j] < 0.05:
                        ax.text(j, i, f"{mat[i, j]:.2f}*", ha="center", va="center",
                                fontsize=7.5, color="white" if abs(mat[i, j]) > 0.55 else INK,
                                fontweight="bold")
            ax.set_xticks(range(len(groups)))
            ax.set_xticklabels(groups)
            ax.set_title(f"{'Score' if target == 'score' else 'Responded'} · {model}",
                         loc="left", color=INK, fontsize=10)
            ax.tick_params(length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
        axes[0].set_yticks(range(len(rows)))
        labels = axes[0].set_yticklabels([_target_label(o, p) for o, p in rows], fontsize=8)
        for lab, (o, _p) in zip(labels, rows):
            if trained_odor and o.casefold().startswith(trained_odor.casefold()):
                lab.set_fontweight("bold")
        cax = fig.add_axes([0.92, 0.25, 0.012, 0.5])
        cb = fig.colorbar(im, cax=cax)
        cb.set_label("Q² (clipped to ±1)")
        fig.suptitle("Level 1 — individual flies, within each starvation group: predict a fly's held-out odor "
                     "from its own other 6 odors (leave-one-fly-out)\nblue = better than the other flies' "
                     "average · red = worse · value shown only where permutation p < 0.05 (uncorrected) · "
                     "white = not testable (every fly gave the same response)",
                     x=0.01, ha="left", fontsize=10.5, color=INK)
        fig.subplots_adjust(left=0.2, right=0.9, top=0.84, bottom=0.08, wspace=0.08)
        path = out / "level1_individual_within_starvation_heatmap.png"
        fig.savefig(path)
        plt.close(fig)
    return path


def figure_coefficients(coefs: pd.DataFrame, out: Path, trained_odor: str) -> Path:
    import matplotlib.pyplot as plt

    from scripts.analysis.pretest_vs_test_comparison import INK, _RC_CONTEXT

    targets = list(dict.fromkeys(coefs["target"]))
    feats = targets
    mat = np.full((len(targets), len(feats)), np.nan)
    for _, r in coefs.iterrows():
        mat[targets.index(r["target"]), feats.index(r["feature"])] = r["coef"]
    v = float(np.nanmax(np.abs(mat))) or 1.0
    label = [_target_label(*t.split("|")) for t in targets]
    with plt.rc_context(_RC_CONTEXT):
        fig, ax = plt.subplots(figsize=(10.5, 8.2))
        im = ax.imshow(mat, cmap=_diverging_cmap(), vmin=-v, vmax=v, aspect="auto")
        ax.set_xticks(range(len(feats)))
        ax.set_xticklabels(label, rotation=55, ha="right", fontsize=7.5)
        ax.set_yticks(range(len(targets)))
        ylab = ax.set_yticklabels(label, fontsize=7.5)
        for lab in list(ylab) + list(ax.get_xticklabels()):
            if trained_odor and lab.get_text().casefold().startswith(trained_odor.casefold()):
                lab.set_fontweight("bold")
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xlabel("Predictor (the fly's response to this odor)")
        ax.set_ylabel("Predicted (held-out odor)")
        cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cb.set_label("Standardised Ridge coefficient")
        ax.set_title("Which odor responses go together, fly by fly (all flies, all starvation times)\n"
                     "descriptive, in-sample Ridge fit · blank = the held-out odor itself",
                     loc="left", color=INK, fontsize=10)
        fig.tight_layout()
        path = out / "relationships_ridge_coefficients.png"
        fig.savefig(path)
        plt.close(fig)
    return path


# ── driver ───────────────────────────────────────────────────────────────


def run_all(paired: pd.DataFrame, *, out_dir: Path, n_perm: int = 200, n_perm_pop: int = 500,
            n_jobs: int = -1, binary_threshold: int = BINARY_THRESHOLD,
            trained_odor: str = "Isoamyl Acetate", figures: bool = True) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    m = fly_matrix(paired)
    written = []

    l1 = _run(level1_tasks(m, n_perm=n_perm, binary_threshold=binary_threshold), n_jobs)
    l2 = _run(level2_tasks(m, n_perm=n_perm, binary_threshold=binary_threshold), n_jobs)
    l3 = _run(level3_tasks(m, n_perm=n_perm_pop, binary_threshold=binary_threshold), n_jobs)
    for name, df in (("level1_individual_within_starvation.csv", l1),
                     ("level2_individual_pooled.csv", l2),
                     ("level3_population_batches.csv", l3)):
        df.to_csv(out_dir / name, index=False)
        written.append(out_dir / name)
    m.to_csv(out_dir / "input_fly_matrix.csv", index=False)
    batch_table(m, binary_threshold=binary_threshold).to_csv(out_dir / "input_batch_table.csv",
                                                             index=False)
    coefs = ridge_coefficients(m)
    coefs.to_csv(out_dir / "relationships_ridge_coefficients.csv", index=False)
    written += [out_dir / "input_fly_matrix.csv", out_dir / "input_batch_table.csv",
                out_dir / "relationships_ridge_coefficients.csv"]
    if figures:
        written.append(figure_overview(l1, l2, l3, out_dir, trained_odor))
        written.append(figure_level1_heatmap(l1, out_dir, trained_odor))
        written.append(figure_coefficients(coefs, out_dir, trained_odor))
    return written


def figures_from_csv(out_dir: Path, *, trained_odor: str = "Isoamyl Acetate") -> list[Path]:
    """Redraw the figures from the tables ``run_all`` saved; no refitting."""
    out_dir = Path(out_dir)
    l1 = pd.read_csv(out_dir / "level1_individual_within_starvation.csv")
    l2 = pd.read_csv(out_dir / "level2_individual_pooled.csv")
    l3 = pd.read_csv(out_dir / "level3_population_batches.csv")
    coefs = pd.read_csv(out_dir / "relationships_ridge_coefficients.csv")
    return [figure_overview(l1, l2, l3, out_dir, trained_odor),
            figure_level1_heatmap(l1, out_dir, trained_odor),
            figure_coefficients(coefs, out_dir, trained_odor)]


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--comparison-dir", required=True, type=Path,
                    help="Folder with paired_scores_per_fly.csv (starvation_pre_post_comparison output).")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Default: <comparison-dir>/../odor_holdout_regression")
    ap.add_argument("--n-perm", type=int, default=200, help="Permutations, individual levels.")
    ap.add_argument("--n-perm-pop", type=int, default=500, help="Permutations, population level.")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--trained-odor", default="Isoamyl Acetate")
    ap.add_argument("--figures-only", action="store_true",
                    help="Redraw the figures from the saved tables in --out-dir.")
    args = ap.parse_args(argv)
    out = args.out_dir or args.comparison_dir.parent / "odor_holdout_regression"
    if args.figures_only:
        for p in figures_from_csv(out, trained_odor=args.trained_odor):
            print(f"[OK] {p}")
        return 0
    paired = pd.read_csv(args.comparison_dir / "paired_scores_per_fly.csv")
    for p in run_all(paired, out_dir=out, n_perm=args.n_perm, n_perm_pop=args.n_perm_pop,
                     n_jobs=args.n_jobs, trained_odor=args.trained_odor):
        print(f"[OK] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
