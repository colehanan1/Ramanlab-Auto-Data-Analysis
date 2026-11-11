#!/usr/bin/env python3
"""Predict a fly's post-exposure PER (yes/no) from its pre-exposure behaviour,
its training block and the sensitized odor, within and across the Sensitivity cohorts.

Unit = one fly x one odor (the window build's ``window_per_fly_per_odor.csv``).
PER = ``responded_*`` from that table (score >= binary_threshold).

Outputs (``--out-dir``, default ``<results>/PER-Prediction``):
  per_prediction_table.csv     modelling table, one row per fly x odor
  transitions_by_cohort.csv    pre->post transition counts (pooled trials)
  gee_pooled.csv / gee_by_cohort.csv  logistic GEE (fly clusters) odds ratios
  prediction_cv.csv            held-out AUC (grouped by fly; leave-one-cohort-out)
  heterogeneity_permutation.json  batch-permutation test of cohort differences
  fig1..fig5 PNG/SVG, FINDINGS.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

RESULTS = Path("/home/ramanlab/Documents/cole/Results/Sensitiviy-Results(11-15)")
TRANSITIONS = ["stay no", "gain", "loss", "stay yes"]
TRANS_COLORS = {"stay no": "#d9d9d9", "gain": "#1b7837", "loss": "#762a83", "stay yes": "#7f7f7f"}
SHORT = {"3Oct-Sensitivity-1-0.1": "3-Octanol", "Hex-Sensitivity-1-0.1": "Hexanol",
         "IAA-Sensitivity-1-1": "Isoamyl acetate", "Paraffin-Sensitivity-1-100": "Paraffin"}


# ---------------------------------------------------------------- features
def add_transitions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    pre, post = df["responded_pre"].astype(bool), df["responded_post"].astype(bool)
    df["transition"] = np.select(
        [~pre & ~post, ~pre & post, pre & ~post], ["stay no", "gain", "loss"], "stay yes")
    return df


def add_fly_features(df: pd.DataFrame) -> pd.DataFrame:
    """Leave-this-odor-out pre-exposure PER rate: the fly's general responsiveness."""
    df = df.copy()
    pre = df["responded_pre"].astype(float)
    tot = pre.groupby(df["fly_id"]).transform("sum")
    n = pre.groupby(df["fly_id"]).transform("count")
    df["pre_rate_other_odors"] = ((tot - pre) / (n - 1)).where(n > 1)
    return df


def merge_training(df: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    t = train.rename(columns={"fly": "batch_folder", "mean_auc": "train_mean_auc",
                              "per_any": "train_per_any"})
    t = t[["batch_folder", "fly_number", "train_mean_auc", "train_per_any"]]
    return df.merge(t, on=["batch_folder", "fly_number"], how="left")


def pooled_rate(df: pd.DataFrame, label: str) -> float:
    """Share of all fly x odor trials with this transition (trials pooled)."""
    return float((df["transition"] == label).mean())


def load_training(results: Path) -> pd.DataFrame:
    parts = []
    for p in results.glob("*/Training-AUC-vs-Testing-Response/*/*_per_fly.csv"):
        parts.append(pd.read_csv(p))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["fly", "fly_number", "mean_auc", "per_any"])


def build_table(results: Path) -> pd.DataFrame:
    df = pd.read_csv(results / "Data/window_per_fly_per_odor.csv")
    df = df[df["in_window"].astype(bool)].dropna(subset=["responded_pre", "responded_post"])
    for c in ("responded_pre", "responded_post", "sensitized_odor"):
        df[c] = df[c].astype(str).str.lower().eq("true")
    df = add_fly_features(add_transitions(df))
    df = merge_training(df, load_training(results))
    df["log_train_auc"] = np.log1p(df["train_mean_auc"])
    df["y"] = df["responded_post"].astype(int)
    df["pre"] = df["responded_pre"].astype(int)
    df["sens"] = df["sensitized_odor"].astype(int)
    df["cohort_s"] = df["cohort"].map(SHORT).fillna(df["cohort"])
    return df


# ---------------------------------------------------------------- models
def gee(formula: str, df: pd.DataFrame):
    return smf.gee(formula, groups="fly_id", data=df, family=sm.families.Binomial(),
                   cov_struct=sm.cov_struct.Exchangeable()).fit()


def or_table(res, label: str) -> pd.DataFrame:
    ci = res.conf_int()
    return pd.DataFrame({"model": label, "term": res.params.index,
                         "odds_ratio": np.exp(res.params.values),
                         "ci_low": np.exp(ci[0].values), "ci_high": np.exp(ci[1].values),
                         "p": res.pvalues.values})


FEATURE_SETS = {
    "base rate": [],
    "pre PER (this odor)": ["pre"],
    "+ fly responsiveness": ["pre", "pre_rate_other_odors"],
    "+ sensitized odor": ["pre", "pre_rate_other_odors", "sens", "pre_x_sens"],
    "+ training block": ["pre", "pre_rate_other_odors", "sens", "pre_x_sens",
                         "log_train_auc", "train_per_any"],
}


def cv_auc(df: pd.DataFrame, feats: list[str], splitter: str) -> float:
    X = df[feats].to_numpy(float) if feats else np.zeros((len(df), 1))
    y, groups = df["y"].to_numpy(), df["fly_id"].to_numpy()
    pred = np.full(len(df), np.nan)
    if splitter == "fly":
        splits = GroupKFold(n_splits=5).split(X, y, groups)
    else:
        cohorts = df["cohort"].to_numpy()
        splits = [(np.where(cohorts != c)[0], np.where(cohorts == c)[0]) for c in np.unique(cohorts)]
    for tr, te in splits:
        if not feats:
            pred[te] = y[tr].mean()
            continue
        m = LogisticRegression(max_iter=1000).fit(X[tr], y[tr])
        pred[te] = m.predict_proba(X[te])[:, 1]
    if np.unique(pred).size == 1:
        return 0.5
    return float(roc_auc_score(y, pred))


def sens_boost_by_cohort(df: pd.DataFrame) -> pd.Series:
    """Gain rate on the sensitized odor minus gain rate on the other odors (pooled trials),
    among trials with no pre-exposure PER."""
    d = df[~df["responded_pre"]]
    g = d.groupby(["cohort", "sensitized_odor"])["responded_post"].mean().unstack()
    return (g.get(True) - g.get(False)).dropna()


def heterogeneity_perm(df: pd.DataFrame, n: int = 2000, seed: int = 0) -> dict:
    """Batch is nested in cohort, so permute whole batches between cohorts (keeping
    cohort sizes in batches) and the sensitized-odor flag with them."""
    obs = sens_boost_by_cohort(df)
    stat = float(obs.var(ddof=0))
    batches = df[["batch_folder", "cohort"]].drop_duplicates()
    rng = np.random.default_rng(seed)
    sens_odor = df[df["sensitized_odor"]].groupby("cohort")["odor"].first()
    hits = 0
    for _ in range(n):
        perm = dict(zip(batches["batch_folder"], rng.permutation(batches["cohort"].to_numpy())))
        d = df.copy()
        d["cohort"] = d["batch_folder"].map(perm)
        d["sensitized_odor"] = d["odor"].to_numpy() == d["cohort"].map(sens_odor).to_numpy()
        # only odors every cohort tested can be compared under relabelling
        if sens_boost_by_cohort(d).var(ddof=0) >= stat:
            hits += 1
    return {"statistic": "variance across cohorts of (sensitized gain rate - other-odor gain rate)",
            "observed_boost": obs.round(4).to_dict(), "observed_stat": stat,
            "n_perm": n, "p": (hits + 1) / (n + 1)}


# ---------------------------------------------------------------- figures
def _save(fig, out: Path, name: str):
    fig.savefig(out / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(out / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def fig_transitions(df, out):
    cohorts = list(SHORT)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x, labels = [], []
    for i, c in enumerate(cohorts):
        for j, s in enumerate([True, False]):
            d = df[(df["cohort"] == c) & (df["sensitized_odor"] == s)]
            if d.empty:
                continue
            pos = i * 2.6 + j
            bottom = 0
            for t in TRANSITIONS:
                v = pooled_rate(d, t) * 100
                ax.bar(pos, v, bottom=bottom, color=TRANS_COLORS[t], width=0.9,
                       label=t if (i == 0 and j == 0) else None, edgecolor="white", lw=0.5)
                bottom += v
            x.append(pos)
            labels.append(f"{SHORT[c].replace('Isoamyl acetate', 'IAA')}\n{'sensitized' if s else 'other odors'}\nn={len(d)} trials")
    ax.set_xticks(x, labels, fontsize=7)
    ax.set_ylabel("% of fly × odor trials")
    ax.set_title("Pre-exposure → post-exposure PER, sensitized odor vs the other odors")
    ax.legend(title="pre → post", bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, out, "fig1_pre_post_transitions")


def fig_forest(tab: pd.DataFrame, out, name, title):
    t = tab[tab["term"] != "Intercept"].iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(t) + 1.2))
    yy = np.arange(len(t))
    ax.errorbar(t["odds_ratio"], yy, xerr=[t["odds_ratio"] - t["ci_low"], t["ci_high"] - t["odds_ratio"]],
                fmt="o", color="black", capsize=3)
    ax.axvline(1, color="grey", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_yticks(yy, [f"{r.term}  (p={r.p:.3g})" for r in t.itertuples()], fontsize=8)
    ax.set_xlabel("odds ratio for post-exposure PER (95% CI, fly-clustered GEE)")
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, out, name)


def fig_cv(cv: pd.DataFrame, out):
    fig, ax = plt.subplots(figsize=(8, 4))
    sets = list(FEATURE_SETS)
    w = 0.38
    for k, (sp, col) in enumerate([("held-out flies", "#4d4d4d"), ("held-out cohort", "#9e9e9e")]):
        v = cv.set_index("features").loc[sets, sp]
        ax.bar(np.arange(len(sets)) + (k - 0.5) * w, v, w, color=col, label=sp)
    ax.axhline(0.5, color="grey", ls="--", lw=1)
    ax.set_ylim(0.4, 1)
    ax.set_xticks(range(len(sets)), sets, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("ROC AUC (out-of-sample)")
    ax.set_title("How well can a fly's post-exposure PER be predicted?")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, out, "fig4_prediction_auc")


def fig_responsiveness(df, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    bins = [-0.01, 0.0, 0.2, 0.4, 1.0]
    lab = ["0", "0–0.2", "0.2–0.4", ">0.4"]
    for ax, pre in zip(axes, [False, True]):
        for s, col in [(True, "#1b7837"), (False, "#762a83")]:
            d = df[(df["responded_pre"] == pre) & (df["sensitized_odor"] == s)].copy()
            d["bin"] = pd.cut(d["pre_rate_other_odors"], bins, labels=lab)
            g = d.groupby("bin", observed=False)["y"].agg(["mean", "count"])
            ax.plot(lab, g["mean"] * 100, "o-", color=col,
                    label=("sensitized odor" if s else "other odors"))
            for xi, (m, n) in enumerate(zip(g["mean"], g["count"])):
                if n:
                    ax.annotate(f"n={n}", (xi, m * 100), textcoords="offset points",
                                xytext=(0, 6), ha="center", fontsize=6, color=col)
        ax.set_title(f"{'PER' if pre else 'no PER'} to this odor before exposure")
        ax.set_xlabel("fly's pre-exposure PER rate on its OTHER odors")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("% trials with post-exposure PER")
    axes[0].legend(frameon=False)
    fig.suptitle("Post-exposure PER vs the fly's general responsiveness (n = fly × odor trials)")
    _save(fig, out, "fig5_fly_responsiveness")


# ---------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", type=Path, default=RESULTS)
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--n-perm", type=int, default=2000)
    a = ap.parse_args(argv)
    out = a.out_dir or a.results / "PER-Prediction"
    out.mkdir(parents=True, exist_ok=True)

    df = build_table(a.results)
    df["pre_x_sens"] = df["pre"] * df["sens"]
    model_df = df.dropna(subset=["pre_rate_other_odors", "log_train_auc", "train_per_any"]).copy()
    df.to_csv(out / "per_prediction_table.csv", index=False)

    # transitions
    tr = (df.groupby(["cohort", "sensitized_odor", "transition"]).size()
            .unstack(fill_value=0).reindex(columns=TRANSITIONS, fill_value=0))
    tr["n_trials"] = tr.sum(axis=1)
    tr.to_csv(out / "transitions_by_cohort.csv")
    fig_transitions(df, out)

    # pooled GEE
    f_pool = ("y ~ pre * sens + pre_rate_other_odors + log_train_auc + train_per_any"
              " + C(cohort_s)")
    pooled = or_table(gee(f_pool, model_df), "pooled")
    pooled.to_csv(out / "gee_pooled.csv", index=False)
    fig_forest(pooled, out, "fig2_pooled_odds_ratios",
               f"All 4 cohorts: what predicts post-exposure PER\n"
               f"(n={len(model_df)} trials, {model_df.fly_id.nunique()} flies)")

    # per-cohort GEE
    per = []
    for c, d in model_df.groupby("cohort_s"):
        try:
            per.append(or_table(gee("y ~ pre + sens + pre_rate_other_odors + log_train_auc", d), c))
        except Exception as e:  # noqa: BLE001 - small cohort can fail to converge
            print(f"GEE failed for {c}: {e}")
    per = pd.concat(per, ignore_index=True)
    per.to_csv(out / "gee_by_cohort.csv", index=False)
    s = per[per["term"] == "sens"].copy()
    s["term"] = s["model"]
    fig_forest(s, out, "fig3_sensitized_effect_by_cohort",
               "Sensitized-odor effect per cohort (adjusted for pre PER,\nresponsiveness, training)")

    # prediction
    rows = []
    for name, feats in FEATURE_SETS.items():
        rows.append({"features": name, "held-out flies": cv_auc(model_df, feats, "fly"),
                     "held-out cohort": cv_auc(model_df, feats, "cohort")})
    cv = pd.DataFrame(rows)
    cv.to_csv(out / "prediction_cv.csv", index=False)
    fig_cv(cv, out)
    fig_responsiveness(model_df, out)

    het = heterogeneity_perm(df, n=a.n_perm)
    (out / "heterogeneity_permutation.json").write_text(json.dumps(het, indent=2))

    write_findings(out, df, model_df, tr, pooled, per, cv, het)
    print(f"wrote {out}")


def _fmt(t, term):
    r = t[t["term"] == term].iloc[0]
    return f"OR {r.odds_ratio:.2f} (95% CI {r.ci_low:.2f}–{r.ci_high:.2f}), p={r.p:.3g}"


def write_findings(out, df, model_df, tr, pooled, per, cv, het):
    L = ["# Predicting post-exposure PER from pre-exposure behaviour and training", "",
         "Built by `scripts/analysis/per_prediction_model.py` from "
         "`Data/window_per_fly_per_odor.csv` (window flies, flagged removed). "
         "Outcome = PER yes/no (score ≥ 2). Unit = one fly × one odor; rates pool trials.", "",
         f"**{len(df)} fly × odor trials, {df.fly_id.nunique()} flies, 4 cohorts.** "
         f"{len(model_df)} trials have training-block data and enter the models.", "",
         "## 1. Pre → post transitions (fig1)", "",
         "| Cohort | Odor set | stay no | gain | loss | stay yes | n trials |", "|---|---|---|---|---|---|---|"]
    for (c, s), r in tr.iterrows():
        n = r["n_trials"]
        L.append(f"| {SHORT.get(c, c)} | {'sensitized' if s else 'other odors'} | "
                 + " | ".join(f"{100 * r[t] / n:.0f}%" for t in TRANSITIONS) + f" | {n} |")
    L += ["", "## 2. What predicts a post-exposure PER, all cohorts (fig2)", "",
          "Logistic GEE, flies as clusters (exchangeable), cohort as fixed effect.", "",
          f"- PER to the same odor before exposure: {_fmt(pooled, 'pre')}",
          f"- Fly's PER rate on its other odors (responsiveness): {_fmt(pooled, 'pre_rate_other_odors')}",
          f"- Sensitized odor (fly had no pre PER): {_fmt(pooled, 'sens')}",
          f"- Sensitized × pre PER interaction: {_fmt(pooled, 'pre:sens')}",
          f"- log(1 + training mean AUC): {_fmt(pooled, 'log_train_auc')}",
          f"- Any PER during training: {_fmt(pooled, 'train_per_any')}", "",
          "## 3. Sensitized-odor effect per cohort (fig3)", ""]
    for r in per[per["term"] == "sens"].itertuples():
        L.append(f"- {r.model}: OR {r.odds_ratio:.2f} ({r.ci_low:.2f}–{r.ci_high:.2f}), p={r.p:.3g}")
    L += ["", f"Batch-permutation test that the sensitized boost differs between cohorts "
          f"(batches shuffled between cohorts, {het['n_perm']} perms): **p={het['p']:.3g}**. "
          "Observed boost (sensitized − other-odor gain rate, trials with no pre PER): "
          + ", ".join(f"{SHORT.get(k, k)} {v * 100:+.0f} pp" for k, v in het["observed_boost"].items()) + ".", "",
          "## 4. How predictable is it? (fig4)", "",
          "Out-of-sample ROC AUC of a logistic model (0.5 = chance).", "",
          "| Features | held-out flies (5-fold) | held-out cohort |", "|---|---|---|"]
    for r in cv.itertuples(index=False):
        L.append(f"| {r[0]} | {r[1]:.3f} | {r[2]:.3f} |")
    L += ["", "## 5. Responsiveness vs sensitization (fig5)", "",
          "Post-exposure PER rate binned by the fly's pre-exposure PER rate on its other odors, "
          "split by whether the fly responded to this odor before exposure.", "",
          "## Caveats", "",
          "- Batch is nested in cohort, so cohort contrasts can carry day/rig/batch effects; "
          "the permutation test above is the honest cross-cohort test.",
          "- One pre and one post trial per odor; per-fly predictions are noisy by construction.",
          "- Each cohort sensitizes one odor, so 'sensitized' and the odor identity overlap within a cohort.",
          "- Paraffin cohort: the 'sensitized odor' is the paraffin carrier itself (control arm)."]
    (out / "FINDINGS.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
