#!/usr/bin/env python3
"""IAA-Sensitivity-1-1: the isoamyl-acetate response on rig 1 vs rig 2 vs rig 3.

Reads production's Sensitivity results (``Sensitiviy-Results(11-15)/Data``,
flagged flies already dropped) and, for comparison, the same tables from the
per-frame-eye-held-while-light-on copy. Per rig: % of flies responding to IAA
before and after exposure, the paired tests, the response through the six
training exposures, and the batches behind each rig (batch is nested in rig:
every batch ran on one rig, so a rig difference can also be a batch/day
difference).

One trial per fly per odor per phase, so every % is responders / flies
(trials pooled, never a mean of per-batch means).

Writes ``Results/Figures/IAA-Rig-Comparison/`` (hand-run analysis).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

COLE = Path("/home/ramanlab/Documents/cole")
PROD = COLE / "Results" / "Sensitiviy-Results(11-15)" / "Data"
LIGHTHOLD = COLE / "Results" / "Figures" / "LiveEye-Comparison" / "livelight" / "Sensitivity-Results" / "Data"
TRAINING = COLE / "Data" / "CSVs-New-Opto-Flys" / "model_predictions_training.csv"
OUT = COLE / "Results" / "Figures" / "IAA-Rig-Comparison"
COHORT = "IAA-Sensitivity-1-1"
ODOR = "Isoamyl Acetate"
THRESHOLD = 2   # binary_threshold: score >= 2 is a response
RIG_HOST = {1: "behaviorlocust", 2: "flybehavior2", 3: "flybehavior3"}
# dataviz categorical slots 1-3, validated (light surface): CVD ΔE >= 9.2; aqua < 3:1 → lines direct-labelled
RIG_COLOR = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a"}
POP_FILES = {"window": "window_per_fly_per_odor.csv", "whole cohort": "whole-cohort_per_fly_per_odor.csv"}


def mcnemar_exact(pre: pd.Series, post: pd.Series) -> float:
    b = int((~pre & post).sum())
    c = int((pre & ~post).sum())
    return float(stats.binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0


def wilcoxon_p(change: pd.Series) -> float:
    d = change[change != 0]
    return float(stats.wilcoxon(d).pvalue) if len(d) else 1.0


def per_rig(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (odor, rig), g in df.groupby(["odor", "rig"]):
        rows.append(_row(g, odor=odor, rig=int(rig)))
    for odor, g in df.groupby("odor"):
        rows.append(_row(g, odor=odor, rig="all"))
    return pd.DataFrame(rows)


def _row(g: pd.DataFrame, **keys) -> dict:
    pre, post = g.responded_pre.astype(bool), g.responded_post.astype(bool)
    return {
        **keys,
        "n_flies": len(g),
        "pct_responding_pre": round(100 * pre.mean(), 1),
        "pct_responding_post": round(100 * post.mean(), 1),
        "change_pct_points": round(100 * (post.mean() - pre.mean()), 1),
        "n_became_responder": int((~pre & post).sum()),
        "n_stopped_responding": int((pre & ~post).sum()),
        "mean_score_pre": round(g.score_pre.mean(), 2),
        "mean_score_post": round(g.score_post.mean(), 2),
        "mean_score_change": round(g.score_change.mean(), 2),
        "p_mcnemar_exact": round(mcnemar_exact(pre, post), 4),
        "p_wilcoxon": round(wilcoxon_p(g.score_change), 4),
    }


def between_rigs(g: pd.DataFrame) -> dict:
    """Do the rigs differ? chi-square on responders (pre, post) and Kruskal-Wallis on score change."""
    out = {}
    for col in ("responded_pre", "responded_post"):
        tab = pd.crosstab(g.rig, g[col].astype(bool))
        if tab.shape == (3, 2):
            chi2, p, _, exp = stats.chi2_contingency(tab)
            out[f"p_chi2_{col}"] = round(float(p), 4)
            out[f"min_expected_{col}"] = round(float(exp.min()), 1)
    groups = [x.score_change.to_numpy() for _, x in g.groupby("rig")]
    out["p_kruskal_score_change"] = round(float(stats.kruskal(*groups).pvalue), 4)
    return out


def training_by_rig(flies: pd.DataFrame) -> pd.DataFrame:
    """% of the rig's flies responding at pre, each training exposure, post (IAA)."""
    t = pd.read_csv(TRAINING)
    t = t[t.dataset == COHORT].copy()
    t["fly_id"] = t.fly.astype(str) + "_fly" + t.fly_number.astype(int).astype(str)
    t = t[t.fly_id.isin(set(flies.fly_id))]
    t["position"] = t.trial_label.str.extract(r"training_(\d+)")[0].astype(int)
    t = t.merge(flies[["fly_id", "rig"]], on="fly_id")
    t["responded"] = t.score >= THRESHOLD
    rows = []
    for rig, g in flies.groupby("rig"):
        rows.append({"rig": int(rig), "position": 0, "label": "pre", "n": len(g),
                     "pct_responding": 100 * g.responded_pre.mean()})
        for pos, h in t[t.rig == rig].groupby("position"):
            rows.append({"rig": int(rig), "position": int(pos), "label": f"T{pos}", "n": len(h),
                         "pct_responding": 100 * h.responded.mean()})
        rows.append({"rig": int(rig), "position": 7, "label": "post", "n": len(g),
                     "pct_responding": 100 * g.responded_post.mean()})
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tables, tests, traj, batches = [], [], {}, []
    for method, root in (("production", PROD), ("per-frame eye, held while light on", LIGHTHOLD)):
        for pop, fname in POP_FILES.items():
            p = root / fname
            if not p.exists():
                continue
            d = pd.read_csv(p)
            d = d[d.cohort == COHORT]
            tables.append(per_rig(d).assign(method=method, population=pop))
            iaa = d[d.odor == ODOR]
            tests.append({"method": method, "population": pop, **between_rigs(iaa)})
            if method == "production":
                traj[pop] = training_by_rig(iaa)
                b = iaa.groupby(["rig", "batch_folder"]).agg(
                    n_flies=("fly_id", "nunique"),
                    first_pretest=("start_time", "min"),
                    pct_pre=("responded_pre", lambda s: round(100 * s.mean(), 1)),
                    pct_post=("responded_post", lambda s: round(100 * s.mean(), 1)),
                ).reset_index()
                batches.append(b.assign(population=pop))
    summary = pd.concat(tables, ignore_index=True)
    lead = ["method", "population", "odor", "rig"]
    summary = summary[lead + [c for c in summary.columns if c not in lead]]
    summary.to_csv(OUT / "iaa_by_rig_all_odors.csv", index=False)
    iaa = summary[summary.odor == ODOR]
    iaa.to_csv(OUT / "iaa_by_rig_sensitized_odor.csv", index=False)
    pd.DataFrame(tests).to_csv(OUT / "between_rig_tests.csv", index=False)
    pd.concat(batches).to_csv(OUT / "batches_per_rig.csv", index=False)
    pd.concat([v.assign(population=k) for k, v in traj.items()]).round(1).to_csv(
        OUT / "iaa_response_by_trial_and_rig.csv", index=False)

    _figure(iaa, traj)
    pd.set_option("display.width", 220)
    print(iaa.to_string(index=False))
    print(pd.DataFrame(tests).to_string(index=False))
    print(pd.concat(batches).to_string(index=False))
    for pop, t in traj.items():
        print(pop)
        print(t.pivot(index="rig", columns="label", values="pct_responding")
              [["pre", "T1", "T2", "T3", "T4", "T5", "T6", "post"]].round(0).to_string())


def _figure(iaa: pd.DataFrame, traj: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    prod = iaa[(iaa.method == "production")]
    pops = [p for p in ("window", "whole cohort") if p in set(prod.population)]
    fig, axes = plt.subplots(len(pops), 2, figsize=(12, 4.4 * len(pops)), squeeze=False,
                             gridspec_kw={"width_ratios": [1, 1.6]})
    for r, pop in enumerate(pops):
        ax = axes[r, 0]
        for rig in (1, 2, 3):
            row = prod[(prod.population == pop) & (prod.rig.astype(str) == str(rig))]
            if row.empty:
                continue
            row = row.iloc[0]
            y = [row.pct_responding_pre, row.pct_responding_post]
            ax.plot([0, 1], y, "-o", color=RIG_COLOR[rig], lw=2, ms=8, mec="white", mew=1.5)
            ax.text(1.05, y[1], f"Rig {rig}  {y[0]:.0f}→{y[1]:.0f}%  (n={row.n_flies}, p={row.p_mcnemar_exact:.2f})",
                    color="#1d1d1b", va="center", fontsize=9)
        allrow = prod[(prod.population == pop) & (prod.rig.astype(str) == "all")].iloc[0]
        ax.plot([0, 1], [allrow.pct_responding_pre, allrow.pct_responding_post], "--", color="#8a8a86", lw=1.5)
        ax.text(-0.05, allrow.pct_responding_pre, f"all {allrow.pct_responding_pre:.0f}%", ha="right",
                va="center", fontsize=9, color="#4a4a47")
        ax.set_xticks([0, 1], ["pre-exposure", "post-exposure"])
        ax.set_xlim(-0.35, 1.9)
        ax.set_ylim(0, 100)
        ax.set_ylabel("% of flies responding to IAA (score ≥ 2)")
        ax.set_title(f"{pop} flies: IAA pre vs post (McNemar exact p)", fontsize=10, loc="left")
        ax.grid(axis="y", color="#ecebe7", lw=0.8)

        ax = axes[r, 1]
        t = traj[pop]
        order = ["pre", "T1", "T2", "T3", "T4", "T5", "T6", "post"]
        for rig in (1, 2, 3):
            g = t[t.rig == rig].set_index("label").reindex(order)
            ax.plot(range(len(order)), g.pct_responding, "-o", color=RIG_COLOR[rig], lw=2, ms=6, mec="white", mew=1.2)
            last = g.pct_responding.dropna()
            ax.text(len(order) - 0.85, last.iloc[-1], f"Rig {rig} (n={int(g.n.max())})", color="#1d1d1b",
                    va="center", fontsize=9)
        ax.set_xticks(range(len(order)), ["pre", "exp 1", "exp 2", "exp 3", "exp 4", "exp 5", "exp 6", "post"])
        ax.set_xlim(-0.3, len(order) + 0.6)
        ax.set_ylim(0, 100)
        ax.set_ylabel("% of flies responding to IAA")
        ax.set_title(f"{pop} flies: IAA through the six exposures", fontsize=10, loc="left")
        ax.grid(axis="y", color="#ecebe7", lw=0.8)
    fig.suptitle("IAA-Sensitivity-1-1: isoamyl-acetate response by rig\nproduction scores · flagged flies dropped · p uncorrected · n = flies")
    fig.tight_layout()
    fig.savefig(OUT / "iaa_response_by_rig.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
