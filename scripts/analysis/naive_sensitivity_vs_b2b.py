#!/usr/bin/env python3
"""Naive (pre-exposure) PER: the 4 Sensitivity cohorts vs the b2b day-window flies.

Both sides are scored on odor seconds 0-10: the Sensitivity pre-test (30 s odor) from
the first-10 s rescore (``per_fly_per_odor_first_10s_odor_window.csv``), the b2b
pre-block (10 s odor) from production (``Results/b2b/pretest/pretest_naive_trials.csv``;
a <= 10 s odor scores identically under the rescore). Same clock window (09:40-16:20
starts), same vials and concentrations.

Plan, fixed before the data were looked at (``PLAN``, pinned by a test):

* groups: the 4 Sensitivity cohorts pooled vs b2b IAA-first, IAA-second, both pooled;
* metric: % of trials responding (score >= binary_threshold) and mean score, trials
  pooled (never a mean of per-fly means), over the 7 odors -- Paraffin Oil is left out
  of the pooled set because the b2b "Paraffin Oil" trial is a master-valve click;
* test: two-sided permutation of the group labels over whole SESSIONS (no session
  holds both a Sensitivity and a b2b fly, so day effects sit at session level);
  batch-folder and fly-level p beside it; Holm over the 3 b2b groups (pooled odors) or
  the 8 odors (per odor); 95 % CI of the difference from a session bootstrap.

Hand-run; writes ``Results/Figures/Naive_Sensitivity-vs-b2b_first-10s/``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from fbpipe.analysis import fly_stats as fs  # noqa: E402
from scripts.analysis import b2b_stats as bs  # noqa: E402

RESULTS = Path("/home/ramanlab/Documents/cole/Results")
SENS_CSV = (RESULTS / "Sensitiviy-Results(11-15)" / "Pre-vs-Post-Sensitivity-Odor-First-10s"
            / "per_fly_per_odor_first_10s_odor_window.csv")
B2B_CSV = RESULTS / "b2b" / "pretest" / "pretest_naive_trials.csv"
OUT = RESULTS / "Figures" / "Naive_Sensitivity-vs-b2b_first-10s"
CONFIG = ROOT / "config" / "config_new.yaml"

SENS = "Sensitivity, 4 cohorts"
FIRST, SECOND, BOTH = "b2b IAA-first", "b2b IAA-second", "b2b both"
B2B_GROUP = {"IAA-PairFirst-Control-1-1": FIRST, "IAA-PairSecond-Control-1-1": SECOND}
SENS_COHORTS = ("IAA-Sensitivity-1-1", "Hex-Sensitivity-1-0.1", "3Oct-Sensitivity-1-0.1",
                "Paraffin-Sensitivity-1-100")
CONTROL = "Paraffin Oil"
ODORS = ("IAA", "Hexanol", "3-Octanol", CONTROL, "Benzaldehyde", "Citral", "Ethyl Butyrate",
         "Linalool")
POOLED = "7 odors pooled"
B2B_GROUPS = (FIRST, SECOND, BOTH)

PLAN = {
    "registered": "2026-11-18, before the comparison was run",
    "groups": f"{SENS} vs {FIRST}, {SECOND}, {BOTH}",
    "metrics": "% of trials responding (score >= binary_threshold) and mean score, trials pooled",
    "pooled_odors": "the 7 odors; Paraffin Oil left out (b2b: master-valve click)",
    "headline_cluster": "session",
    "also": "batch-folder and fly-level permutation p",
    "holm": "over the 3 b2b groups (pooled odors); over the 8 odors (each odor)",
    "ci": "95 % percentile CI of the difference, sessions resampled within each group",
}
HEADLINE = "p_session"
CLUSTERS = {"session": "p_session", "batch": "p_batch", "fly_id": "p_fly"}
N_BOOT, SEED = 10_000, 0

COLORS = {SENS: "#52514e", FIRST: "#6250d6", SECOND: "#eda100", BOTH: "#e34948"}
COHORT_FILL = "#c3c2b7"
INK, INK2, RULE, GRID = "#0b0b0b", "#52514e", "#c3c2b7", "#e1e0d9"
SHORT = {"IAA-Sensitivity-1-1": "IAA-Sens", "Hex-Sensitivity-1-0.1": "Hex-Sens",
         "3Oct-Sensitivity-1-0.1": "3Oct-Sens", "Paraffin-Sensitivity-1-100": "Paraffin-Sens",
         SENS: "Sensitivity\nall 4", FIRST: "b2b\nIAA-first", SECOND: "b2b\nIAA-second",
         BOTH: "b2b\nboth"}
METRICS = {"rate": "% of trials responding", "score": "mean PER score"}
TICK = {"IAA-Sensitivity-1-1": "IAA", "Hex-Sensitivity-1-0.1": "Hex",
        "3Oct-Sensitivity-1-0.1": "3Oct", "Paraffin-Sensitivity-1-100": "Paraffin",
        SENS: "all 4", FIRST: "IAA-\nfirst", SECOND: "IAA-\nsecond", BOTH: "both"}


# ── data ─────────────────────────────────────────────────────────────────────
def binary_threshold(config: Path = CONFIG) -> float:
    import yaml
    return float(yaml.safe_load(Path(config).read_text())["reaction_prediction"]["binary_threshold"])


def load_trials(sens_csv: Path = SENS_CSV, b2b_csv: Path = B2B_CSV, *,
                threshold: float) -> pd.DataFrame:
    """One row per naive trial: source, group, cohort, fly_id, batch, session, odor,
    score, responded. Sensitivity flies outside the clock window are dropped."""
    s = pd.read_csv(sens_csv)
    s = s[s["in_window"].astype(bool) & s["score_pre"].notna()]
    sens = pd.DataFrame({
        "source": "Sensitivity", "group": SENS, "cohort": s["cohort"].astype(str),
        "fly_id": s["cohort"].astype(str) + "|" + s["fly_id"].astype(str),
        "batch": s["batch_folder"].astype(str),
        "odor": s["odor"].replace({"Isoamyl Acetate": "IAA"}).astype(str),
        "score": s["score_pre"].astype(float)})
    b = pd.read_csv(b2b_csv)
    unknown = sorted(set(b["dataset"]) - set(B2B_GROUP))
    if unknown:
        raise ValueError(f"unknown b2b dataset(s): {unknown}")
    b = b[b["score"].notna()]
    pair = pd.DataFrame({
        "source": "b2b", "group": b["dataset"].map(B2B_GROUP), "cohort": b["dataset"].astype(str),
        "fly_id": (b["dataset"].astype(str) + "|" + b["fly"].astype(str) + "_fly"
                   + b["fly_number"].astype(int).astype(str)),
        "batch": b["fly"].astype(str), "odor": b["odor"].astype(str),
        "score": b["score"].astype(float)})
    t = pd.concat([sens, pair], ignore_index=True)
    t["session"] = t["batch"].map(bs.session_of)
    t["responded"] = t["score"] >= threshold
    return t


def groups(trials: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Each Sensitivity cohort, all of them pooled, each b2b group and both pooled."""
    g = {c: trials[trials["cohort"] == c] for c in SENS_COHORTS if (trials["cohort"] == c).any()}
    g[SENS] = trials[trials["group"] == SENS]
    g[FIRST] = trials[trials["group"] == FIRST]
    g[SECOND] = trials[trials["group"] == SECOND]
    g[BOTH] = trials[trials["source"] == "b2b"]
    return g


def odor_rows(rows: pd.DataFrame, odor_set: str) -> pd.DataFrame:
    if odor_set == POOLED:
        return rows[rows["odor"] != CONTROL]
    return rows[rows["odor"] == odor_set]


# ── statistics ───────────────────────────────────────────────────────────────
def _values(rows: pd.DataFrame, metric: str) -> np.ndarray:
    return (rows["responded"].astype(float) * 100.0 if metric == "rate"
            else rows["score"].astype(float)).to_numpy()


def level(rows: pd.DataFrame, metric: str) -> dict:
    """Trial-pooled level with a flies-as-unit 95 % CI, and the counts behind it."""
    if metric == "rate":
        v, lo, hi = fs.fly_cluster_rate_ci(rows["responded"].astype(bool).tolist(),
                                           rows["fly_id"].tolist())
    else:
        v, lo, hi = fs.fly_cluster_mean_ci(rows["score"].astype(float).tolist(),
                                           rows["fly_id"].tolist())
    return {"value": v, "ci_lo": lo, "ci_hi": hi, "n_flies": int(rows["fly_id"].nunique()),
            "n_trials": int(len(rows)), "n_batches": int(rows["batch"].nunique()),
            "n_sessions": int(rows["session"].nunique())}


def _cluster_sums(vals: np.ndarray, clusters: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    d = pd.DataFrame({"v": vals, "c": clusters.to_numpy()}).groupby("c")["v"]
    return d.sum().to_numpy(float), d.size().to_numpy(float)


def boot_diff_ci(a: pd.DataFrame, b: pd.DataFrame, metric: str, *, unit: str = "session",
                 n_boot: int = N_BOOT, seed: int = SEED) -> tuple[float, float]:
    """95 % percentile CI of pooled(a) - pooled(b), clusters resampled within each group."""
    rng = np.random.default_rng(seed)
    out = []
    for rows in (a, b):
        s, n = _cluster_sums(_values(rows, metric), rows[unit])
        idx = rng.integers(0, len(s), size=(n_boot, len(s)))
        out.append(s[idx].sum(axis=1) / n[idx].sum(axis=1))
    d = out[0] - out[1]
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def compare(a: pd.DataFrame, b: pd.DataFrame, metric: str, *, unit: str = "session",
            n_boot: int = N_BOOT) -> dict:
    """a - b on the trial-pooled metric; permutation p over whole sessions, batch folders
    and flies (two-sided); CI from a ``unit`` bootstrap."""
    va, vb = _values(a, metric), _values(b, metric)
    est = float(va.mean() - vb.mean())
    out = {"estimate": est, "a_value": float(va.mean()), "b_value": float(vb.mean()),
           "n_flies_a": int(a["fly_id"].nunique()), "n_flies_b": int(b["fly_id"].nunique()),
           "n_trials_a": int(len(a)), "n_trials_b": int(len(b)),
           "n_sessions_a": int(a["session"].nunique()), "n_sessions_b": int(b["session"].nunique()),
           "n_batches_a": int(a["batch"].nunique()), "n_batches_b": int(b["batch"].nunique())}
    for cl, key in CLUSTERS.items():
        shared = set(a[cl]) & set(b[cl])
        if shared:
            raise ValueError(f"{cl} in both groups: {sorted(shared)[:3]}")
        labels = {**{c: "a" for c in a[cl]}, **{c: "b" for c in b[cl]}}
        r = bs.batch_label_perm(np.concatenate([va, vb]),
                                np.concatenate([a[cl].to_numpy(), b[cl].to_numpy()]), labels)
        out[key] = r["p"]
        out[f"{key}_exact"] = r["exact"]
    out["ci_lo"], out["ci_hi"] = boot_diff_ci(a, b, metric, unit=unit, n_boot=n_boot)
    return out


def levels_table(trials: pd.DataFrame) -> pd.DataFrame:
    rows = []
    g = groups(trials)
    for name, gr in g.items():
        for odor_set in (POOLED, *ODORS):
            sub = odor_rows(gr, odor_set)
            if sub.empty:
                continue
            for metric in METRICS:
                rows.append({"group": name, "odor_set": odor_set, "metric": metric,
                             **level(sub, metric)})
    return pd.DataFrame(rows)


def tests_table(trials: pd.DataFrame, *, n_boot: int = N_BOOT) -> pd.DataFrame:
    """Sensitivity (all 4) vs each b2b group, pooled odors and each odor; Holm families:
    the 3 b2b groups (pooled) and the odors (per group x metric)."""
    g = groups(trials)
    rows = []
    odors = [o for o in ODORS if (trials["odor"] == o).any()]
    for odor_set in (POOLED, *odors):
        for bg in B2B_GROUPS:
            a, b = odor_rows(g[SENS], odor_set), odor_rows(g[bg], odor_set)
            if a.empty or b.empty:
                continue
            for metric in METRICS:
                rows.append({"a_group": SENS, "b_group": bg, "odor_set": odor_set,
                             "metric": metric, "unit": "pp" if metric == "rate" else "score",
                             **compare(a, b, metric, n_boot=n_boot)})
    t = pd.DataFrame(rows)
    t["p_holm"] = np.nan
    pooled = t["odor_set"] == POOLED
    for _, idx in t[pooled].groupby("metric").groups.items():
        t.loc[idx, "p_holm"] = fs.holm_adjust(t.loc[idx, HEADLINE].tolist())
    for _, idx in t[~pooled].groupby(["metric", "b_group"]).groups.items():
        t.loc[idx, "p_holm"] = fs.holm_adjust(t.loc[idx, HEADLINE].tolist())
    t["holm_family"] = np.where(pooled, "3 b2b groups", "odors")
    return t


# ── figure ───────────────────────────────────────────────────────────────────
def _p(p: float) -> str:
    if not np.isfinite(p):
        return "n/a"
    return "<0.001" if p < 0.001 else f"{p:.3f}" if p < 0.1 else f"{p:.2f}"


def _style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(RULE)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def _bar(ax, x, lv, color, width, *, hollow=False) -> None:
    v, lo, hi = lv["value"], lv["ci_lo"], lv["ci_hi"]
    ax.bar(x, v, width=width, color="white" if hollow else color,
           edgecolor=color if hollow else "white", linewidth=1.4 if hollow else 1.0, zorder=2)
    ax.errorbar(x, v, yerr=[[max(v - lo, 0)], [max(hi - v, 0)]], fmt="none", ecolor=INK,
                elinewidth=1.0, capsize=2.5, zorder=3)


def figure(levels: pd.DataFrame, tests: pd.DataFrame, path_png: Path, *,
           window: str = "09:40-16:20") -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    lv = {(r.group, r.odor_set, r.metric): r._asdict() for r in levels.itertuples(index=False)}
    ts = {(r.b_group, r.odor_set, r.metric): r._asdict() for r in tests.itertuples(index=False)}
    odors = [o for o in ODORS if (BOTH, o, "rate") in lv]
    fig = plt.figure(figsize=(17, 10.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 3.0], wspace=0.13, hspace=0.42,
                          left=0.055, right=0.99, top=0.83, bottom=0.17)
    cohorts = [c for c in SENS_COHORTS if (c, POOLED, "rate") in lv]
    for row, metric in enumerate(METRICS):
        # left: the 7 odors pooled
        ax = fig.add_subplot(gs[row, 0])
        _style(ax)
        xs, labels = [], []
        x = 0.0
        for c in cohorts:
            _bar(ax, x, lv[(c, POOLED, metric)], COHORT_FILL, 0.62)
            xs.append(x)
            labels.append(f"{TICK[c]}\nn={lv[(c, POOLED, metric)]['n_flies']}")
            x += 0.8
        x += 0.35
        main_x = {}
        for gname in (SENS, *B2B_GROUPS):
            _bar(ax, x, lv[(gname, POOLED, metric)], COLORS[gname], 0.62)
            main_x[gname] = x
            xs.append(x)
            labels.append(f"{TICK[gname]}\nn={lv[(gname, POOLED, metric)]['n_flies']}")
            x += 1.15 if gname == SENS else 0.8
        ax.set_xticks(xs, labels, fontsize=8)
        ax.set_xlim(xs[0] - 0.5, xs[-1] + 0.5)
        for text, lo, hi in (("Sensitivity cohorts", xs[0], main_x[SENS]),
                             ("b2b flies", main_x[FIRST], main_x[BOTH])):
            ax.annotate(text, xy=((lo + hi) / 2, 0), xycoords=("data", "axes fraction"),
                        xytext=(0, -34), textcoords="offset points", ha="center", va="top",
                        fontsize=8.5, color=INK2)
        tops = [lv[(gn, POOLED, metric)]["ci_hi"] for gn in (*cohorts, SENS, *B2B_GROUPS)]
        ymax = max(tops)
        step = ymax * 0.17
        y = ymax + step * 0.45
        for gname in B2B_GROUPS:
            t = ts.get((gname, POOLED, metric))
            if t is None:
                continue
            x0, x1 = main_x[SENS], main_x[gname]
            ax.plot([x0, x0, x1, x1], [y - step * 0.2, y, y, y - step * 0.2], color=INK2,
                    lw=0.9)
            d = (f"Δ {t['estimate']:+.1f} pp" if metric == "rate"
                 else f"Δ {t['estimate']:+.2f}")
            ax.text(x0 + 0.05, y + step * 0.07,
                    f"{d}  ·  p {_p(t[HEADLINE])}  ·  Holm {_p(t['p_holm'])}",
                    ha="left", va="bottom", fontsize=7.6, color=INK)
            y += step
        ax.set_ylim(min(0, ax.get_ylim()[0]), y + step * 0.2)
        ax.set_ylabel(METRICS[metric] + "\n(95 % CI, flies)", fontsize=10, color=INK2)
        ax.set_title(("A" if row == 0 else "C") + f"  7 odors pooled (no Paraffin Oil)",
                     loc="left", fontsize=10.5, color=INK, fontweight="bold")

        # right: each odor
        ax = fig.add_subplot(gs[row, 1])
        _style(ax)
        w = 0.19
        offs = {SENS: -1.5 * w, FIRST: -0.5 * w, SECOND: 0.5 * w, BOTH: 1.5 * w}
        top = 0.0
        for i, odor in enumerate(odors):
            if i % 2:
                ax.axvspan(i - 0.5, i + 0.5, color="#f4f3f0", zorder=0, lw=0)
            for gname in (SENS, *B2B_GROUPS):
                key = (gname, odor, metric)
                if key not in lv:
                    continue
                _bar(ax, i + offs[gname], lv[key], COLORS[gname], w * 0.92)
                top = max(top, lv[key]["ci_hi"])
        pad = top * 0.035
        for i, odor in enumerate(odors):
            for gname in B2B_GROUPS:
                t = ts.get((gname, odor, metric))
                key = (gname, odor, metric)
                if t is None or key not in lv:
                    continue
                ax.text(i + offs[gname], lv[key]["ci_hi"] + pad, f"p {_p(t['p_holm'])}",
                        rotation=90, ha="center", va="bottom", fontsize=6.8, color=INK)
        ax.set_xticks(range(len(odors)),
                      [o if o != CONTROL else "Paraffin Oil\n(b2b: click)" for o in odors],
                      fontsize=9)
        ax.set_xlim(-0.55, len(odors) - 0.45)
        ax.set_ylim(min(0, ax.get_ylim()[0]), top * 1.3)
        ax.set_title(("B" if row == 0 else "D") + "  each odor  ·  p over each b2b bar: vs "
                     "Sensitivity (all 4), Holm over the odors", loc="left", fontsize=10.5,
                     color=INK, fontweight="bold")
    n_s = lv[(SENS, POOLED, "rate")]
    n_b = lv[(BOTH, POOLED, "rate")]
    fig.suptitle("Naive (pre-exposure) PER, odor seconds 0-10: the Sensitivity cohorts vs the "
                 f"b2b flies (both started {window})", x=0.055, ha="left", y=0.975,
                 fontsize=14, color=INK, fontweight="bold")
    fig.text(0.055, 0.935,
             f"Sensitivity: {n_s['n_flies']} flies, {n_s['n_sessions']} sessions, Oct 9-Nov 12 "
             "(pre-test odor 30 s, rescored on its first 10 s).  b2b: "
             f"{n_b['n_flies']} flies, {n_b['n_sessions']} sessions, Nov 13-18 (pre-block odor "
             "10 s, production score).  Same vials and concentrations.",
             fontsize=9.5, color=INK2, ha="left")
    fig.legend(handles=[Patch(color=COHORT_FILL, label="each Sensitivity cohort"),
                        *[Patch(color=COLORS[g], label=g.replace(", ", " (") +
                                (")" if g == SENS else "")) for g in (SENS, *B2B_GROUPS)]],
               loc="upper left", bbox_to_anchor=(0.055, 0.915), ncol=5, frameon=False,
               fontsize=9.5)
    fig.text(0.055, 0.025,
             "Bars pool trials (never a mean of per-fly means); whiskers 95 % CI with flies as the "
             "unit. p: two-sided permutation of the group labels over whole sessions (no session "
             "holds both a Sensitivity and a b2b fly);\nHolm over the 3 b2b groups (A, C) or over "
             "the 8 odors (B, D). Batch-folder and fly-level p, the 95 % CI of each difference "
             "(session bootstrap) and every count: naive_tests.csv.\nThe b2b 'Paraffin Oil' trial "
             "is a 50 ms master-valve click with no oil (since 2026-11-10), so it is left out of "
             "A and C; IAA-Sensitivity had no Paraffin Oil pre-test. Δ = Sensitivity - b2b.",
             fontsize=8.4, color=INK2, ha="left", va="bottom")
    paths = []
    for ext in (".png", ".pdf"):
        p = path_png.with_suffix(ext)
        fig.savefig(p, dpi=200 if ext == ".png" else None)
        paths.append(p)
    plt.close(fig)
    return paths


# ── README ───────────────────────────────────────────────────────────────────
def readme(out: Path, levels: pd.DataFrame, tests: pd.DataFrame, trials: pd.DataFrame) -> Path:
    L = ["# Naive PER: Sensitivity cohorts vs b2b flies (odor seconds 0-10)", "",
         "Built by `scripts/analysis/naive_sensitivity_vs_b2b.py` (hand-run). Inputs:",
         f"- Sensitivity: `{SENS_CSV}` (pre-test, first-10-s rescore, flies started 09:40-16:20)",
         f"- b2b: `{B2B_CSV}` (the b2b day folder's naive pre-block, production score)", "",
         "## Plan (fixed before the comparison was run)", ""]
    L += [f"- **{k}**: {v}" for k, v in PLAN.items()]
    L += ["", "## Headline: the 7 odors pooled", "",
          "| metric | b2b group | Sensitivity | b2b | Sensitivity − b2b [95 % CI] | p (whole sessions) | Holm | p batch | p fly |",
          "|---|---|---|---|---|---|---|---|---|"]
    for _, r in tests[tests["odor_set"] == POOLED].iterrows():
        f = (lambda v: f"{v:.1f}") if r["metric"] == "rate" else (lambda v: f"{v:.2f}")
        L.append(f"| {METRICS[r['metric']]} | {r['b_group']} ({r['n_flies_b']} flies, "
                 f"{r['n_sessions_b']} sessions) | {f(r['a_value'])} | {f(r['b_value'])} | "
                 f"**{r['estimate']:+.2f}** [{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}] | "
                 f"{_p(r['p_session'])} | {_p(r['p_holm'])} | {_p(r['p_batch'])} | {_p(r['p_fly'])} |")
    sig = tests[(tests["p_holm"] < 0.05)]
    L += ["", "Each odor (Holm over the odors): "
          + ("; ".join(f"{r.odor_set} {r.metric} vs {r.b_group}: {r.estimate:+.2f}, Holm "
                       f"{_p(r.p_holm)}" for r in sig[sig.odor_set != POOLED].itertuples())
             or "no odor clears Holm p < 0.05.")]
    L += ["", "## Read with care", "",
          "- **Group = date.** The Sensitivity flies ran Oct 9-Nov 12 and the b2b flies Nov "
          "13-18; no session holds both, so a difference can be the cohort, the season, the fly "
          "stock or the protocol, not only the b2b design. That is why whole sessions are the "
          "unit; the batch-folder and fly p treat folders or flies of one day as independent and "
          "overstate certainty.",
          "- **Odor length.** The Sensitivity pre-test odor runs 30 s and is cut to its first "
          "10 s for scoring (the rest spliced out, so the model's after-odor window is the "
          "trace after the full 30 s); the b2b odor is 10 s, its after-window starts at 10 s.",
          "- **Paraffin Oil.** b2b: a master-valve click, no oil. Sensitivity: paraffin oil "
          "(some flies the paraffin-off protocol). Shown per odor, kept out of the pooled set.",
          "- **Starvation.** Both about 1.5 h at the first pre-test trial (b2b 1.5-2.5 h, "
          "Sensitivity 1.5-1.7 h).",
          "- **Session layout.** The Sensitivity pre-test is followed by exposure on the same "
          "flies; the b2b pre-block by the odor pairs. Only trials before any exposure are used "
          "on both sides.", ""]
    n = trials.groupby("group")["fly_id"].nunique()
    L += ["## Counts", "", "| group | flies | trials | batch folders | sessions |", "|---|---|---|---|---|"]
    for gname, gr in groups(trials).items():
        L.append(f"| {gname} | {gr['fly_id'].nunique()} | {len(gr)} | {gr['batch'].nunique()} | "
                 f"{gr['session'].nunique()} |")
    L += ["", "Files: `naive_sensitivity_vs_b2b.png/.pdf`, `naive_levels.csv` (every group x odor "
          "x metric with its 95 % CI and counts), `naive_tests.csv` (every test), "
          "`naive_trials.csv` (the trials used).", ""]
    p = out / "README.md"
    p.write_text("\n".join(L))
    return p


def build(sens_csv: Path = SENS_CSV, b2b_csv: Path = B2B_CSV, out: Path = OUT, *,
          threshold: float, n_boot: int = N_BOOT) -> list[Path]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    trials = load_trials(sens_csv, b2b_csv, threshold=threshold)
    levels = levels_table(trials)
    tests = tests_table(trials, n_boot=n_boot)
    written = []
    for name, df in (("naive_levels.csv", levels), ("naive_tests.csv", tests),
                     ("naive_trials.csv", trials)):
        df.to_csv(out / name, index=False)
        written.append(out / name)
    written += figure(levels, tests, out / "naive_sensitivity_vs_b2b.png")
    written.append(readme(out, levels, tests, trials))
    return written


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sensitivity-csv", type=Path, default=SENS_CSV)
    ap.add_argument("--b2b-csv", type=Path, default=B2B_CSV)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--config", type=Path, default=CONFIG)
    a = ap.parse_args(argv)
    written = build(a.sensitivity_csv, a.b2b_csv, a.out, threshold=binary_threshold(a.config))
    print(f"[naive] wrote {len(written)} files under {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
