#!/usr/bin/env python3
"""Naive (pre-test) PER of the b2b flies: day window vs evening window.

User 2026-11-20: "compare the naive response from window in day vs evening window ... to see
if flies respond more or less during evening window or morning window ... combine the two
datasets". Hand-run, not wired into the pipeline.

Input: each window folder's ``pretest/pretest_naive_trials.csv`` (b2b_pair_figures: one row
per fly x odor of the naive pre-test, every odor alone, before any pair; frozen folders and
flagged flies already left out). IAA-PairFirst and IAA-PairSecond flies are pooled: the
pre-test is the same protocol for both.

* Levels pool trials (responses or scores / trials, never a mean of fly means); intervals
  take the fly as the unit (fs.fly_cluster_rate_ci / fly_cluster_mean_ci).
* "7 odors pooled" leaves out the Paraffin Oil partner, a master-valve click with no odor.
* Tests, evening minus day, two-sided: the window is a property of the whole session (one
  date's batch, its rig folders started together), so whole sessions are permuted between
  the windows (bt.label_perm; exact here); batch-folder and fly permutations and a session
  bootstrap 95 % CI are in the CSV. Holm over the 8 odors per metric; the pooled test per
  metric is the headline.

Output: ``Results/Figures/b2b_Naive_Day-vs-Evening/`` (png, pdf, levels and tests CSV).
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

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fbpipe.analysis import fly_stats as fs  # noqa: E402
from scripts.analysis import b2b_baseline_time_of_day as bt  # noqa: E402
from scripts.analysis import b2b_stats as bs  # noqa: E402
from scripts.analysis import naive_sensitivity_vs_b2b as nv  # noqa: E402

RESULTS = Path("/home/ramanlab/Documents/cole/Results")
DAY, EVENING = "Day (started 9:40am–4:20pm)", "Evening (started 4:40pm–9:20pm)"
INPUTS = {DAY: RESULTS / "b2b" / "pretest" / "pretest_naive_trials.csv",
          EVENING: RESULTS / "b2b-Evening" / "pretest" / "pretest_naive_trials.csv"}
OUT = RESULTS / "Figures" / "b2b_Naive_Day-vs-Evening"
CONTROL, POOLED, METRICS = nv.CONTROL, nv.POOLED, nv.METRICS
ODORS = ("IAA", "Hexanol", "Linalool", "Benzaldehyde", "Ethyl Butyrate", "Citral",
         "3-Octanol", CONTROL)
#: dataviz validator, light mode: every check passes (CVD ΔE 30.7, both >= 3:1).
COLORS = {DAY: "#b8860b", EVENING: "#5157c4"}
INK, INK2, RULE, GRID = nv.INK, nv.INK2, nv.RULE, nv.GRID
SHORT = {DAY: "day", EVENING: "evening"}


def load_trials(paths=None, *, threshold: float) -> pd.DataFrame:
    """One row per naive trial of either window: window, dataset, fly_id, batch (the
    batch folder), session, odor, score, responded. A folder in both windows is an error."""
    parts = []
    for window, path in (paths or INPUTS).items():
        t = pd.read_csv(path)
        t = t[pd.to_numeric(t["score"], errors="coerce").notna()]
        parts.append(pd.DataFrame({
            "window": window, "dataset": t["dataset"].astype(str),
            "fly_id": (t["dataset"].astype(str) + "|" + t["fly"].astype(str) + "|"
                       + t["fly_number"].astype(int).astype(str)),
            "batch": t["fly"].astype(str),
            "odor": t["odor"].replace({"Isoamyl Acetate": "IAA"}).astype(str),
            "score": t["score"].astype(float)}))
    out = pd.concat(parts, ignore_index=True)
    split = out.groupby("batch")["window"].nunique()
    if (split > 1).any():
        raise ValueError(f"batch folders in both windows: {sorted(split[split > 1].index)}")
    out["session"] = out["batch"].map(bs.session_of)
    out["responded"] = out["score"] >= threshold
    return out


def odor_sets(trials: pd.DataFrame) -> list[str]:
    return [POOLED] + [o for o in ODORS if (trials["odor"] == o).any()]


def levels_table(trials: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for window in (DAY, EVENING):
        w = trials[trials["window"] == window]
        for odor_set in odor_sets(trials):
            sub = nv.odor_rows(w, odor_set)
            if sub.empty:
                continue
            for metric in METRICS:
                rows.append({"window": window, "odor_set": odor_set, "metric": metric,
                             **nv.level(sub, metric)})
    return pd.DataFrame(rows)


def compare(eve: pd.DataFrame, day: pd.DataFrame, metric: str, *, n_boot: int) -> dict:
    """Evening minus day on the trial-pooled metric; p with whole sessions, batch folders
    and flies permuted between the windows; session-bootstrap 95 % CI."""
    va, vb = nv._values(eve, metric), nv._values(day, metric)
    out = {"estimate": float(va.mean() - vb.mean()), "evening": float(va.mean()),
           "day": float(vb.mean())}
    for name, rows in (("evening", eve), ("day", day)):
        out.update({f"n_flies_{name}": int(rows["fly_id"].nunique()),
                    f"n_trials_{name}": int(len(rows)),
                    f"n_batches_{name}": int(rows["batch"].nunique()),
                    f"n_sessions_{name}": int(rows["session"].nunique())})
    both = pd.concat([eve, day], ignore_index=True)
    vals = np.concatenate([va, vb])
    for cl, key in nv.CLUSTERS.items():
        shared = set(eve[cl]) & set(day[cl])
        if shared:
            raise ValueError(f"{cl} in both windows: {sorted(shared)[:3]}")
        r = bt.label_perm(vals, both[cl], dict(zip(both[cl], both["window"])))
        out[key], out[f"{key}_exact"] = r["p"], bool(r["exact"])
    out["ci_lo"], out["ci_hi"] = nv.boot_diff_ci(eve, day, metric, unit="session",
                                                 n_boot=n_boot)
    return out


def tests_table(trials: pd.DataFrame, *, n_boot: int = nv.N_BOOT) -> pd.DataFrame:
    rows = []
    eve, day = trials[trials["window"] == EVENING], trials[trials["window"] == DAY]
    for odor_set in odor_sets(trials):
        a, b = nv.odor_rows(eve, odor_set), nv.odor_rows(day, odor_set)
        if a.empty or b.empty:
            continue
        for metric in METRICS:
            rows.append({"odor_set": odor_set, "metric": metric,
                         "unit": "pp" if metric == "rate" else "score",
                         **compare(a, b, metric, n_boot=n_boot)})
    t = pd.DataFrame(rows)
    pooled = t["odor_set"] == POOLED
    t["p_holm"] = t["p_session"]
    for _, idx in t[~pooled].groupby("metric").groups.items():
        t.loc[idx, "p_holm"] = fs.holm_adjust(t.loc[idx, "p_session"].tolist())
    t["holm_family"] = np.where(pooled, "headline (one test per metric)", "the odors, per metric")
    return t


# ── figure ───────────────────────────────────────────────────────────────────
def _tick(odor_set: str) -> str:
    if odor_set == POOLED:
        return "7 odors\npooled"
    return f"{CONTROL}\n(click)" if odor_set == CONTROL else odor_set.replace(" ", "\n")


def figure(levels: pd.DataFrame, tests: pd.DataFrame, path: Path) -> list[Path]:
    from matplotlib.patches import Patch

    sets = [s for s in [POOLED, *ODORS] if s in set(levels["odor_set"])]
    xs = {s: (0.0 if s == POOLED else 1.5 + i) for i, s in enumerate(sets[1:])}
    xs[POOLED] = 0.0
    w = 0.36
    lv = levels.set_index(["window", "odor_set", "metric"])
    st = tests.set_index(["odor_set", "metric"])
    n_odors = int((tests["odor_set"] != POOLED).sum() / max(tests["metric"].nunique(), 1))
    with plt.rc_context({"font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
                         "pdf.fonttype": 42, "font.size": 9}):
        fig, axes = plt.subplots(2, 1, figsize=(8.8, 7.0), sharex=False)
        for ax, metric, ylab in ((axes[0], "score", "Mean PER score"),
                                 (axes[1], "rate", "Responding trials (%)\nscore ≥ 2")):
            nv._style(ax)
            ax.axvspan(-0.55, 0.55, color="#f4f3f0", zorder=0, lw=0)
            tops = []
            for s in sets:
                for j, window in enumerate((DAY, EVENING)):
                    if (window, s, metric) not in lv.index:
                        continue
                    nv._bar(ax, xs[s] + (j - 0.5) * (w + 0.02), lv.loc[(window, s, metric)],
                            COLORS[window], w)
                tops.append(max(float(lv.loc[(wd, s, metric), "ci_hi"])
                                for wd in (DAY, EVENING) if (wd, s, metric) in lv.index))
            lo = min(0.0, float(levels.loc[levels["metric"] == metric, "ci_lo"].min()))
            hi = max(tops)
            unit = 0.05 * (hi - lo)
            for s, top in zip(sets, tops):
                if (s, metric) not in st.index:
                    continue
                r = st.loc[(s, metric)]
                star = fs.stars(float(r["p_holm"])).replace("n.s.", "")
                x0, x1 = xs[s] - 0.5 * (w + 0.02), xs[s] + 0.5 * (w + 0.02)
                y = top + 0.8 * unit
                ax.plot([x0, x0, x1, x1], [y - 0.4 * unit, y, y, y - 0.4 * unit], color=INK,
                        lw=0.7)
                txt = f"p={nv._p(float(r['p_session']))}{star}"
                if s == POOLED:
                    d = float(r["estimate"])
                    txt = (f"{d:+.2f}" if metric == "score" else f"{d:+.1f} pp") + f"\n{txt}"
                ax.text(xs[s], y + 0.2 * unit, txt, ha="center", va="bottom", fontsize=7.2,
                        color=INK, fontweight="bold" if star else "normal")
            ax.set_ylim(lo - (unit if lo < 0 else 0), hi + 5.0 * unit)
            ax.axhline(0, color=RULE, lw=0.8, zorder=1)
            ax.axvline(0.95, color=INK2, lw=0.7, ls=(0, (3, 3)))
            ax.set_xlim(-0.6, xs[sets[-1]] + 0.6)
            ax.set_xticks([xs[s] for s in sets])
            ax.set_xticklabels([_tick(s) for s in sets] if ax is axes[1] else [])
            ax.tick_params(axis="x", labelsize=8.5, labelcolor=INK)
            ax.set_ylabel(ylab, color=INK)
        for ax, letter in zip(axes, "AB"):
            ax.text(-0.08, 1.02, letter, transform=ax.transAxes, fontsize=12,
                    fontweight="bold", color=INK, va="bottom")
        n = lv.xs(POOLED, level="odor_set").xs("score", level="metric")
        fig.text(0.08, 0.985, "Naive pre-test (every odor alone, before any pair): day vs "
                 "evening flies, IAA-first and IAA-second pooled", fontsize=10.5,
                 fontweight="bold", color=INK, va="top")
        fig.text(0.08, 0.952, " · ".join(
            f"{SHORT[wd].capitalize()}: {int(n.loc[wd, 'n_flies'])} flies, "
            f"{int(n.loc[wd, 'n_batches'])} batch folders, {int(n.loc[wd, 'n_sessions'])} "
            f"sessions" for wd in (DAY, EVENING) if wd in n.index)
            + " · bars pool trials, whiskers 95 % CI (flies)", fontsize=8.3, color=INK2,
            va="top")
        fig.legend(handles=[Patch(color=COLORS[wd], label=wd) for wd in (DAY, EVENING)],
                   loc="upper left", bbox_to_anchor=(0.075, 0.935), ncol=2, frameon=False,
                   fontsize=8.5)
        foot = ("Evening minus day; p: whole sessions permuted between the windows (exact; "
                "every session lies in one window), two-sided. Shaded: the 7 odors pooled "
                "(the headline; Paraffin Oil, a master-valve click, left out), its difference "
                "printed. Per odor: raw p; * p < 0.05, ** p < 0.01 only after Holm over the "
                f"{n_odors} odors of the panel (exploratory). Batch-folder and fly-level p and "
                "a session-bootstrap 95 % CI of each difference: naive_day_vs_evening_tests.csv "
                "(that CI is too narrow with only 3 evening sessions: go by the session p). "
                "Window = session: day and evening flies differ in date and vial as well as "
                "clock time.")
        import textwrap
        fig.text(0.08, 0.01, textwrap.fill(foot, 150), fontsize=6.8, color=INK2, va="bottom")
        fig.subplots_adjust(left=0.08, right=0.99, top=0.865, bottom=0.17, hspace=0.12)
        path.parent.mkdir(parents=True, exist_ok=True)
        pdf = path.with_suffix(".pdf")
        fig.savefig(path, dpi=300, facecolor="white")
        fig.savefig(pdf, facecolor="white")
        plt.close(fig)
    return [path, pdf]


def build(paths=None, out: Path = OUT, *, threshold: float, n_boot: int = nv.N_BOOT) -> list[Path]:
    trials = load_trials(paths, threshold=threshold)
    levels, tests = levels_table(trials), tests_table(trials, n_boot=n_boot)
    out.mkdir(parents=True, exist_ok=True)
    written = figure(levels, tests, out / "naive_day_vs_evening.png")
    for name, df in (("naive_day_vs_evening_levels.csv", levels),
                     ("naive_day_vs_evening_tests.csv", tests)):
        df.to_csv(out / name, index=False)
        written.append(out / name)
    return written


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--config", type=Path, default=nv.CONFIG)
    a = ap.parse_args(argv)
    for p in build(out=a.out, threshold=nv.binary_threshold(a.config)):
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
