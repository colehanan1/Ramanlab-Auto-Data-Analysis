#!/usr/bin/env python3
"""The sensitized odor at every trial of the session, one cohort (user 2026-11-17).

Eight trials of the odor the cohort was exposed to: the pre-exposure test, the
six exposure trials and the post-exposure test. Per trial, for every fly:

  score            the model score (responded = score >= 2), as on the score bars
  ratio_raw        AUC during / before, area under the raw trace (the AUC folder's
                   raw-area ratio)
  ratio_unsigned   AUC during / before, |deviation from the pre-odor median|
  zpeak_ratio_signed    peak z during, sign kept / |peak z| before
  zpeak_ratio_unsigned  |peak z| during / |peak z| before

The AUC and peak-z values come from ``auc_before_during.trial_auc`` /
``trial_peak_z`` on the trace the AUC and PeakZ folders use, so a pre- or
post-exposure value here is the value there. Exposure trials are odor-only (no
light), with the same 30 s odor, so all eight are scored alike.

Each figure keeps only the flies with every trial, so each bar is the same flies
and every comparison is paired: Friedman across all eight, then Wilcoxon (McNemar
for % responding) on every pair, Holm-corrected within a measure (all exploratory). A fly with a
repeated exposure trial (an aborted session) is left out.

Usage (run by start_window_report.py on a cohort's window inputs):
    python scripts/analysis/sensitized_odor_by_trial.py --dataset IAA-Sensitivity-1-1 \\
        --pretest-wide-csv .../_inputs/wide_pretest.parquet \\
        --training-wide-csv .../_inputs/wide_training.parquet \\
        --testing-wide-csv .../_inputs/wide_testing.parquet \\
        --predictions-csv .../_inputs/predictions.csv \\
        --training-predictions-csv .../_inputs/training_predictions.csv \\
        --out-dir .../Sensitized-Odor-By-Trial
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from itertools import combinations
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT, REPO_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from fbpipe.analysis import fly_stats as fs  # noqa: E402
import scripts.analysis.auc_before_during as abd  # noqa: E402
import scripts.analysis.sensitivity_window_results as swres  # noqa: E402
from scripts.analysis import envelope_visuals as ev  # noqa: E402
from scripts.analysis import pretest_vs_test_traces as pvt  # noqa: E402
from scripts.analysis.phase_terms import EXPOSURE, POST_EXPOSURE, PRE_EXPOSURE  # noqa: E402

OUT_DIR = "Sensitized-Odor-By-Trial"
THRESHOLD = swres.THRESHOLD
TRIALS = [PRE_EXPOSURE, *[f"{EXPOSURE.capitalize()} {n}" for n in range(1, 7)], POST_EXPOSURE]

#: measure -> (y label, value format, null value drawn as a line, or None when the
#: measure has no fixed no-change value: see auc_before_during.NO_FIXED_NULL)
MEASURES = {
    "ratio_raw": ("AUC during / before\n(raw area)", "{:.2f}", 1.0),
    "ratio_unsigned": ("AUC during / before\n(unsigned)", "{:.2f}", None),
    "zpeak_ratio_signed": ("Peak z during, signed\n/ |peak z| before", "{:.2f}", None),
    "zpeak_ratio_unsigned": ("Peak z during / before\n(unsigned)", "{:.2f}", None),
}
FIGURES = {  # file stem -> (measures, title, what the rows are)
    "auc_ratio": (("ratio_raw", "ratio_unsigned"), "AUC during / before",
                  "Raw: area under the trace during the odor / the 30 s before (the AUC "
                  "folder's raw-area ratio). Unsigned: |trace − pre-odor median|, during / "
                  "before. Raw: 1 = no change; unsigned has no fixed no-change value."),
    "peakz_ratio": (("zpeak_ratio_signed", "zpeak_ratio_unsigned"),
                    "Peak z-score during / before, signed and unsigned",
                    "z against the 30 s before the odor. Signed: peak z during with its sign "
                    "/ |peak z| before (a dip reads negative). Unsigned: |peak "
                    "z| during / |peak z| before. Neither has a fixed no-change value."),
}
COLUMNS = ["dataset", "fly_id", "fly", "fly_number", "trial", "trial_index", "trial_label",
           "odor", "score", "responded", *MEASURES]


def sensitized_odor(training: pd.DataFrame) -> str:
    """The one odor of the exposure trials."""
    d, _ = pvt._prepare(training, "training")
    odors = sorted(d["odor_base"].dropna().unique())
    if len(odors) != 1:
        raise ValueError(f"exposure trials carry {len(odors)} odors: {odors}")
    return odors[0]


def _phase_rows(df: pd.DataFrame, phase: str, odor: str) -> tuple[pd.DataFrame, list[str]]:
    d, cols = pvt._prepare(df, phase)
    d = d[d["odor_base"] == odor].copy()
    d["_n"] = d["trial_label"].map(ev._trial_num)
    return d, cols


def per_trial_table(pretest: pd.DataFrame, training: pd.DataFrame, testing: pd.DataFrame,
                    predictions: pd.DataFrame) -> pd.DataFrame:
    """One row per fly x trial of the sensitized odor, flies with a clean session."""
    odor = sensitized_odor(training)
    scores = _scores(predictions)
    frames = []
    for phase, df in (("pretest", pretest), ("training", training), ("testing", testing)):
        d, cols = _phase_rows(df, phase, odor)
        d = d.sort_values("_n", kind="stable")
        key = ["fly", "fly_number"] + (["_n"] if phase == "training" else [])
        dup = d.duplicated(key, keep=False)
        if phase == "training":
            # an exposure trial run twice: the session was restarted, the order is unknown
            bad = d.loc[dup, ["fly", "fly_number"]].drop_duplicates()
            d = d.merge(bad, how="left", indicator=True)
            d = d[d["_merge"] == "left_only"].drop(columns="_merge")
        else:
            d = d.drop_duplicates(key, keep="first")
        for _, row in d.iterrows():
            trace = pvt._to_trace(row, cols, scores)
            if trace is None:
                continue
            on = trace.odor_on_s if trace.odor_on_s is not None else float("nan")
            off = trace.odor_off_s if trace.odor_off_s is not None else float("nan")
            timing = dict(fps=trace.fps, odor_on_s=on, odor_off_s=off)
            auc, z = abd.trial_auc(trace.env, **timing), abd.trial_peak_z(trace.env, **timing)
            if phase == "pretest":
                trial = PRE_EXPOSURE
            elif phase == "testing":
                trial = POST_EXPOSURE
            elif 1 <= int(row["_n"]) <= 6:
                trial = f"{EXPOSURE.capitalize()} {int(row['_n'])}"
            else:
                continue
            frames.append({
                "dataset": str(row["dataset"]), "fly": str(row["fly"]),
                "fly_number": int(row["fly_number"]),
                "fly_id": f"{row['fly']}_fly{int(row['fly_number'])}",
                "trial": trial, "trial_index": TRIALS.index(trial),
                "trial_label": str(row["trial_label"]), "odor": odor,
                "score": trace.score,
                **{m: (auc if m in auc else z).get(m, np.nan) for m in MEASURES},
            })
    t = pd.DataFrame(frames, columns=[c for c in COLUMNS if c != "responded"])
    if t.empty:
        return pd.DataFrame(columns=COLUMNS)
    t["score"] = pd.to_numeric(t["score"], errors="coerce")
    t["responded"] = t["score"] >= THRESHOLD
    flies = t.groupby("fly_id")["trial"].nunique()
    t = t[t["fly_id"].isin(flies[flies == len(TRIALS)].index)]
    return t[COLUMNS].sort_values(["fly_id", "trial_index"], ignore_index=True)


def _scores(predictions: pd.DataFrame) -> dict:
    path_free = predictions.copy()
    path_free = ev._normalise_fly_columns(path_free)
    path_free["score"] = pd.to_numeric(path_free["score"], errors="coerce")
    path_free = path_free[path_free["score"].notna()]
    return {(ev._canon_dataset(str(ds)), str(fly), str(num), str(label)): int(s)
            for ds, fly, num, label, s in path_free[
                ["dataset", "fly", "fly_number", "trial_label", "score"]
            ].itertuples(index=False, name=None)}


def series(table: pd.DataFrame, measure: str) -> pd.DataFrame:
    """Flies x the eight trials for one measure, flies with every value only."""
    w = table.pivot(index="fly_id", columns="trial", values=measure)
    w = w.reindex(columns=TRIALS).astype(float)
    return w[np.isfinite(w).all(axis=1)]


def series_summary(wide: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for trial in wide.columns:
        v = wide[trial].to_numpy(float)
        q25, med, q75 = (np.percentile(v, [25, 50, 75]) if v.size else (np.nan,) * 3)
        rows.append({"trial": trial, "n_flies": int(v.size),
                     "mean": float(v.mean()) if v.size else np.nan,
                     "sem": swres._sem(pd.Series(v)) if v.size else np.nan,
                     "median": float(med), "q25": float(q25), "q75": float(q75)})
    out = pd.DataFrame(rows)
    out[["mean", "sem", "median", "q25", "q75"]] = out[
        ["mean", "sem", "median", "q25", "q75"]].round(4)
    return out


def series_tests(wide: pd.DataFrame) -> pd.DataFrame:
    """Friedman across all trials, then Wilcoxon on every pair, Holm within the measure."""
    from scipy.stats import friedmanchisquare

    n = len(wide)
    try:
        stat, p = friedmanchisquare(*[wide[c].to_numpy() for c in wide.columns])
    except ValueError:
        stat, p = np.nan, np.nan
    rows = [{"test": "Friedman", "trial_a": "all", "trial_b": "all", "n_flies": n,
             "statistic": round(float(stat), 4), "p": float(p), "p_holm": float(p)}]
    pairs = list(combinations(wide.columns, 2))
    # The score panel's minimum (pretest_vs_test_comparison.wilcoxon_p): below 6 non-zero
    # differences a pair is not tested (two-sided exact p could not reach 0.05 anyway)
    # and stays out of the Holm family, as it does there.
    raw = [fs.wilcoxon_signed_rank_p(wide[b].to_numpy(float) - wide[a].to_numpy(float),
                                     min_pairs=swres.pvt.MIN_WILCOXON_PAIRS)
           for a, b in pairs]
    raw = [np.nan if v is None else float(v) for v in raw]
    for (a, b), p_, adj in zip(pairs, raw, fs.holm_adjust(raw)):
        rows.append({"test": "Wilcoxon signed-rank", "trial_a": a, "trial_b": b, "n_flies": n,
                     "statistic": np.nan, "p": p_, "p_holm": float(adj)})
    out = pd.DataFrame(rows)
    out[["p", "p_holm"]] = out[["p", "p_holm"]].astype(float).round(6)
    return out


def plot_measures(wides: dict, summaries: dict, tests: dict, measures: Sequence[str],
                  title: str, note: str, path: Path) -> None:
    """A row per measure: mean ± SEM bars with the median marked, beside it every pair."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    rows = len(measures)
    fig = plt.figure(figsize=(14.5, 4.5 * rows + 1.75))
    grid = fig.add_gridspec(rows, 2, width_ratios=[1.45, 1.0], hspace=0.45, wspace=0.14)
    x = np.arange(len(TRIALS))
    colours = [swres._trial_colour(t) for t in TRIALS]
    for r, m in enumerate(measures):
        ylab, fmt, null = MEASURES[m]
        s, t = summaries[m], tests[m]
        ax = fig.add_subplot(grid[r, 0])
        y, e = s["mean"].to_numpy(), s["sem"].to_numpy()
        med = s["median"].to_numpy()
        ax.bar(x, y, width=0.78, color=colours, zorder=2)
        ax.errorbar(x, y, yerr=e, fmt="none", ecolor=swres.INK_2, elinewidth=1.0, zorder=3)
        ax.plot(x, med, ls="none", marker="D", ms=6, color=swres.INK, mec="white", mew=0.8,
                zorder=4)
        if null is not None:
            ax.axhline(null, color=swres.INK_2, lw=1.0, ls=(0, (3, 2)), zorder=1)
        vs_pre = {row.trial_b: row.p_holm for row in t.itertuples() if row.trial_a == TRIALS[0]}
        tops = np.maximum(y + e, med)
        for i, trial in enumerate(TRIALS):
            ax.annotate(fmt.format(y[i]) + swres._stars(vs_pre.get(trial)), (i, tops[i]),
                        xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
                        fontsize=9.2, color=swres.INK)
        ax.set_xticks(x, [tr.replace(" ", "\n", 1) if tr[:3] == "Exp" else tr.replace("-", "-\n", 1)
                          for tr in TRIALS], fontsize=9.5, color=swres.INK)
        ax.set_xlim(-0.6, len(TRIALS) - 0.4)
        lo = min(0.0, float(np.nanmin(np.minimum(y - e, med))))
        hi = float(np.nanmax(tops))
        ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.16 * (hi - lo))
        ax.set_ylabel(ylab, fontsize=10.5, color=swres.INK)
        ax.yaxis.grid(True, color=swres.GRID, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=swres.INK_2)
        omni = t[t["trial_a"] == "all"].iloc[0]
        ax.set_title(f"n={len(wides[m])} flies with all 8 trials · Friedman across all 8: "
                     f"{swres._p_text(omni['p'])} (exploratory)", loc="left", fontsize=10.5, color=swres.INK)
        swres._p_triangle(fig.add_subplot(grid[r, 1]), TRIALS, t[t["trial_a"] != "all"],
                          "Wilcoxon signed-rank, Holm p")
    h = fig.get_figheight()
    fig.legend(handles=[Line2D([], [], ls="none", marker="D", ms=6, color=swres.INK,
                               mec="white", label="median over flies"),
                        Patch(color=swres.P_TIER_COLOURS[2], label="p < 0.05 after Holm"),
                        Patch(color=swres.P_TIER_COLOURS[1], label="p < 0.05 uncorrected only"),
                        Patch(color=swres.P_TIER_COLOURS[0], label="not significant")],
               loc="upper center", bbox_to_anchor=(0.5, 1 - 1.12 / h), ncol=4, frameon=False,
               fontsize=9.5, labelcolor=swres.INK)
    fig.suptitle(title, fontsize=13.5, color=swres.INK, y=1 - 0.1 / h)
    caption = (f"{note} The same flies in every bar of a row. Bars: mean ± 1 SEM over flies; "
               "diamond: median. Stars: differs from pre-exposure after Holm correction for all "
               "28 pairs (* p<0.05, ** p<0.01, *** p<0.001). Exposure trials are odor-only, the "
               "same 30 s odor as the tests. Ratios are heavy-tailed: a few flies can carry the "
               "mean, so read it against the median; the tests use ranks. All tests here are "
               "exploratory (Friedman omnibus uncorrected; pairs exact tie-aware Wilcoxon, "
               "Holm-adjusted within each measure)."
               + "".join(" " + abd.no_null_sentence(m) for m in measures
                         if MEASURES[m][2] is None))
    fig.text(0.5, 1 - 0.42 / h, textwrap.fill(caption, 190), ha="center", va="top",
             fontsize=8.8, color=swres.INK_2)
    fig.subplots_adjust(top=1 - 1.75 / h, bottom=0.55 / h, left=0.07, right=0.985)
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--pretest-wide-csv", required=True, type=Path)
    p.add_argument("--training-wide-csv", required=True, type=Path)
    p.add_argument("--testing-wide-csv", required=True, type=Path)
    p.add_argument("--predictions-csv", required=True, type=Path)
    p.add_argument("--training-predictions-csv", required=True, type=Path)
    p.add_argument("--dataset", required=True)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--label", default="", help="which flies, for the titles")
    return p


def _read(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    for old in list(out.glob("*.png")) + list(out.glob("*.csv")):
        old.unlink()                                 # this folder is redrawn whole
    preds = pd.concat([pd.read_csv(args.predictions_csv),
                       pd.read_csv(args.training_predictions_csv)], ignore_index=True)
    table = per_trial_table(_read(args.pretest_wide_csv), _read(args.training_wide_csv),
                            _read(args.testing_wide_csv), preds)
    table = table[table["dataset"].map(ev._canon_dataset) == ev._canon_dataset(args.dataset)]
    if table.empty:
        print(f"[by-trial] {args.dataset}: no fly with all 8 trials of the sensitized odor")
        return 0
    ds, odor = args.dataset, table["odor"].iloc[0]
    who = f"{ds}, {args.label}" if args.label else ds
    table.to_csv(out / f"{ds}_by_trial_per_fly.csv", index=False)

    summaries, stats = [], []
    # score and % responding: the Trial-By-Trial figure, on exactly these flies
    score = series(table, "score").astype(int)
    s_sum, s_tests = swres.trial_series_summary(score), swres.trial_series_tests(score)
    swres.plot_trial_series(s_sum, s_tests, f"{odor} across the session, model score: {who} "
                            f"(n={len(score)} flies)", out / f"{ds}_score_by_trial.png")
    summaries.append(s_sum.assign(measure="score"))
    stats.append(s_tests.rename(columns={"measure": "panel"}).assign(measure="score"))

    for stem, (measures, title, note) in FIGURES.items():
        wides = {m: series(table, m) for m in measures}
        sums = {m: series_summary(w) for m, w in wides.items()}
        tests = {m: series_tests(w) for m, w in wides.items()}
        plot_measures(wides, sums, tests, measures, f"{odor} across the session, {title}: "
                      f"{who}", note, out / f"{ds}_{stem}_by_trial.png")
        summaries += [s.assign(measure=m) for m, s in sums.items()]
        stats += [t.assign(measure=m) for m, t in tests.items()]
    pd.concat(summaries, ignore_index=True).to_csv(out / f"{ds}_by_trial_summary.csv",
                                                   index=False)
    pd.concat(stats, ignore_index=True).to_csv(out / f"{ds}_by_trial_stats.csv", index=False)
    print(f"[by-trial] {ds}: {table['fly_id'].nunique()} flies, {odor} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
