#!/usr/bin/env python3
"""PER change and PER response rate as a function of the time of day the experiment ran.

The time-of-day twin of ``starvation_trend_figure.py``: same measures, same
emphasis style, but the x axis is the clock time of each fly's FIRST pre-test
trial instead of its starvation group. The time comes from the raw trial file
names in the batch folder (``output_<batch>_pretest_<n>_<odor>_YYYYMMDD_HHMMSS.csv``,
Pi local time) -- the earliest pre-test file is the start of the experiment.

Flies are binned by that start time, once per ``--bin-hours`` width (default 1 h
and 2 h, each in its own sub-folder). Per bin and odor, pooled over flies (one
pre and one post trial per fly per odor):

* the NAIVE pre-test score and response rate (``time_of_day_pretest.png``,
  ``pretest_stats.csv``) -- does baseline PER differ by time of day? Pre-test
  comes before any training, so this question does not depend on the CS+;
* mean pre-test and post-training score;
* change in PER score (post - pre), mean ± SEM over flies;
* PER response rate (score >= threshold), pre-test and post-training.

Time of day is NOT randomised in these cohorts: each starvation group is run at
its own time (12/18 h mornings, 1 h early afternoon, 6/24/30 h evenings), so a
pooled time-of-day trend is largely the starvation trend re-sorted. Every
figure therefore carries a strip of how many flies of each starvation group sit
in each bin, and the trained-odor figure colours each fly by starvation group.
To look at time of day with starvation held fixed, run it on one dataset
(``--dataset IAA-Sensitivity-24-1``, recorded from 9:41 to 19:09).

Stats (uncorrected): Kruskal-Wallis across bins on the change, Fisher-Freeman-
Halton (Monte Carlo) across bins on the post-training rate, and -- bin-free --
Spearman's rho between start clock time and each fly's pre, post and change.

Usage
-----
    # every IAA sensitivity fly recorded after Oct 5, 1 h and 2 h bins
    python3 scripts/analysis/time_of_day_trend.py --after 2026-10-05

    # one starvation level only (no starvation confound)
    python3 scripts/analysis/time_of_day_trend.py --dataset IAA-Sensitivity-24-1
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import scripts.analysis.starvation_pre_post_comparison as spc  # noqa: E402
from scripts.analysis import phase_terms  # noqa: E402
from scripts.analysis.day_vs_day_report import (  # noqa: E402
    folder_date,
    odor_concentrations,
    odor_text,
    starvation_hours,
    trained_odor,
)
from scripts.analysis.odor_conc_labels import VV  # noqa: E402
from scripts.analysis.pretest_vs_test_comparison import (  # noqa: E402
    BINARY_THRESHOLD,
    INK,
    POST_COLOR,
    POST_LABEL,
    PRE_COLOR,
    PRE_LABEL,
    _RC_CONTEXT,
    _recessive_grid,
    _sem,
    load_paired_scores,
)
from scripts.analysis.starvation_trend_figure import (  # noqa: E402
    CONTEXT_COLOR,
    DELTA_COLOR,
    EMPHASIS_COLOR,
    LINE_W,
    MARKER,
    ZERO_LINE,
    _is_cs,
    _omnibus,
    _p_text,
)

CSV = Path("/home/ramanlab/Documents/cole/Data/CSVs-New-Opto-Flys")
FLAGGED = CSV / "flagged-flys-truth.csv"
DATA_ROOT = Path("/home/ramanlab/Documents/cole/Data/flys_New")
OUT_ROOT = Path("/home/ramanlab/Documents/cole/Results/Figures")
DEFAULT_DATASETS = tuple(f"IAA-Sensitivity-{h}-1" for h in (1, 6, 12, 18, 24, 30))

_PRETEST_RE = re.compile(r"^output_.*_pretest_\d+_.*_(\d{8}_\d{6})\.csv$")
#: Starvation group, light = less starved (repo convention). One hue (warm
#: gray), so it never reads as an odor or a phase. Validated with the dataviz
#: validator: --ordinal, light mode, all checks pass (light end 2.02:1).
STARVATION_RAMP = ("#b5b4ae", "#97968f", "#7a7973", "#5e5d58", "#42413d", "#262522")


# ── start time and bins ──────────────────────────────────────────────────


def first_pretest_time(folder: Path) -> dt.datetime | None:
    """Earliest pre-test trial file in a batch folder, or None."""
    folder = Path(folder)
    if not folder.is_dir():
        return None
    times = [dt.datetime.strptime(m[1], "%Y%m%d_%H%M%S")
             for f in folder.iterdir() if (m := _PRETEST_RE.match(f.name))]
    return min(times) if times else None


def clock_hours(t: dt.datetime) -> float:
    return t.hour + t.minute / 60 + t.second / 3600


def _hhmm(h: float) -> str:
    mins = int(round(h * 60))
    return f"{mins // 60}:{mins % 60:02d}"


def time_bin(t: dt.datetime, width: float) -> tuple[float, str]:
    """(bin start in clock hours, "13:00–14:00") for a start time."""
    start = float(np.floor(clock_hours(t) / width) * width)
    return start, f"{_hhmm(start)}–{_hhmm(start + width)}"


def assign_bins(paired: pd.DataFrame, width: float) -> pd.DataFrame:
    out = paired.copy()
    bins = [time_bin(t, width) for t in pd.to_datetime(out["start_time"])]
    out["bin_start"] = [b[0] for b in bins]
    out["bin_label"] = [b[1] for b in bins]
    out["x"] = out["bin_start"] + width / 2
    out["group"] = out["bin_label"]
    return out


def drop_small_bins(binned: pd.DataFrame, min_flies: int) -> pd.DataFrame:
    """Rows of the bins holding at least ``min_flies`` flies (each fly counted once)."""
    n = (binned.drop_duplicates(["dataset_canon", "fly", "fly_number"])
         .groupby("bin_start").size())
    return binned[binned["bin_start"].isin(n.index[n >= min_flies])]


# ── loading ──────────────────────────────────────────────────────────────


def load_paired(datasets: Sequence[str], *, predictions_csv: Path, flagged_flies_csv: Path,
                tmp_dir: Path, after: dt.date | None = None,
                before: dt.date | None = None) -> pd.DataFrame:
    """Paired pre/post scores per fly and odor, flagged flies dropped, date-filtered.

    ``attrs["odor_conc"]``: each odor's recorded concentration
    (:func:`odor_concentrations`), for the figure text.
    """
    groups = [spc.Group(d, d, "") for d in datasets]
    raw = pd.read_csv(predictions_csv)
    preds = spc.drop_flagged(spc.subset_and_relabel(raw, groups), flagged_flies_csv)
    days = preds["fly"].map(folder_date)
    keep = pd.Series(True, index=preds.index)
    if after is not None:
        keep &= days > after
    if before is not None:
        keep &= days < before
    preds = preds[keep]
    tmp_dir.mkdir(parents=True, exist_ok=True)
    subset = tmp_dir / "predictions_subset_relabelled.csv"
    preds.to_csv(subset, index=False)
    paired = load_paired_scores(subset, flagged_flies_csv=str(flagged_flies_csv))
    # The relabel stripped each label's pins concentration; the kept rows' raw labels have it.
    paired.attrs["odor_conc"] = odor_concentrations(raw.loc[preds.index, "trial_label"])
    return paired


def attach_start_times(paired: pd.DataFrame, data_root: Path) -> tuple[pd.DataFrame, list]:
    """Add start_time and starvation_h; drop (and return) folders with no pre-test file."""
    folders = paired[["dataset_canon", "fly"]].drop_duplicates()
    times = {(d, f): first_pretest_time(Path(data_root) / d / f)
             for d, f in folders.itertuples(index=False)}
    out = paired.copy()
    out["start_time"] = [times[(d, f)] for d, f in zip(out["dataset_canon"], out["fly"])]
    out["starvation_h"] = out["dataset_canon"].map(lambda d: starvation_hours(d)[0])
    missing = sorted(k for k, v in times.items() if v is None)
    return out.dropna(subset=["start_time"]).reset_index(drop=True), missing


# ── tables and stats ─────────────────────────────────────────────────────


def bin_table(binned: pd.DataFrame, *, binary_threshold: int = BINARY_THRESHOLD) -> pd.DataFrame:
    """One row per (bin, odor): n flies, mean ± SEM change, pre/post response %."""
    rows = []
    for (start, odor), g in binned.groupby(["bin_start", "odor"]):
        pre = g["score_pre"].to_numpy(dtype=float)
        post = g["score_post"].to_numpy(dtype=float)
        delta = post - pre
        pct_pre = 100.0 * float(np.mean(pre >= binary_threshold))
        pct_post = 100.0 * float(np.mean(post >= binary_threshold))
        rows.append({
            "bin_start": start, "bin_label": g["bin_label"].iloc[0], "x": g["x"].iloc[0],
            "odor": odor, "n_flies": len(g),
            "starvation_groups_h": ",".join(f"{h:g}" for h in sorted(g["starvation_h"].unique())),
            "mean_pre": float(pre.mean()), "mean_post": float(post.mean()),
            "mean_delta": float(delta.mean()), "sem_delta": _sem(delta),
            "pct_pre": pct_pre, "pct_post": pct_post, "delta_pct": pct_post - pct_pre,
        })
    return (pd.DataFrame(rows)
            .sort_values(["odor", "bin_start"], key=lambda s: s.str.casefold() if s.dtype == object else s)
            .reset_index(drop=True))


def composition(binned: pd.DataFrame) -> pd.DataFrame:
    """Flies per bin (rows) and starvation group (columns), each fly counted once."""
    flies = binned.drop_duplicates(["dataset_canon", "fly", "fly_number"])
    return (flies.groupby(["bin_start", "starvation_h"]).size().unstack(fill_value=0)
            .sort_index().sort_index(axis=1))


_MEASURES = {"delta": lambda g: g["score_post"] - g["score_pre"],
             "pre": lambda g: g["score_pre"], "post": lambda g: g["score_post"]}


def _spearman(clock: np.ndarray, value: np.ndarray) -> tuple[float, float]:
    if len(np.unique(clock)) < 3 or len(np.unique(value)) < 2:
        return np.nan, np.nan
    res = spearmanr(clock, value)
    return float(res.statistic), float(res.pvalue)


def spearman_by_odor(paired: pd.DataFrame, *, measure: str = "delta") -> pd.DataFrame:
    """Spearman rho, start clock time vs a per-fly score, per odor (bin-free).

    ``measure``: "delta" (post - pre), "pre" (naive pre-test) or "post".
    """
    rows = []
    for odor, g in paired.groupby("odor"):
        clock = pd.to_datetime(g["start_time"]).map(clock_hours).to_numpy(dtype=float)
        rho, p = _spearman(clock, _MEASURES[measure](g).to_numpy(dtype=float))
        rows.append({"odor": odor, "n_flies": len(g), "rho": rho, "p": p})
    return pd.DataFrame(rows)


# ── pre-test (naive) by time of day ──────────────────────────────────────

ALL_ODORS = "All odors (per fly)"


def pooled_pretest(binned: pd.DataFrame, *,
                   binary_threshold: int = BINARY_THRESHOLD) -> pd.DataFrame:
    """Per bin, every pre-test trial of every odor pooled: sum / n trials.

    The error bar is SEM over flies of each fly's own pre-test mean -- the fly
    is the independent unit; the plotted mean itself is never a mean of means.
    """
    rows = []
    for start, g in binned.groupby("bin_start"):
        pre = g["score_pre"].to_numpy(dtype=float)
        per_fly = g.groupby(["dataset_canon", "fly", "fly_number"])["score_pre"].mean()
        rows.append({"bin_start": start, "bin_label": g["bin_label"].iloc[0],
                     "x": g["x"].iloc[0], "n_flies": len(per_fly), "n_trials": len(pre),
                     "mean_pre": float(pre.mean()), "sem_pre_flies": _sem(per_fly.to_numpy()),
                     "pct_pre": 100.0 * float(np.mean(pre >= binary_threshold))})
    return pd.DataFrame(rows).sort_values("bin_start").reset_index(drop=True)


def _fly_pretest(binned: pd.DataFrame, binary_threshold: int) -> pd.DataFrame:
    """One row per fly: its own pre-test mean across odors and any-odor response."""
    return (binned.groupby(["dataset_canon", "fly", "fly_number"])
            .agg(bin_label=("bin_label", "first"), bin_start=("bin_start", "first"),
                 start_time=("start_time", "first"), pre_mean=("score_pre", "mean"),
                 responded=("score_pre", lambda s: bool((s >= binary_threshold).any())))
            .reset_index().sort_values("bin_start"))


def pretest_stats(binned: pd.DataFrame, *,
                  binary_threshold: int = BINARY_THRESHOLD) -> pd.DataFrame:
    """Does the NAIVE pre-test differ by time of day? One row per odor, plus all odors.

    Per odor: Kruskal-Wallis on pre-test score and Fisher-Freeman-Halton on
    pre-test response % across bins, Spearman of start clock time vs pre score.
    All-odor row: same tests on one value per fly (its pre-test mean across
    odors; responded = score >= threshold to ANY odor), so no fly counts 7 times.
    """
    labels = list(binned.sort_values("bin_start")["bin_label"].unique())
    rows = []
    flies = _fly_pretest(binned, binary_threshold)
    by = [flies.loc[flies["bin_label"] == b] for b in labels]
    clock = pd.to_datetime(flies["start_time"]).map(clock_hours).to_numpy(dtype=float)
    rho, p = _spearman(clock, flies["pre_mean"].to_numpy(dtype=float))
    rows.append({"odor": ALL_ODORS, "n_flies": len(flies),
                 "p_kw_pre_score": spc._kw([f["pre_mean"].to_numpy(dtype=float) for f in by])
                 if len(labels) > 1 else None,
                 "p_ffh_pre_rate": spc._ffh([f["responded"].to_numpy() for f in by])
                 if len(labels) > 1 else None,
                 "rho_pre": rho, "p_spearman_pre": p})
    om = omnibus(binned, binary_threshold=binary_threshold)
    sp = spearman_by_odor(binned, measure="pre").set_index("odor")
    for odor in sorted(binned["odor"].unique(), key=str.casefold):
        o = om[om["odor"] == odor] if len(om) else om
        rows.append({"odor": odor, "n_flies": int(sp.loc[odor, "n_flies"]),
                     "p_kw_pre_score": o["p_kw_pre_score"].iloc[0] if len(o) else None,
                     "p_ffh_pre_rate": o["p_ffh_pre_rate"].iloc[0] if len(o) else None,
                     "rho_pre": sp.loc[odor, "rho"], "p_spearman_pre": sp.loc[odor, "p"]})
    return pd.DataFrame(rows)


def omnibus(binned: pd.DataFrame, *, binary_threshold: int = BINARY_THRESHOLD) -> pd.DataFrame:
    labels = list(binned.sort_values("bin_start")["bin_label"].unique())
    if len(labels) < 2:
        return pd.DataFrame()
    return spc.omnibus_stats(binned, labels, binary_threshold=binary_threshold)


def starvation_colors(hours: Sequence[float]) -> dict[float, str]:
    levels = sorted(set(float(h) for h in hours))
    if len(levels) > len(STARVATION_RAMP):  # sampled along the same validated ramp
        from scripts.analysis.starvation_pre_post_comparison import _sample_ramp

        return dict(zip(levels, _sample_ramp(STARVATION_RAMP, len(levels))))
    idx = np.round(np.linspace(0, len(STARVATION_RAMP) - 1, len(levels))).astype(int) \
        if len(levels) > 1 else [len(STARVATION_RAMP) - 2]
    return {h: STARVATION_RAMP[i] for h, i in zip(levels, idx)}


# ── drawing ──────────────────────────────────────────────────────────────
# ``cohort`` words the phases (phase_terms): a *-Sensitivity-* name gives
# pre-/post-exposure and "sensitized odor"; None keeps pre-test / CS+.
# ``odor_conc`` (load_paired's attrs["odor_conc"]) adds each odor's recorded
# concentration on a *-Sensitivity-* figure: "Isoamyl Acetate 1% (v/v)".


def _gap_break(t: pd.DataFrame, key: str, width: float) -> tuple[np.ndarray, np.ndarray]:
    """x, y with a NaN wherever bins are not adjacent, so no line spans an empty hour."""
    xs, ys, prev = [], [], None
    for start, x, y in zip(t["bin_start"], t["x"], t[key]):
        if prev is not None and start - prev > width + 1e-9:
            xs.append(np.nan)
            ys.append(np.nan)
        xs.append(x)
        ys.append(y)
        prev = start
    return np.array(xs), np.array(ys)


def _line(ax, t, key, width, **kw):
    x, y = _gap_break(t, key, width)
    ax.plot(x, y, "-", solid_joinstyle="round", solid_capstyle="round", **kw)


def _clock_axis(ax, starts: Sequence[float], width: float, *, extra: float = 0.0,
                n: pd.Series | None = None) -> None:
    """Clock-time ticks at the bin edges; with ``n`` (flies, indexed by bin start)
    a second label row under the times puts "n=17" at the centre of each bin."""
    lo, hi = min(starts), max(starts) + width
    edges = np.arange(lo, hi + 1e-9, width)
    ax.set_xticks(edges)
    ax.set_xticklabels([_hhmm(e) for e in edges])
    if n is not None:
        n = n.sort_index()
        ax.set_xticks(n.index.to_numpy(dtype=float) + width / 2, minor=True)
        ax.set_xticklabels([f"n={int(v)}" for v in n], minor=True)
        ax.tick_params(axis="x", which="minor", length=0, pad=17, labelsize=7.5,
                       labelcolor=INK)
    ax.set_xlim(lo - 0.15 * width, hi + 0.15 * width + extra)


def _composition_strip(ax, comp: pd.DataFrame, width: float, *, legend: bool) -> None:
    colors = starvation_colors(comp.columns)
    bottom = np.zeros(len(comp))
    x = comp.index.to_numpy(dtype=float) + width / 2
    for h in comp.columns:
        vals = comp[h].to_numpy(dtype=float)
        ax.bar(x, vals, 0.8 * width, bottom=bottom, color=colors[h], edgecolor="white",
               linewidth=0.8, label=f"{h:g} h", zorder=2)
        bottom += vals
    for xi, total in zip(x, bottom):
        ax.text(xi, total, f"{int(total)}", ha="center", va="bottom", fontsize=7.5, color=INK)
    ax.set_ylim(0, max(bottom.max(), 1) * 1.3)
    ax.set_ylabel("Flies", fontsize=8.5)
    ax.tick_params(axis="y", labelsize=7.5)
    if legend:
        ax.legend(title="Starvation group", loc="upper left", bbox_to_anchor=(1.01, 1.05),
                  frameon=False, fontsize=7.5, title_fontsize=7.5)


def _rate_lines(ax, t, width, *, labels: bool = True, cohort: str | None = None) -> None:
    pre, post = phase_terms.pre_label(cohort, PRE_LABEL), phase_terms.post_label(cohort, POST_LABEL)
    _line(ax, t, "pct_pre", width, color=PRE_COLOR, linewidth=LINE_W, zorder=3)
    ax.plot(t["x"], t["pct_pre"], "o", color=PRE_COLOR, label=pre if labels else None,
            zorder=3, **MARKER)
    _line(ax, t, "pct_post", width, color=POST_COLOR, linewidth=LINE_W, zorder=4)
    ax.plot(t["x"], t["pct_post"], "s", color=POST_COLOR, label=post if labels else None,
            zorder=4, **MARKER)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])


def _delta_line(ax, t, width, *, color=DELTA_COLOR) -> None:
    ax.axhline(0.0, **ZERO_LINE)
    ax.errorbar(t["x"], t["mean_delta"], yerr=t["sem_delta"], fmt="none", ecolor=color,
                elinewidth=1.0, capsize=2.5, zorder=3)
    _line(ax, t, "mean_delta", width, color=color, linewidth=LINE_W, zorder=4)
    ax.plot(t["x"], t["mean_delta"], "o", color=color, zorder=4, **MARKER)


def build_emphasis(table: pd.DataFrame, comp: pd.DataFrame, *, omnibus, trained_odor: str,
                   title: str, width: float, spearman: pd.DataFrame | None = None,
                   cohort: str | None = None, odor_conc: Mapping[str, str] | None = None):
    """Score change and rate change, every odor (CS+ inked), over the starvation mix."""
    odors = sorted(table["odor"].unique(), key=str.casefold)
    cs = next((o for o in odors if _is_cs(o, trained_odor)), None)
    cs_text = cs if cs is None else odor_text(cs, cohort, odor_conc)
    starts = sorted(table["bin_start"].unique())
    with plt.rc_context(_RC_CONTEXT):
        fig, (a1, a2, a3) = plt.subplots(
            3, 1, figsize=(max(6.4, 0.55 * len(starts) * width + 3.6), 7.6), sharex=True,
            gridspec_kw={"height_ratios": [3, 3, 1.25]})
        for ax, key, ylabel in (
            (a1, "mean_delta", "Change in PER score\n(post − pre)"),
            (a2, "delta_pct", "Change in flies responding\n(post − pre, % points)"),
        ):
            ax.axhline(0.0, **ZERO_LINE)
            for odor in odors:
                if odor == cs:
                    continue
                t = table[table["odor"] == odor].sort_values("bin_start")
                _line(ax, t, key, width, color=CONTEXT_COLOR, linewidth=1.2, zorder=2)
                ax.plot(t["x"], t[key], "o", color=CONTEXT_COLOR, markersize=3, zorder=2)
            if cs is not None:
                t = table[table["odor"] == cs].sort_values("bin_start")
                if key == "mean_delta":
                    ax.errorbar(t["x"], t[key], yerr=t["sem_delta"], fmt="none",
                                ecolor=EMPHASIS_COLOR, elinewidth=1.0, capsize=3, zorder=3)
                _line(ax, t, key, width, color=EMPHASIS_COLOR, linewidth=LINE_W, zorder=4)
                ax.plot(t["x"], t[key], "o", color=EMPHASIS_COLOR, zorder=4, **MARKER)
            ax.set_ylabel(ylabel)
            _recessive_grid(ax)
        a1.set_ylim(-3, 3)
        a2.set_ylim(-80, 80)
        a2.set_yticks(range(-80, 81, 40))
        if cs is not None:
            _p_text(a1, f"{cs_text}: Kruskal-Wallis across bins",
                    _omnibus(omnibus, cs, "p_kw_delta_score"))
            if spearman is not None and len(spearman[spearman["odor"] == cs]):
                r = spearman[spearman["odor"] == cs].iloc[0]
                if pd.notna(r["rho"]):
                    a1.text(0.03, 0.85, f"Spearman (start time vs change): rho={r['rho']:.2f}, "
                            f"p={r['p']:.3f}", transform=a1.transAxes, ha="left", va="top",
                            fontsize=7.5, color=INK,
                            fontweight="bold" if r["p"] < 0.05 else "normal")
            _p_text(a2, f"{cs_text}: "
                    f"{phase_terms.post_label(cohort, 'post-training', case='lower')} "
                    "FFH across bins", _omnibus(omnibus, cs, "p_ffh_post_rate"))
        handles = [plt.Line2D([], [], color=EMPHASIS_COLOR, linewidth=LINE_W, marker="o",
                              label=f"{cs_text} ({phase_terms.odor_tag(cohort, 'trained, CS+')})"),
                   plt.Line2D([], [], color=CONTEXT_COLOR, linewidth=1.2, marker="o",
                              markersize=3,
                              label=f"other {len(odors) - (cs is not None)} odors")]
        a1.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False,
                  fontsize=8)
        _composition_strip(a3, comp, width, legend=True)
        _clock_axis(a3, starts, width, n=table.groupby("bin_start")["n_flies"].max())
        # A fly's first pre-test trial is where its pre-exposure panel starts.
        a3.set_xlabel(f"Start of {phase_terms.pre_label(cohort, 'first pre-test', case='lower')} "
                      f"(time of day, {width:g} h bins; n = flies)")
        effect = phase_terms.block_label(cohort)
        fig.suptitle(f"{title} — how time of day changes the effect of {effect}\n"
                     f"{phase_terms.odor_tag(cohort, 'CS+', capital=True)} mean ± SEM per bin; "
                     "bottom: flies per bin by starvation group "
                     "(time of day and starvation are confounded) · uncorrected",
                     x=0.01, ha="left", fontsize=10, color=INK)
        fig.tight_layout()
    return fig


def build_spotlight(table: pd.DataFrame, paired: pd.DataFrame, comp: pd.DataFrame, *,
                    omnibus, trained_odor: str, title: str, width: float,
                    spearman: pd.DataFrame | None = None, cohort: str | None = None,
                    odor_conc: Mapping[str, str] | None = None):
    """The CS+ alone: every fly's change (coloured by starvation group) and the rates."""
    odor = next((o for o in table["odor"].unique() if _is_cs(o, trained_odor)), None)
    if odor is None:
        return None
    t = table[table["odor"] == odor].sort_values("bin_start")
    starts = sorted(table["bin_start"].unique())
    colors = starvation_colors(comp.columns)
    rng = np.random.default_rng(0)
    sub = paired[paired["odor"] == odor]
    with plt.rc_context(_RC_CONTEXT):
        fig, axes = plt.subplots(2, 2, figsize=(11.0, 5.6), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1.1]})
        left, right = axes[0]
        for h, g in sub.groupby("starvation_h"):
            d = (g["score_post"] - g["score_pre"]).to_numpy(dtype=float)
            left.scatter(g["x"] + rng.uniform(-0.3, 0.3, d.size) * width,
                         d + rng.uniform(-0.12, 0.12, d.size), s=20, color=colors[float(h)],
                         edgecolors="white", linewidths=0.8, zorder=2)
        _delta_line(left, t, width)
        left.set_ylim(-5.6, 5.6)  # a full ±5 change must not sit on the frame
        left.set_yticks(range(-4, 5, 2))
        left.set_ylabel("Change in PER score (post − pre)")
        left.set_title("PER score change per fly (dot colour = starvation group)", loc="left",
                       color=INK, fontsize=9.5)
        _p_text(left, "Kruskal-Wallis", _omnibus(omnibus, odor, "p_kw_delta_score"))
        if spearman is not None and len(spearman[spearman["odor"] == odor]):
            r = spearman[spearman["odor"] == odor].iloc[0]
            if pd.notna(r["rho"]):
                left.text(0.03, 0.87, f"Spearman rho={r['rho']:.2f}, p={r['p']:.3f}",
                          transform=left.transAxes, ha="left", va="top", fontsize=7.5,
                          color=INK, fontweight="bold" if r["p"] < 0.05 else "normal")
        _rate_lines(right, t, width, cohort=cohort)
        right.set_ylabel("PER response % (score ≥ 2)")
        right.set_title("Flies responding", loc="left", color=INK, fontsize=9.5)
        right.legend(loc="upper right", frameon=False, fontsize=8)
        _p_text(right, f"{phase_terms.post_label(cohort, 'post-training', case='lower')} FFH",
                _omnibus(omnibus, odor, "p_ffh_post_rate"))
        for ax in (left, right):
            _recessive_grid(ax)
        first = phase_terms.pre_label(cohort, "first pre-test", case="lower")
        for c, ax in enumerate(axes[1]):
            _composition_strip(ax, comp, width, legend=(c == 1))
            _clock_axis(ax, starts, width, n=t.set_index("bin_start")["n_flies"])
            ax.set_xlabel(f"Start of {first} ({width:g} h bins; n = flies)")
        fig.suptitle(f"{title} — {odor_text(odor, cohort, odor_conc)} "
                     f"({phase_terms.odor_tag(cohort, 'trained odor, CS+')}) by time of day",
                     x=0.01, ha="left", fontsize=10.5, color=INK)
        fig.tight_layout()
    return fig


def build_grid(table: pd.DataFrame, *, omnibus, trained_odor: str, title: str, width: float,
               cohort: str | None = None, odor_conc: Mapping[str, str] | None = None):
    """One column per odor: change (top) and pre/post rate (bottom) vs time of day."""
    odors = sorted(table["odor"].unique(), key=str.casefold)
    pre = phase_terms.pre_label(cohort, "pre-test", case="lower")
    post = phase_terms.post_label(cohort, "post-training", case="lower")
    starts = sorted(table["bin_start"].unique())
    lo = float((table["mean_delta"] - table["sem_delta"].fillna(0)).min())
    hi = float((table["mean_delta"] + table["sem_delta"].fillna(0)).max())
    ylim = (np.floor(min(lo, 0) * 2) / 2 - 0.25, np.ceil(max(hi, 0) * 2) / 2 + 0.5)
    with plt.rc_context(_RC_CONTEXT):
        fig, axes = plt.subplots(3, len(odors), figsize=(2.6 * len(odors) + 0.8, 8.4),
                                 squeeze=False)
        smax = float(np.ceil(max(table["mean_pre"].max(), table["mean_post"].max()) + 0.5))
        for c, odor in enumerate(odors):
            t = table[table["odor"] == odor].sort_values("bin_start")
            score, top, bottom = axes[0, c], axes[1, c], axes[2, c]
            _score_lines(score, t, width, labels=(c == 0), cohort=cohort)
            score.set_ylim(0, smax)
            _p_text(score, "pre KW", _omnibus(omnibus, odor, "p_kw_pre_score"))
            _delta_line(top, t, width)
            top.set_ylim(*ylim)
            _p_text(top, "KW", _omnibus(omnibus, odor, "p_kw_delta_score"))
            _rate_lines(bottom, t, width, labels=(c == 0), cohort=cohort)
            _p_text(bottom, "post FFH", _omnibus(omnibus, odor, "p_ffh_post_rate"))
            cs = _is_cs(odor, trained_odor)
            name = odor_text(odor, cohort, odor_conc)
            if cs:  # with its concentration the name fills the column: the tag wraps under it
                name += ("\n" if name.endswith(VV) else "  ") + f"({phase_terms.odor_tag(cohort)})"
            score.set_title(name, loc="left", fontsize=9.5, color=INK,
                            fontweight="bold" if cs else "normal")
            n = t.set_index("bin_start")["n_flies"]
            for ax in (score, top, bottom):
                # One tick per bin, at its centre; the bottom row names it and its n.
                ax.set_xticks([b + width / 2 for b in starts])
                ax.set_xlim(min(starts) - 0.15 * width, max(starts) + 1.15 * width)
                ax.set_xticklabels([f"{_hhmm(b)}–{_hhmm(b + width)}  n={int(n.get(b, 0))}"
                                    for b in starts] if ax is bottom else [])
                ax.tick_params(axis="x", labelrotation=90, labelsize=7.5)
                _recessive_grid(ax)
                if c:
                    ax.tick_params(labelleft=False)
            # The clock time is the first pre-test trial, i.e. the start of pre-exposure.
            bottom.set_xlabel(f"Start of {pre} (n = flies)" if phase_terms.is_sensitivity(cohort)
                              else "First pre-test (n = flies)")
        axes[0, 0].set_ylabel("Mean PER score")
        axes[1, 0].set_ylabel("Change in PER score\n(post − pre)")
        axes[2, 0].set_ylabel("PER response %")
        # Top right of the figure: under the axes it would sit on the rotated bin labels.
        h, lab = axes[0, 0].get_legend_handles_labels()
        fig.legend(h, lab, loc="upper right", bbox_to_anchor=(0.995, 0.995), ncol=2,
                   frameon=False)
        fig.suptitle(f"{title} — PER change and response rate by time of day ({width:g} h bins)\n"
                     f"top: mean score, {pre} and {post} (pre KW = {pre} across bins) · "
                     "middle: mean ± SEM change per fly (KW = Kruskal-Wallis across bins)\n"
                     "bottom: % flies responding (score ≥ 2), FFH = Fisher-Freeman-Halton · "
                     "uncorrected",
                     x=0.01, ha="left", fontsize=10, color=INK)
        fig.tight_layout()
    return fig


def _score_lines(ax, t, width, *, labels: bool = True, cohort: str | None = None) -> None:
    """Mean PER score, pre-test (blue circles) and post-training (orange squares)."""
    pre, post = phase_terms.pre_label(cohort, PRE_LABEL), phase_terms.post_label(cohort, POST_LABEL)
    for key, color, mk, lab, z in (("mean_pre", PRE_COLOR, "o", pre, 3),
                                   ("mean_post", POST_COLOR, "s", post, 4)):
        _line(ax, t, key, width, color=color, linewidth=LINE_W, zorder=z)
        ax.plot(t["x"], t[key], mk, color=color, label=lab if labels else None, zorder=z,
                **MARKER)


def _stat_line(row: pd.Series) -> tuple[str, bool]:
    def fmt(p):
        return "n/a" if p is None or pd.isna(p) else (f"{p:.3f}" if p >= 0.001 else "<0.001")
    sig = any(pd.notna(row[k]) and row[k] < 0.05
              for k in ("p_kw_pre_score", "p_ffh_pre_rate", "p_spearman_pre"))
    return (f"score KW p={fmt(row['p_kw_pre_score'])} · response FFH "
            f"p={fmt(row['p_ffh_pre_rate'])} · Spearman rho={row['rho_pre']:.2f} "
            f"p={fmt(row['p_spearman_pre'])}" if pd.notna(row["rho_pre"]) else
            f"score KW p={fmt(row['p_kw_pre_score'])} · response FFH "
            f"p={fmt(row['p_ffh_pre_rate'])}"), sig


def build_pretest(table: pd.DataFrame, pooled: pd.DataFrame, comp: pd.DataFrame, *,
                  stats: pd.DataFrame, title: str, width: float, cohort: str | None = None,
                  odor_conc: Mapping[str, str] | None = None):
    """Naive pre-test PER by time of day: all odors pooled (ink), each odor (gray)."""
    odors = sorted(table["odor"].unique(), key=str.casefold)
    starts = sorted(table["bin_start"].unique())
    st = stats.set_index("odor")
    # Nothing is trained in a *-Sensitivity-* cohort, so its first panel is plain
    # pre-exposure, without the "naive, before training" gloss.
    sens = phase_terms.is_sensitivity(cohort)
    pre = phase_terms.pre_label(cohort, "Pre-test")
    with plt.rc_context(_RC_CONTEXT):
        fig, (a1, a2, a3) = plt.subplots(
            3, 1, figsize=(max(6.4, 0.55 * len(starts) * width + 3.6), 7.8), sharex=True,
            gridspec_kw={"height_ratios": [3, 3, 1.25]})
        for ax, key in ((a1, "mean_pre"), (a2, "pct_pre")):
            for odor in odors:
                t = table[table["odor"] == odor].sort_values("bin_start")
                _line(ax, t, key, width, color=CONTEXT_COLOR, linewidth=1.2, zorder=2)
                ax.plot(t["x"], t[key], "o", color=CONTEXT_COLOR, markersize=3, zorder=2)
            if key == "mean_pre":
                ax.errorbar(pooled["x"], pooled[key], yerr=pooled["sem_pre_flies"], fmt="none",
                            ecolor=EMPHASIS_COLOR, elinewidth=1.0, capsize=3, zorder=3)
            _line(ax, pooled, key, width, color=EMPHASIS_COLOR, linewidth=LINE_W, zorder=4)
            ax.plot(pooled["x"], pooled[key], "o", color=EMPHASIS_COLOR, zorder=4, **MARKER)
            _recessive_grid(ax)
        a1.set_ylabel(f"{pre} PER score" if sens
                      else "Pre-test PER score\n(naive, before training)")
        a1.set_ylim(0, max(2.0, float(np.ceil(table["mean_pre"].max() + 0.5))))
        a2.set_ylabel(f"{pre} flies responding %\n(score ≥ 2)")
        a2.set_ylim(0, 100)
        a2.set_yticks([0, 25, 50, 75, 100])
        # Stats: the all-odor (per-fly) row first, then every odor that reaches p<0.05.
        line, sig = _stat_line(st.loc[ALL_ODORS])
        a1.text(0.02, 0.97, f"All odors (one value per fly):\n{line}", transform=a1.transAxes,
                ha="left", va="top", fontsize=7.5, color=INK,
                fontweight="bold" if sig else "normal")
        hits = [odor_text(o, cohort, odor_conc)
                for o in odors if o in st.index and _stat_line(st.loc[o])[1]]
        a1.text(0.02, 0.84, ("Single odors with any p<0.05: " + ", ".join(hits)) if hits
                else "No single odor reaches p<0.05", transform=a1.transAxes, ha="left",
                va="top", fontsize=7.5, color=INK)
        handles = [plt.Line2D([], [], color=EMPHASIS_COLOR, linewidth=LINE_W, marker="o",
                              label="all odors pooled\n(mean ± SEM over flies)"),
                   plt.Line2D([], [], color=CONTEXT_COLOR, linewidth=1.2, marker="o",
                              markersize=3, label=f"each of {len(odors)} odors")]
        a1.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False,
                  fontsize=8)
        _composition_strip(a3, comp, width, legend=True)
        _clock_axis(a3, starts, width, n=pooled.set_index("bin_start")["n_flies"])
        a3.set_xlabel(f"Start of {phase_terms.pre_label(cohort, 'first pre-test', case='lower')} "
                      f"(time of day, {width:g} h bins; n = flies)")
        naive = f"{pre.lower()} PER" if sens else "the naive pre-test"
        fig.suptitle(f"{title} — is {naive} different by time of day?\n"
                     f"per-odor detail: time_of_day_all_odors.png (top row) and "
                     f"pretest_stats.csv · uncorrected",
                     x=0.01, ha="left", fontsize=10, color=INK)
        fig.tight_layout()
    return fig


# ── driver ───────────────────────────────────────────────────────────────


def _save(fig, path: Path) -> Path:
    with plt.rc_context(_RC_CONTEXT):
        fig.savefig(path)
    plt.close(fig)
    return path


def render_width(paired: pd.DataFrame, width: float, *, out_dir: Path, trained: str,
                 title: str, spearman: pd.DataFrame,
                 odor_conc: Mapping[str, str] | None = None,
                 min_flies: int = 0) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    binned = assign_bins(paired, width)
    if min_flies:
        # Small bins leave every table, test and figure, and the Spearman
        # trend is recomputed on the flies that remain.
        binned = drop_small_bins(binned, min_flies)
        spearman = spearman_by_odor(binned)
    table = bin_table(binned)
    comp = composition(binned)
    om = omnibus(binned)
    pooled = pooled_pretest(binned)
    pre_stats = pretest_stats(binned)
    # A pool says pre-/post-exposure only if every dataset in it is a
    # *-Sensitivity-* one; a mixed pool keeps pre-test / post-training / CS+.
    names = sorted(set(paired["dataset_canon"].astype(str)))
    cohort = names[0] if phase_terms.all_sensitivity(names) else None
    written = []
    for name, frame in (("time_of_day_table.csv", table), ("stats_omnibus.csv", om),
                        ("pretest_pooled_by_bin.csv", pooled), ("pretest_stats.csv", pre_stats),
                        ("flies_per_bin_by_starvation.csv", comp.reset_index())):
        frame.to_csv(out_dir / name, index=False)
        written.append(out_dir / name)
    kw = dict(omnibus=om if len(om) else None, trained_odor=trained, title=title, width=width,
              cohort=cohort, odor_conc=odor_conc)
    figs = (("time_of_day_pretest.png",
             build_pretest(table, pooled, comp, stats=pre_stats, title=title, width=width,
                           cohort=cohort, odor_conc=odor_conc)),
            ("time_of_day_emphasis.png", build_emphasis(table, comp, spearman=spearman, **kw)),
            ("time_of_day_trained_odor.png",
             build_spotlight(table, binned, comp, spearman=spearman, **kw)),
            ("time_of_day_all_odors.png", build_grid(table, **kw)))
    for name, fig in figs:
        if fig is not None:
            written.append(_save(fig, out_dir / name))
    return written


def _default_out(datasets: Sequence[str], after, before) -> Path:
    name = datasets[0] if len(datasets) == 1 else "IAA-Sensitivity"
    if len(datasets) > 1 and not all(d.startswith("IAA-Sensitivity") for d in datasets):
        name = "mixed-Sensitivity"
    name += "_time-of-day"
    if after:
        name += f"_after-{after}"
    if before:
        name += f"_before-{before}"
    return OUT_ROOT / name


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", action="append", default=None,
                    help=f"repeatable; default: {', '.join(DEFAULT_DATASETS)}")
    ap.add_argument("--after", type=dt.date.fromisoformat, default=None,
                    help="only batch folders recorded strictly after this date (YYYY-MM-DD)")
    ap.add_argument("--before", type=dt.date.fromisoformat, default=None,
                    help="only batch folders recorded strictly before this date")
    ap.add_argument("--bin-hours", type=float, action="append", default=None,
                    help="bin width in hours, repeatable (default: 1 and 2)")
    ap.add_argument("--title", default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--predictions-csv", type=Path, default=CSV / "model_predictions.csv")
    ap.add_argument("--flagged-flies-csv", type=Path, default=FLAGGED)
    ap.add_argument("--data-root", type=Path, default=DATA_ROOT)
    ap.add_argument("--min-flies", type=int, default=0,
                    help="drop time bins with fewer flies than this (default: keep all)")
    args = ap.parse_args(argv)

    datasets = args.dataset or list(DEFAULT_DATASETS)
    trained = {trained_odor(d) for d in datasets}
    if len(trained) != 1:
        raise SystemExit(f"datasets train different odors ({sorted(trained)}); run them apart")
    trained = trained.pop()
    widths = args.bin_hours or [1.0, 2.0]
    out = args.out_dir or _default_out(datasets, args.after, args.before)
    title = args.title or (datasets[0] if len(datasets) == 1 else "IAA sensitivity")

    paired = load_paired(datasets, predictions_csv=args.predictions_csv,
                         flagged_flies_csv=args.flagged_flies_csv, tmp_dir=out / "inputs",
                         after=args.after, before=args.before)
    odor_conc = paired.attrs.get("odor_conc")
    paired, missing = attach_start_times(paired, args.data_root)
    for d, f in missing:
        print(f"[WARN] no pre-test file found, dropped: {d}/{f}")
    if paired.empty:
        raise SystemExit("no flies with paired pre/post scores and a pre-test time")
    paired["clock_h"] = pd.to_datetime(paired["start_time"]).map(clock_hours)
    paired.to_csv(out / "paired_scores_with_start_time.csv", index=False)
    flies = (paired.drop_duplicates(["dataset_canon", "fly", "fly_number"])
             [["dataset_canon", "fly", "fly_number", "starvation_h", "start_time", "clock_h"]]
             .sort_values("start_time"))
    flies.to_csv(out / "flies_first_pretest_time.csv", index=False)
    spearman = spearman_by_odor(paired)
    pd.concat([spearman_by_odor(paired, measure=m).assign(measure=m)
               for m in ("pre", "post", "delta")]).to_csv(
        out / "spearman_start_time_vs_score.csv", index=False)
    print(f"{len(flies)} flies, {flies['fly'].nunique()} folders, "
          f"first pre-test {_hhmm(flies['clock_h'].min())}–{_hhmm(flies['clock_h'].max())}")

    for w in widths:
        for path in render_width(paired, w, out_dir=out / f"bins_{w:g}h", trained=trained,
                                 title=title, spearman=spearman, odor_conc=odor_conc,
                                 min_flies=args.min_flies):
            print(f"[OK] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
