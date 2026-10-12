#!/usr/bin/env python3
"""PER change and PER response rate as a function of starvation time.

Reads a folder written by ``starvation_pre_post_comparison.py`` (its
``paired_scores_per_fly.csv`` and, if present, ``stats_omnibus.csv``) and plots
each measure against starvation HOURS on a linear axis, so the spacing between
1, 12, 18, 24 and 30 h is real and a trend reads as a line. Group labels must
carry the hours ("12 h").

Two measures, two rows -- never a dual axis:

* change in PER score (post - pre), mean ± SEM over flies;
* PER response rate (score >= threshold), pre-test and post-training lines,
  so the vertical gap at each starvation time IS the change in rate.

Writes ``starvation_trend_emphasis.png`` (the headline: every odor on one axis,
the trained odor in ink, the rest in gray), ``starvation_trend_all_odors.png``
(one column per odor),
``starvation_trend_trained_odor.png`` (the CS+ with every fly shown) and
``starvation_trend_table.csv``. Panel annotations carry the all-group tests
from ``stats_omnibus.csv`` (Kruskal-Wallis on the change, Fisher-Freeman-Halton
on the post-training rate), uncorrected.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.pretest_vs_test_comparison import (  # noqa: E402
    BINARY_THRESHOLD,
    INK,
    INK_MUTED,
    POST_COLOR,
    POST_LABEL,
    PRE_COLOR,
    PRE_LABEL,
    _RC_CONTEXT,
    _recessive_grid,
    _sem,
)

#: The change belongs to neither phase, so it wears ink, not blue or orange.
DELTA_COLOR = "#3d3c39"
DOT_COLOR = "#a8a7a1"
#: Emphasis (dataviz "one series is the point"): the trained odor in ink, every
#: other odor in de-emphasis gray. Ink, not a hue -- each hue is already some
#: odor's identity colour in this repo, and the CS+'s own yellow is too pale
#: for a line on white.
EMPHASIS_COLOR = "#1a1a19"
CONTEXT_COLOR = "#b3b2ad"
#: dataviz mark spec: 2px lines, >=8px markers with a 2px surface ring.
LINE_W = 2.0
MARKER = dict(markersize=6.5, markeredgecolor="white", markeredgewidth=1.5)
ZERO_LINE = dict(color="#9a9994", linewidth=0.9, linestyle="-", zorder=0.6)
_HOURS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*h\b", re.IGNORECASE)


def hours_from_label(label: str) -> float:
    m = _HOURS_RE.search(str(label))
    if not m:
        raise ValueError(f"group label carries no hours: {label!r}")
    return float(m.group(1))


def trend_table(paired: pd.DataFrame, *, binary_threshold: int = BINARY_THRESHOLD) -> pd.DataFrame:
    """One row per (group, odor): n flies, mean ± SEM change, pre/post response %."""
    rows = []
    for (group, odor), g in paired.groupby(["group", "odor"]):
        pre = g["score_pre"].to_numpy(dtype=float)
        post = g["score_post"].to_numpy(dtype=float)
        delta = post - pre
        rows.append({
            "group": group, "hours": hours_from_label(group), "odor": odor,
            "n_flies": len(g),
            "mean_pre": float(np.mean(pre)), "mean_post": float(np.mean(post)),
            "mean_delta": float(np.mean(delta)), "sem_delta": _sem(delta),
            "pct_pre": 100.0 * float(np.mean(pre >= binary_threshold)),
            "pct_post": 100.0 * float(np.mean(post >= binary_threshold)),
        })
        rows[-1]["delta_pct"] = rows[-1]["pct_post"] - rows[-1]["pct_pre"]
    return (pd.DataFrame(rows)
            .sort_values(["odor", "hours"], key=lambda s: s.str.casefold() if s.dtype == object else s)
            .reset_index(drop=True))


def _omnibus(omnibus: pd.DataFrame | None, odor: str, key: str) -> float | None:
    if omnibus is None or key not in omnibus:
        return None
    row = omnibus[omnibus["odor"] == odor]
    if not len(row) or pd.isna(row[key].iloc[0]):
        return None
    return float(row[key].iloc[0])


def _p_text(ax, label: str, p: float | None) -> None:
    if p is None:
        return
    ax.text(0.03, 0.97, f"{label} p={p:.3f}" if p >= 0.001 else f"{label} p<0.001",
            transform=ax.transAxes, ha="left", va="top", fontsize=7.5, color=INK,
            fontweight="bold" if p < 0.05 else "normal")


def _hour_axis(ax, hours: Sequence[float], *, labels: Sequence[str] | None = None) -> None:
    ax.set_xticks(list(hours))
    ax.set_xticklabels(labels if labels is not None else [f"{h:g}" for h in hours])
    ax.set_xlim(min(hours) - 2, max(hours) + 2)


def _is_cs(odor: str, trained: str) -> bool:
    return bool(trained) and odor.casefold().startswith(trained.casefold())


def _delta_panel(ax, t: pd.DataFrame) -> None:
    ax.axhline(0.0, **ZERO_LINE)
    ax.errorbar(t["hours"], t["mean_delta"], yerr=t["sem_delta"], fmt="none",
                ecolor=DELTA_COLOR, elinewidth=1.0, capsize=2.5, zorder=3)
    ax.plot(t["hours"], t["mean_delta"], "-", color=DELTA_COLOR, linewidth=LINE_W,
            marker="o", markerfacecolor=DELTA_COLOR, solid_capstyle="round",
            solid_joinstyle="round", zorder=4, **MARKER)


def _rate_panel(ax, t: pd.DataFrame, *, labels: bool = True) -> None:
    ax.plot(t["hours"], t["pct_pre"], "-", color=PRE_COLOR, linewidth=LINE_W, marker="o",
            label=PRE_LABEL if labels else None, solid_joinstyle="round", zorder=3, **MARKER)
    # Square markers: phase is told by shape as well as hue.
    ax.plot(t["hours"], t["pct_post"], "-", color=POST_COLOR, linewidth=LINE_W, marker="s",
            label=POST_LABEL if labels else None, solid_joinstyle="round", zorder=4, **MARKER)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])


def _n_footnote(t: pd.DataFrame) -> str:
    n = t.groupby("hours")["n_flies"].max().sort_index()
    return "n (flies): " + " · ".join(f"{h:g} h = {int(v)}" for h, v in n.items())


def build_grid(table: pd.DataFrame, *, omnibus: pd.DataFrame | None, trained_odor: str,
               title: str):
    odors = sorted(table["odor"].unique(), key=str.casefold)
    hours = sorted(table["hours"].unique())
    lo = float((table["mean_delta"] - table["sem_delta"]).min())
    hi = float((table["mean_delta"] + table["sem_delta"]).max())
    ylim = (np.floor(min(lo, 0) * 2) / 2 - 0.25, np.ceil(max(hi, 0) * 2) / 2 + 0.5)

    with plt.rc_context(_RC_CONTEXT):
        fig, axes = plt.subplots(2, len(odors), figsize=(2.35 * len(odors) + 0.8, 5.8),
                                 squeeze=False)
        for c, odor in enumerate(odors):
            t = table[table["odor"] == odor].sort_values("hours")
            top, bottom = axes[0, c], axes[1, c]
            _delta_panel(top, t)
            top.set_ylim(*ylim)
            _p_text(top, "KW", _omnibus(omnibus, odor, "p_kw_delta_score"))
            _rate_panel(bottom, t, labels=(c == 0))
            _p_text(bottom, "post FFH", _omnibus(omnibus, odor, "p_ffh_post_rate"))
            cs = _is_cs(odor, trained_odor)
            top.set_title(odor + ("  (CS+)" if cs else ""), loc="left", fontsize=9.5,
                          color=INK, fontweight="bold" if cs else "normal")
            for ax in (top, bottom):
                _hour_axis(ax, hours)
                _recessive_grid(ax)
                if c:
                    ax.tick_params(labelleft=False)
            bottom.set_xlabel("Starvation (h)")
        axes[0, 0].set_ylabel("Change in PER score\n(post − pre)")
        axes[1, 0].set_ylabel("PER response %")
        axes[1, 0].legend(loc="upper left", bbox_to_anchor=(0.0, -0.32), ncol=2, frameon=False)
        fig.suptitle(f"{title} — PER change and response rate vs starvation\n"
                     "top: mean ± SEM change per fly (KW = Kruskal-Wallis across groups) · "
                     "bottom: % flies responding (score ≥ 2), FFH = Fisher-Freeman-Halton · "
                     "uncorrected",
                     x=0.01, ha="left", fontsize=10, color=INK)
        fig.text(0.99, 0.005, _n_footnote(table), ha="right", va="bottom", fontsize=8,
                 color=INK_MUTED)
        fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def build_spotlight(table: pd.DataFrame, paired: pd.DataFrame, *, omnibus, trained_odor: str,
                    title: str):
    odor = next((o for o in table["odor"].unique() if _is_cs(o, trained_odor)), None)
    if odor is None:
        return None
    t = table[table["odor"] == odor].sort_values("hours")
    hours = list(t["hours"])
    labels = [f"{h:g} h\nn={int(n)}" for h, n in zip(t["hours"], t["n_flies"])]
    rng = np.random.default_rng(0)

    with plt.rc_context(_RC_CONTEXT):
        fig, (left, right) = plt.subplots(1, 2, figsize=(9.6, 4.3))
        sub = paired[paired["odor"] == odor]
        for group, g in sub.groupby("group"):
            h = hours_from_label(group)
            d = (g["score_post"] - g["score_pre"]).to_numpy(dtype=float)
            left.scatter(h + rng.uniform(-0.7, 0.7, d.size), d, s=18, color=DOT_COLOR,
                         edgecolors="white", linewidths=0.8, zorder=2)
        _delta_panel(left, t)
        left.set_ylim(-6, 6)
        left.set_yticks(range(-6, 7, 2))
        left.set_ylabel("Change in PER score (post − pre)")
        left.set_title("PER score change per fly", loc="left", color=INK)
        _p_text(left, "Kruskal-Wallis", _omnibus(omnibus, odor, "p_kw_delta_score"))

        _rate_panel(right, t)
        right.set_ylabel("PER response % (score ≥ 2)")
        right.set_title("Flies responding", loc="left", color=INK)
        right.legend(loc="upper right", frameon=False)
        _p_text(right, "post-training FFH", _omnibus(omnibus, odor, "p_ffh_post_rate"))
        for ax in (left, right):
            _hour_axis(ax, hours, labels=labels)
            ax.set_xlabel("Starvation (h)")
            _recessive_grid(ax)
        fig.suptitle(f"{title} — {odor} (trained odor, CS+): effect of starvation",
                     x=0.01, ha="left", fontsize=10.5, color=INK)
        fig.tight_layout()
    return fig


def build_emphasis(table: pd.DataFrame, *, omnibus: pd.DataFrame | None, trained_odor: str,
                   title: str):
    """Every odor on one axis per measure; the trained odor inked, the rest gray.

    Left: change in PER score (CS+ with mean ± SEM). Right: change in % flies
    responding (post − pre, percentage points). Two measures, two panels.
    """
    odors = sorted(table["odor"].unique(), key=str.casefold)
    hours = sorted(table["hours"].unique())
    cs = next((o for o in odors if _is_cs(o, trained_odor)), None)
    n_labels = [f"{h:g} h\nn={int(table.loc[table['hours'] == h, 'n_flies'].max())}"
                for h in hours]

    with plt.rc_context(_RC_CONTEXT):
        fig, (left, right) = plt.subplots(1, 2, figsize=(10.4, 4.4))
        for ax, key, ylabel in (
            (left, "mean_delta", "Change in PER score (post − pre)"),
            (right, "delta_pct", "Change in flies responding\n(post − pre, percentage points)"),
        ):
            ax.axhline(0.0, **ZERO_LINE)
            for odor in odors:
                if odor == cs:
                    continue
                t = table[table["odor"] == odor].sort_values("hours")
                ax.plot(t["hours"], t[key], "-", color=CONTEXT_COLOR, linewidth=1.2,
                        solid_joinstyle="round", zorder=2)
            if cs is not None:
                t = table[table["odor"] == cs].sort_values("hours")
                if key == "mean_delta":
                    ax.errorbar(t["hours"], t[key], yerr=t["sem_delta"], fmt="none",
                                ecolor=EMPHASIS_COLOR, elinewidth=1.0, capsize=3, zorder=3)
                ax.plot(t["hours"], t[key], "-", color=EMPHASIS_COLOR, linewidth=LINE_W,
                        marker="o", markerfacecolor=EMPHASIS_COLOR, solid_joinstyle="round",
                        zorder=4, **MARKER)
                # Direct label at the line end; the gray lines share one label.
                end = t.iloc[-1]
                ax.annotate(f"{cs} (CS+)", (end["hours"], end[key]), xytext=(6, 0),
                            textcoords="offset points", va="center", fontsize=8.5,
                            color=INK, fontweight="bold", annotation_clip=False)
            ax.set_ylabel(ylabel)
            _hour_axis(ax, hours, labels=n_labels)
            ax.set_xlim(min(hours) - 2, max(hours) + 9)  # room for the end label
            ax.set_xlabel("Starvation (h)")
            _recessive_grid(ax)
        left.set_title("PER score change", loc="left", color=INK)
        right.set_title("Response-rate change", loc="left", color=INK)
        left.set_ylim(-3, 3)
        right.set_ylim(-60, 60)
        right.set_yticks(range(-60, 61, 20))
        if cs is not None:
            _p_text(left, f"{cs}: Kruskal-Wallis",
                    _omnibus(omnibus, cs, "p_kw_delta_score"))
        handles = [plt.Line2D([], [], color=EMPHASIS_COLOR, linewidth=LINE_W, marker="o",
                              markerfacecolor=EMPHASIS_COLOR, label=f"{cs} (trained, CS+)"),
                   plt.Line2D([], [], color=CONTEXT_COLOR, linewidth=1.2,
                              label=f"other {len(odors) - (cs is not None)} odors (untrained)")]
        right.legend(handles=handles, loc="upper right", frameon=False, fontsize=8)
        fig.suptitle(f"{title} — how starvation changes the effect of training\n"
                     "CS+ shown as mean ± SEM; per-odor detail in starvation_trend_all_odors.png",
                     x=0.01, ha="left", fontsize=10.5, color=INK)
        fig.tight_layout()
    return fig


def main(argv: Sequence[str] | None = None) -> list[Path]:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--comparison-dir", required=True, type=Path,
                   help="Folder holding paired_scores_per_fly.csv (+ stats_omnibus.csv).")
    p.add_argument("--out-dir", type=Path, default=None, help="Default: --comparison-dir.")
    p.add_argument("--trained-odor", default="Isoamyl Acetate")
    p.add_argument("--title", default="IAA sensitivity")
    p.add_argument("--binary-threshold", type=int, default=BINARY_THRESHOLD)
    args = p.parse_args(argv)

    src = args.comparison_dir
    out = args.out_dir or src
    out.mkdir(parents=True, exist_ok=True)
    paired = pd.read_csv(src / "paired_scores_per_fly.csv")
    omnibus_path = src / "stats_omnibus.csv"
    omnibus = pd.read_csv(omnibus_path) if omnibus_path.exists() else None

    table = trend_table(paired, binary_threshold=args.binary_threshold)
    written = []
    csv_path = out / "starvation_trend_table.csv"
    table.to_csv(csv_path, index=False)
    written.append(csv_path)
    for name, fig in (
        ("starvation_trend_emphasis.png",
         build_emphasis(table, omnibus=omnibus, trained_odor=args.trained_odor,
                        title=args.title)),
        ("starvation_trend_all_odors.png",
         build_grid(table, omnibus=omnibus, trained_odor=args.trained_odor, title=args.title)),
        ("starvation_trend_trained_odor.png",
         build_spotlight(table, paired, omnibus=omnibus, trained_odor=args.trained_odor,
                         title=args.title)),
    ):
        if fig is None:
            continue
        path = out / name
        with plt.rc_context(_RC_CONTEXT):
            fig.savefig(path)
        plt.close(fig)
        written.append(path)
    for path in written:
        print(f"[OK] {path}")
    return written


if __name__ == "__main__":
    main()
