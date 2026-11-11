#!/usr/bin/env python3
"""Recording day vs recording day, every dataset on one page.

Reads the two-group folders ``starvation_pre_post_comparison.py`` wrote with
``--file-tag day`` (``summary_by_group.csv``, ``paired_scores_per_fly.csv``,
``stats_between_groups.csv``) and draws, per dataset (row) and measure (column:
pre-test score, post-training score, change after training), one dumbbell per
odor: the earlier day's mean (light) and the later day's (dark), joined by a
thin rule. A star marks an odor whose two days differ (Mann-Whitney U on the
per-fly values, uncorrected) -- the same p-values as the dataset's own
``day_compared_within_phase`` / ``change_post_minus_pre`` figures.

Colour follows the comparison sets: blue = pre-test, orange = post-training,
gray = change; lighter = the earlier day.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis import phase_terms  # noqa: E402
from scripts.analysis.day_vs_day_report import CSV, odor_concentrations, odor_text  # noqa: E402
from scripts.analysis.pretest_vs_test_comparison import (  # noqa: E402
    INK, _RC_CONTEXT, _recessive_grid, _sem)
from scripts.analysis.significance_brackets import stars  # noqa: E402
from scripts.analysis.starvation_pre_post_comparison import (  # noqa: E402
    DELTA_COLORS, GROUP_COLORS)

MEASURES = (("mean_pre", "sem_pre", "p_pre_score", "Pre-test score"),
            ("mean_post", "sem_post", "p_post_score", "Post-training score"),
            ("mean_delta", "sem_delta", "p_delta_score", "Change (post − pre)"))


def overview_table(comparison_dirs: Sequence[Path]) -> pd.DataFrame:
    """One row per (dataset, odor, day) with means, SEMs and the between-day p."""
    parts = []
    for d in map(Path, comparison_dirs):
        summary = pd.read_csv(d / "summary_by_group.csv")
        paired = pd.read_csv(d / "paired_scores_per_fly.csv")
        between = pd.read_csv(d / "stats_between_groups.csv")
        days = list(dict.fromkeys(summary["group"]))
        delta = (paired.assign(d=paired["score_post"] - paired["score_pre"])
                 .groupby(["group", "odor"])["d"]
                 .agg(mean_delta="mean", sem_delta=lambda s: _sem(s.to_numpy(dtype=float)))
                 .reset_index())
        t = summary.merge(delta, on=["group", "odor"], how="left")
        t = t.merge(between[["odor", "p_pre_score", "p_post_score", "p_delta_score"]],
                    on="odor", how="left")
        t.insert(0, "dataset", d.parent.name)
        t = t.rename(columns={"group": "day"})
        t["day_index"] = t["day"].map({day: i for i, day in enumerate(days)})
        parts.append(t)
    cols = ["dataset", "day", "day_index", "odor", "n_flies", "mean_pre", "sem_pre",
            "mean_post", "sem_post", "mean_delta", "sem_delta", "p_pre_score",
            "p_post_score", "p_delta_score"]
    return pd.concat(parts, ignore_index=True)[cols]


def page_concentrations(comparison_dirs: Sequence[Path], *,
                        predictions_csv: Path | str) -> dict[str, str]:
    """Each odor's recorded concentration over every fly on the page, for the ticks.

    The comparison folders hold labels with the pins concentration stripped, so it
    is read back from the predictions table, for exactly those flies. Every row
    shares the odor ticks, so one map serves the page: an odor whose flies record
    none, or two, gets none (``day_vs_day_report.odor_concentrations``).
    """
    if not Path(predictions_csv).exists():
        return {}
    from fbpipe.config import canon_fly_number
    from scripts.analysis.envelope_visuals import canonical_dataset

    flies = pd.concat([pd.read_csv(Path(d) / "paired_scores_per_fly.csv",
                                   usecols=["dataset_canon", "fly", "fly_number"])
                       for d in comparison_dirs])
    keys = set(zip(flies["dataset_canon"].astype(str), flies["fly"].astype(str),
                   flies["fly_number"].map(canon_fly_number)))
    preds = pd.read_csv(predictions_csv, usecols=["dataset", "fly", "fly_number", "trial_label"])
    canon = {d: canonical_dataset(d) for d in preds["dataset"].astype(str).str.strip().unique()}
    on_page = [(canon[str(d).strip()], str(f).strip(), canon_fly_number(n)) in keys
               for d, f, n in zip(preds["dataset"], preds["fly"], preds["fly_number"])]
    return odor_concentrations(preds.loc[on_page, "trial_label"])


def build_figure(table: pd.DataFrame, *, trained_odor: str = "Isoamyl Acetate",
                 odor_conc: Mapping[str, str] | None = None):
    datasets = list(dict.fromkeys(table["dataset"]))
    odors = sorted(table["odor"].unique(), key=str.casefold)
    y = np.arange(len(odors))
    score_lo = min(0.0, float(np.floor(table[["mean_pre", "mean_post"]].min().min())))
    score_hi = max(4.0, float(np.ceil(table[["mean_pre", "mean_post"]].max().max())))
    dmax = float(np.ceil(table["mean_delta"].abs().max() * 2) / 2) + 0.5
    # The column titles head every dataset's row, so they say pre-/post-exposure
    # only when every dataset is a *-Sensitivity-* one; a mixed page keeps pre-test.
    cohort = datasets[0] if phase_terms.all_sensitivity(datasets) else None
    titles = (f"{phase_terms.pre_label(cohort, 'Pre-test')} score",
              f"{phase_terms.post_label(cohort, 'Post-training')} score", MEASURES[2][3])

    with plt.rc_context(_RC_CONTEXT):
        fig, axes = plt.subplots(len(datasets), 3, figsize=(13.5, 2.9 * len(datasets) + 0.9),
                                 squeeze=False, sharey=True)
        for r, ds in enumerate(datasets):
            sub = table[table["dataset"] == ds]
            days = list(sub.sort_values("day_index")["day"].unique())
            ns = [int(sub.loc[sub["day"] == d, "n_flies"].max()) for d in days]
            for c, (key, _sem_key, pkey, _label) in enumerate(MEASURES):
                ax = axes[r, c]
                if c == 2:
                    ax.axvline(0, color="#9a9994", linewidth=0.9, zorder=1)
                    colors = DELTA_COLORS
                else:
                    colors = [GROUP_COLORS[i][c] for i in range(2)]
                for i, odor in enumerate(odors):
                    o = sub[sub["odor"] == odor].sort_values("day_index")
                    if len(o) < 2:
                        continue
                    vals = o[key].to_numpy(dtype=float)
                    ax.plot(vals, [i, i], color="#c3c2b7", linewidth=1.6, zorder=2,
                            solid_capstyle="round")
                    for di in range(2):
                        ax.scatter(vals[di], i, s=46, color=colors[di], edgecolors="white",
                                   linewidths=1.2, zorder=3)
                    mark = stars(o[pkey].iloc[0])
                    if mark:
                        ax.text(max(vals) + 0.12, i, mark, va="center", ha="left",
                                fontsize=10, fontweight="bold", color=INK)
                ax.set_xlim(*((-dmax, dmax) if c == 2 else (score_lo - 0.3, score_hi + 0.4)))
                _recessive_grid(ax)
                ax.xaxis.grid(True, color="#e4e4e0", linewidth=0.6)
                ax.yaxis.grid(False)
                if r == 0:
                    ax.set_title(titles[c], loc="left", color=INK, fontsize=10)
                if r == len(datasets) - 1:
                    ax.set_xlabel("Mean PER score" if c < 2 else "Mean change per fly")
            axes[r, 0].set_yticks(y)
            labels = axes[r, 0].set_yticklabels([odor_text(o, cohort, odor_conc) for o in odors])
            for lab, odor in zip(labels, odors):
                if trained_odor and odor.casefold().startswith(trained_odor.casefold()):
                    lab.set_fontweight("bold")
            axes[r, 0].set_ylim(len(odors) - 0.5, -0.5)
            axes[r, 0].set_ylabel(f"{ds}\nlight = {days[0]} (n={ns[0]}) · "
                                  f"dark = {days[1]} (n={ns[1]})", fontsize=8.5, color=INK)
        fig.suptitle("Recording day vs recording day, within each dataset — one dumbbell per odor\n"
                     "light dot = earlier day, dark dot = later day · * p<0.05, ** p<0.01 "
                     "(Mann-Whitney U between days, uncorrected) · n = flies",
                     x=0.01, ha="left", fontsize=10.5, color=INK)
        fig.tight_layout()
    return fig


def main(argv: Sequence[str] | None = None) -> list[Path]:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--comparison-dir", action="append", required=True, type=Path,
                    help="A two-day comparison folder; repeat, one per dataset, in plot order.")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--trained-odor", default="Isoamyl Acetate")
    ap.add_argument("--predictions-csv", type=Path, default=CSV / "model_predictions.csv",
                    help="The labels as recorded, for each odor's concentration in the ticks "
                         "(the comparison folders hold them stripped).")
    args = ap.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    table = overview_table(args.comparison_dir)
    csv = args.out_dir / "day_effect_overview.csv"
    table.to_csv(csv, index=False)
    fig = build_figure(table, trained_odor=args.trained_odor,
                       odor_conc=page_concentrations(args.comparison_dir,
                                                     predictions_csv=args.predictions_csv))
    png = args.out_dir / "day_effect_overview.png"
    with plt.rc_context(_RC_CONTEXT):
        fig.savefig(png)
    plt.close(fig)
    for p in (csv, png):
        print(f"[OK] {p}")
    return [csv, png]


if __name__ == "__main__":
    main()
