#!/usr/bin/env python3
"""Pre-exposure baseline by experiment day: do naive flies respond more or less over time?

Reads ``Data/window_per_fly_per_odor.csv`` from a Sensitivity results folder (the
window flies only) and draws, per day:

* A  all cohorts and odors pooled, with every batch's own value behind it;
* B  per cohort;  C  per rig  (cohort mix and rig 1 change with the date);
* D  odor x day heat map.

Estimates pool trials (sum over fly x odor trials / trials, never a mean of
per-fly means); intervals treat the fly as the unit (``fly_stats``). The trend test
is batch-level: one pooled value per batch folder, Spearman against the date, p by
permuting batch values (all batches, then only within rig, then within cohort x
rig), because start date, rig and cohort are batch properties.

Hand-run; reads existing CSVs, needs no pipeline rebuild:

    python3 scripts/analysis/pre_baseline_by_day.py \
        --results "Results/Sensitiviy-Results(11-15)"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
for p in (REPO, REPO / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from fbpipe.analysis import fly_stats as fs  # noqa: E402

THRESHOLD = 2
# Same slots as sensitivity_window_results (cohort by configured position, rigs 5-7),
# copied rather than imported so this hand-run script never pulls the 3.7k-line module.
COHORT_PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                  "#e87ba4", "#008300", "#6250d6", "#e34948")
COHORT_MARKERS = ("o", "s", "D", "^", "v", "P", "X", "h")
RIG_PALETTE = {1: "#e87ba4", 2: "#008300", 3: "#6250d6"}
RIG_MARKERS = {1: "o", 2: "s", 3: "D"}
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
GAP_DAYS = 3          # a break this long or longer is marked on the day axis


# ── tables ──────────────────────────────────────────────────────────────────
def _with_day(rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows.copy()
    rows["day"] = rows["start_time"].astype(str).str[:10]
    return rows


def _estimate(g: pd.DataFrame, metric: str) -> tuple[float, float, float]:
    if metric == "pct":
        return fs.fly_cluster_rate_ci(g["responded_pre"].astype(bool).to_numpy(),
                                      g["fly_id"].to_numpy())
    return fs.fly_cluster_mean_ci(g["score_pre"].astype(float).to_numpy(),
                                  g["fly_id"].to_numpy())


def day_table(rows: pd.DataFrame, *, metric: str = "score",
              by: str | None = None) -> pd.DataFrame:
    """Per day (and group): pooled-trial value with a fly-cluster 95 % CI."""
    rows = _with_day(rows)
    keys = ["day"] + ([by] if by else [])
    out = []
    for key, g in rows.groupby(keys, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        v, lo, hi = _estimate(g, metric)
        out.append({**dict(zip(keys, key)), "value": v, "lo": lo, "hi": hi,
                    "n_flies": int(g["fly_id"].nunique()), "n_trials": len(g),
                    "n_batches": int(g["batch_folder"].nunique())})
    return pd.DataFrame(out)


def odor_day_table(rows: pd.DataFrame, *, metric: str = "score") -> pd.DataFrame:
    rows = _with_day(rows)
    out = []
    for (odor, day), g in rows.groupby(["odor", "day"], sort=True):
        v = (100.0 * g["responded_pre"].astype(bool).mean() if metric == "pct"
             else float(g["score_pre"].astype(float).mean()))
        out.append({"odor": odor, "day": day, "value": v,
                    "n_flies": int(g["fly_id"].nunique())})
    return pd.DataFrame(out)


def batch_values(rows: pd.DataFrame, metric: str = "score") -> pd.DataFrame:
    """One row per batch folder: pooled-trial value, date, rig, cohort."""
    rows = _with_day(rows)
    col = "responded_pre" if metric == "pct" else "score_pre"
    scale = 100.0 if metric == "pct" else 1.0
    b = (rows.groupby("batch_folder")
         .agg(day=("day", "min"), rig=("rig", "first"), cohort=("cohort", "first"),
              value=(col, lambda s: scale * s.astype(float).mean()),
              n_flies=("fly_id", "nunique"))
         .reset_index())
    b["day_num"] = (pd.to_datetime(b["day"]) - pd.Timestamp("2026-01-01")).dt.days
    return b.sort_values(["day", "batch_folder"], ignore_index=True)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = rankdata(x), rankdata(y)
    if rx.std() == 0 or ry.std() == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def _perm_p(x: np.ndarray, y: np.ndarray, strata: np.ndarray, *, n_perm: int,
            seed: int) -> tuple[float, float]:
    """Spearman of x vs y; two-sided p permuting y within strata, (hits+1)/(n+1)."""
    y = y.astype(float).copy()
    for s in np.unique(strata):            # centre within stratum: a rig level is not a trend
        m = strata == s
        y[m] -= y[m].mean()
    rho = _spearman(x, y)
    rng = np.random.default_rng(seed)
    idx = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    hits = 0
    yp = y.copy()
    for _ in range(int(n_perm)):
        for ix in idx:
            yp[ix] = y[rng.permutation(ix)]
        hits += abs(_spearman(x, yp)) >= abs(rho) - 1e-12
    return rho, (hits + 1) / (int(n_perm) + 1)


def batch_trend(rows: pd.DataFrame, *, metric: str = "score", n_perm: int = 20000,
                seed: int = 0) -> dict:
    """Batch-level date trend: unadjusted, within rig, within cohort x rig."""
    b = batch_values(rows, metric)
    x, y = b["day_num"].to_numpy(float), b["value"].to_numpy(float)
    rho, p = _perm_p(x, y, np.zeros(len(b)), n_perm=n_perm, seed=seed)
    rho_r, p_r = _perm_p(x, y, b["rig"].astype(str).to_numpy(), n_perm=n_perm, seed=seed)
    cr = (b["cohort"].astype(str) + "|" + b["rig"].astype(str)).to_numpy()
    rho_cr, p_cr = _perm_p(x, y, cr, n_perm=n_perm, seed=seed)
    return {"metric": metric, "n_batches": len(b), "n_perm": int(n_perm),
            "rho": rho, "p": p, "rho_within_rig": rho_r, "p_within_rig": p_r,
            "rho_within_cohort_rig": rho_cr, "p_within_cohort_rig": p_cr}


def odor_trends(rows: pd.DataFrame, *, metric: str = "score", n_perm: int = 20000,
                seed: int = 0) -> pd.DataFrame:
    """``batch_trend`` per odor, Holm across odors on each p column."""
    out = [{"odor": o, **batch_trend(g, metric=metric, n_perm=n_perm, seed=seed)}
           for o, g in rows.groupby("odor", sort=True)]
    t = pd.DataFrame(out)
    for col in ("p", "p_within_rig", "p_within_cohort_rig"):
        t[f"{col}_holm"] = fs.holm_adjust(t[col].to_numpy(float))
    return t


# ── plot ────────────────────────────────────────────────────────────────────
def _fmt_p(p: float) -> str:
    return "p<0.001" if p < 0.001 else f"p={p:.3f}" if p < 0.1 else f"p={p:.2f}"


def _day_label(d: str) -> str:
    return pd.Timestamp(d).strftime("%b %d")


def _style_axis(ax, days: Sequence[str], ylabel: str, ylim) -> None:
    ax.set_xlim(-0.6, len(days) - 0.4)
    ax.set_ylim(*ylim)
    ax.set_xticks(range(len(days)))
    ax.set_xticklabels([_day_label(d) for d in days], fontsize=8.5, color=INK, rotation=45,
                       ha="right")
    ax.set_ylabel(ylabel, fontsize=10, color=INK)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=8.5)
    dts = pd.to_datetime(list(days))
    for i in range(1, len(days)):
        gap = (dts[i] - dts[i - 1]).days
        if gap >= GAP_DAYS:
            ax.axvline(i - 0.5, color=INK_2, lw=0.8, ls=(0, (2, 3)))
            ax.text(i - 0.5, ylim[1], f" {gap}-day gap", fontsize=7.5, color=INK_2,
                    va="top", ha="left")


def _groups_panel(ax, rows, days, *, by, groups, style, metric, ylim, overall):
    pos = {d: i for i, d in enumerate(days)}
    k = len(groups)
    width = 0.62
    for j, g in enumerate(groups):
        t = day_table(rows[rows[by] == g], metric=metric)
        if t.empty:
            continue
        colour, marker = style[g]
        off = (j - (k - 1) / 2) * width / max(k - 1, 1)
        x = t["day"].map(pos).to_numpy(float) + off
        ax.vlines(x, t["lo"], t["hi"], color=colour, lw=1.4, alpha=0.75, zorder=2)
        ax.scatter(x, t["value"], marker=marker, s=34, color=colour, edgecolor="white",
                   linewidth=0.8, zorder=3)
    ax.axhline(overall, color=INK_2, ls="--", lw=1.1, zorder=1)
    _style_axis(ax, days, ax.get_ylabel(), ylim)


def plot_by_day(rows: pd.DataFrame, cohorts: Sequence[str], path: Path, *,
                metric: str, title: str, trend: dict,
                odor_trend: pd.DataFrame | None = None) -> None:
    rows = _with_day(rows)
    days = sorted(rows["day"].unique())
    pos = {d: i for i, d in enumerate(days)}
    pooled = day_table(rows, metric=metric)
    overall = _estimate(rows, metric)[0]
    ylabel = "Pre-exposure\n% responding" if metric == "pct" else "Pre-exposure\nmean score"
    if metric == "pct":
        top = max(60.0, float(np.nanmax(pooled["hi"])) + 8)
        ylim = (0, top)
    else:
        top = max(2.0, float(np.nanmax(pooled["hi"])) + 0.3)
        ylim = (0, top)

    fig = plt.figure(figsize=(max(14.0, 0.75 * len(days) + 3), 17.5))
    gs = fig.add_gridspec(4, 2, height_ratios=[1.15, 1, 1, 1.05], width_ratios=[60, 1],
                          hspace=0.62, wspace=0.02)
    ax1, ax2, ax3 = (fig.add_subplot(gs[i, 0]) for i in range(3))
    ax4, cax = fig.add_subplot(gs[3, 0]), fig.add_subplot(gs[3, 1])

    # A: pooled, with each batch's own value (the unit of the trend test) behind
    b = batch_values(rows, metric)
    rng = np.random.default_rng(0)
    bx = b["day"].map(pos).to_numpy(float) + rng.uniform(-0.18, 0.18, len(b))
    ax1.scatter(bx, b["value"], s=8 + 6 * b["n_flies"], color="#b9b8b3", edgecolor="white",
                linewidth=0.6, zorder=1, label="one batch (size = flies)")
    x = pooled["day"].map(pos).to_numpy(float)
    ax1.vlines(x, pooled["lo"], pooled["hi"], color=INK, lw=1.6, zorder=3)
    ax1.scatter(x, pooled["value"], s=46, color=INK, edgecolor="white", linewidth=0.9,
                zorder=4, label="day, all cohorts (95% CI, flies as unit)")
    ax1.axhline(overall, color=INK_2, ls="--", lw=1.1, zorder=2, label="all days together")
    for xi, (_, r) in zip(x, pooled.iterrows()):
        ax1.text(xi, ylim[0] + 0.015 * (ylim[1] - ylim[0]), f"{r['n_flies']}",
                 ha="center", va="bottom", fontsize=7.5, color=INK_2)
    _style_axis(ax1, days, ylabel, ylim)
    ax1.legend(loc="upper left", bbox_to_anchor=(0, 1.16), ncol=3, frameon=False,
               fontsize=9, handletextpad=0.4, columnspacing=1.6)
    t = trend
    ax1.set_title(
        f"A  All cohorts, all odors pooled (numbers at the bottom = flies that day)\n"
        f"Batch-level trend over dates ({t['n_batches']} batches): Spearman ρ={t['rho']:+.2f}, "
        f"{_fmt_p(t['p'])};  within rig ρ={t['rho_within_rig']:+.2f}, "
        f"{_fmt_p(t['p_within_rig'])};  within cohort x rig "
        f"ρ={t['rho_within_cohort_rig']:+.2f}, {_fmt_p(t['p_within_cohort_rig'])}",
        loc="left", fontsize=10, color=INK, pad=30, linespacing=1.5)

    # B: per cohort
    present = [c for c in cohorts if (rows["cohort"] == c).any()]
    cstyle = {c: (COHORT_PALETTE[i % len(COHORT_PALETTE)], COHORT_MARKERS[i % len(COHORT_MARKERS)])
              for i, c in enumerate(cohorts)}
    ax2.set_ylabel(ylabel)
    full = (0, 102) if metric == "pct" else (0, 3.05)
    _groups_panel(ax2, rows, days, by="cohort", groups=present, style=cstyle, metric=metric,
                  ylim=full, overall=overall)
    nf = rows.groupby("cohort")["fly_id"].nunique()
    ax2.legend(handles=[Line2D([], [], ls="none", marker=cstyle[c][1], color=cstyle[c][0],
                               markersize=7, label=f"{c} (n={nf[c]})") for c in present],
               loc="upper left", bbox_to_anchor=(0, 1.16), ncol=len(present), frameon=False,
               fontsize=9, handletextpad=0.3, columnspacing=1.4)
    ax2.set_title("B  By cohort (95% CI, flies as unit)", loc="left", fontsize=11,
                  color=INK, pad=24)

    # C: per rig
    rigs = sorted(int(r) for r in rows["rig"].dropna().unique())
    rstyle = {r: (RIG_PALETTE.get(r, INK), RIG_MARKERS.get(r, "o")) for r in rigs}
    ax3.set_ylabel(ylabel)
    _groups_panel(ax3, rows, days, by="rig", groups=rigs, style=rstyle, metric=metric,
                  ylim=full, overall=overall)
    nr = rows.groupby("rig")["fly_id"].nunique()
    ax3.legend(handles=[Line2D([], [], ls="none", marker=rstyle[r][1], color=rstyle[r][0],
                               markersize=7, label=f"Rig {r} (n={nr[r]})") for r in rigs],
               loc="upper left", bbox_to_anchor=(0, 1.16), ncol=len(rigs), frameon=False,
               fontsize=9, handletextpad=0.3, columnspacing=1.4)
    ax3.set_title("C  By rig (Rig 1 = unsuffixed folders, Flybehavior2)", loc="left",
                  fontsize=11, color=INK, pad=24)

    # D: odor x day
    od = odor_day_table(rows, metric=metric)
    odors = sorted(od["odor"].unique())
    grid = np.full((len(odors), len(days)), np.nan)
    for _, r in od.iterrows():
        grid[odors.index(r["odor"]), pos[r["day"]]] = r["value"]
    vmax = 100.0 if metric == "pct" else 3.0
    im = ax4.imshow(np.ma.masked_invalid(grid), cmap="Blues", vmin=0, vmax=vmax,
                    aspect="auto")
    for i in range(len(odors)):
        for j in range(len(days)):
            v = grid[i, j]
            if np.isfinite(v):
                ax4.text(j, i, f"{v:.0f}" if metric == "pct" else f"{v:.1f}",
                         ha="center", va="center", fontsize=7.5,
                         color="white" if v / vmax > 0.55 else INK)
    ax4.set_xticks(range(len(days)))
    ax4.set_xticklabels([_day_label(d) for d in days], fontsize=8.5, color=INK, rotation=45,
                        ha="right")
    ax4.set_yticks(range(len(odors)))
    labels = list(odors)
    if odor_trend is not None and not odor_trend.empty:
        ot = odor_trend.set_index("odor")
        labels = [f"{o}\nρ={ot.at[o, 'rho']:+.2f}  {_fmt_p(ot.at[o, 'p'])}  "
                  f"(within rig {_fmt_p(ot.at[o, 'p_within_rig'])})"
                  f"{'  ' + fs.stars(ot.at[o, 'p_within_rig_holm']) if ot.at[o, 'p_within_rig_holm'] < 0.05 else ''}"
                  if o in ot.index else o for o in odors]
    ax4.set_yticklabels(labels, fontsize=8.5, color=INK, linespacing=1.2)
    ax4.tick_params(length=0)
    for side in ax4.spines.values():
        side.set_visible(False)
    fig.colorbar(im, cax=cax).ax.tick_params(labelsize=8, colors=INK_2)
    what = "% responding" if metric == "pct" else "mean score"
    ax4.set_title(f"D  Each odor by day ({what}; one trial per fly per odor, flies "
                  "as in panel A)\nLeft: each odor's batch-level date trend, uncorrected p; "
                  "a star only if Holm over the odors (within rig) < 0.05", loc="left", fontsize=11, color=INK, pad=8)

    r1 = sorted(rows.loc[rows["rig"] == 1, "day"].unique())
    rig1_days = ", ".join(_day_label(d) for d in r1) if r1 else "no day"
    fig.suptitle(title, fontsize=13, color=INK, y=0.995)
    est = ("% of fly x odor trials scoring ≥2" if metric == "pct"
           else "mean score over fly x odor trials")
    fig.text(0.5, 0.977,
             f"Before exposure every fly is naive and gets the same panel. Value: {est} "
             "(trials pooled, never a mean of fly means). Dashed line: all days together.\n"
             "Trend p: one value per batch, Spearman against the date, batch values "
             f"permuted ({t['n_perm']:,}×; within rig / within cohort x rig after centring "
             "each stratum). Uncorrected, exploratory.\n"
             f"Cohort and rig mix change with the date (Rig 1 runs only on {rig1_days}), "
             "so read panel A against B and C.",
             ha="center", va="top", fontsize=9, color=INK_2, linespacing=1.45)
    fig.subplots_adjust(top=0.895, bottom=0.04, left=0.165, right=0.97)
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


# ── driver ──────────────────────────────────────────────────────────────────
DEFAULT_COHORTS = ("IAA-Sensitivity-1-1", "Hex-Sensitivity-1-0.1",
                   "3Oct-Sensitivity-1-0.1", "Paraffin-Sensitivity-1-100")


def write(results: Path, *, out_dir: Path | None = None,
          cohorts: Sequence[str] = DEFAULT_COHORTS, n_perm: int = 20000,
          seed: int = 0) -> list[Path]:
    rows = pd.read_csv(Path(results) / "Data" / "window_per_fly_per_odor.csv")
    if "in_window" in rows.columns:
        rows = rows[rows["in_window"].astype(bool)]
    cohorts = list(cohorts) + sorted(set(rows["cohort"]) - set(cohorts))
    out_dir = Path(out_dir) if out_dir else Path(results) / "Pre-Exposure-Baseline-By-Day"
    # a sibling of Pre-Exposure-Baseline/: that folder is staged and swapped whole on
    # every sensitivity build, which would delete these
    out_dir.mkdir(parents=True, exist_ok=True)
    span = f"{_day_label(rows['start_time'].min()[:10])}–{_day_label(rows['start_time'].max()[:10])}"
    written, trends, tables = [], [], []
    for metric, suffix, what in (("pct", "", "% responding"), ("score", "_score", "mean score")):
        trend = batch_trend(rows, metric=metric, n_perm=n_perm, seed=seed)
        trends.append(trend)
        otr = odor_trends(rows, metric=metric, n_perm=n_perm, seed=seed)
        otr.to_csv(out_dir / f"pre_exposure_by_day_trend_per_odor{suffix}.csv", index=False)
        for by in (None, "cohort", "rig"):
            t = day_table(rows, metric=metric, by=by)
            t.insert(0, "metric", metric)
            t.insert(1, "group_by", by or "all")
            tables.append(t)
        path = out_dir / f"pre_exposure_by_day{suffix}.png"
        plot_by_day(rows, cohorts, path, metric=metric, trend=trend, odor_trend=otr,
                    title=f"Pre-exposure baseline by day, {what}: window flies, {span}")
        written.append(path)
    pd.concat(tables, ignore_index=True).to_csv(out_dir / "pre_exposure_by_day.csv",
                                                index=False)
    pd.DataFrame(trends).to_csv(out_dir / "pre_exposure_by_day_trend.csv", index=False)
    return written


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", required=True, help="a Sensitivity results folder")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--n-perm", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    for p in write(Path(args.results), out_dir=args.out_dir and Path(args.out_dir),
                   n_perm=args.n_perm, seed=args.seed):
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
