"""Shared plot helpers for the PER analysis package: one look and one set of rules.

Standards these helpers enforce
-------------------------------
* Every figure is saved three ways under one base name: a PNG at 300 dpi, an SVG (text
  kept as text), and a CSV of exactly the numbers drawn (:func:`save_figure`).
* Rate bars show % of trials responding, pooled over trials (sum / trials, never a mean
  of per-fly means), with a 95 % Wilson CI that counts flies. Score bars show the pooled
  mean +/- SEM over flies. Thin, low-alpha per-fly paired lines sit under the bars
  (:func:`paired_bars`; the numbers come from ``stats.group_summary`` / ``stats.per_fly``).
* Stars come from Holm-adjusted p only: * < 0.05, ** < 0.01, *** < 0.001, and nothing
  otherwise. :func:`add_holm_stars` refuses a table without ``p_holm``.
* The title block's subtitle states n flies, n trials, the test and the correction
  (:func:`title_block`), and the caption says what the stars mean (:func:`caption`).
* One metric keeps the same y-limits within an experiment (:func:`shared_ylim` registry).
* Grid and axes are recessive hairlines. Text is set in ink tokens, never in series
  colours. Two or more series always get a legend. There are no dual axes, and nothing
  figure-specific is hard-coded: odor order and colours come from
  ``per_analysis.config`` (ODOR_ORDER / ODOR_COLORS / CONDITION_COLORS / WINDOW_COLORS),
  imported lazily, or from arguments.

Palette validation
------------------
The dataviz skill's ``scripts/validate_palette.js`` was run on 2026-11-18 in light mode
against surface #fcfcfb. Pass/fail is identical on #ffffff, which these figures use;
contrast differs by about 0.07. The palette is fixed by config; the validator records
how it fares and how the figures compensate.

1. The 8-slot categorical palette (ODOR_COLORS, power-curve series),
   "#2a78d6,#eb6834,#1baf7a,#eda100,#e87ba4,#008300,#6250d6,#e34948":

   * ``--pairs all``: FAIL.
     - Lightness band and chroma floor: PASS.
     - CVD separation: FAIL. The worst pair is #008300 / #eb6834 at dE 3.2 (protan;
       tritan 5.1).
     - Normal-vision floor: FAIL. The worst pair is #e34948 / #eb6834 at dE 7.1.
     - Contrast: WARN. #1baf7a is 2.74:1, #eda100 2.11:1 and #e87ba4 2.62:1.
   * Adjacent pairs in the fixed order (the default mode): ALL CHECKS PASS.
     - Worst CVD pair: #eda100 / #1baf7a at dE 9.1.
     - Worst normal-vision pair: #e87ba4 / #eda100 at dE 19.6.
     - Contrast WARN as above.

   Compensation: hue is never the only channel for odor.
     - Bars place each odor at its own x position under its own tick label.
     - Line plots carry a legend, and :func:`power_plot` refuses a 9th series instead of
       generating a hue.
     - Slots are assigned in fixed order. A plot needing all 8 odor hues at once is
       faceted or labelled directly.
     - The contrast WARN slots never carry text, and the CSV is the table view.

2. Condition colours (CONDITION_COLORS), "#52514e,#2a78d6,#1baf7a,#eb6834": FAIL.
   * Chroma floor: FAIL, because #52514e has chroma 0.005. That gray is intentional:
     the pre/alone baseline is a neutral reference, not a series hue.
   * CVD separation: PASS (worst #eb6834 / #1baf7a at dE 9.2 deutan).
   * Normal-vision floor: PASS (worst #2a78d6 / #52514e at dE 21.8).
   * Contrast: WARN (#1baf7a 2.74:1).

   Compensation: conditions also differ by fixed order within each group (baseline
   always left), and the legend names them.

3. Window colours (WINDOW_COLORS), day #2a78d6 vs evening #6250d6: FAIL.
   * CVD separation: dE 4.5 deutan, below the floor of 6.
   * Normal-vision floor: dE 10.5, below 15.

   Colour cannot carry this pair. Compensation: evening marks always take the 45-degree
   hatch in :data:`WINDOW_HATCHES` (pass ``hatches=WINDOW_HATCHES`` to
   :func:`paired_bars`), plus position, tick labels and a legend.
   ``window_style(window)`` returns colour and hatch together.

Figures are light-mode only (PNG/SVG for papers), so the dark-mode run does not apply.
"""
from __future__ import annotations

import textwrap
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.layout_engine import ConstrainedLayoutEngine
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, PercentFormatter
from matplotlib.transforms import blended_transform_factory

from per_analysis import stats as _st

# ── ink and chrome tokens (dataviz reference palette, light mode) ────────────
SURFACE = "#ffffff"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

#: The fixed 8-slot categorical order; used when config gives no colour for a series.
CATEGORICAL = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
               "#e87ba4", "#008300", "#6250d6", "#e34948")

#: Day vs evening fails CVD on hue alone (dE 4.5), so evening marks always carry a
#: 45-degree hatch.
WINDOW_HATCHES: dict[str, str] = {"day": "", "evening": "////"}

HOLM_CAPTION = ("stars: Holm-adjusted p < 0.05 (*), < 0.01 (**), < 0.001 (***); "
                "no star = not significant after correction")

_DEFAULT_YLIMS: dict[str, tuple[float, float]] = {"rate": (0.0, 100.0), "score": (-1.0, 5.0)}
_YLIMS: dict[tuple[Any, str], tuple[float, float]] = {}

#: Share of the data range added above the shared limit for star brackets.
STAR_HEADROOM = 0.16

_YLABELS = {"rate": "PER (% of trials)", "score": "PER score (mean ± SEM)"}


# ── config (lazy) ────────────────────────────────────────────────────────────
def _cfg(name: str, default: Any = None) -> Any:
    return _st.config_value(name, default)


def odor_order(values: Sequence, order: Sequence | None = None) -> list:
    """``values`` (deduplicated) in ``order`` (default: config ODOR_ORDER); values not in
    the order follow, alphabetically."""
    ref = list(order) if order is not None else list(_cfg("ODOR_ORDER", []) or [])
    present = list(dict.fromkeys(v for v in values if not (isinstance(v, float) and np.isnan(v))))
    return [o for o in ref if o in present] + sorted((v for v in present if v not in ref), key=str)


def odor_colors(colors: Mapping | None = None) -> dict:
    """Odor -> colour: ``colors``, else config ODOR_COLORS, else ODOR_ORDER zipped with
    :data:`CATEGORICAL`."""
    if colors is not None:
        return dict(colors)
    cfg = _cfg("ODOR_COLORS")
    if cfg:
        return dict(cfg)
    return dict(zip(_cfg("ODOR_ORDER", []) or [], CATEGORICAL))


def window_style(window: str) -> dict:
    """Colour and hatch of a clock window. Both are always used together, because
    day/evening hue alone fails CVD."""
    colors = _cfg("WINDOW_COLORS", {}) or {}
    return {"color": colors.get(window, INK_MUTED), "hatch": WINDOW_HATCHES.get(window, "")}


def _series_colors(keys: Sequence, colors: Mapping | None, cfg_name: str) -> dict:
    """Colour per key: ``colors``, then config ``cfg_name``, then CATEGORICAL in fixed
    order. Keys match exactly, then as strings (config keys 1 / 2 vs '1' / '2')."""
    src = dict(colors) if colors is not None else dict(_cfg(cfg_name, {}) or {})
    by_str = {str(k): v for k, v in src.items()}
    out = {}
    for i, k in enumerate(keys):
        out[k] = src.get(k, by_str.get(str(k), CATEGORICAL[i % len(CATEGORICAL)]))
    return out


# ── figure, axes, saving ─────────────────────────────────────────────────────
def new_figure(nrows: int = 1, ncols: int = 1, *, figsize: tuple[float, float] | None = None,
               **kwargs: Any) -> tuple[Figure, Any]:
    """``plt.subplots`` with constrained layout and the white surface, which lets
    :func:`title_block` and :func:`caption` reserve their room exactly."""
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, layout="constrained",
                             facecolor=SURFACE, **kwargs)
    return fig, axes


def style_axes(ax: Axes, *, grid: str = "y") -> Axes:
    """Recessive chrome: hairline solid grid on ``grid`` ('x', 'y', 'both' or ''), no
    top/right spines, baseline-grey axes, and tick text in secondary ink."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=BASELINE, labelcolor=INK_SECONDARY, length=3, width=0.8,
                   labelsize=8)
    ax.grid(False)
    if grid:
        ax.grid(True, axis=grid, color=GRID, linewidth=0.6, linestyle="-")
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)
    ax.title.set_color(INK_PRIMARY)
    return ax


def save_figure(fig: Figure, plotted: pd.DataFrame, path_stem: str | Path, *,
                dpi: int = 300, close: bool = True) -> dict[str, Path]:
    """Write ``<stem>.png`` (``dpi``), ``<stem>.svg`` and ``<stem>.csv`` (``plotted``:
    exactly the numbers drawn) side by side, creating parent dirs. SVG text stays text.
    Closes the figure unless ``close=False``. Returns {"png", "svg", "csv"} -> path."""
    stem = Path(path_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = {ext: stem.with_name(f"{stem.name}.{ext}") for ext in ("png", "svg", "csv")}
    with mpl.rc_context({"svg.fonttype": "none"}):
        fig.savefig(paths["png"], dpi=dpi, facecolor=fig.get_facecolor())
        fig.savefig(paths["svg"], facecolor=fig.get_facecolor())
    pd.DataFrame(plotted).to_csv(paths["csv"], index=False)
    if close:
        plt.close(fig)
    return paths


# ── shared y-limits ──────────────────────────────────────────────────────────
def _nice_limits(values: Sequence[float], include_zero: bool = True) -> tuple[float, float]:
    v = np.asarray(values, dtype=float).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        raise ValueError("no finite values to derive limits from")
    lo, hi = float(v.min()), float(v.max())
    if include_zero:
        lo, hi = min(lo, 0.0), max(hi, 0.0)
    if lo == hi:
        hi = lo + 1.0
    ticks = MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]).tick_values(lo, hi)
    return float(ticks[0]), float(ticks[-1])


def register_ylim(metric: str, experiment: Any, lims: tuple[float, float] | None = None, *,
                  data: Sequence[float] | None = None, include_zero: bool = True,
                  overwrite: bool = False) -> tuple[float, float]:
    """Fix ``metric``'s y-limits for every figure of ``experiment``: explicit ``lims``,
    or nice limits around ``data`` (call once per experiment with ALL of its plotted
    values, including per-fly points). Re-registering the same limits is a no-op; a
    different pair raises unless ``overwrite``."""
    if lims is None:
        if data is None:
            raise ValueError("register_ylim needs lims or data")
        lims = _nice_limits(data, include_zero=include_zero)
    lims = (float(lims[0]), float(lims[1]))
    key = (experiment, metric)
    if key in _YLIMS and _YLIMS[key] != lims and not overwrite:
        raise ValueError(f"{metric!r} limits for {experiment!r} already set to {_YLIMS[key]}, "
                         f"not {lims}: one metric keeps one scale within an experiment")
    _YLIMS[key] = lims
    return lims


def shared_ylim(metric: str, experiment: Any = None) -> tuple[float, float]:
    """The registered limits of ``metric`` in ``experiment``, else the fixed defaults
    (rate 0-100 %, score -1..5). An unregistered other metric raises KeyError."""
    if (experiment, metric) in _YLIMS:
        return _YLIMS[(experiment, metric)]
    if metric in _DEFAULT_YLIMS:
        return _DEFAULT_YLIMS[metric]
    raise KeyError(f"no y-limits registered for {metric!r} in {experiment!r}; "
                   "call register_ylim(metric, experiment, data=...) once first")


def reset_ylims(experiment: Any = None) -> None:
    """Forget registered limits (all, or one experiment's)."""
    for key in [k for k in _YLIMS if experiment is None or k[0] == experiment]:
        del _YLIMS[key]


def _apply_ylim(ax: Axes, lims: tuple[float, float], drawn: np.ndarray) -> None:
    lo, hi = lims
    span = hi - lo
    ax.set_ylim(lo, hi + STAR_HEADROOM * span)
    ticks = MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]).tick_values(lo, hi)
    ax.set_yticks([t for t in ticks if lo - 1e-9 <= t <= hi + 1e-9])
    ax.spines["left"].set_bounds(lo, hi)
    if lo < 0 < hi:                     # bars grow from 0, not from the axis floor
        ax.axhline(0.0, color=BASELINE, linewidth=0.8, zorder=1.6)
    d = drawn[np.isfinite(drawn)]
    if d.size and (d.min() < lo - 1e-9 or d.max() > hi + 1e-9):
        warnings.warn(f"plotted values span {d.min():.3g}..{d.max():.3g}, outside the shared "
                      f"limits {lims}; register limits from all of the experiment's data",
                      stacklevel=3)


# ── bars ─────────────────────────────────────────────────────────────────────
def paired_bars(ax: Axes, table: pd.DataFrame, *, conditions: Sequence, metric: str,
                groups: Sequence | None = None, per_fly: pd.DataFrame | None = None,
                colors: Mapping | None = None, group_col: str = "odor",
                condition_col: str = "condition", fly_col: str = "fly_id",
                per_fly_col: str | None = None, order: Sequence | None = None,
                condition_labels: Mapping | None = None, group_labels: Mapping | None = None,
                hatches: Mapping | None = None, experiment: Any = None,
                ylabel: str | None = None, legend: bool = True,
                width: float = 0.8) -> pd.DataFrame:
    """Grouped bars: ``groups`` along x, one bar per condition inside each group.

    ``table``: ``stats.group_summary`` output, so one row per (group, condition) with
    ``value``, ``err_lo`` and ``err_hi`` in plotted units (rate: %, Wilson CI; score:
    pooled mean, +/- SEM). ``per_fly``: ``stats.per_fly`` output, one within-fly mean
    per (group, condition, fly). Rate values arrive as fractions and are drawn x100. Each
    fly becomes a thin, low-alpha line joining its conditions, under the bars.

    ``groups`` defaults to the table's groups in ``order`` (default config ODOR_ORDER).
    ``colors`` defaults to config CONDITION_COLORS. Y-limits come from
    :func:`shared_ylim` (``experiment``), with headroom above for star brackets. Returns
    the plotted numbers: ``layer`` = 'bar' rows (x, value, err_lo, err_hi, n_flies,
    n_trials) plus 'fly' rows (x, fly, value).
    """
    if metric not in ("rate", "score") and experiment is None:
        raise ValueError("a metric other than rate/score needs an experiment with "
                         "registered y-limits")
    conditions = list(conditions)
    k = len(conditions)
    groups = list(groups) if groups is not None else odor_order(table[group_col], order)
    cmap = _series_colors(conditions, colors, "CONDITION_COLORS")
    hatches = dict(hatches or {})
    labels = {c: (condition_labels or {}).get(c, str(c)) for c in conditions}
    slot = width / max(k, 1)
    xpos = {(g, c): gi + (ci - (k - 1) / 2) * slot
            for gi, g in enumerate(groups) for ci, c in enumerate(conditions)}
    tbl = table.set_index([group_col, condition_col])
    style_axes(ax, grid="y")

    rows: list[dict] = []
    drawn: list[float] = []
    for c in conditions:
        xs, vs, los, his = [], [], [], []
        for g in groups:
            if (g, c) not in tbl.index:
                continue
            r = tbl.loc[(g, c)]
            r = r.iloc[0] if isinstance(r, pd.DataFrame) else r
            xs.append(xpos[(g, c)])
            vs.append(float(r["value"]))
            los.append(float(r["err_lo"]))
            his.append(float(r["err_hi"]))
            rows.append({"layer": "bar", group_col: g, condition_col: c, "x": xpos[(g, c)],
                         "value": float(r["value"]), "err_lo": float(r["err_lo"]),
                         "err_hi": float(r["err_hi"]), "interval": r.get("interval", ""),
                         "n_flies": r.get("n_flies", np.nan),
                         "n_trials": r.get("n_trials", np.nan), "metric": metric,
                         "color": cmap[c]})
        if not xs:
            continue
        v, lo, hi = (np.asarray(a, float) for a in (vs, los, his))
        ax.bar(xs, v, width=slot * 0.86, color=cmap[c], hatch=hatches.get(c) or None,
               edgecolor=SURFACE, linewidth=0, zorder=2, label=labels[c])
        yerr = np.vstack([np.clip(v - lo, 0, None), np.clip(hi - v, 0, None)])
        ok = np.isfinite(yerr).all(axis=0)
        if ok.any():
            ax.errorbar(np.asarray(xs)[ok], v[ok], yerr=yerr[:, ok], fmt="none",
                        ecolor=INK_PRIMARY, elinewidth=1.0, capsize=2.5, capthick=1.0,
                        zorder=4)
        drawn += list(lo) + list(hi) + list(v)

    if per_fly is not None and k >= 2:
        col = per_fly_col or next((c for c in ("value", "responded" if metric == "rate"
                                               else "score") if c in per_fly.columns), None)
        if col is None:
            raise KeyError("per_fly has no 'value' column; pass per_fly_col")
        scale = 100.0 if metric == "rate" else 1.0
        sub = per_fly[per_fly[group_col].isin(groups) & per_fly[condition_col].isin(conditions)]
        wide = sub.pivot_table(index=[group_col, fly_col], columns=condition_col,
                               values=col, aggfunc="first", observed=True)
        segments = []
        for (g, fly), r in wide.iterrows():
            pts = [(xpos[(g, c)], float(r[c]) * scale) for c in conditions
                   if c in r.index and np.isfinite(r[c])]
            if len(pts) < 2:
                continue
            segments.append(pts)
            for (x, y), c in zip(pts, [c for c in conditions
                                       if c in r.index and np.isfinite(r[c])]):
                rows.append({"layer": "fly", group_col: g, condition_col: c, "x": x,
                             "value": y, fly_col: fly, "metric": metric})
                drawn.append(y)
        if segments:
            ax.add_collection(LineCollection(segments, colors=INK_MUTED, linewidths=0.6,
                                             alpha=0.3, zorder=1.5))

    lims = (shared_ylim(metric, experiment) if (experiment is not None or metric in _DEFAULT_YLIMS)
            else _nice_limits(drawn))
    _apply_ylim(ax, lims, np.asarray(drawn, float))
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([(group_labels or {}).get(g, str(g)) for g in groups])
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_ylabel(ylabel or _YLABELS.get(metric, metric))
    if legend and k >= 2:
        handles = [Patch(facecolor=cmap[c], hatch=hatches.get(c) or None, edgecolor=SURFACE,
                         linewidth=0, label=labels[c]) for c in conditions]
        ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.0),
                  ncols=min(k, 4), frameon=False, fontsize=8, handlelength=1.2,
                  borderaxespad=0.2, labelcolor=INK_SECONDARY)
    return pd.DataFrame(rows)


# ── significance brackets ────────────────────────────────────────────────────
def add_holm_stars(ax: Axes, comparisons: pd.DataFrame | Sequence[Mapping], *,
                   plotted: pd.DataFrame | None = None, group_col: str = "odor",
                   condition_col: str = "condition") -> pd.DataFrame:
    """Brackets with stars from ``p_holm`` ONLY (* < 0.05, ** < 0.01, *** < 0.001; a
    comparison without a star gets no bracket).

    Each comparison row gives either ``x1``, ``x2`` (and optional ``y``), or ``group``
    (or ``group_col``), ``cond_a`` and ``cond_b`` resolved against ``plotted`` (the
    :func:`paired_bars` return). A bracket sits above the bars, error bars and fly
    points it spans, and stacks above any bracket it overlaps. A table without
    ``p_holm`` raises ValueError: raw p never becomes a star. Returns x1, x2, y,
    p_holm, stars per row (y NaN when nothing was drawn).
    """
    comps = pd.DataFrame(comparisons).reset_index(drop=True)
    if "p_holm" not in comps.columns:
        raise ValueError("add_holm_stars needs a 'p_holm' column: stars come only from "
                         "Holm-adjusted p (stats.add_corrections)")
    ylo, yhi = ax.get_ylim()
    span = yhi - ylo
    pad, tick, step = 0.025 * span, 0.015 * span, 0.07 * span
    pts = None
    if plotted is not None and len(plotted):
        pts = np.column_stack([plotted["x"].to_numpy(float),
                               plotted[["value", "err_hi"]].max(axis=1).to_numpy(float)
                               if "err_hi" in plotted.columns
                               else plotted["value"].to_numpy(float)])
    placed: list[tuple[float, float, float]] = []
    out = []
    for _, r in comps.iterrows():
        if "x1" in comps.columns and pd.notna(r.get("x1")):
            x1, x2 = float(r["x1"]), float(r["x2"])
        else:
            if plotted is None:
                raise ValueError("comparisons by group/condition need plotted=")
            bars = plotted[plotted["layer"] == "bar"]
            g = r["group"] if "group" in comps.columns else r[group_col]
            sel = bars[bars[group_col] == g]
            x1 = float(sel.loc[sel[condition_col] == r["cond_a"], "x"].iloc[0])
            x2 = float(sel.loc[sel[condition_col] == r["cond_b"], "x"].iloc[0])
        x1, x2 = min(x1, x2), max(x1, x2)
        s = _st.stars(r["p_holm"])
        rec = {"x1": x1, "x2": x2, "y": np.nan, "p_holm": r["p_holm"], "stars": s}
        if s:
            if "y" in comps.columns and pd.notna(r.get("y")):
                base = float(r["y"])
            elif pts is not None:
                inside = pts[(pts[:, 0] >= x1 - 1e-9) & (pts[:, 0] <= x2 + 1e-9), 1]
                inside = inside[np.isfinite(inside)]
                base = float(inside.max()) + pad if inside.size else ylo + 0.9 * span
            else:
                base = ylo + 0.85 * span
            for a, b, y_used in placed:
                if a <= x2 and x1 <= b:
                    base = max(base, y_used + step)
            ax.plot([x1, x1, x2, x2], [base, base + tick, base + tick, base],
                    color=INK_SECONDARY, linewidth=0.8, clip_on=False, zorder=5)
            ax.text((x1 + x2) / 2, base + tick, s, ha="center", va="bottom",
                    color=INK_PRIMARY, fontsize=9, clip_on=False, zorder=5)
            placed.append((x1, x2, base))
            rec["y"] = base
        out.append(rec)
    return pd.DataFrame(out)


# ── title, subtitle, caption ─────────────────────────────────────────────────
def stats_subtitle(*, n_flies: int, n_trials: int, test: str, correction: str = "Holm",
                   extra: str | None = None) -> str:
    """'n = 12 flies, 96 trials · test: … · correction: …' (+ extra)."""
    parts = [f"n = {n_flies} flies, {n_trials} trials", f"test: {test}",
             f"correction: {correction}"]
    if extra:
        parts.append(extra)
    return " · ".join(parts)


def _wrap(fig: Figure, text: str, fontsize: float) -> str:
    width_pt = fig.get_figwidth() * 72 - 16
    chars = max(20, int(width_pt / (fontsize * 0.6)))   # ~DejaVu Sans mean advance
    return "\n".join(textwrap.fill(line, chars) for line in str(text).splitlines() or [""])


def _reserve(fig: Figure, *, top: float | None = None, bottom: float | None = None) -> None:
    """Keep ``top`` / ``bottom`` figure fractions free for the title block and caption."""
    m = fig.__dict__.setdefault("_per_margins", {"top": 0.0, "bottom": 0.0})
    if top is not None:
        m["top"] = top
    if bottom is not None:
        m["bottom"] = bottom
    eng = fig.get_layout_engine()
    if isinstance(eng, ConstrainedLayoutEngine):
        eng.set(rect=(0.0, m["bottom"], 1.0, 1.0 - m["top"] - m["bottom"]))
    else:
        sp = fig.subplotpars
        fig.subplots_adjust(top=min(sp.top, 1.0 - m["top"] - 0.05),
                            bottom=max(sp.bottom, m["bottom"] + 0.1))


def title_block(fig: Figure, title: str, subtitle: str | None = None, *,
                n_flies: int | None = None, n_trials: int | None = None,
                test: str | None = None, correction: str | None = None) -> None:
    """Left-aligned title (primary ink) over a subtitle (secondary ink) that states n
    flies, n trials, the test and the correction: pass ``subtitle`` or the four keywords
    (:func:`stats_subtitle`). A subtitle without fly and trial counts warns."""
    if n_flies is not None or n_trials is not None or test is not None:
        built = stats_subtitle(n_flies=n_flies, n_trials=n_trials, test=test or "?",
                               correction=correction or "Holm")
        subtitle = f"{subtitle} · {built}" if subtitle else built
    if not subtitle or "flies" not in subtitle or "trials" not in subtitle:
        warnings.warn("title_block: the subtitle should state n flies, n trials, the test "
                      "and the correction", UserWarning, stacklevel=2)
    h_pt = fig.get_figheight() * 72
    w_pt = fig.get_figwidth() * 72
    sub = _wrap(fig, subtitle or "", 8.5)
    n_lines = sub.count("\n") + 1 if subtitle else 0
    top_pt = 6.0
    fig.text(8 / w_pt, 1 - top_pt / h_pt, title, ha="left", va="top", fontsize=11.5,
             fontweight="semibold", color=INK_PRIMARY)
    used = top_pt + 11.5 * 1.3
    if subtitle:
        fig.text(8 / w_pt, 1 - (used + 2) / h_pt, sub, ha="left", va="top", fontsize=8.5,
                 color=INK_SECONDARY, linespacing=1.3)
        used += 2 + n_lines * 8.5 * 1.35
    _reserve(fig, top=(used + 6) / h_pt)


def caption(fig: Figure, text: str | None = None) -> None:
    """Small muted caption along the bottom; defaults to :data:`HOLM_CAPTION`."""
    h_pt = fig.get_figheight() * 72
    w_pt = fig.get_figwidth() * 72
    body = _wrap(fig, text or HOLM_CAPTION, 7.5)
    n_lines = body.count("\n") + 1
    fig.text(8 / w_pt, 5 / h_pt, body, ha="left", va="bottom", fontsize=7.5,
             color=INK_MUTED, linespacing=1.3)
    _reserve(fig, bottom=(5 + n_lines * 7.5 * 1.35 + 4) / h_pt)


# ── forest and power ─────────────────────────────────────────────────────────
def forest_plot(ax: Axes, table: pd.DataFrame, est: str, lo: str, hi: str, label: str, *,
                ref: float = 0.0, xlabel: str | None = None, color: str | None = None,
                p_holm: str | None = None) -> pd.DataFrame:
    """Point estimates with interval whiskers, one row per ``label``, sorted by effect
    (largest at the top), with a hairline at ``ref``. With ``p_holm`` (a column name)
    each row's Holm stars are printed at the right edge in ink. Returns the plotted rows
    with their ``y``."""
    cols = [label, est, lo, hi] + ([p_holm] if p_holm else [])
    t = table[cols].copy().sort_values(est, kind="stable").reset_index(drop=True)
    t["y"] = np.arange(len(t), dtype=float)
    c = color or CATEGORICAL[0]
    style_axes(ax, grid="x")
    ax.axvline(ref, color=BASELINE, linewidth=0.8, zorder=1)
    ax.hlines(t["y"], t[lo], t[hi], color=c, linewidth=1.5, zorder=2)
    ax.scatter(t[est], t["y"], s=36, color=c, edgecolors=SURFACE, linewidths=1.5, zorder=3)
    ax.set_yticks(t["y"])
    ax.set_yticklabels(t[label].astype(str))
    ax.set_ylim(-0.6, len(t) - 0.4)
    if xlabel:
        ax.set_xlabel(xlabel)
    if p_holm:
        t["stars"] = [_st.stars(p) for p in t[p_holm]]
        tr = blended_transform_factory(ax.transAxes, ax.transData)
        for y, s in zip(t["y"], t["stars"]):
            if s:
                ax.text(1.01, y, s, transform=tr, ha="left", va="center", fontsize=9,
                        color=INK_PRIMARY, clip_on=False)
    return t


def power_plot(ax: Axes, table: pd.DataFrame, *, target: float | None = 0.8,
               delta_label: str = "Δ", n_label: str | None = None) -> pd.DataFrame:
    """Power vs n, one line per delta (``stats.power_curve`` output), with a hairline
    at ``target``. Series take :data:`CATEGORICAL` slots in fixed order; more than 8
    deltas raise rather than generate a hue. One design per axes."""
    if "design" in table.columns and table["design"].nunique() > 1:
        raise ValueError("power_plot draws one design per axes; facet paired / two_group")
    deltas = sorted(table["delta"].unique())
    if len(deltas) > len(CATEGORICAL):
        raise ValueError(f"{len(deltas)} deltas > {len(CATEGORICAL)} palette slots: "
                         "facet instead of generating a 9th hue")
    style_axes(ax, grid="y")
    rows = []
    for i, d in enumerate(deltas):
        sub = table[table["delta"] == d].sort_values("n")
        ax.plot(sub["n"], sub["power"], color=CATEGORICAL[i], linewidth=2,
                solid_capstyle="round", marker="o", markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1.2,
                label=f"{delta_label} = {d:g}", zorder=3)
        rows.append(sub.assign(color=CATEGORICAL[i]))
    if target is not None:
        ax.axhline(target, color=INK_MUTED, linewidth=0.8, zorder=1)
        ax.text(1.0, target, f"{target:.0%} power", transform=blended_transform_factory(
            ax.transAxes, ax.transData), ha="right", va="bottom", fontsize=7.5,
            color=INK_MUTED)
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    design = table["design"].iloc[0] if "design" in table.columns and len(table) else "paired"
    ax.set_xlabel(n_label or ("Flies per group" if design == "two_group" else "Flies"))
    ax.set_ylabel("Power")
    if len(deltas) >= 2:
        ax.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY)
    return pd.concat(rows, ignore_index=True) if rows else table.iloc[0:0]


__all__ = (
    "BASELINE", "CATEGORICAL", "GRID", "HOLM_CAPTION", "INK_MUTED", "INK_PRIMARY",
    "INK_SECONDARY", "SURFACE", "WINDOW_HATCHES", "add_holm_stars", "caption",
    "forest_plot", "new_figure", "odor_colors", "odor_order", "paired_bars", "power_plot",
    "register_ylim", "reset_ylims", "save_figure", "shared_ylim", "stats_subtitle",
    "style_axes", "title_block", "window_style",
)
