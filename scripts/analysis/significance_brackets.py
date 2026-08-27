"""Significance brackets that sit above the labels instead of through them.

Every train-vs-control panel used to pick its bracket height by adding a
hand-tuned constant to the taller bar: ``top + 0.45`` on the score panels,
``top + 22`` on the percent ones. Those constants were guesses about how tall
the value label above the bar would be, and they were wrong often enough that
published figures had "2.55" struck through by the bracket over it.

The height is measurable, so measure it: draw the bars and their labels first,
then ask matplotlib where the ink actually ended up.

Only significant pairs are annotated. A bracket labelled "ns" or "p=1.000"
spends the reader's attention to say nothing happened.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def stars(p: float | None) -> str:
    """``***``/``**``/``*`` for a significant p, ``""`` otherwise.

    Returning empty for non-significant is the whole point: callers skip on
    falsiness, so "ns" never reaches a figure.
    """
    if p is None:
        return ""
    try:
        p = float(p)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _data_top_over(ax, x_lo: float, x_hi: float) -> float:
    """Highest drawn y, in data coords, between two x positions.

    Covers the bars, their error bars and the value labels above them — the
    labels are what the old constants kept clipping.
    """
    fig = ax.get_figure()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    top = -np.inf

    for text in ax.texts:
        if not text.get_text():
            continue
        bb = text.get_window_extent(renderer)
        (tx0, _), (tx1, ty1) = inv.transform(((bb.x0, bb.y0), (bb.x1, bb.y1)))
        if tx0 <= x_hi and tx1 >= x_lo:
            top = max(top, ty1)

    for patch in ax.patches:
        try:
            px0, py0 = patch.get_x(), patch.get_y()
            pw, ph = patch.get_width(), patch.get_height()
        except AttributeError:
            continue
        if px0 <= x_hi and px0 + pw >= x_lo:
            top = max(top, py0 + ph)

    # Error-bar caps and whiskers are Line2Ds; a bracket must clear them too.
    for line in ax.get_lines():
        xd = np.asarray(line.get_xdata(), dtype=float)
        yd = np.asarray(line.get_ydata(), dtype=float)
        if xd.size == 0 or yd.size == 0:
            continue
        inside = (xd >= x_lo) & (xd <= x_hi)
        if inside.any():
            finite = yd[np.isfinite(yd)]
            if finite.size:
                top = max(top, float(np.max(finite)))

    return top if np.isfinite(top) else 0.0


def draw(
    ax,
    x_positions: Sequence[float],
    bar_w: float,
    p_values: Sequence[float | None],
    *,
    fontsize: float = 9,
) -> None:
    """Bracket + stars over each significant pair, clear of everything drawn.

    Call this *after* the bars and their value labels are on the axes — it
    measures them. ``p_values`` is positional: one entry per bar pair, ``None``
    or ``nan`` where there is no test.
    """
    y_lo, y_hi = ax.get_ylim()
    span = float(y_hi - y_lo) or 1.0
    gap = 0.035 * span       # bracket clears the tallest ink by this much
    tick = 0.018 * span      # the little downward tips
    star_pad = 0.008 * span

    highest = -np.inf
    for i, p in enumerate(p_values):
        mark = stars(p)
        if not mark:
            continue

        x_left = x_positions[i] - bar_w / 2
        x_right = x_positions[i] + bar_w / 2
        bracket_y = _data_top_over(ax, x_left - bar_w / 2, x_right + bar_w / 2) + gap

        ax.plot(
            [x_left, x_left, x_right, x_right],
            [bracket_y - tick, bracket_y, bracket_y, bracket_y - tick],
            color="black", linewidth=0.9, clip_on=False, zorder=5,
        )
        ax.text(
            x_positions[i], bracket_y + star_pad, mark,
            ha="center", va="bottom", fontsize=fontsize, fontweight="bold",
            zorder=5,
        )
        highest = max(highest, bracket_y + 3 * star_pad + 0.03 * span)

    # Give the tallest bracket room rather than letting it run off the top.
    if np.isfinite(highest) and highest > y_hi:
        ax.set_ylim(y_lo, highest)
