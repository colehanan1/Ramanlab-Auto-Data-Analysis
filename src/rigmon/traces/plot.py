"""One PNG per trial: every fly's eye–proboscis distance (px) vs time, odor-on shaded."""

from __future__ import annotations

from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

FLY_COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7")  # Okabe–Ito, CVD-safe


def legend_title(n_found: int, expected: int) -> str | None:
    return None if n_found >= expected else f"{n_found} of {expected} flies found"


def render_trace(series, segments, out_png, *, title: str, expected: int = 4) -> Path:
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig = Figure(figsize=(14, 5), dpi=100)  # no pyplot: thread-safe, no global state
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    for on, off, label in segments:
        ax.axvspan(on, off, color="#999999", alpha=0.18, lw=0)
        ax.text((on + off) / 2, 1.0, label, transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=9, color="#555555")
    for fly, values in sorted(series.flies.items()):
        ax.plot(series.t_s, values, color=FLY_COLORS[(fly - 1) % len(FLY_COLORS)],
                lw=1.2, label=f"fly {fly}")
    note = "" if segments else "  (odor timing unavailable)"
    ax.set_title(title + note, loc="left", fontsize=11)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("eye–proboscis distance (px)")
    if series.flies:
        ax.legend(loc="upper right", ncol=4, frameon=False,
                  title=legend_title(series.n_slots, expected), fontsize=9)
    ax.grid(alpha=0.25)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.9, bottom=0.12)
    fig.savefig(out_png, dpi=100, bbox_inches=None)
    return out_png
