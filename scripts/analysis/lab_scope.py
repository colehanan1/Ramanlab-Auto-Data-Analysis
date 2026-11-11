"""Live oscilloscope trace of PER % for the fly-behaviour montages.

Shared by make_per_montage_video.py (training video) and make_repeated_exposure_montage_video.py
(IAA repeated-exposure video). It ports the oscilloscope-stream recipe of
make_per_tracking_scope_clip.py (video-shotcraft card data/chart-live-moves, OscilloscopeStreamV2):

  * the write head is pinned at the plot's right edge; history streams left at a constant
    speed (px/s), so the visible window is plot_w / speed seconds
  * a write-head dot with a soft halo and an afterglow tail behind it
  * the trace switches to the event colour while it is above the PER threshold
  * stimulus bands (odor, light) scroll with the data and are drawn only up to the head
  * no brake and no hold: the stream keeps writing at 1x through the segment's fade-out

Data: PER % is the pipeline's `distance_percentage` column (fbpipe.steps.distance_normalize:
the eye->labellum distance as a percentage of the fly's own min-max range, 0-100), smoothed with
the same centred 3-frame rolling median the montages used for the px trace. It is not
rest-subtracted, so a fly's rest level sits wherever its range puts it (e.g. ~4 % or ~45 %).

PER threshold (event colour): rest median + max(PCT_FLOOR, PCT_NOISE_K x rest SD), rest = the
REST_S seconds before odor onset.

Clip start rule (user 2026-11-02): a clip starts at odor onset, or EARLY_LEAD_S (1 s) before it
when the fly already moves within the first MOVE_WINDOW_S (2 s) of odor. "Moves" = the PER %
exceeds the rest median by more than max(MOVE_FLOOR, MOVE_NOISE_K x rest SD) in that window.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from matplotlib.transforms import blended_transform_factory

# ------------------------------------------------------------ data definitions
PCT_COLUMN = "distance_percentage"
SMOOTH_FRAMES = 3                  # centred rolling median, as the montages' px trace
REST_S = 10.0                      # rest window before odor onset
PCT_FLOOR = 10.0                   # PER threshold: rest median + at least 10 percentage points ...
PCT_NOISE_K = 10.0                 # ... or + 10 rest SD, whichever is larger
MOVE_WINDOW_S = 2.0                # clip start rule: movement within the first 2 s of odor ...
MOVE_FLOOR = 5.0                   # ... = more than 5 points above rest ...
MOVE_NOISE_K = 3.0                 # ... and more than 3 rest SD above rest
EARLY_LEAD_S = 1.0                 # ... then the clip starts 1 s before odor onset

# ------------------------------------------------------------ recipe parameters (scope clip)
SPEED_PX_S = 5.5 * 30              # 165 px/s, the card's perceptible lower bound (5.5 px/f at 30 fps)
HEAD_R_PX = 10.5
HALO_R_PX = 24
TAIL_PX = 160
READOUT_GAIN = 0.32                # readout scale 1 + 0.32 k
K_SPAN = 40.0                      # k reaches 1 at threshold + 40 percentage points
UNIT = "%"
Y_LABEL = "Proboscis Extension (%)"       # card title / y label of every PER % scope
THRESH_LABEL = "PER threshold"


def px_to_pt(px: float) -> float:
    return px * 72 / 100


# ================================================================ pure data helpers
def per_pct(values) -> np.ndarray:
    """PER % trace: distance_percentage, centred 3-frame rolling median."""
    return pd.Series(np.asarray(values, float)).rolling(SMOOTH_FRAMES, center=True, min_periods=1).median().to_numpy()


def rest_stats(v, t_rel) -> tuple[float, float]:
    v, t_rel = np.asarray(v, float), np.asarray(t_rel, float)
    rest = v[(t_rel >= -REST_S) & (t_rel < 0)]
    return float(np.nanmedian(rest)), float(np.nanstd(rest))


def per_threshold(v, t_rel) -> float:
    med, sd = rest_stats(v, t_rel)
    return med + max(PCT_FLOOR, PCT_NOISE_K * sd)


def moves_early(v, t_rel) -> bool:
    """True when the PER % leaves rest within the first MOVE_WINDOW_S of odor onset."""
    v, t_rel = np.asarray(v, float), np.asarray(t_rel, float)
    med, sd = rest_stats(v, t_rel)
    win = v[(t_rel >= 0) & (t_rel < MOVE_WINDOW_S)]
    return bool(win.size and np.nanmax(win) - med > max(MOVE_FLOOR, MOVE_NOISE_K * sd))


def clip_start_rel(v, t_rel) -> float:
    """Start rule: odor onset (0), or -EARLY_LEAD_S when the fly moves within MOVE_WINDOW_S."""
    return -EARLY_LEAD_S if moves_early(v, t_rel) else 0.0


def first_crossing(v, t_rel, thr: float, after: float = 0.0) -> float:
    """First time >= `after` at which the trace is above `thr` (NaN if never)."""
    v, t_rel = np.asarray(v, float), np.asarray(t_rel, float)
    idx = np.where((t_rel >= after) & (v > thr))[0]
    return float(t_rel[idx[0]]) if idx.size else float("nan")


def is_event(v: float, thr: float) -> bool:
    return bool(np.isfinite(v) and v > thr)


def format_readout(v: float) -> str:
    if not np.isfinite(v):
        return "–"
    return str(int(round(float(np.clip(v, 0.0, 100.0)))))


def readout_scale(v: float, thr: float) -> float:
    if not np.isfinite(v):
        return 1.0
    return 1.0 + READOUT_GAIN * float(np.clip((v - thr) / K_SPAN, 0.0, 1.0))


def stream_x(sample_t, head_t: float, plot_w: float, speed: float = SPEED_PX_S):
    """Plot x (px) of a sample at time sample_t when the head (at x = plot_w) is at head_t."""
    return plot_w - (head_t - np.asarray(sample_t, float)) * speed


def band_extent(on_rel: float, off_rel: float, head_t: float, plot_w: float, speed: float = SPEED_PX_S):
    """(x0, x1) of a stimulus band in plot px, drawn only up to the head (never the future)."""
    if head_t < on_rel:
        return None
    return (float(stream_x(on_rel, head_t, plot_w, speed)),
            float(stream_x(min(off_rel, head_t), head_t, plot_w, speed)))


def sticky_x(x0: float, x1: float, label_w: float, plot_w: float, pad: float = 10.0) -> float:
    """Band label x: rides the band's left edge, sticks at the plot's left edge, and is pushed
    out to the left by the band's end once the band has closed."""
    x = max(x0, 0.0) + pad
    if x1 < plot_w - 1e-6:
        x = min(x, x1 - label_w - pad)
    return x


def split_at_threshold(x, y, thr: float):
    """Insert exact threshold crossings; return (x, y_at_or_below, y_at_or_above) with NaN gaps."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    xs, ys = [], []
    for i in range(len(x)):
        if i and (y[i - 1] - thr) * (y[i] - thr) < 0:
            f = (thr - y[i - 1]) / (y[i] - y[i - 1])
            xs.append(x[i - 1] + f * (x[i] - x[i - 1]))
            ys.append(thr)
        xs.append(x[i])
        ys.append(y[i])
    xs, ys = np.array(xs), np.array(ys)
    with np.errstate(invalid="ignore"):
        return xs, np.where(ys <= thr, ys, np.nan), np.where(ys >= thr, ys, np.nan)


def x_tick_seconds(plot_w: float, speed: float = SPEED_PX_S, step: int = 1) -> tuple[int, ...]:
    """Whole seconds back from the head that fit in the window: (N, ..., 1, 0)."""
    return tuple(range(int(plot_w / speed) // step * step, -1, -step))


def x_tick_label(s: int) -> str:
    return f"−{s} s" if s else "now"


# ================================================================ drawing
@dataclass
class Band:
    name: str                       # "odor" | "light"
    color: str
    alpha: float
    label: str                      # "Odor" | "Light"
    region: tuple[float, float]     # (on, off) s from odor onset
    label_region: tuple[float, float] | None = None   # where the label may ride (default: region)


@dataclass
class ScopePanel:
    """Oscilloscope artists on one px axes (x = plot px with the head at plot_w, y = PER %).

    The axes must already be placed (e.g. make_per_montage_video.px_axes); ScopePanel sets its
    limits, ticks, bands, threshold line and trace artists. Call update(t_rel, v, head) per frame.
    """
    ax: object
    plot_w: float
    plot_h: float
    thr: float
    colors: dict                    # keys: idle, event, dot_idle, grid, muted, surface
    bands: list = field(default_factory=list)
    ylim: tuple[float, float] = (-4.0, 104.0)
    yticks: tuple = (0, 50, 100)
    speed: float = SPEED_PX_S
    font_pt: float = 23.04
    band_labels: bool = True        # write band names inside the plot's top lane
    thresh_label: str | None = None # optional label at the threshold line's left end
    x_ticks: bool = True
    line_px: float = 4.0

    def __post_init__(self):
        ax, c = self.ax, self.colors
        ax.set_facecolor("none")
        ax.set_xlim(0, self.plot_w)
        ax.set_ylim(*self.ylim)
        if self.x_ticks:
            secs = x_tick_seconds(self.plot_w, self.speed)
            ax.set_xticks([self.plot_w - s * self.speed for s in secs])
            ax.set_xticklabels([x_tick_label(s) for s in secs])
        else:
            ax.set_xticks([])
        ax.set_yticks(list(self.yticks))
        ax.tick_params(colors=c["muted"], labelsize=self.font_pt, length=4, pad=6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(c["grid"])
        ax.yaxis.grid(True, color=c["grid"], linewidth=1.0)
        ax.xaxis.grid(True, color=c["grid"], linewidth=1.0, alpha=0.6)
        ax.set_axisbelow(True)
        self.thr_line = ax.axhline(self.thr, color=c["event"], linewidth=1.2, linestyle=(0, (5, 4)),
                                   alpha=0.75, zorder=2)
        self.thr_text = None
        if self.thresh_label:
            self.thr_text = ax.text(10, self.thr + 2.0, self.thresh_label, color=c["event"],
                                    fontsize=self.font_pt, va="bottom", zorder=2)
        y0, y1 = self.ylim
        xdata_yaxes = blended_transform_factory(ax.transData, ax.transAxes)
        self.band_patches, self.band_texts = {}, {}
        for b in self.bands:
            self.band_patches[b.name] = ax.add_patch(Rectangle((0, y0), 0, y1 - y0, fc=b.color, ec="none",
                                                               alpha=b.alpha, zorder=1, visible=False))
            if self.band_labels:
                self.band_texts[b.name] = ax.text(0, 1 - 6 / self.plot_h, b.label, transform=xdata_yaxes,
                                                  color=b.color, fontsize=self.font_pt, fontweight="bold",
                                                  va="top", clip_on=True, zorder=3, visible=False)
        self._label_w = {}
        lw = px_to_pt(self.line_px)
        mk = dict(solid_capstyle="round", zorder=3)
        (self.glow2_lo,) = ax.plot([], [], color=c["idle"], lw=px_to_pt(16), alpha=0.0, **mk)
        (self.glow2_hi,) = ax.plot([], [], color=c["event"], lw=px_to_pt(16), alpha=0.0, **mk)
        (self.glow_lo,) = ax.plot([], [], color=c["idle"], lw=px_to_pt(9), alpha=0.0, **mk)
        (self.glow_hi,) = ax.plot([], [], color=c["event"], lw=px_to_pt(9), alpha=0.0, **mk)
        (self.lo,) = ax.plot([], [], color=c["idle"], lw=lw, solid_capstyle="round", solid_joinstyle="round",
                             zorder=4)
        (self.hi,) = ax.plot([], [], color=c["event"], lw=lw, solid_capstyle="round", solid_joinstyle="round",
                             zorder=4)
        self.halo = []
        for r, a in ((HALO_R_PX + 6, 0.08), (HALO_R_PX, 0.12), (HALO_R_PX - 6, 0.16)):
            (h,) = ax.plot([], [], "o", ms=px_to_pt(2 * r), mec="none", alpha=a, clip_on=False, zorder=5)
            self.halo.append((h, a))
        (self.dot,) = ax.plot([], [], "o", ms=px_to_pt(2 * HEAD_R_PX), mec="none", clip_on=False, zorder=6)

    @property
    def window_s(self) -> float:
        return self.plot_w / self.speed

    def _label_width(self, name: str) -> float:
        if name not in self._label_w:
            t = self.band_texts[name]
            r = self.ax.figure.canvas.get_renderer()
            bb = t.get_window_extent(r)
            (x0, _), (x1, _) = self.ax.transData.inverted().transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
            self._label_w[name] = float(x1 - x0)
        return self._label_w[name]

    def value_at(self, t_rel, v, head: float) -> float:
        return float(np.interp(head, t_rel, v))

    def update(self, t_rel, v, head: float, glow: float = 1.0) -> float:
        """Draw the stream with the head at `head` (s from odor onset); returns the head value."""
        c = self.colors
        t_rel, v = np.asarray(t_rel, float), np.asarray(v, float)
        k0 = np.searchsorted(t_rel, head - self.window_s - 0.1)
        k1 = np.searchsorted(t_rel, head, side="right")
        hv = self.value_at(t_rel, v, head)
        xs, lo, hi = split_at_threshold(stream_x(np.r_[t_rel[k0:k1], head], head, self.plot_w, self.speed),
                                        np.r_[v[k0:k1], hv], self.thr)
        self.lo.set_data(xs, lo)
        self.hi.set_data(xs, hi)
        tail = xs >= self.plot_w - TAIL_PX
        for art, a, y in ((self.glow_lo, 0.45, lo), (self.glow_hi, 0.45, hi),
                          (self.glow2_lo, 0.12, lo), (self.glow2_hi, 0.12, hi)):
            art.set_data(xs[tail], y[tail])
            art.set_alpha(a * glow)
            art.set_visible(glow > 0)
        hot = is_event(hv, self.thr)
        for h, a in self.halo:
            h.set_data([self.plot_w], [hv]); h.set_color(c["event"] if hot else c["idle"])
            h.set_alpha(a * glow); h.set_visible(glow > 0)
        self.dot.set_data([self.plot_w], [hv]); self.dot.set_color(c["event"] if hot else c["dot_idle"])
        self.dot.set_alpha(glow); self.dot.set_visible(glow > 0)
        for b in self.bands:
            ext = band_extent(*b.region, head, self.plot_w, self.speed)
            patch = self.band_patches[b.name]
            patch.set_visible(ext is not None)
            if ext:
                patch.set_x(ext[0]); patch.set_width(ext[1] - ext[0])
            if b.name in self.band_texts:
                txt = self.band_texts[b.name]
                lext = band_extent(*(b.label_region or b.region), head, self.plot_w, self.speed)
                txt.set_visible(lext is not None)
                if lext:
                    txt.set_x(sticky_x(lext[0], lext[1], self._label_width(b.name), self.plot_w))
        return hv


@dataclass
class Readout:
    """Large live PER % readout, right-aligned at x_right (figure px) on a baseline (figure px):
    value + unit, scaled 1 + 0.32 k and in the event colour above threshold. fx / fy convert
    top-left frame px to figure fractions (make_per_montage_video.fx / fy)."""
    fig: object
    fx: object
    fy: object
    x_right: float
    baseline: float
    size_pt: float
    unit_pt: float
    unit_w_px: float                # width of the unit text at unit_pt (px)
    colors: dict                    # keys: text, event, muted

    def __post_init__(self):
        c = self.colors
        self.unit = self.fig.text(self.fx(self.x_right), self.fy(self.baseline), UNIT, fontsize=self.unit_pt,
                                  fontweight="bold", color=c["muted"], ha="right", va="baseline")
        self.value = self.fig.text(self.fx(self.x_right), self.fy(self.baseline), "0", fontsize=self.size_pt,
                                   fontweight="bold", color=c["text"], ha="right", va="baseline")

    def update(self, v: float, thr: float) -> None:
        c = self.colors
        s = readout_scale(v, thr)
        hot = is_event(v, thr)
        self.value.set_text(format_readout(v))
        self.value.set_fontsize(self.size_pt * s)
        self.value.set_color(c["event"] if hot else c["text"])
        self.value.set_x(self.fx(self.x_right - (self.unit_w_px + 8) * s))
        self.unit.set_fontsize(self.unit_pt * s)
        self.unit.set_color(c["event"] if hot else c["muted"])
