#!/usr/bin/env python3
"""Standalone "Behavior Tracking" clips: a fly's PER drawn live as an oscilloscope trace.

One 1080 x 1080, 40 fps clip per trial of the PER-montage fly, rendered separately so
they can be spliced into the PER presentation video by hand (the montage itself is not
touched):

  training             training_4  Hexanol paired with light  -> sustained extension
  cs_plus_test         testing_1   Hexanol alone (CS+)        -> odor-evoked extension
  untrained_odor_test  testing_3   Citral (untrained odor)    -> flat trace

Layout: lab header ("Behavior Tracking" + one plain-language line), the stabilised fly
footage with the tracked-labellum ring (make_per_montage_video's loaders and ECC
stabilisation, imported), and an oscilloscope card whose write head is the current
video frame.

Shot recipe: chart-live-moves / oscilloscope-stream (OscilloscopeStreamV2.tsx), tuned
parameters ported, not reinterpreted:
  * write point pinned at the plot's right edge, history streaming left at a constant
    5.5 px/f x 30 fps = 165 px/s (4.125 px per frame at 40 fps); 840 px plot = 5.1 s window.
    The demo runs 8 px/f; the card documents 5.5 as the perceptible lower bound. At 8 px/f
    (3.5 s window) a sustained PER's onset scrolls out before the end, so the closing frames
    showed a flat line. One shared speed keeps the time axis identical across the three
    spliced clips, and 5.1 s keeps each test's odor onset and response in view to the end.
  * 10.5 px write-head dot with a soft 24 px halo, 160 px afterglow tail behind it
  * large live readout at top right, scaled 1 + 0.32 k and switched to the event colour
    while the event passes the head
  * real axes / ticks / units; odor and light periods as bands that scroll with the data
  * no brake and no hold (user rule 2026-10-30: never end on a pause, overriding the demo's
    out-cubic brake + 1.6 s still hold): the stream keeps writing at 1x through the lab
    fade-out, the footage stays on the head's frame, and the write-head dot, halo and
    afterglow tail stay lit while everything fades to the surface together.

Real data only: the trace is the montage's tracked labellum displacement (eye->labellum
distance minus the fly's own pre-odor median). The "spike event" is the fly's PER: the
head value rising above PER_THRESHOLD = max(10 px, rest median + 10 rest SD), which is
10 px for all three trials (rest SD 0.1-0.3 px; 10 px = 2x the pipeline's 5 px
non-reactive span). The event treatment happens once per clip (recipe pitfall): each
window runs 1x real time through one PER and fades out while it is still on screen.
end_rel is where the fade starts; the head keeps going for FADE_OUT_S more. Training
starts at odor onset and starts fading at +7.85 s, with the extension sustained from
+5.1 s (light onset) in the window; its last frame (head +8.675 s) still keeps the onset
~1.5 s inside the left edge and the rising edge clear of the threshold label. The tests
start 1.0 s before odor onset and start fading at +4.7 s: the CS+ peak of its first
bout, which stays above threshold through the fade, and for Citral past the CS+ latency
of +4.1 s (no event). 4.7 s is also the earliest fade start that leaves the header's
one-line description its full reading time.

Nothing on screen identifies the fly, batch, trial or cohort; trial names live only in
this file and the clip file names carry only the condition.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.transforms import blended_transform_factory
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab_video_style as style  # noqa: E402
import make_per_montage_video as montage  # noqa: E402

FFMPEG = "/usr/bin/ffmpeg"
FFPROBE = "/usr/bin/ffprobe"
OUT_DIR = Path("/home/ramanlab/Documents/cole/Results/Figures")

W, H, FPS = 1080, 1080, 40
SURFACE, TEXT_1, TEXT_2, MUTED = style.SURFACE, style.TEXT_1, style.TEXT_2, style.MUTED
GRID, CARD_BG = montage.GRID, montage.RAISED
ODOR_C, LIGHT_C = montage.ODOR_C, montage.LIGHT_C
EVENT_C = montage.LABELLUM_C               # labellum orange: the ring, and the trace above threshold
TEXT_PT = style.MIN_TEXT_PT                # 23.04 pt = 32 px at dpi 100
CROP = montage.CROP

# ------------------------------------------------------------ ported recipe parameters
SPEED_PX_S = 5.5 * 30                      # card's lower bound 5.5 px/frame at 30 fps (demo: 8)
HEAD_R_PX = 10.5                           # write-head dot radius
HALO_R_PX = 24                             # its soft halo
TAIL_PX = 160                              # afterglow behind the head
READOUT_GAIN = 0.32                        # readout scale 1 + 0.32 k
HOLD_IN_S = 1.0                            # still while the content eases in (demo HOLD 12 f + lab reveal)
LEAD_S = 1.0                               # test clips: stream starts this long before odor onset

# ------------------------------------------------------------ event (PER) definition
PER_FLOOR_PX = 10.0                        # 2x the pipeline's NON_REACTIVE_SPAN_PX (5 px)
PER_NOISE_K = 10.0                         # ... or rest median + 10 rest SD, whichever is larger
REST_S = 10.0                              # rest window before odor onset (= montage baseline)
K_SPAN_PX = 50.0                           # readout jump k reaches 1 at threshold + 50 px
LIGHT_ONSET_REL = 5.0                      # protocol: light from +5 s after odor onset (reading-time check)

HEADING = "Behavior Tracking"
BODY = ("Labellum position is tracked frame by frame; proboscis extension appears "
        "as a rise in displacement.")
CARD_TITLE = montage.Y_LABEL               # "Labellum Displacement (px)"
UNIT = "px"
THRESH_LABEL = "PER threshold"
KEY_LABEL = montage.LEGEND_LABEL           # "Labellum (Tracked)"
Y_TICKS = (0, 40, 80)
Y_MIN, Y_MAX = -15.0, 100.0                # top lane above the trace holds the band labels


@dataclass(frozen=True)
class ClipSpec:
    trial: str
    end_rel: float                         # fade-out starts with the head here (s from odor onset)
    title: str
    lead_s: float = LEAD_S                 # stream starts this long before odor onset


_TITLES = {s.trial: s.title for s in montage.SEGMENTS}
CLIPS = {
    # training starts at odor ON and dwells on the sustained extension (user 2026-10-30), and
    # keeps the onset ~1.5 s inside the 5.1 s window to its last frame (review 2026-10-30). The
    # head now keeps streaming through the fade (+0.825 s to the last frame), so the fade starts
    # at 7.85 s: the last frame's head (8.675 s) sits where the old stop (8.7 s) did, the latest
    # whose rising edge clears the "PER threshold" label with visible air
    "training": ClipSpec("training_4", 7.85, _TITLES["training_4"], lead_s=0.0),
    # fade starts at the first bout's peak; it stays above threshold to the last frame (+5.5 s)
    "cs_plus_test": ClipSpec("testing_1", 4.7, _TITLES["testing_1"]),
    "untrained_odor_test": ClipSpec("testing_3", 4.7, _TITLES["testing_3"]),
}
TRIAL_TO_KEY = {s.trial: k for k, s in CLIPS.items()}
PER_ONSET_TRAINING_S = 5.1                 # extension starts at light onset in training_4
CS_PLUS_LATENCY_S = 4.1                    # testing_1 crosses the threshold at +4.10 s


def out_path(key: str) -> Path:
    return OUT_DIR / f"PER_tracking_scope_{key}.mp4"


# ================================================================ pure helpers
class Timeline:
    """Output frame -> head time (s from odor onset): still while the content eases in, then
    1x real time to the last frame. The lab fade-out covers the last n_fade frames and the
    head keeps streaming under it (no brake, no hold): it reaches end_rel at fade_start and
    end_rel + (n_fade - 1) / FPS on the last frame."""

    def __init__(self, end_rel: float, lead_s: float = LEAD_S):
        self.end_rel = float(end_rel)
        self.n_hold_in = int(round(HOLD_IN_S * FPS))
        self.n_stream = int(round((self.end_rel + lead_s) * FPS))
        self.start_rel = self.end_rel - self.n_stream / FPS
        self.stream_start = self.n_hold_in
        self.fade_start = self.stream_start + self.n_stream
        self.n_fade = int(round(style.FADE_OUT_S * FPS))
        self.n_frames = self.fade_start + self.n_fade
        self.duration = self.n_frames / FPS
        self.last_rel = self.head_time(self.n_frames - 1)

    def head_time(self, i: int) -> float:
        return self.start_rel + max(0, i - self.stream_start) / FPS

    def first_frame_at(self, rel: float) -> int:
        """First output frame whose head has reached `rel`."""
        for i in range(self.n_frames):
            if self.head_time(i) >= rel - 1e-9:
                return i
        return self.n_frames


def timeline(key: str, end_rel: float | None = None) -> Timeline:
    spec = CLIPS[key]
    return Timeline(spec.end_rel if end_rel is None else end_rel, spec.lead_s)


def stream_x(sample_t, head_t: float):
    """Plot x (px) of a sample at time sample_t when the head is at head_t."""
    return PLOT_W - (head_t - np.asarray(sample_t, float)) * SPEED_PX_S


def source_frame(head_rel: float, odor_on_s: float, fps_src: float) -> int:
    """Video frame shown when the head is at head_rel: the frame nearest the head time."""
    return int(round((odor_on_s + head_rel) * fps_src))


def footage_positions(tl: Timeline, odor_on_s: float, fps_src: float) -> np.ndarray:
    """Source frame (int) shown at each output frame: the frame nearest the head, as
    source_frame(). The head runs 1x to the last frame, so through the fade every output frame
    shows the next recorded frame (40.02 fps source at 40 fps: never a repeat)."""
    return np.array([source_frame(tl.head_time(i), odor_on_s, fps_src) for i in range(tl.n_frames)], int)


def per_threshold(ext: np.ndarray, t_rel: np.ndarray) -> float:
    rest = ext[(t_rel >= -REST_S) & (t_rel < 0)]
    return float(max(PER_FLOOR_PX, np.nanmedian(rest) + PER_NOISE_K * np.nanstd(rest)))


def is_event(v: float, thr: float) -> bool:
    return bool(np.isfinite(v) and v > thr)


def readout_k(v: float, thr: float) -> float:
    if not np.isfinite(v):
        return 0.0
    return float(np.clip((v - thr) / K_SPAN_PX, 0.0, 1.0))


def format_readout(v: float) -> str:
    if not np.isfinite(v):
        return "–"
    r = int(round(v))
    return f"−{-r}" if r < 0 else str(r)


def readout_state(v: float, thr: float) -> tuple[str, float, str]:
    """(text, scale, colour) of the live readout."""
    # Displacement below the fly's rest position (the labellum retracting a few px, e.g. -3 px
    # under Citral) is shown as 0 on the readout ONLY: a negative number reads as an error.
    # The trace keeps drawing the real, signed data.
    shown = max(v, 0.0) if np.isfinite(v) else v
    return format_readout(shown), 1.0 + READOUT_GAIN * readout_k(v, thr), EVENT_C if is_event(v, thr) else TEXT_1


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


def upward_crossings(values, thr: float) -> int:
    ev = np.asarray(values, float) > thr
    return int(np.sum(~ev[:-1] & ev[1:]))


def band_extent(on_rel: float, off_rel: float, head_t: float):
    """(x0, x1) of a stimulus band in plot px, drawn only up to the head (never the future)."""
    if head_t < on_rel:
        return None
    return float(stream_x(on_rel, head_t)), float(stream_x(min(off_rel, head_t), head_t))


BAND_LABEL_PAD = 10


def sticky_x(x0: float, x1: float, label_w: float, pad: float = BAND_LABEL_PAD) -> float:
    """Band label x: rides the band's left edge, sticks at the plot's left edge, and is
    pushed out to the left by the band's end once the band has closed."""
    x = max(x0, 0.0) + pad
    if x1 < PLOT_W - 1e-6:
        x = min(x, x1 - label_w - pad)
    return x


# ================================================================ layout (px from the top-left)
G = montage.G                               # 60 px outer gutter


def build_header() -> style.SegmentHeader:
    """Lab header without the segment numerals (a standalone clip): brand, heading,
    one-line description. The header's red accent rule belongs beside the "01 / 03"
    numerals, so with no numerals it is left out too (an orphan rule reads as a glitch)."""
    hdr = style.SegmentHeader(1, 1, HEADING, BODY)
    hdr.els = [hdr.brand_el, hdr.heading_el] + ([hdr.body_el] if hdr.body_el else [])
    return hdr


HDR = build_header()
CARD_H = 330
CARD_X, CARD_W = G, W - 2 * G
CARD_Y = H - 32 - CARD_H
VID_Y = HDR.bottom + 22
VID_MAX_H = CARD_Y - 24 - VID_Y
VID_MAX_W = 540                            # leaves a 388 px status column (title, stacked chips, key)
# Footage view: a tighter box inside the stabilised CROP (stabilisation still matches on the
# full crop around the eye anchor). It keeps the head, the eye anchor (crop 165, 120: 145 px
# from the left, 90 from the top) and the labellum's whole path (lowest at crop y 233 when
# extended, 30 px above the bottom), and drops the empty arena at the left, the pin bulb at
# the top and the neighbouring fly's pin at the bottom. 320 x 233 source px -> 540 x 393,
# a 1.69x upscale (checked sharp by eye; the full crop was shown at 1.07x, 386 x 343).
VIEW = (20, 30, 320, 233)                  # x0, y0, w, h in stabilised-crop px
VID_W, VID_H = montage.fit_aspect(VIEW[2], VIEW[3], VID_MAX_W, VID_MAX_H)
COL_X = G + VID_W + 32
COL_W = W - G - COL_X
READ_R = CARD_X + CARD_W - 24              # readout right edge
READ_BASE = CARD_Y + 90                    # readout / card-title baseline
READOUT_PX = 72
PILL_STEP = 72                             # stacked chips: centre to centre
READOUT_PT = READOUT_PX * 72 / 100
PLOT_X, PLOT_Y = CARD_X + 80, CARD_Y + 118
PLOT_W, PLOT_H = 840, 150
WINDOW_S = PLOT_W / SPEED_PX_S
X_TICK_S = tuple(range(int(WINDOW_S), -1, -1))           # every 1 s back from the head
X_TICK_LABELS = tuple(f"−{s} s" if s else "now" for s in X_TICK_S)
assert VID_H >= 380, f"header grew: only {VID_H} px left for the video"


def px_to_pt(px: float) -> float:
    return px * 72 / 100


# ================================================================ figure
def build_figure(key: str, odor_name: str = "", has_light: bool = False, thr: float = PER_FLOOR_PX):
    """Static figure for one clip and the artists updated per frame (no video needed)."""
    montage._setup_fonts()
    spec = CLIPS[key]
    fig = montage._new_fig()
    bg = fig.add_axes([0, 0, 1, 1], zorder=-1)
    bg.set_xlim(0, W); bg.set_ylim(H, 0); bg.axis("off")
    bg.add_patch(FancyBboxPatch((CARD_X, CARD_Y), CARD_W, CARD_H, boxstyle="round,pad=0,rounding_size=14",
                                fc=CARD_BG, ec=GRID, lw=1.5))
    art = {}

    # video + tracked labellum ring
    ax_v = montage.px_axes(fig, G, VID_Y, VID_W, VID_H)
    art["im"] = ax_v.imshow(np.zeros((VIEW[3], VIEW[2]), np.uint8), cmap="gray", vmin=0, vmax=255,
                            interpolation="lanczos")
    ax_v.set_xticks([]); ax_v.set_yticks([])
    for s in ax_v.spines.values():
        s.set_edgecolor(GRID); s.set_linewidth(1)
    (art["ring"],) = ax_v.plot([], [], "o", ms=22, mfc="none", mec=EVENT_C, mew=2.4, path_effects=montage.HALO)

    # status column: condition, then the live stimulus chips stacked right under it, ring key
    _, title_h = montage.text_block(fig, spec.title, COL_X, VID_Y + 2, fontsize=TEXT_PT, max_w=COL_W,
                                    spacing=1.3, balance=True, color=TEXT_1, fontweight="bold")
    m = montage.measurer(fig, TEXT_PT, fontweight="bold")
    pad = montage.pill_pad_px(TEXT_PT)                   # pill bbox pad, px
    pill_x = COL_X + pad + 1
    pill_y = VID_Y + 2 + title_h + 18 + pad + TEXT_PT * 100 / 72 / 2
    art["odor_pill"] = montage._pill(fig, pill_x, pill_y, "ODOR", ODOR_C, False, fontsize=TEXT_PT)
    art["light_pill"] = montage._pill(fig, pill_x, pill_y + PILL_STEP, "LIGHT", LIGHT_C, False, fontsize=TEXT_PT)
    key_y = VID_Y + VID_H - 18
    montage._rule(fig, key_y - 36, COL_X, W - G)
    fig.add_artist(plt.Line2D([montage.fx(COL_X + 9)], [montage.fy(key_y)], transform=fig.transFigure, marker="o",
                              ms=13, mfc="none", mec=EVENT_C, mew=2.4, linestyle="none"))
    fig.text(montage.fx(COL_X + 28), montage.fy(key_y), KEY_LABEL, fontsize=TEXT_PT, color=TEXT_2, va="center")

    # card header: title (= y label, set horizontally) + live readout
    art["card_title"] = fig.text(montage.fx(CARD_X + 24), montage.fy(READ_BASE), CARD_TITLE, fontsize=TEXT_PT,
                                 fontweight="bold", color=TEXT_1, va="baseline")
    art["unit"] = fig.text(montage.fx(READ_R), montage.fy(READ_BASE), UNIT, fontsize=TEXT_PT, fontweight="bold",
                           color=MUTED, ha="right", va="baseline")
    art["readout"] = fig.text(montage.fx(READ_R), montage.fy(READ_BASE), "0", fontsize=READOUT_PT,
                              fontweight="bold", color=TEXT_1, ha="right", va="baseline")
    art["unit_w"] = m(UNIT)

    # oscilloscope plot: x in plot px (head at PLOT_W), y in px of displacement
    ax = montage.px_axes(fig, PLOT_X, PLOT_Y, PLOT_W, PLOT_H)
    ax.set_facecolor("none")
    ax.set_xlim(0, PLOT_W); ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_xticks([PLOT_W - s * SPEED_PX_S for s in X_TICK_S])
    ax.set_xticklabels(X_TICK_LABELS)
    ax.set_yticks(Y_TICKS)
    ax.tick_params(colors=MUTED, labelsize=TEXT_PT, length=4, pad=6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.yaxis.grid(True, color=GRID, linewidth=1.0)
    ax.xaxis.grid(True, color=GRID, linewidth=1.0, alpha=0.6)
    ax.set_axisbelow(True)
    ax.axhline(0, color=MUTED, linewidth=0.9, linestyle=(0, (2, 3)), zorder=2)
    ax.axhline(thr, color=EVENT_C, linewidth=1.2, linestyle=(0, (5, 4)), alpha=0.75, zorder=2)
    # threshold label at the plot's left end: the event's rising edge sweeps right -> left
    # through every x it reaches, so the label sits where the edge never gets to before the
    # clip ends (pinned by test_trace_never_runs_through_the_threshold_label)
    art["thresh_label"] = ax.text(BAND_LABEL_PAD, thr + 2.0, THRESH_LABEL, color=EVENT_C, fontsize=TEXT_PT,
                                  va="bottom", zorder=2)
    art["ax_scope"] = ax

    xdata_yaxes = blended_transform_factory(ax.transData, ax.transAxes)
    bands, labels = {}, {}
    names = [("odor", ODOR_C, 0.16, "Odor")] + ([("light", LIGHT_C, 0.24, "Light")] if has_light else [])
    for name, c, a, lab in names:
        bands[name] = ax.add_patch(Rectangle((0, Y_MIN), 0, Y_MAX - Y_MIN, fc=c, ec="none", alpha=a,
                                             zorder=1, visible=False))
        labels[name] = ax.text(0, 1 - 6 / PLOT_H, lab, transform=xdata_yaxes, color=c, fontsize=TEXT_PT,
                               fontweight="bold", va="top", clip_on=True, zorder=3, visible=False)
    art["bands"], art["labels"] = bands, labels
    art["label_w"] = {k: m(t.get_text()) for k, t in labels.items()}

    lw = px_to_pt(5)
    (art["glow2_lo"],) = ax.plot([], [], color=TEXT_2, lw=px_to_pt(16), alpha=0.0, solid_capstyle="round", zorder=3)
    (art["glow2_hi"],) = ax.plot([], [], color=EVENT_C, lw=px_to_pt(16), alpha=0.0, solid_capstyle="round", zorder=3)
    (art["glow_lo"],) = ax.plot([], [], color=TEXT_2, lw=px_to_pt(9), alpha=0.0, solid_capstyle="round", zorder=3)
    (art["glow_hi"],) = ax.plot([], [], color=EVENT_C, lw=px_to_pt(9), alpha=0.0, solid_capstyle="round", zorder=3)
    (art["lo"],) = ax.plot([], [], color=TEXT_2, lw=lw, solid_capstyle="round", solid_joinstyle="round", zorder=4)
    (art["hi"],) = ax.plot([], [], color=EVENT_C, lw=lw, solid_capstyle="round", solid_joinstyle="round", zorder=4)
    art["halo"] = []
    for r, a in ((HALO_R_PX + 6, 0.08), (HALO_R_PX, 0.12), (HALO_R_PX - 6, 0.16)):
        (h,) = ax.plot([], [], "o", ms=px_to_pt(2 * r), mec="none", alpha=a, clip_on=False, zorder=5)
        art["halo"].append((h, a))
    (art["dot"],) = ax.plot([], [], "o", ms=px_to_pt(2 * HEAD_R_PX), mec="none", clip_on=False, zorder=6)
    return fig, art


def label_data_box(text, ax) -> tuple[float, float, float, float]:
    """(x0, x1, y0, y1) of a drawn text artist in the scope's data units (x = plot px)."""
    bb = text.get_window_extent(ax.figure.canvas.get_renderer())
    (x0, y0), (x1, y1) = ax.transData.inverted().transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
    return float(x0), float(x1), float(y0), float(y1)


def set_readout(art, v: float, thr: float) -> None:
    text, scale, color = readout_state(v, thr)
    art["readout"].set_text(text)
    art["readout"].set_fontsize(READOUT_PT * scale)
    art["readout"].set_color(color)
    art["readout"].set_x(montage.fx(READ_R - (art["unit_w"] + 10) * scale))
    art["unit"].set_fontsize(TEXT_PT * scale)
    art["unit"].set_color(EVENT_C if color == EVENT_C else MUTED)


def head_value(tr: dict, head_rel: float) -> float:
    return float(np.interp(head_rel, tr["t_rel"], tr["ext"]))


def update_scope(art, tr: dict, head: float, glow: float = 1.0) -> None:
    thr = tr["thr"]
    t, e = tr["t_rel"], tr["ext"]
    k0 = np.searchsorted(t, head - WINDOW_S - 0.1)
    k1 = np.searchsorted(t, head, side="right")
    v = head_value(tr, head)
    xs, lo, hi = split_at_threshold(stream_x(np.r_[t[k0:k1], head], head), np.r_[e[k0:k1], v], thr)
    art["lo"].set_data(xs, lo)
    art["hi"].set_data(xs, hi)
    tail = xs >= PLOT_W - TAIL_PX
    for name, a, y in (("glow_lo", 0.45, lo), ("glow_hi", 0.45, hi), ("glow2_lo", 0.12, lo), ("glow2_hi", 0.12, hi)):
        art[name].set_data(xs[tail], y[tail])
        art[name].set_alpha(a * glow)
        art[name].set_visible(glow > 0)
    hot = is_event(v, thr)
    for h, a in art["halo"]:
        h.set_data([PLOT_W], [v]); h.set_color(EVENT_C if hot else TEXT_2)
        h.set_alpha(a * glow); h.set_visible(glow > 0)
    art["dot"].set_data([PLOT_W], [v]); art["dot"].set_color(EVENT_C if hot else TEXT_1)
    art["dot"].set_alpha(glow); art["dot"].set_visible(glow > 0)
    set_readout(art, v, thr)

    regions = {"odor": (0.0, tr["odor_off_rel"])}
    label_regions = {"odor": (0.0, tr["light_rel"][0] if tr["light_rel"] else tr["odor_off_rel"])}
    if tr["light_rel"] and "light" in art["bands"]:
        regions["light"] = label_regions["light"] = tr["light_rel"]
    for name, band in art["bands"].items():
        ext = band_extent(*regions[name], head) if name in regions else None
        band.set_visible(ext is not None)
        if ext:
            band.set_x(ext[0]); band.set_width(ext[1] - ext[0])
        lab = art["labels"][name]
        lext = band_extent(*label_regions[name], head) if name in label_regions else None
        lab.set_visible(lext is not None)
        if lext:
            lab.set_x(sticky_x(lext[0], lext[1], art["label_w"][name]))


def assert_text_floor(fig) -> None:
    from matplotlib.text import Text
    small = [(t.get_text(), t.get_fontsize()) for t in fig.findobj(Text)
             if t.get_text().strip() and t.get_visible() and t.get_fontsize() < TEXT_PT - 1e-6]
    if small:
        raise ValueError(f"text below the {TEXT_PT} pt legibility floor: {small}")


# ================================================================ on-screen text / reading time
def onscreen_strings(key: str) -> list[str]:
    spec = CLIPS[key]
    pills = [f"{p} {s}" for p in ("ODOR", "LIGHT") for s in ("ON", "OFF")]
    bands = ["Odor"] + (["Light"] if key == "training" else [])
    return ([HEADING, BODY, style.LAB_SHORT, spec.title, CARD_TITLE, UNIT, THRESH_LABEL, KEY_LABEL]
            + pills + bands + list(X_TICK_LABELS) + [str(y) for y in Y_TICKS])


def reading_items(key: str) -> list:
    R = style.ReadItem
    tl = timeline(key)
    content = style.CONTENT_T0 + style.CONTENT_DUR
    spec = CLIPS[key]
    items = HDR.read_items() + [R(t, content) for t in
                                (spec.title, CARD_TITLE, THRESH_LABEL, KEY_LABEL, "ODOR OFF", "LIGHT OFF")]
    items.append(R("Odor", tl.first_frame_at(0.0) / FPS))
    if key == "training":
        items.append(R("Light", tl.first_frame_at(LIGHT_ONSET_REL) / FPS))
    return items


# ================================================================ rendering
def load_clip_data(key: str) -> dict:
    spec = CLIPS[key]
    tr = montage.load_trial(spec.trial)
    odor_on, odor_off = tr["odor"]
    tr["t_rel"] = tr["t"] - odor_on
    tr["odor_off_rel"] = odor_off - odor_on
    tr["light_rel"] = (tr["light"][0] - odor_on, tr["light"][1] - odor_on) if tr["light"] else None
    tr["thr"] = per_threshold(tr["ext"], tr["t_rel"])
    return tr


GOP = 1000                                 # > any clip: one keyframe, no quality pop mid-clip


def _encoder(path: Path):
    return subprocess.Popen(
        [FFMPEG, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}", "-r", str(FPS),
         "-i", "-", "-an", "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-g", str(GOP), "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", str(path)],
        stdin=subprocess.PIPE)


def render_clip(key: str, end_rel: float | None = None, path: Path | None = None) -> Path:
    path = path or out_path(key)
    spec = CLIPS[key]
    tr = load_clip_data(key)
    tl = timeline(key, end_rel)
    if tl.n_frames >= GOP:
        raise ValueError(f"{key}: {tl.n_frames} frames; raise GOP so no keyframe lands mid-clip")
    odor_on, fps_src = tr["odor"][0], tr["fps"]
    q = footage_positions(tl, odor_on, fps_src)
    first, last = int(q[0]), int(q[-1])
    if last >= int(tr["frame"].max()):
        raise RuntimeError(f"{key}: stream runs to frame {last}, past the trial's last frame")
    sel = (tr["frame"] >= first) & (tr["frame"] <= last)
    frames, shifts = montage.stabilized_clip(montage.find_video(spec.trial), first, last, CROP, tr["eye_xy"],
                                             tr["lab_xy"][sel])
    if len(frames) != last - first + 1:
        raise RuntimeError(f"{key}: decoded {len(frames)} of {last - first + 1} frames")
    vx, vy, vw, vh = VIEW
    frames = frames[:, vy:vy + vh, vx:vx + vw].astype(np.float32)

    def ring_at(f: int):
        k = min(np.searchsorted(tr["frame"], f), len(tr["frame"]) - 1)
        rx, ry = montage.stabilized_ring(tr["lab_xy"][k], CROP, shifts[f - first])
        return np.array([rx - vx, ry - vy])

    fig, art = build_figure(key, tr["odor_name"], bool(tr["light"]), tr["thr"])
    assert_text_floor(fig)
    enc = _encoder(path)
    for i in range(tl.n_frames):
        head = tl.head_time(i)
        f = int(q[i])                                     # the head's own frame, fade included
        art["im"].set_data(np.clip(frames[f - first] + 0.5, 0, 255).astype(np.uint8))
        r = ring_at(f)
        ok = bool(np.isfinite(r).all())
        art["ring"].set_data([r[0]] if ok else [], [r[1]] if ok else [])
        update_scope(art, tr, head)                       # head glow + tail stay lit as it fades
        montage._set_pill(art["odor_pill"], "ODOR", ODOR_C, 0 <= head < tr["odor_off_rel"])
        lr = tr["light_rel"]
        montage._set_pill(art["light_pill"], "LIGHT", LIGHT_C, bool(lr) and lr[0] <= head < lr[1])
        fig.canvas.draw()
        rgba = style.compose(np.asarray(fig.canvas.buffer_rgba()), i / FPS, tl.duration, HDR.els)
        enc.stdin.write(rgba.tobytes())
    plt.close(fig)
    enc.stdin.close()
    if enc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {path}")
    return path


def probe(path: Path) -> tuple[float, int]:
    out = subprocess.run([FFPROBE, "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                          "stream=nb_read_frames:format=duration", "-of", "default=nw=1:nk=1", str(path)],
                         capture_output=True, text=True, check=True).stdout.split()
    return float(out[1]), int(out[0])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--trial", choices=sorted(TRIAL_TO_KEY) + sorted(CLIPS),
                    help="one clip: a trial (training_4 / testing_1 / testing_3) or its clip key")
    ap.add_argument("--all", action="store_true", help="render all three clips")
    ap.add_argument("--end", type=float, default=None,
                    help="override where the fade-out starts (s from odor onset); single --trial only")
    args = ap.parse_args()
    if args.all == bool(args.trial):
        ap.error("pass exactly one of --trial or --all")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.all:
        with Pool(len(CLIPS)) as pool:
            paths = pool.map(render_clip, list(CLIPS))
    else:
        key = TRIAL_TO_KEY.get(args.trial, args.trial)
        paths = [render_clip(key, args.end)]
    for p in paths:
        dur, n = probe(p)
        print(f"[OK] {p}  {dur:.2f} s  {n} frames")


if __name__ == "__main__":
    main()
