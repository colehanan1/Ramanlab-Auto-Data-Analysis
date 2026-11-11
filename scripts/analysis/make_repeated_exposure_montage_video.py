#!/usr/bin/env python3
"""Website/presentation montage: how repeated odor exposure changes the PER.

Cohort IAA-Sensitivity-1-1.  Protocol: odor panel (before exposure) -> 30 min
rest -> 6 x isoamyl acetate (IAA) back-to-back, no light, no reward -> the same
odor panel again (after exposure).

Square 1080 x 1080 frame (W, H from base). Sequence (real time, 40 fps, 49 s in the simple cut;
segments 1 and 5 are switched off by INCLUDE_PROTOCOL / INCLUDE_SUMMARY):
  0. title card: the IAA fly's own post-exposure footage behind the lab scrim
  1. animated protocol card (the six IAA puffs light up one by one, paced so
     every line can be read)
  2-4. before/after pairs, played simultaneously over the same window relative
       to odor onset: IAA -- the exposed odor, first -- (up), Hexanol (up),
       Benzaldehyde (down).  Each pair is a different fly; nothing on screen
       names flies, batches, trials or the cohort.  Every clip is stabilised on
       the fly's head (see make_per_montage_video.stabilized_clip).  Under each
       video a live oscilloscope of PER % (lab_scope.py, the "Behavior Tracking"
       recipe): write head pinned at the plot's right edge (lined up with the
       video's right edge), history streaming left at SCOPE_SPEED, odor band
       scrolling with the data up to the head, trace orange above that side's own
       PER threshold (dashed), x ticks "-4 s ... now", live "NN %" readout in the
       plot's top lane.  Both columns stream on the same clock, through the fade.
  5. cohort dumbbell: mean PER score +/- SEM, 42 flies, paired Wilcoxon
  6. closing card
Segments 1-6 carry the lab montage's numbered header ("01 / 06" ... "06 / 06").

Visual language (tokens, header, staged reveal, captions, hero card, fade to the
surface) is the lab research-overview montage's via lab_video_style.py (shared
with make_per_montage_video.py, imported as ``base``, which also supplies fonts,
palette, luma reader, trace maths and encoder settings).  Video is the luma plane
with a per-clip percentile stretch (computed once per clip, never per frame).

Usage:
  python3 make_repeated_exposure_montage_video.py            # full render
  python3 make_repeated_exposure_montage_video.py --preview DIR   # one PNG per segment type
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
import json

import make_per_montage_video as base
import lab_scope as scope

style = base.style

DATA_ROOT = Path("/securedstorage/DATAsec/cole/Data-secured-New/IAA-Sensitivity-1-1")
OUT_PATH = Path("/home/ramanlab/Documents/cole/Results/Figures/IAA-Sensitivity-1-1_repeated_exposure_montage.mp4")

W, H, FPS, FADE_S = base.W, base.H, base.FPS, base.FADE_S
FFMPEG, FFPROBE = base.FFMPEG, base.FFPROBE
SURFACE, TEXT_1, TEXT_2, MUTED, GRID = base.SURFACE, base.TEXT_1, base.TEXT_2, base.MUTED, base.GRID
ODOR_C, LABELLUM_C = base.ODOR_C, base.LABELLUM_C          # data colours; the lab red never encodes data
RAISED = "#242422"                      # slightly lifted surface for highlighted blocks
FRAME_WH = base.FRAME_WH

# ------------------------------------------------------------------ timing contract
TITLE_S = 10.0                          # every line of the hero card stays up long enough to read
PROTOCOL_S = 10.5
SUMMARY_S = 10.0
# Slides left out of the simple cut (user 2026-10-30): set True to restore each one.
INCLUDE_PROTOCOL = False                # experimental-protocol card
INCLUDE_FUTURE = False                  # "Future Directions" bullets + Odor -> AL -> Behavior schematic
# closing: 19 s with future directions; the single-statement conclusion needs less
CLOSING_S = 19.0 if INCLUDE_FUTURE else 12.0
# the conclusion is the video's key message: primary text >= 5 % of frame height (54 px)
CLOSING_STATEMENT_T, CLOSING_STATEMENT_SIZE, CLOSING_STATEMENT_LINE = 1.2, 54, 70

BLOCK_T0, BLOCK_DT, BLOCK_DUR = 0.6, 0.6, 0.7   # protocol blocks fade in one after another
PUFF_T0 = 3.0                           # first IAA puff lights at this time in the protocol card
PUFF_DT = 0.8                           # one puff every 0.8 s (was 0.5 s)
PUFF_FADE = 0.4                         # each puff eases on instead of snapping
AFTER_HIGHLIGHT_T = 7.7                 # post-exposure block highlights after the last puff ...
AFTER_HIGHLIGHT_DUR = 0.9               # ... then holds to the end of the card

# summary: each odor's dumbbell slides pre -> post in turn, then its numbers appear
ROW_T0, ROW_DT, ROW_DUR = 2.0, 1.8, 1.5
LABEL_DELAY, LABEL_DUR = 1.2, 0.6

# closing: future directions, then the schematic
FUTURE_RULE_T, FUTURE_HEAD_T = 5.4, 5.8         # the conclusion reads alone first
BULLET_T0, BULLET_DT = 6.8, 2.6                 # bullets fly in from the left, one every 2.6 s
BULLET_FLY_DUR, BULLET_FLY_DX = 1.1, -160
SCHEMATIC_T, SCHEMATIC_DUR = 14.0, 1.2


@dataclass(frozen=True)
class PairSpec:
    odor: str
    direction: str                      # "up" | "down"
    batch: str
    fly: str
    crop: tuple                         # x0, y0, w, h in video pixels
    before_trial: str
    after_trial: str
    start_rel: float                    # seconds relative to odor onset
    end_rel: float
    preview_rel: float                  # response moment used for the preview PNG


# The exposed odor comes first, so it is clear which odor the repeated exposure used.
PAIRS = [
    PairSpec("IAA", "up", "october_09_batch_2_rig_3", "fly1", (560, 0, 340, 280),
             "pretest_6", "testing_6", 10.0, 19.0, 14.5),
    PairSpec("Hexanol", "up", "october_14_batch_1", "fly2", (330, 330, 340, 280),
             "pretest_4", "testing_2", 4.0, 13.0, 9.6),
    PairSpec("Benzaldehyde", "down", "october_12_batch_1_rig_3", "fly1", (480, 0, 340, 280),
             "pretest_2", "testing_2", 15.0, 24.0, 21.4),
]


@dataclass(frozen=True)
class CohortRow:
    odor: str
    pre: float
    post: float
    p: float
    sem_pre: float = 0.0
    sem_post: float = 0.0


# Mean model PER score (0-5), sessions started 12-3 pm; source:
# Results/Show/IAA-Sensitivity-1-1_by-start-time_2026-10-27/12pm-3pm/.../IAA-Sensitivity-1-1_summary.csv
COHORT = [CohortRow("IAA", 1.21, 1.90, 0.044, 0.28, 0.30),
          CohortRow("Hexanol", 0.83, 1.69, 0.009, 0.20, 0.30),
          CohortRow("Benzaldehyde", 1.36, 0.98, 0.039, 0.22, 0.24)]
N_FLIES = 42
SUMMARY_TITLE = f"Population Summary: Pre- vs. Post-Exposure (n = {N_FLIES} Flies)"
SUMMARY_SUBTITLE = "Mean PER score \u00b1 SEM; sessions initiated 12:00\u201315:00"
SUMMARY_FOOTNOTE = "Two-sided Wilcoxon signed-rank test, paired by fly."

# Title card: kicker, then the genotype (italic), shared verbatim with video 1 -- the only
# fly identity anywhere in the video.
TITLE_KICKER_RUNS = [("Proboscis Extension Reflex (PER)", False)]
GENOTYPE_RUNS = base.GENOTYPE_RUNS
EXPOSED_ODOR = "IAA"
ODOR_LABELS = {"IAA": "IAA (Exposed Odor)", "Hexanol": "Hexanol", "Benzaldehyde": "Benzaldehyde"}
TITLE_TEXT = "Odor Experience Bidirectionally Modulates the Proboscis Extension Reflex"
TITLE_SUB = ("Responses to an odor panel were measured before and after six consecutive presentations "
             "of isoamyl acetate (IAA), without light or reward.")
PANEL_TITLES = ("Pre-Exposure", "Post-Exposure")
BADGES = {"up": "\u25b2 Increased Response", "down": "\u25bc Decreased Response"}
Y_LABEL = scope.Y_LABEL                 # "Proboscis Extension (%)": the scopes plot PER %, not px
LEGEND_LABEL = base.LEGEND_LABEL
TEXT_PT = base.TEXT_PT

CLOSING_HEADLINE = "Conclusion"
CLOSING_LINES = [
    "Repeated presentations of a single odor can change a fly\u2019s response to that odor "
    "and to other odors, and not in a uniform way.",
]
FUTURE_HEADING = "Future Directions"
FUTURE_HEAD_SIZE = 40                   # sub-heading between the 32 px body and the 56 px heading
FUTURE_BULLETS = [
    "Functional imaging of antennal lobe (AL) odor representations pre- and post-exposure",
    "Computational modeling linking AL plasticity to behavioral output",
    "Testing model predictions with novel odors",
]


# The population-results slide is left out of the montage (user 2026-10-30); set True to restore it.
INCLUDE_SUMMARY = False

# Lab header numbering: every segment after the title card.
PAIR_NAMES = [f"pair_{p.odor.lower()}" for p in PAIRS]
NAMES = (["title"] + (["protocol"] if INCLUDE_PROTOCOL else []) + PAIR_NAMES
         + (["summary"] if INCLUDE_SUMMARY else []) + ["closing"])
SEGMENT_INDEX = {name: i for i, name in enumerate(NAMES) if i > 0}
N_NUMBERED = len(SEGMENT_INDEX)

# Title card background: the IAA fly's post-exposure trial around its response (14.5 s),
# a wider crop than the pair panel (fly at upper right, burned-in odor label excluded).
TITLE_PAIR = 0
TITLE_FOOTAGE_REL = (9.5, 19.5)
TITLE_CROP = (300, 0, 680, 680)          # x0, y0, w, h in video pixels
TITLE_KB = ((1.04, 0.50, 0.42), (1.11, 0.47, 0.40))   # slow drift: 0.007 zoom / s
PROTOCOL_HEADING = "Experimental Protocol"
PROTOCOL_BODY = "Identical odor panel presented before and after repeated exposure to a single odor"


def total_runtime_s() -> float:
    return (TITLE_S + PROTOCOL_S * INCLUDE_PROTOCOL + SUMMARY_S * INCLUDE_SUMMARY + CLOSING_S
            + sum(p.end_rel - p.start_rel for p in PAIRS))


# ------------------------------------------------------------------- pure helpers
def clip_window(odor_on_frame: int, fps: float, start_rel: float, end_rel: float):
    """First/last source frame (inclusive) of a window given relative to odor onset;
    the span is exactly round(duration * FPS) output frames."""
    first = int(round(odor_on_frame + start_rel * fps))
    n = int(round((end_rel - start_rel) * FPS))
    return first, first + n - 1


def source_frame(i: int, first: int, fps_src: float) -> int:
    """Source frame shown for output frame i (output plays at FPS, source runs at fps_src).
    Every output frame shows a new source frame (no-freeze rule): a source slower than FPS
    (one clip is 39.83 fps) plays one frame per output frame, 0.4 % fast (<= 2 frames over
    a 9 s clip) instead of repeating a frame; a faster source skips one now and then."""
    return first + max(i, int(round(i / FPS * fps_src)))


def head_rel(i: int, first: int, odor_on_frame: int, fps_src: float) -> float:
    """Scope write-head time (s from odor onset) at output frame i: the time of the source frame
    on screen, so each scope stays frame-locked to its own video."""
    return (source_frame(i, first, fps_src) - odor_on_frame) / fps_src


def fmt_change(pre: float, post: float) -> str:
    return f"{pre:.2f} → {post:.2f}"


def fmt_p(p: float) -> str:
    return f"p = {p:.3f}"


def direction_glyph(pre: float, post: float) -> str:
    if post > pre:
        return "▲"
    if post < pre:
        return "▼"
    return "■"


def puffs_lit(t: float) -> int:
    """Number of IAA puffs lit at time t in the protocol card (one every PUFF_DT)."""
    if t < PUFF_T0:
        return 0
    return int(min(6, 1 + (t - PUFF_T0) // PUFF_DT))


def puff_level(j: int, t: float) -> float:
    """0..1 fill of puff j: eases on over PUFF_FADE once its turn comes."""
    return style.smoothstep((t - (PUFF_T0 + j * PUFF_DT)) / PUFF_FADE)


def block_alpha(k: int, t: float) -> float:
    return style.smoothstep((t - (BLOCK_T0 + BLOCK_DT * k)) / BLOCK_DUR)


def contrast_limits(frames: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.8, n_sample: int = 8):
    """Percentile limits from a few frames of ONE clip (fixed for the whole clip: no flicker)."""
    idx = np.linspace(0, len(frames) - 1, min(n_sample, len(frames))).astype(int)
    lo, hi = np.percentile(frames[idx], [lo_pct, hi_pct])
    return float(lo), float(max(hi, lo + 1.0))


def stretch(frame: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (np.clip((frame.astype(np.float32) - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def _ramp(t: float, t0: float, dur: float) -> float:
    return float(np.clip((t - t0) / dur, 0.0, 1.0))


def _mix(c0: str, c1: str, f: float):
    a, b = np.array(to_rgb(c0)), np.array(to_rgb(c1))
    return tuple(a + (b - a) * f)


# ----------------------------------------------------------------------- data
def _stem(batch: str) -> str:
    return batch.rsplit("_rig", 1)[0]


def find_video(batch: str, trial: str) -> Path:
    hits = sorted((DATA_ROOT / batch).glob(f"output_{_stem(batch)}_{trial}_*.mp4"))
    if len(hits) != 1:
        raise RuntimeError(f"Expected one video for {batch}/{trial}, found {hits}")
    return hits[0]


def _trial_table(spec: PairSpec, trial: str):
    tdir = DATA_ROOT / spec.batch / f"{_stem(spec.batch)}_{trial}"
    meta = json.loads((tdir / "_trial_meta.json").read_text())
    df = pd.read_parquet(tdir / f"{tdir.name}_{spec.fly}_distances.parquet")
    return meta, df


def stabilized_window(spec: PairSpec, trial: str, meta: dict, df: pd.DataFrame, first: int, last: int, crop):
    """Stabilised crops + per-frame shifts of source frames first..last (eye-anchored)."""
    frames = df["frame"].to_numpy(int)
    sel = (frames >= first) & (frames <= last)
    return base.stabilized_clip(find_video(spec.batch, trial), first, last, crop, base.eye_position(df),
                                df[["x_class1", "y_class1"]].to_numpy(float)[sel])


def scope_trace(meta: dict, df: pd.DataFrame) -> dict:
    """What a column's scope plots: PER % (lab_scope.per_pct of the pipeline's
    distance_percentage) against time from odor onset, that trial's own PER threshold (rest
    median + max(10 points, 10 rest SD)) and its odor duration."""
    fps = float(meta["fps"])
    on_f, off_f = int(meta["odor_on_frame"]), int(meta["odor_off_frame"])
    rel = df["frame"].to_numpy(dtype=float) / fps - on_f / fps
    pct = scope.per_pct(df[scope.PCT_COLUMN])
    return dict(rel=rel, pct=pct, thr=scope.per_threshold(pct, rel), odor_len=(off_f - on_f) / fps)


def load_clip(spec: PairSpec, trial: str) -> dict:
    meta, df = _trial_table(spec, trial)
    fps = float(meta["fps"])
    on_f = int(meta["odor_on_frame"])
    tr = scope_trace(meta, df)
    first, last = clip_window(on_f, fps, spec.start_rel, spec.end_rel)
    n = last - first + 1
    n_src = source_frame(n - 1, first, fps) - first + 1
    raw, shifts = stabilized_window(spec, trial, meta, df, first, first + n_src - 1, spec.crop)
    if len(raw) < n_src:
        raise RuntimeError(f"{spec.batch}/{trial}: decoded {len(raw)} of {n_src} frames")
    lo, hi = contrast_limits(raw)
    return dict(tr, fps=fps, first=first, on_frame=on_f, frame=df["frame"].to_numpy(int),
                lab_xy=df[["x_class1", "y_class1"]].to_numpy(float),
                video=raw, shifts=shifts, lo=lo, hi=hi, odor_name=meta["odor"])


# --------------------------------------------------------------------- rendering
# Square layout: positions are pixels from the top-left of the W x H frame, converted
# with base.fx / base.fy / base.px_axes.  Gutter G and content width come from base.
G, CONTENT_W = base.G, base.CONTENT_W
fx, fy, px_axes = base.fx, base.fy, base.px_axes


def _pix_axes(fig):
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    return ax


def _rgba(fig) -> np.ndarray:
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba()).copy()


def _frame_count(seconds: float) -> int:
    return int(round(seconds * FPS))


def _set_alpha(artists, a):
    for x in artists:
        x.set_alpha(a)
        bp = x.get_bbox_patch() if hasattr(x, "get_bbox_patch") else None
        if bp is not None:
            bp.set_alpha(a)


# ---- 0. title
def title_footage() -> tuple[np.ndarray, float]:
    """Stabilised, contrast-stretched luma frames of the title background (TITLE_FOOTAGE_REL of
    the IAA fly's post-exposure trial, TITLE_CROP)."""
    spec = PAIRS[TITLE_PAIR]
    meta, df = _trial_table(spec, spec.after_trial)
    fps = float(meta["fps"])
    first, last = clip_window(int(meta["odor_on_frame"]), fps, *TITLE_FOOTAGE_REL)
    raw, _ = stabilized_window(spec, spec.after_trial, meta, df, first, last, TITLE_CROP)
    if not len(raw):
        raise RuntimeError("no title footage decoded")
    lo, hi = contrast_limits(raw)
    return stretch(raw, lo, hi), fps


TITLE_KEYS = [("line", ODOR_C, "Odor"), ("line", LABELLUM_C, LEGEND_LABEL)]


def title_text_els() -> list:
    return style.title_els([TITLE_KICKER_RUNS, GENOTYPE_RUNS], TITLE_TEXT, [TITLE_SUB], TITLE_KEYS)


def build_title():
    frames, fps = title_footage()
    hero = style.FootageHero(frames, TITLE_S, title_text_els(), kb=TITLE_KB, src_fps=fps)
    return _frame_count(TITLE_S), (lambda i: np.asarray(hero.frame(i / FPS).convert("RGBA"))), None


# ---- 1. protocol
BLOCKS = [
    ("Pre-Exposure Test: Odor Panel", "7 odors, 30 s each"),
    ("Rest: 30 min", "No stimulation"),
    ("Repeated Exposure: 6 × IAA", "Consecutive 30 s presentations, no reward"),
    ("Post-Exposure Test: Odor Panel", "Same 7 odors"),
]
PANEL_CHIPS = ("IAA", "Hex", "Benz", "+ 4 more")       # the exposed odor first, as everywhere
# vertical flow for the square frame: (top y, height) per block, px; full content width.
# The first block starts below the lab header (heading + two-line description, bottom 279).
# At 32 px a block's title and detail share one row, except the exposure block (two rows).
BLOCK_YH = [(304, 140), (488, 76), (608, 176), (828, 140)]
BLOCK_PAD = 32
BLOCK_TITLE_PT = TEXT_PT + 3            # block titles (bold) one step above the 32 px floor
TWO_ROW_BLOCKS = (2,)


def build_protocol():
    base._setup_fonts()
    fig = base._new_fig()
    ax = _pix_axes(fig)
    hdr = protocol_header()

    renderer = fig.canvas.get_renderer()
    blocks = []
    for k, ((title, sub), (by, bh)) in enumerate(zip(BLOCKS, BLOCK_YH)):
        patch = FancyBboxPatch((G, by), CONTENT_W, bh, boxstyle="round,pad=0,rounding_size=22",
                               fc=SURFACE, ec=GRID, lw=2)
        ax.add_patch(patch)
        row_y = by + bh / 2 if k == 1 else by + 40
        t1 = ax.text(G + BLOCK_PAD, row_y, title, fontsize=BLOCK_TITLE_PT, fontweight="bold", color=TEXT_1,
                     va="center")
        if k in TWO_ROW_BLOCKS:
            t2 = ax.text(G + BLOCK_PAD, by + 82, sub, fontsize=TEXT_PT, color=TEXT_2, va="center")
        else:
            # title and detail share one row
            fig.canvas.draw()
            t2 = ax.text(G + BLOCK_PAD + t1.get_window_extent(renderer).width + 28, row_y, sub,
                         fontsize=TEXT_PT, color=TEXT_2, va="center")
        blocks.append(dict(patch=patch, texts=[t1, t2], extra=[], y=by, h=bh))
    # down arrows between blocks
    for k in range(3):
        y0 = BLOCK_YH[k][0] + BLOCK_YH[k][1] + 6
        y1 = BLOCK_YH[k + 1][0] - 6
        arr = FancyArrowPatch((W / 2, y0), (W / 2, y1), arrowstyle="-|>", mutation_scale=20, color=MUTED, lw=2.2)
        ax.add_patch(arr)
        blocks[k + 1]["arrow"] = arr

    # odor chips in the before/after blocks
    for k in (0, 3):
        cx = G + BLOCK_PAD + 12
        for label in PANEL_CHIPS:
            tx = ax.text(cx, BLOCK_YH[k][0] + 100, label, fontsize=TEXT_PT,
                         color=TEXT_1 if label != "+ 4 more" else MUTED, va="center", ha="left",
                         bbox=dict(boxstyle="round,pad=0.3,rounding_size=0.8", fc=SURFACE, ec=MUTED, lw=1.4))
            cx += tx.get_window_extent(renderer).width + 36
            blocks[k]["extra"].append(tx)

    # six puff circles in the exposure block
    puffs = []
    for j in range(6):
        cx, cy = G + BLOCK_PAD + 27 + j * 76, BLOCK_YH[2][0] + 136
        c = Circle((cx, cy), 26, fc=SURFACE, ec=GRID, lw=2.2)
        ax.add_patch(c)
        n_txt = ax.text(cx, cy, str(j + 1), fontsize=TEXT_PT, fontweight="bold", color=MUTED, ha="center",
                        va="center")
        puffs.append((c, n_txt))
        blocks[2]["extra"].append(n_txt)

    def frame(i):
        t = i / FPS
        hl = style.smoothstep((t - AFTER_HIGHLIGHT_T) / AFTER_HIGHLIGHT_DUR)
        for k, b in enumerate(blocks):
            a = block_alpha(k, t)
            b["patch"].set_alpha(a)
            _set_alpha(b["texts"] + b["extra"], a)
            if "arrow" in b:
                b["arrow"].set_alpha(a)
            if k == 2:
                # exposure block: border brightens with the first puff
                b["patch"].set_edgecolor(_mix(GRID, ODOR_C, puff_level(0, t)))
            if k == 3:
                b["patch"].set_edgecolor(_mix(GRID, TEXT_1, hl))
                b["patch"].set_facecolor(_mix(SURFACE, RAISED, hl))
                b["patch"].set_linewidth(2 + 2.2 * hl)
        a = block_alpha(2, t)
        for j, (c, n_txt) in enumerate(puffs):
            g = puff_level(j, t)
            c.set_facecolor(_mix(SURFACE, ODOR_C, g))
            c.set_edgecolor(_mix(GRID, ODOR_C, g))
            n_txt.set_color(_mix(MUTED, SURFACE, g))
            c.set_alpha(a)
            n_txt.set_alpha(a)
        # the blocks animate themselves in, so the content is not eased in again
        return style.compose(_rgba(fig), t, PROTOCOL_S, hdr.els, reveal_content=False)

    base.assert_text_floor(fig)
    return _frame_count(PROTOCOL_S), frame, fig


def protocol_header() -> style.SegmentHeader:
    return style.SegmentHeader(SEGMENT_INDEX["protocol"], N_NUMBERED, PROTOCOL_HEADING, PROTOCOL_BODY)


# ---- 2-4. before/after pair: two panels side by side, each with a live PER % scope beneath its video
PANEL_GAP = 40
PANEL_W = (CONTENT_W - PANEL_GAP) // 2                  # 460 px
PANEL_X = (G, G + PANEL_W + PANEL_GAP)
PANEL_LABEL_GAP = 52                                    # header bottom -> panel-label centre
VID_TOP_GAP = 52                                        # panel-label centre -> video top (clears the pill)
TRACE_INSET = 44                                        # y tick labels sit under the video edge
VID_TRACE_GAP = 60                                      # video bottom -> scope y-label bottom (~37 px label)
YLABEL_GAP = 26                                         # y-label bottom -> scope axes (top tick label clears)
SCOPE_BOTTOM = H - 40 - 48                              # 40 px bottom margin under the 32 px "-4 s ... now"
                                                        # ticks (the x-axis title row is gone)
# Live scope (lab_scope.ScopePanel, the "Behavior Tracking" recipe). The head sits at the plot's
# right edge, which is the video's right edge. Every scope is SCOPE_W = 416 px wide; at the
# recipe's 165 px/s that is only a 2.5 s window, too short to keep a response's rise and its
# odor context in view, so all six scopes share one slower speed: 104 px/s = 4.0 s of history
# (2.6 px per output frame, still clearly moving), whole-second ticks landing exactly on the
# plot's left edge (-4 s) and the head ("now").
SCOPE_W = PANEL_W - TRACE_INSET
SCOPE_SPEED = 104.0                                     # px/s, one speed for every column and pair
SCOPE_WINDOW_S = SCOPE_W / SCOPE_SPEED                  # 4.0 s
SCOPE_YLIM = (-8.0, 135.0)                              # 0-100 %; above 100 a text lane for "Odor" (left)
SCOPE_YTICKS = (0, 50, 100)                             # and the readout (right), never crossed by the trace
ODOR_BAND_ALPHA = 0.16
READOUT_PX = 44                                         # live "NN %": value 44 px (up to 58 px at 1.32x) ...
READOUT_PT = READOUT_PX * 72 / 100                      # ... unit at the 32 px floor
READOUT_HEAD_GAP = scope.HALO_R_PX + 16                 # readout right edge -> head: clear of the halo
READOUT_LIFT = 12                                       # readout baseline above the 100 % gridline
SCOPE_COLORS = dict(idle=TEXT_2, event=LABELLUM_C, dot_idle=TEXT_1, grid=GRID, muted=MUTED)
READOUT_COLORS = dict(text=TEXT_1, event=LABELLUM_C, muted=MUTED)


def pair_heading(spec: PairSpec) -> str:
    return ODOR_LABELS[spec.odor]


def pair_header(idx: int) -> style.SegmentHeader:
    spec = PAIRS[idx]
    hdr = style.SegmentHeader(SEGMENT_INDEX[PAIR_NAMES[idx]], N_NUMBERED, pair_heading(spec))
    hdr.add_beside_heading(style.outline_badge(BADGES[spec.direction]), text=BADGES[spec.direction])
    return hdr


def scope_x_tick_labels() -> tuple[str, ...]:
    return tuple(scope.x_tick_label(s) for s in scope.x_tick_seconds(SCOPE_W, SCOPE_SPEED))


def pair_layout(idx: int) -> dict:
    """Pixel geometry of pair `idx` (top-left px): label row, both videos, both scopes, readouts."""
    spec = PAIRS[idx]
    vid_w, vid_h = base.fit_aspect(spec.crop[2], spec.crop[3], PANEL_W, H)
    label_y = pair_header(idx).bottom + PANEL_LABEL_GAP
    vid_y = label_y + VID_TOP_GAP
    trace_label_y = vid_y + vid_h + VID_TRACE_GAP
    scope_y = trace_label_y + YLABEL_GAP
    scope_h = SCOPE_BOTTOM - scope_y
    lo, hi = SCOPE_YLIM
    y100 = scope_y + (hi - 100.0) / (hi - lo) * scope_h          # px row of the 100 % gridline
    scopes = [(px + TRACE_INSET, scope_y, SCOPE_W, scope_h) for px in PANEL_X]
    return dict(label_y=label_y, trace_label_y=trace_label_y,
                vids=[(px, vid_y, vid_w, vid_h) for px in PANEL_X],
                scopes=scopes,
                readouts=[(sx + sw - READOUT_HEAD_GAP, y100 - READOUT_LIFT) for sx, _, sw, _ in scopes],
                pill_xs=[px + PANEL_W - base.pill_pad_px() for px in PANEL_X])


def odor_pill_states(now: float, odor_lens) -> list[bool]:
    """ODOR ON/OFF per column: each clip's pill follows its own odor window (both clips play
    during odor, so the Pre-Exposure column lights up exactly like the Post-Exposure one)."""
    return [0 <= now < L for L in odor_lens]


def pair_figure(idx: int, odor_lens, thrs):
    """Static matplotlib layer of pair `idx`: panel labels, one ODOR pill per column, both videos (clock,
    ring key) and both live scopes with their readouts.  odor_lens = odor duration (s) and thrs = PER
    threshold (%) per side.  Returns (fig, handles) with the artists build_pair updates every frame."""
    spec = PAIRS[idx]
    lay = pair_layout(idx)
    cw, ch = spec.crop[2], spec.crop[3]
    fig = base._new_fig()
    label_y = lay["label_y"]
    # stimulus state on each column's label row, right-aligned with its panel: both clips are
    # shown during odor, so each column header carries its own pill (the heading row carries the
    # odor name and its badge)
    odor_pills = [base._pill(fig, x, label_y, "ODOR", ODOR_C, False, fontsize=TEXT_PT, ha="right")
                  for x in lay["pill_xs"]]
    unit_w = base.measurer(fig, TEXT_PT, fontweight="bold")(scope.UNIT)
    panels = []
    for s_i, (label, odor_len, thr) in enumerate(zip(PANEL_TITLES, odor_lens, thrs)):
        px = PANEL_X[s_i]
        vx, vy, vid_w, vid_h = lay["vids"][s_i]
        fig.text(fx(px), fy(label_y), label, fontsize=TEXT_PT, fontweight="bold", color=TEXT_1, va="center")
        ax_v = px_axes(fig, vx, vy, vid_w, vid_h)
        im = ax_v.imshow(np.zeros((ch, cw), np.uint8), cmap="gray", vmin=0, vmax=255, interpolation="lanczos")
        ax_v.set_xticks([]); ax_v.set_yticks([])
        for sp in ax_v.spines.values():
            sp.set_edgecolor(GRID); sp.set_linewidth(1)
        (ring,) = ax_v.plot([], [], "o", ms=20, mfc="none", mec=LABELLUM_C, mew=2.4, path_effects=base.HALO)
        clock = ax_v.text(0.035, 0.965, "", transform=ax_v.transAxes, fontsize=TEXT_PT, color=TEXT_1, va="top",
                          fontweight="bold", path_effects=base.HALO)
        lab_w = base.measurer(fig, TEXT_PT)(LEGEND_LABEL) / vid_w
        key_y = 0.085
        ax_v.text(0.97, key_y, LEGEND_LABEL, transform=ax_v.transAxes, fontsize=TEXT_PT,
                  color=TEXT_1, ha="right", va="center", path_effects=base.HALO)
        ax_v.plot([0.97 - lab_w - 0.04], [key_y], "o", transform=ax_v.transAxes, ms=11, mfc="none",
                  mec=LABELLUM_C, mew=2, path_effects=base.HALO)

        # live scope: y label over its top-left corner, readout in the plot's top lane by the head
        fig.text(fx(px), fy(lay["trace_label_y"]), Y_LABEL, color=TEXT_2, fontsize=TEXT_PT, va="bottom")
        sx, sy, sw, sh = lay["scopes"][s_i]
        sc = scope.ScopePanel(px_axes(fig, sx, sy, sw, sh), sw, sh, thr, SCOPE_COLORS,
                              bands=[scope.Band("odor", ODOR_C, ODOR_BAND_ALPHA, "Odor", (0.0, odor_len))],
                              ylim=SCOPE_YLIM, yticks=SCOPE_YTICKS, speed=SCOPE_SPEED, font_pt=TEXT_PT)
        rx, rb = lay["readouts"][s_i]
        readout = scope.Readout(fig, fx, fy, rx, rb, READOUT_PT, TEXT_PT, unit_w, READOUT_COLORS)
        panels.append(dict(im=im, ring=ring, clock=clock, ax_v=ax_v, scope=sc, readout=readout))
    return fig, dict(odor_pills=odor_pills, panels=panels)


def build_pair(idx: int):
    spec = PAIRS[idx]
    base._setup_fonts()
    sides = [load_clip(spec, spec.before_trial), load_clip(spec, spec.after_trial)]
    n = _frame_count(spec.end_rel - spec.start_rel)
    hdr = pair_header(idx)
    fig, h = pair_figure(idx, [s["odor_len"] for s in sides], [s["thr"] for s in sides])
    odor_pills = h["odor_pills"]
    panels = [dict(p, clip=c) for p, c in zip(h["panels"], sides)]

    def frame(i):
        # one clock for both columns; each scope's head is its own on-screen frame's time, and
        # both keep streaming at 1x through the fade-out (no brake, no hold)
        now = spec.start_rel + i / FPS
        for p in panels:
            c = p["clip"]
            off = source_frame(i, 0, c["fps"])
            p["im"].set_data(stretch(c["video"][off], c["lo"], c["hi"]))
            k = min(int(np.searchsorted(c["frame"], c["first"] + off)), len(c["frame"]) - 1)
            rx, ry = base.stabilized_ring(c["lab_xy"][k], spec.crop, c["shifts"][off])
            ok = np.isfinite(rx) and np.isfinite(ry)
            p["ring"].set_data([rx] if ok else [], [ry] if ok else [])
            p["clock"].set_text(f"{now:+.1f} s")
            hv = p["scope"].update(c["rel"], c["pct"], head_rel(i, c["first"], c["on_frame"], c["fps"]))
            p["readout"].update(hv, c["thr"])
        for pill, on in zip(odor_pills, odor_pill_states(now, [s["odor_len"] for s in sides])):
            base._set_pill(pill, "ODOR", ODOR_C, on)
        return style.compose(_rgba(fig), i / FPS, n / FPS, hdr.els)

    base.assert_text_floor(fig)
    return n, frame, fig


# ---- 5. cohort dumbbell
SUMMARY_LABEL_W = 330                                   # row-label column (32 px labels)
SUMMARY_AX_GAP = 110                                    # header bottom -> axes top (legend row between)
SUMMARY_AX_BOTTOM = style.CAPTION_Y - 100               # tick labels + x label end above the caption


def summary_ax() -> tuple[int, int, int, int]:
    """x, top y, w, h (px) of the dumbbell axes, below the (three-line) header."""
    top = summary_header().bottom + SUMMARY_AX_GAP
    return G + SUMMARY_LABEL_W, top, W - 2 * G - SUMMARY_LABEL_W, SUMMARY_AX_BOTTOM - top


def build_summary():
    base._setup_fonts()
    fig = base._new_fig()
    hdr = summary_header()
    cap = style.Caption([SUMMARY_FOOTNOTE], None, G + 2, style.CAPTION_Y)
    ax_x, ax_y, ax_w, ax_h = summary_ax()
    # direct legend (marker + text) on its own row above the plot
    base.legend_row(fig, [("dot", MUTED, "Pre-exposure"), ("dot", LABELLUM_C, "Post-exposure"),
                          (None, MUTED, "whiskers: ± SEM")], G, ax_y - 52)
    ax = px_axes(fig, ax_x, ax_y, ax_w, ax_h, facecolor=SURFACE)
    ax.set_xlim(0, 3.7)
    ax.set_ylim(-0.6, 2.6)
    rows = {r.odor: 2 - i for i, r in enumerate(COHORT)}
    ax.set_yticks([rows[r.odor] for r in COHORT])
    ax.set_yticklabels([ODOR_LABELS[r.odor] for r in COHORT], fontsize=TEXT_PT, color=TEXT_1)
    ticks = [0, 0.5, 1, 1.5, 2, 2.5, 3]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{v:g}" for v in ticks])
    ax.tick_params(axis="x", colors=MUTED, labelsize=TEXT_PT, length=4)
    ax.tick_params(axis="y", length=0, pad=16)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("Mean PER score (0–5)", color=TEXT_2, fontsize=TEXT_PT, labelpad=8)

    items = []
    for r in COHORT:
        y = rows[r.odor]
        (conn,) = ax.plot([r.pre, r.pre], [y, y], color=MUTED, linewidth=3, solid_capstyle="round", zorder=2)
        (w0,) = ax.plot([r.pre - r.sem_pre, r.pre + r.sem_pre], [y, y], color=MUTED, linewidth=1.6, zorder=2,
                        marker="|", ms=12, mew=1.6)
        (w1,) = ax.plot([r.pre - r.sem_pre, r.pre + r.sem_pre], [y, y], color=LABELLUM_C, linewidth=1.6, zorder=3,
                        marker="|", ms=12, mew=1.6)
        (d0,) = ax.plot([r.pre], [y], "o", ms=17, color=MUTED, mec=SURFACE, mew=2, zorder=4)
        (d1,) = ax.plot([r.pre], [y], "o", ms=17, color=LABELLUM_C, mec=SURFACE, mew=2, zorder=5)
        lx = max(r.pre + r.sem_pre, r.post + r.sem_post) + 0.12
        lab = ax.text(lx, y + 0.12, f"{fmt_change(r.pre, r.post)}  {direction_glyph(r.pre, r.post)}",
                      fontsize=TEXT_PT, fontweight="bold", color=TEXT_1, va="center", path_effects=base.HALO)
        ptxt = ax.text(lx, y - 0.30, fmt_p(r.p), fontsize=TEXT_PT, color=TEXT_2, va="center",
                       path_effects=base.HALO)
        items.append((r, conn, w1, d1, lab, ptxt))

    def frame(i):
        t = i / FPS
        for k, (r, conn, w1, d1, lab, ptxt) in enumerate(items):
            y = rows[r.odor]
            p = style.smoothstep((t - (ROW_T0 + ROW_DT * k)) / ROW_DUR)   # ease in/out
            x1 = r.pre + (r.post - r.pre) * p
            sem = r.sem_pre + (r.sem_post - r.sem_pre) * p
            conn.set_data([r.pre, x1], [y, y])
            d1.set_data([x1], [y])
            w1.set_data([x1 - sem, x1 + sem], [y, y])
            a = _ramp(t, ROW_T0 + ROW_DT * k + LABEL_DELAY, LABEL_DUR)
            lab.set_alpha(a); ptxt.set_alpha(a)
        return style.compose(_rgba(fig), t, SUMMARY_S, hdr.els, [cap])

    base.assert_text_floor(fig)
    return _frame_count(SUMMARY_S), frame, fig


def summary_header() -> style.SegmentHeader:
    hm = style.run_measure(style.HEAD_SIZE, True)          # balanced lines: no one-word last line
    heading = "\n".join(ln for para in SUMMARY_TITLE.split(": ", 1)
                        for ln in style.wrap_balanced(para, W - 2 * style.MARGIN, hm))
    heading = heading.replace("Summary\n", "Summary:\n", 1)
    return style.SegmentHeader(SEGMENT_INDEX["summary"], N_NUMBERED, heading, SUMMARY_SUBTITLE)


# ---- 6. closing: lab header (headline + conclusion), future directions, then the
#         Odor -> Antennal Lobe -> Behavior schematic
def closing_statement_els(hdr: style.SegmentHeader) -> tuple[list, int]:
    """The simple conclusion: one large statement, centred in the frame below the header,
    flying in from the left like the lab bullets."""
    x, width = style.MARGIN, W - 2 * style.MARGIN
    text = CLOSING_LINES[0]
    lines = style.wrap_balanced(text, width, style.run_measure(CLOSING_STATEMENT_SIZE))   # even line lengths
    img = style.runs_image(style.as_lines(lines), CLOSING_STATEMENT_SIZE, TEXT_1, CLOSING_STATEMENT_LINE)
    y = max(hdr.bottom + 80, int(round(H / 2 - img.height / 2)))          # optical centre of the square
    el = style.tag(style.FlyInEl(img, x, y, CLOSING_STATEMENT_T, BULLET_FLY_DUR, BULLET_FLY_DX), text)
    return [el], y + img.height


def closing_els(hdr: style.SegmentHeader) -> tuple[list, int]:
    if not INCLUDE_FUTURE:
        return closing_statement_els(hdr)
    return future_els(hdr)


def future_els(hdr: style.SegmentHeader) -> tuple[list, int]:
    """Future-directions block in lab type (thin muted rule, bold sub-heading, red-square
    bullets in TEXT_2), keeping the original reveal times. Returns (elements, bottom y)."""
    x, width = style.MARGIN, W - 2 * style.MARGIN - 22
    y = hdr.bottom + 40
    d = style.REVEAL_DUR
    els: list = [style.BarEl(x, y, 300, 1, MUTED, FUTURE_RULE_T, style.RULE_DUR)]
    y += 28
    els.append(style.tag(style.TextEl(style.runs_image([[(FUTURE_HEADING, False)]], FUTURE_HEAD_SIZE, TEXT_1,
                                                       FUTURE_HEAD_SIZE + 10, bold=True),
                                      x, y, FUTURE_HEAD_T, d), FUTURE_HEADING, heading=True))
    y += FUTURE_HEAD_SIZE + 24
    bm = style.run_measure(style.BODY_SIZE)
    for j, b in enumerate(FUTURE_BULLETS):
        lines = style.wrap_text(b, width, bm)
        t0 = BULLET_T0 + BULLET_DT * j
        els.append(style.FlyInEl(style.lab.rect_image(7, 7, style.ACCENT_LINK), x, y + 12, t0,
                                 BULLET_FLY_DUR, BULLET_FLY_DX))
        els.append(style.tag(style.FlyInEl(style.runs_image(style.as_lines(lines), style.BODY_SIZE, TEXT_2,
                                                            style.BODY_LINE), x + 22, y, t0,
                                           BULLET_FLY_DUR, BULLET_FLY_DX), b))
        y += style.BODY_LINE * len(lines) + 14
    return els, y


def build_closing():
    base._setup_fonts()
    fig = base._new_fig()
    ax = _pix_axes(fig)
    hdr = closing_header()
    future, y = closing_els(hdr)

    boxes = []
    labels = SCHEMATIC_LABELS
    bw, bh, gap = 270, 84, 75                            # 32 px labels: "Behavior (PER)" ~230 px
    by = int(round((y + 58 + style.CAPTION_Y) / 2 - bh / 2))   # centred in the space left below the text
    x0 = (W - (3 * bw + 2 * gap)) / 2
    boxes.append(base._rule(fig, by - 58))               # sets the schematic apart from the text column
    for k, label in enumerate(labels):
        bx = x0 + k * (bw + gap)
        patch = FancyBboxPatch((bx, by), bw, bh, boxstyle="round,pad=0,rounding_size=16", fc=SURFACE,
                               ec=MUTED, lw=1.4)
        ax.add_patch(patch)
        tx2 = ax.text(bx + bw / 2, by + bh / 2, label, fontsize=TEXT_PT, color=TEXT_2, ha="center", va="center")
        arr = None
        if k < 2:
            arr = FancyArrowPatch((bx + bw + 10, by + bh / 2), (bx + bw + gap - 10, by + bh / 2),
                                  arrowstyle="-|>", mutation_scale=18, color=MUTED, lw=1.6)
            ax.add_patch(arr)
        boxes += [patch, tx2] + ([arr] if arr else [])

    if not INCLUDE_FUTURE:                                # simple cut: no schematic
        for a in boxes:
            a.remove()
        boxes = []

    def frame(i):
        t = i / FPS
        _set_alpha(boxes, _ramp(t, SCHEMATIC_T, SCHEMATIC_DUR))
        return style.compose(_rgba(fig), t, CLOSING_S, hdr.els + future, reveal_content=False)

    base.assert_text_floor(fig)
    return _frame_count(CLOSING_S), frame, fig


SCHEMATIC_LABELS = ["Odor", "Antennal Lobe", "Behavior (PER)"]


def closing_header() -> style.SegmentHeader:
    body = CLOSING_LINES[0] if INCLUDE_FUTURE else None      # simple cut: the statement is the slide
    return style.SegmentHeader(SEGMENT_INDEX["closing"], N_NUMBERED, CLOSING_HEADLINE, body)


# ------------------------------------------------------------------ reading time
def segment_durations() -> list[tuple[str, float]]:
    return ([("title", TITLE_S)] + ([("protocol", PROTOCOL_S)] if INCLUDE_PROTOCOL else [])
            + [(f"pair_{p.odor.lower()}", p.end_rel - p.start_rel) for p in PAIRS]
            + ([("summary", SUMMARY_S)] if INCLUDE_SUMMARY else []) + [("closing", CLOSING_S)])


def reading_schedule() -> list[tuple[str, float, float, list]]:
    """(segment, duration, fade-out, [ReadItem]): every on-screen text and the time it is fully
    revealed, so tests can check each stays up long enough to read."""
    R = style.ReadItem
    content = style.CONTENT_T0 + style.CONTENT_DUR
    out = [("title", TITLE_S, style.HERO_FADE_OUT, style.read_items(title_text_els()))]
    if INCLUDE_PROTOCOL:
        proto = protocol_header().read_items() + [R(" ".join(PANEL_CHIPS), BLOCK_T0 + BLOCK_DUR)]
        for k, (title, sub) in enumerate(BLOCKS):
            shown = BLOCK_T0 + BLOCK_DT * k + BLOCK_DUR
            proto += [R(title, shown), R(sub, shown)]
        out.append(("protocol", PROTOCOL_S, style.FADE_OUT_S, proto))
    for idx, spec in enumerate(PAIRS):
        items = pair_header(idx).read_items() + [R("ODOR OFF", content)]
        items += [R(t, content) for t in PANEL_TITLES] + [R(LEGEND_LABEL, content), R(Y_LABEL, content),
                                                          R("Odor", content)]
        out.append((PAIR_NAMES[idx], spec.end_rel - spec.start_rel, style.FADE_OUT_S, items))
    if INCLUDE_SUMMARY:
        out.append(_summary_reading(R, content))
    hdr = closing_header()
    close = hdr.read_items() + style.read_items(closing_els(hdr)[0])
    if INCLUDE_FUTURE:
        close += [R(lbl, SCHEMATIC_T + SCHEMATIC_DUR) for lbl in SCHEMATIC_LABELS]
    out.append(("closing", CLOSING_S, style.FADE_OUT_S, close))
    return out


def _summary_reading(R, content) -> tuple[str, float, float, list]:
    summ = summary_header().read_items() + [R(SUMMARY_FOOTNOTE, content), R("Pre-exposure Post-exposure", content),
                                            R("whiskers: ± SEM", content), R("Mean PER score (0–5)", content)]
    summ += [R(ODOR_LABELS[r.odor], content) for r in COHORT]
    for k, r in enumerate(COHORT):
        shown = ROW_T0 + ROW_DT * k + LABEL_DELAY + LABEL_DUR
        summ += [R(f"{fmt_change(r.pre, r.post)} {direction_glyph(r.pre, r.post)}", shown), R(fmt_p(r.p), shown)]
    return ("summary", SUMMARY_S, style.FADE_OUT_S, summ)


# ------------------------------------------------------------------------ driver
BUILDERS = ([build_title] + ([build_protocol] if INCLUDE_PROTOCOL else [])
            + [lambda: build_pair(0), lambda: build_pair(1), lambda: build_pair(2)])
BUILDERS += ([build_summary] if INCLUDE_SUMMARY else []) + [build_closing]


def _encoder(out_path: Path):
    return subprocess.Popen(
        [FFMPEG, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}",
         "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "16",
         "-pix_fmt", "yuv420p", str(out_path)], stdin=subprocess.PIPE)


def render_job(args) -> Path:
    """Every builder's frame(i) is already the composited frame (header, captions, fades)."""
    job, out_path = args
    n, frame, fig = BUILDERS[job]()
    enc = _encoder(out_path)
    for i in range(n):
        enc.stdin.write(np.ascontiguousarray(frame(i)).tobytes())
    enc.stdin.close()
    enc.wait()
    if fig is not None:
        plt.close(fig)
    return out_path


def render_preview(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for job, name in enumerate(NAMES):
        n, frame, fig = BUILDERS[job]()
        shots = {"": n - 1}
        if name.startswith("pair_"):
            spec = PAIRS[PAIR_NAMES.index(name)]
            shots = {"": int(round((spec.preview_rel - spec.start_rel) * FPS)), "_reveal": int(1.0 * FPS),
                     "_fade": n - 1 - int(style.FADE_OUT_S / 2 * FPS)}
        elif name == "protocol":
            shots = {"": int((PROTOCOL_S - 1.2) * FPS), "_mid": int((PUFF_T0 + 2.6 * PUFF_DT) * FPS),
                     "_reveal": int(1.0 * FPS)}
        elif name == "title":
            shots = {"": int(6.5 * FPS), "_reveal": int(1.2 * FPS)}
        elif name == "summary":
            shots = {"": int((SUMMARY_S - 1.2) * FPS), "_mid": int((ROW_T0 + ROW_DT + 0.8) * FPS)}
        elif name == "closing":
            shots = {"": int((CLOSING_S - 1.2) * FPS), "_mid": int((BULLET_T0 + BULLET_DT) * FPS)}
        for suffix, i in shots.items():
            p = out_dir / f"preview_{job}_{name}{suffix}.png"
            plt.imsave(p, frame(min(i, n - 1))[..., :3])
            print("wrote", p)
        if fig is not None:
            plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", type=Path, default=None, help="write one PNG per segment type here and exit")
    args = ap.parse_args()
    if args.preview:
        render_preview(args.preview)
        return
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        jobs = [(j, tmp / f"seg{j}.mp4") for j in range(len(BUILDERS))]
        with Pool(len(jobs)) as pool:
            parts = pool.map(render_job, jobs)
        listing = tmp / "concat.txt"
        listing.write_text("".join(f"file '{p}'\n" for p in parts))
        subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
                        "-c", "copy", "-movflags", "+faststart", str(OUT_PATH)], check=True)
    dur = float(subprocess.run([FFPROBE, "-v", "error", "-show_entries", "format=duration",
                                "-of", "csv=p=0", str(OUT_PATH)], capture_output=True, text=True).stdout)
    print(f"[OK] Wrote {OUT_PATH} ({dur:.1f} s)")


if __name__ == "__main__":
    main()
