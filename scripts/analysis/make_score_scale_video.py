#!/usr/bin/env python3
"""Explainer video of the PER (proboscis extension response) score scale, -1 .. 5.

1920 x 1080, 40 fps, silent. Sequence:
  title card  "PER Score Scale" + genotype, the seven score chips easing in
  one segment per spec entry (display order, normally -1, 0, 1a, 1b, 2, 3, 4, 5):
      left   large black-and-white close-up of ONE fly, ECC-stabilised (the
             make_per_montage_video.py stabiliser), cropped from the 4-fly frame
      right  score badge (fixed score palette from score_summary.py), peak
             readout, the spec's caption, and the fly's PER trace drawn in live
             with a playhead, odor window shaded and labelled "odor"; the same
             y-scale in every segment so a 4 and a 5 differ visibly
  closing card  every scored clip side by side, playing around its peak

Footage plays slowed (SPEED = 0.5x by default: every output frame is a new
source time, shown as the nearest recorded frame -- no blending, which ghosts a moving
proboscis), the crop is fixed (the flies are glued; --stabilize re-aligns each frame), and nothing ever freezes: footage, trace and playhead keep
moving through each fade-out, which the clip window covers. Text gets its
reading time (0.3 s / word) from the moment it is up; captions that cannot be
read in their segment, or that cannot fit, are rejected when the spec loads.

On-screen text never carries fly numbers, batch / folder / trial / dataset
names, sex / species boilerplate or "representative" footers: the genotype is
the only identity shown. Captions come from the spec and are checked for that.

Spec (JSON list, display order), one object per segment:
  score (int -1..5), variant ("a"|"b"|null), caption, dataset, batch,
  fly_number (1-4), trial_label, trial_dir, video, envelope_csv, trace_column,
  fps, odor_on_s, odor_off_s, clip_start_s, clip_end_s, peak_px_over_rest,
  peak_pct, alternates (ignored). Times are seconds from recording start.
  The eye / labellum track (crop + stabilisation) comes from
  <trial_dir>/<trial_dir.name>_fly<N>_distances.parquet and the light window
  (if any) from <trial_dir>/_trial_meta.json. The envelope CSV is read by row:
  row = video frame, so trace time = row / fps (its time_s column is row / 40).

Usage:
  python3 scripts/analysis/make_score_scale_video.py --spec clips.json \
      [--out PER_score_scale.mp4] [--preview-dir DIR] [--speed 0.5] [--no-closing]
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import to_rgb
from matplotlib.patches import FancyBboxPatch
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
for _p in (str(_HERE), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import lab_video_style as style  # noqa: E402
import make_per_montage_video as pm  # noqa: E402  (stabiliser, fonts, text helpers)
from scripts.analysis.score_summary import SCORE_COLORS, SCORE_MAX, SCORE_MIN, _score_cmap  # noqa: E402

FFMPEG, FFPROBE = pm.FFMPEG, pm.FFPROBE
DEFAULT_OUT = Path("/home/ramanlab/Documents/cole/Results/Figures/PER_score_scale.mp4")

W, H = 1920, 1080
FPS = 40
SPEED = 0.5                          # playback rate of the footage (0.5 = half speed)
CLIP_MIN_S, CLIP_MAX_S = 5.0, 10.0   # trial seconds per segment (spec contract)
# Fixed camera by default (user 2026-11-02): the flies are glued, and re-centring each frame
# makes the proboscis movement harder to see. --stabilize brings the ECC alignment back.
STABILIZE = False
# Nearest recorded frame, never a blend of two: blending ghosts a moving proboscis.
BLEND = False
SHARPEN_AMOUNT = 0.6                 # unsharp mask on the source crop (before the ~2.75x upscale)
SHARPEN_SIGMA = 1.2
CROP_WH = (360, 320)                 # source px cropped around the fly (same scale in every segment)
CROP_LEAD = 40                       # crop centre: this far from the eye toward the labellum

SURFACE, TEXT_1, TEXT_2, MUTED = style.SURFACE, style.TEXT_1, style.TEXT_2, style.MUTED
GRID, ODOR_C, LIGHT_C, TRACE_C = pm.GRID, pm.ODOR_C, pm.LIGHT_C, pm.LABELLUM_C
BAND_ALPHA = pm.BAND_ALPHA
TEXT_PT = style.MIN_TEXT_PT          # 32 px floor, in points at dpi 100
HALO = [pe.withStroke(linewidth=4, foreground=SURFACE)]

GENOTYPE = pm.GENOTYPE               # "Gr5a-GAL4 > UAS-CsChrimson"
GENOTYPE_RUNS = [(GENOTYPE, True)]
TITLE_HEADLINE = "PER Score Scale"
TITLE_BODY = "Proboscis extension response, scored from −1 to 5"
HEADER_LABEL = "PER Score Scale"
CLOSING_LABEL = "All Scores Side by Side"
PEAK_LABEL = "Peak response"
DIP_LABEL = "Largest dip"
ODOR_LABEL = "odor"
LIGHT_LABEL = "light"
SCORE_WORD = "Score"
Y_LABEL = {"pct": "PER (% over rest)", "px": "Extension over rest (px)"}


def SPEED_LABEL(speed: float) -> str:  # noqa: N802 (reads as a constant at call sites)
    return f"{speed:g}× speed"


CANONICAL = [(-1, None), (0, None), (1, "a"), (1, "b"), (2, None), (3, None), (4, None), (5, None)]

# --------------------------------------------------------------------------- layout (px, top-left)
G = 72                               # outer gutter
NUM_Y = 44                           # header numeral row top
VID = (G, 128, 990, 880)             # video panel x, y, w, h (crop aspect 360:320)
COL_X = G + VID[2] + 56              # right column
COL_R = W - G
COL_W = COL_R - COL_X
BADGE = (COL_X, VID[1], 236, 236)
PEAK_X = COL_X + BADGE[2] + 44
CAPTION_PX, CAPTION_LINE = 36, 50
CAPTION_W = COL_W
MAX_CAPTION_LINES = 3
CAP_Y = BADGE[1] + BADGE[3] + 36     # first caption block line top (example tag or caption)
TAG_LINE = 44
LANE_Y = 634                         # y label / band labels (vertical centre)
TRACE = (COL_X + 74, 676, COL_R - (COL_X + 74) - 6, 284)    # trace axes x, y, w, h
CHIP_W, CHIP_H, CHIP_GAP = 56, 44, 6
RAMP_RIGHT = COL_R - 200             # header ramp ends here (brand sits to its right)


def to_px_fig(x: float, y: float) -> tuple[float, float]:
    return x / W, 1.0 - y / H


def px_axes(fig, x, y_top, w, h, **kw):
    return fig.add_axes([x / W, 1.0 - (y_top + h) / H, w / W, h / H], **kw)


def pt(px: float) -> float:
    """Font size in points (dpi 100) of a px size."""
    return px * 72 / 100


# =========================================================================== scores
def score_color(score: int) -> str:
    """The fixed, CVD-validated score palette entry (score_summary.SCORE_COLORS)."""
    if not (isinstance(score, (int, np.integer)) and SCORE_MIN <= score <= SCORE_MAX):
        raise ValueError(f"score {score!r} outside {SCORE_MIN}..{SCORE_MAX}")
    cmap, norm = _score_cmap()
    hx = SCORE_COLORS[int(score) - SCORE_MIN]
    assert np.allclose(cmap(norm(score))[:3], to_rgb(hx))
    return hx


def _lum(hex_: str) -> float:
    c = np.array(to_rgb(hex_))
    c = np.where(c <= 0.03928, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return float(0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2])


def badge_text_color(score: int) -> str:
    """Text on a score badge: the surface on pale badges, TEXT_1 on dark ones (higher contrast)."""
    bg = _lum(score_color(score))
    dark, light = _lum(SURFACE), _lum(TEXT_1)
    return SURFACE if (bg + 0.05) / (dark + 0.05) >= (light + 0.05) / (bg + 0.05) else TEXT_1


def score_label(score: int) -> str:
    return f"−{abs(score)}" if score < 0 else str(score)


# =========================================================================== spec
class SpecError(ValueError):
    pass


@dataclass(frozen=True)
class ClipSpec:
    score: int
    variant: str | None
    caption: str
    dataset: str
    batch: str
    fly_number: int
    trial_label: str
    trial_dir: Path
    video: Path
    envelope_csv: Path
    trace_column: str
    fps: float
    odor_on_s: float
    odor_off_s: float
    clip_start_s: float
    clip_end_s: float
    peak_px_over_rest: float
    peak_pct: float

    @property
    def label(self) -> str:
        return f"{self.score}{self.variant or ''}"

    @property
    def duration_s(self) -> float:
        return self.clip_end_s - self.clip_start_s


STR_FIELDS = ("caption", "dataset", "batch", "trial_label", "trace_column")
PATH_FIELDS = ("trial_dir", "video", "envelope_csv")
NUM_FIELDS = ("fps", "odor_on_s", "odor_off_s", "clip_start_s", "clip_end_s", "peak_px_over_rest", "peak_pct")
REQUIRED = ("score", "variant", "fly_number") + STR_FIELDS + PATH_FIELDS + NUM_FIELDS


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


_BANNED = [
    (re.compile(r"\bfly\s*#?\s*\d", re.I), "fly number"),
    (re.compile(r"batch", re.I), "batch name"),
    (re.compile(r"\brig\b|\brig[_ ]?\d", re.I), "rig name"),
    (re.compile(r"(testing|training)_\d", re.I), "trial name"),
    (re.compile(r"\bfemales?\b", re.I), "'Female'"),
    (re.compile(r"single[- ]fly", re.I), "'Single fly'"),
    (re.compile(r"drosophila|melanogaster", re.I), "species name"),
    (re.compile(r"representative", re.I), "'representative individual'"),
    (re.compile(r"\b[\w.]+-(training|control|naive|sensitivity)\b", re.I), "dataset name"),
]


def caption_violations(text: str, clip: ClipSpec | None = None) -> list[str]:
    """Reasons a caption breaks the on-screen text rules (empty list = fine)."""
    out = [f"caption names a {what}" for rx, what in _BANNED if rx.search(text)]
    if clip is not None:
        low = text.lower()
        for name in (clip.dataset, clip.batch, clip.batch.rsplit("_rig", 1)[0], clip.trial_label,
                     clip.trial_dir.name):
            if name and len(name) >= 4 and name.lower() in low:
                out.append(f"caption contains the provenance string {name!r}")
    return out


def caption_lines(text: str, max_w: float = CAPTION_W, size: int = CAPTION_PX) -> list[str]:
    """Word-wrapped caption lines (measured with the Arial the caption is drawn in)."""
    m = style.run_measure(size)
    return [ln for para in text.split("\n") for ln in style.wrap_text(para, max_w * 0.985, m)]


def example_tags(clips) -> list[str | None]:
    """'Example k of n' for scores shown more than once (1a / 1b), else None."""
    counts = {}
    for c in clips:
        counts[c.score] = counts.get(c.score, 0) + 1
    seen, out = {}, []
    for c in clips:
        if counts[c.score] > 1:
            seen[c.score] = seen.get(c.score, 0) + 1
            out.append(f"Example {seen[c.score]} of {counts[c.score]}")
        else:
            out.append(None)
    return out


def _rank(c: ClipSpec) -> int:
    key = (c.score, c.variant if c.score == 1 else None)
    if c.score == 1 and c.variant is None:
        key = (1, "a")
    return CANONICAL.index(key) if key in CANONICAL else 99


def order_warnings(clips) -> list[str]:
    out, seen = [], set()
    ranks = [_rank(c) for c in clips]
    for c in clips:
        if c.label in seen:
            out.append(f"segment {c.label} appears twice")
        seen.add(c.label)
    if any(b < a for a, b in zip(ranks, ranks[1:])):
        out.append("segments are out of the canonical order -1, 0, 1a, 1b, 2, 3, 4, 5: "
                   + ", ".join(c.label for c in clips))
    return out


def _validate(i: int, d) -> list[str]:
    err = []
    if not isinstance(d, dict):
        return [f"[{i}] entry must be an object"]
    for k in REQUIRED:
        if k not in d:
            err.append(f"[{i}] missing field {k}")
    if err:
        return err
    if not (_is_int(d["score"]) and SCORE_MIN <= d["score"] <= SCORE_MAX):
        err.append(f"[{i}] score must be an int in {SCORE_MIN}..{SCORE_MAX}, got {d['score']!r}")
    if d["variant"] not in ("a", "b", None):
        err.append(f"[{i}] variant must be 'a', 'b' or null, got {d['variant']!r}")
    if not (_is_int(d["fly_number"]) and 1 <= d["fly_number"] <= 4):
        err.append(f"[{i}] fly_number must be an int in 1..4, got {d['fly_number']!r}")
    for k in STR_FIELDS:
        if not (isinstance(d[k], str) and d[k].strip()):
            err.append(f"[{i}] {k} must be a non-empty string")
    for k in PATH_FIELDS:
        if not (isinstance(d[k], str) and Path(d[k]).is_absolute()):
            err.append(f"[{i}] {k} must be an absolute path, got {d[k]!r}")
    for k in NUM_FIELDS:
        if not _is_num(d[k]):
            err.append(f"[{i}] {k} must be a finite number, got {d[k]!r}")
    if err:
        return err
    if d["fps"] <= 0:
        err.append(f"[{i}] fps must be > 0")
    if d["clip_start_s"] < 0:
        err.append(f"[{i}] clip_start_s must be >= 0")
    if d["odor_off_s"] <= d["odor_on_s"]:
        err.append(f"[{i}] odor_off_s must be > odor_on_s")
    dur = d["clip_end_s"] - d["clip_start_s"]
    if dur <= 0:
        err.append(f"[{i}] clip_end_s must be > clip_start_s")
    elif not CLIP_MIN_S - 1e-6 <= dur <= CLIP_MAX_S + 1e-6:
        err.append(f"[{i}] clip window is {dur:.2f} s; it must be {CLIP_MIN_S:g}–{CLIP_MAX_S:g} s")
    return err


def parse_spec(obj, check_files: bool = False, speed: float = SPEED) -> list[ClipSpec]:
    """Validate a decoded spec and return its clips; every problem is reported in one SpecError."""
    if not isinstance(obj, list) or not obj:
        raise SpecError("spec must be a non-empty JSON list of clip objects")
    errors, clips = [], []
    for i, d in enumerate(obj):
        e = _validate(i, d)
        if e:
            errors += e
            continue
        c = ClipSpec(
            score=int(d["score"]), variant=d["variant"], caption=d["caption"].strip(),
            dataset=d["dataset"], batch=d["batch"], fly_number=int(d["fly_number"]),
            trial_label=d["trial_label"], trial_dir=Path(d["trial_dir"]), video=Path(d["video"]),
            envelope_csv=Path(d["envelope_csv"]), trace_column=d["trace_column"],
            **{k: float(d[k]) for k in NUM_FIELDS})
        errors += [f"[{i}] {v}" for v in caption_violations(c.caption, c)]
        n_lines = len(caption_lines(c.caption))
        if n_lines > MAX_CAPTION_LINES:
            errors.append(f"[{i}] caption needs {n_lines} lines; at most {MAX_CAPTION_LINES} fit")
        for it, hold, need in reading_shortfalls(c, speed):
            errors.append(f"[{i}] {it.text[:40]!r} is up {hold:.1f} s but needs {need:.1f} s to read; "
                          "shorten the caption or lengthen the clip")
        if check_files:
            for k in PATH_FIELDS:
                if not Path(d[k]).exists():
                    errors.append(f"[{i}] {k} {d[k]} does not exist")
            if Path(d["trial_dir"]).exists() and not distances_path(c).exists():
                errors.append(f"[{i}] eye/labellum track {distances_path(c)} does not exist")
        clips.append(c)
    if errors:
        raise SpecError("invalid score-scale spec:\n  " + "\n  ".join(errors))
    return clips


def load_spec(path: Path, check_files: bool = False, speed: float = SPEED) -> list[ClipSpec]:
    try:
        obj = json.loads(Path(path).read_text())
    except json.JSONDecodeError as e:
        raise SpecError(f"{path}: not valid JSON ({e})") from e
    return parse_spec(obj, check_files=check_files, speed=speed)


def distances_path(c: ClipSpec) -> Path:
    return c.trial_dir / f"{c.trial_dir.name}_fly{c.fly_number}_distances.parquet"


# =========================================================================== timing
def n_output_frames(c: ClipSpec, speed: float = SPEED) -> int:
    """Encoded frames of a segment: the clip window played at `speed` (fade-out included)."""
    return int(round(c.duration_s / speed * FPS))


def on_screen_s(c: ClipSpec, speed: float = SPEED) -> float:
    return n_output_frames(c, speed) / FPS


def source_time(k: int, c: ClipSpec, speed: float = SPEED) -> float:
    """Trial seconds shown at encoded frame k: advances speed / FPS every frame, never held."""
    return c.clip_start_s + k * speed / FPS


def frame_time(frame: float, fps: float) -> float:
    return frame / fps


def frame_pos(t: float, fps: float) -> float:
    return t * fps


def source_frame_range(c: ClipSpec) -> tuple[int, int]:
    """(first, last) recorded frames decoded for the segment: every blend neighbour."""
    return int(math.floor(c.clip_start_s * c.fps + 1e-9)), int(math.ceil(c.clip_end_s * c.fps - 1e-9))


def blend_index(t: float, fps: float, first: int) -> tuple[int, int, float]:
    """Neighbouring decoded frames (relative to `first`) and the blend weight of the later one."""
    p = round(frame_pos(t, fps) - first, 6)
    i0 = int(math.floor(p))
    w = p - i0
    if w < 1e-6:
        return i0, i0, 0.0
    return i0, i0 + 1, float(w)


def frame_for(frames: np.ndarray, i0: int, i1: int, w: float) -> np.ndarray:
    """The frame shown at a sub-frame position: the nearest recorded frame (BLEND False), or the
    linear blend of the two neighbours (BLEND True)."""
    if BLEND:
        return blend(frames, i0, i1, w)
    return frames[i1] if w >= 0.5 else frames[i0]


def sharpen(frames: np.ndarray, amount: float = SHARPEN_AMOUNT, sigma: float = SHARPEN_SIGMA) -> np.ndarray:
    """Unsharp mask, frame by frame: lifts edge contrast lost to compression and the upscale;
    flat regions are unchanged."""
    if amount <= 0:
        return frames
    out = np.empty_like(frames)
    for i, f in enumerate(frames):
        blur = cv2.GaussianBlur(f, (0, 0), sigma)
        out[i] = cv2.addWeighted(f, 1 + amount, blur, -amount, 0)
    return out


def blend(frames: np.ndarray, i0: int, i1: int, w: float) -> np.ndarray:
    if w <= 0 or i0 == i1:
        return frames[i0]
    return (frames[i0].astype(np.float32) * (1 - w) + frames[i1].astype(np.float32) * w + 0.5).astype(np.uint8)


# =========================================================================== traces
def trace_kind(column: str) -> str:
    """'pct' for normalised PER columns (combined_pct, envelope_of_rms, ...), else 'px'."""
    low = column.lower()
    return "pct" if any(s in low for s in ("pct", "percent", "rms", "envelope")) else "px"


def trace_series(env: pd.DataFrame, column: str, fps: float, odor_on_s: float, pre_s: float = 10.0):
    """(t, y): row r is video frame r, so t = r / fps. Every trace is made relative to its own
    pre-odor median (0 = rest), so the line matches the rise-over-rest readout and a
    retraction dips below 0."""
    if column not in env.columns:
        raise KeyError(f"trace column {column!r} not in envelope CSV (has {list(env.columns)})")
    y = env[column].to_numpy(float)
    t = np.arange(len(y)) / fps
    pre = y[(t >= odor_on_s - pre_s) & (t < odor_on_s)]
    y = y - np.nanmedian(pre)
    return t, y


def trace_at(t: float, ts: np.ndarray, ys: np.ndarray) -> float:
    return float(np.interp(t, ts, ys))


def window(ts, ys, t0, t1):
    sel = (ts >= t0) & (ts <= t1)
    return ys[sel]


def peak_index(c: ClipSpec, ts: np.ndarray, ys: np.ndarray) -> int | None:
    """Trace sample of the clip's response peak: the highest point inside odor window and clip window
    (the whole clip window when the two do not overlap)."""
    clip = (ts >= c.clip_start_s) & (ts <= c.clip_end_s) & np.isfinite(ys)
    sel = clip & (ts >= c.odor_on_s) & (ts < c.odor_off_s)
    if not sel.any():
        sel = clip
    if not sel.any():
        return None
    idx = np.flatnonzero(sel)
    return int(idx[np.argmax(ys[idx])])


def _nice_step(span: float) -> float:
    for s in (5, 10, 20, 25, 50, 100, 200, 250, 500):
        if span / s <= 4.4:
            return s
    return 10 ** math.ceil(math.log10(span))


def shared_ylim(windows, kind: str) -> tuple[float, float]:
    """One y-range for every segment: covers each clip's highest point with headroom; a PER %
    axis always shows 0..100."""
    top = max(float(np.nanmax(w)) for w in windows if len(w))
    bot = min(float(np.nanmin(w)) for w in windows if len(w))
    if kind == "pct":
        top = max(top, 100.0)
    hi = top * 1.08
    lo = min(-0.06 * hi, bot - 0.04 * hi)
    if bot < -0.1 * hi:                     # a real dip (retraction): give it a labelled tick
        step = _nice_step(hi)
        lo = -math.ceil(-bot / step) * step - 0.04 * hi
    return lo, hi


def y_ticks(lo: float, hi: float) -> list[float]:
    step = _nice_step(hi)
    start = math.ceil(lo / step) * step if lo < -step / 2 else 0.0
    return [float(v) for v in np.arange(start, hi + 1e-9, step) if v >= lo]


def _signed(v: float) -> str:
    return f"{'+' if v >= 0 else chr(0x2212)}{abs(v):.0f}"


def is_dip(c: ClipSpec) -> bool:
    """A retraction clip: its readout is the trough below rest, not a peak above it."""
    return c.peak_px_over_rest < 0 or c.peak_pct < 0


def peak_label(c: ClipSpec) -> str:
    return DIP_LABEL if is_dip(c) else PEAK_LABEL


def peak_text(c: ClipSpec, kind: str) -> tuple[str, str]:
    """(big readout, small line under it)."""
    if is_dip(c):
        px = f"{abs(c.peak_px_over_rest):.0f} px below rest"
        if kind == "pct":
            return f"{_signed(c.peak_pct)} %", px
        return f"{_signed(c.peak_px_over_rest)} px", ""
    if kind == "pct":
        return f"{round(c.peak_pct):.0f} %", f"{_signed(c.peak_px_over_rest)} px over rest"
    return f"{_signed(c.peak_px_over_rest)} px", ""


def stretch_contrast(frames: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.5) -> np.ndarray:
    """One linear grey-level stretch for a whole clip (percentiles over every frame), so dim,
    low-contrast footage fills the range without per-frame flicker."""
    lo, hi = np.percentile(frames, [lo_pct, hi_pct])
    if hi - lo < 1:
        return frames
    out = (frames.astype(np.float32) - lo) * (255.0 / (hi - lo))
    return np.clip(out, 0, 255).astype(np.uint8)


# =========================================================================== crop
def crop_for(eye_xy, lab_xy, frame_wh, crop_wh=CROP_WH, lead: float = CROP_LEAD):
    """Fixed-size crop around one fly: centred `lead` px from the eye toward the labellum's
    median position (so a mirrored rig leads the other way), clamped inside the frame."""
    eye = np.asarray(eye_xy, float)
    lab = np.asarray(lab_xy, float).reshape(-1, 2)
    lab = lab[np.isfinite(lab).all(1)]
    u = np.array([0.0, 1.0])
    if len(lab):
        v = np.median(lab, axis=0) - eye
        if np.hypot(*v) > 1:
            u = v / np.hypot(*v)
    cx, cy = eye + u * lead
    w, h = crop_wh
    x0 = int(np.clip(round(cx - w / 2), 0, frame_wh[0] - w))
    y0 = int(np.clip(round(cy - h / 2), 0, frame_wh[1] - h))
    return x0, y0, w, h


# =========================================================================== reading time
CONTENT_SHOWN = style.CONTENT_T0 + style.CONTENT_DUR      # matplotlib layer fully in


def segment_read_items(c: ClipSpec, tag: str | None = None, kind: str = "pct") -> list[style.ReadItem]:
    R = style.ReadItem
    big, small = peak_text(c, kind)
    items = [R(c.caption, CONTENT_SHOWN), R(f"{peak_label(c)} {big} {small}", CONTENT_SHOWN),
             R(Y_LABEL[kind], CONTENT_SHOWN), R(ODOR_LABEL, CONTENT_SHOWN)]
    if tag:
        items.append(R(tag, CONTENT_SHOWN))
    return items


def reading_shortfalls(c: ClipSpec, speed: float = SPEED, tag: str | None = None):
    return style.reading_shortfalls(segment_read_items(c, tag), on_screen_s(c, speed), style.FADE_OUT_S)


# =========================================================================== PIL pieces
def chip_image(score: int, w: int, h: int, size: int, radius: int | None = None) -> Image.Image:
    """Rounded score chip in the fixed palette with its number (PIL, for the title / closing)."""
    ss = 4
    radius = radius if radius is not None else h // 5
    mask = Image.new("L", (w * ss, h * ss), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, w * ss - 1, h * ss - 1), radius * ss, fill=255)
    chip = Image.new("RGBA", (w, h), style.rgb(score_color(score)) + (255,))
    chip.putalpha(mask.resize((w, h), Image.LANCZOS))
    num = style.runs_image([[(score_label(score), False)]], size, badge_text_color(score), size, bold=True)
    bbox = num.getbbox() or (0, 0, num.width, num.height)
    gw, gh = bbox[2] - bbox[0], bbox[3] - bbox[1]
    chip.alpha_composite(num, (int(round((w - gw) / 2 - bbox[0])), int(round((h - gh) / 2 - bbox[1]))))
    return chip


def header_els(idx: int | None, n: int, label: str = HEADER_LABEL) -> list:
    """Lab header row at 1920 px: '0N / 0M' numeral + accent rule + label, brand at the right.
    idx None = an unnumbered card (closing): rule + label only."""
    T0, d = style.HEADER_T0, style.REVEAL_DUR
    els = []
    x = G
    if idx is not None:
        num_s, of_s = style.header_numerals(idx + 1, n)
        num = style.runs_image([[(num_s, False)]], style.lab.NUMERAL_SIZE, style.ACCENT_LINK,
                               style.lab.NUMERAL_SIZE + 4, bold=True)
        of = style.runs_image([[(of_s, False)]], style.CAPTION_SIZE, MUTED, style.CAPTION_LINE)
        els += [style.TextEl(num, x, NUM_Y, T0["num"], d, rise=10),
                style.TextEl(of, x + num.width + 4, NUM_Y + 3, T0["of"], d, rise=10)]
        x += num.width + of.width + 18
    els.append(style.BarEl(x, NUM_Y + 15, style.HEADER_RULE_W, 2, style.ACCENT, T0["rule"], style.RULE_DUR))
    x += style.HEADER_RULE_W + 18
    lab_img = style.runs_image([[(label, False)]], style.CAPTION_SIZE, TEXT_2, style.CAPTION_LINE)
    els.append(style.tag(style.TextEl(lab_img, x, NUM_Y + 3, T0["heading"], d, rise=10), label))
    brand = style.runs_image([[(style.LAB_SHORT, False)]], style.CAPTION_SIZE, MUTED, style.CAPTION_LINE)
    els.append(style.TextEl(brand, W - G - brand.width + 4, NUM_Y + 3, T0["brand"], d, rise=10))
    return els


# =========================================================================== segment figure
def _new_fig():
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=SURFACE)
    pm.FigureCanvasAgg(fig)
    return fig


def _ftext(fig, x, y, s, size_px, **kw):
    fx, fy = to_px_fig(x, y)
    return fig.text(fx, fy, s, fontsize=pt(size_px), **kw)


def _box(fig, x, y, w, h, fc, ec="none", lw=0.0, r=0.0):
    fx, fy = to_px_fig(x, y + h)
    p = FancyBboxPatch((fx, fy), w / W, h / H, boxstyle=f"round,pad=0,rounding_size={r / H}",
                       transform=fig.transFigure, fc=fc, ec=ec, lw=lw, mutation_aspect=H / W)
    fig.add_artist(p)
    return p


def _measure(fig, size_px, **kw):
    return pm.measurer(fig, pt(size_px), **kw)


def ramp(fig, active: int | None, x_right: float = RAMP_RIGHT, y: float = NUM_Y + 18):
    """Header score ramp: the seven chips -1..5, the current score outlined (TEXT_1) and marked
    by the lab accent beneath (a marker, not data)."""
    n = SCORE_MAX - SCORE_MIN + 1
    x = x_right - n * CHIP_W - (n - 1) * CHIP_GAP
    for s in range(SCORE_MIN, SCORE_MAX + 1):
        on = s == active
        _box(fig, x, y - CHIP_H / 2, CHIP_W, CHIP_H, score_color(s), ec=TEXT_1 if on else "none",
             lw=3.0 if on else 0.0, r=8)
        _ftext(fig, x + CHIP_W / 2, y, score_label(s), 32, ha="center", va="center", fontweight="bold",
               color=badge_text_color(s))
        if on:
            fig.add_artist(plt.Line2D([(x + 8) / W, (x + CHIP_W - 8) / W], [1 - (y + CHIP_H / 2 + 9) / H] * 2,
                                      transform=fig.transFigure, color=style.ACCENT_LINK, lw=4,
                                      solid_capstyle="butt"))
        x += CHIP_W + CHIP_GAP


def segment_figure(c: ClipSpec, idx: int, n: int, ylim, data: dict, tag: str | None,
                   crop_wh=CROP_WH, speed: float = SPEED):
    """Static matplotlib layer of one segment; returns (fig, handles) updated per frame.
    data: t, y (full trace), light ((on, off) trial s | None), kind ('pct'|'px')."""
    pm._setup_fonts()
    fig = _new_fig()
    kind = data["kind"]
    odor_on, odor_off = c.odor_on_s, c.odor_off_s

    # ---- video panel
    vx, vy, vw, vh = VID
    ax_v = px_axes(fig, vx, vy, vw, vh)
    im = ax_v.imshow(np.zeros(crop_wh[::-1], np.uint8), cmap="gray", vmin=0, vmax=255, interpolation="lanczos")
    ax_v.set_xticks([]); ax_v.set_yticks([])
    for s in ax_v.spines.values():
        s.set_edgecolor(GRID); s.set_linewidth(1)
    clock = ax_v.text(0.025, 0.972, "", transform=ax_v.transAxes, fontsize=TEXT_PT, color=TEXT_1, va="top",
                      fontweight="bold", path_effects=HALO)
    ax_v.text(0.975, 0.972, SPEED_LABEL(speed), transform=ax_v.transAxes, fontsize=TEXT_PT, color=TEXT_2,
              va="top", ha="right", path_effects=HALO)
    pill = ax_v.text(0.025, 0.035, "", transform=ax_v.transAxes, fontsize=TEXT_PT, fontweight="bold",
                     va="bottom", ha="left", bbox=dict(boxstyle="round,pad=0.45,rounding_size=0.9", lw=1.6))
    light_pill = None
    if data.get("light"):
        light_pill = ax_v.text(0.025 + 0.215, 0.035, "", transform=ax_v.transAxes, fontsize=TEXT_PT,
                               fontweight="bold", va="bottom", ha="left",
                               bbox=dict(boxstyle="round,pad=0.45,rounding_size=0.9", lw=1.6))

    # ---- header ramp
    ramp(fig, c.score)

    # ---- badge + peak readout
    bx, by, bw, bh = BADGE
    _box(fig, bx, by, bw, bh, score_color(c.score), r=26)
    fg = badge_text_color(c.score)
    _ftext(fig, bx + bw / 2, by + 24, SCORE_WORD, 32, ha="center", va="top", color=fg)
    _ftext(fig, bx + bw / 2, by + bh / 2 + 24, score_label(c.score), 140, ha="center", va="center",
           fontweight="bold", color=fg)
    big, small = peak_text(c, kind)
    _ftext(fig, PEAK_X, by + 30, peak_label(c), 32, va="center", color=TEXT_2)
    _ftext(fig, PEAK_X - 4, by + 110, big, 88, va="center", fontweight="bold", color=TEXT_1)
    if small:
        _ftext(fig, PEAK_X, by + 196, small, 32, va="center", color=TEXT_2)

    # ---- caption block
    y = CAP_Y
    if tag:
        _ftext(fig, COL_X, y, tag, 32, va="top", color=MUTED)
        y += TAG_LINE
    for i, ln in enumerate(caption_lines(c.caption)):         # one artist per line: exact line pitch
        _ftext(fig, COL_X, y + i * CAPTION_LINE, ln, CAPTION_PX, va="top", color=TEXT_1)

    # ---- trace
    tx, ty, tw, th = TRACE
    ax_t = px_axes(fig, tx, ty, tw, th, facecolor=SURFACE)
    x0, x1 = c.clip_start_s - odor_on, c.clip_end_s - odor_on
    lo, hi = ylim
    ax_t.set_xlim(x0, x1)
    ax_t.set_ylim(lo, hi)

    def xpx(v):
        return tx + (np.clip(v, x0, x1) - x0) / (x1 - x0) * tw

    ylab = _ftext(fig, COL_X, LANE_Y, Y_LABEL[kind], 32, va="center", color=TEXT_2)
    ylab_r = COL_X + _measure(fig, 32)(Y_LABEL[kind])
    bands = [("odor", ODOR_C, ODOR_LABEL, 0.0, odor_off - odor_on)]
    if data.get("light"):
        bands.append(("light", LIGHT_C, LIGHT_LABEL, data["light"][0] - odor_on, data["light"][1] - odor_on))
    mb = _measure(fig, 32, fontweight="bold")
    label_x = ylab_r + 28
    for key, col, lab, b0, b1 in bands:
        if b1 <= x0 or b0 >= x1:
            continue
        ax_t.axvspan(b0, b1, color=col, alpha=BAND_ALPHA[key], lw=0)
        p0, p1 = xpx(b0), xpx(b1)
        fig.add_artist(plt.Line2D([p0 / W, p1 / W], [1 - (ty - 6) / H] * 2, transform=fig.transFigure,
                                  color=col, lw=5, solid_capstyle="butt"))
        lx = max(p0 + 2, label_x)
        _ftext(fig, lx, LANE_Y, lab, 32, va="center", fontweight="bold", color=col)
        label_x = lx + mb(lab) + 28
    pps = tw / (x1 - x0)
    ticks = [v for v in pm.x_ticks(x0, x1) if min(v - x0, x1 - v) * pps >= pm.X_TICK_EDGE]
    ax_t.set_xticks(ticks)
    ax_t.set_xticklabels([pm.x_tick_label(v) for v in ticks])
    ax_t.set_yticks(y_ticks(lo, hi))
    ax_t.set_yticklabels([f"{v:.0f}" for v in y_ticks(lo, hi)])
    ax_t.tick_params(colors=MUTED, labelsize=TEXT_PT, length=4)
    for side in ("top", "right"):
        ax_t.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax_t.spines[side].set_color(GRID)
    ax_t.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax_t.set_axisbelow(True)
    ax_t.axhline(0, color=MUTED, linewidth=0.8, linestyle=(0, (2, 3)))
    (line,) = ax_t.plot([], [], color=TRACE_C, linewidth=3.0, solid_capstyle="round")
    playhead = ax_t.axvline(x0, color=TEXT_2, linewidth=1.2)
    (dot,) = ax_t.plot([], [], "o", ms=11, color=TRACE_C, mec=SURFACE, mew=2)
    (peak_mark,) = ax_t.plot([], [], "o", ms=16, mfc="none", mec=TEXT_1, mew=2)
    ts, ys = data["t"], data["y"]
    k_peak = peak_index(c, ts, ys)
    return fig, dict(ax_v=ax_v, ax_t=ax_t, im=im, clock=clock, pill=pill, light_pill=light_pill, line=line,
                     playhead=playhead, dot=dot, peak_mark=peak_mark, x_range=(x0, x1), ylab=ylab,
                     peak=(ts[k_peak] - odor_on, ys[k_peak]) if k_peak is not None else None)


def _set_pill(txt, label: str, color: str, active: bool) -> None:
    txt.set_text(f"{label} {'ON' if active else 'OFF'}")
    txt.set_color(SURFACE if active else TEXT_2)
    p = txt.get_bbox_patch()
    p.set_edgecolor(color if active else MUTED)
    p.set_facecolor(color if active else SURFACE)
    p.set_alpha(1.0 if active else 0.85)


def text_boxes(fig):
    return pm.text_boxes(fig)


def assert_text_floor(fig) -> None:
    pm.assert_text_floor(fig)


def update_frame(hd: dict, c: ClipSpec, data: dict, t_src: float) -> None:
    """Point every live artist at trial time t_src (sub-frame: the trace is interpolated)."""
    ts, ys = data["t"], data["y"]
    now = t_src - c.odor_on_s
    x0, _ = hd["x_range"]
    a = np.searchsorted(ts, c.clip_start_s)
    b = np.searchsorted(ts, t_src, side="right")
    y_now = trace_at(t_src, ts, ys)
    hd["line"].set_data(np.r_[ts[a:b] - c.odor_on_s, now], np.r_[ys[a:b], y_now])
    hd["dot"].set_data([now], [y_now])
    hd["playhead"].set_xdata([now, now])
    hd["clock"].set_text(f"{now:+.1f} s".replace("-", "−"))
    _set_pill(hd["pill"], "ODOR", ODOR_C, 0 <= now < c.odor_off_s - c.odor_on_s)
    if hd["light_pill"] is not None:
        lo, lf = data["light"]
        _set_pill(hd["light_pill"], "LIGHT", LIGHT_C, lo - c.odor_on_s <= now < lf - c.odor_on_s)
    if hd["peak"] is not None and now >= hd["peak"][0]:
        hd["peak_mark"].set_data([hd["peak"][0]], [hd["peak"][1]])
    else:
        hd["peak_mark"].set_data([], [])


# =========================================================================== data
def probe_wh(video: Path) -> tuple[int, int]:
    out = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height", "-of", "csv=p=0", str(video)],
                         capture_output=True, text=True, check=True).stdout.strip().split(",")
    return int(out[0]), int(out[1])


def trial_light(c: ClipSpec):
    meta = c.trial_dir / "_trial_meta.json"
    if not meta.exists():
        return None, None
    m = json.loads(meta.read_text())
    fps = m.get("fps")
    if m.get("light_on_s") is not None and m.get("light_off_s") is not None:
        return (float(m["light_on_s"]), float(m["light_off_s"])), fps
    return None, fps


def load_trace(c: ClipSpec):
    env = pd.read_csv(c.envelope_csv)
    t, y = trace_series(env, c.trace_column, c.fps, c.odor_on_s)
    if t[-1] < c.clip_end_s:
        raise RuntimeError(f"{c.label}: trace ends at {t[-1]:.2f} s, before the clip end {c.clip_end_s} s")
    return t, y


def load_segment_data(c: ClipSpec) -> dict:
    t, y = load_trace(c)
    light, meta_fps = trial_light(c)
    if meta_fps is not None and abs(float(meta_fps) - c.fps) > 1e-3:
        raise RuntimeError(f"{c.label}: spec fps {c.fps} != trial fps {meta_fps}")
    df = pd.read_parquet(distances_path(c))
    first, last = source_frame_range(c)
    sel = (df["frame"] >= first) & (df["frame"] <= last)
    return dict(t=t, y=y, light=light, kind=trace_kind(c.trace_column), eye_xy=pm.eye_position(df),
                lab_xy=df.loc[sel, ["x_class1", "y_class1"]].to_numpy(float))


def clip_frames(c: ClipSpec, data: dict, first: int, last: int):
    """Luma crops of frames first..last: a fixed crop (default) or ECC-stabilised (STABILIZE),
    then one contrast stretch per clip and a light sharpen."""
    if STABILIZE:
        return sharpen(stabilized_frames(c, data, first, last))
    frame_wh = probe_wh(c.video)
    crop = crop_for(data["eye_xy"], data["lab_xy"], frame_wh)
    win = pm.read_window(c.video, first, last, crop, margin=0, frame_wh=frame_wh)
    if len(win) != last - first + 1:
        raise RuntimeError(f"{c.label}: decoded {len(win)} of {last - first + 1} frames from {c.video}")
    print(f"  [{c.label}] fixed crop {crop}", flush=True)
    return sharpen(stretch_contrast(win))


def stabilized_frames(c: ClipSpec, data: dict, first: int, last: int):
    """Stabilised luma crops of frames first..last (pm.stabilized_clip with the real frame size)."""
    frame_wh = probe_wh(c.video)
    crop = crop_for(data["eye_xy"], data["lab_xy"], frame_wh)
    win = pm.read_window(c.video, first, last, crop, frame_wh=frame_wh)
    if len(win) != last - first + 1:
        raise RuntimeError(f"{c.label}: decoded {len(win)} of {last - first + 1} frames from {c.video}")
    origin = (crop[0] - pm.STAB_MARGIN, crop[1] - pm.STAB_MARGIN)
    mask = pm.stab_mask(win.shape[1:], origin, data["eye_xy"], data["lab_xy"])
    d = pm.estimate_shifts(win, mask)
    rms, step = pm.jitter_stats(d)
    print(f"  [{c.label}] crop {crop}, shake removed: rms {rms:.2f} px, step {step:.2f} px", flush=True)
    return stretch_contrast(pm.stabilize(win, d))


# =========================================================================== encoding
def _encoder(out_path: Path):
    return subprocess.Popen(
        [FFMPEG, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}",
         "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264", "-preset", "slow", "-crf", "16",
         "-pix_fmt", "yuv420p", str(out_path)],
        stdin=subprocess.PIPE)


def render_segment(job) -> list[Path]:
    """job = (clip, idx, n, ylim, tag, out, speed, preview): preview None encodes the segment to
    `out` (mp4); else preview = [(name, on-screen s)] and PNG stills land in directory `out`."""
    c, idx, n, ylim, tag, out, speed, preview = job
    data = load_segment_data(c)
    first, last = source_frame_range(c)
    frames = clip_frames(c, data, first, last)
    fig, hd = segment_figure(c, idx, n, ylim, data, tag, crop_wh=frames.shape[1:][::-1], speed=speed)
    assert_text_floor(fig)
    els = header_els(idx, n)
    n_out = n_output_frames(c, speed)
    dur = n_out / FPS
    written = []

    def frame(k):
        t_src = source_time(k, c, speed)
        i0, i1, w = blend_index(t_src, c.fps, first)
        hd["im"].set_data(frame_for(frames, i0, i1, w))
        update_frame(hd, c, data, t_src)
        fig.canvas.draw()
        return style.compose(np.asarray(fig.canvas.buffer_rgba()), k / FPS, dur, els)

    if preview is not None:
        for name, s in preview:
            k = int(np.clip(round(s * FPS), 0, n_out - 1))
            p = Path(out) / name
            Image.fromarray(frame(k)[..., :3]).save(p)
            written.append(p)
    else:
        enc = _encoder(out)
        for k in range(n_out):
            enc.stdin.write(frame(k).tobytes())
        enc.stdin.close()
        if enc.wait():
            raise RuntimeError(f"ffmpeg failed on segment {c.label}")
        written.append(Path(out))
    plt.close(fig)
    return written


# =========================================================================== title card
TITLE_X = 160
TITLE_CHIP = (132, 104, 52)          # chip w, h, number px


def title_els() -> list:
    sp = style.HERO_SPACING
    d = style.REVEAL_DUR
    kicker = style.runs_image([GENOTYPE_RUNS], 40, TEXT_2, 52)
    head = style.runs_image([[(TITLE_HEADLINE, False)]], 104, TEXT_1, 120, bold=True)
    body = style.runs_image([[(TITLE_BODY, False)]], 44, TEXT_2, 58)
    cw, ch, cs = TITLE_CHIP
    stack_h = 6 + 40 + 52 + 18 + 120 + 24 + 58 + 64 + ch
    y = int((H - stack_h) / 2)
    els = [style.BarEl(TITLE_X, y, 72, 6, style.ACCENT, 0.30 * sp, style.RULE_DUR)]
    y += 6 + 40
    els.append(style.tag(style.TextEl(kicker, TITLE_X, y, 0.45 * sp, d), GENOTYPE))
    y += 52 + 18
    els.append(style.tag(style.TextEl(head, TITLE_X - 5, y, 0.70 * sp, d, rise=22), TITLE_HEADLINE, heading=True))
    y += 120 + 24
    els.append(style.tag(style.TextEl(body, TITLE_X, y, 1.00 * sp, d), TITLE_BODY))
    y += 58 + 64
    for j, s in enumerate(range(SCORE_MIN, SCORE_MAX + 1)):
        els.append(style.TextEl(chip_image(s, cw, ch, cs), TITLE_X + j * (cw + 16), y, (1.35 + 0.12 * j) * sp, d,
                                rise=14))
    return els


def title_duration(els) -> float:
    items = style.read_items(els)
    landed = max(e.t0 + e.dur for e in els)
    need = max(landed, max(it.shown for it in items)) + style.card_reading_time(items) + style.FADE_OUT_S
    return math.ceil((need + 0.2) * 10) / 10


def title_frame(els, t: float, dur: float) -> np.ndarray:
    fr = Image.new("RGBA", (W, H), style.rgb(SURFACE) + (255,))
    for el in els:
        el.draw(fr, t)
    return np.asarray(style.fade_to_surface(fr, style.segment_fade(t, dur, 0.0, style.FADE_OUT_S)))


def render_title(job) -> list[Path]:
    out, preview = job
    els = title_els()
    dur = title_duration(els)
    if preview is not None:
        paths = []
        for name, t in preview:
            p = Path(out) / name
            Image.fromarray(title_frame(els, t, dur)[..., :3]).save(p)
            paths.append(p)
        return paths
    enc = _encoder(out)
    for k in range(int(round(dur * FPS))):
        enc.stdin.write(title_frame(els, k / FPS, dur).tobytes())
    enc.stdin.close()
    if enc.wait():
        raise RuntimeError("ffmpeg failed on the title card")
    return [Path(out)]


# =========================================================================== closing card
CLOSING_S = 8.0
TILE_GAP = 24
CLOSE_TOP, CLOSE_BOTTOM = 136, H - 56


def tile_grid(n: int) -> list[tuple[int, int, int, int]]:
    """Tile boxes (x, y, w, h) for n clips: up to 4 per row, rows balanced (5 -> 3 + 2), crop aspect,
    centred."""
    rows = 1 if n <= 4 else math.ceil(n / 4)
    cols = math.ceil(n / rows)
    aspect = CROP_WH[0] / CROP_WH[1]
    w = (W - 2 * G - (cols - 1) * TILE_GAP) / cols
    h = w / aspect
    h_max = (CLOSE_BOTTOM - CLOSE_TOP - (rows - 1) * TILE_GAP) / rows
    if h > h_max:
        h, w = h_max, h_max * aspect
    w, h = int(w), int(h)
    total_h = rows * h + (rows - 1) * TILE_GAP
    y0 = CLOSE_TOP + (CLOSE_BOTTOM - CLOSE_TOP - total_h) // 2
    boxes = []
    for i in range(n):
        r, col = divmod(i, cols)
        in_row = min(cols, n - r * cols)
        x0 = (W - (in_row * w + (in_row - 1) * TILE_GAP)) // 2
        boxes.append((x0 + col * (w + TILE_GAP), y0 + r * (h + TILE_GAP), w, h))
    return boxes


def closing_window(c: ClipSpec, data: dict, speed: float) -> tuple[float, float]:
    """Trial window a tile plays: CLOSING_S on screen at `speed`, centred on the clip's peak."""
    span = CLOSING_S * speed
    k = peak_index(c, data["t"], data["y"])
    t_pk = data["t"][k] if k is not None else (c.clip_start_s + c.clip_end_s) / 2
    t0 = float(np.clip(t_pk - span / 2, c.clip_start_s, max(c.clip_start_s, c.clip_end_s - span)))
    return t0, t0 + span


def render_closing(job) -> list[Path]:
    clips, out, speed, preview = job
    boxes = tile_grid(len(clips))
    tiles = []
    for c, box in zip(clips, boxes):
        data = load_segment_data(c)
        t0, t1 = closing_window(c, data, speed)
        first, last = int(math.floor(t0 * c.fps)), int(math.ceil(t1 * c.fps))
        frames = clip_frames(c, data, first, last)
        tiles.append((c, box, frames, first, t0))
    els = header_els(None, 0, CLOSING_LABEL)
    dur = CLOSING_S
    chips = {c.score: chip_image(c.score, 76, 60, 40, radius=12) for c in clips}

    def frame(k):
        fr = Image.new("RGBA", (W, H), style.rgb(SURFACE) + (255,))
        for c, (x, y, w, h), frames, first, t0 in tiles:
            t_src = min(t0 + k * speed / FPS, t0 + CLOSING_S * speed)
            i0, i1, wgt = blend_index(t_src, c.fps, first)
            i1 = min(i1, len(frames) - 1)
            i0 = min(i0, len(frames) - 1)
            img = cv2.resize(frame_for(frames, i0, i1, wgt), (w, h), interpolation=cv2.INTER_LANCZOS4)
            fr.paste(Image.fromarray(img).convert("RGBA"), (x, y))
            ImageDraw.Draw(fr).rectangle((x - 1, y - 1, x + w, y + h), outline=style.rgb(GRID), width=1)
            fr.alpha_composite(chips[c.score], (x + 14, y + 14))
        return style.compose(np.asarray(fr), k / FPS, dur, els)

    if preview is not None:
        paths = []
        for name, t in preview:
            p = Path(out) / name
            Image.fromarray(frame(int(round(t * FPS)))[..., :3]).save(p)
            paths.append(p)
        return paths
    enc = _encoder(out)
    for k in range(int(round(dur * FPS))):
        enc.stdin.write(frame(k).tobytes())
    enc.stdin.close()
    if enc.wait():
        raise RuntimeError("ffmpeg failed on the closing card")
    return [Path(out)]


# =========================================================================== driver
def prepare(clips, speed):
    """Shared y-range (one for every segment) and the trace kind; refuses mixed trace kinds."""
    kinds = {trace_kind(c.trace_column) for c in clips}
    if len(kinds) != 1:
        raise SpecError(f"segments mix trace kinds {sorted(kinds)}: one shared y-scale needs one kind")
    kind = kinds.pop()
    wins = []
    for c in clips:
        t, y = load_trace(c)
        wins.append(window(t, y, c.clip_start_s, c.clip_end_s))
        mx = float(np.nanmax(wins[-1]))
        if kind == "pct" and c.peak_pct > 0 and abs(mx - c.peak_pct) > 0.15 * max(c.peak_pct, 10):
            print(f"  [warn] {c.label}: spec peak_pct {c.peak_pct:.1f} but the {c.trace_column} trace peaks at "
                  f"{mx:.1f} inside the clip window", flush=True)
    return shared_ylim(wins, kind), kind


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--spec", type=Path, required=True, help="clips.json (list, display order)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"output mp4 (default {DEFAULT_OUT})")
    ap.add_argument("--preview-dir", type=Path, default=None, help="write PNG stills here instead of the video")
    ap.add_argument("--speed", type=float, default=SPEED, help="footage playback rate (default 0.5)")
    ap.add_argument("--no-closing", action="store_true", help="leave out the closing side-by-side card")
    ap.add_argument("--jobs", type=int, default=6, help="parallel render processes")
    ap.add_argument("--stabilize", action="store_true",
                    help="re-align every frame to the clip's first (ECC); default is a fixed crop")
    return ap


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    global STABILIZE
    STABILIZE = bool(args.stabilize)      # read by the forked render workers
    if not 0 < args.speed <= 1:
        raise SystemExit("--speed must be in (0, 1]")
    clips = load_spec(args.spec, check_files=True, speed=args.speed)
    for w in order_warnings(clips):
        print(f"[warn] {w}", flush=True)
    ylim, kind = prepare(clips, args.speed)
    tags = example_tags(clips)
    n = len(clips)
    print(f"{n} segments, trace kind {kind}, shared y-range {ylim[0]:.1f}..{ylim[1]:.1f}", flush=True)

    if args.preview_dir:
        out = args.preview_dir
        out.mkdir(parents=True, exist_ok=True)
        jobs = []
        for i, (c, tag) in enumerate(zip(clips, tags)):
            data_t, data_y = load_trace(c)
            t_pk = data_t[peak_index(c, data_t, data_y)]
            dur = on_screen_s(c, args.speed)
            stem = f"{i + 1:02d}_score{c.label}"
            pv = [(f"{stem}_a_early.png", 2.6), (f"{stem}_b_peak.png", (t_pk - c.clip_start_s) / args.speed),
                  (f"{stem}_c_fade.png", dur - 0.35)]
            jobs.append((c, i, n, ylim, tag, out, args.speed, pv))
        with Pool(min(args.jobs, len(jobs) + 2)) as pool:
            title = pool.apply_async(render_title, ((out, [("00_title_reveal.png", 1.6), ("00_title.png", 6.0)]),))
            close = None if args.no_closing else pool.apply_async(
                render_closing, ((clips, out, args.speed, [("99_closing.png", 3.0)]),))
            paths = [p for ps in pool.map(render_segment, jobs) for p in ps]
            paths = title.get() + paths + (close.get() if close else [])
        print("previews:\n  " + "\n  ".join(str(p) for p in paths))
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        jobs = [(c, i, n, ylim, tag, tmp / f"seg{i + 1:02d}.mp4", args.speed, None)
                for i, (c, tag) in enumerate(zip(clips, tags))]
        with Pool(min(args.jobs, len(jobs) + 2)) as pool:
            title = pool.apply_async(render_title, ((tmp / "seg00.mp4", None),))
            close = None if args.no_closing else pool.apply_async(
                render_closing, ((clips, tmp / "seg99.mp4", args.speed, None),))
            parts = [p for ps in pool.map(render_segment, jobs) for p in ps]
            parts = title.get() + parts + (close.get() if close else [])
        listing = tmp / "concat.txt"
        listing.write_text("".join(f"file '{p}'\n" for p in parts))
        subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
                        "-c", "copy", "-an", "-movflags", "+faststart", str(args.out)], check=True)
    dur = float(subprocess.run([FFPROBE, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                str(args.out)], capture_output=True, text=True).stdout)
    print(f"[OK] wrote {args.out} ({dur:.1f} s)")


if __name__ == "__main__":
    main()
