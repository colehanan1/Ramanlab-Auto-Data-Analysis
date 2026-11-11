#!/usr/bin/env python3
"""Website/presentation montage of our best optogenetically trained fly.

Fly: Hex-Training-24-0.1 / august_11_batch_1_rig_2 / fly3 -- picked by scanning
every training-cohort trial: full light-evoked extension in all 6 training
trials (weakest +61 px over rest) and Hexanol-specific odor PER on testing.

Square 1080 x 1080 frame. Segments (real time, 40 fps, ~46 s total):
  0. title card: the fly's own training_4 footage drifting behind the lab scrim,
     held until every word on it can be read (0.3 s / word + 1 s for the headline)
  1. animated protocol card (six CS-US pairing glyphs light up one by one,
     paced so every line can be read)
  2. training_4  Hexanol + light   -> immediate, sustained full extension   (odor onset .. +9.7 s)
  3. testing_1   Hexanol, no light -> repeated odor-evoked PER             (odor onset .. +12.2 s)
  4. testing_3   Citral,  no light -> no response (specificity)            (odor onset .. +9.1 s)
Every clip opens at odor onset, or 1 s before it when the fly already moves within the first
2 s of odor (lab_scope.clip_start_rel; user 2026-11-02), and ends as soon as the response
(or, for Citral, the header's reading time) allows -- see SEGMENTS.
Nothing ever ends on a pause (user rule 2026-10-30, overriding the earlier 1.2 s end
holds): footage, scope stream, readout, clock and pills keep running in real time while each
segment fades out, and the title card's footage keeps playing through its fade. Every
clip's window (start_rel..end_rel) therefore covers its fade too: the last FADE_OUT_S of
real frames play under the fade, and every output frame shows a new source frame.

Visual language (tokens, numbered segment header, staged reveal, captions, hero
card, fade to the surface) is the lab research-overview montage's, via
lab_video_style.py, so this video plays in one set with it. The lab red accent
marks numerals, rules and the active step only; it never encodes data.

Video is the encoded luma plane (full-res, does not clip under the red
stimulus light). Under it runs a live oscilloscope of PER % (lab_scope.py, the
recipe of the approved Behavior Tracking clips): the pipeline's distance_percentage
(eye->labellum distance as % of the fly's own range, 3-frame median), write head pinned
at the plot's right edge, history streaming left at 165 px/s ('-N s ... now'), head dot
with halo and afterglow, orange above the PER threshold (rest median + max(10 points,
10 rest SD); dashed line), odor / light bands scrolling with the data up to the head,
and a large live 'NN %' readout beside the head. Colours: dataviz reference palette,
dark steps, validated all-pairs (light = blue, odor = aqua, labellum = orange);
every coloured element also carries a text label.

Nothing on screen identifies the fly, batch, trial or cohort: those names stay
in this file. Every fly clip is stabilised before it is cropped: the fly
vibrates on its pin in the airflow (up to ~3 px frame to frame in the
repeated-exposure clips), so each frame is re-aligned (sub-pixel ECC
translation) to the clip's own reference on an eye-anchored region of the head
and pin that excludes the labellum's path, and the tracked-labellum ring gets
the same per-frame shift so it stays on the labellum.
"""

from __future__ import annotations

import argparse
import json
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
from matplotlib import font_manager
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import to_rgb
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab_video_style as style  # noqa: E402
import lab_scope  # noqa: E402


BATCH = Path("/securedstorage/DATAsec/cole/Data-secured-New/Hex-Training-24-0.1/august_11_batch_1_rig_2")
FLY = "fly3"
OUT_PATH = Path("/home/ramanlab/Documents/cole/Results/Figures/august_11_batch_1_rig_2_fly3_PER_montage.mp4")

# Anaconda's ffmpeg 4.3 has no libx264; the system build does.
FFMPEG = "/usr/bin/ffmpeg"
FFPROBE = "/usr/bin/ffprobe"

W, H = 1080, 1080                    # square frame; every layout size derives from these
FPS = 40
FADE_S = 0.4
TITLE_S = 14.8                       # hero card: once all up, held for the whole card's reading time
PROTOCOL_S = 12.5
BLOCK_T0, BLOCK_DT, BLOCK_DUR = 0.6, 0.8, 0.7    # protocol blocks fade in one after another
PAIR_T0 = 3.0                        # first pairing glyph lights here (protocol card)
PAIR_DT = 0.85                       # one pairing every 0.85 s (was 0.45 s)
GLYPH_FADE = 0.4                     # each glyph eases on instead of snapping
TEST_HIGHLIGHT_T = 8.6               # test block highlights after the pairings finish ...
TEST_HIGHLIGHT_DUR = 1.0             # ... and then holds to the end of the card
CROP = (400, 500, 360, 320)          # x0, y0, w, h in video pixels
FRAME_WH = (1080, 1080)
SRC_FPS = 40.02                      # the recordings' frame rate (_trial_meta.json "fps"); checked at render

SURFACE, TEXT_1, TEXT_2, MUTED = style.SURFACE, style.TEXT_1, style.TEXT_2, style.MUTED
GRID = "#383835"
LIGHT_C = "#3987e5"
ODOR_C = "#199e70"
LABELLUM_C = "#d95926"
RAISED = "#242422"                   # slightly lifted surface for the highlighted block

HEADER_TITLE = "Optogenetic Conditioning of PER (Proboscis Extension Reflex)"
PROTOCOL_HEADING = "Experimental Protocol"
PROTOCOL_BODY = "Flies food-deprived 24 h before training"
PROTOCOL_FOOTNOTE = "CS, conditioned stimulus; US, unconditioned stimulus."
Y_LABEL = "Labellum Displacement (px)"
X_LABEL = "Time from Odor Onset (s)"     # not drawn here (unit-suffixed ticks); the IAA montage uses it


@dataclass(frozen=True)
class Segment:
    """A fly clip.  The window is set in seconds from odor onset (start_rel..end_rel, fade-out
    included); odor_on_frame is the trial's recorded odor-onset frame (_trial_meta.json, checked
    at render), so the clip's frames and its absolute start_s / end_s follow without the data."""
    trial: str
    start_rel: float
    end_rel: float
    step: str
    title: str
    subtitle: str
    odor_on_frame: int

    @property
    def first_frame(self) -> int:
        return self.odor_on_frame + int(round(self.start_rel * SRC_FPS))

    @property
    def last_frame(self) -> int:
        return self.odor_on_frame + int(round(self.end_rel * SRC_FPS))

    @property
    def start_s(self) -> float:
        return self.first_frame / SRC_FPS

    @property
    def end_s(self) -> float:
        return self.last_frame / SRC_FPS


# Clip windows (user 2026-11-02): no clip opens on a long still stretch.  Start rule
# (lab_scope.clip_start_rel): odor onset, or 1 s before it when the fly moves within the first
# 2 s of odor -- none of these three does, so every clip opens at odor onset.  Ends are tight:
#   training_4  PER at light onset (+5.07 s), sustained; the clip's length is set by the header's
#               reading time (heading + description read as one block: 9.66 s), fade from +8.9 s
#   testing_1   four bouts that grow (+4.1 s ~30 %, +10.6 s ~49 %, +14.6 s ~50 %, then the full
#               extension at +19-21.5 s, peak 95 % at +20.2 s); every clip must show its trial's
#               maximal extension (user 2026-11-02), so the fade starts +21.75 s, as the proboscis
#               retracts after the full extension
#   testing_3   no response; the shortest window whose header can be read (9.06 s), well past the
#               CS+ latency (+4.1 s)
# Every window covers its fade-out: real frames keep playing under it, nothing is held.
SEGMENTS = [
    Segment("training_4", 0.0, 9.7, "Training",
            "Training: Hexanol Paired with Light",
            "Proboscis extension is elicited at light onset and sustained for the duration of illumination.",
            odor_on_frame=1203),
    Segment("testing_1", 0.0, 22.6, "Test: CS\u207a",
            "Test: Hexanol Alone (CS\u207a, No Light)",
            "Following training, the conditioned odor alone elicits repeated proboscis extension.",
            odor_on_frame=1204),
    Segment("testing_3", 0.0, 9.1, "Test: Untrained Odor",
            "Test: Citral (Untrained Odor)",
            "No response to an untrained odor, indicating that the conditioned response is odor-specific.",
            odor_on_frame=1203),
]

# The experimental-protocol slide is left out (user 2026-10-30); set True to restore it.
# Its 24 h food-deprivation note moves to the title card's kicker instead.
INCLUDE_PROTOCOL = False

# Lab header numbering: every segment after the title card ("01 / 03" ... "03 / 03").
PROTOCOL_INDEX = 1 if INCLUDE_PROTOCOL else 0
N_NUMBERED = INCLUDE_PROTOCOL + len(SEGMENTS)

# Title card background: the fly's own training_4 footage across light onset (35.06 s),
# a wider square crop than the clip (fly in the right third, burned-in label excluded).
TITLE_FOOTAGE_TRIAL = "training_4"
TITLE_FOOTAGE_S = (28.0, 42.8)
TITLE_CROP = (100, 300, 700, 700)    # x0, y0, w, h in video pixels
TITLE_KB = ((1.04, 0.50, 0.50), (1.12, 0.47, 0.50))   # slow drift: 0.009 zoom / s


def segment_index(idx: int) -> int:
    """Header numeral of fly clip `idx` (0-based): the protocol card is 01."""
    return PROTOCOL_INDEX + 1 + idx


# --------------------------------------------------------------------------- data

def trial_dir(trial: str) -> Path:
    return BATCH / f"{BATCH.name.rsplit('_rig', 1)[0]}_{trial}"


def find_video(trial: str) -> Path:
    stem = BATCH.name.rsplit("_rig", 1)[0]
    hits = sorted(BATCH.glob(f"output_{stem}_{trial}_*.mp4"))
    if len(hits) != 1:
        raise RuntimeError(f"Expected one video for {trial}, found {hits}")
    return hits[0]


def extension_trace(dist: np.ndarray, t: np.ndarray, odor_on_s: float, pre_s: float = 10.0) -> np.ndarray:
    """Eye->labellum distance minus the pre-odor median (0 = rest), 3-frame median-smoothed."""
    pre = dist[(t >= odor_on_s - pre_s) & (t < odor_on_s)]
    base = np.nanmedian(pre)
    ext = pd.Series(dist - base).rolling(3, center=True, min_periods=1).median().to_numpy()
    return ext


def load_trial(trial: str) -> dict:
    tdir = trial_dir(trial)
    meta = json.loads((tdir / "_trial_meta.json").read_text())
    df = pd.read_parquet(tdir / f"{tdir.name}_{FLY}_distances.parquet")
    fps = float(meta["fps"])
    t = df["frame"].to_numpy(dtype=float) / fps
    odor_on, odor_off = meta["odor_on_frame"] / fps, meta["odor_off_frame"] / fps
    return dict(
        meta=meta, fps=fps, frame=df["frame"].to_numpy(int), t=t,
        ext=extension_trace(df["distance_0_1"].to_numpy(float), t, odor_on),
        pct=lab_scope.per_pct(df[lab_scope.PCT_COLUMN]),            # PER %: the live scope's trace
        lab_xy=df[["x_class1", "y_class1"]].to_numpy(float),
        eye_xy=eye_position(df),
        odor=(odor_on, odor_off),
        light=(meta["light_on_s"], meta["light_off_s"]) if meta.get("light_on_s") else None,
        odor_name=meta["odor"],
    )


def read_clip(video: Path, first: int, last: int, crop=CROP, frame_wh=FRAME_WH):
    """Yield luma crops for frames first..last inclusive."""
    x0, y0, cw, ch = crop
    cmd = [FFMPEG, "-v", "error", "-i", str(video), "-vf",
           f"select=between(n\\,{first}\\,{last}),extractplanes=y,crop={cw}:{ch}:{x0}:{y0}",
           "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    n = cw * ch
    try:
        while True:
            buf = proc.stdout.read(n)
            if len(buf) < n:
                break
            yield np.frombuffer(buf, np.uint8).reshape(ch, cw)
    finally:
        proc.stdout.close()
        proc.wait()


def eye_position(df: pd.DataFrame) -> tuple[float, float]:
    """The fly's (anchored) eye position in raw video pixels: the stabilisation anchor."""
    return float(np.nanmedian(df["x_class0"])), float(np.nanmedian(df["y_class0"]))


# ------------------------------------------------------------------ stabilisation
# The flies vibrate on their pins in the airflow while the camera stays put (the
# arena wall moves < 0.2 px), so the shake is removed per fly: every frame is
# translated (sub-pixel) back onto the clip's own reference, measured on the
# fly's head / pin around the anchored eye -- never on the labellum, which is
# what the viewer should see move.
STAB_MARGIN = 14                     # px read around a crop so a re-aligned crop never shows an edge
STAB_ROI = (140, 120, 30)            # eye-anchored region: half width, rows above, rows below the eye
STAB_LAB_PAD = 30                    # px kept clear around the labellum's whole path
STAB_BLUR = 1.5                      # Gaussian sigma before matching (sensor noise)


def read_window(video: Path, first: int, last: int, crop, margin: int = STAB_MARGIN,
                frame_wh=FRAME_WH) -> np.ndarray:
    """Luma frames of `crop` grown by `margin` px on every side (n, ch + 2m, cw + 2m);
    rows / columns beyond the video frame are mirrored in."""
    x0, y0, cw, ch = crop
    wx0, wy0 = max(0, x0 - margin), max(0, y0 - margin)
    wx1, wy1 = min(frame_wh[0], x0 + cw + margin), min(frame_wh[1], y0 + ch + margin)
    frames = np.stack(list(read_clip(video, first, last, crop=(wx0, wy0, wx1 - wx0, wy1 - wy0))))
    pad = ((0, 0), (wy0 - (y0 - margin), y0 + ch + margin - wy1), (wx0 - (x0 - margin), x0 + cw + margin - wx1))
    return np.pad(frames, pad, mode="reflect") if any(p != (0, 0) for p in pad) else frames


def stab_mask(shape, origin, eye_xy, lab_xy, roi=STAB_ROI, lab_pad=STAB_LAB_PAD,
              border: int = STAB_MARGIN) -> np.ndarray:
    """Boolean matching region in window pixels (window top-left = `origin` in video pixels):
    the box around the eye, minus the labellum's path (+ lab_pad) and the window border."""
    h, w = shape
    ox, oy = origin
    ex, ey = eye_xy[0] - ox, eye_xy[1] - oy
    half, up, down = roi
    m = np.zeros((h, w), bool)
    m[max(0, int(ey - up)):max(0, int(ey + down)), max(0, int(ex - half)):max(0, int(ex + half))] = True
    lab = np.asarray(lab_xy, float).reshape(-1, 2)
    lab = lab[np.isfinite(lab).all(1)]
    if len(lab):
        lx0, ly0 = lab.min(0) - (ox, oy) - lab_pad
        lx1, ly1 = lab.max(0) - (ox, oy) + lab_pad
        m[max(0, int(ly0)):max(0, int(np.ceil(ly1))), max(0, int(lx0)):max(0, int(np.ceil(lx1)))] = False
    if border:
        m[:border] = False; m[-border:] = False; m[:, :border] = False; m[:, -border:] = False
    if m.sum() < 1500:
        raise ValueError(f"stabilisation region too small ({m.sum()} px)")
    return m


def align(img: np.ndarray, d) -> np.ndarray:
    """Sample `img` at x + d (sub-pixel, cubic): content displaced by +d returns to its reference place."""
    M = np.float32([[1, 0, d[0]], [0, 1, d[1]]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_REFLECT101)


def _ecc_track(frames, ref, mask8, limit: float) -> np.ndarray:
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)
    out = np.zeros((len(frames), 2))
    warp = np.eye(2, 3, dtype=np.float32)
    for i, f in enumerate(frames):
        try:
            _, w = cv2.findTransformECC(ref, f, warp.copy(), cv2.MOTION_TRANSLATION, crit, mask8, 1)
            if np.abs(w[:, 2]).max() <= limit:
                warp = w                   # else keep the previous frame's estimate
        except cv2.error:
            pass
        out[i] = warp[0, 2], warp[1, 2]
    return out


def estimate_shifts(frames: np.ndarray, mask: np.ndarray, blur: float = STAB_BLUR, passes: int = 2,
                    limit: float = STAB_MARGIN) -> np.ndarray:
    """Per-frame translation d (n, 2) of the masked scene, frame(x) ~ ref(x - d), against the
    clip's own reference (temporal median, then the mean of the re-aligned frames). ECC is
    sub-pixel and invariant to brightness changes (the red stimulus light)."""
    f32 = [cv2.GaussianBlur(f.astype(np.float32), (0, 0), blur) for f in frames]
    ref = np.median(np.stack(f32), axis=0).astype(np.float32)
    mask8 = mask.astype(np.uint8)
    d = np.zeros((len(frames), 2))
    for p in range(passes):
        d = _ecc_track(f32, ref, mask8, limit)
        if p < passes - 1:
            ref = np.mean([align(f, dd) for f, dd in zip(f32, d)], axis=0).astype(np.float32)
    return d - np.median(d, axis=0)        # lock to the clip's typical position


def stabilize(window: np.ndarray, shifts: np.ndarray, margin: int = STAB_MARGIN) -> np.ndarray:
    """Re-align every window frame by its shift, then cut the margin off: (n, ch, cw), same dtype."""
    sl = (slice(margin, -margin or None), slice(margin, -margin or None))
    return np.stack([align(f, d)[sl] for f, d in zip(window, shifts)])


def stabilized_ring(lab_xy, crop_origin, shift) -> tuple[float, float]:
    """Tracked labellum (raw video px) in stabilised-crop px: the frame's shift applies to it too."""
    return lab_xy[0] - crop_origin[0] - shift[0], lab_xy[1] - crop_origin[1] - shift[1]


def jitter_stats(d: np.ndarray) -> tuple[float, float]:
    """(RMS displacement about the mean, RMS frame-to-frame step) of a shift track, px."""
    d = np.asarray(d, float)
    ff = np.diff(d, axis=0)
    return float(np.sqrt(((d - d.mean(0)) ** 2).sum(1).mean())), float(np.sqrt((ff ** 2).sum(1).mean()))


def stabilized_clip(video: Path, first: int, last: int, crop, eye_xy, lab_xy):
    """Stabilised luma crops (n, ch, cw) of frames first..last and their shifts (n, 2).
    lab_xy: labellum positions (raw px) over the clip, kept out of the matching region."""
    win = read_window(video, first, last, crop)
    origin = (crop[0] - STAB_MARGIN, crop[1] - STAB_MARGIN)
    mask = stab_mask(win.shape[1:], origin, eye_xy, lab_xy)
    d = estimate_shifts(win, mask)
    return stabilize(win, d), d


def pairings_lit(t: float) -> int:
    """Number of CS-US pairing glyphs lit at time t on the protocol card (one every PAIR_DT)."""
    if t < PAIR_T0:
        return 0
    return int(min(6, 1 + (t - PAIR_T0) // PAIR_DT))


def glyph_level(j: int, t: float) -> float:
    """0..1 fill of pairing glyph j: eases on over GLYPH_FADE once its turn comes."""
    return style.smoothstep((t - (PAIR_T0 + j * PAIR_DT)) / GLYPH_FADE)


def fade_alpha(i: int, n: int, fade_frames: int) -> float:
    """1 in the middle of a clip, ramping linearly from/to 0 over fade_frames at each end."""
    if fade_frames <= 0:
        return 1.0
    return float(min(1.0, (i + 1) / fade_frames, (n - i) / fade_frames))


def clip_frames(idx: int) -> tuple[int, int]:
    """(data frames, encoded frames) of fly clip `idx`: one encoded frame per recorded frame
    start_s..end_s, fade-out included -- the data keeps running under the fade, nothing is held,
    so the two counts are equal."""
    seg = SEGMENTS[idx]
    n_data = seg.last_frame - seg.first_frame + 1
    return n_data, n_data


def source_frame(i: int, n_data: int) -> int:
    """Data frame shown at encoded frame i: real time to the last frame, fade included (no hold)."""
    if not 0 <= i < n_data:
        raise IndexError(f"encoded frame {i} outside the clip's {n_data} data frames")
    return i


def head_rel(idx: int, i: int) -> float:
    """The scope's write head at encoded frame i of fly clip idx: the shown frame's time from
    odor onset (s).  It advances on every frame, fade-out included."""
    seg = SEGMENTS[idx]
    j = source_frame(i, clip_frames(idx)[0])
    return (seg.first_frame + j - seg.odor_on_frame) / SRC_FPS


# ------------------------------------------------------------------------ layout
# All positions below are pixels from the TOP-LEFT of the W x H frame; fx/fy and
# px_axes convert them to matplotlib figure fractions.  Gutters derive from W.

G = W // 18                          # 60 px outer gutter
CONTENT_W = W - 2 * G
SEP = "  ·  "                   # phrase separator used in the metadata lines


def fx(px: float) -> float:
    return px / W


def fy(px: float) -> float:
    """Figure-fraction y of a pixel row measured from the top."""
    return 1.0 - px / H


def px_axes(fig, x: float, y_top: float, w: float, h: float, **kw):
    return fig.add_axes([x / W, 1.0 - (y_top + h) / H, w / W, h / H], **kw)


def fit_aspect(cw: int, ch: int, max_w: float, max_h: float) -> tuple[int, int]:
    """Largest (w, h) with the crop's aspect ratio that fits inside max_w x max_h."""
    s = min(max_w / cw, max_h / ch)
    return int(round(cw * s)), int(round(ch * s))


def wrap_words(text: str, max_w: float, measure, balance: bool = False) -> list[str]:
    """Greedy word wrap by measured width (measure(text) -> px).  balance=True keeps the
    greedy line count but evens the line lengths (for headlines)."""
    words = text.split()

    def greedy(limit):
        lines, cur = [], ""
        for w in words:
            cand = f"{cur} {w}" if cur else w
            if cur and measure(cand) > limit:
                lines.append(cur)
                cur = w
            else:
                cur = cand
        return lines + ([cur] if cur else [])

    lines = greedy(max_w)
    if balance and len(lines) > 1:
        lo, hi = max(measure(w) for w in words), max_w
        for _ in range(24):
            mid = (lo + hi) / 2
            if len(greedy(mid)) <= len(lines):
                hi = mid
            else:
                lo = mid
        lines = greedy(hi)
    return lines


def wrap_runs(runs, max_w: float, measure, sep: str = SEP, balance: bool = False):
    """Wrap styled (text, italic) runs into lines, breaking only at `sep`.
    measure(text, italic) -> px.  Returns a list of lines, each a list of runs.
    balance=True keeps the greedy line count but evens the line widths."""
    phrases = [[]]
    for text, italic in runs:
        for j, part in enumerate(text.split(sep)):
            if j > 0:
                phrases.append([])
            if part:
                phrases[-1].append((part, italic))
    phrases = [ph for ph in phrases if ph]
    sep_w = measure(sep, False)
    widths = [sum(measure(t, it) for t, it in ph) for ph in phrases]

    def greedy(limit):
        lines, cur, cur_w = [], [], 0.0
        for ph, w in zip(phrases, widths):
            if cur and cur_w + sep_w + w > limit:
                lines.append(cur)
                cur, cur_w = [], 0.0
            if cur:
                cur.append((sep, False))
                cur_w += sep_w
            cur.extend(ph)
            cur_w += w
        return lines + ([cur] if cur else [])

    lines = greedy(max_w)
    if balance and len(lines) > 1:
        lo, hi = max(widths), max_w
        for _ in range(24):
            mid = (lo + hi) / 2
            if len(greedy(mid)) <= len(lines):
                hi = mid
            else:
                lo = mid
        lines = greedy(hi)
    return lines


# ------------------------------------------------------------------------ drawing

HALO = [pe.withStroke(linewidth=3, foreground=SURFACE)]


def _setup_fonts() -> None:
    for p in ("/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",
              "/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf",
              "/usr/share/fonts/truetype/msttcorefonts/Arial_Italic.ttf"):
        if Path(p).exists():
            font_manager.fontManager.addfont(p)
    plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "font.sans-serif": ["Arial", "DejaVu Sans"]})


def _new_fig():
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=SURFACE)
    FigureCanvasAgg(fig)
    return fig


def line_px(fontsize: float, spacing: float = 1.2) -> float:
    """Baseline-to-baseline distance in px for a font size in points (dpi 100)."""
    return fontsize * 100 / 72 * spacing


def measurer(fig, fontsize: float, **kw):
    """measure(text, italic=False) -> rendered width in px for this font."""
    renderer = fig.canvas.get_renderer()

    def measure(text, italic=False):
        t = fig.text(0, 0, text, fontsize=fontsize, style="italic" if italic else "normal", **kw)
        w = t.get_window_extent(renderer).width
        t.remove()
        return w
    return measure


def text_block(fig, text: str, x: float, y_top: float, *, fontsize: float, max_w: float = None,
               spacing: float = 1.35, ha: str = "left", balance: bool = False, **kw):
    """Word-wrapped text whose first line's top sits at y_top (px).  Paragraphs split on
    '\\n' are wrapped separately.  Returns (artist, height_px)."""
    m = measurer(fig, fontsize, **{k: v for k, v in kw.items() if k in ("fontweight",)})
    lines = []
    for para in text.split("\n"):
        lines += wrap_words(para, max_w or CONTENT_W, m, balance=balance)
    art = fig.text(fx(x), fy(y_top), "\n".join(lines), fontsize=fontsize, ha=ha, va="top",
                   linespacing=spacing, multialignment=ha, **kw)
    return art, len(lines) * line_px(fontsize, spacing)


def draw_runs(fig, runs, x: float, y: float, *, fontsize: float, color: str, max_w: float = None,
              spacing: float = 1.45, ha: str = "left", balance: bool = True):
    """Styled runs (italic species/genotype) wrapped at separators; line centres from y (px).
    Returns (artists, n_lines)."""
    m = measurer(fig, fontsize)
    lines = wrap_runs(runs, max_w or CONTENT_W, m, balance=balance)
    arts = []
    for li, line in enumerate(lines):
        widths = [m(t, it) for t, it in line]
        cx = x - (sum(widths) / 2 if ha == "center" else 0.0)
        for (t, it), w in zip(line, widths):
            arts.append(fig.text(fx(cx), fy(y + li * line_px(fontsize, spacing)), t, fontsize=fontsize,
                                 color=color, va="center", style="italic" if it else "normal"))
            cx += w
    return arts, len(lines)


def legend_row(fig, entries, x: float, y: float, *, fontsize: float = style.MIN_TEXT_PT, color: str = TEXT_2,
               gap: float = 36, ha: str = "left"):
    """Direct legend: entries = [(kind, colour, label)], kind in {'line', 'dot', 'ring', None};
    None is a plain note set in `colour` with no swatch."""
    m = measurer(fig, fontsize)
    sw = {"line": 26, "dot": 14, "ring": 16, None: -10}
    widths = [sw[k] + 10 + m(label) for k, _, label in entries]
    cx = x - ((sum(widths) + gap * (len(entries) - 1)) / 2 if ha == "center" else 0.0)
    arts = []
    for (kind, c, label), w in zip(entries, widths):
        if kind is None:
            pass
        elif kind == "line":
            arts.append(fig.add_artist(plt.Line2D([fx(cx), fx(cx + sw[kind])], [fy(y), fy(y)],
                                                  transform=fig.transFigure, color=c, linewidth=5,
                                                  solid_capstyle="butt")))
        else:
            arts.append(fig.add_artist(plt.Line2D([fx(cx + sw[kind] / 2)], [fy(y)], transform=fig.transFigure,
                                                  marker="o", ms=sw[kind] * 0.72 if kind == "dot" else 11,
                                                  color=c, mfc=c if kind == "dot" else "none", mec=c,
                                                  mew=2, linestyle="none")))
        arts.append(fig.text(fx(cx + sw[kind] + 10), fy(y), label, fontsize=fontsize,
                             color=c if kind is None else color, va="center"))
        cx += w + gap
    return arts


def _rule(fig, y: float, x0: float = None, x1: float = None, color: str = GRID, lw: float = 1.0):
    x0 = G if x0 is None else x0
    x1 = W - G if x1 is None else x1
    return fig.add_artist(plt.Line2D([fx(x0), fx(x1)], [fy(y), fy(y)], transform=fig.transFigure,
                                     color=color, linewidth=lw))


PILL_PAD = 0.5                       # pill padding, in units of its font size


def pill_pad_px(fontsize: float = style.MIN_TEXT_PT) -> float:
    """Pill padding in px: the box reaches this far past the label on every side."""
    return PILL_PAD * fontsize * 100 / 72


def _pill(fig, x, y, label, color, active, fontsize=style.MIN_TEXT_PT, ha="left"):
    """Stimulus-state pill; x, y in px (left/right edge of the label, vertical centre)."""
    txt = fig.text(fx(x), fy(y), "", ha=ha, va="center", fontsize=fontsize, fontweight="bold",
                   bbox=dict(boxstyle=f"round,pad={PILL_PAD},rounding_size=0.95", linewidth=1.6))
    _set_pill(txt, label, color, active)
    return txt


def _set_pill(txt, label, color, active):
    txt.set_text(f"{label} {'ON' if active else 'OFF'}")
    txt.set_color(SURFACE if active else MUTED)
    patch = txt.get_bbox_patch()
    patch.set_edgecolor(color if active else GRID)
    patch.set_facecolor(color if active else SURFACE)


# (text, italic) runs.  The genotype (italic by convention) is the only fly identity
# shown, once, on the title card; shared with the repeated-exposure montage.
GENOTYPE = "Gr5a-GAL4 > UAS-CsChrimson"
GENOTYPE_RUNS = [(GENOTYPE, True)]
STARVATION_NOTE = "  \u00b7  Food-deprived 24 h"     # on the title when the protocol card is left out

# single-fly segment layout (px from top).  The header block (numeral row, heading,
# description) is the lab montage's.  The video snaps to each clip's own header (a one-line
# heading gives the footage room instead of leaving a gap) and keeps one size, 520 x 462 (1.44x
# the 360 x 320 crop), in every clip; the status column fills x 604-1020 beside it.  Under the
# video a label lane carries the y label (left) and the threshold / stimulus-band key (right), so
# no label sits inside the plot, where the PER % streams.  The live scope (lab_scope.ScopePanel)
# reaches down to a fixed bottom ('-N s ... now' ticks, ~40 px margin), so it is taller in the
# clip with the shorter header; its write head is pinned at the plot's right edge and the large
# live readout ('NN %') sits just right of it, vertically centred on the plot.  The plot width
# (= the 4 s history window at 165 px/s) is what is left once the readout at its widest fits.
VID_GAP = 16                         # this clip's header -> video
VID_W = 520                          # footage width, px (height follows the crop's aspect)
COL_GAP = 24                         # video -> status column
BOTTOM_MARGIN = 40                   # lowest text (x tick labels) -> frame bottom
TRACE_BOTTOM = H - BOTTOM_MARGIN - 48    # trace axes bottom: ticks (5.6 px) + pad (4.9) + 32 px labels
LANE_GAP = 12                        # video bottom -> label lane (text box top)
LANE_H = 38                          # label lane: one 32 px line
LANE_AXES_GAP = 16                   # label lane -> trace axes top (clears the top y tick label)
TRACE_X = G + 64                     # room for the '100' y tick label (53 px + tick + pad) inside the gutter
SCOPE_SPEED = lab_scope.SPEED_PX_S   # 165 px/s in every clip: the window is the plot width / speed (>= 4 s)
SCOPE_YLIM = (-10.0, 110.0)          # PER % 0-100 with room for the head dot (and its halo clears the lane)
SCOPE_YTICKS = (0, 50, 100)          # sparse: 32 px labels need >= 40 px between ticks
READOUT_PX = 64                      # live PER % readout, right of the write head (2x the caption size)
READOUT_PT = READOUT_PX * 72 / 100
READOUT_GAP = 40                     # plot's right edge -> readout at its widest: clears the halo and 'now'
X_TICK_STEP = 2                      # s between x tick labels (make_score_scale_video)
X_TICK_EDGE = 44                     # px: no x tick label closer than this to the plot's (left) end
STEP_DY = 48                         # stepper pitch (32 px labels)
PILL_GAP = 12                        # ODOR and LIGHT pills share one row
TEXT_PT = style.MIN_TEXT_PT          # matplotlib text floor: 23.04 pt = 32 px, the lab caption size
BAND_ALPHA = {"odor": 0.16, "light": 0.22}
BAND_SWATCH = 22                     # px, band key swatch
LEGEND_LABEL = "Labellum (Tracked)"


def rect_box(rect) -> tuple[float, float, float, float]:
    """(x, y, w, h) -> (x0, y0, x1, y1), top-left pixels."""
    x, y, w, h = rect
    return x, y, x + w, y + h


def text_boxes(fig) -> list[tuple[str, tuple[float, float, float, float]]]:
    """(text, (x0, y0, x1, y1)) in top-left frame pixels for every visible, non-empty text the
    figure draws (figure texts, axes texts, axis labels, tick labels inside the view); a pill's
    box is its rounded patch.  Layout tests check these never overlap."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    arts = list(fig.texts)
    for ax in fig.axes:
        arts += list(ax.texts) + [ax.xaxis.label, ax.yaxis.label]
        (x0, x1), (y0, y1) = sorted(ax.get_xlim()), sorted(ax.get_ylim())
        arts += [t.label1 for t in ax.xaxis.get_major_ticks()[:len(ax.get_xticks())]
                 if x0 - 1e-9 <= t.get_loc() <= x1 + 1e-9]
        arts += [t.label1 for t in ax.yaxis.get_major_ticks()[:len(ax.get_yticks())]
                 if y0 - 1e-9 <= t.get_loc() <= y1 + 1e-9]
    out = []
    for a in arts:
        if not a.get_visible() or not a.get_text().strip():
            continue
        if a.axes is not None and not a.axes.get_visible():
            continue
        patch = a.get_bbox_patch()
        bb = (patch if patch is not None else a).get_window_extent(r)
        out.append((a.get_text(), (bb.x0, H - bb.y1, bb.x1, H - bb.y0)))
    return out


def x_ticks(x0: float, x1: float, step: float = X_TICK_STEP) -> np.ndarray:
    """Tick positions every `step` s inside [x0, x1] (explicit, so none is drawn off the axes)."""
    return np.arange(np.ceil(x0 / step) * step, x1 + 1e-9, step)


def x_tick_label(v: float) -> str:
    """Unit-suffixed tick label replacing the x-axis title: '\u22122 s', '0 s', '2 s' (no '+', so
    18 s of ticks every 2 s keep clear gaps at 32 px)."""
    v = int(round(v))
    return f"{chr(0x2212) if v < 0 else ''}{abs(v)} s"


def segment_layout(idx: int | None = None) -> dict:
    """Pixel geometry of fly clip `idx` (top-left px); None = the clip with the tallest header."""
    if idx is None:
        idx = max(range(len(SEGMENTS)), key=lambda i: clip_header(i).bottom)
    top = clip_header(idx).bottom + VID_GAP
    vid_w, vid_h = VID_W, int(round(VID_W * CROP[3] / CROP[2]))
    col_x = G + vid_w + COL_GAP
    pad = pill_pad_px()
    _setup_fonts()
    fig = _new_fig()
    bold = measurer(fig, TEXT_PT, fontweight="bold")
    pill_w = max(bold(f"{s} OFF") for s in ("ODOR", "LIGHT")) + 2 * pad
    unit_w = bold(lab_scope.UNIT)
    # the readout at its widest: '100' + unit, scaled by the event jump (1 + READOUT_GAIN)
    readout_w = (measurer(fig, READOUT_PT, fontweight="bold")("100") + 8 + unit_w) * (1 + lab_scope.READOUT_GAIN)
    plt.close(fig)
    steps_y0 = top + 20
    rule_y = steps_y0 + (len(SEGMENTS) - 1) * STEP_DY + 40
    pill_y = rule_y + 22 + pad + 18
    lane_y = top + vid_h + LANE_GAP + LANE_H / 2
    trace_y = lane_y + LANE_H / 2 + LANE_AXES_GAP
    trace_r = W - G - readout_w - READOUT_GAP
    trace_h = TRACE_BOTTOM - trace_y
    cap = 0.716 * READOUT_PX                         # Arial digit height at scale 1
    return dict(vid=(G, top, vid_w, vid_h), col_x=col_x, steps_y0=steps_y0, rule_y=rule_y,
                pill_y=pill_y, pill_x=(col_x + pad, col_x + pill_w + PILL_GAP + pad), pill_w=pill_w,
                odor_y=pill_y + 18 + pad + 34, key_y=top + vid_h - 16, lane_y=lane_y,
                trace=(TRACE_X, trace_y, trace_r - TRACE_X, trace_h),
                readout_x=W - G, readout_w=readout_w, readout_base=trace_y + trace_h / 2 + cap / 2,
                unit_w=unit_w)


def clip_header(idx: int) -> style.SegmentHeader:
    """Lab header; a heading that wraps at 56 px is balanced (no one-word last line)."""
    seg = SEGMENTS[idx]
    heading = "\n".join(style.wrap_balanced(seg.title, W - 2 * style.MARGIN,
                                             style.run_measure(style.HEAD_SIZE, True)))
    return style.SegmentHeader(segment_index(idx), N_NUMBERED, heading, seg.subtitle)


def _steps(fig, active_idx, x, y0, dy=STEP_DY):
    """Vertical stepper: a rail with the active step marked by the lab accent (a marker, not data)."""
    fig.add_artist(plt.Line2D([fx(x), fx(x)], [fy(y0 - 16), fy(y0 + (len(SEGMENTS) - 1) * dy + 16)],
                              transform=fig.transFigure, color=GRID, linewidth=2))
    for i, seg in enumerate(SEGMENTS):
        on = i == active_idx
        y = y0 + i * dy
        if on:
            fig.add_artist(plt.Line2D([fx(x), fx(x)], [fy(y - 16), fy(y + 16)], transform=fig.transFigure,
                                      color=style.ACCENT_LINK, linewidth=4, solid_capstyle="butt"))
        fig.text(fx(x + 20), fy(y), f"{i + 1}  {seg.step}", fontsize=TEXT_PT,
                 fontweight="bold" if on else "normal", color=TEXT_1 if on else MUTED, va="center")


THRESH_SWATCH = 38                   # px, dashed threshold-line swatch in the lane key: three dashes


def band_key(fig, bands, x_right: float, y: float, gap: float = 24, sw: float = BAND_SWATCH):
    """Right-aligned key of the stimulus bands: a swatch filled like the band (full-colour edge)
    and the band's name in bold, in its colour.  bands = [(kind, colour, label)]; kind 'thresh'
    is the scope's dashed threshold line instead of a band swatch."""
    m = measurer(fig, TEXT_PT, fontweight="bold")
    sws = [THRESH_SWATCH if kind == "thresh" else sw for kind, _, _ in bands]
    widths = [s + 8 + m(label) for s, (_, _, label) in zip(sws, bands)]
    x = x_right - sum(widths) - gap * (len(bands) - 1)
    arts = []
    for (kind, c, label), s, w in zip(bands, sws, widths):
        if kind == "thresh":
            arts.append(fig.add_artist(plt.Line2D([fx(x), fx(x + s)], [fy(y), fy(y)], transform=fig.transFigure,
                                                  color=c, linewidth=2.4, linestyle=(0, (2.5, 2)))))   # the plot line's 8 / 7 px dashes
        else:
            arts.append(fig.add_artist(FancyBboxPatch(
                (fx(x), fy(y + s / 2)), s / W, s / H, boxstyle="square,pad=0", transform=fig.transFigure,
                fc=_mix(SURFACE, c, BAND_ALPHA[kind]), ec=c, lw=1.6)))
        arts.append(fig.text(fx(x + s + 8), fy(y), label, color=c, fontsize=TEXT_PT, fontweight="bold",
                             va="center"))
        x += w + gap
    return arts


SCOPE_COLORS = dict(idle=TEXT_2, event=LABELLUM_C, dot_idle=TEXT_1, grid=GRID, muted=MUTED)
READOUT_COLORS = dict(text=TEXT_1, event=LABELLUM_C, muted=MUTED)


def segment_figure(idx: int, odor: tuple[float, float], light, odor_name: str,
                   thr: float = lab_scope.PCT_FLOOR):
    """Static matplotlib layer of fly clip `idx` (video panel, status column, live PER % scope)
    for a trial with odor = (on, off) and light = (on, off) | None in trial seconds and PER
    threshold `thr` (%).  Returns (fig, handles): the artists render_segment updates every frame
    (scope.update(t_rel, pct, head) and readout.update(value, thr) drive the scope)."""
    lay = segment_layout(idx)
    odor_on, odor_off = odor
    fig = _new_fig()

    # video panel (the hero) + status column to its right
    cw, ch = CROP[2], CROP[3]
    _, vid_y, vid_w, vid_h = lay["vid"]
    ax_v = px_axes(fig, G, vid_y, vid_w, vid_h)
    im = ax_v.imshow(np.zeros((ch, cw), np.uint8), cmap="gray", vmin=0, vmax=255, interpolation="lanczos")
    ax_v.set_xticks([]); ax_v.set_yticks([])
    for s in ax_v.spines.values():
        s.set_edgecolor(GRID); s.set_linewidth(1)
    (marker,) = ax_v.plot([], [], "o", ms=22, mfc="none", mec=LABELLUM_C, mew=2.4, path_effects=HALO)
    clock = ax_v.text(0.03, 0.97, "", transform=ax_v.transAxes, fontsize=TEXT_PT, color=TEXT_1,
                      va="top", fontweight="bold", path_effects=HALO)
    # status column: stepper, stimulus pills (one row), odor name; the ring's key sits at its
    # foot, bottom-aligned with the video
    col_x = lay["col_x"]
    _steps(fig, idx, col_x, lay["steps_y0"])
    _rule(fig, lay["rule_y"], col_x, W - G)
    odor_pill = _pill(fig, lay["pill_x"][0], lay["pill_y"], "ODOR", ODOR_C, False)
    light_pill = _pill(fig, lay["pill_x"][1], lay["pill_y"], "LIGHT", LIGHT_C, True)
    fig.text(fx(col_x), fy(lay["odor_y"]), f"Odor: {odor_name}", fontsize=TEXT_PT, color=TEXT_2,
             va="center")
    key_y = lay["key_y"]
    _rule(fig, key_y - 36, col_x, W - G)
    fig.add_artist(plt.Line2D([fx(col_x + 9)], [fy(key_y)], transform=fig.transFigure, marker="o", ms=13,
                              mfc="none", mec=LABELLUM_C, mew=2.4, linestyle="none"))
    fig.text(fx(col_x + 28), fy(key_y), LEGEND_LABEL, fontsize=TEXT_PT, color=TEXT_2, va="center")

    # label lane under the video: y label (left), threshold + stimulus-band key (right).  Nothing
    # is written inside the plot, where the PER % streams (a 100 % trace runs along its top).
    lane_y = lay["lane_y"]
    fig.text(fx(G), fy(lane_y), lab_scope.Y_LABEL, color=TEXT_2, fontsize=TEXT_PT, va="center")
    keys = ([("thresh", LABELLUM_C, lab_scope.THRESH_LABEL), ("odor", ODOR_C, "Odor")]
            + ([("light", LIGHT_C, "Light")] if light else []))
    band_key(fig, keys, W - G, lane_y)

    # live scope beneath the lane: write head pinned at the plot's right edge, history streaming
    # left at SCOPE_SPEED; the bands scroll with the data and stop at the head
    tx, ty, tw, th = lay["trace"]
    ax_t = px_axes(fig, tx, ty, tw, th)
    bands = [lab_scope.Band("odor", ODOR_C, BAND_ALPHA["odor"], "Odor", (0.0, odor_off - odor_on))]
    if light:
        bands.append(lab_scope.Band("light", LIGHT_C, BAND_ALPHA["light"], "Light",
                                    (light[0] - odor_on, light[1] - odor_on)))
    scope = lab_scope.ScopePanel(ax_t, tw, th, thr, SCOPE_COLORS, bands=bands, ylim=SCOPE_YLIM,
                                 yticks=SCOPE_YTICKS, speed=SCOPE_SPEED, font_pt=TEXT_PT, band_labels=False)
    # '-N s ... now': drop a tick whose label would hang off the plot's left end
    secs = [s for s in lab_scope.x_tick_seconds(tw, SCOPE_SPEED) if tw - s * SCOPE_SPEED >= X_TICK_EDGE]
    ax_t.set_xticks([tw - s * SCOPE_SPEED for s in secs])
    ax_t.set_xticklabels([lab_scope.x_tick_label(s) for s in secs])
    readout = lab_scope.Readout(fig, fx, fy, lay["readout_x"], lay["readout_base"], READOUT_PT, TEXT_PT,
                                lay["unit_w"], READOUT_COLORS)
    return fig, dict(ax_v=ax_v, ax_t=ax_t, im=im, marker=marker, clock=clock, odor_pill=odor_pill,
                     light_pill=light_pill, scope=scope, readout=readout)


def render_segment(args) -> Path:
    """args = (idx, out_path) to encode a clip, or (idx, png_path, abs_s) to save one preview PNG
    (abs_s inside the last FADE_OUT_S previews a fade frame, data still running)."""
    idx, out_path = args[0], args[1]
    preview_abs = args[2] if len(args) > 2 else None
    seg = SEGMENTS[idx]
    _setup_fonts()
    tr = load_trial(seg.trial)
    if abs(tr["fps"] - SRC_FPS) > 1e-3:
        raise RuntimeError(f"{seg.trial}: recorded at {tr['fps']} fps, layout assumes {SRC_FPS}")
    if tr["meta"]["odor_on_frame"] != seg.odor_on_frame:
        raise RuntimeError(f"{seg.trial}: odor onset at frame {tr['meta']['odor_on_frame']}, "
                           f"SEGMENTS says {seg.odor_on_frame}")
    first, last = seg.first_frame, seg.last_frame
    n_data, n_total = clip_frames(idx)
    odor_on, odor_off = tr["odor"]
    rel = tr["t"] - odor_on                                    # time from odor onset
    pct = tr["pct"]
    thr = lab_scope.per_threshold(pct, rel)
    if seg.start_rel != lab_scope.clip_start_rel(pct, rel):     # the start rule (user 2026-11-02)
        raise RuntimeError(f"{seg.trial}: start rule gives {lab_scope.clip_start_rel(pct, rel)} s, "
                           f"SEGMENTS says {seg.start_rel} s")

    hdr = clip_header(idx)
    sel = (tr["frame"] >= first) & (tr["frame"] <= last)
    frames, shifts = stabilized_clip(find_video(seg.trial), first, last, CROP, tr["eye_xy"], tr["lab_xy"][sel])
    if len(frames) != n_data:
        raise RuntimeError(f"{seg.trial}: decoded {len(frames)} of {n_data} frames")
    fig, hd = segment_figure(idx, tr["odor"], tr["light"], tr["odor_name"], thr=thr)
    im, marker, clock, scope, readout = (hd[k] for k in ("im", "marker", "clock", "scope", "readout"))
    odor_pill, light_pill = hd["odor_pill"], hd["light_pill"]

    assert_text_floor(fig)
    duration = n_total / FPS
    if preview_abs is not None:
        i = int(round((preview_abs - seg.start_s) * FPS))
        if not 0 <= i < n_total:
            raise RuntimeError(f"{seg.trial}: preview time {preview_abs} s outside the clip")
        targets = [i]
    else:
        targets = range(n_total)
    enc = None if preview_abs is not None else _encoder(out_path)
    drawn = None
    for i in targets:
        j = source_frame(i, n_data)                            # real time through the fade, never held
        if j != drawn:
            fidx = first + j
            k = min(np.searchsorted(tr["frame"], fidx), len(tr["frame"]) - 1)
            im.set_data(frames[j])
            rx, ry = stabilized_ring(tr["lab_xy"][k], CROP, shifts[j])
            ok = np.isfinite(rx) and np.isfinite(ry)
            marker.set_data([rx] if ok else [], [ry] if ok else [])
            now = head_rel(idx, i)                             # the shown frame's time from odor onset
            clock.set_text(f"{now:+.1f} s")
            readout.update(scope.update(rel, pct, now), thr)   # stream + head + live readout
            _set_pill(odor_pill, "ODOR", ODOR_C, 0 <= now < odor_off - odor_on)
            light_on = bool(tr["light"]) and tr["light"][0] - odor_on <= now < tr["light"][1] - odor_on
            _set_pill(light_pill, "LIGHT", LIGHT_C, light_on)
            fig.canvas.draw()
            content = np.asarray(fig.canvas.buffer_rgba()).copy()
            drawn = j
        rgba = style.compose(content, i / FPS, duration, hdr.els)
        if preview_abs is not None:
            plt.imsave(out_path, rgba[..., :3])
            plt.close(fig)
            return out_path
        enc.stdin.write(rgba.tobytes())
    plt.close(fig)
    enc.stdin.close()
    enc.wait()
    return out_path


def assert_text_floor(fig) -> None:
    """Every visible matplotlib text is at least the lab caption size (TEXT_PT = 32 px)."""
    from matplotlib.text import Text
    small = [(t.get_text(), t.get_fontsize()) for t in fig.findobj(Text)
             if t.get_text().strip() and t.get_visible() and t.get_fontsize() < TEXT_PT - 1e-6]
    if small:
        raise ValueError(f"text below the {TEXT_PT} pt legibility floor: {small}")


def _encoder(out_path: Path):
    return subprocess.Popen(
        [FFMPEG, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}",
         "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "16",
         "-pix_fmt", "yuv420p", str(out_path)],
        stdin=subprocess.PIPE)


# ------------------------------------------------------------------- title card
TITLE_HEADLINE = "Optogenetic Conditioning of the\nProboscis Extension Reflex"
TITLE_BODY = ("Conditioned stimulus (CS): hexanol. Unconditioned stimulus (US): red-light "
              "activation of Gr5a⁺ sugar-sensing neurons via CsChrimson.")
TITLE_KEYS = [("line", ODOR_C, "Odor (CS)"), ("line", LIGHT_C, "Light (US)"), ("line", LABELLUM_C, "Labellum")]


def title_footage() -> tuple[np.ndarray, float]:
    """Stabilised luma frames of the title-card background (TITLE_FOOTAGE_S of training_4,
    TITLE_CROP), re-aligned on the featured fly's head like the clips."""
    tr = load_trial(TITLE_FOOTAGE_TRIAL)
    fps = tr["fps"]
    first, last = (int(round(s * fps)) for s in TITLE_FOOTAGE_S)
    sel = (tr["frame"] >= first) & (tr["frame"] <= last)
    frames, _ = stabilized_clip(find_video(TITLE_FOOTAGE_TRIAL), first, last, TITLE_CROP, tr["eye_xy"],
                                tr["lab_xy"][sel])
    if not len(frames):
        raise RuntimeError("no title footage decoded")
    return frames, fps


def title_text_els() -> list:
    """The hero card's staggered text stack (no footage needed: used for reading-time checks)."""
    kicker = GENOTYPE_RUNS if INCLUDE_PROTOCOL else GENOTYPE_RUNS + [(STARVATION_NOTE, False)]
    return style.title_els([kicker], TITLE_HEADLINE.replace("\n", " "), TITLE_BODY.split("\n"),
                           TITLE_KEYS)


def build_title() -> style.FootageHero:
    """Lab hero card: the fly's own footage behind the scrim, genotype kicker, headline,
    body, muted rule and the colour key (data colours, text-labelled)."""
    frames, fps = title_footage()
    return style.FootageHero(frames, TITLE_S, title_text_els(), kb=TITLE_KB, src_fps=fps)


def render_title(out_path: Path) -> Path:
    hero = build_title()
    enc = _encoder(out_path)
    for i in range(int(round(TITLE_S * FPS))):
        enc.stdin.write(np.asarray(hero.frame(i / FPS).convert("RGBA")).tobytes())
    enc.stdin.close()
    enc.wait()
    return out_path


# ----------------------------------------------------------------- protocol card
# Vertical flow for the square frame: full-width blocks stacked top to bottom with
# down arrows between them.  (top y, height) per block, px; the first block starts
# just below the lab header (heading + one-line description, bottom 224).
BLOCK_X, BLOCK_W = G, CONTENT_W
BLOCK_YH = [(262, 262), (580, 96), (730, 140)]     # header (heading + one-line description) ends 233
BLOCK_PAD = 32
BLOCK_TITLE_PT = TEXT_PT + 3                         # block titles (bold) one step above the 32 px floor


def _ramp(t: float, t0: float, dur: float) -> float:
    return float(np.clip((t - t0) / dur, 0.0, 1.0))


def _mix(c0: str, c1: str, f: float):
    a, b = np.array(to_rgb(c0)), np.array(to_rgb(c1))
    return tuple(a + (b - a) * f)


def protocol_header() -> style.SegmentHeader:
    return style.SegmentHeader(PROTOCOL_INDEX, N_NUMBERED, PROTOCOL_HEADING, PROTOCOL_BODY)


def protocol_caption() -> style.Caption:
    return style.Caption([PROTOCOL_FOOTNOTE], None, G + 2, style.CAPTION_Y)


PROTOCOL_TEXTS = [
    ("Training: 6 × CS–US Pairings", "Hexanol 30 s · light from +5 s to +30 s · 5.5 min inter-trial interval"),
    ("Rest: ≈30 min", "No stimulation"),
    ("Test: Odor Panel without Light", "Hexanol (CS⁺) and six untrained odors"),
]


def block_alpha(k: int, t: float) -> float:
    """Fade-in of protocol block k (blocks appear one after another, BLOCK_DT apart)."""
    return style.smoothstep((t - (BLOCK_T0 + BLOCK_DT * k)) / BLOCK_DUR)


def build_protocol():
    """Animated protocol card: returns (n_frames, frame(i) -> rgba, fig).  frame(i) is the
    full composited frame (lab header + caption over the animated blocks)."""
    _setup_fonts()
    fig = _new_fig()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
    hdr, cap = protocol_header(), protocol_caption()
    texts = PROTOCOL_TEXTS
    renderer = fig.canvas.get_renderer()
    blocks = []
    for k, ((title, sub), (by, bh)) in enumerate(zip(texts, BLOCK_YH)):
        patch = FancyBboxPatch((BLOCK_X, by), BLOCK_W, bh, boxstyle="round,pad=0,rounding_size=22",
                               fc=SURFACE, ec=GRID, lw=2)
        ax.add_patch(patch)
        if k == 1:
            # compact rest block: title and detail share one line
            t1 = ax.text(BLOCK_X + BLOCK_PAD, by + bh / 2, title, fontsize=BLOCK_TITLE_PT, fontweight="bold",
                         color=TEXT_1, va="center")
            fig.canvas.draw()
            t2 = ax.text(BLOCK_X + BLOCK_PAD + t1.get_window_extent(renderer).width + 28, by + bh / 2,
                         sub, fontsize=TEXT_PT, color=TEXT_2, va="center")
        else:
            t1 = ax.text(BLOCK_X + BLOCK_PAD, by + 46, title, fontsize=BLOCK_TITLE_PT, fontweight="bold",
                         color=TEXT_1, va="center")
            sub_lines = wrap_runs([(sub, False)], BLOCK_W - 2 * BLOCK_PAD, measurer(fig, TEXT_PT), sep=" · ",
                                  balance=True)
            t2 = ax.text(BLOCK_X + BLOCK_PAD, by + 76, "\n".join("".join(t for t, _ in ln) for ln in sub_lines),
                         fontsize=TEXT_PT, color=TEXT_2, va="top", linespacing=1.3)
        blocks.append(dict(patch=patch, texts=[t1, t2], extra=[], y=by, h=bh))
    for k in range(2):
        y0 = BLOCK_YH[k][0] + BLOCK_YH[k][1] + 8
        y1 = BLOCK_YH[k + 1][0] - 8
        arr = FancyArrowPatch((W / 2, y0), (W / 2, y1), arrowstyle="-|>", mutation_scale=22, color=MUTED, lw=2.4)
        ax.add_patch(arr)
        blocks[k + 1]["arrow"] = arr

    # six pairing glyphs: aqua odor bar, blue light bar over its last 5/6
    by, bh = BLOCK_YH[0]
    gw, gh = 128, 34
    pitch = (BLOCK_W - 2 * BLOCK_PAD - gw) / 5
    gy = by + bh - 46
    glyphs = []
    for j in range(6):
        gx = BLOCK_X + BLOCK_PAD + j * pitch
        odor = plt.Rectangle((gx, gy - gh / 2), gw, gh, fc=SURFACE, ec=GRID, lw=2)
        # inset from the glyph's right edge so an unlit glyph keeps its whole outline
        light = plt.Rectangle((gx + gw / 6, gy - gh * 0.27), gw * 5 / 6 - 4, gh * 0.54, fc=SURFACE, ec="none")
        ax.add_patch(odor); ax.add_patch(light)
        glyphs.append((odor, light))

    def frame(i):
        t = i / FPS
        hl = style.smoothstep((t - TEST_HIGHLIGHT_T) / TEST_HIGHLIGHT_DUR)
        for k, b in enumerate(blocks):
            a = block_alpha(k, t)
            b["patch"].set_alpha(a)
            for x in b["texts"]:
                x.set_alpha(a)
            if "arrow" in b:
                b["arrow"].set_alpha(a)
            if k == 2:
                b["patch"].set_edgecolor(_mix(GRID, TEXT_1, hl))
                b["patch"].set_facecolor(_mix(SURFACE, RAISED, hl))
                b["patch"].set_linewidth(2 + 2.2 * hl)
        a0 = block_alpha(0, t)
        for j, (odor, light) in enumerate(glyphs):
            g = glyph_level(j, t)
            odor.set_facecolor(_mix(SURFACE, ODOR_C, g))
            odor.set_edgecolor(_mix(GRID, ODOR_C, g))
            light.set_facecolor(_mix(SURFACE, LIGHT_C, g))
            odor.set_alpha(a0); light.set_alpha(a0)
        fig.canvas.draw()
        # the blocks animate themselves in, so the content is not eased in again
        return style.compose(np.asarray(fig.canvas.buffer_rgba()), t, PROTOCOL_S, hdr.els, [cap],
                             reveal_content=False)

    assert_text_floor(fig)
    return int(round(PROTOCOL_S * FPS)), frame, fig


def render_protocol(out_path: Path) -> Path:
    n, frame, fig = build_protocol()
    enc = _encoder(out_path)
    for i in range(n):
        enc.stdin.write(frame(i).tobytes())
    enc.stdin.close()
    enc.wait()
    plt.close(fig)
    return out_path


# ------------------------------------------------------------------ reading time
def segment_durations() -> list[tuple[str, float]]:
    return ([("title", TITLE_S)] + ([("protocol", PROTOCOL_S)] if INCLUDE_PROTOCOL else [])
            + [(s.trial, clip_frames(i)[1] / FPS) for i, s in enumerate(SEGMENTS)])


def total_runtime_s() -> float:
    return sum(d for _, d in segment_durations())


def reading_schedule() -> list[tuple[str, float, float, list]]:
    """(segment, duration, fade-out, [ReadItem]) for every segment: each on-screen text
    with the time it is fully revealed, so tests can check it stays up long enough to read."""
    R = style.ReadItem
    content = style.CONTENT_T0 + style.CONTENT_DUR          # matplotlib layer fully in
    out = [("title", TITLE_S, style.HERO_FADE_OUT, style.read_items(title_text_els()))]
    if INCLUDE_PROTOCOL:
        proto = protocol_header().read_items() + [R(PROTOCOL_FOOTNOTE, content)]
        for k, (title, sub) in enumerate(PROTOCOL_TEXTS):
            shown = BLOCK_T0 + BLOCK_DT * k + BLOCK_DUR
            proto += [R(title, shown), R(sub, shown)]
        out.append(("protocol", PROTOCOL_S, style.FADE_OUT_S, proto))
    for idx, seg in enumerate(SEGMENTS):
        items = clip_header(idx).read_items() + [R(f"{i + 1} {s.step}", content) for i, s in enumerate(SEGMENTS)]
        items += [R("ODOR OFF", content), R("LIGHT OFF", content), R("Odor: Hexanol", content),
                  R(LEGEND_LABEL, content), R(lab_scope.Y_LABEL, content), R(lab_scope.THRESH_LABEL, content),
                  R("Odor", content)] + ([R("Light", content)] if idx == 0 else [])
        out.append((seg.trial, clip_frames(idx)[1] / FPS, style.FADE_OUT_S, items))
    return out


PREVIEW_REL = (6.0, 11.1, 6.0)       # seconds from odor onset; seg 1 at light-evoked PER, seg 2 at its 2nd bout


def render_preview(out_dir: Path) -> None:
    """One PNG per segment type: title (mid-reveal + settled), protocol (reveal/mid/late), three
    clips (at the response; the first also mid-reveal)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    hero = build_title()
    hero.frame(1.2).save(out_dir / "v1_title_reveal.png")
    hero.frame(6.0).save(out_dir / "v1_title.png")
    if INCLUDE_PROTOCOL:
        n, frame, fig = build_protocol()
        for label, t in (("reveal", 1.0), ("mid", PAIR_T0 + 2.6 * PAIR_DT), ("late", PROTOCOL_S - 1.5)):
            plt.imsave(out_dir / f"v1_protocol_{label}.png", frame(int(t * FPS))[..., :3])
        plt.close(fig)
    for idx, rel in enumerate(PREVIEW_REL):
        seg = SEGMENTS[idx]
        tr = load_trial(seg.trial)
        render_segment((idx, out_dir / f"v1_seg{idx + 1}.png", tr["odor"][0] + rel))
    render_segment((0, out_dir / "v1_seg1_reveal.png", SEGMENTS[0].start_s + 1.0))
    print("previews in", out_dir)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", type=Path, default=None, help="write preview PNGs here and exit")
    args = ap.parse_args()
    if args.preview:
        render_preview(args.preview)
        return
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        jobs = [(i, tmp / f"seg{i + 1}.mp4") for i in range(len(SEGMENTS))]
        with Pool(len(jobs) + 2) as pool:
            title = pool.apply_async(render_title, (tmp / "seg0.mp4",))
            proto = pool.apply_async(render_protocol, (tmp / "seg0b.mp4",)) if INCLUDE_PROTOCOL else None
            parts = pool.map(render_segment, jobs)
            parts = [title.get()] + ([proto.get()] if proto else []) + parts
        listing = tmp / "concat.txt"
        listing.write_text("".join(f"file '{p}'\n" for p in parts))
        subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
                        "-c", "copy", "-movflags", "+faststart", str(OUT_PATH)], check=True)
    dur = float(subprocess.run([FFPROBE, "-v", "error", "-show_entries", "format=duration",
                                "-of", "csv=p=0", str(OUT_PATH)], capture_output=True, text=True).stdout)
    print(f"[OK] Wrote {OUT_PATH} ({dur:.1f} s)")


if __name__ == "__main__":
    main()
