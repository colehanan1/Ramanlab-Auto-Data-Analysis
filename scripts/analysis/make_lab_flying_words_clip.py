#!/usr/bin/env python3
"""Raman Lab "flying words" title clip (1080x1080, 40 fps, H.264, no audio).

A standalone clip for the user to splice into the research-overview video:
research keywords fly out of a depth tunnel toward the camera behind a calm
foreground lockup (accent rule, "Raman Lab", full lab name) that matches the
lab intro card in make_lab_overview_montage.py. That script is not touched;
its tokens, fonts, TextEl / BarEl easing and segment fade are imported.

Motion is a Python port of the video-shotcraft `flying-words` card
(references/shots/typography/flying-words.md, demos/typography/flying-words/
FlyingWords.tsx), keeping its tuned math in the demo's 480 px design units:

  * CSS perspective 1100; each word moves linearly along z from -1750 to +800,
    i.e. it passes through the camera plane;
  * golden-angle placement a = i*2.39996 + rand(i*7+1)*0.8, radius
    82 + rand(i*13+2)*165, y flattened by 0.6; outward drift 0.5 + u*1.35;
  * life opacity [0,1,0.5,0.2,0] at [0,0.25,0.6,0.85,1];
  * near-end blur (u-0.86)*26 for u > 0.86; rotateZ (rand(i+21)-0.5)*14 deg;
  * evenly spaced phases k/N (assigned to placements by PHASE_SLOTS, see
    below), deterministic rand(), integer CYCLES -> the word layer loops
    exactly (frame 0 == frame N);
  * a breathing glow at the tunnel end, drawn over the words.

Re-skin and the user's standing preferences (they override the card's
high-energy defaults):
  * lab surface, words in TEXT_1 / TEXT_2 with a few in link red; the glow is a
    faint WashU-red tint instead of the demo's blue; the vignette only darkens
    SURFACE toward the edges;
  * one pass per word in 12 s (the demo: one every 3 s), so every keyword sits
    at >= 32 px and >= 0.8 opacity for well over 0.5 s; font sizes are raised
    (30-42 design px instead of 20-36) so that holds even for the smallest;
  * the vanishing point sits above centre; each word fades out as a whole
    (opacity -> 0) before its ink box reaches the lower-third title block, so
    no half-word ghost crosses the accent bar or the lab name;
  * each word also fades as its box nears the frame edge, so a readable word
    is never cut by the frame (the demo lets words fly off-screen);
  * words are centred on their anchor (the demo's fixed -60/-14 px margin
    approximates centring); the list keeps the card's 4-9 letter words and its
    order + PHASE_OFFSET are chosen so that no two readable words (>= 32 px,
    >= 0.5 opacity, unblurred) overlap;
  * no shake, no full-frame pulses; fade in from / out to the surface.

    python3 scripts/analysis/make_lab_flying_words_clip.py
    python3 scripts/analysis/make_lab_flying_words_clip.py --preview DIR
"""

from __future__ import annotations

import argparse
import math
import sys
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path
from typing import NamedTuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import make_lab_overview_montage as lab  # noqa: E402

# ------------------------------------------------------------------ lab tokens
W, H, FPS = lab.W, lab.H, lab.FPS
SURFACE, TEXT_1, TEXT_2, MUTED = lab.SURFACE, lab.TEXT_1, lab.TEXT_2, lab.MUTED
ACCENT, ACCENT_LINK = lab.ACCENT, lab.ACCENT_LINK
LAB_SHORT, LAB_FULL = lab.LAB_SHORT, lab.LAB_FULL
FFMPEG = lab.FFMPEG
rgb = lab.rgb

OUT_PATH = lab.OUT_DIR / "raman_lab_flying_words.mp4"

# ------------------------------------------------------------------ timing
DURATION_S = 12.0
N_FRAMES = int(round(DURATION_S * FPS))      # 480
CYCLES = 1                                   # integer -> the word layer loops exactly
BREATHS = 2                                  # glow breaths per clip (demo: sin(t*4*pi))
FADE_IN_S = 0.6
FADE_OUT_S = 0.9                             # = the intro card's fade to the surface
# Global shift of where the loop starts. Phases stay i/N; this only picks which
# moment of the loop is frame 0 so no word's readable pass is cut by the fades.
PHASE_OFFSET = 0.029                         # annealed with WORDS order and PHASE_SLOTS

# ------------------------------------------------------------------ demo motion constants (480 px design units)
DESIGN_W = 480
S = W / DESIGN_W                             # design px -> output px (2.25)
PERSPECTIVE = 1100
Z_FAR, Z_NEAR = -1750, 800
OP = (0.0, 1.0, 0.5, 0.2, 0.0)
OT = (0.0, 0.25, 0.6, 0.85, 1.0)
BLUR_FROM, BLUR_GAIN = 0.86, 26.0
FONT_BASE, FONT_RANGE = 30.0, 6.0            # demo 20 + rand*16; raised for the 32 px floor
HALO_A = 0.28                                # demo text-shadow 0 0 16px at .45 alpha, toned down
VP = (W / 2, 400.0)                          # vanishing point: above centre, clear of the title
SPREAD = 1.25                                # extra x/y spread of the tunnel (square frame)

# ------------------------------------------------------------------ legibility
LEGIBLE_PX = 32
LEGIBLE_ALPHA = 0.8
MIN_LEGIBLE_S = 0.5

# ------------------------------------------------------------------ words (lab terms, 4-9 letters as on the card)
WORDS = (   # order annealed with PHASE_SLOTS + PHASE_OFFSET: readable words never overlap (>= 8 px apart)
    "Implants", "Sensors", "Circuits", "Olfaction", "Learning", "Locust", "Biorobots",
    "Glomeruli", "Fruit Fly", "Spikes", "Neurons", "Bio-Hybrid", "E-Nose", "Memory",
    "Behavior", "Odorants", "Encoding", "Insects", "MEMS", "Antennae",
)
RED_WORDS = frozenset({"Olfaction", "Sensors", "Bio-Hybrid", "Locust"})
# Phase slot of each placement: phases stay the demo's evenly spaced k/N (same density), but
# which golden-angle placement gets which phase is re-chosen (seeded annealing, together with
# WORDS order and PHASE_OFFSET) so no two readable words ever overlap.
PHASE_SLOTS = (14, 9, 11, 8, 0, 10, 19, 3, 5, 17, 16, 1, 4, 2, 15, 18, 12, 13, 7, 6)
GLOW_COLOR = ACCENT


# ================================================================== pure helpers
def rand(seed: float) -> float:
    """Motion.tsx rand(): fract(sin(seed*127.1 + 311.7) * 43758.5453)."""
    x = math.sin(seed * 127.1 + 311.7) * 43758.5453
    return x - math.floor(x)


def lerp(u: float, a: float, b: float) -> float:
    return a + (b - a) * u


def z_at(u: float) -> float:
    return lerp(u, Z_FAR, Z_NEAR)


def drift(u: float) -> float:
    return 0.5 + u * 1.35


def near_blur(u: float) -> float:
    """Blur (design px, before projection) for the last stretch of travel."""
    return (u - BLUR_FROM) * BLUR_GAIN if u > BLUR_FROM else 0.0


def life_opacity(u: float) -> float:
    """Piecewise-linear [0,1,0.5,0.2,0] at [0,0.25,0.6,0.85,1]."""
    for k in range(4):
        if u <= OT[k + 1]:
            p = (u - OT[k]) / (OT[k + 1] - OT[k])
            return OP[k] + (OP[k + 1] - OP[k]) * p
    return 0.0


def project(x: float, y: float, z: float, p: float = PERSPECTIVE) -> tuple[float, float, float]:
    """CSS perspective: a point at depth z is scaled by p / (p - z) about the origin."""
    if z >= p:
        raise ValueError("point is at or behind the camera plane")
    k = p / (p - z)
    return x * k, y * k, k


class Item(NamedTuple):
    text: str
    font_design: float
    color: str
    x: float
    y: float
    rz: float
    ph: float


def _make_items() -> tuple[Item, ...]:
    n = len(WORDS)
    items = []
    for i, w in enumerate(WORDS):
        a = i * 2.39996 + rand(i * 7 + 1) * 0.8
        r = 82 + rand(i * 13 + 2) * 165
        if w in RED_WORDS:
            color = ACCENT_LINK          # link red holds contrast on the dark surface
        else:
            color = TEXT_1 if rand(i * 11) < 0.55 else TEXT_2
        items.append(Item(w, FONT_BASE + rand(i + 3) * FONT_RANGE, color,
                          math.cos(a) * r, math.sin(a) * r * 0.6, (rand(i + 21) - 0.5) * 14, PHASE_SLOTS[i] / n))
    return tuple(items)


ITEMS = _make_items()


def word_u(i: int, frame: int) -> float:
    """Travel progress of word i at a frame; integer arithmetic so frame N == frame 0 exactly."""
    cyc = (frame * CYCLES) % N_FRAMES / N_FRAMES
    return (cyc + ITEMS[i].ph + PHASE_OFFSET) % 1.0


def clip_fade(frame: int) -> float:
    return lab.segment_fade(frame / FPS, DURATION_S, FADE_IN_S, FADE_OUT_S)


# ------------------------------------------------------------------ per-word fades: title block + frame edge
# A whole word fades (never a row band through it): fully visible while the bottom
# of its ink box is above SCRIM_Y0, gone by SCRIM_Y1, which is 40 px above the
# accent bar (TITLE_Y = 700) so the halo never reaches the title either.
SCRIM_Y0, SCRIM_Y1 = 560.0, 660.0
EDGE_MARGIN, EDGE_FADE = 18.0, 80.0          # px: gone EDGE_MARGIN from the edge, full EDGE_MARGIN+EDGE_FADE in


def word_attenuation(x: float, y: float) -> float:
    """Title-block fade: y is the bottom of a word's ink box. 1 above SCRIM_Y0, 0 from SCRIM_Y1 down."""
    return 1.0 - lab.smoothstep((y - SCRIM_Y0) / (SCRIM_Y1 - SCRIM_Y0))


def edge_attenuation(bbox: tuple[float, float, float, float]) -> float:
    """Frame-edge fade: 0 once the ink box is within EDGE_MARGIN of any edge."""
    l, t, r, b = bbox
    d = min(l, t, W - r, H - b)
    return lab.smoothstep((d - EDGE_MARGIN) / EDGE_FADE)


@lru_cache(maxsize=None)
def _ref_metrics(i: int) -> tuple[float, float, float, float]:
    """Ink box of word i at 100 px relative to its 'mm' anchor, in units of font px."""
    f = lab.font(100, True)
    l, t, r, b = f.getbbox(ITEMS[i].text, anchor="mm")
    return l / 100, t / 100, r / 100, b / 100


class WordState(NamedTuple):
    u: float
    z: float
    cx: float
    cy: float
    px: float           # on-screen font size
    alpha: float        # life opacity
    alpha_eff: float    # life x scrim x clip fade
    blur_px: float      # on-screen blur sigma
    rz: float
    bbox: tuple[float, float, float, float]
    atten: float = 1.0  # title-block x frame-edge fade (part of alpha_eff, applied when drawing)
    quad: tuple = ()    # rotated ink box corners (tl, tr, br, bl) in frame px


def word_state(i: int, frame: int) -> WordState:
    it = ITEMS[i]
    u = word_u(i, frame)
    z = z_at(u)
    d = drift(u)
    sx, sy, k = project(it.x * d, it.y * d, z)
    cx, cy = VP[0] + S * SPREAD * sx, VP[1] + S * SPREAD * sy
    px = it.font_design * S * k
    a = life_opacity(u)
    blur = near_blur(u) * S * k
    l, t, r, b = (v * px for v in _ref_metrics(i))
    th = math.radians(it.rz)
    c, s = math.cos(th), math.sin(th)
    quad = tuple((cx + c * x - s * y, cy + s * x + c * y) for x, y in ((l, t), (r, t), (r, b), (l, b)))
    xs = [q[0] for q in quad]
    ys = [q[1] for q in quad]
    bbox = (min(xs), min(ys), max(xs), max(ys))
    atten = word_attenuation(cx, bbox[3]) * edge_attenuation(bbox)
    a_eff = a * atten * clip_fade(frame)
    return WordState(u, z, cx, cy, px, a, a_eff, blur, it.rz, bbox, atten, quad)


def is_legible(st: WordState) -> bool:
    l, t, r, b = st.bbox
    return (st.px >= LEGIBLE_PX and st.alpha_eff >= LEGIBLE_ALPHA and st.blur_px == 0.0
            and l >= 0 and t >= 0 and r <= W and b <= H)


def legible_frames(i: int) -> list[int]:
    """Longest contiguous run of in-clip frames where word i is readable."""
    best: list[int] = []
    cur: list[int] = []
    for f in range(N_FRAMES):
        if is_legible(word_state(i, f)):
            cur.append(f)
            if len(cur) > len(best):
                best = list(cur)
        else:
            cur = []
    return best


def legible_window_s(i: int) -> float:
    return len(legible_frames(i)) / FPS


# ================================================================== drawing
MIP_LEVELS = (24, 48, 96, 192, 384)


@lru_cache(maxsize=None)
def _sprite(i: int, level: int) -> Image.Image:
    """Alpha mask of word i (text + soft halo) at font size `level`, text centred."""
    it = ITEMS[i]
    f = lab.font(level, True)
    sigma = 8.0 / it.font_design * level          # demo: text-shadow blur 16px -> sigma 8 at font size fs
    pad = int(math.ceil(3 * sigma)) + 4
    l, t, r, b = f.getbbox(it.text, anchor="mm")
    hw = int(math.ceil(max(-l, r))) + pad
    hh = int(math.ceil(max(-t, b))) + pad
    mask = Image.new("L", (2 * hw, 2 * hh), 0)
    ImageDraw.Draw(mask).text((hw, hh), it.text, font=f, fill=255, anchor="mm")
    halo = np.asarray(mask.filter(ImageFilter.GaussianBlur(sigma)), dtype=np.float32) / 255.0
    txt = np.asarray(mask, dtype=np.float32) / 255.0
    comb = txt + (1.0 - txt) * np.clip(halo * 1.6, 0, 1) * HALO_A
    return Image.fromarray(np.round(comb * 255).astype(np.uint8), "L")


def _word_crop(i: int, st: WordState) -> tuple[np.ndarray, int, int] | None:
    """Transformed alpha crop (float 0..1) and its frame offset, or None if off-frame."""
    level = next((lv for lv in MIP_LEVELS if lv >= st.px), MIP_LEVELS[-1])
    spr = _sprite(i, level)
    sw, sh = spr.size
    s = st.px / level
    th = math.radians(st.rz)
    c, sn = math.cos(th), math.sin(th)
    hw = s * (sw / 2 * abs(c) + sh / 2 * abs(sn)) + 2 + 3 * st.blur_px
    hh = s * (sw / 2 * abs(sn) + sh / 2 * abs(c)) + 2 + 3 * st.blur_px
    x0, y0 = max(0, int(math.floor(st.cx - hw))), max(0, int(math.floor(st.cy - hh)))
    x1, y1 = min(W, int(math.ceil(st.cx + hw))), min(H, int(math.ceil(st.cy + hh)))
    if x1 <= x0 or y1 <= y0:
        return None
    ox, oy = x0 - st.cx, y0 - st.cy
    coeffs = (c / s, sn / s, (c * ox + sn * oy) / s + sw / 2,
              -sn / s, c / s, (-sn * ox + c * oy) / s + sh / 2)
    crop = spr.transform((x1 - x0, y1 - y0), Image.AFFINE, coeffs, resample=Image.BICUBIC, fillcolor=0)
    if st.blur_px > 0.05:
        crop = crop.filter(ImageFilter.GaussianBlur(st.blur_px))
    return np.asarray(crop, dtype=np.float32) / 255.0, x0, y0


def _background() -> np.ndarray:
    """SURFACE, darkened toward the edges about the vanishing point (demo's radial vignette)."""
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32) + 0.5
    d = np.hypot((xs - VP[0]) / (0.6 * W), (ys - VP[1]) / (0.6 * H)) / 0.75
    k = 1.0 - 0.32 * np.clip(d, 0, 1) ** 1.5
    return np.asarray(rgb(SURFACE), dtype=np.float32)[None, None, :] * k[..., None]


def _glow_map() -> np.ndarray:
    """Demo glow: 220 design px circle, rgba(c,.22) -> transparent at 68 % of farthest-corner, blur 6."""
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32) + 0.5
    r = np.hypot(xs - VP[0], ys - VP[1]) / S                       # design px
    r_end = 0.68 * 110 * math.sqrt(2)
    a = 0.22 * np.clip(1 - r / r_end, 0, 1)
    img = Image.fromarray(np.round(a * 255 / 0.22 * 0.9).astype(np.uint8), "L").filter(ImageFilter.GaussianBlur(6 * S))
    return np.asarray(img, dtype=np.float32) / 255.0 / 0.9 * 0.22


BG = _background()
GLOW = _glow_map()


def glow_breath(frame: int) -> float:
    return 0.75 + math.sin(2 * math.pi * BREATHS * ((frame % N_FRAMES) / N_FRAMES)) * 0.12


def _word_layer_f(frame: int, words: bool = True) -> np.ndarray:
    out = BG.copy()
    states = sorted(((word_state(i, frame), i) for i in range(len(ITEMS))), key=lambda p: p[0].z)
    for st, i in states if words else ():                          # far -> near
        if st.alpha * st.atten <= 0.003:
            continue
        got = _word_crop(i, st)
        if got is None:
            continue
        m, x0, y0 = got
        h, w = m.shape
        a = m * (st.alpha * st.atten)
        reg = out[y0:y0 + h, x0:x0 + w]
        col = np.asarray(rgb(ITEMS[i].color), dtype=np.float32)
        reg += (col - reg) * a[..., None]
    g = GLOW * glow_breath(frame)
    out += (np.asarray(rgb(GLOW_COLOR), dtype=np.float32) - out) * g[..., None]
    return out


def render_word_layer(frame: int) -> np.ndarray:
    """Background + words + glow (no title, no fades): the seamlessly looping layer."""
    return np.clip(np.round(_word_layer_f(frame)), 0, 255).astype(np.uint8)


# ------------------------------------------------------------------ foreground title (matches the intro card)
TITLE_X = 96
TITLE_Y = 700
TITLE_T0 = 2.4
REVEAL_DUR = 1.1          # lab TextEl 0.7 s, slowed for a calm title
TITLE_SIZE, FULL_SIZE, FULL_LINE = 124, 38, 50


@lru_cache(maxsize=1)
def title_lines() -> tuple[str, ...]:
    """Balanced wrap, as on the intro card: 'Systems Neuroscience &' / 'Neuromorphic Engineering Lab'."""
    return tuple(lab.wrap_balanced(LAB_FULL, 760, lab.measure_fn(lab.font(FULL_SIZE))))


@lru_cache(maxsize=1)
def title_els() -> tuple:
    x, y = TITLE_X, TITLE_Y
    return (
        lab.BarEl(x, y, 64, 6, ACCENT, TITLE_T0, 1.0),
        lab.TextEl(lab.text_image([LAB_SHORT], lab.font(TITLE_SIZE, True), TEXT_1, 130),
                   x - 6, y + 22, TITLE_T0 + 0.25, REVEAL_DUR, rise=22),
        lab.TextEl(lab.text_image(list(title_lines()), lab.font(FULL_SIZE), TEXT_1, FULL_LINE),
                   x, y + 174, TITLE_T0 + 0.6, REVEAL_DUR),
    )


def title_landed_s() -> float:
    return max(el.t0 + el.dur for el in title_els())


def title_font_sizes() -> tuple[int, ...]:
    return (TITLE_SIZE, FULL_SIZE)


def rendered_strings() -> list[str]:
    return [it.text for it in ITEMS] + [LAB_SHORT] + list(title_lines())


def palette_used() -> set[str]:
    return {SURFACE, GLOW_COLOR, ACCENT, TEXT_1} | {it.color for it in ITEMS}


def render_frame(frame: int) -> Image.Image:
    fr = Image.fromarray(render_word_layer(frame), "RGB").convert("RGBA")
    t = frame / FPS
    for el in title_els():
        el.draw(fr, t)
    return lab.fade_to_surface(fr, clip_fade(frame)).convert("RGB")


# ================================================================== output
def _frame_bytes(frame: int) -> bytes:
    return render_frame(frame).tobytes()


def render_video(out_path: Path = OUT_PATH, workers: int = 12) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [FFMPEG, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264", "-preset", "slow", "-crf", "16",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_path)]
    import subprocess
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    with Pool(workers) as pool:
        for b in pool.imap(_frame_bytes, range(N_FRAMES), chunksize=4):
            enc.stdin.write(b)
    enc.stdin.close()
    if enc.wait() != 0:
        raise RuntimeError("ffmpeg failed")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preview", type=Path, metavar="DIR", help="write a few PNG frames instead of the video")
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args()
    short = [(it.text, legible_window_s(i)) for i, it in enumerate(ITEMS) if legible_window_s(i) < MIN_LEGIBLE_S]
    if short:
        raise SystemExit(f"words without a {MIN_LEGIBLE_S} s readable pass: {short}")
    if args.preview:
        args.preview.mkdir(parents=True, exist_ok=True)
        for f in np.linspace(0, N_FRAMES - 1, 8).round().astype(int):
            render_frame(int(f)).save(args.preview / f"flying_words_f{int(f):03d}.png")
        print(f"previews -> {args.preview}")
        return
    render_video(args.out)
    print(f"{args.out} ({N_FRAMES} frames)")


if __name__ == "__main__":
    main()
