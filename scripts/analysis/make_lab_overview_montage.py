#!/usr/bin/env python3
"""Raman Lab intro card + research-overview montage (1080x1080, H.264).

Two square videos for the lab's talks/website, styled to sit between the
fly-behaviour montages (make_per_montage_video.py,
make_repeated_exposure_montage_video.py): dark surface #1a1a19, Arial,
TEXT_1/TEXT_2/MUTED text tokens, 40 fps, libx264 crf 16, yuv420p, +faststart.

  1. Intro card (~7 s): drifting lab imagery behind a dark scrim, staggered
     text reveal, an odor-plume particle field and a pulsing
     glomerulus-like dot cluster. Starts and ends on the bare surface so it
     loops / splices cleanly.
  2. Overview montage (~52 s): opening (short intro card) -> 01 Biological
     Olfaction -> 02 Artificial Olfaction -> 03 Insect-Based Brain-Machine
     Interface for Bio-Hybrid Olfaction -> closing (lab name + URL).
     Each section: numbered heading, two sentences, 2-3 images with Ken Burns
     motion, every image captioned and credited. Images hand over by dipping
     through the surface (outgoing card + its caption/credit fade out, then the
     incoming ones fade in), so two differently shaped cards never ghost.

Accent colour: WashU red #a51417, taken from the lab site's theme stylesheet
(wp-content/themes/wutheme_chauvenet/style.css: `h2.is-style-underline::before
{background:#a51417}`, pagination buttons, a:hover). The theme's link red
#d01c29 (`a{color:#d01c29}`) is used for the small section numerals because
it holds contrast on the dark surface. Both used sparingly.

Assets are downloaded once (images only) into ASSET_DIR and reused; every
image's URL and citation is in ASSETS and is written to
raman_lab_montage_sources.md next to the outputs.

    python3 scripts/analysis/make_lab_overview_montage.py            # both videos
    python3 scripts/analysis/make_lab_overview_montage.py --preview DIR
"""

from __future__ import annotations

import argparse
import math
import subprocess
import tempfile
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

# Anaconda's ffmpeg has no libx264; the system build does.
FFMPEG = "/usr/bin/ffmpeg"

W = H = 1080
FPS = 40

SURFACE = "#1a1a19"
TEXT_1 = "#f0efec"
TEXT_2 = "#c3c2b7"
MUTED = "#8a8984"
ACCENT = "#a51417"        # chauvenet style.css: h2 underline rule, pagination, a:hover
ACCENT_LINK = "#d01c29"   # chauvenet style.css: a{color:#d01c29}
CARD_LIGHT = "#e4e3df"    # TEXT_1 at ~95 %: light card for white-background figures
CARD_DARK = "#000000"     # matches the black backgrounds of the dark figures

FONT_DIR = Path("/usr/share/fonts/truetype/msttcorefonts")
FONT_REG = FONT_DIR / "Arial.ttf"
FONT_BOLD = FONT_DIR / "Arial_Bold.ttf"

ASSET_DIR = Path("/home/ramanlab/Documents/cole/Results/Figures/raman_lab_montage_assets/lab_assets")
OUT_DIR = Path("/home/ramanlab/Documents/cole/Results/Figures")
CARD_OUT = OUT_DIR / "raman_lab_intro_card.mp4"
MONTAGE_OUT = OUT_DIR / "raman_lab_research_overview_montage.mp4"
SOURCES_OUT = OUT_DIR / "raman_lab_montage_sources.md"

# ------------------------------------------------------------------ timing
CARD_S = 7.0
OPENING_S = 5.0
CLOSING_S = 5.5
DISSOLVE_S = 1.2          # image-to-image hand-over inside a section: dip through the surface
                          # (first half: outgoing fades out; second half: incoming fades in)
BG_DISSOLVE_S = 1.6       # background cross-dissolve in hero scenes
SEG_FADE_OUT_S = 0.55     # every montage segment ends on the bare surface
BG_DESATURATE = 0.35      # hero-scene backgrounds: fraction blended toward greyscale

# ------------------------------------------------------------------ layout
MARGIN = 72
AREA_X, AREA_W = 60, 960        # image area in sections; its top follows the text block
AREA_BOTTOM = 972               # sibling montages' footnote band: CAPTION_Y below sits under it
AREA_GAP = 34                   # text block -> image area
MIN_AREA_H = 480
CAPTION_Y = AREA_BOTTOM + 16    # one-line footnote top (lab_video_style re-exports it); lab sections
                                # place their own caption block via section_caption_y()
CAPTION_GAP = 16                # image area -> section caption block
CAPTION_X = MARGIN              # section captions/credits share the heading + body left edge
SAFE_BOTTOM = 40                # no text in the bottom 40 px of the frame
CARD_RADIUS = 16
MIN_TEXT_PX = 32                # legibility floor: >= 3 % of the 1080 px frame height for any text
HEAD_SIZE, HEAD_LINE = 56, 66   # headings >= 5 % of frame height
BODY_SIZE, BODY_LINE = 32, 46
CAPTION_SIZE, CAPTION_LINE = MIN_TEXT_PX, 42
NUMERAL_SIZE = 34               # segment numeral "01"; "/ 03" and the brand use CAPTION_SIZE


# ================================================================== assets
@dataclass(frozen=True)
class Asset:
    key: str
    url: str
    rel: str                     # path under ASSET_DIR
    caption: str
    credit: str                  # on-screen credit
    citation: str                # full citation for the sources file
    page: str                    # page the image was found on
    mode: str                    # "cover" (photo) | "light" | "dark" (figure on a card)
    crop: tuple[float, float, float, float] | None = None   # fractional l, t, r, b
    kb: tuple[tuple[float, float, float], tuple[float, float, float]] = ((1.0, 0.5, 0.5), (1.08, 0.5, 0.5))
    max_up: float = 2.0          # cards: never upscale a small source more than this
    soften: float = 0.0          # Gaussian radius (source px) to soften GIF dithering


SITE = "Image: ramanlab.wustl.edu"
CHANDAK = "Chandak & Raman, Nature Communications, 2023"
SAHA = "Saha et al., Biosensors and Bioelectronics: X, 2020"

ASSETS: dict[str, Asset] = {a.key: a for a in (
    Asset("locust", "https://ramanlab.wustl.edu/files/2023/03/577ed6cb5c40c.imageSTLToday.jpg",
          "site/577ed6cb5c40c.imageSTLToday.jpg", "Locust with a neural implant", SITE,
          "Raman Lab website, homepage and Research page (full-size original of the "
          "1024x683 homepage image; file name indicates an STLToday photo).",
          "https://ramanlab.wustl.edu/", "cover",
          kb=((1.04, 0.42, 0.50), (1.16, 0.40, 0.42))),
    Asset("circuits", "https://ramanlab.wustl.edu/files/2023/03/Screen-Shot-2020-05-25-at-10.38.43-PM-1.png",
          "site/Screen-Shot-2020-05-25-at-10.38.43-PM-1.png", "Olfactory circuits", SITE,
          "Raman Lab website, homepage gallery (full-size original).",
          "https://ramanlab.wustl.edu/", "cover",
          kb=((1.0, 0.5, 0.30), (1.06, 0.5, 0.62))),
    Asset("neural", "https://ramanlab.wustl.edu/files/2023/03/NeuralImage2.jpg",
          "site/NeuralImage2.jpg", "Imaging neural signals", SITE,
          "Raman Lab website, homepage gallery.",
          "https://ramanlab.wustl.edu/", "cover",
          kb=((1.0, 0.55, 0.5), (1.10, 0.62, 0.45))),
    Asset("chandak", "https://media.springernature.com/full/springer-static/image/"
          "art%3A10.1038%2Fs41467-023-40443-2/MediaObjects/41467_2023_40443_Fig2_HTML.png",
          "pubs/chandak2023_fig2.png",
          "Odor-evoked projection-neuron responses in the locust antennal lobe", CHANDAK,
          "Chandak, R. & Raman, B. Neural manifolds for odor-driven innate and acquired appetitive "
          "preferences. Nature Communications 14, 4719 (2023). doi:10.1038/s41467-023-40443-2 "
          "(open access, CC BY 4.0; PMC10404252). Fig. 2, panels a-c (cropped).",
          "https://www.nature.com/articles/s41467-023-40443-2", "light",
          crop=(0.0, 0.0, 1.0, 1185 / 2550), kb=((0.955, 0.5, 0.5), (1.0, 0.5, 0.5))),
    Asset("enose", "https://ramanlab.wustl.edu/files/2022/09/image-1.png",
          "site/image-1.png", "An electronic nose", SITE,
          "Raman Lab website, Research page (Artificial Olfaction).",
          "https://ramanlab.wustl.edu/research/", "light",
          kb=((0.955, 0.5, 0.5), (1.0, 0.5, 0.5))),
    Asset("mems", "https://ramanlab.wustl.edu/files/2022/09/image-2.png",
          "site/image-2.png", "MEMS microsensor array for sensing toxic industrial chemicals", SITE,
          "Raman Lab website, Research page (Artificial Olfaction).",
          "https://ramanlab.wustl.edu/research/", "light",
          kb=((0.955, 0.5, 0.5), (1.0, 0.5, 0.5))),
    Asset("chip", "https://ramanlab.wustl.edu/files/2023/03/MicrosensorArray.gif",
          "site/MicrosensorArray.gif", "Electronic nose: microsensor array chip", SITE,
          "Raman Lab website, homepage gallery (the GIF is a single static frame, 230x219).",
          "https://ramanlab.wustl.edu/", "dark",
          kb=((0.955, 0.5, 0.5), (1.0, 0.5, 0.5)), max_up=2.0, soften=0.7),
    Asset("saha_concept", "https://ars.els-cdn.com/content/image/1-s2.0-S2590137020300169-gr1.jpg",
          "pubs/biosx_gr1.jpg", "Insect-based biorobots for static and dynamic odor detection", SAHA,
          "Saha, D., Mehta, D., Altan, E., Chandak, R., Traner, M., Lo, R., Gupta, P., Singamaneni, S., "
          "Chakrabartty, S. & Raman, B. Explosive sensing with insect-based biorobots. Biosensors and "
          "Bioelectronics: X 6, 100050 (2020). doi:10.1016/j.biosx.2020.100050 (open access, CC BY 4.0). Fig. 1.",
          "https://doi.org/10.1016/j.biosx.2020.100050", "light",
          kb=((0.955, 0.5, 0.5), (1.0, 0.5, 0.5))),
    Asset("saha_record", "https://ars.els-cdn.com/content/image/1-s2.0-S2590137020300169-gr3.jpg",
          "pubs/biosx_gr3.jpg", "Minimally invasive recording of odor-evoked neural activity", SAHA,
          "Saha, D. et al. Explosive sensing with insect-based biorobots. Biosensors and "
          "Bioelectronics: X 6, 100050 (2020). doi:10.1016/j.biosx.2020.100050 (open access, CC BY 4.0). Fig. 3.",
          "https://doi.org/10.1016/j.biosx.2020.100050", "light",
          kb=((0.955, 0.5, 0.5), (1.0, 0.5, 0.5))),
)}


@dataclass(frozen=True)
class Section:
    number: str
    heading: str
    body: str
    assets: tuple[str, ...]
    duration: float


SECTIONS: tuple[Section, ...] = (
    Section("01", "Biological Olfaction",
            "How does a nervous system turn odorants into neural representations? We record and "
            "image invertebrate olfactory circuits to uncover the design and computing principles "
            "of biological olfaction.",
            ("circuits", "neural", "chandak"), 14.0),
    Section("02", "Artificial Olfaction",
            "An electronic nose pairs an array of cross-selective chemical sensors with a "
            "pattern-recognition engine. We design neuromorphic, MEMS-based microsensor arrays "
            "for non-invasive chemical sensing.",
            ("enose", "mems", "chip"), 13.0),
    Section("03", "Insect-Based Brain–Machine Interface for Bio-Hybrid Olfaction",
            "Biological olfaction still outperforms its engineered analogues. We build insect–"
            "machine interfaces, reading odor-evoked activity from locusts with neural implants, "
            "for bio-hybrid chemical sensing.",
            ("locust", "saha_concept", "saha_record"), 15.0),
)

LAB_SHORT = "Raman Lab"
LAB_FULL = "Systems Neuroscience & Neuromorphic Engineering Lab"
UNIVERSITY = "Washington University in St. Louis"
THEMES = ("Fruit Fly and Insect Olfaction", "Neuromorphic Sensing", "Bio-Hybrid Systems")
URL = "ramanlab.wustl.edu"


# ================================================================== pure helpers
def clamp01(u: float) -> float:
    return 0.0 if u <= 0 else 1.0 if u >= 1 else float(u)


def ease_out_cubic(u: float) -> float:
    u = clamp01(u)
    return 1.0 - (1.0 - u) ** 3


def smoothstep(u: float) -> float:
    u = clamp01(u)
    return u * u * (3.0 - 2.0 * u)


def reveal(t: float, t0: float, dur: float) -> float:
    """Eased 0->1 progress of an element that starts revealing at t0."""
    return ease_out_cubic((t - t0) / dur) if dur > 0 else float(t >= t0)


def segment_fade(t: float, duration: float, fade_in: float, fade_out: float) -> float:
    """Whole-frame opacity against the surface: ramps up at the start, down at the end."""
    a_in = smoothstep(t / fade_in) if fade_in > 0 else 1.0
    a_out = smoothstep((duration - t) / fade_out) if fade_out > 0 else 1.0
    return min(a_in, a_out)


def ken_burns_rect(src_w: float, src_h: float, out_aspect: float,
                   zoom: float, cx: float, cy: float) -> tuple[float, float, float, float]:
    """Source crop box (l, t, r, b) for a cover-fit Ken Burns frame.

    zoom=1 is the largest crop of aspect `out_aspect` that fits the source;
    zoom>1 shrinks it. (cx, cy) is the desired centre as a fraction of the
    source; the box is clamped so it never leaves the image (no borders).
    """
    if zoom < 1.0:
        raise ValueError("cover Ken Burns zoom must be >= 1 (smaller would show borders)")
    if src_w / src_h > out_aspect:
        ch = float(src_h)
        cw = ch * out_aspect
    else:
        cw = float(src_w)
        ch = cw / out_aspect
    cw, ch = cw / zoom, ch / zoom
    x0 = min(max(cx * src_w - cw / 2.0, 0.0), src_w - cw)
    y0 = min(max(cy * src_h - ch / 2.0, 0.0), src_h - ch)
    return (x0, y0, x0 + cw, y0 + ch)


def kb_params(kb: tuple[tuple[float, float, float], tuple[float, float, float]], u: float) -> tuple[float, float, float]:
    """Linear (constant-velocity) interpolation of (zoom, cx, cy) at progress u."""
    u = clamp01(u)
    (z0, x0, y0), (z1, x1, y1) = kb
    return (z0 + (z1 - z0) * u, x0 + (x1 - x0) * u, y0 + (y1 - y0) * u)


def fit_contain(src_w: float, src_h: float, box_w: float, box_h: float, max_up: float = math.inf) -> tuple[int, int]:
    """Largest size with the source aspect that fits the box (letterbox), capped upscale."""
    s = min(box_w / src_w, box_h / src_h, max_up)
    return max(1, round(src_w * s)), max(1, round(src_h * s))


def image_slots(n: int, duration: float, dissolve: float) -> list[tuple[float, float]]:
    """n equal, overlapping (start, end) slots covering [0, duration]; overlaps = dissolve."""
    if n < 1:
        raise ValueError("need at least one image")
    length = (duration + (n - 1) * dissolve) / n
    if length <= 2 * dissolve:
        raise ValueError("slots too short for the dissolve")
    return [(i * (length - dissolve), i * (length - dissolve) + length) for i in range(n)]


def slot_weights(slots: Sequence[tuple[float, float]], t: float, dissolve: float) -> list[float]:
    """Visibility of each image at t; adjacent weights sum to 1 during a dissolve."""
    w = []
    n = len(slots)
    for i, (s, e) in enumerate(slots):
        a_in = 1.0 if i == 0 else smoothstep((t - s) / dissolve)
        a_out = 1.0 if i == n - 1 else 1.0 - smoothstep((t - (e - dissolve)) / dissolve)
        inside = (s <= t < e) or (i == n - 1 and t >= e) or (i == 0 and t < s)
        w.append(min(a_in, a_out) if inside else 0.0)
    return w


def dip_weights(slots: Sequence[tuple[float, float]], t: float, dissolve: float) -> list[float]:
    """Visibility of each image at t when hand-overs dip through the surface: within each
    `dissolve`-long overlap the outgoing image fades out over the first half and the incoming
    one fades in over the second, so at most one image is ever visible."""
    half = dissolve / 2
    w = []
    n = len(slots)
    for i, (s, e) in enumerate(slots):
        a_in = 1.0 if i == 0 else smoothstep((t - (s + half)) / half)
        a_out = 1.0 if i == n - 1 else 1.0 - smoothstep((t - (e - dissolve)) / half)
        inside = (s <= t < e) or (i == n - 1 and t >= e) or (i == 0 and t < s)
        w.append(min(a_in, a_out) if inside else 0.0)
    return w


def wrap_text(text: str, max_width: float, measure: Callable[[str], float]) -> list[str]:
    """Greedy word wrap; a single over-long word gets its own line rather than being split."""
    lines: list[str] = []
    cur = ""
    for word in text.split():
        trial = f"{cur} {word}" if cur else word
        if cur and measure(trial) > max_width:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines


def wrap_balanced(text: str, max_width: float, measure: Callable[[str], float]) -> list[str]:
    """Same line count as the greedy wrap at max_width, but at the narrowest width
    that still achieves it, so the last line is not a one-word orphan."""
    lines = wrap_text(text, max_width, measure)
    if len(lines) <= 1:
        return lines
    lo = max(measure(w) for w in text.split())
    hi = float(max_width)
    while hi - lo > 1.0:
        mid = (lo + hi) / 2
        if len(wrap_text(text, mid, measure)) <= len(lines):
            hi = mid
        else:
            lo = mid
    return wrap_text(text, hi, measure)


@dataclass(frozen=True)
class Seg:
    name: str
    start: float
    duration: float


def montage_timeline() -> list[Seg]:
    segs, t = [], 0.0
    for name, d in [("opening", OPENING_S)] + [(f"section{s.number}", s.duration) for s in SECTIONS] + [("closing", CLOSING_S)]:
        segs.append(Seg(name, t, d))
        t += d
    return segs


def montage_duration() -> float:
    return sum(s.duration for s in montage_timeline())


def frame_count(duration: float, fps: int = FPS) -> int:
    return int(round(duration * fps))


# ================================================================== drawing helpers
def rgb(hex_: str) -> tuple[int, int, int]:
    return tuple(int(hex_[i:i + 2], 16) for i in (1, 3, 5))


@lru_cache(maxsize=None)
def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_BOLD if bold else FONT_REG), size)


def measure_fn(f: ImageFont.FreeTypeFont) -> Callable[[str], float]:
    return lambda s: f.getlength(s)


def text_image(lines: Sequence[str], f: ImageFont.FreeTypeFont, color: str, line_h: int) -> Image.Image:
    """RGBA text block (colour + coverage alpha, no dark fringes)."""
    if f.size < MIN_TEXT_PX:
        raise ValueError(f"{f.size} px text is below the {MIN_TEXT_PX} px legibility floor")
    width = int(math.ceil(max(f.getlength(s) for s in lines))) + 4
    height = int(line_h * (len(lines) - 1) + f.size * 1.3) + 2
    mask = Image.new("L", (width, height), 0)
    d = ImageDraw.Draw(mask)
    for i, s in enumerate(lines):
        d.text((0, i * line_h), s, font=f, fill=255, anchor="la")
    layer = Image.new("RGBA", (width, height), rgb(color) + (255,))
    layer.putalpha(mask)
    return layer


def rect_image(w: int, h: int, color: str) -> Image.Image:
    return Image.new("RGBA", (w, h), rgb(color) + (255,))


@lru_cache(maxsize=32)
def rounded_mask(size: tuple[int, int], radius: int) -> Image.Image:
    ss = 4
    big = Image.new("L", (size[0] * ss, size[1] * ss), 0)
    ImageDraw.Draw(big).rounded_rectangle((0, 0, big.width - 1, big.height - 1), radius * ss, fill=255)
    return big.resize(size, Image.LANCZOS)


def with_alpha(layer: Image.Image, a: float) -> Image.Image:
    if a >= 0.999:
        return layer
    out = layer.copy()
    out.putalpha(layer.getchannel("A").point(lambda v: int(v * a + 0.5)))
    return out


def fade_to_surface(frame: Image.Image, a: float) -> Image.Image:
    if a >= 0.999:
        return frame
    return Image.blend(Image.new(frame.mode, frame.size, rgb(SURFACE) + ((255,) if frame.mode == "RGBA" else ())), frame, max(a, 0.0))


@dataclass
class TextEl:
    img: Image.Image
    x: int
    y: int
    t0: float
    dur: float = 0.7
    rise: int = 16

    def draw(self, frame: Image.Image, t: float, alpha: float = 1.0) -> None:
        u = reveal(t, self.t0, self.dur)
        a = u * alpha
        if a <= 0.003:
            return
        frame.alpha_composite(with_alpha(self.img, a), (self.x, self.y + int(round((1 - u) * self.rise))))


@dataclass
class BarEl:
    """Accent rule that grows from the left."""
    x: int
    y: int
    w: int
    h: int
    color: str
    t0: float
    dur: float = 0.8

    def draw(self, frame: Image.Image, t: float, alpha: float = 1.0) -> None:
        u = reveal(t, self.t0, self.dur)
        w = int(round(self.w * u))
        if w < 1 or alpha <= 0.003:
            return
        frame.alpha_composite(with_alpha(rect_image(w, self.h, self.color), alpha), (self.x, self.y))


# ================================================================== assets I/O
def ensure_assets(asset_dir: Path = ASSET_DIR) -> None:
    """Download any missing asset (images only). Existing files are never re-fetched."""
    for a in ASSETS.values():
        p = asset_dir / a.rel
        if p.exists() and p.stat().st_size > 0:
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(a.url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read()
        tmp = p.with_suffix(p.suffix + ".part")
        tmp.write_bytes(data)
        Image.open(tmp).verify()          # must decode as an image, else fail loudly
        tmp.rename(p)


def load_asset(a: Asset, asset_dir: Path = ASSET_DIR) -> Image.Image:
    im = Image.open(asset_dir / a.rel)
    im.seek(0)
    im = im.convert("RGBA")
    if a.crop:
        l, t, r, b = a.crop
        im = im.crop((round(l * im.width), round(t * im.height), round(r * im.width), round(b * im.height)))
    if a.soften > 0:
        im = im.filter(ImageFilter.GaussianBlur(a.soften))
    return im


class ImageLayer:
    """One section image rendered into the AREA box at Ken Burns progress u."""

    def __init__(self, asset: Asset, box: tuple[int, int], asset_dir: Path = ASSET_DIR):
        self.a = asset
        self.bw, self.bh = box
        src = load_asset(asset, asset_dir)
        self.bg = Image.new("RGB", box, rgb(SURFACE))
        if asset.mode == "cover":
            self.src = src.convert("RGB")
            self.mask = rounded_mask(box, CARD_RADIUS)
        else:
            self.plate = self._plate(src)

    def _plate(self, src: Image.Image) -> Image.Image:
        pad = 22
        iw, ih = fit_contain(src.width, src.height, self.bw - 2 * pad, self.bh - 2 * pad, self.a.max_up)
        img = src.resize((iw, ih), Image.LANCZOS)
        if self.a.mode == "light":
            base = Image.new("RGBA", img.size, (255, 255, 255, 255))
            base.alpha_composite(img)
            # Multiply onto the card colour: figure whites become the card, no white box.
            img_rgb = ImageChops.multiply(base.convert("RGB"), Image.new("RGB", img.size, rgb(CARD_LIGHT)))
            card_col = CARD_LIGHT
        else:
            base = Image.new("RGBA", img.size, rgb(CARD_DARK) + (255,))
            base.alpha_composite(img)
            img_rgb = base.convert("RGB")
            card_col = CARD_DARK
        card = Image.new("RGB", (iw + 2 * pad, ih + 2 * pad), rgb(card_col))
        card.paste(img_rgb, (pad, pad))
        card = card.convert("RGBA")
        card.putalpha(rounded_mask(card.size, CARD_RADIUS))
        plate = Image.new("RGBA", (self.bw, self.bh), (0, 0, 0, 0))
        plate.alpha_composite(card, ((self.bw - card.width) // 2, (self.bh - card.height) // 2))
        return plate

    def render(self, u: float) -> Image.Image:
        z, cx, cy = kb_params(self.a.kb, u)
        out = self.bg.copy()
        if self.a.mode == "cover":
            box = ken_burns_rect(self.src.width, self.src.height, self.bw / self.bh, z, cx, cy)
            out.paste(self.src.resize((self.bw, self.bh), Image.BICUBIC, box=box), (0, 0), self.mask)
        else:
            # Card grows z (0.955 -> 1.0) about the box centre, sub-pixel via affine.
            dx, dy = (cx - 0.5) * self.bw, (cy - 0.5) * self.bh
            c = self.bw / 2 - (self.bw / 2 + dx) / z
            f = self.bh / 2 - (self.bh / 2 + dy) / z
            layer = self.plate.transform((self.bw, self.bh), Image.AFFINE, (1 / z, 0, c, 0, 1 / z, f),
                                         resample=Image.BICUBIC, fillcolor=(0, 0, 0, 0))
            out.paste(layer, (0, 0), layer)
        return out


# ================================================================== hero scenes
class Motif:
    """Odor-plume particles drifting left->right plus a pulsing glomerulus cluster."""

    def __init__(self, seed: int = 7, n: int = 210):
        rng = np.random.default_rng(seed)
        self.p = rng.random(n)
        self.v = rng.uniform(34, 72, n)
        self.z = np.clip(rng.standard_normal(n), -2.2, 2.2)
        self.phi = rng.random(n)
        self.r = rng.uniform(1.0, 2.9, n)
        self.base = rng.uniform(0.25, 0.75, n)
        self.accent = rng.random(n) < 0.07
        # glomeruli: phyllotaxis disc
        k = 34
        self.gc = (W * 0.835, H * 0.205)
        R = 160
        idx = np.arange(k)
        rad = R * np.sqrt((idx + 0.5) / k)
        th = idx * math.radians(137.508)
        self.gx = self.gc[0] + rad * np.cos(th)
        self.gy = self.gc[1] + rad * np.sin(th)
        self.gr = rng.uniform(10, 14, k)
        self.gphi = rng.random(k)
        self.gactive = rng.random(k) < 0.45
        ring = Image.new("L", (W, H), 0)
        d = ImageDraw.Draw(ring)
        for x, y, r in zip(self.gx, self.gy, self.gr):
            d.ellipse((x - r, y - r, x + r, y + r), outline=int(255 * 0.20), width=2)
        self.ring = ring.filter(ImageFilter.GaussianBlur(0.6))

    def draw(self, frame: Image.Image, t: float, alpha: float = 1.0) -> None:
        if alpha <= 0.003:
            return
        L = W + 80
        x = -40 + ((self.p + self.v * t / L) % 1.0) * L
        xf = np.clip(x / W, 0, 1)
        y_src = H * 0.86 - H * 0.14 * xf
        spread = 26 + 170 * xf
        y = y_src + self.z * spread + 20 * np.sin(2 * np.pi * (x / 380 + self.phi) + 0.8 * t)
        edge = np.clip((x + 40) / 160, 0, 1) * np.clip((W + 40 - x) / 160, 0, 1)
        inten = self.base * edge * (1 - 0.45 * xf) * 0.55 * alpha
        m_neu = Image.new("L", (W, H), 0)
        m_acc = Image.new("L", (W, H), 0)
        dn, da = ImageDraw.Draw(m_neu), ImageDraw.Draw(m_acc)
        for xi, yi, ri, ii, ac in zip(x, y, self.r, inten, self.accent):
            if ii <= 0.01:
                continue
            (da if ac else dn).ellipse((xi - ri, yi - ri, xi + ri, yi + ri), fill=int(255 * min(ii * (1.5 if ac else 1.0), 1)))
        frame.paste(rgb(TEXT_2), (0, 0, W, H), m_neu.filter(ImageFilter.GaussianBlur(0.8)))
        frame.paste(rgb(ACCENT_LINK), (0, 0, W, H), m_acc.filter(ImageFilter.GaussianBlur(0.8)))
        # glomeruli: faint outlines, sparse accent "activations"
        frame.paste(rgb(MUTED), (0, 0, W, H), self.ring.point(lambda v: int(v * alpha)))
        fill = Image.new("L", (W, H), 0)
        d = ImageDraw.Draw(fill)
        for gx, gy, gr, ph, act in zip(self.gx, self.gy, self.gr, self.gphi, self.gactive):
            if not act:
                continue
            a = max(0.0, math.sin(2 * math.pi * (t / 3.5 + ph))) ** 6 * 0.5 * alpha
            if a > 0.01:
                d.ellipse((gx - gr + 2, gy - gr + 2, gx + gr - 2, gy + gr - 2), fill=int(255 * a))
        frame.paste(rgb(ACCENT_LINK), (0, 0, W, H), fill.filter(ImageFilter.GaussianBlur(1.0)))


def make_scrim() -> Image.Image:
    """Dark surface overlay: 74 % base, darker toward the left (text side) and edges."""
    xs = np.linspace(0, 1, W)[None, :]
    ys = np.linspace(-1, 1, H)[:, None]
    xc = xs * 2 - 1
    a = 0.74 + 0.16 * (1 - xs) ** 1.5 + 0.10 * np.clip(np.sqrt(xc ** 2 + ys ** 2) - 0.6, 0, 1)
    a = np.clip(a, 0, 0.96)
    s = Image.new("RGBA", (W, H), rgb(SURFACE) + (0,))
    s.putalpha(Image.fromarray((a * 255).astype(np.uint8)))
    return s


# Hero-card type scale (px). Everything is >= MIN_TEXT_PX; "Raman Lab" leads, the
# full lab name is second, the university / theme list / URL are the quiet tier.
HERO_SIZES = {
    "card":    dict(lab_short=124, lab_full=40, university=32, themes=32),
    "opening": dict(lab_short=124, lab_full=40, university=32, themes=32),
    "closing": dict(lab_short=112, lab_full=38, university=32, url=36),
}


def hero_font_sizes(variant: str) -> dict:
    return dict(HERO_SIZES[variant])


def hero_layout(variant: str) -> list:
    """Text + rule elements of the intro card / opening / closing (pure given fonts)."""
    z = HERO_SIZES[variant]
    x = 96
    sp = 0.72 if variant == "opening" else 1.0     # stagger compression
    els: list = []
    uni_line, th_line = z["university"] + 8, z.get("themes", 32) + 8
    full_line = z["lab_full"] + 12
    if variant in ("card", "opening"):
        y = 300
        els.append(BarEl(x, y, 64, 6, ACCENT, 0.30 * sp))
        els.append(TextEl(text_image([UNIVERSITY], font(z["university"]), TEXT_2, uni_line), x, y + 28, 0.45 * sp))
        els.append(TextEl(text_image([LAB_SHORT], font(z["lab_short"], True), TEXT_1, 130), x - 6, y + 80, 0.70 * sp, rise=22))
        full = wrap_balanced(LAB_FULL, 760, measure_fn(font(z["lab_full"])))
        fy = y + 236
        els.append(TextEl(text_image(full, font(z["lab_full"]), TEXT_1, full_line), x, fy, 1.00 * sp))
        ry = fy + full_line * (len(full) - 1) + int(z["lab_full"] * 1.3) + 34
        els.append(BarEl(x, ry, 300, 1, MUTED, 1.30 * sp, 1.0))
        tsz = z["themes"]
        dot = max(8, round(tsz * 0.25))
        for i, th in enumerate(THEMES):
            ty = ry + 30 + i * (th_line + 12)
            dy = round(tsz * 0.905 - tsz * 0.716 / 2 - dot / 2)       # dot centred on the cap height
            els.append(TextEl(rect_image(dot, dot, ACCENT_LINK), x, ty + dy, (1.50 + 0.2 * i) * sp, rise=10))
            els.append(TextEl(text_image([th], font(tsz), TEXT_2, th_line), x + dot + 16, ty, (1.50 + 0.2 * i) * sp, rise=10))
    else:  # closing
        y = 330
        els.append(BarEl(x, y, 64, 6, ACCENT, 0.25))
        els.append(TextEl(text_image([LAB_SHORT], font(z["lab_short"], True), TEXT_1, 120), x - 5, y + 30, 0.40, rise=22))
        full = wrap_balanced(LAB_FULL, 760, measure_fn(font(z["lab_full"])))
        fy = y + 174
        els.append(TextEl(text_image(full, font(z["lab_full"]), TEXT_1, full_line), x, fy, 0.65))
        uy = fy + full_line * (len(full) - 1) + int(z["lab_full"] * 1.3) + 16
        els.append(TextEl(text_image([UNIVERSITY], font(z["university"]), TEXT_2, uni_line), x, uy, 0.85))
        ry = uy + int(z["university"] * 1.3) + 34
        els.append(BarEl(x, ry, 300, 1, MUTED, 1.05, 1.0))
        els.append(TextEl(text_image([URL], font(z["url"], True), TEXT_1, z["url"] + 6), x, ry + 30, 1.25))
    return els


class HeroScene:
    """Intro card / montage opening / montage closing."""

    def __init__(self, variant: str, duration: float, bg_keys: Sequence[str],
                 bg_kbs: Sequence[tuple] | None = None, asset_dir: Path = ASSET_DIR,
                 fade_in: float = 0.9, fade_out: float = 0.8):
        self.variant, self.duration = variant, duration
        self.fade_in, self.fade_out = fade_in, fade_out
        # Backgrounds are partly desaturated so saturated micrographs stay behind the type.
        self.bgs = [load_asset(ASSETS[k], asset_dir).convert("RGB") for k in bg_keys]
        self.bgs = [Image.blend(im, im.convert("L").convert("RGB"), BG_DESATURATE) for im in self.bgs]
        self.kbs = list(bg_kbs) if bg_kbs else [((1.0, 0.5, 0.5), (1.12, 0.5, 0.5))] * len(self.bgs)
        self.slots = image_slots(len(self.bgs), duration, BG_DISSOLVE_S)
        self.scrim = make_scrim()
        self.motif = Motif()
        self.els = self._layout()

    def _layout(self) -> list:
        return hero_layout(self.variant)

    def background(self, t: float) -> Image.Image:
        ws = slot_weights(self.slots, t, BG_DISSOLVE_S)
        layers = []
        for i, (img, kb, (s, e)) in enumerate(zip(self.bgs, self.kbs, self.slots)):
            if ws[i] <= 0:
                continue
            z, cx, cy = kb_params(kb, (t - s) / (e - s))
            box = ken_burns_rect(img.width, img.height, 1.0, z, cx, cy)
            layers.append((ws[i], img.resize((W, H), Image.BICUBIC, box=box)))
        if len(layers) == 1:
            out = layers[0][1]
        else:
            (w0, a), (w1, b) = layers[:2]
            out = Image.blend(a, b, w1 / (w0 + w1))
        out = out.convert("RGBA")
        out.alpha_composite(self.scrim)
        return out

    def frame(self, t: float) -> Image.Image:
        fr = self.background(t)
        self.motif.draw(fr, t, alpha=smoothstep((t - 0.2) / 1.2))
        for el in self.els:
            el.draw(fr, t)
        return fade_to_surface(fr, segment_fade(t, self.duration, self.fade_in, self.fade_out)).convert("RGB")


CARD_BG = (("locust", ((1.06, 0.46, 0.50), (1.20, 0.52, 0.44))),
           ("circuits", ((1.0, 0.50, 0.40), (1.10, 0.56, 0.55))))
CLOSING_BG = (("neural", ((1.0, 0.45, 0.5), (1.12, 0.6, 0.5))),)


def card_scene(asset_dir: Path = ASSET_DIR) -> HeroScene:
    return HeroScene("card", CARD_S, [k for k, _ in CARD_BG], [kb for _, kb in CARD_BG], asset_dir)


def opening_scene(asset_dir: Path = ASSET_DIR) -> HeroScene:
    return HeroScene("opening", OPENING_S, [k for k, _ in CARD_BG], [kb for _, kb in CARD_BG], asset_dir,
                     fade_in=0.7, fade_out=SEG_FADE_OUT_S)


def closing_scene(asset_dir: Path = ASSET_DIR) -> HeroScene:
    return HeroScene("closing", CLOSING_S, [k for k, _ in CLOSING_BG], [kb for _, kb in CLOSING_BG], asset_dir,
                     fade_in=0.5, fade_out=0.9)


# ================================================================== section scenes
def section_text_layout(sec: Section) -> dict:
    """Wrapped lines and y positions for a section's header block (pure given fonts)."""
    hf, bf = font(HEAD_SIZE, True), font(BODY_SIZE)
    width = W - 2 * MARGIN
    head = wrap_text(sec.heading, width, measure_fn(hf))
    body = wrap_text(sec.body, width, measure_fn(bf))   # greedy: lab_video_style mirrors this layout
    num_y = 62
    head_y = 112
    body_y = head_y + HEAD_LINE * len(head) + 14
    bottom = body_y + BODY_LINE * (len(body) - 1) + int(BODY_SIZE * 1.3)
    return dict(head=head, body=body, num_y=num_y, head_y=head_y, body_y=body_y, bottom=bottom)


def caption_lines(a: Asset) -> list[str]:
    """An image caption wrapped (balanced) to the text column (CAPTION_X .. W - MARGIN); never truncated."""
    return wrap_balanced(a.caption, W - MARGIN - CAPTION_X, measure_fn(font(CAPTION_SIZE)))


def caption_block_h(n_lines: int) -> int:
    """Height of an n-line caption block at CAPTION_SIZE/CAPTION_LINE (top of line 1 to glyph bottom)."""
    return CAPTION_LINE * (n_lines - 1) + int(math.ceil(CAPTION_SIZE * 1.3))


def section_caption_y(sec: Section) -> int:
    """Top of the section's caption block: its tallest caption (+ credit line) ends SAFE_BOTTOM above the edge."""
    n = max(len(caption_lines(ASSETS[k])) for k in sec.assets) + 1
    return H - SAFE_BOTTOM - caption_block_h(n)


def section_area(sec: Section) -> tuple[int, int, int, int]:
    """Image area (x, y, w, h): from just below the text block down to the caption block."""
    y = section_text_layout(sec)["bottom"] + AREA_GAP
    h = section_caption_y(sec) - CAPTION_GAP - y
    if h < MIN_AREA_H:
        raise ValueError(f"section {sec.number}: text leaves only {h} px for images")
    return (AREA_X, y, AREA_W, h)


class SectionScene:
    def __init__(self, sec: Section, asset_dir: Path = ASSET_DIR):
        self.sec, self.duration = sec, sec.duration
        self.ax, self.ay, aw, ah = section_area(sec)
        self.layers = [ImageLayer(ASSETS[k], (aw, ah), asset_dir) for k in sec.assets]
        self.slots = image_slots(len(self.layers), sec.duration, DISSOLVE_S)
        lay = section_text_layout(sec)
        num = text_image([sec.number], font(NUMERAL_SIZE, True), ACCENT_LINK, NUMERAL_SIZE + 4)
        of = text_image([f"/ {len(SECTIONS):02d}"], font(CAPTION_SIZE), MUTED, CAPTION_LINE)
        brand = text_image([LAB_SHORT], font(CAPTION_SIZE), MUTED, CAPTION_LINE)
        self.els = [
            TextEl(num, MARGIN, lay["num_y"], 0.10, rise=10),
            TextEl(of, MARGIN + num.width + 4, lay["num_y"] + 3, 0.18, rise=10),
            BarEl(MARGIN + num.width + of.width + 18, lay["num_y"] + 15, 90, 2, ACCENT, 0.25),
            TextEl(brand, W - MARGIN - brand.width + 4, lay["num_y"] + 3, 0.18, rise=10),
            TextEl(text_image(lay["head"], font(HEAD_SIZE, True), TEXT_1, HEAD_LINE), MARGIN - 3, lay["head_y"], 0.32, rise=20),
            TextEl(text_image(lay["body"], font(BODY_SIZE), TEXT_2, BODY_LINE), MARGIN, lay["body_y"], 0.62),
        ]
        self.captions = []
        for k in sec.assets:
            a = ASSETS[k]
            lines = caption_lines(a)
            cap = text_image(lines, font(CAPTION_SIZE), TEXT_2, CAPTION_LINE)
            cred = text_image([a.credit], font(CAPTION_SIZE), MUTED, CAPTION_LINE)
            self.captions.append((cap, cred, len(lines)))
        self.cap_y = section_caption_y(sec)

    def frame(self, t: float) -> Image.Image:
        fr = Image.new("RGBA", (W, H), rgb(SURFACE) + (255,))
        img_in = reveal(t, 0.40, 0.9)
        ws = dip_weights(self.slots, t, DISSOLVE_S)
        # At most one image is visible (hand-overs dip through the surface); its caption and
        # credit fade with it, so a caption never outlives or precedes its own card.
        for i, w in enumerate(ws):
            a = w * img_in
            if a <= 0.003:
                continue
            region = self.layers[i].render((t - self.slots[i][0]) / (self.slots[i][1] - self.slots[i][0]))
            if a < 0.999:
                region = Image.blend(Image.new("RGB", region.size, rgb(SURFACE)), region, a)
            fr.paste(region, (self.ax, self.ay))
            cap, cred, n_cap = self.captions[i]
            fr.alpha_composite(with_alpha(cap, a), (CAPTION_X, self.cap_y))
            fr.alpha_composite(with_alpha(cred, a), (CAPTION_X, self.cap_y + CAPTION_LINE * n_cap + 1))
        for el in self.els:
            el.draw(fr, t)
        return fade_to_surface(fr, segment_fade(t, self.duration, 0.0, SEG_FADE_OUT_S)).convert("RGB")


# ================================================================== rendering
def build_scene(name: str, asset_dir: Path = ASSET_DIR):
    if name == "card":
        return card_scene(asset_dir)
    if name == "opening":
        return opening_scene(asset_dir)
    if name == "closing":
        return closing_scene(asset_dir)
    for s in SECTIONS:
        if name == f"section{s.number}":
            return SectionScene(s, asset_dir)
    raise KeyError(name)


def encoder(out_path: Path, faststart: bool = False) -> subprocess.Popen:
    cmd = [FFMPEG, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "16",
           "-pix_fmt", "yuv420p"]
    if faststart:
        cmd += ["-movflags", "+faststart"]
    return subprocess.Popen(cmd + [str(out_path)], stdin=subprocess.PIPE)


def render_scene(args: tuple[str, str, bool, str]) -> str:
    name, out_path, faststart, asset_dir = args
    scene = build_scene(name, Path(asset_dir))
    enc = encoder(Path(out_path), faststart)
    for i in range(frame_count(scene.duration)):
        enc.stdin.write(scene.frame(i / FPS).tobytes())
    enc.stdin.close()
    if enc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {name}")
    return out_path


def write_sources(path: Path = SOURCES_OUT) -> None:
    used = [k for s in SECTIONS for k in s.assets]
    bg = [k for k, _ in CARD_BG + CLOSING_BG]
    lines = ["# Raman Lab montage — image sources", "",
             "Generated by `scripts/analysis/make_lab_overview_montage.py`. Every image shown in",
             "`raman_lab_intro_card.mp4` and `raman_lab_research_overview_montage.mp4`, with the URL it",
             "was downloaded from and its citation. On-screen credits use the short form.", "",
             f"Accent colour: WashU red `{ACCENT}` (and link red `{ACCENT_LINK}` for section numerals), "
             "from https://ramanlab.wustl.edu/wp-content/themes/wutheme_chauvenet/style.css "
             "(`h2.is-style-underline::before{background:#a51417}`, pagination, `a:hover`; `a{color:#d01c29}`).", ""]
    for title, keys in (("Montage sections", used), ("Intro card / opening / closing backgrounds", bg)):
        lines += [f"## {title}", ""]
        for k in dict.fromkeys(keys):
            a = ASSETS[k]
            where = [s.heading for s in SECTIONS if k in s.assets]
            lines += [f"- **{a.caption}** ({', '.join(where) or 'background'})",
                      f"  - Image URL: {a.url}",
                      f"  - Found on: {a.page}",
                      f"  - On-screen credit: {a.credit}",
                      f"  - Citation: {a.citation}"]
        lines.append("")
    lines += ["## Not used", "",
              "- Electronic-nose papers listed on /research/ and /publications/ (Gupta et al. 2022 ACS AMI; "
              "Yilmaz et al. 2019 ACS ANM; Raman et al. 2009 Sens. Actuators B) are not open access, so "
              "the Artificial Olfaction section uses the lab website's own images only.",
              "- Gupta et al. 2024 Nature Nanotechnology (nano-neuromodulation) is not open access; skipped.", ""]
    path.write_text("\n".join(lines))


def render_preview(out_dir: Path, asset_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    card = card_scene(asset_dir)
    for label, t in (("early", 0.9), ("hero", 3.6), ("late", 6.5)):
        card.frame(t).save(out_dir / f"card_{label}.png")
    opening = opening_scene(asset_dir)
    opening.frame(3.0).save(out_dir / "m0_opening.png")
    for sec in SECTIONS:
        sc = SectionScene(sec, asset_dir)
        sc.frame(0.75).save(out_dir / f"m{sec.number}_text_reveal.png")
        for i, (s, e) in enumerate(sc.slots):
            sc.frame((s + e) / 2).save(out_dir / f"m{sec.number}_img{i + 1}_{sec.assets[i]}.png")
        s1 = sc.slots[1][0]
        sc.frame(s1 + DISSOLVE_S / 2).save(out_dir / f"m{sec.number}_dissolve.png")
    closing_scene(asset_dir).frame(3.5).save(out_dir / "m9_closing.png")
    print(f"previews -> {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preview", type=Path, metavar="DIR", help="write a PNG per segment instead of rendering")
    ap.add_argument("--assets", type=Path, default=ASSET_DIR, help="asset cache directory")
    ap.add_argument("--only", choices=["card", "montage"], help="render just one video")
    args = ap.parse_args()
    ensure_assets(args.assets)
    if args.preview:
        render_preview(args.preview, args.assets)
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        jobs = []
        if args.only in (None, "card"):
            jobs.append(("card", str(CARD_OUT), True, str(args.assets)))
        seg_paths = []
        if args.only in (None, "montage"):
            for seg in montage_timeline():
                p = str(Path(td) / f"{seg.name}.mp4")
                seg_paths.append(p)
                jobs.append((seg.name, p, False, str(args.assets)))
        with Pool(min(len(jobs), 8)) as pool:
            pool.map(render_scene, jobs)
        if seg_paths:
            lst = Path(td) / "list.txt"
            lst.write_text("".join(f"file '{p}'\n" for p in seg_paths))
            subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
                            "-c", "copy", "-movflags", "+faststart", str(MONTAGE_OUT)], check=True)
    if args.only in (None, "card"):
        card_scene(args.assets).frame(3.6).save(CARD_OUT.with_name(CARD_OUT.stem + "_poster.png"))
    if args.only in (None, "montage"):
        opening_scene(args.assets).frame(3.0).save(MONTAGE_OUT.with_name(MONTAGE_OUT.stem + "_poster.png"))
        write_sources()
    print("done")


if __name__ == "__main__":
    main()
