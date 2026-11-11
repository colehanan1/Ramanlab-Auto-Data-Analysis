"""Shared lab-montage visual language for the square fly-behaviour videos.

make_per_montage_video.py and make_repeated_exposure_montage_video.py use this so
they read as one set with make_lab_overview_montage.py: same tokens, typography,
numbered segment header, staged text reveal, caption style, hero title card and
fade-to-surface between segments.

The primitives (tokens, TextEl / BarEl, Motif, make_scrim, Ken Burns, easing)
are imported from make_lab_overview_montage.py unchanged, so that script's
output is untouched. This module only adds what the behaviour videos need on
top: a header built from arbitrary strings, italic / superscript-plus text runs
(Arial has no U+207A glyph), lab-style captions, a hero card whose background
is the fly's own footage, and a compositor that lays the PIL header over a
matplotlib content frame.

Pacing: the behaviour videos are read, not skimmed, so every staged reveal,
fade and drift runs PACE x slower than the lab montage (same order, same
easing), and every text element carries its words so tests can check that it
stays on screen long enough to read (READ_S_PER_WORD, HEADING_EXTRA_S).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import make_lab_overview_montage as lab  # noqa: E402

# ------------------------------------------------------------ re-exported lab tokens
W, H, FPS = lab.W, lab.H, lab.FPS
SURFACE, TEXT_1, TEXT_2, MUTED = lab.SURFACE, lab.TEXT_1, lab.TEXT_2, lab.MUTED
ACCENT, ACCENT_LINK = lab.ACCENT, lab.ACCENT_LINK      # rules / numerals and markers only, never data
MARGIN = lab.MARGIN
HEAD_SIZE, HEAD_LINE = lab.HEAD_SIZE, lab.HEAD_LINE
BODY_SIZE, BODY_LINE = lab.BODY_SIZE, lab.BODY_LINE
CAPTION_SIZE, CAPTION_LINE = lab.CAPTION_SIZE, lab.CAPTION_LINE
CAPTION_Y = lab.CAPTION_Y
SEG_FADE_OUT_S = lab.SEG_FADE_OUT_S
LAB_SHORT = lab.LAB_SHORT

TextEl, BarEl, Motif = lab.TextEl, lab.BarEl, lab.Motif
font, rgb, reveal, smoothstep = lab.font, lab.rgb, lab.reveal, lab.smoothstep
segment_fade, fade_to_surface, with_alpha = lab.segment_fade, lab.fade_to_surface, lab.with_alpha
ken_burns_rect, kb_params, make_scrim, wrap_text = lab.ken_burns_rect, lab.kb_params, lab.make_scrim, lab.wrap_text

SEP = "  ·  "                        # phrase separator
SUP_PLUS = "⁺"
FONT_ITALIC = lab.FONT_DIR / "Arial_Italic.ttf"
FONT_BOLD_ITALIC = lab.FONT_DIR / "Arial_Bold_Italic.ttf"

# ------------------------------------------------------------ pacing (lab timings x PACE)
PACE = 1.6                                 # every reveal / fade / stagger runs this much slower than the lab's
REVEAL_DUR = round(0.7 * PACE, 2)          # text element ease-in (lab TextEl.dur 0.7)
RULE_DUR = round(0.8 * PACE, 2)            # accent rule growth (lab BarEl.dur 0.8)
FADE_OUT_S = 0.85                          # segment fade to the surface (lab 0.55)
READ_S_PER_WORD = 0.30                     # a text element stays this long per word once revealed ...
HEADING_EXTRA_S = 1.0                      # ... plus this for a heading
MIN_TEXT_PX = lab.MIN_TEXT_PX              # legibility floor: >= 3 % of frame height (32 px); = the lab caption size
MIN_TEXT_PT = round(MIN_TEXT_PX * 72 / 100, 2)   # the same floor in matplotlib points at dpi 100 (23.04 pt)

# ------------------------------------------------------------ header system (lab SectionScene)
NUM_Y, HEAD_Y = 62, 112
HEADER_RULE_W = 90
LAB_HEADER_T0 = {"num": 0.10, "of": 0.18, "brand": 0.18, "rule": 0.25, "heading": 0.32, "body": 0.62}
HEADER_T0 = {k: round(v * PACE, 3) for k, v in LAB_HEADER_T0.items()}
CONTENT_T0, CONTENT_DUR = round(0.40 * PACE, 3), round(0.9 * PACE, 3)   # image area eases in under the header

# ------------------------------------------------------------ hero title card (lab opening)
HERO_X = 96
HERO_SPACING = round(0.72 * PACE, 3)       # lab opening's stagger, slowed
HERO_FADE_IN, HERO_FADE_OUT = round(0.7 * PACE, 2), FADE_OUT_S
HERO_HEAD_SIZE, HERO_HEAD_LINE = 60, 70
HERO_BODY_SIZE, HERO_BODY_LINE = 34, 48
KICKER_SIZE, KICKER_LINE = 32, 44

Run = tuple[str, bool]                     # (text, italic)


# ================================================================ pure helpers
def header_numerals(index: int, total: int) -> tuple[str, str]:
    """('0N', '/ 0M') for segment `index` of `total` (1-based)."""
    if total < 1 or not 1 <= index <= total:
        raise ValueError(f"segment {index} of {total}")
    return f"{index:02d}", f"/ {total:02d}"


# ================================================================ reading time
@dataclass(frozen=True)
class ReadItem:
    """A piece of on-screen text, the time it is fully revealed, and whether it is a heading."""
    text: str
    shown: float
    heading: bool = False


def word_count(text: str) -> int:
    """Words a viewer reads: whitespace tokens containing a letter or digit."""
    return sum(any(c.isalnum() for c in tok) for tok in text.split())


def required_hold(item: ReadItem) -> float:
    return READ_S_PER_WORD * word_count(item.text) + (HEADING_EXTRA_S if item.heading else 0.0)


@dataclass
class FlyInEl:
    """Element that slides in horizontally from `dx` px away while fading in (lab easing).
    Unlike TextEl it may start off-frame, so the paste is clipped to the frame."""
    img: Image.Image
    x: int
    y: int
    t0: float
    dur: float = 0.7
    dx: int = -120

    def draw(self, frame: Image.Image, t: float, alpha: float = 1.0) -> None:
        u = lab.reveal(t, self.t0, self.dur)
        a = u * alpha
        if a <= 0.003:
            return
        src = lab.with_alpha(self.img, a)
        x, y = self.x + int(round((1 - u) * self.dx)), self.y
        l, tp = max(0, -x), max(0, -y)
        r, b = min(src.width, frame.width - x), min(src.height, frame.height - y)
        if r <= l or b <= tp:
            return
        frame.alpha_composite(src.crop((l, tp, r, b)), (x + l, y + tp))


def el_box(el) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1) frame pixels an element covers once it has landed (TextEl / FlyInEl
    image box, BarEl rectangle): layout tests use it to check that nothing overlaps."""
    if isinstance(el, BarEl):
        return el.x, el.y, el.x + el.w, el.y + el.h
    return el.x, el.y, el.x + el.img.width, el.y + el.img.height


def tag(el, text: str, heading: bool = False):
    """Attach the words an element shows (PIL TextEl) so its reading time can be checked."""
    el.text, el.heading = text, heading
    return el


def read_items(els) -> list[ReadItem]:
    """ReadItems of every tagged element (shown = end of its reveal)."""
    return [ReadItem(e.text, e.t0 + e.dur, e.heading) for e in els if getattr(e, "text", None)]


def reading_shortfalls(items, duration: float, fade_out: float = FADE_OUT_S) -> list[tuple[ReadItem, float, float]]:
    """(item, hold, needed) for every item that leaves the screen (fade-out start) before it can be read."""
    end = duration - fade_out
    return [(it, end - it.shown, required_hold(it)) for it in items if end - it.shown < required_hold(it) - 1e-9]


def card_reading_time(items) -> float:
    """Seconds to read every element of a card (sum of the per-element holds)."""
    return sum(required_hold(it) for it in items)


def card_hold(items, duration: float, fade_out: float = FADE_OUT_S) -> float:
    """Seconds the whole card stays fully revealed: last element landed -> fade-out start."""
    return duration - fade_out - max(it.shown for it in items)


def card_reading_shortfall(items, duration: float, fade_out: float = FADE_OUT_S) -> float:
    """Seconds missing for a viewer to read the whole card once it is all up (0 when it fits).
    Stricter than reading_shortfalls: a hero card is read as one block, not element by element."""
    return max(0.0, card_reading_time(items) - card_hold(items, duration, fade_out))


def hero_reveal_times(n_kicker: int, n_items: int, sp: float = HERO_SPACING) -> dict:
    """Start times of the hero card's staggered reveal (lab card timings x sp)."""
    return {
        "rule": 0.30 * sp,
        "kicker": [(0.45 + 0.08 * i) * sp for i in range(n_kicker)],
        "headline": 0.70 * sp,
        "body": 1.00 * sp,
        "divider": 1.30 * sp,
        "items": [(1.50 + 0.2 * i) * sp for i in range(n_items)],
    }


# ================================================================ rich text (italic runs, superscript plus)
@lru_cache(maxsize=None)
def font_style(size: int, bold: bool = False, italic: bool = False) -> ImageFont.FreeTypeFont:
    if not italic:
        return font(size, bold)
    return ImageFont.truetype(str(FONT_BOLD_ITALIC if bold else FONT_ITALIC), size)


def _chunks(text: str):
    """Split on U+207A (missing from Arial): yields (chunk, is_superscript_plus)."""
    parts = text.split(SUP_PLUS)
    for i, p in enumerate(parts):
        if p:
            yield p, False
        if i < len(parts) - 1:
            yield "+", True


def run_measure(size: int, bold: bool = False) -> Callable[..., float]:
    """measure(text, italic=False) -> rendered width in px (superscript plus aware)."""
    sup = max(8, round(size * 0.62))

    def measure(text: str, italic: bool = False) -> float:
        return sum(font_style(sup if s else size, bold, italic).getlength(c) for c, s in _chunks(text))
    return measure


def as_lines(text_or_lines) -> list[list[Run]]:
    if isinstance(text_or_lines, str):
        return [[(text_or_lines, False)]]
    return [[(r, False)] if isinstance(r, str) else list(r) for r in text_or_lines]


def runs_image(lines: Sequence[Sequence[Run]], size: int, color: str, line_h: int,
               bold: bool = False) -> Image.Image:
    """RGBA text block like lab.text_image, but each line is a list of (text, italic) runs."""
    if size < MIN_TEXT_PX:
        raise ValueError(f"{size} px text is below the {MIN_TEXT_PX} px legibility floor")
    m = run_measure(size, bold)
    widths = [sum(m(t, it) for t, it in line) for line in lines]
    width = int(np.ceil(max(widths))) + 4
    height = int(line_h * (len(lines) - 1) + size * 1.3) + 2
    mask = Image.new("L", (width, height), 0)
    d = ImageDraw.Draw(mask)
    sup = max(8, round(size * 0.62))
    for i, line in enumerate(lines):
        x = 0.0
        for text, italic in line:
            for chunk, is_sup in _chunks(text):
                f = font_style(sup if is_sup else size, bold, italic)
                d.text((x, i * line_h), chunk, font=f, fill=255, anchor="la")
                x += f.getlength(chunk)
    layer = Image.new("RGBA", (width, height), rgb(color) + (255,))
    layer.putalpha(mask)
    return layer


def wrap_balanced(text: str, max_w: float, measure: Callable[[str], float]) -> list[str]:
    """Greedy line count, line lengths evened out (for headlines)."""
    lines = wrap_text(text, max_w, measure)
    if len(lines) < 2:
        return lines
    lo, hi = max(measure(w) for w in text.split()), max_w
    for _ in range(24):
        mid = (lo + hi) / 2
        if len(wrap_text(text, mid, measure)) <= len(lines):
            hi = mid
        else:
            lo = mid
    return wrap_text(text, hi, measure)


# ================================================================ segment header
class SegmentHeader:
    """Lab section header: '0N / 0M' + red rule, 'Raman Lab' at the right, bold
    heading and TEXT_2 body, revealed in the lab's order (PACE x slower). `body`
    is a string (word-wrapped like the lab), pre-wrapped lines of (text, italic)
    runs, or None for a heading-only header."""

    def __init__(self, index: int, total: int, heading: str, body=None, width: float | None = None):
        width = width or W - 2 * MARGIN
        hm, bm = run_measure(HEAD_SIZE, True), run_measure(BODY_SIZE)
        self.head_lines = [ln for para in heading.split("\n") for ln in wrap_text(para, width, hm)]
        if not body:
            body_runs = []
        elif isinstance(body, str):
            body_runs = [[(ln, False)] for ln in wrap_text(body, width, bm)]
        else:
            body_runs = as_lines(body)
        self.body_lines = ["".join(t for t, _ in ln) for ln in body_runs]
        self.head_y = HEAD_Y
        self.body_y = HEAD_Y + HEAD_LINE * len(self.head_lines) + 14
        if body_runs:
            self.bottom = self.body_y + BODY_LINE * (len(body_runs) - 1) + int(BODY_SIZE * 1.3)
        else:
            self.bottom = HEAD_Y + HEAD_LINE * (len(self.head_lines) - 1) + int(HEAD_SIZE * 1.3)
        self.head_width = max(hm(ln) for ln in self.head_lines)

        n, of = header_numerals(index, total)
        num = runs_image([[(n, False)]], lab.NUMERAL_SIZE, ACCENT_LINK, lab.NUMERAL_SIZE + 4, bold=True)
        ofi = runs_image([[(of, False)]], CAPTION_SIZE, MUTED, CAPTION_LINE)
        brand = runs_image([[(LAB_SHORT, False)]], CAPTION_SIZE, MUTED, CAPTION_LINE)
        t0, d = HEADER_T0, REVEAL_DUR
        self.numeral_el = TextEl(num, MARGIN, NUM_Y, t0["num"], d, rise=10)
        self.of_el = TextEl(ofi, MARGIN + num.width + 4, NUM_Y + 3, t0["of"], d, rise=10)
        self.rule_el = BarEl(MARGIN + num.width + ofi.width + 18, NUM_Y + 15, HEADER_RULE_W, 2, ACCENT, t0["rule"],
                             RULE_DUR)
        self.brand_el = TextEl(brand, W - MARGIN - brand.width + 4, NUM_Y + 3, t0["brand"], d, rise=10)
        self.heading_el = tag(TextEl(runs_image(as_lines(self.head_lines), HEAD_SIZE, TEXT_1, HEAD_LINE, bold=True),
                                     MARGIN - 3, self.head_y, t0["heading"], d, rise=20),
                              " ".join(self.head_lines), heading=True)
        self.els: list = [self.numeral_el, self.of_el, self.rule_el, self.brand_el, self.heading_el]
        self.body_el = None
        if body_runs:
            self.body_el = tag(TextEl(runs_image(body_runs, BODY_SIZE, TEXT_2, BODY_LINE), MARGIN, self.body_y,
                                      t0["body"], d), " ".join(self.body_lines))
            self.els.append(self.body_el)

    def add_beside_heading(self, img: Image.Image, gap: int = 26, text: str = "") -> TextEl:
        """An element (e.g. a badge) after the heading's first line, revealed with it."""
        cap_mid = self.head_y + round(HEAD_SIZE * 0.55)
        el = TextEl(img, int(MARGIN - 3 + self.head_width + gap), cap_mid - img.height // 2,
                    HEADER_T0["heading"], REVEAL_DUR, rise=20)
        if text:
            tag(el, text)
        self.els.append(el)
        return el

    def read_items(self) -> list[ReadItem]:
        return read_items(self.els)

    def draw(self, frame: Image.Image, t: float, alpha: float = 1.0) -> None:
        for el in self.els:
            el.draw(frame, t, alpha)


def outline_badge(text: str, size: int = CAPTION_SIZE, color: str = TEXT_1, pad_x: int = 14, pad_y: int = 8) -> Image.Image:
    """Rounded outline pill with bold text (e.g. '▲ Increased Response')."""
    label = runs_image([[(text, False)]], size, color, size, bold=True)
    w, h = label.width + 2 * pad_x, int(size * 1.15) + 2 * pad_y
    ss = 4
    big = Image.new("L", (w * ss, h * ss), 0)
    ImageDraw.Draw(big).rounded_rectangle((ss, ss, w * ss - ss - 1, h * ss - ss - 1), radius=h * ss // 2,
                                          outline=255, width=2 * ss)
    out = Image.new("RGBA", (w, h), rgb(color) + (0,))
    out.putalpha(big.resize((w, h), Image.LANCZOS))
    out.alpha_composite(label, (pad_x, pad_y - round(size * 0.12)))
    return out


# ================================================================ captions
class Caption:
    """Lab caption: TEXT_2 description line(s), then MUTED provenance line(s)."""

    def __init__(self, desc_lines, prov_lines, x: int, y: int,
                 size: int = CAPTION_SIZE, line: int = CAPTION_LINE, gap: int = 1):
        desc, prov = as_lines(desc_lines) if desc_lines else [], as_lines(prov_lines) if prov_lines else []
        self.x, self.y = x, y
        self.n_desc, self.n_prov = len(desc), len(prov)
        self.desc_img = runs_image(desc, size, TEXT_2, line) if desc else None
        self.prov_y = y + line * len(desc) + (gap if desc else 0)
        self.prov_img = runs_image(prov, size, MUTED, line) if prov else None
        self.bottom = self.prov_y + line * max(len(prov), 0) if prov else y + line * len(desc)

    def draw(self, frame: Image.Image, alpha: float = 1.0) -> None:
        if alpha <= 0.003:
            return
        if self.desc_img is not None:
            frame.alpha_composite(with_alpha(self.desc_img, alpha), (self.x, self.y))
        if self.prov_img is not None:
            frame.alpha_composite(with_alpha(self.prov_img, alpha), (self.x, self.prov_y))


# ================================================================ compositing
def compose(content: np.ndarray, t: float, duration: float, els: Sequence = (),
            captions: Sequence[Caption] = (), reveal_content: bool = True) -> np.ndarray:
    """Matplotlib content frame (RGB/RGBA array) + PIL header/captions -> RGBA array.

    The content eases in under the header (lab image-area reveal) unless the
    content animates itself in; the whole frame fades to the surface over the
    last FADE_OUT_S, like every lab montage segment (slowed)."""
    fr = Image.fromarray(np.ascontiguousarray(content[..., :3])).convert("RGBA")
    a = reveal(t, CONTENT_T0, CONTENT_DUR)        # captions always ease in with the content slot
    if reveal_content and a < 0.999:
        fr = Image.blend(Image.new("RGBA", fr.size, rgb(SURFACE) + (255,)), fr, a)
    for c in captions:
        c.draw(fr, a)
    for el in els:
        el.draw(fr, t)
    fr = fade_to_surface(fr, segment_fade(t, duration, 0.0, FADE_OUT_S))
    return np.asarray(fr)


# ================================================================ hero title card
def key_swatch(kind: str, color: str) -> Image.Image:
    """Data-key swatch in its data colour: 'line' (26x5 bar) or 'ring' (16 px outline)."""
    if kind == "line":
        return Image.new("RGBA", (26, 5), rgb(color) + (255,))
    ss, d = 4, 18
    big = Image.new("L", (d * ss, d * ss), 0)
    ImageDraw.Draw(big).ellipse((ss, ss, d * ss - ss, d * ss - ss), outline=255, width=int(2.2 * ss))
    out = Image.new("RGBA", (d, d), rgb(color) + (0,))
    out.putalpha(big.resize((d, d), Image.LANCZOS))
    return out


def title_els(kicker_lines: Sequence[Sequence[Run]], headline: str, body_paras: Sequence[str],
              keys: Sequence[tuple[str, str, str]], x: int = HERO_X, sp: float = HERO_SPACING) -> list:
    """Left-aligned hero stack (lab card): short red rule, TEXT_2 kicker line(s), bold
    headline, body, thin muted rule, key items -- vertically centred, staggered."""
    max_w = W - 2 * x
    tt = hero_reveal_times(len(kicker_lines), len(keys), sp)
    head = wrap_balanced(headline, max_w, run_measure(HERO_HEAD_SIZE, True))
    bm = run_measure(HERO_BODY_SIZE)
    paras = [wrap_text(p, max_w, bm) for p in body_paras]
    # vertical stack, relative to the top rule
    k_top = 28
    h_top = k_top + KICKER_LINE * (len(kicker_lines) - 1) + 50
    b_top = h_top + HERO_HEAD_LINE * len(head) + 24
    para_gap = 14
    b_tops, y = [], b_top
    for p in paras:
        b_tops.append(y)
        y += HERO_BODY_LINE * len(p) + para_gap
    d_top = y - para_gap - HERO_BODY_LINE + int(HERO_BODY_SIZE * 1.3) + 34
    i_tops = [d_top + 30 + i * 44 for i in range(len(keys))]
    total = (i_tops[-1] + 36) if keys else d_top + 1
    y0 = int(round((H - total) / 2))

    d = REVEAL_DUR
    els: list = [BarEl(x, y0, 64, 6, ACCENT, tt["rule"], RULE_DUR)]
    for i, ln in enumerate(kicker_lines):
        els.append(tag(TextEl(runs_image([ln], KICKER_SIZE, TEXT_2, KICKER_LINE), x, y0 + k_top + i * KICKER_LINE,
                              tt["kicker"][i], d), "".join(t for t, _ in ln)))
    els.append(tag(TextEl(runs_image(as_lines(head), HERO_HEAD_SIZE, TEXT_1, HERO_HEAD_LINE, bold=True),
                          x - 3, y0 + h_top, tt["headline"], d, rise=22), headline, heading=True))
    for p, bt, txt in zip(paras, b_tops, body_paras):
        els.append(tag(TextEl(runs_image(as_lines(p), HERO_BODY_SIZE, TEXT_1, HERO_BODY_LINE), x, y0 + bt,
                              tt["body"], d), txt))
    els.append(BarEl(x, y0 + d_top, 300, 1, MUTED, tt["divider"], RULE_DUR))
    for (kind, color, label), it, t0 in zip(keys, i_tops, tt["items"]):
        sw = key_swatch(kind, color)
        els.append(TextEl(sw, x, y0 + it + 17 - sw.height // 2, t0, d, rise=10))
        els.append(tag(TextEl(runs_image([[(label, False)]], KICKER_SIZE, TEXT_2, KICKER_LINE), x + 40, y0 + it, t0,
                              d, rise=10), label))
    return els


class FootageHero:
    """Hero card over the fly's own footage: luma frames drifting (Ken Burns) behind
    the lab scrim, the odor-plume motif, and staggered text elements."""

    def __init__(self, frames: np.ndarray, duration: float, els: Sequence,
                 kb=((1.0, 0.5, 0.5), (1.12, 0.5, 0.5)), src_fps: float = FPS,
                 motif_alpha: float = 0.8, fade_in: float = HERO_FADE_IN, fade_out: float = HERO_FADE_OUT):
        self.frames, self.duration, self.els, self.kb = frames, duration, list(els), kb
        self.src_fps, self.motif_alpha = src_fps, motif_alpha
        self.fade_in, self.fade_out = fade_in, fade_out
        self.scrim = make_scrim()
        self.motif = Motif()

    def background(self, t: float) -> Image.Image:
        i = min(int(round(t * self.src_fps)), len(self.frames) - 1)
        src = Image.fromarray(np.ascontiguousarray(self.frames[i])).convert("RGB")
        z, cx, cy = kb_params(self.kb, t / self.duration)
        box = ken_burns_rect(src.width, src.height, 1.0, z, cx, cy)
        out = src.resize((W, H), Image.BICUBIC, box=box).convert("RGBA")
        out.alpha_composite(self.scrim)
        return out

    def frame(self, t: float) -> Image.Image:
        fr = self.background(t)
        self.motif.draw(fr, t / PACE, alpha=smoothstep((t - 0.2 * PACE) / (1.2 * PACE)) * self.motif_alpha)  # slower drift
        for el in self.els:
            el.draw(fr, t)
        return fade_to_surface(fr, segment_fade(t, self.duration, self.fade_in, self.fade_out)).convert("RGB")
