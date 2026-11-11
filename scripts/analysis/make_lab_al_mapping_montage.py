#!/usr/bin/env python3
"""Lab presentation montage: mapping the circuitry of the locust antennal lobe (1080x1080, H.264).

Standalone square video in the lab montage style (make_lab_overview_montage.py tokens and
hero/closing layouts, lab_video_style.py section headers, reading-time rules, PACE):

  title       hero card over a dimmed, slowly rotating projection-neuron fill
  01 / 04     optical z-series through one antennal lobe (slice movie, real time)
  02 / 04     two projection-neuron fills, rotating side by side
  03 / 04     local-neuron fill, rotating, with a re-drawn legible scale bar
  04 / 04     research objective: one statement flying in at 54 px -- off (INCLUDE_OBJECTIVE =
              False, user 2026-11-02), so the headers read 01 / 03 .. 03 / 03
  closing     lab closing card (lab name, university, URL) over a rotating PN fill; the video
              ends here

Source movies are MJPEG AVIs the user downloaded. They are copied once into ASSET_DIR (a new
folder next to the lab montage's asset cache) and only ever decoded as images with OpenCV.

Brightness: every movie gets a LINEAR black/white-point lift, out = clip((x - black) * 255 /
(white - black), 0, 255), the same map on all three channels. No gamma or curve, so no faint
structure is invented; values above `white` clip. The slice movie needs the strongest lift (its
background sits near 6/255 and its tissue near 15-20/255).

Rotation movies are 7 fps; at 40 fps each output frame is a linear cross-fade of the two nearest
source frames (blend_frames). No optical-flow / motion interpolation, so no warping artefacts.

No segment ends on a pause: the slice keeps stepping in depth, the rotations (hero and closing
backgrounds included) keep turning and the closing push-in keeps zooming through every fade-out, so motion carries each segment into the next.
Only text holds still once revealed (reading time: READ_S_PER_WORD per word, + HEADING_EXTRA_S).

    python3 scripts/analysis/make_lab_al_mapping_montage.py            # render the video
    python3 scripts/analysis/make_lab_al_mapping_montage.py --preview DIR
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import make_lab_overview_montage as lab  # noqa: E402  (read-only: never edited from here)
import lab_video_style as style  # noqa: E402

# ------------------------------------------------------------------ format + lab tokens
W, H, FPS = lab.W, lab.H, lab.FPS
FFMPEG = lab.FFMPEG
SURFACE, TEXT_1, TEXT_2, MUTED = lab.SURFACE, lab.TEXT_1, lab.TEXT_2, lab.MUTED
ACCENT, ACCENT_LINK = lab.ACCENT, lab.ACCENT_LINK
CARD_DARK = lab.CARD_DARK
MARGIN = lab.MARGIN
rgb = lab.rgb

ASSET_DIR = lab.ASSET_DIR.parent / "al_mapping"          # copies of the downloaded movies
DOWNLOAD_DIR = Path("/home/ramanlab/Downloads")
OUT_PATH = lab.OUT_DIR / "raman_lab_antennal_lobe_mapping_montage.mp4"

# ================================================================== COPY (all on-screen text)
# Fact-checked wording (coordinator, 2026-10-30). No neuron or glomerulus counts anywhere.
# Formal academic register (user, 2026-10-30): always "the antennal lobe", never the bare "lobe".
COPY = {
    "kicker": "Raman Lab",
    "headline": "Mapping the Locust Antennal Lobe",
    "title_line": ("Is the spatial arrangement of antennal lobe glomeruli organized by odor chemical "
                   "identity or by odor valence?"),
    "sections_heading": "Optical Sections",
    "sections_body": ("Serial confocal sections through the antennal lobe reveal the glomeruli innervated "
                      "by a single filled neuron."),
    "sections_caption": "Serial optical sections through the antennal lobe",
    "projection_heading": "Projection Neurons",
    "projection_body": ("Projection neurons innervate multiple glomeruli and send axons to the mushroom "
                        "body calyces and the lateral protocerebrum."),
    "projection_caption": "Rotating 3D projections of filled projection neurons",
    "local_heading": "Local Neurons",
    "local_body": ("Local neurons arborize extensively throughout the antennal lobe, interconnecting "
                   "many glomeruli."),
    "local_caption": "Rotating 3D projection of a filled local neuron",
    "objective_heading": "Research Objective",
    "objective": ("Map the glomerular innervation of individual antennal lobe neurons and determine "
                  "whether its spatial organization reflects odor chemical identity or behavioral valence."),
    "scale": "100 µm",
}
# Extra patterns that must never reach the screen (people, specimen / file IDs, species).
FORBIDDEN_PATTERNS = [r"drosophila", r"melanogaster", r"projections of composite", r"\.avi", r"\b20x\b",
                      r"\b40x\b"]

# ================================================================== cell types (swap here)
# PN / LN assignment is inferred from morphology (user: "AL LNs and PNs in movie one or other"):
#   PN = soma outside the antennal lobe, a few discrete glomerular tufts, an axon leaving toward the calyx;
#   LN = soma at the antennal lobe edge, branches spread across most of the antennal lobe, no axon out.
# To swap, change `cell_type` below; each section picks its movies by cell type.
CELL_TYPES = {"PN": "projection neuron", "LN": "local neuron", "fill": "filled neuron (type not labelled)"}


@dataclass(frozen=True)
class Movie:
    file: str                       # name inside ASSET_DIR
    download: str                   # original name in DOWNLOAD_DIR (copied once, never read in place)
    cell_type: str                  # key of CELL_TYPES
    n_frames: int
    src_fps: float
    size: tuple[int, int]           # source (w, h)
    crop: tuple[int, int, int, int]  # source px (l, t, r, b)
    black: int                      # linear lift: black point ...
    white: int                      # ... and white point (0-255)
    start_frame: float = 0.0        # rotation movies: source frame (turn angle) when playback starts
    order: int = 0                  # left -> right within a section
    use: bool = True


MOVIES: dict[str, Movie] = {
    # Optical z-series, one antennal lobe. Very dark (background ~6, tissue 15-20, tufts 60-120 of 255).
    "slice": Movie("slice.avi", "slicemov.avi", "fill", 457, 25.0, (970, 949), (0, 273, 970, 949), 6, 64),
    # 3D rotation movies (120 frames = one full turn; frame 119 -> 0 is an ordinary step).
    "pn_a": Movie("pn_a.avi", "Projections of Composite_20x (2).avi", "PN", 120, 7.0, (1072, 1447),
                  (0, 0, 1072, 1447), 8, 200, start_frame=56.65, order=0),
    "pn_b": Movie("pn_b.avi", "Projections of Composite_20x (1).avi", "PN", 120, 7.0, (1066, 1435),
                  (0, 0, 1066, 1435), 6, 150, start_frame=101.65, order=1),
    # 40x: burned-in L scale bar + tiny labels occupy rows 753-946; the crop stops above them.
    "ln_a": Movie("ln_a.avi", "Projections of Composite_40x.avi", "LN", 120, 7.0, (1008, 965),
                  (0, 40, 1008, 745), 10, 200, start_frame=112.15),
    # Lower-resolution LN-like fill: not used (one LN movie keeps the section calm and short).
    "ln_b": Movie("ln_b.avi", "Projections of Composite_20x.avi", "LN", 120, 7.0, (650, 512),
                  (0, 0, 650, 512), 10, 200, use=False),
}


def _by_type(cell_type: str, rotating: bool = True) -> tuple[str, ...]:
    keys = [k for k, mv in MOVIES.items() if mv.cell_type == cell_type and mv.use and (mv.src_fps < 20) == rotating]
    return tuple(sorted(keys, key=lambda k: MOVIES[k].order))


SECTION_MEDIA = {"sections": ("slice",), "projection": _by_type("PN"), "local": _by_type("LN")}
TITLE_MOVIE = SECTION_MEDIA["projection"][0]           # hero background
CLOSING_MOVIE = SECTION_MEDIA["projection"][-1]        # slowly rotating behind the closing card

# Scale bar, measured in the 40x source: the horizontal and vertical arms of the burned-in L are
# both 176 px (cols 796-971, rows 753-928) and labelled 100 um. Re-drawn at the display scale.
SCALE_BAR_SRC_PX, SCALE_BAR_UM = 176, 100
SCALE_BAR_MOVIE = "ln_a"
SCALE_BAR_THICK = 4
SCALE_BAR_INSET = 22                  # px from the card's bottom-right corner (micrograph convention)

# ================================================================== timing
# No freeze anywhere: footage keeps moving until the last frame of each fade-out.
FADE_OUT_S = style.FADE_OUT_S        # 0.85 (lab pacing x PACE)
PLAY_T0 = 1.4                        # media starts moving once the image area has mostly eased in
TITLE_S = 9.5
TITLE_SPEED = 0.6                    # hero rotation, slowed
TITLE_START_FRAME = 88.71            # hero rotation angle at t = 0 (the approved opening view)
CLOSING_START_FRAME = 45.0           # closing rotation starts on the angle of the former closing still
PROJECTION_S = 12.5
LOCAL_S = 11.0
OBJECTIVE_READ_MARGIN_S = 1.0        # extra hold on the objective statement beyond its word-count time
INCLUDE_OBJECTIVE = False            # 04 objective slide removed (user 2026-11-02)
INCLUDE_CLOSING = True               # the video ends on the lab closing card
CLOSING_S = 6.0
CLOSING_FADE_OUT = 0.9               # lab closing scene
CLOSING_ZOOM = 0.05                  # closing push-in: 1.0 -> 1.05 over the whole segment, never stops
SLICE_START = 90.0                   # first frame shown: just before the first tuft (~frame 95)
SLICE_SPEED = 1.0                    # real time (25 planes/s); every plane is shown, the last one at the
                                     # segment's very end, so the slice is still stepping during the fade
ROT_SPEED = 1.0                      # 7 source frames/s = 21 deg/s
TITLE_SHIFT = -90                    # hero text block sits above the neuron in the background
TITLE_MOTIF_ALPHA = 0.6
STATEMENT_T, STATEMENT_DUR, STATEMENT_DX = 1.2, 1.1, -160   # IAA montage conclusion timings
STATEMENT_SIZE, STATEMENT_LINE = 54, 70

# ================================================================== layout
AREA_X, AREA_W = lab.AREA_X, lab.AREA_W
AREA_GAP, CAPTION_GAP = lab.AREA_GAP, lab.CAPTION_GAP
SAFE_BOTTOM = lab.SAFE_BOTTOM
CAPTION_SIZE, CAPTION_LINE = style.CAPTION_SIZE, style.CAPTION_LINE
CAPTION_Y = H - SAFE_BOTTOM - lab.caption_block_h(1)    # one caption line, ends 40 px above the edge
PANEL_GAP = 24
CARD_RADIUS = lab.CARD_RADIUS
TITLE_CROP = (0, 300, 1072, 1372)                         # square of the PN movie: neuron in the lower half
SECTION_INDEX = {"sections": 1, "projection": 2, "local": 3, "objective": 4}
N_NUMBERED = len(SECTION_INDEX) - (0 if INCLUDE_OBJECTIVE else 1)   # 01 / 03 .. 03 / 03 without it


# ================================================================== pure helpers
def lift(arr: np.ndarray, black: float, white: float) -> np.ndarray:
    """Linear brightness/contrast lift, clipped: (x - black) * 255 / (white - black) -> uint8.
    The same affine map for every channel; no gamma, so nothing faint is invented."""
    if not white > black:
        raise ValueError(f"white point {white} must exceed black point {black}")
    out = (np.asarray(arr, dtype=np.float32) - np.float32(black)) * np.float32(255.0 / (white - black))
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def blend_frames(frames: np.ndarray, pos: float, loop: bool) -> np.ndarray:
    """Linear cross-fade between the two source frames around fractional index `pos`.
    Integer positions return the source frame unchanged; `loop` wraps the last frame to the first,
    otherwise positions past the end hold the last frame."""
    n = len(frames)
    if loop:
        pos = pos % n
    else:
        pos = min(max(pos, 0.0), n - 1.0)
    i0 = int(math.floor(pos))
    w = pos - i0
    if w < 1e-9:
        return frames[i0 % n].copy()
    i1 = (i0 + 1) % n if loop else min(i0 + 1, n - 1)
    out = frames[i0].astype(np.float32) * (1.0 - w) + frames[i1].astype(np.float32) * w
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


@dataclass(frozen=True)
class Clip:
    key: str
    t0: float            # segment time playback starts
    start: float         # source frame at t0
    speed: float
    src_fps: float
    n_src: int
    loop: bool


def clip_pos(clip: Clip, t: float) -> float:
    """Fractional source frame at segment time t (held at `start` until t0, then plays on)."""
    el = max(0.0, t - clip.t0)
    pos = clip.start + el * clip.src_fps * clip.speed
    return pos % clip.n_src if clip.loop else min(pos, clip.n_src - 1.0)


@dataclass(frozen=True)
class Seg:
    name: str
    duration: float
    fade_out: float


def slice_duration() -> float:
    """The slice reaches its last plane exactly at the segment end: every rendered frame, the fade
    included, shows a plane strictly deeper than the frame before."""
    mv = MOVIES["slice"]
    play = (mv.n_frames - 1 - SLICE_START) / (mv.src_fps * SLICE_SPEED)
    return round(PLAY_T0 + play, 4)


def objective_duration() -> float:
    """Statement lands, is held for its word count (+ margin), then fades: grows with the copy."""
    shown = STATEMENT_T + STATEMENT_DUR
    hold = style.required_hold(style.ReadItem(COPY["objective"], shown))
    return math.ceil((shown + hold + OBJECTIVE_READ_MARGIN_S + FADE_OUT_S) * 10 - 1e-6) / 10


def _all_segments() -> list[Seg]:
    return [Seg("title", TITLE_S, style.HERO_FADE_OUT),
            Seg("sections", slice_duration(), FADE_OUT_S),
            Seg("projection", PROJECTION_S, FADE_OUT_S),
            Seg("local", LOCAL_S, FADE_OUT_S),
            Seg("objective", objective_duration(), FADE_OUT_S),
            Seg("closing", CLOSING_S, CLOSING_FADE_OUT)]


def segments() -> list[Seg]:
    """Segments rendered into the video (objective / closing only when switched on)."""
    off = {n for n, on in (("objective", INCLUDE_OBJECTIVE), ("closing", INCLUDE_CLOSING)) if not on}
    return [s for s in _all_segments() if s.name not in off]


def seg(name: str) -> Seg:
    return next(s for s in _all_segments() if s.name == name)


def fade_start(name: str) -> float:
    s = seg(name)
    return s.duration - s.fade_out


def frame_count(duration: float) -> int:
    return lab.frame_count(duration)


def _loop_clip(key: str, t0: float, speed: float, start: float | None = None) -> Clip:
    mv = MOVIES[key]
    start = mv.start_frame if start is None else start
    return Clip(key, t0, start % mv.n_frames, speed, mv.src_fps, mv.n_frames, True)


def segment_clips(name: str) -> list[Clip]:
    if name == "title":
        return [_loop_clip(TITLE_MOVIE, 0.0, TITLE_SPEED, TITLE_START_FRAME)]
    if name == "sections":
        mv = MOVIES["slice"]
        return [Clip("slice", PLAY_T0, SLICE_START, SLICE_SPEED, mv.src_fps, mv.n_frames, False)]
    if name in ("projection", "local"):
        return [_loop_clip(k, PLAY_T0, ROT_SPEED) for k in SECTION_MEDIA[name]]
    if name == "closing":
        return [_loop_clip(CLOSING_MOVIE, 0.0, TITLE_SPEED, CLOSING_START_FRAME)]
    return []


def closing_zoom(t: float) -> float:
    """Closing push-in: grows linearly over the whole segment (fade-out included), never stops."""
    return 1.0 + CLOSING_ZOOM * max(t, 0.0) / seg("closing").duration


def crop_size(mv: Movie) -> tuple[int, int]:
    l, t, r, b = mv.crop
    return r - l, b - t


def display_scale(mv: Movie, box: tuple[int, int, int, int]) -> float:
    return box[2] / crop_size(mv)[0]


def scale_bar_px(scale: float) -> int:
    return int(round(SCALE_BAR_SRC_PX * scale))


# ================================================================== text elements (pure)
def mark(el, label: str, px: int, role: str = "text", read: bool = True):
    """Record what an element shows and at what size; `read` also tags it for reading time."""
    el.label, el.px, el.role = label, px, role
    if read:
        style.tag(el, label, heading=(role == "heading"))
    return el


def title_text_els() -> list:
    els = style.title_els([[(COPY["kicker"], False)]], COPY["headline"], [COPY["title_line"]], keys=[])
    sizes = {COPY["kicker"]: (style.KICKER_SIZE, "text"), COPY["headline"]: (style.HERO_HEAD_SIZE, "heading"),
             COPY["title_line"]: (style.HERO_BODY_SIZE, "text")}
    for el in els:
        el.y += TITLE_SHIFT
        if getattr(el, "text", None):
            px, role = sizes[el.text]
            mark(el, el.text, px, role)
    return els


def section_header(index: int) -> style.SegmentHeader:
    name = next(k for k, v in SECTION_INDEX.items() if v == index)
    body = COPY.get(f"{name}_body")
    total = max(N_NUMBERED, index)          # the (switched-off) objective still builds as 04 / 04
    hdr = style.SegmentHeader(index, total, COPY[f"{name}_heading"], body)
    num, of = style.header_numerals(index, total)
    mark(hdr.numeral_el, num, lab.NUMERAL_SIZE, read=False)
    mark(hdr.of_el, of, CAPTION_SIZE, read=False)
    mark(hdr.brand_el, style.LAB_SHORT, CAPTION_SIZE, read=False)
    mark(hdr.heading_el, COPY[f"{name}_heading"], style.HEAD_SIZE, "heading")
    if hdr.body_el is not None:
        mark(hdr.body_el, body, style.BODY_SIZE)
    return hdr


def statement_els(hdr: style.SegmentHeader) -> list:
    """The key statement, balanced over the text width, flying in from the left (IAA conclusion)."""
    text = COPY["objective"]
    width = W - 2 * MARGIN
    lines = style.wrap_balanced(text, width, style.run_measure(STATEMENT_SIZE))
    img = style.runs_image(style.as_lines(lines), STATEMENT_SIZE, TEXT_1, STATEMENT_LINE)
    y = max(hdr.bottom + 80, int(round(H / 2 - img.height / 2)))
    el = style.FlyInEl(img, MARGIN, y, STATEMENT_T, STATEMENT_DUR, STATEMENT_DX)
    return [mark(el, text, STATEMENT_SIZE, "statement")]


def closing_text_els() -> list:
    """Lab closing card (make_lab_overview_montage closing layout, lab timings)."""
    els = lab.hero_layout("closing")
    z = lab.HERO_SIZES["closing"]
    labels = [(lab.LAB_SHORT, z["lab_short"], "heading"), (lab.LAB_FULL, z["lab_full"], "text"),
              (lab.UNIVERSITY, z["university"], "text"), (lab.URL, z["url"], "text")]
    texts = [el for el in els if isinstance(el, lab.TextEl)]
    assert len(texts) == len(labels)
    for el, (label, px, role) in zip(texts, labels):
        mark(el, label, px, role)
    return els


def section_area(name: str) -> tuple[int, int, int, int]:
    top = section_header(SECTION_INDEX[name]).bottom + AREA_GAP
    return (AREA_X, top, AREA_W, CAPTION_Y - CAPTION_GAP - top)


def media_boxes(name: str) -> list[tuple[int, int, int, int]]:
    """(x, y, w, h) of each movie card: contain-fit, side by side, centred in the image area."""
    ax, ay, aw, ah = section_area(name)
    keys = SECTION_MEDIA[name]
    n = len(keys)
    slot_w = (aw - PANEL_GAP * (n - 1)) / n
    sizes = []
    for k in keys:
        cw, ch = crop_size(MOVIES[k])
        s = min(slot_w / cw, ah / ch)
        sizes.append((int(round(cw * s)), int(round(ch * s))))
    total = sum(w for w, _ in sizes) + PANEL_GAP * (n - 1)
    x = ax + (aw - total) // 2
    boxes = []
    for w, h in sizes:
        boxes.append((x, ay + (ah - h) // 2, w, h))
        x += w + PANEL_GAP
    return boxes


def caption_el(name: str):
    img = style.runs_image([[(COPY[f"{name}_caption"], False)]], CAPTION_SIZE, TEXT_2, CAPTION_LINE)
    el = lab.TextEl(img, MARGIN, CAPTION_Y, style.CONTENT_T0, style.CONTENT_DUR, rise=0)
    return mark(el, COPY[f"{name}_caption"], CAPTION_SIZE)


def scale_bar_el():
    """Legible '100 um' label + bar at the 40x movie's display scale, inset in the card's bottom-right
    corner (standard micrograph placement: the formal local-neuron caption leaves no room for the
    bar on the caption row). Revealed with the card."""
    mv = MOVIES[SCALE_BAR_MOVIE]
    box = media_boxes("local")[SECTION_MEDIA["local"].index(SCALE_BAR_MOVIE)]
    bar = scale_bar_px(display_scale(mv, box))
    label = style.runs_image([[(COPY["scale"], False)]], CAPTION_SIZE, TEXT_1, CAPTION_LINE)
    gap = 18
    img = Image.new("RGBA", (label.width + gap + bar, label.height), rgb(TEXT_1) + (0,))
    img.alpha_composite(label, (0, 0))
    by = round(CAPTION_SIZE * 0.62) - SCALE_BAR_THICK // 2          # mid cap-height of the label
    ImageDraw.Draw(img).rectangle((label.width + gap, by, label.width + gap + bar - 1, by + SCALE_BAR_THICK - 1),
                                  fill=rgb(TEXT_1) + (255,))
    x = box[0] + box[2] - img.width - SCALE_BAR_INSET
    y = box[1] + box[3] - img.height - SCALE_BAR_INSET
    el = lab.TextEl(img, x, y, style.CONTENT_T0, style.CONTENT_DUR, rise=0)
    el.bar_px = bar
    return mark(el, COPY["scale"], CAPTION_SIZE)


def all_text_elements() -> dict[str, list]:
    out = {"title": title_text_els()}
    for name in ("sections", "projection", "local"):
        els = section_header(SECTION_INDEX[name]).els + [caption_el(name)]
        if name == "local":
            els.append(scale_bar_el())
        out[name] = els
    hdr = section_header(SECTION_INDEX["objective"])
    out["objective"] = hdr.els + statement_els(hdr)
    out["closing"] = closing_text_els()
    _check_scale_bar(out["local"])
    return out


def _check_scale_bar(els) -> None:
    """The scale bar sits wholly inside the LN card and clear of the caption row below it."""
    bar = next(e for e in els if getattr(e, "label", "") == COPY["scale"])
    bx, by, bw, bh = media_boxes("local")[SECTION_MEDIA["local"].index(SCALE_BAR_MOVIE)]
    if not (bx <= bar.x and bar.x + bar.img.width <= bx + bw and by <= bar.y and bar.y + bar.img.height <= by + bh):
        raise ValueError("scale bar leaves the local-neuron card")
    if bar.y + bar.img.height > CAPTION_Y:
        raise ValueError("scale bar runs into the caption row")


def text_inventory() -> list[tuple[str, str, int, str]]:
    """(segment, text, px, role) of every text element drawn."""
    return [(name, el.label, el.px, el.role) for name, els in all_text_elements().items()
            for el in els if hasattr(el, "img")]


def onscreen_strings() -> list[str]:
    return [text for _, text, _, _ in text_inventory()]


def reading_schedule() -> list[tuple[str, float, float, list]]:
    els = all_text_elements()
    return [(s.name, s.duration, s.fade_out, style.read_items(els[s.name])) for s in segments()]


# ================================================================== movie I/O
def ensure_assets(asset_dir: Path = ASSET_DIR) -> None:
    """Copy each downloaded movie once into the asset folder (bytes only, never executed)."""
    asset_dir.mkdir(parents=True, exist_ok=True)
    for mv in MOVIES.values():
        dst = asset_dir / mv.file
        if not dst.exists():
            shutil.copyfile(DOWNLOAD_DIR / mv.download, dst)


@lru_cache(maxsize=8)
def load_movie(key: str, size: tuple[int, int], asset_dir: str = str(ASSET_DIR),
               crop: tuple[int, int, int, int] | None = None, only: int | None = None) -> np.ndarray:
    """Decode (OpenCV), crop, linearly lift and resize every frame -> (n, h, w, 3) uint8 RGB.
    `crop` overrides the movie's crop; `only` keeps just that one source frame."""
    import cv2
    mv = MOVIES[key]
    cap = cv2.VideoCapture(str(Path(asset_dir) / mv.file))
    l, t, r, b = crop or mv.crop
    frames, n = [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if (f.shape[1], f.shape[0]) != mv.size:
            raise ValueError(f"{mv.file}: frame size {f.shape[1]}x{f.shape[0]} != {mv.size}")
        if only is None or n == only:
            f = lift(f[t:b, l:r, ::-1], mv.black, mv.white)
            frames.append(cv2.resize(f, size, interpolation=cv2.INTER_AREA))
        n += 1
    cap.release()
    if n != mv.n_frames:
        raise ValueError(f"{mv.file}: {n} frames, expected {mv.n_frames}")
    return np.stack(frames)


@lru_cache(maxsize=16)
def card_mask(size: tuple[int, int]) -> Image.Image:
    return lab.rounded_mask(size, CARD_RADIUS)


def desaturate(img: np.ndarray, amount: float) -> np.ndarray:
    if img.ndim == 4:                                   # a whole movie: frame by frame (bounded memory)
        return np.stack([desaturate(f, amount) for f in img])
    grey = img.astype(np.float32).mean(-1, keepdims=True)
    return np.clip(np.rint(img * (1 - amount) + grey * amount), 0, 255).astype(np.uint8)


# ================================================================== scenes
class _Base:
    name = ""

    def __init__(self):
        s = seg(self.name)
        self.duration, self.fade_out = s.duration, s.fade_out

    def _fade(self, fr: Image.Image, t: float, fade_in: float = 0.0) -> Image.Image:
        a = lab.segment_fade(t, self.duration, fade_in, self.fade_out)
        return lab.fade_to_surface(fr, a).convert("RGB")


class TitleScene(_Base):
    """Lab hero card over a dimmed, slowly rotating projection-neuron fill."""
    name = "title"

    def __init__(self, asset_dir: Path = ASSET_DIR):
        super().__init__()
        self.clip = segment_clips("title")[0]
        frames = load_movie(TITLE_MOVIE, (W, H), str(asset_dir), crop=TITLE_CROP)
        self.frames = desaturate(frames, lab.BG_DESATURATE)
        self.scrim = lab.make_scrim()
        self.motif = lab.Motif()
        self.els = title_text_els()

    def frame(self, t: float) -> Image.Image:
        bg = blend_frames(self.frames, clip_pos(self.clip, t), True)
        fr = Image.fromarray(bg).convert("RGBA")
        fr.alpha_composite(self.scrim)
        self.motif.draw(fr, t / style.PACE, alpha=lab.smoothstep((t - 0.2 * style.PACE) / (1.2 * style.PACE))
                        * TITLE_MOTIF_ALPHA)
        for el in self.els:
            el.draw(fr, t)
        return self._fade(fr, t, style.HERO_FADE_IN)


class MediaScene(_Base):
    """Numbered lab section: header, movie card(s) easing in, caption row; footage plays through the fade."""

    def __init__(self, name: str, asset_dir: Path = ASSET_DIR):
        self.name = name
        super().__init__()
        self.hdr = section_header(SECTION_INDEX[name])
        self.clips = segment_clips(name)
        self.boxes = media_boxes(name)
        self.frames = [load_movie(c.key, (bw, bh), str(asset_dir)) for c, (_, _, bw, bh) in zip(self.clips, self.boxes)]
        self.els = all_text_elements()[name]

    def frame(self, t: float) -> Image.Image:
        fr = Image.new("RGBA", (W, H), rgb(SURFACE) + (255,))
        a = lab.reveal(t, style.CONTENT_T0, style.CONTENT_DUR)
        if a > 0.003:
            for clip, (bx, by, bw, bh), frames in zip(self.clips, self.boxes, self.frames):
                img = Image.fromarray(blend_frames(frames, clip_pos(clip, t), clip.loop))
                if a < 0.999:
                    img = Image.blend(Image.new("RGB", img.size, rgb(SURFACE)), img, a)
                fr.paste(img, (bx, by), card_mask((bw, bh)))
        for el in self.els:
            el.draw(fr, t)
        return self._fade(fr, t)


class ObjectiveScene(_Base):
    name = "objective"

    def __init__(self, asset_dir: Path = ASSET_DIR):
        super().__init__()
        self.els = all_text_elements()["objective"]

    def frame(self, t: float) -> Image.Image:
        fr = Image.new("RGBA", (W, H), rgb(SURFACE) + (255,))
        for el in self.els:
            el.draw(fr, t)
        return self._fade(fr, t)


class ClosingScene(_Base):
    """Lab closing card over a slowly rotating, pushing-in projection-neuron fill (scrim + motif)."""
    name = "closing"

    def __init__(self, asset_dir: Path = ASSET_DIR):
        super().__init__()
        mv = MOVIES[CLOSING_MOVIE]
        self.clip = segment_clips("closing")[0]
        self.frames = desaturate(load_movie(CLOSING_MOVIE, _cover_size(mv), str(asset_dir)), lab.BG_DESATURATE)
        self.scrim = lab.make_scrim()
        self.motif = lab.Motif()
        self.els = all_text_elements()["closing"]

    def background(self, t: float) -> Image.Image:
        z = closing_zoom(t)                                          # slow push-in, runs through the fade
        img = Image.fromarray(blend_frames(self.frames, clip_pos(self.clip, t), True))
        box = lab.ken_burns_rect(img.width, img.height, 1.0, z, 0.5, 0.62)
        return img.resize((W, H), Image.BICUBIC, box=box).convert("RGBA")

    def frame(self, t: float) -> Image.Image:
        fr = self.background(t)
        fr.alpha_composite(self.scrim)
        self.motif.draw(fr, t / style.PACE, alpha=lab.smoothstep((t - 0.2) / 1.2) * TITLE_MOTIF_ALPHA)
        for el in self.els:
            el.draw(fr, t)
        return self._fade(fr, t, 0.5)


def _cover_size(mv: Movie) -> tuple[int, int]:
    cw, ch = crop_size(mv)
    s = W / min(cw, ch)
    return int(round(cw * s)), int(round(ch * s))


def build_scene(name: str, asset_dir: Path = ASSET_DIR):
    if name == "title":
        return TitleScene(asset_dir)
    if name in ("sections", "projection", "local"):
        return MediaScene(name, asset_dir)
    if name == "objective":
        return ObjectiveScene(asset_dir)
    if name == "closing":
        return ClosingScene(asset_dir)
    raise KeyError(name)


# ================================================================== rendering
def encoder_cmd(out_path: Path) -> list[str]:
    return [FFMPEG, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
            "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264", "-preset", "slow", "-crf", "16",
            "-pix_fmt", "yuv420p", str(out_path)]


def render_segment(args: tuple[str, str, str]) -> str:
    name, out_path, asset_dir = args
    scene = build_scene(name, Path(asset_dir))
    enc = subprocess.Popen(encoder_cmd(Path(out_path)), stdin=subprocess.PIPE)
    for i in range(frame_count(scene.duration)):
        enc.stdin.write(scene.frame(i / FPS).tobytes())
    enc.stdin.close()
    if enc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {name}")
    return out_path


def render(out_path: Path = OUT_PATH, asset_dir: Path = ASSET_DIR) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        jobs = [(s.name, str(Path(td) / f"{i}_{s.name}.mp4"), str(asset_dir)) for i, s in enumerate(segments())]
        with Pool(len(jobs)) as pool:
            pool.map(render_segment, jobs)
        lst = Path(td) / "list.txt"
        lst.write_text("".join(f"file '{p}'\n" for _, p, _ in jobs))
        subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
                        "-c", "copy", "-an", "-movflags", "+faststart", str(out_path)], check=True)
    print(f"wrote {out_path}")


def preview_times(name: str) -> dict[str, float]:
    s = seg(name)
    fs = fade_start(name)
    return {"a_reveal": 1.2, "b_mid": (PLAY_T0 + fs) / 2, "c_fade_start": fs,
            "d_fade": s.duration - s.fade_out / 2}


def render_preview(out_dir: Path, asset_dir: Path = ASSET_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, s in enumerate(segments()):
        sc = build_scene(s.name, asset_dir)
        for label, t in preview_times(s.name).items():
            p = out_dir / f"{i}_{s.name}_{label}.png"
            sc.frame(t).save(p)
            print("wrote", p)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preview", type=Path, metavar="DIR", help="write key frames per segment instead of rendering")
    ap.add_argument("--assets", type=Path, default=ASSET_DIR, help="folder holding the copied movies")
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args()
    ensure_assets(args.assets)
    if args.preview:
        render_preview(args.preview, args.assets)
        return
    render(args.out, args.assets)


if __name__ == "__main__":
    main()
