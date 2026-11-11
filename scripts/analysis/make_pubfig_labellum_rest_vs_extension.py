#!/usr/bin/env python3
"""Two-panel publication figure: labellum at rest vs full proboscis extension.

Fly: Hex-Training-24-0.1 / august_11_batch_1_rig_2 / fly3, training_4
(Hexanol + light). Chosen by scanning every training-cohort trial for the
strongest light-evoked extension (+61 px in its weakest training trial) that
also shows CS+-specific odor PER on testing.

Frames come straight from the raw video as the encoded luma (Y) plane: full
resolution (B/R are 4:2:0 half-res chroma and look blocky) and it does not clip
under the red stimulus light, whereas the decoded R channel saturates.

Eye / labellum boxes are YOLO26-OBB detections (train12 best.pt, run under
yolo-env -- the .engine returns garbage confidences outside the pipeline),
frozen here so the figure rebuilds without the model.

Each panel is ONE unaveraged frame: the sharpest (Laplacian variance of the head
region) among frames with the labellum at rest (before odor onset) or at full
extension (top decile of eye-labellum distance inside the light window). Then a
mild unsharp mask (sigma=1.0 px, amount=0.6), the linear percentile stretch and
one Lanczos pre-upscale (shown with interpolation="none", so nothing resamples
twice).

Boxes are shrunk to 0.8x about their own centroid so they sit inside the structure.

Colours are slots 1-2 of the dataviz reference palette (dark steps), validated
CVD-safe (worst protan dE 26.8); each mark carries a dark halo because both
fall below 3:1 against the gray cuticle.
"""

from __future__ import annotations

from pathlib import Path

import subprocess

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib import font_manager
from matplotlib.patches import Polygon
import numpy as np


ROOT = Path("/home/ramanlab/Documents/cole")
VIDEO = Path(
    "/securedstorage/DATAsec/cole/Data-secured-New/Hex-Training-24-0.1/august_11_batch_1_rig_2"
    "/output_august_11_batch_1_training_4_Hexanol_20260811_130413.mp4"
)
OUT_PATH = ROOT / "Results/Figures/august_11_batch_1_rig_2_fly3_labellum_rest_vs_extension.png"

# (x0, y0, width, height) of the crop around fly3's head, in video pixels.
CROP = (420, 540, 320, 260)
FRAME_WH = (1080, 1080)

EYE_COLOR = "#3987e5"
LABELLUM_COLOR = "#d95926"
BOX_SCALE = 0.8       # polygon shrink about its centroid
BOX_LW, HALO_LW, LABEL_PT = 1.0, 2.0, 8.5
HALO = [pe.withStroke(linewidth=HALO_LW, foreground="black")]

UNSHARP_SIGMA, UNSHARP_AMOUNT = 1.0, 0.6
UPSCALE = 6           # ~ printed size at 600 dpi: 320 px * 6 = 1920 px over ~3.2 in

# frame -> (title, eye OBB corners, labellum OBB corners), full-frame pixels.
PANELS = {
    913: (  # conf eye 0.94, labellum 0.88
        "No extension",
        [[586.9, 645.1], [584.2, 596.3], [544.8, 598.5], [547.5, 647.3]],
        [[580.2, 675.1], [578.3, 656.1], [551.3, 658.8], [553.2, 677.8]],
    ),
    1524: (  # conf eye 0.89, labellum 0.93
        "Full extension",
        [[583.9, 643.5], [584.8, 598.6], [547.7, 597.9], [546.8, 642.7]],
        [[549.6, 749.9], [559.3, 728.1], [530.3, 715.1], [520.6, 736.9]],
    ),
}


def read_luma(video: Path, frame_idx: int, frame_wh: tuple[int, int] = FRAME_WH) -> np.ndarray:
    """Return the Y plane of one frame as a uint8 (H, W) array."""
    w, h = frame_wh
    raw = subprocess.run(
        ["/usr/bin/ffmpeg", "-v", "error", "-i", str(video),
         "-vf", f"select=eq(n\\,{frame_idx}),extractplanes=y",
         "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        check=True, capture_output=True,
    ).stdout
    if len(raw) != w * h:
        raise RuntimeError(f"Could not read frame {frame_idx} from {video}")
    return np.frombuffer(raw, dtype=np.uint8).reshape(h, w)


def unsharp(img: np.ndarray, sigma: float = UNSHARP_SIGMA, amount: float = UNSHARP_AMOUNT) -> np.ndarray:
    """Unsharp mask: img + amount * (img - gaussian(img, sigma)); float in, float out."""
    img = np.asarray(img, dtype=float)
    return img + amount * (img - cv2.GaussianBlur(img, (0, 0), sigma))


def stretch(a: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.9) -> np.ndarray:
    """Linear percentile contrast stretch to uint8 0..255."""
    lo, hi = np.percentile(a, [lo_pct, hi_pct])
    return np.clip((a - lo) / max(hi - lo, 1e-9) * 255.0, 0, 255).astype(np.uint8)


def crop_stretch(luma: np.ndarray, crop: tuple[int, int, int, int],
                 lo_pct: float = 1.0, hi_pct: float = 99.9) -> np.ndarray:
    """Crop and linearly contrast-stretch to 0..255 (no denoise: it smears cuticle detail)."""
    x0, y0, w, h = crop
    return stretch(luma[y0:y0 + h, x0:x0 + w].astype(float), lo_pct, hi_pct)


def shrink_polygon(poly, scale: float = BOX_SCALE) -> np.ndarray:
    """Scale a polygon about its vertex centroid (orientation preserved)."""
    p = np.asarray(poly, dtype=float)
    c = p.mean(axis=0)
    return c + (p - c) * scale


def panel_image(frame_idx: int) -> np.ndarray:
    """Single frame: cropped, sharpened, stretched and upscaled."""
    x0, y0, w, h = CROP
    crop = read_luma(VIDEO, frame_idx)[y0:y0 + h, x0:x0 + w]
    img = stretch(unsharp(crop))
    return cv2.resize(img, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_LANCZOS4)


def to_crop_coords(corners, crop: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0 = crop[:2]
    return np.asarray(corners, dtype=float) - [x0, y0]


def _box(ax: plt.Axes, poly: np.ndarray, color: str, text: str, *, side: str) -> None:
    ax.add_patch(Polygon(poly, closed=True, fill=False, edgecolor=color,
                         linewidth=BOX_LW, path_effects=HALO))
    if side == "above":
        x, y, ha, va = poly[:, 0].mean(), poly[:, 1].min() - 3, "center", "bottom"
    else:
        x, y, ha, va = poly[:, 0].max() + 4, poly[:, 1].mean(), "left", "center"
    ax.text(x, y, text, color=color, fontsize=LABEL_PT, fontweight="bold",
            ha=ha, va=va, path_effects=HALO)


def main() -> None:
    for font_path in (
        "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf",
    ):
        if Path(font_path).exists():
            font_manager.fontManager.addfont(font_path)

    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 600,
            "font.family": "Arial",
            "font.sans-serif": ["Arial"],
            "font.size": 10,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.75), gridspec_kw={"wspace": 0.05})
    for ax, (frame_idx, (title, eye, labellum)) in zip(axes, PANELS.items()):
        img = panel_image(frame_idx)
        w, h = CROP[2:]
        ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="none",
                  extent=(0, w, h, 0))  # axes stay in crop pixels for the boxes
        _box(ax, shrink_polygon(to_crop_coords(eye, CROP)), EYE_COLOR, "Eye", side="above")
        _box(ax, shrink_polygon(to_crop_coords(labellum, CROP)), LABELLUM_COLOR, "Labellum", side="right")
        ax.set_title(title, fontsize=11, pad=4)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(0.8)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".svg", ".pdf"):
        fig.savefig(OUT_PATH.with_suffix(suffix), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"[OK] Wrote {OUT_PATH} (+ .svg, .pdf)")


if __name__ == "__main__":
    main()
