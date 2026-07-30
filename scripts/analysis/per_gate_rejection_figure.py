"""Methods figure: rejecting physically impossible proboscis measurements.

One hero panel (a real video frame with the acceptance boundary traced on it)
plus four small panels, covering the five gates the pipeline applies:

  cap        proboscis detections capped at the number of resolved flies,
             highest confidence kept        (yolo_infer._limit_proboscis_detections)
  spatial    anisotropic gate around each eye, 160 px lateral/ventral,
             40 px dorsal                   (distance_sanity.anisotropic_semi_axes)
  release    with >=3 flies, pairings beyond 160 px are dropped and the
             binding released               (yolo_infer._max_valid_eye_prob_distance_px)
  jump       accepted positions must be within 80 px of the previous
             ACCEPTED position              (distance_sanity.sanitize_proboscis_velocity_dataframe)
  normalize  only 10-160 px contributes to each fly's normalization range
                                            (config_new.yaml distance_limits)

Gate values are read from config/config_new.yaml at runtime; the boundary is
traced by the production function, not re-derived here.

Run:
    python scripts/analysis/per_gate_rejection_figure.py
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for _p in (str(REPO_ROOT), str(SRC_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fbpipe.config import load_raw_config  # noqa: E402

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patheffects import withStroke  # noqa: E402

# Validated all-pairs (light mode): worst CVD dE 13.0, normal-vision 16.3, all
# >= 3:1. A green/red accept/reject pair was tested first and FAILED at deutan
# dE 4.1 -- the same failure that ruled out the red/green score palette.
ACCEPTED = "#2a78d6"   # blue
REJECTED = "#eb6834"   # orange
EYE = "#4a3aa7"        # violet
INK = "#0b0b0b"
MUTED = "#898781"
SURFACE = "#fcfcfb"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "text.color": INK,
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,       # editable text in Illustrator / Inkscape
        "ps.fonttype": 42,
        "svg.fonttype": "none",   # keep <text> as text, not paths, in the SVG
    }
)

# Every string the hero panel is allowed to draw. The test asserts this set
# exactly, which is what keeps the slide from accreting prose.
HERO_TEXTS = frozenset(
    {"dorsal", "ventral", "ACCEPTANCE BOUNDARY", "146 px", "220 px", "160", "40"}
)

CONFIG_PATH = REPO_ROOT / "config" / "config_new.yaml"
OUT_DIR = REPO_ROOT / "figures"


@dataclass(frozen=True)
class GateSettings:
    """The five gate values the figure reports, as read from the config."""

    max_px: float
    up_divisor: float
    max_jump_px: float
    norm_min_px: float
    norm_max_px: float

    @property
    def dorsal_px(self) -> float:
        """Upward (dorsal) allowance -- the tightened semi-axis."""
        return self.max_px / self.up_divisor


def load_gate_settings(config_path: Path | str = CONFIG_PATH) -> GateSettings:
    """Read the gate constants from *config_path*.

    Raises KeyError if a gate block is missing: a figure that silently fell back
    to defaults would print numbers the pipeline does not use.
    """
    raw = load_raw_config(config_path)
    try:
        pf = raw["proboscis_filter"]
        dl = raw["distance_limits"]
        return GateSettings(
            max_px=float(pf["max_eye_prob_distance_px"]),
            up_divisor=float(pf["up_divisor"]),
            max_jump_px=float(pf["max_jump_px"]),
            norm_min_px=float(dl["class2_min"]),
            norm_max_px=float(dl["class2_max"]),
        )
    except KeyError as exc:
        raise KeyError(
            f"{config_path} is missing gate settings: {exc}. "
            "The figure must not fall back to repo defaults (150/180/250)."
        ) from exc


import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fbpipe.utils.columns import (  # noqa: E402
    find_eye_xy_columns,
    find_proboscis_xy_columns,
)
from fbpipe.utils.distance_sanity import (  # noqa: E402
    anisotropic_boundary_offsets,
    anisotropic_semi_axes,
    sanitize_eye_prob_geometry_dataframe,
)
from fbpipe.utils.tables import read_table  # noqa: E402

DATA_ROOT = Path("/home/ramanlab/Documents/cole/Data/flys_New")
VIDEO_ROOT = Path("/securedstorage/DATAsec/cole/Data-secured-New")


@dataclass(frozen=True)
class Subject:
    """The one fly the hero panel zooms in on.

    Chosen from a scan of 2029 fly-trials that still have their source video: it
    is the closest-to-boundary *odor testing* trial in the set, and the only
    near-edge candidate with a perfect detection record (3605/3605).
    """

    dataset: str
    trial_rel: str
    slot: str
    odor: str
    eye_xy: tuple[float, float]
    peak_frame: int
    video_name: str


SUBJECT = Subject(
    dataset="3Oct-Control-24-0.1",
    trial_rel="july_26_batch_1/july_26_batch_1_testing_1",
    slot="fly1",
    odor="3-Octonol",
    eye_xy=(845.0, 183.0),
    peak_frame=1102,
    video_name="output_july_26_batch_1_testing_1_3-Octonol_20260726_161155.mp4",
)


@dataclass(frozen=True)
class Offsets:
    """Accepted eye->proboscis offsets, in pixels, for one fly-trial."""

    dx: np.ndarray
    dy: np.ndarray
    frames: np.ndarray
    eye_xy: tuple[float, float]


def subject_parquet_path(subject: Subject = SUBJECT) -> Path:
    trial_name = subject.trial_rel.split("/")[-1]
    return (
        DATA_ROOT
        / subject.dataset
        / subject.trial_rel
        / f"{trial_name}_{subject.slot}_distances.parquet"
    )


def subject_video_path(subject: Subject = SUBJECT) -> Path:
    batch_dir = subject.trial_rel.split("/")[0]
    return VIDEO_ROOT / subject.dataset / batch_dir / subject.video_name


def _resolve_frame_numbers(df: pd.DataFrame, ok: np.ndarray) -> np.ndarray:
    """Resolve the video frame number for each accepted row.

    Prefers the dataframe's own frame column ("frame", then the aliases
    "frame_number" / "frame_idx") over row position: row index and frame
    number coincide only when the rows are contiguous, but `frames` is what
    later tasks use to seek the video to the right frame -- silently deriving
    it from row position would be wrong for any subject with dropped or
    non-contiguous rows. Falls back to row position (`np.flatnonzero(ok)`)
    when no such column exists, or when the column exists but cannot be
    cleanly converted to integers (e.g. it contains NaN/unparsable values):
    degrading to row position there is safer than raising, since row
    position is still a legitimate frame index for a fully contiguous
    recording.
    """
    frame_col = next(
        (c for c in ("frame", "frame_number", "frame_idx") if c in df.columns),
        None,
    )
    if frame_col is not None:
        raw = pd.to_numeric(df[frame_col], errors="coerce").to_numpy()[ok]
        if not np.any(np.isnan(raw)):
            return raw.astype(int)
    return np.flatnonzero(ok)


def load_subject_offsets(subject: Subject = SUBJECT) -> Offsets:
    """Read the fly's accepted proboscis positions as offsets from its frozen eye."""
    df = read_table(subject_parquet_path(subject))
    ex_col, ey_col = find_eye_xy_columns(df)
    px_col, py_col = find_proboscis_xy_columns(df)
    if not (ex_col and ey_col and px_col and py_col):
        raise ValueError(f"missing eye/proboscis columns in {subject_parquet_path(subject)}")

    ex = pd.to_numeric(df[ex_col], errors="coerce").to_numpy(float)
    ey = pd.to_numeric(df[ey_col], errors="coerce").to_numpy(float)
    px = pd.to_numeric(df[px_col], errors="coerce").to_numpy(float)
    py = pd.to_numeric(df[py_col], errors="coerce").to_numpy(float)

    dx, dy = px - ex, py - ey
    ok = np.isfinite(dx) & np.isfinite(dy)

    return Offsets(
        dx=dx[ok],
        dy=dy[ok],
        frames=_resolve_frame_numbers(df, ok),
        eye_xy=(float(np.nanmedian(ex)), float(np.nanmedian(ey))),
    )


# The constructed bad detection, as an offset from the eye. Lateral-ventral
# quadrant, well outside the gate: r = 220.5 px, gate norm 1.90. Task 3's tests
# push this through the real production gate to prove it is genuinely rejected.
REJECTED_OFFSET: tuple[float, float] = (185.0, 120.0)


def gate_boundary_offsets(settings: GateSettings, n: int = 360) -> np.ndarray:
    """Trace the acceptance boundary as (n, 2) dx/dy offsets from the eye.

    Delegates to the production drawing function so the figure cannot drift from
    the implementation.
    """
    pts = anisotropic_boundary_offsets(settings.max_px, settings.up_divisor, n)
    return np.asarray(pts, dtype=float)


def gate_norm(dx, dy, settings: GateSettings) -> np.ndarray:
    """Gate-normalised radius. 1.0 is exactly on the boundary; > 1.0 is rejected."""
    dx = np.asarray(dx, dtype=float)
    dy = np.asarray(dy, dtype=float)
    a, b = anisotropic_semi_axes(dx, dy, settings.max_px, settings.up_divisor)
    return (dx / a) ** 2 + (dy / b) ** 2


def offsets_survive_geometry_gate(dx, dy, settings: GateSettings) -> np.ndarray:
    """Run offsets through the REAL production geometry gate.

    Returns a boolean mask: True where the point survives, False where the
    pipeline would blank it. Used to guarantee the figure's rejected example is
    a rejection the model actually makes.
    """
    dx = np.asarray(dx, dtype=float)
    dy = np.asarray(dy, dtype=float)
    frame = pd.DataFrame(
        {
            "x_class0": np.zeros_like(dx),
            "y_class0": np.zeros_like(dy),
            "x_class1": dx,
            "y_class1": dy,
        }
    )
    cleaned, _ = sanitize_eye_prob_geometry_dataframe(
        frame, settings.max_px, settings.up_divisor
    )
    return pd.to_numeric(cleaned["x_class1"], errors="coerce").notna().to_numpy()


# Crop window around the frozen eye at (845, 183): the full gate (160 px lateral
# and ventral, 40 px dorsal) plus margin, and room for the constructed rejection
# at eye + (185, 120). Verified to sit inside the 1080x1080 frame.
CROP: tuple[int, int, int, int] = (645, 63, 1045, 383)

# Blend fraction toward white, applied AFTER the contrast stretch and
# inversion below. Not stylistic: the palette validator WARNs at every
# mid-gray surface tested (orange falls to 1.61:1 on #b8b8b6), so a raw
# photographic background would void the contrast guarantees.
#
# This footage is near-black IR video (raw crop mean 17.0, min 5, max 79):
# a naive "blend toward white" (gray + (255-gray)*ghost) at any ghost strong
# enough to pale the surface also flattens that whole narrow raw range into a
# handful of grey levels -- at ghost=0.80 the naive approach produced mean
# 206.8 but a range of only 14 (205..219): a blank pale rectangle with no
# visible fly. So the pipeline contrast-stretches by percentile first (to use
# the full 0-255 range) and INVERTS (the footage is dark-background/
# bright-fly; inverting puts the fly dark-on-pale, which is the correct
# polarity for a light surface the coloured overlay marks need). GHOST_BLEND
# is then a much gentler final blend on top of that already-pale surface.
# Verified at 0.40: mean 199.3, min 102, max 255, range 153.
GHOST_BLEND = 0.40


def load_frame_crop(
    video_path: Path,
    frame_index: int,
    crop: tuple[int, int, int, int] = CROP,
    ghost: float = GHOST_BLEND,
) -> np.ndarray:
    """Pull one frame, crop it, grayscale it, contrast-stretch, invert, and
    ghost it toward white.

    The stretch and inversion are what keep the fly visible: this footage is
    near-black IR video occupying a narrow raw range, so a naive blend toward
    white would flatten it to a handful of grey levels before it reads as
    pale enough for the overlay palette. Stretching first uses the full 0-255
    range, and inverting puts the (naturally bright) fly down as a dark shape
    on a light surface -- the polarity the coloured overlay marks need.

    Returns an RGB uint8 array so matplotlib can draw it directly.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_path}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok or frame is None:
        raise ValueError(f"cannot read frame {frame_index} of {video_path}")

    x0, y0, x1, y1 = crop
    gray = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(float)

    lo, hi = np.percentile(gray, 1), np.percentile(gray, 99.5)
    stretched = np.clip((gray - lo) / max(hi - lo, 1e-6), 0, 1) * 255.0
    inverted = 255.0 - stretched

    pale = inverted + (255.0 - inverted) * float(ghost)
    return np.repeat(np.clip(pale, 0, 255).astype(np.uint8)[:, :, None], 3, axis=2)


def _halo(width: float = 3.0):
    """White relief so a mark stays legible over the ghosted frame."""
    return [withStroke(linewidth=width, foreground="white")]


def draw_hero(ax, offsets: Offsets, settings: GateSettings, image: np.ndarray | None) -> None:
    """Panel A: the acceptance boundary drawn over a real frame, in eye-centred px."""
    ex, ey = offsets.eye_xy
    x0, y0, x1, y1 = CROP

    if image is not None:
        ax.imshow(image, extent=(x0 - ex, x1 - ex, y1 - ey, y0 - ey),
                  interpolation="bilinear", zorder=0)

    # accepted cloud -- all real detections
    ax.scatter(offsets.dx, offsets.dy, s=9, c=ACCEPTED, alpha=0.10,
               linewidths=0, zorder=2)

    # the acceptance boundary, traced by the production function
    pts = gate_boundary_offsets(settings, n=360)
    ring = np.vstack([pts, pts[:1]])
    ax.plot(ring[:, 0], ring[:, 1], color=INK, lw=2.5, zorder=4,
            path_effects=_halo(5.0))

    # frozen eye anchor
    ax.plot([0], [0], marker="P", ms=11, color=EYE, mew=0, zorder=6,
            path_effects=_halo())

    # peak PER -- the accepted detection nearest the boundary
    r = np.hypot(offsets.dx, offsets.dy)
    pk = int(np.argmax(r))
    pdx, pdy = float(offsets.dx[pk]), float(offsets.dy[pk])
    ax.plot([0, pdx], [0, pdy], color=ACCEPTED, lw=1.6, zorder=5)
    ax.plot([pdx], [pdy], marker="o", ms=11, color=ACCEPTED, mew=2.0,
            mec="white", zorder=7)
    ax.text(pdx + 12, pdy, "146 px", color=ACCEPTED, fontsize=12,
            fontweight="bold", va="center", ha="left", path_effects=_halo())

    # constructed rejection -- verified against the real gate in the tests
    rdx, rdy = REJECTED_OFFSET
    ax.plot([0, rdx], [0, rdy], color=REJECTED, lw=1.6, ls=(0, (4, 3)), zorder=5)
    ax.plot([rdx], [rdy], marker="X", ms=15, mfc="none", mec=REJECTED, mew=3.0,
            zorder=7, path_effects=_halo())
    ax.text(rdx, rdy + 16, "220 px", color=REJECTED, fontsize=12,
            fontweight="bold", va="top", ha="center", path_effects=_halo())

    # gate values, on the boundary itself
    lat, dorsal = settings.max_px, settings.dorsal_px
    ax.text(lat + 6, 0, str(int(lat)), color=INK, fontsize=12, va="center",
            ha="left", path_effects=_halo())
    ax.text(0, lat + 6, str(int(lat)), color=INK, fontsize=12, va="top",
            ha="center", path_effects=_halo())
    ax.text(0, -dorsal - 6, str(int(dorsal)), color=INK, fontsize=12,
            va="bottom", ha="center", path_effects=_halo())

    ax.text(0, -dorsal - 34, "dorsal", color=MUTED, fontsize=11, va="bottom",
            ha="center", style="italic")
    ax.text(0, lat + 34, "ventral", color=MUTED, fontsize=11, va="top",
            ha="center", style="italic")
    ax.text(0.02, 0.02, "ACCEPTANCE BOUNDARY", transform=ax.transAxes,
            color=INK, fontsize=13, fontweight="bold", va="bottom", ha="left")

    ax.set_xlim(x0 - ex, x1 - ex)
    ax.set_ylim(y1 - ey, y0 - ey)   # image convention: +dy is ventral, downward
    ax.set_aspect(1.0)
    ax.axis("off")


# Panels whose marks are illustrations rather than this fly's measurements. The
# subject has 2 flies (so the >=3-fly release never fires) and a max real
# displacement of 5.7 px (so the 80 px gate is never approached). These get a
# dashed border, and the caption says so once.
SCHEMATIC_PANELS = frozenset({"cap", "release", "jump"})


def _mark_schematic(ax) -> None:
    """Hairline dashed border: this panel is an illustration, not measured data."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linestyle((0, (3, 3)))
        spine.set_linewidth(0.8)
        spine.set_edgecolor(MUTED)


def _panel_title(ax, text: str) -> None:
    ax.text(0.0, 1.0, text, transform=ax.transAxes, color=INK, fontsize=11,
            fontweight="bold", va="bottom", ha="left")


def draw_cap_panel(ax, n_flies: int) -> None:
    """Panel B: detections ranked by confidence, capped at the resolved fly count."""
    confidences = [0.93, 0.88, 0.71, 0.40]
    for i, conf in enumerate(confidences):
        x = 0.12 + 0.25 * i
        keep = i < n_flies
        if keep:
            ax.plot([x], [0.55], marker="o", ms=13, color=ACCEPTED, mew=2.0,
                    mec="white")
        else:
            ax.plot([x], [0.55], marker="X", ms=13, mfc="none", mec=REJECTED,
                    mew=2.5)
        ax.text(x, 0.28, f"{conf:.2f}",
                color=ACCEPTED if keep else REJECTED,
                fontsize=10, ha="center", va="top")

    _panel_title(ax, f"{n_flies}-fly cap")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    _mark_schematic(ax)


def draw_release_panel(ax, settings: GateSettings) -> None:
    """Panel C: with >=3 flies, an over-distance pairing is dropped, not stretched."""
    ax.plot([0.15], [0.55], marker="P", ms=12, color=EYE, mew=0)
    ax.plot([0.85], [0.55], marker="P", ms=12, color=EYE, mew=0)
    ax.plot([0.42], [0.55], marker="o", ms=11, color=ACCEPTED, mew=2.0, mec="white")

    ax.plot([0.15, 0.42], [0.55, 0.55], color=ACCEPTED, lw=1.8)
    ax.plot([0.85, 0.56], [0.55, 0.55], color=REJECTED, lw=1.8, ls=(0, (4, 3)))
    ax.plot([0.55], [0.55], marker="X", ms=13, mfc="none", mec=REJECTED, mew=2.5)

    ax.text(0.70, 0.30, str(int(settings.max_px)), color=REJECTED, fontsize=11,
            ha="center", va="top")
    _panel_title(ax, "≥3 flies")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    _mark_schematic(ax)


def draw_jump_panel(ax, settings: GateSettings) -> dict:
    """Panel D: the displacement gate, measured from the last ACCEPTED position.

    Drawn in PIXEL data coordinates so the ring radius on screen is literally
    ``settings.max_jump_px``. Returns the ring geometry so the tests can assert
    both that the drawn circle uses the gate value and that it is centred on the
    last accepted point rather than on the rejected one.
    """
    accepted_x = [40.0, 70.0, 100.0, 130.0]
    accepted_y = [110.0, 118.0, 106.0, 114.0]
    ax.plot(accepted_x, accepted_y, color=ACCEPTED, lw=1.8, marker="o", ms=9,
            mew=1.6, mec="white")

    last_x, last_y = accepted_x[-1], accepted_y[-1]
    rej_x, rej_y = 250.0, 112.0  # 120 px away -- outside the 80 px gate
    ax.plot([last_x, rej_x], [last_y, rej_y], color=REJECTED, lw=1.8,
            ls=(0, (4, 3)))
    ax.plot([rej_x], [rej_y], marker="X", ms=13, mfc="none", mec=REJECTED,
            mew=2.5)

    ax.add_patch(plt.Circle((last_x, last_y), settings.max_jump_px, fill=False,
                            color=INK, lw=1.2, ls=(0, (2, 2))))
    ax.text(last_x, last_y - settings.max_jump_px - 6, str(int(settings.max_jump_px)),
            color=INK, fontsize=11, ha="center", va="top")

    _panel_title(ax, "jump gate")
    ax.set_xlim(0, 320)
    ax.set_ylim(0, 210)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect(1.0)
    _mark_schematic(ax)
    return {"ring_centre": (last_x, last_y), "ring_radius": settings.max_jump_px}


def draw_norm_panel(ax, offsets: Offsets, settings: GateSettings) -> dict:
    """Panel E: the normalization window, with this fly's real spread inside it."""
    lo, hi = settings.norm_min_px, settings.norm_max_px
    axis_max = 200.0

    ax.axhspan(0.42, 0.68, xmin=lo / axis_max, xmax=hi / axis_max,
               color=ACCEPTED, alpha=0.16, lw=0)

    r = np.hypot(offsets.dx, offsets.dy)
    ax.plot([np.percentile(r, 1), np.percentile(r, 99)], [0.55, 0.55],
            color=ACCEPTED, lw=4.0, solid_capstyle="round")
    ax.plot([r.max()], [0.55], marker="o", ms=9, color=ACCEPTED, mew=1.6,
            mec="white")

    for value in (lo, hi):
        ax.plot([value, value], [0.42, 0.68], color=INK, lw=1.2)
        ax.text(value, 0.34, str(int(value)), color=INK, fontsize=11,
                ha="center", va="top")

    _panel_title(ax, "normalize")
    ax.set_xlim(0, axis_max)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return {"band": (lo, hi)}
