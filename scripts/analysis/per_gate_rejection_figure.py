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


import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fbpipe.utils.columns import (  # noqa: E402
    find_eye_xy_columns,
    find_proboscis_xy_columns,
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

    # Prefer the parquet's own frame column over row position: row index and
    # frame number coincide for this subject (rows are contiguous), but need
    # not in general, and `frames` is what later tasks use to seek the video
    # to the right frame -- silently using row position would be wrong for
    # any subject with dropped/non-contiguous rows.
    frame_col = next(
        (c for c in ("frame", "frame_number", "frame_idx") if c in df.columns),
        None,
    )
    if frame_col is not None:
        frames = pd.to_numeric(df[frame_col], errors="coerce").to_numpy()[ok].astype(int)
    else:
        frames = np.flatnonzero(ok)

    return Offsets(
        dx=dx[ok],
        dy=dy[ok],
        frames=frames,
        eye_xy=(float(np.nanmedian(ex)), float(np.nanmedian(ey))),
    )
