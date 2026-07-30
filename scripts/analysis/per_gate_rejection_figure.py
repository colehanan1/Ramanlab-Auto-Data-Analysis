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
