"""Tests for get_push_folder() folder-name construction.

Real odor conditions encode the training/control mode in the folder name
(e.g. ``Hex-Control-24-0.005``). The non-odor pseudo-pins (RandomPanel and
LightSweep) are mode-invariant — ``expand_config`` builds the same cycles for
both training and control — so the mode segment is dropped, yielding names like
``RandomPanel-24-10`` instead of ``RandomPanel-Training-24-10``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "PiCode"))

from experiment_scheduler import get_push_folder  # noqa: E402

BASE = "ramanlab@host:/data/flys"


def _folder(remote: str) -> str:
    """Extract just the trailing folder name from a remote push spec."""
    return remote.rstrip("/").rsplit("/", 1)[-1]


def test_real_odor_keeps_mode_segment():
    remote = get_push_folder(
        "OFM_H", "Control", BASE, starvation_hours=24, odor_vial_conc="0.005%"
    )
    assert _folder(remote) == "Hex-Control-24-0.005"


def test_random_panel_drops_mode_segment():
    """RandomPanel is mode-invariant, so the folder omits Training/Control."""
    for mode in ("Training", "Control"):
        remote = get_push_folder(
            "OFM_PANEL", mode, BASE, starvation_hours=24, odor_vial_conc="10%"
        )
        assert _folder(remote) == "RandomPanel-24-10"


def test_light_sweep_drops_mode_segment():
    """LightSweep is also mode-invariant (light fires in both modes)."""
    for mode in ("Training", "Control"):
        remote = get_push_folder(
            "OFM_LIGHT", mode, BASE, starvation_hours=24, odor_vial_conc="10%"
        )
        assert _folder(remote) == "LightSweep-24-10"


def test_random_panel_without_optional_params():
    remote = get_push_folder("OFM_PANEL", "Training", BASE)
    assert _folder(remote) == "RandomPanel"
