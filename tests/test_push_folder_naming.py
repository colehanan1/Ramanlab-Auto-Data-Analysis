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


# ── manual (non-optogenetic) sessions ────────────────────────────────────────
# Manual runs reuse every other naming rule and only append a "-Manual" segment,
# so the manual cohort stays a separate dataset from its opto counterpart while
# still canonicalizing to the same odor downstream.


def test_manual_appends_suffix_to_real_odor():
    remote = get_push_folder(
        "OFM_H", "Control", BASE, starvation_hours=24, odor_vial_conc="0.005%",
        manual=True,
    )
    assert _folder(remote) == "Hex-Control-24-0.005-Manual"


def test_manual_appends_suffix_to_mode_invariant_pin():
    remote = get_push_folder(
        "OFM_PANEL", "Training", BASE, starvation_hours=24, odor_vial_conc="10%",
        manual=True,
    )
    assert _folder(remote) == "RandomPanel-24-10-Manual"


def test_manual_appends_suffix_without_optional_params():
    remote = get_push_folder("OFM_PANEL", "Training", BASE, manual=True)
    assert _folder(remote) == "RandomPanel-Manual"


def test_manual_defaults_off():
    """Omitting manual must reproduce the opto folder name byte-for-byte."""
    kwargs = dict(starvation_hours=24, odor_vial_conc="0.005%")
    assert get_push_folder("OFM_H", "Training", BASE, **kwargs) == get_push_folder(
        "OFM_H", "Training", BASE, manual=False, **kwargs
    )
    assert _folder(get_push_folder("OFM_H", "Training", BASE, **kwargs)) == \
        "Hex-Training-24-0.005"


def test_manual_preserves_host_and_base_path():
    remote = get_push_folder(
        "OFM_E", "Training", BASE, starvation_hours=36, odor_vial_conc="10%",
        manual=True,
    )
    assert remote == "ramanlab@host:/data/flys/EB-Training-36-10-Manual/"


def test_manual_local_path_base_has_no_host_split():
    remote = get_push_folder(
        "OFM_E", "Training", "/data/flys", starvation_hours=36, manual=True,
    )
    assert remote == "/data/flys/EB-Training-36-Manual/"


def test_manual_dataset_name_still_canonicalizes_downstream():
    """The suffixed folder must survive fbpipe's dataset canonicalization."""
    from fbpipe.odor_constants import canon_dataset, resolve_dataset_label

    folder = _folder(get_push_folder(
        "OFM_H", "Training", BASE, starvation_hours=24, odor_vial_conc="0.005%",
        manual=True,
    ))
    canon = canon_dataset(folder)
    # Distinct cohort from the opto run…
    assert canon != canon_dataset("Hex-Training-24-0.005")
    # …but still resolves to the same trained odor.
    assert resolve_dataset_label(canon) == "Hexanol"
