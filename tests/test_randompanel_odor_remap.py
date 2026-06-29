"""Regression tests for the RandomPanel-Training-24-10 odor correction.

The rig labelled every ``training_N_Linalool`` trial in
``RandomPanel-Training-24-10`` as Linalool, but the odor actually delivered was
isoamyl acetate. Rather than rewriting the raw recording files, the figures
relabel that one dataset's "Linalool" as "Isoamyl Acetate" via the per-dataset
``odor_remap`` override (``DatasetOverride.odor_remap`` in the config, applied by
``apply_dataset_odor_remap`` at every display-odor resolution site).

These tests pin both halves of the wiring:
  1. the real ``config/config.yaml`` carries the override, and
  2. it actually changes the figure label for this dataset only.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fbpipe.config import load_settings
from scripts.analysis import envelope_visuals as ev

# RandomPanel datasets live in config_new.yaml (the active config for this work),
# not the legacy config.yaml.
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "config_new.yaml"
DATASET = "RandomPanel-Training-24-10"
WRONG_ODOR = "Linalool"
CORRECT_LABEL = "Isoamyl Acetate"


@pytest.fixture()
def restore_protocol_and_remap():
    """Snapshot and restore the module-global protocol + remap state."""
    saved_protocol = ev.get_protocol()
    saved_remap = {ds: dict(m) for ds, m in ev._DATASET_ODOR_REMAP.items()}
    try:
        yield
    finally:
        ev.set_protocol(saved_protocol)
        ev.set_dataset_odor_remap(saved_remap)


def test_config_declares_isoamyl_acetate_remap() -> None:
    """The shipped config must relabel this dataset's Linalool -> Isoamyl Acetate."""
    settings = load_settings(CONFIG_PATH)
    override = settings.dataset_overrides.get(DATASET)
    assert override is not None, (
        f"no dataset_overrides entry for {DATASET}; the odor correction is missing"
    )
    assert override.odor_remap.get(WRONG_ODOR) == CORRECT_LABEL


def test_display_label_uses_isoamyl_acetate(restore_protocol_and_remap) -> None:
    """End-to-end: the config remap relabels the figure odor for this dataset."""
    settings = load_settings(CONFIG_PATH)
    # Mimic run_workflows: register the config's remap, run under v2 (RandomPanel).
    ev.set_dataset_odor_remap(
        {
            ds: dict(ov.odor_remap)
            for ds, ov in settings.dataset_overrides.items()
            if ov.odor_remap
        }
    )
    ev.set_protocol("v2")

    # The Linalool trials now render as Isoamyl Acetate.
    assert ev._display_odor(DATASET, "training_6_Linalool") == CORRECT_LABEL
    assert ev._display_odor(DATASET, "training_13_Linalool") == CORRECT_LABEL


def test_remap_is_scoped_to_this_dataset_and_odor(restore_protocol_and_remap) -> None:
    """Correction must not leak to other odors or other datasets."""
    ev.set_protocol("v2")
    ev.set_dataset_odor_remap({DATASET: {WRONG_ODOR: CORRECT_LABEL}})

    # Other odors in the same dataset are untouched.
    assert ev._display_odor(DATASET, "training_2_Citral") == "Citral"
    # Linalool in a *different* RandomPanel dataset stays Linalool.
    assert ev._display_odor("RandomPanel-Training-24-0.01", "training_3_Linalool") == "Linalool"


def test_apply_dataset_odor_remap_unit() -> None:
    """The substitution helper maps the display name for the configured dataset."""
    saved = {ds: dict(m) for ds, m in ev._DATASET_ODOR_REMAP.items()}
    try:
        ev.set_dataset_odor_remap({DATASET: {WRONG_ODOR: CORRECT_LABEL}})
        assert ev.apply_dataset_odor_remap(DATASET, WRONG_ODOR) == CORRECT_LABEL
        # Unconfigured odor/dataset pass through unchanged.
        assert ev.apply_dataset_odor_remap(DATASET, "Citral") == "Citral"
        assert ev.apply_dataset_odor_remap("Other-Dataset", WRONG_ODOR) == WRONG_ODOR
    finally:
        ev.set_dataset_odor_remap(saved)
