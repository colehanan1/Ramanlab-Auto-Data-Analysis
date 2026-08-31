"""Pins for config_new.yaml's dataset roster and per-dataset freeze state.

Two config decisions live here:

1. ``3Oct-Training-24-0.1-Manual`` is a registered dataset. The folder exists
   on disk (july_20/21 + august_11 batches) and runs the same rig plumbing as
   its ``3Oct-Training-24-0.1`` sibling, so it must appear in the ``datasets:``
   list (path expansion) and carry the same odor remap (figures pair cohorts by
   display label — a missing remap splits odors into unpaired columns; the
   remap content itself is pinned by test_3oct_odor_remap.py).

2. The three finished RandomPanel cohorts were thawed for FIGURES on
   2026-08-11, then RE-FROZEN on 2026-08-27 when every dataset except the four
   ``*-Sensitivity-*`` cohorts was frozen for both data and figures. Thawing one
   again is a deliberate act — flip it back here and in config_new.yaml
   together, or the figures silently stop updating.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from fbpipe.config import load_settings

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "config_new.yaml"

MANUAL_DATASET = "3Oct-Training-24-0.1-Manual"

# Re-frozen 2026-08-27 along with every other finished dataset.
REFROZEN_FIGURE_DATASETS = (
    "RandomPanel-Training-24-10",
    "RandomPanel-24-1",
    "RandomPanel-24-0.1",
)

STILL_FROZEN_RANDOMPANEL = (
    "RandomPanel-Training-24-0.01",
    "RandomPanel-Training-24-0.01-Gr5aOld",
    "RandomPanel-Control-24-10",
)


def _raw_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_manual_3oct_dataset_is_registered() -> None:
    """The Manual cohort must be in the datasets: list so its roots expand."""
    assert MANUAL_DATASET in _raw_config()["datasets"]


def test_manual_3oct_dataset_carries_the_3oct_remap() -> None:
    """Same rig plumbing as 3Oct-Training-24-0.1 -> identical remap block."""
    settings = load_settings(CONFIG_PATH)
    manual = settings.dataset_overrides.get(MANUAL_DATASET)
    assert manual is not None, (
        f"no dataset_overrides entry for {MANUAL_DATASET}; the odor correction "
        "is missing and its figures would carry uncorrected labels"
    )
    sibling = settings.dataset_overrides["3Oct-Training-24-0.1"]
    assert dict(manual.odor_remap) == dict(sibling.odor_remap)


@pytest.mark.parametrize("dataset", REFROZEN_FIGURE_DATASETS)
def test_randompanel_figures_refrozen(dataset: str) -> None:
    """Frozen for both: a run touches only the *-Sensitivity-* cohorts."""
    settings = load_settings(CONFIG_PATH)
    ov = settings.dataset_overrides[dataset]
    assert ov.freeze_figures is True, (
        f"{dataset} figures thawed; every non-Sensitivity dataset is frozen "
        f"as of 2026-08-27 — thaw deliberately, not by accident"
    )
    assert ov.freeze_data is True, (
        f"{dataset} raw data left the freeze cache"
    )


@pytest.mark.parametrize("dataset", STILL_FROZEN_RANDOMPANEL)
def test_other_randompanel_datasets_stay_fully_frozen(dataset: str) -> None:
    """The thaw is scoped: the -0.01 pair and the empty Control stay frozen."""
    settings = load_settings(CONFIG_PATH)
    ov = settings.dataset_overrides[dataset]
    assert ov.freeze_data is True
    assert ov.freeze_figures is True
