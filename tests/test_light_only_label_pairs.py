"""Light-only trial detection in envelope_visuals must follow the label, not
only the trial number.

The odor-pair protocol (PiCode config_v2_pair*_A.yaml) runs 16 pair trials
``testing_<N>_<odor1>_then_<odor2>`` and, since 2026-11-17, ends with a
light-only probe ``testing_17_lightonly``. The v2 rule "trial 9 is the light
trial" dropped pair trial 9 from every generic reaction plot and would treat the
probe as an odor trial. Every other protocol keeps the old behaviour.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.analysis import envelope_visuals as ev  # noqa: E402


@pytest.fixture(autouse=True)
def _restore_protocol():
    saved = ev.get_protocol()
    yield
    ev.set_protocol(saved)


@pytest.mark.parametrize("label, expected", [
    # unchanged v2 behaviour for every non-pair dataset
    ("testing_9_lightonly", True),
    ("testing_9_LightOnly", True),
    ("testing_9", True),                 # bare label (Hex-Control-24-0.01)
    ("testing_9_Citral", True),          # pinned in test_train_vs_ctrl_split
    ("testing_1_hexanol", False),
    ("testing_8_isoamylacetate-1", False),
    ("pretest_9_hexanol", False),
    # odor-pair datasets
    ("testing_9_isoamylacetate-1_then_citral-1", False),
    ("testing_9_paraffinoil-100_then_isoamylacetate-1", False),
    ("testing_16_isoamylacetate-1_then_isoamylacetate-1", False),
    ("testing_17_lightonly", True),
    ("testing_17_LightOnly", True),
])
def test_v2_light_only_label(label, expected):
    ev.set_protocol("v2")
    assert ev._is_light_only_label(label) is expected


@pytest.mark.parametrize("label, expected", [
    ("testing_11_citral", True),
    ("testing_11", True),
    ("testing_9_citral", False),
    ("testing_11_isoamylacetate-1_then_citral-1", False),
    ("testing_17_lightonly", True),
])
def test_legacy_light_only_label(label, expected):
    ev.set_protocol("legacy")
    assert ev._is_light_only_label(label) is expected


def test_drop_keeps_pair_9_and_drops_the_probe():
    ev.set_protocol("v2")
    trials = [f"testing_{n}_isoamylacetate-1_then_citral-1" for n in range(1, 17)]
    trials.append("testing_17_lightonly")
    kept = ev._drop_testing_11(trials)
    assert kept == trials[:16]


def test_probe_displays_as_light_control():
    ev.set_protocol("v2")
    assert ev._display_odor_v2("IAA-PairFirst-Control-1-1",
                               "testing_17_lightonly") == "Optogenetic Light Control"
