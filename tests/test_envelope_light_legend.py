"""Tests for the green "Light pulsing starts" legend gating in envelope_visuals.

The green light marker is sourced from each trial's sensors_output_*.csv (the
measured first light-on time). Control datasets have no optogenetic light, so
their trials carry no light-on time and no green line is ever drawn — the legend
must therefore NOT advertise a "Light pulsing starts" entry for them.

Conversely, a training dataset whose sensors logs provide a light-on time must
keep both the green line and its legend entry.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis import envelope_visuals as ev  # noqa: E402


X_MAX = 90.0


def test_no_legend_when_no_trial_has_light_on_time():
    # Control dataset: every trial's light_start_s is None (no light in sensors).
    assert (
        ev._should_show_light_legend("line", [None, None, None], X_MAX) is False
    )


def test_legend_when_at_least_one_trial_has_light_on_time():
    # Training dataset: sensors-derived light-on at ~35 s.
    assert (
        ev._should_show_light_legend("line", [None, 35.06, None], X_MAX) is True
    )


def test_no_legend_when_light_start_beyond_axis():
    # A light-on time past the visible x-axis draws no line, so no legend entry.
    assert ev._should_show_light_legend("line", [120.0], X_MAX) is False


def test_no_legend_for_non_line_modes():
    assert ev._should_show_light_legend("none", [35.0], X_MAX) is False
    assert ev._should_show_light_legend("paired-span", [35.0], X_MAX) is False


def test_no_discriminate_legend_when_no_discriminate_trial():
    # v2 protocol: all six training trials use the same odor — none discriminate.
    assert (
        ev._should_show_discriminate_legend("training", [False, False, False])
        is False
    )


def test_discriminate_legend_when_a_discriminate_trial_exists():
    assert (
        ev._should_show_discriminate_legend("training", [False, True, False])
        is True
    )


def test_no_discriminate_legend_for_testing():
    assert ev._should_show_discriminate_legend("testing", [True, True]) is False


def test_control_dataset_detected_by_name():
    # Control datasets carry no optogenetic light — detected from the dataset
    # name, not from any single trial's sensors log.
    assert ev._is_control_dataset("Hex-Control-24-0.01") is True
    assert ev._is_control_dataset("hex_control_24") is True


def test_light_training_dataset_is_not_control():
    assert ev._is_control_dataset("Hex-Training-24-0.01") is False
    assert ev._is_control_dataset("EB-Training") is False
