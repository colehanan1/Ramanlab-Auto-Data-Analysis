#!/usr/bin/env python3
"""Tests for random_panel light-only trailer trials in expand_config.py.

The light trials used to be hard-coded inside ``_build_random_panel_cycles``.
These tests pin down two things:
  1. The default (no ``light_trials`` key in the YAML) still emits the original
     1 Hz-solid + 5 Hz-pulse pair — backward compatibility.
  2. A ``light_trials`` list in the ``experiment`` block drives the trailer
     trials, in order, with the configured hz/duty.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "PiCode"))

from expand_config import expand_config


def _panel_cfg(experiment_extra=None):
    exp = {"mode": "random_panel"}
    if experiment_extra:
        exp.update(experiment_extra)
    return {"format": "v2", "experiment": exp}


def _light_steps(result):
    """Light trailer steps are the ones carrying a light_schedule."""
    steps = result["cycles"][0]["steps"]
    return [s for s in steps if "light_schedule" in s]


def test_default_light_trials():
    """No light_trials key → original 1 Hz solid + 5 Hz pulse pair."""
    result = expand_config(_panel_cfg())
    lights = _light_steps(result)
    assert [(s["light_hertz"], s["light_duty"]) for s in lights] == [
        (1.0, 100.0),
        (5.0, 50.0),
    ]
    print("PASS: test_default_light_trials")


def test_custom_light_trials_order_and_values():
    """light_trials list drives the trailer trials, in order, with hz/duty."""
    trials = [
        {"label": "Light_Solid_1Hz_100pct", "hertz": 1.0, "duty": 100.0},
        {"label": "Light_Pulse_5Hz_50pct", "hertz": 5.0, "duty": 50.0},
        {"label": "Light_Pulse_10Hz_50pct", "hertz": 10.0, "duty": 50.0},
        {"label": "Light_Pulse_20Hz_50pct", "hertz": 20.0, "duty": 50.0},
        {"label": "Light_Pulse_50Hz_50pct", "hertz": 50.0, "duty": 50.0},
    ]
    result = expand_config(_panel_cfg({"light_trials": trials}))
    lights = _light_steps(result)
    assert len(lights) == 5
    assert [(s["light_hertz"], s["light_duty"]) for s in lights] == [
        (1.0, 100.0), (5.0, 50.0), (10.0, 50.0), (20.0, 50.0), (50.0, 50.0)
    ]
    # Labels propagate to both the light step and its Start Recording marker.
    assert [s["odor_label"] for s in lights] == [t["label"] for t in trials]
    print("PASS: test_custom_light_trials_order_and_values")


def test_custom_light_trials_recording_markers():
    """Each light trial is bracketed by its own Start/Stop Recording pair."""
    trials = [
        {"label": "Light_Pulse_10Hz_50pct", "hertz": 10.0, "duty": 50.0},
        {"label": "Light_Pulse_20Hz_50pct", "hertz": 20.0, "duty": 50.0},
        {"label": "Light_Pulse_50Hz_50pct", "hertz": 50.0, "duty": 50.0},
    ]
    result = expand_config(_panel_cfg({"light_trials": trials}))
    steps = result["cycles"][0]["steps"]
    light_starts = [
        s for s in steps
        if s["name"] == "Start Recording"
        and str(s.get("odor_label", "")).startswith("Light_")
    ]
    assert len(light_starts) == 3
    print("PASS: test_custom_light_trials_recording_markers")


def _staircase_cfg(extra=None):
    sc = {"flash_duration": 5, "gaps": [10, 20, 30, 40, 50, 60], "pre": 30, "post": 30}
    if extra:
        sc.update(extra)
    return _panel_cfg({"light_staircase": sc})


def test_staircase_single_video():
    """The staircase probe is ONE recording: a single Start/Stop pair wrapping
    a single light step (not one recording per flash)."""
    result = expand_config(_staircase_cfg())
    steps = result["cycles"][0]["steps"]
    sc_starts = [
        s for s in steps
        if s["name"] == "Start Recording" and s.get("odor_label") == "Light_Staircase"
    ]
    sc_light = [
        s for s in steps
        if "light_schedule" in s and s.get("odor_label") == "Light_Staircase"
    ]
    assert len(sc_starts) == 1, "staircase must be a single video"
    assert len(sc_light) == 1, "staircase must be a single light step"
    print("PASS: test_staircase_single_video")


def test_staircase_segments_and_timing():
    """7 flashes (one per gap + a trailing flash), 5 s each, solid (duty 100),
    at the right offsets given pre=30 and gaps 10/20/30/40/50/60."""
    result = expand_config(_staircase_cfg())
    light = next(
        s for s in result["cycles"][0]["steps"]
        if "light_schedule" in s and s.get("odor_label") == "Light_Staircase"
    )
    segs = light["light_schedule"]
    assert len(segs) == 7
    assert light["light_duty"] == 100.0          # solid
    # Every flash is exactly 5 s long.
    assert all(s["end"] - s["start"] == 5 for s in segs)
    # Offsets: pre=30, then +5 flash +gap each time.
    expected_starts = [30, 45, 70, 105, 150, 205, 270]
    assert [s["start"] for s in segs] == expected_starts
    # Total step duration covers the last flash end (275) + post-roll (30).
    assert light["duration"] == 305
    print("PASS: test_staircase_segments_and_timing")


def test_no_staircase_when_absent():
    """No light_staircase key → no staircase recording is emitted."""
    result = expand_config(_panel_cfg())
    steps = result["cycles"][0]["steps"]
    assert not any(s.get("odor_label") == "Light_Staircase" for s in steps)
    print("PASS: test_no_staircase_when_absent")


# ── Pi-1 light-intensity ramp ─────────────────────────────────────────
# On Pi 1 (LDD-L dimming) the ramp REPLACES the freq/duty trailers; on Pi 2 the
# trailers run unchanged and the ramp is omitted entirely.

_DEFAULT_TRAILERS = [
    {"label": "Light_Solid_1Hz_100pct", "hertz": 1.0, "duty": 100.0},
    {"label": "Light_Pulse_5Hz_50pct", "hertz": 5.0, "duty": 50.0},
]


def _ramp_cfg(extra=None, trailers=True):
    exp = {}
    if trailers:
        exp["light_trials"] = _DEFAULT_TRAILERS
    ramp = {
        "enabled": True,
        "intensities": [10, 20, 30, 40, 50],
        "on_duration": 10,
        "gap": 30,
        "pre": 15,
        "post": 15,
        "gap_before": 245,
    }
    if extra:
        ramp.update(extra)
    exp["light_intensity_ramp"] = ramp
    return _panel_cfg(exp)


def _ramp_light_step(result):
    return [
        s for s in result["cycles"][0]["steps"]
        if "light_schedule" in s and s.get("odor_label") == "Light_Intensity_Ramp"
    ]


def test_intensity_ramp_pi1_single_video():
    """pi1=True + ramp block → one Start/Stop pair, one light step, 5 solid
    segments at brightness 10..50, correct offsets, total 200 s."""
    result = expand_config(_ramp_cfg(), pi1=True)
    steps = result["cycles"][0]["steps"]
    starts = [
        s for s in steps
        if s["name"] == "Start Recording" and s.get("odor_label") == "Light_Intensity_Ramp"
    ]
    ramp = _ramp_light_step(result)
    assert len(starts) == 1, "ramp must be a single video"
    assert len(ramp) == 1, "ramp must be a single light step"
    light = ramp[0]
    segs = light["light_schedule"]
    assert len(segs) == 5
    assert [s["brightness"] for s in segs] == [10, 20, 30, 40, 50]
    assert light["light_duty"] == 100.0                 # solid
    assert [s["start"] for s in segs] == [15, 55, 95, 135, 175]
    assert all(s["end"] - s["start"] == 10 for s in segs)
    assert light["duration"] == 200
    print("PASS: test_intensity_ramp_pi1_single_video")


def test_intensity_ramp_pi1_replaces_trailers():
    """pi1=True + ramp block → the freq/duty trailers are NOT emitted."""
    result = expand_config(_ramp_cfg(), pi1=True)
    labels = [
        str(s.get("odor_label", "")) for s in result["cycles"][0]["steps"]
        if "light_schedule" in s
    ]
    assert labels == ["Light_Intensity_Ramp"], f"expected ramp only, got {labels}"
    print("PASS: test_intensity_ramp_pi1_replaces_trailers")


def test_intensity_ramp_absent_on_pi2():
    """pi1=False + ramp block → no ramp; trailers run unchanged."""
    result = expand_config(_ramp_cfg(), pi1=False)
    steps = result["cycles"][0]["steps"]
    assert not any(s.get("odor_label") == "Light_Intensity_Ramp" for s in steps)
    light_labels = [s.get("odor_label") for s in steps if "light_schedule" in s]
    assert "Light_Solid_1Hz_100pct" in light_labels
    assert "Light_Pulse_5Hz_50pct" in light_labels
    print("PASS: test_intensity_ramp_absent_on_pi2")


def test_no_ramp_block_keeps_trailers_both_pis():
    """No light_intensity_ramp key → trailers on both Pis, no ramp anywhere."""
    for pi in (True, False):
        result = expand_config(_panel_cfg(), pi1=pi)
        steps = result["cycles"][0]["steps"]
        assert not any(s.get("odor_label") == "Light_Intensity_Ramp" for s in steps)
        assert any(s.get("odor_label") == "Light_Solid_1Hz_100pct" for s in steps)
    print("PASS: test_no_ramp_block_keeps_trailers_both_pis")


def test_intensity_ramp_disabled_falls_back_to_trailers():
    """enabled: false → treated as absent; Pi 1 runs trailers, no ramp."""
    result = expand_config(_ramp_cfg({"enabled": False}), pi1=True)
    steps = result["cycles"][0]["steps"]
    assert not any(s.get("odor_label") == "Light_Intensity_Ramp" for s in steps)
    assert any(s.get("odor_label") == "Light_Solid_1Hz_100pct" for s in steps)
    print("PASS: test_intensity_ramp_disabled_falls_back_to_trailers")


if __name__ == "__main__":
    test_default_light_trials()
    test_custom_light_trials_order_and_values()
    test_custom_light_trials_recording_markers()
    test_staircase_single_video()
    test_staircase_segments_and_timing()
    test_no_staircase_when_absent()
    test_intensity_ramp_pi1_single_video()
    test_intensity_ramp_pi1_replaces_trailers()
    test_intensity_ramp_absent_on_pi2()
    test_no_ramp_block_keeps_trailers_both_pis()
    test_intensity_ramp_disabled_falls_back_to_trailers()
    print("\n=== ALL TESTS PASSED ===")
