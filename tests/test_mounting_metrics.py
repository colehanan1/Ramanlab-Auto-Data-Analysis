"""The mounting / posture / abdomen-movement metrics and their statistics.

``fbpipe.analysis.mounting`` is pure maths, so these tests pin the definitions a
figure silently inherits rather than any file layout:

* the head frame -- "forward" is away from the anchor, so a mirrored rig reads
  the same; the pitch angles take the ABSOLUTE x difference for the same reason;
* the stimulus window -- a LightOnly trial carries a filler ``odor_on_frame``
  (1201) that must never become its onset;
* the run / event counters -- NaN breaks a run and a run is clipped to its window;
* the score table's duplicate rule -- exact duplicates collapse, conflicting
  scores are dropped and counted;
* the permutation tests -- shuffle only within a stratum (or within a fly), so a
  stratum-level (fly-level) confound alone cannot produce a small p.
"""
from __future__ import annotations

import itertools
import math
import os
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from fbpipe.analysis import mounting as M  # noqa: E402
from fbpipe.geometry.metrics import interior_angle  # noqa: E402


def _excursion(n=100, at=20, length=12, height=0.4, base=0.0):
    """A flat trace with one rectangular excursion of ``length`` frames."""
    x = np.full(n, base, dtype=float)
    x[at:at + length] = base + height
    return x


# ---------------------------------------------------------------------------
# Head frame: eye -> proboscis
# ---------------------------------------------------------------------------

def test_a_proboscis_behind_and_below_the_eye_has_negative_theta():
    # anchor on the left edge, so forward is +x and a proboscis 36 px left is behind
    dist, theta = M.head_frame_vector(500.0, 300.0, 464.0, 371.0, 0.0)
    assert theta == pytest.approx(-26.9, abs=0.1)
    assert dist == pytest.approx(79.6, abs=0.1)


def test_the_mirrored_frame_gives_the_same_theta_and_distance():
    dist, theta = M.head_frame_vector(500.0, 300.0, 464.0, 371.0, 0.0)
    dist_m, theta_m = M.head_frame_vector(500.0, 300.0, 536.0, 371.0, 1080.0)
    assert theta_m == pytest.approx(-26.9, abs=0.1)
    assert theta_m == pytest.approx(theta, abs=1e-9)
    assert dist_m == pytest.approx(dist, abs=1e-9)


@pytest.mark.parametrize("anchor_x, prob_x", [(0.0, 536.0), (1080.0, 464.0)])
def test_a_proboscis_ahead_of_the_eye_has_positive_theta_on_either_rig(anchor_x, prob_x):
    _, theta = M.head_frame_vector(500.0, 300.0, prob_x, 371.0, anchor_x)
    assert theta == pytest.approx(26.9, abs=0.1)


def test_a_proboscis_straight_below_the_eye_has_zero_theta():
    dist, theta = M.head_frame_vector(500.0, 300.0, 500.0, 371.0, 0.0)
    assert theta == pytest.approx(0.0, abs=1e-9)
    assert dist == pytest.approx(71.0)


def test_the_frame_midline_decides_which_side_the_anchor_is_on():
    # an anchor at x=300 is on the left of a 1080-wide frame but on the right of a 400-wide one
    _, theta_wide = M.head_frame_vector(500.0, 300.0, 536.0, 371.0, 300.0)
    _, theta_narrow = M.head_frame_vector(500.0, 300.0, 536.0, 371.0, 300.0, frame_mid=200.0)
    assert theta_wide == pytest.approx(26.9, abs=0.1)
    assert theta_narrow == pytest.approx(-26.9, abs=0.1)


def test_the_default_midline_is_540_and_an_anchor_on_it_counts_as_the_right_side():
    def theta_at(anchor):
        return M.head_frame_vector(500.0, 300.0, 536.0, 371.0, anchor)[1]

    assert theta_at(539.9) == pytest.approx(26.9, abs=0.1)
    assert theta_at(540.0) == pytest.approx(-26.9, abs=0.1)
    assert theta_at(540.1) == pytest.approx(-26.9, abs=0.1)


def test_head_frame_vector_works_on_whole_columns_and_keeps_missing_frames_missing():
    dist, theta = M.head_frame_vector(
        [500.0, 500.0, np.nan, 500.0], [300.0] * 4, [464.0, 500.0, 464.0, np.nan], [371.0] * 4, 0.0)
    assert dist[0] == pytest.approx(79.6, abs=0.1) and theta[0] == pytest.approx(-26.9, abs=0.1)
    assert dist[1] == pytest.approx(71.0) and theta[1] == pytest.approx(0.0, abs=1e-9)
    assert np.isnan(dist[2]) and np.isnan(theta[2])
    assert np.isnan(dist[3]) and np.isnan(theta[3])


def test_an_unknown_anchor_gives_no_theta_but_still_a_distance():
    # without the anchor the forward side is unknown; guessing it would flip the sign silently
    dist, theta = M.head_frame_vector(500.0, 300.0, 464.0, 371.0, np.nan)
    assert dist == pytest.approx(79.6, abs=0.1)
    assert np.isnan(theta)


# ---------------------------------------------------------------------------
# Body pitch and tip pitch
# ---------------------------------------------------------------------------

def test_body_pitch_is_positive_when_the_abdomen_is_below_the_eye():
    assert M.body_pitch_deg(100.0, 100.0, 300.0, 120.0) == pytest.approx(5.71, abs=0.01)


def test_body_pitch_is_the_same_on_a_mirrored_rig():
    assert M.body_pitch_deg(100.0, 100.0, -100.0, 120.0) == pytest.approx(5.71, abs=0.01)
    assert M.body_pitch_deg(100.0, 100.0, -100.0, 120.0) == pytest.approx(
        M.body_pitch_deg(100.0, 100.0, 300.0, 120.0), abs=1e-12)


def test_body_pitch_is_negative_when_the_abdomen_is_above_the_eye():
    assert M.body_pitch_deg(100.0, 100.0, 300.0, 80.0) == pytest.approx(-5.71, abs=0.01)


def test_body_pitch_works_per_frame_and_keeps_missing_frames_missing():
    out = M.body_pitch_deg([100.0, 100.0, np.nan], [100.0] * 3, [300.0, 300.0, 300.0], [120.0, np.nan, 120.0])
    assert out.shape == (3,)
    assert out[0] == pytest.approx(5.71, abs=0.01)
    assert np.isnan(out[1]) and np.isnan(out[2])


def test_tip_pitch_is_plus_45_for_a_tip_down_and_behind_the_abdomen():
    assert M.tip_pitch_deg(300.0, 120.0, 420.0, 240.0) == pytest.approx(45.0, abs=1e-9)


def test_tip_pitch_is_the_same_when_the_rig_is_mirrored():
    assert M.tip_pitch_deg(300.0, 120.0, 180.0, 240.0) == pytest.approx(45.0, abs=1e-9)


def test_tip_pitch_is_negative_when_the_tip_is_above_the_abdomen():
    assert M.tip_pitch_deg(300.0, 120.0, 420.0, 0.0) == pytest.approx(-45.0, abs=1e-9)


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("mirror", [1.0, -1.0])
def test_the_gap_between_tip_pitch_and_body_pitch_is_the_geometry_stage_bend(seed, mirror):
    """For an in-plane fly (head, abdomen, tip run away from the head horizontally)
    |tip pitch - body pitch| is 180 minus the interior angle `abdomen_bend_deg` uses."""
    rng = np.random.default_rng(seed)
    eye = rng.uniform(0.0, 100.0, 2)
    abd = eye + np.array([rng.uniform(100.0, 300.0), rng.uniform(-60.0, 60.0)])
    tip = abd + np.array([rng.uniform(20.0, 200.0), rng.uniform(-150.0, 150.0)])
    flip = np.array([mirror, 1.0])
    eye, abd, tip = eye * flip, abd * flip, tip * flip
    gap = abs(M.tip_pitch_deg(abd[0], abd[1], tip[0], tip[1])
              - M.body_pitch_deg(eye[0], eye[1], abd[0], abd[1]))
    assert gap == pytest.approx(180.0 - interior_angle(eye, abd, tip), abs=1e-6)


# ---------------------------------------------------------------------------
# Angle between two vectors
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("u, v, expected", [
    ((1.0, 0.0), (0.0, 1.0), 90.0),
    ((1.0, 0.0), (-1.0, 0.0), 180.0),
    ((2.0, 0.0), (5.0, 0.0), 0.0),
    ((1.0, 0.0), (1.0, 1.0), 45.0),
    ((1.0, 0.0), (1.0, -1.0), 45.0),        # unsigned
    ((0.0, -3.0), (4.0, 0.0), 90.0),
])
def test_angle_between_is_the_unsigned_angle_in_degrees(u, v, expected):
    assert M.angle_between_deg(u, v) == pytest.approx(expected, abs=1e-9)


def test_angle_between_is_accurate_for_nearly_parallel_vectors():
    # an arccos formulation returns 0 or noise here
    assert M.angle_between_deg((1.0, 0.0), (1.0, 1e-9)) == pytest.approx(math.degrees(1e-9), rel=1e-6)


@pytest.mark.parametrize("u, v", [((0.0, 0.0), (1.0, 0.0)), ((np.nan, 0.0), (1.0, 0.0)), ((1.0, 0.0), (1.0, np.nan))])
def test_angle_between_is_nan_for_a_zero_or_missing_vector(u, v):
    assert math.isnan(M.angle_between_deg(u, v))


# ---------------------------------------------------------------------------
# Stimulus window
# ---------------------------------------------------------------------------

def test_a_light_only_trial_never_uses_its_filler_odor_frame():
    trial = {"has_odor": False, "odor_on_frame": 1201, "light_on_s": 5.0, "light_off_s": 10.0}
    assert M.stimulus_frames(trial, fps=40.0) == (200, 400)


def test_a_training_trial_starts_at_whichever_stimulus_comes_first():
    trial = {"has_odor": True, "odor_on_frame": 300, "light_on_s": 4.0}
    onset, offset = M.stimulus_frames(trial, fps=40.0)
    assert onset == 160
    assert offset is None


def test_an_odor_only_trial_runs_from_odor_on_to_odor_off():
    trial = {"has_odor": True, "odor_on_frame": 300, "odor_off_frame": 700}
    assert M.stimulus_frames(trial, fps=40.0) == (300, 700)


def test_the_stimulus_ends_when_the_last_of_odor_and_light_ends():
    base = {"has_odor": True, "odor_on_frame": 300, "odor_off_frame": 700, "light_on_s": 4.0}
    assert M.stimulus_frames({**base, "light_off_s": 20.0}, fps=40.0) == (160, 800)
    assert M.stimulus_frames({**base, "light_off_s": 10.0}, fps=40.0) == (160, 700)


def test_a_trial_without_an_odor_ignores_the_odor_off_frame_too():
    trial = {"has_odor": False, "odor_on_frame": 1201, "odor_off_frame": 1500, "light_on_s": 5.0}
    assert M.stimulus_frames(trial, fps=40.0) == (200, None)


def test_a_trial_with_nothing_known_has_no_window():
    assert M.stimulus_frames({}, fps=40.0) == (None, None)
    assert M.stimulus_frames({"has_odor": False, "odor_on_frame": 1201}, fps=40.0) == (None, None)
    assert M.stimulus_frames({"has_odor": True, "light_on_s": None, "light_off_s": None}, fps=40.0) == (None, None)


def test_missing_values_in_a_table_row_count_as_absent():
    # an unknown has_odor is not "odor": the earlier odor_on_frame must not win
    row = pd.Series({"has_odor": np.nan, "odor_on_frame": 100.0, "odor_off_frame": np.nan,
                     "light_on_s": 4.0, "light_off_s": np.nan})
    assert M.stimulus_frames(row, fps=40.0) == (160, None)
    row = pd.Series({"has_odor": True, "odor_on_frame": 300.0, "odor_off_frame": np.nan,
                     "light_on_s": np.nan, "light_off_s": np.nan})
    assert M.stimulus_frames(row, fps=40.0) == (300, None)


def test_stimulus_frames_are_plain_integers():
    for trial in ({"has_odor": True, "odor_on_frame": 300.0, "odor_off_frame": 700.0},
                  {"has_odor": False, "light_on_s": 5.0, "light_off_s": 10.0}):
        onset, offset = M.stimulus_frames(trial, fps=40.0)
        assert type(onset) is int and type(offset) is int


def test_light_times_are_rounded_to_the_nearest_frame():
    assert M.stimulus_frames({"light_on_s": 4.35, "light_off_s": 4.29}, fps=25.0) == (109, 107)


def test_the_windows_split_the_trial_at_stimulus_onset_and_offset():
    pre, stim = M.window_slice(1000, 200, 400)
    assert (pre, stim) == (slice(0, 200), slice(200, 400))


def test_a_stimulus_with_no_known_end_runs_to_the_last_frame():
    assert M.window_slice(1000, 200, None) == (slice(0, 200), slice(200, 1000))


def test_a_trial_without_an_onset_has_two_empty_windows():
    pre, stim = M.window_slice(1000, None, None)
    assert pre == slice(0, 0) and stim == slice(0, 0)
    frames = np.arange(1000)
    assert frames[pre].size == 0 and frames[stim].size == 0


# ---------------------------------------------------------------------------
# Rest level, held out, held sub-type, retracted reference
# ---------------------------------------------------------------------------

def test_rest_level_is_the_lower_decile_of_the_finite_frames():
    x = np.arange(100.0)
    assert M.rest_level(x) == pytest.approx(np.quantile(x, 0.10))
    assert M.rest_level(np.concatenate([x, np.full(30, np.nan)])) == pytest.approx(np.quantile(x, 0.10))
    assert M.rest_level(np.arange(10.0), min_finite=5, q=0.5) == pytest.approx(4.5)


def test_rest_level_needs_enough_finite_frames():
    assert math.isnan(M.rest_level(np.arange(39.0)))
    assert not math.isnan(M.rest_level(np.arange(40.0)))
    assert math.isnan(M.rest_level(np.full(100, np.nan)))
    assert math.isnan(M.rest_level(np.concatenate([np.arange(39.0), np.full(200, np.nan)])))


def _pre_window(frac_out, n=100, ref=1.0, out=0.3):
    dist = np.full(n, ref)
    dist[:int(round(frac_out * n))] = ref + out
    return dist


def test_a_fly_held_out_for_85_percent_of_the_window_is_held_out():
    assert M.held_out(_pre_window(0.85), ref_eu=1.0, margin_eu=0.2, frac_min=0.8) is True


def test_a_fly_held_out_for_70_percent_of_the_window_is_not():
    assert M.held_out(_pre_window(0.70), ref_eu=1.0, margin_eu=0.2, frac_min=0.8) is False


def test_held_out_counts_the_frame_that_sits_exactly_on_the_margin_and_the_exact_fraction():
    ref, margin = 1.0, 0.2
    assert M.held_out(np.full(10, ref + margin), ref_eu=ref, margin_eu=margin, frac_min=1.0) is True
    assert M.held_out(_pre_window(0.80), ref_eu=ref, margin_eu=margin, frac_min=0.8) is True
    assert M.held_out(np.full(10, ref + margin - 1e-9), ref_eu=ref, margin_eu=margin, frac_min=0.1) is False


def test_held_out_is_judged_on_the_finite_frames_only():
    dist = np.concatenate([np.full(50, 1.3), np.full(50, np.nan)])
    assert M.held_out(dist, ref_eu=1.0, margin_eu=0.2, frac_min=0.8) is True
    dist = np.concatenate([np.full(10, 1.3), np.full(10, 1.0), np.full(80, np.nan)])
    assert M.held_out(dist, ref_eu=1.0, margin_eu=0.2, frac_min=0.8) is False


def test_a_window_with_no_finite_frames_is_not_held_out():
    assert M.held_out(np.full(50, np.nan), ref_eu=1.0, margin_eu=0.2, frac_min=0.0) is False
    assert M.held_out(np.array([]), ref_eu=1.0, margin_eu=0.2, frac_min=0.5) is False
    assert M.held_out(np.full(50, 5.0), ref_eu=np.nan, margin_eu=0.2, frac_min=0.0) is False


def test_held_subtype_names_back_down_and_forward():
    assert M.held_subtype(-13.0) == "held_back"
    assert M.held_subtype(1.1) == "held_down"
    assert M.held_subtype(13.0) == "held_forward"


def test_held_subtype_boundaries_belong_to_held_down_and_thresholds_are_configurable():
    assert M.held_subtype(-10.0) == "held_down"
    assert M.held_subtype(10.0) == "held_down"
    assert M.held_subtype(-13.0, back_deg=-20.0) == "held_down"
    assert M.held_subtype(13.0, forward_deg=5.0) == "held_forward"


def test_held_subtype_always_answers_with_a_label_even_for_a_missing_angle():
    assert isinstance(M.held_subtype(float("nan")), str)


def test_the_retracted_reference_is_the_median_of_the_finite_minima():
    assert M.retracted_reference(pd.Series([1.0, 3.0, np.nan, 2.0])) == pytest.approx(2.0)
    assert M.retracted_reference(np.array([1.0, 2.0, 10.0, 20.0, np.nan])) == pytest.approx(6.0)


def test_the_retracted_reference_is_nan_without_any_finite_minimum():
    assert math.isnan(M.retracted_reference(pd.Series([np.nan, np.nan])))
    assert math.isnan(M.retracted_reference(pd.Series([], dtype=float)))


def test_anchor_spread_is_the_largest_pairwise_distance():
    assert M.anchor_spread_px(np.array([(0, 0), (3, 4), (0, 0)])) == pytest.approx(5.0)
    # the widest pair is not first-to-last
    assert M.anchor_spread_px(np.array([(0.0, 0.0), (10.0, 0.0), (5.0, 0.0)])) == pytest.approx(10.0)


def test_anchor_spread_is_zero_for_one_or_no_point_and_ignores_missing_rows():
    assert M.anchor_spread_px(np.array([(7.0, 9.0)])) == 0.0
    assert M.anchor_spread_px(np.empty((0, 2))) == 0.0
    assert M.anchor_spread_px(np.array([(0.0, 0.0), (np.nan, 5.0), (3.0, 4.0)])) == pytest.approx(5.0)
    assert M.anchor_spread_px(np.full((4, 2), np.nan)) == 0.0


# ---------------------------------------------------------------------------
# Runs and events
# ---------------------------------------------------------------------------

def test_a_12_frame_excursion_is_one_run_at_min_frames_12_and_none_at_13():
    x = _excursion()
    assert M.runs_above(x, 0.2, min_frames=12) == [(20, 32)]
    assert M.runs_above(x, 0.2, min_frames=13) == []


def test_an_excursion_after_the_window_end_is_not_counted():
    assert M.runs_above(_excursion(), 0.2, min_frames=12, end=15) == []


def test_a_run_is_clipped_to_the_window_before_its_length_is_judged():
    x = _excursion()
    assert M.runs_above(x, 0.2, min_frames=12, end=26) == []
    assert M.runs_above(x, 0.2, min_frames=6, end=26) == [(20, 26)]
    assert M.runs_above(x, 0.2, min_frames=6, start=26) == [(26, 32)]
    assert M.runs_above(x, 0.2, min_frames=12, start=26) == []


def test_two_excursions_split_by_one_sub_threshold_frame_are_two_runs():
    x = _excursion(length=25)
    x[32] = 0.0
    assert M.runs_above(x, 0.2, min_frames=12) == [(20, 32), (33, 45)]


def test_a_missing_frame_breaks_a_run_it_is_never_bridged():
    x = _excursion()
    x[26] = np.nan
    assert M.runs_above(x, 0.2, min_frames=12) == []
    assert M.runs_above(x, 0.2, min_frames=6) == [(20, 26)]


def test_a_frame_sitting_exactly_on_the_level_is_not_above_it():
    assert M.runs_above(np.full(30, 0.2), 0.2, min_frames=3) == []
    assert M.runs_below(np.full(30, 0.2), 0.2, min_frames=3) == []


def test_runs_ignore_a_window_that_lies_outside_the_trace():
    x = _excursion()
    assert M.runs_above(x, 0.2, min_frames=3, start=500) == []
    assert M.runs_above(x, 0.2, min_frames=3, end=0) == []
    assert M.runs_above(x, 0.2, min_frames=12, start=0, end=10_000) == [(20, 32)]
    assert M.runs_above(np.array([]), 0.2, min_frames=3) == []


def test_runs_are_plain_integer_pairs():
    ((i0, i1),) = M.runs_above(_excursion(), 0.2, min_frames=12)
    assert type(i0) is int and type(i1) is int


def test_a_12_frame_dip_is_one_run_below_at_min_frames_12_and_none_at_13():
    x = -_excursion()
    assert M.runs_below(x, -0.2, min_frames=12) == [(20, 32)]
    assert M.runs_below(x, -0.2, min_frames=13) == []


def test_runs_below_mirror_the_window_split_and_missing_frame_rules():
    x = -_excursion(length=25)
    x[32] = 0.0
    assert M.runs_below(x, -0.2, min_frames=12) == [(20, 32), (33, 45)]
    assert M.runs_below(x, -0.2, min_frames=12, end=15) == []
    x = -_excursion()
    x[26] = np.nan
    assert M.runs_below(x, -0.2, min_frames=12) == []


def test_a_curl_is_a_sustained_rise_above_the_resting_bend():
    bend = np.full(200, 5.0)
    bend[50:65] = 20.0                         # +15 deg for 15 frames, rise needed is 10
    assert M.curl_events(bend, 5.0, rise_deg=10.0, min_frames=12) == (1, 0)


def test_a_straightening_is_a_sustained_dip_below_the_resting_bend():
    bend = np.full(200, 5.0)
    bend[50:65] = -8.0
    assert M.curl_events(bend, 5.0, rise_deg=10.0, min_frames=12) == (0, 1)


def test_an_excursion_outside_the_window_is_not_a_curl_event():
    bend = np.full(200, 5.0)
    bend[150:165] = 20.0
    assert M.curl_events(bend, 5.0, rise_deg=10.0, min_frames=12, end=100) == (0, 0)
    assert M.curl_events(bend, 5.0, rise_deg=10.0, min_frames=12, start=100, end=200) == (1, 0)
    bend = np.full(200, 5.0)
    bend[50:65] = -8.0
    assert M.curl_events(bend, 5.0, rise_deg=10.0, min_frames=12, start=100) == (0, 0)


def test_a_short_or_shallow_excursion_is_not_a_curl_event():
    short = np.full(200, 5.0)
    short[50:61] = 20.0                        # 11 frames
    assert M.curl_events(short, 5.0, rise_deg=10.0, min_frames=12) == (0, 0)
    shallow = np.full(200, 5.0)
    shallow[50:65] = 14.0                      # +9 deg, rise needed is 10
    assert M.curl_events(shallow, 5.0, rise_deg=10.0, min_frames=12) == (0, 0)


def test_curls_and_straightenings_are_counted_together_over_the_window():
    bend = np.full(300, 5.0)
    bend[20:40] = 25.0
    bend[60:80] = -10.0
    bend[100:120] = 25.0
    assert M.curl_events(bend, 5.0, rise_deg=10.0, min_frames=12) == (2, 1)


def test_curl_events_cannot_be_judged_against_a_missing_resting_bend():
    bend = np.full(200, 20.0)
    assert M.curl_events(bend, np.nan, rise_deg=10.0, min_frames=12) == (0, 0)


# ---------------------------------------------------------------------------
# Point speed (eye units per second)
# ---------------------------------------------------------------------------

def test_a_tip_moving_48_px_per_frame_at_40_fps_with_a_48_px_eye_is_40_eu_per_s():
    n = 60
    speed = M.point_speed_eu_s(np.arange(n) * 48.0, np.full(n, 100.0), eye_scale_px=48.0, fps=40.0)
    assert speed.shape == (n,)
    assert np.isnan(speed[0])
    # the two frames at either end see a truncated median window, so check the interior
    assert speed[3:-3] == pytest.approx(40.0)


def test_diagonal_motion_is_measured_along_the_path():
    n = 40
    speed = M.point_speed_eu_s(np.arange(n) * 30.0, np.arange(n) * 40.0, eye_scale_px=25.0, fps=10.0)
    assert speed[3:-3] == pytest.approx(20.0)             # 50 px/frame = 2 eu/frame * 10 fps


def test_a_missing_frame_gives_missing_speeds_on_both_sides_never_an_interpolation():
    x = np.arange(60) * 48.0
    x[20] = np.nan
    speed = M.point_speed_eu_s(x, np.full(60, 100.0), eye_scale_px=48.0, fps=40.0)
    assert set(np.flatnonzero(np.isnan(speed))) == {0, 20, 21}


def test_a_gap_of_three_frames_blanks_the_four_speeds_that_touch_it():
    x = np.arange(60) * 48.0
    x[20:23] = np.nan
    speed = M.point_speed_eu_s(x, np.full(60, 100.0), eye_scale_px=48.0, fps=40.0)
    assert set(np.flatnonzero(np.isnan(speed))) == {0, 20, 21, 22, 23}


def test_a_frame_missing_in_one_coordinate_is_missing_as_a_point():
    y = np.full(60, 100.0)
    y[20] = np.nan
    speed = M.point_speed_eu_s(np.arange(60) * 48.0, y, eye_scale_px=48.0, fps=40.0)
    assert set(np.flatnonzero(np.isnan(speed))) == {0, 20, 21}


def test_the_median_is_centred_so_a_step_is_not_reported_late():
    x = np.r_[np.zeros(30), np.full(30, 96.0)]
    speed = M.point_speed_eu_s(x, np.zeros(60), eye_scale_px=48.0, fps=40.0)
    assert int(np.nanargmax(speed)) == 30
    assert speed[30] == pytest.approx(96.0 / 48.0 * 40.0)


def test_a_one_frame_30_px_jitter_on_a_still_point_is_removed_by_the_median():
    x = np.full(60, 200.0)
    x[30] += 30.0
    speed = M.point_speed_eu_s(x, np.full(60, 100.0), eye_scale_px=48.0, fps=40.0)
    assert np.nanmax(np.abs(speed)) == 0.0


def test_the_default_five_frame_median_also_removes_a_two_frame_jitter():
    x = np.full(60, 200.0)
    x[30:32] += 30.0
    speed = M.point_speed_eu_s(x, np.full(60, 100.0), eye_scale_px=48.0, fps=40.0)
    assert np.nanmax(np.abs(speed)) == 0.0
    three = M.point_speed_eu_s(x, np.full(60, 100.0), eye_scale_px=48.0, fps=40.0, smooth_frames=3)
    assert np.nanmax(three) > 0.0                          # a 3-frame median would let it through


def test_without_smoothing_the_same_jitter_shows_up_as_speed():
    x = np.full(60, 200.0)
    x[30] += 30.0
    speed = M.point_speed_eu_s(x, np.full(60, 100.0), eye_scale_px=48.0, fps=40.0, smooth_frames=1)
    assert speed[30] == pytest.approx(30.0 / 48.0 * 40.0)
    assert speed[31] == pytest.approx(30.0 / 48.0 * 40.0)


@pytest.mark.parametrize("bad_scale", [0.0, -48.0, np.nan, np.inf])
def test_an_unusable_eye_scale_gives_no_speeds(bad_scale):
    speed = M.point_speed_eu_s(np.arange(30) * 48.0, np.zeros(30), eye_scale_px=bad_scale, fps=40.0)
    assert speed.shape == (30,) and np.isnan(speed).all()


def test_a_trace_too_short_for_a_speed_is_all_missing():
    assert np.isnan(M.point_speed_eu_s([1.0], [1.0], eye_scale_px=48.0, fps=40.0)).all()
    assert M.point_speed_eu_s([], [], eye_scale_px=48.0, fps=40.0).shape == (0,)


def test_point_speed_leaves_its_inputs_alone():
    x = np.arange(30) * 48.0
    x[5] = np.nan
    before = x.copy()
    M.point_speed_eu_s(x, np.zeros(30), eye_scale_px=48.0, fps=40.0)
    assert np.array_equal(x, before, equal_nan=True)


# ---------------------------------------------------------------------------
# Spearman and coupling
# ---------------------------------------------------------------------------

def test_spearman_of_a_monotone_relation_is_plus_or_minus_one():
    rho, p = M.spearman([1, 2, 3, 4, 5, 6], [2, 4, 8, 16, 32, 64])
    assert rho == pytest.approx(1.0) and p < 0.01
    rho, _ = M.spearman([1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1])
    assert rho == pytest.approx(-1.0)


def test_spearman_agrees_with_scipy_on_a_noisy_sample():
    rng = np.random.default_rng(3)
    x = rng.normal(size=60)
    y = 0.5 * x + rng.normal(size=60)
    ref = stats.spearmanr(x, y)
    rho, p = M.spearman(x, y)
    assert rho == pytest.approx(ref.statistic, abs=1e-12)
    assert p == pytest.approx(ref.pvalue, abs=1e-12)


def test_spearman_drops_a_pair_when_either_member_is_missing():
    x = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0])
    y = np.array([1.0, 3.0, 2.0, np.nan, 5.0, 7.0, 6.0])
    keep = [0, 1, 4, 5, 6]
    ref = stats.spearmanr(x[keep], y[keep])
    rho, p = M.spearman(x, y)
    assert (rho, p) == pytest.approx((ref.statistic, ref.pvalue))


def test_spearman_needs_three_pairs_and_some_variation():
    for x, y in [([1.0, 2.0], [1.0, 2.0]), ([1.0, 1.0, 1.0, 1.0], [1.0, 2.0, 3.0, 4.0]),
                 ([1.0, 2.0, 3.0, 4.0], [7.0, 7.0, 7.0, 7.0]), ([], []),
                 ([1.0, np.nan, 3.0, np.nan], [1.0, 2.0, np.nan, 4.0])]:
        rho, p = M.spearman(x, y)
        assert math.isnan(rho) and math.isnan(p)


@pytest.mark.parametrize("y", [[1.0, 2.0], [1.0]])
def test_spearman_refuses_series_of_different_lengths(y):
    with pytest.raises(ValueError, match="same length"):
        M.spearman([1.0, 2.0, 3.0], y)
    with pytest.raises(ValueError, match="same length"):
        M.coupling_rho([1.0, 2.0, 3.0], y, min_frames=1)


def test_identical_ramps_couple_perfectly():
    ramp = np.arange(300.0)
    assert M.coupling_rho(ramp, ramp) == pytest.approx(1.0)
    assert M.coupling_rho(ramp, -ramp) == pytest.approx(-1.0)


def test_coupling_needs_200_frames_where_both_series_are_finite():
    ramp = np.arange(300.0)
    short = ramp.copy()
    short[150:] = np.nan                       # 150 finite frames
    assert math.isnan(M.coupling_rho(ramp, short))
    enough = ramp.copy()
    enough[250:] = np.nan                      # 250 finite frames
    assert M.coupling_rho(ramp, enough) == pytest.approx(1.0)
    assert math.isnan(M.coupling_rho(ramp, short, min_frames=151))
    assert M.coupling_rho(ramp, short, min_frames=150) == pytest.approx(1.0)


def test_coupling_counts_only_frames_where_both_series_are_finite():
    ramp = np.arange(400.0)
    a, b = ramp.copy(), ramp.copy()
    a[:120] = np.nan
    b[300:] = np.nan                           # both finite on frames 120..299 only = 180 frames
    assert math.isnan(M.coupling_rho(a, b))
    assert M.coupling_rho(a, b, min_frames=180) == pytest.approx(1.0)


def test_coupling_of_a_flat_series_is_missing_not_zero():
    assert math.isnan(M.coupling_rho(np.arange(300.0), np.full(300, 12.0)))


# ---------------------------------------------------------------------------
# Permutation tests
# ---------------------------------------------------------------------------

def _exact_stratified_p(x, y, strata):
    """Every within-stratum rearrangement of x, ranked by |Spearman rho| -- the definition."""
    x, y, strata = np.asarray(x, float), np.asarray(y, float), np.asarray(strata)
    groups = [np.flatnonzero(strata == label) for label in dict.fromkeys(strata.tolist())]
    observed = abs(stats.spearmanr(x, y).statistic)
    hits = total = 0
    for orders in itertools.product(*[itertools.permutations(range(g.size)) for g in groups]):
        shuffled = x.copy()
        for g, order in zip(groups, orders):
            shuffled[g] = x[g][list(order)]
        hits += abs(stats.spearmanr(shuffled, y).statistic) >= observed - 1e-12
        total += 1
    return hits / total


def test_a_stratum_level_confound_alone_is_not_significant():
    """x and y are both low in stratum A and both high in B, with nothing going on inside either."""
    x = np.r_[np.arange(10.0), 100.0 + np.arange(10.0)]
    y = np.r_[[4, 9, 2, 7, 0, 5, 8, 1, 6, 3], 100 + np.array([6, 1, 8, 3, 9, 0, 5, 2, 7, 4])].astype(float)
    strata = ["A"] * 10 + ["B"] * 10
    assert M.spearman(x, y)[1] < 0.01                      # pooled, the confound looks highly significant
    assert M.stratified_permutation_p(x, y, strata, n=4000) > 0.2
    # and an unstratified shuffle would have called it significant
    assert M.stratified_permutation_p(x, y, ["all"] * 20, n=4000) < 0.01


def test_an_association_inside_every_stratum_is_significant():
    x = np.r_[np.arange(10.0), 100.0 + np.arange(10.0)]
    strata = ["A"] * 10 + ["B"] * 10
    assert M.stratified_permutation_p(x, x + 3.0, strata, n=4000) < 0.01


def test_the_stratified_p_is_two_sided():
    x = np.r_[np.arange(10.0), 100.0 + np.arange(10.0)]
    assert M.stratified_permutation_p(x, -x, ["A"] * 10 + ["B"] * 10, n=2000) < 0.01


def test_the_stratified_p_matches_the_exact_within_stratum_distribution():
    x = [1, 2, 2, 3, 5, 6, 6, 7]                           # ties in both variables
    y = [2, 1, 3, 1, 5, 4, 5, 6]
    strata = list("AAAABBBB")
    exact = _exact_stratified_p(x, y, strata)              # 24 * 24 rearrangements
    assert 0.3 < exact < 0.8                               # a fixture that can tell a wrong null apart
    assert M.stratified_permutation_p(x, y, strata, n=20000, seed=1) == pytest.approx(exact, abs=0.02)


def test_the_stratified_p_counts_both_tails_of_the_within_stratum_distribution():
    x = [1, 2, 2, 4, 1, 3, 3, 4]                           # strata at the same level: the null is two-tailed
    y = [1, 3, 2, 4, 2, 1, 4, 3]
    strata = list("AAAABBBB")
    exact = _exact_stratified_p(x, y, strata)
    assert exact == pytest.approx(0.1944, abs=1e-4)        # a one-tailed count would give half of this
    assert M.stratified_permutation_p(x, y, strata, n=20000, seed=1) == pytest.approx(exact, abs=0.02)


def test_the_stratified_p_is_reproducible_and_lies_on_the_permutation_grid():
    x = np.arange(20.0)
    y = np.r_[np.arange(10.0)[::-1], np.arange(10.0)] + np.arange(20.0) % 3
    strata = ["A"] * 10 + ["B"] * 10
    first = M.stratified_permutation_p(x, y, strata, n=999, seed=5)
    assert M.stratified_permutation_p(x, y, strata, n=999, seed=5) == first
    assert (first * 1000) == pytest.approx(round(first * 1000))      # (1 + hits) / (n + 1)
    assert 1 / 1000 <= first <= 1.0


def test_the_stratified_p_ignores_incomplete_cases_and_works_with_any_label_type():
    """Same draws with and without rows that lack an x, a y or a stratum label."""
    x = [1, 2, 2, 3, 5, 6, 6, 7]
    y = [2, 1, 3, 1, 5, 4, 5, 6]
    labels = [("rig_1", "A")] * 4 + [("rig_2", "A")] * 4
    clean = M.stratified_permutation_p(x, y, labels, n=2000, seed=2)
    assert 0.3 < clean < 0.8
    x2 = [*x, np.nan, 4.0, 3.0]
    y2 = [*y, 7.0, np.nan, 2.0]
    labels2 = [*labels, ("rig_1", "A"), ("rig_1", "A"), None]
    assert M.stratified_permutation_p(x2, y2, labels2, n=2000, seed=2) == clean
    assert M.stratified_permutation_p(x2, y2, labels2[:-1] + [np.nan], n=2000, seed=2) == clean


def test_the_stratified_p_is_missing_without_enough_variation():
    assert math.isnan(M.stratified_permutation_p([1.0, 2.0], [1.0, 2.0], ["A", "A"], n=99))
    assert math.isnan(M.stratified_permutation_p([1.0] * 6, np.arange(6.0), ["A"] * 6, n=99))
    assert math.isnan(M.stratified_permutation_p(np.arange(6.0), [3.0] * 6, ["A"] * 6, n=99))


def test_when_every_stratum_has_one_member_nothing_can_be_shuffled_and_p_is_one():
    x = np.arange(8.0)
    assert M.stratified_permutation_p(x, x, list("ABCDEFGH"), n=99) == 1.0


def _fly_table(rows):
    return pd.DataFrame(rows, columns=["fly", "state", "outcome"])


def _determined_flies():
    """10 flies x 6 trials; the outcome IS the state (plus a per-fly offset that cancels)."""
    rng = np.random.default_rng(0)
    rows = []
    for f in range(10):
        for s in rng.permutation([1, 1, 1, 0, 0, 0]):
            rows.append((f"fly{f}", int(s), float(s) + 0.05 * f))
    return _fly_table(rows)


# (state, outcome) per trial for 12 flies x 8 trials. The outcomes were drawn without looking at the
# state, so any difference between the states is chance. Written out rather than generated so a
# change in NumPy's random streams cannot move the fixture.
_NULL_FLIES = [
    ("10110001", "00110000"), ("01001110", "01010100"), ("10110001", "11111011"),
    ("00011011", "00011010"), ("01101001", "00001010"), ("11101000", "00000010"),
    ("10100101", "11111001"), ("11101000", "10010100"), ("11001001", "11110001"),
    ("11110000", "01001111"), ("10010011", "11110011"), ("01011001", "10111111"),
]


def _independent_flies():
    return _fly_table([(f"fly{f}", int(s), float(o))
                       for f, (states, outcomes) in enumerate(_NULL_FLIES) for s, o in zip(states, outcomes)])


def test_an_outcome_set_by_the_state_inside_every_fly_is_significant():
    diff, p = M.within_fly_permutation(_determined_flies(), state="state", outcome="outcome", fly="fly", n=2000)
    assert diff == pytest.approx(1.0)
    assert p < 0.01


def test_the_within_fly_test_is_two_sided():
    df = _determined_flies()
    diff, p = M.within_fly_permutation(df.assign(outcome=-df["outcome"]), state="state", outcome="outcome",
                                       fly="fly", n=2000)
    assert diff == pytest.approx(-1.0)
    assert p < 0.01


def test_an_outcome_unrelated_to_the_state_is_not_significant():
    diff, p = M.within_fly_permutation(_independent_flies(), state="state", outcome="outcome", fly="fly", n=2000)
    assert abs(diff) < 0.3
    assert p > 0.2


def test_the_difference_is_the_pooled_mean_with_the_state_minus_the_pooled_mean_without():
    df = _fly_table([("a", 1, 1.0), ("a", 1, 0.0), ("a", 0, 0.0),
                     ("b", 1, 1.0), ("b", 0, 0.0), ("b", 0, 0.0), ("b", 0, 1.0)])
    diff, _ = M.within_fly_permutation(df, state="state", outcome="outcome", fly="fly", n=99)
    # trials pooled: state 1 -> (1, 0, 1) = 2/3, state 0 -> (0, 0, 0, 1) = 1/4.
    # (The mean of the two per-fly differences, 1/2 and 2/3, would be 7/12: an average of averages.)
    assert diff == pytest.approx(2 / 3 - 1 / 4)


def _single_state_flies(outcome):
    """One fly that was only ever in the state and one that never was, at extreme outcomes."""
    return _fly_table([("solo_on", 1, outcome)] * 5 + [("solo_off", 0, 1.0 - outcome)] * 5)


def test_flies_seen_in_only_one_state_are_ignored():
    """They are dropped before any draw, so the same seed gives the identical (diff, p)."""
    both = _independent_flies()
    base = M.within_fly_permutation(both, state="state", outcome="outcome", fly="fly", n=1500, seed=4)
    for extreme in (1.0, 0.0, 1e12):       # a huge ignored outcome must not widen any tolerance either
        mixed = pd.concat([both, _single_state_flies(extreme)], ignore_index=True)
        assert M.within_fly_permutation(mixed, state="state", outcome="outcome", fly="fly", n=1500, seed=4) == base


def test_a_fly_level_confound_alone_is_not_significant():
    """Responsive flies are mostly in the state, unresponsive ones mostly out of it; inside a fly it makes no difference."""
    rows = []
    for f in range(4):
        rows += [(f"responsive{f}", s, 1.0) for s in (1, 1, 1, 1, 1, 1, 0, 0)]
    for f in range(4):
        rows += [(f"quiet{f}", s, 0.0) for s in (1, 1, 0, 0, 0, 0, 0, 0)]
    df = _fly_table(rows)
    assert df.groupby("state")["outcome"].mean().to_dict() == {0: 0.25, 1: 0.75}   # pooled, a big "effect"
    diff, p = M.within_fly_permutation(df, state="state", outcome="outcome", fly="fly", n=500)
    assert diff == pytest.approx(0.5)
    assert p == 1.0


def test_arrangements_that_tie_the_observed_difference_count_despite_rounding():
    """0.1 + 0.2 and 0.3 + 0.0 are the same sum on paper, so the difference is 0 whichever pair is 'on'.

    In floating point the observed one is -5.6e-17, and the arrangement that is 'on' for 0.1 and 0.2
    gives exactly 0; without slack that tie would be missed and p would fall to about 0.83."""
    df = _fly_table([("a", 0, 0.1), ("a", 0, 0.2), ("a", 1, 0.3), ("a", 1, 0.0)])
    diff, p = M.within_fly_permutation(df, state="state", outcome="outcome", fly="fly", n=300)
    assert diff == pytest.approx(0.0, abs=1e-12)
    assert p == 1.0


def test_the_within_fly_p_matches_the_exact_distribution_of_label_shuffles():
    df = _fly_table([(f, s, o) for f, ss, oo in [
        ("A", (1, 0, 0, 1), (1, 0, 1, 1)),
        ("B", (0, 1, 1, 0), (0, 0, 1, 0)),
        ("C", (1, 0, 1, 0), (1, 1, 0, 0))] for s, o in zip(ss, oo)])
    flies = [g for _, g in df.groupby("fly")]
    n1 = sum(int(g["state"].sum()) for g in flies)
    n0 = len(df) - n1

    def pooled_diff(arrangement):
        on = sum(g["outcome"].to_numpy()[list(c)].sum() for g, c in zip(flies, arrangement))
        return on / n1 - (df["outcome"].sum() - on) / n0

    observed = pooled_diff([np.flatnonzero(g["state"].to_numpy()) for g in flies])
    arrangements = list(itertools.product(*[itertools.combinations(range(len(g)), int(g["state"].sum())) for g in flies]))
    exact = np.mean([abs(pooled_diff(a)) >= abs(observed) - 1e-12 for a in arrangements])   # 6 * 6 * 6 shuffles
    assert 0.3 < exact < 0.8
    diff, p = M.within_fly_permutation(df, state="state", outcome="outcome", fly="fly", n=20000, seed=3)
    assert diff == pytest.approx(observed)
    assert p == pytest.approx(exact, abs=0.02)


def test_the_within_fly_result_is_reproducible_from_the_seed():
    df = _independent_flies()
    a = M.within_fly_permutation(df, state="state", outcome="outcome", fly="fly", n=500, seed=9)
    b = M.within_fly_permutation(df, state="state", outcome="outcome", fly="fly", n=500, seed=9)
    assert a == b
    assert M.within_fly_permutation(df, state="state", outcome="outcome", fly="fly", n=500) == \
        M.within_fly_permutation(df, state="state", outcome="outcome", fly="fly", n=500, seed=0)


def test_a_boolean_state_column_is_the_same_as_zero_one():
    df = _independent_flies()
    as_bool = df.assign(state=df["state"].astype(bool))
    assert M.within_fly_permutation(as_bool, state="state", outcome="outcome", fly="fly", n=300, seed=1) == \
        M.within_fly_permutation(df, state="state", outcome="outcome", fly="fly", n=300, seed=1)


def test_trials_with_a_missing_state_or_outcome_are_left_out_before_flies_are_judged():
    df = _fly_table([("a", 1, 1.0), ("a", 0, 0.0), ("a", 0, 1.0), ("a", 1, 0.0),
                     ("b", 1, 1.0), ("b", 0, 0.0),
                     ("c", 1, np.nan), ("c", 0, 1.0), ("c", 0, 0.0),        # fly c loses its only state-1 trial
                     ("d", np.nan, 1.0), ("d", 0, 0.0), ("d", 1, 1.0), ("d", 0, 0.0)])
    kept = df[~df["fly"].isin(["c"]) & df["state"].notna()]
    assert M.within_fly_permutation(df, state="state", outcome="outcome", fly="fly", n=400, seed=2) == \
        M.within_fly_permutation(kept, state="state", outcome="outcome", fly="fly", n=400, seed=2)


def test_without_a_fly_seen_in_both_states_there_is_no_result():
    df = _fly_table([("a", 1, 1.0), ("a", 1, 0.0), ("b", 0, 1.0), ("b", 0, 0.0)])
    diff, p = M.within_fly_permutation(df, state="state", outcome="outcome", fly="fly", n=99)
    assert math.isnan(diff) and math.isnan(p)
    diff, p = M.within_fly_permutation(df.iloc[0:0], state="state", outcome="outcome", fly="fly", n=99)
    assert math.isnan(diff) and math.isnan(p)


def test_the_within_fly_p_lies_on_the_permutation_grid():
    df = _independent_flies()
    _, p = M.within_fly_permutation(df, state="state", outcome="outcome", fly="fly", n=199, seed=1)
    assert (p * 200) == pytest.approx(round(p * 200))
    assert 1 / 200 <= p <= 1.0


# ---------------------------------------------------------------------------
# Score table: duplicates and conflicts
# ---------------------------------------------------------------------------

_SCORE_COLUMNS = ["dataset", "fly", "fly_number", "trial_label", "score", "prediction"]


def _scores(*rows):
    return pd.DataFrame(list(rows), columns=_SCORE_COLUMNS)


def test_two_exact_duplicate_rows_are_kept_once():
    df = _scores(("d", "b1", 1, "testing_1_hex", 2, 1),
                 ("d", "b1", 1, "testing_1_hex", 2, 1),
                 ("d", "b1", 1, "testing_2_oct", 0, 0))
    out, counts = M.dedupe_predictions(df)
    assert out["trial_label"].tolist() == ["testing_1_hex", "testing_2_oct"]
    assert counts["exact_duplicates"] == 1
    assert counts["conflicts"] == 0


def test_a_key_whose_rows_disagree_on_the_score_loses_all_its_rows():
    df = _scores(("d", "b1", 1, "testing_1_hex", 2, 1),
                 ("d", "b1", 1, "testing_1_hex", 4, 1),
                 ("d", "b1", 1, "testing_2_oct", 0, 0))
    out, counts = M.dedupe_predictions(df)
    assert out["trial_label"].tolist() == ["testing_2_oct"]
    assert counts["conflicts"] == 1
    assert counts["exact_duplicates"] == 0


def test_the_same_trial_label_in_another_fly_or_dataset_is_not_a_duplicate():
    df = _scores(("d1", "b1", 1, "testing_1_hex", 2, 1),
                 ("d2", "b1", 1, "testing_1_hex", 4, 1),
                 ("d1", "b2", 1, "testing_1_hex", 3, 1),
                 ("d1", "b1", 2, "testing_1_hex", 0, 0))
    out, counts = M.dedupe_predictions(df)
    assert len(out) == 4
    assert counts["exact_duplicates"] == 0 and counts["conflicts"] == 0


def test_duplicates_are_collapsed_before_conflicts_are_judged_and_both_are_counted():
    df = _scores(("d", "b1", 1, "testing_1_hex", 2, 1),
                 ("d", "b1", 1, "testing_1_hex", 2, 1),
                 ("d", "b1", 1, "testing_1_hex", 4, 1),
                 ("d", "b1", 1, "testing_3_acv", 1, 0),
                 ("d", "b1", 1, "testing_4_cit", 1, 0),
                 ("d", "b1", 1, "testing_4_cit", 3, 0),
                 ("d", "b1", 1, "testing_4_cit", 3, 0))
    out, counts = M.dedupe_predictions(df)
    assert out["trial_label"].tolist() == ["testing_3_acv"]
    assert counts["exact_duplicates"] == 2
    assert counts["conflicts"] == 2


def test_rows_that_agree_on_the_score_but_differ_elsewhere_collapse_to_the_first():
    df = _scores(("d", "b1", 1, "testing_1_hex", 2, 1),
                 ("d", "b1", 1, "testing_1_hex", 2, 0))
    out, counts = M.dedupe_predictions(df)
    assert out["prediction"].tolist() == [1]
    assert counts["exact_duplicates"] == 0
    assert counts["conflicts"] == 0
    assert counts["same_score_duplicates"] == 1


def test_a_missing_score_disagrees_with_a_real_one_but_not_with_another_missing_one():
    df = _scores(("d", "b1", 1, "testing_1_hex", np.nan, 1),
                 ("d", "b1", 1, "testing_1_hex", 2.0, 1),
                 ("d", "b1", 1, "testing_2_oct", np.nan, 0),
                 ("d", "b1", 1, "testing_2_oct", np.nan, 0))
    out, counts = M.dedupe_predictions(df)
    assert out["trial_label"].tolist() == ["testing_2_oct"]
    assert counts["conflicts"] == 1 and counts["exact_duplicates"] == 1


def test_a_conflict_is_found_even_when_part_of_the_key_is_missing():
    df = _scores(("d", "b1", np.nan, "testing_1_hex", 1, 0),
                 ("d", "b1", np.nan, "testing_1_hex", 3, 0),
                 ("d", "b1", 2.0, "testing_1_hex", 4, 0),
                 ("d", "b1", np.nan, "testing_2_oct", 1, 0),
                 ("d", "b1", np.nan, "testing_2_oct", 1, 0))
    out, counts = M.dedupe_predictions(df)
    assert sorted(zip(out["trial_label"], out["score"])) == [("testing_1_hex", 4), ("testing_2_oct", 1)]
    assert counts["conflicts"] == 1 and counts["exact_duplicates"] == 1


def test_dedupe_returns_a_fresh_frame_and_leaves_the_input_alone():
    df = _scores(("d", "b1", 1, "testing_1_hex", 2, 1),
                 ("d", "b1", 1, "testing_1_hex", 4, 1),
                 ("d", "b1", 1, "testing_2_oct", 0, 0),
                 ("d", "b1", 1, "testing_2_oct", 0, 0),
                 ("d", "b1", 1, "testing_3_acv", 1, 0),
                 ("d", "b1", 1, "testing_3_acv", 1, 1))
    df.index = [10, 10, 30, 30, 50, 50]                    # not even unique
    snapshot = df.copy()
    out, _ = M.dedupe_predictions(df)
    pd.testing.assert_frame_equal(df, snapshot)
    assert out["trial_label"].tolist() == ["testing_2_oct", "testing_3_acv"]
    assert out.index.tolist() == [0, 1]
    assert list(out.columns) == list(df.columns)


def test_an_empty_score_table_has_nothing_to_dedupe():
    out, counts = M.dedupe_predictions(_scores())
    assert len(out) == 0
    assert counts["exact_duplicates"] == 0 and counts["conflicts"] == 0


def test_a_score_table_missing_a_key_column_is_refused_by_name():
    with pytest.raises(KeyError, match="trial_label"):
        M.dedupe_predictions(_scores(("d", "b1", 1, "testing_1_hex", 2, 1)).drop(columns="trial_label"))


# ---------------------------------------------------------------------------
# Purity
# ---------------------------------------------------------------------------

def test_the_metrics_survive_all_missing_inputs_without_a_numpy_or_scipy_warning():
    nan = np.full(300, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        M.head_frame_vector(nan, nan, nan, nan, 0.0)
        M.body_pitch_deg(nan, nan, nan, nan)
        M.tip_pitch_deg(nan, nan, nan, nan)
        M.point_speed_eu_s(nan, nan, eye_scale_px=48.0, fps=40.0)
        assert math.isnan(M.rest_level(nan))
        assert M.runs_above(nan, 0.0, min_frames=3) == [] and M.runs_below(nan, 0.0, min_frames=3) == []
        assert M.curl_events(nan, 5.0, rise_deg=10.0, min_frames=12) == (0, 0)
        assert M.held_out(nan, ref_eu=1.0, margin_eu=0.2, frac_min=0.8) is False
        assert math.isnan(M.retracted_reference(nan))
        assert M.anchor_spread_px(np.full((5, 2), np.nan)) == 0.0
        assert math.isnan(M.angle_between_deg((np.nan, 1.0), (1.0, 0.0)))
        assert all(math.isnan(v) for v in M.spearman(nan, nan))
        assert math.isnan(M.coupling_rho(nan, nan))
        assert math.isnan(M.stratified_permutation_p(nan, nan, ["a"] * 300, n=10))


def test_the_module_imports_nothing_from_the_pipeline():
    """Pure maths: importing it must not pull in any other fbpipe module (steps, config, geometry...)."""
    code = ("import sys; import fbpipe.analysis.mounting; "
            "print(sorted(m for m in sys.modules if m == 'fbpipe' or m.startswith('fbpipe.')))")
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "['fbpipe', 'fbpipe.analysis', 'fbpipe.analysis.mounting']"
