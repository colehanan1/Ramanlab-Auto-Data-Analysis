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
  stratum-level (fly-level) confound alone cannot produce a small p;
* the stratified risk difference -- the effect size that goes with them: a
  Mantel-Haenszel-weighted mean of the within-stratum differences, so a stratum
  confound cannot make it large either;
* the paired window permutation -- the statistic is the difference of the POOLED
  rates (events over minutes, before vs during the stimulus), never a mean of
  rates. The default, ``scheme="split"``, re-assigns a trial's events between its
  two windows in proportion to their lengths: a valid test only if a trial's events
  are independent and come at a constant rate, which real (bursty) events do not,
  so it is no cure for windows of different lengths; ``scheme="swap"`` exchanges
  the two windows with probability 1/2, is valid only for equal-length windows that
  are interchangeable when nothing happens, and refuses windows that differ by more
  than 1 %. Both p-values are Monte-Carlo estimates;
* the placebo design -- the stimulus is tested against an EQUAL-length window just
  before it, and both windows of a trial are scored against ONE resting bend taken
  from the fly's OTHER trials (the median of their early baseline windows: never the
  trial itself, never a placebo or a during frame), so with a stationary bend "the
  stimulus does nothing" does mean "the same events per minute". A baseline fitted
  to the pre-stimulus window itself (the first design), or taken from the same trial
  next to the placebo window (the second), makes the placebo quieter than the start
  of the stimulus; the calibration tests pool 200 to 400 trials, where that shows.
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
from scipy import signal, stats

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
# Stratified (Mantel-Haenszel) risk difference
# ---------------------------------------------------------------------------

def _strata_table(rows):
    return pd.DataFrame(rows, columns=["stratum", "state", "outcome"])


def _two_strata():
    """A: state 1 -> [1, 1], state 0 -> [1, 0] (n1 = n0 = 2, rates 1.0 / 0.5, weight 1, difference +0.5).
    B: state 1 -> [0], state 0 -> [1, 0, 0, 0, 0, 0] (n1 = 1, n0 = 6, rates 0 / 1/6, weight 6/7, difference -1/6)."""
    return _strata_table([("A", 1, 1.0), ("A", 1, 1.0), ("A", 0, 1.0), ("A", 0, 0.0),
                          ("B", 1, 0.0), ("B", 0, 1.0), ("B", 0, 0.0), ("B", 0, 0.0), ("B", 0, 0.0), ("B", 0, 0.0),
                          ("B", 0, 0.0)])


def _rd(df):
    return M.stratified_risk_difference(df, state="state", outcome="outcome", stratum="stratum")


def test_the_stratified_risk_difference_is_the_mantel_haenszel_weighted_mean_of_the_stratum_differences():
    # (1 * 0.5 + 6/7 * (-1/6)) / (1 + 6/7) = (5/14) / (13/7) = 5/26
    assert _rd(_two_strata()) == pytest.approx(5 / 26)


def test_the_pooled_difference_of_the_same_trials_is_a_different_number():
    df = _two_strata()
    pooled = df.loc[df["state"] == 1, "outcome"].mean() - df.loc[df["state"] == 0, "outcome"].mean()
    assert pooled == pytest.approx(5 / 12) and _rd(df) != pytest.approx(pooled)


def test_a_stratum_is_weighted_by_n1_n0_over_n1_plus_n0_not_by_its_size_or_equally():
    small = [("X", 1, 1.0)] + [("X", 0, 0.0)] * 9                     # n1 = 1, n0 = 9: weight 0.9, difference +1
    even = [("Y", 1, 0.0)] * 25 + [("Y", 1, 1.0)] * 25 + [("Y", 0, 0.0)] * 25 + [("Y", 0, 1.0)] * 25   # weight 25, difference 0
    assert _rd(_strata_table(small + even)) == pytest.approx(0.9 / 25.9)
    # n1 * n0 as the weight would give 9 / 2509, the number of trials 10 / 110 and equal weights 1 / 2


def test_with_one_stratum_it_is_the_plain_difference_of_the_two_rates():
    df = _strata_table([("A", 1, 1.0), ("A", 1, 0.0), ("A", 1, 1.0), ("A", 0, 0.0), ("A", 0, 1.0)])
    assert _rd(df) == pytest.approx(2 / 3 - 1 / 2)


def test_a_stratum_seen_in_only_one_state_does_not_count():
    extra = _strata_table([("C", 1, 1.0), ("C", 1, 1.0), ("C", 1, 1.0), ("D", 0, 0.0), ("D", 0, 1.0)])
    assert _rd(pd.concat([_two_strata(), extra], ignore_index=True)) == pytest.approx(5 / 26)


def test_without_a_stratum_seen_in_both_states_there_is_no_difference():
    df = _strata_table([("A", 1, 1.0), ("A", 1, 0.0), ("B", 0, 1.0), ("B", 0, 0.0)])
    assert math.isnan(_rd(df)) and math.isnan(_rd(df.iloc[0:0]))


def test_rows_with_a_missing_state_outcome_or_stratum_are_left_out_before_strata_are_judged():
    messy = pd.concat([_two_strata(), _strata_table([
        ("C", 1, np.nan), ("C", 0, 1.0), ("C", 0, 0.0),             # stratum C loses its only state-1 trial
        ("E", np.nan, 1.0), ("E", 1, 1.0), ("E", 0, 0.0),           # E keeps a state-1 and a state-0 trial (the NaN row goes)
        (None, 1, 1.0), (None, 0, 0.0)]).astype({"stratum": object})], ignore_index=True)
    kept = pd.concat([_two_strata(), _strata_table([("E", 1, 1.0), ("E", 0, 0.0)])], ignore_index=True)
    assert _rd(messy) == pytest.approx(_rd(kept))
    assert _rd(kept) != pytest.approx(5 / 26)                        # E really counts (weight 1/2, difference +1)


def test_the_order_of_the_rows_and_the_kind_of_stratum_label_do_not_matter():
    df = _two_strata()
    assert _rd(df.sample(frac=1.0, random_state=3)) == pytest.approx(_rd(df))
    assert _rd(df.assign(stratum=df["stratum"].map({"A": 7, "B": 2}))) == pytest.approx(_rd(df))
    assert _rd(df.assign(stratum=df["stratum"].map({"A": ("pretest", 1), "B": ("testing", 1)}))) == pytest.approx(_rd(df))


def test_any_nonzero_state_counts_as_in_the_state_as_in_the_permutation_test():
    df = _two_strata()
    assert _rd(df.assign(state=df["state"] * 2)) == pytest.approx(_rd(df))


def test_a_boolean_state_is_the_same_as_zero_one_for_the_stratified_difference():
    df = _two_strata()
    assert _rd(df.assign(state=df["state"].astype(bool))) == pytest.approx(_rd(df))


def test_a_trial_type_confound_leaves_the_pooled_difference_large_and_the_stratified_one_zero():
    """Pretest: state 25 % and response 25 %; testing: 75 % and 75 %; unrelated inside each, so only the type moves both."""
    rows = []
    for stratum, n_on, on_responding, off_responding in (("pretest", 4, 1, 3), ("testing", 12, 9, 3)):
        for t in range(16):
            on = t < n_on
            rows.append((stratum, int(on), float(t < on_responding if on else t - n_on < off_responding)))
    df = _strata_table(rows)
    pooled = df.loc[df["state"] == 1, "outcome"].mean() - df.loc[df["state"] == 0, "outcome"].mean()
    assert pooled == pytest.approx(0.25) and _rd(df) == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Paired window permutation (events per minute, before vs during the stimulus)
# ---------------------------------------------------------------------------

SCHEMES = ("swap", "split")


def _windows(rows):
    """One row per trial: (events, minutes) of its first window, then of its second."""
    return pd.DataFrame(rows, columns=["e_pre", "m_pre", "e_stim", "m_stim"])


def _perm(df, **kw):
    """``paired_window_permutation`` of a ``_windows`` table, with the scheme the caller names (the default if none)."""
    return M.paired_window_permutation(df, events_pre="e_pre", min_pre="m_pre", events_stim="e_stim",
                                       min_stim="m_stim", **kw)


def _swap(df, **kw):
    return _perm(df, scheme="swap", **kw)


def _split(df, **kw):
    return _perm(df, scheme="split", **kw)


def _exact_swap_p(df):
    """P(|d| >= |d_obs|) over every one of the 2^k ways to exchange the windows of whole trials -- the definition."""
    e0, m0, e1, m1 = (df[c].to_numpy(float) for c in ("e_pre", "m_pre", "e_stim", "m_stim"))

    def diff(swap):
        s = np.array(swap, bool)
        stim_e, stim_m = np.where(s, e0, e1).sum(), np.where(s, m0, m1).sum()
        return stim_e / stim_m - (e0.sum() + e1.sum() - stim_e) / (m0.sum() + m1.sum() - stim_m)

    observed = diff([False] * len(df))
    every = [diff(s) for s in itertools.product([False, True], repeat=len(df))]
    return float(np.mean([abs(d) >= abs(observed) - 1e-12 for d in every]))


def _exact_split_p(df):
    """P(|d| >= |d_obs|) when every trial's events are re-assigned to its windows independently, each one to the stimulus
    window with probability m_stim / (m_pre + m_stim) -- the definition: the stimulus window's total is a sum of
    independent binomials (convolved exactly), and the difference is a function of that total alone."""
    e0, m0, e1, m1 = (df[c].to_numpy(float) for c in ("e_pre", "m_pre", "e_stim", "m_stim"))
    total, share = e0 + e1, m1 / (m0 + m1)
    dist = np.array([1.0])
    for k, q in zip(total.astype(int), share):
        dist = np.convolve(dist, stats.binom.pmf(np.arange(k + 1), k, q))
    stim = np.arange(len(dist), dtype=float)
    d = stim / m1.sum() - (total.sum() - stim) / m0.sum()
    observed = e1.sum() / m1.sum() - e0.sum() / m0.sum()
    return float(dist[np.abs(d) >= abs(observed) - 1e-12].sum())


EXACT = {"swap": _exact_swap_p, "split": _exact_split_p}


def _mirror(df):
    """The same trials with the two windows called the other way round (before <-> during)."""
    return df.rename(columns={"e_pre": "e_stim", "m_pre": "m_stim", "e_stim": "e_pre", "m_stim": "m_pre"})


def _burst_trials(k=30, stim_events=3, stim_min=0.1):
    """Every trial: 0 or 1 events in 0.5 min before the stimulus, ``stim_events`` in the ``stim_min`` minutes of it."""
    return _windows([(i % 2, 0.5, stim_events, stim_min) for i in range(k)])


def _same_rate_trials(k=40, rate=2.0, boost=1.0, seed=0, pre=0.5, stim=1 / 15):
    """Poisson counts with the SAME rate per minute before and during the stimulus (times ``boost`` during it), in windows
    of ``pre`` and ``stim`` minutes (30 s and 4 s unless told otherwise): what a stimulus that changes nothing looks like."""
    rng = np.random.default_rng(seed)
    return _windows([(rng.poisson(rate * pre), pre, rng.poisson(rate * boost * stim), stim) for _ in range(k)])


# a table the two schemes both accept: each trial's two windows are equally long (the lengths differ BETWEEN trials)
EQUAL = _windows([(2, 0.5, 1, 0.5), (0, 0.3, 2, 0.3), (1, 1.0, 0, 1.0), (3, 0.4, 1, 0.4), (1, 0.5, 1, 0.5),
                  (0, 0.6, 1, 0.6), (2, 0.5, 3, 0.5), (1, 0.7, 0, 0.7), (4, 0.5, 2, 0.5)])
# the same trials with a short stimulus window, which only the split can test
UNEQUAL = _windows([(2, 0.5, 1, 0.4), (0, 0.5, 1, 0.3), (1, 0.8, 0, 0.3), (3, 0.4, 1, 0.1), (1, 0.5, 1, 0.25),
                    (0, 0.6, 1, 0.3), (2, 0.5, 1, 0.2), (1, 0.7, 0, 0.1), (4, 0.5, 2, 0.2)])


# ---- the default and the options ------------------------------------------------------------------------------------

def test_the_default_is_the_length_proportional_split_and_the_swap_is_an_explicit_option():
    unequal = _same_rate_trials(k=20, boost=2.0, seed=4)
    assert _perm(unequal, n=300, seed=2) == _split(unequal, n=300, seed=2)
    assert _perm(EQUAL, n=2000, seed=2) == _split(EQUAL, n=2000, seed=2) != _swap(EQUAL, n=2000, seed=2)


def test_an_unknown_scheme_is_refused():
    for scheme in ("shuffle", "", None, "Split"):
        with pytest.raises(ValueError):
            _perm(EQUAL, n=9, scheme=scheme)


def test_the_swap_refuses_windows_that_differ_in_length_by_more_than_one_percent():
    for stim in (0.1, 0.45, 0.495, 0.51, 2.0):            # a short window, 10 % either way, just over 1 %, 2 %, a long one
        df = _windows([(1, 0.5, 2, 0.5), (0, 0.5, 1, 0.5), (2, 0.5, 1, stim)])           # ONE trial in three is off
        with pytest.raises(ValueError, match="same length"):
            _swap(df, n=99)
        assert not math.isnan(_perm(df, n=99)[1]) and not math.isnan(_split(df, n=99)[1]), stim   # the split takes any lengths


def test_the_swap_accepts_windows_within_one_percent_and_never_looks_at_the_trials_it_leaves_out():
    near = _windows([(1, 0.5, 2, 0.503), (0, 0.5, 1, 0.4975), (2, 0.5, 1, 0.5)])
    assert not math.isnan(_swap(near, n=99, seed=1)[1])
    left_out = pd.concat([near, _windows([(np.nan, 0.5, 3, 0.1), (2, 0.5, np.nan, 0.1),        # a missing count
                                          (2, 0.0, 3, 0.1), (2, np.nan, 3, 0.1), (2, -0.5, 3, 0.1)])],   # an unusable length
                         ignore_index=True)
    assert _swap(left_out, n=99, seed=1) == _swap(near, n=99, seed=1)                          # unequal, but never eligible


def test_the_difference_is_the_pooled_stimulus_rate_minus_the_pooled_rate_before():
    df = _windows([(2, 0.5, 1, 0.1), (0, 0.5, 3, 0.2), (1, 1.0, 0, 0.3)])
    assert _perm(df, n=99)[0] == pytest.approx(4 / 0.6 - 3 / 2.0)           # (1 + 3 + 0) / 0.6 min  -  (2 + 0 + 1) / 2.0 min
    per_trial = np.mean([1 / 0.1 - 2 / 0.5, 3 / 0.2 - 0 / 0.5, 0 / 0.3 - 1 / 1.0])
    assert per_trial == pytest.approx(20 / 3) and _perm(df, n=99)[0] != pytest.approx(per_trial)   # a mean of rates: 6.67
    equal = _windows([(2, 0.5, 1, 0.5), (0, 1.0, 3, 1.0)])
    assert _swap(equal, n=99)[0] == _split(equal, n=99)[0] == pytest.approx(4 / 1.5 - 2 / 1.5)     # the same for both


def test_the_difference_does_not_depend_on_the_order_of_the_trials():
    df = _windows([(2, 0.5, 1, 0.1), (0, 0.5, 2, 0.2), (1, 1.0, 0, 0.3), (3, 0.4, 1, 0.1)])
    assert _perm(df.sample(frac=1.0, random_state=3), n=99)[0] == pytest.approx(_perm(df, n=99)[0])


def test_the_columns_are_named_by_the_caller_and_other_columns_are_ignored():
    df = _windows([(2, 0.5, 1, 0.1), (0, 0.5, 2, 0.2), (1, 1.0, 0, 0.3)])
    renamed = df.rename(columns={"e_pre": "a", "m_pre": "b", "e_stim": "c", "m_stim": "d"}).assign(fly="x", junk=np.nan)
    assert M.paired_window_permutation(renamed, events_pre="a", min_pre="b", events_stim="c", min_stim="d", n=300,
                                       seed=1) == _perm(df, n=300, seed=1)


# ---- what both schemes must do (equal windows, so that both apply) -------------------------------------------------

@pytest.mark.parametrize("scheme", SCHEMES)
def test_a_strong_effect_of_the_stimulus_inside_every_trial_is_significant_and_so_is_a_lowered_rate(scheme):
    df = _burst_trials(stim_min=0.5)
    diff, p = _perm(df, n=999, scheme=scheme)
    assert diff == pytest.approx(90 / 15 - 15 / 15) and p == pytest.approx(1 / 1000)       # 6 events/min during, 1 before
    diff, p = _perm(_mirror(df), n=999, scheme=scheme)
    assert diff == pytest.approx(15 / 15 - 90 / 15) and p == pytest.approx(1 / 1000)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_windows_with_the_same_events_and_minutes_show_nothing_and_p_is_one(scheme):
    df = _windows([(2, 0.5, 2, 0.5), (0, 0.5, 0, 0.5), (3, 0.4, 3, 0.4), (1, 1.0, 1, 1.0)])
    diff, p = _perm(df, n=500, scheme=scheme)
    assert diff == pytest.approx(0.0, abs=1e-12) and p == 1.0


@pytest.mark.parametrize("scheme", SCHEMES)
def test_windows_that_differ_only_by_chance_are_not_significant(scheme):
    df = _windows([(2, 0.5, 1, 0.5), (0, 0.5, 1, 0.5), (1, 0.5, 2, 0.5), (3, 0.5, 2, 0.5), (1, 0.5, 1, 0.5),
                   (2, 0.5, 0, 0.5), (0, 0.5, 1, 0.5), (1, 0.5, 2, 0.5), (2, 0.5, 1, 0.5), (1, 0.5, 2, 0.5)])
    diff, p = _perm(df, n=4000, seed=1, scheme=scheme)
    assert abs(diff) < 0.6 and p > 0.4


@pytest.mark.parametrize("scheme", SCHEMES)
def test_the_p_matches_the_exact_distribution_of_the_scheme(scheme):
    exact = EXACT[scheme](EQUAL)                                  # 2^9 swaps; 3^2 * 2^7 ... outcomes of the split
    assert 0.2 < exact < 0.9                                      # a fixture that can tell a wrong null apart
    diff, p = _perm(EQUAL, n=40000, seed=3, scheme=scheme)
    assert p == pytest.approx(exact, abs=0.015)
    assert diff == pytest.approx(EQUAL["e_stim"].sum() / EQUAL["m_stim"].sum() - EQUAL["e_pre"].sum() / EQUAL["m_pre"].sum())


@pytest.mark.parametrize("scheme", SCHEMES)
def test_the_p_counts_both_tails(scheme):
    """Pushing the same data the other way gives the same two-sided p (one tail alone would give about half of it)."""
    df = _windows([(2, 0.5, 1, 0.5), (0, 0.5, 1, 0.5), (1, 0.5, 2, 0.5), (3, 0.5, 0, 0.5)])
    (d1, p1), (d2, p2) = (_perm(d, n=40000, seed=4, scheme=scheme) for d in (df, _mirror(df)))
    assert d1 == pytest.approx(-d2) and d1 != pytest.approx(0.0)
    exact = EXACT[scheme](df)
    assert exact == pytest.approx(EXACT[scheme](_mirror(df))) and 0.6 < exact < 0.9
    assert p1 == pytest.approx(exact, abs=0.015) and p2 == pytest.approx(exact, abs=0.015)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_trials_that_all_move_the_same_way_have_the_p_of_a_fair_coin(scheme):
    """Four identical trials (one event before, none during, equal windows): the stimulus window ends up with 0 to 4 of
    the events, binomially under either scheme, so only 0 and 4 are as extreme as what was seen: p = 2 / 2^4."""
    assert _perm(_windows([(1, 0.5, 0, 0.5)] * 4), n=20000, seed=1, scheme=scheme)[1] == pytest.approx(2 / 16, abs=0.01)


def test_a_single_trial_cannot_be_told_from_its_swap_so_the_swap_has_p_one():
    diff, p = _swap(_windows([(0, 0.5, 4, 0.5)]), n=999, seed=1)
    assert diff == pytest.approx(8.0) and p == 1.0


def test_a_single_trial_split_between_equal_windows_is_as_extreme_only_with_all_or_none_of_its_events():
    diff, p = _split(_windows([(0, 0.5, 4, 0.5)]), n=20000, seed=1)
    assert diff == pytest.approx(8.0) and p == pytest.approx(2 / 16, abs=0.01)      # all 4 in the stimulus, or none


@pytest.mark.parametrize("scheme", SCHEMES)
def test_the_result_is_reproducible_from_the_seed_lies_on_the_permutation_grid_and_changes_with_the_seed(scheme):
    df = _burst_trials(k=12, stim_events=1, stim_min=0.5)
    first = _perm(df, n=499, seed=5, scheme=scheme)
    assert _perm(df, n=499, seed=5, scheme=scheme) == first
    assert _perm(df, n=499, scheme=scheme) == _perm(df, n=499, seed=0, scheme=scheme)
    assert (first[1] * 500) == pytest.approx(round(first[1] * 500)) and 1 / 500 <= first[1] <= 1.0
    assert len({_perm(EQUAL, n=60, seed=s, scheme=scheme)[1] for s in range(8)}) > 1


@pytest.mark.parametrize("scheme", SCHEMES)
def test_trials_missing_a_window_or_an_event_count_are_left_out(scheme):
    clean = _windows([(2, 0.5, 1, 0.5), (0, 0.5, 2, 0.5), (1, 1.0, 0, 1.0), (3, 0.4, 1, 0.4), (1, 0.5, 1, 0.5)])
    messy = pd.concat([clean, _windows([(np.nan, 0.5, 3, 0.1),       # no events counted before
                                        (2, 0.5, np.nan, 0.1),       # no events counted during
                                        (2, np.nan, 3, 0.1),         # no length before
                                        (2, 0.5, 3, np.nan),         # no length during
                                        (2, 0.0, 3, 0.1),            # an empty window before
                                        (2, 0.5, 3, 0.0),            # an empty window during
                                        (2, -0.5, 3, 0.1)])], ignore_index=True)
    assert _perm(messy, n=400, seed=2, scheme=scheme) == _perm(clean, n=400, seed=2, scheme=scheme)
    complete = pd.concat([clean, _windows([(2, 0.5, 3, 0.5)])], ignore_index=True)       # a whole trial does count
    assert _perm(complete, n=400, seed=2, scheme=scheme)[0] != pytest.approx(_perm(clean, n=400, seed=2, scheme=scheme)[0])


@pytest.mark.parametrize("scheme", SCHEMES)
def test_without_a_qualifying_trial_there_is_no_result(scheme):
    for df in (_windows([]), _windows([(np.nan, 0.5, 1, 0.5), (1, 0.5, np.nan, 0.5)]), _windows([(1, 0.0, 1, 0.5)])):
        diff, p = _perm(df, n=99, scheme=scheme)
        assert math.isnan(diff) and math.isnan(p)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_the_input_table_is_left_alone_and_a_permutation_count_below_one_is_refused(scheme):
    df = _burst_trials(k=6, stim_min=0.5)
    before = df.copy()
    _perm(df, n=50, scheme=scheme)
    pd.testing.assert_frame_equal(df, before)
    with pytest.raises(ValueError):
        _perm(df, n=0, scheme=scheme)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_when_nothing_differs_every_draw_is_as_extreme_so_p_is_exactly_one_across_chunks_of_draws(scheme):
    df = _windows([(2, 0.5, 2, 0.5)] * 3000)                                  # n = 2000 is not a multiple of the chunk size
    diff, p = _perm(df, n=2000, seed=3, scheme=scheme)
    assert diff == pytest.approx(0.0, abs=1e-12) and p == 1.0


@pytest.mark.parametrize("scheme", SCHEMES)
def test_many_trials_are_handled_in_chunks_and_the_result_does_not_depend_on_the_chunking(scheme, monkeypatch):
    df = _burst_trials(k=3000, stim_events=0, stim_min=0.5)                  # 3000 x 2000 draws do not fit in one block
    first = _perm(df, n=2000, seed=7, scheme=scheme)
    assert first == _perm(df, n=2000, seed=7, scheme=scheme) and first[0] < 0 and first[1] < 0.01
    few = _same_rate_trials(k=50, boost=2.0, seed=2, stim=0.5)
    whole = _perm(few, n=600, seed=5, scheme=scheme)
    with monkeypatch.context() as small:
        small.setattr(M, "_CHUNK_CELLS", 700)                                 # 14 draws at a time instead of 600
        assert _perm(few, n=600, seed=5, scheme=scheme) == whole


@pytest.mark.parametrize("scheme", SCHEMES)
def test_a_stimulus_that_changes_nothing_is_not_called_significant_in_equal_windows(scheme):
    """Same rate before and during, in 30 s and 30 s windows: either scheme keeps the false-positive rate at its level."""
    calls = sum(_perm(_same_rate_trials(seed=rep, stim=0.5), n=199, seed=rep, scheme=scheme)[1] <= 0.05 for rep in range(300))
    assert calls / 300 < 0.09                                    # 0.05 expected, 3 standard errors of slack


def test_swaps_that_tie_the_observed_difference_count_despite_rounding():
    """Two of the 8 swaps give, in exact arithmetic, the observed |difference|, but one falls 1 ulp short in floating point."""
    df = _windows([(0.4, 0.5, 0.4, 0.5), (0.4, 0.5, 0.7, 0.5), (0.5, 0.5, 0.6, 0.5)])
    exact = _exact_swap_p(df)
    assert 0.2 < exact < 0.95
    assert _swap(df, n=20000, seed=2)[1] == pytest.approx(exact, abs=0.02)


# ---- the default, which also takes windows of different lengths ----------------------------------------------------

def test_a_strong_effect_in_a_short_stimulus_window_is_significant_and_so_is_a_lowered_rate():
    diff, p = _perm(_burst_trials(), n=999)
    assert diff == pytest.approx(90 / 3.0 - 15 / 15.0) and p == pytest.approx(1 / 1000)         # 30 events/min vs 1/min
    diff, p = _perm(_mirror(_burst_trials()), n=999)
    assert diff == pytest.approx(15 / 15.0 - 90 / 3.0) and diff < 0 and p == pytest.approx(1 / 1000)


def test_the_p_matches_the_exact_distribution_when_the_windows_differ_in_length():
    exact = _exact_split_p(_windows([(2, 0.5, 1, 0.1), (0, 0.5, 2, 0.2), (1, 1.0, 0, 0.3), (3, 0.4, 1, 0.1)]))   # 5*5*3*5
    assert 0.05 < exact < 0.9
    assert _perm(_windows([(2, 0.5, 1, 0.1), (0, 0.5, 2, 0.2), (1, 1.0, 0, 0.3), (3, 0.4, 1, 0.1)]), n=40000,
                 seed=2)[1] == pytest.approx(exact, abs=0.015)
    exact = _exact_split_p(UNEQUAL.iloc[:6])
    assert 0.05 < exact < 0.95
    assert _perm(UNEQUAL.iloc[:6], n=40000, seed=6)[1] == pytest.approx(exact, abs=0.015)


def test_a_stimulus_that_changes_nothing_is_not_called_significant_just_because_its_window_is_short():
    """Same rate before and during, in 30 s and 4 s windows (the odor): the default keeps the false-positive rate at its
    level, plain or with trials of very different rates; exchanging the two whole windows would call about one in five of
    these significant at 0.05 (and is refused for windows that differ in length)."""
    for rate_spread in (0.0, 1.0):                              # every trial at the same rate, then gamma-distributed rates
        calls = 0
        for rep in range(300):
            rng = np.random.default_rng(1000 + rep)
            rates = rng.gamma(1 / rate_spread, rate_spread, 40) * 2.0 if rate_spread else np.full(40, 2.0)
            df = _windows([(rng.poisson(r * 0.5), 0.5, rng.poisson(r / 15), 1 / 15) for r in rates])
            calls += _perm(df, n=199, seed=rep)[1] <= 0.05
        assert calls / 300 < 0.09, rate_spread                  # 0.05 expected, 3 standard errors of slack


def test_a_real_stimulus_effect_is_still_found_in_short_windows():
    found = sum(_perm(_same_rate_trials(boost=3.0, seed=rep), n=199, seed=rep)[1] <= 0.05 for rep in range(100))
    assert found >= 70


def test_one_trial_with_every_event_in_the_short_window_is_unlikely():
    diff, p = _perm(_windows([(0, 0.5, 4, 0.1)]), n=999, seed=1)         # all 4 events in 0.1 of the 0.6 minutes
    assert diff == pytest.approx(40.0) and p < 0.01                      # (1/6)^4 = 0.0008: the swap could not tell


def test_a_trial_without_events_does_not_change_the_p_but_its_minutes_still_count_in_the_rates():
    df = _same_rate_trials(k=10, boost=2.0, seed=3)
    quiet = pd.concat([df, _windows([(0, 0.5, 0, 0.0667)] * 5)], ignore_index=True)
    assert _perm(quiet, n=300, seed=1)[0] == pytest.approx(
        quiet["e_stim"].sum() / quiet["m_stim"].sum() - quiet["e_pre"].sum() / quiet["m_pre"].sum())
    assert _perm(quiet, n=300, seed=1)[0] != pytest.approx(_perm(df, n=300, seed=1)[0])


def test_the_split_needs_whole_counts_of_events():
    with pytest.raises(ValueError):
        _perm(_windows([(1.5, 0.5, 2, 0.1)]), n=9)
    with pytest.raises(ValueError):
        _perm(_windows([(-1, 0.5, 2, 0.1), (1, 0.5, 2, 0.1)]), n=9)
    assert not math.isnan(_swap(_windows([(1.5, 0.5, 2, 0.5)]), n=9)[0])    # the swap has no such need


# ---------------------------------------------------------------------------
# Placebo design: equal-length windows scored against the fly's other trials
# ---------------------------------------------------------------------------

FPS10 = 10.0                      # 3 frames = the default curl_min_s of 0.3 s


def _placebo(bend, *, onset=300, offset=340, fps=FPS10, gate=0.5, rise=10.0, min_s=0.3, rest=5.0):
    return M.placebo_events(bend, onset=onset, offset=offset, fps=fps, rise_deg=rise, min_s=min_s, min_tracked_frac=gate,
                            rest_deg=rest)


def _flat(n=400, level=5.0):
    return np.full(n, level)


def _excursion_at(a, b, *, n=400, height=20.0, level=5.0):
    """A flat bend with one excursion of ``height`` degrees over frames ``a:b``."""
    bend = _flat(n, level)
    bend[a:b] += height
    return bend


def _counts(out):
    return (out["abd_curl_events_placebo"], out["abd_curl_events_during"],
            out["abd_straighten_events_placebo"], out["abd_straighten_events_during"])


def test_the_placebo_columns_are_named_in_one_place():
    assert tuple(_placebo(_flat())) == M.PLACEBO_COLUMNS
    assert M.PLACEBO_COLUMNS == ("pb_window_min", "abd_tracked_frac_base", "abd_tracked_frac_placebo",
                                 "abd_tracked_frac_during", "abd_base_ok", "abd_rest_bend_loo_deg",
                                 "abd_curl_events_placebo", "abd_straighten_events_placebo", "abd_curl_events_during",
                                 "abd_straighten_events_during")


@pytest.mark.parametrize("n, onset, offset, frames", [
    (400, 300, 340, 40),        # a stimulus shorter than half the pre-stimulus window: all of it
    (400, 300, None, 100),      # a stimulus that runs to the end of the trial
    (400, 300, 450, 100),       # ... or past it: only the frames that exist
    (400, 300, 400, 100),
    (400, 60, 100, 30),         # a short pre-stimulus window: half of it
    (400, 61, 400, 30),         # an odd one: floor(61 / 2)
    (400, 2, 100, 1),
    (400, 300, 330, 30),        # stimulus and half the pre-window both 30: either way 30
])
def test_the_placebo_window_is_the_shorter_of_the_stimulus_and_half_the_pre_stimulus_window(n, onset, offset, frames):
    assert _placebo(_flat(n), onset=onset, offset=offset)["pb_window_min"] == pytest.approx(frames / FPS10 / 60)


NO_DESIGN = [
    (400, None, None),          # no stimulus known
    (400, 1, 100),              # half of a one-frame pre-stimulus window is nothing
    (400, 0, 100),              # no pre-stimulus window at all
    (400, 300, 300),            # an empty stimulus
    (400, 300, 250),            # a stimulus that ends before it starts
    (400, 400, 450),            # a stimulus that starts after the trial
    (0, 300, 340),              # no trial
]


@pytest.mark.parametrize("n, onset, offset", NO_DESIGN)
def test_without_a_pre_stimulus_window_and_a_stimulus_there_is_nothing_to_compare(n, onset, offset):
    out = _placebo(_flat(n), onset=onset, offset=offset)
    assert out["abd_base_ok"] is False
    assert all(math.isnan(v) for k, v in out.items() if k != "abd_base_ok"), out
    assert M.baseline_values(_flat(n), onset=onset, offset=offset, min_tracked_frac=0.0) is None


def test_the_placebo_is_the_window_just_before_the_stimulus_and_the_during_window_its_first_frames():
    """onset 300, stimulus 40 frames: baseline 0-260, placebo 260-300, during 300-340 (3 frames = curl_min_s)."""
    expected = {(257, 260): (0, 0), (260, 263): (1, 0), (297, 300): (1, 0), (300, 303): (0, 1),
                (337, 340): (0, 1), (340, 343): (0, 0), (259, 262): (0, 0), (298, 301): (0, 0)}   # a straddle: 2 + 1 frames
    for (a, b), (placebo, during) in expected.items():
        out = _placebo(_excursion_at(a, b))
        assert (out["abd_curl_events_placebo"], out["abd_curl_events_during"]) == (placebo, during), (a, b)


def test_a_run_that_straddles_the_onset_is_clipped_to_each_window_and_counts_in_each_when_each_part_is_long_enough():
    out = _placebo(_excursion_at(296, 304))                    # 4 frames before the onset, 4 after
    assert (out["abd_curl_events_placebo"], out["abd_curl_events_during"]) == (1, 1)
    out = _placebo(_excursion_at(298, 306))                    # 2 frames before: too short; 6 after
    assert (out["abd_curl_events_placebo"], out["abd_curl_events_during"]) == (0, 1)


def test_both_windows_are_scored_against_the_resting_bend_that_was_given():
    bend = _flat()
    bend[260:340] = 25.0                                        # the placebo and the stimulus window both sit 20 degrees up
    assert _counts(_placebo(bend, rest=5.0)) == (1, 1, 0, 0)    # against a rest of 5: a curl in each window ...
    assert _counts(_placebo(bend, rest=25.0)) == (0, 0, 0, 0)   # ... against one of 25 nothing happens
    assert _counts(_placebo(bend, rest=45.0)) == (0, 0, 1, 1)   # ... and against 45 the bend has dropped


def test_the_resting_bend_that_was_given_is_reported_and_the_events_follow_it():
    out = _placebo(_flat(), rest=-6.0)                          # a bend of 5 is 11 degrees above a rest of -6
    assert out["abd_rest_bend_loo_deg"] == -6.0 and _counts(out) == (1, 1, 0, 0)
    out = _placebo(_flat(), rest=16.0)                          # ... and 11 below 16
    assert out["abd_rest_bend_loo_deg"] == 16.0 and _counts(out) == (0, 0, 1, 1)


@pytest.mark.parametrize("rest", [float("nan"), float("inf"), float("-inf")])
def test_without_a_usable_resting_bend_the_counts_are_empty_but_the_window_facts_are_not(rest):
    out = _placebo(_excursion_at(262, 270), rest=rest)
    assert math.isnan(out["abd_rest_bend_loo_deg"]) and all(math.isnan(v) for v in _counts(out)), out
    assert out["pb_window_min"] == pytest.approx(40 / FPS10 / 60) and out["abd_base_ok"] is True
    assert (out["abd_tracked_frac_base"], out["abd_tracked_frac_placebo"], out["abd_tracked_frac_during"]) == (1, 1, 1)


def test_curls_and_straightenings_are_counted_apart_in_each_window():
    bend = _flat()
    bend[262:268] += 20.0                                       # a curl in the placebo window
    bend[310:316] -= 20.0                                       # a straightening in the stimulus window
    bend[320:326] -= 20.0                                       # and another one
    out = _placebo(bend)
    assert (out["abd_curl_events_placebo"], out["abd_straighten_events_placebo"]) == (1, 0)
    assert (out["abd_curl_events_during"], out["abd_straighten_events_during"]) == (0, 2)


def test_the_rise_and_the_minimum_duration_are_the_curl_settings():
    shallow = _excursion_at(262, 270, height=9.0)               # below rise_deg = 10
    assert _placebo(shallow)["abd_curl_events_placebo"] == 0 and _placebo(shallow, rise=8.0)["abd_curl_events_placebo"] == 1
    brief = _excursion_at(262, 264)                             # 2 frames = 0.2 s
    assert _placebo(brief)["abd_curl_events_placebo"] == 0 and _placebo(brief, min_s=0.2)["abd_curl_events_placebo"] == 1
    assert _placebo(_excursion_at(262, 264), fps=20.0)["abd_curl_events_placebo"] == 0       # 2 frames at 20 fps are 0.1 s


def test_a_missing_frame_breaks_a_placebo_window_run_it_is_never_bridged():
    bend = _excursion_at(262, 272)
    bend[267] = np.nan
    assert _placebo(bend)["abd_curl_events_placebo"] == 2         # 5 frames and 4 frames, each long enough
    bend[[263, 266, 269]] = np.nan
    assert _placebo(bend)["abd_curl_events_placebo"] == 0         # pieces of 1, 2, 2 and 2 frames: a bridged run would count


def test_the_tracked_fractions_are_those_of_each_window():
    bend = _flat()
    bend[0:65] = np.nan                                         # a quarter of the 260-frame baseline
    bend[260:270] = np.nan                                      # a quarter of the 40-frame placebo window
    out = _placebo(bend)
    assert out["abd_tracked_frac_base"] == pytest.approx(0.75) and out["abd_tracked_frac_placebo"] == pytest.approx(0.75)
    assert out["abd_tracked_frac_during"] == 1.0


def test_whether_the_trials_own_baseline_can_be_lent_to_the_others_does_not_decide_its_own_scoring():
    ok, poor = _flat(), _flat()
    ok[:130] = np.nan                                           # exactly half of the 260 frames: at the gate, so usable
    poor[:131] = np.nan
    assert _placebo(ok)["abd_base_ok"] is True and _placebo(poor)["abd_base_ok"] is False
    out = _placebo(poor)
    assert out["abd_tracked_frac_base"] == pytest.approx(129 / 260)
    assert _counts(out) == (0, 0, 0, 0)                         # scored all the same: its rest came from the other trials


def test_each_window_is_judged_on_its_own_tracking_against_the_given_resting_bend():
    bend = _flat()
    bend[260:281] = np.nan                                      # 21 of the 40 placebo frames missing: 0.475
    out = _placebo(bend)
    assert out["abd_rest_bend_loo_deg"] == 5.0 and out["abd_tracked_frac_placebo"] == pytest.approx(0.475)
    assert math.isnan(out["abd_curl_events_placebo"]) and math.isnan(out["abd_straighten_events_placebo"])
    assert out["abd_curl_events_during"] == 0 and out["abd_straighten_events_during"] == 0     # measured, not blank
    bend = _flat()
    bend[300:321] = np.nan
    out = _placebo(bend)
    assert out["abd_curl_events_placebo"] == 0 and math.isnan(out["abd_curl_events_during"])
    bend = _flat()
    bend[300:320] = np.nan                                      # exactly half: usable
    assert _placebo(bend)["abd_curl_events_during"] == 0


def test_a_window_with_no_tracked_bend_has_no_events_even_when_the_gate_is_open():
    bend = _flat()
    bend[300:340] = np.nan
    out = _placebo(bend, gate=0.0)
    assert out["abd_tracked_frac_during"] == 0 and math.isnan(out["abd_curl_events_during"])     # not 0: nothing was seen
    assert out["abd_curl_events_placebo"] == 0                                                   # the other window is fine
    bend = _flat()
    bend[:260] = np.nan                                         # no baseline frame at all: nothing to lend, any gate
    assert _placebo(bend, gate=0.0)["abd_base_ok"] is False


@pytest.mark.parametrize("fps", [0.0, -10.0, float("nan"), float("inf")])
def test_a_frame_rate_that_cannot_be_used_leaves_every_column_empty(fps):
    out = _placebo(_excursion_at(262, 270), fps=fps)
    assert out["abd_base_ok"] is False and tuple(out) == M.PLACEBO_COLUMNS
    assert all(math.isnan(v) for k, v in out.items() if k != "abd_base_ok"), out


def test_the_input_is_left_alone_and_a_list_is_as_good_as_an_array():
    bend = _excursion_at(262, 270)
    before = bend.copy()
    as_array = _placebo(bend)
    np.testing.assert_array_equal(bend, before)
    assert _placebo(list(bend)) == as_array


# ---- the early baseline a trial lends to the other trials of its fly --------------------------------------------------

def test_the_early_baseline_is_everything_before_the_placebo_window():
    ramp = (0.05 * np.arange(400.0)) ** 2                   # every frame has its own value, so a one-frame shift shows
    for onset, offset, frames in ((300, 340, 40), (100, 400, 50), (300, None, 100)):
        got = M.baseline_values(ramp, onset=onset, offset=offset, min_tracked_frac=0.5)
        np.testing.assert_array_equal(got, ramp[:onset - frames])


def test_the_early_baseline_never_holds_a_placebo_or_a_during_frame():
    bend = _flat()
    bend[260:340] = 1e6                                     # the placebo and the during window of this trial
    got = M.baseline_values(bend, onset=300, offset=340, min_tracked_frac=0.5)
    assert got.size == 260 and got.max() == 5.0


def test_the_early_baseline_keeps_only_the_frames_that_were_tracked():
    bend = np.arange(400.0)
    bend[[3, 10, 11, 100]] = np.nan
    got = M.baseline_values(bend, onset=300, offset=340, min_tracked_frac=0.5)
    np.testing.assert_array_equal(got, np.delete(np.arange(260.0), [3, 10, 11, 100]))


def test_an_early_baseline_is_lent_only_when_it_is_tracked_in_the_gate_fraction_of_its_frames():
    bend = _flat()
    bend[:130] = np.nan                                     # exactly half of the 260 frames
    assert M.baseline_values(bend, onset=300, offset=340, min_tracked_frac=0.5).size == 130
    bend[130] = np.nan
    assert M.baseline_values(bend, onset=300, offset=340, min_tracked_frac=0.5) is None
    assert M.baseline_values(bend, onset=300, offset=340, min_tracked_frac=0.4).size == 129


def test_an_early_baseline_without_a_single_tracked_frame_is_never_lent():
    bend = _flat()
    bend[:260] = np.nan
    assert M.baseline_values(bend, onset=300, offset=340, min_tracked_frac=0.0) is None


def test_the_trial_says_whether_its_baseline_can_be_lent():
    for bend in (_flat(), np.where(np.arange(400) < 130, np.nan, 5.0), np.where(np.arange(400) < 131, np.nan, 5.0),
                 np.full(400, np.nan)):
        lent = M.baseline_values(bend, onset=300, offset=340, min_tracked_frac=0.5) is not None
        assert _placebo(bend)["abd_base_ok"] is lent


# ---- the leave-one-out resting bend --------------------------------------------------------------------------------------

def test_the_resting_bend_of_a_trial_is_the_median_of_the_other_trials_baselines_pooled():
    baselines = [np.full(10, 1.0), np.full(10, 2.0), np.full(10, 3.0)]
    assert M.leave_one_out_rest(baselines) == [2.5, 2.0, 1.5]       # 2 and 3 pooled; 1 and 3 pooled; 1 and 2 pooled


def test_the_pooled_frames_are_the_median_not_the_median_of_the_trial_medians():
    long, short = np.full(100, 1.0), np.full(10, 9.0)
    assert M.leave_one_out_rest([np.full(10, 4.0), long, short])[0] == 1.0        # the median of 110 frames, not of (1, 9)


def test_a_trial_never_enters_its_own_resting_bend():
    rest = M.leave_one_out_rest([np.full(10, 5.0), np.full(10, 5.0), np.full(10, 1000.0)])
    assert rest[2] == 5.0                                           # the extreme trial's own rest does not hold it ...
    assert rest[0] == rest[1] == 502.5                              # ... the rest of the others does (5s and 1000s pooled)


def test_a_resting_bend_needs_at_least_two_other_trials_with_a_baseline():
    a, b, c = np.full(10, 1.0), np.full(10, 2.0), np.full(10, 3.0)
    assert all(math.isnan(r) for r in M.leave_one_out_rest([a])) and all(math.isnan(r) for r in M.leave_one_out_rest([a, b]))
    assert M.leave_one_out_rest([a, b, c]) == [2.5, 2.0, 1.5]
    only_one = M.leave_one_out_rest([a, None, c])                    # a has c alone; c has a alone; the middle trial both
    assert math.isnan(only_one[0]) and math.isnan(only_one[2]) and only_one[1] == 2.0


def test_a_trial_that_cannot_lend_a_baseline_still_gets_a_resting_bend_from_the_others():
    rest = M.leave_one_out_rest([None, np.full(10, 4.0), np.full(10, 6.0), None])
    assert rest[0] == rest[3] == 5.0 and math.isnan(rest[1]) and math.isnan(rest[2])


def test_the_number_of_other_trials_that_are_required_can_be_changed():
    a, b = np.full(10, 1.0), np.full(10, 3.0)
    assert M.leave_one_out_rest([a, b], min_trials=1) == [3.0, 1.0]
    assert all(math.isnan(r) for r in M.leave_one_out_rest([a, b, a], min_trials=3))


def test_the_resting_bends_are_returned_in_the_order_of_the_trials_and_the_input_is_left_alone():
    baselines = [np.array([1.0, 2.0, 3.0]), np.array([10.0, 20.0, 30.0]), np.array([100.0, 200.0, 300.0])]
    before = [b.copy() for b in baselines]
    assert M.leave_one_out_rest(baselines) == [np.median([10, 20, 30, 100, 200, 300]), np.median([1, 2, 3, 100, 200, 300]),
                                               np.median([1, 2, 3, 10, 20, 30])]
    assert all(np.array_equal(a, b) for a, b in zip(baselines, before))


def test_no_trials_and_no_baselines_give_no_resting_bends():
    assert M.leave_one_out_rest([]) == []
    assert len(M.leave_one_out_rest([None, None, None])) == 3
    assert all(math.isnan(r) for r in M.leave_one_out_rest([None, None, None]))


def test_frames_that_are_not_finite_never_enter_a_resting_bend():
    rest = M.leave_one_out_rest([np.array([np.nan, 1.0, 1.0]), np.array([5.0, np.inf, 5.0]), np.array([3.0, 3.0, np.nan])])
    assert rest == [np.median([5.0, 5.0, 3.0, 3.0]), np.median([1.0, 1.0, 3.0, 3.0]), np.median([1.0, 1.0, 5.0, 5.0])]
    assert all(math.isnan(r) for r in M.leave_one_out_rest([np.array([np.nan]), np.array([1.0]), np.array([2.0])])[1:])


# ---- one fly: two passes ------------------------------------------------------------------------------------------------

def _trial(bend, onset=300, offset=340, fps=FPS10):
    return M.PlaceboTrial(bend, onset, offset, fps)


def _fly(trials, **kw):
    return M.placebo_fly(trials, **{"rise_deg": 10.0, "min_s": 0.3, "min_tracked_frac": 0.5, **kw})


def test_one_fly_gives_one_row_of_the_documented_columns_per_trial_in_order():
    rows = _fly([_trial(_flat(level=5.0)), _trial(_flat(level=7.0)), _trial(_flat(level=9.0))])
    assert len(rows) == 3 and all(tuple(r) == M.PLACEBO_COLUMNS for r in rows)
    assert [r["abd_rest_bend_loo_deg"] for r in rows] == [8.0, 7.0, 6.0]          # the other two trials' baselines, pooled
    assert _fly([]) == []


def test_each_trial_is_scored_against_the_other_trials_and_not_against_itself():
    rows = _fly([_trial(_flat(level=5.0)), _trial(_flat(level=7.0)), _trial(_flat(level=9.0))], rise_deg=1.0)
    assert _counts(rows[0]) == (0, 0, 1, 1)                      # 5 is below the 8 of the others: a straightening
    assert _counts(rows[1]) == (0, 0, 0, 0)                      # 7 is the 7 of the others
    assert _counts(rows[2]) == (1, 1, 0, 0)                      # 9 is above the 6 of the others: a curl


def test_no_placebo_or_during_frame_of_any_trial_enters_a_resting_bend():
    def trial():                                                  # 30 frames of baseline at 5, then 60 frames at 25
        bend = np.full(120, 25.0)
        bend[:30] = 5.0
        return _trial(bend, onset=60, offset=100)                 # L = 30: baseline 0-30, placebo 30-60, during 60-90
    rows = _fly([trial(), trial(), trial()])
    assert [r["abd_rest_bend_loo_deg"] for r in rows] == [5.0, 5.0, 5.0]
    assert all(_counts(r) == (1, 1, 0, 0) for r in rows)          # 25 is 20 above 5 in both windows of every trial


def test_a_fly_needs_three_trials_with_a_usable_baseline_for_any_of_them_to_be_scored():
    rows = _fly([_trial(_flat()), _trial(_flat())])
    assert all(math.isnan(r["abd_rest_bend_loo_deg"]) and all(math.isnan(v) for v in _counts(r)) for r in rows)
    assert all(r["abd_base_ok"] is True and r["pb_window_min"] == pytest.approx(40 / FPS10 / 60) for r in rows)
    assert all(_counts(r) == (0, 0, 0, 0) for r in _fly([_trial(_flat()), _trial(_flat()), _trial(_flat())]))


def test_a_baseline_tracked_in_too_few_frames_lends_nothing_to_the_other_trials():
    poor = _flat(level=6.0)
    poor[:131] = np.nan                                           # 129 of the 260 baseline frames: under the gate
    poor[131:260] = -100.0                                        # ... and what is left of it is far from everything
    rows = _fly([_trial(_flat(level=5.0)), _trial(_flat(level=5.0)), _trial(_flat(level=7.0)), _trial(poor)])
    assert [r["abd_base_ok"] for r in rows] == [True, True, True, False]
    assert [r["abd_rest_bend_loo_deg"] for r in rows] == [6.0, 6.0, 5.0, 5.0]    # no -100 in any of them
    assert _counts(rows[3]) == (0, 0, 0, 0)                       # and it is scored all the same, against the others


def test_a_poor_baseline_that_leaves_one_other_trial_gives_no_resting_bend():
    poor = _flat()
    poor[:131] = np.nan
    rows = _fly([_trial(_flat()), _trial(_flat()), _trial(poor)])
    assert math.isnan(rows[0]["abd_rest_bend_loo_deg"]) and math.isnan(rows[1]["abd_rest_bend_loo_deg"])   # one lender each
    assert rows[2]["abd_rest_bend_loo_deg"] == 5.0 and _counts(rows[2]) == (0, 0, 0, 0)                  # two lenders


def test_a_trial_without_a_known_stimulus_lends_nothing_and_is_not_scored():
    rows = _fly([_trial(_flat(), onset=None, offset=None), _trial(_flat()), _trial(_flat())])
    assert rows[0]["abd_base_ok"] is False and all(math.isnan(v) for k, v in rows[0].items() if k != "abd_base_ok")
    assert math.isnan(rows[1]["abd_rest_bend_loo_deg"])           # only one other trial can lend: not enough
    rows = _fly([_trial(_flat(), onset=None, offset=None), _trial(_flat()), _trial(_flat()), _trial(_flat())])
    assert [r["abd_rest_bend_loo_deg"] for r in rows[1:]] == [5.0, 5.0, 5.0]


def test_a_trial_without_a_usable_frame_rate_lends_nothing_and_is_not_scored():
    rows = _fly([_trial(_flat(), fps=0.0), _trial(_flat()), _trial(_flat())])
    assert rows[0]["abd_base_ok"] is False and all(math.isnan(v) for k, v in rows[0].items() if k != "abd_base_ok")
    assert math.isnan(rows[1]["abd_rest_bend_loo_deg"]) and math.isnan(rows[2]["abd_rest_bend_loo_deg"])   # one lender each
    rows = _fly([_trial(_flat(), fps=float("nan")), _trial(_flat()), _trial(_flat()), _trial(_flat())])
    assert [r["abd_rest_bend_loo_deg"] for r in rows[1:]] == [5.0, 5.0, 5.0]


def test_the_minimum_number_of_other_trials_is_a_setting_of_the_fly():
    rows = _fly([_trial(_flat()), _trial(_flat())], min_trials=1)
    assert [r["abd_rest_bend_loo_deg"] for r in rows] == [5.0, 5.0]


def test_the_settings_of_a_fly_reach_every_trial():
    bump = _excursion_at(262, 264, height=15.0)                   # 2 frames of 15 degrees in the placebo window
    flat = [_trial(_flat()), _trial(_flat())]
    assert _counts(_fly([_trial(bump), *flat])[0]) == (0, 0, 0, 0)
    assert _counts(_fly([_trial(bump), *flat], min_s=0.2)[0]) == (1, 0, 0, 0)
    assert _counts(_fly([_trial(bump), *flat], min_s=0.2, rise_deg=16.0)[0]) == (0, 0, 0, 0)
    assert math.isnan(_fly([_trial(bump), *flat], min_tracked_frac=1.1)[0]["abd_rest_bend_loo_deg"])


def test_each_trial_of_a_fly_keeps_its_own_frame_rate_and_stimulus():
    rows = _fly([_trial(_flat(), fps=20.0), _trial(_flat(400), onset=200, offset=240), _trial(_flat())])
    assert [r["pb_window_min"] for r in rows] == [pytest.approx(40 / 20.0 / 60), pytest.approx(40 / 10.0 / 60),
                                                  pytest.approx(40 / 10.0 / 60)]


# ---- calibration: a stimulus that changes nothing ----------------------------------------------------------------------

FPS_SIM = 10.0
ONSET_SIM, N_SIM = 300, 450      # 30 s before the stimulus; the trace ends 15 s into it, so L = 15 s = 150 frames
TRIALS_PER_FLY = 10


def _ar1_bend(rng, *, tau_s=3.0, sd=4.0, drift=0.0, n=N_SIM, level=6.0, fps=FPS_SIM):
    """A bend trace: an AR(1) with time constant ``tau_s`` and stationary SD ``sd`` degrees, plus (optionally) a random walk
    of ``drift`` degrees per sqrt(second) -- a fly that slowly changes its resting posture inside the trial."""
    a = math.exp(-1.0 / (tau_s * fps))
    noise = rng.normal(0.0, sd * math.sqrt(1.0 - a * a), n)
    bend, _ = signal.lfilter([1.0], [1.0, -a], noise, zi=[a * rng.normal(0.0, sd)])
    if drift:
        bend = bend + np.cumsum(rng.normal(0.0, drift / math.sqrt(fps), n))
    return level + bend


def _cohort(rng, flies, *, trials=TRIALS_PER_FLY, fly_sd=3.0, evoked=0.0, **sim):
    """``flies`` flies of ``trials`` trials: each fly has a resting level of its own (SD ``fly_sd`` degrees) and every trial
    an AR(1) bend around it; ``evoked`` = mean number of 0.5 s, +15 degree curls added to the first 15 s of the stimulus."""
    cohort = []
    for _ in range(flies):
        level, traces = rng.normal(0.0, fly_sd), []
        for _ in range(trials):
            bend = _ar1_bend(rng, **sim) + level
            for start in rng.integers(ONSET_SIM, ONSET_SIM + 130, rng.poisson(evoked)):
                bend[start:start + 5] += 15.0
            traces.append(bend)
        cohort.append(traces)
    return cohort


def _leave_one_out_rows(cohort):
    """The production path: every fly's trials through ``placebo_fly``."""
    rows = []
    for traces in cohort:
        rows += M.placebo_fly([M.PlaceboTrial(b, ONSET_SIM, None, FPS_SIM) for b in traces], rise_deg=10.0, min_s=0.3,
                              min_tracked_frac=0.5)
    return pd.DataFrame(rows)


def _adjacent_rows(cohort):
    """The design before this commit, for contrast: each trial scored against the median of ITS OWN early baseline."""
    rows = []
    for traces in cohort:
        for bend in traces:
            rest = float(np.median(M.baseline_values(bend, onset=ONSET_SIM, offset=None, min_tracked_frac=0.5)))
            rows.append(M.placebo_events(bend, onset=ONSET_SIM, offset=None, fps=FPS_SIM, rise_deg=10.0, min_s=0.3,
                                         min_tracked_frac=0.5, rest_deg=rest))
    return pd.DataFrame(rows)


def _swap_p(df, event, seed=0, n=199):
    return M.paired_window_permutation(df, events_pre=f"abd_{event}_events_placebo", min_pre="pb_window_min",
                                       events_stim=f"abd_{event}_events_during", min_stim="pb_window_min", n=n, seed=seed,
                                       scheme="swap")[1]


def _false_positive_rates(reps, *, flies, design=_leave_one_out_rows, **sim):
    hits = dict.fromkeys(("curl", "straighten"), 0)
    for rep in range(reps):
        df = design(_cohort(np.random.default_rng(5000 + rep), flies, **sim))
        for event in hits:
            hits[event] += _swap_p(df, event, seed=rep) <= 0.05
    return {event: count / reps for event, count in hits.items()}


def _mean(rates):
    return sum(rates.values()) / len(rates)


@pytest.mark.parametrize("flies, tau_s, reps", [(20, 3.0, 80), (20, 10.0, 80), (40, 10.0, 40)],
                         ids=["200 trials, tau 3 s", "200 trials, tau 10 s", "400 trials, tau 10 s"])
def test_a_stimulus_that_changes_nothing_is_not_called_significant_with_a_leave_one_out_baseline(flies, tau_s, reps):
    """Flies x 10 trials of a stationary AR(1) bend with a resting level per fly and no stimulus effect: the false-positive
    rate (the mean of the curl and the straightening comparison) of the real events + the leave-one-out placebo design +
    the window-swap test at 0.05 stays under 0.09 with 200 and 400 pooled trials, at a bend time constant of 3 and 10 s.
    The baseline of the same trial next to the placebo window gave 0.35 and 0.64 here (see the next test); measured over
    1000 replicates the design gives 0.003 to 0.05 from 40 to 400 trials at 3, 5 and 10 s."""
    rates = _false_positive_rates(reps, flies=flies, tau_s=tau_s)
    assert _mean(rates) <= 0.09, rates


def test_a_baseline_taken_from_the_same_trial_next_to_the_placebo_window_is_not_calibrated():
    """For contrast, and so that the test above can fail: with the baseline of the SAME trial the placebo window sits next
    to it, a slowly wandering bend (tau 10 s) is still near its median there and farther from it a window later, and
    pooling 200 trials turns that into a stimulus effect that is not there (0.35 over 1000 replicates)."""
    rates = _false_positive_rates(30, flies=20, tau_s=10.0, design=_adjacent_rows)
    assert _mean(rates) > 0.15, rates


def test_a_stimulus_that_adds_curls_is_found_with_a_leave_one_out_baseline():
    for rep in range(3):
        df = _leave_one_out_rows(_cohort(np.random.default_rng(700 + rep), 10, evoked=0.5))
        assert _swap_p(df, "curl", seed=rep, n=999) < 0.01, rep
        assert df["abd_curl_events_during"].sum() > 3 * df["abd_curl_events_placebo"].sum()
        assert _swap_p(df, "straighten", seed=rep) > 0.01                    # and nothing else moved


def test_a_bend_that_drifts_steadily_inside_the_trial_is_the_known_limit_of_the_design():
    """Documents the limitation rather than a goal: a random-walk drift of 1 degree / sqrt(s) makes the start of the
    stimulus differ from the window before it with no stimulus effect (0.35 over 1000 replicates of 200 trials; 0.07 and 0.11
    at 0.3 degree / sqrt(s) with 200 and 400 trials). If this ever passes the other way, the docstrings that quote these
    numbers need the new ones."""
    rates = _false_positive_rates(30, flies=20, tau_s=3.0, drift=1.0)
    assert _mean(rates) > 0.15, rates


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
        for scheme in ("swap", "split"):
            assert all(math.isnan(v) for v in M.paired_window_permutation(
                pd.DataFrame({"a": nan, "b": nan, "c": nan, "d": nan}), events_pre="a", min_pre="b", events_stim="c",
                min_stim="d", n=10, scheme=scheme))


def test_the_module_imports_nothing_from_the_pipeline():
    """Pure maths: importing it must not pull in any other fbpipe module (steps, config, geometry...)."""
    code = ("import sys; import fbpipe.analysis.mounting; "
            "print(sorted(m for m in sys.modules if m == 'fbpipe' or m.startswith('fbpipe.')))")
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "['fbpipe', 'fbpipe.analysis', 'fbpipe.analysis.mounting']"
