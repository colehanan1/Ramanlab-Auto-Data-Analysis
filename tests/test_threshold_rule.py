"""The one θ implementation the whole pipeline shares.

Before this module there were nine separate copies of "median + k·MAD" across
scripts/, and they did not agree:

* ``envelope_visuals`` used the **one-sided upper** MAD with k from config (2.0);
* ``envelope_combined._compute_trial_metrics`` -- the function that produces the
  ``AUC-*`` columns of the wide table, i.e. the numbers every downstream figure
  is built on -- used the **symmetric** MAD with k hardcoded to **3.0**, and
  optionally a *fly-level* rather than per-trial baseline;
* ``per_folder_envelope_traces*.py``, which draw the red line the user reads off
  the trace figures, used the **symmetric** MAD.

So the red line in a figure and the AUC that figure's caption quotes were
computed from different thresholds. These tests pin the shared rule.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fbpipe.analysis.threshold import (  # noqa: E402
    ThresholdRule,
    baseline_theta,
    compute_theta,
    rolling_baseline,
    upper_sigma,
)


# --------------------------------------------------------------------------- #
# upper_sigma
# --------------------------------------------------------------------------- #


def test_upper_sigma_matches_mad_for_symmetric_noise():
    """1.4826 * median(positive deviations) == 1.4826 * MAD when noise is symmetric.

    For X ~ N(0, s) the median of the positive half is the 75th percentile,
    0.6745 s, which is also median|X|. So the scaling constant carries over and
    ``k`` keeps its "number of SDs" reading.
    """
    rng = np.random.default_rng(0)
    x = rng.normal(50.0, 4.0, 20000)
    assert upper_sigma(x) == pytest.approx(4.0, rel=0.05)


def test_upper_sigma_ignores_downward_dips():
    base = np.array([30, 30, 31, 30, 32, 30, 31, 30, 33, 30], dtype=float)
    dipped = base.copy()
    dipped[[1, 5]] = [2.0, 1.0]
    assert upper_sigma(base) == pytest.approx(upper_sigma(dipped))


def test_upper_sigma_is_zero_without_upward_samples():
    assert upper_sigma(np.full(10, 7.0)) == 0.0


# --------------------------------------------------------------------------- #
# baseline_theta / the floor
# --------------------------------------------------------------------------- #


def test_baseline_theta_floor_lifts_a_flat_baseline():
    flat = np.full(40, 6.6)
    flat[::7] += 0.3
    assert baseline_theta(flat, 2.0) - np.median(flat) < 1.5
    assert baseline_theta(flat, 2.0, min_delta=5.0) == pytest.approx(np.median(flat) + 5.0)


def test_baseline_theta_floor_never_clamps_down():
    noisy = np.concatenate([np.full(30, 30.0), np.linspace(30.0, 90.0, 30)])
    bare = baseline_theta(noisy, 2.0)
    assert bare > np.median(noisy) + 5.0
    assert baseline_theta(noisy, 2.0, min_delta=5.0) == pytest.approx(bare)


def test_baseline_theta_empty_window_is_nan():
    assert math.isnan(baseline_theta(np.array([]), 2.0))


# --------------------------------------------------------------------------- #
# rolling_baseline
# --------------------------------------------------------------------------- #


def test_rolling_anchor_reports_the_held_position_not_the_resting_one():
    """The benzaldehyde case: fly extends mid-baseline and holds until odor on."""
    before = np.concatenate([np.full(150, 3.6), np.full(150, 23.6)])
    loc, scale = rolling_baseline(before, fps=10.0, anchor_s=5.0,
                                  noise_block_s=2.0, noise_pctl=10.0)
    assert loc == pytest.approx(23.6)
    assert scale < 0.5


def test_rolling_noise_floor_ignores_a_minority_of_loud_windows():
    calm = np.full(200, 20.0) + np.tile([0.0, 0.2], 100)
    loud = np.full(100, 20.0) + np.tile([0.0, 30.0], 50)
    _, scale = rolling_baseline(np.concatenate([calm, loud]), fps=10.0, anchor_s=5.0,
                                noise_block_s=2.0, noise_pctl=10.0)
    assert scale < 1.0


def test_rolling_baseline_handles_an_all_nan_window():
    loc, scale = rolling_baseline(np.full(50, np.nan), fps=10.0, anchor_s=5.0,
                                  noise_block_s=2.0, noise_pctl=10.0)
    assert math.isnan(loc)
    assert scale == 0.0


# --------------------------------------------------------------------------- #
# ThresholdRule
# --------------------------------------------------------------------------- #


def test_legacy_rule_is_the_shipped_behaviour():
    """ThresholdRule() with no arguments must reproduce median + k*upper-sigma."""
    env = np.concatenate([np.array([30, 30, 31, 30, 32, 30, 31, 30, 33, 30], float),
                          np.full(10, 60.0)])
    rule = ThresholdRule(std_mult=2.0)
    assert rule.theta(env, fps=1.0, baseline_until_s=10.0) == pytest.approx(
        baseline_theta(env[:10], 2.0)
    )


def test_rule_round_trips_through_a_config_mapping():
    """The pipeline passes these as plain YAML values; parsing must be lossless."""
    rule = ThresholdRule.from_mapping(
        {"threshold_std_mult": 2.0, "threshold_min_delta": 5.0,
         "threshold_anchor_s": 5.0, "threshold_noise_block_s": 2.0,
         "threshold_noise_pctl": 10.0}
    )
    assert rule == ThresholdRule(2.0, 5.0, 5.0, 2.0, 10.0)


def test_rule_from_mapping_ignores_unrelated_keys_and_fills_defaults():
    rule = ThresholdRule.from_mapping({"out_dir": "/tmp", "threshold_std_mult": 3.0})
    assert rule.std_mult == 3.0
    assert rule.min_delta == 0.0
    assert rule.anchor_s is None


def test_rule_from_mapping_treats_null_anchor_as_legacy():
    """YAML `threshold_anchor_s:` with no value must not become 0.0 seconds."""
    assert ThresholdRule.from_mapping({"threshold_anchor_s": None}).anchor_s is None


def test_rule_theta_is_nan_for_an_empty_trace():
    assert math.isnan(ThresholdRule().theta(np.array([]), fps=40.0, baseline_until_s=30.0))


def test_rule_theta_is_nan_for_nonpositive_fps():
    env = np.full(100, 5.0)
    assert math.isnan(ThresholdRule().theta(env, fps=0.0, baseline_until_s=30.0))


def test_new_rule_lowers_theta_when_a_burst_contaminates_the_baseline():
    fps = 10.0
    env = np.concatenate([np.full(150, 30.0), np.full(50, 74.0), np.full(100, 36.0),
                          np.linspace(30.0, 85.0, 300)])
    legacy = ThresholdRule(std_mult=2.0).theta(env, fps=fps, baseline_until_s=30.0)
    new = ThresholdRule(2.0, 5.0, 5.0).theta(env, fps=fps, baseline_until_s=30.0)
    assert new < legacy


def test_new_rule_raises_theta_when_the_baseline_is_flat():
    fps = 10.0
    env = np.concatenate([np.full(300, 6.6), np.full(300, 14.0)])
    legacy = ThresholdRule(std_mult=2.0).theta(env, fps=fps, baseline_until_s=30.0)
    new = ThresholdRule(2.0, 5.0, 5.0).theta(env, fps=fps, baseline_until_s=30.0)
    assert new > legacy
    assert new == pytest.approx(11.6)


def test_describe_names_the_active_rule():
    assert "legacy" in ThresholdRule(2.0).describe().lower()
    assert "anchor" in ThresholdRule(2.0, 5.0, 5.0).describe().lower()
