"""Tests for the one-sided (upper) MAD response threshold in envelope_visuals.

The per-trial "red line" threshold must be raised only by *upward* variability in
the pre-odor baseline. Downward dips below the resting median (the opposite of a
proboscis extension) must not push the threshold up.
"""

import sys
from pathlib import Path

import math

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)

from scripts.analysis import envelope_visuals as ev  # noqa: E402

K = 3.0  # threshold_std_mult default


def _symmetric_theta(window: np.ndarray, std_mult: float = K) -> float:
    """The OLD symmetric-MAD formula, for documenting the behavior change."""
    baseline = float(np.nanmedian(window))
    mad = float(np.nanmedian(np.abs(window - baseline)))
    return float(baseline + std_mult * 1.4826 * mad)


# Resting ~30 with tight *upward* jitter; no downward excursions.
_BASE = np.array(
    [30, 30, 30, 31, 30, 30, 32, 30, 31, 30, 30, 33, 30, 30, 31], dtype=float
)
# Same trace, but three resting samples replaced by deep downward dips.
_DIPPED = _BASE.copy()
_DIPPED[[2, 7, 13]] = np.array([3, 1, 5], dtype=float)


def _theta(window: np.ndarray) -> float:
    # fps=1.0 and baseline_until_s=len -> the whole window is the baseline.
    return ev._compute_theta(window, fps=1.0, baseline_until_s=window.size, std_mult=K)


def test_downward_dips_do_not_change_threshold():
    """One-sided theta is invariant to downward dips below the resting median."""
    assert np.isclose(_theta(_BASE), _theta(_DIPPED))
    # The dips don't move the median, so the resting center is unchanged too.
    assert np.isclose(np.nanmedian(_BASE), np.nanmedian(_DIPPED))


def test_old_symmetric_formula_was_inflated_by_dips():
    """Documents the bug being fixed: symmetric MAD *was* raised by the dips."""
    assert _symmetric_theta(_DIPPED) > _symmetric_theta(_BASE)


def test_upward_jitter_still_raises_threshold():
    """Upward variability must still lift the line above the resting median."""
    flat = np.full(12, 30.0)
    assert np.isclose(_theta(flat), 30.0)  # no upward spread -> theta == median
    assert _theta(_BASE) > 30.0  # upward jitter present -> theta above resting


def test_raising_an_above_median_sample_raises_threshold():
    """Larger upward deviations increase theta (sensitivity preserved)."""
    higher = _BASE.copy()
    higher[higher > 30] += 10.0  # push the upward samples further up
    assert _theta(higher) > _theta(_BASE)


# --------------------------------------------------------------------------- #
# Floor and quiet-window guards.
#
# Two independent failure modes seen in Hex-Control-24-0.1:
#
#   1. A *flat* baseline drives sigma_up toward zero, so theta collapses onto the
#      resting median and a few units of drift during the odor read as a full
#      response. Real case: august_15_batch_2_rig_2 fly 1 training_6, baseline
#      6.6 +- ~1, theta = 9.5, odor-window peak 17.7 -> 51% of frames "over" with
#      no visible extension.
#
#   2. A single spontaneous burst *inside* the baseline inflates sigma_up and
#      lifts theta above the real response. Real case: august_11_batch_1_rig_3
#      fly 4 testing_4, baseline flat at ~30 for 15 s then one excursion to 74,
#      theta = 56.4, odor-window peak 84.6 -> only 23% over.
#
# Neither is fixable by moving k: it moves both thresholds the same direction.
# --------------------------------------------------------------------------- #


def test_min_delta_defaults_to_no_change():
    """The floor is opt-in; omitting it reproduces the shipped threshold."""
    assert np.isclose(ev._baseline_theta(_BASE, K), ev._baseline_theta(_BASE, K, min_delta=0.0))


def test_min_delta_raises_a_flat_baseline_threshold():
    """Failure mode 1: a quiet baseline must not yield a hair-trigger line."""
    flat = np.full(40, 6.6)
    flat[::7] += 0.3  # sigma_up tiny but non-zero, as in the real trace
    bare = ev._baseline_theta(flat, K)
    floored = ev._baseline_theta(flat, K, min_delta=10.0)
    # The problem: theta lands ~1 unit above rest, so a 7-unit drift reads as PER.
    assert bare - float(np.median(flat)) < 1.5
    assert np.isclose(floored, float(np.median(flat)) + 10.0)


def test_min_delta_does_not_lower_an_already_wide_threshold():
    """The floor is a max(), never a clamp down onto a noisy baseline."""
    noisy = np.concatenate([np.full(30, 30.0), np.linspace(30.0, 90.0, 30)])
    bare = ev._baseline_theta(noisy, K)
    assert bare > float(np.median(noisy)) + 10.0
    assert np.isclose(ev._baseline_theta(noisy, K, min_delta=10.0), bare)


def test_rolling_anchor_ignores_a_burst_late_in_the_baseline():
    """Failure mode 2: one pre-odor burst must not lift theta over the response."""
    fps = 10.0
    quiet = np.full(150, 30.0) + np.tile([0.0, 0.4], 75)   # 15 s steady at rest
    burst = np.concatenate([np.full(50, 74.0), np.full(100, 36.0)])  # 15 s active
    during = np.linspace(30.0, 85.0, 300)
    env = np.concatenate([quiet, burst, during])

    contaminated = ev._compute_theta(env, fps=fps, baseline_until_s=30.0, std_mult=2.0)
    rolled = ev._compute_theta(
        env, fps=fps, baseline_until_s=30.0, std_mult=2.0, anchor_s=5.0
    )
    assert rolled < contaminated
    assert float(np.mean(during > rolled)) > float(np.mean(during > contaminated))


def test_rolling_anchor_tracks_a_step_the_fly_never_came_down_from():
    """Failure mode 3, the benzaldehyde case.

    The fly extends partway through the baseline and *holds* there until odor
    onset. The whole-window median then describes a resting position the proboscis
    left 13 s ago, and the step inflates sigma on top of that. Anchoring on the
    last few seconds reports where the proboscis actually is at onset.
    """
    fps = 10.0
    rest = np.full(150, 3.6)                      # 15 s at rest
    step = np.concatenate([np.full(20, 61.0), np.full(130, 23.6)])  # extend, then hold
    during = np.full(300, 42.0)                   # a further, real extension
    env = np.concatenate([rest, step, during])

    contaminated = ev._compute_theta(env, fps=fps, baseline_until_s=30.0, std_mult=2.0)
    rolled = ev._compute_theta(
        env, fps=fps, baseline_until_s=30.0, std_mult=2.0, min_delta=6.0, anchor_s=5.0
    )
    assert contaminated > 42.0            # the shipped line sits above the response
    assert np.isclose(rolled, 23.6 + 6.0, atol=0.5)  # held position + the floor
    assert rolled < 42.0                  # ...so the response is detected


def test_rolling_anchor_does_not_chase_the_resting_median_down():
    """The anchor is the *held* position, not the quietest stretch of the window.

    Choosing the calmest sub-window instead would pick the 3.6 rest stretch and
    put theta near zero, marking the entire odor window as a response.
    """
    fps = 10.0
    env = np.concatenate(
        [np.full(150, 3.6), np.full(150, 23.6), np.full(300, 42.0)]
    )
    rolled = ev._compute_theta(
        env, fps=fps, baseline_until_s=30.0, std_mult=2.0, min_delta=6.0, anchor_s=5.0
    )
    assert rolled > 20.0


def test_rolling_defaults_to_the_whole_baseline():
    """Opt-in: no anchor_s means the shipped whole-window estimate."""
    env = np.concatenate([_BASE, np.full(20, 60.0)])
    assert np.isclose(
        ev._compute_theta(env, fps=1.0, baseline_until_s=_BASE.size, std_mult=K),
        ev._compute_theta(
            env, fps=1.0, baseline_until_s=_BASE.size, std_mult=K, anchor_s=None
        ),
    )


def test_rolling_anchor_longer_than_baseline_uses_what_exists():
    """An anchor longer than the baseline must not return NaN or crash."""
    env = np.concatenate([_BASE, np.full(20, 60.0)])
    th = ev._compute_theta(
        env, fps=1.0, baseline_until_s=_BASE.size, std_mult=K, anchor_s=999.0
    )
    assert math.isfinite(th)
    # The anchor spans the whole baseline, so the location is just its median; the
    # rolling scale then adds this window's own upward jitter on top.
    assert th >= float(np.median(_BASE))
    # ...and it lands in the same neighbourhood as the whole-window estimate --
    # a 25th-percentile rolling sigma is a slightly wider read of the same jitter.
    assert np.isclose(th, ev._baseline_theta(_BASE, K), rtol=0.10)


def test_rolling_noise_floor_ignores_a_minority_of_loud_windows():
    """The scale is a low percentile of rolling sigma, not the mean of it."""
    fps = 10.0
    calm = np.full(200, 20.0) + np.tile([0.0, 0.2], 100)
    loud = np.full(100, 20.0) + np.tile([0.0, 30.0], 50)
    _, scale = ev._rolling_baseline(
        np.concatenate([calm, loud]), fps, anchor_s=5.0, noise_block_s=2.0, noise_pctl=10.0
    )
    assert scale < 1.0


def test_floor_and_rolling_compose():
    """Both guards together: local location, floored excursion."""
    fps = 10.0
    env = np.concatenate(
        [np.full(150, 6.6), np.full(50, 74.0), np.full(100, 6.6), np.full(300, 14.0)]
    )
    th = ev._compute_theta(
        env, fps=fps, baseline_until_s=30.0, std_mult=2.0, min_delta=10.0, anchor_s=5.0
    )
    assert np.isclose(th, 16.6)  # anchor median 6.6, noise floor 0 -> floored to +10
