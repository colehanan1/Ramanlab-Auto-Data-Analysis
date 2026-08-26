"""The AUC columns must use the same θ as everything else.

``envelope_combined._compute_trial_metrics`` produces ``AUC-Before/During/After``
-- the numbers the wide table carries and every downstream figure is built from.
It used to compute its own threshold: **symmetric** MAD, k hardcoded to **3.0**,
against the one-sided MAD at k = 2.0 that the trace figures drew. These tests pin
it to the shared :class:`ThresholdRule`.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fbpipe.analysis.threshold import ThresholdRule  # noqa: E402
from scripts.analysis import envelope_combined as ec  # noqa: E402

FPS = 10.0
BEFORE_N, DURING_N = 300, 300


def _trace(before_level, during_level, *, jitter=0.0, burst=None):
    rng = np.random.default_rng(0)
    before = np.full(BEFORE_N, float(before_level))
    if jitter:
        before = before + np.abs(rng.normal(0.0, jitter, BEFORE_N))
    if burst is not None:
        start, n, level = burst
        before[start : start + n] = level
    during = np.full(DURING_N, float(during_level))
    return np.concatenate([before, during, np.full(DURING_N, float(before_level))])


def _metrics(env, rule, **kw):
    return ec._compute_trial_metrics(
        env, FPS, fallback_fps=FPS, default_fps=FPS,
        fly_before_median=float("nan"), use_per_trial_baseline=True,
        odor_on_frame=BEFORE_N, odor_off_frame=BEFORE_N + DURING_N,
        threshold_rule=rule, **kw
    )


def test_metrics_accept_a_threshold_rule():
    m = _metrics(_trace(10.0, 40.0), ThresholdRule(std_mult=2.0))
    assert m["AUC-During"] > 0.0


def test_floor_suppresses_a_small_drift_on_a_flat_baseline():
    """The flat-baseline false positive, measured in AUC rather than in pixels."""
    env = _trace(6.6, 13.0, jitter=0.15)
    loose = _metrics(env, ThresholdRule(std_mult=2.0))
    floored = _metrics(env, ThresholdRule(std_mult=2.0, min_delta=10.0))
    assert loose["AUC-During"] > 0.0
    assert floored["AUC-During"] == 0.0


def test_floor_does_not_touch_a_large_real_response():
    env = _trace(6.6, 60.0, jitter=0.15)
    loose = _metrics(env, ThresholdRule(std_mult=2.0))
    floored = _metrics(env, ThresholdRule(std_mult=2.0, min_delta=5.0))
    assert floored["AUC-During"] == pytest.approx(loose["AUC-During"], rel=0.15)


def test_anchor_recovers_a_response_a_baseline_burst_had_masked():
    """A spontaneous extension before odor on must not suppress the AUC after it.

    Shaped after august_11_batch_1_rig_3 fly 4 testing_4: flat at 30 for 15 s, one
    excursion to 74, then settled *elevated* at ~38 for the rest of the baseline.
    Enough of the window sits above the median that even the one-sided spread is
    inflated, so legacy theta lands at ~56 and clips most of a real response.
    """
    before = np.concatenate([
        np.full(150, 30.0),      # 15 s at rest
        np.full(10, 74.0),       # the excursion
        np.full(140, 38.0),      # settled higher, and still there at odor on
    ])
    env = np.concatenate([before, np.full(DURING_N, 60.0), np.full(DURING_N, 38.0)])

    legacy = _metrics(env, ThresholdRule(std_mult=2.0))
    anchored = _metrics(env, ThresholdRule(std_mult=2.0, min_delta=5.0, anchor_s=5.0))
    assert legacy["AUC-During"] < anchored["AUC-During"]


def test_rule_is_required_so_no_caller_silently_keeps_the_old_hardcoded_k():
    """The old code hardcoded k=3 on a symmetric MAD. That path must be gone."""
    import inspect

    sig = inspect.signature(ec._compute_trial_metrics)
    assert "threshold_rule" in sig.parameters
    src = inspect.getsource(ec._compute_trial_metrics)
    assert "3.0 * before_sigma" not in src


def test_per_trial_and_fly_level_baselines_still_differ_under_the_legacy_rule():
    """use_per_trial_baseline must keep working; the rule only changes the spread."""
    env = _trace(10.0, 40.0, jitter=0.3)
    per_trial = ec._compute_trial_metrics(
        env, FPS, fallback_fps=FPS, default_fps=FPS, fly_before_median=25.0,
        use_per_trial_baseline=True, odor_on_frame=BEFORE_N,
        odor_off_frame=BEFORE_N + DURING_N, threshold_rule=ThresholdRule(2.0),
    )
    fly_level = ec._compute_trial_metrics(
        env, FPS, fallback_fps=FPS, default_fps=FPS, fly_before_median=25.0,
        use_per_trial_baseline=False, odor_on_frame=BEFORE_N,
        odor_off_frame=BEFORE_N + DURING_N, threshold_rule=ThresholdRule(2.0),
    )
    assert per_trial["AUC-During"] > fly_level["AUC-During"]


def test_anchor_ignores_the_fly_level_baseline():
    """Anchoring is per-trial by construction, so the fly median must not leak in."""
    env = _trace(10.0, 40.0, jitter=0.3)
    rule = ThresholdRule(2.0, 5.0, 5.0)
    a = ec._compute_trial_metrics(
        env, FPS, fallback_fps=FPS, default_fps=FPS, fly_before_median=25.0,
        use_per_trial_baseline=False, odor_on_frame=BEFORE_N,
        odor_off_frame=BEFORE_N + DURING_N, threshold_rule=rule,
    )
    b = ec._compute_trial_metrics(
        env, FPS, fallback_fps=FPS, default_fps=FPS, fly_before_median=float("nan"),
        use_per_trial_baseline=True, odor_on_frame=BEFORE_N,
        odor_off_frame=BEFORE_N + DURING_N, threshold_rule=rule,
    )
    assert a["AUC-During"] == pytest.approx(b["AUC-During"])
