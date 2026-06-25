"""Tests for the one-sided (upper) MAD response threshold in envelope_visuals.

The per-trial "red line" threshold must be raised only by *upward* variability in
the pre-odor baseline. Downward dips below the resting median (the opposite of a
proboscis extension) must not push the threshold up.
"""

import sys
from pathlib import Path

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
