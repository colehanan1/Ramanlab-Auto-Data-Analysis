"""Unit tests for angle-distance combination helpers."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)

from scripts import envelope_combined as ec  # noqa: E402


def test_angle_multiplier_never_below_unity():
    """Ensure dir_val multipliers never attenuate the distance percentage."""

    angles = np.array([-80, -30, -12, 0, 15, 35, 55, 90], dtype=float)
    multipliers = ec._angle_multiplier(angles)

    assert np.all(multipliers[:4] == 1.0)
    assert multipliers[4] == 1.25
    assert multipliers[5] == 1.50
    assert multipliers[6] == 1.75
    assert multipliers[7] == 2.00


def test_angle_multiplier_handles_scalars():
    """Scalar inputs should still respect the unity floor."""

    assert ec._angle_multiplier(np.array([-100.0])).item() == 1.0
    assert ec._angle_multiplier(np.array([5.0])).item() == 1.0
