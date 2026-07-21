"""The mirrored rig_3 anchor must invert the measured angle.

Flipping the anchor from the right edge to the left edge is equivalent to
``angle -> 180 - angle`` for a fly on the mid-line, which is what restores the
correct sign of the extension->angle relationship for rig_3.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.analysis.envelope_combined import _compute_angle_deg
from fbpipe.utils.rig_anchor import DEFAULT_ANCHOR, MIRRORED_ANCHOR


def _frame():
    """One fly on the mid-line (y=540) with the proboscis extended right."""
    return pd.DataFrame(
        {
            "x_class0": [500.0],
            "y_class0": [540.0],
            "x_class1": [550.0],
            "y_class1": [540.0],
        }
    )


def test_default_anchor_matches_module_constant_behaviour():
    df = _frame()
    assert np.allclose(
        _compute_angle_deg(df).to_numpy(dtype=float),
        _compute_angle_deg(df, DEFAULT_ANCHOR).to_numpy(dtype=float),
        equal_nan=True,
    )


def test_mirrored_anchor_is_supplement_of_default():
    df = _frame()
    right = _compute_angle_deg(df, DEFAULT_ANCHOR).to_numpy(dtype=float)
    left = _compute_angle_deg(df, MIRRORED_ANCHOR).to_numpy(dtype=float)
    assert np.allclose(left, 180.0 - right, atol=1e-9)


def test_extension_toward_anchor_reads_as_zero_degrees():
    """Proboscis pointing at the anchor is 0 deg; away from it is 180 deg."""
    df = _frame()
    assert _compute_angle_deg(df, DEFAULT_ANCHOR).iloc[0] == pytest.approx(0.0, abs=1e-9)
    assert _compute_angle_deg(df, MIRRORED_ANCHOR).iloc[0] == pytest.approx(180.0, abs=1e-9)


def test_reference_and_measurement_share_one_anchor(monkeypatch, tmp_path):
    """The baseline and the measurement must be computed with the same anchor.

    A mismatch is silently wrong rather than an error, so pin it with a test.
    """
    from scripts.analysis import envelope_combined as ec

    seen: list[tuple[float, float] | None] = []
    real = ec._compute_angle_deg

    def spy(df, anchor=None):
        seen.append(anchor)
        return real(df, anchor)

    monkeypatch.setattr(ec, "_compute_angle_deg", spy)

    fly_dir = tmp_path / "july_17_batch_2_rig_3"
    fly_dir.mkdir()
    df = pd.DataFrame(
        {
            "frame": [0, 1],
            "x_class0": [500.0, 500.0],
            "y_class0": [540.0, 540.0],
            "x_class1": [550.0, 560.0],
            "y_class1": [540.0, 540.0],
            "distance_percentage_0_1": [10.0, 20.0],
        }
    )
    df.to_parquet(fly_dir / "updated_t1_fly1_distances.parquet")

    monkeypatch.setattr(ec, "_trial_csv_candidates",
                        lambda d, s: [fly_dir / "updated_t1_fly1_distances.parquet"])

    ec._ensure_angle_percentages(fly_dir, "*.parquet")

    assert seen, "_compute_angle_deg was never called"
    assert set(seen) == {MIRRORED_ANCHOR}, f"anchors disagreed: {set(seen)}"


@pytest.mark.parametrize(
    "dir_name", ["july_17_batch_2_rig_2", "july_18_batch_1"]
)
def test_non_rig3_fly_resolves_to_default_anchor(monkeypatch, dir_name):
    """A rig_2 fly (and a fly with no rig token at all) must use DEFAULT_ANCHOR.

    A reviewer once hardcoded ``anchor = (0.0, 540.0)`` (the rig_3 mirrored
    anchor) for every fly in ``_ensure_angle_percentages``. That silently
    inverts angles for all non-rig_3 data, and the pre-existing agreement
    test (which only checks the three call sites agree with each other, not
    that the resolved anchor is correct for the rig) still passed. Pin the
    resolved anchor explicitly so that regression can't sneak back in.

    Uses its own ``tempfile.TemporaryDirectory`` rather than the ``tmp_path``
    fixture: pytest derives ``tmp_path``'s directory name from this test's
    node id, which (via "non_rig3_fly") itself contains the substring
    "rig3" -- ``rig_token`` scans every ancestor path component, so for the
    no-rig-token case that accidental substring in an ancestor directory
    would get misread as a rig_3 trial. A dedicated temp dir keeps the
    fixture's own naming out of the path under test.
    """
    from scripts.analysis import envelope_combined as ec

    seen: list[tuple[float, float] | None] = []
    real = ec._compute_angle_deg

    def spy(df, anchor=None):
        seen.append(anchor)
        return real(df, anchor)

    monkeypatch.setattr(ec, "_compute_angle_deg", spy)

    with tempfile.TemporaryDirectory(prefix="anchor_test_") as base:
        fly_dir = Path(base) / dir_name
        fly_dir.mkdir()
        df = pd.DataFrame(
            {
                "frame": [0, 1],
                "x_class0": [500.0, 500.0],
                "y_class0": [540.0, 540.0],
                "x_class1": [550.0, 560.0],
                "y_class1": [540.0, 540.0],
                "distance_percentage_0_1": [10.0, 20.0],
            }
        )
        df.to_parquet(fly_dir / "updated_t1_fly1_distances.parquet")

        monkeypatch.setattr(ec, "_trial_csv_candidates",
                            lambda d, s: [fly_dir / "updated_t1_fly1_distances.parquet"])

        ec._ensure_angle_percentages(fly_dir, "*.parquet")

    assert seen, "_compute_angle_deg was never called"
    assert set(seen) == {DEFAULT_ANCHOR}, f"anchors disagreed: {set(seen)}"


def test_off_axis_angle_stays_unsigned_for_both_anchors():
    """The angle must never go negative, for either anchor.

    The shared fixture (``_frame``) is collinear (eye and proboscis share
    y=540), so ``cross`` is exactly ``-0.0`` there and a sign bug in the
    ``arctan2`` call wouldn't be caught. Use a genuinely off-axis proboscis
    (y_class1 != y_class0) so a signed angle (i.e. reverting the ``np.abs``
    around ``cross``) would actually go negative and fail this test.
    """
    df = pd.DataFrame(
        {
            "x_class0": [500.0],
            "y_class0": [540.0],
            "x_class1": [550.0],
            "y_class1": [480.0],
        }
    )
    for anchor in (DEFAULT_ANCHOR, MIRRORED_ANCHOR):
        angle = _compute_angle_deg(df, anchor).iloc[0]
        assert 0.0 <= angle <= 180.0, f"angle {angle} out of [0, 180] for anchor {anchor}"
