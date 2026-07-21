"""compose_videos_rms must honour the same per-rig anchor as envelope_combined."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fbpipe.steps.compose_videos_rms import compute_angle_deg_at_point2
from fbpipe.utils.rig_anchor import DEFAULT_ANCHOR, MIRRORED_ANCHOR


def _frame():
    return pd.DataFrame(
        {
            "x_class0": [500.0],
            "y_class0": [540.0],
            "x_class1": [550.0],
            "y_class1": [540.0],
        }
    )


def _non_collinear_frame():
    """Eye and proboscis NOT on the same horizontal line as each other.

    ``_frame()`` puts the eye, the proboscis, AND both anchors all on
    y=540, so every cross product in ``compute_angle_deg_at_point2`` is
    exactly zero there -- a genuine geometric divergence between this
    module's angle math and ``envelope_combined``'s duplicate
    implementation (e.g. a sign or scale error on the ``vy`` component)
    would go completely undetected by a test built on that fixture. Only
    ``test_agrees_with_envelope_combined_implementation`` should use this
    fixture; ``test_mirrored_anchor_is_supplement_of_default`` genuinely
    requires p2y == anchor_y for the ``180 - angle`` supplement identity to
    hold and must keep using the collinear ``_frame()``.
    """
    return pd.DataFrame(
        {
            "x_class0": [500.0],
            "y_class0": [540.0],
            "x_class1": [550.0],
            "y_class1": [500.0],
        }
    )


def test_default_argument_preserves_current_values():
    df = _frame()
    assert np.allclose(
        compute_angle_deg_at_point2(df).to_numpy(dtype=float),
        compute_angle_deg_at_point2(df, DEFAULT_ANCHOR).to_numpy(dtype=float),
        equal_nan=True,
    )


def test_mirrored_anchor_is_supplement_of_default():
    df = _frame()
    right = compute_angle_deg_at_point2(df, DEFAULT_ANCHOR).to_numpy(dtype=float)
    left = compute_angle_deg_at_point2(df, MIRRORED_ANCHOR).to_numpy(dtype=float)
    assert np.allclose(left, 180.0 - right, atol=1e-9)


def test_agrees_with_envelope_combined_implementation():
    """The two duplicate implementations must not drift apart.

    Uses ``_non_collinear_frame()``, not ``_frame()``: on the collinear
    fixture every cross product is zero for both implementations
    regardless of how ``vy``/``vx`` are computed, so a real divergence
    (e.g. scaling ``vy`` in one implementation but not the other) would
    silently pass. See ``_non_collinear_frame``'s docstring.
    """
    from scripts.analysis.envelope_combined import _compute_angle_deg

    df = _non_collinear_frame()
    for anchor in (DEFAULT_ANCHOR, MIRRORED_ANCHOR):
        a = compute_angle_deg_at_point2(df, anchor).to_numpy(dtype=float)
        b = _compute_angle_deg(df, anchor).to_numpy(dtype=float)
        assert np.allclose(a, b, atol=1e-9, equal_nan=True)


def _fly_dir_with_trial(base: Path, dir_name: str) -> Path:
    """Build a minimal fly_dir/RMS_calculations/<trial>.parquet fixture."""
    fly_dir = base / dir_name
    rms_dir = fly_dir / "RMS_calculations"
    rms_dir.mkdir(parents=True)
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
    df.to_parquet(rms_dir / "training_1_distances.parquet")
    return fly_dir


def test_reference_and_measurement_share_one_anchor(monkeypatch, tmp_path):
    """The baseline and the measurement must be computed with the same anchor.

    Mirrors envelope_combined's ``test_reference_and_measurement_share_one_anchor``
    (tests/test_rig3_anchor_angles.py) for this module's duplicate
    implementation. A mismatch is silently wrong rather than an error, so pin
    it with a spy on every call to ``compute_angle_deg_at_point2`` made while
    processing a rig_3 fly (once inside ``find_fly_reference_angle``, once per
    file in the main loop).
    """
    import fbpipe.steps.compose_videos_rms as module

    seen: list[tuple[float, float] | None] = []
    real = module.compute_angle_deg_at_point2

    def spy(df, anchor=None):
        seen.append(anchor)
        return real(df, anchor)

    monkeypatch.setattr(module, "compute_angle_deg_at_point2", spy)

    fly_dir = _fly_dir_with_trial(tmp_path, "july_17_batch_2_rig_3")
    module._process_fly_angles(fly_dir)

    assert seen, "compute_angle_deg_at_point2 was never called"
    assert set(seen) == {MIRRORED_ANCHOR}, f"anchors disagreed: {set(seen)}"


@pytest.mark.parametrize("dir_name", ["july_17_batch_2_rig_2", "july_18_batch_1"])
def test_non_rig3_fly_resolves_to_default_anchor(monkeypatch, dir_name):
    """A rig_2 fly (and a fly with no rig token at all) must use DEFAULT_ANCHOR.

    Task 2 (commit 7385012) found that a reviewer mutation hardcoding
    ``anchor = MIRRORED_ANCHOR`` for every fly in the analogous
    ``_ensure_angle_percentages`` passed all pre-existing tests, because the
    only prior coverage checked the call sites agree with *each other*, not
    that the resolved anchor is correct for the rig. The same shape of
    mutation in this module's ``_process_fly_angles`` (hardcoding
    ``anchor = MIRRORED_ANCHOR`` instead of ``resolve_anchor(fly_dir)``) was
    verified during development to pass every test in this file and in
    tests/test_compose_videos_rms.py + tests/test_compose_videos_rms_parquet.py
    before this test was added. Pin the resolved anchor explicitly so that
    regression can't sneak back in here too.

    Uses its own ``tempfile.TemporaryDirectory`` rather than the ``tmp_path``
    fixture: pytest derives ``tmp_path``'s directory name from this test's
    node id, which (via "non_rig3_fly") itself contains the substring
    "rig3" -- ``rig_token`` scans every ancestor path component, so for the
    no-rig-token case that accidental substring in an ancestor directory
    would get misread as a rig_3 trial. A dedicated temp dir keeps the
    fixture's own naming out of the path under test.
    """
    import fbpipe.steps.compose_videos_rms as module

    seen: list[tuple[float, float] | None] = []
    real = module.compute_angle_deg_at_point2

    def spy(df, anchor=None):
        seen.append(anchor)
        return real(df, anchor)

    monkeypatch.setattr(module, "compute_angle_deg_at_point2", spy)

    with tempfile.TemporaryDirectory(prefix="anchor_test_") as base:
        fly_dir = _fly_dir_with_trial(Path(base), dir_name)
        module._process_fly_angles(fly_dir)

    assert seen, "compute_angle_deg_at_point2 was never called"
    assert set(seen) == {DEFAULT_ANCHOR}, f"anchors disagreed: {set(seen)}"
