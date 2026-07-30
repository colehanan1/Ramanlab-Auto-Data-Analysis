"""Tests for the PER gate rejection methods figure.

The gate values asserted here are the ones the thesis text cites, and they live
in ``config/config_new.yaml`` -- NOT in the repo defaults (``config.yaml`` and
``src/fbpipe/config.py`` carry 150 / 180 / 250). The figure must read them at
runtime so it can never silently disagree with the pipeline.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from scripts.analysis.per_gate_rejection_figure import (
    CONFIG_PATH,
    GateSettings,
    load_gate_settings,
)


def test_gate_settings_match_config_new() -> None:
    """The five gate values the figure reports come from config_new.yaml."""
    s = load_gate_settings()
    assert s.max_px == 160.0
    assert s.up_divisor == 4.0
    assert s.dorsal_px == 40.0
    assert s.max_jump_px == 80.0
    assert s.norm_min_px == 10.0
    assert s.norm_max_px == 160.0


def test_config_path_points_at_config_new() -> None:
    assert CONFIG_PATH.name == "config_new.yaml"
    assert CONFIG_PATH.exists()


def test_gate_settings_are_not_hardcoded(tmp_path: Path) -> None:
    """Feeding a different config must change the values -- proving they are read,
    not baked into the script."""
    alt = tmp_path / "alt.yaml"
    alt.write_text(
        "proboscis_filter:\n"
        "  max_eye_prob_distance_px: 99.0\n"
        "  up_divisor: 3.0\n"
        "  max_jump_px: 55.0\n"
        "distance_limits:\n"
        "  class2_min: 5.0\n"
        "  class2_max: 99.0\n",
        encoding="utf-8",
    )
    s = load_gate_settings(alt)
    assert s.max_px == 99.0
    assert s.up_divisor == 3.0
    assert s.dorsal_px == 33.0
    assert s.max_jump_px == 55.0
    assert s.norm_min_px == 5.0
    assert s.norm_max_px == 99.0


def test_gate_settings_rejects_missing_keys(tmp_path: Path) -> None:
    """A config without the gate blocks must fail loudly, not silently default."""
    empty = tmp_path / "empty.yaml"
    empty.write_text("{}\n", encoding="utf-8")
    with pytest.raises(KeyError):
        load_gate_settings(empty)


import numpy as np

from scripts.analysis.per_gate_rejection_figure import (
    SUBJECT,
    Offsets,
    load_subject_offsets,
    subject_parquet_path,
    subject_video_path,
)

requires_data = pytest.mark.skipif(
    not subject_parquet_path().exists(),
    reason="subject parquet not mounted on this machine",
)


def test_subject_identity() -> None:
    """The subject is pinned so the figure caption cannot drift from the data."""
    assert SUBJECT.dataset == "3Oct-Control-24-0.1"
    assert SUBJECT.trial_rel == "july_26_batch_1/july_26_batch_1_testing_1"
    assert SUBJECT.slot == "fly1"
    assert SUBJECT.odor == "3-Octonol"
    assert SUBJECT.eye_xy == (845.0, 183.0)
    assert SUBJECT.peak_frame == 1102


@requires_data
def test_subject_offsets_match_spec() -> None:
    """Golden numbers from the candidate scan. If the parquet is ever
    reprocessed, this fails loudly rather than the figure quietly changing."""
    off = load_subject_offsets()
    assert len(off.dx) == 3605, "subject was chosen for its perfect detection record"
    assert len(off.dx) == len(off.dy) == len(off.frames)

    ex, ey = off.eye_xy
    assert round(ex) == 845 and round(ey) == 183

    r = np.hypot(off.dx, off.dy)
    assert r.max() == pytest.approx(145.8, abs=0.1)

    peak_i = int(np.argmax(r))
    assert off.frames[peak_i] == SUBJECT.peak_frame
    assert off.dx[peak_i] == pytest.approx(28.9, abs=0.1)
    assert off.dy[peak_i] == pytest.approx(142.8, abs=0.1)


@requires_data
def test_subject_per_is_ventral() -> None:
    """The figure's argument: PER is a near-vertical ventral excursion, so the
    gate is generous ventrally and tight dorsally."""
    off = load_subject_offsets()
    assert off.dy.min() > 0, "this fly never goes dorsal"
    assert np.abs(off.dx).max() < 40.0
    assert off.dy.max() > 140.0


def test_subject_video_path_is_the_raw_recording() -> None:
    """The '*_distance_annotated.mp4' sibling is the pipeline's own overlay and
    must not be used -- we draw our own."""
    path = subject_video_path()
    assert path.name.startswith("output_")
    assert "distance_annotated" not in path.name


@requires_data
def test_subject_offsets_frames_match_parquet_frame_column() -> None:
    """`frames` must come from the parquet's own frame column, not row position.

    This asserts against the column itself (not against arange/contiguity), so
    it would fail if the implementation silently reverted to
    ``np.flatnonzero(ok)`` and the column ever diverged from row position --
    unlike an ``arange(3605)`` check, which both implementations satisfy today.
    """
    import pandas as pd

    from fbpipe.utils.columns import find_eye_xy_columns, find_proboscis_xy_columns
    from fbpipe.utils.tables import read_table

    df = read_table(subject_parquet_path())
    ex_col, ey_col = find_eye_xy_columns(df)
    px_col, py_col = find_proboscis_xy_columns(df)

    ex = pd.to_numeric(df[ex_col], errors="coerce").to_numpy(float)
    ey = pd.to_numeric(df[ey_col], errors="coerce").to_numpy(float)
    px = pd.to_numeric(df[px_col], errors="coerce").to_numpy(float)
    py = pd.to_numeric(df[py_col], errors="coerce").to_numpy(float)
    ok = np.isfinite(px - ex) & np.isfinite(py - ey)
    expected_frames = pd.to_numeric(df["frame"], errors="coerce").to_numpy()[ok].astype(int)

    off = load_subject_offsets()
    np.testing.assert_array_equal(off.frames, expected_frames)
