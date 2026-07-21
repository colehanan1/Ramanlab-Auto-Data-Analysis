"""Invalidation pass for cached rig_3 angle columns.

_process_fly_angles and _ensure_angle_percentages both short-circuit when the
cached columns are already present, so they must be dropped before a recompute.
"""
from __future__ import annotations

import pandas as pd

from scripts.pipeline.recompute_rig3_angles import (
    ANGLE_COLUMNS,
    find_rig3_tables,
    invalidate_angle_columns,
)


def test_angle_columns_cover_both_producers():
    assert set(ANGLE_COLUMNS) == {
        "angle_ARB_deg",
        "angle_centered_deg",
        "angle_centered_pct",
        "angle_multiplier",
    }


def test_invalidate_drops_only_angle_columns():
    df = pd.DataFrame(
        {
            "frame": [0, 1],
            "x_class0": [1.0, 2.0],
            "angle_ARB_deg": [10.0, 20.0],
            "angle_centered_deg": [1.0, 2.0],
            "angle_centered_pct": [5.0, 6.0],
            "angle_multiplier": [1.1, 1.2],
        }
    )
    out = invalidate_angle_columns(df)
    assert list(out.columns) == ["frame", "x_class0"]
    # input must not be mutated
    assert "angle_ARB_deg" in df.columns


def test_invalidate_is_safe_when_columns_absent():
    df = pd.DataFrame({"frame": [0], "x_class0": [1.0]})
    assert list(invalidate_angle_columns(df).columns) == ["frame", "x_class0"]


def test_find_rig3_tables_selects_only_rig3(tmp_path):
    r3 = tmp_path / "july_17_batch_2_rig_3" / "trial_1"
    r2 = tmp_path / "july_17_batch_2_rig_2" / "trial_1"
    r3.mkdir(parents=True)
    r2.mkdir(parents=True)
    (r3 / "updated_t_fly1_distances.parquet").write_bytes(b"")
    (r2 / "updated_t_fly1_distances.parquet").write_bytes(b"")

    found = find_rig3_tables([tmp_path])
    assert len(found) == 1
    assert "rig_3" in str(found[0])
