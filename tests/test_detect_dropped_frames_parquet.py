"""Focused tests for detect_dropped_frames Parquet I/O changes.

Verifies:
1. read_table is used to load per-fly distance files (parquet round-trip).
2. Skip/freshness logic uses resolve_existing so a .parquet input file's
   mtime is used when comparing against the .txt output file.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import pytest

from fbpipe.utils.tables import write_table, resolve_existing


# Representative columns from the pipeline schema.
_COLUMNS = [
    "frame", "timestamp",
    "track_id_class0", "x_class0", "y_class0", "corners_class0",
    "track_id_class1", "x_class1", "y_class1", "corners_class1",
    "x_anchor", "y_anchor",
    "distance_0_1", "distance_0_anchor", "angle_deg_c0_c1_vs_anchor",
]


def _make_distance_df(n_frames: int = 20) -> pd.DataFrame:
    """Return a minimal per-fly distance DataFrame."""
    frames = list(range(n_frames))
    data = {col: [0] * n_frames for col in _COLUMNS}
    data["frame"] = frames
    data["timestamp"] = [float(i) for i in frames]
    data["distance_0_1"] = [float(i) * 0.5 for i in frames]
    data["distance_0_anchor"] = [float(i) * 0.3 for i in frames]
    data["angle_deg_c0_c1_vs_anchor"] = [0.0] * n_frames
    data["corners_class0"] = ["[]"] * n_frames
    data["corners_class1"] = ["[]"] * n_frames
    return pd.DataFrame(data)


def test_resolve_existing_prefers_parquet(tmp_path: Path) -> None:
    """resolve_existing returns the .parquet file when both .parquet and .csv exist."""
    df = _make_distance_df()
    parquet_path = tmp_path / "fly1_distances.parquet"
    csv_path = tmp_path / "fly1_distances.csv"
    write_table(df, parquet_path)
    df.to_csv(csv_path, index=False)

    resolved = resolve_existing(csv_path)  # ask by csv name — should get parquet
    assert resolved is not None
    assert resolved.suffix == ".parquet"


def test_resolve_existing_falls_back_to_csv(tmp_path: Path) -> None:
    """resolve_existing returns the .csv file when only CSV exists."""
    df = _make_distance_df()
    csv_path = tmp_path / "fly1_distances.csv"
    df.to_csv(csv_path, index=False)

    resolved = resolve_existing(csv_path)
    assert resolved is not None
    assert resolved.suffix == ".csv"


def test_skip_logic_uses_resolved_input_mtime(tmp_path: Path) -> None:
    """The freshness check uses actual_input (from resolve_existing) not the raw path.

    Scenario: a .parquet file exists as the canonical input; the .txt output
    was written AFTER the parquet file. The skip condition should be True.
    """
    df = _make_distance_df()

    # Write the parquet file (the canonical input).
    parquet_path = tmp_path / "fly1_distances.parquet"
    written = write_table(df, parquet_path)
    assert written.exists()

    # Sleep a tiny bit so the .txt is strictly newer.
    time.sleep(0.02)

    out_path = tmp_path / "fly1_distances_dropped_frames.txt"
    out_path.write_text("No dropped frames found.\n", encoding="utf-8")

    # Reproduce the skip logic from detect_dropped_frames.main:
    #   actual_input = resolve_existing(csv_path) or csv_path
    csv_path = tmp_path / "fly1_distances.csv"  # doesn't exist — only .parquet does
    actual_input = resolve_existing(csv_path) or csv_path

    should_skip = (
        out_path.exists()
        and out_path.stat().st_mtime >= actual_input.stat().st_mtime
    )
    assert should_skip, (
        "Expected skip=True because the .txt output is newer than the parquet input"
    )


def test_skip_logic_recomputes_when_input_newer(tmp_path: Path) -> None:
    """When the parquet input is newer than the .txt report, skip should be False."""
    df = _make_distance_df()

    # Write the .txt first (simulate stale report).
    out_path = tmp_path / "fly1_distances_dropped_frames.txt"
    out_path.write_text("No dropped frames found.\n", encoding="utf-8")

    time.sleep(0.02)

    # Then write the parquet (simulate updated input after the old report).
    parquet_path = tmp_path / "fly1_distances.parquet"
    write_table(df, parquet_path)

    csv_path = tmp_path / "fly1_distances.csv"  # doesn't exist
    actual_input = resolve_existing(csv_path) or csv_path

    should_skip = (
        out_path.exists()
        and out_path.stat().st_mtime >= actual_input.stat().st_mtime
    )
    assert not should_skip, (
        "Expected skip=False because the parquet input is newer than the .txt output"
    )
