"""Phase 2 integration + regression tests for the CSV->Parquet step migration.

These exercise the wired pipeline END-TO-END (which the existing
test_multi_fly_pipeline integration tests no longer do, since they are pinned to
the old Settings(main_directory=...) API and sit in the frozen-failing baseline),
plus a focused regression test for the distance_sanity 3-fly detection that must
work across .parquet siblings (a value-changing bug found in central review).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fbpipe.config import ForceSettings, Settings
from fbpipe.steps import distance_normalize, distance_stats
from fbpipe.utils.distance_sanity import (
    csv_requires_three_fly_distance_sanitization,
    is_three_fly_trial_csv,
    sanitize_three_fly_distance_dataframe,
)
from fbpipe.utils.tables import read_table, table_path, write_table

PER_FLY_COLUMNS = [
    "frame", "timestamp", "track_id_class0", "x_class0", "y_class0",
    "corners_class0", "track_id_class1", "x_class1", "y_class1",
    "corners_class1", "x_anchor", "y_anchor", "distance_0_1",
    "distance_0_anchor", "angle_deg_c0_c1_vs_anchor",
]


def _make_df(distances: list[float]) -> pd.DataFrame:
    n = len(distances)
    return pd.DataFrame({
        "frame": np.arange(n, dtype="int64"),
        "timestamp": np.arange(n, dtype="float64") / 40.0,
        "track_id_class0": np.ones(n, dtype="int64"),
        "x_class0": np.full(n, 100.0),
        "y_class0": np.full(n, 200.0),
        "corners_class0": ["[]"] * n,
        "track_id_class1": np.ones(n, dtype="int64"),
        "x_class1": np.full(n, 110.0),
        "y_class1": np.full(n, 205.0),
        "corners_class1": ["[]"] * n,
        "x_anchor": np.full(n, 1079.0),
        "y_anchor": np.full(n, 540.0),
        "distance_0_1": np.asarray(distances, dtype="float64"),
        "distance_0_anchor": np.full(n, 500.0),
        "angle_deg_c0_c1_vs_anchor": np.full(n, 10.0),
    })[PER_FLY_COLUMNS]


def _cfg(root: Path) -> Settings:
    # force.pipeline defaults to True (always-recompute); disable it so the
    # per-step "already done" skip logic is exercised.
    return Settings(
        model_path="",
        main_directories=[str(root)],
        force=ForceSettings(pipeline=False),
    )


# --------------------------------------------------------------------------
# Regression: distance_sanity 3-fly detection must work on .parquet siblings
# --------------------------------------------------------------------------

def test_three_fly_detected_across_parquet_siblings(tmp_path: Path):
    trial = tmp_path / "flyA" / "trial1"
    trial.mkdir(parents=True)
    for slot in (1, 2, 3):
        write_table(_make_df([90.0, 120.0]), trial / f"apr13_fly{slot}_distances.parquet")
    target = trial / "apr13_fly1_distances.parquet"
    assert target.exists()
    # Before the fix this returned False (regex + glob were .csv-only).
    assert is_three_fly_trial_csv(target) is True


def test_three_fly_detected_across_mixed_formats(tmp_path: Path):
    trial = tmp_path / "flyA" / "trial1"
    trial.mkdir(parents=True)
    write_table(_make_df([90.0]), trial / "apr13_fly1_distances.parquet")
    write_table(_make_df([90.0]), trial / "apr13_fly2_distances.parquet")
    # one legacy csv sibling
    _make_df([90.0]).to_csv(trial / "apr13_fly3_distances.csv", index=False)
    assert is_three_fly_trial_csv(trial / "apr13_fly1_distances.parquet") is True


def test_sanitization_required_reads_parquet_values(tmp_path: Path):
    trial = tmp_path / "flyA" / "trial1"
    trial.mkdir(parents=True)
    # 3 slots, one row over the 180px three-fly limit
    for slot in (1, 2, 3):
        write_table(_make_df([90.0, 250.0]), trial / f"apr13_fly{slot}_distances.parquet")
    target = trial / "apr13_fly1_distances.parquet"
    # Reads distance values out of the .parquet (was pd.read_csv -> crash -> False).
    assert csv_requires_three_fly_distance_sanitization(target, 180.0) is True


def test_sanitization_blanks_over_limit_rows_on_parquet(tmp_path: Path):
    trial = tmp_path / "flyA" / "trial1"
    trial.mkdir(parents=True)
    for slot in (1, 2, 3):
        write_table(_make_df([90.0, 250.0]), trial / f"apr13_fly{slot}_distances.parquet")
    target = trial / "apr13_fly1_distances.parquet"
    df = read_table(target)
    cleaned, n = sanitize_three_fly_distance_dataframe(df, target, 180.0)
    assert n >= 1
    # The 250px row's proboscis distance must be blanked (NaN), the 90px row kept.
    assert pd.isna(cleaned.loc[1, "distance_0_1"])
    assert cleaned.loc[0, "distance_0_1"] == 90.0


# --------------------------------------------------------------------------
# End-to-end: distance_stats -> distance_normalize on .csv input -> .parquet out
# --------------------------------------------------------------------------

def test_stats_then_normalize_end_to_end_parquet(tmp_path: Path):
    trial = tmp_path / "flyA" / "trial1"
    trial.mkdir(parents=True)
    csv_in = trial / "apr13_fly1_distances.csv"
    _make_df([80.0, 100.0, 140.0, 200.0]).to_csv(csv_in, index=False)
    cfg = _cfg(tmp_path)

    distance_stats.main(cfg)
    stats_json = tmp_path / "flyA" / "fly1_global_distance_stats_class_0.json"
    assert stats_json.exists(), "distance_stats should write the per-slot stats JSON"

    distance_normalize.main(cfg)
    pq_out = table_path(csv_in)
    assert pq_out.exists(), "distance_normalize should write a .parquet output"

    out = read_table(pq_out)
    assert "distance_percentage_0_1" in out.columns
    pct = pd.to_numeric(out["distance_percentage_0_1"], errors="coerce")
    assert pct.notna().any()
    # Monotonic with raw distance (higher distance -> higher percentage).
    order = np.argsort(out["distance_0_1"].to_numpy())
    assert np.all(np.diff(pct.to_numpy()[order]) >= -1e-9)


def test_normalize_is_idempotent_on_second_run(tmp_path: Path):
    trial = tmp_path / "flyA" / "trial1"
    trial.mkdir(parents=True)
    csv_in = trial / "apr13_fly1_distances.csv"
    _make_df([80.0, 100.0, 140.0, 200.0]).to_csv(csv_in, index=False)
    cfg = _cfg(tmp_path)

    distance_stats.main(cfg)
    distance_normalize.main(cfg)
    pq_out = table_path(csv_in)
    first = read_table(pq_out)
    mtime1 = pq_out.stat().st_mtime_ns

    # Second normalize run must detect "already normalized" and not rewrite.
    distance_normalize.main(cfg)
    second = read_table(pq_out)
    pd.testing.assert_frame_equal(first, second)
    assert pq_out.stat().st_mtime_ns == mtime1, "second run should skip (no rewrite)"
