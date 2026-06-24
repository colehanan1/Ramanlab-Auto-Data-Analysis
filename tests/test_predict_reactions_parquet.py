"""Focused tests for the Parquet I/O layer in predict_reactions.

Covers:
- _write_empty_predictions writes .parquet (not .csv)
- _augment_prediction_csv uses resolve_existing for the existence check so it
  finds the parquet output and reads/writes via read_table/write_table
- main reads data_csv via read_table (parquet-transparent)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fbpipe.steps.predict_reactions import (
    _augment_prediction_csv,
    _write_empty_predictions,
)
from fbpipe.utils.tables import table_path, write_table


# ---------------------------------------------------------------------------
# _write_empty_predictions -> write_table -> .parquet
# ---------------------------------------------------------------------------


def test_write_empty_predictions_creates_parquet(tmp_path):
    out = tmp_path / "predictions.csv"  # caller still supplies .csv name
    _write_empty_predictions(out, ["dataset", "fly", "prediction"])
    parquet_out = table_path(out)
    assert parquet_out.exists(), "Expected .parquet file to be written"
    df = pd.read_parquet(parquet_out)
    assert list(df.columns) == ["dataset", "fly", "prediction"]
    assert len(df) == 0


def test_write_empty_predictions_adds_prediction_column(tmp_path):
    out = tmp_path / "predictions.csv"
    _write_empty_predictions(out, ["dataset", "fly"])
    parquet_out = table_path(out)
    assert parquet_out.exists()
    df = pd.read_parquet(parquet_out)
    assert "prediction" in df.columns


# ---------------------------------------------------------------------------
# _augment_prediction_csv existence check via resolve_existing
# ---------------------------------------------------------------------------


def test_augment_prediction_csv_skips_when_no_output_exists(tmp_path, capsys):
    """If neither .parquet nor .csv exists, augment should return early."""
    out = tmp_path / "predictions.csv"  # nothing written yet
    source_df = pd.DataFrame({"dataset": ["d1"], "fly": ["f1"], "fly_number": ["1"]})
    # Should return without error or printing the annotation line
    _augment_prediction_csv(out, source_df, threshold=5.0)
    captured = capsys.readouterr()
    assert "Annotated predictions" not in captured.out


def test_augment_prediction_csv_reads_and_rewrites_parquet(tmp_path):
    """When the CLI writes a CSV, augment reads it (via read_table) and
    rewrites the result as parquet."""
    out = tmp_path / "predictions.csv"

    # Simulate the external CLI writing a CSV at output_csv
    pred_df = pd.DataFrame(
        {
            "dataset": ["d1"],
            "fly": ["f1"],
            "fly_number": ["1"],
            "trial_label": ["t1"],
            "prediction": [1],
        }
    )
    pred_df.to_csv(out, index=False)
    assert out.exists()

    source_df = pd.DataFrame(
        {
            "dataset": ["d1"],
            "fly": ["f1"],
            "fly_number": ["1"],
            "trial_label": ["t1"],
            "global_min": [10.0],
            "global_max": [60.0],
            "trial_type": ["testing"],
        }
    )

    _augment_prediction_csv(out, source_df, threshold=5.0)

    # After augmentation the result should be a parquet file
    parquet_out = table_path(out)
    assert parquet_out.exists(), "Expected parquet output after augmentation"
    result = pd.read_parquet(parquet_out)
    assert "non_reactive_flag" in result.columns
    assert "_non_reactive" in result.columns
    assert len(result) == 1


def test_augment_prediction_csv_finds_existing_parquet(tmp_path):
    """If the CLI output was already converted to parquet (e.g. second run),
    resolve_existing should still find it and augment should succeed."""
    out = tmp_path / "predictions.csv"

    pred_df = pd.DataFrame(
        {
            "dataset": ["d1"],
            "fly": ["f1"],
            "fly_number": ["1"],
            "trial_label": ["t1"],
            "prediction": [0],
        }
    )
    # Write directly as parquet (no csv on disk)
    written = write_table(pred_df, out)
    assert written == table_path(out)
    assert not out.exists(), "Sanity: CSV should not exist"

    source_df = pd.DataFrame(
        {
            "dataset": ["d1"],
            "fly": ["f1"],
            "fly_number": ["1"],
            "trial_label": ["t1"],
            "global_min": [5.0],
            "global_max": [8.0],  # span=3 < 5 -> non-reactive
            "trial_type": ["testing"],
        }
    )

    _augment_prediction_csv(out, source_df, threshold=5.0)

    parquet_out = table_path(out)
    assert parquet_out.exists()
    result = pd.read_parquet(parquet_out)
    assert "non_reactive_flag" in result.columns
    # span = 3 < 5 -> non_reactive_flag should be 1.0
    assert result["non_reactive_flag"].iloc[0] == 1.0
