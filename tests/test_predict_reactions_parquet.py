"""Tests for predict_reactions I/O.

model_predictions.csv is an EXTERNAL/human-facing output: it is written by the
external flybehavior-response CLI and read by literal-`.csv` consumers (the
run_workflows reaction-prediction gate + SMB sync, the reaction_matrix analysis,
and backup scripts). It therefore stays CSV. The data_csv INPUT may be Parquet
or CSV and is read via read_table.
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
from fbpipe.utils.tables import table_path


# ---------------------------------------------------------------------------
# _write_empty_predictions -> CSV (external output), never Parquet
# ---------------------------------------------------------------------------


def test_write_empty_predictions_creates_csv_not_parquet(tmp_path):
    out = tmp_path / "predictions.csv"
    _write_empty_predictions(out, ["dataset", "fly", "prediction"])
    assert out.exists(), "Expected the .csv to be written"
    assert not table_path(out).exists(), "Must NOT write a .parquet for an external output"
    df = pd.read_csv(out)
    assert list(df.columns) == ["dataset", "fly", "prediction"]
    assert len(df) == 0


def test_write_empty_predictions_adds_prediction_column(tmp_path):
    out = tmp_path / "predictions.csv"
    _write_empty_predictions(out, ["dataset", "fly"])
    assert out.exists()
    df = pd.read_csv(out)
    assert "prediction" in df.columns


# ---------------------------------------------------------------------------
# _augment_prediction_csv reads + rewrites the external CSV in place
# ---------------------------------------------------------------------------


def test_augment_skips_when_no_output_exists(tmp_path, capsys):
    out = tmp_path / "predictions.csv"  # nothing written yet
    source_df = pd.DataFrame({"dataset": ["d1"], "fly": ["f1"], "fly_number": ["1"]})
    _augment_prediction_csv(out, source_df, threshold=5.0)
    captured = capsys.readouterr()
    assert "Annotated predictions" not in captured.out


def test_augment_reads_and_rewrites_csv(tmp_path):
    out = tmp_path / "predictions.csv"
    pd.DataFrame(
        {"dataset": ["d1"], "fly": ["f1"], "fly_number": ["1"],
         "trial_label": ["t1"], "prediction": [1]}
    ).to_csv(out, index=False)

    source_df = pd.DataFrame(
        {"dataset": ["d1"], "fly": ["f1"], "fly_number": ["1"], "trial_label": ["t1"],
         "global_min": [10.0], "global_max": [60.0], "trial_type": ["testing"]}
    )
    _augment_prediction_csv(out, source_df, threshold=5.0)

    # Result stays CSV (no parquet sidecar) and carries the annotations.
    assert out.exists()
    assert not table_path(out).exists(), "augment must not produce a .parquet"
    result = pd.read_csv(out)
    assert "non_reactive_flag" in result.columns
    assert "_non_reactive" in result.columns
    assert len(result) == 1


def test_augment_marks_non_reactive_low_span(tmp_path):
    out = tmp_path / "predictions.csv"
    pd.DataFrame(
        {"dataset": ["d1"], "fly": ["f1"], "fly_number": ["1"],
         "trial_label": ["t1"], "prediction": [0]}
    ).to_csv(out, index=False)

    source_df = pd.DataFrame(
        {"dataset": ["d1"], "fly": ["f1"], "fly_number": ["1"], "trial_label": ["t1"],
         "global_min": [5.0], "global_max": [8.0],  # span 3 < 5 -> non-reactive
         "trial_type": ["testing"]}
    )
    _augment_prediction_csv(out, source_df, threshold=5.0)

    result = pd.read_csv(out)
    assert result["non_reactive_flag"].iloc[0] == 1.0
