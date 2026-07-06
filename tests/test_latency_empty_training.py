"""Regression tests for latency analysis on testing-only datasets.

RandomPanel datasets carry a ``trial_type_override: testing`` — every trial is a
testing trial even though the folders are named ``training_N``. Filtering such a
CSV to ``trial_type == "training"`` yields an empty frame.

Previously, when a fly-state CSV was also supplied, the empty-row fly-state mask
became an empty Python list and ``df[[]]`` was interpreted by pandas as *column*
selection, dropping every column and surfacing a cryptic
``KeyError: 'trial_label'``. The intended behaviour is a clear "no training rows"
error (which the pipeline then turns into a graceful skip).
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "analysis"))

from envelope_training import _latency_records_from_csv  # noqa: E402

FLY_STATE_COL = "FLY-State(1, 0, -1)"


def _write_testing_only_csv(tmp_path: Path) -> Path:
    """A CSV whose every row is testing (trial_type) despite training_N labels."""
    df = pd.DataFrame(
        {
            "dataset": ["RandomPanel-Training-24-10"] * 3,
            "fly": ["fly1", "fly1", "fly2"],
            "trial_label": ["training_1_acv", "training_2_hexanol", "training_4_citral"],
            "trial_type": ["testing", "testing", "testing"],
            "fps": [40.0, 40.0, 40.0],
        }
    )
    csv = tmp_path / "all_envelope_rows_wide_training.csv"
    df.to_csv(csv, index=False)
    return csv


def _write_fly_state_csv(tmp_path: Path) -> Path:
    truth = pd.DataFrame(
        {
            "dataset": ["RandomPanel-Training-24-10"],
            "fly": ["fly2"],
            FLY_STATE_COL: [0],
        }
    )
    csv = tmp_path / "flagged-flys-truth.csv"
    truth.to_csv(csv, index=False)
    return csv


def _call(csv_path: Path, fly_state_csv: Path | None):
    return _latency_records_from_csv(
        csv_path,
        before_sec=30.0,
        during_sec=35.0,
        threshold_mult=2.0,
        latency_ceiling=10.0,
        trials_of_interest=(4, 6),
        fps_default=40.0,
        odor_on_s=30.0,
        odor_off_s=60.0,
        odor_latency_s=2.15,
        fly_state_csv=fly_state_csv,
        fly_state_column=FLY_STATE_COL,
    )


def test_no_training_rows_raises_clear_error_with_fly_state(tmp_path):
    """With a fly-state CSV present, must NOT raise KeyError on dropped columns."""
    csv_path = _write_testing_only_csv(tmp_path)
    fly_state = _write_fly_state_csv(tmp_path)
    with pytest.raises(RuntimeError, match="[Nn]o training rows"):
        _call(csv_path, fly_state)


def test_no_training_rows_raises_clear_error_without_fly_state(tmp_path):
    csv_path = _write_testing_only_csv(tmp_path)
    with pytest.raises(RuntimeError, match="[Nn]o training rows"):
        _call(csv_path, None)
