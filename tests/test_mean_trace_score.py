"""Tests for scoring a cohort-mean trace with the ordinal PER model.

The CLI is stubbed here — the real model is exercised by
``test_mean_trace_score_matches_pipeline`` below, which is skipped when the
model file or the CLI is unavailable.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import pytest

from scripts.analysis.mean_trace_score import (
    build_mean_row,
    score_group_means,
    score_rows,
)

MODEL_PATH = Path(
    "/home/ramanlab/Documents/cole/VSCode/FlyBehaviorScoring/outputs/"
    "ordinal_scorer/model_ordinal_xgb.json"
)
WIDE_PARQUET = Path(
    "/home/ramanlab/Documents/cole/Data/CSVs-New-Opto-Flys/"
    "all_envelope_rows_wide_combined_base.parquet"
)
PREDICTIONS_CSV = Path(
    "/home/ramanlab/Documents/cole/Data/CSVs-New-Opto-Flys/model_predictions.csv"
)


def _rows(n: int = 3, n_frames: int = 8) -> pd.DataFrame:
    rows = []
    for i in range(n):
        row = {
            "dataset": "EB-Training-24-1",
            "fly": f"fly_{i}",
            "fly_number": 1,
            "trial_type": "testing",
            "trial_label": "testing_1_ethylbutyrate",
            "global_max": float(i),
            "trimmed_global_min": float(2 * i),
        }
        row.update({f"dir_val_{j}": float(i + j) for j in range(n_frames)})
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Mean row construction
# ---------------------------------------------------------------------------


def test_build_mean_row_averages_the_trace() -> None:
    row = build_mean_row(_rows())
    # dir_val_j = i + j over i in {0,1,2} -> mean = 1 + j
    assert row["dir_val_0"] == pytest.approx(1.0)
    assert row["dir_val_7"] == pytest.approx(8.0)


def test_build_mean_row_averages_the_engineered_features() -> None:
    """The engineered block is averaged, not recomputed — pinning that keeps
    the documented method from drifting into a silent reimplementation."""
    row = build_mean_row(_rows())
    assert row["global_max"] == pytest.approx(1.0)
    assert row["trimmed_global_min"] == pytest.approx(2.0)


def test_build_mean_row_keeps_non_numeric_metadata() -> None:
    row = build_mean_row(_rows())
    assert row["dataset"] == "EB-Training-24-1"
    assert row["trial_type"] == "testing"


def test_build_mean_row_skips_nans() -> None:
    """One fly missing a frame must not blank that frame of the mean."""
    rows = _rows()
    rows.loc[0, "dir_val_3"] = np.nan
    row = build_mean_row(rows)
    assert row["dir_val_3"] == pytest.approx(np.mean([1 + 3, 2 + 3]))


def test_build_mean_row_rejects_an_empty_frame() -> None:
    with pytest.raises(ValueError):
        build_mean_row(_rows().iloc[0:0])


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------


class _FakeCli:
    """Stand-in for flybehavior-response that echoes fixed scores."""

    def __init__(self, scores: Sequence[int]) -> None:
        self.scores = list(scores)
        self.calls: list[list[str]] = []
        self.seen: list[pd.DataFrame] = []

    def __call__(self, cmd: Sequence[str]) -> None:
        self.calls.append(list(cmd))
        args = dict(zip(cmd[2::2], cmd[3::2]))
        data = pd.read_csv(args["--data-csv"])
        self.seen.append(data.copy())
        # Emit one row per available score, so a short score list models a CLI
        # that dropped rows.
        n = min(len(data), len(self.scores))
        out = data.iloc[:n][
            [c for c in ("dataset", "fly", "fly_number", "trial_label")
             if c in data.columns]
        ].copy()
        out["prediction"] = [int(s >= 2) for s in self.scores[:n]]
        out["score"] = self.scores[:n]
        out.to_csv(args["--output-csv"], index=False)


def test_score_rows_invokes_predict_ordinal_with_the_model_and_threshold() -> None:
    fake = _FakeCli([3])
    score_rows(_rows(1), model_path=Path("/models/m.json"), binary_threshold=2,
               runner=fake)
    cmd = fake.calls[0]
    assert cmd[1] == "predict-ordinal"
    assert "--model-path" in cmd and "/models/m.json" in cmd
    assert cmd[cmd.index("--binary-threshold") + 1] == "2"


def test_score_rows_returns_empty_without_rows() -> None:
    fake = _FakeCli([])
    out = score_rows(_rows(1).iloc[0:0], model_path=Path("/m.json"), runner=fake)
    assert out.empty
    assert not fake.calls


def test_score_group_means_scores_one_row_per_group() -> None:
    fake = _FakeCli([4, 0])
    result = score_group_means(
        {"Trained|Hexanol": _rows(3), "Control|Hexanol": _rows(2)},
        model_path=Path("/m.json"), runner=fake,
    )
    assert result["Trained|Hexanol"] == {"score": 4, "prediction": 1, "n_flies": 3}
    assert result["Control|Hexanol"] == {"score": 0, "prediction": 0, "n_flies": 2}
    # one CLI call for all groups, not one per group
    assert len(fake.calls) == 1


def test_score_group_means_skips_empty_groups() -> None:
    fake = _FakeCli([2])
    result = score_group_means(
        {"Trained|Hexanol": _rows(2), "Control|Hexanol": _rows(0)},
        model_path=Path("/m.json"), runner=fake,
    )
    assert set(result) == {"Trained|Hexanol"}


def test_score_group_means_labels_synthetic_rows_by_group() -> None:
    """Each synthetic row carries its group key in ``fly`` so a misaligned CLI
    output is detectable rather than silently mapped to the wrong cohort."""
    fake = _FakeCli([1, 1])
    score_group_means(
        {"A": _rows(2), "B": _rows(2)}, model_path=Path("/m.json"), runner=fake
    )
    sent = fake.seen[0]
    assert list(sent["fly"]) == ["A", "B"]
    assert set(sent["fly_number"]) == {0}


def test_score_group_means_raises_when_the_cli_drops_rows() -> None:
    fake = _FakeCli([1])          # only one score for two groups
    with pytest.raises(RuntimeError, match="misaligned"):
        score_group_means(
            {"A": _rows(2), "B": _rows(2)}, model_path=Path("/m.json"), runner=fake
        )


# ---------------------------------------------------------------------------
# Real-model round trip
# ---------------------------------------------------------------------------


requires_model = pytest.mark.skipif(
    not MODEL_PATH.exists()
    or not WIDE_PARQUET.exists()
    or not PREDICTIONS_CSV.exists()
    or shutil.which("flybehavior-response") is None,
    reason="ordinal model / wide table / CLI not available",
)


@requires_model
def test_scoring_real_rows_reproduces_the_pipeline_scores() -> None:
    """Scoring untouched wide rows through this module must return exactly what
    the pipeline wrote to model_predictions.csv — otherwise the synthetic-row
    scores are not on the same scale as the per-fly ones."""
    wide = pd.read_parquet(WIDE_PARQUET)
    sample = wide[
        (wide["dataset"] == "3Oct-Training-24-0.1")
        & (wide["trial_type"] == "testing")
    ].head(8)
    scored = score_rows(sample, model_path=MODEL_PATH, binary_threshold=2)

    published = pd.read_csv(PREDICTIONS_CSV)
    merged = scored.merge(
        published, on=["dataset", "fly", "fly_number", "trial_label"],
        suffixes=("_ours", "_pipeline"),
    )
    assert len(merged) > 0
    assert (merged["score_ours"] == merged["score_pipeline"]).all()
