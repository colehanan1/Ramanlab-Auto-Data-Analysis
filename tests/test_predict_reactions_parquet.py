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


# ---------------------------------------------------------------------------
# main(): odor-pair trials reach the scorer as two rows (user rule 2026-11-13)
# ---------------------------------------------------------------------------


def test_main_scores_each_pair_odor_as_its_own_row(tmp_path, monkeypatch):
    import numpy as np
    from types import SimpleNamespace
    from fbpipe.steps import predict_reactions as pr

    n = 3250
    root = tmp_path / "IAA-PairFirst-Control-1-1"
    batch = root / "b1"
    batch.mkdir(parents=True)
    state = np.array(["off"] * n, dtype=object)
    state[1203:1603] = "OFM_A"
    state[1647:2047] = "OFM_L"
    pd.DataFrame({"FrameNumber": np.arange(1, n + 1), "ActiveOFM": state}).to_csv(
        batch / "output_b1_testing_3_IsoamylAcetate-1_then_Linalool-1_20261113_125306.csv",
        index=False)

    def row(label, fly_number):
        r = {"dataset": root.name, "fly": "b1", "fly_number": fly_number, "trial_type": "testing",
             "trial_label": label, "fps": 40.0, "trial_odor_on_s": 30.1, "trial_odor_off_s": 40.1,
             "trial_duration_s": n / 40.0, "global_min": 0.0, "global_max": 100.0,
             "trimmed_global_min": 0.0, "trimmed_global_max": 100.0}
        r.update({f"dir_val_{i}": float(i % 50) for i in range(n)})
        return r
    data = tmp_path / "wide.csv"
    pd.DataFrame([row("testing_3_isoamylacetate-1_then_linalool-1", 1),
                  row("testing_1_hexanol-1", 1)]).to_csv(data, index=False)
    model = tmp_path / "m.json"
    model.write_text("{}")
    out = tmp_path / "pred.csv"
    seen = {}

    def fake_run(cmd, check, env):
        src = pd.read_csv(cmd[cmd.index("--data-csv") + 1])
        seen["labels"] = list(src["trial_label"])
        seen["on_off"] = src[["trial_odor_on_s", "trial_odor_off_s", "trial_duration_s"]].values.tolist()
        src.assign(prediction=1)[["dataset", "fly", "fly_number", "trial_label", "prediction"]].to_csv(
            cmd[cmd.index("--output-csv") + 1], index=False)

    monkeypatch.setattr(pr.subprocess, "run", fake_run)
    monkeypatch.setattr(pr, "_pair_roots", lambda cfg: [root])
    cfg = SimpleNamespace(
        reaction_prediction=SimpleNamespace(
            data_csv=str(data), extra_data_csvs=[], model_path=str(model), output_csv=str(out),
            trial_types=["testing"], model_type="ordinal", binary_threshold=2, threshold=None),
        non_reactive_span_px=5.0, flagged_flies_csv="", include_frozen=False)
    monkeypatch.setattr("fbpipe.utils.frozen_folders.warn_if_unmarked", lambda *a, **k: None)
    pr.main(cfg)

    assert seen["labels"] == ["testing_3_isoamylacetate-1_then_linalool-1",
                              "testing_3_isoamylacetate-1_then_linalool-1__odor2",
                              "testing_1_hexanol-1"]
    (o1_on, o1_off, o1_dur), (o2_on, o2_off, o2_dur), _ = seen["on_off"]
    assert (o1_on, o1_off) == pytest.approx((30.1, 40.1)) and o1_dur == pytest.approx(42.2)
    assert (o2_on, o2_off) == pytest.approx((30.1, 40.1))
    assert o2_dur == pytest.approx((1204 + n - 1648) / 40.0)
    pred = pd.read_csv(out)
    assert sorted(pred["trial_label"]) == sorted(seen["labels"])
