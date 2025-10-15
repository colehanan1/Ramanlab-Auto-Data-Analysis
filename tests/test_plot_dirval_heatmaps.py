import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts import plot_dirval_heatmaps as heatmaps


def test_detect_dirval_columns_sorted():
    df = pd.DataFrame(columns=["dir_val_10", "dir_val_2", "dir_val_1", "other"])
    cols = heatmaps.detect_dirval_columns(df, "dir_val_")
    assert cols == ["dir_val_1", "dir_val_2", "dir_val_10"]


def test_prepare_heatmap_matrix_normalise_and_sort():
    df = pd.DataFrame(
        {
            "dataset": ["d", "d"],
            "fly": ["f", "f"],
            "trial_label": [2, 2],
            "fps": [20, 20],
            "dir_val_0": [1.0, 3.0],
            "dir_val_1": [4.0, 1.0],
            "dir_val_2": [0.0, 5.0],
        }
    )
    tracker = heatmaps.MissingFpsTracker()
    heatmap = heatmaps.prepare_heatmap_matrix(
        df,
        ["dir_val_0", "dir_val_1", "dir_val_2"],
        normalise="zscore",
        sort_by="peak",
        fps_tracker=tracker,
        dataset="d",
        fly="f",
        dry_trials=0,
    )
    assert heatmap is not None
    # Each row should be zero-mean after z-scoring
    assert np.allclose(np.nanmean(heatmap.matrix, axis=1), 0.0)
    # Sorting by peak places the second row (peak at index 2) after the first (peak at index 1)
    assert heatmap.trial_indices[0] == df.index[0]
    assert heatmap.trial_indices[1] == df.index[1]


def test_main_smoke(tmp_path):
    csv_path = tmp_path / "wide.csv"
    df = pd.DataFrame(
        {
            "dataset": ["d1", "d1", "d1"],
            "fly": ["A", "A", "A"],
            "trial_label": [2, 1, 4],
            "fps": [30, 30, 30],
            "dir_val_0": [0.1, 0.2, 0.3],
            "dir_val_1": [0.2, 0.3, 0.4],
            "dir_val_2": [0.3, 0.4, 0.5],
        }
    )
    df.to_csv(csv_path, index=False)

    outdir = tmp_path / "out"
    code = heatmaps.main(
        [
            "--csv",
            str(csv_path),
            "--outdir",
            str(outdir),
            "--dataset",
            "d1",
            "--dry-run",
            "0",
            "--log-level",
            "WARNING",
        ]
    )
    assert code == 0
    assert (outdir / "d1" / "A" / "odor_2_heatmap.png").exists()
    assert (outdir / "d1" / "A" / "train_combined_heatmap.png").exists()
    assert (outdir / "d1" / "combined" / "dataset_combined.png").exists()

    summary_json = outdir / "d1" / "A" / "odor_2_heatmap.json"
    data = json.loads(summary_json.read_text())
    assert data["n_trials"] >= 1
    assert "mean_length" in data
