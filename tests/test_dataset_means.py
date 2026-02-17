"""Tests for the dataset_means analysis utility."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)

src_str = str(ROOT / "src")
if src_str not in sys.path:
    sys.path.insert(0, src_str)

from fbpipe.utils.nanstats import (  # noqa: E402
    count_finite_contributors,
    nan_pad_stack,
    nanmean_sem,
)
from scripts.analysis.dataset_means import (  # noqa: E402
    _read_distance_pct,
    compute_dataset_means,
    plot_dataset_means,
    write_sidecar,
)


# ---------------------------------------------------------------------------
# nanstats helpers
# ---------------------------------------------------------------------------


class TestNanPadStack:
    def test_equal_length(self):
        a = np.array([1.0, 2.0])
        b = np.array([3.0, 4.0])
        result = nan_pad_stack([a, b])
        assert result.shape == (2, 2)
        np.testing.assert_array_equal(result[0], [1.0, 2.0])
        np.testing.assert_array_equal(result[1], [3.0, 4.0])

    def test_unequal_length_pads_with_nan(self):
        short = np.array([1.0])
        long = np.array([2.0, 3.0, 4.0])
        result = nan_pad_stack([short, long])
        assert result.shape == (2, 3)
        assert result[0, 0] == 1.0
        assert np.isnan(result[0, 1])
        assert np.isnan(result[0, 2])

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            nan_pad_stack([])


class TestNanmeanSem:
    def test_shapes_match(self):
        stacked = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        mean, sem = nanmean_sem(stacked)
        assert mean.shape == (3,)
        assert sem.shape == (3,)

    def test_mean_correct(self):
        stacked = np.array([[10.0, 20.0], [30.0, 40.0]])
        mean, sem = nanmean_sem(stacked)
        np.testing.assert_allclose(mean, [20.0, 30.0])

    def test_single_row_sem_zero(self):
        stacked = np.array([[5.0, 10.0, 15.0]])
        mean, sem = nanmean_sem(stacked)
        np.testing.assert_array_equal(sem, [0.0, 0.0, 0.0])

    def test_nan_aware(self):
        stacked = np.array([[1.0, np.nan], [3.0, 4.0]])
        mean, sem = nanmean_sem(stacked)
        assert mean[0] == 2.0
        assert mean[1] == 4.0
        # Only one contributor at col 1 so SEM should be 0
        assert sem[1] == 0.0


class TestCountFiniteContributors:
    def test_all_finite(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        assert count_finite_contributors(arr) == 3

    def test_one_all_nan(self):
        arr = np.array([[np.nan, np.nan], [1.0, 2.0]])
        assert count_finite_contributors(arr) == 1


# ---------------------------------------------------------------------------
# _read_distance_pct
# ---------------------------------------------------------------------------


def test_read_distance_pct_uses_existing_column(tmp_path):
    csv = tmp_path / "fly1_testing_1_distances.csv"
    df = pd.DataFrame({"distance_percentage": [10.0, 50.0, 90.0]})
    df.to_csv(csv, index=False)

    result = _read_distance_pct(csv, min_floor_px=9.5)
    assert result is not None
    np.testing.assert_allclose(result, [10.0, 50.0, 90.0])


def test_read_distance_pct_normalises_fallback(tmp_path):
    csv = tmp_path / "fly1_testing_1_distances.csv"
    df = pd.DataFrame({
        "distance_0_1": [10.0, 55.0, 100.0],
        "min_distance_0_1": [10.0, 10.0, 10.0],
        "max_distance_0_1": [100.0, 100.0, 100.0],
    })
    df.to_csv(csv, index=False)

    result = _read_distance_pct(csv, min_floor_px=9.5)
    assert result is not None
    # eff_min = max(10.0, 9.5) = 10.0; span = 90
    # pct = 100 * (raw - 10) / 90 → [0.0, 50.0, 100.0]
    np.testing.assert_allclose(result, [0.0, 50.0, 100.0])


def test_read_distance_pct_returns_none_for_missing_columns(tmp_path):
    csv = tmp_path / "bad.csv"
    pd.DataFrame({"unrelated_col": [1, 2, 3]}).to_csv(csv, index=False)
    assert _read_distance_pct(csv, min_floor_px=9.5) is None


# ---------------------------------------------------------------------------
# compute_dataset_means – integration test with synthetic data
# ---------------------------------------------------------------------------


def _build_fly_tree(dataset_root: Path, fly_name: str, trials: dict[str, np.ndarray]) -> None:
    """Create a fly directory with trial CSVs containing distance_percentage."""
    fly_dir = dataset_root / fly_name
    fly_dir.mkdir(parents=True, exist_ok=True)
    for trial_label, values in trials.items():
        csv_name = f"{fly_name}_{trial_label}_fly1_distances.csv"
        df = pd.DataFrame({"distance_percentage": values})
        df.to_csv(fly_dir / csv_name, index=False)


def test_compute_dataset_means_basic(tmp_path):
    dataset = tmp_path / "hex_control"
    rng = np.random.default_rng(42)

    for fly_id in range(1, 4):
        fly_name = f"october_0{fly_id}_fly_{fly_id}"
        _build_fly_tree(dataset, fly_name, {
            "testing_6": rng.uniform(20, 80, size=100),
            "testing_7": rng.uniform(20, 80, size=100),
        })

    results = compute_dataset_means(dataset, trial_type="testing", fps=40.0)
    assert len(results) > 0

    for odor, data in results.items():
        assert data["mean"].shape == data["sem"].shape
        assert data["n_flies"] == 3
        assert len(data["mean"]) == 100


def test_compute_dataset_means_handles_missing_trials(tmp_path):
    dataset = tmp_path / "EB_control"

    # Fly 1 has testing_6, fly 2 has nothing useful
    _build_fly_tree(dataset, "fly_1", {
        "testing_6": np.array([10.0, 20.0, 30.0]),
    })
    fly2 = dataset / "fly_2"
    fly2.mkdir()
    pd.DataFrame({"unrelated": [1, 2]}).to_csv(fly2 / "fly_2_testing_6_fly1_distances.csv", index=False)

    results = compute_dataset_means(dataset, trial_type="testing", fps=40.0)
    # Should still produce results from fly_1
    assert len(results) > 0
    for data in results.values():
        assert data["n_flies"] >= 1


# ---------------------------------------------------------------------------
# Plotting smoke test
# ---------------------------------------------------------------------------


def test_plot_dataset_means_returns_figure():
    results = {
        "Benzaldehyde": {
            "mean": np.linspace(0, 100, 200),
            "sem": np.full(200, 5.0),
            "n_flies": 4,
            "fly_names": ["fly_1", "fly_2", "fly_3", "fly_4"],
            "source_csvs": [],
        },
        "Hexanol": {
            "mean": np.linspace(10, 80, 200),
            "sem": np.full(200, 3.0),
            "n_flies": 3,
            "fly_names": ["fly_1", "fly_2", "fly_3"],
            "source_csvs": [],
        },
    }
    fig = plot_dataset_means(
        results,
        dataset_name="test_dataset",
        fps=40.0,
        odor_on_s=30.0,
        odor_off_s=60.0,
    )
    assert fig is not None
    axes = fig.get_axes()
    # Single plot with all odors overlaid
    assert len(axes) == 1
    # 2 odor traces + 2 vlines = at least 4 lines
    assert len(axes[0].lines) >= 4
    # Legend should list both odors
    legend_texts = [t.get_text() for t in axes[0].get_legend().get_texts()]
    assert len(legend_texts) == 2
    plt.close(fig)


def test_plot_returns_none_for_empty():
    fig = plot_dataset_means(
        {},
        dataset_name="empty",
        fps=40.0,
        odor_on_s=30.0,
        odor_off_s=60.0,
    )
    assert fig is None


# ---------------------------------------------------------------------------
# JSON sidecar
# ---------------------------------------------------------------------------


def test_write_sidecar_creates_valid_json(tmp_path):
    results = {
        "Hexanol": {
            "mean": np.ones(50),
            "sem": np.zeros(50),
            "n_flies": 2,
            "fly_names": ["fly_a", "fly_b"],
            "source_csvs": ["/a/b.csv", "/c/d.csv"],
        },
    }
    path = tmp_path / "sidecar.json"
    write_sidecar(
        path,
        dataset_name="hex_control",
        fps=40.0,
        odor_on_s=30.0,
        odor_off_s=60.0,
        min_floor_px=9.5,
        trial_type="testing",
        results=results,
    )
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["dataset"] == "hex_control"
    assert "Hexanol" in data["odors"]
    assert data["odors"]["Hexanol"]["n_flies"] == 2
    assert data["odors"]["Hexanol"]["n_timepoints"] == 50


# ---------------------------------------------------------------------------
# matplotlib import guard
# ---------------------------------------------------------------------------

import matplotlib.pyplot as plt  # noqa: E402, F811
