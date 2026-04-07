"""Tests for the dataset_means analysis utility."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
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
import scripts.analysis.dataset_means as dataset_means_mod  # noqa: E402
from scripts.analysis.dataset_means import (  # noqa: E402
    _odor_colour,
    _select_flagged_csv_arg,
    _select_outdir_arg,
    _select_wide_csv_arg,
    _resolve_flagged_csv_path,
    _resolve_wide_csv_path,
    compute_dataset_means,
    load_excluded_flies,
    plot_dataset_means,
    write_sidecar,
)


def _make_wide_df(rows: list[dict]) -> pd.DataFrame:
    """Build a synthetic wide-format DataFrame with dir_val_* columns."""

    max_len = max(len(row["trace"]) for row in rows)
    records = []
    for row in rows:
        record = {
            "dataset": row["dataset"],
            "fly": row["fly"],
            "fly_number": row["fly_number"],
            "trial_label": row["trial_label"],
            "trial_type": row.get("trial_type", "testing"),
        }
        trace = row["trace"]
        for idx in range(max_len):
            record[f"dir_val_{idx}"] = float(trace[idx]) if idx < len(trace) else np.nan
        records.append(record)
    return pd.DataFrame.from_records(records)


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
        assert sem[1] == 0.0


class TestCountFiniteContributors:
    def test_all_finite(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        assert count_finite_contributors(arr) == 3

    def test_one_all_nan(self):
        arr = np.array([[np.nan, np.nan], [1.0, 2.0]])
        assert count_finite_contributors(arr) == 1


def test_compute_dataset_means_baseline_corrects_each_odor_and_skips_unmapped_testing():
    wide_df = _make_wide_df([
        {
            "dataset": "ACV",
            "fly": "fly_a",
            "fly_number": 1,
            "trial_label": "testing_6",
            "trace": [10.0, 10.0, 13.0, 15.0],
        },
        {
            "dataset": "ACV",
            "fly": "fly_a",
            "fly_number": 1,
            "trial_label": "testing_11",
            "trace": [100.0, 100.0, 100.0, 100.0],
        },
        {
            "dataset": "ACV",
            "fly": "fly_b",
            "fly_number": 2,
            "trial_label": "testing_6",
            "trace": [20.0, 20.0, 24.0, 26.0],
        },
    ])

    results = compute_dataset_means(
        wide_df,
        "ACV",
        fps=2.0,
        odor_on_s=1.0,
        subtract_baseline=True,
    )

    assert set(results) == {"3-Octonol"}
    np.testing.assert_allclose(results["3-Octonol"]["mean"], [0.0, 0.0, 3.5, 5.5])
    np.testing.assert_allclose(
        results["3-Octonol"]["sem"],
        [0.0, 0.0, 0.35355339, 0.35355339],
    )
    assert results["3-Octonol"]["n_flies"] == 2
    assert "testing_11" not in results


def test_compute_dataset_means_training_maps_odors_correctly():
    """Training trials use per-dataset odor schedules."""
    wide_df = _make_wide_df([
        {
            "dataset": "Benz-Training",
            "fly": "fly_a",
            "fly_number": 1,
            "trial_label": "training_1",
            "trial_type": "training",
            "trace": [10.0, 10.0, 15.0, 20.0],
        },
        {
            "dataset": "Benz-Training",
            "fly": "fly_a",
            "fly_number": 1,
            "trial_label": "training_5",
            "trial_type": "training",
            "trace": [10.0, 10.0, 12.0, 14.0],
        },
    ])

    results = compute_dataset_means(
        wide_df,
        "Benz-Training",
        fps=2.0,
        odor_on_s=1.0,
        subtract_baseline=True,
    )

    # training_1 → Benzaldehyde (default schedule), training_5 → Hexanol
    assert "Benzaldehyde" in results
    assert "Hexanol" in results


def test_compute_dataset_means_training_skips_unmapped_trials():
    """Training trials beyond the schedule (e.g. training_9) are skipped."""
    wide_df = _make_wide_df([
        {
            "dataset": "Benz-Training",
            "fly": "fly_a",
            "fly_number": 1,
            "trial_label": "training_1",
            "trial_type": "training",
            "trace": [10.0, 10.0, 15.0, 20.0],
        },
        {
            "dataset": "Benz-Training",
            "fly": "fly_a",
            "fly_number": 1,
            "trial_label": "training_9",
            "trial_type": "training",
            "trace": [99.0, 99.0, 99.0, 99.0],
        },
    ])

    results = compute_dataset_means(
        wide_df,
        "Benz-Training",
        fps=2.0,
        odor_on_s=1.0,
        subtract_baseline=True,
    )

    assert "Benzaldehyde" in results
    assert "training_9" not in results


def test_compute_dataset_means_24_02_variant_maps_correctly():
    """The 24-02 dataset variants resolve their odor mappings."""
    wide_df = _make_wide_df([
        {
            "dataset": "Benz-Training-24-02",
            "fly": "fly_a",
            "fly_number": 1,
            "trial_label": "training_1",
            "trial_type": "training",
            "trace": [10.0, 10.0, 15.0, 20.0],
        },
        {
            "dataset": "Benz-Control-24-02",
            "fly": "fly_b",
            "fly_number": 1,
            "trial_label": "testing_2",
            "trial_type": "testing",
            "trace": [5.0, 5.0, 8.0, 12.0],
        },
    ])

    training_results = compute_dataset_means(
        wide_df[wide_df["dataset"] == "Benz-Training-24-02"],
        "Benz-Training-24-02",
        fps=2.0,
        odor_on_s=1.0,
        subtract_baseline=True,
    )
    assert "Benzaldehyde" in training_results

    testing_results = compute_dataset_means(
        wide_df[wide_df["dataset"] == "Benz-Control-24-02"],
        "Benz-Control-24-02",
        fps=2.0,
        odor_on_s=1.0,
        subtract_baseline=True,
    )
    assert "Benzaldehyde" in testing_results


def test_plot_dataset_means_training_title():
    """Training plots use 'Training Odors Mean' in the title."""
    results = {
        "Benzaldehyde": {
            "mean": np.linspace(0, 100, 200),
            "sem": np.full(200, 5.0),
            "n_flies": 4,
            "fly_names": ["fly_1", "fly_2", "fly_3", "fly_4"],
        },
    }
    fig = plot_dataset_means(
        results,
        dataset_name="Benz-Training",
        fps=40.0,
        odor_on_s=30.0,
        odor_off_s=60.0,
        baseline_subtracted=True,
        trial_type="training",
    )
    assert fig is not None
    title = fig.get_axes()[0].get_title()
    assert "Training" in title
    assert "Testing" not in title
    plt.close(fig)


def test_compute_dataset_means_can_leave_raw_values_when_baseline_disabled():
    wide_df = _make_wide_df([
        {
            "dataset": "ACV",
            "fly": "fly_a",
            "fly_number": 1,
            "trial_label": "testing_6",
            "trace": [10.0, 10.0, 13.0, 15.0],
        },
        {
            "dataset": "ACV",
            "fly": "fly_b",
            "fly_number": 2,
            "trial_label": "testing_6",
            "trace": [20.0, 20.0, 24.0, 26.0],
        },
    ])

    results = compute_dataset_means(
        wide_df,
        "ACV",
        fps=2.0,
        odor_on_s=1.0,
        subtract_baseline=False,
    )

    np.testing.assert_allclose(results["3-Octonol"]["mean"], [15.0, 15.0, 18.5, 20.5])


def test_resolve_wide_csv_path_uses_fallback_default(monkeypatch, tmp_path):
    preferred = tmp_path / "preferred.csv"
    legacy = tmp_path / "legacy.csv"
    legacy.write_text("dataset,fly,fly_number,trial_label,trial_type,dir_val_0\n")

    monkeypatch.setattr(dataset_means_mod, "DEFAULT_WIDE_CSV", preferred)
    monkeypatch.setattr(dataset_means_mod, "LEGACY_WIDE_CSV", legacy)
    monkeypatch.setattr(dataset_means_mod, "DEFAULT_WIDE_CSV_CANDIDATES", (preferred, legacy))

    assert _resolve_wide_csv_path(preferred) == legacy.resolve()


def test_resolve_flagged_csv_path_uses_fallback_default(monkeypatch, tmp_path):
    preferred = tmp_path / "preferred_flagged.csv"
    legacy = tmp_path / "legacy_flagged.csv"
    legacy.write_text("dataset,fly,fly_number,FLY-State(1, 0, -1)\n")

    monkeypatch.setattr(dataset_means_mod, "DEFAULT_FLAGGED_CSV", preferred)
    monkeypatch.setattr(dataset_means_mod, "LEGACY_FLAGGED_CSV", legacy)
    monkeypatch.setattr(dataset_means_mod, "DEFAULT_FLAGGED_CSV_CANDIDATES", (preferred, legacy))

    assert _resolve_flagged_csv_path(preferred) == legacy.resolve()


def test_select_runtime_paths_from_config():
    cfg_data = {
        "analysis": {
            "dataset_means": {
                "wide_csv": "/tmp/dataset_means/wide.csv",
                "flagged_csv": "/tmp/dataset_means/flagged.csv",
                "out_dir": "/tmp/dataset_means/out",
            },
            "combined": {
                "combined_base": {
                    "wide": {
                        "output_csv": "/tmp/dataset_means/combined_base.csv",
                    }
                }
            },
        },
        "flagged_flies_csv": "/tmp/dataset_means/top_level_flagged.csv",
    }

    assert _select_wide_csv_arg(None, cfg_data) == Path("/tmp/dataset_means/wide.csv")
    assert _select_flagged_csv_arg(None, cfg_data) == Path("/tmp/dataset_means/flagged.csv")
    assert _select_outdir_arg(None, cfg_data) == Path("/tmp/dataset_means/out")


def test_select_runtime_paths_fallback_to_shared_config_keys():
    cfg_data = {
        "analysis": {
            "combined": {
                "combined_base": {
                    "wide": {
                        "output_csv": "/tmp/dataset_means/combined_base.csv",
                    }
                }
            }
        },
        "flagged_flies_csv": "/tmp/dataset_means/top_level_flagged.csv",
    }

    assert _select_wide_csv_arg(None, cfg_data) == Path("/tmp/dataset_means/combined_base.csv")
    assert _select_flagged_csv_arg(None, cfg_data) == Path("/tmp/dataset_means/top_level_flagged.csv")


def test_load_excluded_flies_excludes_nonpositive_states(tmp_path):
    flagged_csv = tmp_path / "flagged.csv"
    pd.DataFrame({
        "dataset": ["ACV", "ACV", "Benz"],
        "fly": ["fly_a", "fly_b", "fly_c"],
        "fly_number": [1, 2, 3],
        "FLY-State(1, 0, -1)": [1, 0, -2],
    }).to_csv(flagged_csv, index=False)

    excluded = load_excluded_flies(flagged_csv)

    assert ("ACV", "fly_b", 2) in excluded
    assert ("Benz", "fly_c", 3) in excluded
    assert ("ACV", "fly_a", 1) not in excluded


def test_plot_dataset_means_returns_baseline_labeled_figure():
    results = {
        "Benzaldehyde": {
            "mean": np.linspace(0, 100, 200),
            "sem": np.full(200, 5.0),
            "n_flies": 4,
            "fly_names": ["fly_1", "fly_2", "fly_3", "fly_4"],
        },
    }
    fig = plot_dataset_means(
        results,
        dataset_name="test_dataset",
        fps=40.0,
        odor_on_s=30.0,
        odor_off_s=60.0,
        baseline_subtracted=True,
    )
    assert fig is not None
    axes = fig.get_axes()
    assert len(axes) == 1
    assert axes[0].get_ylabel() == "Distance % (baseline-subtracted)"
    assert "Pre-odor centered" in axes[0].get_title()
    assert len(axes[0].collections) == 1
    assert len(axes[0].lines) >= 4
    plt.close(fig)


def test_plot_dataset_means_uses_stable_odor_palette():
    results = {
        "Hexanol": {
            "mean": np.linspace(0, 20, 50),
            "sem": np.zeros(50),
            "n_flies": 2,
            "fly_names": ["fly_1", "fly_2"],
        },
        "Benzaldehyde": {
            "mean": np.linspace(5, 25, 50),
            "sem": np.zeros(50),
            "n_flies": 2,
            "fly_names": ["fly_1", "fly_2"],
        },
    }
    fig = plot_dataset_means(
        results,
        dataset_name="palette_test",
        fps=40.0,
        odor_on_s=30.0,
        odor_off_s=60.0,
        baseline_subtracted=True,
    )
    assert fig is not None
    ax = fig.get_axes()[0]
    odor_lines = {line.get_label().split(" (n=")[0]: line for line in ax.lines if " (n=" in line.get_label()}
    assert odor_lines["Hexanol"].get_color() == _odor_colour("Hexanol")
    assert odor_lines["Benzaldehyde"].get_color() == _odor_colour("Benzaldehyde")
    assert _odor_colour("Hexanol") == "#2ca02c"
    assert _odor_colour("Benzaldehyde") == "#1f77b4"
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


def test_write_sidecar_creates_valid_json(tmp_path):
    results = {
        "Hexanol": {
            "mean": np.ones(50),
            "sem": np.zeros(50),
            "n_flies": 2,
            "fly_names": ["fly_a", "fly_b"],
        },
    }
    path = tmp_path / "sidecar.json"
    write_sidecar(
        path,
        dataset_name="hex_control",
        fps=40.0,
        odor_on_s=30.0,
        odor_off_s=60.0,
        trial_type="testing",
        results=results,
        baseline_subtracted=True,
    )
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["dataset"] == "hex_control"
    assert data["baseline_subtracted"] is True
    assert data["baseline_window_s"] == [0.0, 30.0]
    assert "Hexanol" in data["odors"]
    assert data["odors"]["Hexanol"]["n_flies"] == 2
    assert data["odors"]["Hexanol"]["n_timepoints"] == 50
