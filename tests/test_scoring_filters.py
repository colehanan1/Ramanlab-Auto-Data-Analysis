"""Tests for pure scoring-filter logic used by the config_new scoring GUI."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.label import scoring_filters as sf  # noqa: E402


# --- compute_theta (one-sided upper MAD) ----------------------------------

def test_compute_theta_ignores_downward_dips():
    fps = 1.0
    base = np.array(
        [30, 30, 30, 31, 30, 30, 32, 30, 31, 30, 30, 33, 30, 30, 31], dtype=float
    )
    dipped = base.copy()
    dipped[[2, 7, 13]] = np.array([3, 1, 5], dtype=float)  # deep downward dips
    theta_base = sf.compute_theta(base, fps, odor_on_s=len(base), std_mult=3.0)
    theta_dipped = sf.compute_theta(dipped, fps, odor_on_s=len(dipped), std_mult=3.0)
    assert np.isclose(theta_base, theta_dipped)  # dips do not raise the line
    assert theta_base > 30.0  # upward jitter still lifts it above resting


def test_compute_theta_flat_returns_baseline():
    flat = np.full(40, 30.0)
    assert np.isclose(sf.compute_theta(flat, 1.0, odor_on_s=40, std_mult=3.0), 30.0)


def test_compute_theta_too_few_baseline_samples_is_nan():
    assert np.isnan(sf.compute_theta(np.array([30.0, 31.0]), 1.0, odor_on_s=2))


# --- trial filtering -------------------------------------------------------

def _matrix():
    return pd.DataFrame(
        {
            "dataset": [
                "RandomPanel-Training-24-0.01",
                "RandomPanel-Training-24-0.01-Gr5aOld",
                "Hex-Training-24-0.01",
                "LightSweep-Control-24-0.01",
                "RandomPanel-Training-24-0.01",
            ],
            "fly": ["f1", "f2", "f3", "f4", "f5"],
            "fly_number": [1, 1, 1, 1, 2],
            "trial_type": ["testing"] * 5,
            "trial_label": [
                "testing_3_hexanol",
                "testing_3_hexanol",
                "testing_11_citral",   # must be KEPT
                "testing_2_light",
                "testing_4_acv",
            ],
        }
    )


def test_filter_exact_dataset_excludes_gr5aold():
    out = sf.filter_trials(_matrix(), dataset="RandomPanel-Training-24-0.01")
    assert set(out["fly"]) == {"f1", "f5"}  # f2 (-Gr5aOld) excluded by exact match


def test_filter_drops_light_only_dataset_keeps_testing_11():
    out = sf.filter_trials(_matrix(), dataset=None)
    assert "LightSweep-Control-24-0.01" not in set(out["dataset"])  # light-only gone
    assert "testing_11_citral" in set(out["trial_label"])           # testing_11 kept


def test_available_datasets_sorted_unique():
    assert sf.available_datasets(_matrix()) == [
        "Hex-Training-24-0.01",
        "LightSweep-Control-24-0.01",
        "RandomPanel-Training-24-0.01",
        "RandomPanel-Training-24-0.01-Gr5aOld",
    ]


def test_apply_exclusions_removes_flagged_fly():
    out = sf.apply_exclusions(_matrix(), {("f3", 1)})
    assert "f3" not in set(out["fly"])


def test_override_dataset_with_training_labels_is_kept():
    # RandomPanel is scored as testing (trial_type=testing) but keeps training_<N>
    # labels — those rows must survive the label filter.
    df = pd.DataFrame(
        {
            "dataset": ["RandomPanel-Training-24-0.01-Gr5aOld"],
            "fly": ["fr"],
            "fly_number": [1],
            "trial_type": ["testing"],
            "trial_label": ["training_10_benzaldehyde"],
        }
    )
    out = sf.filter_trials(df, dataset="RandomPanel-Training-24-0.01-Gr5aOld")
    assert list(out["trial_label"]) == ["training_10_benzaldehyde"]


# --- --list summary --------------------------------------------------------

def test_summarize_datasets_counts():
    df = sf.filter_trials(_matrix(), dataset=None)
    scored = {("RandomPanel-Training-24-0.01", "f1", 1, "testing_3")}
    rows = {r["dataset"]: r for r in sf.summarize_datasets(df, scored)}
    rp = rows["RandomPanel-Training-24-0.01"]
    assert rp["trials"] == 2 and rp["flies"] == 2 and rp["scored"] == 1
    assert rows["Hex-Training-24-0.01"]["scored"] == 0
