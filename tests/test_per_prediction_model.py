"""Feature construction for scripts/analysis/per_prediction_model.py."""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

SPEC = importlib.util.spec_from_file_location(
    "per_prediction_model",
    Path(__file__).resolve().parents[1] / "scripts/analysis/per_prediction_model.py")
ppm = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ppm)


def _rows():
    # one fly, 3 odors; odor C is sensitized
    return pd.DataFrame({
        "cohort": ["X"] * 3, "batch_folder": ["b1"] * 3, "fly_number": [1] * 3,
        "fly_id": ["b1_fly1"] * 3, "rig": [2] * 3, "in_window": [True] * 3,
        "odor": ["A", "B", "C"], "sensitized_odor": [False, False, True],
        "responded_pre": [True, False, False], "responded_post": [True, True, False],
        "hours_starved_at_pretest": [24.0] * 3,
    })


def test_transition_labels():
    out = ppm.add_transitions(_rows())
    assert list(out["transition"]) == ["stay yes", "gain", "stay no"]


def test_other_odor_pre_rate_leaves_this_odor_out():
    out = ppm.add_fly_features(_rows())
    # A: others B,C -> 0/2; B: A,C -> 1/2; C: A,B -> 1/2
    assert list(out["pre_rate_other_odors"]) == [0.0, 0.5, 0.5]


def test_training_merge_by_batch_and_fly_number():
    train = pd.DataFrame({"fly": ["b1"], "fly_number": [1], "mean_auc": [10.0], "per_any": [1]})
    out = ppm.merge_training(_rows(), train)
    assert out["train_mean_auc"].tolist() == [10.0] * 3
    assert out["train_per_any"].tolist() == [1] * 3


def test_rates_pool_trials_not_flies():
    # fly1 2 odors both gain, fly2 1 odor stays no -> 2/3, not mean(1, 0)=0.5
    df = pd.DataFrame({"fly_id": ["f1", "f1", "f2"], "transition": ["gain", "gain", "stay no"]})
    assert ppm.pooled_rate(df, "gain") == pytest.approx(2 / 3)
