"""Tests for the training-vigor-vs-learning driver.

Asks whether how hard a fly extended during conditioning predicts whether it
responded to the trained odor at test. Two figures per cohort role:

* ``training_vs_learning_scatter_<role>.png``    per-presentation scatter + PER split
* ``training_vs_learning_trajectory_<role>.png`` across-trial dynamics + slope

The join is the fragile part: training AUC comes from the *training* wide table
and testing scores from the predictions CSV, keyed on (fly, fly_number). A
mis-join silently pairs one fly's vigor with another's learning, which looks
like a real correlation, so the join and the drop rules are pinned below.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.analysis.training_vs_learning import (
    build_fly_summary,
    slope_per_fly,
    trained_odor_token,
    trained_presentations,
)

TRAIN = "EB-Training-24-1"
CTRL = "EB-Control-24-1"


# ---------------------------------------------------------------------------
# Trained-odor resolution
# ---------------------------------------------------------------------------


def _training_frame(flies=("july_13_batch_1", "july_14_batch_1"), n_trials=6):
    rows = []
    for fly in flies:
        for fly_number in (1, 2):
            for trial in range(1, n_trials + 1):
                rows.append({
                    "dataset": TRAIN,
                    "fly": fly,
                    "fly_number": fly_number,
                    "trial_type": "training",
                    "trial_label": f"training_{trial}_ethylbutyrate",
                    "AUC-During": 100.0 + 10.0 * trial,
                })
    return pd.DataFrame(rows)


def _testing_frame(flies=("july_13_batch_1", "july_14_batch_1")):
    """testing_1 and testing_8 are ethyl butyrate; the rest are other odors."""
    schedule = [
        (1, "ethylbutyrate"), (2, "hexanol"), (3, "citral"), (4, "acv"),
        (5, "benzaldehyde"), (6, "linalool"), (7, "3-octonol"),
        (8, "ethylbutyrate"), (9, "lightonly"),
    ]
    rows = []
    for fly in flies:
        for fly_number in (1, 2):
            for trial, odor in schedule:
                score = 4.0 if (odor == "ethylbutyrate" and fly_number == 1) else 0.0
                rows.append({
                    "dataset": TRAIN,
                    "fly": fly,
                    "fly_number": fly_number,
                    "trial_type": "testing",
                    "trial_label": f"testing_{trial}_{odor}",
                    "score": score,
                    "prediction": int(score >= 2),
                })
    return pd.DataFrame(rows)


def test_trained_odor_token_comes_from_the_training_trials() -> None:
    """The conditioning trials name the trained odor unambiguously — inferring
    it from the testing panel instead would depend on which odors repeat."""
    assert trained_odor_token(_training_frame()) == "ethylbutyrate"


def test_trained_presentations_are_the_trained_odor_in_trial_order() -> None:
    pres = trained_presentations(_testing_frame(), "ethylbutyrate")
    assert pres == [1, 8]


def test_trained_presentations_ignores_other_odors() -> None:
    pres = trained_presentations(_testing_frame(), "hexanol")
    assert pres == [2]


# ---------------------------------------------------------------------------
# Slope
# ---------------------------------------------------------------------------


def test_slope_is_positive_when_extension_rises_across_trials() -> None:
    auc = pd.Series([100.0, 110, 120, 130, 140, 150], index=[1, 2, 3, 4, 5, 6])
    assert slope_per_fly(auc) == pytest.approx(10.0)


def test_slope_is_negative_when_the_fly_habituates() -> None:
    auc = pd.Series([200.0, 180, 160, 140, 120, 100], index=[1, 2, 3, 4, 5, 6])
    assert slope_per_fly(auc) == pytest.approx(-20.0)


def test_slope_is_nan_with_a_single_trial() -> None:
    """One point has no trend. Returning 0 would put the fly on the "flat"
    pile alongside genuinely flat responders."""
    assert np.isnan(slope_per_fly(pd.Series([100.0], index=[1])))


# ---------------------------------------------------------------------------
# Join
# ---------------------------------------------------------------------------


def test_build_fly_summary_joins_on_fly_and_fly_number() -> None:
    summary = build_fly_summary(_training_frame(), _testing_frame(), TRAIN)
    assert len(summary) == 4                       # 2 folders x 2 fly numbers
    assert set(summary.columns) >= {
        "fly", "fly_number", "mean_auc", "slope", "mean_score",
        "score_1", "per_1", "score_8", "per_8",
    }
    # fly_number 1 scored 4 on both presentations, fly_number 2 scored 0
    ones = summary[summary["fly_number"] == 1]
    assert set(ones["score_1"]) == {4.0} and set(ones["score_8"]) == {4.0}
    assert set(summary[summary["fly_number"] == 2]["score_1"]) == {0.0}


def test_build_fly_summary_mean_auc_is_over_the_training_trials() -> None:
    summary = build_fly_summary(_training_frame(), _testing_frame(), TRAIN)
    # AUC = 100 + 10*trial for trials 1..6 -> mean 135
    assert set(summary["mean_auc"].round(6)) == {135.0}
    assert set(summary["slope"].round(6)) == {10.0}


def test_build_fly_summary_drops_flies_without_training_data() -> None:
    """A fly with test scores but no conditioning record has no vigor to
    correlate; it must drop out rather than enter as NaN and skew n."""
    training = _training_frame(flies=("july_13_batch_1",))
    testing = _testing_frame(flies=("july_13_batch_1", "july_14_batch_1"))
    summary = build_fly_summary(training, testing, TRAIN)
    assert set(summary["fly"]) == {"july_13_batch_1"}


def test_build_fly_summary_drops_flies_without_testing_data() -> None:
    training = _training_frame(flies=("july_13_batch_1", "july_14_batch_1"))
    testing = _testing_frame(flies=("july_13_batch_1",))
    summary = build_fly_summary(training, testing, TRAIN)
    assert set(summary["fly"]) == {"july_13_batch_1"}


def test_build_fly_summary_mean_score_averages_the_presentations() -> None:
    summary = build_fly_summary(_training_frame(), _testing_frame(), TRAIN)
    ones = summary[summary["fly_number"] == 1]
    assert set(ones["mean_score"]) == {4.0}
    assert set(summary[summary["fly_number"] == 2]["mean_score"]) == {0.0}


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    rng = np.random.default_rng(0)
    flies = ["july_13_batch_1", "july_14_batch_1", "july_15_batch_1_rig_2"]
    training_rows, testing_rows = [], []
    for dataset in (TRAIN, CTRL):
        for fly in flies:
            for fly_number in (1, 2, 3):
                for trial in range(1, 7):
                    training_rows.append({
                        "dataset": dataset, "fly": fly, "fly_number": fly_number,
                        "trial_type": "training",
                        "trial_label": f"training_{trial}_ethylbutyrate",
                        "AUC-During": float(rng.integers(0, 400)),
                    })
                for trial, odor in [
                    (1, "ethylbutyrate"), (2, "hexanol"), (3, "citral"),
                    (4, "acv"), (5, "benzaldehyde"), (6, "linalool"),
                    (7, "3-octonol"), (8, "ethylbutyrate"),
                ]:
                    score = float(rng.integers(-1, 6))
                    testing_rows.append({
                        "dataset": dataset, "fly": fly, "fly_number": fly_number,
                        "trial_type": "testing",
                        "trial_label": f"testing_{trial}_{odor}",
                        "score": score, "prediction": int(score >= 2),
                    })
    train_path = tmp_path / "training.parquet"
    pred_path = tmp_path / "model_predictions.csv"
    pd.DataFrame(training_rows).to_parquet(train_path, index=False)
    pd.DataFrame(testing_rows).to_csv(pred_path, index=False)
    return train_path, pred_path


def test_main_writes_scatter_and_trajectory_for_both_roles(tmp_path: Path) -> None:
    from scripts.analysis.training_vs_learning import main

    train_path, pred_path = _write_inputs(tmp_path)
    out_dir = tmp_path / "training_vs_learning"
    main([
        "--training-wide-csv", str(train_path),
        "--predictions-csv", str(pred_path),
        "--train-dataset", TRAIN,
        "--control-dataset", CTRL,
        "--odor-short", "EB",
        "--out-dir", str(out_dir),
    ])
    names = {p.name for p in out_dir.iterdir()}
    assert names >= {
        "training_vs_learning_scatter_training.png",
        "training_vs_learning_scatter_control.png",
        "training_vs_learning_trajectory_training.png",
        "training_vs_learning_trajectory_control.png",
    }


def test_main_writes_a_sidecar_with_the_stats(tmp_path: Path) -> None:
    import json

    from scripts.analysis.training_vs_learning import main

    train_path, pred_path = _write_inputs(tmp_path)
    out_dir = tmp_path / "training_vs_learning"
    main([
        "--training-wide-csv", str(train_path),
        "--predictions-csv", str(pred_path),
        "--train-dataset", TRAIN,
        "--control-dataset", CTRL,
        "--odor-short", "EB",
        "--out-dir", str(out_dir),
    ])
    meta = json.loads((out_dir / "training_vs_learning.json").read_text())
    assert set(meta) >= {"training", "control"}
    role = meta["training"]
    assert role["n_flies"] == 9
    assert role["learner_threshold"] == 1.0
    assert "presentations" in role and len(role["presentations"]) == 2
    for pres in role["presentations"]:
        assert set(pres) >= {"label", "spearman_rho", "spearman_p", "mannwhitney_p"}


# ---------------------------------------------------------------------------
# Learner colouring
# ---------------------------------------------------------------------------


def test_learner_mask_splits_on_the_threshold() -> None:
    from scripts.analysis.training_vs_learning import learner_mask

    summary = pd.DataFrame({"mean_score": [0.0, 1.0, 1.5, 4.0]})
    assert list(learner_mask(summary, 1.0)) == [False, False, True, True]


def test_learner_mask_treats_exactly_the_threshold_as_non_learner() -> None:
    """The trajectory legend reads "> 1.0" / "<= 1.0"; a fly sitting exactly on
    the line must land in the same group in both the legend and the points."""
    from scripts.analysis.training_vs_learning import learner_mask

    assert list(learner_mask(pd.DataFrame({"mean_score": [1.0]}), 1.0)) == [False]


def test_learner_mask_puts_nan_scores_with_the_non_learners() -> None:
    from scripts.analysis.training_vs_learning import learner_mask

    summary = pd.DataFrame({"mean_score": [np.nan, 3.0]})
    assert list(learner_mask(summary, 1.0)) == [False, True]


def test_point_colors_are_green_for_learners_red_for_non_learners() -> None:
    from scripts.analysis.training_vs_learning import (
        LEARNER_COLOR,
        NONLEARNER_COLOR,
        point_colors,
    )

    summary = pd.DataFrame({"mean_score": [4.0, 0.0]})
    assert point_colors(summary, 1.0) == [LEARNER_COLOR, NONLEARNER_COLOR]
