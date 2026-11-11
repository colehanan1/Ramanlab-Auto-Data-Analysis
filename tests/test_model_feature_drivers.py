"""Which of the scoring model's inputs drive its scores on one cohort.

Three views per input: its share of the model's training gain, its share of
the per-trial attribution (SHAP contributions to the class each trial got),
and its Spearman correlation with the score.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

import scripts.analysis.model_feature_drivers as mfd


@pytest.fixture(scope="module")
def toy():
    """7-class softmax model whose label is driven by ``strong`` alone (plus a
    little ``weak``); ``noise`` carries nothing."""
    rng = np.random.default_rng(0)
    n = 600
    X = pd.DataFrame({"strong": rng.normal(size=n), "weak": rng.normal(size=n),
                      "noise": rng.normal(size=n)})
    y = np.clip(np.round(3 + 1.5 * X["strong"] + 0.3 * X["weak"]), 0, 6).astype(int)
    booster = xgb.train(
        {"objective": "multi:softmax", "num_class": 7, "max_depth": 3, "eta": 0.3,
         "verbosity": 0},
        xgb.DMatrix(X, label=y, feature_names=list(X.columns)), num_boost_round=30)
    return booster, X


def test_the_driving_input_ranks_first_in_every_view(toy):
    booster, X = toy
    score = mfd.predict_classes(booster, X) - 1           # class index -> -1..5
    phase = np.where(np.arange(len(X)) % 2 == 0, "pretest", "testing")
    t = mfd.driver_table(booster, X, score, phase).set_index("feature")
    assert t.index[0] == "strong"                         # sorted by attribution
    assert t["gain_share"].idxmax() == "strong"
    assert t["rho_all"].idxmax() == "strong"
    assert t.loc["strong", "rho_all"] > 0.8
    assert abs(t.loc["noise", "rho_all"]) < 0.2
    assert t["shap_share"].sum() == pytest.approx(1.0)
    assert t["gain_share"].sum() == pytest.approx(1.0)
    assert {"rho_pretest", "rho_testing"} <= set(t.columns)


def test_attribution_reads_the_class_each_trial_was_given():
    # (trials, classes, features + bias)
    contrib = np.zeros((2, 3, 3))
    contrib[0, 1, 0] = 2.0      # trial 0 was given class 1: feature 0 pushed +2
    contrib[0, 2, 1] = 9.0      # ...class 2 is not trial 0's class: ignored
    contrib[1, 2, 1] = -4.0     # trial 1 was given class 2: feature 1 pushed -4
    contrib[1, 2, 2] = 100.0    # the bias column is never a feature
    out = mfd.attribution(contrib, np.array([1, 2]))
    assert out.tolist() == [1.0, 2.0]                     # mean |contribution|


def test_scores_must_match_before_the_ranking_is_trusted():
    assert mfd.score_agreement(np.array([0, 1, 2]), np.array([0, 1, 2])) == (3, 3)
    assert mfd.score_agreement(np.array([0, 1, 2]), np.array([0, 2, np.nan])) == (1, 2)


def test_rho_legend_names_the_phases_by_cohort(toy):
    booster, X = toy
    score = mfd.predict_classes(booster, X) - 1
    phase = np.where(np.arange(len(X)) % 2 == 0, "pretest", "testing")
    table = mfd.driver_table(booster, X, score, phase)

    def legend(cohorts):
        fig = mfd.plot_drivers(table, "title", "note", cohorts=cohorts)
        texts = [t.get_text() for t in fig.axes[2].get_legend().get_texts()]
        plt.close(fig)
        return texts

    assert legend(["IAA-Sensitivity-1-1"]) == ["Pre-exposure", "Post-exposure",
                                               "Pre + post"]
    assert legend(["Hex-HexTest-Training-32-1"]) == ["Pre-test", "Post-training", "Pre + post"]
    # a mix keeps the conditioning words
    assert legend(["IAA-Sensitivity-1-1", "Hex-HexTest-Training-32-1"])[:2] == [
        "Pre-test", "Post-training"]
