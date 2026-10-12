"""Hold-one-odor-out Lasso/Ridge prediction (scripts/analysis/odor_holdout_regression.py).

Pins the parts that make the numbers honest: the held-out unit never reaches
the fit, the baseline is the TRAINING mean, Q² is relative to that baseline,
and a planted odor-odor relationship is found while pure noise is not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import scripts.analysis.odor_holdout_regression as ohr

ODORS = ["Benzaldehyde", "Citral", "Isoamyl Acetate"]


def _paired(n_per_group=6, seed=0, groups=("1 h", "12 h")):
    rng = np.random.default_rng(seed)
    rows = []
    for g in groups:
        for i in range(n_per_group):
            for odor in ODORS:
                rows.append(dict(group=g, dataset_canon="X", fly=f"{g}_batch_{i // 3}",
                                 fly_number=i % 3 + 1, odor=odor,
                                 score_pre=int(rng.integers(0, 5)),
                                 score_post=int(rng.integers(0, 5))))
    return pd.DataFrame(rows)


def test_fly_matrix_has_one_row_per_fly_and_a_column_per_odor_phase():
    m = ohr.fly_matrix(_paired())
    assert len(m) == 12
    assert ohr.score_columns(m) == [f"{o}|{p}" for o in ODORS for p in ("pre", "post")]
    assert set(m["hours"]) == {1.0, 12.0}
    assert m["batch"].nunique() == 4


def test_feature_target_drops_both_phases_of_the_held_out_odor():
    m = ohr.fly_matrix(_paired())
    X, y, feats = ohr.feature_target(m, "Citral", "post")
    assert not any(f.startswith("Citral|") for f in feats)
    assert len(feats) == 4 and X.shape == (12, 4)
    assert np.array_equal(y, m["Citral|post"].to_numpy(float))


def test_cv_predict_never_fits_on_the_held_out_row_and_baseline_is_training_mean():
    seen = []

    class Spy:
        def fit(self, X, y):
            seen.append(set(map(tuple, X)))
            self.m = float(np.mean(y))
            return self

        def predict(self, X):
            return np.full(len(X), self.m)

    X = np.arange(10, dtype=float).reshape(-1, 1) * np.array([[1.0, 2.0]])
    y = np.arange(10, dtype=float)
    splits = list(ohr.loo_splits(10))
    pred, base = ohr.cv_predict(X, y, splits, make=lambda n: Spy())
    for (train, test), rows in zip(splits, seen):
        assert tuple(X[test[0]]) not in rows
    for train, test in splits:
        assert base[test[0]] == pytest.approx(y[train].mean())


def test_q2_is_one_for_perfect_and_zero_for_baseline():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    base = np.array([3.0, 3.0, 2.0, 2.0])
    assert ohr.q2(y, y, base) == pytest.approx(1.0)
    assert ohr.q2(y, base, base) == pytest.approx(0.0)


def test_permutation_p_counts_the_observed_value():
    assert ohr.permutation_p(0.5, np.array([0.1, 0.6, 0.2, 0.3])) == pytest.approx(2 / 5)


def test_a_planted_relationship_is_found_and_noise_is_not():
    rng = np.random.default_rng(1)
    n = 24
    a = rng.normal(size=n)
    df = pd.DataFrame({"x|pre": a, "x|post": rng.normal(size=n),
                       "noise|pre": rng.normal(size=n), "noise|post": rng.normal(size=n),
                       "y|pre": a + 0.1 * rng.normal(size=n), "y|post": rng.normal(size=n)})
    splits = list(ohr.loo_splits(n))
    for kind in ("ridge", "lasso"):
        X, y, _ = ohr.feature_target(df, "y", "pre")
        res = ohr.evaluate(X, y, splits, kind=kind, n_perm=49, seed=0)
        assert res["q2"] > 0.7 and res["p_perm"] <= 0.05
        X, y, _ = ohr.feature_target(df, "y", "post")
        res = ohr.evaluate(X, y, splits, kind=kind, n_perm=49, seed=0)
        assert res["q2"] < 0.2 and res["p_perm"] > 0.05


def test_binary_targets_fall_back_to_prevalence_when_a_fold_has_one_class():
    X = np.random.default_rng(0).normal(size=(6, 2))
    y = np.array([0, 0, 0, 0, 0, 1], dtype=float)
    pred, base = ohr.cv_predict(X, y, list(ohr.loo_splits(6)),
                                make=lambda n: ohr.make_model("logit_l2", n), binary=True)
    assert np.allclose(pred, base)


def test_batch_table_averages_flies_and_counts_responders():
    m = ohr.fly_matrix(_paired())
    b = ohr.batch_table(m, binary_threshold=2)
    assert len(b) == 4 and set(b["n_flies"]) == {3}
    one = m[m["batch"] == b["batch"].iloc[0]]
    assert b["Citral|pre"].iloc[0] == pytest.approx(one["Citral|pre"].mean())
    assert b["Citral|pre|pct"].iloc[0] == pytest.approx(100 * (one["Citral|pre"] >= 2).mean())


def test_run_all_writes_the_three_level_tables(tmp_path):
    paired = _paired(n_per_group=6, groups=("1 h", "12 h", "24 h"))
    out = ohr.run_all(paired, out_dir=tmp_path, n_perm=9, n_perm_pop=9, n_jobs=1)
    names = sorted(p.name for p in out if p.suffix == ".csv")
    assert {"level1_individual_within_starvation.csv", "level2_individual_pooled.csv",
            "level3_population_batches.csv"} <= set(names)
    l1 = pd.read_csv(tmp_path / "level1_individual_within_starvation.csv")
    assert {"group", "odor", "phase", "target", "model", "q2", "p_perm"} <= set(l1.columns)


def test_figures_only_redraws_from_the_saved_tables(tmp_path):
    paired = _paired(n_per_group=6, groups=("1 h", "12 h", "24 h"))
    ohr.run_all(paired, out_dir=tmp_path, n_perm=4, n_perm_pop=4, n_jobs=1, figures=False)
    written = ohr.figures_from_csv(tmp_path, trained_odor="Isoamyl Acetate")
    assert sorted(p.name for p in written) == [
        "level1_individual_within_starvation_heatmap.png", "overview_q2_all_levels.png",
        "relationships_ridge_coefficients.png"]
