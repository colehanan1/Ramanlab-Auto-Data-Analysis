"""Explainer figures for the hold-one-odor-out Ridge/Lasso results
(scripts/analysis/regression_explainer_figures.py)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import scripts.analysis.odor_holdout_regression as ohr
import scripts.analysis.regression_explainer_figures as rx

ODORS = ["Benzaldehyde", "Citral", "Hexanol", "Isoamyl Acetate"]


def _matrix(n=30, seed=1):
    """Flies with a shared 'responsiveness' so odors are predictable from each other."""
    rng = np.random.default_rng(seed)
    resp = rng.uniform(0, 3, n)
    rows = []
    for i in range(n):
        row = {"group": "1 h" if i < n // 2 else "24 h", "hours": 1.0 if i < n // 2 else 24.0,
               "fly": f"b{i // 3}", "fly_number": i % 3 + 1, "batch": f"g/b{i // 3}"}
        for o in ODORS:
            for ph in ("pre", "post"):
                row[f"{o}|{ph}"] = float(np.clip(np.round(resp[i] + rng.normal(0, 0.8)), 0, 5))
        rows.append(row)
    return pd.DataFrame(rows)


def _results():
    rows = []
    for level, target in (("level2", "score"), ("level3", "mean score")):
        for o in ODORS:
            for ph in ("pre", "post"):
                for model in ("Ridge", "Lasso"):
                    for variant, q in (("odors", 0.2), ("odors + starvation", 0.18),
                                       ("starvation only", -0.01)):
                        rows.append({"level": level, "group": "all", "odor": o, "phase": ph,
                                     "target": target, "model": model,
                                     "kind": model.lower(), "scheme": "leave-one-batch-out",
                                     "variant": variant, "n": 30, "q2": q,
                                     "p_perm": 0.01 if q > 0 else 0.6})
    d = pd.DataFrame(rows)
    l1 = d[d["level"] == "level2"].assign(group="1 h", scheme="leave-one-fly-out",
                                           q2=-0.2, p_perm=0.7)
    l1 = l1[l1["variant"] == "odors"]
    return l1, d[d["level"] == "level2"], d[d["level"] == "level3"]


def test_worked_example_matches_the_regression_q2():
    m = _matrix()
    ex = rx.worked_example(m, "Benzaldehyde", "pre")
    assert len(ex.table) == len(m)
    # identical to the regression's own formula on the same predictions
    assert ex.q2 == pytest.approx(ohr.q2(ex.table["actual"].to_numpy(),
                                         ex.table["predicted"].to_numpy(),
                                         ex.table["average_guess"].to_numpy()))
    assert ex.q2 > 0  # shared responsiveness makes it predictable
    assert ex.rmse_model < ex.rmse_average


def test_null_distribution_is_reproducible_and_centred_near_zero():
    m = _matrix()
    null = rx.null_q2(m, "Benzaldehyde", "pre", n_perm=20, seed=0)
    assert null.shape == (20,)
    assert np.allclose(null, rx.null_q2(m, "Benzaldehyde", "pre", n_perm=20, seed=0))
    assert np.nanmedian(null) < 0.1


def test_pick_example_is_the_best_predicted_pre_test_odor():
    _l1, l2, _l3 = _results()
    l2 = l2.copy()
    l2.loc[(l2["odor"] == "Citral") & (l2["phase"] == "pre") & (l2["model"] == "Ridge")
           & (l2["variant"] == "odors"), "q2"] = 0.5
    assert rx.pick_example(l2) == ("Citral", "pre")


def test_coefficients_lasso_can_drop_predictors_ridge_keeps_all():
    c = rx.coefficients(_matrix(), "Benzaldehyde", "pre")
    assert list(c.columns) == ["feature", "Ridge", "Lasso"]
    assert len(c) == 2 * (len(ODORS) - 1)  # both phases of every OTHER odor
    assert (c["Ridge"] != 0).all()


def test_odor_correlations_is_a_symmetric_odor_matrix():
    r = rx.odor_correlations(_matrix(), phase="pre")
    assert list(r.index) == list(r.columns) == sorted(ODORS)
    assert np.allclose(r.to_numpy(), r.to_numpy().T)


def test_skill_table_keeps_ridge_odors_only():
    _l1, l2, l3 = _results()
    s = rx.skill_table(l2, l3)
    assert set(s["level"]) == {"single fly", "batch average"}
    assert len(s) == 2 * len(ODORS) * 2


def test_all_figures_build(tmp_path):
    m = _matrix()
    l1, l2, l3 = _results()
    ex = rx.worked_example(m, "Benzaldehyde", "pre")
    figs = [rx.fig_how_it_works(m, ex), rx.fig_is_it_luck(ex, rx.null_q2(m, "Benzaldehyde", "pre", n_perm=10)),
            rx.fig_skill_by_odor(rx.skill_table(l2, l3)), rx.fig_within_group(l1),
            rx.fig_ridge_vs_lasso(rx.coefficients(m, "Benzaldehyde", "pre"), l1, l2, l3, ex),
            rx.fig_odors_together(rx.odor_correlations(m)), rx.fig_starvation(l2, l3)]
    for f in figs:
        assert "?" in f._suptitle.get_text()
        plt.close(f)


@pytest.mark.parametrize("cohorts, sensitization", [
    (["IAA-Sensitivity-1-1", "IAA-Sensitivity-24-1"], True),
    (["Hex-HexTest-Training-32-1"], False),
    ([], False),
])
def test_phase_wording_follows_the_cohorts(cohorts, sensitization):
    from matplotlib.text import Text

    m = _matrix()
    l1, l2, l3 = _results()
    ex = rx.worked_example(m, "Benzaldehyde", "pre")
    figs = [rx.fig_how_it_works(m, ex, cohorts=cohorts),
            rx.fig_skill_by_odor(rx.skill_table(l2, l3), cohorts=cohorts),
            rx.fig_within_group(l1, cohorts=cohorts),
            rx.fig_ridge_vs_lasso(rx.coefficients(m, "Benzaldehyde", "pre"), l1, l2, l3, ex,
                                  cohorts=cohorts),
            rx.fig_odors_together(rx.odor_correlations(m), cohorts=cohorts),
            rx.fig_starvation(l2, l3, cohorts=cohorts)]
    for f in figs:
        text = "\n".join(t.get_text() for t in f.findobj(Text)).lower()
        plt.close(f)
        assert ("pre-exposure" in text) is sensitization
        assert ("pre-test" in text) is not sensitization


def test_cohorts_come_from_the_comparison_next_to_the_regression(tmp_path):
    reg = tmp_path / "odor_holdout_regression"
    reg.mkdir()
    assert rx.regression_cohorts(reg) == []
    (tmp_path / "1h_vs_24h").mkdir()
    pd.DataFrame({"dataset_canon": ["IAA-Sensitivity-24-1", "IAA-Sensitivity-1-1"]}).to_csv(
        tmp_path / "1h_vs_24h" / "paired_scores_per_fly.csv", index=False)
    assert rx.regression_cohorts(reg) == ["IAA-Sensitivity-1-1", "IAA-Sensitivity-24-1"]
