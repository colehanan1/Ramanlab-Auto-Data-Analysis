"""Predicting a fly's PER change after training (scripts/analysis/change_prediction.py)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import scripts.analysis.change_prediction as cp
import scripts.analysis.odor_holdout_regression as ohr

ODORS = ["Benzaldehyde", "Citral", "Hexanol", "Isoamyl Acetate"]


def _paired(n=36, seed=0):
    """Flies whose changes share a per-fly 'arousal' term, so changes are predictable
    from each other; pre-test is pure noise, so pre-test should NOT predict the change."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        arousal = rng.normal(0, 1.5)
        for o in ODORS:
            pre = int(rng.integers(0, 3))
            post = int(np.clip(np.round(pre + arousal + rng.normal(0, 0.4)), -1, 5))
            rows.append({"dataset_canon": "IAA-Sensitivity-1-1", "fly": f"october_{10 + i // 4:02d}_batch_1",
                         "fly_number": i % 4 + 1, "odor": o, "score_pre": pre, "score_post": post})
    return pd.DataFrame(rows)


def test_change_matrix_adds_post_minus_pre_per_odor():
    m = cp.change_matrix(_paired())
    for o in ODORS:
        assert np.allclose(m[f"{o}|delta"], m[f"{o}|post"] - m[f"{o}|pre"])


@pytest.mark.parametrize("which, n_cols", [("pre_all", 4), ("pre_others", 3), ("pre_own", 1),
                                           ("delta_others", 3)])
def test_predictor_sets(which, n_cols):
    m = cp.change_matrix(_paired())
    X = cp.predictors(m, "Hexanol", which)
    assert X.shape == (len(m), n_cols)
    if which == "delta_others":   # the target odor's own change is never a predictor
        own = m["Hexanol|delta"].to_numpy(float)
        assert not any(np.allclose(X[:, j], own) for j in range(X.shape[1]))


def test_every_fold_holds_out_one_whole_batch():
    m = cp.change_matrix(_paired())
    for train, test in ohr.group_splits(m["batch"]):
        held = set(m.loc[test, "batch"])
        assert len(held) == 1 and not held & set(m.loc[train, "batch"])


def test_changes_predict_each_other_but_noise_pre_test_does_not():
    m = cp.change_matrix(_paired())
    subset = m["fly"].str.startswith("october_1").to_numpy()
    y = m["Hexanol|delta"].to_numpy(float)
    splits = ohr.group_splits(m["batch"])
    good = cp.evaluate(cp.predictors(m, "Hexanol", "delta_others"), y, splits, subset,
                       kind="ridge", n_perm=20, seed=0)
    assert good["q2_all"] > 0.5 and good["p_all"] < 0.1
    assert {"q2_subset", "p_subset", "n_subset"} <= set(good)
    bad = cp.evaluate(cp.predictors(m, "Hexanol", "pre_others"), y, splits, subset,
                      kind="ridge", n_perm=20, seed=0)
    assert bad["q2_all"] < 0.2


def test_subset_q2_uses_only_the_subset_flies():
    y = np.array([0, 1, 2, 3, 4, 5], float)
    pred = np.array([0, 1, 2, 0, 0, 0], float)   # perfect on the first three only
    base = np.full(6, 2.5)
    s = np.array([True, True, True, False, False, False])
    assert cp.subset_q2(y, pred, base, s) == pytest.approx(1.0)
    assert cp.subset_q2(y, pred, base, ~s) < 0


def test_run_and_figures(tmp_path):
    m = cp.change_matrix(_paired())
    subset = m["fly"].isin(["october_10_batch_1", "october_11_batch_1"]).to_numpy()
    res, preds = cp.run(m, subset, n_perm=5, n_jobs=1, models=("ridge",))
    assert set(res["predictors"]) == set(cp.PREDICTOR_SETS)
    assert len(res) == len(ODORS) * len(cp.PREDICTOR_SETS)
    figs = [cp.fig_skill(res, trained="Isoamyl Acetate", subset_label="1–2 pm"),
            cp.fig_trained_example(preds, res, m, subset, trained="Isoamyl Acetate",
                                   subset_label="1–2 pm"),
            cp.fig_change_correlations(m, trained="Isoamyl Acetate")]
    for f in figs:
        assert "?" in f._suptitle.get_text()
        plt.close(f)


# ── dataset-level consistency ──────────────────────────────────────────────


def _odor_specific(n=40, seed=1):
    """Every fly raises Isoamyl Acetate by ~2 and leaves the rest alone: a perfectly
    consistent dataset pattern, so the odor average should predict individual flies."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        for o in ODORS:
            pre = 0
            post = int(np.clip(np.round((2 if o == "Isoamyl Acetate" else 0) + rng.normal(0, 0.3)), -1, 5))
            rows.append({"dataset_canon": "D", "fly": f"october_{10 + i // 8:02d}_batch_1"
                         + ("" if i % 2 else "_rig_2"), "fly_number": i % 8 + 1, "odor": o,
                         "score_pre": pre, "score_post": post})
    return pd.DataFrame(rows)


def test_odor_average_predicts_a_consistent_dataset_pattern():
    m = cp.change_matrix(_odor_specific())
    out = cp.odor_average(m)
    assert out["q2_all"] > 0.8
    assert out["p_friedman"] < 0.001
    t = out["per_odor"].set_index("odor")
    assert t.loc["Isoamyl Acetate", "pct_up"] > 90 and t.loc["Hexanol", "pct_same"] > 80


def test_sessions_pool_rigs_and_batch_profiles_agree_when_consistent():
    m = cp.change_matrix(_odor_specific())
    assert set(m["session"]) == {f"october_{d}_batch_1" for d in range(10, 15)}
    prof, r = cp.batch_consistency(m)
    assert list(prof.columns)[0] == "Isoamyl Acetate"
    assert (r["r_with_rest"] > 0.9).all()


def test_dataset_figures_build():
    m = cp.change_matrix(_odor_specific())
    subset = m["session"].eq("october_10_batch_1").to_numpy()
    for f in (cp.fig_dataset_consistency(m, subset, trained="Isoamyl Acetate", subset_label="x"),
              cp.fig_batch_consistency(m, trained="Isoamyl Acetate")):
        assert "?" in f._suptitle.get_text()
        plt.close(f)


def test_answers_do_not_claim_a_link_when_there_is_none():
    """Independent changes (odor-specific data): the wording must say 'no', not 'go together'."""
    m = cp.change_matrix(_odor_specific())
    f = cp.fig_change_correlations(m, trained="Isoamyl Acetate")
    assert "tends to rise" not in f.answer
    plt.close(f)
    assert "no better than" in cp.skill_words(-0.03)
    assert "no better than" in cp.skill_words(0.01)
    assert cp.skill_words(0.25).startswith("25% better")


@pytest.mark.parametrize("dataset, pre, tag, block, sep", [
    ("IAA-Sensitivity-1-1", "pre-exposure", "(sensitized odor)", "exposure", "\n"),
    ("Hex-HexTest-Training-32-1", "pre-test", "(trained)", "training", " "),
    (None, "pre-test", "(trained)", "training", " "),
])
def test_figure_wording_follows_the_dataset(dataset, pre, tag, block, sep):
    iaa = "Isoamyl Acetate"
    m = cp.change_matrix(_paired())
    subset = m["fly"].eq("october_10_batch_1").to_numpy()
    res, _preds = cp.run(m, subset, n_perm=0, n_jobs=1, models=("ridge",))
    f = cp.fig_skill(res, trained=iaa, subset_label="x", dataset=dataset)
    assert f"from its {pre} to the other odors" in [t.get_text() for t in f.legends[0].get_texts()]
    assert f"{iaa} {tag}" in [t.get_text() for t in f.axes[0].get_yticklabels()]
    assert f"From its {pre} to the other odors" in f.answer
    plt.close(f)
    f = cp.fig_dataset_consistency(m, subset, trained=iaa, subset_label="x", dataset=dataset)
    assert f.axes[0].get_ylabel().startswith(f"Change in PER score after {block}\n")
    assert f.axes[0].get_xticklabels()[0].get_text().startswith(f"{iaa}{sep}{tag}\n↑")
    plt.close(f)
    f = cp.fig_batch_consistency(m, trained=iaa, dataset=dataset)
    assert f.axes[0].get_xticklabels()[0].get_text() == f"{iaa}\n{tag}"
    plt.close(f)


@pytest.mark.parametrize("dataset, change", [
    ("IAA-Sensitivity-1-1", "Change = post-exposure minus pre-exposure PER score."),
    ("Hex-HexTest-Training-32-1", "Change = post-training minus pre-test PER score."),
])
def test_answers_headline_names_the_phases(tmp_path, dataset, change):
    m = cp.change_matrix(_paired())
    subset = m["fly"].eq("october_10_batch_1").to_numpy()
    res, preds = cp.run(m, subset, n_perm=0, n_jobs=1, models=("ridge",))
    cp._render(m, subset, res, preds, out=tmp_path, dataset=dataset,
               trained="Isoamyl Acetate", label="x")
    assert change in (tmp_path / "ANSWERS.txt").read_text().splitlines()[0]
