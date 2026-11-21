"""Naive (pre-exposure) PER: Sensitivity cohorts vs the b2b day-window flies."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from scripts.analysis import naive_sensitivity_vs_b2b as nv  # noqa: E402

THR = 2.0


def _sens_csv(tmp: Path) -> Path:
    rows = []
    # cohort A: 2 sessions (one with two rig folders); cohort B: 1 session
    for cohort, folder, n_fly in (("IAA-Sensitivity-1-1", "october_09_batch_1", 2),
                                  ("IAA-Sensitivity-1-1", "october_09_batch_1_rig_3", 1),
                                  ("Hex-Sensitivity-1-0.1", "october_10_batch_1", 2)):
        for f in range(1, n_fly + 1):
            for odor, score in (("Isoamyl Acetate", 3), ("Hexanol", 0), ("Paraffin Oil", 4)):
                rows.append(dict(cohort=cohort, batch_folder=folder, fly_number=f,
                                 fly_id=f"{folder}_fly{f}", in_window=True, odor=odor,
                                 score_pre=score, responded_pre=score >= THR))
    # a fly outside the clock window is dropped
    rows.append(dict(cohort="IAA-Sensitivity-1-1", batch_folder="october_11_batch_2",
                     fly_number=1, fly_id="october_11_batch_2_fly1", in_window=False,
                     odor="Hexanol", score_pre=5, responded_pre=True))
    p = tmp / "sens.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def _b2b_csv(tmp: Path) -> Path:
    rows = []
    for ds, folder, n_fly in (("IAA-PairFirst-Control-1-1", "november_13_batch_1_rig_2", 2),
                              ("IAA-PairSecond-Control-1-1", "november_13_batch_1", 3)):
        for f in range(1, n_fly + 1):
            for odor, score in (("IAA", 0), ("Hexanol", 1), ("Paraffin Oil", 0)):
                rows.append(dict(dataset=ds, fly=folder, fly_number=f, odor=odor,
                                 label=f"pretest_x_{odor}", score=score))
    p = tmp / "b2b.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


@pytest.fixture
def trials(tmp_path):
    return nv.load_trials(_sens_csv(tmp_path), _b2b_csv(tmp_path), threshold=THR)


def test_loader_unifies_both_sources(trials):
    assert set(trials["group"]) == {nv.SENS, nv.FIRST, nv.SECOND}
    assert "IAA" in set(trials["odor"]) and "Isoamyl Acetate" not in set(trials["odor"])
    assert not trials["batch"].str.contains("october_11").any()          # out of window
    assert trials.loc[trials.batch == "october_09_batch_1_rig_3", "session"].iloc[0] \
        == "october_09_batch_1"
    assert trials["responded"].equals(trials["score"] >= THR)
    # fly ids never collide across cohorts / datasets
    assert trials.groupby("fly_id")["cohort"].nunique().max() == 1


def test_groups_include_b2b_combined_and_each_sensitivity_cohort(trials):
    g = nv.groups(trials)
    assert len(g[nv.BOTH]) == len(g[nv.FIRST]) + len(g[nv.SECOND])
    assert len(g[nv.SENS]) == len(g["IAA-Sensitivity-1-1"]) + len(g["Hex-Sensitivity-1-0.1"])


def test_pooled_odor_set_leaves_out_the_paraffin_partner(trials):
    pooled = nv.odor_rows(trials, nv.POOLED)
    assert nv.CONTROL not in set(pooled["odor"])
    assert set(nv.odor_rows(trials, "Hexanol")["odor"]) == {"Hexanol"}


def test_levels_pool_trials_not_fly_means():
    t = pd.DataFrame({"fly_id": ["a", "a", "a", "b"], "score": [2.0, 2.0, 2.0, 0.0],
                      "responded": [True, True, True, False], "batch": "x", "session": "x"})
    rate = nv.level(t, "rate")
    assert rate["value"] == pytest.approx(75.0)          # 3 of 4 trials, not (100 + 0) / 2
    assert nv.level(t, "score")["value"] == pytest.approx(1.5)
    assert rate["n_flies"] == 2 and rate["n_trials"] == 4


def test_compare_reports_the_difference_and_three_cluster_levels(trials):
    g = nv.groups(trials)
    r = nv.compare(nv.odor_rows(g[nv.SENS], nv.POOLED), nv.odor_rows(g[nv.BOTH], nv.POOLED),
                   "rate")
    # Sensitivity: IAA 3 (yes) + Hexanol 0 (no) -> 50 %; b2b: none respond -> 0 %
    assert r["estimate"] == pytest.approx(50.0)
    for k in ("p_session", "p_batch", "p_fly", "ci_lo", "ci_hi", "n_sessions_a", "n_sessions_b"):
        assert k in r
    assert 0.0 < r["p_session"] <= 1.0 and r["p_fly"] <= r["p_session"] + 1e-12


def test_compare_refuses_a_cluster_in_both_groups(trials):
    a = trials[trials.group == nv.FIRST]
    b = a.assign(group=nv.SECOND)
    with pytest.raises(ValueError, match="both groups"):
        nv.compare(a, b, "rate", unit="session")


def test_tests_table_holm_families(trials):
    t = nv.tests_table(trials)
    pooled = t[t.odor_set == nv.POOLED]
    assert set(pooled["b_group"]) == {nv.FIRST, nv.SECOND, nv.BOTH}
    assert (pooled["a_group"] == nv.SENS).all()
    for (_, _), fam in pooled.groupby(["metric", "odor_set"]):
        assert (fam["p_holm"] >= fam["p_session"] - 1e-12).all() and len(fam) == 3
    per = t[t.odor_set != nv.POOLED]
    assert per.groupby(["metric", "b_group"]).size().max() == len(set(trials["odor"]))


def test_pre_registered_plan_is_pinned():
    assert nv.PLAN["headline_cluster"] == "session"
    assert nv.PLAN["pooled_odors"] == "the 7 odors; Paraffin Oil left out (b2b: master-valve click)"
    assert nv.HEADLINE == "p_session"


def test_build_writes_figure_tables_and_readme(tmp_path):
    out = tmp_path / "out"
    written = nv.build(_sens_csv(tmp_path), _b2b_csv(tmp_path), out, threshold=THR, n_boot=200)
    names = {p.name for p in written}
    assert {"naive_sensitivity_vs_b2b.png", "naive_sensitivity_vs_b2b.pdf",
            "naive_levels.csv", "naive_tests.csv", "naive_trials.csv", "README.md"} <= names
    readme = (out / "README.md").read_text()
    assert "whole sessions" in readme and "click" in readme
    lv = pd.read_csv(out / "naive_levels.csv")
    assert {nv.SENS, nv.FIRST, nv.SECOND, nv.BOTH} <= set(lv["group"])
