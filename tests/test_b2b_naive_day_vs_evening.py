"""Naive pre-test PER, b2b day window vs evening window, both datasets pooled."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from fbpipe.analysis import fly_stats as fs  # noqa: E402
from scripts.analysis import b2b_baseline_time_of_day as bt  # noqa: E402
from scripts.analysis import b2b_naive_day_vs_evening as dv  # noqa: E402

THR = 2.0
FIRST, SECOND = "IAA-PairFirst-Control-1-1", "IAA-PairSecond-Control-1-1"


def _csv(tmp: Path, name: str, folders, score_of) -> Path:
    rows = []
    for ds, folder in folders:
        for fn in (1, 2):
            for odor in ("IAA", "Hexanol", "Paraffin Oil"):
                rows.append(dict(dataset=ds, fly=folder, fly_number=fn, odor=odor,
                                 label=f"pretest_x_{odor}", score=score_of(folder, fn, odor)))
    p = tmp / f"{name}.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


DAY_FOLDERS = ((FIRST, "november_13_batch_1_rig_2"), (SECOND, "november_13_batch_1"),
               (FIRST, "november_14_batch_1"), (SECOND, "november_16_batch_1_rig_3"))
EVE_FOLDERS = ((FIRST, "november_17_batch_2"), (SECOND, "november_18_batch_2_rig_2"))


@pytest.fixture
def paths(tmp_path):
    day = _csv(tmp_path, "day", DAY_FOLDERS, lambda f, n, o: 0 if o != "IAA" else n)
    eve = _csv(tmp_path, "eve", EVE_FOLDERS, lambda f, n, o: 3 if o == "Hexanol" else 0)
    return {dv.DAY: day, dv.EVENING: eve}


def test_loader_pools_both_datasets_by_window(paths):
    t = dv.load_trials(paths, threshold=THR)
    assert set(t["window"]) == {dv.DAY, dv.EVENING}
    assert set(t.loc[t["window"] == dv.DAY, "dataset"]) == {FIRST, SECOND}
    assert t.loc[t["batch"] == "november_16_batch_1_rig_3", "session"].iloc[0] == \
        "november_16_batch_1"
    assert t["responded"].equals(t["score"] >= THR)
    assert t.groupby("fly_id")["window"].nunique().max() == 1


def test_a_folder_in_both_windows_is_refused(tmp_path):
    same = ((FIRST, "november_13_batch_1_rig_2"),)
    p = {dv.DAY: _csv(tmp_path, "a", same, lambda *a: 0),
         dv.EVENING: _csv(tmp_path, "b", same, lambda *a: 0)}
    with pytest.raises(ValueError, match="both windows"):
        dv.load_trials(p, threshold=THR)


def test_levels_pool_trials_and_leave_the_click_out_of_the_pooled_set(paths):
    t = dv.load_trials(paths, threshold=THR)
    lv = dv.levels_table(t).set_index(["window", "odor_set", "metric"])
    # day, IAA + Hexanol: IAA scores 1, 2 per folder (4 folders), Hexanol 0 -> 12 / 16
    assert lv.loc[(dv.DAY, dv.POOLED, "score"), "value"] == pytest.approx(12 / 16)
    assert lv.loc[(dv.DAY, dv.POOLED, "score"), "n_trials"] == 16
    # evening: Hexanol 3 (responds), IAA 0 -> 50 % of the 8 pooled trials
    assert lv.loc[(dv.EVENING, dv.POOLED, "rate"), "value"] == pytest.approx(50.0)
    assert lv.loc[(dv.EVENING, "Paraffin Oil", "rate"), "value"] == pytest.approx(0.0)


def test_tests_are_evening_minus_day_with_whole_sessions_permuted(paths):
    t = dv.load_trials(paths, threshold=THR)
    st = dv.tests_table(t, n_boot=200)
    r = st[(st["odor_set"] == dv.POOLED) & (st["metric"] == "score")].iloc[0]
    eve, day = t[t["window"] == dv.EVENING], t[t["window"] == dv.DAY]
    pooled = lambda d: d[d["odor"] != dv.CONTROL]["score"].mean()  # noqa: E731
    assert r["estimate"] == pytest.approx(pooled(eve) - pooled(day))
    both = t[t["odor"] != dv.CONTROL]
    want = bt.label_perm(both["score"].to_numpy(dtype=float), both["session"],
                         dict(zip(both["session"], both["window"])))
    assert r["p_session"] == pytest.approx(want["p"])
    assert r["n_sessions_day"] == 3 and r["n_sessions_evening"] == 2
    per = st[(st["odor_set"] != dv.POOLED) & (st["metric"] == "score")]
    np.testing.assert_allclose(per["p_holm"], fs.holm_adjust(per["p_session"].tolist()))
    assert (st.loc[st["odor_set"] == dv.POOLED, "p_holm"]
            == st.loc[st["odor_set"] == dv.POOLED, "p_session"]).all()   # the headline


def test_build_writes_the_figure_and_tables(paths, tmp_path):
    out = tmp_path / "out"
    written = dv.build(paths, out, threshold=THR, n_boot=200)
    names = {p.name for p in written}
    assert {"naive_day_vs_evening.png", "naive_day_vs_evening.pdf",
            "naive_day_vs_evening_levels.csv", "naive_day_vs_evening_tests.csv"} <= names
