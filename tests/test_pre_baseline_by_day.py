"""Pre-exposure baseline by day: pooled-trial estimates and batch-level trend tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.analysis import pre_baseline_by_day as pbd


def _rows(spec):
    """spec: (day, batch, rig, cohort, fly, odor, score)."""
    out = pd.DataFrame(spec, columns=["day", "batch_folder", "rig", "cohort", "fly_number",
                                      "odor", "score_pre"])
    out["start_time"] = out["day"] + " 10:00"
    out["fly_id"] = out["batch_folder"] + "_fly" + out["fly_number"].astype(str)
    out["responded_pre"] = out["score_pre"] >= 2
    return out.drop(columns="day")


def test_day_mean_pools_trials_not_fly_means():
    # fly 1: two trials (0, 0); fly 2: one trial (3). Pooled 3/3 = 1.0, mean of means 1.5.
    rows = _rows([("2026-10-01", "b1", 2, "C", 1, "A", 0),
                  ("2026-10-01", "b1", 2, "C", 1, "B", 0),
                  ("2026-10-01", "b1", 2, "C", 2, "A", 3)])
    t = pbd.day_table(rows, metric="score")
    assert t.loc[0, "value"] == pytest.approx(1.0)
    assert t.loc[0, "n_flies"] == 2 and t.loc[0, "n_trials"] == 3


def test_day_rate_is_percent_of_trials():
    rows = _rows([("2026-10-01", "b1", 2, "C", 1, "A", 2),
                  ("2026-10-01", "b1", 2, "C", 1, "B", 0),
                  ("2026-10-01", "b1", 2, "C", 2, "A", 0),
                  ("2026-10-01", "b1", 2, "C", 2, "B", 0)])
    t = pbd.day_table(rows, metric="pct")
    assert t.loc[0, "value"] == pytest.approx(25.0)
    assert t.loc[0, "lo"] <= 25.0 <= t.loc[0, "hi"]


def test_day_table_by_group_splits_rows():
    rows = _rows([("2026-10-01", "b1", 1, "C", 1, "A", 0),
                  ("2026-10-01", "b2", 2, "C", 1, "A", 3),
                  ("2026-10-02", "b3", 2, "C", 1, "A", 1)])
    t = pbd.day_table(rows, metric="score", by="rig")
    assert len(t) == 3
    assert set(t["rig"]) == {1, 2}


def _trend_rows(values_by_batch):
    """One batch per day, two flies per batch, one odor, each fly scoring ``v``."""
    spec = []
    for i, (rig, v) in enumerate(values_by_batch):
        day = f"2026-10-{i + 1:02d}"
        for fly in (1, 2):
            spec.append((day, f"b{i}", rig, "C", fly, "A", v))
    return _rows(spec)


def test_batch_trend_detects_monotone_rise():
    rows = _trend_rows([(2, v) for v in (0, 0, 1, 1, 2, 2, 3, 3, 3, 3)])
    res = pbd.batch_trend(rows, metric="score", n_perm=4000, seed=1)
    assert res["n_batches"] == 10
    assert res["rho"] > 0.9
    assert res["p"] < 0.01


def test_batch_trend_within_rig_removes_rig_confound():
    # Early batches are rig 1 (low), late ones rig 2 (high); flat within each rig.
    rows = _trend_rows([(1, 0)] * 5 + [(2, 3)] * 5)
    res = pbd.batch_trend(rows, metric="score", n_perm=4000, seed=1)
    assert res["p"] < 0.05                 # pooled: looks like a trend
    assert res["p_within_rig"] > 0.5       # within rig: nothing


def test_batch_trend_uses_one_value_per_batch():
    # 30 flies in one batch must not count as 30 points
    spec = [("2026-10-01", "big", 2, "C", f, "A", 3) for f in range(30)]
    spec += [(f"2026-10-{d:02d}", f"b{d}", 2, "C", 1, "A", 0) for d in range(2, 6)]
    res = pbd.batch_trend(_rows(spec), metric="score", n_perm=2000, seed=0)
    assert res["n_batches"] == 5


def test_odor_day_table_has_one_cell_per_odor_day():
    rows = _rows([("2026-10-01", "b1", 2, "C", 1, "A", 2),
                  ("2026-10-01", "b1", 2, "C", 1, "B", 0),
                  ("2026-10-02", "b2", 2, "C", 1, "A", 1)])
    t = pbd.odor_day_table(rows, metric="score")
    assert len(t) == 3
    cell = t[(t["odor"] == "A") & (t["day"] == "2026-10-01")].iloc[0]
    assert cell["value"] == 2 and cell["n_flies"] == 1


def test_permutation_p_never_zero():
    rows = _trend_rows([(2, v) for v in range(4)])
    res = pbd.batch_trend(rows, metric="score", n_perm=500, seed=0)
    assert res["p"] > 0


def test_odor_trends_one_row_per_odor_with_holm():
    spec = []
    for i in range(8):
        day = f"2026-10-{i + 1:02d}"
        for fly in (1, 2):
            spec.append((day, f"b{i}", 2, "C", fly, "A", min(3, i // 2)))   # rises
            spec.append((day, f"b{i}", 2, "C", fly, "B", (i * 5) % 3))      # no trend
    t = pbd.odor_trends(_rows(spec), metric="score", n_perm=2000, seed=0)
    assert list(t["odor"]) == ["A", "B"]
    a = t.set_index("odor").loc["A"]
    assert a["rho"] > 0.9 and a["p"] < 0.05
    assert (t["p_within_rig_holm"] >= t["p_within_rig"]).all()
