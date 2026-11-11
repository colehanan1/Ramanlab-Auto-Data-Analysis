"""When each experiment ran (scripts/analysis/experiment_start_times.py)."""

from __future__ import annotations

import datetime as dt

import scripts.analysis.experiment_start_times as est


def _folder(root, name, start, starved_iso=None):
    f = root / "IAA-Sensitivity-1-1" / name
    f.mkdir(parents=True)
    (f / f"output_{name}_pretest_1_Hexanol-0.1_{start}.csv").write_text("x")
    (f / f"output_{name}_testing_1_Hexanol-0.1_20261008_150000.csv").write_text("x")
    if starved_iso:
        (f / "session_metadata.txt").write_text(
            "Starvation Input\n----------------\n"
            f"Starved Datetime (ISO, local naive): {starved_iso}\n"
            "Target starvation duration (hours): 1.0\n")
    return f


def test_starved_at_and_target_are_read_from_session_metadata(tmp_path):
    f = _folder(tmp_path, "october_08_batch_2", "20261008_132345", "2026-10-08T11:33:00")
    assert est.starved_at(f) == dt.datetime(2026, 10, 8, 11, 33)
    assert est.target_hours(f) == 1.0
    assert est.starved_at(tmp_path / "missing") is None


def test_table_sorted_day_batch_rig_with_hours_starved(tmp_path):
    _folder(tmp_path, "october_08_batch_2_rig_2", "20261008_130959", "2026-10-08T11:33:00")
    _folder(tmp_path, "october_08_batch_2", "20261008_132345", "2026-10-08T11:33:00")
    _folder(tmp_path, "october_09_batch_2_rig_3", "20261009_131834")
    t = est.start_table(tmp_path, "IAA-Sensitivity-1-1")
    assert list(t["folder"]) == ["october_08_batch_2", "october_08_batch_2_rig_2",
                                 "october_09_batch_2_rig_3"]
    assert list(t["rig"]) == [1, 2, 3]
    r0 = t.iloc[0]
    assert r0["first_pretest"] == dt.datetime(2026, 10, 8, 13, 23, 45)
    assert r0["start_clock"] == "13:23"
    assert abs(r0["hours_starved_at_pretest"] - (1 + 50 / 60 + 45 / 3600)) < 1e-6
    assert t.iloc[2]["hours_starved_at_pretest"] != t.iloc[2]["hours_starved_at_pretest"]  # NaN


def test_batch_summary_spans_its_rigs(tmp_path):
    _folder(tmp_path, "october_08_batch_2_rig_2", "20261008_130959")
    _folder(tmp_path, "october_08_batch_2", "20261008_132345")
    b = est.batch_summary(est.start_table(tmp_path, "IAA-Sensitivity-1-1"))
    row = b.iloc[0]
    assert row["batch"] == "october_08_batch_2"
    assert (row["earliest"], row["latest"], row["n_rigs"]) == ("13:09", "13:23", 2)
