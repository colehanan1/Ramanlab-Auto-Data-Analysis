"""Day-vs-day auto-detection (scripts/analysis/day_vs_day_report.py).

Only the pure logic is pinned here: parsing a batch folder's recording day,
picking which two days to compare, and the starvation hours a dataset name
carries. The report itself shells out to already-tested analysis scripts
(pretest_vs_test_comparison, starvation_pre_post_comparison, ...), the same
pattern as the other *_vs_* driver scripts in this repo.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

import scripts.analysis.day_vs_day_report as dvd


@pytest.mark.parametrize(
    "folder, expected",
    [
        ("october_08_batch_2", dt.date(2026, 10, 8)),
        ("october_08_batch_2_rig_2", dt.date(2026, 10, 8)),
        ("august_28_batch_1_rig_2", dt.date(2026, 8, 28)),
        ("september_02_batch_1_rig_2", dt.date(2026, 9, 2)),
    ],
)
def test_folder_date(folder, expected):
    assert dvd.folder_date(folder) == expected


def test_folder_date_rejects_an_unparseable_name():
    with pytest.raises(ValueError):
        dvd.folder_date("not_a_recording_folder")


def test_day_prefix_and_label():
    assert dvd.day_prefix(dt.date(2026, 10, 8)) == "october_08"
    assert dvd.day_label(dt.date(2026, 10, 8)) == "Oct 8"


@pytest.mark.parametrize(
    "dataset, hours, conc",
    [("IAA-Sensitivity-18-1", 18.0, "1"), ("IAA-Sensitivity-24-1", 24.0, "1"),
     ("Hex-Sensitivity-24-0.1", 24.0, "0.1")],
)
def test_starvation_hours(dataset, hours, conc):
    h, c = dvd.starvation_hours(dataset)
    assert h == hours and c == conc


def test_usable_days_counts_unflagged_flies_per_calendar_day(tmp_path):
    wide = pd.DataFrame({
        "dataset": ["D"] * 6,
        "fly": ["october_08_batch_1", "october_08_batch_1", "october_08_batch_1_rig_2",
                "october_09_batch_1", "october_09_batch_1", "october_10_batch_1"],
        "fly_number": [1, 2, 1, 1, 2, 1],
    })
    flagged = tmp_path / "flagged.csv"
    flagged.write_text(
        'dataset,fly,fly_number,"FLY-State(1, 0, -1)",comment,,,,\n'
        "D,october_09_batch_1,1,0,bad,,,,\n"
        "D,october_09_batch_1,2,0,bad,,,,\n"
    )
    days = dvd.usable_days(wide, flagged_flies_csv=flagged)
    counts = {d.date: d.n_flies for d in days}
    assert counts == {dt.date(2026, 10, 8): 3, dt.date(2026, 10, 10): 1}
    # Oct 9 had both its flies flagged, so it drops out entirely.
    assert dt.date(2026, 10, 9) not in counts


def test_pick_most_recent_pair_orders_earlier_first():
    days = [dvd.DayInfo(dt.date(2026, 10, 8), 5), dvd.DayInfo(dt.date(2026, 10, 10), 3),
            dvd.DayInfo(dt.date(2026, 9, 1), 4)]
    a, b = dvd.pick_days(days)
    assert (a.date, b.date) == (dt.date(2026, 10, 8), dt.date(2026, 10, 10))


def test_pick_days_with_fewer_than_two_usable_days_returns_none():
    assert dvd.pick_days([dvd.DayInfo(dt.date(2026, 10, 8), 5)]) is None
    assert dvd.pick_days([]) is None


def test_pick_days_honours_an_explicit_pair():
    days = [dvd.DayInfo(dt.date(2026, 10, 8), 5), dvd.DayInfo(dt.date(2026, 10, 9), 2),
            dvd.DayInfo(dt.date(2026, 10, 10), 3)]
    a, b = dvd.pick_days(days, explicit=("october_08", "october_10"))
    assert (a.date, b.date) == (dt.date(2026, 10, 8), dt.date(2026, 10, 10))


def test_pick_days_explicit_pair_not_present_raises():
    days = [dvd.DayInfo(dt.date(2026, 10, 8), 5)]
    with pytest.raises(ValueError):
        dvd.pick_days(days, explicit=("october_08", "october_09"))


def test_out_dir_name_and_comparison_text():
    a, b = dvd.DayInfo(dt.date(2026, 10, 8), 12), dvd.DayInfo(dt.date(2026, 10, 10), 7)
    assert dvd.out_dir_name(a, b) == "Oct08-vs-Oct10"
    assert dvd.comparison_text(a, b, hours=18.0) == "Oct 8 vs Oct 10 (both 18 h starved)"


# ── batch mode, multi-unit comparisons, name resolution ────────────────────


@pytest.mark.parametrize("folder, batch", [
    ("october_11_batch_1", 1), ("october_11_batch_2_rig_3", 2), ("october_11_batch_12_rig_2", 12)])
def test_folder_batch(folder, batch):
    assert dvd.folder_batch(folder) == batch


def _wide_batches():
    return pd.DataFrame({
        "dataset": ["D"] * 7,
        "fly": ["october_11_batch_1", "october_11_batch_1_rig_2", "october_11_batch_2",
                "october_11_batch_2_rig_3", "october_11_batch_3", "october_11_batch_3",
                "october_10_batch_1"],
        "fly_number": [1, 1, 1, 2, 1, 2, 1],
    })


def test_usable_units_by_batch_groups_rigs_of_the_same_batch(tmp_path):
    units = dvd.usable_units(_wide_batches(), flagged_flies_csv=tmp_path / "none.csv", by="batch")
    got = {(u.date, u.batch): u.n_flies for u in units}
    assert got == {(dt.date(2026, 10, 10), 1): 1, (dt.date(2026, 10, 11), 1): 2,
                   (dt.date(2026, 10, 11), 2): 2, (dt.date(2026, 10, 11), 3): 2}


def test_batch_unit_prefix_label_and_tag():
    u = dvd.DayInfo(dt.date(2026, 10, 11), 4, batch=2)
    assert u.prefix == "october_11_batch_2"
    assert u.label == "Oct 11 batch 2"
    assert u.tag == "Oct11b2"


def test_pick_units_all_returns_every_unit_in_order_with_no_upper_limit():
    units = [dvd.DayInfo(dt.date(2026, 10, 11), 3, batch=b) for b in (3, 1, 2)]
    picked = dvd.pick_units(units, all_units=True)
    assert [u.batch for u in picked] == [1, 2, 3]
    # IAA-Sensitivity-1-1 had eight batches on 2026-10-15 and grows by ~2 a day.
    many = [dvd.DayInfo(dt.date(2026, 10, d), 2) for d in range(1, 13)]
    assert [u.date.day for u in dvd.pick_units(many, all_units=True)] == list(range(1, 13))


def test_pick_units_explicit_list():
    units = [dvd.DayInfo(dt.date(2026, 10, 11), 3, batch=b) for b in (1, 2, 3)]
    picked = dvd.pick_units(units, explicit=["october_11_batch_3", "october_11_batch_1"])
    assert [u.batch for u in picked] == [1, 3]


def test_out_dir_name_and_comparison_text_for_several_units():
    units = [dvd.DayInfo(dt.date(2026, 10, 11), 3, batch=b) for b in (1, 2, 3)]
    assert dvd.out_dir_name(*units) == "Oct11b1-vs-Oct11b2-vs-Oct11b3"
    # Same-day batches: the date once, short group names (they label every bar pair).
    assert dvd.display_labels(units) == ["Batch 1", "Batch 2", "Batch 3"]
    assert dvd.comparison_text(*units, hours=0.0) == (
        "Oct 11: Batch 1 vs Batch 2 vs Batch 3 (all 0 h starved)")
    assert dvd.comparison_text(*units[:2], hours=None) == "Oct 11: Batch 1 vs Batch 2"


def test_display_labels_across_days_and_for_plain_days():
    mixed = [dvd.DayInfo(dt.date(2026, 10, 10), 3, batch=1),
             dvd.DayInfo(dt.date(2026, 10, 11), 3, batch=2)]
    assert dvd.display_labels(mixed) == ["Oct 10 b1", "Oct 11 b2"]
    days = [dvd.DayInfo(dt.date(2026, 10, 8), 3), dvd.DayInfo(dt.date(2026, 10, 9), 3)]
    assert dvd.display_labels(days) == ["Oct 8", "Oct 9"]


def test_resolve_dataset_is_case_insensitive_and_suggests_on_a_miss():
    known = ["3Oct-Sensitivity-1-0.1", "IAA-Sensitivity-1-1"]
    assert dvd.resolve_dataset("3OCT-Sensitivity-1-0.1", known) == "3Oct-Sensitivity-1-0.1"
    with pytest.raises(SystemExit, match="3Oct-Sensitivity-1-0.1"):
        dvd.resolve_dataset("3OCT-Sensitivity-1-1", known)


# ── comparison-only by default ─────────────────────────────────────────────


def _fake_run(monkeypatch, tmp_path, dataset="IAA-Sensitivity-1-1"):
    calls = {"day": [], "comparison": []}
    wide = pd.DataFrame({"dataset": [dataset] * 4,
                         "fly": ["october_08_batch_2", "october_09_batch_2_rig_2",
                                 "october_12_batch_1", "october_12_batch_1_rig_3"],
                         "fly_number": [1, 1, 1, 2]})
    monkeypatch.setattr(dvd.pd, "read_parquet", lambda *a, **k: wide)
    monkeypatch.setattr(dvd, "FLAGGED", tmp_path / "none.csv")
    monkeypatch.setattr(dvd, "_build_day", lambda out, group, log, pubfig: calls["day"].append(out))
    monkeypatch.setattr(dvd, "_build_comparison",
                        lambda out, units, dataset, **k: calls["comparison"].append(out))
    return calls


def test_run_dataset_builds_only_the_comparison_by_default(monkeypatch, tmp_path):
    calls = _fake_run(monkeypatch, tmp_path)
    out = dvd.run_dataset("IAA-Sensitivity-1-1", out_root=tmp_path, explicit=None, pubfig=True,
                          by="batch", all_units=True, hours=0)
    assert calls["day"] == []                      # no per-batch figure sets
    assert len(calls["comparison"]) == 1
    readme = (out / "README.md").read_text()
    assert "--per-unit" in readme                  # says how to get the per-batch sets


def test_run_dataset_per_unit_builds_every_unit_too(monkeypatch, tmp_path):
    calls = _fake_run(monkeypatch, tmp_path)
    dvd.run_dataset("IAA-Sensitivity-1-1", out_root=tmp_path, explicit=None, pubfig=False,
                    by="batch", all_units=True, hours=0, per_unit=True)
    assert len(calls["day"]) == 3 and len(calls["comparison"]) == 1


def test_starvation_hours_reads_sensitivity_variants_like_5s():
    assert dvd.starvation_hours("IAA-Sensitivity5s-1-1") == (1.0, "1")


@pytest.mark.parametrize("dataset, present, absent", [
    ("IAA-Sensitivity-1-1",
     ["Sensitized odor: ", "blue = pre-exposure, orange = post-exposure"],
     ["Trained odor (CS+)", "blue = pre-test", "post-training"]),
    # Not reachable from the CLI (only *-Sensitivity-* names parse); pins that the
    # README's figure key follows the cohort instead of hard-coding either wording.
    ("Hex-HexTest-Training-32-1",
     ["Trained odor (CS+): ", "blue = pre-test, orange = post-training"],
     ["Sensitized odor", "pre-exposure", "post-exposure"]),
])
def test_readme_figure_key_words_the_panels_and_odor_by_protocol(monkeypatch, tmp_path, dataset,
                                                                present, absent):
    _fake_run(monkeypatch, tmp_path, dataset=dataset)
    monkeypatch.setattr(dvd, "starvation_hours", lambda d: (0.0, "1"))
    out = dvd.run_dataset(dataset, out_root=tmp_path, explicit=None, pubfig=False, by="batch",
                          all_units=True, hours=0)
    readme = (out / "README.md").read_text()
    assert all(s in readme for s in present)
    assert not any(s in readme for s in absent)


# ── odor concentration in the figure text (scripts/analysis/odor_conc_labels.py) ──


def test_odor_concentrations_reads_the_pins_suffix_back_per_display_odor():
    concs = dvd.odor_concentrations([
        "pretest_1_isoamylacetate-1", "testing_2_isoamylacetate-1",
        "testing_3_3octanol-0.1", "testing_4_3-octonol-0.1",     # two pins spellings, one odor
        "testing_5_hexanol-0.1", "testing_6_hexanol",            # one batch recorded none
        "testing_7_citral-1", "testing_8_citral-0.1",            # two vials disagree
        "testing_9_linalool"])
    assert concs == {"Isoamyl Acetate": "1", "3-Octanol": "0.1"}


@pytest.mark.parametrize("odor, cohort, concs, drawn", [
    ("Isoamyl Acetate", "IAA-Sensitivity-1-1", {"Isoamyl Acetate": "1"}, "Isoamyl Acetate 1% (v/v)"),
    ("3-Octanol", "3Oct-Sensitivity-1-0.1", {"3-Octanol": "0.1"}, "3-Octanol 0.1% (v/v)"),
    # none recorded (pre-pins batches): the display name, never a guessed percent
    ("Isoamyl Acetate", "IAA-Sensitivity-40-1", {}, "Isoamyl Acetate"),
    ("Hexanol", "IAA-Sensitivity-24-1", None, "Hexanol"),
    # every other cohort draws the name exactly as loaded
    ("Isoamyl Acetate", "Hex-HexTest-Training-32-1", {"Isoamyl Acetate": "1"}, "Isoamyl Acetate"),
    ("Hexanol", None, {"Hexanol": "0.1"}, "Hexanol"),
])
def test_odor_text_adds_the_concentration_on_sensitivity_figures_only(odor, cohort, concs, drawn):
    assert dvd.odor_text(odor, cohort, concs) == drawn
