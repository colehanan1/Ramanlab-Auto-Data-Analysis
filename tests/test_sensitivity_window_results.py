"""Sensitiviy-Results(11-15): the sensitivity cohorts for one start-time window,
rebuilt by the pipeline on every run instead of by hand.

Covers the pure pieces of ``scripts/analysis/sensitivity_window_results.py``
(which cohorts can be built, per-fly figure matching, the data tables) and the
``run_workflows`` command builder, which follows the ``pretest_vs_test`` shape:
None when unconfigured.
"""

import datetime as dt
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import scripts.analysis.sensitivity_window_results as swr
from scripts.pipeline.run_workflows import _sensitivity_window_command

CONFIG = Path(__file__).resolve().parent.parent / "config" / "config_new.yaml"
LO, HI = dt.time(10, 50), dt.time(15, 10)


def _batch(root: Path, dataset: str, folder: str, stamp: str) -> None:
    d = root / dataset / folder
    d.mkdir(parents=True)
    (d / f"output_{folder}_pretest_1_Hexanol-0.1_{stamp}.csv").write_text("x\n")


# ── which cohorts can be built ────────────────────────────────────────────


def test_a_cohort_with_no_folder_yet_is_waiting_not_an_error(tmp_path):
    status, folders = swr.dataset_status(tmp_path, "Paraffin-Sensitivity-1-100", LO, HI)
    assert status == swr.NO_DATA and folders == []


def test_only_folders_started_in_the_window_are_used(tmp_path):
    ds = "Hex-Sensitivity-1-0.1"
    _batch(tmp_path, ds, "november_03_batch_1_rig_2", "20261103_130941")
    _batch(tmp_path, ds, "november_03_batch_2_rig_2", "20261103_171500")
    status, folders = swr.dataset_status(tmp_path, ds, LO, HI)
    assert status == swr.OK
    assert folders == ["november_03_batch_1_rig_2"]


def test_a_cohort_with_nothing_in_the_window_says_so(tmp_path):
    ds = "Hex-Sensitivity-1-0.1"
    _batch(tmp_path, ds, "november_03_batch_2_rig_2", "20261103_171500")
    assert swr.dataset_status(tmp_path, ds, LO, HI) == (swr.NONE_IN_WINDOW, [])


def test_a_cohort_on_disk_but_not_scored_yet_is_waiting_not_flagged():
    """2026-11-05: Paraffin's first batch was copied in before any pipeline run had
    scored it, and the step called it 'every fly flagged' and exited 2."""
    preds = pd.DataFrame({"dataset": ["Hex-Sensitivity-1-0.1"], "fly": ["b1"]})
    assert swr.scoring_status(preds, "Paraffin-Sensitivity-1-100",
                              ["november_05_batch_1"]) == swr.NOT_SCORED
    assert swr.scoring_status(preds, "Hex-Sensitivity-1-0.1", ["b1", "b2"]) == swr.OK


# ── per-fly figures ───────────────────────────────────────────────────────


def test_per_fly_figures_match_the_exact_folder_only(tmp_path):
    """october_14_batch_1 must not pull in october_14_batch_1_rig_2's flies."""
    ds = "IAA-Sensitivity-1-1"
    d = tmp_path / "Raw-Testing-PER-Traces" / ds
    d.mkdir(parents=True)
    for name in ("october_14_batch_1_fly1_testing.png",
                 "october_14_batch_1_rig_2_fly1_testing.png",
                 "october_14_batch_1_fly2_testing.png"):
        (d / name).write_text("")
    links = swr.per_fly_sources(tmp_path, ds, [("october_14_batch_1", 1)])
    got = [p.name for p in links["3_Post-exposure-Raw-Traces"]]
    assert got == ["october_14_batch_1_fly1_testing.png"]


def test_relinking_replaces_only_symlinks(tmp_path):
    src = tmp_path / "a.png"
    src.write_text("")
    dst = tmp_path / "links"
    dst.mkdir()
    (dst / "old.png").symlink_to(src)
    (dst / "mine.txt").write_text("user file")
    swr.replace_links(dst, [src])
    assert sorted(p.name for p in dst.iterdir()) == ["a.png", "mine.txt"]
    assert (dst / "a.png").is_symlink()


# ── data tables ───────────────────────────────────────────────────────────


def _paired():
    return pd.DataFrame({
        "fly": ["b1", "b1", "b2", "b2"],
        "fly_number": [1, 1, 1, 1],
        "odor": ["Hexanol", "Citral", "Hexanol", "Citral"],
        "score_pre": [0, 3, 1, 0],
        "score_post": [3, 3, 2, 0],
    })


def _starts():
    return pd.DataFrame({
        "folder": ["b1", "b2"],
        "first_pretest": pd.to_datetime(["2026-11-03 13:09", "2026-11-03 17:30"]),
        "rig": ["rig_2", "rig_2"],
        "hours_starved_at_pretest": [1.04, 1.2],
    })


def test_tidy_marks_window_bin_and_sensitized_odor():
    long = swr.tidy_pairs(_paired(), "Hex-Sensitivity-1-0.1", _starts(), "Hexanol",
                          LO, HI, bin_hours=2)
    b1 = long[long["batch_folder"] == "b1"].iloc[0]
    assert bool(b1["in_window"]) and b1["start_bin"] == "12:00-14:00"
    assert not long.loc[long["batch_folder"] == "b2", "in_window"].any()
    hex_rows = long[long["odor"] == "Hexanol"]
    assert hex_rows["sensitized_odor"].all()
    assert list(hex_rows["score_change"]) == [3, 1]
    assert list(hex_rows["responded_post"]) == [True, True]


def test_summary_pools_flies_without_averaging_averages():
    long = swr.tidy_pairs(_paired(), "Hex-Sensitivity-1-0.1", _starts(), "Hexanol",
                          LO, HI, bin_hours=2)
    s = swr.summarise(long, ["cohort", "odor"]).set_index("odor")
    assert s.loc["Hexanol", "n_flies"] == 2
    assert s.loc["Hexanol", "pct_responding_pre"] == 0.0
    assert s.loc["Hexanol", "pct_responding_post"] == 100.0
    assert s.loc["Hexanol", "mean_score_pre"] == 0.5
    assert s.loc["Citral", "n_responding_pre"] == 1


def test_wide_table_has_a_pre_and_post_column_per_odor():
    long = swr.tidy_pairs(_paired(), "Hex-Sensitivity-1-0.1", _starts(), "Hexanol",
                          LO, HI, bin_hours=2)
    w = swr.wide_table(long)
    assert [c for c in w.columns if c.endswith((" pre", " post"))] == [
        "Citral pre", "Citral post", "Hexanol pre", "Hexanol post"]
    assert len(w) == 2


# ── cross-cohort change figure ────────────────────────────────────────────


def _long_two_cohorts():
    a = swr.tidy_pairs(_paired(), "Hex-Sensitivity-1-0.1", _starts(), "Hexanol",
                       LO, HI, bin_hours=2)
    b = swr.tidy_pairs(_paired().assign(odor=lambda d: d["odor"].replace(
        {"Hexanol": "Isoamyl Acetate"})), "IAA-Sensitivity-1-1", _starts(),
        "Isoamyl Acetate", LO, HI, bin_hours=2)
    return pd.concat([a, b], ignore_index=True)


def test_change_table_pools_flies_and_gives_a_paired_sem():
    t = swr.change_table(_long_two_cohorts()).set_index(["cohort", "odor"])
    hex_ = t.loc[("Hex-Sensitivity-1-0.1", "Hexanol")]
    # responded pre [F, F] -> post [T, T]: +100 points, every fly moved the same way
    assert hex_["n_flies"] == 2 and hex_["pct_change"] == 100.0
    assert hex_["sem_pct_change"] == 0.0
    # score change [3, 1]: mean 2, sem = sd(ddof=1) / sqrt(n) = 1.414 / 1.414
    assert hex_["mean_score_change"] == 2.0
    assert hex_["sem_score_change"] == pytest.approx(1.0)
    assert bool(hex_["sensitized_odor"])
    assert not bool(t.loc[("Hex-Sensitivity-1-0.1", "Citral"), "sensitized_odor"])


def test_odor_order_puts_each_cohorts_sensitized_odor_first():
    t = swr.change_table(_long_two_cohorts())
    order = swr.odor_order(t, ["IAA-Sensitivity-1-1", "Hex-Sensitivity-1-0.1"])
    assert order == ["Isoamyl Acetate", "Hexanol", "Citral"]


def test_cohort_colours_follow_the_configured_list_not_whats_on_disk():
    cohorts = ["IAA-Sensitivity-1-1", "Hex-Sensitivity-1-0.1",
               "3Oct-Sensitivity-1-0.1", "Paraffin-Sensitivity-1-100"]
    full = swr.cohort_colours(cohorts)
    # slots in configured order: a cohort with no data yet (Paraffin) still holds
    # its slot, so the others are never repainted when it lands
    assert [full[c] for c in cohorts] == list(swr.COHORT_PALETTE[:4])
    assert len(set(full.values())) == 4


def test_cross_cohort_plots_the_window_flies_only(tmp_path):
    """User 2026-11-05: the cross-cohort figure is for the 11-3 window groups only."""
    long = _long_two_cohorts()
    written = swr.write_cross_cohort(tmp_path, long, ["IAA-Sensitivity-1-1",
                                                      "Hex-Sensitivity-1-0.1"],
                                     window="10:40-15:20")
    d = tmp_path / "Cross-Cohort"
    assert sorted(p.name for p in d.iterdir()) == ["cross_cohort_change_window.csv",
                                                    "cross_cohort_change_window.png",
                                                    "cross_cohort_pre_post_window.csv",
                                                    "cross_cohort_pre_post_window.png"]
    assert [p.name for p in written] == ["cross_cohort_change_window.png",
                                         "cross_cohort_pre_post_window.png"]
    win = pd.read_csv(d / "cross_cohort_change_window.csv")
    assert set(win["cohort"]) == {"IAA-Sensitivity-1-1", "Hex-Sensitivity-1-0.1"}
    assert (win["n_flies"] == 1).all()  # b2 started 17:30, outside the window


def test_cross_cohort_draws_one_bar_per_cohort_and_odor(tmp_path, monkeypatch):
    """User 2026-11-09: bars instead of the lollipop markers."""
    from matplotlib.colors import to_hex
    from matplotlib.container import BarContainer
    from matplotlib.figure import Figure

    figs = []
    monkeypatch.setattr(Figure, "savefig", lambda self, *a, **k: figs.append(self))
    long = _long_two_cohorts()
    table = swr.change_table(long[long["in_window"]])
    cohorts = ["IAA-Sensitivity-1-1", "Hex-Sensitivity-1-0.1"]
    colours = swr.cohort_colours(cohorts)
    swr.plot_cross_cohort(table, cohorts, colours, "t", tmp_path / "x.png")
    axes = figs[0].axes
    assert len(axes) == 2
    for ax in axes:
        bars = [b for c in ax.containers if isinstance(c, BarContainer) for b in c]
        assert len(bars) == len(table)
        assert {to_hex(b.get_facecolor()) for b in bars} == set(colours.values())
        assert sum(b.get_linewidth() > 0 for b in bars) == int(table["sensitized_odor"].sum())
        assert all(line.get_marker() in ("None", None, "") for line in ax.lines)


def test_cross_cohort_also_shows_the_pre_and_post_levels(tmp_path, monkeypatch):
    """User 2026-11-09: besides the change, the pre- and post-exposure % for every cohort."""
    from matplotlib.container import BarContainer
    from matplotlib.figure import Figure

    figs = []
    real_savefig = Figure.savefig

    def keep(self, *a, **k):
        figs.append(self)
        return real_savefig(self, *a, **k)

    monkeypatch.setattr(Figure, "savefig", keep)
    long = _long_two_cohorts()
    swr.write_cross_cohort(tmp_path, long, ["IAA-Sensitivity-1-1", "Hex-Sensitivity-1-0.1"],
                           window="09:40-16:20")
    levels = pd.read_csv(tmp_path / "Cross-Cohort" / "cross_cohort_pre_post_window.csv")
    win = long[long["in_window"]]
    for r in levels.itertuples():
        g = win[(win["cohort"] == r.cohort) & (win["odor"] == r.odor)]
        assert r.n_flies == len(g)
        assert r.pct_responding_pre == pytest.approx(100 * g["responded_pre"].mean())
        assert r.pct_responding_post == pytest.approx(100 * g["responded_post"].mean())
        assert r.mean_score_post == pytest.approx(g["score_post"].mean())
        assert r.ci95_low_pre <= r.pct_responding_pre <= r.ci95_high_pre
    pre_post = figs[-1]
    assert len(pre_post.axes) == 2
    for ax in pre_post.axes:
        bars = [b for c in ax.containers if isinstance(c, BarContainer) for b in c]
        assert len(bars) == 2 * len(levels)            # a pre and a post bar per cohort x odor


def test_cross_cohort_writes_nothing_when_no_fly_is_in_the_window(tmp_path):
    long = _long_two_cohorts().assign(in_window=False)
    written = swr.write_cross_cohort(tmp_path, long, ["Hex-Sensitivity-1-0.1"],
                                     window="10:40-15:20")
    assert written == []
    assert not any((tmp_path / "Cross-Cohort").glob("*.png"))


def test_the_folder_reports_the_paired_rate_test(tmp_path, monkeypatch):
    """Same test as the pipeline's Pre-vs-Post figures (McNemar, flies paired with
    themselves): the windowed figure set is asked for it, and the README quotes it."""
    calls = []
    monkeypatch.setattr(swr, "_run", lambda cmd, log: calls.append(cmd) or 1)
    args = swr.build_parser().parse_args(
        ["--out-dir", str(tmp_path), "--start", "10:40", "--end", "15:20",
         "--dataset", "Hex-Sensitivity-1-0.1"])
    swr.build_cohort(tmp_path, "Hex-Sensitivity-1-0.1", args, log=None)
    cmd = [str(c) for c in calls[0]]
    assert cmd[cmd.index("--rate-test") + 1] == "mcnemar"

    long = _long_two_cohorts()
    tables = {"window_summary_per_odor.csv": swr.summarise(long, ["cohort", "odor"]),
              "by_2h": swr.summarise(long, ["cohort", "odor", "start_bin"])}
    swr.write_readme(tmp_path, args, {"Hex-Sensitivity-1-0.1": swr.OK}, tables)
    text = (tmp_path / "README.md").read_text()
    assert "p (rate, McNemar)" in text and "p (rate, Fisher)" not in text


# ── pre-exposure baseline across cohorts and rigs ─────────────────────────


def _pre_long(groups):
    """groups: {(cohort, rig): [[pre score per odor A, B], ...one list per fly]}."""
    rows = []
    for (cohort, rig), flies in groups.items():
        for i, scores in enumerate(flies):
            for odor, sc in zip(("Citral", "Hexanol"), scores):
                rows.append({"cohort": cohort, "rig": rig, "fly_id": f"{cohort}_{rig}_{i}",
                             "odor": odor, "score_pre": sc, "responded_pre": sc >= 2,
                             "in_window": True})
    return pd.DataFrame(rows)


def test_wilson_interval_matches_known_values():
    lo, hi = swr.wilson_ci(0, 10)
    assert lo == 0.0 and hi == pytest.approx(27.75, abs=0.01)
    lo, hi = swr.wilson_ci(5, 10)
    assert (lo, hi) == (pytest.approx(23.66, abs=0.01), pytest.approx(76.34, abs=0.01))


def test_baseline_table_pools_trials_for_all_odors_never_a_mean_of_means():
    long = _pre_long({("A", 1): [[3, 0]],              # 1 fly, responds to 1 of 2
                      ("A", 2): [[0, 0], [0, 0], [0, 0]]})  # 3 flies, none
    t = swr.baseline_table(long, "cohort").set_index("odor")
    allr = t.loc[swr.ALL_ODORS]
    # pooled: 1 response in 4 flies x 2 odors = 8 trials -> 12.5 %
    assert allr["n_flies"] == 4 and allr["n_trials"] == 8
    assert allr["pct_responding"] == 12.5
    assert t.loc["Citral", "n_trials"] == 4 and t.loc["Citral", "pct_responding"] == 25.0
    r = swr.baseline_table(long, "rig").set_index(["rig", "odor"])
    assert r.loc[(1, swr.ALL_ODORS), "pct_responding"] == 50.0
    assert r.loc[(2, swr.ALL_ODORS), "pct_responding"] == 0.0


def test_baseline_stats_find_no_difference_between_identical_groups():
    same = [[3, 0], [0, 3], [0, 0], [2, 2]]
    st = swr.baseline_stats(_pre_long({("A", 1): same, ("B", 1): same}), "cohort")
    st = st.set_index("odor")
    assert list(st.index) == [swr.ALL_ODORS, "Citral", "Hexanol"]
    assert st.loc["Citral", "p_rate_ffh"] == pytest.approx(1.0)
    assert st.loc[swr.ALL_ODORS, "p_score_kw"] == pytest.approx(1.0)
    assert st.loc[swr.ALL_ODORS, "n_flies"] == 8   # one value per fly, not per trial


def test_baseline_stats_flag_a_real_difference():
    st = swr.baseline_stats(_pre_long({("A", 1): [[3, 3]] * 12, ("B", 1): [[0, 0]] * 12}),
                            "cohort").set_index("odor")
    assert st.loc["Citral", "p_rate_ffh"] < 0.001


def test_pre_baseline_folder_holds_figure_and_tables(tmp_path):
    long = _pre_long({("A", 1): [[3, 0], [0, 0]], ("B", 2): [[2, 2], [0, 3]]})
    written = swr.write_pre_baseline(tmp_path, long, ["A", "B"], window="10:40-15:20")
    d = tmp_path / "Pre-Exposure-Baseline"
    assert [p.name for p in written] == ["pre_exposure_baseline.png"]
    assert sorted(p.name for p in d.iterdir()) == [
        "pre_exposure_baseline.png", "pre_exposure_by_cohort.csv",
        "pre_exposure_by_cohort_x_rig.csv", "pre_exposure_by_rig.csv",
        "pre_exposure_stats.csv"]
    stats = pd.read_csv(d / "pre_exposure_stats.csv")
    assert set(stats["compared"]) == {"cohort", "rig"}


def test_pre_baseline_skips_when_no_fly_is_in_the_window(tmp_path):
    long = _pre_long({("A", 1): [[3, 0]]}).assign(in_window=False)
    assert swr.write_pre_baseline(tmp_path, long, ["A"], window="10:40-15:20") == []


# ── before vs from a split date ────────────────────────────────────────────


SPLIT = dt.date(2026, 10, 31)


def test_period_label_puts_the_split_day_itself_in_the_later_group():
    assert swr.period_label("2026-10-30 23:59", SPLIT) == "Before Oct 31"
    assert swr.period_label("2026-10-31 00:00", SPLIT) == "Oct 31 onward"
    assert swr.period_label("2026-11-04 13:09", SPLIT) == "Oct 31 onward"


def _dated_long():
    long = _long_two_cohorts()
    # real batch folders belong to one cohort, so make the fly ids unique across them
    return long.assign(start_time=long["batch_folder"].map(
        {"b1": "2026-10-20 13:09", "b2": "2026-11-02 13:30"}), in_window=True,
        fly_id=long["cohort"].str[:3] + "_" + long["fly_id"])


def test_date_split_compares_the_pre_exposure_of_all_flies_and_named_cohorts(tmp_path):
    """User 2026-11-05: just the pre-exposure, all flies before vs from the date, plus IAA."""
    written = swr.write_date_split(tmp_path, _dated_long(),
                                   ["IAA-Sensitivity-1-1", "Hex-Sensitivity-1-0.1"],
                                   split=SPLIT, window="10:40-15:20",
                                   panels=["IAA-Sensitivity-1-1"])
    d = tmp_path / "Before-vs-From-2026-10-31"
    # user 2026-11-05: for IAA, pre vs post within each period as well
    assert [p.name for p in written] == ["pre_exposure_by_period.png",
                                         "pre_vs_post_by_period_IAA-Sensitivity-1-1.png"]
    assert sorted(p.name for p in d.iterdir()) == [
        "pre_exposure_by_period.csv", "pre_exposure_by_period.png",
        "pre_exposure_by_period_stats.csv",
        "pre_vs_post_by_period_IAA-Sensitivity-1-1.csv",
        "pre_vs_post_by_period_IAA-Sensitivity-1-1.png"]
    pp = pd.read_csv(d / "pre_vs_post_by_period_IAA-Sensitivity-1-1.csv")
    iaa = pp.set_index(["period", "odor"]).loc[("Before Oct 31", "Isoamyl Acetate")]
    assert (iaa["pct_responding_pre"], iaa["pct_responding_post"]) == (0.0, 100.0)
    assert "p_rate_mcnemar_paired" in pp.columns
    pre = pd.read_csv(d / "pre_exposure_by_period.csv")
    assert set(pre["period"]) == {"Before Oct 31", "Oct 31 onward"}
    assert set(pre["cohort"]) == {swr.ALL_COHORTS, "IAA-Sensitivity-1-1"}
    pooled = pre[(pre["cohort"] == swr.ALL_COHORTS) & (pre["odor"] == swr.ALL_ODORS)]
    assert pooled["n_flies"].sum() == 4   # b1 + b2, both cohorts, one fly each


def test_a_moved_split_date_leaves_no_stale_folder(tmp_path):
    (tmp_path / "Before-vs-From-2026-10-01").mkdir()
    swr.write_date_split(tmp_path, _dated_long(), ["Hex-Sensitivity-1-0.1"], split=SPLIT,
                         window="10:40-15:20")
    assert [p.name for p in tmp_path.glob("Before-vs-From-*")] == ["Before-vs-From-2026-10-31"]


def test_date_split_skips_without_window_flies(tmp_path):
    long = _dated_long().assign(in_window=False)
    assert swr.write_date_split(tmp_path, long, ["Hex-Sensitivity-1-0.1"], split=SPLIT,
                                window="10:40-15:20") == []


def test_split_date_reaches_the_command_only_when_configured():
    assert "--split-date" not in _sensitivity_window_command(_cfg(), _Settings(),
                                                             python_exec="py")
    cmd = _sensitivity_window_command(
        _cfg(split_date="2026-10-31", split_cohorts=["IAA-Sensitivity-1-1"]), _Settings(),
        python_exec="py")
    assert cmd[cmd.index("--split-date") + 1] == "2026-10-31"
    assert cmd[cmd.index("--split-cohort") + 1] == "IAA-Sensitivity-1-1"


# ── one odor across the whole session: pre, each exposure trial, post ──────


IAA = "IAA-Sensitivity-1-1"
PRE_POST = {"f1": (0, 4), "f2": (2, 3), "f3": (1, 5), "f4": (0, 0)}


def _series_long():
    rows = [{"cohort": IAA, "batch_folder": "b1", "fly_number": i + 1, "fly_id": f"b1_fly{i + 1}",
             "in_window": True, "odor": "Isoamyl Acetate", "sensitized_odor": True,
             "score_pre": pre, "score_post": post}
            for i, (pre, post) in enumerate(PRE_POST.values())]
    other_odor = dict(rows[0], odor="Hexanol", sensitized_odor=False, score_pre=5)
    late_fly = dict(rows[0], fly_number=9, fly_id="b1_fly9", in_window=False)
    return pd.DataFrame(rows + [other_odor, late_fly])


def _series_training(flies=(1, 2, 3, 4, 9), trials=6):
    return pd.DataFrame([{"dataset": IAA, "fly": "b1", "fly_number": f,
                          "trial_label": f"training_{t}_isoamylacetate-1", "score": (f + t) % 4}
                         for f in flies for t in range(1, trials + 1)])


def test_trial_series_is_eight_ordered_trials_per_window_fly():
    wide = swr.trial_series(_series_long(), _series_training(), IAA)
    assert list(wide.columns) == ["Pre-exposure"] + [f"Exposure {i}" for i in range(1, 7)] + [
        "Post-exposure"]
    assert list(wide.index) == ["b1_fly1", "b1_fly2", "b1_fly3", "b1_fly4"]  # no late fly
    assert wide.loc["b1_fly1", "Pre-exposure"] == 0       # not the Hexanol row's 5
    assert wide.loc["b1_fly1", "Post-exposure"] == 4
    assert wide.loc["b1_fly2", "Exposure 3"] == (2 + 3) % 4


def test_a_fly_missing_an_exposure_trial_is_left_out_of_every_bar():
    training = _series_training()
    training = training[~((training["fly_number"] == 2) & training["trial_label"].str.startswith("training_4"))]
    assert "b1_fly2" not in swr.trial_series(_series_long(), training, IAA).index


def test_a_fly_with_a_repeated_exposure_trial_is_left_out():
    # aborted/restarted sessions leave a second copy of a trial index
    training = _series_training()
    training = pd.concat([training, training[(training["fly_number"] == 3)
                                             & training["trial_label"].str.startswith("training_2")]])
    assert "b1_fly3" not in swr.trial_series(_series_long(), training, IAA).index


def test_trial_series_is_empty_without_exposure_scores():
    assert swr.trial_series(_series_long(), _series_training().iloc[0:0], IAA).empty


def test_cochran_q_matches_the_hand_calculation():
    x = np.array([[1, 1, 0], [1, 0, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    q, p = swr.cochran_q(x)
    assert q == pytest.approx(2.0 * 8 / 6)
    assert p == pytest.approx(float(np.exp(-q / 2)))


def test_trial_series_summary_pools_one_trial_per_fly():
    wide = swr.trial_series(_series_long(), _series_training(), IAA)
    s = swr.trial_series_summary(wide).set_index("trial")
    assert s.loc["Pre-exposure", "n_flies"] == 4
    assert s.loc["Pre-exposure", "n_responding"] == 1     # only the score of 2
    assert s.loc["Pre-exposure", "pct_responding"] == pytest.approx(25.0)
    assert s.loc["Post-exposure", "pct_responding"] == pytest.approx(75.0)
    assert s.loc["Post-exposure", "mean_score"] == pytest.approx(3.0)


def test_trial_series_tests_cover_every_pair_paired_and_holm_corrected():
    wide = swr.trial_series(_series_long(), _series_training(), IAA)
    t = swr.trial_series_tests(wide)
    for measure, n in (("pct_responding", 28), ("score", 28)):
        pairs = t[(t["measure"] == measure) & (t["trial_a"] != "all")]
        assert len(pairs) == n
        ok = pairs.dropna(subset=["p"])
        assert (ok["p_holm"] >= ok["p"]).all()
    assert set(t.loc[t["trial_a"] == "all", "test"]) == {"Cochran Q", "Friedman"}
    row = t[(t["measure"] == "pct_responding") & (t["trial_a"] == "Pre-exposure")
            & (t["trial_b"] == "Post-exposure")].iloc[0]
    binary = wide >= swr.THRESHOLD
    assert row["test"] == "McNemar exact"
    expected = swr.pvt.mcnemar_p(binary["Pre-exposure"], binary["Post-exposure"])
    assert row["p"] == pytest.approx(expected, abs=1e-4)


def test_write_trial_series_draws_eight_bars_per_measure(tmp_path, monkeypatch):
    from matplotlib.container import BarContainer
    from matplotlib.figure import Figure

    figs = []
    monkeypatch.setattr(Figure, "savefig", lambda self, *a, **k: figs.append(self))
    written = swr.write_trial_series(tmp_path, _series_long(), _series_training(), [IAA],
                                     window="09:40-16:20")
    d = tmp_path / "Trial-By-Trial"
    assert [p.name for p in written] == [f"{IAA}_sensitized_odor_by_trial.png"]
    assert sorted(p.name for p in d.iterdir()) == [f"{IAA}_by_trial_per_fly.csv",
                                                    f"{IAA}_by_trial_stats.csv",
                                                    f"{IAA}_by_trial_summary.csv"]
    with_bars = [ax for ax in figs[0].axes if any(isinstance(c, BarContainer) for c in ax.containers)]
    assert len(with_bars) == 2
    for ax in with_bars:
        assert sum(len(c) for c in ax.containers if isinstance(c, BarContainer)) == 8


def test_write_trial_series_writes_nothing_without_exposure_scores(tmp_path):
    assert swr.write_trial_series(tmp_path, _series_long(), _series_training().iloc[0:0], [IAA],
                                  window="09:40-16:20") == []
    assert not (tmp_path / "Trial-By-Trial").exists()


# ── only cohorts whose rows or code changed are rebuilt ───────────────────


class _Args:
    bin_hours = 2.0


TOD = "Time-of-Day_2h-bins_whole-dataset"
INPUTS = ("predictions.csv", "training_predictions.csv", "wide_pretest.parquet",
          "wide_testing.parquet", "wide_training.parquet")


def _built_cohort(out, ds, *, without=()):
    """What a build leaves: the cohort set with its _inputs, and its time-of-day set."""
    (out / ds / "_inputs").mkdir(parents=True, exist_ok=True)
    (out / ds / "FINDINGS.md").write_text("x")
    for name in INPUTS:
        if name not in without:
            (out / ds / "_inputs" / name).write_text("rows")
    (out / ds / "Graded-PER-Rasters").mkdir(exist_ok=True)
    (out / ds / "Graded-PER-Rasters" / "raster.png").write_text("png")
    (out / TOD / ds / "bins_2h").mkdir(parents=True, exist_ok=True)
    (out / TOD / ds / "bins_2h" / "time_of_day_emphasis.png").write_text("png")


def _spy_builds(monkeypatch, *, cohort_status=swr.OK, tod=0, without=()):
    calls = []

    def fake_cohort(out, ds, args, log):
        calls.append(ds)
        _built_cohort(out, ds, without=without)
        return cohort_status

    monkeypatch.setattr(swr, "build_cohort", fake_cohort)
    monkeypatch.setattr(swr, "build_time_of_day", lambda *a, **k: tod)
    monkeypatch.setattr(swr, "link_per_fly", lambda *a, **k: None)
    return calls


def _cache(tmp_path):
    from fbpipe.utils.figure_cache import FigureCache
    return FigureCache(tmp_path / swr.CACHE_FILE)


def test_an_unchanged_cohort_is_not_rebuilt(tmp_path, monkeypatch):
    calls = _spy_builds(monkeypatch)
    ds = "IAA-Sensitivity-1-1"
    # (status, reused): main checks a kept set again before trusting it
    assert swr.build_or_reuse(tmp_path, ds, _Args, None, _cache(tmp_path), "f1") == (swr.OK, False)
    assert swr.build_or_reuse(tmp_path, ds, _Args, None, _cache(tmp_path), "f1") == (swr.OK, True)
    assert calls == [ds]


def test_a_cohort_with_new_rows_is_rebuilt_and_the_others_kept(tmp_path, monkeypatch):
    calls = _spy_builds(monkeypatch)
    for ds in ("A", "B"):
        swr.build_or_reuse(tmp_path, ds, _Args, None, _cache(tmp_path), f"{ds}1")
    calls.clear()
    swr.build_or_reuse(tmp_path, "A", _Args, None, _cache(tmp_path), "A2")
    swr.build_or_reuse(tmp_path, "B", _Args, None, _cache(tmp_path), "B1")
    assert calls == ["A"]


@pytest.mark.parametrize("kwargs", [{"cohort_status": "built; some figure steps failed (exit 2)"},
                                    {"tod": 1}])
def test_a_failed_build_is_retried_next_run(tmp_path, monkeypatch, kwargs):
    calls = _spy_builds(monkeypatch, **kwargs)
    for _ in range(2):
        swr.build_or_reuse(tmp_path, "A", _Args, None, _cache(tmp_path), "f1")
    assert calls == ["A", "A"]


@pytest.mark.parametrize("missing", ["A/FINDINGS.md", "Time-of-Day_2h-bins_whole-dataset/A"])
def test_a_missing_output_is_rebuilt(tmp_path, monkeypatch, missing):
    import shutil
    calls = _spy_builds(monkeypatch)
    swr.build_or_reuse(tmp_path, "A", _Args, None, _cache(tmp_path), "f1")
    target = tmp_path / missing
    shutil.rmtree(target) if target.is_dir() else target.unlink()
    swr.build_or_reuse(tmp_path, "A", _Args, None, _cache(tmp_path), "f1")
    assert calls == ["A", "A"]


@pytest.mark.parametrize("path", ["A/Graded-PER-Rasters/raster.png",
                                  f"{TOD}/A/bins_2h/time_of_day_emphasis.png"])
@pytest.mark.parametrize("change", ["deleted", "truncated", "joined by a stray file"])
def test_a_set_changed_since_it_was_built_is_rebuilt(tmp_path, monkeypatch, path, change):
    """Reuse trusts the two folders only as the build left them (paths and sizes)."""
    calls = _spy_builds(monkeypatch)
    swr.build_or_reuse(tmp_path, "A", _Args, None, _cache(tmp_path), "f1")
    target = tmp_path / path
    if change == "deleted":
        target.unlink()
    elif change == "truncated":
        target.write_text("")
    else:
        (target.parent / "stray.png").write_text("png")
    assert swr.build_or_reuse(tmp_path, "A", _Args, None, _cache(tmp_path), "f1") == (swr.OK, False)
    assert calls == ["A", "A"]


def test_per_fly_links_do_not_count_as_a_change(tmp_path, monkeypatch):
    """link_per_fly refreshes them every run from the pipeline's own figures."""
    calls = _spy_builds(monkeypatch)
    swr.build_or_reuse(tmp_path, "A", _Args, None, _cache(tmp_path), "f1")
    links = tmp_path / "A" / "Per-Fly" / "1_Pre-exposure-Raw-Traces"
    links.mkdir(parents=True)
    (links / "b1_fly1_pretest.png").symlink_to(tmp_path / "A" / "FINDINGS.md")
    assert swr.build_or_reuse(tmp_path, "A", _Args, None, _cache(tmp_path), "f1") == (swr.OK, True)
    assert calls == ["A"]


@pytest.mark.parametrize("name", INPUTS)
def test_a_set_without_one_of_its_input_tables_is_rebuilt(tmp_path, monkeypatch, name):
    """The data check, the Per-Fly links and the cross-cohort traces read _inputs back:
    a set lacking one is rebuilt, even one that was built that way."""
    calls = _spy_builds(monkeypatch, without=(name,))
    for _ in range(2):
        swr.build_or_reuse(tmp_path, "A", _Args, None, _cache(tmp_path), "f1")
    assert calls == ["A", "A"]


def test_an_entry_recorded_without_the_folders_is_not_trusted(tmp_path, monkeypatch):
    # what a cache written before the folders were digested holds: the fingerprint alone
    calls = _spy_builds(monkeypatch)
    _built_cohort(tmp_path, "A")
    old = _cache(tmp_path)
    old.record("A", "f1")
    old.save()
    assert swr.build_or_reuse(tmp_path, "A", _Args, None, _cache(tmp_path), "f1") == (swr.OK, False)
    assert calls == ["A"]


def _fingerprint_inputs():
    keys = {"dataset": ["A", "A", "B"], "fly": ["b1", "b2", "b1"], "fly_number": [1, 1, 1],
            "trial_label": ["t", "t", "t"]}
    pred = pd.DataFrame({**keys, "score": [1, 2, 3]})
    wide = pd.DataFrame({**keys, "dir_val_0": [0.1, 0.2, 0.3]})
    starts = pd.DataFrame({"folder": ["b1", "b2"], "first_pretest": ["x", "y"]})
    return pred, wide, starts


def _flagged():
    """The truth table as read_flagged reads it: every cell as text."""
    return pd.DataFrame({"dataset": ["A", "B"], "fly": ["b1", "b1"], "fly_number": ["2", "2"],
                         "FLY-State(1, 0, -1)": ["0", "0"], "comment": ["dead", "dead"]})


def _fp(pred, wide, starts, *, folders=("b1",), code="c", training=None, flagged=None):
    return swr.cohort_fingerprint("A", folders, predictions=pred,
                                  training=pred.iloc[0:0] if training is None else training,
                                  wide=[wide], starts=starts,
                                  flagged=_flagged() if flagged is None else flagged,
                                  settings=code)


def test_fingerprint_ignores_other_cohorts_rows():
    pred, wide, starts = _fingerprint_inputs()
    base = _fp(pred, wide, starts)
    p2, w2 = pred.copy(), wide.copy()
    p2.loc[2, "score"] = 5
    w2.loc[2, "dir_val_0"] = 9.9
    assert _fp(p2, w2, starts) == base


def test_fingerprint_moves_with_a_score_new_window_trace_or_code():
    pred, wide, starts = _fingerprint_inputs()
    base = _fp(pred, wide, starts)
    p2 = pred.copy()
    p2.loc[0, "score"] = 5
    assert _fp(p2, wide, starts) != base
    w2 = wide.copy()
    w2.loc[0, "dir_val_0"] = 9.9
    assert _fp(pred, w2, starts) != base
    assert _fp(pred, wide, starts, folders=("b1", "b2")) != base
    assert _fp(pred, wide, starts, code="other") != base


def test_fingerprint_ignores_a_trace_outside_the_window_folders():
    # those flies are only in the time-of-day set, which reads predictions, not traces
    pred, wide, starts = _fingerprint_inputs()
    base = _fp(pred, wide, starts)
    w2 = wide.copy()
    w2.loc[1, "dir_val_0"] = 9.9
    assert _fp(pred, w2, starts) == base


def test_fingerprint_reads_the_start_table_the_same_whatever_its_dtypes():
    """A pandas upgrade changes the units and dtypes it infers (datetime64[ns] -> [us],
    object -> str), not the start times, so it must not redraw every cohort."""
    pred, wide, _ = _fingerprint_inputs()
    starts = pd.DataFrame({"folder": ["b1", "b2"],
                           "first_pretest": pd.to_datetime(["2026-11-03 13:09:41",
                                                            "2026-11-03 17:30:02"]),
                           "rig": [2, 2], "hours_starved_at_pretest": [1.04, np.nan]})
    base = _fp(pred, wide, starts)
    other = starts.astype({"first_pretest": "datetime64[us]", "folder": "string"}).iloc[::-1]
    assert _fp(pred, wide, other) == base
    moved = starts.assign(first_pretest=starts["first_pretest"] + pd.Timedelta(minutes=1))
    assert _fp(pred, wide, moved) != base


def test_fingerprint_matches_the_dataset_name_as_the_build_does():
    """subset_rows strips the dataset name, so a row it draws must count here too."""
    pred, wide, starts = _fingerprint_inputs()
    pred = pred.assign(dataset=["A ", "A", "B"])
    base = _fp(pred, wide, starts, training=pred)
    rescored = pred.copy()
    rescored.loc[0, "score"] = 5
    assert _fp(rescored, wide, starts, training=pred) != base       # predictions
    assert _fp(pred, wide, starts, training=rescored) != base       # training predictions


def test_fingerprint_ignores_trace_columns_this_cohort_leaves_empty():
    """Another cohort's longer trace widens the shared wide table for every cohort."""
    pred, wide, starts = _fingerprint_inputs()
    base = _fp(pred, wide, starts)
    # row 1 is A's folder b2, outside the window; row 2 is cohort B
    assert _fp(pred, wide.assign(dir_val_1=[np.nan, 0.5, 0.7]), starts) == base
    assert _fp(pred, wide.assign(dir_val_1=[0.4, np.nan, np.nan]), starts) != base


def test_fingerprint_counts_only_this_cohorts_flagged_flies():
    pred, wide, starts = _fingerprint_inputs()
    base = _fp(pred, wide, starts)
    others = _flagged()
    others.loc[1, "comment"] = "escaped"                             # cohort B's fly
    others = pd.concat([others, others.iloc[[1]].assign(fly="b9")], ignore_index=True)
    assert _fp(pred, wide, starts, flagged=others) == base
    assert _fp(pred, wide, starts, flagged=_flagged().iloc[::-1]) == base
    mine = _flagged()
    mine.loc[0, "FLY-State(1, 0, -1)"] = "1"                         # A's fly back in
    assert _fp(pred, wide, starts, flagged=mine) != base
    # the truth table's own parser strips the name too
    spaced = _flagged().assign(dataset=[" A", "B"])
    unflagged = spaced.copy()
    unflagged.loc[0, "FLY-State(1, 0, -1)"] = "1"
    assert _fp(pred, wide, starts, flagged=spaced) != _fp(pred, wide, starts, flagged=unflagged)


def test_read_flagged_keeps_every_cell_as_written(tmp_path):
    """A blank in another cohort's row must not turn this cohort's "1" into "1.0"."""
    path = tmp_path / "flagged.csv"
    path.write_text('dataset,fly,fly_number,"FLY-State(1, 0, -1)",comment\n'
                    "A,b1,1,0,\nB,b1,,-1,dead\n")
    table = swr.read_flagged(path)
    assert table.loc[0, "fly_number"] == "1" and table.loc[0, "comment"] == ""
    assert swr.read_flagged(tmp_path / "missing.csv").empty


# ── what every cohort shares: this code, the config, the scorer and its model ──


def _model(tmp_path, *, ensemble="ensemble_v1"):
    """A config naming a scorer model whose JSON names its guard ensemble, beside it."""
    model = tmp_path / "models" / "model.json"
    (model.parent / ensemble).mkdir(parents=True, exist_ok=True)
    model.write_text(json.dumps({"learner": {"attributes": {
        "fbr_two_guard_ensemble": ensemble, "fbr_auc_eps": "2.85"}}}))
    for i in range(2):
        (model.parent / ensemble / f"member_{i:02d}.json").write_text(f"member {i}")
    (model.parent / ensemble / "manifest.json").write_text("{}")
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"reaction_prediction": {"model_path": str(model)}}))
    return config, model


def _settings(monkeypatch, config, *extra):
    monkeypatch.setattr(swr.fc, "code_digest", lambda *a, **k: "code")
    args = swr.build_parser().parse_args(["--out-dir", "/x", "--start", "10:40", "--end",
                                          "15:20", "--dataset", "A", "--config", str(config),
                                          *extra])
    return swr.settings_digest(args)


def test_settings_move_with_the_model_file(tmp_path, monkeypatch):
    """The config names the model by path: one retrained in place changes no byte of it."""
    config, model = _model(tmp_path)
    base = _settings(monkeypatch, config)
    assert _settings(monkeypatch, config) == base
    model.write_text(model.read_text().replace("2.85", "2.86"))
    assert _settings(monkeypatch, config) != base


@pytest.mark.parametrize("edit", ["changed", "added", "removed"])
def test_settings_move_with_every_file_of_the_guard_ensemble(tmp_path, monkeypatch, edit):
    config, model = _model(tmp_path)
    ensemble = model.parent / "ensemble_v1"
    base = _settings(monkeypatch, config)
    if edit == "changed":
        (ensemble / "member_01.json").write_text("retrained")
    elif edit == "added":
        (ensemble / "member_02.json").write_text("member 2")
    else:
        (ensemble / "manifest.json").unlink()
    assert _settings(monkeypatch, config) != base


def test_settings_ignore_files_beside_the_model_outside_its_ensemble(tmp_path, monkeypatch):
    config, model = _model(tmp_path)
    (model.parent / "ensemble_v0").mkdir()
    base = _settings(monkeypatch, config)
    (model.parent / "ensemble_v0" / "member_00.json").write_text("an older ensemble")
    (model.parent / "notes.txt").write_text("x")
    assert _settings(monkeypatch, config) == base


def test_settings_move_with_the_scorer_source(tmp_path, monkeypatch):
    config, _ = _model(tmp_path)
    monkeypatch.setattr(swr, "scorer_digest", lambda *a, **k: "v1")
    base = _settings(monkeypatch, config)
    monkeypatch.setattr(swr, "scorer_digest", lambda *a, **k: "v2")
    assert _settings(monkeypatch, config) != base


def test_settings_leave_the_flagged_table_to_each_cohort(tmp_path, monkeypatch):
    """A fly flagged in one cohort must not redraw them all (cohort_fingerprint has it)."""
    config, _ = _model(tmp_path)
    flagged = tmp_path / "flagged.csv"
    flagged.write_text("dataset,fly,fly_number,FLY-State\nB,b1,1,0\n")
    base = _settings(monkeypatch, config, "--flagged-flies-csv", str(flagged))
    flagged.write_text("dataset,fly,fly_number,FLY-State\nB,b1,1,0\nB,b2,1,-1\n")
    assert _settings(monkeypatch, config, "--flagged-flies-csv", str(flagged)) == base


def test_scorer_digest_reads_the_package_source_without_importing_it(tmp_path, monkeypatch):
    """An editable install: its code changes without a byte of this repo changing."""
    pkg = tmp_path / "site" / "fake_scorer_pkg"
    (pkg / "sub").mkdir(parents=True)
    (pkg / "__init__.py").write_text("raise ImportError('found, never imported')\n")
    (pkg / "sub" / "model.py").write_text("X = 1\n")
    (pkg / "README.txt").write_text("not code")
    monkeypatch.syspath_prepend(str(tmp_path / "site"))
    base = swr.scorer_digest("fake_scorer_pkg")
    assert base and "fake_scorer_pkg" not in sys.modules
    (pkg / "README.txt").write_text("still not code")
    assert swr.scorer_digest("fake_scorer_pkg") == base
    (pkg / "sub" / "model.py").write_text("X = 2\n")
    assert swr.scorer_digest("fake_scorer_pkg") != base
    assert swr.scorer_digest("no_such_scorer_pkg_here") == ""


def test_the_scorer_digest_finds_the_installed_scorer():
    import importlib.util
    if importlib.util.find_spec("flybehavior_response") is None:
        pytest.skip("flybehavior_response is not installed here")
    assert swr.scorer_digest() != ""


@pytest.mark.parametrize("breakage", ["no config", "no reaction_prediction", "no model file",
                                      "model not JSON", "no ensemble attribute",
                                      "no ensemble folder"])
def test_a_missing_scorer_piece_is_an_empty_part_not_an_error(tmp_path, monkeypatch, breakage):
    config, model = _model(tmp_path)
    if breakage == "no config":
        config.unlink()
    elif breakage == "no reaction_prediction":
        config.write_text(yaml.safe_dump({"analysis": {}}))
    elif breakage == "no model file":
        model.unlink()
    elif breakage == "model not JSON":
        model.write_bytes(b"\x00\xffbinary booster")
    elif breakage == "no ensemble attribute":
        model.write_text(json.dumps({"learner": {"attributes": {}}}))
    else:
        shutil.rmtree(model.parent / "ensemble_v1")
    parts = list(swr.model_digests(config))
    if breakage in ("no config", "no reaction_prediction", "no model file"):
        assert parts == ["", ""]
    else:
        assert parts[0] != "" and parts[1] == ""
    _settings(monkeypatch, config)          # and the step still runs


# ── main: a kept set that disagrees with the data tables is redrawn ───────


def _main_runner(tmp_path, monkeypatch, *, check=lambda *a: None):
    """main for one cohort "A" with one window folder, its builds faked by _spy_builds."""
    csv = tmp_path / "csv"
    csv.mkdir()
    pd.DataFrame({"dataset": ["A"], "fly": ["b1"], "fly_number": [1], "trial_label": ["t"],
                  "score": [1]}).to_csv(csv / "model_predictions.csv", index=False)
    flagged = tmp_path / "flagged.csv"
    flagged.write_text("dataset,fly,fly_number,FLY-State\n")
    out = tmp_path / "out"
    monkeypatch.setattr(swr, "settings_digest", lambda args: "settings")
    monkeypatch.setattr(swr, "dataset_status", lambda *a: (swr.OK, ["b1"]))
    monkeypatch.setattr(swr, "start_table", lambda *a: pd.DataFrame({"folder": ["b1"]}))
    monkeypatch.setattr(swr, "read_wide_for_fingerprints", lambda *a: [])
    monkeypatch.setattr(swr, "cohort_pairs", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(swr, "check_matches_figures", check)
    argv = ["--out-dir", str(out), "--start", "10:40", "--end", "15:20", "--dataset", "A",
            "--csv-dir", str(csv), "--flagged-flies-csv", str(flagged),
            "--data-root", str(tmp_path)]
    return (lambda: swr.main(argv)), out, flagged


def test_a_kept_set_that_disagrees_with_the_data_is_rebuilt_and_checked_again(tmp_path,
                                                                               monkeypatch):
    """Something the fingerprint does not cover changed: redraw instead of stopping the step."""
    checks = []

    def check(long, built, flagged):
        checks.append(built)
        if len(checks) == 2:                   # the kept set, on the second run
            raise RuntimeError("data tables disagree with the figure build")

    calls = _spy_builds(monkeypatch)
    run, out, _ = _main_runner(tmp_path, monkeypatch, check=check)
    assert run() == 0 and calls == ["A"]
    assert run() == 0
    assert calls == ["A", "A"] and len(checks) == 3
    assert run() == 0 and calls == ["A", "A"]   # the redraw was recorded


def test_a_fresh_set_that_disagrees_with_the_data_still_stops_the_step(tmp_path, monkeypatch):
    def check(*a):
        raise RuntimeError("data tables disagree with the figure build")

    calls = _spy_builds(monkeypatch)
    run, *_ = _main_runner(tmp_path, monkeypatch, check=check)
    with pytest.raises(RuntimeError, match="disagree"):
        run()
    assert calls == ["A"]


def test_a_failed_build_keeps_the_last_set_unchecked_and_fails_the_step(tmp_path, monkeypatch):
    """The kept set was drawn from older rows: checking it against today's would stop
    the step over a failure it already reports."""
    checks = []
    calls = _spy_builds(monkeypatch, cohort_status="failed (exit 1, see _build_log.txt)")
    run, out, _ = _main_runner(tmp_path, monkeypatch, check=lambda *a: checks.append(a))
    assert run() == 2
    assert calls == ["A"] and checks == []
    assert (out / "A" / "FINDINGS.md").exists()
    assert "failed (exit 1" in (out / "README.md").read_text()


def test_flagging_a_fly_of_another_cohort_keeps_this_one(tmp_path, monkeypatch):
    calls = _spy_builds(monkeypatch)
    run, _, flagged = _main_runner(tmp_path, monkeypatch)
    run()
    flagged.write_text("dataset,fly,fly_number,FLY-State\nB,b1,1,0\n")
    run()
    assert calls == ["A"]
    flagged.write_text("dataset,fly,fly_number,FLY-State\nB,b1,1,0\nA,b1,1,0\n")
    run()
    assert calls == ["A", "A"]


# ── a crashed report is not "every fly flagged" ───────────────────────────


def _report_args(tmp_path):
    return swr.build_parser().parse_args(["--out-dir", str(tmp_path), "--start", "10:40",
                                          "--end", "15:20", "--dataset", "A"])


def _fake_report(monkeypatch, code, *, findings):
    """start_window_report.py exiting with ``code``, with or without its FINDINGS.md."""
    def run(cmd, log):
        cmd = [str(c) for c in cmd]
        staging = Path(cmd[cmd.index("--out-dir") + 1])
        (staging / "_inputs").mkdir(parents=True, exist_ok=True)   # got that far
        if findings:
            (staging / "FINDINGS.md").write_text("new")
        return code

    monkeypatch.setattr(swr, "_run", run)
    monkeypatch.setattr(swr, "link_per_fly", lambda *a, **k: None)


def _last_good_set(out):
    (out / "A").mkdir(parents=True)
    (out / "A" / "FINDINGS.md").write_text("old")


def test_nothing_to_plot_drops_the_last_set(tmp_path, monkeypatch):
    """No folder in the window, or every fly in it flagged: the old set would show flies
    that no longer qualify."""
    _last_good_set(tmp_path)
    _fake_report(monkeypatch, swr.swr.NOTHING_TO_PLOT, findings=False)
    assert swr.build_cohort(tmp_path, "A", _report_args(tmp_path), log=None) == swr.ALL_FLAGGED
    assert not (tmp_path / "A").exists() and not (tmp_path / ".building-A").exists()


@pytest.mark.parametrize("code", [1, 2, -9, 0])
def test_a_report_that_crashed_keeps_the_last_set_and_fails(tmp_path, monkeypatch, code):
    """Python exits 1 on an uncaught exception, argparse 2, a killed run below 0."""
    _last_good_set(tmp_path)
    _fake_report(monkeypatch, code, findings=False)
    status = swr.build_cohort(tmp_path, "A", _report_args(tmp_path), log=None)
    assert status.startswith(f"failed (exit {code}")
    assert (tmp_path / "A" / "FINDINGS.md").read_text() == "old"
    assert not (tmp_path / ".building-A").exists()


@pytest.mark.parametrize("code, status", [(0, swr.OK),
                                          (2, f"{swr.OK}; some figure steps failed (exit 2)")])
def test_a_report_with_findings_replaces_the_last_set(tmp_path, monkeypatch, code, status):
    _last_good_set(tmp_path)
    _fake_report(monkeypatch, code, findings=True)
    assert swr.build_cohort(tmp_path, "A", _report_args(tmp_path), log=None) == status
    assert (tmp_path / "A" / "FINDINGS.md").read_text() == "new"


def _real_report(tmp_path):
    """build_cohort running start_window_report.py itself, on tmp_path's data only."""
    out = tmp_path / "out"
    _last_good_set(out)
    args = swr.build_parser().parse_args(
        ["--out-dir", str(out), "--start", "10:40", "--end", "15:20", "--dataset", "A",
         "--data-root", str(tmp_path / "data"), "--csv-dir", str(tmp_path / "csv"),
         "--flagged-flies-csv", str(tmp_path / "flagged.csv")])
    with open(tmp_path / "log.txt", "w") as log:
        return out, swr.build_cohort(out, "A", args, log)


def test_a_real_crash_of_the_report_keeps_the_last_set(tmp_path):
    # no data root: an uncaught FileNotFoundError, Python's exit 1
    out, status = _real_report(tmp_path)
    assert status.startswith("failed (exit 1")
    assert (out / "A" / "FINDINGS.md").read_text() == "old"


def test_a_real_report_with_nothing_in_the_window_drops_the_last_set(tmp_path):
    _batch(tmp_path / "data", "A", "november_03_batch_2_rig_2", "20261103_171500")
    out, status = _real_report(tmp_path)
    assert status == swr.ALL_FLAGGED and not (out / "A").exists()


# ── one build at a time; a missing Excel engine costs only the workbook ───


def test_a_second_build_waits_for_the_first(tmp_path):
    import threading
    import time

    first = swr.acquire_build_lock(tmp_path)
    got = []
    waiter = threading.Thread(target=lambda: got.append(swr.acquire_build_lock(tmp_path, timeout_s=20)))
    waiter.start()
    time.sleep(0.4)
    assert not got                       # still waiting for the first build
    first.close()                        # what its process exiting does
    waiter.join(10)
    assert got and got[0] is not None


def test_a_build_gives_up_when_another_holds_the_folder_too_long(tmp_path):
    first = swr.acquire_build_lock(tmp_path)
    assert swr.acquire_build_lock(tmp_path, timeout_s=0) is None
    first.close()


def test_main_touches_nothing_when_it_cannot_get_the_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(swr, "acquire_build_lock", lambda out, **kw: None)
    code = swr.main(["--out-dir", str(tmp_path), "--start", "10:40", "--end", "15:20",
                     "--dataset", "A"])
    assert code == 3
    assert list(tmp_path.iterdir()) == []


def test_a_missing_excel_engine_costs_only_the_workbook(tmp_path, monkeypatch):
    """Nightly runs lacked openpyxl for 5 nights: the crash cost every summary folder."""
    def no_engine(*a, **k):
        raise ModuleNotFoundError("No module named 'openpyxl'")

    monkeypatch.setattr(swr.pd, "ExcelWriter", no_engine)
    cohorts = ["IAA-Sensitivity-1-1", "Hex-Sensitivity-1-0.1"]
    swr.write_data(tmp_path, _long_two_cohorts(), cohorts, 2.0)
    d = tmp_path / "Data"
    assert (d / "window_summary_per_odor.csv").exists()
    assert (d / "README.md").exists()
    assert not (d / "Sensitivity_pre_post_scores.xlsx").exists()


# ── mean traces of every cohort on one axis, pre and post ─────────────────

TRACE_COHORTS = ["IAA-Sensitivity-1-1", "Hex-Sensitivity-1-0.1", "3Oct-Sensitivity-1-0.1"]


def _trace_inputs(out, *, short=False):
    """3 cohorts x 4 flies; post-exposure Hexanol is raised by 20 in the Hex cohort only."""
    rng = np.random.default_rng(0)
    frames = 3600
    cols = [f"dir_val_{i}" for i in range(frames)]
    for cohort in TRACE_COHORTS:
        for phase, table in (("pretest", "wide_pretest"), ("testing", "wide_testing")):
            rows = []
            for fly in range(1, 5):
                labels = [f"{phase}_1_hexanol", f"{phase}_2_citral"]
                if phase == "testing":
                    labels.append("testing_9_lightonly")
                for label in labels:
                    trace = rng.normal(0, 1, frames)
                    if phase == "testing" and "hexanol" in label and cohort.startswith("Hex"):
                        trace[1200:2400] += 20
                    if short and cohort.startswith("IAA") and fly == 1 and "citral" in label:
                        trace[200:] = np.nan
                    rows.append({"dataset": cohort, "fly": "november_01_batch_1", "fly_number": fly,
                                 "trial_label": label, "trial_type": phase,
                                 **dict(zip(cols, trace))})
            inputs = out / cohort / "_inputs"
            inputs.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_parquet(inputs / f"{table}.parquet", index=False)


def test_cohort_traces_compare_every_cohort_per_odor_and_phase(tmp_path):
    _trace_inputs(tmp_path, short=True)
    written = swr.write_cohort_traces(tmp_path, TRACE_COHORTS, window="09:40-16:20")
    d = tmp_path / "Mean-Traces-By-Cohort"
    assert sorted(p.name for p in written) == ["odor_window_means_by_cohort.png",
                                               "post_exposure_mean_traces_by_cohort.png",
                                               "pre_exposure_mean_traces_by_cohort.png"]
    stats = pd.read_csv(d / "cohort_trace_stats.csv")
    assert set(stats["odor"]) == {"Hexanol", "Citral"}          # the light probe is no odor
    assert set(stats["phase"]) == {"pre", "post"}
    one = stats.set_index(["phase", "odor", "cohort"])
    assert one.loc[("pre", "Citral", "IAA-Sensitivity-1-1"), "n_flies"] == 3   # short trace dropped
    assert one.loc[("pre", "Hexanol", "IAA-Sensitivity-1-1"), "n_flies"] == 4
    post_hex = stats[(stats["phase"] == "post") & (stats["odor"] == "Hexanol")]
    pre_hex = stats[(stats["phase"] == "pre") & (stats["odor"] == "Hexanol")]
    assert post_hex["kw_p"].iloc[0] < 0.05 < pre_hex["kw_p"].iloc[0]
    assert (stats["kw_p_holm"] >= stats["kw_p"] - 1e-12).all()
    excluded = pd.read_csv(d / "excluded_traces_tracking_gaps.csv")
    assert len(excluded) == 2                                    # pre and post Citral, IAA fly 1
    assert set(excluded["fly"]) == {"november_01_batch_1_fly1"}
    assert (excluded["baseline_finite_frac"] < 0.9).all()
    # whole batch folders shuffled between cohorts: flies of one session are not independent
    assert stats["kw_p_folder_permutation"].between(0, 1).all()
    assert (stats["kw_p_folder_permutation_holm"] >= stats["kw_p_folder_permutation"] - 1e-12).all()


def test_folder_permutation_respects_sessions():
    """Two cohorts, each one folder: no shuffle of folders can beat the observed split
    better than chance, however different the flies look."""
    values = {"A": {f"b1_fly{i}": 10.0 + i for i in range(5)},
              "B": {f"b2_fly{i}": float(i) for i in range(5)}}
    assert swr.folder_permutation_p(values, n_perm=200) >= 0.4
    split = {"A": {**{f"b{j}_fly{i}": 10.0 + i for j in (1, 3) for i in range(3)}},
             "B": {**{f"b{j}_fly{i}": float(i) for j in (2, 4) for i in range(3)}}}
    assert swr.folder_permutation_p(split, n_perm=500) < 0.4


def test_frame_mean_sem_blanks_thin_frames_and_uses_the_finite_count():
    arr = np.array([[1.0, 2.0, 3.0], [3.0, np.nan, np.nan], [5.0, 4.0, np.nan]])
    mean, sem = swr.frame_mean_sem(arr, min_frac=0.6)
    assert mean[0] == pytest.approx(3.0)
    assert sem[0] == pytest.approx(np.std([1, 3, 5], ddof=1) / np.sqrt(3))
    assert sem[1] == pytest.approx(np.std([2, 4], ddof=1) / np.sqrt(2))   # 2 of 3 flies
    assert np.isnan(mean[2]) and np.isnan(sem[2])                       # 1 of 3 < 60 %


def test_cohort_traces_need_two_cohorts(tmp_path):
    _trace_inputs(tmp_path)
    (tmp_path / "Mean-Traces-By-Cohort").mkdir()
    (tmp_path / "Mean-Traces-By-Cohort" / "stale.png").write_text("x")
    assert swr.write_cohort_traces(tmp_path, TRACE_COHORTS[:1], window="09:40-16:20") == []
    assert not (tmp_path / "Mean-Traces-By-Cohort").exists()


def test_model_digests_find_the_ensemble_beside_the_real_model(tmp_path):
    """The scorer resolves the model path before looking for its ensemble; so must we."""
    import json

    real = tmp_path / "models"
    (real / "ens").mkdir(parents=True)
    model = real / "m.json"
    model.write_text(json.dumps({"learner": {"attributes": {swr.GUARD_ENSEMBLE_ATTR: "ens"}}}))
    (real / "ens" / "member_00.json").write_text("1")
    (tmp_path / "current").mkdir()
    (tmp_path / "current" / "m.json").symlink_to(model)
    cfg = tmp_path / "c.yaml"
    cfg.write_text(f"reaction_prediction:\n  model_path: {tmp_path / 'current' / 'm.json'}\n")
    assert swr.model_digests(cfg)[1] != ""


def test_path_arguments_are_made_absolute_before_children_see_them(tmp_path, monkeypatch):
    """The children run with cwd=REPO_ROOT, so a relative path would name another file."""
    monkeypatch.chdir(tmp_path)
    args = swr.resolve_path_args(swr.build_parser().parse_args(
        ["--out-dir", "out", "--start", "10:40", "--end", "15:20", "--dataset", "A",
         "--flagged-flies-csv", "flags.csv"]))
    assert args.out_dir == tmp_path / "out"
    assert args.flagged_flies_csv == tmp_path / "flags.csv"
    assert all(Path(getattr(args, a)).is_absolute() for a in swr.PATH_ARGS)


# ── run_workflows wiring ──────────────────────────────────────────────────


class _Reaction:
    def __init__(self, output_csv="/csv/model_predictions.csv"):
        self.output_csv = output_csv
        self.python = ""


class _Settings:
    flagged_flies_csv = "/csv/flagged-flys-truth.csv"

    def __init__(self, **kw):
        self.reaction_prediction = _Reaction(**kw)


def _cfg(**over):
    base = {"enabled": True, "out_dir": "/res/Sens", "start": "10:50", "end": "15:10",
            "datasets": ["IAA-Sensitivity-1-1", "Paraffin-Sensitivity-1-100"]}
    base.update(over)
    return {"sensitivity_window_results": base}


def test_no_command_when_unconfigured_or_disabled():
    assert _sensitivity_window_command({}, _Settings(), python_exec="py") is None
    assert _sensitivity_window_command(_cfg(enabled=False), _Settings(),
                                       python_exec="py") is None


@pytest.mark.parametrize("missing", ["out_dir", "datasets"])
def test_no_command_without_out_dir_or_datasets(missing):
    assert _sensitivity_window_command(_cfg(**{missing: None}), _Settings(),
                                       python_exec="py") is None


def test_command_carries_window_datasets_and_inputs():
    cmd = _sensitivity_window_command(_cfg(), _Settings(), python_exec="py")
    assert cmd[0] == "py" and cmd[1].endswith("sensitivity_window_results.py")
    assert cmd[cmd.index("--out-dir") + 1] == "/res/Sens"
    assert cmd[cmd.index("--start") + 1] == "10:50"
    assert cmd[cmd.index("--end") + 1] == "15:10"
    assert [cmd[i + 1] for i, a in enumerate(cmd) if a == "--dataset"] == [
        "IAA-Sensitivity-1-1", "Paraffin-Sensitivity-1-100"]
    assert cmd[cmd.index("--csv-dir") + 1] == "/csv"
    assert cmd[cmd.index("--flagged-flies-csv") + 1] == "/csv/flagged-flys-truth.csv"


def test_trial_series_cohorts_reach_the_command_only_when_configured():
    assert "--trial-series-cohort" not in _sensitivity_window_command(_cfg(), _Settings(),
                                                                      python_exec="py")
    cmd = _sensitivity_window_command(_cfg(trial_series_cohorts=[IAA]), _Settings(),
                                      python_exec="py")
    assert cmd[cmd.index("--trial-series-cohort") + 1] == IAA


# ── config/config_new.yaml ────────────────────────────────────────────────


def test_operative_config_writes_the_11_to_3_set():
    if not CONFIG.exists():
        pytest.skip("config/config_new.yaml not present (gitignored)")
    block = yaml.safe_load(CONFIG.read_text())["analysis"]["sensitivity_window_results"]
    assert block["enabled"] is True
    assert block["out_dir"] == "/home/ramanlab/Documents/cole/Results/Sensitiviy-Results(11-15)"
    assert (block["start"], block["end"]) == ("09:40", "16:20")  # 10am-4pm +/- 20 min
    assert str(block["split_date"]) == "2026-10-31"  # before vs from Oct 31
    assert block["split_cohorts"] == ["IAA-Sensitivity-1-1"]
    assert block["trial_series_cohorts"] == ["IAA-Sensitivity-1-1"]
    assert {"IAA-Sensitivity-1-1", "Hex-Sensitivity-1-0.1",
            "3Oct-Sensitivity-1-0.1"} <= set(block["datasets"])


def test_the_pipeline_rebuilds_the_folder_on_every_run():
    # every folder in the results directory comes from this one step, so a cache
    # gate on it would leave all of them stale between data changes
    src = (Path(__file__).parents[1] / "scripts/pipeline/run_workflows.py").read_text()
    assert '"sensitivity_window_results", "analysis"' not in src
    assert "_run_sensitivity_window_results(analysis_cfg, settings)" in src


# ── cross-cohort: change in the peak-z during/before ratio ───────────────


def _peakz_csv(out, cohort, rows):
    """A cohort's PeakZ-Before-vs-During/peakz_per_fly_change.csv; ``rows`` are
    (fly, fly number, odor, signed pre, signed post, unsigned pre, unsigned post).
    ``trained`` is left blank, as auc_before_during leaves it on a fly missing a phase."""
    d = out / cohort / "PeakZ-Before-vs-During"
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "dataset": cohort, "fly": f, "fly_number": n, "odor": o, "trained": "",
        "zpeak_ratio_signed_pre": sp, "zpeak_ratio_signed_post": sq,
        "zpeak_ratio_signed_post_minus_pre": sq - sp,
        "zpeak_ratio_unsigned_pre": up, "zpeak_ratio_unsigned_post": uq,
        "zpeak_ratio_unsigned_post_minus_pre": uq - up,
    } for f, n, o, sp, sq, up, uq in rows]).to_csv(d / "peakz_per_fly_change.csv", index=False)


def test_peakz_change_takes_each_flys_own_post_minus_pre(tmp_path):
    """User 2026-11-11: the signed pre-to-post change of the peak-z during/before ratio,
    every cohort side by side per odor."""
    from scripts.analysis.auc_before_during import _wilcoxon

    _peakz_csv(tmp_path, "Hex-Sensitivity-1-0.1", [
        ("b1", 1, "Hexanol", -1.0, 2.0, 1.0, 2.0),
        ("b1", 2, "Hexanol", 0.5, 1.5, 0.5, 1.5),
        ("b1", 3, "Hexanol", 0.2, np.nan, 0.2, np.nan),       # no post-exposure trial
        ("b1", 1, "Citral", 1.0, 0.0, 1.0, 1.0),
        ("b1", 2, "Citral", 2.0, 1.0, 2.0, 3.0),
    ])
    t = swr.peakz_change_table(tmp_path, ["IAA-Sensitivity-1-1", "Hex-Sensitivity-1-0.1"])
    assert set(t["cohort"]) == {"Hex-Sensitivity-1-0.1"}     # IAA: no PeakZ table, left out
    by = t.set_index(["cohort", "odor"])
    hexanol, citral = by.loc[("Hex-Sensitivity-1-0.1", "Hexanol")], by.loc[("Hex-Sensitivity-1-0.1", "Citral")]
    assert hexanol["n_flies"] == 2 and citral["n_flies"] == 2
    assert hexanol["signed_ratio_change"] == pytest.approx(2.0)       # +3 and +1
    assert hexanol["sem_signed_ratio_change"] == pytest.approx(1.0)   # sd sqrt(2) / sqrt(2)
    assert hexanol["unsigned_ratio_change"] == pytest.approx(1.0)
    assert citral["signed_ratio_change"] == pytest.approx(-1.0)
    assert citral["sem_signed_ratio_change"] == pytest.approx(0.0)
    assert citral["unsigned_ratio_change"] == pytest.approx(0.5)
    # the sensitized odor is the cohort's own, not the blank "trained" column
    assert bool(hexanol["sensitized_odor"]) and not bool(citral["sensitized_odor"])
    # the p each cohort's own PeakZ figure prints
    assert hexanol["p_signed_ratio_wilcoxon"] == pytest.approx(
        _wilcoxon(np.array([2.0, 1.5]), np.array([-1.0, 0.5])), nan_ok=True)


def test_cross_cohort_draws_the_peakz_ratio_change(tmp_path, monkeypatch):
    from matplotlib.container import BarContainer
    from matplotlib.figure import Figure

    figs = []
    real_savefig = Figure.savefig

    def keep(self, *a, **k):
        figs.append(self)
        return real_savefig(self, *a, **k)

    monkeypatch.setattr(Figure, "savefig", keep)
    for cohort, sensitized in (("Hex-Sensitivity-1-0.1", "Hexanol"),
                               ("IAA-Sensitivity-1-1", "Isoamyl Acetate")):
        _peakz_csv(tmp_path, cohort, [(f"b{i}", 1, o, 0.1 * i, 0.4 * i, 0.2 * i, 0.1 * i)
                                      for i in (1, 2, 3) for o in (sensitized, "Citral")])
    written = swr.write_cross_cohort(tmp_path, _long_two_cohorts(),
                                     ["IAA-Sensitivity-1-1", "Hex-Sensitivity-1-0.1"],
                                     window="09:40-16:20")
    name = "cross_cohort_peakz_ratio_change_window"
    table = pd.read_csv(tmp_path / "Cross-Cohort" / f"{name}.csv")
    assert len(table) == 4 and (table["n_flies"] == 3).all()
    assert written[-1] == tmp_path / "Cross-Cohort" / f"{name}.png" and written[-1].is_file()
    fig = figs[-1]
    assert len(fig.axes) == 2                                 # signed and unsigned ratio
    for ax in fig.axes:
        bars = [b for c in ax.containers if isinstance(c, BarContainer) for b in c]
        assert len(bars) == len(table)
        assert sum(b.get_linewidth() > 0 for b in bars) == 2  # each cohort's sensitized odor


def test_cross_cohort_skips_the_peakz_figure_without_peakz_tables(tmp_path):
    written = swr.write_cross_cohort(tmp_path, _long_two_cohorts(),
                                     ["IAA-Sensitivity-1-1", "Hex-Sensitivity-1-0.1"],
                                     window="09:40-16:20")
    assert not (tmp_path / "Cross-Cohort" / "cross_cohort_peakz_ratio_change_window.png").exists()
    assert "cross_cohort_peakz_ratio_change_window.png" not in [p.name for p in written]


# ── cross-cohort: isoamyl acetate, first vs last 15 s of the odor ────────


#: Every per-half measure in a cohort's cumulative_responders_flies.csv.
HALF_MEASURES = ("auc", "auc_ratio", "auc_unsigned_ratio", "zpeak", "zpeak_ratio",
                 "zpeak_unsigned_ratio")


def _halves_csv(out, cohort, rows):
    """A cohort's cumulative_responders_flies.csv, isoamyl acetate rows only;
    ``rows`` are (fly, "pre"/"post", score, onset s, first-half value, second-half
    value), the two values written for every per-half measure."""
    import scripts.analysis.cumulative_responders as cr

    labels = cr.phase_labels(cohort)
    d = out / cohort / cr.FOLDER
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"fly": f, "fly_number": 1, "odor": "Isoamyl Acetate", "phase": labels[ph],
                   "trial": 1, "score": s, "responding": s >= 2, "onset_s": o,
                   "odor_on_s": 30.0, "odor_off_s": 60.0, "theta": 15.0,
                   **{f"{m}_{half}_half": v for m in HALF_MEASURES
                      for half, v in (("first", a1), ("second", a2))}}
                  for f, ph, s, o, a1, a2 in rows]).to_csv(d / f"{cr.TAG}_flies.csv", index=False)


def _two_halves_cohorts(out):
    nan = np.nan
    _halves_csv(out, "IAA-Sensitivity-1-1", [
        ("a", "pre", 3, 5.0, 40.0, 0.0), ("a", "post", 3, 20.0, 5.0, 60.0),
        ("b", "pre", 0, nan, 0.0, 0.0), ("b", "post", 2, 18.0, 0.0, 30.0),
    ])
    _halves_csv(out, "Hex-Sensitivity-1-0.1", [
        ("c", "pre", 2, 3.0, 20.0, 10.0), ("c", "post", 2, nan, 1.0, 1.0),  # no clear onset
        ("d", "pre", 0, 25.0, 0.0, 9.0), ("d", "post", 0, nan, 0.0, 0.0),   # late, not a responder
        ("e", "pre", 0, nan, 0.0, 0.0), ("e", "post", 1, nan, 0.0, 0.0),
    ])


def test_isoamyl_acetate_halves_count_when_each_responder_started(tmp_path):
    """User 2026-11-11: did responses move to the end of the odor after exposure?"""
    _two_halves_cohorts(tmp_path)
    cohorts = ["IAA-Sensitivity-1-1", "Hex-Sensitivity-1-0.1", "3Oct-Sensitivity-1-0.1"]
    flies = swr.odor_halves_flies(tmp_path, cohorts)
    assert set(flies["cohort"]) == set(cohorts[:2])          # 3Oct: no table, left out
    t = swr.odor_halves_table(flies, cohorts).set_index("group")
    iaa = t.loc["IAA-Sensitivity-1-1"]
    assert (iaa["n_flies"], iaa["n_first_pre"], iaa["n_second_pre"],
            iaa["n_first_post"], iaa["n_second_post"]) == (2, 1, 0, 0, 2)
    assert iaa["pct_second_post"] == 100.0
    # user 2026-11-11: AUC as during / before; peak z as during / before and alone
    for measure in ("auc_ratio", "zpeak_ratio", "zpeak"):
        assert iaa[f"{measure}_second_post"] == pytest.approx(45.0)   # responders a and b
        assert iaa[f"{measure}_first_pre"] == pytest.approx(40.0)     # responder a only
    hex_ = t.loc["Hex-Sensitivity-1-0.1"]
    assert (hex_["n_first_pre"], hex_["n_second_pre"], hex_["n_no_onset_post"]) == (1, 0, 1)
    every = t.loc[swr.ALL_FLIES]
    assert every["n_flies"] == 5
    # pooled: 2 late starters of 5 flies, not the mean of 100 % and 0 %
    assert every["pct_second_post"] == pytest.approx(40.0)


def test_cross_cohort_draws_the_isoamyl_acetate_halves(tmp_path, monkeypatch):
    from matplotlib.container import BarContainer
    from matplotlib.figure import Figure

    figs = []
    real_savefig = Figure.savefig

    def keep(self, *a, **k):
        figs.append(self)
        return real_savefig(self, *a, **k)

    monkeypatch.setattr(Figure, "savefig", keep)
    _two_halves_cohorts(tmp_path)
    written = swr.write_cross_cohort(tmp_path, _long_two_cohorts(),
                                     ["IAA-Sensitivity-1-1", "Hex-Sensitivity-1-0.1"],
                                     window="09:40-16:20")
    name = "cross_cohort_isoamyl_acetate_halves_window"
    assert (tmp_path / "Cross-Cohort" / f"{name}.csv").is_file()
    assert tmp_path / "Cross-Cohort" / f"{name}.png" in written
    fig = next(f for f in figs if f._suptitle and "15 s" in f._suptitle.get_text())
    # responders; AUC during / before; peak z during / before; peak z during
    assert len(fig.axes) == 4
    for ax in fig.axes:
        bars = [b for c in ax.containers if isinstance(c, BarContainer) for b in c]
        assert len(bars) == 3 * 2 * 2                 # (2 cohorts + all flies) x halves x phases


def test_cross_cohort_skips_the_halves_without_responder_tables(tmp_path):
    swr.write_cross_cohort(tmp_path, _long_two_cohorts(), ["Hex-Sensitivity-1-0.1"],
                           window="09:40-16:20")
    assert not list((tmp_path / "Cross-Cohort").glob("*halves*"))


def _peakz_during_csv(out, cohort, rows):
    """A cohort's peakz_per_fly_change.csv with the during-odor peak z as well;
    ``rows`` are (fly, odor, signed pre, signed post); the ratios are left blank."""
    d = out / cohort / "PeakZ-Before-vs-During"
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "dataset": cohort, "fly": f, "fly_number": 1, "odor": o, "trained": "",
        "zpeak_during_signed_pre": sp, "zpeak_during_signed_post": sq,
        "zpeak_during_unsigned_pre": abs(sp), "zpeak_during_unsigned_post": abs(sq),
        **{f"zpeak_ratio_{k}_{ph}": np.nan for k in ("signed", "unsigned")
           for ph in ("pre", "post")},
    } for f, o, sp, sq in rows]).to_csv(d / "peakz_per_fly_change.csv", index=False)


def test_peakz_during_change_takes_the_odor_windows_own_peak(tmp_path):
    """User 2026-11-11: the cross-cohort peak-z change, during the odor alone."""
    _peakz_during_csv(tmp_path, "Hex-Sensitivity-1-0.1", [
        ("b1", "Hexanol", 2.0, 8.0), ("b2", "Hexanol", -3.0, 1.0), ("b3", "Citral", 1.0, 1.0),
    ])
    t = swr.peakz_change_table(tmp_path, ["Hex-Sensitivity-1-0.1"],
                               measures=swr.PEAKZ_DURING).set_index("odor")
    assert t.loc["Hexanol", "n_flies"] == 2
    assert t.loc["Hexanol", "signed_during_change"] == pytest.approx(5.0)      # +6 and +4
    assert t.loc["Hexanol", "unsigned_during_change"] == pytest.approx(2.0)    # +6 and -2
    # the ratio table leaves out flies whose ratios are blank, the during one does not
    assert swr.peakz_change_table(tmp_path, ["Hex-Sensitivity-1-0.1"]).empty


def test_cross_cohort_draws_the_peakz_change_during_the_odor_alone(tmp_path):
    for cohort, odor in (("Hex-Sensitivity-1-0.1", "Hexanol"),
                         ("IAA-Sensitivity-1-1", "Isoamyl Acetate")):
        _peakz_during_csv(tmp_path, cohort, [(f"b{i}", odor, 1.0 * i, 2.0 * i)
                                             for i in (1, 2, 3)])
    written = swr.write_cross_cohort(tmp_path, _long_two_cohorts(),
                                     ["IAA-Sensitivity-1-1", "Hex-Sensitivity-1-0.1"],
                                     window="09:40-16:20")
    name = "cross_cohort_peakz_during_change_window"
    assert (tmp_path / "Cross-Cohort" / f"{name}.csv").is_file()
    assert tmp_path / "Cross-Cohort" / f"{name}.png" in written
