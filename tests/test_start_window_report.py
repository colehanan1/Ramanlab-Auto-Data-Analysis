"""Figure set for the flies whose experiment started inside a clock window.

The window is half-open, [start, end), on the first pre-test trial's clock
time. Folders are matched by exact name: a batch prefix like
``october_14_batch_1`` would also pull in ``october_14_batch_1_rig_2``, which
may have started outside the window.
"""

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest

import scripts.analysis.start_window_report as swr

DS = "IAA-Sensitivity-1-1"


def _starts(rows):
    return pd.DataFrame(
        [{"dataset": DS, "folder": f, "first_pretest": dt.datetime(2026, 10, 14, *hms)}
         for f, hms in rows])


@pytest.mark.parametrize("text, expected", [
    ("13:00", dt.time(13, 0)), ("9:05", dt.time(9, 5)), ("1pm", dt.time(13, 0)),
    ("12pm", dt.time(12, 0)), ("12am", dt.time(0, 0)), ("11:30am", dt.time(11, 30)),
])
def test_clock_parsing(text, expected):
    assert swr.parse_clock(text) == expected


@pytest.mark.parametrize("lo, hi, label", [
    ("13:00", "14:00", "1pm-2pm"), ("12:00", "14:00", "12pm-2pm"),
    ("12:00", "15:00", "12pm-3pm"), ("11:30", "13:00", "11:30am-1pm"),
])
def test_window_label(lo, hi, label):
    assert swr.window_label(swr.parse_clock(lo), swr.parse_clock(hi)) == label


def test_window_is_half_open_on_the_first_pretest_clock_time():
    starts = _starts([
        ("a", (11, 59, 59)), ("b", (12, 0, 0)), ("c", (13, 59, 59)), ("d", (14, 0, 0)),
    ])
    picked = swr.folders_in_window(starts, dt.time(12, 0), dt.time(14, 0))
    assert picked["folder"].tolist() == ["b", "c"]


def test_excluded_folders_leave_the_window_by_exact_name():
    starts = _starts([("october_27_batch_1", (13, 0, 0)),
                      ("october_27_batch_1_rig_2", (13, 8, 0)),
                      ("october_27_batch_1_rig_3", (13, 14, 0))])
    window = swr.folders_in_window(starts, dt.time(12, 0), dt.time(15, 0))
    kept = swr.drop_folders(window, ["october_27_batch_1_rig_2"])
    assert kept["folder"].tolist() == ["october_27_batch_1", "october_27_batch_1_rig_3"]
    assert swr.drop_folders(window, [])["folder"].tolist() == window["folder"].tolist()


def test_excluding_a_folder_outside_the_window_is_an_error():
    window = _starts([("october_27_batch_1_rig_3", (13, 14, 0))])
    with pytest.raises(ValueError, match="october_27_batch_1_rig_2"):
        swr.drop_folders(window, ["october_27_batch_1_rig_2"])


def test_findings_name_the_excluded_folders(tmp_path):
    flies = pd.DataFrame([{"folder": "october_27_batch_1_rig_3",
                           "start": dt.datetime(2026, 10, 27, 13, 14),
                           "flies": 3, "flagged_dropped": 0}])
    path = swr.write_findings(tmp_path, DS, "12pm-3pm", flies, {},
                              excluded=["october_27_batch_1_rig_2"])
    text = path.read_text()
    assert "Excluded by hand: october_27_batch_1_rig_2" in text
    assert "Excluded by hand" not in swr.write_findings(tmp_path, DS, "12pm-3pm", flies,
                                                        {}).read_text()


def _wide(rows):
    return pd.DataFrame(
        [{"dataset": ds, "fly": fly, "fly_number": num, "trial_label": label, "x": 1.0}
         for ds, fly, num, label in rows])


def test_subset_matches_exact_folders_drops_flagged_and_normalises_labels(tmp_path):
    flagged = tmp_path / "flagged.csv"
    flagged.write_text('dataset,fly,fly_number,"FLY-State(1, 0, -1)",comment\n'
                       f"{DS},october_14_batch_1,2,0,dead\n")
    df = _wide([
        (DS, "october_14_batch_1", 1, "testing_2_hexanol-0.1"),
        (DS, "october_14_batch_1", 2, "testing_2_hexanol-0.1"),        # flagged
        (DS, "october_14_batch_1_rig_2", 1, "testing_2_hexanol-0.1"),  # other rig
        ("IAA-Sensitivity-12-1", "october_14_batch_1", 1, "testing_2_hexanol"),
    ])
    out = swr.subset_rows(df, DS, ["october_14_batch_1"], flagged)
    assert out[["fly", "fly_number"]].values.tolist() == [["october_14_batch_1", 1]]
    assert out["trial_label"].tolist() == ["testing_2_hexanol"]
    assert out["dataset"].tolist() == [DS]


def test_findings_count_only_per_odor_tests(tmp_path):
    summary = pd.DataFrame({
        "phase": ["Pre-test", "Post-training"] * 3,
        "odor": ["A", "A", "B", "B", "All odors", "All odors"],
        "measure": ["delta_signed"] * 6,
        "p_pre_vs_post": [0.01, 0.01, 0.5, 0.5, float("nan"), float("nan")],
    })
    assert swr.count_significant(summary) == (1, 2)


# ── rate test ─────────────────────────────────────────────────────────────


def test_rate_test_defaults_to_fisher_and_accepts_mcnemar():
    base = ["--dataset", DS, "--start", "13:00", "--end", "14:00", "--out-dir", "/x"]
    assert swr.build_parser().parse_args(base).rate_test == "fisher"
    assert swr.build_parser().parse_args(base + ["--rate-test", "mcnemar"]).rate_test == "mcnemar"
    with pytest.raises(SystemExit):
        swr.build_parser().parse_args(base + ["--rate-test", "chi2"])


@pytest.mark.parametrize("rate_test", ["fisher", "mcnemar"])
def test_the_score_figures_get_the_chosen_rate_test(monkeypatch, tmp_path, rate_test):
    seen = {}
    monkeypatch.setattr(swr, "_run", lambda cmd, log: seen.setdefault(str(cmd[1]), cmd) and 0)
    p = {k: tmp_path / k for k in ("predictions", "testing", "pretest", "training")}
    swr.run_figures(tmp_path, p, DS, config=tmp_path / "c.yaml", label="start x", log=None,
                    rate_test=rate_test)
    cmd = next(c for k, c in seen.items() if k.endswith("pretest_vs_test_comparison.py"))
    assert cmd[cmd.index("--rate-test") + 1] == rate_test


# ── flagged flies: the table given on the command line, in every step ─────


def test_every_figure_step_that_drops_flagged_flies_gets_the_given_table(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(swr, "_run", lambda cmd, log: seen.setdefault(str(cmd[1]), cmd) and 0)
    p = {k: tmp_path / k for k in ("predictions", "testing", "pretest", "training")}
    mine = tmp_path / "my_flagged.csv"
    swr.run_figures(tmp_path, p, DS, config=tmp_path / "c.yaml", label="start x", log=None,
                    flagged=mine)
    takers = {Path(k).name: c for k, c in seen.items() if "--flagged-flies-csv" in c}
    assert {"pretest_vs_test_comparison.py", "dataset_mean_traces_tvc.py"} <= set(takers)
    for cmd in takers.values():
        assert cmd[cmd.index("--flagged-flies-csv") + 1] == mine
    assert all(str(swr.FLAGGED) not in map(str, cmd) for cmd in seen.values())


# ── exit codes: "nothing to plot" is not a crash ──────────────────────────

FOLDER = "october_14_batch_1"


def _window_data(tmp_path, *, flag=()):
    """One DS batch folder started 13:00, flies 1 and 2 in every pipeline table;
    ``flag``: the fly numbers the truth table drops."""
    d = tmp_path / "data" / DS / FOLDER
    d.mkdir(parents=True)
    (d / f"output_{FOLDER}_pretest_1_Hexanol-0.1_20261014_130000.csv").write_text("x\n")
    csv = tmp_path / "csv"
    csv.mkdir()
    rows = pd.DataFrame([{"dataset": DS, "fly": FOLDER, "fly_number": n, "trial_type": phase,
                          "trial_label": f"{phase}_1_hexanol-0.1", "score": 1, "dir_val_0": 0.5}
                         for n in (1, 2) for phase in ("pretest", "testing", "training")])
    for key, name in swr._TABLES:
        rows[rows["trial_type"] == key].to_parquet(csv / name, index=False)
    rows[rows["trial_type"] != "training"].to_csv(csv / "model_predictions.csv", index=False)
    rows[rows["trial_type"] == "training"].to_csv(csv / "model_predictions_training.csv",
                                                  index=False)
    flagged = tmp_path / "flagged.csv"
    flagged.write_text('dataset,fly,fly_number,"FLY-State(1, 0, -1)",comment\n'
                       + "".join(f"{DS},{FOLDER},{n},0,dead\n" for n in flag))
    return flagged


def _main(tmp_path, flagged, start="12:00", end="14:00"):
    return swr.main(["--dataset", DS, "--start", start, "--end", end,
                     "--out-dir", str(tmp_path / "out"), "--data-root", str(tmp_path / "data"),
                     "--csv-dir", str(tmp_path / "csv"), "--flagged-flies-csv", str(flagged)])


def test_nothing_to_plot_has_an_exit_code_of_its_own():
    # 1 is what Python exits with on an uncaught exception, 2 a failed figure step (and
    # an argparse error): the caller drops the previous figure set on this code only
    assert swr.NOTHING_TO_PLOT not in (0, 1, 2)


def test_no_folder_in_the_window_is_nothing_to_plot(tmp_path):
    flagged = _window_data(tmp_path)
    assert _main(tmp_path, flagged, start="15:00", end="16:00") == swr.NOTHING_TO_PLOT


def test_every_fly_flagged_is_nothing_to_plot(tmp_path):
    flagged = _window_data(tmp_path, flag=(1, 2))
    assert _main(tmp_path, flagged) == swr.NOTHING_TO_PLOT
    assert not (tmp_path / "out" / "FINDINGS.md").exists()


def test_main_hands_its_flagged_table_to_the_figure_steps(tmp_path, monkeypatch):
    flagged = _window_data(tmp_path, flag=(2,))
    seen = {}
    monkeypatch.setattr(swr, "run_figures", lambda *a, **k: seen.update(k) or {})
    assert _main(tmp_path, flagged) == 0
    assert Path(seen["flagged"]) == flagged
    assert "| october_14_batch_1 | 2026-10-14 13:00 | 1 | 1 |" in (
        tmp_path / "out" / "FINDINGS.md").read_text()
