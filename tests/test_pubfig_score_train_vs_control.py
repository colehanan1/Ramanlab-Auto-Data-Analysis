"""Publication styling for the training-vs-control mean-score bars.

House rules, shared with the other publication bar figures:
* training bars coloured by odor, control bars gray;
* y axis "Mean PER Score", topping out at the score maximum;
* the cohort n stated once in the legend, not repeated on every bar;
* significance shown as stars only — no p-values, and no bracket at all for a
  comparison that is not significant.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.colors import to_hex

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts.analysis import odor_bar_palette as pal
from scripts.analysis import pubfig_score_train_vs_control as pub


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            # label, trained, train mean/sem/n, ctrl mean/sem/n, p
            ("3-Octanol (0.1%) 1", True, 1.3913, 0.3706, 23, 0.3478, 0.2377, 23, 0.0157),
            ("3-Octanol (0.1%) 2", True, 0.8696, 0.3518, 23, 0.5652, 0.3106, 23, 0.4381),
            ("Isoamyl Acetate (1%)", False, 1.2174, 0.4072, 23, 2.6087, 0.4076, 23, 0.0180),
            ("Linalool (1%)", False, 0.6522, 0.3303, 23, 0.1739, 0.1141, 23, 0.5896),
        ],
        columns=[
            "odor", "is_trained",
            "mean_train", "sem_train", "n_train",
            "mean_ctrl", "sem_ctrl", "n_ctrl", "p_value",
        ],
    )


@pytest.fixture()
def ax():
    fig, ax = plt.subplots()
    pub.plot_train_vs_control(ax, _rows(), title="t")
    yield ax
    plt.close(fig)


def _bar_groups(ax):
    from matplotlib.container import BarContainer

    return [c for c in ax.containers if isinstance(c, BarContainer)]


def _train_bars(ax):
    return list(_bar_groups(ax)[0].patches)


def test_training_bars_are_coloured_by_odor(ax):
    assert [to_hex(b.get_facecolor()) for b in _train_bars(ax)] == [
        pal.OCTANOL_BLUE,
        pal.OCTANOL_BLUE,
        pal.ISOAMYL_YELLOW,
        pal.LINALOOL_PURPLE,
    ]


def test_control_bars_are_gray(ax):
    assert {to_hex(b.get_facecolor()) for b in _bar_groups(ax)[1].patches} == {
        pal.CTRL_COLOR
    }


def test_y_axis_matches_the_other_publication_score_figures(ax):
    assert ax.get_ylabel() == "Mean PER Score"
    assert ax.get_ylim()[1] == 5.0


def test_cohort_n_is_stated_once_in_the_legend(ax):
    assert [t.get_text() for t in ax.get_legend().get_texts()] == [
        "Training (n=23)",
        "Control (n=23)",
    ]


def test_tick_labels_do_not_repeat_the_n(ax):
    assert all("n=" not in t.get_text() for t in ax.get_xticklabels())


def test_only_significant_pairs_get_a_bracket_and_it_shows_stars_only(ax):
    texts = [t.get_text() for t in ax.texts]
    assert "*" in texts
    assert texts.count("*") == 2, "one star per significant pair (p=0.0157, 0.0180)"
    assert not any("ns" == t for t in texts), "non-significant pairs get no bracket"
    assert not any("p" in t and "=" in t for t in texts), "no p-values on the figure"


# ---------------------------------------------------------------------------
# Fly-date filtering (--fly-months)
# ---------------------------------------------------------------------------
# The Hex-*-24-0.01 cohorts were collected in two blocks separated by a gap:
# April/May, then a July/August restart on rebuilt rigs. The month lives only
# in the fly *folder* name, so these cover reading it back and cutting the
# cohorts on it.


def test_fly_month_reads_the_folder_prefix():
    assert pub.fly_month("july_31_batch_1_rig_2") == "july"
    assert pub.fly_month("august_04_batch_1_rig_3") == "august"
    assert pub.fly_month("april_22_batch_3_rig_2") == "april"


def test_fly_month_is_none_when_the_folder_carries_no_date():
    assert pub.fly_month("flagged") is None
    assert pub.fly_month("") is None
    assert pub.fly_month("julyish_01_batch_1") is None, "prefix must be a whole token"


def test_fly_month_ignores_case_and_padding():
    assert pub.fly_month("  July_31_batch_1  ") == "july"


def _dated() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fly": [
                "april_22_batch_1", "may_20_batch_1",
                "july_31_batch_1_rig_2", "august_04_batch_1_rig_3",
                "flagged",
            ],
            "score": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )


def test_filter_by_fly_months_keeps_only_the_named_months():
    kept = pub.filter_by_fly_months(_dated(), ("july", "august"))
    assert list(kept["fly"]) == ["july_31_batch_1_rig_2", "august_04_batch_1_rig_3"]


def test_filter_by_fly_months_drops_undated_flies():
    kept = pub.filter_by_fly_months(_dated(), ("april", "may"))
    assert "flagged" not in list(kept["fly"]), (
        "a fly with no date folder cannot be placed in an era, so it is out"
    )
    assert list(kept["fly"]) == ["april_22_batch_1", "may_20_batch_1"]


def test_filter_by_fly_months_accepts_any_case():
    kept = pub.filter_by_fly_months(_dated(), ("July", "AUGUST"))
    assert len(kept) == 2


def test_filter_by_fly_months_rejects_a_month_that_is_not_a_month():
    with pytest.raises(SystemExit, match="junuary"):
        pub.filter_by_fly_months(_dated(), ("junuary",))


def test_filter_by_fly_months_rejects_an_empty_selection():
    with pytest.raises(SystemExit, match="no flies"):
        pub.filter_by_fly_months(_dated(), ("december",))


def test_filter_by_fly_months_returns_everything_when_no_months_given():
    kept = pub.filter_by_fly_months(_dated(), None)
    assert len(kept) == len(_dated()), "no filter means no rows dropped"


# ---------------------------------------------------------------------------
# End to end: the month filter must change the cohort n, not just the rows
# ---------------------------------------------------------------------------


def _synthetic_predictions(tmp_path: Path) -> Path:
    """Two Hex-24-0.01 cohorts, half April flies and half July flies.

    April flies score 0 in both cohorts; July flies score 4 when trained and 0
    when control. So an unfiltered run dilutes the effect and a July-only run
    shows it at full strength — which is what the filter has to prove it does.
    """
    rows = []
    for dataset, trained in (("Hex-Training-24-0.01", True), ("Hex-Control-24-0.01", False)):
        for fly, month_score in (("april_22_batch_1", 0.0), ("july_31_batch_1_rig_2", 4.0)):
            for fly_number in (1, 2):
                for trial in ("testing_2_acv", "testing_3_benzaldehyde"):
                    rows.append(
                        {
                            "dataset": dataset,
                            "fly": fly,
                            "fly_number": fly_number,
                            "trial_label": trial,
                            "score": month_score if trained else 0.0,
                            "trial_type": "testing",
                            "fly_type": "GR5a-Old",
                        }
                    )
    out = tmp_path / "model_predictions.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def test_rows_from_score_summary_without_months_pools_both_eras(tmp_path):
    rows = pub.rows_from_score_summary(
        _synthetic_predictions(tmp_path), "Hex-Training-24-0.01", config=None
    )
    assert set(rows["n_train"]) == {4} and set(rows["n_ctrl"]) == {4}
    assert rows["mean_train"].tolist() == pytest.approx([2.0, 2.0]), (
        "two April flies at 0 and two July flies at 4 average to 2"
    )


def test_rows_from_score_summary_with_months_keeps_only_that_era(tmp_path):
    rows = pub.rows_from_score_summary(
        _synthetic_predictions(tmp_path), "Hex-Training-24-0.01", config=None,
        fly_months=("july", "august"),
    )
    assert set(rows["n_train"]) == {2} and set(rows["n_ctrl"]) == {2}
    assert rows["mean_train"].tolist() == pytest.approx([4.0, 4.0]), (
        "the April flies must be gone, not merely down-weighted"
    )
    assert rows["mean_ctrl"].tolist() == pytest.approx([0.0, 0.0])


def test_cli_accepts_comma_separated_months(tmp_path):
    out_dir = tmp_path / "figs"
    pub.main(
        [
            "dataset",
            "--train-dataset", "Hex-Training-24-0.01",
            "--predictions-csv", str(_synthetic_predictions(tmp_path)),
            "--figures-dir", str(out_dir),
            "--fly-months", "july,august",
            "--out-stem", "era",
        ]
    )
    written = pd.read_csv(out_dir / "era.csv")
    assert set(written["n_train"]) == {2}


# ---------------------------------------------------------------------------
# Flagged-fly exclusions (--flagged-flies-csv)
# ---------------------------------------------------------------------------
# flagged-flys-truth.csv is an EXCLUSION table: FLY-State != 1 means drop the
# fly. Matching is on the (dataset, fly, fly_number) triple, so a fly folder
# misspelled in the truth CSV silently excludes nothing — which is worse than
# erroring, because the figure still renders and just quietly keeps a fly the
# truth table says is dead. These cover both the exclusion and the miss.


def _truth_csv(tmp_path: Path, rows) -> Path:
    out = tmp_path / "flagged-flys-truth.csv"
    pd.DataFrame(
        [
            {"dataset": ds, "fly": fly, "fly_number": n,
             "FLY-State(1, 0, -1)": state, "comment": note}
            for ds, fly, n, state, note in rows
        ]
    ).to_csv(out, index=False)
    return out


def test_flagged_flies_are_dropped_from_the_cohort(tmp_path):
    preds = _synthetic_predictions(tmp_path)
    truth = _truth_csv(
        tmp_path,
        [("Hex-Training-24-0.01", "july_31_batch_1_rig_2", 1, -1, "dead")],
    )
    rows = pub.rows_from_score_summary(
        preds, "Hex-Training-24-0.01", config=None,
        fly_months=("july", "august"), flagged_flies_csv=str(truth),
    )
    assert set(rows["n_train"]) == {1}, "the dead fly must be gone (2 July flies -> 1)"


def test_a_state_of_one_is_kept(tmp_path):
    """FLY-State 1 means alive — being listed is not the same as being excluded."""
    preds = _synthetic_predictions(tmp_path)
    truth = _truth_csv(
        tmp_path,
        [("Hex-Training-24-0.01", "july_31_batch_1_rig_2", 1, 1, "alive")],
    )
    rows = pub.rows_from_score_summary(
        preds, "Hex-Training-24-0.01", config=None,
        fly_months=("july", "august"), flagged_flies_csv=str(truth),
    )
    assert set(rows["n_train"]) == {2}


def test_exclusions_apply_to_the_control_cohort_too(tmp_path):
    preds = _synthetic_predictions(tmp_path)
    truth = _truth_csv(
        tmp_path,
        [("Hex-Control-24-0.01", "july_31_batch_1_rig_2", 2, 0, "weak")],
    )
    rows = pub.rows_from_score_summary(
        preds, "Hex-Training-24-0.01", config=None,
        fly_months=("july", "august"), flagged_flies_csv=str(truth),
    )
    assert set(rows["n_train"]) == {2} and set(rows["n_ctrl"]) == {1}


def test_unmatched_exclusion_rows_are_reported(tmp_path, capsys):
    """A misspelled fly folder must be surfaced, not silently ignored.

    This is the ``august_4`` / ``august_04`` case in the real truth CSV.
    """
    preds = _synthetic_predictions(tmp_path)
    truth = _truth_csv(
        tmp_path,
        [
            ("Hex-Training-24-0.01", "july_31_batch_1_rig_2", 1, -1, "dead"),
            ("Hex-Training-24-0.01", "july_3_batch_1_rig_2", 2, -1, "typo"),
        ],
    )
    pub.rows_from_score_summary(
        preds, "Hex-Training-24-0.01", config=None,
        fly_months=("july", "august"), flagged_flies_csv=str(truth),
    )
    reported = [
        line for line in capsys.readouterr().out.splitlines()
        if line.startswith("[UNMATCHED]")
    ]
    assert reported, "a truth-CSV row matching no fly must be reported"
    assert any("july_3_batch_1_rig_2" in line for line in reported)
    assert not any("july_31_batch_1_rig_2" in line for line in reported), (
        "the row that did match must not be listed as unmatched"
    )


def test_no_flagged_csv_means_no_exclusions(tmp_path):
    rows = pub.rows_from_score_summary(
        _synthetic_predictions(tmp_path), "Hex-Training-24-0.01", config=None,
        fly_months=("july", "august"),
    )
    assert set(rows["n_train"]) == {2}


def test_cli_accepts_flagged_flies_csv(tmp_path):
    preds = _synthetic_predictions(tmp_path)
    truth = _truth_csv(
        tmp_path,
        [("Hex-Training-24-0.01", "july_31_batch_1_rig_2", 1, -1, "dead")],
    )
    out_dir = tmp_path / "figs"
    pub.main(
        [
            "dataset",
            "--train-dataset", "Hex-Training-24-0.01",
            "--predictions-csv", str(preds),
            "--figures-dir", str(out_dir),
            "--fly-months", "july,august",
            "--flagged-flies-csv", str(truth),
            "--out-stem", "flagged",
        ]
    )
    assert set(pd.read_csv(out_dir / "flagged.csv")["n_train"]) == {1}


# ---------------------------------------------------------------------------
# Hand-picked flies (--fly) and odor_pins labels
# ---------------------------------------------------------------------------
# A figure over a named subset of both cohorts (e.g. the unglued-leg flies),
# given as the DATASET/FOLDER_flyN paths the user copies out of flys_New. These
# cohorts were recorded under ~/odor_pins.txt, so every label carries its vial
# concentration ("testing_8_isoamylacetate-1"), which the shared label code
# does not parse: without handling, the odors render raw and the trained odor's
# two presentations pool into one column.

IAA_TRAIN, IAA_CTRL = "IAA-Odor10s-Training-24-1", "IAA-Odor10s-Control-24-1"
PICKED = [
    f"{IAA_TRAIN}/october_21_batch_1_rig_3_fly1",
    f"{IAA_TRAIN}/october_22_batch_1_rig_2_fly2",
    f"{IAA_CTRL}/october_22_batch_1_fly1",
    f"{IAA_CTRL}/october_24_batch_1_fly1",
]


def _pins_predictions(tmp_path: Path, extra=()) -> Path:
    """Picked training flies score 4 then 2 on IAA, 0 on hexanol; controls 0.

    Every fly NOT in ``PICKED`` scores 5 on everything, so one leaking into the
    figure moves a mean.
    """
    flies = [
        (IAA_TRAIN, "october_21_batch_1_rig_3", 1),
        (IAA_TRAIN, "october_22_batch_1_rig_2", 1),
        (IAA_TRAIN, "october_22_batch_1_rig_2", 2),
        (IAA_CTRL, "october_22_batch_1", 1),
        (IAA_CTRL, "october_24_batch_1", 1),
        (IAA_CTRL, "october_24_batch_1_rig_2", 2),
    ]
    picked = {pub.parse_fly_spec(s) for s in PICKED}
    rows = []
    for dataset, fly, n in flies:
        for trial, trained_score in (
            ("testing_1_isoamylacetate-1", 4.0),
            ("testing_2_hexanol-0.1", 0.0),
            ("testing_8_isoamylacetate-1", 2.0),
            ("testing_9_lightonly", 5.0),
        ):
            if (dataset, fly, n) not in picked:
                score = 5.0
            else:
                score = trained_score if dataset == IAA_TRAIN else 0.0
            rows.append(
                {"dataset": dataset, "fly": fly, "fly_number": n,
                 "trial_label": trial, "score": score,
                 "trial_type": "testing", "fly_type": "GR5a-Old"}
            )
    rows.extend(extra)
    out = tmp_path / "model_predictions.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def _by_odor(rows: pd.DataFrame) -> dict:
    return {r.odor: r for r in rows.itertuples(index=False)}


def test_parse_fly_spec_splits_dataset_folder_and_number():
    assert pub.parse_fly_spec(f"{IAA_CTRL}/october_24_batch_1_rig_2_fly2") == (
        IAA_CTRL, "october_24_batch_1_rig_2", 2,
    )
    assert pub.parse_fly_spec(f" {IAA_CTRL}/october_22_batch_1_fly1 ") == (
        IAA_CTRL, "october_22_batch_1", 1,
    )


@pytest.mark.parametrize(
    "spec",
    [f"{IAA_CTRL}/october_22_batch_1", "october_22_batch_1_fly1", f"{IAA_CTRL}/_fly1"],
)
def test_parse_fly_spec_rejects_a_path_it_cannot_split(spec):
    with pytest.raises(SystemExit):
        pub.parse_fly_spec(spec)


def test_only_the_picked_flies_are_drawn(tmp_path):
    rows = pub.rows_from_score_summary(
        _pins_predictions(tmp_path), IAA_TRAIN, config=None, flies=PICKED,
    )
    assert set(rows["n_train"]) == {2} and set(rows["n_ctrl"]) == {2}
    assert set(rows["mean_ctrl"]) == {0.0}, "an unpicked control fly leaked in"


def test_pins_labels_resolve_to_named_odors_with_their_concentration(tmp_path):
    rows = pub.rows_from_score_summary(
        _pins_predictions(tmp_path), IAA_TRAIN, config=None, flies=PICKED,
    )
    assert sorted(rows["odor"]) == [
        "Hexanol (0.1%)", "Isoamyl Acetate (1%) 1", "Isoamyl Acetate (1%) 2",
    ]


def test_the_trained_odor_keeps_its_two_presentations_apart(tmp_path):
    by_odor = _by_odor(pub.rows_from_score_summary(
        _pins_predictions(tmp_path), IAA_TRAIN, config=None, flies=PICKED,
    ))
    first, second = by_odor["Isoamyl Acetate (1%) 1"], by_odor["Isoamyl Acetate (1%) 2"]
    assert (first.mean_train, second.mean_train) == (4.0, 2.0)
    assert first.is_trained and second.is_trained
    assert not by_odor["Hexanol (0.1%)"].is_trained


def test_percent_responding_draws_the_same_flies(tmp_path):
    by_odor = _by_odor(pub.rows_from_percent_responding(
        _pins_predictions(tmp_path), IAA_TRAIN, config=None, flies=PICKED,
    ))
    assert by_odor["Isoamyl Acetate (1%) 1"].n_train == 2
    assert by_odor["Isoamyl Acetate (1%) 1"].mean_train == 100.0
    assert by_odor["Isoamyl Acetate (1%) 1"].mean_ctrl == 0.0
    assert by_odor["Isoamyl Acetate (1%) 1"].is_trained


def test_a_picked_fly_missing_from_the_csv_is_fatal(tmp_path, capsys):
    typo = f"{IAA_CTRL}/october_23_batch_1_fly1"
    with pytest.raises(SystemExit):
        pub.rows_from_score_summary(
            _pins_predictions(tmp_path), IAA_TRAIN, config=None,
            flies=[*PICKED, typo],
        )
    assert "october_23_batch_1" in capsys.readouterr().out


def test_two_vials_of_one_odor_are_never_merged(tmp_path):
    """The hex-test panel carries hexanol at 0.1 and 1 -- one column would lie."""
    stronger_hex = {
        "dataset": IAA_CTRL, "fly": "october_22_batch_1", "fly_number": 1,
        "trial_label": "testing_3_hexanol-1", "score": 0.0,
        "trial_type": "testing", "fly_type": "GR5a-Old",
    }
    with pytest.raises(SystemExit):
        pub.rows_from_score_summary(
            _pins_predictions(tmp_path, extra=[stronger_hex]), IAA_TRAIN,
            config=None, flies=PICKED,
        )


def test_cli_takes_repeated_fly_flags(tmp_path):
    out_dir = tmp_path / "figs"
    argv = ["dataset", "--train-dataset", IAA_TRAIN,
            "--predictions-csv", str(_pins_predictions(tmp_path)),
            "--figures-dir", str(out_dir), "--out-stem", "picked"]
    for spec in PICKED:
        argv += ["--fly", spec]
    pub.main(argv)
    written = pd.read_csv(out_dir / "picked.csv")
    assert set(written["n_train"]) == {2} and set(written["n_ctrl"]) == {2}


# ---------------------------------------------------------------------------
# SEM drawn upward only (--sem-up)
# ---------------------------------------------------------------------------
# The pre-vs-post publication figures draw each SEM from the bar top up to
# mean + SEM, capped at the top only. Opt-in here, so the figures already
# published from this script regenerate unchanged.


def _vertical_segments(ax):
    return [
        (line.get_xdata()[0], sorted(line.get_ydata()))
        for line in ax.get_lines()
        if len(line.get_xdata()) == 2 and line.get_xdata()[0] == line.get_xdata()[1]
    ]


def test_sem_up_draws_each_error_bar_from_the_bar_top_upward():
    fig, ax = plt.subplots()
    rows = _rows()
    pub.plot_train_vs_control(ax, rows, title="t", sem_up=True)
    try:
        assert all(g.errorbar is None for g in _bar_groups(ax)), (
            "the symmetric matplotlib error bars must not be drawn as well"
        )
        tops = {
            round(b.get_x() + b.get_width() / 2, 6): b.get_height()
            for g in _bar_groups(ax) for b in g.patches
        }
        segments = [(round(x, 6), ys) for x, ys in _vertical_segments(ax)
                    if round(x, 6) in tops]
        assert len(segments) == 2 * len(rows), "one upward SEM per bar"
        sems = dict(zip(
            [round(b.get_x() + b.get_width() / 2, 6) for b in _train_bars(ax)],
            rows["sem_train"],
        ))
        for x, (low, high) in segments:
            assert low == pytest.approx(tops[x]), "starts at the bar top, not below it"
            if x in sems:
                assert high - low == pytest.approx(sems[x])
    finally:
        plt.close(fig)


def test_error_bars_stay_symmetric_by_default(ax):
    assert all(g.errorbar is not None for g in _bar_groups(ax))


def test_cli_sem_up_flag(tmp_path):
    out_dir = tmp_path / "figs"
    argv = ["dataset", "--train-dataset", IAA_TRAIN,
            "--predictions-csv", str(_pins_predictions(tmp_path)),
            "--figures-dir", str(out_dir), "--out-stem", "up", "--sem-up"]
    for spec in PICKED:
        argv += ["--fly", spec]
    pub.main(argv)
    assert (out_dir / "up.png").exists()
