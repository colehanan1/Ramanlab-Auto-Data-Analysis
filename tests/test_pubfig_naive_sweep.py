"""Naive vs trained vs control, swept over every dataset, odor and presentation.

``pubfig_naive_vs_trained`` shipped with two hand-written comparisons (3-octanol
at 0.1%, ethyl butyrate at 1% batch 1). The same figure is wanted for every
odor in every trained/control cohort, and for the second presentation of the
trained odor as well as the first.

The naive arm has to come from the RandomPanel panel delivered at the *same*
concentration -- 1% odors against ``RandomPanel-24-1``, 0.1% against
``RandomPanel-24-0.1`` -- so an odor whose label carries no concentration, or
whose concentration has no naive panel, is skipped and reported rather than
silently compared against the wrong dose.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analysis.pubfig_naive_vs_trained import (  # noqa: E402
    Comparison,
    cohort_scores,
    naive_dataset_for,
    parse_concentration,
    render_comparison,
    sweep_comparisons,
    training_control_pairs,
)

NAIVE_BY_CONC = {
    10.0: "RandomPanel-Training-24-10",
    1.0: "RandomPanel-24-1",
    0.1: "RandomPanel-24-0.1",
}


def _scores(rows):
    """rows: (dataset_canon, fly, odor_display, occurrence, score)."""
    return pd.DataFrame(
        [
            {
                "dataset_canon": ds,
                "fly": fly,
                "fly_number": 1,
                "fly_type": "GR5a-Old",
                "odor_display": odor,
                "occurrence": occ,
                "score": score,
            }
            for ds, fly, odor, occ, score in rows
        ]
    )


def _panel_rows(dataset, odors, flies=("july_20_batch_1", "july_20_batch_2")):
    return [
        (dataset, fly, odor, occ, 2.0)
        for fly in flies
        for odor, occurrences in odors
        for occ in occurrences
    ]


def _frame():
    """3-octanol cohort: the trained odor is presented twice, hexanol once."""
    rows = []
    for dataset in ("3OCT-Training-24-0.1", "3OCT-Control-24-0.1"):
        rows += _panel_rows(
            dataset,
            [("3-Octanol (0.1%)", (1, 2)), ("Hexanol (0.1%)", (1,)),
             ("Citral (1%)", (1,))],
        )
    rows += _panel_rows(
        "RandomPanel-24-0.1", [("3-Octanol", (1, 2)), ("Hexanol", (1, 2))]
    )
    rows += _panel_rows("RandomPanel-24-1", [("Citral", (1, 2))])
    return _scores(rows)


# ---------------------------------------------------------------------------
# Concentration parsing and naive matching
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label,expected",
    [
        ("Ethyl Butyrate (1%)", 1.0),
        ("Hexanol (0.1%)", 0.1),
        ("Hexanol (0.01%)", 0.01),
        ("Sour Dough Yeast (25%)", 25.0),
        ("3-Octanol (0.1%) 2", 0.1),
        ("Apple Cider Vinegar", None),
        ("Hexanol ()", None),
    ],
)
def test_concentration_is_read_off_the_label(label, expected):
    assert parse_concentration(label) == expected


def test_naive_panel_is_chosen_by_concentration():
    assert naive_dataset_for(1.0, NAIVE_BY_CONC) == "RandomPanel-24-1"
    assert naive_dataset_for(0.1, NAIVE_BY_CONC) == "RandomPanel-24-0.1"


def test_a_concentration_with_no_naive_panel_has_no_match():
    """25% sourdough was never in a random panel; guessing a dose would be a
    fabricated comparison."""
    assert naive_dataset_for(25.0, NAIVE_BY_CONC) is None


def test_only_the_three_sanctioned_panels_are_naive_arms():
    """The naive comparison is restricted to RandomPanel-24-0.1 / -24-1 /
    -Training-24-10; the 0.01% panels are not a baseline, so a 0.01% odor
    gets no figure rather than one against a neighbouring dose."""
    from scripts.analysis.pubfig_naive_vs_trained import NAIVE_BY_CONC as shipped

    assert set(shipped.values()) == {
        "RandomPanel-24-0.1",
        "RandomPanel-24-1",
        "RandomPanel-Training-24-10",
    }
    assert naive_dataset_for(0.01, shipped) is None


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------


def test_pairs_come_from_the_datasets_actually_scored():
    pairs = training_control_pairs(_frame())
    assert pairs == [("3OCT-Training-24-0.1", "3OCT-Control-24-0.1", "3OCT-24-0.1")]


def test_a_training_dataset_without_a_control_is_not_paired():
    df = _scores(_panel_rows("EB-Training-24-1", [("Hexanol (0.1%)", (1,))]))
    assert training_control_pairs(df) == []


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------


def _sweep(df=None):
    return sweep_comparisons(df if df is not None else _frame(), naive_by_conc=NAIVE_BY_CONC)


def test_one_comparison_per_odor_and_presentation():
    built, _ = _sweep()
    assert [(c.odor, c.presentation) for c in built] == [
        ("3-Octanol", 1),
        ("3-Octanol", 2),
        ("Citral", 1),
        ("Hexanol", 1),
    ]


def test_each_comparison_names_its_three_datasets():
    built, _ = _sweep()
    first = built[0]
    assert first.train_dataset == "3OCT-Training-24-0.1"
    assert first.control_dataset == "3OCT-Control-24-0.1"
    assert first.naive_dataset == "RandomPanel-24-0.1"
    # Citral is a 1% odor inside a 0.1% cohort: its naive arm is the 1% panel.
    citral = [c for c in built if c.odor == "Citral"][0]
    assert citral.naive_dataset == "RandomPanel-24-1"


def test_repeated_odors_are_marked_so_the_two_figures_differ():
    built, _ = _sweep()
    oct1, oct2 = [c for c in built if c.odor == "3-Octanol"]
    assert oct1.stem != oct2.stem
    assert oct2.stem.endswith("_p2")
    assert "presentation 2" in oct2.title
    # A once-presented odor keeps the plain name.
    hexanol = [c for c in built if c.odor == "Hexanol"][0]
    assert not hexanol.stem.endswith("_p1")
    assert "presentation" not in hexanol.title


def test_an_untagged_odor_is_skipped_with_a_reason():
    df = _frame()
    df = pd.concat(
        [
            df,
            _scores(_panel_rows("3OCT-Training-24-0.1", [("Apple Cider Vinegar", (1,))])),
            _scores(_panel_rows("3OCT-Control-24-0.1", [("Apple Cider Vinegar", (1,))])),
        ],
        ignore_index=True,
    )
    built, skipped = _sweep(df)
    assert "Apple Cider Vinegar" not in [c.odor for c in built]
    reasons = {s["odor"]: s["reason"] for s in skipped}
    assert "no concentration" in reasons["Apple Cider Vinegar"]


def test_an_odor_absent_from_the_naive_panel_is_skipped():
    """Linalool never appears in the random panels -- the naive bar would be
    empty and the figure would imply a zero response."""
    df = pd.concat(
        [
            _frame(),
            _scores(_panel_rows("3OCT-Training-24-0.1", [("Linalool (1%)", (1,))])),
            _scores(_panel_rows("3OCT-Control-24-0.1", [("Linalool (1%)", (1,))])),
        ],
        ignore_index=True,
    )
    built, skipped = _sweep(df)
    assert "Linalool" not in [c.odor for c in built]
    assert any("naive" in s["reason"] for s in skipped if s["odor"] == "Linalool")


def test_a_presentation_the_control_never_saw_is_skipped():
    df = _frame()
    drop = (df["dataset_canon"] == "3OCT-Control-24-0.1") & (df["occurrence"] == 2)
    built, _ = _sweep(df[~drop])
    assert ("3-Octanol", 2) not in [(c.odor, c.presentation) for c in built]


def test_comparisons_land_in_a_per_cohort_subfolder():
    built, _ = _sweep()
    assert {c.out_subdir for c in built} == {"3OCT-24-0.1"}


# ---------------------------------------------------------------------------
# Stale outputs
# ---------------------------------------------------------------------------


def test_a_figure_the_sweep_no_longer_builds_is_removed(tmp_path):
    """A rule change (a naive panel withdrawn, an odor retagged) must not leave
    the old figure sitting in the folder looking current."""
    from scripts.analysis.pubfig_naive_vs_trained import build_sweep

    stale_dir = tmp_path / "3OCT-24-0.1"
    stale_dir.mkdir(parents=True)
    stale = stale_dir / "pubfig_naive_vs_trained_Hexanol_0-01pct_p1.png"
    stale.write_bytes(b"old")
    kept = stale_dir / "notes.txt"
    kept.write_text("not mine to delete")

    # thaw_all: this test is about PRUNING, and build_sweep defaults to the
    # shipped config, where every non-sensitivity cohort is frozen for figures.
    # Freeze behaviour has its own file (test_pubfig_naive_freeze.py).
    written, _ = build_sweep(out_dir=tmp_path, df=_frame(), thaw_all=True)
    assert written, "nothing was built, so the prune proves nothing"
    assert not stale.exists()
    assert kept.exists(), "only this driver's own outputs may be pruned"


def test_pruned_files_are_reported(tmp_path, capsys):
    from scripts.analysis.pubfig_naive_vs_trained import build_sweep

    stale = tmp_path / "3OCT-24-0.1" / "pubfig_naive_vs_trained_Gone_1pct.svg"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"old")
    build_sweep(out_dir=tmp_path, df=_frame(), thaw_all=True)
    assert "Gone_1pct" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Presentation selection and the footnote
# ---------------------------------------------------------------------------


def test_cohort_scores_can_take_the_second_presentation():
    df = _frame()
    first = cohort_scores(df, "3OCT-Training-24-0.1", "3-Octanol", presentation=1)
    second = cohort_scores(df, "3OCT-Training-24-0.1", "3-Octanol", presentation=2)
    assert len(first) == len(second) == 2


def test_the_footnote_can_be_dropped():
    comparison = Comparison(
        odor="Hexanol", concentration="0.1%",
        naive_dataset="RandomPanel-24-0.1",
        train_dataset="T", control_dataset="C",
    )
    groups = {
        "Naive": np.array([1.0, 2.0, 3.0]),
        "Trained": np.array([2.0, 3.0, 4.0]),
        "Control": np.array([0.0, 1.0, 1.0]),
    }
    with_note = render_comparison(comparison, groups)
    without = render_comparison(comparison, groups, footnote=False)
    assert len(without.texts) < len(with_note.texts)
    assert not any("Wilson" in t.get_text() for t in without.texts)
    plt.close("all")
