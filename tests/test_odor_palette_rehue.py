"""The August 2026 odor re-hue, and the pubfig restyle of the matrix bars.

Three odors moved off the green family they shared with each other, and the two
warm slots swapped:

* 3-Octanol   green  -> blue
* Linalool    green  -> purple   (was Isoamyl Acetate's)
* Citral      yellow -> orange
* Isoamyl Ac. purple -> yellow   (was Citral's)
* Benzaldehyde  -    -> brown    (it had no entry and fell back to blue)
* Apple Cider V. orange -> red    (it collided with Citral's new orange)
* Sour Dough Y. -    -> teal     (it had no entry)

Hexanol's green and Ethyl Butyrate's pink are unchanged.

The reaction-matrix bar panels used to paint every bar gray with the trained
odor in ``tab:blue``; they now wear the same per-odor palette + gray control as
``pubfig_score_train_vs_control``, and both metrics label their y axis the same
way everywhere.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from scripts.analysis import odor_bar_palette as pal  # noqa: E402
from scripts.analysis.envelope_visuals import plot_reaction_rate_bars  # noqa: E402
from scripts.analysis.reaction_matrix_training_vs_control import (  # noqa: E402
    plot_training_vs_control_bars,
)

PERCENT_Y_LABEL = "Mean PER response %"
SCORE_Y_LABEL = "Mean PER Score"


def _rgba(color):
    return mcolors.to_rgba(color)


def _hue_deg(color: str) -> float:
    """Hue in degrees, so a test can say "brown" without pinning a hex."""
    import colorsys

    r, g, b = mcolors.to_rgb(color)
    return colorsys.rgb_to_hsv(r, g, b)[0] * 360.0


def _value(color: str) -> float:
    import colorsys

    r, g, b = mcolors.to_rgb(color)
    return colorsys.rgb_to_hsv(r, g, b)[2]


# ---------------------------------------------------------------------------
# The re-hue itself
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "odor, lo, hi",
    [
        # (odor, hue window in degrees)
        ("3-Octanol", 190.0, 250.0),        # blue
        ("Benzaldehyde", 15.0, 45.0),       # brown (a dark, low-value orange)
        ("Citral", 25.0, 50.0),             # orange
        ("Isoamyl Acetate", 45.0, 65.0),    # yellow
        ("Linalool", 265.0, 320.0),         # purple
        ("Apple Cider Vinegar", 0.0, 20.0),  # red
        ("Sour Dough Yeast", 170.0, 200.0),  # teal
        ("Hexanol", 90.0, 150.0),           # green, unchanged
        ("Ethyl Butyrate", 300.0, 340.0),   # pink, unchanged
    ],
)
def test_odor_sits_in_its_new_hue_family(odor, lo, hi):
    color = pal.odor_color(odor)
    assert color is not None, f"{odor} has no palette entry"
    assert lo <= _hue_deg(color) <= hi, f"{odor} is {color}, hue {_hue_deg(color):.0f}deg"


def test_benzaldehyde_is_brown_not_just_orange():
    """Brown is a dark orange: the hue alone does not separate it from Citral."""
    assert _value(pal.odor_color("Benzaldehyde")) < 0.75
    assert _value(pal.odor_color("Citral")) > 0.85


def test_no_two_odors_share_a_colour():
    """``3-octonol`` is the older spelling of the same odor, not a second one."""
    colors = [c for k, c in pal.ODOR_BAR_COLORS.items() if k != "3-octonol"]
    assert len(set(colors)) == len(colors)


def test_octanol_and_linalool_no_longer_share_the_green_family():
    """They were DARK_GREEN and DARKER_GREEN — two steps of one hue."""
    octanol = _hue_deg(pal.odor_color("3-Octanol"))
    linalool = _hue_deg(pal.odor_color("Linalool"))
    hexanol = _hue_deg(pal.odor_color("Hexanol"))
    assert abs(octanol - hexanol) > 45.0
    assert abs(linalool - hexanol) > 45.0


def test_citral_and_isoamyl_acetate_swapped():
    """Citral took the warm slot; Isoamyl Acetate took Citral's yellow."""
    assert _hue_deg(pal.odor_color("Citral")) < _hue_deg(pal.odor_color("Isoamyl Acetate"))


def test_acv_moved_off_orange_because_citral_took_it():
    """They appear together in Hex-24-0.005, EB-Control-24-0.1 and RandomPanel."""
    assert _rgba(pal.odor_color("Apple Cider Vinegar")) != _rgba(pal.odor_color("Citral"))
    assert _hue_deg(pal.odor_color("Apple Cider Vinegar")) < 20.0


@pytest.mark.parametrize(
    "label, odor",
    [
        ("3-Octanol (0.1%) 2", "3-Octanol"),
        ("Benzaldehyde (0.1%)", "Benzaldehyde"),
        ("Citral (1%)", "Citral"),
        ("Isoamyl Acetate (1%)", "Isoamyl Acetate"),
        ("Linalool (1%)", "Linalool"),
        ("Sour Dough Yeast (25%)", "Sour Dough Yeast"),
        ("Apple Cider Vinegar 1", "Apple Cider Vinegar"),
    ],
)
def test_decorated_labels_still_resolve(label, odor):
    assert pal.odor_color(label) == pal.odor_color(odor)


def test_older_octanol_spelling_follows_the_new_blue():
    assert pal.odor_color("3-Octonol") == pal.odor_color("3-Octanol")


# ---------------------------------------------------------------------------
# The single-cohort matrix bar panel now wears the pubfig palette
# ---------------------------------------------------------------------------

def _stats(odors, *, trained: str | None = None) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "odor": odors,
            "rate": [0.4] * len(odors),
            "num_trials": [10] * len(odors),
            "is_trained": [o == trained for o in odors],
            "trial_num": list(range(1, len(odors) + 1)),
        }
    )


def test_matrix_bars_take_the_odor_palette_not_flat_gray():
    odors = ["3-Octanol", "Benzaldehyde", "Citral", "Hexanol"]
    fig, ax = plt.subplots()
    try:
        plot_reaction_rate_bars(ax, _stats(odors, trained="Hexanol"), title="t")
        got = [_rgba(p.get_facecolor()) for p in ax.patches]
        assert got == [_rgba(pal.odor_color(o)) for o in odors]
    finally:
        plt.close(fig)


def test_matrix_bars_no_longer_paint_the_trained_odor_tab_blue():
    fig, ax = plt.subplots()
    try:
        plot_reaction_rate_bars(ax, _stats(["Hexanol"], trained="Hexanol"), title="t")
        assert _rgba(ax.patches[0].get_facecolor()) != _rgba("tab:blue")
    finally:
        plt.close(fig)


def test_matrix_bars_fall_back_for_an_unknown_odor():
    fig, ax = plt.subplots()
    try:
        plot_reaction_rate_bars(ax, _stats(["Mystery Odor"], trained="Mystery Odor"), title="t")
        assert _rgba(ax.patches[0].get_facecolor()) == _rgba(pal.TRAIN_COLOR)
    finally:
        plt.close(fig)


def test_matrix_bars_mark_the_trained_odor_by_weight_not_by_shouting():
    """The pubfig bolds the trained tick; it does not upper-case it."""
    fig, ax = plt.subplots()
    try:
        plot_reaction_rate_bars(ax, _stats(["Hexanol", "Citral"], trained="Hexanol"), title="t")
        ticks = {t.get_text(): t for t in ax.get_xticklabels()}
        assert "Hexanol" in ticks, f"trained tick was rewritten: {list(ticks)}"
        assert ticks["Hexanol"].get_weight() == "bold"
        assert ticks["Citral"].get_weight() != "bold"
    finally:
        plt.close(fig)


def test_matrix_bars_carry_a_legend():
    fig, ax = plt.subplots()
    try:
        plot_reaction_rate_bars(ax, _stats(["Hexanol", "Citral"], trained="Hexanol"), title="t")
        assert ax.get_legend() is not None
    finally:
        plt.close(fig)


def test_matrix_bars_use_the_shared_percent_y_label():
    fig, ax = plt.subplots()
    try:
        plot_reaction_rate_bars(ax, _stats(["Hexanol"], trained="Hexanol"), title="t")
        assert ax.get_ylabel() == PERCENT_Y_LABEL
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# The paired training-vs-control matrix bars
# ---------------------------------------------------------------------------

def _tvc_stats(odors, trained: str):
    train = pd.DataFrame(
        {
            "odor": odors,
            "rate": [40.0] * len(odors),
            "num_trials": [10] * len(odors),
            "is_trained": [o == trained for o in odors],
        }
    )
    ctrl = pd.DataFrame(
        {"odor": odors, "rate": [20.0] * len(odors), "num_trials": [12] * len(odors)}
    )
    return train, ctrl


def test_tvc_training_bars_take_the_odor_palette():
    odors = ["3-Octanol", "Citral", "Linalool"]
    fig, ax = plt.subplots()
    try:
        plot_training_vs_control_bars(ax, *_tvc_stats(odors, "3-Octanol"), title="t")
        train_bars = ax.patches[: len(odors)]
        assert [_rgba(b.get_facecolor()) for b in train_bars] == [
            _rgba(pal.odor_color(o)) for o in odors
        ]
    finally:
        plt.close(fig)


def test_tvc_control_bars_are_the_one_shared_gray():
    odors = ["3-Octanol", "Citral"]
    fig, ax = plt.subplots()
    try:
        plot_training_vs_control_bars(ax, *_tvc_stats(odors, "3-Octanol"), title="t")
        ctrl_bars = ax.patches[len(odors):]
        assert {_rgba(b.get_facecolor()) for b in ctrl_bars} == {_rgba(pal.CTRL_COLOR)}
    finally:
        plt.close(fig)


def test_tvc_bars_mark_the_trained_odor_by_weight_not_by_shouting():
    odors = ["3-Octanol", "Citral"]
    fig, ax = plt.subplots()
    try:
        plot_training_vs_control_bars(ax, *_tvc_stats(odors, "3-Octanol"), title="t")
        ticks = {t.get_text(): t for t in ax.get_xticklabels()}
        assert "3-Octanol" in ticks, f"trained tick was rewritten: {list(ticks)}"
        assert ticks["3-Octanol"].get_weight() == "bold"
    finally:
        plt.close(fig)


def test_tvc_bars_use_the_shared_percent_y_label():
    fig, ax = plt.subplots()
    try:
        plot_training_vs_control_bars(ax, *_tvc_stats(["Citral"], "Citral"), title="t")
        assert ax.get_ylabel() == PERCENT_Y_LABEL
    finally:
        plt.close(fig)


def test_tvc_legend_keys_every_odor_colour_and_the_control_gray():
    """The pubfig legend: one swatch per training colour, then the gray."""
    odors = ["3-Octanol", "Citral", "Linalool"]
    fig, ax = plt.subplots()
    try:
        plot_training_vs_control_bars(ax, *_tvc_stats(odors, "3-Octanol"), title="t")
        legend = ax.get_legend()
        assert legend is not None
        labels = [t.get_text() for t in legend.get_texts()]
        assert len(labels) == 2
        assert labels[0].startswith("Training")
        assert labels[1].startswith("Control")
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# One y-axis label per metric, everywhere
# ---------------------------------------------------------------------------

def test_percent_metric_y_label_is_shared():
    from scripts.analysis.pubfig_score_train_vs_control import PERCENT_METRIC, SCORE_METRIC

    assert PERCENT_METRIC.y_label == PERCENT_Y_LABEL
    assert SCORE_METRIC.y_label == SCORE_Y_LABEL


def test_naive_vs_trained_shares_both_labels():
    from scripts.analysis import pubfig_naive_vs_trained as nvt

    assert nvt.RATE_Y_LABEL == PERCENT_Y_LABEL
    assert nvt.SCORE_Y_LABEL == SCORE_Y_LABEL


# ---------------------------------------------------------------------------
# The heatmap panel's ticks match the bar panel directly beneath them
# ---------------------------------------------------------------------------

def test_heatmap_ticks_mark_the_trained_odor_the_same_way_as_the_bars():
    """One figure, two panels: they cannot disagree about how to say "trained"."""
    from scripts.analysis.envelope_visuals import _style_trained_xticks

    fig, ax = plt.subplots()
    try:
        _style_trained_xticks(ax, ["Hexanol", "Citral"], "Hexanol", 9)
        ticks = {t.get_text(): t for t in ax.get_xticklabels()}
        assert "Hexanol" in ticks, f"trained tick was rewritten: {list(ticks)}"
        assert ticks["Hexanol"].get_weight() == "bold"
        assert ticks["Citral"].get_weight() != "bold"
        assert _rgba(ticks["Hexanol"].get_color()) != _rgba("tab:blue")
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# The no-control fallback used to read 3846%
# ---------------------------------------------------------------------------

def test_training_only_bars_do_not_multiply_percent_by_a_hundred():
    """``_rate_stats_from_binary`` returns 0-100; ``plot_reaction_rate_bars``
    takes 0-1 and scales it itself. Feeding one to the other straight through
    turned 38% into 3846% on every dataset with no matching control cohort
    (Hex-Training-24-0.01, RandomPanel-*, ...)."""
    from scripts.analysis.reaction_matrix_training_vs_control import (
        plot_training_only_bars,
    )

    stats = pd.DataFrame(
        {
            "odor": ["Citral", "Linalool"],
            "rate": [38.46, 15.38],  # percent, as the binary CSV reader emits
            "num_trials": [13, 13],
            "is_trained": [False, False],
        }
    )
    fig, ax = plt.subplots()
    try:
        plot_training_only_bars(ax, stats, title="t")
        heights = sorted(round(p.get_height(), 2) for p in ax.patches)
        assert heights == [15.38, 38.46]
        assert ax.get_ylim()[1] <= 115.0
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# "For all" has to keep being true
# ---------------------------------------------------------------------------

def test_no_analysis_script_spells_its_own_per_axis():
    """Six spellings of one axis is how this drifted the first time.

    Any new PER percent or PER score axis must come from ``per_axis_labels``.
    """
    import re
    from pathlib import Path

    banned = re.compile(
        r'["\'](?:'
        r'PER\s*%|PER%|Average PER%|Average PER Response %|'
        r'%\s*of\s*Flies\s*Responding|Responding flies \(%\)|'
        r'Mean Score|Mean [Oo]rdinal [Ss]core'
        r')["\']'
    )
    root = Path(__file__).resolve().parents[1] / "scripts" / "analysis"
    offenders = []
    for path in sorted(root.glob("*.py")):
        if path.name == "per_axis_labels.py":
            continue
        for n, line in enumerate(path.read_text().splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue  # a comment quoting the old name is fine
            if not re.search(r"set_ylabel\s*\(|\bylabel\s*=|_Y_LABEL\s*=|y_label\s*=", line):
                continue  # only axis assignments — other scripts label other quantities
            if banned.search(line):
                offenders.append(f"{path.name}:{n}: {line.strip()}")
    assert not offenders, "hardcoded PER axis labels:\n  " + "\n  ".join(offenders)
