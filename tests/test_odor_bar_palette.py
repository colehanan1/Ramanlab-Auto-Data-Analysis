"""Shared odor palette for trained bar plots.

Hexanol bars are the trace figure's green and Apple Cider Vinegar bars its
orange, in every figure that draws training bars. Odors with no palette entry
keep the old dark/light blue split between trained and untrained.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest
from matplotlib.colors import to_hex

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts.analysis import odor_bar_palette as pal


def test_palette_matches_the_trace_figure():
    assert pal.HEX_COLOR == "#6cc070"
    assert pal.ACV_COLOR == "#ffc685"


def test_hexanol_and_acv_are_coloured_by_odor():
    colors = pal.training_bar_colors(
        ["Hexanol", "Apple Cider Vinegar"], [True, False]
    )
    assert colors == [pal.HEX_COLOR, pal.ACV_COLOR]


def test_odor_colour_wins_whether_or_not_the_odor_was_trained():
    trained = pal.training_bar_colors(["Apple Cider Vinegar"], [True])
    untrained = pal.training_bar_colors(["  apple cider VINEGAR "], [False])
    assert trained == untrained == [pal.ACV_COLOR]


def test_every_panel_odor_has_its_own_entry():
    colors = pal.training_bar_colors(
        ["3-Octanol", "Linalool", "Ethyl Butyrate", "Citral"],
        [False, False, False, False],
    )
    assert colors == [
        pal.DARK_GREEN,
        pal.DARKER_GREEN,
        pal.PINK,
        pal.CITRAL_YELLOW,
    ]


def test_linalool_is_darker_than_3_octanol():
    import matplotlib.colors as mcolors

    def _lightness(hex_color):
        return sum(mcolors.to_rgb(hex_color))

    assert _lightness(pal.DARKER_GREEN) < _lightness(pal.DARK_GREEN)


def test_citral_is_distinct_from_the_acv_orange():
    assert pal.CITRAL_YELLOW != pal.ACV_COLOR


def test_odors_outside_the_palette_keep_the_trained_untrained_blues():
    colors = pal.training_bar_colors(["Benzaldehyde", "Benzaldehyde"], [True, False])
    assert colors == [pal.TRAIN_COLOR, pal.NON_TRAIN_COLOR]


def test_trained_tick_colour_is_black_for_palette_odors_blue_otherwise():
    assert pal.trained_tick_color("Hexanol") == "black"
    assert pal.trained_tick_color("Benzaldehyde") == pal.TRAIN_COLOR


@pytest.fixture()
def ax():
    fig, ax = plt.subplots()
    yield ax
    plt.close(fig)


def test_legend_training_entry_shows_each_odor_colour_once(ax):
    swatches = pal.training_legend_swatches(
        [pal.ACV_COLOR, pal.HEX_COLOR, pal.NON_TRAIN_COLOR, pal.NON_TRAIN_COLOR]
    )
    assert [to_hex(p.get_facecolor()) for p in swatches] == [
        pal.ACV_COLOR,
        pal.HEX_COLOR,
        pal.NON_TRAIN_COLOR,
    ]


def test_legend_names_training_and_control(ax):
    ax.bar([0, 1, 2], [1, 2, 3])
    pal.add_training_legend(
        ax, [pal.ACV_COLOR, pal.HEX_COLOR], ctrl_color=pal.CTRL_COLOR
    )
    assert [t.get_text() for t in ax.get_legend().get_texts()] == [
        "Training",
        "Control",
    ]


def test_legend_omits_control_when_there_is_no_control_cohort(ax):
    ax.bar([0], [1])
    pal.add_training_legend(ax, [pal.HEX_COLOR])
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["Training"]


# --------------------------------------------------------------------------
# Isoamyl acetate — the v2 rigs' odor, purple
# --------------------------------------------------------------------------


def test_isoamyl_acetate_is_the_bar_figures_purple():
    """Pinned to the value the bar figures ship, so traces cannot drift off it."""
    assert pal.ISOAMYL_PURPLE == "#8e6bbf"
    assert pal.odor_color("Isoamyl Acetate") == pal.ISOAMYL_PURPLE


def test_isoamyl_acetate_is_not_the_acv_orange():
    """It is relabelled ACV on the v2 rigs, but it is a different odor."""
    assert pal.odor_color("Isoamyl Acetate") != pal.odor_color("Apple Cider Vinegar")


def test_isoamyl_lookup_tolerates_concentration_and_presentation_labels():
    for label in (
        "isoamyl acetate",
        "Isoamyl Acetate (1%)",
        "Isoamyl Acetate (1%) 2",
        "  Isoamyl Acetate  ",
    ):
        assert pal.odor_color(label) == pal.ISOAMYL_PURPLE, label


def test_concentration_stripping_applies_to_every_palette_odor():
    assert pal.odor_color("Hexanol (0.1%)") == pal.HEX_COLOR
    assert pal.odor_color("3-Octanol (0.1%) 2") == pal.DARK_GREEN
    assert pal.odor_color("Ethyl Butyrate (1%) 1") == pal.PINK


def test_stripping_does_not_invent_a_colour_for_an_unknown_odor():
    assert pal.odor_color("Benzaldehyde (0.1%)") is None
    assert pal.odor_color("(1%)") is None


def test_purple_is_distinct_from_every_other_palette_entry():
    others = set(pal.ODOR_BAR_COLORS.values()) - {pal.ISOAMYL_PURPLE}
    assert pal.ISOAMYL_PURPLE not in others
    assert pal.ISOAMYL_PURPLE not in {pal.TRAIN_COLOR, pal.NON_TRAIN_COLOR, pal.CTRL_COLOR}


def test_an_empty_cohort_draws_no_legend(ax):
    """An empty dataset used to crash HandlerTuple mid-draw."""
    pal.add_training_legend(ax, [])
    assert ax.get_legend() is None
