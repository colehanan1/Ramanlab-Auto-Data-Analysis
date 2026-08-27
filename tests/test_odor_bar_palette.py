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


def test_acv_moved_off_the_orange_citral_now_owns():
    """They share a panel in Hex-24-0.005, EB-Control-24-0.1 and RandomPanel."""
    assert pal.ACV_COLOR != "#ffc685"
    assert pal.ACV_COLOR != pal.CITRAL_ORANGE


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
        pal.OCTANOL_BLUE,
        pal.LINALOOL_PURPLE,
        pal.PINK,
        pal.CITRAL_ORANGE,
    ]


def test_linalool_and_3_octanol_no_longer_share_one_hue():
    """They used to be DARK_GREEN and DARKER_GREEN — two steps of one green,
    alongside Hexanol's. Telling three greens apart in one panel asked too much
    of the reader, so each took its own hue."""
    import colorsys

    import matplotlib.colors as mcolors

    def _hue(hex_color):
        return colorsys.rgb_to_hsv(*mcolors.to_rgb(hex_color))[0] * 360.0

    hues = [_hue(c) for c in (pal.LINALOOL_PURPLE, pal.OCTANOL_BLUE, pal.HEX_COLOR)]
    for a, b in ((0, 1), (0, 2), (1, 2)):
        assert abs(hues[a] - hues[b]) > 45.0


def test_citral_is_distinct_from_the_acv_orange():
    assert pal.CITRAL_ORANGE != pal.ACV_COLOR


def test_odors_outside_the_palette_keep_the_trained_untrained_blues():
    colors = pal.training_bar_colors(["Nonanal", "Nonanal"], [True, False])
    assert colors == [pal.TRAIN_COLOR, pal.NON_TRAIN_COLOR]


def test_trained_tick_colour_is_black_for_palette_odors_blue_otherwise():
    assert pal.trained_tick_color("Hexanol") == "black"
    assert pal.trained_tick_color("Nonanal") == pal.TRAIN_COLOR


def test_benzaldehyde_has_an_entry_and_no_longer_falls_back_to_blue():
    """It was the one panel odor with no colour of its own."""
    assert pal.odor_color("Benzaldehyde") == pal.BENZALDEHYDE_BROWN
    assert pal.odor_color("Benzaldehyde") not in {pal.TRAIN_COLOR, pal.NON_TRAIN_COLOR}


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
# Isoamyl acetate — the v2 rigs' odor, yellow (it swapped with Citral)
# --------------------------------------------------------------------------


def test_isoamyl_acetate_is_the_bar_figures_yellow():
    """Pinned to the value the bar figures ship, so traces cannot drift off it."""
    assert pal.ISOAMYL_YELLOW == "#f2d43c"
    assert pal.odor_color("Isoamyl Acetate") == pal.ISOAMYL_YELLOW


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
        assert pal.odor_color(label) == pal.ISOAMYL_YELLOW, label


def test_concentration_stripping_applies_to_every_palette_odor():
    assert pal.odor_color("Hexanol (0.1%)") == pal.HEX_COLOR
    assert pal.odor_color("3-Octanol (0.1%) 2") == pal.OCTANOL_BLUE
    assert pal.odor_color("Ethyl Butyrate (1%) 1") == pal.PINK


def test_stripping_does_not_invent_a_colour_for_an_unknown_odor():
    assert pal.odor_color("Nonanal (0.1%)") is None
    assert pal.odor_color("(1%)") is None


def test_yellow_is_distinct_from_every_other_palette_entry():
    others = set(pal.ODOR_BAR_COLORS.values()) - {pal.ISOAMYL_YELLOW}
    assert pal.ISOAMYL_YELLOW not in others
    assert pal.ISOAMYL_YELLOW not in {pal.TRAIN_COLOR, pal.NON_TRAIN_COLOR, pal.CTRL_COLOR}


def test_an_empty_cohort_draws_no_legend(ax):
    """An empty dataset used to crash HandlerTuple mid-draw."""
    pal.add_training_legend(ax, [])
    assert ax.get_legend() is None
