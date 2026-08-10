"""The Training_vs_Control trace figures share the bar figures' odor palette.

``score_bars_*`` paints each training bar with its odor colour and every
control bar grey. These trace figures used to invent their own hues
(``ODOR_COLOURS``: citral red, ethyl butyrate blue) so the same odor read as
two different things depending on the panel. They now pull from
``odor_bar_palette`` — trained line in the odor colour, control line grey — and
label the y axis "Average PER%".
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from scripts.analysis.dataset_means_specific_flies import (
    CTRL_BAND_COLOR,
    CTRL_LINE_COLOR,
    Y_LABEL,
    _plot_odors_single_role,
    _plot_pair_single_role,
    _plot_training_vs_control_for_odor,
)
from scripts.analysis.odor_bar_palette import (
    ACV_COLOR,
    CTRL_COLOR,
    DARK_GREEN,
    DARKER_GREEN,
    HEX_COLOR,
    PINK,
    TRAIN_COLOR,
)

FPS = 10.0
N_FRAMES = 200
ODOR_ON_S = 5.0
ODOR_OFF_S = 10.0


def _per_fly(n_flies: int = 3, value: float = 1.0) -> dict[str, np.ndarray]:
    return {
        f"fly_{i}": np.full(N_FRAMES, value + i, dtype=float) for i in range(n_flies)
    }


def _rgba(color) -> tuple[float, float, float, float]:
    return matplotlib.colors.to_rgba(color)


def _line_colors(ax) -> list[tuple[float, float, float, float]]:
    """Colours of the mean lines, skipping the odor-window guide lines."""
    return [
        _rgba(line.get_color())
        for line in ax.get_lines()
        if line.get_label() and not line.get_label().startswith("_")
    ]


def _tvc_fig(odor: str, **kwargs):
    return _plot_training_vs_control_for_odor(
        odor=odor,
        train_per_fly=_per_fly(3, 1.0),
        ctrl_per_fly=_per_fly(4, 0.5),
        fps=FPS,
        odor_on_s=ODOR_ON_S,
        odor_off_s=ODOR_OFF_S,
        ylim=(-10.0, 40.0),
        **kwargs,
    )


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# --------------------------------------------------------------------------
# Trained line takes the bar figure's odor colour
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("odor", "expected"),
    [
        ("Hexanol", HEX_COLOR),
        ("Apple Cider Vinegar", ACV_COLOR),
        ("3-Octanol", DARK_GREEN),
        ("3-Octonol", DARK_GREEN),  # the older spelling in the OctNov exports
        ("Linalool", DARKER_GREEN),
        ("Ethyl Butyrate", PINK),
    ],
)
def test_trained_line_uses_bar_palette_colour(odor, expected):
    fig = _tvc_fig(odor)
    ax = fig.axes[0]
    trained = [
        line for line in ax.get_lines() if str(line.get_label()).startswith("Trained")
    ]
    assert len(trained) == 1
    assert _rgba(trained[0].get_color()) == _rgba(expected)


def test_trained_colour_is_not_the_old_odor_colours_hue():
    """Citral was red here and yellow in the bars — the palette now wins."""
    fig = _tvc_fig("Citral")
    trained = [
        line for line in fig.axes[0].get_lines()
        if str(line.get_label()).startswith("Trained")
    ][0]
    assert _rgba(trained.get_color()) != _rgba("#d62728")


def test_unpalettised_odor_falls_back_to_the_bar_train_colour():
    """Benzaldehyde has no palette entry; the bars paint it TRAIN_COLOR."""
    fig = _tvc_fig("Benzaldehyde")
    trained = [
        line for line in fig.axes[0].get_lines()
        if str(line.get_label()).startswith("Trained")
    ][0]
    assert _rgba(trained.get_color()) == _rgba(TRAIN_COLOR)


def test_color_key_overrides_a_decorated_label():
    """A caller whose label carries a suffix still hits the palette."""
    fig = _tvc_fig("Hexanol (0.1%) 2", color_key="Hexanol")
    trained = [
        line for line in fig.axes[0].get_lines()
        if str(line.get_label()).startswith("Trained")
    ][0]
    assert _rgba(trained.get_color()) == _rgba(HEX_COLOR)
    assert "Hexanol (0.1%) 2" in fig.axes[0].get_title()


# --------------------------------------------------------------------------
# Control cohort is grey, like the control bars
# --------------------------------------------------------------------------


@pytest.mark.parametrize("odor", ["Hexanol", "Apple Cider Vinegar", "Benzaldehyde"])
def test_control_line_is_grey_for_every_odor(odor):
    fig = _tvc_fig(odor)
    control = [
        line for line in fig.axes[0].get_lines()
        if str(line.get_label()).startswith("Control")
    ]
    assert len(control) == 1
    r, g, b, _ = _rgba(control[0].get_color())
    assert r == g == b, f"control line should be neutral grey, got {(r, g, b)}"
    assert _rgba(control[0].get_color()) == _rgba(CTRL_LINE_COLOR)


def test_control_band_uses_the_bar_control_grey():
    fig = _tvc_fig("Hexanol")
    bands = fig.axes[0].collections
    assert len(bands) == 2  # control band drawn first, then trained
    assert _rgba(bands[0].get_facecolor()[0])[:3] == _rgba(CTRL_BAND_COLOR)[:3]
    assert _rgba(CTRL_BAND_COLOR) == _rgba(CTRL_COLOR)


def test_control_line_is_darker_than_its_band_so_it_reads_on_the_shading():
    """The odor window is shaded grey; a #b0b0b0 line would vanish into it."""
    assert _rgba(CTRL_LINE_COLOR)[0] < _rgba(CTRL_BAND_COLOR)[0]


def test_trained_and_control_bands_are_not_the_same_colour():
    fig = _tvc_fig("Hexanol")
    ctrl_band, train_band = fig.axes[0].collections
    assert _rgba(ctrl_band.get_facecolor()[0])[:3] != _rgba(
        train_band.get_facecolor()[0]
    )[:3]


# --------------------------------------------------------------------------
# Y axis label
# --------------------------------------------------------------------------


def test_y_label_constant_is_average_per_percent():
    assert Y_LABEL == "Average PER%"


def test_training_vs_control_y_label():
    assert _tvc_fig("Hexanol").axes[0].get_ylabel() == "Average PER%"


def test_pair_plot_y_label():
    fig = _plot_pair_single_role(
        pair=("Hexanol", "Ethyl Butyrate"),
        role="trained",
        per_odor={"Hexanol": _per_fly(), "Ethyl Butyrate": _per_fly()},
        fps=FPS,
        odor_on_s=ODOR_ON_S,
        odor_off_s=ODOR_OFF_S,
        ylim=None,
    )
    assert fig.axes[0].get_ylabel() == "Average PER%"


def test_group_plot_y_label():
    fig = _plot_odors_single_role(
        odors=("Hexanol", "Linalool", "3-Octanol"),
        role="control",
        per_odor={
            "Hexanol": _per_fly(),
            "Linalool": _per_fly(),
            "3-Octonol": _per_fly(),
        },
        fps=FPS,
        odor_on_s=ODOR_ON_S,
        odor_off_s=ODOR_OFF_S,
        ylim=None,
    )
    assert fig.axes[0].get_ylabel() == "Average PER%"


# --------------------------------------------------------------------------
# Pair / group overlays use the same palette, in both roles
# --------------------------------------------------------------------------


@pytest.mark.parametrize("role", ["trained", "control"])
def test_pair_plot_lines_use_bar_palette(role):
    fig = _plot_pair_single_role(
        pair=("Hexanol", "Ethyl Butyrate"),
        role=role,
        per_odor={"Hexanol": _per_fly(), "Ethyl Butyrate": _per_fly(value=5.0)},
        fps=FPS,
        odor_on_s=ODOR_ON_S,
        odor_off_s=ODOR_OFF_S,
        ylim=None,
    )
    assert _line_colors(fig.axes[0]) == [_rgba(HEX_COLOR), _rgba(PINK)]


def test_pair_plot_colour_does_not_depend_on_role():
    """Hexanol is one green whether the figure is trained-only or control-only."""
    per_odor = {"Hexanol": _per_fly(), "Ethyl Butyrate": _per_fly()}
    kwargs = dict(
        pair=("Hexanol", "Ethyl Butyrate"),
        per_odor=per_odor,
        fps=FPS,
        odor_on_s=ODOR_ON_S,
        odor_off_s=ODOR_OFF_S,
        ylim=None,
    )
    trained = _line_colors(_plot_pair_single_role(role="trained", **kwargs).axes[0])
    control = _line_colors(_plot_pair_single_role(role="control", **kwargs).axes[0])
    assert trained == control


def test_group_plot_lines_use_bar_palette():
    fig = _plot_odors_single_role(
        odors=("Hexanol", "Linalool", "3-Octanol"),
        role="control",
        per_odor={
            "Hexanol": _per_fly(),
            "Linalool": _per_fly(value=3.0),
            "3-Octonol": _per_fly(value=6.0),
        },
        fps=FPS,
        odor_on_s=ODOR_ON_S,
        odor_off_s=ODOR_OFF_S,
        ylim=None,
    )
    assert _line_colors(fig.axes[0]) == [
        _rgba(HEX_COLOR),
        _rgba(DARKER_GREEN),
        _rgba(DARK_GREEN),
    ]


def test_group_plot_band_matches_its_line():
    fig = _plot_odors_single_role(
        odors=("Hexanol",),
        role="control",
        per_odor={"Hexanol": _per_fly()},
        fps=FPS,
        odor_on_s=ODOR_ON_S,
        odor_off_s=ODOR_OFF_S,
        ylim=None,
    )
    ax = fig.axes[0]
    assert _rgba(ax.collections[0].get_facecolor()[0])[:3] == _rgba(HEX_COLOR)[:3]


# --------------------------------------------------------------------------
# Everything else about the figure is unchanged
# --------------------------------------------------------------------------


def test_legend_still_reports_cohort_sizes():
    fig = _tvc_fig("Hexanol")
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert labels == ["Control (n=4)", "Trained (n=3)"]


def test_odor_window_guides_survive():
    ax = _tvc_fig("Hexanol").axes[0]
    dashed_x = sorted(
        line.get_xdata()[0]
        for line in ax.get_lines()
        if line.get_linestyle() == "--"
    )
    assert dashed_x == [ODOR_ON_S, ODOR_OFF_S]
    assert ax.get_xlabel() == "Time (s)"


def test_isoamyl_acetate_trace_is_purple():
    """The v2 cohorts relabel ACV as isoamyl acetate; it gets its own purple."""
    from scripts.analysis.odor_bar_palette import ISOAMYL_PURPLE

    fig = _tvc_fig("Isoamyl Acetate (1%)")
    trained = [
        line for line in fig.axes[0].get_lines()
        if str(line.get_label()).startswith("Trained")
    ][0]
    assert _rgba(trained.get_color()) == _rgba(ISOAMYL_PURPLE)


def test_isoamyl_acetate_via_base_odor_key_color_key():
    """The per-presentation driver passes base_odor_key() as color_key."""
    from scripts.analysis.dataset_mean_traces_tvc import base_odor_key
    from scripts.analysis.odor_bar_palette import ISOAMYL_PURPLE

    fig = _tvc_fig(
        "Isoamyl Acetate (1%) 2", color_key=base_odor_key("Isoamyl Acetate (1%) 2")
    )
    trained = [
        line for line in fig.axes[0].get_lines()
        if str(line.get_label()).startswith("Trained")
    ][0]
    assert _rgba(trained.get_color()) == _rgba(ISOAMYL_PURPLE)


def test_concentration_tagged_labels_still_hit_the_palette_without_a_color_key():
    """v2 labels carry "(0.1%)"; the figure must not fall back to blue."""
    fig = _tvc_fig("Hexanol (0.1%)")
    trained = [
        line for line in fig.axes[0].get_lines()
        if str(line.get_label()).startswith("Trained")
    ][0]
    assert _rgba(trained.get_color()) == _rgba(HEX_COLOR)
