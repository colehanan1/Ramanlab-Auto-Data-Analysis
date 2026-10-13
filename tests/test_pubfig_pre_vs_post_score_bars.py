"""Publication version of the pre-vs-post score bars (scripts/analysis/pubfig_pre_vs_post_score_bars.py).

Pins the requested styling: upper-only SEM, black-outlined bars, no title, all
text black, a fixed y max -- and that a bar or bracket never silently runs off
that fixed max.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import pytest

import scripts.analysis.pubfig_pre_vs_post_score_bars as pub


def _summary(**overrides):
    rows = {
        "odor": ["Citral", "Isoamyl Acetate"],
        "n_flies": [10, 10],
        "mean_pre": [0.9, 0.7],
        "sem_pre": [0.3, 0.4],
        "mean_post": [1.4, 2.2],
        "sem_post": [0.6, 0.5],
        "p_score": [0.56, 0.008],
    }
    rows.update(overrides)
    return pd.DataFrame(rows)


def _black(color) -> bool:
    return np.allclose(mcolors.to_rgb(color), (0, 0, 0))


@pytest.fixture
def fig_ax():
    fig = pub.build_figure(_summary(), trained_odor="Isoamyl Acetate", ymax=4)
    yield fig, fig.axes[0]
    matplotlib.pyplot.close(fig)


def test_y_axis_runs_to_the_requested_max(fig_ax):
    _fig, ax = fig_ax
    assert ax.get_ylim()[1] == pytest.approx(4)
    assert max(ax.get_yticks()) == pytest.approx(4)


def test_no_title_anywhere(fig_ax):
    fig, ax = fig_ax
    assert ax.get_title() == "" and ax.get_title(loc="left") == ""
    assert fig._suptitle is None


def test_bars_have_black_outlines(fig_ax):
    _fig, ax = fig_ax
    assert len(ax.patches) == 4
    for bar in ax.patches:
        assert _black(bar.get_edgecolor()) and bar.get_linewidth() > 0


def test_error_bars_point_up_only(fig_ax):
    _fig, ax = fig_ax
    tops = {round(b.get_x() + b.get_width() / 2, 6): b.get_height() for b in ax.patches}
    for line in ax.lines:
        xs, ys = line.get_xdata(), line.get_ydata()
        if len(xs) == 2 and xs[0] == xs[1] and round(xs[0], 6) in tops:
            # Vertical error segment: starts at the bar top, goes up.
            assert min(ys) == pytest.approx(tops[round(xs[0], 6)])
            assert max(ys) > min(ys)


def test_every_text_element_is_black(fig_ax):
    fig, ax = fig_ax
    texts = list(ax.texts) + ax.get_xticklabels() + ax.get_yticklabels()
    texts += [ax.yaxis.label, ax.xaxis.label]
    legend = ax.get_legend()
    if legend is not None:
        texts += legend.get_texts()
    assert texts
    for t in texts:
        assert _black(t.get_color()), t.get_text()


def test_significant_odor_gets_stars(fig_ax):
    _fig, ax = fig_ax
    assert "**" in [t.get_text() for t in ax.texts]


def test_star_sits_inside_the_axis_just_above_its_error_bar(fig_ax):
    """The n label must not be mistaken for data and push the bracket to the ceiling."""
    _fig, ax = fig_ax
    star = next(t for t in ax.texts if t.get_text() == "**")
    # Isoamyl Acetate post: 2.2 + 0.5 SEM = 2.7.
    assert 2.7 < star.get_position()[1] < 3.3


def test_data_above_the_fixed_max_is_an_error_not_a_clip():
    with pytest.raises(ValueError):
        pub.build_figure(_summary(mean_post=[1.4, 3.8]), trained_odor="", ymax=4)


def test_writes_png_pdf_svg(tmp_path):
    paths = pub.render(_summary(), out_stem=tmp_path / "fig", trained_odor="Isoamyl Acetate", ymax=4)
    assert sorted(p.suffix for p in paths) == [".pdf", ".png", ".svg"]
    assert all(p.stat().st_size > 0 for p in paths)
