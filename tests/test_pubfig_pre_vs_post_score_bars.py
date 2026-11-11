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


def _legend_texts(fig):
    return fig.axes[0].get_legend().get_texts()


def test_sensitivity_cohort_legend_names_the_sensitization_phases():
    """Odor-only exposure, nothing paired with it: not pre-test / post-training."""
    fig = pub.build_figure(_summary(), trained_odor="Isoamyl Acetate", ymax=4,
                           cohort="IAA-Sensitivity-1-1")
    assert [t.get_text() for t in _legend_texts(fig)] == ["Pre-exposure", "Post-exposure"]
    matplotlib.pyplot.close(fig)


@pytest.mark.parametrize("cohort", ["Hex-HexTest-Training-32-1", ""])
def test_conditioned_cohort_legend_keeps_its_one_row_of_pre_test_post_training(cohort):
    fig = pub.build_figure(_summary(), trained_odor="Isoamyl Acetate", ymax=4, cohort=cohort)
    texts = _legend_texts(fig)
    assert [t.get_text() for t in texts] == ["Pre-test", "Post-training"]
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    pre, post = (t.get_window_extent(r) for t in texts)
    assert abs(pre.y0 - post.y0) < pre.height / 2  # side by side, one row
    matplotlib.pyplot.close(fig)


def test_longer_sensitization_legend_clears_the_n_label():
    fig = pub.build_figure(_summary(), ymax=4, cohort="IAA-Sensitivity-1-1")
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    ax = fig.axes[0]
    n_label = next(t for t in ax.texts if "flies" in t.get_text())
    assert not ax.get_legend().get_window_extent(r).overlaps(n_label.get_window_extent(r))
    matplotlib.pyplot.close(fig)


@pytest.mark.parametrize("cohort, drawn, gone", [
    ("IAA-Sensitivity-1-1", ("Pre-exposure", "Post-exposure"), ("Pre-test", "Post-training")),
    ("Hex-HexTest-Training-32-1", ("Pre-test", "Post-training"), ("sensitization", "exposure")),
])
def test_cli_names_the_phases_from_the_summary_file_name(tmp_path, cohort, drawn, gone):
    summary_csv = tmp_path / f"{cohort}_summary.csv"
    _summary().to_csv(summary_csv, index=False)
    assert pub.main(["--summary-csv", str(summary_csv), "--out-stem", str(tmp_path / "fig"),
                     "--trained-odor", "Isoamyl Acetate"]) == 0
    svg = (tmp_path / "fig.svg").read_text()  # svg.fonttype "none" keeps text as text
    assert all(s in svg for s in drawn) and not any(s in svg for s in gone)


def _ticks(fig):
    return fig.axes[0].get_xticklabels()


def test_sensitivity_ticks_write_each_odor_with_its_concentration_as_v_v():
    """Pins tokens read "<Odor> <c>% (v/v)"; the sensitized odor is still bolded."""
    fig = pub.build_figure(_summary(odor=["3-octonol-0.1", "isoamylacetate-1"]),
                           trained_odor="Isoamyl Acetate", ymax=4, cohort="IAA-Sensitivity-1-1")
    assert [t.get_text() for t in _ticks(fig)] == ["3-Octanol 0.1% (v/v)", "Isoamyl Acetate 1% (v/v)"]
    assert [t.get_fontweight() for t in _ticks(fig)] == ["normal", "bold"]
    matplotlib.pyplot.close(fig)


def test_both_label_kinds_keep_their_own_bars_each_formatted():
    """IAA-Sensitivity-24-1 has pre-pins and pins batches: two IAA bars, both bold."""
    fig = pub.build_figure(_summary(odor=["Isoamyl Acetate", "isoamylacetate-1"]),
                           trained_odor="Isoamyl Acetate", ymax=4, cohort="IAA-Sensitivity-24-1")
    assert [t.get_text() for t in _ticks(fig)] == ["Isoamyl Acetate", "Isoamyl Acetate 1% (v/v)"]
    assert all(t.get_fontweight() == "bold" for t in _ticks(fig))
    matplotlib.pyplot.close(fig)


@pytest.mark.parametrize("cohort, odors", [
    # Pre-pins batches recorded no concentration: the name alone, no percent.
    ("3Oct-Sensitivity-24-0.1", ["3-Octanol", "Isoamyl Acetate"]),
    # Conditioning cohorts keep the summary's labels exactly.
    ("Hex-HexTest-Training-32-1", ["benzaldehyde-1", "isoamylacetate-1"]),
    ("", ["benzaldehyde-1", "isoamylacetate-1"]),
])
def test_ticks_without_a_concentration_or_outside_sensitivity_are_unchanged(cohort, odors):
    fig = pub.build_figure(_summary(odor=odors), ymax=4, cohort=cohort)
    assert [t.get_text() for t in _ticks(fig)] == odors
    matplotlib.pyplot.close(fig)


@pytest.mark.parametrize("cohort, drawn, gone", [
    ("IAA-Sensitivity-1-1", "Isoamyl Acetate 1% (v/v)", "isoamylacetate-1"),
    ("Hex-HexTest-Training-32-1", "isoamylacetate-1", "(v/v)"),
])
def test_cli_formats_odor_names_for_sensitivity_cohorts_only(tmp_path, cohort, drawn, gone):
    summary_csv = tmp_path / f"{cohort}_summary.csv"
    _summary(odor=["3-octonol-0.1", "isoamylacetate-1"]).to_csv(summary_csv, index=False)
    assert pub.main(["--summary-csv", str(summary_csv), "--out-stem", str(tmp_path / "fig")]) == 0
    svg = (tmp_path / "fig.svg").read_text()
    assert drawn in svg and gone not in svg
