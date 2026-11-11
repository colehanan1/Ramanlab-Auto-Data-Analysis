"""Trained-odor pre/post publication figure across datasets (scripts/analysis/pubfig_cs_plus_pre_post.py)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import pytest

import scripts.analysis.pubfig_cs_plus_pre_post as cs


def _paired(dataset, odor, pre, post):
    return pd.DataFrame({"dataset_canon": dataset, "fly": [f"f{i}" for i in range(len(pre))],
                         "fly_number": 1, "odor": odor, "score_pre": pre, "score_post": post})


def _panels():
    a = cs.panel_summary(_paired("IAA-Sensitivity-1-1", "Isoamyl Acetate",
                                 [0, 1, 0, 1, 2, 0, 1, 0], [3, 4, 2, 4, 3, 3, 4, 2]),
                         dataset="IAA-Sensitivity-1-1", odor="Isoamyl Acetate")
    b = cs.panel_summary(_paired("3Oct-Sensitivity-1-0.1", "3-Octanol",
                                 [1, 0, 2, 1], [1, 1, 2, 0]),
                         dataset="3Oct-Sensitivity-1-0.1", odor="3-Octanol")
    return [a, b]


def test_panel_summary_means_sems_n_and_paired_p():
    a = _panels()[0]
    assert a.n == 8
    assert a.mean_pre == pytest.approx(5 / 8) and a.mean_post == pytest.approx(25 / 8)
    assert a.sem_pre == pytest.approx(np.std([0, 1, 0, 1, 2, 0, 1, 0], ddof=1) / np.sqrt(8))
    assert a.p is not None and a.p < 0.05
    assert a.hours == 1.0


def test_panel_summary_rejects_a_missing_odor():
    with pytest.raises(ValueError):
        cs.panel_summary(_paired("D-Sensitivity-1-1", "Citral", [0], [1]),
                         dataset="D-Sensitivity-1-1", odor="Hexanol")


def test_each_odor_gets_its_own_hue_with_pre_lighter_than_post():
    lum = lambda h: sum(mcolors.to_rgb(h))  # noqa: E731
    for odor in ("Isoamyl Acetate", "3-Octanol"):
        pre, post = cs.odor_colors(odor)
        assert lum(pre) > lum(post)
    assert cs.odor_colors("Isoamyl Acetate") != cs.odor_colors("3-Octanol")


def test_figure_style(tmp_path):
    fig = cs.build_figure(_panels())
    axes = fig.axes
    assert len(axes) == 2
    for ax in axes:
        assert ax.get_title() == "" and len(ax.patches) == 2
        for bar in ax.patches:
            assert np.allclose(mcolors.to_rgb(bar.get_edgecolor()), (0, 0, 0))
        tops = {round(b.get_x() + b.get_width() / 2, 6): b.get_height() for b in ax.patches}
        for line in ax.lines:  # error bars start at the bar top and go up
            xs, ys = line.get_xdata(), line.get_ydata()
            if len(xs) == 2 and xs[0] == xs[1] and round(xs[0], 6) in tops:
                assert min(ys) == pytest.approx(tops[round(xs[0], 6)])
    assert axes[0].get_ylim() == axes[1].get_ylim()
    texts = [t.get_text() for ax in axes for t in ax.texts]
    assert "Isoamyl Acetate 1% (v/v)" in texts and "3-Octanol 0.1% (v/v)" in texts
    assert "n = 8 flies" in texts and "n = 4 flies" in texts
    joined = " ".join(texts)
    for gone in ("Sensitivity", "CS+", "starved"):  # header is odor + conc, then n
        assert gone not in joined


def test_render_writes_png_pdf_svg_and_table(tmp_path):
    paths = cs.render(_panels(), out_stem=tmp_path / "fig")
    assert sorted(p.suffix for p in paths) == [".csv", ".pdf", ".png", ".svg"]
    t = pd.read_csv(tmp_path / "fig.csv")
    assert list(t["odor"]) == ["Isoamyl Acetate", "3-Octanol"]


def test_starvation_hours_override_beats_the_dataset_name():
    p = cs.panel_summary(_paired("IAA-Sensitivity-1-1", "Isoamyl Acetate", [0, 1], [2, 3]),
                         dataset="IAA-Sensitivity-1-1", odor="Isoamyl Acetate", hours=0.0)
    assert p.hours == 0.0
    # Kept in the table, but no longer drawn: the header is name + n only.
    fig = cs.build_figure([p])
    assert "starved" not in " ".join(t.get_text() for t in fig.axes[0].texts)


@pytest.mark.parametrize("spec, expected", [
    ("IAA-Sensitivity-1-1=Isoamyl Acetate", ("IAA-Sensitivity-1-1", "Isoamyl Acetate", None)),
    ("3Oct-Sensitivity-1-0.1=3-Octanol=0", ("3Oct-Sensitivity-1-0.1", "3-Octanol", 0.0)),
])
def test_parse_panel_spec(spec, expected):
    assert cs.parse_panel(spec) == expected


def test_default_panels_are_zero_hours():
    assert all(cs.parse_panel(s)[2] == 0.0 for s in cs.DEFAULT_PANELS)


def test_headers_sit_inside_the_axes_for_a_compact_figure():
    fig = cs.build_figure(_panels())
    for ax in fig.axes:
        heads = [t for t in ax.texts if "%" in t.get_text() or "flies" in t.get_text()]
        assert len(heads) == 2
        for t in heads:
            assert t.get_transform() == ax.transAxes and 0 < t.get_position()[1] <= 1.0
    assert fig.get_size_inches()[1] <= 2.8


def test_bracket_is_not_pushed_up_by_the_header():
    """Header drawn after the bracket, so the star stays just above its error bar."""
    fig = cs.build_figure(_panels())
    star = next(t for t in fig.axes[0].texts if t.get_text().startswith("*"))
    p = _panels()[0]
    assert star.get_position()[1] < p.mean_post + p.sem_post + 0.9


def test_headers_fit_inside_their_panel_horizontally():
    fig = cs.build_figure(_panels())
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    for ax in fig.axes:
        box = ax.get_window_extent(r)
        for t in ax.texts:
            if "%" in t.get_text() or "flies" in t.get_text():
                tb = t.get_window_extent(r)
                assert tb.x0 >= box.x0 + 2 and tb.x1 <= box.x1 + 1, t.get_text()


def test_axis_runs_to_3_75_but_the_last_tick_and_spine_stop_at_3():
    # Real-sized bars (tallest bar + SEM ~2.3, as in the IAA/3-Octanol data).
    real = [cs.Panel("IAA-Sensitivity-1-1", "Isoamyl Acetate", 20, 0.6, 0.3, 1.8, 0.45,
                     0.01, 0.0, "1"),
            cs.Panel("3Oct-Sensitivity-1-0.1", "3-Octanol", 11, 0.9, 0.3, 0.3, 0.2,
                     None, 0.0, "0.1")]
    fig = cs.build_figure(real)
    ax = fig.axes[0]
    assert ax.get_ylim()[1] == pytest.approx(3.75)
    assert max(ax.get_yticks()) == pytest.approx(3)
    assert ax.spines["left"].get_bounds()[1] == pytest.approx(3)
    assert fig.get_size_inches()[1] <= 2.45


def test_data_taller_than_the_default_top_still_fits():
    tall = cs.Panel("X-Sensitivity-1-1", "Hexanol", 5, 1.0, 0.1, 3.4, 0.4, None, 0.0, "1")
    ax = cs.build_figure([tall]).axes[0]
    assert ax.get_ylim()[1] > 3.8 + 0.5  # bar + SEM + room for the header


def test_sensitivity_panels_draw_no_conditioning_words():
    """Nothing is paired with the odor in these datasets: no CS+, pre-test or post-training."""
    fig = cs.build_figure(_panels())
    fig.canvas.draw()
    drawn = [t.get_text() for ax in fig.axes
             for t in (*ax.texts, *ax.get_xticklabels(), ax.yaxis.label)]
    joined = " ".join(drawn).casefold()
    assert "pre" in joined and "post" in joined
    for gone in ("cs+", "trained", "training", "test"):
        assert gone not in joined


def _panel(dataset, odor, conc):
    return cs.Panel(dataset, odor, 5, 0.5, 0.1, 1.0, 0.2, None, 0.0, conc)


@pytest.mark.parametrize("dataset, odor, conc, header", [
    ("IAA-Sensitivity-1-1", "Isoamyl Acetate", "1", "Isoamyl Acetate 1% (v/v)"),
    ("3Oct-Sensitivity-24-0.1", "3-Octanol", "0.1", "3-Octanol 0.1% (v/v)"),
    # _HOURS_RE (the .csv's conc) misses the 5s name; the header still reads its 1%.
    ("IAA-Sensitivity5s-1-1", "Isoamyl Acetate", None, "Isoamyl Acetate 1% (v/v)"),
    # The name's concentration is the sensitized odor's: another odor gets none.
    ("IAA-Sensitivity-1-1", "3-Octanol", "1", "3-Octanol"),
    # No concentration in the name: the odor alone, none guessed.
    ("IAA-Sensitivity", "Isoamyl Acetate", None, "Isoamyl Acetate"),
    # A conditioning cohort keeps its old header text exactly.
    ("Hex-HexTest-Training-32-1", "Hexanol", None, "Hexanol"),
    ("Hex-HexTest-Training-32-1", "Hexanol", "1", "Hexanol 1%"),
])
def test_header_writes_a_sensitivity_concentration_as_v_v(dataset, odor, conc, header):
    fig = cs.build_figure([_panel(dataset, odor, conc)])
    assert [t.get_text() for t in fig.axes[0].texts if "flies" not in t.get_text()] == [header]
    matplotlib.pyplot.close(fig)


def test_a_long_header_widens_the_figure_instead_of_shrinking_its_type():
    default_w = 2.05 * 2 + 0.5
    short = cs.build_figure([_panel("Hex-HexTest-Training-32-1", "Hexanol", None)] * 2)
    assert short.get_size_inches()[0] == pytest.approx(default_w)  # fits already: untouched
    wide = cs.build_figure(_panels())
    assert wide.get_size_inches()[0] > default_w
    head = next(t for t in wide.axes[0].texts if t.get_text().endswith("(v/v)"))
    assert head.get_fontsize() == cs.HEAD_PT
    for fig in (short, wide):
        matplotlib.pyplot.close(fig)
