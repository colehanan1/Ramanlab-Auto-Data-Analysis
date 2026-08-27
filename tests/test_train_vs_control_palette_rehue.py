"""Every training-vs-control panel wears the pubfig's split.

``pubfig_score_train_vs_control`` is the reference: the first series takes each
odor's palette colour, the second is the one shared gray, and the trained odor
is marked by a bold x tick rather than by shouting its name in dark blue
capitals. The reaction-matrix panels were brought onto that footing first; this
covers the score-side panels that still painted every trained bar ``#1a3a6b``
and every other bar one of three flat blues and grays.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _candidate in (PROJECT_ROOT, PROJECT_ROOT / "scripts", PROJECT_ROOT / "src"):
    if str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

from scripts.analysis import odor_bar_palette as pal  # noqa: E402
from scripts.analysis.per_axis_labels import SCORE_Y_LABEL  # noqa: E402

_SPEC = importlib.util.spec_from_file_location(
    "score_summary_rehue", PROJECT_ROOT / "scripts" / "analysis" / "score_summary.py"
)
score_summary = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = score_summary
_SPEC.loader.exec_module(score_summary)

# The three flat colours these panels used to hand out regardless of odor.
OLD_TRAIN_TRAINED = "#1a3a6b"
OLD_TRAIN_OTHER = "#7bafd4"
OLD_CTRL_OTHER = "#c8c8c8"


def _rgba(color):
    return mcolors.to_rgba(color)


def _facecolors(bars):
    return [_rgba(b.get_facecolor()) for b in bars]


# ---------------------------------------------------------------------------
# score_summary: mean_score_train_vs_ctrl_<dataset>
# ---------------------------------------------------------------------------

def _tvc_rows():
    """One row per presented odor, as the train-vs-control summary builds them."""
    odors = ["Hexanol", "3-Octanol", "Citral", "Linalool"]
    return pd.DataFrame(
        {
            "odor": odors,
            "mean_score_train": [3.1, 1.2, 0.8, 0.4],
            "sem_score_train": [0.3, 0.2, 0.2, 0.1],
            "mean_score_ctrl": [1.0, 1.1, 0.9, 0.5],
            "sem_score_ctrl": [0.2, 0.2, 0.2, 0.1],
            "n_flies_train": [12, 12, 12, 12],
            "n_flies_ctrl": [14, 14, 14, 14],
            "is_trained": [True, False, False, False],
        }
    )


def _render_tvc():
    rows = _tvc_rows()
    fig, ax = plt.subplots()
    score_summary.plot_score_train_vs_control(
        ax, rows, title="t", label="Hexanol"
    )
    return fig, ax, rows


def test_score_tvc_training_bars_take_the_odor_palette():
    fig, ax, rows = _render_tvc()
    try:
        train_bars = ax.patches[: len(rows)]
        assert _facecolors(train_bars) == [
            _rgba(pal.odor_color(o)) for o in rows["odor"]
        ]
    finally:
        plt.close(fig)


def test_score_tvc_control_bars_are_the_one_shared_gray():
    """They used to split into #808080 / #c8c8c8 on trained-ness."""
    fig, ax, rows = _render_tvc()
    try:
        ctrl_bars = ax.patches[len(rows):]
        assert set(_facecolors(ctrl_bars)) == {_rgba(pal.CTRL_COLOR)}
    finally:
        plt.close(fig)


def test_score_tvc_drops_the_three_flat_blues():
    fig, ax, rows = _render_tvc()
    try:
        seen = set(_facecolors(ax.patches))
        for retired in (OLD_TRAIN_TRAINED, OLD_TRAIN_OTHER, OLD_CTRL_OTHER):
            assert _rgba(retired) not in seen, retired
    finally:
        plt.close(fig)


def test_score_tvc_marks_the_trained_odor_by_weight_not_by_shouting():
    fig, ax, rows = _render_tvc()
    try:
        ticks = list(ax.get_xticklabels())
        trained = ticks[0]
        assert trained.get_text().startswith("Hexanol"), trained.get_text()
        assert "HEXANOL" not in trained.get_text()
        assert trained.get_weight() == "bold"
        assert ticks[1].get_weight() != "bold"
    finally:
        plt.close(fig)


def test_score_tvc_puts_n_in_the_legend_not_on_every_tick():
    fig, ax, rows = _render_tvc()
    try:
        ticks = [t.get_text() for t in ax.get_xticklabels()]
        assert not any("n=" in t for t in ticks), ticks
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert labels == ["Training (n=12)", "Control (n=14)"], labels
    finally:
        plt.close(fig)


def test_score_tvc_keeps_n_on_the_tick_when_it_varies_by_odor():
    """A single legend number would be a lie, so nothing is silently dropped."""
    rows = _tvc_rows()
    rows["n_flies_train"] = [12, 12, 9, 12]
    fig, ax = plt.subplots()
    try:
        score_summary.plot_score_train_vs_control(ax, rows, title="t", label="Hexanol")
        ticks = [t.get_text() for t in ax.get_xticklabels()]
        assert any("n=" in t for t in ticks), ticks
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert labels == ["Training", "Control (n=14)"], labels
    finally:
        plt.close(fig)


def test_score_tvc_uses_the_shared_score_y_label():
    fig, ax, _ = _render_tvc()
    try:
        assert ax.get_ylabel() == SCORE_Y_LABEL
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# score_summary: the single-cohort mean_score_<dataset> bars
# ---------------------------------------------------------------------------

def test_single_cohort_score_bars_take_the_odor_palette():
    """They were dark blue for the trained odor and flat gray for the rest."""
    odors = ["Hexanol", "Citral", "Linalool"]
    colors = score_summary.single_cohort_bar_colors(
        odors, [True, False, False]
    )
    assert colors == [pal.odor_color(o) for o in odors]


def test_single_cohort_score_bars_fall_back_for_an_unknown_odor():
    colors = score_summary.single_cohort_bar_colors(["Nonanal"], [True])
    assert colors == [pal.TRAIN_COLOR]


# ---------------------------------------------------------------------------
# The score matrix's odor ticks match the bars beneath them
# ---------------------------------------------------------------------------

def test_score_matrix_ticks_are_bold_not_upper_cased():
    from scripts.analysis.envelope_visuals import set_protocol

    set_protocol("v2")
    rows = []
    for idx, fly in enumerate(("f1", "f2", "f3"), start=1):
        for label, s in (("testing_1_hexanol", 4), ("testing_2_citral", 1)):
            rows.append({
                "dataset": "Hex-Training", "fly": fly, "fly_number": str(idx),
                "trial_label": label, "score": s, "trial_type": "testing",
            })
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        pd.DataFrame(rows).to_csv(td / "s.csv", index=False)
        plt.close("all")
        real_close = plt.close
        plt.close = lambda *a, **k: None
        try:
            score_summary.generate_score_summary(
                csv_path=td / "s.csv", out_dir=td / "out", overwrite=True
            )
        finally:
            plt.close = real_close
        fig = next(
            f for f in map(plt.figure, plt.get_fignums())
            if any(a.get_ylabel() == SCORE_Y_LABEL for a in f.axes)
        )
        try:
            texts = [
                t.get_text()
                for a in (*fig.axes, *sum((list(a.child_axes) for a in fig.axes), []))
                for t in a.get_xticklabels()
                if t.get_text()
            ]
            assert texts, "no odor ticks found"
            assert not any(t.isupper() and len(t) > 2 for t in texts), texts
        finally:
            plt.close(fig)


# ---------------------------------------------------------------------------
# rig_batch_breakdowns: same split, and the series keep their own names
# ---------------------------------------------------------------------------

def test_rig_batch_score_bars_take_the_palette_and_the_shared_gray():
    from scripts.analysis import rig_batch_breakdowns as rbb

    odors = ["Hexanol", "Citral"]
    stats_a = pd.DataFrame(
        {"mean_score": [3.0, 1.0], "sem_score": [0.2, 0.2], "n_flies": [8, 8]}
    )
    stats_b = pd.DataFrame(
        {"mean_score": [1.0, 0.9], "sem_score": [0.2, 0.2], "n_flies": [9, 9]}
    )
    fig, ax = plt.subplots()
    try:
        rbb.plot_score_comparison_bars(
            ax, odors, [True, False], stats_a, stats_b,
            label_a="Rig 2", label_b="Rig 3", title="t",
        )
        assert _facecolors(ax.patches[:2]) == [_rgba(pal.odor_color(o)) for o in odors]
        assert set(_facecolors(ax.patches[2:])) == {_rgba(pal.CTRL_COLOR)}
    finally:
        plt.close(fig)


def test_rig_batch_legend_keeps_the_comparison_names():
    """A rig-2-vs-rig-3 panel must not claim to show "Training vs Control"."""
    from scripts.analysis import rig_batch_breakdowns as rbb

    stats_a = pd.DataFrame({"mean_score": [3.0], "sem_score": [0.2], "n_flies": [8]})
    stats_b = pd.DataFrame({"mean_score": [1.0], "sem_score": [0.2], "n_flies": [9]})
    fig, ax = plt.subplots()
    try:
        rbb.plot_score_comparison_bars(
            ax, ["Hexanol"], [True], stats_a, stats_b,
            label_a="Rig 2", label_b="Rig 3", title="t",
        )
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert [l.split(" (n=")[0] for l in labels] == ["Rig 2", "Rig 3"], labels
        assert "Training" not in labels and "Control" not in labels, labels
    finally:
        plt.close(fig)


def test_matrix_bars_accept_custom_series_names():
    """``rig_batch_breakdowns`` reuses the reaction-rate panel for rig-vs-rig."""
    from scripts.analysis.reaction_matrix_training_vs_control import (
        plot_training_vs_control_bars,
    )

    odors = ["Hexanol", "Citral"]
    train = pd.DataFrame({
        "odor": odors, "rate": [40.0, 20.0],
        "num_trials": [10, 10], "is_trained": [True, False],
    })
    ctrl = pd.DataFrame({
        "odor": odors, "rate": [20.0, 10.0], "num_trials": [12, 12],
    })
    fig, ax = plt.subplots()
    try:
        plot_training_vs_control_bars(
            ax, train, ctrl, title="t", train_label="Rig 2", ctrl_label="Rig 3"
        )
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert labels[0].startswith("Rig 2"), labels
        assert labels[1].startswith("Rig 3"), labels
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# make_pubfig_control_vs_trained_bars
# ---------------------------------------------------------------------------

def test_control_vs_trained_pubfig_uses_the_palette_and_the_shared_gray():
    import importlib.util as iu

    path = PROJECT_ROOT / "scripts" / "analysis" / "make_pubfig_control_vs_trained_bars.py"
    spec = iu.spec_from_file_location("cvt_bars", path)
    mod = iu.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    labels = [label for _, label in mod.BAR_ORDER]
    fig, ax = plt.subplots()
    try:
        mod.draw_comparison_bars(
            ax,
            labels=labels,
            control=[10.0] * len(labels),
            trained=[50.0] * len(labels),
            control_n=9,
            trained_n=11,
            control_label="Control",
            trained_label="Trained",
        )
        n = len(labels)
        assert set(_facecolors(ax.patches[:n])) == {_rgba(pal.CTRL_COLOR)}
        assert _facecolors(ax.patches[n:]) == [
            _rgba(pal.odor_color(l) or pal.TRAIN_COLOR) for l in labels
        ]
        assert _rgba("#1f77b4") not in set(_facecolors(ax.patches))
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# The reaction-boundary line is gone
# ---------------------------------------------------------------------------

def _boundary_lines(ax):
    return [
        ln for ln in ax.get_lines()
        if len(set(ln.get_ydata())) == 1 and abs(ln.get_ydata()[0] - 1.5) < 1e-9
    ]


def test_score_tvc_draws_no_reaction_boundary_line():
    fig, ax, _ = _render_tvc()
    try:
        assert not _boundary_lines(ax), "the red dotted line at 1.5 is still drawn"
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert "Reaction Boundary" not in labels, labels
    finally:
        plt.close(fig)


def test_rig_batch_draws_no_reaction_boundary_line():
    from scripts.analysis import rig_batch_breakdowns as rbb

    stats_a = pd.DataFrame({"mean_score": [3.0], "sem_score": [0.2], "n_flies": [8]})
    stats_b = pd.DataFrame({"mean_score": [1.0], "sem_score": [0.2], "n_flies": [9]})
    fig, ax = plt.subplots()
    try:
        rbb.plot_score_comparison_bars(
            ax, ["Hexanol"], [True], stats_a, stats_b,
            label_a="Rig 2", label_b="Rig 3", title="t",
        )
        assert not _boundary_lines(ax)
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert [l.split(" (n=")[0] for l in labels] == ["Rig 2", "Rig 3"], labels
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Significance: only when it is significant, and never on top of a value label
# ---------------------------------------------------------------------------

def _bracket_lines(ax):
    """The bracket polylines, by shape: y runs [tip, top, top, tip].

    Matching on "4 points and black" also catches the errorbar caps, which are
    Line2Ds with one point per bar.
    """
    out = []
    for ln in ax.get_lines():
        y = list(ln.get_ydata())
        x = list(ln.get_xdata())
        if len(y) != 4 or len(x) != 4:
            continue
        if y[0] == y[3] and y[1] == y[2] and y[1] > y[0] and x[0] == x[1] and x[2] == x[3]:
            out.append(ln)
    return out


def _annotation_texts(ax):
    return [t.get_text() for t in ax.texts]


def test_score_tvc_annotates_only_the_significant_pairs():
    rows = _tvc_rows()
    rows["score_p_value"] = [0.0004, 0.92, 1.0, 0.31]
    fig, ax = plt.subplots()
    try:
        x, bar_w = score_summary.plot_score_train_vs_control(
            ax, rows, title="t", label="Hexanol"
        )
        score_summary._draw_score_significance_brackets(ax, x, bar_w, rows)
        assert len(_bracket_lines(ax)) == 1, "one significant pair -> one bracket"
        texts = _annotation_texts(ax)
        assert "***" in texts, texts
        assert not any(t.startswith("p=") for t in texts), texts
        assert "ns" not in texts, texts
    finally:
        plt.close(fig)


def test_score_tvc_draws_nothing_when_nothing_is_significant():
    rows = _tvc_rows()
    rows["score_p_value"] = [0.9, 0.9, 1.0, 0.4]
    fig, ax = plt.subplots()
    try:
        x, bar_w = score_summary.plot_score_train_vs_control(
            ax, rows, title="t", label="Hexanol"
        )
        score_summary._draw_score_significance_brackets(ax, x, bar_w, rows)
        assert _bracket_lines(ax) == []
        assert "ns" not in _annotation_texts(ax)
    finally:
        plt.close(fig)


def _text_tops_near(ax, fig, x_centre, half_width):
    """Data-space tops of every value label sitting over one bar pair."""
    fig.canvas.draw()
    inv = ax.transData.inverted()
    tops = []
    for t in ax.texts:
        bb = t.get_window_extent(fig.canvas.get_renderer())
        (x0, _), (x1, y1) = inv.transform(((bb.x0, bb.y0), (bb.x1, bb.y1)))
        if x0 <= x_centre + half_width and x1 >= x_centre - half_width:
            if t.get_text().replace(".", "").isdigit():
                tops.append(y1)
    return tops


def test_score_tvc_bracket_clears_the_value_labels():
    """The rotated means used to be struck through by the bracket above them."""
    rows = _tvc_rows()
    rows["score_p_value"] = [0.0004, 0.9, 0.9, 0.9]
    fig, ax = plt.subplots()
    try:
        x, bar_w = score_summary.plot_score_train_vs_control(
            ax, rows, title="t", label="Hexanol"
        )
        score_summary._draw_score_significance_brackets(ax, x, bar_w, rows)
        bracket = _bracket_lines(ax)[0]
        bracket_bottom = min(bracket.get_ydata())
        tops = _text_tops_near(ax, fig, x[0], bar_w)
        assert tops, "no value labels found over the significant pair"
        assert bracket_bottom > max(tops), (
            f"bracket at {bracket_bottom:.3f} sits on a label topping {max(tops):.3f}"
        )
    finally:
        plt.close(fig)


def test_matrix_bars_annotate_only_the_significant_pairs():
    from scripts.analysis.reaction_matrix_training_vs_control import (
        plot_training_vs_control_bars,
    )

    odors = ["Hexanol", "Citral"]
    train = pd.DataFrame({
        "odor": odors, "rate": [64.0, 20.0],
        "num_trials": [11, 11], "is_trained": [True, False],
    })
    ctrl = pd.DataFrame({
        "odor": odors, "rate": [16.0, 18.0], "num_trials": [19, 19],
    })
    fig, ax = plt.subplots()
    try:
        plot_training_vs_control_bars(
            ax, train, ctrl, title="t",
            p_values={(0, "Hexanol"): 0.004, (0, "Citral"): 0.47},
        )
        assert len(_bracket_lines(ax)) == 1
        texts = _annotation_texts(ax)
        assert "**" in texts, texts
        assert not any("p=" in t for t in texts), texts
        assert "ns" not in texts, texts
    finally:
        plt.close(fig)


def test_matrix_bracket_clears_the_percent_labels():
    from scripts.analysis.reaction_matrix_training_vs_control import (
        plot_training_vs_control_bars,
    )

    train = pd.DataFrame({
        "odor": ["Hexanol"], "rate": [64.0], "num_trials": [11], "is_trained": [True],
    })
    ctrl = pd.DataFrame({"odor": ["Hexanol"], "rate": [16.0], "num_trials": [19]})
    fig, ax = plt.subplots()
    try:
        plot_training_vs_control_bars(
            ax, train, ctrl, title="t", p_values={(0, "Hexanol"): 0.004},
        )
        bracket = _bracket_lines(ax)[0]
        fig.canvas.draw()
        inv = ax.transData.inverted()
        tops = []
        for t in ax.texts:
            if t.get_text() in {"**", "*", "***"}:
                continue
            bb = t.get_window_extent(fig.canvas.get_renderer())
            tops.append(inv.transform((bb.x1, bb.y1))[1])
        assert min(bracket.get_ydata()) > max(tops)
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# n lives in the legend, not on every bar
# ---------------------------------------------------------------------------

def test_matrix_single_cohort_bars_print_the_rate_without_the_n():
    from scripts.analysis.envelope_visuals import plot_reaction_rate_bars

    stats = pd.DataFrame({
        "odor": ["Hexanol", "Citral"],
        "rate": [0.62, 0.31],
        "num_trials": [13, 13],
        "is_trained": [True, False],
        "trial_num": [1, 2],
    })
    fig, ax = plt.subplots()
    try:
        plot_reaction_rate_bars(ax, stats, title="t")
        texts = [t.get_text() for t in ax.texts]
        assert sorted(texts) == ["31%", "62%"], texts
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert labels == ["Training (n=13)"], labels
    finally:
        plt.close(fig)


def test_matrix_tvc_bars_print_the_rate_without_the_n():
    from scripts.analysis.reaction_matrix_training_vs_control import (
        plot_training_vs_control_bars,
    )

    odors = ["Hexanol", "Citral"]
    train = pd.DataFrame({
        "odor": odors, "rate": [64.0, 20.0],
        "num_trials": [11, 11], "is_trained": [True, False],
    })
    ctrl = pd.DataFrame({"odor": odors, "rate": [16.0, 18.0], "num_trials": [19, 19]})
    fig, ax = plt.subplots()
    try:
        plot_training_vs_control_bars(ax, train, ctrl, title="t")
        texts = [t.get_text() for t in ax.texts]
        assert not any("n=" in t for t in texts), texts
        assert sorted(texts) == ["16%", "18%", "20%", "64%"], texts
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert labels == ["Training (n=11)", "Control (n=19)"], labels
    finally:
        plt.close(fig)


def test_rig_batch_puts_n_in_the_legend():
    from scripts.analysis import rig_batch_breakdowns as rbb

    stats_a = pd.DataFrame({"mean_score": [3.0, 1.0], "sem_score": [0.2, 0.2],
                            "n_flies": [8, 8]})
    stats_b = pd.DataFrame({"mean_score": [1.0, 0.9], "sem_score": [0.2, 0.2],
                            "n_flies": [9, 9]})
    fig, ax = plt.subplots()
    try:
        rbb.plot_score_comparison_bars(
            ax, ["Hexanol", "Citral"], [True, False], stats_a, stats_b,
            label_a="Rig 2", label_b="Rig 3", title="t",
        )
        ticks = [t.get_text() for t in ax.get_xticklabels()]
        assert not any("n=" in t for t in ticks), ticks
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert labels == ["Rig 2 (n=8)", "Rig 3 (n=9)"], labels
    finally:
        plt.close(fig)
