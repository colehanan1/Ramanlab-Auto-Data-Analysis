"""Tests for the one-line, plain-words statement of the PER trace.

The sheet says: PER % = distance from eye to proboscis x f(angle), with f on a
log scale between 1 and 2 for positive angles. It exists to be read by somebody
who will never open the code, so the risk is not that it crashes -- it is that
it drifts from the pipeline and quietly teaches the wrong thing. The curve and
both bounds it prints are therefore pinned to
``envelope_combined._angle_multiplier`` (the function the pipeline calls) and
the axis it names to ``envelope_visuals._envelope_ylabel``, never to literals
typed into the figure.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis import distance_angle_multiplier_simple as fig_mod  # noqa: E402
from scripts.analysis import envelope_combined as ec  # noqa: E402
from scripts.analysis import envelope_visuals as ev  # noqa: E402


@pytest.fixture(scope="module")
def figure():
    fig = fig_mod.build_figure()
    yield fig
    fig_mod.plt.close(fig)


def _all_text(fig) -> str:
    texts = [t.get_text() for t in fig.findobj(match=lambda o: hasattr(o, "get_text"))]
    return "\n".join(texts)


# ── the curve is the production function, not a lookalike ────────────────────
def test_multiplier_curve_is_the_production_function():
    angle_pct, mult = fig_mod.multiplier_curve()
    assert np.allclose(mult, ec._angle_multiplier(angle_pct))


def test_curve_spans_the_full_clipped_angle_range():
    angle_pct, _ = fig_mod.multiplier_curve()
    assert angle_pct.min() == -100.0
    assert angle_pct.max() == 100.0


def test_negative_angles_never_attenuate():
    """The claim the sheet makes about f: only positive angles do anything."""
    angle_pct, mult = fig_mod.multiplier_curve()
    assert np.all(mult[angle_pct <= 0.0] == fig_mod.MULT_MIN)
    assert np.all(mult >= fig_mod.MULT_MIN)


def test_curve_is_bounded_by_the_printed_limits():
    _, mult = fig_mod.multiplier_curve()
    assert mult.max() == pytest.approx(fig_mod.MULT_MAX)


def test_curve_is_log_shaped_not_linear():
    """Half the angle must already buy more than half the gain."""
    angle_pct, mult = fig_mod.multiplier_curve()
    half = float(np.interp(50.0, angle_pct, mult))
    midpoint = (fig_mod.MULT_MIN + fig_mod.MULT_MAX) / 2.0
    assert half > midpoint
    assert np.all(np.diff(mult[angle_pct >= 0.0]) >= 0.0)


# ── the printed bounds are read from the pipeline ────────────────────────────
def test_bounds_match_production_multiplier():
    assert fig_mod.MULT_MIN == pytest.approx(float(ec._angle_multiplier(np.array([0.0]))[0]))
    assert fig_mod.MULT_MAX == pytest.approx(float(ec._angle_multiplier(np.array([100.0]))[0]))


def test_bounds_appear_on_the_sheet(figure):
    text = _all_text(figure)
    assert f"{fig_mod.MULT_MIN:.0f} → {fig_mod.MULT_MAX:.0f} on a log scale" in text


# ── the sheet reads as the one line it promises ──────────────────────────────
def test_equation_reads_as_one_line(figure):
    text = _all_text(figure)
    for term in fig_mod.EQUATION_TERMS:
        assert term.text in text
    spelled = " ".join(term.text for term in fig_mod.EQUATION_TERMS)
    assert spelled == "PER % = distance from eye to proboscis × f(angle)"


def test_equation_terms_are_laid_out_left_to_right():
    xs = [term.x for term in fig_mod.EQUATION_TERMS]
    assert xs == sorted(xs)


def test_axis_name_matches_the_trace_figures(figure):
    """The sheet names the y-axis it explains, spelled as production spells it."""
    produced = ev._envelope_ylabel(
        SimpleNamespace(y_label_override=None, matrix_npy="combined_base.npy")
    )
    assert produced in _all_text(figure)


def test_takeaway_states_the_direction_of_the_effect(figure):
    assert fig_mod.TAKEAWAY in _all_text(figure)
    assert "positive angle" in fig_mod.TAKEAWAY
    assert "the other way" in fig_mod.TAKEAWAY


def test_sheet_carries_words_not_symbols(figure):
    """The maths lives on the other sheet; this one stays readable."""
    text = _all_text(figure)
    for token in ("ln(", "$", "∑", "\\", "_{"):
        assert token not in text


# ── rendering ────────────────────────────────────────────────────────────────
def test_render_writes_png_and_svg(tmp_path):
    written = fig_mod.render(out_dir=tmp_path)
    assert {p.suffix for p in written} == {".png", ".svg"}
    for path in written:
        assert path.exists() and path.stat().st_size > 0


def test_render_names_files_from_the_stem(tmp_path):
    written = fig_mod.render(out_dir=tmp_path)
    assert {p.stem for p in written} == {fig_mod.STEM}
