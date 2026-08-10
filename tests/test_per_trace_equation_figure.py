"""Tests for the raw-PER-trace equation + symbol figures.

Two sheets are rendered: one carrying only the seven equations, one defining
every symbol in them. Their whole value is that the printed maths agrees with
the production code, so these tests pin the constants and the multiplier curve
to ``envelope_combined`` itself rather than to literals typed into the figure,
and they keep the two sheets from bleeding into each other.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis import envelope_combined as ec  # noqa: E402
from scripts.analysis import per_trace_equation_figure as fig_mod  # noqa: E402
from fbpipe.config import load_raw_config  # noqa: E402
from fbpipe.utils.columns import EYE_CLASS, PROBOSCIS_CLASS  # noqa: E402


@pytest.fixture(scope="module")
def settings():
    return fig_mod.load_settings(fig_mod.CONFIG_PATH)


def _all_text(fig) -> str:
    texts = [t.get_text() for t in fig.findobj(match=lambda o: hasattr(o, "get_text"))]
    return "\n".join(texts)


# ── constants come from the config / production module, never from literals ──
def test_anchor_matches_config(settings):
    raw = load_raw_config(fig_mod.CONFIG_PATH)
    assert settings.anchor_x == float(raw["anchor_x"])
    assert settings.anchor_y == float(raw["anchor_y"])


def test_normalization_floor_matches_production(settings):
    assert settings.norm_floor_px == ec.WEIGHTED_EFFECTIVE_MAX_FLOOR


def test_gate_values_match_config(settings):
    raw = load_raw_config(fig_mod.CONFIG_PATH)
    limits = raw["distance_limits"]
    assert settings.norm_min_px == float(limits["class2_min"])
    assert settings.norm_max_px == float(limits["class2_max"])


def test_measure_columns_match_config(settings):
    raw = load_raw_config(fig_mod.CONFIG_PATH)
    wide = raw["analysis"]["combined"]["combined_base"]["wide"]
    assert settings.measure_cols == tuple(wide["measure_cols"])


def test_class_ids_match_pipeline(settings):
    assert settings.eye_class == EYE_CLASS
    assert settings.proboscis_class == PROBOSCIS_CLASS


# ── the drawn multiplier curve is the production function, not a redraw ──
def test_multiplier_curve_is_production_function():
    angle_pct, multiplier = fig_mod.multiplier_curve()
    np.testing.assert_allclose(multiplier, ec._angle_multiplier(angle_pct))
    assert multiplier[angle_pct <= 0].max() == 1.0
    np.testing.assert_allclose(multiplier[-1], 2.0)


def test_multiplier_curve_spans_full_angle_range():
    angle_pct, _ = fig_mod.multiplier_curve()
    assert angle_pct[0] == -100.0
    assert angle_pct[-1] == 100.0


# ── sheet 1: equations only ──
def test_equation_sheet_renders_every_step(settings):
    fig = fig_mod.build_equation_figure(settings)
    text = _all_text(fig)
    for step in fig_mod.steps(settings):
        assert step.equation in text, f"missing equation for step {step.number}"
        assert step.title.upper() in text, f"missing title for step {step.number}"
    fig_mod.plt.close(fig)


def test_equation_sheet_carries_no_symbol_prose(settings):
    """The glossary lives on the other sheet; this one stays equations."""
    fig = fig_mod.build_equation_figure(settings)
    text = _all_text(fig)
    for symbol in fig_mod.symbols(settings):
        assert symbol.meaning not in text, f"prose leaked onto the equation sheet: {symbol.name}"
    fig_mod.plt.close(fig)


def test_final_equation_is_the_plotted_column(settings):
    final = fig_mod.steps(settings)[-1]
    assert "PER" in final.equation
    assert _fmt_floor(settings) in final.equation


def _fmt_floor(settings) -> str:
    return f"{settings.norm_floor_px:g}"


# ── sheet 2: symbol key ──
def test_symbol_sheet_defines_every_core_symbol(settings):
    fig = fig_mod.build_symbol_figure(settings)
    text = _all_text(fig)
    for symbol in fig_mod.symbols(settings):
        assert symbol.latex in text, f"missing symbol {symbol.name}"
        assert symbol.meaning in text, f"missing meaning for {symbol.name}"
    fig_mod.plt.close(fig)


@pytest.mark.parametrize(
    "token",
    ["d(t)", r"\theta(t)", r"\Delta\theta(t)", "A(t)", "m(t)", "d_w(t)", r"\mathrm{PER}(t)"],
)
def test_every_equation_variable_has_a_glossary_row(settings, token):
    latex = [s.latex for s in fig_mod.symbols(settings)]
    assert any(token in entry for entry in latex), f"{token} is used but never defined"


def test_symbol_sheet_reports_the_config_constants(settings):
    fig = fig_mod.build_symbol_figure(settings)
    text = _all_text(fig)
    for token in ("1079", "540", "150", "config_new.yaml", "combined_pct"):
        assert token in text, f"{token} missing from the symbol sheet"
    fig_mod.plt.close(fig)


def test_symbol_sheet_states_raw_trace_is_unsmoothed(settings):
    """RMS + Hilbert exist in the code but are NOT applied to the raw trace."""
    fig = fig_mod.build_symbol_figure(settings)
    text = _all_text(fig).lower()
    assert "rms" in text and "hilbert" in text
    assert "not applied" in text
    fig_mod.plt.close(fig)


# ── output plumbing ──
@pytest.mark.parametrize("builder", ["build_equation_figure", "build_symbol_figure"])
def test_svg_keeps_text_editable(tmp_path, settings, builder):
    fig = getattr(fig_mod, builder)(settings)
    out = tmp_path / f"{builder}.svg"
    fig.savefig(out)
    fig_mod.plt.close(fig)
    body = out.read_text()
    assert "<text" in body
    assert "PER" in body


def test_main_writes_both_sheets_in_all_formats(tmp_path):
    rc = fig_mod.main(["--out-dir", str(tmp_path)])
    assert rc == 0
    for stem in (fig_mod.STEM_EQUATIONS, fig_mod.STEM_SYMBOLS):
        for ext in ("png", "svg", "pdf"):
            path = tmp_path / f"{stem}.{ext}"
            assert path.exists() and path.stat().st_size > 0


def test_the_two_stems_are_distinct_files():
    assert fig_mod.STEM_EQUATIONS != fig_mod.STEM_SYMBOLS
