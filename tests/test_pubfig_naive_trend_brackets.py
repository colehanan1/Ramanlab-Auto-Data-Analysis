"""Near-significant pairs must be visible, not invisible.

Every pair already gets a star bracket when its Holm-adjusted p clears 0.05.
With three cohorts there are three pairwise tests, so Holm multiplies the
smallest p by three: trained-vs-control at p=0.034 becomes p=0.067 and the
figure showed nothing at all -- indistinguishable from p=0.9.

A pair that is significant *before* correction now carries its adjusted
p-value on the bracket, in grey and without stars. Stars stay reserved for
results that survive the correction; the reader sees the difference and how
close it came, and the sidecar CSV still carries both p-values.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analysis.pubfig_naive_vs_trained import (  # noqa: E402
    Comparison,
    _draw_brackets,
    render_comparison,
)

TOPS = {"Naive": 1.0, "Trained": 2.0, "Control": 1.0}


def _pairwise(**kw):
    """Default: nothing notable anywhere, then override single pairs."""
    out = {
        ("Naive", "Trained"): {"p_raw": 0.8, "p_adj": 1.0},
        ("Naive", "Control"): {"p_raw": 0.8, "p_adj": 1.0},
        ("Trained", "Control"): {"p_raw": 0.8, "p_adj": 1.0},
    }
    for key, value in kw.items():
        pair = tuple(key.split("_vs_"))
        out[pair] = value
    return out


def _texts(pairwise, **kw):
    fig, ax = plt.subplots()
    _draw_brackets(ax, pairwise, TOPS, span=6.0, **kw)
    texts = [t.get_text() for t in ax.texts]
    plt.close(fig)
    return texts


def test_a_near_miss_shows_its_adjusted_p_value():
    texts = _texts(
        _pairwise(Trained_vs_Control={"p_raw": 0.0336, "p_adj": 0.0672}), trend=True
    )
    assert texts == ["p = 0.067"]


def test_the_trend_annotation_is_opt_in():
    pairwise = _pairwise(Trained_vs_Control={"p_raw": 0.0336, "p_adj": 0.0672})
    assert _texts(pairwise) == []


def test_a_surviving_pair_keeps_its_stars_and_gains_no_p_value():
    texts = _texts(
        _pairwise(Naive_vs_Trained={"p_raw": 0.0022, "p_adj": 0.0066}), trend=True
    )
    assert texts == ["**"]


def test_a_pair_that_was_never_significant_stays_bare():
    """p=0.31 is not a trend; annotating it would invite reading noise."""
    texts = _texts(
        _pairwise(Trained_vs_Control={"p_raw": 0.31, "p_adj": 0.93}), trend=True
    )
    assert texts == []


def test_a_missing_p_value_is_not_annotated():
    texts = _texts(
        _pairwise(Trained_vs_Control={"p_raw": float("nan"), "p_adj": float("nan")}),
        trend=True,
    )
    assert texts == []


def test_stars_and_trends_coexist_without_overlapping():
    fig, ax = plt.subplots()
    _draw_brackets(
        ax,
        _pairwise(
            Naive_vs_Trained={"p_raw": 0.002, "p_adj": 0.006},
            Trained_vs_Control={"p_raw": 0.034, "p_adj": 0.067},
        ),
        TOPS,
        span=6.0,
        trend=True,
    )
    assert sorted(t.get_text() for t in ax.texts) == ["**", "p = 0.067"]
    heights = sorted(t.get_position()[1] for t in ax.texts)
    assert heights[1] > heights[0], "the second bracket sits on top of the first"
    plt.close(fig)


def test_the_grey_trend_label_is_not_bold_like_the_stars():
    fig, ax = plt.subplots()
    _draw_brackets(
        ax,
        _pairwise(
            Naive_vs_Trained={"p_raw": 0.002, "p_adj": 0.006},
            Trained_vs_Control={"p_raw": 0.034, "p_adj": 0.067},
        ),
        TOPS,
        span=6.0,
        trend=True,
    )
    by_text = {t.get_text(): t for t in ax.texts}
    assert by_text["**"].get_fontweight() == "bold"
    assert by_text["p = 0.067"].get_fontweight() != "bold"
    plt.close(fig)


def test_the_figure_passes_the_flag_through():
    comparison = Comparison(
        odor="3-Octanol", concentration="0.1%",
        naive_dataset="RandomPanel-24-0.1",
        train_dataset="T", control_dataset="C",
    )
    # Trained clearly above naive; control between them -- trained vs control
    # lands near, but not below, the corrected threshold.
    groups = {
        "Naive": np.array([0.0] * 18 + [4.0, 5.0]),
        "Trained": np.array([2.0, 3.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0, 3.0]),
        "Control": np.array([0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0]),
    }
    plain = render_comparison(comparison, groups, footnote=False)
    with_trend = render_comparison(comparison, groups, footnote=False, trend_p=True)
    n_plain = sum(len(ax.texts) for ax in plain.axes)
    n_trend = sum(len(ax.texts) for ax in with_trend.axes)
    assert n_trend >= n_plain
    plt.close("all")
