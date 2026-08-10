"""Every Hexanol and ACV exposure gets its own bar, not just the first.

``score_bars_variants`` collapses each odor to its first presentation. This
figure keeps all of them: ACV at trials 1 and 3, Hexanol at trials 2, 4 and 5.
The tests below guard the three things that make that legible — one bar pair
per presentation, exposure-numbered tick labels, and a p-value keyed to the
presentation rather than to the odor.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts.analysis import score_bars_all_exposures as sba


ACV = "Apple Cider Vinegar"
HEX = "Hexanol"


def _stats(rows):
    """(trial_num, odor, mean, sem) -> a summary frame."""
    return pd.DataFrame(
        [
            {
                "trial_num": t,
                "odor": o,
                "mean_score": m,
                "sem_score": s,
                "n_flies": 13,
                "is_trained": o == HEX,
            }
            for t, o, m, s in rows
        ]
    )


def _octnov_stats():
    """The real Oct/Nov testing panel: ACV at 1 and 3, Hexanol at 2, 4 and 5."""
    return _stats(
        [
            (1, ACV, -0.31, 0.13),
            (2, HEX, 3.23, 0.63),
            (3, ACV, -0.23, 0.12),
            (4, HEX, 3.77, 0.52),
            (5, HEX, 3.62, 0.62),
            (6, "Benzaldehyde", 0.15, 0.25),
            (7, "3-Octanol", 1.15, 0.55),
        ]
    )


# --- subset ---------------------------------------------------------------

def test_hex_and_acv_keep_every_presentation():
    kept = sba._keep_hex_acv(_octnov_stats())
    assert list(zip(kept["trial_num"], kept["odor"])) == [
        (1, ACV), (2, HEX), (3, ACV), (4, HEX), (5, HEX)
    ]


def test_other_odors_are_dropped():
    kept = sba._keep_hex_acv(_octnov_stats())
    assert set(kept["odor"]) == {ACV, HEX}


# --- exposure numbering ---------------------------------------------------

def test_exposures_are_numbered_per_odor():
    kept = sba._keep_hex_acv(_octnov_stats())
    assert sba._exposure_labels(kept) == [
        f"{ACV} 1", f"{HEX} 1", f"{ACV} 2", f"{HEX} 2", f"{HEX} 3"
    ]


def test_exposure_number_follows_trial_order_not_row_order():
    """Numbering must survive the bars being re-sorted for display."""
    kept = sba._keep_hex_acv(_octnov_stats())
    shuffled = kept.iloc[::-1].reset_index(drop=True)
    assert sba._exposure_labels(shuffled) == [
        f"{HEX} 3", f"{HEX} 2", f"{ACV} 2", f"{HEX} 1", f"{ACV} 1"
    ]


# --- bar ordering ---------------------------------------------------------

def test_odor_grouped_order_puts_both_acv_bars_before_the_hexanol_block():
    ordered = sba._order_bars(sba._keep_hex_acv(_octnov_stats()), group_by_odor=True)
    assert list(ordered["odor"]) == [ACV, ACV, HEX, HEX, HEX]
    assert list(ordered["trial_num"]) == [1, 3, 2, 4, 5]


def test_odor_grouped_order_keeps_the_first_presented_odor_first():
    """ACV leads because it is presented first, not because of its name."""
    hex_first = _stats([(1, HEX, 3.2, 0.6), (2, ACV, -0.3, 0.1), (3, HEX, 3.8, 0.5)])
    ordered = sba._order_bars(hex_first, group_by_odor=True)
    assert list(ordered["odor"]) == [HEX, HEX, ACV]


def test_presentation_order_interleaves_the_two_odors():
    ordered = sba._order_bars(sba._keep_hex_acv(_octnov_stats()), group_by_odor=False)
    assert list(ordered["trial_num"]) == [1, 2, 3, 4, 5]
    assert list(ordered["odor"]) == [ACV, HEX, ACV, HEX, HEX]


# --- statistics -----------------------------------------------------------

def _fly_scores(rows):
    return pd.DataFrame(
        [
            {"trial_num": t, "odor": o, "fly": f"fly_{i}", "fly_mean_score": v}
            for t, o, vals in rows
            for i, v in enumerate(vals)
        ]
    )


def test_each_hexanol_exposure_is_tested_separately():
    """A significant exposure must not borrow its p-value from another one."""
    train = _fly_scores([(2, HEX, [5, 5, 5, 5]), (4, HEX, [0, 0, 0, 0])])
    ctrl = _fly_scores([(2, HEX, [0, 0, 0, 0]), (4, HEX, [0, 0, 0, 0])])
    p = sba._mannwhitney_per_presentation(train, ctrl, [(2, HEX), (4, HEX)])
    assert p[(2, HEX)] < 0.05
    assert not (p[(4, HEX)] < 0.05)


def test_missing_cohort_yields_no_bracket():
    p = sba._mannwhitney_per_presentation(
        _fly_scores([(2, HEX, [5, 5, 5])]), _fly_scores([]), [(2, HEX)]
    )
    assert sba._stars(p[(2, HEX)]) == ""


@pytest.mark.parametrize(
    "p,expected",
    [(0.0004, "***"), (0.005, "**"), (0.04, "*"), (0.2, ""), (float("nan"), "")],
)
def test_star_thresholds(p, expected):
    assert sba._stars(p) == expected


# --- rendering ------------------------------------------------------------

def _rendered_axes():
    kept = sba._keep_hex_acv(_octnov_stats())
    train = sba._order_bars(kept, group_by_odor=True)
    ctrl = train.assign(mean_score=0.1, sem_score=0.2, n_flies=15)
    fig, ax = plt.subplots()
    sba._plot_train_vs_ctrl(
        ax,
        train,
        ctrl,
        title="test",
        p_values={(t, o): 0.0001 for t, o in zip(train["trial_num"], train["odor"])},
    )
    return fig, ax


def test_five_bar_pairs_are_drawn():
    fig, ax = _rendered_axes()
    try:
        assert len(ax.get_xticklabels()) == 5
        assert [t.get_text() for t in ax.get_xticklabels()] == [
            f"{ACV} 1", f"{ACV} 2", f"{HEX} 1", f"{HEX} 2", f"{HEX} 3"
        ]
    finally:
        plt.close(fig)


def test_every_hexanol_tick_is_bold_and_green():
    fig, ax = _rendered_axes()
    try:
        from matplotlib.colors import to_hex

        from scripts.analysis import odor_bar_palette as pal

        hex_ticks = [t for t in ax.get_xticklabels() if t.get_text().startswith(HEX)]
        assert len(hex_ticks) == 3
        for tick in hex_ticks:
            assert tick.get_weight() == "bold"
            assert to_hex(tick.get_color()) == to_hex(pal.trained_tick_color(HEX))
    finally:
        plt.close(fig)


def _bracket_tops(ax):
    return [max(ln.get_ydata()) for ln in ax.get_lines() if len(ln.get_ydata()) == 4]


def test_significance_brackets_stay_inside_the_axes():
    """The tallest Hexanol bar sits at 4.3; its bracket must not run off-axis."""
    fig, ax = _rendered_axes()
    try:
        _, top = ax.get_ylim()
        tops = _bracket_tops(ax)
        assert tops, "no significance brackets drawn"
        assert max(tops) < top
    finally:
        plt.close(fig)


def test_brackets_share_one_height():
    """Five pairs of differing height would otherwise staircase across the panel."""
    fig, ax = _rendered_axes()
    try:
        tops = _bracket_tops(ax)
        assert len(tops) == 5
        assert max(tops) == pytest.approx(min(tops))
    finally:
        plt.close(fig)


def test_brackets_clear_the_tallest_value_annotation():
    """The bracket row sits above every "3.77"-style label, not through them."""
    fig, ax = _rendered_axes()
    try:
        annotations = [
            t.get_position()[1] for t in ax.texts if t.get_text().replace(
                "-", "").replace(".", "").isdigit()
        ]
        assert annotations
        assert min(_bracket_tops(ax)) > max(annotations) + 0.2
    finally:
        plt.close(fig)


def test_y_axis_is_ticked_over_the_full_score_range():
    fig, ax = _rendered_axes()
    try:
        assert list(ax.get_yticks()) == [-1, 0, 1, 2, 3, 4, 5]
    finally:
        plt.close(fig)
