"""Plain-language findings figures for a starvation comparison
(scripts/analysis/starvation_findings_figures.py)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import scripts.analysis.starvation_findings_figures as sf

IAA = "Isoamyl Acetate"


def _paired():
    """Two groups x 4 flies x 3 odors. 1 h: IAA goes up in 3 of 4 flies, others flat."""
    rows = []
    for group, iaa in (("1 h", [(0, 2), (0, 3), (1, 2), (2, 2)]),
                       ("24 h", [(1, 0), (0, 0), (2, 1), (1, 1)])):
        for i, (pre, post) in enumerate(iaa):
            fly = f"{group}-f{i}"
            rows.append((group, fly, IAA, pre, post))
            rows.append((group, fly, "Hexanol", 1, 1))
            rows.append((group, fly, "Citral", 0, 1 if i == 0 else 0))
    df = pd.DataFrame(rows, columns=["group", "fly", "odor", "score_pre", "score_post"])
    df["dataset_canon"] = "D"
    df["fly_number"] = 1
    return df


def test_groups_sort_by_hours_not_text():
    assert sf.group_order(["24 h", "1 h", "6 h", "12 h"]) == ["1 h", "6 h", "12 h", "24 h"]


def test_direction_shares_up_same_down():
    d = sf.direction_shares(_paired(), odors=[IAA])
    one = d.loc["1 h"]
    assert one["n"] == 4
    assert (one["up"], one["same"], one["down"]) == (75.0, 25.0, 0.0)
    assert d.loc["24 h", "down"] == 50.0
    # rows follow starvation order
    assert list(d.index) == ["1 h", "24 h"]


def test_direction_shares_for_other_odors_counts_every_fly_odor_pair():
    d = sf.direction_shares(_paired(), odors=["Hexanol", "Citral"])
    assert d.loc["1 h", "n"] == 8
    assert d.loc["1 h", "up"] == pytest.approx(100 / 8)


def test_specificity_index_is_trained_change_minus_mean_other_change():
    s = sf.specificity(_paired(), trained=IAA)
    f0 = s[s["fly"] == "1 h-f0"].iloc[0]
    # IAA +2; others: Hexanol 0, Citral +1 -> mean 0.5 -> index 1.5
    assert f0["index"] == pytest.approx(1.5)
    assert len(s) == 8


def test_boot_ci_brackets_the_mean_and_is_reproducible():
    v = np.array([0, 1, 2, 3, 4, 5], dtype=float)
    lo, hi = sf.boot_ci(v)
    assert lo < v.mean() < hi
    assert sf.boot_ci(v) == (lo, hi)
    assert sf.boot_ci(np.array([2.0])) == (2.0, 2.0)


def test_change_matrix_has_mean_change_and_paired_p():
    m, p = sf.change_matrix(_paired(), trained=IAA)
    assert list(m.columns) == ["1 h", "24 h"]
    assert m.index[0] == IAA  # trained odor first
    assert m.loc[IAA, "1 h"] == pytest.approx(1.5)
    assert m.loc["Hexanol", "1 h"] == 0.0
    assert p.shape == m.shape


def test_baseline_uses_pooled_trials_and_per_fly_ci():
    b = sf.baseline(_paired()).set_index("group")
    # 1 h pre: IAA 0,0,1,2 + Hexanol 1x4 + Citral 0x4 = 7 / 12
    assert b.loc["1 h", "mean_pre"] == pytest.approx(7 / 12)
    assert b.loc["1 h", "n_flies"] == 4
    assert b.loc["1 h", "ci_lo"] <= b.loc["1 h", "mean_pre"] <= b.loc["1 h", "ci_hi"]


def test_responsiveness_bins_relate_other_odors_to_held_out_odor():
    r = sf.responsiveness(_paired())
    assert {"bin", "n", "pct_responding"} <= set(r.columns)
    assert r["n"].sum() == 8 * 3  # every fly x odor pair, pre-test


def test_all_figures_build_with_question_titles(tmp_path):
    p = _paired()
    figs = [sf.fig_direction(p, trained=IAA), sf.fig_specificity(p, trained=IAA),
            sf.fig_change_map(p, trained=IAA), sf.fig_baseline(p),
            sf.fig_responsiveness(p, regression=None)]
    for fig in figs:
        assert fig._suptitle is not None and "?" in fig._suptitle.get_text()
        plt.close(fig)


def test_time_blocks_split_on_gaps_not_clock_bins():
    clock = pd.Series([9.5, 10.9, 13.2, 17.7, 18.4, 18.8, 10.1])
    b = sf.time_blocks(clock)
    assert b.tolist() == [0, 0, 1, 2, 2, 2, 0]
