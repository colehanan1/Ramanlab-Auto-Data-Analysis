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


# ── phase wording: *-Sensitivity-* cohorts are pre-/post-exposure ─────


def _texts(fig) -> str:
    """Every string drawn on the figure, plus the unwrapped question and answer."""
    from matplotlib.text import Text

    fig.canvas.draw()
    return "\n".join([t.get_text() for t in fig.findobj(Text)] + [fig.answer])


def _regression():
    """A minimal odor_holdout_regression table: each series, pre phase, every odor."""
    rows = [(series, odor, "pre", 4, 0.2, 0.05)
            for series in ("single fly, from its other odors", "batch average, from its other odors",
                           "single fly, from starvation time only")
            for odor in (IAA, "Hexanol", "Citral")]
    return pd.DataFrame(rows, columns=["series", "odor", "phase", "n", "q2", "p_perm"])


def _figure_texts(paired, monkeypatch) -> dict[str, str]:
    """Text of all six figures; Q6's raw-data lookup is stubbed (no batch folders)."""
    import sys
    import types
    from pathlib import Path

    hour = {f: 9 + i for i, f in enumerate(sorted(paired["fly"].unique()))}
    starts = paired.assign(start_time=paired["fly"].map(lambda f: pd.Timestamp(2026, 10, 8, hour[f])))
    tod = types.ModuleType("scripts.analysis.time_of_day_trend")
    tod.attach_start_times = lambda p, root: (starts, [])
    tod.clock_hours = lambda t: t.hour + t.minute / 60
    monkeypatch.setitem(sys.modules, "scripts.analysis.time_of_day_trend", tod)
    figs = {"direction": sf.fig_direction(paired, trained=IAA),
            "specificity": sf.fig_specificity(paired, trained=IAA),
            "change_map": sf.fig_change_map(paired, trained=IAA),
            "baseline": sf.fig_baseline(paired),
            "responsiveness": sf.fig_responsiveness(paired, regression=_regression()),
            "time_of_day": sf.fig_time_of_day(paired, data_root=Path("unused"))}
    texts = {k: _texts(f) for k, f in figs.items()}
    for f in figs.values():
        plt.close(f)
    return texts


def test_sensitivity_cohorts_say_sensitized_odor_and_pre_sensitization(monkeypatch):
    p = _paired()
    p["dataset_canon"] = p["group"].map({"1 h": "IAA-Sensitivity-1-1", "24 h": "IAA-Sensitivity-24-1"})
    t = _figure_texts(p, monkeypatch)
    assert "Did exposure raise the PER to the sensitized odor (Isoamyl Acetate)?" in t["direction"]
    assert "Isoamyl Acetate — the sensitized odor" in t["direction"]
    assert "PER went up after exposure" in t["direction"]
    assert "Was the rise specific to the sensitized odor?" in t["specificity"]
    assert "↑ rose more for the sensitized odor" in t["specificity"]
    assert "Isoamyl Acetate  (sensitized odor)" in t["change_map"]
    assert "Where did PER change after exposure?" in t["change_map"]
    assert "Before exposure, does starvation change how much flies respond?" in t["baseline"]
    assert "Pre-exposure PER score, all odors" in t["baseline"]
    assert "the average pre-exposure score is" in t["baseline"]
    assert "Pre-exposure, every fly × odor" in t["responsiveness"]
    assert "Predicting the pre-exposure score" in t["responsiveness"]
    assert "Pre-exposure: the other odors predict" in t["responsiveness"]
    assert "first pre-exposure trial" in t["time_of_day"]
    assert "Before exposure, does the time of day" in t["time_of_day"]
    for text in t.values():
        for old in ("trained odor", "(trained)", "pre-test", "Pre-test", "after training",
                    "Before training"):
            assert old not in text, old


@pytest.mark.parametrize("datasets", [
    {"1 h": "Hex-HexTest-Training-32-1", "24 h": "Hex-HexTest-Training-32-1"},
    {"1 h": "IAA-Sensitivity-1-1", "24 h": "Hex-HexTest-Training-24-1"},  # a mix keeps it too
], ids=["conditioning", "mixed"])
def test_conditioning_cohorts_keep_trained_odor_and_pre_test(monkeypatch, datasets):
    p = _paired()
    p["dataset_canon"] = p["group"].map(datasets)
    t = _figure_texts(p, monkeypatch)
    assert "Did training raise the PER to the trained odor (Isoamyl Acetate)?" in t["direction"]
    assert "Isoamyl Acetate — the trained odor" in t["direction"]
    assert "PER went up after training" in t["direction"]
    assert "Was the rise specific to the trained odor?" in t["specificity"]
    assert "Isoamyl Acetate  (trained)" in t["change_map"]
    assert "Where did PER change after training?" in t["change_map"]
    assert "Before training, does starvation change how much flies respond?" in t["baseline"]
    assert "Pre-test PER score, all odors" in t["baseline"]
    assert "Pre-test, every fly × odor" in t["responsiveness"]
    assert "Predicting the pre-test score" in t["responsiveness"]
    assert "first pre-test trial" in t["time_of_day"]
    assert not any(w in text.lower() for w in ("sensitiz", "exposure") for text in t.values())
