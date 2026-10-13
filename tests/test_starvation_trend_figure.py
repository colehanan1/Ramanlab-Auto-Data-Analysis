"""Starvation trend figure (scripts/analysis/starvation_trend_figure.py).

Reads a starvation comparison folder (paired_scores_per_fly.csv + stats_omnibus.csv)
and plots each measure against starvation HOURS on a linear axis.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

import scripts.analysis.starvation_trend_figure as stf


def _paired():
    rows = []
    spec = {"1 h": ([0, 0, 1], [0, 1, 1]), "12 h": ([0, 0, 1, 0], [2, 3, 3, 2]),
            "24 h": ([1, 2], [1, 2])}
    for g, (pre, post) in spec.items():
        for i, (a, b) in enumerate(zip(pre, post)):
            for odor in ("Isoamyl Acetate", "Citral"):
                rows.append(dict(group=g, dataset_canon="X", fly=f"{g}_{i}", fly_number=1,
                                 odor=odor, score_pre=a, score_post=b))
    return pd.DataFrame(rows)


@pytest.mark.parametrize("label, hours", [("1 h", 1.0), ("12 h", 12.0), ("18h", 18.0),
                                          ("24 h starved", 24.0), ("0.5 h", 0.5)])
def test_hours_from_label(label, hours):
    assert stf.hours_from_label(label) == hours


def test_hours_from_label_rejects_labels_without_hours():
    with pytest.raises(ValueError):
        stf.hours_from_label("Oct 6")


def test_trend_table_is_per_group_and_odor_in_hour_order():
    t = stf.trend_table(_paired(), binary_threshold=2)
    iaa = t[t.odor == "Isoamyl Acetate"]
    assert list(iaa.hours) == [1.0, 12.0, 24.0]
    row = iaa[iaa.hours == 12.0].iloc[0]
    assert row.n_flies == 4
    assert row.mean_delta == pytest.approx(np.mean([2, 3, 2, 2]))
    assert row.sem_delta == pytest.approx(np.std([2, 3, 2, 2], ddof=1) / 2)
    assert row.pct_pre == pytest.approx(0.0) and row.pct_post == pytest.approx(100.0)


def test_points_sit_at_their_hours_on_a_linear_axis(tmp_path):
    t = stf.trend_table(_paired(), binary_threshold=2)
    fig = stf.build_grid(t, omnibus=None, trained_odor="Isoamyl Acetate", title="t")
    ax = fig.axes[0]
    xs = sorted({round(x, 6) for line in ax.get_lines() for x in line.get_xdata()
                 if line.get_linestyle() == "-" and len(line.get_xdata()) == 3})
    assert xs == [1.0, 12.0, 24.0]


def test_figures_are_written(tmp_path):
    paired = _paired()
    paired.to_csv(tmp_path / "paired_scores_per_fly.csv", index=False)
    pd.DataFrame({"odor": ["Isoamyl Acetate", "Citral"], "p_kw_delta_score": [0.04, 0.5],
                  "p_ffh_post_rate": [0.2, 0.9]}).to_csv(tmp_path / "stats_omnibus.csv", index=False)
    written = stf.main(["--comparison-dir", str(tmp_path), "--trained-odor", "Isoamyl Acetate"])
    names = sorted(p.name for p in written)
    assert names == ["starvation_trend_all_odors.png", "starvation_trend_emphasis.png",
                     "starvation_trend_table.csv", "starvation_trend_trained_odor.png"]
    assert all(p.stat().st_size > 0 for p in written)


# ── dataviz mark specs ─────────────────────────────────────────────────────

import matplotlib.colors as mcolors  # noqa: E402


def _same(a, b) -> bool:
    return np.allclose(mcolors.to_rgb(a), mcolors.to_rgb(b))


def test_delta_pct_is_post_minus_pre_rate():
    t = stf.trend_table(_paired(), binary_threshold=2)
    row = t[(t.odor == "Isoamyl Acetate") & (t.hours == 12.0)].iloc[0]
    assert row.delta_pct == pytest.approx(row.pct_post - row.pct_pre)


def test_emphasis_puts_the_trained_odor_in_ink_and_the_rest_in_gray():
    t = stf.trend_table(_paired(), binary_threshold=2)
    fig = stf.build_emphasis(t, omnibus=None, trained_odor="Isoamyl Acetate", title="t")
    data_lines = [ln for ln in fig.axes[0].get_lines()
                  if len(ln.get_xdata()) == 3 and ln.get_linestyle() == "-"]
    inked = [ln for ln in data_lines if _same(ln.get_color(), stf.EMPHASIS_COLOR)]
    grays = [ln for ln in data_lines if _same(ln.get_color(), stf.CONTEXT_COLOR)]
    assert len(inked) == 1 and len(grays) == 1
    assert inked[0].get_linewidth() >= 2.0


def test_reference_lines_are_solid_and_markers_carry_a_surface_ring():
    t = stf.trend_table(_paired(), binary_threshold=2)
    fig = stf.build_grid(t, omnibus=None, trained_odor="Isoamyl Acetate", title="t")
    for ax in fig.axes:
        for ln in ax.get_lines():
            assert ln.get_linestyle() in ("-", "None"), ln.get_linestyle()
            # Error-bar caps ("_") are ticks, not data markers.
            if ln.get_marker() not in (None, "None", "", "_", "|"):
                assert _same(ln.get_markeredgecolor(), "white")
