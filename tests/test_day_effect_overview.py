"""Day-vs-day overview across datasets (scripts/analysis/day_effect_overview.py)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import pytest

import scripts.analysis.day_effect_overview as deo


def _comparison_dir(root, name, days, p_pre=0.5):
    d = root / name / f"{days[0].replace(' ', '')}_vs_{days[1].replace(' ', '')}"
    d.mkdir(parents=True)
    rows = []
    for i, day in enumerate(days):
        for odor in ("Citral", "Isoamyl Acetate"):
            rows.append(dict(group=day, odor=odor, n_flies=5 + i, mean_pre=1.0 + i,
                             sem_pre=0.2, mean_post=2.0 - i, sem_post=0.3, pct_pre=20.0,
                             pct_post=40.0, p_score=0.5, p_rate=0.5))
    pd.DataFrame(rows).to_csv(d / "summary_by_group.csv", index=False)
    paired = []
    for i, day in enumerate(days):
        for f in range(5 + i):
            for odor in ("Citral", "Isoamyl Acetate"):
                paired.append(dict(group=day, dataset_canon=name, fly=f"b{f}", fly_number=1,
                                   odor=odor, score_pre=1 + i, score_post=2 - i))
    pd.DataFrame(paired).to_csv(d / "paired_scores_per_fly.csv", index=False)
    pd.DataFrame([dict(odor=o, group_a=days[0], group_b=days[1], n_a=5, n_b=6,
                       mean_delta_a=1.0, sem_delta_a=0.1, mean_delta_b=-1.0, sem_delta_b=0.1,
                       p_pre_score=p_pre, p_post_score=0.5, p_delta_score=0.01,
                       p_pre_rate=1.0, p_post_rate=1.0)
                  for o in ("Citral", "Isoamyl Acetate")]).to_csv(
        d / "stats_between_groups.csv", index=False)
    return d


def test_overview_table_has_one_row_per_dataset_odor_and_day(tmp_path):
    a = _comparison_dir(tmp_path, "IAA-Sensitivity-24-1", ("Oct 6", "Oct 7"))
    b = _comparison_dir(tmp_path, "IAA-Sensitivity-12-1", ("Oct 6", "Oct 9"), p_pre=0.01)
    t = deo.overview_table([a, b])
    assert len(t) == 2 * 2 * 2
    row = t[(t.dataset == "IAA-Sensitivity-12-1") & (t.odor == "Citral") & (t.day == "Oct 9")].iloc[0]
    assert row.day_index == 1 and row.mean_pre == pytest.approx(2.0)
    assert row.mean_delta == pytest.approx(-1.0) and row.n_flies == 6
    assert row.p_pre_score == pytest.approx(0.01)


def test_overview_figure_renders_with_a_row_per_dataset(tmp_path):
    a = _comparison_dir(tmp_path, "IAA-Sensitivity-24-1", ("Oct 6", "Oct 7"))
    b = _comparison_dir(tmp_path, "IAA-Sensitivity-12-1", ("Oct 6", "Oct 9"), p_pre=0.01)
    out = deo.main(["--comparison-dir", str(a), "--comparison-dir", str(b),
                    "--out-dir", str(tmp_path / "out")])
    assert sorted(p.name for p in out) == ["day_effect_overview.csv", "day_effect_overview.png"]
    fig = deo.build_figure(deo.overview_table([a, b]), trained_odor="Isoamyl Acetate")
    assert len(fig.axes) == 2 * 3
    stars = [t.get_text() for ax in fig.axes for t in ax.texts]
    assert "*" in stars  # 12-1 pre-test p=0.01 and every delta p=0.01
