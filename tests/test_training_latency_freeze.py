"""Training latency figures must respect ``freeze.figures`` too.

``envelope_training.latency_reports`` draws one per-fly figure and one mean
figure per dataset, plus a pooled grand-mean panel. The per-dataset figures are
that cohort's figures, so a frozen cohort must not redraw them; the grand mean
pools every dataset, so it follows the all-contributors rule.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import scripts.analysis.envelope_training as et  # noqa: E402

TRIALS = (4, 6)


def _lat_df() -> pd.DataFrame:
    rows = []
    # Config spells it "3Oct-Training-24-0.1"; the frame carries the canonical
    # "3OCT-Training-24-0.1" -- the mismatch that let the cohort keep redrawing.
    for dataset in ("3OCT-Training-24-0.1", "EB-Training-24-1"):
        for fly in ("july_29_batch_1_rig_2", "july_29_batch_2_rig_2"):
            for trial in TRIALS:
                rows.append({
                    "dataset_canon": dataset, "fly": fly, "fly_number": "1",
                    "trial_num": trial, "latency": 3.0, "latency_any": 3.0,
                    "plot_latency": 3.0, "lat_for_mean": 3.0,
                    "response_kind": "response_within_ceiling",
                })
    return pd.DataFrame(rows)


def _run(tmp_path, frozen=()):
    out = tmp_path / ("out_" + ("-".join(frozen) or "live"))
    et._plot_latency_per_fly(
        _lat_df(), out, latency_ceiling=10.0, trials_of_interest=TRIALS,
        overwrite=True, frozen_datasets=frozen,
    )
    et._plot_latency_by_odor(
        _lat_df(), out, latency_ceiling=10.0, trials_of_interest=TRIALS,
        overwrite=True, frozen_datasets=frozen,
    )
    et._plot_latency_grand_means(
        _lat_df(), out, 10.0, True, frozen_datasets=frozen,
    )
    return {str(p.relative_to(out)) for p in out.rglob("*.png")}


def test_frozen_cohort_draws_no_training_latency_figures(tmp_path):
    names = _run(tmp_path, frozen=("3Oct-Training-24-0.1",))
    assert not any("3OCT" in n for n in names), names
    assert any("EB-Training" in n for n in names), names


def test_live_cohorts_still_draw(tmp_path):
    names = _run(tmp_path)
    assert any("3OCT" in n for n in names), names
    assert any("EB-Training" in n for n in names), names


def test_grand_mean_survives_one_live_cohort(tmp_path):
    names = _run(tmp_path, frozen=("3Oct-Training-24-0.1",))
    assert "grand_mean_by_odor_latency.png" in names, names


def test_grand_mean_skipped_when_every_cohort_is_frozen(tmp_path):
    names = _run(tmp_path, frozen=("3Oct-Training-24-0.1", "EB-Training-24-1"))
    assert names == set(), names


def test_latency_reports_forwards_the_freeze(tmp_path, monkeypatch):
    seen = {}
    for fn in ("_plot_latency_per_fly", "_plot_latency_by_odor"):
        monkeypatch.setattr(
            et, fn,
            lambda df, out, *, frozen_datasets=(), _n=fn, **kw:
                seen.__setitem__(_n, set(frozen_datasets)),
        )
    monkeypatch.setattr(
        et, "_plot_latency_grand_means",
        lambda df, out, ceiling, overwrite, frozen_datasets=():
            seen.__setitem__("grand", set(frozen_datasets)),
    )
    csv = tmp_path / "lat.csv"
    _lat_df().to_csv(csv, index=False)
    monkeypatch.setattr(et, "_latency_records_from_csv", lambda *a, **k: _lat_df())
    et.latency_reports(
        None, None, tmp_path / "out", csv_path=csv,
        before_sec=5.0, during_sec=30.0, threshold_mult=4.0,
        latency_ceiling=10.0, trials_of_interest=TRIALS,
        fps_default=40.0, overwrite=True,
        frozen_datasets=("3Oct-Training-24-0.1",),
    )
    assert seen["_plot_latency_per_fly"] == {"3Oct-Training-24-0.1"}
    assert seen["_plot_latency_by_odor"] == {"3Oct-Training-24-0.1"}
    assert seen["grand"] == {"3Oct-Training-24-0.1"}
