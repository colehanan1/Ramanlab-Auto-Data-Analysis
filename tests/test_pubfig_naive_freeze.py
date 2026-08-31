"""``pubfig_naive_vs_trained --sweep`` must not redraw frozen cohorts.

The sweep enumerates every trained/control pair present in model_predictions.csv,
and frozen datasets stay in that CSV. Without a freeze check the sweep redraws
every published naive-vs-trained figure on every run.

A comparison has three arms, so it is dropped only when ALL THREE (naive,
trained, control) are frozen for figures -- the same all-contributors rule as
``envelope_visuals.should_skip_frozen_figure``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.pubfig_naive_vs_trained import sweep_comparisons  # noqa: E402

NAIVE_BY_CONC = {1.0: "RandomPanel-24-1", 0.1: "RandomPanel-24-0.1"}


def _frame() -> pd.DataFrame:
    rows = []

    def add(ds, odor, occs):
        for fly in ("july_20_batch_1", "july_20_batch_2"):
            for occ in occs:
                rows.append({
                    "dataset_canon": ds, "fly": fly, "fly_number": 1,
                    "fly_type": "GR5a-Old", "odor_display": odor,
                    "occurrence": occ, "score": 2.0,
                })

    for ds in ("3OCT-Training-24-0.1", "3OCT-Control-24-0.1"):
        add(ds, "3-Octanol (0.1%)", (1,))
    for ds in ("EB-Training-24-1", "EB-Control-24-1"):
        add(ds, "Ethyl Butyrate (1%)", (1,))
    add("RandomPanel-24-0.1", "3-Octanol", (1,))
    add("RandomPanel-24-1", "Ethyl Butyrate", (1,))
    return pd.DataFrame(rows)


def _cohorts(frozen=()):
    comps, _ = sweep_comparisons(
        _frame(), naive_by_conc=NAIVE_BY_CONC, frozen_datasets=frozen
    )
    return {c.train_dataset for c in comps}


def test_sweep_renders_every_cohort_by_default():
    assert _cohorts() == {"3OCT-Training-24-0.1", "EB-Training-24-1"}


def test_all_three_arms_frozen_drops_the_comparison():
    frozen = ("EB-Training-24-1", "EB-Control-24-1", "RandomPanel-24-1")
    assert _cohorts(frozen) == {"3OCT-Training-24-0.1"}


def test_a_live_naive_panel_keeps_the_comparison():
    """Both conditioned arms frozen but a live naive panel still redraws --
    the figure is the three-arm comparison."""
    assert "EB-Training-24-1" in _cohorts(("EB-Training-24-1", "EB-Control-24-1"))


def test_a_live_trained_arm_keeps_the_comparison():
    assert "EB-Training-24-1" in _cohorts(("EB-Control-24-1", "RandomPanel-24-1"))


def test_skips_are_reported_not_silent():
    frozen = ("EB-Training-24-1", "EB-Control-24-1", "RandomPanel-24-1")
    _, skipped = sweep_comparisons(
        _frame(), naive_by_conc=NAIVE_BY_CONC, frozen_datasets=frozen
    )
    reasons = [s["reason"] for s in skipped]
    assert any("frozen" in r for r in reasons), skipped


def test_build_sweep_reads_the_freeze_from_config(tmp_path, monkeypatch):
    """The --config file is the freeze's source of truth for the sweep, and
    --thaw-all lifts it for one run."""
    import scripts.analysis.pubfig_naive_vs_trained as pnt

    config = tmp_path / "config.yaml"
    config.write_text(
        "dataset_overrides:\n"
        "  EB-Training-24-1:\n    freeze: {figures: true}\n"
        "  EB-Control-24-1:\n    freeze: {figures: true}\n"
        "  RandomPanel-24-1:\n    freeze: {figures: true}\n"
    )
    seen = {}
    real = pnt.sweep_comparisons

    def _spy(frame, **kw):
        seen["frozen"] = set(kw.get("frozen_datasets") or ())
        return real(frame, **kw)

    monkeypatch.setattr(pnt, "sweep_comparisons", _spy)
    monkeypatch.setattr(pnt, "NAIVE_BY_CONC", NAIVE_BY_CONC, raising=False)

    pnt.build_sweep(config=config, out_dir=tmp_path / "figs", df=_frame())
    assert seen["frozen"] == {
        "EB-Training-24-1", "EB-Control-24-1", "RandomPanel-24-1",
    }
    written = {p.parent.name for p in (tmp_path / "figs").rglob("*.png")}
    assert not any(w.startswith("EB-") for w in written), written

    pnt.build_sweep(config=config, out_dir=tmp_path / "figs2", df=_frame(),
                    thaw_all=True)
    assert seen["frozen"] == set()


def test_freezing_a_cohort_does_not_prune_its_published_figures(tmp_path, monkeypatch):
    """Freeze means "stop redrawing", never "delete what is already there".

    ``build_sweep`` prunes figures it no longer builds, so a frozen cohort --
    which by definition is no longer built -- would otherwise have every
    published PNG/PDF/SVG/CSV deleted on the next run.
    """
    import scripts.analysis.pubfig_naive_vs_trained as pnt

    config = tmp_path / "config.yaml"
    config.write_text(
        "dataset_overrides:\n"
        "  EB-Training-24-1:\n    freeze: {figures: true}\n"
        "  EB-Control-24-1:\n    freeze: {figures: true}\n"
        "  RandomPanel-24-1:\n    freeze: {figures: true}\n"
    )
    out = tmp_path / "figs"
    published = out / "EB-24-1" / "pubfig_naive_vs_trained_Ethyl_Butyrate_1pct_p1.png"
    published.parent.mkdir(parents=True)
    published.write_bytes(b"published")
    stats = published.with_name(
        "pubfig_naive_vs_trained_Ethyl_Butyrate_1pct_p1_stats.csv"
    )
    stats.write_text("published")

    monkeypatch.setattr(pnt, "NAIVE_BY_CONC", NAIVE_BY_CONC, raising=False)
    pnt.build_sweep(config=config, out_dir=out, df=_frame())

    assert published.exists(), "a frozen cohort's figure was deleted by the prune"
    assert stats.exists()


def test_a_live_cohorts_stale_figure_is_still_pruned(tmp_path, monkeypatch):
    import scripts.analysis.pubfig_naive_vs_trained as pnt

    config = tmp_path / "config.yaml"
    config.write_text("dataset_overrides: {}\n")
    out = tmp_path / "figs"
    stale = out / "EB-24-1" / "pubfig_naive_vs_trained_Gone_1pct_p1.png"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"stale")

    monkeypatch.setattr(pnt, "NAIVE_BY_CONC", NAIVE_BY_CONC, raising=False)
    pnt.build_sweep(config=config, out_dir=out, df=_frame())
    assert not stale.exists()
