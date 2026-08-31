"""The RandomPanel concentration comparison must respect ``freeze.figures``.

It draws ONE figure set from all three RandomPanel datasets, so the
all-contributors rule applies: it is skipped only when every contributing
dataset is frozen. All three are frozen in the shipped config, so without this
check the whole set is redrawn on every run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import scripts.analysis.randompanel_conc_comparison as rc  # noqa: E402


def _config(tmp_path, frozen) -> Path:
    lines = ["dataset_overrides:"]
    for name in frozen:
        lines += [f"  {name}:", "    freeze: {figures: true}"]
    path = tmp_path / "config.yaml"
    path.write_text("\n".join(lines) + "\n")
    return path


def _run(tmp_path, frozen=(), monkeypatch=None):
    """Call generate_conc_comparison with the loader stubbed to a full panel."""

    class _Reached(Exception):
        """Raised at the point where the figure set would start being drawn."""

    def _stop(*a, **k):
        raise _Reached

    monkeypatch.setattr(rc, "_load_panel", lambda *a, **k: _panel())
    monkeypatch.setattr(rc, "_resolve_out_dir", _stop)
    try:
        rc.generate_conc_comparison(
            csv_path=tmp_path / "model_predictions.csv",
            out_dir=tmp_path / "figs",
            config=str(_config(tmp_path, frozen)),
            n_iter=10,
        )
    except _Reached:
        return True
    return False


def _panel() -> pd.DataFrame:
    """Stands in for ``_load_panel``'s RETURN value, so it must match that shape.

    ``odor_display`` and ``fly_type`` are inputs the real loader consumes and
    drops in its per-fly groupby; ``reacted`` and ``n_trials`` are what it adds.
    An earlier version of this stub had it backwards, which is how the
    ``KeyError: 'dataset_canon'`` crash reached production green.
    ``test_randompanel_conc_panel_contract.py`` now pins the two together.
    """
    rows = []
    for dataset, conc in rc.CONC_BY_DATASET.items():
        for fly in ("f1", "f2"):
            rows.append({
                "odor": "Hexanol", "conc": conc, "dataset_canon": dataset,
                "fly": fly, "fly_number": 1,
                "score": 2.0, "reacted": 1.0, "n_trials": 2,
            })
    return pd.DataFrame(rows)


def test_all_three_frozen_skips_the_figure_set(tmp_path, monkeypatch):
    assert _run(tmp_path, tuple(rc.CONC_BY_DATASET), monkeypatch) is False


def test_one_live_dataset_still_draws(tmp_path, monkeypatch):
    frozen = list(rc.CONC_BY_DATASET)[:-1]
    assert _run(tmp_path, tuple(frozen), monkeypatch) is True


def test_no_freeze_draws(tmp_path, monkeypatch):
    assert _run(tmp_path, (), monkeypatch) is True
