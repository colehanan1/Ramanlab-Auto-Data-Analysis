"""``_load_panel``'s real return shape, pinned against the freeze test's stub.

The all-contributors freeze check reads ``df["dataset_canon"]``, but
``_load_panel`` returns a groupby result keyed on (odor, conc, fly, fly_number)
— which drops that column. Production crashed with ``KeyError: 'dataset_canon'``
the moment all three concentrations were present.

``test_randompanel_conc_freeze.py`` did not catch it: it monkeypatches
``_load_panel`` with a hand-built frame that *does* carry ``dataset_canon``. The
stub and the real function had drifted apart, so the test exercised a shape the
code never produces. These tests call the real loader, and assert the stub still
matches it.
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
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))
from test_randompanel_conc_freeze import _panel as _stub_panel  # noqa: E402

ODORS = ["Hexanol", "Citral", "Linalool"]


def _predictions(tmp_path, datasets=None) -> Path:
    """A predictions CSV in the real shape: two exposures per odor per fly."""
    rows = []
    for dataset in (datasets or list(rc.CONC_BY_DATASET)):
        for fly_number in (1, 2):
            trial = 0
            for odor in ODORS:
                for _exposure in (1, 2):
                    trial += 1
                    rows.append({
                        "dataset": dataset,
                        "fly": f"{dataset}_batch_1",
                        "fly_number": fly_number,
                        "trial_label": f"testing_{trial}_{odor.lower()}",
                        "score": 3,
                        "trial_type": "testing",
                        "fly_type": "GR5a-Old",
                    })
    path = tmp_path / "model_predictions.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _load(tmp_path, **kw):
    return rc._load_panel(_predictions(tmp_path, **kw), fly_type="GR5a-Old", config="")


# ── the real return shape ─────────────────────────────────────────────────


def test_load_panel_returns_rows_for_a_full_panel(tmp_path):
    assert not _load(tmp_path).empty


def test_load_panel_keeps_dataset_canon(tmp_path):
    """The freeze check reads this column off the returned frame."""
    assert "dataset_canon" in _load(tmp_path).columns


def test_every_contributing_dataset_is_recoverable(tmp_path):
    contributors = set(_load(tmp_path)["dataset_canon"].unique())
    assert contributors == set(rc.CONC_BY_DATASET)


def test_load_panel_still_pools_each_flys_two_exposures(tmp_path):
    """The unit of analysis stays the FLY, not the presentation.

    Carrying dataset_canon must not split a fly's two exposures into two rows —
    that would be the pseudoreplication the pooling exists to prevent.
    """
    panel = _load(tmp_path)
    assert set(panel["n_trials"]) == {2}
    # 3 datasets x 2 flies x 3 odors, one row each.
    assert len(panel) == 3 * 2 * len(ODORS)


def test_conc_still_maps_one_to_one_with_the_dataset(tmp_path):
    panel = _load(tmp_path)
    for dataset, conc in rc.CONC_BY_DATASET.items():
        sub = panel[panel["dataset_canon"] == dataset]
        assert set(sub["conc"]) == {conc}


def test_the_expected_columns_are_all_present(tmp_path):
    assert set(_load(tmp_path).columns) >= {
        "odor", "conc", "fly", "fly_number", "score", "reacted", "n_trials",
        "dataset_canon",
    }


# ── the stub must not drift from the real thing again ─────────────────────


def test_the_freeze_tests_stub_matches_the_real_loader(tmp_path):
    """The bug in one sentence: the stub had columns the real loader lacked."""
    real = set(_load(tmp_path).columns)
    stub = set(_stub_panel().columns)
    missing = {c for c in stub if c not in real}
    assert not missing, (
        f"test_randompanel_conc_freeze._panel() supplies {sorted(missing)}, "
        "which _load_panel does not return — the stub is testing a shape "
        "production never produces."
    )


def test_the_stub_supplies_everything_the_freeze_check_reads():
    assert "dataset_canon" in _stub_panel().columns


# ── the crash itself ──────────────────────────────────────────────────────


def test_generate_conc_comparison_survives_a_full_panel(tmp_path, monkeypatch):
    """End to end on the real loader: this is the call that aborted the run."""

    class _Reached(Exception):
        pass

    def _stop(*a, **k):
        raise _Reached

    monkeypatch.setattr(rc, "_resolve_out_dir", _stop)
    with pytest.raises(_Reached):
        rc.generate_conc_comparison(
            csv_path=_predictions(tmp_path),
            out_dir=tmp_path / "figs",
            fly_type="GR5a-Old",
            config="",
            n_iter=10,
        )


def test_a_partial_panel_still_skips_without_touching_dataset_canon(tmp_path):
    """Fewer than three concentrations returns before the freeze check."""
    rc.generate_conc_comparison(
        csv_path=_predictions(tmp_path, datasets=["RandomPanel-24-0.1"]),
        out_dir=tmp_path / "figs",
        fly_type="GR5a-Old",
        config="",
        n_iter=10,
    )
    assert not (tmp_path / "figs").exists()
