"""Trace figures show both odor scores of an IAA odor-pair trial (2026-11-13)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.analysis import envelope_visuals as ev  # noqa: E402

PAIR = "testing_3_isoamylacetate-1_then_linalool-1"


def _scores(monkeypatch, table):
    monkeypatch.setattr(ev, "_lookup_model_score", lambda ds, fly, n, label: table.get(label))


def test_single_odor_trial_text_unchanged(monkeypatch):
    _scores(monkeypatch, {"testing_1_hexanol-1": 3})
    assert ev._score_text("d", "f", "1", "testing_1_hexanol-1") == "Score: 3"
    assert ev._score_text("d", "f", "1", "testing_2_hexanol-1") is None


def test_pair_trial_shows_both_odor_scores(monkeypatch):
    _scores(monkeypatch, {PAIR: 1, PAIR + "__odor2": 4})
    assert ev._score_text("d", "f", "1", PAIR) == "Odor 1: 1   Odor 2: 4"


def test_pair_trial_missing_odor2_score(monkeypatch):
    _scores(monkeypatch, {PAIR: 2})
    assert ev._score_text("d", "f", "1", PAIR) == "Odor 1: 2   Odor 2: –"


# ── odor 2 shading: its valve window comes from the rig per-frame log ─────────
def test_odor2_window_from_frames_log(tmp_path, monkeypatch):
    import numpy as np
    import pandas as pd
    batch = tmp_path / "IAA-PairX-Control-1-1" / "b1"
    batch.mkdir(parents=True)
    n = 3250
    state = np.array(["off"] * n, dtype=object)
    state[1203:1603] = "OFM_A"
    state[1647:2047] = "OFM_L"
    pd.DataFrame({"FrameNumber": np.arange(1, n + 1), "ActiveOFM": state}).to_csv(
        batch / "output_b1_testing_3_IsoamylAcetate-1_then_Linalool-1_20261113_125306.csv", index=False)
    monkeypatch.setattr(ev, "_PAIR_DATA_ROOTS", (tmp_path,))
    ev._ODOR2_WINDOW_CACHE.clear()
    on, off = ev._odor2_window("IAA-PairX-Control-1-1", "b1", PAIR, 40.0)
    assert (on, off) == (1648 / 40.0, 2048 / 40.0)
    assert ev._odor2_window("IAA-PairX-Control-1-1", "b1", "testing_1_hexanol-1", 40.0) is None
    assert ev._odor2_window("IAA-PairX-Control-1-1", "missing", PAIR, 40.0) is None
