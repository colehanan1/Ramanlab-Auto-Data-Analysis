"""src/per_analysis/daynight.py: day vs evening sessions, on synthetic tidy trials."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "src", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from per_analysis import config as C  # noqa: E402
from test_per_analysis_cross import CLICK, make_trials, session  # noqa: E402

daynight = pytest.importorskip("per_analysis.daynight")
cross = pytest.importorskip("per_analysis.cross")

PF = "IAA-PairFirst-Control-1-1"


@pytest.fixture(autouse=True)
def _fast(monkeypatch):
    monkeypatch.setattr(cross, "N_BOOT", 300)
    monkeypatch.setattr(cross, "N_PERM_FLY", 300)


def _pairfirst(score, n_evening_sessions=1):
    day = [session(f"november_{13 + i}_batch_1", "day", 11.5, f"2026-11-{13 + i}",
                   batches=(3, 3)) for i in range(4)]
    eve = [session(f"november_{17 - i}_batch_2", "evening", 18.0, f"2026-11-{17 - i}",
                   batches=(3, 3)) for i in range(n_evening_sessions)]
    return make_trials({PF: day + eve}, score)


def test_window_colours_are_validated_and_distinct():
    w = C.WINDOW_COLORS
    assert w["day"] != w["evening"]
    assert {w["day"], w["evening"]}.isdisjoint(
        {str(v).lower() for v in C.CONDITION_COLORS.values()})


def test_window_comparison_finds_a_planted_effect_and_not_an_absent_one():
    """Naive (alone) responses: evening flies answer everything (score 3), day flies
    nothing; the pair block is identical in both windows."""
    def score(ctx):
        if ctx["block"] == "alone":
            return 3 if ctx["window"] == "evening" else 0
        return 0
    t = cross.odor_trials(_pairfirst(score, n_evening_sessions=2))
    tab = daynight.window_table(t, PF)
    naive = tab[(tab.measure == "naive") & (tab.odor == daynight.ALL_ODORS)
                & (tab.metric == "rate")].iloc[0]
    assert naive["effect"] == pytest.approx(100.0)            # evening - day, pp
    assert naive["p_batch"] < 0.05 and naive["p"] < 0.05
    # 4 day + 2 evening sessions: C(6, 2) = 15 session arrangements, 1 as extreme
    assert naive["p_session"] == pytest.approx(1 / 15)
    assert naive["n_flies_day"] == 24 and naive["n_flies_evening"] == 12
    assert naive["n_batches_evening"] == 4 and naive["n_sessions_evening"] == 2
    pair = tab[(tab.measure == "pair") & (tab.odor == daynight.ALL_ODORS)
               & (tab.metric == "rate")].iloc[0]
    assert pair["effect"] == pytest.approx(0.0)
    assert pair["p_batch"] == pytest.approx(1.0)
    assert {"n_flies", "effect", "ci_lo", "ci_hi", "p", "p_holm", "q_bh", "p_batch",
            "p_session", "test", "correction"} <= set(tab.columns)


def test_controlled_change_per_fly_mean_over_partners():
    """Every partner X: alone 0, pair 2 (round 1) and 0 (round 2); click: alone 0, pair 0.
    Round 1 = 2 - 0 = 2, round 2 = 0, pooled = 1, for every fly."""
    def score(ctx):
        if ctx["block"] == "pair" and ctx["odor"] != CLICK:
            return 2 if ctx["presentation"] == 1 else 0
        return 0
    t = cross.odor_trials(_pairfirst(score))
    d = daynight.exp2_controlled_change(t[t.cohort == PF], "score")
    got = d.groupby("round")["value"].agg(["min", "max"])
    assert got.loc["round 1"].tolist() == pytest.approx([2.0, 2.0])
    assert got.loc["round 2"].tolist() == pytest.approx([0.0, 0.0])
    assert got.loc["pooled"].tolist() == pytest.approx([1.0, 1.0])


def test_other_window_is_left_out():
    t = _pairfirst(lambda ctx: 0)
    t.loc[t.batch == "november_13_batch_1", "window"] = C.OTHER_WINDOW
    counts = daynight.window_counts(cross.odor_trials(t))
    assert set(counts["window"]) == {"day", "evening"}
    assert int(counts.loc[counts.window == "day", "n_flies"].sum()) == 24 - 3


def test_one_evening_session_confound_note_everywhere(tmp_path):
    def score(ctx):
        return int(ctx["rng"].integers(0, 4))
    cohorts = {PF: [session("november_13_batch_1", "day", 11.5, "2026-11-13", (3, 3)),
                    session("november_14_batch_1", "day", 13.5, "2026-11-14", (3, 3)),
                    session("november_17_batch_1", "day", 12.7, "2026-11-17", (3, 3)),
                    session("november_17_batch_2", "evening", 18.0, "2026-11-17", (3, 3))],
               "IAA-Sensitivity-1-1": [
                   session("october_12_batch_1", "day", 14.0, "2026-10-12", (3, 3)),
                   session("october_13_batch_1", "day", 11.5, "2026-10-13", (3, 3)),
                   session("october_12_batch_2", "evening", 17.5, "2026-10-12", (3, 3)),
                   session("october_14_batch_2", "evening", 17.3, "2026-10-14", (3,))],
               "IAA-PairSecond-Control-1-1": [
                   session("november_16_batch_1", "day", 12.0, "2026-11-16", (3, 3))]}
    t = make_trials(cohorts, score)
    out, tables = tmp_path / "results", tmp_path / "results" / "tables"
    findings = daynight.run(t, out, tables=tables)
    assert findings and all(f["section"] == "daynight" for f in findings)
    primary = [f for f in findings if f["primary"]]
    assert len(primary) == 1 and "Exp 2" in primary[0]["text"]
    text = primary[0]["text"].lower()
    assert "one evening session" in text and "confounded" in text
    assert "batches" in text and "sessions" in text
    iaa = [f for f in findings if "IAA-Sensitivity" in f["analysis"]]
    assert iaa and "one evening session" not in iaa[0]["text"].lower()
    root = out / "08_day_vs_evening"
    assert (root / "README.md").exists()
    svgs = [p for p in root.rglob("*.svg") if "pairfirst" in str(p).lower()]
    assert svgs and all("confounded" in p.read_text() for p in svgs)
    assert any("no evening" in f["text"].lower() and "Exp 3" in f["text"] for f in findings)
    assert list((root / "models").glob("*_coefs.csv"))
    for csv in tables.glob("daynight_*.csv"):
        d = pd.read_csv(csv)
        assert {"n_flies", "effect", "ci_lo", "ci_hi", "p", "p_holm", "q_bh", "p_batch",
                "test", "correction"} <= set(d.columns), csv.name


def _probes(windows, odor_window_s):
    return pd.DataFrame({"cohort": "IAA-PairFirst-Control-1-1", "is_light_probe": True,
                         "window": windows, "odor_window_s": odor_window_s,
                         "fly_id": [f"f{i}" for i in range(len(windows))]})


def test_pair_probe_note_reports_short_windows_from_the_table():
    note = daynight.probe_note(_probes(["evening"] * 9, 0.025), "IAA-PairFirst-Control-1-1")
    assert "9 of 9" in note and "under 1 s" in note and "full pipeline run" in note


def test_pair_probe_note_counts_valid_probes_per_window():
    note = daynight.probe_note(_probes(["evening"] * 9 + ["day"] * 3, 10.0),
                               "IAA-PairFirst-Control-1-1")
    assert "day 3, evening 9" in note and "under 1 s" not in note


def test_pair_probe_note_without_probes():
    t = _probes(["day"], 10.0).assign(is_light_probe=False)
    assert "no light probe" in daynight.probe_note(t, "IAA-PairFirst-Control-1-1")
