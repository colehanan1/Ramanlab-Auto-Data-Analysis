"""per_analysis QC rules on small synthetic tidy tables."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from per_analysis import io, qc

THR = 2


def _fly(fly_id, rows, *, experiment="exp1", cohort="C1", window="day", batch=None):
    """rows: (block, label, odor, score, extra-dict)."""
    batch = batch or fly_id.split("|")[1]
    out = []
    for block, label, odor, score, *extra in rows:
        p = io.parse_trial_label(label)
        r = dict(experiment=experiment, cohort=cohort, fly_id=fly_id, batch=batch,
                 session=batch, rig=1, host="Flybehavior2", window=window, block=block,
                 trial_label=label, trial_index=p.trial_index, odor=odor, prime=None,
                 partner=odor, presentation=np.nan, is_light_probe=p.is_light_probe,
                 is_click=False, is_odor2=p.is_odor2 or len(p.odors) == 2,
                 odor_window_s=10.0 if experiment != "exp1" else 30.0,
                 score=score, responded=score >= THR, odor1_score=np.nan, dataset=cohort)
        if extra:
            r.update(extra[0])
        out.append(r)
    return out


def _frame(*flies):
    return pd.DataFrame([r for f in flies for r in f])


LIGHT = ("post", "testing_9_lightonly", None, 4)


# ── responders ────────────────────────────────────────────────────────────────
def test_never_responder_odors_only_and_light_probe_reported():
    a = _fly("C1|a|1", [("pre", "pretest_1_isoamylacetate-1", "IAA", 0),
                        ("post", "testing_1_isoamylacetate-1", "IAA", 1),
                        ("post", "testing_8_paraffinoil-100", "Paraffin Oil", 4),
                        LIGHT])
    b = _fly("C1|b|1", [("pre", "pretest_1_isoamylacetate-1", "IAA", 0),
                        ("post", "testing_1_isoamylacetate-1", "IAA", 2),
                        ("post", "testing_9_lightonly", None, 0)])
    per_fly, _ = qc.run_qc(_frame(a, b), threshold=THR)
    pf = per_fly.set_index("fly_id")
    assert pf.loc["C1|a|1", "never_responder"]          # paraffin and light do not count
    assert pf.loc["C1|a|1", "light_probe_responded"]
    assert not pf.loc["C1|b|1", "never_responder"]
    assert not pf.loc["C1|b|1", "light_probe_responded"]


def test_never_responder_counts_pair_odor1_but_not_the_click():
    kw = dict(experiment="exp2", cohort="P1")
    c = _fly("P1|c|1", [("alone", "pretest_1_hexanol-0.1", "Hexanol", 0),
                        ("pair", "testing_1_isoamylacetate-1_then_hexanol-0.1", "Hexanol", 0,
                         dict(prime="IAA", odor1_score=3))], **kw)
    kw3 = dict(experiment="exp3", cohort="P2")
    d = _fly("P2|d|1", [("alone", "pretest_1_isoamylacetate-1", "IAA", 1),
                        ("pair", "testing_1_paraffinoil-100_then_isoamylacetate-1", "IAA", 0,
                         dict(prime="Paraffin Oil", partner="Paraffin Oil", is_click=True,
                              odor1_score=4))], **kw3)
    pf = qc.run_qc(_frame(c, d), threshold=THR)[0].set_index("fly_id")
    assert not pf.loc["P1|c|1", "never_responder"]      # responded to odor 1 (IAA)
    assert pf.loc["P2|d|1", "never_responder"]          # only the click got a response
    assert np.isnan(pf.loc["P2|d|1", "light_probe_score"])


def test_always_responder_includes_paraffin_and_click():
    e = _fly("C1|e|1", [("pre", "pretest_1_isoamylacetate-1", "IAA", 2),
                        ("post", "testing_8_paraffinoil-100", "Paraffin Oil", 3),
                        ("post", "testing_9_lightonly", None, 0)])
    f = _fly("C1|f|1", [("pre", "pretest_1_isoamylacetate-1", "IAA", 2),
                        ("post", "testing_8_paraffinoil-100", "Paraffin Oil", 1)])
    kw = dict(experiment="exp2", cohort="P1")
    g = _fly("P1|g|1", [("alone", "pretest_1_hexanol-0.1", "Hexanol", 3),
                        ("pair", "testing_1_isoamylacetate-1_then_paraffinoil-100",
                         "Paraffin Oil", 2, dict(prime="IAA", is_click=True, odor1_score=1))],
             **kw)
    pf = qc.run_qc(_frame(e, f, g), threshold=THR)[0].set_index("fly_id")
    assert pf.loc["C1|e|1", "always_responder"]         # the light probe does not count
    assert not pf.loc["C1|f|1", "always_responder"]     # paraffin below threshold
    assert not pf.loc["P1|g|1", "always_responder"]     # odor 1 of the pair below threshold


# ── completeness ──────────────────────────────────────────────────────────────
def _raw(rows):
    return pd.DataFrame([dict(dataset=d, folder=f, trial_label=l) for d, f, l in rows])


def test_missing_trials_against_the_folder_raw_files():
    a = _fly("C1|a|1", [("pre", "pretest_1_isoamylacetate-1", "IAA", 0),
                        ("post", "testing_1_isoamylacetate-1", "IAA", 0)])
    raw = _raw([("C1", "a", "pretest_1_isoamylacetate-1"), ("C1", "a", "pretest_2_citral-1"),
                ("C1", "a", "testing_1_isoamylacetate-1")])
    pf = qc.run_qc(_frame(a), raw, threshold=THR)[0].set_index("fly_id")
    assert pf.loc["C1|a|1", "n_missing_trials"] == 1
    assert pf.loc["C1|a|1", "missing_trials"] == "pretest_2_citral-1"
    assert pf.loc["C1|a|1", "incomplete"]


def test_missing_trials_without_raw_uses_the_folder_mates():
    a1 = _fly("C1|a|1", [("pre", "pretest_1_isoamylacetate-1", "IAA", 0),
                         ("pre", "pretest_2_citral-1", "Citral", 0)])
    a2 = _fly("C1|a|2", [("pre", "pretest_1_isoamylacetate-1", "IAA", 0)])
    pf = qc.run_qc(_frame(a1, a2), threshold=THR)[0].set_index("fly_id")
    assert pf.loc["C1|a|1", "n_missing_trials"] == 0
    assert pf.loc["C1|a|2", "n_missing_trials"] == 1


def test_protocol_gaps_known_vs_unexpected():
    full = [("pre", "pretest_1_paraffinoil-100", "Paraffin Oil", 0),
            ("pre", "pretest_2_isoamylacetate-1", "IAA", 0),
            ("exposure", "training_1_isoamylacetate-1", "IAA", 0),
            ("exposure", "training_2_isoamylacetate-1", "IAA", 0),
            ("post", "testing_1_isoamylacetate-1", "IAA", 0),
            ("post", "testing_8_paraffinoil-100", "Paraffin Oil", 0)]
    a = _fly("C1|a|1", full)
    b = _fly("C1|b|1", [r for r in full if "paraffin" not in r[1]])     # older panel
    c = _fly("C1|c|1", [r for r in full if r[1] != "training_2_isoamylacetate-1"])
    pf = qc.run_qc(_frame(a, b, c), threshold=THR)[0].set_index("fly_id")
    assert pf.loc["C1|a|1", "protocol_gaps_known"] == ""
    assert pf.loc["C1|b|1", "protocol_gaps_known"] == "post:Paraffin Oil; pre:Paraffin Oil"
    assert not pf.loc["C1|b|1", "incomplete"]
    assert pf.loc["C1|c|1", "protocol_gaps_unexpected"] == "exposure:IAA (1 of 2)"
    assert pf.loc["C1|c|1", "incomplete"]


def test_pair_light_probe_gap_is_known():
    kw = dict(experiment="exp2", cohort="P1")
    rows = [("alone", "pretest_1_isoamylacetate-1", "IAA", 0),
            ("pair", "testing_1_isoamylacetate-1_then_hexanol-0.1", "Hexanol", 0,
             dict(prime="IAA", odor1_score=0)),
            ("pair", "testing_2_isoamylacetate-1_then_hexanol-0.1", "Hexanol", 0,
             dict(prime="IAA", odor1_score=0))]
    a = _fly("P1|a|1", rows + [("pair", "testing_3_lightonly", None, 3)], **kw)
    b = _fly("P1|b|1", rows, **kw)
    pf = qc.run_qc(_frame(a, b), threshold=THR)[0].set_index("fly_id")
    assert pf.loc["P1|b|1", "protocol_gaps_known"] == "pair:light probe"
    assert not pf.loc["P1|b|1", "incomplete"]


# ── table integrity ───────────────────────────────────────────────────────────
def test_duplicates_out_of_range_and_short_windows():
    rows = [("alone", "pretest_1_paraffinoil-100", "Paraffin Oil", 0,
             dict(odor_window_s=0.025, is_click=True)),
            ("alone", "pretest_2_isoamylacetate-1", "IAA", 7),
            ("alone", "pretest_3_citral-1", "Citral", 0),
            ("pair", "testing_17_lightonly", None, 2, dict(odor_window_s=0.025))]
    a = _fly("P1|a|1", rows, experiment="exp2", cohort="P1")
    t = _frame(a)
    t = pd.concat([t, t.iloc[[2]]], ignore_index=True)
    pf, md = qc.run_qc(t, threshold=THR)
    r = pf.set_index("fly_id").loc["P1|a|1"]
    assert r.n_duplicate_trials == 1
    assert r.n_out_of_range_scores == 1
    assert r.n_short_odor_windows == 1                  # the light probe is not an odor window
    assert r.short_window_trials == "pretest_1_paraffinoil-100"
    for heading in ("Duplicate trials", "Out-of-range scores", "Short odor windows",
                    "Never-responders", "Always-responders", "Flies per cohort"):
        assert heading in md


def test_flies_per_cohort_window():
    a = _fly("C1|a|1", [LIGHT], window="day")
    b = _fly("C1|b|1", [LIGHT], window="evening")
    c = _fly("C1|c|1", [LIGHT], window="day")
    d = _fly("C2|d|1", [LIGHT], window="other", cohort="C2")
    tab = qc.flies_per_cohort_window(_frame(a, b, c, d))
    assert tab.loc["C1", "day"] == 2 and tab.loc["C1", "evening"] == 1
    assert tab.loc["C2", "other"] == 1 and tab.loc["C2", "day"] == 0
    assert tab.loc["C1", "total"] == 3


# ── exclusions and outputs ────────────────────────────────────────────────────
def _resp_frame():
    never = _fly("C1|n|1", [("pre", "pretest_1_isoamylacetate-1", "IAA", 0)])
    always = _fly("C1|a|1", [("pre", "pretest_1_isoamylacetate-1", "IAA", 4)])
    mixed = _fly("C1|m|1", [("pre", "pretest_1_isoamylacetate-1", "IAA", 0),
                            ("post", "testing_1_isoamylacetate-1", "IAA", 3)])
    return _frame(never, always, mixed)


def test_apply_exclusions():
    t = _resp_frame()
    pf, _ = qc.run_qc(t, threshold=THR)
    assert qc.apply_exclusions(t, pf, {}).equals(t)
    kept = qc.apply_exclusions(t, pf, {"never_responders": True})
    assert set(kept.fly_id) == {"C1|a|1", "C1|m|1"}
    kept = qc.apply_exclusions(t, pf, {"never_responders": True, "always_responders": True})
    assert set(kept.fly_id) == {"C1|m|1"}
    with pytest.raises(KeyError):
        qc.apply_exclusions(t, pf, {"sleepy": True})


def test_apply_exclusions_defaults_follow_config(monkeypatch):
    t = _resp_frame()
    pf, _ = qc.run_qc(t, threshold=THR)
    assert len(qc.apply_exclusions(t, pf)) == len(t)    # all switches off by default
    monkeypatch.setattr(qc.C, "EXCLUDE_ALWAYS_RESPONDERS", True)
    assert "C1|a|1" not in set(qc.apply_exclusions(t, pf).fly_id)


def test_write_qc_outputs(tmp_path):
    pf, md = qc.run_qc(_resp_frame(), threshold=THR)
    paths = qc.write_qc_outputs(pf, md, tmp_path)
    assert {p.name for p in paths} >= {"qc_report.csv", "qc_report.md", "README.md"}
    readme = (tmp_path / "README.md").read_text().strip().splitlines()
    assert 2 <= len(readme) <= 4
    assert len(pd.read_csv(tmp_path / "qc_report.csv")) == 3
