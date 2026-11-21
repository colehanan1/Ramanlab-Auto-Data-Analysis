"""src/per_analysis/cross.py: cross-experiment analyses, on synthetic tidy trials.

``make_trials`` (also imported by the power / daynight tests) builds a small tidy table
with the contract columns the analysis modules read.
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from per_analysis import config as C  # noqa: E402

ODORS = list(C.ODOR_ORDER)
CLICK = C.CLICK_ODOR
THRESHOLD = 2
COHORT_EXP = {"IAA-Sensitivity-1-1": "exp1", "Hex-Sensitivity-1-0.1": "exp1",
              "3Oct-Sensitivity-1-0.1": "exp1", "Paraffin-Sensitivity-1-100": "exp1",
              "IAA-PairFirst-Control-1-1": "exp2", "IAA-PairSecond-Control-1-1": "exp3"}
EXPOSURE = {"IAA-Sensitivity-1-1": "IAA", "Hex-Sensitivity-1-0.1": "Hexanol",
            "3Oct-Sensitivity-1-0.1": "3-Octanol", "Paraffin-Sensitivity-1-100": "Paraffin Oil"}


def session(name, window="day", start_h=11.0, date="2026-11-10", batches=(3, 3)):
    """A session spec: one batch folder per entry of ``batches`` (flies per folder)."""
    return dict(session=name, window=window, start_h=start_h, date=date, batches=list(batches))


def zero_score(ctx):
    return 0


def make_trials(cohorts, score_fn=zero_score, *, no_paraffin_batches=(), bad_click_flies=(),
                seed=0):
    """Synthetic tidy trials.

    ``cohorts``: {cohort: [session(...), ...]}. ``score_fn(ctx)`` -> int score, where
    ``ctx`` holds cohort, experiment, window, session, batch, fly_id, fly_index (0.. in
    the cohort), block, odor, prime, presentation, light (bool), rng. Batch folders named
    in ``no_paraffin_batches`` have no Paraffin Oil in pre/post (exp1); flies in
    ``bad_click_flies`` get a 0.025 s alone click (the known-bad trials)."""
    rng = np.random.default_rng(seed)
    rows = []
    for cohort, sessions in cohorts.items():
        exp = COHORT_EXP[cohort]
        fly_index = 0
        for s in sessions:
            for bi, n_flies in enumerate(s["batches"]):
                batch = s["session"] + ("" if bi == 0 else f"_rig_{bi + 1}")
                start_h = s["start_h"] + 0.1 * bi
                start = pd.Timestamp(s["date"]) + pd.Timedelta(hours=start_h)
                for f in range(1, n_flies + 1):
                    fly_id = f"{cohort}|{batch}|{f}"
                    base = dict(experiment=exp, cohort=cohort, fly_id=fly_id, batch=batch,
                                session=s["session"], window=s["window"], rig=bi + 1,
                                start_time=start, start_time_h=start_h,
                                date=pd.Timestamp(s["date"]),
                                exposure_odor=EXPOSURE.get(cohort))
                    trials = _protocol(exp, cohort, batch in no_paraffin_batches)
                    pos = {}
                    for t, (phase, block, odor, prime, pres, light) in enumerate(trials):
                        pos[block] = pos.get(block, 0) + 1
                        ctx = dict(base, fly_index=fly_index, block=block, odor=odor,
                                   prime=prime, presentation=pres, light=light, rng=rng)
                        score = int(score_fn(ctx))
                        o1 = np.nan
                        if prime is not None:
                            o1 = float(score_fn(dict(ctx, odor=prime, prime=None,
                                                     block="prime", presentation=None)))
                        window_s = 30.0
                        if light and exp != "exp1":
                            window_s = 0.025                  # the invalid pair probes
                        if (fly_id in bad_click_flies and block == "alone"
                                and odor == CLICK):
                            window_s = 0.025
                        is_click = (exp != "exp1") and (odor == CLICK or prime == CLICK)
                        label = (f"{phase}_{pos[block]}_lightonly" if light else
                                 f"{phase}_{pos[block]}_{odor}")
                        rows.append(dict(base, block=block, trial_label=label.lower(),
                                         trial_index=pos[block], position_in_block=pos[block],
                                         odor=None if light else odor,
                                         prime=prime,
                                         partner=(None if light else
                                                  (prime if exp == "exp3" and prime else odor)),
                                         presentation=pres, is_light_probe=light,
                                         is_click=is_click, paraffin_off_protocol=False,
                                         odor_window_s=window_s, score=score,
                                         responded=score >= THRESHOLD, odor1_score=o1,
                                         trial_time=start + pd.Timedelta(minutes=t),
                                         phase=phase))
                    fly_index += 1
    df = pd.DataFrame(rows)
    for c in ("odor", "prime", "partner", "exposure_odor"):
        df[c] = df[c].astype("str")
    df["presentation"] = df["presentation"].astype(float)
    df["score"] = df["score"].astype("int64")
    return df


def _protocol(exp, cohort, no_paraffin):
    """(phase, block, odor, prime, presentation, light) per trial."""
    out = []
    if exp == "exp1":
        panel = [o for o in ODORS if not (o == CLICK and (no_paraffin
                                                          or cohort == "IAA-Sensitivity-1-1"))]
        out += [("pretest", "pre", o, None, None, False) for o in panel]
        out += [("training", "exposure", EXPOSURE[cohort], None, None, False)] * 6
        out += [("testing", "post", o, None, None, False) for o in panel]
        out += [("testing", "post", None, None, None, True)]
    else:
        out += [("pretest", "alone", o, None, None, False) for o in ODORS]
        for pres in (1.0, 2.0):
            for x in ODORS:
                if exp == "exp2":
                    out.append(("testing", "pair", x, "IAA", pres, False))
                else:
                    out.append(("testing", "pair", "IAA", x, pres, False))
        if exp == "exp2":
            out.append(("testing", "pair", None, None, None, True))
    return out


# ── the generator itself ─────────────────────────────────────────────────────
def test_make_trials_shapes():
    t = make_trials({"IAA-PairFirst-Control-1-1": [session("november_13_batch_1")],
                     "IAA-PairSecond-Control-1-1": [session("november_13_batch_1",
                                                            batches=(0, 0, 2))],
                     "Hex-Sensitivity-1-0.1": [session("november_04_batch_1", batches=(2,))]})
    e2 = t[t.experiment == "exp2"]
    assert e2.fly_id.nunique() == 6 and len(e2) == 6 * (8 + 16 + 1)
    e3 = t[(t.experiment == "exp3") & (t.block == "pair")]
    assert set(e3.odor) == {"IAA"} and set(e3.prime) == set(ODORS)
    assert t.loc[t.is_light_probe, "odor"].isna().all()


# ── the module ───────────────────────────────────────────────────────────────
cross = pytest.importorskip("per_analysis.cross")


@pytest.fixture(autouse=True)
def _fast(monkeypatch):
    monkeypatch.setattr(cross, "N_BOOT", 500)
    monkeypatch.setattr(cross, "N_PERM_FLY", 500)


def test_odor_trials_drop_light_probes_and_short_click_windows():
    t = make_trials({"IAA-PairSecond-Control-1-1": [session("november_13_batch_1")]},
                    bad_click_flies={"IAA-PairSecond-Control-1-1|november_13_batch_1|1"})
    o = cross.odor_trials(t)
    assert not o["is_light_probe"].any()
    assert (o["odor_window_s"] >= 1).all()
    assert len(t) - len(o) == 1


def test_noise_floor_is_the_within_fly_difference():
    """Hexanol = the fly's own paraffin + 2 for flies tested on paraffin; flies never
    tested on paraffin answer Hexanol with 5 and must not enter. Citral = paraffin."""
    def score(ctx):
        k = ctx["fly_index"] % 3                       # baselines differ between flies
        if ctx["odor"] == "Hexanol":
            return 5 if ctx["batch"] == "november_04_batch_1" else k + 2
        return k
    cohorts = {"Hex-Sensitivity-1-0.1": [session("november_04_batch_1", batches=(4, 10))],
               "IAA-Sensitivity-1-1": [session("october_12_batch_1", batches=(4,))],
               "IAA-PairSecond-Control-1-1": [session("november_13_batch_1",
                                                      batches=(4, 4))]}
    bad = {f"IAA-PairSecond-Control-1-1|november_13_batch_1|{i}" for i in (1, 2)}
    t = make_trials(cohorts, score, no_paraffin_batches={"november_04_batch_1"},
                    bad_click_flies=bad)
    tab = cross.noise_floor_table(cross.odor_trials(t))
    hx = tab[(tab.cohort == "Hex-Sensitivity-1-0.1") & (tab.odor == "Hexanol")
             & (tab.metric == "score")].iloc[0]
    assert hx["effect"] == pytest.approx(2.0)          # within fly, not 5 - mean(paraffin)
    assert hx["n_flies"] == 10                         # only the flies with paraffin
    assert hx["flag"] != "not distinguishable from paraffin"
    ci = tab[(tab.cohort == "Hex-Sensitivity-1-0.1") & (tab.odor == "Citral")
             & (tab.metric == "score")].iloc[0]
    assert ci["effect"] == pytest.approx(0.0)
    assert ci["flag"] == "not distinguishable from paraffin"
    assert {"p", "p_holm", "q_bh", "p_batch", "ci_lo", "ci_hi", "test", "correction"} <= set(tab)
    iaa = tab[tab.cohort == "IAA-Sensitivity-1-1"]
    assert len(iaa) == 1 and iaa.iloc[0]["flag"] == "skipped"
    assert "paraffin" in iaa.iloc[0]["note"].lower()
    p2 = tab[(tab.cohort == "IAA-PairSecond-Control-1-1") & (tab.metric == "score")]
    assert (p2["n_flies"] == 6).all()                  # the 2 bad-click flies are out


def _pair_cohorts(score):
    cohorts = {"IAA-PairFirst-Control-1-1": [session("november_13_batch_1", batches=(4, 4)),
                                             session("november_14_batch_1", batches=(4,))],
               "IAA-PairSecond-Control-1-1": [session("november_13_batch_1",
                                                      batches=(0, 0, 4)),
                                              session("november_16_batch_1", batches=(4, 4))]}
    return make_trials(cohorts, score)


def test_forest_table_rows_and_sort():
    """Exp 2 IAA->EB raised by 3 score points against the click; Exp 3 Hexanol->IAA
    lowered by 1 against click->IAA. Every prime->target pair of both experiments is a
    row, sorted by effect, largest first, with Holm over the whole forest."""
    def score(ctx):
        if ctx["block"] == "pair" and ctx["experiment"] == "exp2" \
                and ctx["odor"] == "Ethyl Butyrate":
            return 4
        if ctx["block"] == "pair" and ctx["experiment"] == "exp3":
            return 1 if ctx["prime"] == "Hexanol" else 2
        return 1
    f = cross.forest_table(cross.odor_trials(_pair_cohorts(score)), metric="score")
    assert len(f) == 14
    assert list(f["effect"]) == sorted(f["effect"], reverse=True)
    top, bottom = f.iloc[0], f.iloc[-1]
    assert top["experiment"] == "exp2" and top["target"] == "Ethyl Butyrate"
    assert top["effect"] == pytest.approx(3.0)
    assert bottom["experiment"] == "exp3" and bottom["prime"] == "Hexanol"
    assert bottom["effect"] == pytest.approx(-1.0)
    assert (f["p_holm"] >= f["p"] - 1e-12).all()
    assert not ((f["experiment"] == "exp2") & (f["target"] == CLICK)).any()
    assert not ((f["experiment"] == "exp3") & (f["prime"] == CLICK)).any()


def test_iaa_conditions_exp2_has_no_click_primed():
    t = cross.odor_trials(_pair_cohorts(zero_score))
    d = cross.iaa_condition_trials(t)
    got = d.groupby("experiment")["condition"].agg(set).to_dict()
    assert got["exp2"] == {"alone", "IAA-primed"}
    assert got["exp3"] == {"alone", "IAA-primed", "click-primed"}
    # each fly: one alone IAA trial and two IAA->IAA presentations
    n = d.groupby(["fly_id", "condition"]).size()
    assert set(n.xs("alone", level="condition")) == {1}
    assert set(n.xs("IAA-primed", level="condition")) == {2}


def test_two_group_effect_and_session_strata():
    """b - a in pp for rates; the batch p keeps a shared session's labels inside it."""
    a = pd.Series([0, 0, 0, 0, 0, 0], index=[f"a{i}" for i in range(6)], dtype=float)
    b = pd.Series([1, 1, 1, 1, 1, 1], index=[f"b{i}" for i in range(6)], dtype=float)
    batch_of = {**{f"a{i}": f"A|s{i // 2}_x" for i in range(6)},
                **{f"b{i}": f"B|s{i // 2}_y" for i in range(6)}}
    session_of_batch = {f"A|s{k}_x": f"s{k}" for k in range(3)}
    session_of_batch.update({f"B|s{k}_y": f"s{k}" for k in range(3)})
    r = cross.two_group(a, b, batch_of, session_of_batch, scale=100.0)
    assert r["effect"] == pytest.approx(100.0)
    assert r["n_flies"] == 12 and r["n_batches"] == 6
    assert r["strata"] == "session"
    # 3 shared sessions, one batch per group each: 2^3 arrangements, 2 as extreme
    assert r["p_batch"] == pytest.approx(2 / 8)
    # without shared sessions the 6 batches swap freely: C(6, 3) = 20, 2 as extreme
    free = cross.two_group(a, b, batch_of, {k: k for k in session_of_batch}, scale=100.0)
    assert free["strata"] == "none" and free["p_batch"] == pytest.approx(2 / 20)


def test_run_writes_outputs(tmp_path):
    def score(ctx):
        r = ctx["rng"]
        return int(r.integers(0, 4))
    cohorts = {"IAA-Sensitivity-1-1": [session("october_12_batch_1", batches=(3, 3))],
               "Hex-Sensitivity-1-0.1": [session("november_04_batch_1", batches=(3, 3))],
               "3Oct-Sensitivity-1-0.1": [session("november_04_batch_2", batches=(3, 3))],
               "Paraffin-Sensitivity-1-100": [session("november_05_batch_1",
                                                      batches=(3, 3))],
               "IAA-PairFirst-Control-1-1": [session("november_13_batch_1", batches=(3, 3)),
                                             session("november_14_batch_1", batches=(3,))],
               "IAA-PairSecond-Control-1-1": [session("november_13_batch_1",
                                                      batches=(0, 0, 3)),
                                              session("november_16_batch_1", batches=(3, 3))]}
    t = make_trials(cohorts, score)
    out, tables = tmp_path / "results", tmp_path / "results" / "tables"
    findings = cross.run(t, out, tables=tables)
    assert findings and all(f["section"] == "cross" for f in findings)
    for f in findings:
        assert {"section", "analysis", "primary", "text", "numbers"} <= set(f)
    root = out / "05_cross_experiment"
    for sub in ("a_iaa_baseline_by_cohort", "b_iaa_iaa_across_experiments",
                "c_eb_iaa_bidirectional", "d_paraffin_noise_floor", "models"):
        assert (root / sub / "README.md").exists()
        assert any((root / sub).iterdir())
    assert (root / "README.md").exists()
    pngs = list(root.rglob("*.png"))
    assert pngs and all(p.with_suffix(".svg").exists() and p.with_suffix(".csv").exists()
                        for p in pngs)
    for csv in tables.glob("cross_*.csv"):
        d = pd.read_csv(csv)
        assert {"n_flies", "effect", "ci_lo", "ci_hi", "p", "p_holm", "q_bh", "p_batch",
                "test", "correction"} <= set(d.columns), csv.name
    assert list((root / "models").glob("*_coefs.csv"))


def test_mixed_logistic_is_sensitivity_only_and_sign_checked(tmp_path):
    assert cross.sign_check({"estimate": -0.4}, {"estimate": 0.36}) == "DISAGREES"
    assert cross.sign_check({"estimate": 0.4}, {"estimate": 0.1}) == "agrees"
    assert cross.sign_check({"estimate": np.nan}, {"estimate": 0.1}) == "n/a"
    t = cross.odor_trials(_pair_cohorts(lambda ctx: int(ctx["rng"].integers(0, 4))))
    cross.iaa_iaa_models(cross.iaa_condition_trials(t), tmp_path)
    txt = (tmp_path / "cross_b_mixed_logistic_responded_condition_x_experiment.txt").read_text()
    assert "SENSITIVITY ONLY" in txt and "GEE is the model of record" in txt
    gee = (tmp_path / "cross_b_gee_responded_condition_x_experiment.txt").read_text()
    assert "SENSITIVITY ONLY" not in gee
