"""src/per_analysis/exp1.py: Exp 1 (sensitivity exposure cohorts) analyses."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from per_analysis import exp1  # noqa: E402
from per_analysis import plots as pl  # noqa: E402
from per_analysis import stats as st  # noqa: E402

CONTROL = "Paraffin-Sensitivity-1-100"
#: cohort -> (exposure odor, exposure concentration, batch-name token)
SPEC = {"IAA-Sensitivity-1-1": ("IAA", 1.0, "iaa"),
        "Hex-Sensitivity-1-0.1": ("Hexanol", 0.1, "hex"),
        "3Oct-Sensitivity-1-0.1": ("3-Octanol", 0.1, "oct"),
        CONTROL: ("Paraffin Oil", 100.0, "par")}
ODORS = ["IAA", "Hexanol", "3-Octanol", "Benzaldehyde", "Paraffin Oil"]
FAST = {"n_boot": 300, "n_perm": 300}


def _score(rng, responded: bool) -> int:
    return int(rng.integers(2, 6)) if responded else int(rng.integers(-1, 2))


def make_trials(n_flies: int = 12, flies_per_batch: int = 4, *, effects=None, seed: int = 0,
                no_paraffin: dict | None = None, base: float = 0.3) -> pd.DataFrame:
    """Synthetic tidy trials shaped like ``io.load_trials()`` for exp1 (plus noise rows).

    ``effects``: (cohort, odor) -> (p_pre, p_post) response probabilities; others
    ``base`` in both blocks. The IAA cohort never runs Paraffin Oil; ``no_paraffin``:
    cohort -> number of its flies (from the first) that never ran it either."""
    rng = np.random.default_rng(seed)
    effects = dict(effects or {})
    no_paraffin = dict(no_paraffin or {})
    rows = []
    for ci, (cohort, (exp_odor, conc, tok)) in enumerate(SPEC.items()):
        for f in range(n_flies):
            b = f // flies_per_batch
            batch = f"{tok}_batch_{b}"
            start = 10.0 + 0.9 * b + 0.3 * ci
            has_par = cohort != "IAA-Sensitivity-1-1" and f >= no_paraffin.get(cohort, 0)
            odors = [o for o in ODORS if o != "Paraffin Oil" or has_par]
            common = dict(experiment="exp1", cohort=cohort, dataset=cohort,
                          exposure_odor=exp_odor, exposure_conc=conc, starvation_h=1,
                          fly_id=f"{cohort}|{batch}|{f}", batch=batch, fly=batch,
                          fly_number=f, rig=1 + b % 3, host="Flybehavior2", window="day",
                          start_time_h=start, is_click=False, odor_window_s=30.0,
                          paraffin_off_protocol=cohort == CONTROL, odor1_score=np.nan)
            pre_order = (["Paraffin Oil"] if has_par else []) + \
                [o for o in odors if o != "Paraffin Oil"]
            post_order = list(np.random.default_rng(100 + b).permutation(
                [o for o in odors if o != "Paraffin Oil"])) + (["Paraffin Oil"] if has_par else [])
            for block, order in (("pre", pre_order), ("post", post_order)):
                for pos, odor in enumerate(order, start=1):
                    p_pre, p_post = effects.get((cohort, odor), (base, base))
                    r = bool(rng.random() < (p_pre if block == "pre" else p_post))
                    rows.append({**common, "block": block, "odor": odor, "position_in_block": pos,
                                 "is_light_probe": False, "responded": r,
                                 "score": _score(rng, r),
                                 "minutes_since_exposure_end": (1.5 + 5.5 * (pos - 1)
                                                                if block == "post" else np.nan)})
            for pos in range(1, 7):                       # exposure pulses
                r = bool(rng.random() < 0.5)
                rows.append({**common, "block": "exposure", "odor": exp_odor,
                             "position_in_block": pos, "is_light_probe": False, "responded": r,
                             "score": _score(rng, r), "minutes_since_exposure_end": np.nan})
            r = bool(rng.random() < 0.6)                   # light probe ends the post block
            rows.append({**common, "block": "post", "odor": np.nan,
                         "position_in_block": len(post_order) + 1, "is_light_probe": True,
                         "responded": r, "score": _score(rng, r),
                         "minutes_since_exposure_end": 1.5 + 5.5 * len(post_order)})
    # noise: another experiment, and one exp1 trial with a broken 0.025 s window
    rows.append({**rows[0], "experiment": "exp2", "cohort": "IAA-PairFirst-Control-1-1",
                 "fly_id": "pair|x|1", "batch": "pair_batch"})
    rows.append({**rows[0], "odor": "Benzaldehyde", "odor_window_s": 0.025,
                 "position_in_block": 99})
    return pd.DataFrame(rows)


PLANTED = {("IAA-Sensitivity-1-1", "IAA"): (0.1, 0.9)}


@pytest.fixture(autouse=True)
def _reset_ylims():
    pl.reset_ylims()
    yield
    pl.reset_ylims()


# ── data rules ───────────────────────────────────────────────────────────────
def test_prepare_keeps_exp1_odor_trials_and_logs_short_windows(caplog):
    t = make_trials()
    with caplog.at_level(logging.INFO, logger="per_analysis.exp1"):
        d = exp1.prepare(t)
    assert set(d["experiment"]) == {"exp1"}
    assert not d["is_light_probe"].any()
    assert (d["odor_window_s"] >= 1).all()
    assert len(d) == len(t) - 1 - 1 - int(t["is_light_probe"].sum())
    assert any("odor_window_s" in r.getMessage() for r in caplog.records)
    # start time centred over flies
    per_fly = d.drop_duplicates("fly_id")
    assert per_fly["start_time_c"].mean() == pytest.approx(0.0, abs=1e-9)


def test_fly_changes_one_row_per_fly_and_odor():
    d = exp1.prepare(make_trials())
    ch = exp1.fly_changes(d, "rate")
    assert not ch.duplicated(["cohort", "odor", "fly_id"]).any()
    assert set(ch["change"].unique()) <= {-1.0, 0.0, 1.0}
    assert (ch["change"] == ch["post"] - ch["pre"]).all()
    sc = exp1.fly_changes(d, "score")
    assert sc["change"].abs().max() <= 6


# ── covariates ───────────────────────────────────────────────────────────────
def test_add_covariates_drops_aliased_and_undefined_terms():
    d = exp1.prepare(make_trials())
    d = d[d["block"].isin(["pre", "post"])]
    formula, notes = exp1.add_covariates(
        d, "responded ~ post * C(grp, Treatment('Paraffin'))",
        ("start_time_c", "position_in_block", "exposure_conc", "minutes_since_exposure_end"))
    assert "start_time_c" in formula and "position_in_block" in formula
    assert "exposure_conc" not in formula                  # one concentration per cohort
    assert "minutes_since_exposure_end" not in formula     # undefined on pre rows
    joined = " ".join(notes)
    assert "exposure_conc" in joined and "minutes_since_exposure_end" in joined

    post = d[(d["block"] == "post") & (d["cohort"] == "IAA-Sensitivity-1-1")].copy()
    post["min10"] = post["minutes_since_exposure_end"] / 10
    formula, notes = exp1.add_covariates(post, "responded ~ min10",
                                         ("start_time_c", "position_in_block"))
    assert "position_in_block" not in formula               # the same clock as min10
    assert any("position_in_block" in n for n in notes)


# ── difference in differences ────────────────────────────────────────────────
def _did(trials, metric="rate"):
    return exp1.did_vs_control_table(exp1.prepare(trials), metric, **FAST)


def test_did_recovers_a_planted_effect():
    tab = _did(make_trials(n_flies=24, effects=PLANTED, seed=1))
    row = tab[(tab["cohort"] == "IAA-Sensitivity-1-1") & (tab["odor"] == "IAA")].iloc[0]
    assert row["primary"]
    assert row["effect"] == pytest.approx(80, abs=25)        # percentage points
    assert row["ci_lo"] > 0
    assert row["p"] < 0.01
    assert row["n_flies_exposed"] == 24 and row["n_flies_control"] == 24
    score = _did(make_trials(n_flies=24, effects=PLANTED, seed=1), "score")
    srow = score[(score["cohort"] == "IAA-Sensitivity-1-1") & (score["odor"] == "IAA")].iloc[0]
    assert srow["ci_lo"] > 0


def test_did_finds_nothing_when_nothing_is_planted():
    tab = _did(make_trials(n_flies=24, seed=2))
    prim = tab[tab["primary"]]
    assert len(prim) == 3
    for _, r in prim.iterrows():
        assert r["ci_lo"] < 0 < r["ci_hi"], r[["cohort", "odor", "ci_lo", "ci_hi"]]
        assert r["p_holm"] > 0.05


def test_primary_contrasts_are_holm_corrected_over_three():
    tab = _did(make_trials(n_flies=12, effects=PLANTED, seed=3))
    prim = tab[tab["primary"]]
    assert sorted(zip(prim["cohort"], prim["odor"])) == sorted(
        [("IAA-Sensitivity-1-1", "IAA"), ("Hex-Sensitivity-1-0.1", "Hexanol"),
         ("3Oct-Sensitivity-1-0.1", "3-Octanol")])
    np.testing.assert_allclose(prim["p_holm"], st.holm(prim["p"].tolist()))
    assert (prim["family"] == "primary").all()
    # exploratory rows: Holm within each cohort's panel, primaries left out
    for cohort, sub in tab[~tab["primary"] & tab["p"].notna()].groupby("cohort"):
        np.testing.assert_allclose(sub["p_holm"], st.holm(sub["p"].tolist()))
        assert sub["family"].nunique() == 1 and "exploratory" in sub["family"].iloc[0]
    for col in exp1.TABLE_COLUMNS + ["p_batch"]:
        assert col in tab.columns


def test_paraffin_rows_use_only_flies_that_ran_paraffin():
    t = make_trials(n_flies=12, no_paraffin={"Hex-Sensitivity-1-0.1": 5}, seed=4)
    d = exp1.prepare(t)
    ch = exp1.fly_changes(d, "rate")
    par = ch[ch["odor"] == "Paraffin Oil"]
    assert "IAA-Sensitivity-1-1" not in set(par["cohort"])
    assert (par["cohort"] == "Hex-Sensitivity-1-0.1").sum() == 7
    tab = exp1.did_vs_control_table(d, "rate", **FAST)
    hex_par = tab[(tab["cohort"] == "Hex-Sensitivity-1-0.1") & (tab["odor"] == "Paraffin Oil")]
    assert hex_par["n_flies_exposed"].iloc[0] == 7
    assert "7 of 12" in hex_par["note"].iloc[0]
    iaa_par = tab[(tab["cohort"] == "IAA-Sensitivity-1-1") & (tab["odor"] == "Paraffin Oil")]
    assert len(iaa_par) == 1 and iaa_par["n_flies_exposed"].iloc[0] == 0
    assert np.isnan(iaa_par["p"].iloc[0]) and "never" in iaa_par["note"].iloc[0].lower()
    # a 'not tested' row does not enter its panel's Holm family
    expl = tab[(tab["cohort"] == "IAA-Sensitivity-1-1") & ~tab["primary"] & tab["p"].notna()]
    np.testing.assert_allclose(expl["p_holm"], st.holm(expl["p"].tolist()))


# ── other analyses ───────────────────────────────────────────────────────────
def test_pre_vs_post_holm_within_each_cohort_panel():
    d = exp1.prepare(make_trials(effects=PLANTED, seed=5))
    tab = exp1.pre_vs_post_table(d, "rate", **FAST)
    for cohort, sub in tab[tab["p"].notna()].groupby("cohort"):
        np.testing.assert_allclose(sub["p_holm"], st.holm(sub["p"].tolist()))
    iaa = tab[tab["cohort"] == "IAA-Sensitivity-1-1"]
    assert iaa.loc[iaa["odor"] == "IAA", "exposed_odor"].iloc[0]
    assert np.isnan(iaa.loc[iaa["odor"] == "Paraffin Oil", "p"].iloc[0])


def test_drift_uses_the_odors_every_cohort_ran():
    d = exp1.prepare(make_trials(no_paraffin={"Hex-Sensitivity-1-0.1": 3}))
    drift = exp1.fly_drift(d, "rate")
    assert (drift["n_odors"] == 4).all()                    # Paraffin Oil left out
    assert drift["fly_id"].is_unique
    ch = exp1.fly_changes(d, "rate")
    one = drift.iloc[0]
    sub = ch[(ch["fly_id"] == one["fly_id"]) & (ch["odor"] != "Paraffin Oil")]
    assert one["drift"] == pytest.approx(sub["change"].mean())


def test_suppression_pools_every_other_cohort():
    d = exp1.prepare(make_trials(n_flies=12, effects=PLANTED, seed=6))
    tab = exp1.suppression_table(d, "rate", **FAST)
    assert list(tab["odor"]) == ["IAA", "Hexanol", "3-Octanol"]
    iaa = tab[tab["odor"] == "IAA"].iloc[0]
    assert iaa["n_flies_exposed"] == 12 and iaa["n_flies_control"] == 36
    np.testing.assert_allclose(tab["p_holm"], st.holm(tab["p"].tolist()))


# ── run() end to end ─────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def ran(tmp_path_factory):
    pl.reset_ylims()
    out = tmp_path_factory.mktemp("results")
    tables = out / "tables"
    findings = exp1.run(make_trials(n_flies=8, effects=PLANTED, seed=7), out, tables=tables,
                        n_boot=200, n_perm=200, ordinal_boot=4)
    return out, tables, findings


EXPECTED = {
    "a_pre_vs_post": ["exp1_pre_vs_post_rate", "exp1_pre_vs_post_score"],
    "b_did_vs_paraffin": ["exp1_did_vs_paraffin_rate", "exp1_did_vs_paraffin_score"],
    "c_global_drift": ["exp1_global_drift_rate", "exp1_global_drift_score",
                       "exp1_global_drift_vs_start_rate", "exp1_global_drift_vs_start_score"],
    "d_time_since_exposure": ["exp1_time_since_exposure_rate",
                              "exp1_time_since_exposure_score"],
    "e_odor_specific_suppression": ["exp1_odor_specific_suppression_rate",
                                    "exp1_odor_specific_suppression_score"],
}


def test_run_writes_the_expected_files(ran):
    out, tables, _ = ran
    root = out / "02_exp1_exposure_cohorts"
    assert (root / "README.md").exists()
    for sub, stems in EXPECTED.items():
        assert (root / sub / "README.md").exists(), sub
        for stem in stems:
            for ext in ("png", "svg", "csv"):
                assert (root / sub / f"{stem}.{ext}").exists(), f"{sub}/{stem}.{ext}"
            tab = pd.read_csv(tables / f"{stem}.csv")
            for col in exp1.TABLE_COLUMNS:
                assert col in tab.columns, (stem, col)
    assert (tables / "exp1_did_vs_paraffin_models.csv").exists()
    models = root / "models"
    assert (models / "README.md").exists()
    txts = sorted(p.name for p in models.glob("*.txt"))
    assert any("full_gee" in n for n in txts) and any("full_glmm" in n for n in txts)
    assert any("ordinal" in n for n in txts)
    for txt in models.glob("*.txt"):
        assert txt.with_name(txt.stem + "_coefs.csv").exists()


def test_run_returns_findings_with_the_contract_keys(ran):
    _, _, findings = ran
    assert findings
    for f in findings:
        assert set(f) >= {"section", "analysis", "primary", "text", "numbers"}
        assert f["section"] == "exp1" and isinstance(f["numbers"], dict)
        assert isinstance(f["text"], str) and f["text"]
    prim = [f for f in findings if f["primary"]]
    assert len(prim) == 3
    assert all(f["analysis"] == "b_did_vs_paraffin" for f in prim)
    analyses = {f["analysis"] for f in findings}
    assert {"a_pre_vs_post", "c_global_drift", "d_time_since_exposure",
            "e_odor_specific_suppression"} <= analyses
