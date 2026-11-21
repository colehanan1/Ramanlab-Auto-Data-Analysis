"""src/per_analysis/pairs.py, exp2.py, exp3.py: the IAA odor-pair analyses.

Synthetic pair trials in the tidy-table shape (``per_analysis.io`` contract columns):
an alone block (8 odors, the click first) and a pair block (each partner twice).
exp2: IAA then X (X scored); exp3: X then IAA (IAA scored).
"""
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

from per_analysis import exp2, exp3, pairs  # noqa: E402
from per_analysis import plots as pl  # noqa: E402
from per_analysis import stats as st  # noqa: E402

ODORS = ["IAA", "Hexanol", "3-Octanol", "Benzaldehyde", "Citral", "Ethyl Butyrate",
         "Linalool"]
CLICK = "Paraffin Oil"
PARTNERS = ODORS + [CLICK]
COHORT = {"exp2": "IAA-PairFirst-Control-1-1", "exp3": "IAA-PairSecond-Control-1-1"}
TABLE_COLS = ["n_flies", "effect", "ci_lo", "ci_hi", "p", "p_holm", "q_bh", "p_batch",
              "test", "correction"]
FAST = {"n_boot": 400}


def _zero(*_):
    return 0


def make_trials(experiment, *, n_batches=3, flies_per_batch=4, window="day",
                start_h=11.0, prefix="", score_fn=_zero, odor1_fn=_zero,
                light_probe=False, short_click=False):
    """Tidy-shaped rows. ``score_fn(fly_i, condition, partner, presentation)`` -> score
    of the scored odor; ``odor1_fn(fly_i, partner, presentation)`` -> odor-1 score."""
    rows = []
    fly_i = 0
    for b in range(n_batches):
        batch = f"{prefix}november_1{b}_batch_1"
        for n in range(1, flies_per_batch + 1):
            base = dict(experiment=experiment, cohort=COHORT[experiment],
                        fly_id=f"{COHORT[experiment]}|{batch}|{n}", batch=batch,
                        session=batch, window=window, start_time_h=start_h + 0.25 * b,
                        fly_number=n, dataset=COHORT[experiment], fly=batch)
            alone = [CLICK] + ODORS[::-1] if fly_i % 2 else [CLICK] + ODORS
            for pos, odor in enumerate(alone, start=1):
                s = int(score_fn(fly_i, "alone", odor, None))
                label = f"pretest_{pos}_{odor.lower()}"
                rows.append({**base, "block": "alone", "trial_label": label,
                             "scored_label": label, "trial_index": pos,
                             "position_in_block": pos, "odor": odor, "prime": np.nan,
                             "partner": odor, "presentation": np.nan,
                             "is_light_probe": False, "is_click": odor == CLICK,
                             "odor_window_s": 0.025 if (short_click and odor == CLICK)
                             else 10.0,
                             "score": s, "odor1_score": np.nan,
                             "minutes_since_block_start": 5.0 * (pos - 1)})
            order = [(x, 1) for x in PARTNERS] + [(x, 2) for x in PARTNERS[::-1]]
            rot = fly_i % 8
            order = order[rot:8] + order[:rot] + order[8:]
            for pos, (x, pres) in enumerate(order, start=1):
                odor, prime = (x, "IAA") if experiment == "exp2" else ("IAA", x)
                s = int(score_fn(fly_i, "pair", x, pres))
                label = f"testing_{pos}_{prime.lower()}_then_{odor.lower()}"
                rows.append({**base, "block": "pair", "trial_label": label,
                             "scored_label": label + "__odor2", "trial_index": pos,
                             "position_in_block": pos, "odor": odor, "prime": prime,
                             "partner": x, "presentation": float(pres),
                             "is_light_probe": False, "is_click": x == CLICK,
                             "odor_window_s": 10.0, "score": s,
                             "odor1_score": float(odor1_fn(fly_i, x, pres)),
                             "minutes_since_block_start": 5.0 * (pos - 1)})
            if light_probe:
                rows.append({**base, "block": "pair", "trial_label": "testing_17_lightonly",
                             "scored_label": "testing_17_lightonly", "trial_index": 17,
                             "position_in_block": 17, "odor": np.nan, "prime": np.nan,
                             "partner": np.nan, "presentation": np.nan,
                             "is_light_probe": True, "is_click": False,
                             "odor_window_s": 0.025, "score": 3, "odor1_score": np.nan,
                             "minutes_since_block_start": 80.0})
            fly_i += 1
    df = pd.DataFrame(rows)
    df["responded"] = df["score"] >= 2
    return df


@pytest.fixture(autouse=True)
def _reset():
    pl.reset_ylims()
    yield
    pl.reset_ylims()


# ── data rules ───────────────────────────────────────────────────────────────
def test_prepare_drops_light_probes_and_short_alone_clicks(caplog):
    df = make_trials("exp3", light_probe=True, short_click=True)
    other = make_trials("exp2", n_batches=1)
    with caplog.at_level(logging.INFO):
        d = pairs.prepare(pd.concat([df, other]), "exp3")
    assert set(d["experiment"]) == {"exp3"}
    assert not d["is_light_probe"].any()
    assert not ((d["condition"] == "alone") & (d["odor_window_s"] < 1)).any()
    assert len(d) == len(df) - 12 - 12            # 12 light probes, 12 short clicks
    text = caplog.text.lower()
    assert "light" in text and "12" in text and "window" in text


def test_prepare_codes_condition_presentation_and_odor1():
    df = make_trials("exp3", odor1_fn=lambda f, x, pres: 3 if x == "Linalool" else 0)
    d = pairs.prepare(df, "exp3")
    assert set(d["condition"]) == {"alone", "pair"}
    assert set(d.loc[d.condition == "pair", "pres"]) == {"1st", "2nd"}
    assert set(d.loc[d.condition == "alone", "pres"]) == {"alone"}
    alone = d[d.condition == "alone"]
    assert (alone["odor1_responded"] == 0).all()          # no odor 1 before an alone trial
    lin = d[(d.condition == "pair") & (d.partner == "Linalool")]
    assert (lin["odor1_responded"] == 1).all()
    assert (d.loc[d.pres == "2nd", "second_presentation"] == 1).all()
    assert (d.loc[d.pres != "2nd", "second_presentation"] == 0).all()


def test_scored_long_exp3_copies_the_iaa_alone_baseline_per_prime():
    d = pairs.prepare(make_trials("exp3"), "exp3")
    long = pairs.scored_long(d, "exp3")
    alone = long[long.condition == "alone"]
    assert set(alone["odor"]) == {"IAA"}                  # only IAA is the baseline
    assert set(alone["group"]) == set(PARTNERS)
    assert alone.groupby("group").size().nunique() == 1   # one copy per prime
    assert long.loc[long.condition == "alone", "_uid"].nunique() == 12


# ── windows ──────────────────────────────────────────────────────────────────
def test_split_windows_and_file_stem(caplog):
    df = pd.concat([make_trials("exp2"), make_trials("exp2", n_batches=1, window="evening",
                                                     prefix="eve_", start_h=18.0),
                    make_trials("exp2", n_batches=1, window="other", prefix="late_",
                                start_h=22.0)])
    d = pairs.prepare(df, "exp2")
    with caplog.at_level(logging.INFO):
        parts = pairs.split_windows(d)
    assert [w for w, _ in parts] == ["day", "evening"]
    assert dict(parts)["evening"]["fly_id"].nunique() == 4
    assert "other" in caplog.text
    assert pairs.stem("exp3", "vs_paraffin_reference", "rate", "day") == \
        "exp3_vs_paraffin_reference_rate_day"


def test_empty_window_is_skipped_with_a_log_line(caplog):
    d = pairs.prepare(make_trials("exp3"), "exp3")
    with caplog.at_level(logging.INFO):
        parts = pairs.split_windows(d)
    assert [w for w, _ in parts] == ["day"]
    assert "evening" in caplog.text and "skip" in caplog.text.lower()


# ── the within-fly reference contrast ────────────────────────────────────────
def test_reference_contrast_recovers_a_planted_effect():
    # IAA right after Hexanol always responds, right after the click never; other primes 0
    df = make_trials("exp3", score_fn=lambda f, c, x, p: 4 if (c == "pair"
                                                                and x == "Hexanol") else 0)
    long = pairs.scored_long(pairs.prepare(df, "exp3"), "exp3")
    res = pairs.reference_contrast(long, "rate", **FAST)
    t = res.table.set_index("group")
    assert list(t.index) == ODORS                     # the click is the reference, no row
    for c in TABLE_COLS:
        assert c in t.columns
    hx = t.loc["Hexanol"]
    assert hx["effect"] == pytest.approx(100.0) and hx["n_flies"] == 12
    assert hx["p"] == pytest.approx(2 / 2 ** 12)       # every fly +1: exact sign-flip
    assert hx["p_holm"] == pytest.approx(min(1.0, 7 * 2 / 2 ** 12))
    assert hx["p_holm"] < 0.05 and hx["n_batches"] == 3
    assert hx["pooled_b"] == pytest.approx(100.0) and hx["pooled_a"] == pytest.approx(0.0)
    assert t.loc["IAA", "effect"] == 0 and t.loc["IAA", "p"] == 1
    assert "Holm" in hx["correction"] and "7" in hx["correction"]


def test_reference_contrast_is_within_fly():
    # flies differ in baseline (0..3) but IAA after Citral is always +2 over the click
    def score(f, c, x, p):
        return (f % 4) + (2 if (c == "pair" and x == "Citral") else 0)
    long = pairs.scored_long(pairs.prepare(make_trials("exp3", score_fn=score), "exp3"),
                             "exp3")
    t = pairs.reference_contrast(long, "score", **FAST).table.set_index("group")
    assert t.loc["Citral", "effect"] == pytest.approx(2.0)
    assert t.loc["Citral", "ci_lo"] == pytest.approx(2.0)   # no between-fly noise left
    assert t.loc["Citral", "ci_hi"] == pytest.approx(2.0)
    assert t.loc["Hexanol", "effect"] == pytest.approx(0.0)


def test_exp2_direct_form_is_x_in_pair_minus_click_in_pair():
    def score(f, c, x, p):
        if c == "alone":
            return 1 if x == "Hexanol" else 0
        return {"Hexanol": 3 + f % 2, CLICK: 1}.get(x, 0)
    long = pairs.scored_long(pairs.prepare(make_trials("exp2", score_fn=score), "exp2"),
                             "exp2")
    t = pairs.reference_contrast(long, "score", **FAST).table.set_index("group")
    assert t.loc["Hexanol", "effect"] == pytest.approx(2.5)   # mean(3 + f%2) - 1


# ── the controlled change ────────────────────────────────────────────────────
def test_controlled_change_removes_odor_identity():
    # fly f: Hexanol alone 1, in pair 3 + f%2; click alone 0, in pair 1
    def score(f, c, x, p):
        if c == "alone":
            return 1 if x == "Hexanol" else 0
        return {"Hexanol": 3 + f % 2, CLICK: 1}.get(x, 0)
    long = pairs.scored_long(pairs.prepare(make_trials("exp2", score_fn=score), "exp2"),
                             "exp2")
    res = pairs.controlled_change(long, "score", **FAST)
    t = res.table.set_index("group")
    assert list(t.index) == ODORS
    # per fly: (3 + f%2 - 1) - (1 - 0) = 1 + f%2 -> mean 1.5 over 12 flies
    assert t.loc["Hexanol", "effect"] == pytest.approx(1.5)
    assert t.loc["Hexanol", "n_flies"] == 12
    assert t.loc["Hexanol", "pooled_b"] == pytest.approx(2.5)  # pooled Hexanol change
    assert t.loc["Hexanol", "pooled_a"] == pytest.approx(1.0)  # pooled click change
    assert t.loc["Hexanol", "effect_pooled"] == pytest.approx(1.5)
    # IAA: change 0 - 0, minus the click's +1 -> -1 in every fly
    assert t.loc["IAA", "effect"] == pytest.approx(-1.0)
    # the change bars and per-fly lines are in plotted units
    bars = res.bars.set_index(["group", "condition"])
    assert bars.loc[("Hexanol", "odor"), "value"] == pytest.approx(2.5)
    pf = res.per_fly
    assert set(pf.loc[(pf.group == "Hexanol") & (pf.condition == "odor"), "value"]) == {2, 3}


def test_controlled_change_rate_is_in_percentage_points():
    def score(f, c, x, p):
        if c == "alone":
            return 0
        return {"Benzaldehyde": 3, CLICK: 3 if p == 1 else 0}.get(x, 0)
    long = pairs.scored_long(pairs.prepare(make_trials("exp2", score_fn=score), "exp2"),
                             "exp2")
    t = pairs.controlled_change(long, "rate", **FAST).table.set_index("group")
    assert t.loc["Benzaldehyde", "effect"] == pytest.approx(50.0)   # 100 - 50 pp
    assert t.loc["Benzaldehyde", "unit"] == "percentage points"


# ── carry-over ───────────────────────────────────────────────────────────────
def test_carryover_exclusion_removes_a_leak_from_odor1():
    # Linalool (odor 1) responds on its 1st presentation and IAA "responds" with it;
    # the 2nd click's odor 1 also responds (the click itself).
    def odor1(f, x, p):
        return 3 if (x == "Linalool" and p == 1) or (x == CLICK and p == 2) else 0

    def score(f, c, x, p):
        return 3 if (c == "pair" and x == "Linalool" and p == 1) else 0
    d = pairs.prepare(make_trials("exp3", score_fn=score, odor1_fn=odor1), "exp3")
    long = pairs.scored_long(d, "exp3")
    main = pairs.reference_contrast(long, "rate", **FAST).table.set_index("group")
    assert main.loc["Linalool", "effect"] == pytest.approx(50.0)
    kept, n_dropped = pairs.drop_carryover(d)
    assert n_dropped == 24                            # 12 Linalool + 12 click trials
    assert not ((kept.condition == "pair") & (kept.odor1_responded == 1)).any()
    assert len(kept[kept.condition == "alone"]) == len(d[d.condition == "alone"])
    sens = pairs.reference_contrast(pairs.scored_long(kept, "exp3"), "rate",
                                    **FAST).table.set_index("group")
    assert sens.loc["Linalool", "effect"] == pytest.approx(0.0)
    assert sens.loc["Linalool", "n_flies"] == 12


def test_carryover_exclusion_keeps_a_prime_row_without_flies():
    def odor1(f, x, p):
        return 3 if x == "Hexanol" else 0
    d = pairs.prepare(make_trials("exp3", odor1_fn=odor1), "exp3")
    kept, _ = pairs.drop_carryover(d)
    t = pairs.reference_contrast(pairs.scored_long(kept, "exp3"), "rate",
                                 **FAST).table.set_index("group")
    assert list(t.index) == ODORS
    assert t.loc["Hexanol", "n_flies"] == 0 and np.isnan(t.loc["Hexanol", "p"])
    assert "6" in t.loc["IAA", "correction"]          # Holm over the primes with flies


# ── shared-baseline diagnostic ───────────────────────────────────────────────
def test_shared_baseline_counts_directions_and_batch_p():
    def score(f, c, x, p):
        return (3 if c == "pair" else 0) if x in ODORS[:5] else 0
    long = pairs.scored_long(pairs.prepare(make_trials("exp3", score_fn=score), "exp3"),
                             "exp3")
    res = pairs.shared_baseline(long, "rate", **FAST)
    dirn = res.extra["direction"]
    assert dirn["n_up"] == 5 and dirn["n_down"] == 0 and dirn["n_zero"] == 2
    t = res.table.set_index("group")
    assert "direction count" in t.index
    row = t.loc["direction count"]
    assert row["effect"] == 5 and row["n_flies"] == 12 and row["n_batches"] == 3
    # Every fly has the same row, so a joint flip keeps all 5 odors together: the count
    # is 5 unless the 12 signs cancel (C(12, 6) of 2^12 patterns). The shared baseline
    # explains the 5-0 count completely.
    assert row["p"] == pytest.approx(1 - 924 / 4096)
    assert dirn["p_naive_binomial"] < 0.07                # the naive count overstates it
    # batch p: whole batches' rows flipped jointly (3 batches: the sum never cancels)
    assert row["p_batch"] == pytest.approx(1.0)
    per_odor = t.drop(index="direction count")
    assert list(per_odor.index) == ODORS and (per_odor["n_flies"] == 12).all()


# ── end to end ───────────────────────────────────────────────────────────────
def _two_windows(experiment, **kw):
    rng = np.random.default_rng(1)

    def score(f, c, x, p):
        return int(rng.integers(-1, 6))

    def odor1(f, x, p):
        return int(rng.integers(-1, 6))
    return pd.concat([make_trials(experiment, score_fn=score, odor1_fn=odor1, **kw),
                      make_trials(experiment, n_batches=2, window="evening", prefix="eve_",
                                  start_h=18.0, score_fn=score, odor1_fn=odor1, **kw)],
                     ignore_index=True)


def test_exp3_run_writes_window_labelled_outputs(tmp_path):
    out = tmp_path / "results"
    findings = exp3.run(_two_windows("exp3", short_click=True), out, tables=out / "tables",
                        fit_models=False, **FAST)
    folder = out / "04_exp3_iaa_second"
    for w in ("day", "evening"):
        for metric in ("rate", "score"):
            base = folder / "c_vs_paraffin_reference" / f"exp3_vs_paraffin_reference_{metric}_{w}"
            for ext in ("png", "svg", "csv"):
                assert base.with_suffix(f".{ext}").exists()
            tbl = pd.read_csv(out / "tables" / f"exp3_vs_paraffin_reference_{metric}_{w}.csv")
            for c in TABLE_COLS + ["p_batch_disagrees"]:
                assert c in tbl.columns
            assert (tbl["window"] == w).all()
            assert (out / "tables" /
                    f"exp3_vs_paraffin_reference_carryover_excluded_{metric}_{w}.csv").exists()
            for sub, analysis in (("a_alone_vs_pair_pooled", "alone_vs_pair_pooled"),
                                  ("b_presentation_1_vs_2", "presentation_1_vs_2"),
                                  ("d_shared_baseline_diagnostic",
                                   "shared_baseline_diagnostic")):
                assert (folder / sub / f"exp3_{analysis}_{metric}_{w}.png").exists()
    for sub in ("", "a_alone_vs_pair_pooled", "b_presentation_1_vs_2",
                "c_vs_paraffin_reference", "d_shared_baseline_diagnostic"):
        readme = folder / sub / "README.md"
        assert readme.exists() and 2 <= len(readme.read_text().strip().splitlines()) <= 4
    assert findings and all({"section", "analysis", "primary", "text", "numbers"} <= set(f)
                            for f in findings)
    assert {f["section"] for f in findings} == {"exp3"}
    prim = [f for f in findings if f["primary"]]
    assert prim and all(f["analysis"] == "vs_paraffin_reference" for f in prim)
    assert all("Holm" in f["text"] and "CI" in f["text"] for f in prim)


def test_exp2_run_skips_an_empty_window(tmp_path, caplog):
    out = tmp_path / "results"
    df = make_trials("exp2", score_fn=lambda f, c, x, p: (f + len(x)) % 6 - 1)
    with caplog.at_level(logging.INFO, logger="per_analysis.exp2"):
        findings = exp2.run(df, out, tables=out / "tables", fit_models=False, **FAST)
    folder = out / "03_exp2_iaa_first"
    files = [p.name for p in folder.rglob("*.png")]
    assert files and not any("evening" in f for f in files)
    assert (folder / "c_vs_paraffin_reference" /
            "exp2_vs_paraffin_reference_direct_rate_day.png").exists()
    assert (folder / "c_vs_paraffin_reference" /
            "exp2_vs_paraffin_reference_controlled_score_day.png").exists()
    assert "evening" in caplog.text and "skip" in caplog.text.lower()
    assert {f["section"] for f in findings} == {"exp2"}
    assert not any(f["primary"] for f in findings)


def test_models_record_status_without_crashing(tmp_path):
    out = tmp_path / "results"
    df = _two_windows("exp3")
    exp3.run(df[df.window == "evening"], out, tables=out / "tables", fit_models=True,
             ordinal_boot=4, **FAST)
    models = out / "04_exp3_iaa_second" / "models"
    status = pd.read_csv(models / "exp3_model_status_evening.csv")
    assert set(status["fit"]) == {"gee", "glmm", "mixedlm", "ordinal"}
    assert set(status["model"]) == {"alone_vs_pair", "vs_paraffin_reference"}
    assert status["status"].isin(["ok", "not_converged", "failed"]).all()
    for r in status.itertuples():
        assert (models / f"exp3_model_{r.model}_{r.fit}_evening.txt").exists()
        assert (models / f"exp3_model_{r.model}_{r.fit}_evening_coefs.csv").exists()
    assert (models / "README.md").exists()
    # odor 1 enters the exp3 models as a covariate
    coefs = pd.read_csv(models / "exp3_model_vs_paraffin_reference_gee_evening_coefs.csv")
    assert coefs["term"].str.contains("odor1_responded").any() or \
        (coefs["status"] == "failed").all()


def test_prune_terms_drops_constant_covariates():
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [5.0, 5.0, 5.0], "g": ["x", "x", "x"],
                          "h": ["x", "y", "x"]})
    T = pairs.Term
    kept, dropped = pairs.prune_terms(frame, [T("a", ("a",)), T("b", ("b",)),
                                              T("C(g)", ("g",), True),
                                              T("C(h) * a", ("h", "a"), True)])
    assert kept == ["a", "C(h) * a"]
    assert len(dropped) == 2 and "b" in dropped[0] and "g" in dropped[1]


def test_fit_with_nan_estimates_is_recorded_as_failed(tmp_path, monkeypatch):
    # statsmodels can return all-NaN GEE estimates without raising (separation in a small
    # window); stats then says 'ok'. The status table and coefficient CSV must not.
    frame = pairs.model_frame(pairs.prepare(make_trials("exp3"), "exp3")
                              .query("condition == 'pair'"))

    def nan_fit(df, formula, *a, **k):
        coefs = pd.DataFrame({"term": ["Intercept", "x"], "estimate": [np.nan, np.nan],
                              "se": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "p": np.nan,
                              "model": "GEE", "n_obs": len(df), "n_groups": 12,
                              "status": "ok", "note": ""})
        return object(), coefs, "summary\n"
    monkeypatch.setattr(pairs.st, "fit_gee_logistic", nan_fit)
    rows = pairs.fit_model_set(frame, [pairs.Term("position_in_block",
                                                  ("position_in_block",))],
                               name="m", experiment="exp3", window="day", folder=tmp_path,
                               ordinal_boot=2)
    gee = next(r for r in rows if r["fit"] == "gee")
    assert gee["status"] == "failed" and "non-finite" in gee["note"]
    saved = pd.read_csv(tmp_path / "exp3_model_m_gee_day_coefs.csv")
    assert (saved["status"] == "failed").all()


def _fit_row(fit, estimates):
    terms = list(estimates)
    coefs = pd.DataFrame({"term": terms, "estimate": [estimates[t] for t in terms],
                          "status": "ok"})
    return {"model": "m", "fit": fit, "status": "ok", "_coefs": coefs}


def test_gee_is_the_model_of_record_and_sign_flips_are_flagged():
    rows = [_fit_row("gee", {"Intercept": -1.0, "C(x)[T.a]": -0.41, "C(x)[T.b]": 0.3,
                             "C(x)[T.c]": np.nan}),
            _fit_row("glmm", {"Intercept": -1.2, "C(x)[T.a]": 0.36, "C(x)[T.b]": 0.2,
                              "C(x)[T.c]": 0.5, "VC fly (log SD)": -0.5})]
    flips = pairs.sign_disagreements(rows, name="m", pattern="C(x)")
    assert [f["term"] for f in flips] == ["C(x)[T.a]"]
    assert flips[0]["gee"] == pytest.approx(-0.41) and flips[0]["glmm"] == pytest.approx(0.36)
    assert pairs.FIT_ROLE["gee"] == "model of record"
    assert all(pairs.FIT_ROLE[f] == "sensitivity check" for f in ("glmm", "mixedlm", "ordinal"))
    assert "a" in pairs.describe_sign_flips(flips)
    assert pairs.describe_sign_flips([]) == ""


def test_model_status_table_marks_roles(tmp_path):
    out = tmp_path / "results"
    exp3.run(_two_windows("exp3")[lambda d: d.window == "evening"], out,
             tables=out / "tables", fit_models=True, ordinal_boot=2, **FAST)
    status = pd.read_csv(out / "04_exp3_iaa_second" / "models" / "exp3_model_status_evening.csv")
    roles = dict(zip(status["fit"], status["role"]))
    assert roles["gee"] == "model of record" and roles["glmm"] == "sensitivity check"
    readme = (out / "04_exp3_iaa_second" / "models" / "README.md").read_text()
    assert "model of record" in readme


def _exp2_frame(alone_zero):
    rng = np.random.default_rng(3)

    def score(f, c, x, p):
        if c == "alone" and x in alone_zero:
            return 0
        return int(rng.integers(-1, 6))
    return pairs.model_frame(pairs.prepare(make_trials("exp2", n_batches=4,
                                                       score_fn=score), "exp2"))


EXP2_TERMS = [pairs.Term("C(condition, Treatment('alone')) * "
                         "C(partner, Treatment('Paraffin Oil'))", ("condition", "partner"),
                         True),
              pairs.Term("position_in_block", ("position_in_block",))]
EXP2_SEP = pairs.Separation(cells=("condition", "partner"), level="partner",
                            reference="Paraffin Oil")


def test_separated_levels_finds_all_zero_cells():
    frame = _exp2_frame({"IAA"})
    assert pairs.separated_levels(frame, EXP2_SEP) == ["IAA"]


def test_gee_leaves_out_a_separated_partner(tmp_path):
    # IAA alone never responds: its interaction is infinite and, left in, drags every
    # GEE estimate to NaN. The model of record leaves IAA out and says so.
    rows = pairs.fit_model_set(_exp2_frame({"IAA"}), EXP2_TERMS, name="m",
                               experiment="exp2", window="day", folder=tmp_path,
                               ordinal_boot=2, separation=EXP2_SEP)
    gee = next(r for r in rows if r["fit"] == "gee")
    assert gee["status"] != "failed" and gee["gee_left_out"] == "IAA"
    assert "IAA" in gee["note"] and "separation" in gee["note"]
    est = gee["_coefs"]
    assert np.isfinite(est["estimate"]).all()
    assert not est["term"].str.contains(r"\[T\.IAA\]", regex=True).any()
    assert est["term"].str.contains(r"\[T\.pair\]:C\(partner", regex=True).sum() == 6


def test_gee_with_a_separated_reference_fails_with_a_reason(tmp_path):
    rows = pairs.fit_model_set(_exp2_frame({"Paraffin Oil"}), EXP2_TERMS, name="m",
                               experiment="exp2", window="evening", folder=tmp_path,
                               ordinal_boot=2, separation=EXP2_SEP)
    gee = next(r for r in rows if r["fit"] == "gee")
    assert gee["status"] == "failed" and "reference" in gee["note"]
    assert gee["gee_left_out"] == "" and pairs.gee_left_out(rows, "m") == ""
    assert (tmp_path / "exp2_model_m_gee_evening.txt").read_text().count("reference") >= 1
