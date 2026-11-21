"""src/per_analysis/stats.py: fly-unit statistics for the per_analysis package."""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from per_analysis import stats as st  # noqa: E402


# ── paired sign-flip ─────────────────────────────────────────────────────────
def test_sign_flip_exact_on_known_data():
    r = st.sign_flip_test([1, 1, 1, 1])
    assert r["p"] == pytest.approx(2 / 16) and r["exact"] and r["n"] == 4
    # a zero carries no sign: 3 informative flies -> 2 of 8 patterns
    r0 = st.sign_flip_test([1, 1, 1, 0])
    assert r0["p"] == pytest.approx(2 / 8) and r0["exact"]
    assert r0["n"] == 4 and r0["n_nonzero"] == 3
    assert st.sign_flip_test([0.5, 1.5, -0.5])["p"] == pytest.approx(6 / 8)


def test_sign_flip_degenerate_inputs():
    assert st.sign_flip_test([0, 0, 0])["p"] == pytest.approx(1.0)
    empty = st.sign_flip_test([])
    assert np.isnan(empty["p"]) and empty["n"] == 0
    assert st.sign_flip_test([np.nan, 1.0])["n"] == 1


def _brute_force_sign_flip(d):
    d = np.asarray(d, dtype=float)
    obs = abs(d.sum())
    hits = 0
    for signs in itertools.product((-1.0, 1.0), repeat=d.size):
        hits += abs(np.dot(signs, np.abs(d))) >= obs - 1e-12
    return hits / 2 ** d.size


def test_sign_flip_off_grid_is_enumerated_exactly():
    d = np.random.default_rng(1).normal(0.4, 1.0, 9)       # means of 3 trials: off grid
    r = st.sign_flip_test(d)
    assert r["exact"] and r["method"] == "exact-enumeration"
    assert r["p"] == pytest.approx(_brute_force_sign_flip(d))


def test_sign_flip_means_of_three_trials_are_exact_at_any_n():
    small = np.array([1, 2, -1, 4, 3, 2, 1, 5, -2, 1]) / 3      # per-fly means of 3 trials
    r = st.sign_flip_test(small)
    assert r["method"] == "exact-grid" and r["p"] == pytest.approx(_brute_force_sign_flip(small))
    mixed = np.array([1 / 3, 0.5, -0.25, 2 / 3, 1.0, 0.75, -1 / 6, 1.5])   # 2-4 trials
    assert st.sign_flip_test(mixed)["p"] == pytest.approx(_brute_force_sign_flip(mixed))
    big = np.random.default_rng(0).integers(-6, 10, 45) / 3
    rb = st.sign_flip_test(big)
    assert rb["exact"] and rb["method"] == "exact-grid"
    a = np.abs(big[big != 0])                                     # same values, by Monte Carlo
    assert st._mc_p(a, abs(big.sum()), 100_000, 0) == pytest.approx(rb["p"], abs=0.01)


def test_sign_flip_large_off_grid_uses_seeded_monte_carlo():
    d = np.random.default_rng(2).normal(0.3, 1.0, 30)
    r1 = st.sign_flip_test(d, n_perm=5000, seed=7)
    r2 = st.sign_flip_test(d, n_perm=5000, seed=7)
    assert not r1["exact"] and r1["method"] == "monte-carlo"
    assert r1["p"] == r2["p"]                                 # seeded -> reproducible
    hits = r1["p"] * 5001 - 1                                 # p = (hits + 1) / (n + 1)
    assert hits == pytest.approx(round(hits))


def test_batch_sign_flip_wraps_b2b():
    vals = [1, 1, 2, 0.5, 0.5, 3]
    batches = ["a", "a", "b", "b", "b", "c"]
    r = st.batch_sign_flip_test(vals, batches)
    assert r["p"] == pytest.approx(2 / 8) and r["n_batches"] == 3 and r["exact"]


# ── bootstrap ────────────────────────────────────────────────────────────────
def test_bca_covers_true_mean_and_matches_scipy():
    from scipy.stats import bootstrap
    x = np.random.default_rng(0).normal(2.0, 1.0, 40)
    lo, hi = st.bootstrap_ci(x, n_boot=10_000, seed=0)
    assert lo < 2.0 < hi and lo < x.mean() < hi
    ref = bootstrap((x,), np.mean, n_resamples=10_000, method="BCa",
                    random_state=1).confidence_interval
    assert lo == pytest.approx(ref.low, abs=0.06) and hi == pytest.approx(ref.high, abs=0.06)


def test_bca_degenerate_constant_sample():
    assert st.bootstrap_ci([3.0, 3.0, 3.0, 3.0], n_boot=500, seed=0) == (3.0, 3.0)
    lo, hi = st.bootstrap_ci([1.0], n_boot=100, seed=0)
    assert np.isnan(lo) and np.isnan(hi)


def test_bootstrap_percentile_and_custom_stat():
    x = np.random.default_rng(3).exponential(1.0, 60)
    lo, hi = st.bootstrap_ci(x, stat=np.median, n_boot=2000, seed=1, method="percentile")
    assert lo < np.median(x) < hi


def test_paired_effect_inner_joins_and_reports_batch_p():
    a = pd.Series([0.0, 0.2, 0.4, 0.1, 0.0, 0.3], index=list("abcdef"))
    b = pd.Series([1.0, 0.8, 0.9, 0.7, 0.6, 1.0, 9.9], index=list("abcdefg"))  # g unpaired
    batch = {"a": "b1", "b": "b1", "c": "b2", "d": "b2", "e": "b3", "f": "b3"}
    r = st.paired_effect(a, b, batch_of=batch, n_boot=2000, seed=0)
    d = (b - a).dropna()
    assert r["n_flies"] == 6 and r["n_batches"] == 3
    assert r["mean_diff"] == pytest.approx(d.mean())
    assert r["mean_a"] == pytest.approx(a.mean()) and r["mean_b"] == pytest.approx(b.iloc[:6].mean())
    assert r["p"] == pytest.approx(st.sign_flip_test(d.to_numpy())["p"])
    assert r["p_batch"] == pytest.approx(2 / 8)               # 3 batches, all up
    assert r["ci_lo"] <= r["mean_diff"] <= r["ci_hi"]
    no_batch = st.paired_effect(a, b, n_boot=500, seed=0)
    assert np.isnan(no_batch["p_batch"])


def test_paired_effect_rejects_duplicate_fly_index():
    a = pd.Series([0.0, 1.0], index=["x", "x"])
    with pytest.raises(ValueError):
        st.paired_effect(a, a)


# ── per-fly aggregation ──────────────────────────────────────────────────────
@pytest.fixture
def trials():
    return pd.DataFrame({
        "fly_id": ["A"] * 4 + ["B"],
        "batch": ["b1"] * 4 + ["b2"],
        "odor": ["IAA"] * 5,
        "condition": ["pre", "pre", "post", "post", "pre"],
        "score": [3, 1, 4, 5, 0],
        "responded": [True, False, True, True, False],
    })


def test_per_fly_is_the_within_fly_mean(trials):
    pf = st.per_fly(trials, "responded", by=["condition"])
    row = pf[(pf.fly_id == "A") & (pf.condition == "pre")].iloc[0]
    assert row["responded"] == pytest.approx(0.5) and row["n_trials"] == 2
    assert row["batch"] == "b1"
    assert len(pf) == 3


def test_pooled_rate_pools_trials_not_fly_means(trials):
    r = st.pooled_rate_ci(trials)
    assert r["rate"] == pytest.approx(60.0)                   # 3 / 5 trials, not mean of 75 % and 0 %
    assert r["n_flies"] == 2 and r["n_trials"] == 5
    assert 0 <= r["ci_lo"] <= r["rate"] <= r["ci_hi"] <= 100


def test_pooled_mean_sem_one_trial_per_fly_is_the_plain_sem():
    x = np.array([1.0, 2.0, 4.0, 5.0, 3.0])
    df = pd.DataFrame({"fly_id": list("abcde"), "score": x})
    r = st.pooled_mean_sem(df)
    assert r["mean"] == pytest.approx(x.mean())
    assert r["sem"] == pytest.approx(x.std(ddof=1) / np.sqrt(x.size))


def test_group_summary_units(trials):
    rate = st.group_summary(trials, by=["condition"], metric="rate")
    post = rate.set_index("condition").loc["post"]
    assert post["value"] == pytest.approx(100.0) and post["n_flies"] == 1
    score = st.group_summary(trials, by=["condition"], metric="score")
    pre = score.set_index("condition").loc["pre"]
    assert pre["value"] == pytest.approx(4 / 3)              # (3 + 1 + 0) / 3 trials
    assert pre["err_lo"] < pre["value"] < pre["err_hi"]


def test_paired_comparisons_table_carries_batch_p():
    rng = np.random.default_rng(0)
    rows = []
    for f in range(12):
        for odor in ("IAA", "EB"):
            for cond, base in (("alone", 0.2), ("pair", 0.7)):
                for t in range(2):
                    rows.append({"fly_id": f"f{f}", "batch": f"b{f // 3}", "odor": odor,
                                 "condition": cond, "responded": rng.random() < base})
    tbl = st.paired_comparisons(pd.DataFrame(rows), "responded", condition_col="condition",
                                a="alone", b="pair", by=["odor"], n_boot=500)
    assert set(tbl["odor"]) == {"IAA", "EB"}
    for col in ("p", "p_batch", "p_holm", "q_bh", "ci_lo", "ci_hi", "n_flies"):
        assert col in tbl.columns


# ── multiplicity ─────────────────────────────────────────────────────────────
def test_holm_and_bh():
    assert st.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert st.bh([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.04, 0.04])
    h = st.holm([0.01, np.nan])
    assert h[0] == pytest.approx(0.01) and np.isnan(h[1])


def test_stars_only_three_levels_or_blank():
    assert [st.stars(p) for p in (0.0005, 0.005, 0.03, 0.2, np.nan, None)] == \
        ["***", "**", "*", "", "", ""]


def test_add_corrections_within_family():
    t = pd.DataFrame({"fam": ["x", "x", "x", "y"], "p": [0.01, 0.04, 0.03, 0.04]})
    out = st.add_corrections(t, p_col="p", family_col="fam")
    assert out["p_holm"].tolist() == pytest.approx([0.03, 0.06, 0.06, 0.04])
    assert out["q_bh"].tolist() == pytest.approx([0.03, 0.04, 0.04, 0.04])
    assert "p_holm" not in t.columns                          # input untouched
    whole = st.add_corrections(t, p_col="p")
    assert whole["p_holm"].iloc[0] == pytest.approx(0.04)


# ── difference in differences ────────────────────────────────────────────────
def test_did_planted_effect():
    rng = np.random.default_rng(4)
    x = pd.Series(rng.normal(1.0, 1.0, 60), index=[f"x{i}" for i in range(60)])
    c = pd.Series(rng.normal(0.0, 1.0, 60), index=[f"c{i}" for i in range(60)])
    r = st.did_bootstrap(x, c, n_boot=4000, n_perm=4000, seed=0)
    assert r["estimate"] == pytest.approx(x.mean() - c.mean())
    assert r["ci_lo"] < r["estimate"] < r["ci_hi"]
    assert r["p"] < 0.05 and r["ci_lo"] > 0


def test_did_no_effect():
    rng = np.random.default_rng(5)
    x = rng.normal(0.0, 1.0, 40)
    c = rng.normal(0.0, 1.0, 40)
    r = st.did_bootstrap(x, c, n_boot=4000, n_perm=4000, seed=0)
    assert r["p"] > 0.05 and r["ci_lo"] < 0 < r["ci_hi"]


def test_did_batch_permutation_p():
    rng = np.random.default_rng(6)
    x = pd.Series(rng.normal(1.0, 1.0, 24), index=[f"x{i}" for i in range(24)])
    c = pd.Series(rng.normal(0.0, 1.0, 24), index=[f"c{i}" for i in range(24)])
    batch = {f: f"bx{i // 4}" for i, f in enumerate(x.index)}
    batch.update({f: f"bc{i // 4}" for i, f in enumerate(c.index)})
    r = st.did_bootstrap(x, c, batch_of=batch, n_boot=2000, n_perm=2000, seed=0)
    assert r["n_batches_exposed"] == 6 and r["n_batches_control"] == 6
    assert 0 < r["p_batch"] <= 1 and r["perm_exact_batch"]   # C(12, 6) = 924 splits, exact
    from fbpipe.analysis.fly_stats import fly_permutation_p
    assert r["p"] == pytest.approx(fly_permutation_p(x, c, n_perm=2000, seed=0))


# ── models ───────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def model_frame():
    rng = np.random.default_rng(0)
    n_fly, per = 40, 10
    fly = np.repeat([f"f{i}" for i in range(n_fly)], per)
    cond = np.tile(["pre", "post"], n_fly * per // 2)
    u = np.repeat(rng.normal(0, 0.7, n_fly), per)
    eta = -0.5 + 1.5 * (cond == "post") + u
    responded = rng.random(fly.size) < 1 / (1 + np.exp(-eta))
    score = np.clip(np.round(1.5 + 1.2 * eta + rng.logistic(size=fly.size)), -1, 5).astype(int)
    return pd.DataFrame({"fly_id": fly, "condition": cond, "responded": responded,
                         "score": score, "cont": 2.0 + 1.0 * (cond == "post") + u
                         + rng.normal(0, 0.5, fly.size)})


POST = "C(condition, Treatment('pre'))[T.post]"
COEF_COLS = ["term", "estimate", "se", "ci_lo", "ci_hi", "p", "model", "n_obs", "n_groups"]


def _post_row(coefs):
    return coefs.set_index("term").loc[POST]


def test_gee_recovers_sign_with_bool_response(model_frame):
    res, coefs, text = st.fit_gee_logistic(
        model_frame, "responded ~ C(condition, Treatment('pre'))")
    assert res is not None and (coefs["status"] == "ok").all()
    assert _post_row(coefs)["estimate"] > 0                  # bool LHS would flip this sign
    assert list(coefs.columns[:len(COEF_COLS)]) == COEF_COLS
    assert coefs["n_groups"].iloc[0] == 40 and coefs["n_obs"].iloc[0] == 400
    assert "GEE" in text


def test_mixed_logistic_is_labelled_variational(model_frame):
    res, coefs, text = st.fit_mixed_logistic(
        model_frame, "responded ~ C(condition, Treatment('pre'))")
    assert _post_row(coefs)["estimate"] > 0
    assert "variational" in coefs["model"].iloc[0].lower()
    assert "variational" in text.lower()


def test_mixedlm_recovers_sign(model_frame):
    res, coefs, _ = st.fit_mixedlm(model_frame, "cont ~ C(condition, Treatment('pre'))")
    row = _post_row(coefs)
    assert row["estimate"] == pytest.approx(1.0, abs=0.25) and row["p"] < 0.05


def test_ordinal_recovers_sign_with_cluster_bootstrap(model_frame):
    res, coefs, text = st.fit_ordinal(model_frame, "score ~ C(condition, Treatment('pre'))",
                                      n_boot=25, seed=0)
    row = _post_row(coefs)
    assert row["estimate"] > 0 and np.isfinite(row["se"]) and row["se"] > 0
    assert "bootstrap" in coefs["model"].iloc[0].lower()
    assert "bootstrap" in text.lower()


@pytest.mark.parametrize("fit", ["fit_gee_logistic", "fit_mixed_logistic",
                                 "fit_mixedlm", "fit_ordinal"])
def test_failing_fit_returns_a_failed_row(model_frame, fit):
    res, coefs, text = getattr(st, fit)(model_frame, "no_such_column ~ condition")
    assert res is None
    assert len(coefs) == 1 and coefs["status"].iloc[0] == "failed"
    assert "fail" in text.lower()
    assert list(coefs.columns[:len(COEF_COLS)]) == COEF_COLS


def test_save_model_writes_both_files(model_frame, tmp_path):
    out = st.fit_gee_logistic(model_frame, "responded ~ C(condition, Treatment('pre'))")
    txt, csv = st.save_model(out, tmp_path / "deep" / "gee_model")
    assert txt.exists() and csv.exists()
    assert txt.name == "gee_model.txt" and csv.name == "gee_model_coefs.csv"
    back = pd.read_csv(csv)
    assert set(COEF_COLS) <= set(back.columns)


# ── power ────────────────────────────────────────────────────────────────────
def test_power_curve_is_monotone_in_n():
    d = np.random.default_rng(0).normal(0.5, 1.0, 30)
    tbl = st.power_curve(d, deltas=[0.0, 0.6], ns=[4, 8, 16, 32], n_sim=400, seed=0)
    assert set(tbl.columns) >= {"delta", "n", "power"}
    eff = tbl[tbl.delta == 0.6].sort_values("n")["power"].to_numpy()
    assert np.all(np.diff(eff) >= -0.03) and eff[-1] > eff[0] + 0.4
    size = tbl[tbl.delta == 0.0]["power"].to_numpy()
    assert np.all(size <= 0.1)                                # delta 0 = the test's size


def test_power_curve_two_group_and_min_n():
    rng = np.random.default_rng(1)
    a, b = rng.normal(0, 1, 30), rng.normal(0, 1, 30)
    tbl = st.power_curve((a, b), deltas=[1.0], ns=[5, 10, 20, 40], test="two_group",
                         n_sim=300, seed=0)
    pw = tbl.sort_values("n")["power"].to_numpy()
    assert np.all(np.diff(pw) >= -0.03) and pw[-1] > 0.8
    m = st.min_n_for_power(tbl, target=0.8)
    assert m.loc[m.delta == 1.0, "min_n"].iloc[0] in (10, 20, 40)
    never = st.min_n_for_power(tbl, target=1.01)
    assert np.isnan(never["min_n"].iloc[0])


# ── shared-baseline diagnostic ───────────────────────────────────────────────
def test_direction_count_test_shared_baseline():
    rng = np.random.default_rng(0)
    n_fly, odors = 14, ["IAA", "EB", "Hex", "Lin", "Ben"]
    alone = rng.normal(0, 1, n_fly)
    pair = rng.normal(0, 0.3, (n_fly, len(odors))) + 0.4      # every odor shares the baseline
    changes = pd.DataFrame(pair - alone[:, None], columns=odors,
                           index=[f"f{i}" for i in range(n_fly)])
    r = st.direction_count_test(changes)
    assert 0.0 <= r["p"] <= 1.0 and r["exact"]
    assert r["n_up"] + r["n_down"] + r["n_zero"] == len(odors)
    assert 0.0 <= r["p_naive_binomial"] <= 1.0
    assert "null" in r and "jointly" in r["null"]
    assert sum(r["null_n_up"].values()) == pytest.approx(1.0)


def test_direction_count_test_handles_missing_and_mc():
    rng = np.random.default_rng(1)
    changes = pd.DataFrame(rng.normal(0, 1, (25, 4)), columns=list("abcd"))
    changes.iloc[0, 0] = np.nan
    r = st.direction_count_test(changes, n_perm=3000, seed=0)
    assert not r["exact"] and 0 < r["p"] <= 1.0
