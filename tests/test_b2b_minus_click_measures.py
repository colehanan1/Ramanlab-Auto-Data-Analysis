"""scripts/analysis/b2b_minus_click_measures.py: each partner's change minus the paraffin
click's change on log AUC and the peak-z ratio (user 2026-11-18)."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from fbpipe.analysis import fly_stats as fs  # noqa: E402
from scripts.analysis import b2b_minus_click_measures as m  # noqa: E402
from scripts.analysis import b2b_pair_figures as b  # noqa: E402
from scripts.analysis import b2b_stats as bs  # noqa: E402

FIRST, SECOND = b.FIRST, b.SECOND
N = 100


# ── tidy per-trial rows: the contrast arithmetic ─────────────────────────────
def _t(ds, fly, fn, role, scored, partner, value, rnd=0):
    return {"dataset": ds, "fly": fly, "fly_number": fn, "trial_label": f"{role}_{partner}_{rnd}",
            "role": role, "scored_odor": scored, "partner": partner, "round": rnd,
            "log_auc": value}


def _pair(ds, fly, fn, scored, partner, values):
    return [_t(ds, fly, fn, m.IN_PAIR, scored, partner, v, i + 1) for i, v in enumerate(values)]


def _first_trials():
    """IAA-first: Hexanol scored after IAA, the click after IAA; three flies with both,
    one with Hexanol only (cC), one with the click only (cD)."""
    H, C = "Hexanol", m.CLICK
    rows = []
    for fly, fn, h_naive, h_pair, c_naive, c_pair in (
            ("bA", 1, 1.0, [3.0, 5.0], 0.0, [1.0, 1.0]),
            ("bA", 2, 2.0, [2.0, 4.0], 1.0, [1.0, 3.0]),
            ("bB", 1, 0.0, [6.0, np.nan], 0.0, [2.0, 0.0])):    # one in-pair trial lost
        rows += [_t(FIRST, fly, fn, m.NAIVE, H, "", h_naive), *_pair(FIRST, fly, fn, H, H, h_pair),
                 _t(FIRST, fly, fn, m.NAIVE, C, "", c_naive), *_pair(FIRST, fly, fn, C, C, c_pair)]
    rows += [_t(FIRST, "bC", 1, m.NAIVE, H, "", 0.0), *_pair(FIRST, "bC", 1, H, H, [9.0, 9.0])]
    rows += [_t(FIRST, "bD", 1, m.NAIVE, C, "", 0.0), *_pair(FIRST, "bD", 1, C, C, [4.0, 4.0])]
    return rows


def _row_of(t, ds, partner, panel="log_auc"):
    hit = t[(t["dataset"] == ds) & (t["partner"] == partner) & (t["panel"] == panel)]
    assert len(hit) == 1
    return hit.iloc[0]


def test_bar_pools_trials_and_the_test_takes_one_value_per_fly():
    t = m.contrast_table(pd.DataFrame(_first_trials()), "log_auc")
    r = _row_of(t, FIRST, "Hexanol")
    # Hexanol: in-pair (3+5+2+4+6)/5 = 4, naive (1+2+0)/3 = 1 -> 3; click: in-pair
    # (1+1+1+3+2+0)/6 = 4/3, naive (0+1+0)/3 = 1/3 -> 1. Bar 3 - 1 = 2 (trials pooled).
    assert r["partner_change"] == pytest.approx(3.0)
    assert r["click_change"] == pytest.approx(1.0)
    assert r["value"] == pytest.approx(2.0)
    assert r["kind"] == m.KINDS[0] and r["scored_odor"] == "Hexanol"
    # per fly V = (mean in-pair - naive) - (click's): 2, 0, 5; only flies with both
    v, batches = [2.0, 0.0, 5.0], ["bA", "bA", "bB"]
    assert r["n_flies"] == 3 and r["n_batches"] == 2
    assert r["mean_per_fly"] == pytest.approx(7 / 3)          # not the bar: never averaged
    want = bs.batch_sign_flip(v, batches, ci=True)
    assert r["p_batch"] == pytest.approx(want["p"])
    assert r["ci_lo"] == pytest.approx(want["ci_lo"], nan_ok=True)
    assert r["ci_hi"] == pytest.approx(want["ci_hi"], nan_ok=True)
    assert r["p_fly"] == pytest.approx(bs.sign_flip_p(v))
    assert (r["n_trials_partner_pair"], r["n_trials_partner_naive"]) == (5, 3)
    assert (r["n_trials_click_pair"], r["n_trials_click_naive"]) == (6, 3)
    assert r["batches"] == "bA;bB"


def test_click_bar_is_its_own_change_over_every_fly_with_it():
    t = m.contrast_table(pd.DataFrame(_first_trials()), "log_auc")
    r = _row_of(t, FIRST, m.CLICK)
    assert r["kind"] == m.KINDS[1]
    # bA1, bA2, bB1, bD1: in-pair (1+1+1+3+2+0+4+4)/8 = 2, naive (0+1+0+0)/4 = 0.25
    assert r["value"] == pytest.approx(1.75) and r["click_change"] == pytest.approx(1.75)
    assert np.isnan(r["partner_change"])
    assert r["n_flies"] == 4 and r["n_batches"] == 3
    assert r["mean_per_fly"] == pytest.approx((1 + 1 + 1 + 4) / 4)
    assert r["p_batch"] == pytest.approx(
        bs.batch_sign_flip([1.0, 1.0, 1.0, 4.0], ["bA", "bA", "bB", "bD"])["p"])


def _second_trials():
    """IAA-second: IAA scored after Citral and after the click; naive = the IAA pre-test
    trial (the fly's naive Citral must not be used)."""
    rows = []
    for fly, fn, iaa, cit, clk in (("sA", 1, 1.0, [3.0, 5.0], [2.0, 2.0]),
                                   ("sB", 1, 0.0, [1.0, 1.0], [0.0, 1.0])):
        rows += [_t(SECOND, fly, fn, m.NAIVE, "IAA", "", iaa),
                 _t(SECOND, fly, fn, m.NAIVE, "Citral", "", 100.0),
                 *_pair(SECOND, fly, fn, "IAA", "Citral", cit),
                 *_pair(SECOND, fly, fn, "IAA", m.CLICK, clk)]
    return rows


def test_second_scores_iaa_against_its_own_naive_trial():
    t = m.contrast_table(pd.DataFrame(_second_trials()), "log_auc")
    r = _row_of(t, SECOND, "Citral")
    assert r["scored_odor"] == "IAA"
    # Citral: in-pair 10/4 - naive IAA 1/2 = 2; click: 5/4 - 1/2 = 0.75
    assert r["partner_change"] == pytest.approx(2.0)
    assert r["click_change"] == pytest.approx(0.75)
    assert r["value"] == pytest.approx(1.25)
    # per fly the naive cancels: V = mean(IAA after Citral) - mean(IAA after the click)
    assert r["mean_per_fly"] == pytest.approx(((4 - 2) + (1 - 0.5)) / 2)
    c = _row_of(t, SECOND, m.CLICK)
    assert c["value"] == pytest.approx(0.75)


def test_holm_spans_every_bar_of_a_panel_and_only_that_panel():
    trials = pd.DataFrame(_first_trials() + _second_trials())
    trials["zpeak_ratio_signed"] = trials["log_auc"]
    trials["zpeak_ratio_unsigned"] = trials["log_auc"] * 2 - 1
    t = m.contrast_table(trials, "peakz_ratio")
    assert set(t["panel"]) == {"zpeak_ratio_signed", "zpeak_ratio_unsigned"}
    for panel, g in t.groupby("panel"):
        # both datasets' partners and both click bars: one family of 4
        assert sorted(zip(g["dataset"], g["partner"])) == sorted(
            [(FIRST, "Hexanol"), (FIRST, m.CLICK), (SECOND, "Citral"), (SECOND, m.CLICK)])
        assert (g["n_tests_in_family"] == 4).all()
        np.testing.assert_allclose(g["p_batch_holm"], fs.holm_adjust(g["p_batch"].tolist()))
        assert g["stars"].tolist() == [fs.stars(h).replace("n.s.", "") for h in g["p_batch_holm"]]
    assert (t["measure"] == "peakz_ratio").all()


# ── naive vs in pair, no click subtraction (user 2026-11-20) ─────────────────
def _peak(rows):
    t = pd.DataFrame(rows)
    return t.assign(zpeak_ratio_signed=t["log_auc"], zpeak_ratio_unsigned=t["log_auc"].abs() + 1)


def test_naive_vs_pair_bars_pool_trials_of_the_flies_with_both():
    bars, st = m.naive_vs_pair_table(_peak(_first_trials()), FIRST)
    h = bars[(bars["panel"] == "zpeak_ratio_signed") & (bars["partner"] == "Hexanol")]
    naive, pair = (h[h["key"] == k].iloc[0] for k in ("pre", "pair"))
    # flies with both: bA1, bA2, bB1, bC1 (the lost in-pair trial drops, the fly stays)
    assert naive["mean"] == pytest.approx((1 + 2 + 0 + 0) / 4)
    assert pair["mean"] == pytest.approx((3 + 5 + 2 + 4 + 6 + 9 + 9) / 7)   # trials, not flies
    assert (naive["n"], pair["n"], naive["n_flies"]) == (4, 7, 4)
    ids = ["bA|1", "bA|2", "bB|1", "bC|1"]
    _, lo, hi = fs.fly_cluster_mean_ci([1, 2, 0, 0], ids)
    assert (naive["ci_lo"], naive["ci_hi"]) == pytest.approx((lo, hi))
    s = st[(st["panel"] == "zpeak_ratio_signed") & (st["partner"] == "Hexanol")].iloc[0]
    d = [4 - 1, 3 - 2, 6 - 0, 9 - 0]                      # per fly: in-pair mean - naive
    assert s["n_flies"] == 4 and s["mean_diff"] == pytest.approx(np.mean(d))
    assert s["p"] == pytest.approx(bs.sign_flip_p(np.array(d, dtype=float)))
    c = bars[(bars["panel"] == "zpeak_ratio_signed") & (bars["partner"] == m.CLICK)]
    assert c.set_index("key")["mean"].to_dict() == pytest.approx({"pre": 0.25, "pair": 2.0})


def test_naive_vs_pair_second_compares_iaa_alone_with_iaa_after_each_partner():
    bars, _ = m.naive_vs_pair_table(_peak(_second_trials()), SECOND)
    b_ = bars[bars["panel"] == "zpeak_ratio_signed"].set_index(["partner", "key"])["mean"]
    assert b_[("Citral", "pre")] == pytest.approx(0.5)      # the IAA naive trials, 1/2
    assert b_[("Citral", "pair")] == pytest.approx(10 / 4)
    assert b_[(m.CLICK, "pre")] == pytest.approx(0.5)
    assert set(bars["scored_odor"]) == {"IAA"}


def test_naive_vs_pair_p_is_holm_adjusted_over_each_panel_s_odors():
    _, st = m.naive_vs_pair_table(_peak(_first_trials()), FIRST)
    for _, g in st.groupby("panel"):
        assert len(g) == 2                                   # Hexanol and the click
        np.testing.assert_allclose(g["p_holm"], fs.holm_adjust(g["p"].tolist()))
        assert g["stars"].tolist() == [b.stars(p) for p in g["p_holm"]]
    assert "batch_level_p" in st.columns and "p_batch" not in st.columns


def test_naive_vs_pair_figure_prints_p_on_every_odor(tmp_path):
    import matplotlib.text

    seen = {}
    real = m._save_figure

    def capture(fig, path):
        seen["text"] = [t.get_text() for t in fig.findobj(matplotlib.text.Text)]
        return real(fig, path)
    m._save_figure = capture
    try:
        bars, st = m.naive_vs_pair_table(_peak(_first_trials()), FIRST)
        paths = m.naive_vs_pair_figure(bars, st, FIRST, tmp_path / "x_peakz_pooled.png",
                                       started="flies started 9:40am–4:20pm")
    finally:
        m._save_figure = real
    assert {p.name for p in paths} == {"x_peakz_pooled.png", "x_peakz_pooled.pdf",
                                       "x_peakz_pooled.csv", "x_peakz_pooled_stats.csv"}
    text = "\n".join(seen["text"])
    for p in st["p"]:
        assert m._p_short(p) in text
    assert "NOT controlled" in text and "Paraffin Oil\n(click)" in text


def test_a_dataset_without_the_click_has_no_bars():
    rows = [r for r in _first_trials() if r["partner"] != m.CLICK and r["scored_odor"] != m.CLICK]
    assert m.contrast_table(pd.DataFrame(rows), "log_auc").empty


def test_partners_follow_the_b2b_order_with_the_click_last():
    trials = pd.DataFrame(_first_trials() + _second_trials())
    order = m.partner_order(trials)
    assert order == ["Hexanol", "Citral", m.CLICK]
    assert [p for p in b.ORDER if p in order] == order


# ── wide rows: which trials, and production's odor-2 trace ───────────────────
def _row(ds, label, fly="b1", fn=1, seed=0, on=30.0):
    rng = np.random.default_rng(seed)
    trace = 10 + rng.normal(0, 1, N)
    trace[int(on):int(on) + 10] += 3 + seed % 4          # odor 1 (or the naive odor)
    trace[41:51] += 1 + seed % 3                          # odor 2 of a pair trial
    r = {"dataset": ds, "fly": fly, "fly_number": fn, "trial_label": label, "fps": 1.0,
         "trial_odor_on_s": on, "trial_odor_off_s": on + 10.0, "trial_duration_s": float(N),
         "trace_len": N}
    r.update({f"dir_val_{i}": float(v) for i, v in enumerate(trace)})
    return r


def _odor2(_row):
    return (41.0, 51.0)


PRE_FIRST = ("pretest_2_linalool-1", "pretest_1_paraffinoil-100", "pretest_8_isoamylacetate-1")
PAIR_FIRST = ("testing_3_isoamylacetate-1_then_linalool-1",
              "testing_10_isoamylacetate-1_then_linalool-1",
              "testing_5_isoamylacetate-1_then_paraffinoil-100",
              "testing_16_isoamylacetate-1_then_paraffinoil-100")
PRE_SECOND = ("pretest_8_isoamylacetate-1", "pretest_2_citral-1", "pretest_1_paraffinoil-100")
PAIR_SECOND = ("testing_5_citral-1_then_isoamylacetate-1",
               "testing_12_citral-1_then_isoamylacetate-1",
               "testing_4_paraffinoil-100_then_isoamylacetate-1",
               "testing_14_paraffinoil-100_then_isoamylacetate-1")


def _wide(flies=(("b1", 1), ("b1", 2), ("b2", 1), ("b3", 1)), datasets=(FIRST, SECOND)):
    pre, pair = [], []
    seed = 0
    for ds in datasets:
        pre_l, pair_l = (PRE_FIRST, PAIR_FIRST) if ds == FIRST else (PRE_SECOND, PAIR_SECOND)
        for fly, fn in flies:
            for label in pre_l:
                seed += 1
                pre.append(_row(ds, label, fly, fn, seed))
            for label in pair_l:
                seed += 1
                pair.append(_row(ds, label, fly, fn, seed))
    return pd.DataFrame(pre), pd.DataFrame(pair)


def test_naive_and_pair_rows_pick_the_scored_odor_and_its_naive_trial():
    pre, pair = _wide(flies=(("b1", 1),))
    naive, o2 = m.naive_and_pair_rows(pre, pair, _odor2)
    got = sorted(zip(o2["dataset"], o2["trial_label"], o2["partner"], o2["scored_odor"],
                     o2["round"]))
    assert got == sorted([
        (FIRST, PAIR_FIRST[0] + "__odor2", "Linalool", "Linalool", 1),
        (FIRST, PAIR_FIRST[1] + "__odor2", "Linalool", "Linalool", 2),
        (FIRST, PAIR_FIRST[2] + "__odor2", m.CLICK, m.CLICK, 1),
        (FIRST, PAIR_FIRST[3] + "__odor2", m.CLICK, m.CLICK, 2),
        (SECOND, PAIR_SECOND[0] + "__odor2", "Citral", "IAA", 1),
        (SECOND, PAIR_SECOND[1] + "__odor2", "Citral", "IAA", 2),
        (SECOND, PAIR_SECOND[2] + "__odor2", m.CLICK, "IAA", 1),
        (SECOND, PAIR_SECOND[3] + "__odor2", m.CLICK, "IAA", 2)])
    assert (o2["role"] == m.IN_PAIR).all() and (naive["role"] == m.NAIVE).all()
    # IAA-first: every odor's naive trial; IAA-second: only IAA's is ever a baseline
    assert sorted(zip(naive["dataset"], naive["scored_odor"])) == sorted(
        [(FIRST, "Linalool"), (FIRST, m.CLICK), (FIRST, "IAA"), (SECOND, "IAA")])


def test_odor2_peak_z_is_taken_on_productions_split_trace():
    pre, pair = _wide(flies=(("b1", 1),), datasets=(FIRST,))
    trials = m.trial_measures(m.naive_and_pair_rows(pre, pair, _odor2), config=None)
    from scripts.analysis import auc_before_during as abd

    label = PAIR_FIRST[0]
    raw = pair[pair["trial_label"] == label].iloc[0]
    trace = raw[[f"dir_val_{i}" for i in range(N)]].to_numpy(float)
    # the pre-odor-1 block, then everything from odor 2's onset, timed as a normal trial
    want = abd.trial_peak_z(np.concatenate([trace[:30], trace[41:]]), fps=1.0, odor_on_s=30.0,
                            odor_off_s=40.0)
    got = trials[trials["trial_label"] == label + "__odor2"].iloc[0]
    for col in ("zpeak_ratio_signed", "zpeak_ratio_unsigned", "zpeak_during_signed"):
        assert got[col] == pytest.approx(want[col])
    naive = trials[(trials["role"] == m.NAIVE) & (trials["scored_odor"] == "Linalool")].iloc[0]
    raw = pre[pre["trial_label"] == "pretest_2_linalool-1"].iloc[0]
    want = abd.trial_peak_z(raw[[f"dir_val_{i}" for i in range(N)]].to_numpy(float), fps=1.0,
                            odor_on_s=30.0, odor_off_s=40.0)
    assert naive["zpeak_ratio_signed"] == pytest.approx(want["zpeak_ratio_signed"])
    assert "log_auc" not in trials.columns                 # no config: no model


def test_log_auc_is_the_model_input_column():
    rows = pd.DataFrame({"x": [0, 1]})
    seen = {}

    def model_features(r, spec, cfg, rule=None):
        seen["spec"] = spec
        return pd.DataFrame({"peak_z": [9.0, 9.0], m.LOG_AUC_FEATURE: [0.5, -1.0]})

    wr = SimpleNamespace(window_spec=lambda name: f"spec:{name}", model_features=model_features)
    cfg = SimpleNamespace(features=["peak_z", m.LOG_AUC_FEATURE], auc_eps=1.0, odor_delay_s=1.0)
    assert m._log_auc(rows, cfg, None, wr).tolist() == [0.5, -1.0]
    assert seen["spec"] == "spec:odor"


def test_log_auc_falls_back_to_the_raw_areas_when_the_model_lacks_it():
    rows = pd.DataFrame({"x": [0, 1]})
    raw = pd.DataFrame({"AUC-Before": [0.0, 1.0], "AUC-During": [np.e - 1, -3.0]})
    wr = SimpleNamespace(window_spec=lambda name: name,
                         model_features=lambda *a, **k: pd.DataFrame({"peak_z": [1.0, 1.0]}),
                         raw_features=lambda r, spec, rule=None, odor_delay_s=None: raw)
    cfg = SimpleNamespace(features=["peak_z"], auc_eps=1.0, odor_delay_s=1.0)
    np.testing.assert_allclose(m._log_auc(rows, cfg, None, wr), [1.0, -np.log(2.0)])


# ── write(): files, a missing model, a missing dataset ───────────────────────
def _config(tmp_path, model=None) -> Path:
    cfg = tmp_path / "config.yaml"
    rp = {"binary_threshold": 2}
    if model is not None:
        rp["model_path"] = str(model)
    cfg.write_text(yaml.safe_dump({"reaction_prediction": rp}))
    return cfg


def test_write_without_a_model_skips_log_auc_and_still_draws_peak_z(tmp_path, capsys):
    pre, pair = _wide()
    paths = m.write(tmp_path, pre, pair, _odor2, config=_config(tmp_path),
                    started="flies started 9:40am–4:20pm")
    d = tmp_path / m.OUT_DIR
    assert m.OUT_DIR == "Change-Minus-Click"
    group = [f"{g}_peakz_pooled{s}" for g in ("IAA-first", "IAA-second")
             for s in (".png", ".pdf", ".csv", "_stats.csv")]
    assert sorted(p.name for p in paths) == sorted([
        "b2b_peakz_ratio_change_minus_click.png", "b2b_peakz_ratio_change_minus_click.pdf",
        "b2b_peakz_ratio_change_minus_click.csv", "b2b_measures_per_trial.csv", *group])
    assert all(p.is_file() for p in paths)
    assert all(p.parent == d for p in paths if p.name not in group)
    assert {p.parent.name for p in paths if p.name in group} == {"IAA-first", "IAA-second"}
    assert "[b2b][WARN]" in capsys.readouterr().out
    t = pd.read_csv(d / "b2b_peakz_ratio_change_minus_click.csv")
    assert set(t["dataset"]) == {FIRST, SECOND}
    assert set(t["partner"]) == {"Linalool", "Citral", m.CLICK}
    per_trial = pd.read_csv(d / "b2b_measures_per_trial.csv")
    assert len(per_trial) == 4 * (3 + 4) + 4 * (1 + 4)      # naive + odor-2 trials


def test_write_with_a_model_adds_the_log_auc_figure(tmp_path, monkeypatch):
    model = tmp_path / "model.json"
    model.write_text("{}")
    monkeypatch.setattr(m, "_log_auc_scorer",
                        lambda model_path, config: (lambda rows: np.arange(len(rows), dtype=float)))
    pre, pair = _wide()
    paths = m.write(tmp_path, pre, pair, _odor2, config=_config(tmp_path, model))
    names = {p.name for p in paths}
    for stem in ("b2b_log_auc_change_minus_click", "b2b_peakz_ratio_change_minus_click"):
        assert {f"{stem}.png", f"{stem}.pdf", f"{stem}.csv"} <= names
    t = pd.read_csv(tmp_path / m.OUT_DIR / "b2b_log_auc_change_minus_click.csv")
    assert (t["panel"] == "log_auc").all() and set(t["dataset"]) == {FIRST, SECOND}
    assert "log_auc" in pd.read_csv(tmp_path / m.OUT_DIR / "b2b_measures_per_trial.csv").columns


def test_write_with_one_dataset_missing_does_not_raise(tmp_path):
    pre, pair = _wide(datasets=(FIRST,))
    paths = m.write(tmp_path, pre, pair, _odor2, config=_config(tmp_path))
    t = pd.read_csv(tmp_path / m.OUT_DIR / "b2b_peakz_ratio_change_minus_click.csv")
    assert set(t["dataset"]) == {FIRST}
    assert any(p.suffix == ".png" for p in paths)


def test_write_with_no_rows_writes_nothing(tmp_path):
    pre, pair = _wide()
    assert m.write(tmp_path, pre.iloc[0:0], pair.iloc[0:0], _odor2,
                   config=_config(tmp_path)) == []
