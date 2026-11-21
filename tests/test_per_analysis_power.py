"""src/per_analysis/power.py: bootstrap power curves from the observed per-fly data."""
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

from test_per_analysis_cross import CLICK, make_trials, session  # noqa: E402

power = pytest.importorskip("per_analysis.power")
cross = pytest.importorskip("per_analysis.cross")


@pytest.fixture(autouse=True)
def _fast(monkeypatch):
    monkeypatch.setattr(power, "N_SIM", 300)
    monkeypatch.setattr(power, "N_PERM_INNER", 300)
    monkeypatch.setattr(cross, "N_BOOT", 300)
    monkeypatch.setattr(cross, "N_PERM_FLY", 300)


# ── per-fly design values ────────────────────────────────────────────────────
def test_exp2_value_is_x_change_minus_click_change():
    """Benzaldehyde: alone 0, pair 2 and 4 (mean 3); click: alone 1, pair 1 and 3 (mean
    2) -> (3 - 0) - (2 - 1) = 2. Every other partner scores 0 throughout -> 0 - 1 = -1."""
    def score(ctx):
        if ctx["block"] == "prime":
            return 0
        if ctx["odor"] == "Benzaldehyde" and ctx["block"] == "pair":
            return 2 if ctx["presentation"] == 1 else 4
        if ctx["odor"] == CLICK:
            return 1 if ctx["block"] == "alone" else (1 if ctx["presentation"] == 1 else 3)
        return 0
    t = cross.odor_trials(make_trials(
        {"IAA-PairFirst-Control-1-1": [session("november_13_batch_1", batches=(2,))]}, score))
    d = power.exp2_values(t, "score")
    assert set(d["partner"]) == set(cross.PARTNERS)
    assert len(d) == 2 * 7
    assert d.loc[d.partner == "Benzaldehyde", "value"].tolist() == pytest.approx([2.0, 2.0])
    assert d.loc[d.partner == "Hexanol", "value"].tolist() == pytest.approx([-1.0, -1.0])


def test_exp3_value_is_iaa_after_x_minus_after_click():
    def score(ctx):
        if ctx["block"] == "pair":
            return {"Ethyl Butyrate": 4, CLICK: 1}.get(ctx["prime"], 2)
        return 0
    t = cross.odor_trials(make_trials(
        {"IAA-PairSecond-Control-1-1": [session("november_13_batch_1", batches=(3,))]}, score))
    d = power.exp3_values(t, "score")
    assert len(d) == 3 * 7
    assert d.loc[d.partner == "Ethyl Butyrate", "value"].tolist() == pytest.approx([3.0] * 3)
    assert d.loc[d.partner == "Citral", "value"].tolist() == pytest.approx([1.0] * 3)


def test_exp1_values_are_post_minus_pre_on_the_exposed_odor():
    def score(ctx):
        if ctx["odor"] == "Hexanol" and ctx["block"] == "post":
            return 3 if ctx["cohort"].startswith("Hex") else 1
        return 0
    t = cross.odor_trials(make_trials(
        {"Hex-Sensitivity-1-0.1": [session("november_04_batch_1", batches=(3,))],
         "Paraffin-Sensitivity-1-100": [session("november_05_batch_1", batches=(2,))]},
        score))
    designs = power.exp1_values(t, "score")
    hex_ = designs["Hexanol"]
    assert hex_["exposed"].tolist() == pytest.approx([3.0] * 3)
    assert hex_["control"].tolist() == pytest.approx([1.0] * 2)


# ── power ────────────────────────────────────────────────────────────────────
def test_power_is_monotone_in_n_and_delta():
    d = np.random.default_rng(3).normal(0.0, 1.0, 60)
    ns = list(range(20, 151, 10))
    tab = power.simulate(d, kind="paired", deltas=[0.0, 0.25, 0.5], ns=ns)
    tol = 0.08                                     # Monte Carlo slack (n_sim 300)
    for delta, g in tab.groupby("delta"):
        p = g.sort_values("n")["power"].to_numpy()
        if delta > 0:
            assert np.all(np.diff(p) >= -tol), (delta, p)
            assert p[-1] > p[0]
    wide = tab.pivot(index="n", columns="delta", values="power")
    assert (wide[0.5] >= wide[0.25] - tol).all()
    assert (wide[0.25] >= wide[0.0] - tol).all()
    assert wide[0.0].max() < 0.15                  # size near alpha
    two = power.simulate((d[:30], d[30:]), kind="two_group", deltas=[0.5], ns=[20, 80, 150])
    p2 = two.sort_values("n")["power"].to_numpy()
    assert np.all(np.diff(p2) >= -tol) and p2[-1] > p2[0]


def test_binomial_model_is_less_optimistic_on_zero_heavy_contrasts():
    """Most flies never respond: the per-fly contrast is 0 for 50 of 56 flies. Shifting
    every fly by 0.15 is detected far more often at n = 20 than making some flies respond."""
    d = np.r_[np.zeros(50), [1.0, -1.0] * 3]
    shift = power.simulate(d, kind="paired", deltas=[0.15], ns=[20])
    binom = power.simulate_binomial(d, kind="paired", k=2, deltas=[0.0, 0.15], ns=[20, 150])
    s20 = float(shift["power"].iloc[0])
    b = binom.set_index(["delta", "n"])["power"]
    assert s20 > 0.6 and b[(0.15, 20)] < s20 - 0.3
    assert b[(0.0, 20)] < 0.15 and b[(0.15, 150)] > b[(0.15, 20)]
    two = power.simulate_binomial((d, d), kind="two_group", k=1, deltas=[0.0, 0.2],
                                  ns=[20, 150])
    t = two.set_index(["delta", "n"])["power"]
    assert t[(0.0, 20)] < 0.15 and t[(0.2, 150)] > t[(0.2, 20)]


def test_min_n_table_flags_the_15pp_line():
    curves = pd.DataFrame({"experiment": "exp2", "design": "x", "metric": "rate",
                           "delta": [0.15] * 3 + [0.10] * 3, "n": [20, 30, 40] * 2,
                           "power": [0.5, 0.81, 0.9, 0.2, 0.5, 0.7]})
    m = power.min_n_table(curves)
    row = m[m.delta == 0.15].iloc[0]
    assert row["min_n"] == 30 and bool(row["highlight_15pp"])
    assert np.isnan(m[m.delta == 0.10].iloc[0]["min_n"])
    assert not bool(m[m.delta == 0.10].iloc[0]["highlight_15pp"])


def test_run_writes_curves_and_min_n(tmp_path, monkeypatch):
    monkeypatch.setattr(power, "NS", (20, 60, 100))
    def score(ctx):
        return int(ctx["rng"].integers(0, 4))
    cohorts = {"IAA-Sensitivity-1-1": [session("october_12_batch_1", batches=(4, 4))],
               "Hex-Sensitivity-1-0.1": [session("november_04_batch_1", batches=(4,))],
               "3Oct-Sensitivity-1-0.1": [session("november_04_batch_2", batches=(4,))],
               "Paraffin-Sensitivity-1-100": [session("november_05_batch_1", batches=(4, 4))],
               "IAA-PairFirst-Control-1-1": [session("november_13_batch_1", batches=(4, 4))],
               "IAA-PairSecond-Control-1-1": [session("november_16_batch_1", batches=(4, 4))]}
    t = make_trials(cohorts, score)
    out, tables = tmp_path / "results", tmp_path / "results" / "tables"
    findings = power.run(t, out, tables=tables)
    assert findings and all(f["section"] == "power" for f in findings)
    m = pd.read_csv(tables / "power_min_n.csv")
    assert set(m["experiment"]) == {"exp1", "exp2", "exp3"}
    assert {"n_flies", "effect", "ci_lo", "ci_hi", "p", "p_holm", "q_bh", "p_batch", "test",
            "correction", "min_n", "highlight_15pp"} <= set(m.columns)
    assert (out / "06_power" / "README.md").exists()
    assert list((out / "06_power").glob("*.png"))
    assert any("15 pp" in f["text"] for f in findings)
