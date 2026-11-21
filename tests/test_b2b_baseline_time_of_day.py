"""scripts/analysis/b2b_baseline_time_of_day.py: batch-level tests (user 2026-11-18).

Start time, rig and group are batch properties (one batch folder = one rig's run of one
session), so every test the Pre-Exposure-Baseline and Time-of-Day sets draw permutes
whole batch folders, and every figure says "exploratory · Holm-adjusted over N tests".
The fly-level tests stay only as secondary ``*fly_level*`` CSV columns.
"""
import sys
import time
from pathlib import Path

import matplotlib.text
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from fbpipe.analysis import fly_stats as fs  # noqa: E402
import scripts.analysis.starvation_pre_post_comparison as spc  # noqa: E402
from scripts.analysis import b2b_baseline_time_of_day as bt  # noqa: E402
from scripts.analysis import b2b_stats as bs  # noqa: E402
from scripts.analysis import sensitivity_window_results as swres  # noqa: E402
from scripts.analysis import time_of_day_trend as tod  # noqa: E402

FIRST, SECOND = "IAA-PairFirst-Control-1-1", "IAA-PairSecond-Control-1-1"
NAMES = {FIRST: "IAA-first", SECOND: "IAA-second"}
SESSIONS = ("november_13_batch_1", "november_14_batch_1", "november_16_batch_1")
TAILS = ("", "_rig_2", "_rig_3")
ODORS = ("Hexanol", "IAA", "Linalool")
#: each session: one IAA-second folder, on a different rig each time
FOLDERS = {s + t: (SECOND if r == i else FIRST)
           for i, s in enumerate(SESSIONS) for r, t in enumerate(TAILS)}


def _pre(seed: int = 1) -> pd.DataFrame:
    """b2b ``pretest_records``: 9 batches x 3 flies x 3 odors, one trial each."""
    rng = np.random.default_rng(seed)
    rows = [dict(dataset=ds, fly=folder, fly_number=fn, odor=odor,
                 label=f"pretest_{odor}", score=int(rng.integers(0, 6)))
            for folder, ds in FOLDERS.items() for fn in (1, 2, 3) for odor in ODORS]
    return pd.DataFrame(rows)


def _texts(fig) -> list[str]:
    return [t.get_text() for t in fig.findobj(matplotlib.text.Text)]


# ── Pre-Exposure-Baseline ────────────────────────────────────────────────────
def test_baseline_long_carries_each_trial_s_batch_folder():
    long = bt.baseline_long(_pre(), NAMES, 2)
    assert set(long["batch"]) == set(FOLDERS)
    assert len(long) == 9 * 3 * 3


@pytest.mark.parametrize("by", ["cohort", "rig"])
def test_baseline_p_is_batch_level_permuted_within_session(by):
    long = bt.baseline_long(_pre(), NAMES, 2)
    st = bt.baseline_stats(long, by).set_index("odor")
    labels = {b: (NAMES[ds] if by == "cohort" else bt.rig_of(b)) for b, ds in FOLDERS.items()}
    strata = {b: bs.session_of(b) for b in FOLDERS}
    for odor in [*ODORS, swres.ALL_ODORS]:
        g = long if odor == swres.ALL_ODORS else long[long["odor"] == odor]
        for metric, col in (("rate", "responded_pre"), ("score", "score_pre")):
            want = bs.batch_label_perm(g[col].astype(float), g["batch"], labels, strata)
            assert st.loc[odor, f"p_{metric}_batch"] == pytest.approx(want["p"])
        assert st.loc[odor, "n_batches"] == 9
    # 3 sessions: group 3 x 3 x 3 arrangements, rig 3! ** 3, both enumerated exactly
    assert bool(st["perm_exact"].all())
    assert set(st["n_perm"]) == {27 if by == "cohort" else 216}


def test_baseline_p_is_holm_adjusted_over_the_panel_s_odors():
    long = bt.baseline_long(_pre(), NAMES, 2)
    for by in ("cohort", "rig"):
        st = bt.baseline_stats(long, by)
        for metric in ("rate", "score"):
            raw = st[f"p_{metric}_batch"].to_numpy(dtype=float)
            assert np.allclose(st[f"p_{metric}_batch_holm"], fs.holm_adjust(raw), equal_nan=True)
            assert set(st[f"holm_n_{metric}"]) == {int(np.isfinite(raw).sum())} == {4}


def test_baseline_stats_csv_columns_keep_the_fly_level_p_secondary():
    long = bt.baseline_long(_pre(), NAMES, 2)
    st = bt.baseline_stats(long, "cohort")
    assert list(st.columns) == bt.BASELINE_STATS_COLS
    assert st["odor"].iloc[0] == swres.ALL_ODORS
    for col in ("p_rate_ffh", "p_score_kw"):          # no bare fly-level column
        assert col not in st.columns
    # the fly-level column is the old test: KW across groups on one trial per fly
    hex_ = long[long["odor"] == "Hexanol"]
    want = spc._kw([hex_.loc[hex_["cohort"] == c, "score_pre"].to_numpy(dtype=float)
                    for c in sorted(hex_["cohort"].unique())])
    assert st.set_index("odor").loc["Hexanol", "fly_level_p_score_kw"] == pytest.approx(want)


def test_a_batch_in_two_groups_is_refused():
    long = bt.baseline_long(_pre(), NAMES, 2)
    long.loc[long.index[0], "cohort"] = "IAA-second" if long["cohort"].iloc[0] == "IAA-first" \
        else "IAA-first"
    with pytest.raises(ValueError, match="batch"):
        bt.baseline_stats(long, "cohort")


def test_baseline_all_odors_interval_counts_flies_not_trials():
    # 4 flies x 3 odors; two answer every odor, two none: 6 of 12 trials, 2 of 4 flies
    rows = [dict(cohort="IAA-first", odor=o, rig=1, batch="b", fly_id=f"f{i}",
                 score_pre=4.0 if i < 2 else 0.0, responded_pre=i < 2)
            for i in range(4) for o in ODORS]
    long = pd.DataFrame(rows)
    t = bt.baseline_table(long, "cohort").set_index("odor")
    allr = t.loc[swres.ALL_ODORS]
    assert (allr.n_trials, allr.n_flies, allr.pct_responding) == (12, 4, 50.0)  # pooled
    pct, lo, hi = fs.fly_cluster_rate_ci(long["responded_pre"], long["fly_id"])
    assert (allr.ci95_lo, allr.ci95_hi) == (round(lo, 2), round(hi, 2))
    assert allr.ci95_hi - allr.ci95_lo > np.subtract(*swres.wilson_ci(6, 12)[::-1])
    m, slo, shi = fs.fly_cluster_mean_ci(long["score_pre"], long["fly_id"])
    assert (allr.mean_score_pre, allr.score_ci95_lo, allr.score_ci95_hi) == (
        round(m, 4), round(slo, 4), round(shi, 4))
    # one trial per fly: the plain Wilson interval on the count
    one = t.loc["Hexanol"]
    assert (one.ci95_lo, one.ci95_hi) == pytest.approx(swres.wilson_ci(2, 4), abs=0.011)


def test_baseline_grid_has_a_percent_version():
    long = pd.DataFrame({"cohort": ["IAA-first"] * 3, "odor": ["Linalool"] * 3,
                         "rig": [1, 1, 2], "fly_id": ["a", "b", "c"],
                         "score_pre": [0.0, 2.0, 5.0], "responded_pre": [False, True, True]})
    vals, notes, cols = bt.score_grid(long, ["IAA-first"], metric="pct")
    assert cols == ["Rig 1", "Rig 2", "All rigs"]
    assert vals.tolist() == [[50.0, 100.0, pytest.approx(200 / 3)]]
    assert notes[0][0] == "50%\n2 flies"


@pytest.mark.parametrize("metric", ["pct", "score"])
def test_baseline_figure_prints_only_adjusted_batch_level_p(metric):
    long = bt.baseline_long(_pre(), NAMES, 2)
    tables = bt.baseline_tables(long)
    fig = bt.baseline_figure(long, list(NAMES.values()), tables, "t", metric=metric)
    text = "\n".join(_texts(fig))
    assert "batches permuted within session" in text
    assert "exploratory · Holm-adjusted over 4 tests" in text
    assert "*" not in text and "uncorrected" not in text
    st = tables["pre_exposure_stats.csv"]
    col = {"pct": "p_rate_batch_holm", "score": "p_score_batch_holm"}[metric]
    shown = {bt.p_label(p) for p in st[col]} - {""}
    assert shown and shown <= set(_texts(fig))       # the Holm-adjusted p is what is drawn


def test_write_baseline_draws_the_percent_figure_itself(tmp_path, monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("swres.plot_pre_baseline draws fly-level, unadjusted p")
    monkeypatch.setattr(swres, "plot_pre_baseline", refuse)
    written = bt.write_baseline(tmp_path, _pre(), NAMES, threshold=2, window="")
    d = tmp_path / bt.BASELINE_DIR
    assert {p.name for p in written} >= {"pre_exposure_baseline.png",
                                          "pre_exposure_baseline_score.png",
                                          "pre_exposure_stats.csv"}
    st = pd.read_csv(d / "pre_exposure_stats.csv")
    assert list(st.columns) == bt.BASELINE_STATS_COLS
    assert set(st["compared"]) == {"cohort", "rig"}


# ── label permutation that scales ────────────────────────────────────────────
def test_label_perm_is_b2b_stats_own_test_when_it_can_be_enumerated():
    rng = np.random.default_rng(4)
    batches = np.repeat([f"s{i // 3}_{i}" for i in range(9)], 2)
    vals = rng.normal(size=batches.size)
    labels = {b: int(b.split("_")[1]) % 3 for b in set(batches)}
    strata = {b: b.split("_")[0] for b in set(batches)}
    for st in (None, strata):
        assert bt.label_perm(vals, batches, labels, st) == bs.batch_label_perm(
            vals, batches, labels, st)


def test_label_perm_lists_only_the_distinct_arrangements(monkeypatch):
    # bt's own exact enumeration (forced) gives b2b_stats' exact answer
    rng = np.random.default_rng(5)
    batches = np.repeat([f"b{i}" for i in range(10)], 2)
    vals = rng.normal(size=batches.size)
    labels = {f"b{i}": i % 3 for i in range(10)}
    want = bs.batch_label_perm(vals, batches, labels)
    monkeypatch.setattr(bt, "BS_MAX_STRATUM", 0)
    got = bt.label_perm(vals, batches, labels)
    assert got["exact"] and got["n_perm"] == want["n_perm"] == 4200   # 10! / (4! 3! 3!)
    assert got["p"] == pytest.approx(want["p"]) and got["means"] == want["means"]


def test_label_perm_is_exact_and_fast_for_12_batches_in_one_stratum():
    # 12 batches in 4 bins: 12! = 479 M orderings, 166 320 distinct arrangements
    batches = [f"b{i:02d}" for i in range(12)]
    labels = {b: (0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 3, 3)[i] for i, b in enumerate(batches)}
    vals = [float(i) for i in range(12)]
    t0 = time.time()
    r = bt.label_perm(vals, batches, labels)
    assert time.time() - t0 < 10
    assert r["exact"] and r["n_perm"] == 166_320


def test_label_perm_does_not_enumerate_every_ordering_of_a_large_stratum():
    # 16 batches in 4 bins: 16! orderings (bs would list them all), 63 M arrangements
    batches = [f"b{i:02d}" for i in range(16)]
    labels = {b: i % 4 for i, b in enumerate(batches)}
    vals = [float(i % 4 == 0) * 3 + i * 0.01 for i in range(16)]
    t0 = time.time()
    r = bt.label_perm(vals, batches, labels)
    assert time.time() - t0 < 30
    assert not r["exact"] and r["n_perm"] == bs.N_MC
    assert 0 < r["p"] < 0.01
    assert r["n_batches"] == 16 and set(r["means"]) == {0, 1, 2, 3}


# ── Time of day: the test block ──────────────────────────────────────────────
STARTS = {"a": "09:30", "b": "10:15", "c": "12:05", "d": "12:40", "e": "14:10",
          "f": "15:20", "g": "16:45"}


def _changes(seed: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for k, (batch, hhmm) in enumerate(STARTS.items()):
        t = pd.Timestamp(f"2026-11-13 {hhmm}")
        start, label = tod.time_bin(t.to_pydatetime(), 2.0)
        for fn in (1, 2, 3):
            naive = float(rng.integers(0, 3))
            pair = naive + 0.5 * int(rng.integers(-1, 4)) + 0.1 * k
            rows.append(dict(dataset=FIRST, fly=batch, fly_number=fn, bin_start=start,
                             bin_label=label, start_time=t, naive=naive, pair=pair,
                             change=pair - naive))
    return pd.DataFrame(rows)


def test_test_block_across_bins_and_trend_are_batch_level():
    ch = _changes()
    st = bt.test_block_stats(ch, bs.sign_flip_p).set_index("test")
    across, trend = st.loc[bt.TB_ACROSS], st.loc[bt.TB_TREND]
    want = bs.batch_label_perm(ch["change"], ch["fly"], dict(zip(ch["fly"], ch["bin_start"])))
    assert across["p"] == pytest.approx(want["p"]) and across["n_batches"] == 7
    clock = dict(zip(ch["fly"], ch["start_time"].map(tod.clock_hours)))
    wt = bs.batch_trend(ch["change"], ch["fly"], clock)
    assert (trend["rho"], trend["p"]) == (pytest.approx(wt["rho"]), pytest.approx(wt["p"]))
    # fly-level KW / Spearman kept, secondary
    groups = [g["change"].to_numpy(dtype=float) for _, g in ch.groupby("bin_start")]
    assert across["p_fly_level"] == pytest.approx(spc._kw(groups))
    rho_f, p_f = tod._spearman(ch["start_time"].map(tod.clock_hours).to_numpy(dtype=float),
                               ch["change"].to_numpy(dtype=float))
    assert (trend["rho_fly_level"], trend["p_fly_level"]) == (pytest.approx(rho_f),
                                                              pytest.approx(p_f))
    # the two drawn tests are one Holm family of 2
    assert [across["p_holm"], trend["p_holm"]] == pytest.approx(
        list(fs.holm_adjust([across["p"], trend["p"]])))
    assert across["holm_n"] == trend["holm_n"] == 2


def test_test_block_per_bin_paired_p_is_holm_adjusted_across_bins():
    ch = _changes()
    st = bt.test_block_stats(ch, bs.sign_flip_p)
    per = st[st["test"] == bt.TB_PAIRED]
    raw = [bs.sign_flip_p(g["change"]) for _, g in ch.groupby("bin_start")]
    assert per["p"].tolist() == pytest.approx(raw)
    assert per["p_holm"].tolist() == pytest.approx(list(fs.holm_adjust(raw)))
    assert set(per["holm_n"]) == {len(raw)} == {5}
    assert list(st.columns) == bt.TB_COLS


def test_test_block_figure_states_the_batch_level_adjusted_p():
    ch = _changes()
    st = bt.test_block_stats(ch, bs.sign_flip_p)
    recs = pd.DataFrame([dict(dataset=r.dataset, fly=r.fly, fly_number=r.fly_number,
                              role=role, score=s)
                         for r in ch.itertuples() for role, s in (("pre", r.naive),
                                                                  ("post", r.pair))])
    start_of = {(FIRST, b): pd.Timestamp(f"2026-11-13 {h}") for b, h in STARTS.items()}
    table = bt.test_block_table(recs, start_of, width=2.0, threshold=2)
    comp = pd.DataFrame({0.0: [3] * 5}, index=sorted(ch["bin_start"].unique()))
    fig = bt.test_block_figure(table, comp, st, title="t", width=2.0, threshold=2,
                               colors={"naive": "k", "pair": "m"},
                               labels={"naive": "alone", "pair": "pair"})
    text = "\n".join(_texts(fig))
    a = st.set_index("test")
    assert "batch level" in text and "7 batches" in text
    assert "exploratory · Holm-adjusted over 2 tests" in text
    assert bt._p(a.loc[bt.TB_ACROSS, "p_holm"]) in text
    assert bt._p(a.loc[bt.TB_TREND, "p_holm"]) in text
    assert "Kruskal-Wallis" not in text and "uncorrected" not in text and "*" not in text


# ── Time of day: the naive pre-test odor by odor ─────────────────────────────
HOURS = ("09:30", "10:15", "12:05", "12:40", "14:10", "15:20")


def _binned(seed: int = 3, pre: pd.DataFrame | None = None) -> pd.DataFrame:
    """IAA-first's 6 folders x 3 flies x 3 odors, binned 2 h by start time."""
    pre = _pre(seed) if pre is None else pre
    pre = pre[pre["dataset"] == FIRST]
    folders = sorted(pre["fly"].unique())
    start_of = {(FIRST, f): pd.Timestamp(f"2026-11-13 {h}") for f, h in zip(folders, HOURS)}
    frame = bt.pretest_frame(pre, start_of, {k: 1.0 for k in start_of})
    return tod.assign_bins(frame, 2.0)


def test_pretest_odor_table_pools_trials_with_flies_as_the_unit():
    pre = _pre(3)
    # one fly gave IAA twice (an orphan re-run): its two trials pool, it counts once
    extra = pre[(pre["dataset"] == FIRST) & (pre["odor"] == "IAA")].iloc[[0]].assign(score=5)
    b = _binned(pre=pd.concat([pre, extra], ignore_index=True))
    t = bt.pretest_odor_table(b, threshold=2)
    assert len(t) == b.groupby(["bin_start", "odor"]).ngroups
    for r in t.itertuples():
        g = b[(b["bin_start"] == r.bin_start) & (b["odor"] == r.odor)]
        s = g["score_pre"].to_numpy(dtype=float)
        fly = (g["fly"].astype(str) + "|" + g["fly_number"].astype(str)).to_numpy()
        per_fly = pd.Series(s).groupby(fly).mean().to_numpy()
        assert (r.n_flies, r.n_trials) == (len(per_fly), len(s))
        assert r.mean_pre == pytest.approx(s.sum() / len(s))        # trials, not fly means
        assert r.sem_pre_flies == pytest.approx(
            np.std(per_fly, ddof=1) / np.sqrt(len(per_fly)) if len(per_fly) > 1 else np.nan,
            nan_ok=True)
        pct, lo, hi = fs.fly_cluster_rate_ci(s >= 2, fly)
        assert (r.pct_pre, r.pct_ci95_lo, r.pct_ci95_hi) == pytest.approx((pct, lo, hi))
    dup = t[(t["odor"] == "IAA") & (t["n_trials"] > t["n_flies"])]
    assert len(dup) == 1 and int(dup["n_trials"].iloc[0]) == int(dup["n_flies"].iloc[0]) + 1


def test_pretest_odor_order_scored_odor_first_click_last():
    got = bt.odor_order(["Paraffin Oil", "Zeta", "Citral", "IAA", "Hexanol"])
    assert got == ["IAA", "Hexanol", "Citral", "Zeta", "Paraffin Oil"]
    assert "click" in bt.odor_title("Paraffin Oil") and bt.odor_title("Hexanol") == "Hexanol"


def test_pretest_by_odor_figure_one_panel_pair_per_odor_with_its_adjusted_p():
    b = _binned()
    table = bt.pretest_odor_table(b, threshold=2)
    pooled = tod.pooled_pretest(b, binary_threshold=2)
    stats = tod.pretest_stats(b, binary_threshold=2)
    fig = bt.pretest_odor_figure(table, pooled, stats, title="IAA-first", width=2.0,
                                 threshold=2)
    assert len(fig.axes) == 2 * len(ODORS)                 # score + % per odor, no empty cells
    text = "\n".join(_texts(fig))
    st = stats.set_index("odor")
    for odor in ODORS:
        line, _ = bt.odor_stat_line(st.loc[odor])
        assert odor in text and line in text
        for test in tod.PRETEST_TESTS:                     # every drawn p is the Holm one
            p = st.loc[odor, f"{test}_batch_holm"]
            assert not np.isfinite(p) or bt.p_label(p) in line
    assert f"exploratory · Holm-adjusted over {len(ODORS)} tests" in text
    assert "uncorrected" not in text and "*" not in text


def test_pretest_by_odor_score_axis_shows_negative_means():
    """Scores run -1..5: a bin averaging -0.33 (SEM 0.33) must not be cut at 0, and a
    0 % or 0-score point must not sit half under the axis frame."""
    b = _binned()
    table = bt.pretest_odor_table(b, threshold=2)
    first = table.index[0]
    table.loc[first, ["mean_pre", "sem_pre_flies", "pct_pre"]] = [-1 / 3, 1 / 3, 0.0]
    pooled = tod.pooled_pretest(b, binary_threshold=2)
    stats = tod.pretest_stats(b, binary_threshold=2)
    fig = bt.pretest_odor_figure(table, pooled, stats, title="t", width=2.0, threshold=2)
    score_axes, pct_axes = fig.axes[0::2], fig.axes[1::2]
    for ax in score_axes:
        assert ax.get_ylim()[0] <= -2 / 3 - 1e-9
    for ax in pct_axes:
        assert ax.get_ylim()[0] < 0 and ax.get_ylim()[1] >= 100


# ── Time of day: both figures through write_time_of_day ──────────────────────
def test_write_time_of_day_relabels_the_pretest_figure_as_holm_adjusted(tmp_path, monkeypatch):
    pre = _pre()
    pre = pre[pre["dataset"] == FIRST]
    folders = sorted(pre["fly"].unique())
    hours = ("09:30", "10:15", "12:05", "12:40", "14:10", "15:20")
    starts = pd.DataFrame({"dataset": FIRST, "folder": folders,
                           "first_pretest": [pd.Timestamp(f"2026-11-13 {h}") for h in hours]})
    recs = pd.concat([pre.assign(role="pre"),
                      pre.assign(role="post", score=pre["score"].clip(upper=4) + 1)])
    seen = {}
    real_save = tod._save

    def capture(fig, path):
        seen[Path(path).name] = "\n".join(_texts(fig))
        return real_save(fig, path)
    monkeypatch.setattr(tod, "_save", capture)
    written = bt.write_time_of_day(
        tmp_path, pre, recs, starts, [tmp_path], {FIRST: "IAA-first"}, {FIRST: "head"},
        threshold=2, paired_p=bs.sign_flip_p, colors={"naive": "k", "pair": "m"},
        labels={"naive": "alone", "pair": "pair"})
    assert {"time_of_day_pretest.png", "time_of_day_test_block.png",
            "time_of_day_pretest_by_odor.png", "pretest_by_odor_by_bin.csv"} \
        <= {p.name for p in written}
    by_odor = seen["time_of_day_pretest_by_odor.png"]
    assert all(o in by_odor for o in ODORS)
    assert f"exploratory · Holm-adjusted over {len(ODORS)} tests" in by_odor
    pretest = seen["time_of_day_pretest.png"]
    assert "Naive pre-test PER score" in pretest
    assert f"exploratory · Holm-adjusted over {len(ODORS)} tests" in pretest
    assert "uncorrected" not in pretest
    block = seen["time_of_day_test_block.png"]
    assert "exploratory · Holm-adjusted over 2 tests" in block and "*" not in block
    d = tmp_path / bt.time_of_day_dir(2.0) / FIRST
    assert list(pd.read_csv(d / "test_block_stats.csv").columns) == bt.TB_COLS
