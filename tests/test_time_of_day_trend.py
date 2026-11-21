"""PER change vs time of day of the first pre-test (scripts/analysis/time_of_day_trend.py)."""

from __future__ import annotations

import datetime as dt

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import pytest

import scripts.analysis.time_of_day_trend as tod


def _touch(folder, *names):
    folder.mkdir(parents=True, exist_ok=True)
    for n in names:
        (folder / n).write_text("x")


def test_first_pretest_time_is_the_earliest_pretest_file(tmp_path):
    f = tmp_path / "october_08_batch_2"
    _touch(f,
           "output_october_08_batch_2_pretest_2_Citral-1_20261008_132915.csv",
           "output_october_08_batch_2_pretest_1_Hexanol-0.1_20261008_132345.csv",
           "sensors_output_october_08_batch_2_pretest_1_Hexanol-0.1_20261008_132340.csv",
           # training/testing are later in the day and must never win
           "output_october_08_batch_2_training_1_IsoamylAcetate-1_20261008_090000.csv")
    # sensors_* files are logged by a second process; the trial's own file is the clock.
    assert tod.first_pretest_time(f) == dt.datetime(2026, 10, 8, 13, 23, 45)


def test_first_pretest_time_without_pretest_files_is_none(tmp_path):
    f = tmp_path / "x"
    _touch(f, "output_x_testing_1_Hexanol_20261008_132345.csv")
    assert tod.first_pretest_time(f) is None
    assert tod.first_pretest_time(tmp_path / "missing") is None


@pytest.mark.parametrize("t, width, start, label", [
    (dt.datetime(2026, 10, 8, 13, 23), 1, 13.0, "13:00–14:00"),
    (dt.datetime(2026, 10, 8, 13, 23), 2, 12.0, "12:00–14:00"),
    (dt.datetime(2026, 10, 8, 9, 59), 2, 8.0, "8:00–10:00"),
    (dt.datetime(2026, 10, 8, 18, 0), 1, 18.0, "18:00–19:00"),
])
def test_time_bin(t, width, start, label):
    assert tod.time_bin(t, width) == (start, label)


def test_clock_hours():
    assert tod.clock_hours(dt.datetime(2026, 1, 1, 13, 30)) == pytest.approx(13.5)


def _paired():
    # Two folders at 9:10 and 9:50 (same 1 h bin), one at 13:20; two odors each.
    rows = []
    for fly, pre, post in (("a", 0, 3), ("b", 1, 4), ("c", 2, 0)):
        for odor in ("Isoamyl Acetate", "Hexanol"):
            rows.append({"dataset_canon": "IAA-Sensitivity-24-1", "fly": fly, "fly_number": 1,
                         "odor": odor, "score_pre": pre, "score_post": post})
    p = pd.DataFrame(rows)
    times = {"a": dt.datetime(2026, 10, 8, 9, 10), "b": dt.datetime(2026, 10, 8, 9, 50),
             "c": dt.datetime(2026, 10, 8, 13, 20)}
    p["start_time"] = p["fly"].map(times)
    p["starvation_h"] = 24.0
    return p


def test_bin_table_pools_flies_per_bin_and_odor():
    t = tod.bin_table(tod.assign_bins(_paired(), 1))
    iaa = t[t["odor"] == "Isoamyl Acetate"].set_index("bin_start")
    assert list(iaa.index) == [9.0, 13.0]
    assert iaa.loc[9.0, "n_flies"] == 2
    assert iaa.loc[9.0, "mean_delta"] == pytest.approx(3.0)
    assert iaa.loc[9.0, "pct_pre"] == 0.0 and iaa.loc[9.0, "pct_post"] == 100.0
    assert iaa.loc[13.0, "mean_delta"] == pytest.approx(-2.0)
    assert iaa.loc[9.0, "bin_label"] == "9:00–10:00"
    # the x position is the bin centre, so 1 h and 2 h bins line up on one clock axis
    assert iaa.loc[9.0, "x"] == pytest.approx(9.5)


def test_composition_counts_flies_per_bin_and_starvation_group():
    p = _paired()
    p.loc[p["fly"] == "c", "starvation_h"] = 1.0
    comp = tod.composition(tod.assign_bins(p, 2))
    assert comp.loc[8.0].to_dict() == {1.0: 0, 24.0: 2}
    assert comp.loc[12.0].to_dict() == {1.0: 1, 24.0: 0}


def test_spearman_needs_at_least_three_distinct_times():
    p = _paired()
    r = tod.spearman_by_odor(p)
    assert set(r["odor"]) == {"Isoamyl Acetate", "Hexanol"}
    row = r[r["odor"] == "Isoamyl Acetate"].iloc[0]
    assert row["n_flies"] == 3 and -1 <= row["rho"] <= 1
    two = p[p["fly"] != "c"]
    assert tod.spearman_by_odor(two)["p"].isna().all()


def test_starvation_colors_are_ordered_light_to_dark():
    import matplotlib.colors as mcolors

    cols = tod.starvation_colors([1.0, 12.0, 24.0])
    lum = [sum(mcolors.to_rgb(cols[h])) for h in (1.0, 12.0, 24.0)]
    assert lum == sorted(lum, reverse=True)
    assert len(set(tod.starvation_colors([1.0, 6, 12, 18, 24, 30]).values())) == 6


def test_figures_build_with_clock_ticks():
    p = tod.assign_bins(_paired(), 1)
    table = tod.bin_table(p)
    comp = tod.composition(p)
    fig = tod.build_emphasis(table, comp, omnibus=None, trained_odor="Isoamyl Acetate",
                             title="T", width=1)
    labels = " ".join(t.get_text() for ax in fig.axes for t in ax.get_xticklabels())
    assert "9:00" in labels and "13:00" in labels
    assert "Starvation" not in " ".join(ax.get_xlabel() for ax in fig.axes)
    spot = tod.build_spotlight(table, p, comp, omnibus=None, trained_odor="Isoamyl Acetate",
                               title="T", width=1)
    assert spot is not None
    grid = tod.build_grid(table, omnibus=None, trained_odor="Isoamyl Acetate", title="T", width=1)
    assert len(grid.axes) == 6  # 2 odors x (pre/post score, change, rate)
    import matplotlib.pyplot as plt

    for f in (fig, spot, grid):
        plt.close(f)


def test_omnibus_runs_over_bin_labels():
    p = tod.assign_bins(_paired(), 1)
    om = tod.omnibus(p)
    assert set(om["odor"]) == {"Isoamyl Acetate", "Hexanol"}
    assert "p_kw_delta_score" in om
    # one bin -> nothing to compare
    assert tod.omnibus(p[p["bin_start"] == 9.0]).empty


# ── pre-test (naive) by time of day ────────────────────────────────────────


def test_pooled_pretest_pools_trials_not_fly_means():
    p = _paired()
    # fly a misses Hexanol: pooled mean must weight by trials, not average fly means
    p = p[~((p["fly"] == "a") & (p["odor"] == "Hexanol"))]
    p.loc[(p["fly"] == "b") & (p["odor"] == "Hexanol"), "score_pre"] = 3
    t = tod.pooled_pretest(tod.assign_bins(p, 1)).set_index("bin_start")
    # bin 9: trials a/IAA=0, b/IAA=1, b/Hex=3 -> 4/3, not mean(0, 2)=1
    assert t.loc[9.0, "n_trials"] == 3 and t.loc[9.0, "n_flies"] == 2
    assert t.loc[9.0, "mean_pre"] == pytest.approx(4 / 3)
    assert t.loc[9.0, "pct_pre"] == pytest.approx(100 / 3)


def test_pretest_stats_has_every_odor_plus_the_all_odor_row():
    p = tod.assign_bins(_paired(), 1)
    st = tod.pretest_stats(p)
    assert list(st["odor"])[0] == tod.ALL_ODORS
    assert set(st["odor"]) == {tod.ALL_ODORS, "Isoamyl Acetate", "Hexanol"}
    for col in ("p_kw_pre_score", "p_ffh_pre_rate", "rho_pre", "p_spearman_pre"):
        assert col in st


def test_spearman_measure_pre():
    r = tod.spearman_by_odor(_paired(), measure="pre")
    row = r[r["odor"] == "Isoamyl Acetate"].iloc[0]
    # pre scores 0, 1, 2 at 9:10, 9:50, 13:20 -> perfectly monotone
    assert row["rho"] == pytest.approx(1.0)


def test_pretest_figure_builds():
    import matplotlib.pyplot as plt

    p = tod.assign_bins(_paired(), 1)
    fig = tod.build_pretest(tod.bin_table(p), tod.pooled_pretest(p), tod.composition(p),
                            stats=tod.pretest_stats(p), title="T", width=1)
    text = " ".join(t.get_text() for ax in fig.axes for t in ax.texts)
    assert "all odors" in text.lower()
    assert any("Pre-test" in ax.get_ylabel() for ax in fig.axes)
    plt.close(fig)


def _minor_labels(ax):
    return [t.get_text() for t in ax.get_xticklabels(minor=True) if t.get_text()]


def test_every_figure_shows_n_under_each_bin():
    import matplotlib.pyplot as plt

    p = tod.assign_bins(_paired(), 1)
    table, comp, pooled = tod.bin_table(p), tod.composition(p), tod.pooled_pretest(p)
    kw = dict(omnibus=None, trained_odor="Isoamyl Acetate", title="T", width=1)
    figs = [tod.build_pretest(table, pooled, comp, stats=tod.pretest_stats(p), title="T", width=1),
            tod.build_emphasis(table, comp, **kw), tod.build_spotlight(table, p, comp, **kw)]
    for fig in figs:
        assert _minor_labels(fig.axes[-1] if fig is not figs[2] else fig.axes[2]) == ["n=2", "n=1"]
        plt.close(fig)
    grid = tod.build_grid(table, **kw)
    labels = [t.get_text() for t in grid.axes[-1].get_xticklabels()]
    assert any("n=2" in s for s in labels) and any("n=1" in s for s in labels)
    plt.close(grid)


def test_starvation_colors_past_six_levels_stay_ordered():
    import matplotlib.colors as mcolors

    levels = [1.0, 6, 7, 12, 18, 24, 30, 40]
    cols = tod.starvation_colors(levels)
    lum = [sum(mcolors.to_rgb(cols[float(h)])) for h in levels]
    assert lum == sorted(lum, reverse=True) and len(set(cols.values())) == 8


# ── phase wording by protocol (scripts/analysis/phase_terms.py) ────────────


def _drawn_text(figs) -> str:
    """Every string the figures draw (suptitle, titles, labels, notes, legends); closes them."""
    import matplotlib.pyplot as plt

    parts = []
    for fig in figs:
        parts.append(fig.get_suptitle())
        for ax in fig.axes:
            parts += [ax.get_title(loc=loc) for loc in ("left", "center", "right")]
            parts += [ax.get_xlabel(), ax.get_ylabel()] + [t.get_text() for t in ax.texts]
            if ax.get_legend():
                parts += [t.get_text() for t in ax.get_legend().get_texts()]
        parts += [t.get_text() for leg in fig.legends for t in leg.get_texts()]
        plt.close(fig)
    return "\n".join(parts)


def _every_figure(cohort):
    p = tod.assign_bins(_paired(), 1)
    table, comp, pooled = tod.bin_table(p), tod.composition(p), tod.pooled_pretest(p)
    # a real omnibus, so the "post-... FFH" notes are drawn too
    kw = dict(omnibus=tod.omnibus(p), trained_odor="Isoamyl Acetate", title="T", width=1,
              cohort=cohort)
    return [tod.build_pretest(table, pooled, comp, stats=tod.pretest_stats(p), title="T",
                              width=1, cohort=cohort),
            tod.build_emphasis(table, comp, **kw), tod.build_spotlight(table, p, comp, **kw),
            tod.build_grid(table, **kw)]


def test_sensitivity_figures_say_pre_and_post_sensitization_and_sensitized_odor():
    text = _drawn_text(_every_figure("IAA-Sensitivity-24-1"))
    for new in ("Pre-exposure", "Post-exposure", "post-exposure FFH",
                "Pre-exposure PER score", "is pre-exposure PER different",
                "Start of pre-exposure", "effect of exposure",
                "Sensitized odor mean ± SEM", "Isoamyl Acetate (sensitized odor)",
                "Isoamyl Acetate  (sensitized odor)"):
        assert new in text
    # nothing is paired with the odor: no conditioning words anywhere
    for old in ("Pre-test", "pre-test", "Post-training", "post-training", "CS+", "naive", "train"):
        assert old not in text


def test_conditioning_cohort_keeps_pre_test_post_training_and_cs_plus():
    text = _drawn_text(_every_figure("Hex-HexTest-Training-32-1"))
    for old in ("Pre-test (naive)", "Post-training", "post-training FFH",
                "Pre-test PER score\n(naive, before training)", "is the naive pre-test different",
                "Start of first pre-test", "First pre-test (n = flies)", "effect of training",
                "CS+ mean ± SEM", "Isoamyl Acetate (trained, CS+)",
                "Isoamyl Acetate (trained odor, CS+)", "Isoamyl Acetate  (CS+)"):
        assert old in text
    assert "sensitiz" not in text.lower() and "exposure" not in text.lower()


def test_render_width_words_figures_by_the_datasets_it_draws(monkeypatch, tmp_path):
    drawn = []
    monkeypatch.setattr(tod, "_save", lambda fig, path: drawn.append(fig) or path)
    p = _paired()
    kw = dict(trained="Isoamyl Acetate", title="T", spearman=tod.spearman_by_odor(p))
    tod.render_width(p, 1, out_dir=tmp_path / "sensitivity", **kw)
    assert "Post-exposure" in _drawn_text(drawn)  # from the data, not the title
    drawn.clear()
    mixed = p.assign(dataset_canon=p["dataset_canon"].where(p["fly"] != "c",
                                                            "Hex-HexTest-Training-32-1"))
    tod.render_width(mixed, 1, out_dir=tmp_path / "mixed", **kw)
    text = _drawn_text(drawn)
    assert "Post-training" in text and "sensitiz" not in text.lower() and "exposure" not in text.lower()


# ── odor concentration in the figure text (scripts/analysis/odor_conc_labels.py) ──

_CONC = {"Isoamyl Acetate": "1", "Hexanol": "0.1"}


def _every_figure_with_conc(cohort, odor_conc):
    p = tod.assign_bins(_paired(), 1)
    table, comp, pooled = tod.bin_table(p), tod.composition(p), tod.pooled_pretest(p)
    kw = dict(omnibus=tod.omnibus(p), trained_odor="Isoamyl Acetate", title="T", width=1,
              cohort=cohort, odor_conc=odor_conc)
    figs = [tod.build_pretest(table, pooled, comp, stats=tod.pretest_stats(p), title="T",
                              width=1, cohort=cohort, odor_conc=odor_conc),
            tod.build_emphasis(table, comp, **kw), tod.build_spotlight(table, p, comp, **kw),
            tod.build_grid(table, **kw)]
    return _drawn_text(figs)


def test_sensitivity_figures_name_each_odor_with_its_concentration():
    text = _every_figure_with_conc("IAA-Sensitivity-1-1", _CONC)
    for new in ("Isoamyl Acetate 1% (v/v) (sensitized odor)",          # legend, suptitle
                "Isoamyl Acetate 1% (v/v): Kruskal-Wallis across bins",
                "Isoamyl Acetate 1% (v/v)\n(sensitized odor)",          # grid column, wrapped
                "Hexanol 0.1% (v/v)"):
        assert new in text
    assert "Isoamyl Acetate (sensitized odor)" not in text


def test_odor_with_no_recorded_concentration_keeps_its_bare_name():
    text = _every_figure_with_conc("IAA-Sensitivity-40-1", {"Hexanol": "0.1"})
    assert "Isoamyl Acetate  (sensitized odor)" in text  # one line, as before
    assert "Isoamyl Acetate (sensitized odor)" in text
    assert "Isoamyl Acetate 1%" not in text and "Hexanol 0.1% (v/v)" in text


def test_conditioning_figures_never_add_a_concentration():
    text = _every_figure_with_conc("Hex-HexTest-Training-32-1", _CONC)
    assert "% (v/v)" not in text
    assert "Isoamyl Acetate  (CS+)" in text and "Isoamyl Acetate (trained, CS+)" in text


def test_load_paired_keeps_the_stripped_concentration_for_the_figures(tmp_path):
    rows = []
    for fly, conc in (("october_08_batch_1", "-1"), ("october_09_batch_1", "-1")):
        for phase, n in (("pretest", 1), ("testing", 2)):
            for odor, c in (("isoamylacetate", conc), ("hexanol", "")):
                rows.append(dict(dataset="IAA-Sensitivity-1-1", fly=fly, fly_number=1,
                                 trial_label=f"{phase}_{n}_{odor}{c}", score=1,
                                 trial_type=phase))
    csv = tmp_path / "preds.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    paired = tod.load_paired(["IAA-Sensitivity-1-1"], predictions_csv=csv,
                             flagged_flies_csv=tmp_path / "none.csv", tmp_dir=tmp_path / "in")
    assert paired.attrs["odor_conc"] == {"Isoamyl Acetate": "1"}
    assert set(paired["odor"]) == {"Isoamyl Acetate", "Hexanol"}  # the key stays bare


# ── --min-flies: only bins with enough flies are drawn and tested ──


def test_drop_small_bins_counts_flies_not_rows():
    # bin 9:00 holds flies a and b (2 odors each -> 4 rows), bin 13:00 holds fly c
    binned = tod.assign_bins(_paired(), 1)
    kept = tod.drop_small_bins(binned, 2)
    assert sorted(kept["fly"].unique()) == ["a", "b"]
    assert set(kept["bin_start"]) == {9.0}
    assert tod.drop_small_bins(binned, 3).empty
    assert len(tod.drop_small_bins(binned, 0)) == len(binned)


def test_render_width_min_flies_drops_the_bin_from_tables_and_stats(monkeypatch, tmp_path):
    monkeypatch.setattr(tod, "_save", lambda fig, path: path)
    p = _paired()
    tod.render_width(p, 1, out_dir=tmp_path, trained="Isoamyl Acetate", title="T",
                     spearman=tod.spearman_by_odor(p), min_flies=2)
    table = pd.read_csv(tmp_path / "time_of_day_table.csv")
    assert set(table["bin_start"]) == {9.0}
    comp = pd.read_csv(tmp_path / "flies_per_bin_by_starvation.csv")
    assert comp["bin_start"].tolist() == [9.0]
    pooled = pd.read_csv(tmp_path / "pretest_pooled_by_bin.csv")
    assert set(pooled["bin_start"]) == {9.0}


# ── batch-aware tests (2026-11-18): start time is a property of the batch ──────
# Every fly of a batch folder starts together, so a fly-level trend or bin test counts
# one start time 4-10 times over. The trend is tested on one pooled value per batch;
# the bin comparisons get a p from permuting whole batches between the bins.

import itertools  # noqa: E402
import math  # noqa: E402

import numpy as np  # noqa: E402
from scipy.stats import kruskal, spearmanr  # noqa: E402

from fbpipe.analysis import fly_stats as fs  # noqa: E402

IAA = "Isoamyl Acetate"
#: Six batches of ten flies, one batch a day, starting 8:05, 9:05 ... 13:05, so 2 h bins
#: hold two batches each. The batch (its day) shifts every fly's IAA change together:
#: by bin 8:00 (0, 0.5), 10:00 (2, 3), 12:00 (1, 1.5). Within a batch flies barely differ.
_EFFECTS = (0.0, 0.5, 2.0, 3.0, 1.0, 1.5)


def _batch_folder(b):
    return f"october_{b + 1:02d}_batch_1"


def _batched(effects=_EFFECTS, flies=10, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for b, eff in enumerate(effects):
        start = dt.datetime(2026, 10, 1 + b, 8 + b, 5)
        for n in range(1, flies + 1):
            pre = float(rng.integers(0, 2))
            for odor, shift in ((IAA, eff), ("Hexanol", 0.0)):
                rows.append({"dataset_canon": "IAA-Sensitivity-1-1", "fly": _batch_folder(b),
                             "fly_number": n, "odor": odor, "score_pre": pre,
                             "score_post": pre + shift + rng.normal(0, 0.05),
                             "start_time": start, "starvation_h": 1.0})
    return pd.DataFrame(rows)


def _clock_of_batches(n=6):
    return [8 + b + 5 / 60 for b in range(n)]


def test_trend_is_tested_on_one_pooled_value_per_batch():
    p = _batched()
    r = tod.spearman_by_odor(p).set_index("odor").loc[IAA]
    assert r["p"] < 1e-4                       # 60 flies, but only 6 start times
    assert r["n_batches"] == 6
    iaa = p[p["odor"] == IAA]
    means = [float((g["score_post"] - g["score_pre"]).sum() / len(g))
             for _, g in iaa.groupby("fly", sort=True)]
    rho, pv = spearmanr(_clock_of_batches(), means)
    assert r["rho_batch"] == pytest.approx(rho) and r["p_batch"] == pytest.approx(pv)
    assert r["rho_batch"] == pytest.approx(1 - 6 * 16 / (6 * 35))   # ranks 1 2 5 6 3 4
    assert r["p_batch"] > 0.05                 # the fly-level "trend" is six batches


def test_trend_pools_trials_within_a_batch_not_fly_means():
    # batch b's flies: one with two IAA rows (0 and 3), one with a single row (3):
    # pooled (0 + 3 + 3) / 3 = 2, never the mean of fly means (1.5 + 3) / 2 = 2.25
    rows = []
    for b, values in enumerate(([0.0, 3.0, 3.0], [1.0], [2.5], [4.0])):
        for i, v in enumerate(values):
            rows.append({"dataset_canon": "D", "fly": _batch_folder(b),
                         "fly_number": 1 if (b == 0 and i < 2) else 2 + i,
                         "odor": IAA, "score_pre": 0.0, "score_post": v,
                         "start_time": dt.datetime(2026, 10, 1, 9 + b, 0)})
    rho, p, n = tod.batch_spearman(
        [9.0, 9.0, 9.0, 10.0, 11.0, 12.0], [0.0, 3.0, 3.0, 1.0, 2.5, 4.0],
        ["b0", "b0", "b0", "b1", "b2", "b3"])
    assert n == 4
    expected = spearmanr([9.0, 10.0, 11.0, 12.0], [2.0, 1.0, 2.5, 4.0])
    assert rho == pytest.approx(expected.statistic) and p == pytest.approx(expected.pvalue)
    r = tod.spearman_by_odor(pd.DataFrame(rows)).iloc[0]
    assert r["rho_batch"] == pytest.approx(expected.statistic)


def _exact_batch_kw_p(values, bins, batches):
    """Every way of dealing the batches into the bins (bin sizes in batches kept),
    scipy's Kruskal-Wallis H on the flies each time: the exact batch-permutation p."""
    df = pd.DataFrame({"v": values, "bin": bins, "batch": batches})
    batch_bin = df.groupby("batch")["bin"].first()
    ids = list(batch_bin.index)
    h_obs = kruskal(*[g["v"].to_numpy() for _, g in df.groupby("bin")]).statistic
    deals = set(itertools.permutations(batch_bin.to_list()))
    hits = 0
    for deal in deals:
        relabel = dict(zip(ids, deal))
        new = df["batch"].map(relabel)
        h = kruskal(*[df.loc[new == lab, "v"].to_numpy() for lab in sorted(set(deal))]).statistic
        hits += h >= h_obs - 1e-9
    return hits / len(deals)


def test_batch_permutation_p_moves_every_fly_of_a_batch_together():
    # 4 batches x 3 flies, every fly of a batch alike; bins {b1, b2} vs {b3, b4}.
    values = np.repeat([1.0, 2.0, 3.0, 4.0], 3)
    batches = np.repeat(["b1", "b2", "b3", "b4"], 3)
    bins = np.repeat(["early", "early", "late", "late"], 3)
    p_batch = tod.batch_permutation_p(values, bins, batches)
    # 6 ways to split 4 batches 2 + 2; this split and its mirror are the most extreme
    assert p_batch == pytest.approx(2 / 6, abs=0.01)
    assert p_batch == pytest.approx(_exact_batch_kw_p(values, bins, batches), abs=0.01)
    # each fly its own batch: 924 ways, 2 as extreme -- the fly-level p is far smaller
    alone = tod.batch_permutation_p(values, bins, np.arange(12))
    assert alone == pytest.approx(2 / 924, abs=0.002)
    # fixed seed: the same p every run; (hits + 1) / (n + 1) is never 0
    assert tod.batch_permutation_p(values, bins, batches) == p_batch
    assert tod.batch_permutation_p(values, bins, np.arange(12), n_perm=50) >= 1 / 51


def test_batch_permutation_p_is_nan_when_nothing_can_be_compared():
    assert np.isnan(tod.batch_permutation_p([1.0, 2.0], ["a", "a"], ["b1", "b2"]))  # one bin
    assert np.isnan(tod.batch_permutation_p([2.0] * 4, ["a", "a", "b", "b"],
                                            ["b1", "b2", "b3", "b4"]))              # no spread
    assert np.isnan(tod.batch_permutation_p([0, 0, 0, 0], ["a", "a", "b", "b"],
                                            ["b1", "b2", "b3", "b4"], statistic="ffh"))
    # a batch whose flies sit in two bins cannot be moved as one
    assert np.isnan(tod.batch_permutation_p([1.0, 2.0, 3.0], ["a", "b", "b"],
                                            ["b1", "b1", "b2"]))


def test_bin_comparison_p_permutes_whole_batches():
    binned = tod.assign_bins(_batched(), 2)
    om = tod.omnibus(binned).set_index("odor")
    iaa = om.loc[IAA]
    assert iaa["p_kw_delta_score"] < 1e-6      # flies as the unit
    assert iaa["n_batches"] == 6
    # 90 ways to deal 6 batches 2 + 2 + 2; the observed split is the most extreme and
    # every relabelling of the three bins gives the same H: exactly 6 / 90
    assert iaa["p_kw_delta_score_batch"] == pytest.approx(6 / 90, abs=0.01)
    sub = binned[binned["odor"] == IAA]
    exact = _exact_batch_kw_p((sub["score_post"] - sub["score_pre"]).to_numpy(),
                              sub["bin_label"].to_numpy(), sub["fly"].to_numpy())
    assert iaa["p_kw_delta_score_batch"] == pytest.approx(exact, abs=0.01)
    assert iaa["p_kw_delta_score_batch"] > 0.05


def _rate_batches(post_responders=(0, 1, 9, 10, 5, 4), flies=10):
    """Six batches of ten, two per 2 h bin; batch b has ``post_responders[b]`` flies
    responding post (score 3), the rest 0; nobody responds pre."""
    rows = []
    for b, k in enumerate(post_responders):
        for n in range(flies):
            rows.append({"dataset_canon": "D", "fly": _batch_folder(b), "fly_number": n + 1,
                         "odor": IAA, "score_pre": 0.0, "score_post": 3.0 if n < k else 0.0,
                         "start_time": dt.datetime(2026, 10, 1 + b, 8 + b, 5),
                         "starvation_h": 1.0})
    return pd.DataFrame(rows)


def _exact_batch_ffh_p(responders, n_per_batch, deal_bins):
    """Exact: every deal of the batches into bins (two per bin), the bins x responded
    table's hypergeometric probability given ITS margins; p = P(as improbable)."""
    def prob(ks, ns):
        return math.prod(math.comb(n, k) for k, n in zip(ks, ns))
    def table(deal):
        lab = sorted(set(deal))
        return ([sum(r for r, d in zip(responders, deal) if d == g) for g in lab],
                [sum(n for n, d in zip(n_per_batch, deal) if d == g) for g in lab])
    p_obs = prob(*table(deal_bins))
    deals = set(itertools.permutations(deal_bins))
    return sum(prob(*table(d)) <= p_obs * (1 + 1e-12) for d in deals) / len(deals)


def test_rate_comparison_p_permutes_whole_batches_with_the_ffh_statistic():
    binned = tod.assign_bins(_rate_batches(), 2)
    iaa = tod.omnibus(binned).set_index("odor").loc[IAA]
    assert iaa["p_ffh_post_rate"] < 0.001      # flies as the unit
    exact = _exact_batch_ffh_p((0, 1, 9, 10, 5, 4), (10,) * 6, (8, 8, 10, 10, 12, 12))
    assert exact == pytest.approx(6 / 90)
    assert iaa["p_ffh_post_rate_batch"] == pytest.approx(exact, abs=0.01)
    assert np.isnan(iaa["p_ffh_pre_rate_batch"])   # nobody responds pre: nothing to test


def test_per_odor_bin_tests_and_trends_are_holm_adjusted_across_odors():
    binned = tod.assign_bins(_batched(), 2)
    om = tod.omnibus(binned)
    assert len(om) == 2
    for col in ("p_kw_pre_score", "p_kw_post_score", "p_kw_delta_score",
                "p_ffh_pre_rate", "p_ffh_post_rate"):
        np.testing.assert_allclose(om[f"{col}_batch_holm"], fs.holm_adjust(om[f"{col}_batch"]),
                                   equal_nan=True)
    sp = tod.spearman_by_odor(binned)
    np.testing.assert_allclose(sp["p_batch_holm"], fs.holm_adjust(sp["p_batch"]), equal_nan=True)


def test_pretest_stats_carry_batch_level_columns_and_holm():
    binned = tod.assign_bins(_batched(), 2)
    st = tod.pretest_stats(binned).set_index("odor")
    for col in ("n_batches", "p_kw_pre_score_batch", "p_ffh_pre_rate_batch", "rho_pre_batch",
                "p_spearman_pre_batch", "p_kw_pre_score_batch_holm",
                "p_ffh_pre_rate_batch_holm", "p_spearman_pre_batch_holm"):
        assert col in st
    # the fly-level values stay, as before
    for col in ("p_kw_pre_score", "p_ffh_pre_rate", "rho_pre", "p_spearman_pre"):
        assert col in st
    allrow = st.loc[tod.ALL_ODORS]
    assert allrow["n_batches"] == 6
    # all odors: each batch's every pre trial pooled (sum / trials), against its start time
    pooled = [float(g["score_pre"].sum() / len(g)) for _, g in binned.groupby("fly", sort=True)]
    exp = spearmanr(_clock_of_batches(), pooled)
    assert allrow["rho_pre_batch"] == pytest.approx(exp.statistic)
    assert allrow["p_spearman_pre_batch"] == pytest.approx(exp.pvalue)
    # per odor: Holm across the odors; the all-odor row: Holm over its own three tests
    odors = st.drop(index=tod.ALL_ODORS)
    for col in ("p_kw_pre_score_batch", "p_ffh_pre_rate_batch", "p_spearman_pre_batch"):
        np.testing.assert_allclose(odors[f"{col}_holm"], fs.holm_adjust(odors[col]),
                                   equal_nan=True)
    three = ["p_kw_pre_score_batch", "p_ffh_pre_rate_batch", "p_spearman_pre_batch"]
    np.testing.assert_allclose(allrow[[f"{c}_holm" for c in three]].to_numpy(dtype=float),
                               fs.holm_adjust(allrow[three].to_numpy(dtype=float)),
                               equal_nan=True)


def test_without_a_batch_column_the_batch_columns_are_blank_and_one_warning_prints(capsys):
    p = _batched().drop(columns=["fly", "dataset_canon"])
    sp = tod.spearman_by_odor(p)
    assert sp["p"].notna().all()               # the fly-level values are still there
    assert sp["p_batch"].isna().all() and sp["rho_batch"].isna().all()
    out = capsys.readouterr().out
    assert out.count("[WARN]") == 1 and "batch" in out
    om = tod.omnibus(tod.assign_bins(p, 2))
    assert om["p_kw_delta_score"].notna().all()
    assert om["p_kw_delta_score_batch"].isna().all()
    assert capsys.readouterr().out.count("[WARN]") == 1


def test_spearman_helper_keeps_its_plain_two_array_signature():
    """b2b_baseline_time_of_day calls ``tod._spearman(clock, change)`` directly."""
    import inspect

    assert list(inspect.signature(tod._spearman).parameters) == ["clock", "value"]
    x, y = np.array([9.0, 10.0, 11.0, 12.0]), np.array([1.0, 3.0, 2.0, 4.0])
    exp = spearmanr(x, y)
    assert tod._spearman(x, y) == pytest.approx((exp.statistic, exp.pvalue))
    assert all(np.isnan(tod._spearman(np.array([9.0, 9.0, 10.0]), np.array([1.0, 2.0, 3.0]))))


def test_b2b_still_imports_and_its_pretest_frame_gets_batch_level_stats():
    """The b2b scripts belong to another session: what they import must keep working."""
    from scripts.analysis import b2b_baseline_time_of_day as bt

    pre = pd.DataFrame([
        {"dataset": "PairFirst", "fly": _batch_folder(b), "fly_number": n, "odor": odor,
         "score": float((b + n) % 3)}
        for b in range(4) for n in (1, 2) for odor in (IAA, "Hexanol")])
    start_of = {("PairFirst", _batch_folder(b)): pd.Timestamp(2026, 11, 13 + b, 9 + 2 * b, 0)
                for b in range(4)}
    starv = {k: 1.0 for k in start_of}
    frame = bt.pretest_frame(pre, start_of, starv)
    binned = tod.assign_bins(frame, 2.0)
    st = tod.pretest_stats(binned, binary_threshold=2)
    assert st.set_index("odor").loc[tod.ALL_ODORS, "n_batches"] == 4
    assert np.isfinite(st["p_spearman_pre_batch"].iloc[0])
    tod.bin_table(binned, binary_threshold=2)
    tod.pooled_pretest(binned, binary_threshold=2)


def _crafted_tables():
    """Fly-level p tiny, batch-level p large: a figure must draw the batch-level one."""
    p = tod.assign_bins(_paired(), 1)
    table, comp, pooled = tod.bin_table(p), tod.composition(p), tod.pooled_pretest(p)
    om = pd.DataFrame({"odor": ["Isoamyl Acetate", "Hexanol"]})
    for col in ("p_kw_pre_score", "p_kw_post_score", "p_kw_delta_score",
                "p_ffh_pre_rate", "p_ffh_post_rate"):
        om[col] = 0.0001
        om[f"{col}_batch"] = 0.31
        om[f"{col}_batch_holm"] = 0.62
    sp = pd.DataFrame({"odor": ["Isoamyl Acetate", "Hexanol"], "n_flies": 3, "rho": 0.99,
                       "p": 0.0001, "n_batches": 3, "rho_batch": 0.5, "p_batch": 0.33,
                       "p_batch_holm": 0.66})
    st = tod.pretest_stats(p)
    for c in ("p_kw_pre_score", "p_ffh_pre_rate", "p_spearman_pre"):
        st[c] = 0.0001
        st[f"{c}_batch"] = 0.35
        st[f"{c}_batch_holm"] = 0.7
    st["rho_pre_batch"] = 0.25
    return p, table, comp, pooled, om, sp, st


def test_figures_draw_the_batch_level_holm_adjusted_p():
    p, table, comp, pooled, om, sp, st = _crafted_tables()
    kw = dict(omnibus=om, trained_odor="Isoamyl Acetate", title="T", width=1,
              cohort="IAA-Sensitivity-1-1")
    figs = [tod.build_emphasis(table, comp, spearman=sp, **kw),
            tod.build_spotlight(table, p, comp, spearman=sp, **kw),
            tod.build_grid(table, **kw),
            tod.build_pretest(table, pooled, comp, stats=st, title="T", width=1,
                              cohort="IAA-Sensitivity-1-1")]
    text = _drawn_text(figs)
    assert "p=0.620" in text and "p=0.660" in text and "p=0.700" in text
    assert "p<0.001" not in text and "p=0.000" not in text
    assert "3 batches" in text                 # the trend's n is batches, not flies
    assert text.count("exploratory · Holm-adjusted") >= 4
    assert "uncorrected" not in text


def test_render_width_writes_the_batch_level_columns(monkeypatch, tmp_path):
    monkeypatch.setattr(tod, "_save", lambda fig, path: path)
    p = _batched()
    tod.render_width(p, 2, out_dir=tmp_path, trained=IAA, title="T",
                     spearman=tod.spearman_by_odor(p))
    om = pd.read_csv(tmp_path / "stats_omnibus.csv")
    for col in ("n_batches", "p_kw_delta_score_batch", "p_kw_delta_score_batch_holm",
                "p_ffh_post_rate_batch_holm", "p_kw_delta_score"):
        assert col in om
    st = pd.read_csv(tmp_path / "pretest_stats.csv")
    assert {"p_spearman_pre_batch_holm", "rho_pre_batch", "n_batches"} <= set(st.columns)
