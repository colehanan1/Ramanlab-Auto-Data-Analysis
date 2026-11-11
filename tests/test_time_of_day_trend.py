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
