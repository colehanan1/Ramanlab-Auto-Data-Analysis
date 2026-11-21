"""Area under the PER trace before vs during the odor, signed and unsigned.

Each trial's trace is referenced to its own pre-odor median. The signed area
counts a dip below that baseline as negative; the unsigned area counts every
excursion as positive. ``before`` is the window of the same length just ahead
of odor onset, so ``during - before`` is the odor-evoked change on that fly's
own trial.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.text import Text

import scripts.analysis.auc_before_during as abd

FPS = 40.0


def _trace(before, during, after=None, fps=FPS, on_s=30.0, off_s=60.0):
    """Piecewise trace: ``before`` for [0, on), ``during`` for [on, off)."""
    n_before, n_during = int(on_s * fps), int((off_s - on_s) * fps)
    parts = [np.broadcast_to(np.asarray(before, float), n_before),
             np.broadcast_to(np.asarray(during, float), n_during)]
    if after is not None:
        parts.append(np.full(int(30 * fps), float(after)))
    return np.concatenate(parts).copy()


# ── one trial ──────────────────────────────────────────────────────────────


def test_step_up_during_odor_is_positive_in_both_measures():
    r = abd.trial_auc(_trace(10.0, 30.0), fps=FPS, odor_on_s=30.0, odor_off_s=60.0)
    assert r["baseline"] == pytest.approx(10.0)
    assert r["before_signed"] == pytest.approx(0.0)
    assert r["before_unsigned"] == pytest.approx(0.0)
    # 20 % above baseline for 30 s.
    assert r["during_signed"] == pytest.approx(600.0)
    assert r["during_unsigned"] == pytest.approx(600.0)
    assert r["delta_signed"] == pytest.approx(600.0)
    assert r["delta_unsigned"] == pytest.approx(600.0)


def test_dip_during_odor_is_negative_signed_but_positive_unsigned():
    r = abd.trial_auc(_trace(30.0, 10.0), fps=FPS, odor_on_s=30.0, odor_off_s=60.0)
    assert r["during_signed"] == pytest.approx(-600.0)
    assert r["during_unsigned"] == pytest.approx(600.0)
    assert r["delta_signed"] == pytest.approx(-600.0)
    assert r["delta_unsigned"] == pytest.approx(600.0)


def test_symmetric_wobble_cancels_signed_but_not_unsigned():
    n = int(30 * FPS)
    during = 20.0 + np.where(np.arange(n) % 2 == 0, 5.0, -5.0)
    r = abd.trial_auc(_trace(20.0, during), fps=FPS, odor_on_s=30.0, odor_off_s=60.0)
    assert r["during_signed"] == pytest.approx(0.0, abs=1e-9)
    assert r["during_unsigned"] == pytest.approx(5.0 * 30.0)


def test_baseline_is_the_pre_odor_median_so_a_spontaneous_per_shows_before():
    # A 2 s spontaneous extension to 80 % in an otherwise flat 20 % baseline:
    # the median ignores it, so it is counted as before-odor area.
    trace = _trace(20.0, 20.0)
    trace[400:480] = 80.0
    r = abd.trial_auc(trace, fps=FPS, odor_on_s=30.0, odor_off_s=60.0)
    assert r["baseline"] == pytest.approx(20.0)
    assert r["before_signed"] == pytest.approx(60.0 * 2.0)
    assert r["before_unsigned"] == pytest.approx(60.0 * 2.0)
    assert r["delta_signed"] == pytest.approx(-120.0)


def test_signed_delta_does_not_depend_on_the_baseline():
    # Equal-length windows: (during - b) - (before - b) integrates to the raw
    # difference whatever b is.
    rng = np.random.default_rng(0)
    trace = rng.uniform(0, 100, size=int(90 * FPS))
    r = abd.trial_auc(trace, fps=FPS, odor_on_s=30.0, odor_off_s=60.0)
    n = int(30 * FPS)
    raw = (trace[n:2 * n].mean() - trace[:n].mean()) * 30.0
    assert r["delta_signed"] == pytest.approx(raw)


def test_before_window_matches_the_during_window_length():
    # Odor on at 40 s for 20 s: before is [20, 40), not [0, 40).
    trace = np.full(int(90 * FPS), 10.0)
    trace[: int(20 * FPS)] = 99.0           # outside the before window
    trace[int(40 * FPS): int(60 * FPS)] = 30.0
    r = abd.trial_auc(trace, fps=FPS, odor_on_s=40.0, odor_off_s=60.0)
    assert r["baseline"] == pytest.approx(10.0)
    assert r["before_unsigned"] == pytest.approx(0.0)
    assert r["during_signed"] == pytest.approx(20.0 * 20.0)


def test_uses_the_trials_own_odor_timing():
    trace = _trace(10.0, 30.0, on_s=29.9, off_s=59.9)
    r = abd.trial_auc(trace, fps=FPS, odor_on_s=29.9, odor_off_s=59.9)
    assert r["during_signed"] == pytest.approx(600.0)


def test_nan_frames_are_skipped_and_the_area_keeps_the_window_length():
    trace = _trace(10.0, 30.0)
    trace[int(30 * FPS): int(30 * FPS) + 100] = np.nan    # 2.5 s of dropout
    r = abd.trial_auc(trace, fps=FPS, odor_on_s=30.0, odor_off_s=60.0)
    assert r["during_signed"] == pytest.approx(600.0)


def test_a_mostly_missing_window_is_not_scored():
    trace = _trace(10.0, 30.0)
    trace[int(30 * FPS): int(50 * FPS)] = np.nan          # 2/3 of the window
    r = abd.trial_auc(trace, fps=FPS, odor_on_s=30.0, odor_off_s=60.0)
    assert np.isnan(r["during_signed"])
    assert np.isnan(r["delta_signed"])
    assert np.isnan(r["delta_unsigned"])


def test_trace_shorter_than_odor_off_is_not_scored():
    r = abd.trial_auc(np.full(int(45 * FPS), 10.0), fps=FPS,
                      odor_on_s=30.0, odor_off_s=60.0)
    assert np.isnan(r["during_signed"])


# ── during / before ratios ─────────────────────────────────────────────────


def test_raw_ratio_is_the_raster_csvs_auc_ratio():
    # Area under the raw trace, no baseline: 30 % x 30 s over 10 % x 30 s.
    r = abd.trial_auc(_trace(10.0, 30.0), fps=FPS, odor_on_s=30.0, odor_off_s=60.0)
    assert r["auc_before_raw"] == pytest.approx(300.0)
    assert r["auc_during_raw"] == pytest.approx(900.0)
    assert r["ratio_raw"] == pytest.approx(3.0)


def test_raw_ratio_uses_the_raster_zero_substitute():
    r = abd.trial_auc(_trace(0.0, 5.0), fps=FPS, odor_on_s=30.0, odor_off_s=60.0)
    assert r["ratio_raw"] == pytest.approx(150.0 / 1.0)


def test_unsigned_ratio_is_movement_during_over_movement_before():
    n = int(30 * FPS)
    alt = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    r = abd.trial_auc(_trace(20.0 + 2 * alt, 20.0 + 6 * alt), fps=FPS,
                      odor_on_s=30.0, odor_off_s=60.0)
    assert r["before_unsigned"] == pytest.approx(60.0)
    assert r["during_unsigned"] == pytest.approx(180.0)
    assert r["ratio_unsigned"] == pytest.approx(3.0)


# ── peak z-score ───────────────────────────────────────────────────────────


def _alt(n_s, amp, centre=20.0):
    """centre +/- amp alternating every frame: mean centre, SD ~amp."""
    n = int(n_s * FPS)
    return centre + amp * np.where(np.arange(n) % 2 == 0, 1.0, -1.0)


def _z(**kw):
    return abd.trial_peak_z(fps=FPS, odor_on_s=30.0, odor_off_s=60.0, **kw)


def test_peak_z_is_in_sds_of_the_pre_odor_window():
    r = _z(trace=_trace(_alt(30, 2.0), 30.0))
    assert r["z_baseline_mean"] == pytest.approx(20.0)
    assert r["z_baseline_sd"] == pytest.approx(2.0, rel=1e-3)
    assert r["zpeak_during_signed"] == pytest.approx(5.0, rel=1e-3)   # +10 / 2
    assert r["zpeak_during_unsigned"] == pytest.approx(5.0, rel=1e-3)
    # The before window's own peak is its noise ceiling: 1 SD here.
    assert r["zpeak_before_unsigned"] == pytest.approx(1.0, rel=1e-3)
    assert r["zpeak_delta_unsigned"] == pytest.approx(4.0, rel=1e-3)
    assert r["zpeak_ratio_unsigned"] == pytest.approx(5.0, rel=1e-3)


def test_peak_z_dip_is_negative_signed_positive_unsigned():
    r = _z(trace=_trace(_alt(30, 2.0), 10.0))
    assert r["zpeak_during_signed"] == pytest.approx(-5.0, rel=1e-3)
    assert r["zpeak_during_unsigned"] == pytest.approx(5.0, rel=1e-3)


def test_signed_ratio_keeps_the_odor_peaks_sign_over_the_baseline_peaks_size():
    # Signed during peak / |before peak|: the denominator is never negative,
    # so a dip reads negative and a rise positive, in baseline-ceiling units.
    rise = _z(trace=_trace(_alt(30, 2.0), 30.0))
    dip = _z(trace=_trace(_alt(30, 2.0), 10.0))
    assert rise["zpeak_ratio_signed"] == pytest.approx(5.0, rel=1e-3)
    assert dip["zpeak_ratio_signed"] == pytest.approx(-5.0, rel=1e-3)
    # No fixed no-change value: sham windows read about +0.23..+0.34 with no odor.
    assert abd.null_value("zpeak_ratio_signed") is None
    assert "zpeak_ratio_signed" in abd.PEAKZ.measures


def test_signed_peak_z_keeps_the_sign_of_the_larger_excursion():
    during = np.full(int(30 * FPS), 20.0)
    during[100:140] = 26.0     # +3 SD
    during[600:640] = 8.0      # -6 SD: wins
    r = _z(trace=_trace(_alt(30, 2.0), during))
    assert r["zpeak_during_signed"] == pytest.approx(-6.0, rel=1e-3)
    assert r["zpeak_during_unsigned"] == pytest.approx(6.0, rel=1e-3)


def test_peak_z_is_not_scored_on_a_flat_baseline():
    r = _z(trace=_trace(20.0, 30.0))
    assert np.isnan(r["zpeak_during_signed"])
    assert np.isnan(r["zpeak_ratio_unsigned"])


def test_peak_z_uses_the_same_before_window_as_the_auc():
    trace = np.full(int(90 * FPS), 20.0)
    trace[: int(20 * FPS)] = 99.0                          # outside before
    trace[int(20 * FPS): int(40 * FPS)] = _alt(20, 2.0)
    trace[int(40 * FPS): int(60 * FPS)] = 30.0
    r = abd.trial_peak_z(trace, fps=FPS, odor_on_s=40.0, odor_off_s=60.0)
    assert r["z_baseline_mean"] == pytest.approx(20.0)
    assert r["zpeak_during_signed"] == pytest.approx(5.0, rel=1e-3)


def test_main_peakz_writes_its_own_figures_and_tables(tables, tmp_path):
    pre, post = tables
    for df in (pre, post):         # give every baseline some noise to scale by
        cols = [f"dir_val_{i}" for i in range(int(30 * FPS))]
        df[cols] = df[cols].to_numpy() + np.where(np.arange(len(cols)) % 2 == 0, 1.0, -1.0)
    pre.to_parquet(tmp_path / "pre.parquet", index=False)
    post.to_parquet(tmp_path / "post.parquet", index=False)
    out = tmp_path / "out"
    assert abd.main(["--pretest-wide-csv", str(tmp_path / "pre.parquet"),
                     "--testing-wide-csv", str(tmp_path / "post.parquet"),
                     "--metric", "peakz", "--out-dir", str(out)]) == 0
    for name in ("peakz_per_trial.csv", "peakz_summary.csv", "peakz_per_fly_change.csv",
                 "PeakZ_before_vs_during_dataset.png",
                 "PeakZ_during_dataset.png",
                 "PeakZ_during_minus_before_dataset.png",
                 "PeakZ_ratio_during_over_before_dataset.png",
                 "PeakZ_during_minus_before_per_fly.png"):
        assert (out / name).is_file(), name
    per_trial = pd.read_csv(out / "peakz_per_trial.csv")
    assert "zpeak_delta_signed" in per_trial.columns
    assert "delta_signed" not in per_trial.columns       # AUC columns stay out


# ── thresholded AUC / the scorer's AUC-During-Before-LogRatio ─────────────


def test_log_ratio_is_the_scorers_auc_during_before_logratio(tables):
    pre, post = tables
    pre = pre.assign(**{"AUC-Before": 0.0, "AUC-During": 80.0})
    post = post.assign(**{"AUC-Before": 5.0, "AUC-During": 5.0})
    t = abd.build_auc_table(pre, post, auc_eps=8.0)
    r = t.loc[t["phase"] == "Pre-test"].iloc[0]
    assert r["thr_auc_before"] == 0.0
    assert r["thr_auc_during"] == 80.0
    assert r["thr_auc_delta"] == pytest.approx(80.0)
    # log((AUC-During + eps) / (AUC-Before + eps)), as xgb_ordinal.auc_log_ratios
    assert r["thr_auc_log_ratio"] == pytest.approx(np.log(88.0 / 8.0))
    post_rows = t.loc[t["phase"] == "Post-training", "thr_auc_log_ratio"]
    assert np.allclose(post_rows, 0.0)


def test_log_ratio_needs_eps_and_the_threshold_columns(tables):
    pre, post = (df.assign(**{"AUC-Before": 1.0, "AUC-During": 9.0}) for df in tables)
    assert abd.build_auc_table(pre, post)["thr_auc_log_ratio"].isna().all()
    no_cols = abd.build_auc_table(*tables, auc_eps=8.0)
    assert no_cols["thr_auc_before"].isna().all()


def test_null_line_is_one_for_ratios_and_zero_for_the_log_ratio():
    assert abd.null_value("ratio_raw") == 1.0
    assert abd.null_value("thr_auc_log_ratio") == 0.0
    assert abd.null_value("delta_signed") == 0.0


def test_eps_is_read_from_the_model_the_config_points_at(tmp_path):
    model = tmp_path / "model.json"
    model.write_text('{"learner": {"attributes": {"fbr_auc_eps": "8.5"}}}')
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(f"reaction_prediction:\n  model_path: {model}\n")
    assert abd.model_auc_eps(cfg) == pytest.approx(8.5)


def test_main_logratio_writes_its_own_figures(tables, tmp_path):
    pre, post = (df.assign(**{"AUC-Before": 1.0, "AUC-During": 9.0}) for df in tables)
    pre.to_parquet(tmp_path / "pre.parquet", index=False)
    post.to_parquet(tmp_path / "post.parquet", index=False)
    out = tmp_path / "out"
    args = ["--pretest-wide-csv", str(tmp_path / "pre.parquet"),
            "--testing-wide-csv", str(tmp_path / "post.parquet"),
            "--metric", "logratio", "--out-dir", str(out)]
    with pytest.raises(SystemExit):                 # no eps anywhere: refuse
        abd.main(args)
    assert abd.main(args + ["--auc-eps", "1.0"]) == 0
    for name in ("logratio_per_trial.csv", "logratio_summary.csv",
                 "AUCLogRatio_before_vs_during_dataset.png",
                 "AUCLogRatio_during_dataset.png",
                 "AUCLogRatio_during_minus_before_dataset.png",
                 "AUCLogRatio_ratio_during_over_before_dataset.png",
                 "AUCLogRatio_log_ratio_per_fly.png"):
        assert (out / name).is_file(), name
    per_trial = pd.read_csv(out / "logratio_per_trial.csv")
    assert per_trial["thr_auc_log_ratio"].dropna().round(6).eq(round(np.log(10 / 2), 6)).all()


# ── table and summary ─────────────────────────────────────────────────────


def _wide(rows, fps=FPS):
    """Wide rows whose trace is ``before`` for 30 s then ``during`` for 30 s."""
    recs = []
    for ds, fly, num, label, before, during in rows:
        trace = _trace(before, during, after=before)
        rec = {"dataset": ds, "fly": fly, "fly_number": num,
               "trial_type": label.split("_")[0], "trial_label": label,
               "fps": fps, "trace_len": trace.size,
               "trial_odor_on_s": 30.0, "trial_odor_off_s": 60.0}
        rec.update({f"dir_val_{i}": v for i, v in enumerate(trace)})
        recs.append(rec)
    return pd.DataFrame.from_records(recs)


DS = "IAA-Sensitivity-1-1"


@pytest.fixture
def tables():
    pre = _wide([
        (DS, "oct_09_batch_1", 1, "pretest_1_hexanol", 10, 20),
        (DS, "oct_09_batch_1", 1, "pretest_2_isoamylacetate", 10, 10),
        (DS, "oct_09_batch_1", 2, "pretest_1_hexanol", 10, 30),
        (DS, "oct_09_batch_1", 2, "pretest_2_isoamylacetate", 10, 0),
    ])
    post = _wide([
        (DS, "oct_09_batch_1", 1, "testing_1_isoamylacetate", 10, 40),
        (DS, "oct_09_batch_1", 1, "testing_2_hexanol", 10, 20),
        (DS, "oct_09_batch_1", 1, "testing_9_lightonly", 10, 90),
        (DS, "oct_09_batch_1", 2, "testing_1_isoamylacetate", 10, 30),
        (DS, "oct_09_batch_1", 2, "testing_2_hexanol", 10, 10),
    ])
    return pre, post


def test_table_has_one_row_per_fly_odor_phase(tables):
    t = abd.build_auc_table(*tables)
    assert len(t) == 8                                # light-only trial dropped
    assert set(t["phase"]) == {"Pre-test", "Post-training"}
    hex_pre_fly2 = t.loc[(t["phase"] == "Pre-test") & (t["odor"] == "Hexanol")
                         & (t["fly_number"] == "2")].iloc[0]
    assert hex_pre_fly2["delta_signed"] == pytest.approx(600.0)
    iaa = t.loc[t["odor"] == "Isoamyl Acetate"]
    assert iaa["trained"].all()
    assert not t.loc[t["odor"] == "Hexanol", "trained"].any()


def test_dataset_mean_pools_trials_not_odor_means(tables):
    t = abd.build_auc_table(*tables)
    # Drop one Hexanol trial so the odors are unbalanced (1 vs 2 trials).
    t = t.drop(t.index[(t["phase"] == "Pre-test") & (t["odor"] == "Hexanol")
                       & (t["fly_number"] == "2")])
    s = abd.summarise(t)
    row = s.loc[(s["phase"] == "Pre-test") & (s["odor"] == abd.ALL_ODORS)
                & (s["measure"] == "delta_signed")].iloc[0]
    pre = t.loc[t["phase"] == "Pre-test", "delta_signed"]
    assert row["n_trials"] == 3
    assert row["mean"] == pytest.approx(pre.mean())   # (300 + 0 - 300) / 3
    assert row["mean"] != pytest.approx(
        pre.groupby(t["odor"]).mean().mean())


def test_paired_change_pairs_each_fly_with_itself(tables):
    t = abd.build_auc_table(*tables)
    paired = abd.paired_change(t, "delta_signed")
    iaa = paired.loc[paired["odor"] == "Isoamyl Acetate"].set_index("fly_number")
    # fly 1: pre 0 -> post 900; fly 2: pre -300 -> post 600.
    assert iaa.loc["1", "Pre-test"] == pytest.approx(0.0)
    assert iaa.loc["1", "Post-training"] == pytest.approx(900.0)
    assert iaa.loc["2", "change"] == pytest.approx(900.0)


# ── AUC vs model score ─────────────────────────────────────────────────────


def _predictions(path, rows):
    pd.DataFrame(rows, columns=["dataset", "fly", "fly_number", "trial_label", "score"]
                 ).to_csv(path, index=False)
    return path


def test_scores_join_on_trial_label_even_with_a_concentration_suffix(tables, tmp_path):
    t = abd.build_auc_table(*tables)
    csv = _predictions(tmp_path / "p.csv", [
        (DS, "oct_09_batch_1", 1, "pretest_1_hexanol", 0),              # exact
        (DS, "oct_09_batch_1", 1, "testing_1_isoamylacetate-1", 4),     # + conc
    ])
    out = abd.attach_scores(t, csv)
    by = out.set_index(["phase", "fly_number", "odor"])["score"]
    assert by[("Pre-test", "1", "Hexanol")] == 0
    assert by[("Post-training", "1", "Isoamyl Acetate")] == 4
    assert by.isna().sum() == len(t) - 2          # unmatched stay missing


def test_score_correlation_is_spearman_per_phase_and_pooled():
    t = pd.DataFrame({
        "phase": ["Pre-test"] * 4 + ["Post-training"] * 4,
        "score": [0, 1, 2, 3, 0, 2, 4, 5],
        # Pre: monotone up. Post: monotone down.
        "delta_signed": [1.0, 5.0, 9.0, 20.0, 50.0, 40.0, 30.0, 10.0],
    })
    c = abd.score_correlations(t, ["delta_signed"]).set_index("phase")
    assert c.loc["Pre-test", "rho"] == pytest.approx(1.0)
    assert c.loc["Post-training", "rho"] == pytest.approx(-1.0)
    assert c.loc[abd.COMBINED, "n_trials"] == 8


def test_main_writes_auc_vs_score_figures(tables, tmp_path):
    pre, post = tables
    pre.to_parquet(tmp_path / "pre.parquet", index=False)
    post.to_parquet(tmp_path / "post.parquet", index=False)
    csv = _predictions(tmp_path / "p.csv", [
        (DS, "oct_09_batch_1", n, label, s)
        for n in (1, 2)
        for label, s in (("pretest_1_hexanol", 0), ("pretest_2_isoamylacetate", 1),
                         ("testing_1_isoamylacetate", 3), ("testing_2_hexanol", 2))
    ])
    out = tmp_path / "out"
    assert abd.main(["--pretest-wide-csv", str(tmp_path / "pre.parquet"),
                     "--testing-wide-csv", str(tmp_path / "post.parquet"),
                     "--predictions-csv", str(csv), "--out-dir", str(out)]) == 0
    for name in ("AUC_vs_score_during_minus_before.png", "AUC_vs_score_during.png",
                 "AUC_vs_score_ratio.png",
                 "auc_vs_score_correlation.csv", "auc_vs_score_summary.csv"):
        assert (out / name).is_file(), name
    assert pd.read_csv(out / "auc_per_trial.csv")["score"].notna().sum() == 8


def test_main_writes_figures_and_tables(tables, tmp_path):
    pre, post = tables
    pre.to_parquet(tmp_path / "pre.parquet", index=False)
    post.to_parquet(tmp_path / "post.parquet", index=False)
    out = tmp_path / "out"
    assert abd.main(["--pretest-wide-csv", str(tmp_path / "pre.parquet"),
                     "--testing-wide-csv", str(tmp_path / "post.parquet"),
                     "--out-dir", str(out)]) == 0
    for name in ("auc_per_trial.csv", "auc_summary.csv", "auc_per_fly_change.csv",
                 "AUC_before_vs_during_dataset.png",
                 "AUC_during_minus_before_dataset.png",
                 "AUC_ratio_during_over_before_dataset.png",
                 "AUC_during_minus_before_per_fly.png"):
        assert (out / name).is_file(), name


# ── wording: sensitivity cohorts vs conditioning cohorts ───────────────────


def _drawn_text(fig) -> str:
    """Every string on a figure: titles, ticks, legends, footnotes."""
    return "\n".join(a.get_text() for a in fig.findobj(Text))


def _figures(t):
    """The figures that name the phases; the first three also tag the trained odor."""
    t = t.assign(score=pd.array([i % 3 for i in range(len(t))], dtype="Int64"))
    return [abd.plot_before_vs_during(t, "title"),
            abd.plot_paired_dataset(t, abd.AUC.rows("delta"), "title"),
            abd.plot_per_fly_matrix(t, "title"),
            abd.plot_auc_vs_score(t, abd.AUC.rows("delta"), "title")]


def test_sensitivity_figures_say_pre_post_sensitization_and_sensitized_odor(tables):
    t = abd.build_auc_table(*tables)                      # IAA-Sensitivity-1-1
    for i, fig in enumerate(_figures(t)):
        text = _drawn_text(fig)
        plt.close(fig)
        assert "Pre-exposure" in text and "Post-exposure" in text
        assert "pre-test" not in text.lower() and "post-training" not in text.lower()
        assert "CS+" not in text
        if i < 3:
            assert "(sensitized odor)" in text
    assert set(t["phase"]) == set(abd.PHASES)             # table values stay as they were


def test_conditioning_cohorts_keep_pre_test_post_training_and_cs_plus(tables):
    t = abd.build_auc_table(*tables).assign(dataset="Hex-HexTest-Training-32-1")
    for i, fig in enumerate(_figures(t)):
        text = _drawn_text(fig)
        plt.close(fig)
        assert "Pre-test" in text and "Post-training" in text
        assert "sensitiz" not in text.lower() and "exposure" not in text.lower()
        if i < 3:
            assert "(CS+)" in text


def test_a_mix_of_cohorts_keeps_the_conditioning_words(tables):
    t = abd.build_auc_table(*tables)
    t.loc[t["fly_number"] == "2", "dataset"] = "Hex-HexTest-Training-32-1"
    fig = abd.plot_paired_dataset(t, abd.AUC.rows("delta"), "title")
    text = _drawn_text(fig)
    plt.close(fig)
    assert "Pre-test" in text and "(CS+)" in text and "sensitiz" not in text.lower() and "exposure" not in text.lower()


@pytest.mark.parametrize("dataset, phases", [
    (DS, "pre-exposure vs post-exposure"),
    ("Hex-HexTest-Training-32-1", "pre-test vs post-training"),
])
def test_main_titles_name_the_phases_by_cohort(tables, tmp_path, monkeypatch, dataset, phases):
    titles = []

    def keep_title(fig, path):
        titles.append(fig._suptitle.get_text())
        plt.close(fig)

    monkeypatch.setattr(abd, "_save", keep_title)
    for name, df in zip(("pre", "post"), tables):
        df.assign(dataset=dataset).to_parquet(tmp_path / f"{name}.parquet", index=False)
    assert abd.main(["--pretest-wide-csv", str(tmp_path / "pre.parquet"),
                     "--testing-wide-csv", str(tmp_path / "post.parquet"),
                     "--out-dir", str(tmp_path / "out")]) == 0
    assert len(titles) == 5
    assert all(s.endswith(f", {phases} (2 flies)") for s in titles), titles


# ── no fixed null for the five biased measures; exact Wilcoxon; Holm ──────

NO_NULL = ("delta_unsigned", "ratio_unsigned", "zpeak_delta_unsigned",
           "zpeak_ratio_unsigned", "zpeak_ratio_signed")
KEEP_NULL = {"delta_signed": 0.0, "ratio_raw": 1.0, "zpeak_delta_signed": 0.0,
             "thr_auc_log_ratio": 0.0}


@pytest.mark.parametrize("measure", NO_NULL)
def test_biased_measures_have_no_fixed_null(measure):
    assert abd.null_value(measure) is None
    metric = next(m for m in abd.METRICS.values() if measure in m.measures)
    assert measure not in metric.tested
    rows = [r for r in (*metric.rows("delta"), *metric.ratios) if r.measure == measure]
    assert rows and all(r.null is None for r in rows)


@pytest.mark.parametrize("measure,null", KEEP_NULL.items())
def test_unbiased_measures_keep_their_null_and_test(measure, null):
    assert abd.null_value(measure) == null
    metric = next(m for m in abd.METRICS.values() if measure in m.measures)
    assert measure in metric.tested


def test_summary_has_no_p_vs_null_for_the_biased_measures(tables):
    t = abd.build_auc_table(*tables)
    s = abd.summarise(t)
    for m in NO_NULL[:2]:
        assert s.loc[s["measure"] == m, "p_vs_null"].isna().all()
    assert "p_pre_vs_post" in s.columns


def test_wilcoxon_is_exact_with_ties():
    d = np.array([-2, -2, -2, -2, -1, -1, 1], float)
    assert abd._wilcoxon(d) == pytest.approx(0.0625)
    # paired form: x - y
    assert abd._wilcoxon(d + 5, np.full(7, 5.0)) == pytest.approx(0.0625)
    assert np.isnan(abd._wilcoxon(np.zeros(4)))
    assert np.isnan(abd._wilcoxon(np.array([np.nan])))


def test_summary_holm_columns_and_all_odors_per_fly(tables):
    t = abd.build_auc_table(*tables)
    s = abd.summarise(t)
    assert "p_pre_vs_post_holm" in s.columns
    from fbpipe.analysis import fly_stats as fs
    one = s.loc[(s["measure"] == "delta_signed") & (s["phase"] == "Pre-test")
                & (s["odor"] != abd.ALL_ODORS)]
    raw = one["p_pre_vs_post"].to_numpy(float)
    np.testing.assert_allclose(one["p_pre_vs_post_holm"].to_numpy(float),
                               fs.holm_adjust(raw), equal_nan=True)
    allrow = s.loc[(s["odor"] == abd.ALL_ODORS) & (s["measure"] == "delta_signed")]
    assert allrow["p_pre_vs_post_holm"].isna().all()


def test_all_odors_test_uses_one_value_per_fly():
    rows = []
    # 6 flies x 7 odors, every fly x odor pair shifts by +1 but ONE fly-level
    # change per fly: the test must see 6 values, not 42.
    for fl in range(6):
        for o in range(7):
            for ph, base in (("Pre-test", 0.0), ("Post-training", 1.0 + 0.1 * fl)):
                rows.append(dict(dataset="D", fly="b", fly_number=str(fl), odor=f"o{o}",
                                 phase=ph, trained=o == 0, delta_signed=base))
    t = pd.DataFrame(rows)
    p = abd.all_odors_pre_vs_post_p(t, "delta_signed")
    # six positive fly changes: exact two-sided p = 2/2**6
    assert p == pytest.approx(2 / 64)


def test_paired_plot_captions_name_the_sham_bias_and_holm(tables):
    t = abd.build_auc_table(*tables)
    rows = (abd.AUC.rows("delta")[1],)
    fig = abd.plot_paired_dataset(t, rows, "x", metric=abd.AUC)
    text = " ".join(x.get_text() for x in fig.findobj(Text))
    assert "No fixed no-change value" in text
    assert "Holm-adjusted across" in text and "exploratory" in text
    # no horizontal reference line is drawn for the biased measure
    ax = fig.axes[0]
    assert not any(list(l.get_xdata()) == [0, 1] and set(l.get_ydata()) == {0.0}
                   for l in ax.lines)      # axhline spans axes coords 0..1
    plt.close(fig)


# ── review: pooled unit, pooled error bars, Holm on the figure, lines ──────


def _fly_rows(values, measure="delta_signed"):
    """``values``: odor -> [(pre, post), ...] one pair per fly (None = trial missing)."""
    rows = []
    for odor, pairs in values.items():
        for fl, pair in enumerate(pairs):
            for phase, v in zip(abd.PHASES, pair):
                if v is not None:
                    rows.append(dict(dataset=DS, fly="oct_09_batch_1", fly_number=str(fl + 1),
                                     odor=odor, phase=phase, trained=odor == "Hexanol",
                                     **{measure: float(v)}))
    return pd.DataFrame(rows)


def _hlines(ax):
    """The axhline reference lines of an axes (x spans axes coords 0..1)."""
    return [l for l in ax.lines if list(l.get_xdata()) == [0, 1]]


def test_all_odors_test_uses_only_odors_a_fly_has_in_both_phases():
    # Every fly rises on Hexanol. Flies 1-3 also gave a large Citral pre-test trial
    # but none after training: pooling each phase separately would leave that 10 in
    # the pre mean and turn their change negative. Paired per odor first, every
    # fly's change is positive -> exact two-sided p = 2 / 2**6.
    t = _fly_rows({
        "Hexanol": [(0.0, 1.0 + 0.1 * f) for f in range(6)],
        "Citral": [(10.0, None)] * 3 + [(0.0, 0.5)] * 3,
    })
    assert abd.all_odors_pre_vs_post_p(t, "delta_signed") == pytest.approx(2 / 64)


def test_pooled_sem_counts_flies_not_trials():
    from scipy.stats import t as student_t

    from fbpipe.analysis import fly_stats as fs
    rng = np.random.default_rng(3)
    levels = rng.normal(0.0, 10.0, size=5)            # each fly's own level, all odors
    t = _fly_rows({f"o{o}": [(lv + rng.normal(), lv + rng.normal()) for lv in levels]
                   for o in range(4)})
    for m in abd.AUC.measures:
        if m not in t:
            t[m] = t["delta_signed"]
    s = abd.summarise(t).set_index(["phase", "odor", "measure"])
    pre = t.loc[t["phase"] == "Pre-test"]
    _, lo, hi = fs.fly_cluster_mean_ci(pre["delta_signed"], pre["fly_number"])
    want = (hi - lo) / 2 / student_t.ppf(0.975, 5 - 1)
    got = s.loc[("Pre-test", abd.ALL_ODORS, "delta_signed"), "sem"]
    assert got == pytest.approx(want)
    trial_sem = pre["delta_signed"].std(ddof=1) / np.sqrt(len(pre))
    assert got > 1.5 * trial_sem                       # 20 trials are not 20 flies
    one = pre.loc[pre["odor"] == "o1", "delta_signed"]  # one trial per fly: plain SEM
    assert s.loc[("Pre-test", "o1", "delta_signed"), "sem"] == pytest.approx(
        one.std(ddof=1) / np.sqrt(len(one)))
    # the figure's pooled error bar is that SEM too
    fig = abd.plot_paired_dataset(t, (abd.Row("delta_signed", "y", "t"),), "x")
    x_pre = abd._group_x(abd._odor_order(t) + [abd.ALL_ODORS])[-1] - 0.2
    halves = [abs(seg[1][1] - seg[0][1]) / 2 for c in fig.axes[0].containers
              if hasattr(c, "has_yerr") and c.has_yerr
              for seg in c.lines[2][0].get_segments() if abs(seg[0][0] - x_pre) < 1e-9]
    assert halves == [pytest.approx(want)]
    plt.close(fig)


def test_paired_plot_prints_the_holm_p_and_takes_its_stars_from_it():
    # 6 flies all up on Hexanol: raw p = 2/64 = 0.031, Holm over 3 odors = 0.094.
    t = _fly_rows({"Hexanol": [(0.0, c) for c in (1, 2, 3, 4, 5, 6)],
                   "Citral": [(0.0, c) for c in (1, -1, 2, -2, 3, -3)],
                   "Linalool": [(0.0, c) for c in (1, -2, 3, -4, 5, -6)]})
    fig = abd.plot_paired_dataset(t, (abd.Row("delta_signed", "y", "t"),), "x")
    labels = [x.get_text() for x in fig.axes[0].texts]
    plt.close(fig)
    assert "p=0.094" in labels and not any("*" in s for s in labels)
    # 8 flies all up: raw 2/256 = 0.0078 -> Holm x3 = 0.023, one star
    t = _fly_rows({"Hexanol": [(0.0, c) for c in range(1, 9)],
                   "Citral": [(0.0, c) for c in (1, -1, 2, -2, 3, -3, 4, -4)],
                   "Linalool": [(0.0, c) for c in (1, -2, 3, -4, 5, -6, 7, -8)]})
    fig = abd.plot_paired_dataset(t, (abd.Row("delta_signed", "y", "t"),), "x")
    labels = [x.get_text() for x in fig.axes[0].texts]
    plt.close(fig)
    assert "p=0.023 *" in labels


def test_only_the_measures_with_a_fixed_null_get_a_reference_line(tables):
    t = abd.build_auc_table(*tables)
    fig = abd.plot_paired_dataset(t, abd.AUC.ratios, "x", metric=abd.AUC)
    raw_ax, unsigned_ax = fig.axes[:2]
    assert [set(l.get_ydata()) for l in _hlines(raw_ax)] == [{1.0}]
    assert _hlines(unsigned_ax) == []
    assert {p.get_y() for p in unsigned_ax.patches} == {0.0}    # bars rise from 0, not 1
    plt.close(fig)
    pz = _fly_rows({"Hexanol": [(1.0, 2.0), (0.5, 1.5), (0.2, 0.9)]}, "zpeak_ratio_signed")
    pz["zpeak_ratio_unsigned"] = pz["zpeak_ratio_signed"] + 1.0
    fig = abd.plot_paired_dataset(pz, abd.PEAKZ.ratios, "x", metric=abd.PEAKZ)
    assert all(_hlines(ax) == [] for ax in fig.axes[:2])
    plt.close(fig)


def test_p_vs_null_only_for_the_measures_with_a_fixed_null():
    rng = np.random.default_rng(7)
    cols = set(abd.AUC.measures) | set(abd.PEAKZ.measures)
    t = _fly_rows({o: [(0.0, 0.0)] * 7 for o in ("Hexanol", "Citral")})
    for m in cols:
        t[m] = rng.normal(0.5, 1.0, size=len(t))
    for metric in (abd.AUC, abd.PEAKZ):
        s = abd.summarise(t, metric)
        s = s.loc[s["odor"] != abd.ALL_ODORS]
        for m in metric.measures:
            p = s.loc[s["measure"] == m, "p_vs_null"]
            if m in NO_NULL or m not in metric.deltas + abd.RATIOS + tuple(
                    r.measure for r in metric.ratios):
                assert p.isna().all(), m
            else:
                assert p.notna().all(), m
