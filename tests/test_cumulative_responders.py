"""Cumulative % of flies responding, pre- vs post-exposure, one figure per odor.

A fly counts as responding when the model scores it >= the binary threshold, the
same call (and the same fly-by-odor pairs) as the window's score figures, so every
curve ends on the % responding those figures report. The curve says *when* each
responder started: its first ``MIN_RUN_S`` above its own red line after odor onset.
A responder with no such moment during the odor is counted after odor off, in a
zone of its own, rather than left out.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import scripts.analysis.cumulative_responders as cr
import scripts.analysis.pretest_vs_test_comparison as pvt
import scripts.analysis.start_window_report as swr

REPO = Path(__file__).resolve().parents[1]
DS = "Hex-Sensitivity-1-0.1"          # sensitized odor: Hexanol
FPS = 40.0


# ── when a response starts ────────────────────────────────────────────────


def _binary(n=400, on=()):
    b = np.zeros(n, dtype=bool)
    for a, z in on:
        b[a:z] = True
    return b


def test_first_run_is_timed_from_the_window_start():
    b = _binary(on=[(100, 120)])                      # 20 frames = 0.5 s, from 2.5 s
    assert cr.first_run_s(b, fps=FPS, start_s=1.0, end_s=9.0) == pytest.approx(1.5)


def test_a_blip_shorter_than_the_minimum_run_is_not_a_response():
    b = _binary(on=[(100, 119), (200, 230)])          # 19 frames, then 30 frames
    assert cr.first_run_s(b, fps=FPS, start_s=0.0, end_s=9.0) == pytest.approx(5.0)
    assert np.isnan(cr.first_run_s(_binary(on=[(100, 119)]), fps=FPS, start_s=0.0, end_s=9.0))


def test_only_the_part_of_a_run_inside_the_window_counts():
    b = _binary(on=[(30, 60)])                        # 0.75-1.5 s
    # in progress at the window start, 20 frames left inside: responding from 0 s
    assert cr.first_run_s(b, fps=FPS, start_s=1.0, end_s=9.0) == pytest.approx(0.0)
    # cut by the window end, 10 frames inside: no response
    assert np.isnan(cr.first_run_s(b, fps=FPS, start_s=0.0, end_s=1.0))
    # entirely before the window: no response
    assert np.isnan(cr.first_run_s(b, fps=FPS, start_s=2.0, end_s=9.0))


def test_the_minimum_run_is_a_setting():
    b = _binary(on=[(100, 110)])                      # 10 frames = 0.25 s
    assert np.isnan(cr.first_run_s(b, fps=FPS, start_s=0.0, end_s=9.0))
    assert cr.first_run_s(b, fps=FPS, start_s=0.0, end_s=9.0,
                          min_run_s=0.25) == pytest.approx(2.5)


# ── the curve ─────────────────────────────────────────────────────────────


def test_the_curve_counts_responders_only_and_divides_by_every_fly():
    onsets = np.array([0.0, 2.5, np.nan, 1.0, np.nan])
    responding = np.array([True, True, True, False, False])
    pct = cr.cumulative_percent(onsets, responding, np.array([0.0, 1.0, 2.5, 30.0]))
    # the non-responder that crossed at 1 s never counts; the responder with no
    # onset is not on the timed curve (it is drawn after odor off)
    assert pct.tolist() == pytest.approx([20.0, 20.0, 40.0, 40.0])


# ── pairing: exactly the score figures' fly x odor pairs ──────────────────


def _pred(rows, ds=DS):
    return pd.DataFrame([{"dataset": ds, "fly": f, "fly_number": n, "trial_type": ph,
                          "trial_label": f"{ph}_{i}_{odor}", "score": s}
                         for f, n, ph, i, odor, s in rows])


def _flagged(path, flies=()):
    path.write_text('dataset,fly,fly_number,"FLY-State(1, 0, -1)",comment\n'
                    + "".join(f"{DS},{f},{n},0,dead\n" for f, n in flies))
    return path


def test_pairs_each_fly_and_odor_across_phases_as_the_score_figures_do(tmp_path):
    rows = [
        ("b1", 1, "pretest", 1, "hexanol", 0), ("b1", 1, "testing", 3, "hexanol", 3),
        ("b1", 1, "pretest", 2, "citral", 2), ("b1", 1, "testing", 1, "citral", 1),
        ("b1", 2, "pretest", 1, "hexanol", 1),                 # no post-exposure trial
        ("b1", 1, "testing", 8, "lightonly", 4),               # no pre-exposure trial
        ("b2", 1, "pretest", 1, "hexanol", 4), ("b2", 1, "testing", 2, "hexanol", 4),
    ]
    path = tmp_path / "pred.csv"
    _pred(rows).to_csv(path, index=False)
    flagged = _flagged(tmp_path / "flagged.csv", [("b2", 1)])
    got = cr.paired_trials(path, dataset=DS, flagged_flies_csv=str(flagged))
    cols = ["fly", "fly_number", "odor", "trial_pre", "trial_post", "score_pre", "score_post"]
    assert got[cols].values.tolist() == [["b1", 1, "Citral", 2, 1, 2, 1],
                                         ["b1", 1, "Hexanol", 1, 3, 0, 3]]
    ref = pvt.load_paired_scores(path, flagged_flies_csv=str(flagged))
    assert (sorted(zip(ref["odor"], ref["score_pre"], ref["score_post"]))
            == sorted(zip(got["odor"], got["score_pre"], got["score_post"])))


def test_another_cohorts_rows_are_left_out(tmp_path):
    rows = _pred([("b1", 1, "pretest", 1, "hexanol", 0), ("b1", 1, "testing", 1, "hexanol", 3)])
    other = _pred([("b9", 1, "pretest", 1, "hexanol", 4), ("b9", 1, "testing", 1, "hexanol", 4)],
                  ds="IAA-Sensitivity-1-1")
    path = tmp_path / "pred.csv"
    pd.concat([rows, other]).to_csv(path, index=False)
    assert cr.paired_trials(path, dataset=DS)["fly"].tolist() == ["b1"]


# ── end to end: tables in, figures and numbers out ────────────────────────


THRESHOLD_BLOCK = {"threshold_std_mult": 3.0, "threshold_min_delta": 5.0,
                   "threshold_anchor_s": 5.0, "threshold_noise_block_s": 5.0,
                   "threshold_noise_pctl": 25.0}


def _trace(start_s=None, length_s=2.0, n=3600):
    """Flat at 10 % (so theta = 10 + the 5 % floor), a 60 % extension from ``start_s``."""
    x = np.full(n, 10.0)
    if start_s is not None:
        a = int(round(start_s * FPS))
        x[a:a + int(round(length_s * FPS))] = 60.0
    return x


def _wide(rows):
    recs = []
    for fly, num, phase, i, odor, trace in rows:
        rec = {"dataset": DS, "fly": fly, "fly_number": num, "trial_type": phase,
               "trial_label": f"{phase}_{i}_{odor}", "fps": FPS, "trace_len": len(trace),
               "trial_odor_on_s": 30.0, "trial_odor_off_s": 60.0}
        rec.update({f"dir_val_{k}": v for k, v in enumerate(trace)})
        recs.append(rec)
    return pd.DataFrame(recs)


@pytest.fixture
def cohort(tmp_path):
    """Four flies, Hexanol (the sensitized odor) and Citral, both phases.

    Hexanol pre:  fly 2 responds (score 2, from 2 s); fly 4 crosses at 10 s but
                  scores 1, so it is not responding.
    Hexanol post: fly 1 (from 5 s), fly 2 (from 1 s) and fly 3, whose score is 2
                  but whose trace never clears theta -> counted after odor off.
    Citral: nobody, either phase.
    """
    spec = {  # fly: (pre score, pre onset, post score, post onset)
        1: (0, None, 3, 35.0), 2: (2, 32.0, 4, 31.0), 3: (0, None, 2, None), 4: (1, 40.0, 0, None),
    }
    wide = {"pretest": [], "testing": []}
    pred = []
    for n, (s_pre, t_pre, s_post, t_post) in spec.items():
        for phase, idx, odor, score, onset in (
                ("pretest", 1, "hexanol", s_pre, t_pre), ("testing", 2, "hexanol", s_post, t_post),
                ("pretest", 2, "citral", 0, None), ("testing", 1, "citral", 0, None)):
            wide[phase].append(("b1", n, phase, idx, odor, _trace(onset)))
            pred.append(("b1", n, phase, idx, odor, score))
    paths = {"pretest": tmp_path / "pre.parquet", "testing": tmp_path / "post.parquet",
             "predictions": tmp_path / "pred.csv", "config": tmp_path / "config.yaml"}
    _wide(wide["pretest"]).to_parquet(paths["pretest"], index=False)
    _wide(wide["testing"]).to_parquet(paths["testing"], index=False)
    _pred(pred).to_csv(paths["predictions"], index=False)
    paths["config"].write_text(yaml.safe_dump(
        {"analysis": {"combined": {"combined_base": {"wide": THRESHOLD_BLOCK}}}}))
    return paths


def _main(paths, out, *extra):
    return cr.main(["--pretest-wide-csv", str(paths["pretest"]),
                    "--testing-wide-csv", str(paths["testing"]),
                    "--predictions-csv", str(paths["predictions"]),
                    "--config", str(paths["config"]), "--dataset", DS,
                    "--out-dir", str(out), "--label", "start 9:40am-4:20pm", *extra])


def test_one_figure_per_odor_and_one_with_every_odor(cohort, tmp_path):
    out = tmp_path / "out"
    assert _main(cohort, out) == 0
    assert (out / "Hexanol_cumulative_responders.png").is_file()
    assert (out / "Citral_cumulative_responders.png").is_file()
    assert (out / "All_odors_cumulative_responders.png").is_file()


def test_the_summary_matches_the_score_figures_and_times_each_responder(cohort, tmp_path):
    out = tmp_path / "out"
    _main(cohort, out)
    summary = pd.read_csv(out / "cumulative_responders_summary.csv")
    assert summary["odor"].tolist() == ["Hexanol", "Citral"]        # sensitized odor first
    hex_ = summary.iloc[0]
    assert bool(hex_["sensitized"]) and not bool(summary.iloc[1]["sensitized"])
    assert (hex_["n_flies"], hex_["responding_pre"], hex_["responding_post"]) == (4, 1, 3)
    assert (hex_["pct_pre"], hex_["pct_post"]) == (25.0, 75.0)
    assert (hex_["no_onset_pre"], hex_["no_onset_post"]) == (0, 1)
    assert hex_["median_onset_pre_s"] == pytest.approx(2.0)
    assert hex_["median_onset_post_s"] == pytest.approx(3.0)       # flies 1 (5 s) and 2 (1 s)
    p = pvt.mcnemar_p([False, True, False, False], [True, True, True, False])
    assert hex_["p_mcnemar"] == pytest.approx(p)
    ref = pvt.per_odor_summary(pvt.load_paired_scores(cohort["predictions"]),
                               rate_test="mcnemar").set_index("odor")
    for _, row in summary.iterrows():
        assert row["pct_pre"] == pytest.approx(ref.loc[row["odor"], "pct_pre"])
        assert row["pct_post"] == pytest.approx(ref.loc[row["odor"], "pct_post"])


def test_every_fly_is_listed_with_its_score_and_onset(cohort, tmp_path):
    out = tmp_path / "out"
    _main(cohort, out)
    flies = pd.read_csv(out / "cumulative_responders_flies.csv")
    assert len(flies) == 4 * 2 * 2                                  # flies x odors x phases
    row = flies[(flies["odor"] == "Hexanol") & (flies["fly_number"] == 4)
                & (flies["phase"] == "Pre-exposure")].iloc[0]
    # crossed at 10 s, but the model did not call it a response
    assert row["onset_s"] == pytest.approx(10.0) and not bool(row["responding"])


def test_nothing_paired_is_nothing_to_draw(cohort, tmp_path):
    pred = pd.read_csv(cohort["predictions"])
    pred[pred["trial_type"] == "testing"].to_csv(cohort["predictions"], index=False)
    out = tmp_path / "out"
    assert _main(cohort, out) == 0
    assert not list(out.glob("*.png"))


# ── wired into the window figure set ──────────────────────────────────────


def _arg(cmd, flag):
    return cmd[cmd.index(flag) + 1]


def test_the_window_set_draws_it_with_the_score_figures_threshold(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(swr, "_run", lambda cmd, log: seen.setdefault(Path(str(cmd[1])).name, cmd)
                        and 0)
    p = {k: tmp_path / k for k in ("predictions", "testing", "pretest", "training")}
    flagged = tmp_path / "flagged.csv"
    swr.run_figures(tmp_path, p, DS, config=tmp_path / "c.yaml", label="start x", log=None,
                    flagged=flagged)
    cmd = seen["cumulative_responders.py"]
    score = seen["pretest_vs_test_comparison.py"]
    assert _arg(cmd, "--binary-threshold") == _arg(score, "--binary-threshold")
    assert _arg(cmd, "--out-dir") == tmp_path / cr.FOLDER
    for flag, value in (("--pretest-wide-csv", p["pretest"]), ("--testing-wide-csv", p["testing"]),
                        ("--predictions-csv", p["predictions"]), ("--flagged-flies-csv", flagged),
                        ("--dataset", DS), ("--config", tmp_path / "c.yaml"),
                        ("--label", "start x")):
        assert _arg(cmd, flag) == value


def test_editing_it_redraws_the_window_set():
    from fbpipe.utils import figure_cache as fc

    files = fc.code_files([REPO / "scripts" / "analysis" / "sensitivity_window_results.py"])
    assert REPO / "scripts" / "analysis" / "cumulative_responders.py" in files


# ── area in each half of the odor (for the cross-cohort halves figure) ───


def test_the_two_halves_split_the_auc_figures_during_area():
    from scripts.analysis.auc_before_during import trial_auc

    rng = np.random.default_rng(0)
    x = 10 + rng.normal(0, 1, 3600)
    x[int(50 * FPS):int(52 * FPS)] += 50                 # 50 % for 2 s, 20 s into the odor
    first, second = cr.half_aucs(x, fps=FPS, odor_on_s=30.0, odor_off_s=60.0)
    during = trial_auc(x, fps=FPS, odor_on_s=30.0, odor_off_s=60.0)["during_signed"]
    assert first + second == pytest.approx(during)
    assert second == pytest.approx(100.0, abs=5) and abs(first) < 5


def test_every_fly_row_carries_its_area_in_each_half(cohort, tmp_path):
    out = tmp_path / "out"
    _main(cohort, out)
    flies = pd.read_csv(out / "cumulative_responders_flies.csv")
    row = flies[(flies["odor"] == "Hexanol") & (flies["fly_number"] == 1)
                & (flies["phase"] == "Post-exposure")].iloc[0]
    # 60 % from 35 s for 2 s over a flat 10 %: 50 % x 2 s, all of it in the first 15 s
    assert row["auc_first_half"] == pytest.approx(100.0)
    assert row["auc_second_half"] == pytest.approx(0.0)


def test_each_half_gets_the_ratio_and_peak_z_measures_of_the_cohort_figures():
    """User 2026-11-11: AUC as during / before, and peak z (during / before and during
    alone), in each half of the odor, as each cohort's AUC and PeakZ figures define them."""
    from scripts.analysis.auc_before_during import trial_auc, trial_peak_z

    rng = np.random.default_rng(1)
    x = 10 + rng.normal(0, 1, 3600)
    x[int(50 * FPS):int(52 * FPS)] += 50                  # 20 s into the odor
    m = cr.half_measures(x, fps=FPS, odor_on_s=30.0, odor_off_s=60.0)
    whole = trial_peak_z(x, fps=FPS, odor_on_s=30.0, odor_off_s=60.0)
    # the odor window's peak is in the second half, so there the half's peak is the window's
    assert m["zpeak_second_half"] == pytest.approx(whole["zpeak_during_signed"])
    assert m["zpeak_ratio_second_half"] == pytest.approx(whole["zpeak_ratio_signed"])
    assert m["zpeak_unsigned_ratio_second_half"] == pytest.approx(whole["zpeak_ratio_unsigned"])
    assert abs(m["zpeak_first_half"]) < 5 < m["zpeak_second_half"]
    # raw-area ratio per second: the half's mean level / the mean level before the odor
    for half, (lo, hi) in (("first", (1200, 1800)), ("second", (1800, 2400))):
        assert m[f"auc_ratio_{half}_half"] == pytest.approx(x[lo:hi].mean() / x[:1200].mean())
    # so the two halves average to the AUC figure's own during / before raw ratio
    raw = trial_auc(x, fps=FPS, odor_on_s=30.0, odor_off_s=60.0)["ratio_raw"]
    assert (m["auc_ratio_first_half"] + m["auc_ratio_second_half"]) / 2 == pytest.approx(raw)
    # and the signed areas are still there
    assert (m["auc_first_half"], m["auc_second_half"]) == pytest.approx(
        cr.half_aucs(x, fps=FPS, odor_on_s=30.0, odor_off_s=60.0))
