"""Mean PER trace of the flies scored as responding vs not, pre- vs post-exposure.

The two groups are the score figures' own (same fly x odor pairs, same threshold),
and each trace is the mean-trace figures' own (each trial minus its first 30 s,
averaged over flies), so the two panels are the Mean-Traces curve, split.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scripts.analysis.responder_mean_traces as rmt
import scripts.analysis.start_window_report as swr
from scripts.analysis.dataset_means_specific_flies import _mean_sem_trace

REPO = Path(__file__).resolve().parents[1]
DS = "Hex-Sensitivity-1-0.1"          # sensitized odor: Hexanol
FPS = 40.0
N = 3600                              # 90 s


def _trace(rest=10.0, during=None):
    """``rest`` everywhere, ``during`` from 30 to 60 s."""
    x = np.full(N, float(rest))
    if during is not None:
        x[int(30 * FPS):int(60 * FPS)] = float(during)
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


def _pred(rows):
    return pd.DataFrame([{"dataset": DS, "fly": f, "fly_number": n, "trial_type": ph,
                          "trial_label": f"{ph}_{i}_{odor}", "score": s}
                         for f, n, ph, i, odor, s in rows])


@pytest.fixture
def cohort(tmp_path):
    """Hexanol: pre, fly 2 responds (+10 during the odor); post, flies 1 (+20) and
    2 (+30) respond and fly 3 (resting at 12) dips by 4 without responding.
    Citral: flat, nobody responds."""
    spec = {  # fly: ((pre score, pre trace), (post score, post trace))
        1: ((0, _trace()), (3, _trace(during=30))),
        2: ((2, _trace(during=20)), (4, _trace(during=40))),
        3: ((0, _trace(rest=12)), (0, _trace(rest=12, during=8))),
    }
    wide = {"pretest": [], "testing": []}
    pred = []
    for n, ((s_pre, t_pre), (s_post, t_post)) in spec.items():
        for phase, idx, odor, score, trace in (
                ("pretest", 1, "hexanol", s_pre, t_pre), ("testing", 2, "hexanol", s_post, t_post),
                ("pretest", 2, "citral", 0, _trace()), ("testing", 1, "citral", 0, _trace())):
            wide[phase].append(("b1", n, phase, idx, odor, trace))
            pred.append(("b1", n, phase, idx, odor, score))
    paths = {"pretest": tmp_path / "pre.parquet", "testing": tmp_path / "post.parquet",
             "predictions": tmp_path / "pred.csv"}
    _wide(wide["pretest"]).to_parquet(paths["pretest"], index=False)
    _wide(wide["testing"]).to_parquet(paths["testing"], index=False)
    _pred(pred).to_csv(paths["predictions"], index=False)
    return paths


def _main(paths, out, *extra):
    return rmt.main(["--pretest-wide-csv", str(paths["pretest"]),
                     "--testing-wide-csv", str(paths["testing"]),
                     "--predictions-csv", str(paths["predictions"]), "--dataset", DS,
                     "--out-dir", str(out), "--label", "start 9:40am-4:20pm", *extra])


# ── the traces: each trial minus its own first 30 s ──────────────────────


def test_each_trial_loses_its_own_pre_odor_mean(cohort):
    paired = rmt.paired_trials(cohort["predictions"], dataset=DS)
    traces = rmt.phase_traces(pd.read_parquet(cohort["pretest"]), paired, phase="pre",
                              baseline_frames=int(30 * FPS))
    t3 = traces[("b1", 3, "Hexanol")]                  # rests at 12, flat
    assert np.allclose(t3, 0.0)
    t2 = traces[("b1", 2, "Hexanol")]
    assert np.allclose(t2[: int(30 * FPS)], 0.0) and np.allclose(t2[1200:2400], 10.0)


def test_trailing_frames_with_no_tracking_are_cut_as_on_the_mean_traces(cohort):
    wide = pd.read_parquet(cohort["pretest"])
    wide.loc[0, [f"dir_val_{k}" for k in range(3000, N)]] = np.nan
    paired = rmt.paired_trials(cohort["predictions"], dataset=DS)
    traces = rmt.phase_traces(wide, paired, phase="pre", baseline_frames=1200)
    key = (wide.loc[0, "fly"], int(wide.loc[0, "fly_number"]), "Hexanol")
    assert traces[key].size == 3000


# ── the split ─────────────────────────────────────────────────────────────


def test_the_summary_splits_by_the_score_of_the_same_trial(cohort, tmp_path):
    out = tmp_path / "out"
    assert _main(cohort, out) == 0
    s = pd.read_csv(out / "responder_mean_traces_summary.csv")
    hexanol = s[s["odor"] == "Hexanol"].set_index(["phase", "group"])
    expect = {("Pre-exposure", "responding"): (1, 10.0),
              ("Pre-exposure", "not responding"): (2, 0.0),
              ("Post-exposure", "responding"): (2, 25.0),
              ("Post-exposure", "not responding"): (1, -4.0)}
    for key, (n, during) in expect.items():
        assert hexanol.loc[key, "n_flies"] == n
        assert hexanol.loc[key, "mean_during"] == pytest.approx(during)
        assert hexanol.loc[key, "mean_before"] == pytest.approx(0.0)
    assert s["odor"].iloc[0] == "Hexanol"                  # sensitized odor first
    citral = s[s["odor"] == "Citral"].set_index(["phase", "group"])
    assert citral.loc[("Pre-exposure", "responding"), "n_flies"] == 0


def test_the_two_groups_add_up_to_the_mean_trace_of_every_fly(cohort):
    paired = rmt.paired_trials(cohort["predictions"], dataset=DS)
    traces = rmt.phase_traces(pd.read_parquet(cohort["testing"]), paired, phase="post",
                              baseline_frames=1200)
    hexanol = {k: v for k, v in traces.items() if k[2] == "Hexanol"}
    responding = {(f, n, o): bool(s >= 2) for f, n, o, s in
                  paired[["fly", "fly_number", "odor", "score_post"]].itertuples(index=False)}
    curves = rmt.group_curves(hexanol, responding, max_frames=N)
    every, _ = _mean_sem_trace(hexanol, N)
    (m1, _, n1), (m0, _, n0) = curves["responding"], curves["not responding"]
    assert np.allclose((n1 * m1 + n0 * m0) / (n1 + n0), every)


def test_one_figure_per_odor_and_one_with_every_odor(cohort, tmp_path):
    out = tmp_path / "out"
    _main(cohort, out)
    assert (out / "Hexanol_responder_mean_traces.png").is_file()
    assert (out / "Citral_responder_mean_traces.png").is_file()
    assert (out / "All_odors_responder_mean_traces.png").is_file()


def test_nothing_paired_is_nothing_to_draw(cohort, tmp_path):
    pred = pd.read_csv(cohort["predictions"])
    pred[pred["trial_type"] == "testing"].to_csv(cohort["predictions"], index=False)
    out = tmp_path / "out"
    assert _main(cohort, out) == 0
    assert not list(out.glob("*.png"))


# ── --with-median: median twins and the mean-vs-median comparison ─────────


def _skewed_traces():
    """Five flies, one group; four rest at 0 and one outlier jumps to 100 from 30 s."""
    traces = {("b1", i, "Hexanol"): _trace(rest=0.0, during=(100.0 if i == 5 else 2.0 * i))
              for i in range(1, 6)}
    return traces, {k: True for k in traces}


def test_the_median_curve_is_nanmedian_over_the_same_per_fly_traces():
    traces, responding = _skewed_traces()
    traces[("b1", 2, "Hexanol")] = traces[("b1", 2, "Hexanol")][:3000]   # a shorter fly
    bands = rmt.group_bands(traces, responding, max_frames=N, stat="median")
    centre, lo, hi, n = bands["responding"]
    stack = np.vstack([rmt._fit(v, N, np.nan) for v in traces.values()])
    assert n == 5
    assert np.allclose(centre, np.nanmedian(stack, axis=0), equal_nan=True)
    assert np.allclose(lo, np.nanpercentile(stack, 25, axis=0), equal_nan=True)
    assert np.allclose(hi, np.nanpercentile(stack, 75, axis=0), equal_nan=True)
    assert bands["not responding"][3] == 0 and np.isnan(bands["not responding"][0]).all()


def test_median_differs_from_mean_when_one_fly_is_an_outlier():
    traces, responding = _skewed_traces()
    mean = rmt.group_bands(traces, responding, max_frames=N, stat="mean")["responding"]
    med = rmt.group_bands(traces, responding, max_frames=N, stat="median")["responding"]
    t = 45 * int(FPS)
    assert med[0][t] == pytest.approx(6.0)                  # 2,4,6,8,100 -> 6
    assert mean[0][t] == pytest.approx(24.0)
    assert (med[2][t], med[1][t]) == (pytest.approx(8.0), pytest.approx(4.0))   # IQR 4..8
    curves = rmt.group_curves(traces, responding, max_frames=N)["responding"]
    assert np.allclose(mean[0], curves[0]) and np.allclose(mean[1], curves[0] - curves[1])
    assert np.allclose(mean[2], curves[0] + curves[1])      # mean band is mean +/- SEM


def test_mean_vs_median_gives_each_phase_its_own_panel(tmp_path, monkeypatch):
    """User 2026-11-13: with pre and post on one axis the dashed medians hid under the
    other phase's lines; each panel now holds one phase: its mean solid, median black dashed."""
    from matplotlib.figure import Figure

    figs = []
    monkeypatch.setattr(Figure, "savefig", lambda self, *a, **k: figs.append(self))
    traces, responding = _skewed_traces()
    mean = {k: rmt.group_bands(traces, responding, max_frames=N, stat="mean") for k in ("pre", "post")}
    med = {k: rmt.group_bands(traces, responding, max_frames=N, stat="median") for k in ("pre", "post")}
    kw = dict(odor="Hexanol", sensitized=True, dataset="Hex-Sensitivity-1-0.1", label="",
              labels={"pre": "Pre-exposure", "post": "Post-exposure"}, fps=FPS,
              odor_on_s=30.0, odor_off_s=60.0, threshold=2)
    rmt.plot_odor(tmp_path / "x.png", mean, stat="both", other=med, **kw)
    rmt.plot_all(tmp_path / "y.png", {"Hexanol": mean}, ["Hexanol"], "Hexanol", stat="both",
                 others={"Hexanol": med}, **{k: v for k, v in kw.items()
                                             if k not in ("odor", "sensitized")})
    for fig, n_axes in zip(figs, (4, 4)):
        assert len(fig.axes) == n_axes
        for ax in fig.axes:
            lines = [ln for ln in ax.get_lines() if len(ln.get_xdata()) > 2]
            solid = [ln for ln in lines if ln.get_linestyle() == "-"]
            dashed = [ln for ln in lines if ln.get_linestyle() == "--"]
            if not solid:                      # an empty group (no flies) draws nothing
                continue
            assert len(solid) == 1 and len(dashed) == 1
            assert dashed[0].get_color() == rmt.INK


def test_y_range_covers_the_iqr_band():
    band = {"pre": {"responding": (np.zeros(10), np.full(10, -50.0), np.full(10, 80.0), 3)}}
    lo, hi = rmt.y_range(band)
    assert lo < -50.0 and hi > 80.0


SIBLINGS = ("_median", "_mean_vs_median")


def test_without_the_flag_no_median_files_are_written(cohort, tmp_path):
    out = tmp_path / "out"
    _main(cohort, out)
    assert not [f for f in out.iterdir() if "median" in f.name]
    cols = pd.read_csv(out / "responder_mean_traces_summary.csv").columns
    assert not [c for c in cols if "median" in c]


def test_with_the_flag_each_figure_gets_both_siblings(cohort, tmp_path):
    out = tmp_path / "out"
    _main(cohort, out, "--with-median")
    for stem in ("Hexanol_responder_mean_traces", "Citral_responder_mean_traces",
                 "All_odors_responder_mean_traces"):
        assert (out / f"{stem}.png").is_file()
        for sib in SIBLINGS:
            assert (out / f"{stem}{sib}.png").is_file()


def test_the_flag_leaves_the_mean_figures_and_summary_columns_alone(cohort, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    _main(cohort, a)
    _main(cohort, b, "--with-median")
    base = pd.read_csv(a / "responder_mean_traces_summary.csv")
    both = pd.read_csv(b / "responder_mean_traces_summary.csv")
    pd.testing.assert_frame_equal(both[list(base.columns)], base)
    assert list(both.columns[:len(base.columns)]) == list(base.columns)
    assert list(both.columns[len(base.columns):]) == [
        "median_before", "median_during", "median_after", "median_peak_during"]
    assert (a / "Hexanol_responder_mean_traces.png").read_bytes() == \
        (b / "Hexanol_responder_mean_traces.png").read_bytes()


def test_summary_median_columns_are_the_window_average_of_the_median_curve(cohort, tmp_path):
    out = tmp_path / "out"
    _main(cohort, out, "--with-median")
    s = pd.read_csv(out / "responder_mean_traces_summary.csv")
    row = s[(s["odor"] == "Hexanol") & (s["phase"] == "Post-exposure")
            & (s["group"] == "responding")].iloc[0]
    assert row["median_during"] == pytest.approx(25.0)      # median of 20 and 30
    assert row["median_before"] == pytest.approx(0.0)


def test_a_run_without_the_flag_clears_the_median_files_of_an_earlier_run(cohort, tmp_path):
    out = tmp_path / "out"
    _main(cohort, out, "--with-median")
    _main(cohort, out)
    assert not [f for f in out.iterdir() if "median" in f.name]
    assert (out / "Hexanol_responder_mean_traces.png").is_file()


# ── wired into the window figure set ──────────────────────────────────────


def _arg(cmd, flag):
    return cmd[cmd.index(flag) + 1]


def test_the_window_set_draws_it_with_the_mean_traces_baseline_and_score_threshold(
        monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(swr, "_run", lambda cmd, log: seen.setdefault(Path(str(cmd[1])).name, cmd)
                        and 0)
    p = {k: tmp_path / k for k in ("predictions", "testing", "pretest", "training")}
    flagged = tmp_path / "flagged.csv"
    swr.run_figures(tmp_path, p, DS, config=tmp_path / "c.yaml", label="start x", log=None,
                    flagged=flagged)
    cmd = seen["responder_mean_traces.py"]
    means = seen["dataset_mean_traces_tvc.py"]
    score = seen["pretest_vs_test_comparison.py"]
    assert _arg(cmd, "--binary-threshold") == _arg(score, "--binary-threshold")
    for flag in ("--fps", "--odor-on-s"):
        assert _arg(cmd, flag) == _arg(means, flag)
    assert _arg(cmd, "--out-dir") == tmp_path / rmt.FOLDER
    for flag, value in (("--pretest-wide-csv", p["pretest"]), ("--testing-wide-csv", p["testing"]),
                        ("--predictions-csv", p["predictions"]), ("--flagged-flies-csv", flagged),
                        ("--dataset", DS), ("--label", "start x")):
        assert _arg(cmd, flag) == value


def test_editing_it_redraws_the_window_set():
    from fbpipe.utils import figure_cache as fc

    files = fc.code_files([REPO / "scripts" / "analysis" / "sensitivity_window_results.py"])
    assert REPO / "scripts" / "analysis" / "responder_mean_traces.py" in files
