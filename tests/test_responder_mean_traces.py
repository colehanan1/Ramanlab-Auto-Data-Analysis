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
