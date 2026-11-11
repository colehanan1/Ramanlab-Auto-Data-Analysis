"""How much, and how many flies, above their red line at each moment (drafts B and C).

B: the mean over flies of the graded value (extension above rest while above the
red line, 0 otherwise), so a dip below rest cannot cancel a response. C: the % of
flies above their red line, 1 s running mean, Wilson 95 % band.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import scripts.analysis.above_threshold_traces as at
import scripts.analysis.every_fly_heatmaps as eh
import scripts.analysis.start_window_report as swr

REPO = Path(__file__).resolve().parents[1]
DS = "Hex-Sensitivity-1-0.1"
FPS = 40.0
N = 3600


def _trace(start_s=None, length_s=2.0):
    x = np.full(N, 10.0)                  # flat at 10 % (theta = 15)
    if start_s is not None:
        a = int(round(start_s * FPS))
        x[a:a + int(round(length_s * FPS))] = 60.0
    return x


@pytest.fixture
def cohort(tmp_path):
    """Hexanol post: flies 1 (from 35 s) and 2 (from 31 s) extend 50 % for 2 s; 3, 4 never."""
    spec = {1: (0, None, 3, 35.0), 2: (2, 32.0, 4, 31.0), 3: (0, None, 2, None),
            4: (1, 40.0, 0, None)}
    wide = {"pretest": [], "testing": []}
    pred = []
    for n, (s_pre, t_pre, s_post, t_post) in spec.items():
        for phase, idx, odor, score, onset in (
                ("pretest", 1, "hexanol", s_pre, t_pre), ("testing", 2, "hexanol", s_post, t_post),
                ("pretest", 2, "citral", 0, None), ("testing", 1, "citral", 0, None)):
            rec = {"dataset": DS, "fly": "b1", "fly_number": n, "trial_type": phase,
                   "trial_label": f"{phase}_{idx}_{odor}", "fps": FPS, "trace_len": N,
                   "trial_odor_on_s": 30.0, "trial_odor_off_s": 60.0}
            rec.update({f"dir_val_{k}": v for k, v in enumerate(_trace(onset))})
            wide[phase].append(rec)
            pred.append({k: rec[k] for k in ("dataset", "fly", "fly_number", "trial_type",
                                             "trial_label")} | {"score": score})
    paths = {"pretest": tmp_path / "pre.parquet", "testing": tmp_path / "post.parquet",
             "predictions": tmp_path / "pred.csv", "config": tmp_path / "config.yaml"}
    pd.DataFrame(wide["pretest"]).to_parquet(paths["pretest"], index=False)
    pd.DataFrame(wide["testing"]).to_parquet(paths["testing"], index=False)
    pd.DataFrame(pred).to_csv(paths["predictions"], index=False)
    paths["config"].write_text(yaml.safe_dump({"analysis": {"combined": {"combined_base": {
        "wide": {"threshold_std_mult": 3.0, "threshold_min_delta": 5.0,
                 "threshold_anchor_s": 5.0, "threshold_noise_block_s": 5.0,
                 "threshold_noise_pctl": 25.0}}}}}))
    return paths


def test_b_averages_the_graded_value_over_every_fly(cohort):
    rows = eh.load_rows(cohort["pretest"], cohort["testing"], cohort["predictions"],
                        config=cohort["config"], dataset=DS, n_frames=N)
    hexanol = {k: v for k, v in rows["post"].items() if k[2] == "Hexanol"}
    mean, sem, n = at.mean_extension(hexanol)
    assert n == 4
    assert mean[int(31.5 * FPS)] == pytest.approx(50.0 / 4)   # only fly 2 is up, 50 %
    assert mean[int(20 * FPS)] == 0.0 and sem[int(20 * FPS)] == 0.0


def test_c_is_the_share_of_tracked_flies_above_their_line():
    above = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 0], [0, 0, 0]], dtype=bool)
    tracked = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 0], [1, 1, 0]], dtype=bool)
    pct, lo, hi = at.percent_above(above, tracked, smooth_frames=1)
    assert pct.tolist() == pytest.approx([25.0, 50.0, 0.0])   # frame 3: 2 flies tracked
    assert (lo <= pct).all() and (pct <= hi).all()


def test_one_figure_per_odor_one_with_every_odor_and_a_summary(cohort, tmp_path):
    out = tmp_path / "out"
    assert at.main(["--pretest-wide-csv", str(cohort["pretest"]),
                    "--testing-wide-csv", str(cohort["testing"]),
                    "--predictions-csv", str(cohort["predictions"]),
                    "--config", str(cohort["config"]), "--dataset", DS,
                    "--out-dir", str(out), "--label", "start 9:40am-4:20pm"]) == 0
    for name in ("Hexanol_above_threshold.png", "Citral_above_threshold.png",
                 "All_odors_above_threshold.png"):
        assert (out / name).is_file()
    s = pd.read_csv(out / "above_threshold_summary.csv").set_index(["odor", "phase"])
    # post: flies 1 and 2 each 50 % up for 2 s of the 30 s odor, flies 3 and 4 never
    assert s.loc[("Hexanol", "Post-exposure"), "n_flies"] == 4
    assert s.loc[("Hexanol", "Post-exposure"), "mean_extension_during"] == pytest.approx(
        2 * 50.0 * 2.0 / 30.0 / 4, rel=0.02)
    assert s.loc[("Citral", "Pre-exposure"), "mean_extension_during"] == 0.0


def _arg(cmd, flag):
    return cmd[cmd.index(flag) + 1]


def test_the_window_set_draws_it(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(swr, "_run", lambda cmd, log: seen.setdefault(Path(str(cmd[1])).name, cmd)
                        and 0)
    p = {k: tmp_path / k for k in ("predictions", "testing", "pretest", "training")}
    swr.run_figures(tmp_path, p, DS, config=tmp_path / "c.yaml", label="start x", log=None,
                    flagged=tmp_path / "flagged.csv")
    cmd = seen["above_threshold_traces.py"]
    assert _arg(cmd, "--out-dir") == tmp_path / at.FOLDER
    assert _arg(cmd, "--config") == tmp_path / "c.yaml"
    assert _arg(cmd, "--binary-threshold") == _arg(seen["pretest_vs_test_comparison.py"],
                                                   "--binary-threshold")


def test_editing_it_redraws_the_window_set():
    from fbpipe.utils import figure_cache as fc

    files = fc.code_files([REPO / "scripts" / "analysis" / "sensitivity_window_results.py"])
    assert REPO / "scripts" / "analysis" / "above_threshold_traces.py" in files
