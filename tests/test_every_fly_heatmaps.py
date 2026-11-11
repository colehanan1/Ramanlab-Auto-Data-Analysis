"""Every fly's response to one odor, pre- and post-exposure side by side (draft E).

Cells are the graded rasters' value (extension above rest while above the red
line), grey where the tracker lost the proboscis; the flies are the score
figures' pairs, in one order for both panels, strongest during the odor first.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import scripts.analysis.every_fly_heatmaps as eh
import scripts.analysis.start_window_report as swr

REPO = Path(__file__).resolve().parents[1]
DS = "Hex-Sensitivity-1-0.1"          # sensitized odor: Hexanol
FPS = 40.0
N = 3600
THRESHOLD_BLOCK = {"threshold_std_mult": 3.0, "threshold_min_delta": 5.0,
                   "threshold_anchor_s": 5.0, "threshold_noise_block_s": 5.0,
                   "threshold_noise_pctl": 25.0}


def _trace(start_s=None, length_s=2.0):
    """Flat at 10 % (theta = 15), a 60 % extension from ``start_s``."""
    x = np.full(N, 10.0)
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
    """Hexanol: fly 2 extends in both phases, fly 1 post only, fly 4 pre only (scored
    1, so not a responder), fly 3 never; fly 3's post trace loses tracking 70-80 s."""
    spec = {1: (0, None, 3, 35.0), 2: (2, 32.0, 4, 31.0), 3: (0, None, 2, None),
            4: (1, 40.0, 0, None)}
    wide = {"pretest": [], "testing": []}
    pred = []
    for n, (s_pre, t_pre, s_post, t_post) in spec.items():
        for phase, idx, odor, score, onset in (
                ("pretest", 1, "hexanol", s_pre, t_pre), ("testing", 2, "hexanol", s_post, t_post),
                ("pretest", 2, "citral", 0, None), ("testing", 1, "citral", 0, None)):
            trace = _trace(onset)
            if n == 3 and phase == "testing" and odor == "hexanol":
                trace[int(70 * FPS):int(80 * FPS)] = np.nan
            wide[phase].append(("b1", n, phase, idx, odor, trace))
            pred.append({"dataset": DS, "fly": "b1", "fly_number": n, "trial_type": phase,
                         "trial_label": f"{phase}_{idx}_{odor}", "score": score})
    paths = {"pretest": tmp_path / "pre.parquet", "testing": tmp_path / "post.parquet",
             "predictions": tmp_path / "pred.csv", "config": tmp_path / "config.yaml"}
    _wide(wide["pretest"]).to_parquet(paths["pretest"], index=False)
    _wide(wide["testing"]).to_parquet(paths["testing"], index=False)
    pd.DataFrame(pred).to_csv(paths["predictions"], index=False)
    paths["config"].write_text(yaml.safe_dump(
        {"analysis": {"combined": {"combined_base": {"wide": THRESHOLD_BLOCK}}}}))
    return paths


def _main(paths, out):
    return eh.main(["--pretest-wide-csv", str(paths["pretest"]),
                    "--testing-wide-csv", str(paths["testing"]),
                    "--predictions-csv", str(paths["predictions"]),
                    "--config", str(paths["config"]), "--dataset", DS,
                    "--out-dir", str(out), "--label", "start 9:40am-4:20pm"])


def test_rows_are_the_graded_value_and_grey_where_tracking_was_lost(cohort):
    rows = eh.load_rows(cohort["pretest"], cohort["testing"], cohort["predictions"],
                        config=cohort["config"], dataset=DS, n_frames=N)
    post = rows["post"][("b1", 2, "Hexanol")]
    a = int(31 * FPS)
    assert post[a] == pytest.approx(50.0) and post[a - 10] == 0.0   # 60 % over a 10 % rest
    lost = rows["post"][("b1", 3, "Hexanol")]
    assert np.isnan(lost[int(75 * FPS)]) and lost[int(65 * FPS)] == 0.0


def test_flies_are_ordered_by_their_extension_during_the_odor_pre_plus_post(cohort):
    rows = eh.load_rows(cohort["pretest"], cohort["testing"], cohort["predictions"],
                        config=cohort["config"], dataset=DS, n_frames=N)
    order = eh.fly_order(rows["pre"], rows["post"], odor="Hexanol",
                         on=int(30 * FPS), off=int(60 * FPS))
    # fly 2 extends in both phases; flies 1 and 4 once each (tie: fly number); fly 3 never
    assert [n for _, n in order] == [2, 1, 4, 3]


def test_one_figure_per_odor_and_the_row_order(cohort, tmp_path):
    out = tmp_path / "out"
    assert _main(cohort, out) == 0
    assert (out / "Hexanol_every_fly_heatmap.png").is_file()
    assert (out / "Citral_every_fly_heatmap.png").is_file()
    order = pd.read_csv(out / "every_fly_heatmap_order.csv")
    hexanol = order[order["odor"] == "Hexanol"].sort_values("rank")
    assert hexanol["fly_number"].tolist() == [2, 1, 4, 3]
    assert hexanol["responding_post"].tolist() == [True, True, False, True]


def test_nothing_paired_is_nothing_to_draw(cohort, tmp_path):
    pred = pd.read_csv(cohort["predictions"])
    pred[pred["trial_type"] == "testing"].to_csv(cohort["predictions"], index=False)
    out = tmp_path / "out"
    assert _main(cohort, out) == 0
    assert not list(out.glob("*.png"))


def _arg(cmd, flag):
    return cmd[cmd.index(flag) + 1]


def test_the_window_set_draws_it_with_the_score_figures_threshold(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(swr, "_run", lambda cmd, log: seen.setdefault(Path(str(cmd[1])).name, cmd)
                        and 0)
    p = {k: tmp_path / k for k in ("predictions", "testing", "pretest", "training")}
    swr.run_figures(tmp_path, p, DS, config=tmp_path / "c.yaml", label="start x", log=None,
                    flagged=tmp_path / "flagged.csv")
    cmd = seen["every_fly_heatmaps.py"]
    assert _arg(cmd, "--binary-threshold") == _arg(seen["pretest_vs_test_comparison.py"],
                                                   "--binary-threshold")
    assert _arg(cmd, "--out-dir") == tmp_path / eh.FOLDER
    for flag, value in (("--pretest-wide-csv", p["pretest"]), ("--testing-wide-csv", p["testing"]),
                        ("--predictions-csv", p["predictions"]), ("--config", tmp_path / "c.yaml"),
                        ("--flagged-flies-csv", tmp_path / "flagged.csv"), ("--dataset", DS)):
        assert _arg(cmd, flag) == value


def test_editing_it_redraws_the_window_set():
    from fbpipe.utils import figure_cache as fc

    files = fc.code_files([REPO / "scripts" / "analysis" / "sensitivity_window_results.py"])
    assert REPO / "scripts" / "analysis" / "every_fly_heatmaps.py" in files
