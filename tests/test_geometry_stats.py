"""Statistics over the geometry pipeline's outputs: per fly-trial metrics, label joins,
correlations, status comparison, blinded-comment check, and the end-to-end report."""
import json

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest

from fbpipe.config import Settings
from fbpipe.geometry import columns as C
from fbpipe.geometry import stats as ST
from fbpipe.geometry.config import GeometrySettings
from fbpipe.geometry.trial_meta import TrialInfo

N = 120   # frames per trial at 40 fps: window = frames 40..80 (1.0-2.0 s)


def _info(batch, name, ttype, idx, odor, rec, has_odor=True):
    return TrialInfo(video=None, batch_dir=str(batch), dataset="Hex-Sensitivity-24-0.1", trial_name=name,
                     out_dir=str(batch / "geometry" / name), trial_type=ttype, trial_index=idx, odor=odor,
                     label=f"{ttype}_{idx}_{odor}", n_frames=N, fps=40.0, has_odor=has_odor,
                     odor_on_frame=40 if has_odor else None, odor_off_frame=80 if has_odor else None,
                     light_on_s=None if has_odor else 1.0, light_off_s=None if has_odor else 2.0,
                     fly_type="GR5a-Old", sidecar_txt=None, sidecar_csv=None, recorded_at=rec)


def _table(responder=True, angle_ok=True):
    df = pd.DataFrame({c: np.nan for c in C.PER_FRAME_COLUMNS + C.NORMALIZED_COLUMNS}, index=range(N))
    frame = np.arange(N)
    win = (frame >= 40) & (frame < 80)
    df[C.FRAME] = frame
    df[C.TIMESTAMP] = frame / 40.0
    df[C.ODOR_ON] = win
    df[C.PRO_STATUS] = C.PRO_OK
    df[C.N_EYES] = 2
    df[C.PRO_DIST] = np.where(win, 90.0, 45.0) if responder else 45.0
    df[C.PRO_PCT] = np.where(win, 80.0, 10.0) if responder else 10.0
    if angle_ok:
        df[C.STATUS] = C.STATUS_OK
        angle = np.where(win, 60.0, 5.0) + np.linspace(0, 4, N)          # small ramp -> no ties
        df[C.ABD_ANGLE_PCT] = angle
        df[C.ABD_LEN_PCT] = 110.0 - 0.2 * angle                          # anti-correlated with the angle
        df[C.ABD_WID_PCT] = 95.0 + 0.1 * angle                           # correlated
        df[C.ABD_LEN] = 200.0 * df[C.ABD_LEN_PCT] / 100.0
    else:
        df[C.STATUS] = "predicted tip -0.4 eye-widths outside frame - abdomen not fully visible"
    return df


@pytest.fixture
def batch(tmp_path):
    b = tmp_path / "flys_Abdomen" / "Hex-Sensitivity-24-0.1" / "september_02_batch_1_rig_3"
    trials = [_info(b, "september_02_batch_1_testing_1", "testing", 1, "Hexanol", "20260902_120000"),
              _info(b, "september_02_batch_1_testing_9", "testing", 9, "LightOnly", "20260902_130000", has_odor=False)]
    for info in trials:
        d = b / "geometry" / info.trial_name
        d.mkdir(parents=True)
        C.write_table_fmt(_table(True, True), C.geometry_table_path(d, info.trial_name, 1, "csv"), "csv")
        C.write_table_fmt(_table(False, False), C.geometry_table_path(d, info.trial_name, 2, "csv"), "csv")
        C.meta_path(d, info.trial_name).write_text(json.dumps({"trial": info.to_dict()}))
    return b


def _labels(tmp_path):
    pred = tmp_path / "model_predictions.csv"
    pd.DataFrame([{"dataset": "Hex-Sensitivity-24-0.1", "fly": "september_02_batch_1_rig_3", "fly_number": 1,
                   "trial_label": "testing_1_hexanol", "prediction": 1, "score": 3, "trial_type": "testing"}]).to_csv(pred, index=False)
    blind = tmp_path / "blinded_video_scores.csv"
    pd.DataFrame([{"dataset": "Hex-Sensitivity-24-0.1", "fly": "september_02_batch_1_rig_3", "fly_number": 1,
                   "trial_type": "testing", "trial_label": "testing_1", "user_score": 2, "comment": "abdomen"},
                  {"dataset": "Hex-Sensitivity-24-0.1", "fly": "september_02_batch_1_rig_3", "fly_number": 2,
                   "trial_type": "testing", "trial_label": "testing_1", "user_score": 0, "comment": np.nan}]).to_csv(blind, index=False)
    flagged = tmp_path / "flagged-flys-truth.csv"
    flagged.write_text('dataset,fly,fly_number,"FLY-State(1, 0, -1)",comment,,,,\n'
                       'Hex-Sensitivity-24-0.1,september_02_batch_1_rig_3,2,0,Bad,ret,,,\n')
    return pred, blind, flagged


def test_fly_trial_metrics_for_batch(batch):
    df = ST.fly_trial_metrics_for_batch(batch, "Hex-Sensitivity-24-0.1", False, GeometrySettings(), Settings(model_path="x", main_directories=[]))
    assert len(df) == 4 and set(df.fly_number) == {1, 2}
    r = df[(df.fly_number == 1) & (df.trial_label == "testing_1_Hexanol")].iloc[0]
    assert r["trial_label_lc"] == "testing_1_hexanol" and r["odor_norm"] == "hexanol"
    assert r["per_base"] == pytest.approx(10.0) and r["per_win_mean"] == pytest.approx(80.0)
    assert r["per_delta"] == pytest.approx(70.0) and r["per_win_max"] == pytest.approx(80.0)
    assert 50.0 < r["angle_delta"] < 60.0 and r["angle_win_max"] > 60.0
    assert r["angle_activity_win"] < 3.0 and r["angle_tracked_frac"] == pytest.approx(1.0)
    assert r["rho_length_angle"] < -0.9 and r["rho_width_angle"] > 0.9
    assert r["window_on_s"] == pytest.approx(1.0) and r["window_off_s"] == pytest.approx(2.0) and r["has_odor"]
    r2 = df[(df.fly_number == 2) & (df.trial_label == "testing_1_Hexanol")].iloc[0]
    assert np.isnan(r2["angle_delta"]) and r2["angle_tracked_frac"] == 0.0 and r2["per_delta"] == pytest.approx(0.0)
    light = df[(df.fly_number == 1) & (df.trial_type == "testing") & (df.trial_index == 9)].iloc[0]
    assert not light["has_odor"] and light["window_on_s"] == pytest.approx(1.0)
    assert (df["frozen"] == False).all()


def test_attach_labels_joins_predictions_blinded_and_flags(batch, tmp_path):
    df = ST.fly_trial_metrics_for_batch(batch, "Hex-Sensitivity-24-0.1", False, GeometrySettings(), Settings(model_path="x", main_directories=[]))
    pred, blind, flagged = _labels(tmp_path)
    out = ST.attach_labels(df, predictions_csv=pred, blinded_csv=blind, flagged_csv=flagged)
    r = out[(out.fly_number == 1) & (out.trial_index == 1)].iloc[0]
    assert r["model_score"] == 3 and r["model_prediction"] == 1
    assert r["user_score"] == 2 and bool(r["abdomen_comment"]) and "abdomen" in str(r["comment"])
    assert r["fly_state"] == 1 and r["fly_status"] == "alive"
    r2 = out[(out.fly_number == 2) & (out.trial_index == 1)].iloc[0]
    assert r2["fly_state"] == 0 and r2["fly_status"] == "weak" and not bool(r2["abdomen_comment"])
    assert np.isnan(r2["model_score"])
    light = out[(out.fly_number == 1) & (out.trial_index == 9)].iloc[0]
    assert np.isnan(light["user_score"]) and not bool(light["abdomen_comment"])


def test_spearman_table_and_bh():
    rng = np.random.default_rng(0)
    x = rng.normal(size=40)
    df = pd.DataFrame({"x": x, "y": 2 * x + rng.normal(scale=0.1, size=40), "g": ["a"] * 20 + ["b"] * 20})
    t = ST.spearman_table(df, "x", "y", group_cols=("g",), min_n=8)
    assert list(t["g"]) == ["a", "b"] and (t["rho"] > 0.9).all() and (t["p"] < 1e-5).all() and (t["p_bh"] >= t["p"]).all()
    pooled = ST.spearman_table(df, "x", "y")
    assert len(pooled) == 1 and pooled["n"].iloc[0] == 40
    small = ST.spearman_table(df.head(5), "x", "y")
    assert small.empty
    assert ST.bh_adjust(np.array([0.01, 0.04, 0.03])).round(4).tolist() == [0.03, 0.04, 0.04]
    assert ST.auc_from_groups(np.array([5, 6, 7]), np.array([1, 2, 3])) == pytest.approx(1.0)
    assert ST.auc_from_groups(np.array([1, 2, 3]), np.array([1, 2, 3])) == pytest.approx(0.5)


def test_run_analysis_writes_report_and_figures(batch, tmp_path):
    settings = Settings(model_path="x", main_directories=[str(batch.parent)], datasets=("Hex-Sensitivity-24-0.1",))
    pred, blind, flagged = _labels(tmp_path)
    df = ST.collect_fly_trial_metrics(settings, GeometrySettings())
    assert len(df) == 4
    df = ST.attach_labels(df, predictions_csv=pred, blinded_csv=blind, flagged_csv=flagged)
    out = tmp_path / "stats"
    res = ST.run_analysis(df, out)
    assert (out / "fly_trial_metrics.csv").exists() and (out / "report.md").exists()
    for name in ("correlations.csv", "status_comparison.csv", "blinded_abdomen_comments.csv", "box_vs_angle.csv", "odor_summary.csv"):
        assert (out / name).exists(), name
    assert res["n_fly_trials"] == 4 and res["figures"]
    report = (out / "report.md").read_text()
    assert "abdomen" in report.lower() and "Spearman" in report
