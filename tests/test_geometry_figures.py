"""Trace / size / QC figures and the trial summary of the geometry pipeline."""
import json

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest

from fbpipe.config import Settings
from fbpipe.geometry import columns as C
from fbpipe.geometry import figures as F
from fbpipe.geometry import steps as S
from fbpipe.geometry.config import GeometrySettings
from fbpipe.geometry.trial_meta import TrialInfo


def _info(batch, name, ttype, idx, odor, rec, n=120, has_odor=True):
    return TrialInfo(video=None, batch_dir=str(batch), dataset="Hex-Sensitivity-24-0.1", trial_name=name,
                     out_dir=str(batch / "geometry" / name), trial_type=ttype, trial_index=idx, odor=odor,
                     label=f"{ttype}_{idx}_{odor}", n_frames=n, fps=40.0, has_odor=has_odor,
                     odor_on_frame=40 if has_odor else None, odor_off_frame=80 if has_odor else None,
                     light_on_s=None if has_odor else 1.0, light_off_s=None if has_odor else 1.6,
                     fly_type="GR5a-Old", sidecar_txt=None, sidecar_csv=None, recorded_at=rec)


def _table(n, angle_ok=True):
    df = pd.DataFrame({c: np.nan for c in C.PER_FRAME_COLUMNS + C.NORMALIZED_COLUMNS}, index=range(n))
    df[C.FRAME] = np.arange(n)
    df[C.TIMESTAMP] = np.arange(n) / 40.0
    df[C.ODOR_ON] = (df[C.FRAME] >= 40) & (df[C.FRAME] < 80)
    df[C.PRO_STATUS] = C.PRO_OK
    df[C.N_EYES] = 3
    df[C.PRO_PCT] = np.where(df[C.ODOR_ON], 80.0, 10.0)
    if angle_ok:
        df[C.STATUS] = C.STATUS_OK
        df[C.ABD_ANGLE_PCT] = np.where(df[C.ODOR_ON], 60.0, 5.0)
    else:
        df[C.STATUS] = "predicted tip -0.4 eye-widths outside frame - abdomen not fully visible"
    df[C.ABD_LEN_PCT] = 100.0
    df[C.ABD_WID_PCT] = 100.0
    return df


@pytest.fixture
def batch(tmp_path):
    b = tmp_path / "flys_Abdomen" / "Hex-Sensitivity-24-0.1" / "september_02_batch_1_rig_3_copy"
    trials = [
        _info(b, "september_02_batch_1_testing_9", "testing", 9, "LightOnly", "20260902_130000", has_odor=False),
        _info(b, "september_02_batch_1_pretest_1", "pretest", 1, "Citral", "20260902_110000"),
        _info(b, "september_02_batch_1_training_1", "training", 1, "Hexanol", "20260902_113000"),
        _info(b, "september_02_batch_1_training_2", "training", 2, "Hexanol", "20260902_114000"),
        _info(b, "september_02_batch_1_testing_1", "testing", 1, "Hexanol", "20260902_120000"),
    ]
    for info in trials:
        d = b / "geometry" / info.trial_name
        d.mkdir(parents=True)
        C.write_table_fmt(_table(info.n_frames), C.geometry_table_path(d, info.trial_name, 1, "csv"), "csv")
        C.write_table_fmt(_table(info.n_frames, angle_ok=False), C.geometry_table_path(d, info.trial_name, 2, "csv"), "csv")
        C.meta_path(d, info.trial_name).write_text(json.dumps({"trial": info.to_dict()}))
    return b


def test_panels_are_loaded_in_protocol_order_and_trained_odor_is_found(batch):
    panels = F.load_panels(batch, GeometrySettings())
    assert sorted(panels) == [1, 2]
    assert [p.info.label for p in panels[1]] == ["pretest_1_Citral", "training_1_Hexanol", "training_2_Hexanol",
                                                 "testing_1_Hexanol", "testing_9_LightOnly"]
    assert F.trained_odor(panels[1]) == "Hexanol"
    assert panels[1][0].t[1] == pytest.approx(0.025)


def test_generate_writes_figures_and_summary(batch, tmp_path):
    settings = Settings(model_path="x", main_directories=[str(batch.parent)], datasets=("Hex-Sensitivity-24-0.1",))
    out_root = tmp_path / "figs"
    res = F.generate(batch, "Hex-Sensitivity-24-0.1", settings, GeometrySettings(trace_max_s=3.0), out_root)
    out_dir = out_root / "Hex-Sensitivity-24-0.1" / "september_02_batch_1_rig_3_copy"
    names = sorted(p.name for p in out_dir.glob("*.png"))
    assert names == sorted([
        "september_02_batch_1_rig_3_copy_fly1_per_abdomen_traces.png",
        "september_02_batch_1_rig_3_copy_fly2_per_abdomen_traces.png",
        "september_02_batch_1_rig_3_copy_fly1_abdomen_size_traces.png",
        "september_02_batch_1_rig_3_copy_fly2_abdomen_size_traces.png",
        "september_02_batch_1_rig_3_copy_geometry_qc.png",
    ])
    assert res["n_figures"] == 5 and set(res) >= {"figures", "summary_csv", "out_dir"}
    summary = pd.read_csv(res["summary_csv"])
    assert len(summary) == 10                                   # 5 trials x 2 flies
    row = summary[(summary.fly_number == 1) & (summary.trial_label == "testing_1_Hexanol")].iloc[0]
    assert row["per_max_pct_window"] == pytest.approx(80.0)
    assert row["angle_max_pct_window"] == pytest.approx(60.0)
    assert row["window_on_s"] == pytest.approx(1.0) and row["window_off_s"] == pytest.approx(2.0)
    assert row["angle_tracked_frac"] == pytest.approx(1.0)
    light = summary[(summary.fly_number == 1) & (summary.trial_label == "testing_9_LightOnly")].iloc[0]
    assert not light["has_odor"] and light["window_on_s"] == pytest.approx(1.0) and light["window_off_s"] == pytest.approx(1.6)
    fly2 = summary[summary.fly_number == 2]
    assert (fly2["angle_tracked_frac"] == 0.0).all()
    assert "outside frame" in str(fly2["dominant_status"].iloc[0])


def test_run_figures_uses_the_configured_root(batch, tmp_path):
    settings = Settings(model_path="x", main_directories=[str(batch.parent)], datasets=("Hex-Sensitivity-24-0.1",))
    gcfg = GeometrySettings(wide_out_dir=str(tmp_path / "wide"), figures_out_dir=str(tmp_path / "figs"), trace_max_s=3.0)
    result = S.run_figures(settings, gcfg)
    assert result["n_figures"] == 5
    assert (tmp_path / "figs" / "Hex-Sensitivity-24-0.1" / "september_02_batch_1_rig_3_copy").is_dir()
