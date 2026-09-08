"""Wide tables in production column order."""
import json

import numpy as np
import pandas as pd
import pytest

from fbpipe.analysis.traces import read_wide_table
from fbpipe.config import Settings
from fbpipe.geometry import columns as C
from fbpipe.geometry import steps as S
from fbpipe.geometry import wide as Wd
from fbpipe.geometry.config import GeometrySettings
from fbpipe.geometry.trial_meta import TrialInfo


def _info(batch, name, ttype, idx, rec, n=30):
    return TrialInfo(video=None, batch_dir=str(batch), dataset="Hex-Sensitivity-24-0.1", trial_name=name,
                     out_dir=str(batch / "geometry" / name), trial_type=ttype, trial_index=idx, odor="Citral",
                     label=f"{ttype}_{idx}_Citral", n_frames=n, fps=40.0, has_odor=True, odor_on_frame=10,
                     odor_off_frame=20, light_on_s=None, light_off_s=None, fly_type="GR5a-Old",
                     sidecar_txt=None, sidecar_csv=None, recorded_at=rec)


def _table(n, seed):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: np.nan for c in C.PER_FRAME_COLUMNS + C.NORMALIZED_COLUMNS}, index=range(n))
    df[C.FRAME] = np.arange(n)
    df[C.TIMESTAMP] = np.arange(n) / 40.0
    df[C.ODOR_ON] = False
    df[C.STATUS] = C.STATUS_OK
    df[C.PRO_STATUS] = C.PRO_OK
    df[C.N_EYES] = 3
    df[C.PRO_PCT] = rng.uniform(0, 100, n)
    df[C.ABD_ANGLE_PCT] = rng.uniform(0, 100, n)
    df.loc[n - 3:, C.ABD_ANGLE_PCT] = np.nan
    df[C.ABD_LEN_PCT] = 100.0
    df[C.ABD_WID_PCT] = 100.0
    return df


@pytest.fixture
def batch(tmp_path):
    b = tmp_path / "flys_Abdomen" / "Hex-Sensitivity-24-0.1" / "september_02_batch_1_rig_3_copy"
    later = _info(b, "september_02_batch_1_testing_1", "testing", 1, "20260902_120000", n=30)
    earlier = _info(b, "september_02_batch_1_pretest_1", "pretest", 1, "20260902_110000", n=25)
    for info, seed in ((later, 1), (earlier, 2)):
        d = b / "geometry" / info.trial_name
        d.mkdir(parents=True)
        for slot in (1, 2):
            C.write_table_fmt(_table(info.n_frames, seed + slot), C.geometry_table_path(d, info.trial_name, slot, "csv"), "csv")
        C.meta_path(d, info.trial_name).write_text(json.dumps({"trial": info.to_dict()}))
    return b


def test_wide_rows_follow_production_column_order_and_protocol_order(batch, tmp_path):
    rows = Wd.batch_wide_rows(batch, "Hex-Sensitivity-24-0.1", False, GeometrySettings())
    assert set(rows) == set(Wd.WIDE_MEASURES)
    tables = Wd.build_wide_tables(rows)
    per = tables["per"]
    meta = list(per.columns[: len(Wd.WIDE_META_COLUMNS)])
    assert meta == Wd.WIDE_META_COLUMNS
    assert meta[:3] == ["dataset", "fly", "fly_number"] and meta[-1] == "frozen"
    assert meta[-4:] == ["trial_type", "trial_label", "fps", "frozen"]
    vals = [c for c in per.columns if c.startswith("dir_val_")]
    assert vals == [f"dir_val_{i}" for i in range(30)], "padded to the longest trial, 0-indexed"
    assert per["fly_number"].tolist() == ["1", "2", "1", "2"]
    assert per["trial_label"].tolist() == ["pretest_1_Citral", "pretest_1_Citral", "testing_1_Citral", "testing_1_Citral"]
    assert per["fly"].iloc[0] == "september_02_batch_1_rig_3_copy" and per["dataset"].iloc[0] == "Hex-Sensitivity-24-0.1"
    assert per["trace_len"].tolist() == [25, 25, 30, 30]
    assert np.isnan(per["dir_val_29"].iloc[0]) and np.isfinite(per["dir_val_29"].iloc[2])
    assert per["trial_odor_on_s"].iloc[0] == pytest.approx(0.25) and per["frozen"].tolist() == [False] * 4
    ang = tables["abdomen_angle"]
    assert ang["tracked_fraction"].iloc[0] == pytest.approx(22 / 25)
    out = Wd.write_wide_tables(tables, tmp_path / "wide", "csv")
    assert out["per"].name == "geometry_rows_wide_per.csv"
    back = read_wide_table(out["per"])
    assert len(back) == 4 and "dir_val_0" in back.columns


def test_frozen_batches_are_marked_and_run_wide_writes_every_measure(batch, tmp_path):
    settings = Settings(model_path="x", main_directories=[str(batch.parent)], datasets=("Hex-Sensitivity-24-0.1",))
    gcfg = GeometrySettings(table_format="parquet", wide_out_dir=str(tmp_path / "wide"), figures_out_dir=str(tmp_path / "figs"))
    result = S.run_wide(settings, gcfg)
    assert sorted(result["files"]) == sorted(Wd.WIDE_MEASURES)
    for path in result["files"].values():
        assert path.suffix == ".parquet" and path.exists()
    rows = Wd.batch_wide_rows(batch, "Hex-Sensitivity-24-0.1", True, gcfg)
    assert all(meta["frozen"] is True for meta, _ in rows["per"])
