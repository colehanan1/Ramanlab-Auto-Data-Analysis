"""Per-fly baselines over a batch and the % columns they produce."""
import json

import numpy as np
import pandas as pd
import pytest

from fbpipe.config import Settings
from fbpipe.geometry import columns as C
from fbpipe.geometry import normalize as N
from fbpipe.geometry.config import GeometrySettings
from fbpipe.geometry.trial_meta import TrialInfo


def _info(batch, name, ttype, idx, rec, has_odor=True):
    return TrialInfo(video=None, batch_dir=str(batch), dataset="Hex-Sensitivity-24-0.1", trial_name=name,
                     out_dir=str(batch / "geometry" / name), trial_type=ttype, trial_index=idx, odor="Hexanol",
                     label=f"{ttype}_{idx}_Hexanol", n_frames=40, fps=40.0, has_odor=has_odor,
                     odor_on_frame=20 if has_odor else None, odor_off_frame=30 if has_odor else None,
                     light_on_s=None if has_odor else 0.5, light_off_s=None if has_odor else 0.75,
                     fly_type="GR5a-Old", sidecar_txt=None, sidecar_csv=None, recorded_at=rec)


def _table(pro, angle, length, width, status=None):
    n = len(pro)
    df = pd.DataFrame({c: np.nan for c in C.PER_FRAME_COLUMNS}, index=range(n))
    df[C.FRAME] = np.arange(n)
    df[C.TIMESTAMP] = np.arange(n) / 40.0
    df[C.ODOR_ON] = False
    df[C.PRO_DIST] = pro
    df[C.PRO_STATUS] = np.where(np.isfinite(pro), C.PRO_OK, C.PRO_MISSING)
    df[C.ABD_ANGLE] = angle
    df[C.ABD_LEN] = length
    df[C.ABD_WID] = width
    df[C.STATUS] = status if status is not None else np.where(np.isfinite(angle), C.STATUS_OK, C.STATUS_ABDOMEN_ONLY)
    df[C.N_EYES] = 3
    return df


def _write(batch, info, tables, fmt="csv"):
    d = batch / "geometry" / info.trial_name
    d.mkdir(parents=True, exist_ok=True)
    for slot, df in tables.items():
        C.write_table_fmt(df, C.geometry_table_path(d, info.trial_name, slot, fmt), fmt)
    C.meta_path(d, info.trial_name).write_text(json.dumps({"trial": info.to_dict()}))


@pytest.fixture
def batch(tmp_path):
    b = tmp_path / "flys_Abdomen" / "Hex-Sensitivity-24-0.1" / "september_02_batch_1_rig_3_copy"
    b.mkdir(parents=True)
    n = 40
    pro_a = np.full(n, 40.0); pro_a[25] = 100.0; pro_a[26] = 5.0          # 5 px is below per_min_px
    ang_a = np.full(n, 175.0); ang_a[25] = 120.0                            # most curled in trial A frame 25
    len_a = np.full(n, 150.0); len_a[20:] = 180.0                            # pre-odor rest = 150
    wid_a = np.full(n, 60.0); wid_a[20:] = 70.0
    pro_b = np.full(n, 42.0); pro_b[:3] = np.nan
    ang_b = np.full(n, 178.0); ang_b[5:] = np.nan                            # tip lost after frame 5
    len_b = np.full(n, 152.0); wid_b = np.full(n, 61.0)
    a = _info(b, "september_02_batch_1_pretest_1", "pretest", 1, "20260902_110000")
    bb = _info(b, "september_02_batch_1_testing_1", "testing", 1, "20260902_120000")
    _write(b, a, {1: _table(pro_a, ang_a, len_a, wid_a)})
    _write(b, bb, {1: _table(pro_b, ang_b, len_b, wid_b)})
    return b


def _settings():
    return Settings(model_path="x", main_directories=[], odor_on_s=30.0, fps_default=40.0)


def test_pre_odor_mask_uses_the_trial_window_or_the_config_default(batch):
    info = _info(batch, "x", "testing", 1, None)
    df = _table(np.zeros(40), np.zeros(40), np.zeros(40), np.zeros(40))
    assert N.pre_odor_mask(df, info, _settings()).sum() == 20
    light = _info(batch, "x", "testing", 9, None, has_odor=False)
    assert N.pre_odor_mask(df, light, _settings()).sum() == 40      # 30 s * 40 fps > 40 frames -> all frames


def test_normalize_batch_writes_stats_and_percent_columns(batch):
    gcfg = GeometrySettings(table_format="csv", rest_min_samples=5)
    out = N.normalize_batch(batch, _settings(), gcfg)
    assert out["slots"] == [1] and out["n_trials"] == 2
    stats = json.loads(C.stats_path(batch / "geometry", 1).read_text())
    assert stats["proboscis"]["global_min"] == pytest.approx(40.0)     # 5 px floored out
    assert stats["proboscis"]["global_max"] == pytest.approx(100.0)
    assert stats["proboscis"]["effective_max"] == pytest.approx(100.0)
    assert stats["abdomen_angle"]["min_deg"] == pytest.approx(120.0)
    # 5 of the 45 angle frames sit at 178 deg, so the 5 % trim of the straight
    # (resting) end lands on 178 anyway -- the trim only bites on a lone outlier.
    assert stats["abdomen_angle"]["max_deg"] == pytest.approx(178.0)
    assert stats["abdomen_angle"]["raw_max"] == pytest.approx(178.0)
    assert stats["abdomen_angle"]["min_at"] == {"trial": "pretest_1_Hexanol", "frame": 25}
    assert stats["abdomen_angle"]["max_at"]["trial"] == "testing_1_Hexanol"
    assert stats["abdomen_angle"]["n_frames"] == 40 + 5
    assert stats["abdomen_length"]["rest_px"] == pytest.approx(np.median(np.r_[np.full(20, 150.0), np.full(20, 152.0)]))
    assert stats["abdomen_length"]["source"] == "pre_odor"
    a = C.read_table_any(batch / "geometry" / "september_02_batch_1_pretest_1" / "september_02_batch_1_pretest_1_fly1_geometry.csv")
    for col in C.NORMALIZED_COLUMNS:
        assert col in a.columns
    assert list(a.columns) == C.PER_FRAME_COLUMNS + C.NORMALIZED_COLUMNS
    assert a[C.PRO_PCT].iloc[0] == pytest.approx(0.0)
    assert a[C.PRO_PCT].iloc[25] == pytest.approx(100.0)
    # Changed 2026-09-10: out-of-range values CLAMP instead of going missing, because a
    # trimmed gmin puts ~5 % of frames below the floor by design. per_min_px still decides
    # which frames define the SCALE, not which frames get a value.
    assert a[C.PRO_PCT].iloc[26] == pytest.approx(0.0), "below the floor -> 0 %, not NaN"
    assert a[C.ABD_ANGLE_PCT].iloc[25] == pytest.approx(100.0)
    assert a[C.ABD_ANGLE_PCT].iloc[0] == pytest.approx(100.0 * (178.0 - 175.0) / (178.0 - 120.0))
    assert a[C.ABD_LEN_PCT].iloc[30] == pytest.approx(100.0 * 180.0 / stats["abdomen_length"]["rest_px"])
    b = C.read_table_any(batch / "geometry" / "september_02_batch_1_testing_1" / "september_02_batch_1_testing_1_fly1_geometry.csv")
    assert b[C.ABD_ANGLE_PCT].iloc[0] == pytest.approx(0.0)
    assert np.isnan(b[C.ABD_ANGLE_PCT].iloc[10])


def test_normalize_is_idempotent_and_keeps_csv_when_switching_to_parquet(batch):
    gcfg = GeometrySettings(table_format="csv", rest_min_samples=5)
    N.normalize_batch(batch, _settings(), gcfg)
    first = json.loads(C.stats_path(batch / "geometry", 1).read_text())
    N.normalize_batch(batch, _settings(), gcfg)
    assert json.loads(C.stats_path(batch / "geometry", 1).read_text()) == first
    N.normalize_batch(batch, _settings(), GeometrySettings(table_format="parquet", rest_min_samples=5))
    d = batch / "geometry" / "september_02_batch_1_pretest_1"
    assert (d / "september_02_batch_1_pretest_1_fly1_geometry.csv").exists()
    assert (d / "september_02_batch_1_pretest_1_fly1_geometry.parquet").exists()


def test_rest_falls_back_to_all_frames_when_pre_odor_is_short(batch):
    stats = N.normalize_batch(batch, _settings(), GeometrySettings(rest_min_samples=1000))["stats"][1]
    assert stats["abdomen_length"]["source"] == "all"


# --------------------------------------------------------------------------- #
# Robust (trimmed) scale endpoints -- user spec 2026-09-10
#
# One mis-tracked frame used to define a whole fly's scale. The RESTING end of each
# signal is dense (the fly sits there most of the time), so a percentile there is
# stable; the SIGNAL end is sparse and real -- above the 99th percentile of PER,
# frames are 2.6x enriched for odor-on -- so it stays at the literal extreme.
#
#   PER    0 % = resting  -> trimmed to the per_trim_pct-th percentile of px
#          100 % = extended -> literal max (never trimmed)
#   ANGLE  0 % = straight (resting) -> trimmed to the (100-angle_trim_pct)-th pct of deg
#          100 % = most curled      -> literal min (never trimmed)
# --------------------------------------------------------------------------- #
def _stats_for(pro, ang, gcfg):
    n = len(pro)
    info = TrialInfo(video=None, batch_dir="/b", dataset="d", trial_name="t", out_dir="/b/geometry/t",
                     trial_type="testing", trial_index=1, odor="Hexanol", label="testing_1_Hexanol",
                     n_frames=n, fps=40.0, has_odor=True, odor_on_frame=n, odor_off_frame=n,
                     light_on_s=None, light_off_s=None, fly_type="GR5a-Old",
                     sidecar_txt=None, sidecar_csv=None, recorded_at=None)
    df = _table(np.asarray(pro, float), np.asarray(ang, float),
                np.full(n, 150.0), np.full(n, 60.0))
    return N.compute_slot_stats(1, [(info, df)], gcfg, _settings())


def test_per_floor_is_trimmed_but_the_ceiling_stays_at_the_literal_max():
    """100 resting frames at 40 px plus one real 120 px extension."""
    pro = np.full(101, 40.0); pro[0] = 30.0; pro[100] = 120.0
    ang = np.full(101, 170.0)
    s = _stats_for(pro, ang, GeometrySettings(table_format="csv", rest_min_samples=1, per_trim_pct=5.0))
    p = s["proboscis"]
    assert p["global_min"] > 30.0, "the 30 px outlier no longer sets the floor"
    assert p["global_min"] == pytest.approx(40.0), "the dense resting level does"
    assert p["global_max"] == pytest.approx(120.0), "the real extension MUST survive the trim"
    assert p["raw_min"] == pytest.approx(30.0) and p["raw_max"] == pytest.approx(120.0)


def test_angle_straight_end_is_trimmed_but_the_curled_end_stays_at_the_literal_min():
    """100 resting frames near straight plus one real deep curl."""
    ang = np.full(101, 150.0); ang[0] = 179.0; ang[100] = 95.0
    pro = np.full(101, 40.0)
    s = _stats_for(pro, ang, GeometrySettings(table_format="csv", rest_min_samples=1, angle_trim_pct=5.0))
    a = s["abdomen_angle"]
    assert a["max_deg"] < 179.0, "the lone 179 deg frame no longer sets 0 %"
    assert a["max_deg"] == pytest.approx(150.0), "the dense resting posture does"
    assert a["min_deg"] == pytest.approx(95.0), "the real curl MUST survive the trim"
    assert a["raw_max"] == pytest.approx(179.0) and a["raw_min"] == pytest.approx(95.0)


def test_a_zero_trim_reproduces_the_untrimmed_literal_extremes():
    pro = np.full(101, 40.0); pro[0] = 30.0; pro[100] = 120.0
    ang = np.full(101, 150.0); ang[0] = 179.0; ang[100] = 95.0
    s = _stats_for(pro, ang, GeometrySettings(table_format="csv", rest_min_samples=1,
                                              per_trim_pct=0.0, angle_trim_pct=0.0))
    assert s["proboscis"]["global_min"] == pytest.approx(30.0)
    assert s["abdomen_angle"]["max_deg"] == pytest.approx(179.0)


def test_frames_below_the_trimmed_per_floor_clamp_to_zero_rather_than_going_missing():
    """The whole point of clamping: the ~5 % of frames under the floor stay in the trace."""
    pro = np.full(101, 40.0); pro[0] = 30.0; pro[100] = 120.0
    ang = np.full(101, 150.0)
    gcfg = GeometrySettings(table_format="csv", rest_min_samples=1, per_trim_pct=5.0)
    s = _stats_for(pro, ang, gcfg)
    df = _table(pro, ang, np.full(101, 150.0), np.full(101, 60.0))
    out = N.apply_slot_stats(df, s, gcfg)
    pct = out[C.PRO_PCT].to_numpy(float)
    assert pct[0] == pytest.approx(0.0), "the sub-floor frame is 0 %, not NaN"
    assert pct[100] == pytest.approx(100.0)
    assert np.isfinite(pct).all()
