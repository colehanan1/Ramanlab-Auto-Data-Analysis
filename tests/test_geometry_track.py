"""Per-video tracking with a fake detector: slot freezing, per-frame assembly,
metrics in native pixels, the tip-less fallback, the velocity gate, alignment."""
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from fbpipe.config import Settings
from fbpipe.geometry import columns as C
from fbpipe.geometry import track as T
from fbpipe.geometry.config import GeometrySettings
from fbpipe.geometry.detect import id_to_label_map
from fbpipe.geometry.trial_meta import TrialInfo
from fbpipe.utils.fly_geometry import ABDOMEN, ABDOMEN_TIP, EYE, PROBOSCIS, Part
from test_fly_geometry import box, normal_fly, EYE_SCALE

NAMES = {0: "abdomen", 1: "abdomenTip", 2: "eye", 3: "proboscis"}
CLS = {ABDOMEN: 0, ABDOMEN_TIP: 1, EYE: 2, PROBOSCIS: 3}
W, H = 1152, 1080          # native frame; model frame is 1080x1080 (sx = 1152/1080)
N_FRAMES = 40
MARK = 5                    # frame index is stored as a grey block of value idx*MARK


class _OBB:
    def __init__(self, parts):
        self.xyxyxyxy = np.asarray([p.pts for p in parts], dtype=np.float32).reshape(-1, 4, 2)
        self.cls = np.asarray([CLS[p.label] for p in parts], dtype=np.float32)
        self.conf = np.asarray([p.score for p in parts], dtype=np.float32)

    def __len__(self):
        return len(self.cls)


class _Result:
    def __init__(self, parts):
        self.obb = _OBB(parts) if parts else None


def scene(idx):
    """Model-space detections for frame ``idx`` (three stacked flies + events)."""
    parts = []
    for slot, y in ((1, 200.0), (2, 500.0), (3, 800.0)):
        fly = normal_fly(eye_c=(300.0, y), scale=EYE_SCALE)
        if slot == 3 and idx >= 10:                       # fly 3 loses its tip
            fly = [p for p in fly if p.label != ABDOMEN_TIP]
        if slot == 1 and idx >= 30:                       # fly 1 curls its tip toward the proboscis side (+y)
            eye = np.array([300.0, y])
            fly = [p for p in fly if p.label != ABDOMEN_TIP]
            fly.append(Part(ABDOMEN_TIP, box(eye + np.array([5.9 * EYE_SCALE, 1.4 * EYE_SCALE]), 0.93 * EYE_SCALE)))
        if slot == 2 and idx == 20:                       # fly 2's proboscis jumps 90 px for one frame
            # (perpendicular: stays inside the eye-relative gate, so only the velocity gate can catch it)
            fly = [p if p.label != PROBOSCIS else Part(PROBOSCIS, p.pts + np.array([0.0, 90.0])) for p in fly]
        parts += [Part(p.label, p.pts, 0.9) for p in fly]
    if idx == 3:                                          # spurious bare eye, one frame only
        parts.append(Part(EYE, box((1040.0, 120.0), EYE_SCALE), 0.4))
    return parts


def fake_predict(images, conf):
    out = []
    for im in images:
        assert im.shape[:2] == (1080, 1080), "frames must reach the model at 1080x1080"
        idx = int(round(float(im[10:30, 10:30, 0].mean()) / MARK))
        out.append(_Result(scene(idx)))
    return out


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "output_september_02_batch_1_testing_3_Hexanol_20260902_120000.avi"
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 40.0, (W, H))
    assert wr.isOpened()
    for idx in range(N_FRAMES):
        fr = np.zeros((H, W, 3), np.uint8)
        fr[0:40, 0:40] = idx * MARK
        wr.write(fr)
    wr.release()
    rows = [{"FrameNumber": k + 1, "UTC_ISO": f"2026-09-02T12:00:00.{k * 25000:06d}Z",
             "ActiveOFM": "OFM_H" if 10 <= k < 20 else "off"} for k in range(N_FRAMES)]
    csv = path.with_suffix(".csv")
    pd.DataFrame(rows).to_csv(csv, index=False)
    return path


def _info(video):
    return TrialInfo(video=str(video), batch_dir=str(video.parent), dataset="Hex-Sensitivity-24-0.1",
                     trial_name="september_02_batch_1_testing_3", out_dir=str(video.parent / "geometry" / "t"),
                     trial_type="testing", trial_index=3, odor="Hexanol", label="testing_3_Hexanol",
                     n_frames=N_FRAMES, fps=40.0, has_odor=True, odor_on_frame=10, odor_off_frame=20,
                     light_on_s=None, light_off_s=None, fly_type="GR5a-Old", sidecar_txt=None,
                     sidecar_csv=str(video.with_suffix(".csv")), recorded_at="20260902_120000")


def _settings():
    return Settings(model_path="x", main_directories=[], conf_thres=0.25, max_flies=5, inference_batch_size=7,
                    allow_cpu=True, fps_default=40.0)


def _gcfg(**kw):
    return GeometrySettings(warmup_frames=5, annotated_video=False, **kw)


def test_freeze_slots_two_pass_sorted_and_presence_filtered():
    vg = T.VideoGeometry(W, H, 1080, 1080, measure_native=True)
    warm = [scene(i) for i in range(5)]
    frozen, axis, chir = T.freeze_slots(warm, vg, _gcfg().limits, max_flies=5, presence_min=0.5)
    assert [f.slot for f in frozen] == [1, 2, 3]
    ys = [f.eye_model.centroid[1] for f in frozen]
    assert ys == sorted(ys) and ys[0] == pytest.approx(200.0, abs=1.0)
    assert all(f.presence == 1.0 for f in frozen)
    assert frozen[0].eye_report.centroid[0] == pytest.approx(300.0 * W / 1080, abs=1.0)
    assert axis is not None and abs(math.degrees(math.atan2(axis[1], axis[0]))) < 5.0
    assert chir == -1.0
    assert T.freeze_slots([[]], vg, _gcfg().limits, max_flies=5, presence_min=0.5)[0] == []


def test_track_video_end_to_end(video):
    res = T.track_video(video, _info(video), predict_fn=fake_predict, id_to_label=id_to_label_map(NAMES),
                        settings=_settings(), gcfg=_gcfg())
    assert sorted(res.tables) == [1, 2, 3]
    df1, df2, df3 = (res.tables[s] for s in (1, 2, 3))
    for df in (df1, df2, df3):
        assert list(df.columns) == C.PER_FRAME_COLUMNS
        assert len(df) == N_FRAMES and df[C.FRAME].tolist() == list(range(N_FRAMES))
    # frozen eye in native pixels, constant
    assert df1[C.EYE_X].nunique() == 1 and df1[C.EYE_X].iloc[0] == pytest.approx(300.0 * W / 1080, abs=1.0)
    assert df1[C.EYE_Y].iloc[0] == pytest.approx(200.0, abs=1.0)
    # alignment: frame k <-> FrameNumber k+1; odor on for frames 10..19
    assert df1[C.TIMESTAMP].iloc[1] == pytest.approx(0.025, abs=1e-6)
    assert df1[C.ODOR_ON].tolist() == [10 <= k < 20 for k in range(N_FRAMES)]
    # straight flies: interior angle 180, bend 0, box size in native px (x-scaled length)
    assert df2[C.ABD_ANGLE].iloc[0] == pytest.approx(180.0, abs=0.5)
    assert df2[C.ABD_BEND].iloc[0] == pytest.approx(0.0, abs=0.5)
    assert df2[C.ABD_LEN].iloc[0] == pytest.approx(3.4 * EYE_SCALE * W / 1080, rel=0.02)
    assert df2[C.ABD_WID].iloc[0] == pytest.approx(2.0 * EYE_SCALE, rel=0.02)
    assert (df2[C.STATUS].iloc[:10] == C.STATUS_OK).all()
    # fly 1 curls toward its proboscis side from frame 30: angle < 180, bend > 0
    assert df1[C.ABD_ANGLE].iloc[35] < 170.0 and df1[C.ABD_BEND].iloc[35] > 5.0
    assert (df1[C.STATUS].iloc[30:] == C.STATUS_OK).all()
    # fly 3 loses its tip from frame 10: box still recorded, angle NaN, status says why
    assert df3[C.ABD_ANGLE].iloc[5] == pytest.approx(180.0, abs=0.5)
    assert np.isnan(df3[C.ABD_ANGLE].iloc[15]) and np.isnan(df3[C.TIP_X].iloc[15])
    assert np.isfinite(df3[C.ABD_LEN].iloc[15]) and df3[C.STATUS].iloc[15] == C.STATUS_ABDOMEN_ONLY
    # fly 2's 90 px proboscis jump is blanked by the velocity gate, the next frame is kept
    assert np.isnan(df2[C.PRO_X].iloc[20]) and df2[C.PRO_STATUS].iloc[20] == C.PRO_VELOCITY
    assert np.isfinite(df2[C.PRO_X].iloc[21]) and df2[C.PRO_STATUS].iloc[21] == C.PRO_OK
    assert df2[C.PRO_DIST].iloc[0] == pytest.approx(1.29 * EYE_SCALE, rel=0.08)
    # meta
    m = res.meta
    assert m["n_frames_processed"] == N_FRAMES and m["frame"]["width"] == W and m["frame"]["sx"] == pytest.approx(W / 1080)
    assert [s["slot"] for s in m["slots"]] == [1, 2, 3] and m["chirality"] == -1.0
    assert m["tracked_fraction"]["3"]["abdomen_tip"] == pytest.approx(10 / N_FRAMES)
    assert m["tracked_fraction"]["2"]["proboscis"] == pytest.approx((N_FRAMES - 1) / N_FRAMES)
    json.dumps(m)   # must be JSON-serialisable


def test_track_video_model_space_and_max_frames(video):
    res = T.track_video(video, _info(video), predict_fn=fake_predict, id_to_label=id_to_label_map(NAMES),
                        settings=_settings(), gcfg=_gcfg(measure_in="model"), max_frames=12)
    df = res.tables[1]
    assert len(df) == 12 and df[C.EYE_X].iloc[0] == pytest.approx(300.0, abs=1.0)
    assert res.meta["n_frames_processed"] == 12


def test_track_video_with_no_flies_returns_no_tables(video):
    def nothing(images, conf):
        return [_Result([]) for _ in images]
    res = T.track_video(video, _info(video), predict_fn=nothing, id_to_label=id_to_label_map(NAMES),
                        settings=_settings(), gcfg=_gcfg())
    assert res.tables == {} and res.frozen == [] and res.meta["n_frames_processed"] == 0


def test_slot_palette_matches_the_lab_per_fly_colours():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "analysis"))
    import light_trial_traces as lt
    from matplotlib.colors import to_rgb
    for slot, name in lt.FLY_COLORS.items():
        r, g, b = (int(round(v * 255)) for v in to_rgb(name))
        assert T.SLOT_COLORS_BGR[slot] == (b, g, r)
