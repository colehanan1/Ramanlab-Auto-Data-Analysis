import math

import pytest

from fbpipe.utils.mount_qc import Thresholds
from rigmon.traces.series import build_series, odor_segments


def det(cls, x, y):
    return {"cls": cls, "x": x, "y": y, "w": 40, "h": 40, "r": 0.0, "conf": 0.9}


def frame(i, extend=0.0, drop_prob_of=None):
    dets = []
    for k, y in enumerate((150, 400, 650, 850), start=1):
        dets.append(det("eye", 700, y))
        if k != drop_prob_of:
            dets.append(det("proboscis", 700, y + 80 + (extend if k == 2 else 0)))
    return {"i": i, "dets": dets}


def test_series_px_per_fly_top_to_bottom_with_time_axis():
    frames = [frame(i * 2, extend=(60 if i >= 5 else 0)) for i in range(10)]
    s = build_series(frames, 40.0, gate_px=160.0)
    assert s.n_slots == 4 and sorted(s.flies) == [1, 2, 3, 4]
    assert s.t_s[:3] == pytest.approx([0.0, 0.05, 0.10])
    assert s.flies[1] == pytest.approx([80.0] * 10)
    assert s.flies[2][:5] == pytest.approx([80.0] * 5) and s.flies[2][5:] == pytest.approx([140.0] * 5)


def test_unpaired_proboscis_is_nan():
    s = build_series([frame(0, drop_prob_of=3), frame(2)], 40.0, gate_px=160.0)
    assert math.isnan(s.flies[3][0]) and s.flies[3][1] == pytest.approx(80.0)


def test_proboscis_beyond_gate_is_nan():
    s = build_series([frame(0, extend=200)], 40.0, gate_px=160.0)   # fly 2 at 280 px > 160 gate
    assert math.isnan(s.flies[2][0])


def test_three_flies():
    frames = [{"i": 0, "dets": [d for d in frame(0)["dets"] if d["y"] < 800]}]
    assert build_series(frames, 40.0, gate_px=160.0).n_slots == 3


def test_empty_frames():
    s = build_series([], 40.0, gate_px=160.0)
    assert s.n_slots == 0 and s.t_s == [] and s.flies == {}


def _csv(tmp_path, labels, step_ns=25_000_000):
    p = tmp_path / "t.csv"
    rows = ["FrameNumber,SensorNs,UTC_ISO,ActiveOFM,FrameInterval_s,Deviation_s"]
    for k, lab in enumerate(labels):
        rows.append(f"{k + 1},{1_000_000_000 + k * step_ns},x,{lab},0,0")
    p.write_text("\n".join(rows) + "\n")
    return p


def test_odor_segments_one_block(tmp_path):
    p = _csv(tmp_path, ["off"] * 40 + ["OFM_I"] * 80 + ["off"] * 40)
    assert [(round(a, 3), round(b, 3), c) for a, b, c in odor_segments(p)] == [(1.0, 3.0, "OFM_I")]


def test_odor_segments_two_blocks_and_open_end(tmp_path):
    p = _csv(tmp_path, ["off"] * 4 + ["A"] * 4 + ["off"] * 4 + ["B"] * 4)
    segs = [(round(a, 3), round(b, 3), c) for a, b, c in odor_segments(p)]
    assert segs == [(0.1, 0.2, "A"), (0.3, 0.375, "B")]


def test_odor_segments_none_or_missing(tmp_path):
    assert odor_segments(_csv(tmp_path, ["off"] * 10)) == []
    assert odor_segments(tmp_path / "nope.csv") == []
