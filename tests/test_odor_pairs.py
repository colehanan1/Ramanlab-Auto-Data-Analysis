"""fbpipe.utils.odor_pairs: score each odor of an IAA odor-pair trial on its own.

A pair trial (``testing_<N>_<odor1>_then_<odor2>``, 2026-11-13) is odor 1 for 10 s,
1 s of nothing, odor 2 for 10 s, in one 81 s recording. The user's rule
(2026-11-13): give each odor its own model score, each against the trial's
pre-odor block, never the two odors combined. Before scoring, each pair row is
split into two rows:

* odor 1: the label unchanged, the trace cut off where odor 2 reaches the fly
  (odor 2 valve onset + the scorer's 1 s arrival delay);
* odor 2: ``<label>__odor2``, the trace is the pre-odor block [0, odor-1 onset)
  followed by everything from odor 2's valve onset, timed as a normal trial
  (on = odor-1 onset, off = on + odor-2 length).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fbpipe.utils import odor_pairs as op

FPS = 40.0
N = 3250
LABEL = "testing_3_isoamylacetate-1_then_linalool-1"


def _frames_csv(path: Path, first="OFM_A", second="OFM_L", *, n=N, second_frames=400) -> Path:
    """Rig log: odor 1 rows 1203..1602 (frames 1204-1603), off 44 frames, odor 2."""
    frames = np.arange(1, n + 1)
    state = np.array(["off"] * n, dtype=object)

    def put(ch, a, b):
        if ch == "OFM_P":
            state[a:a + 2] = "OFM_P"
            state[b - 2:b] = "OFM_P"
        else:
            state[a:b] = ch
    put(first, 1203, 1603)
    if second_frames:
        put(second, 1647, 1647 + second_frames)
    pd.DataFrame({"FrameNumber": frames, "SensorNs": ((frames - 1) / FPS * 1e9).astype(np.int64),
                  "ActiveOFM": state}).to_csv(path, index=False)
    return path


@pytest.mark.parametrize("first,second", [("OFM_A", "OFM_L"), ("OFM_A", "OFM_A"),
                                          ("OFM_A", "OFM_P"), ("OFM_P", "OFM_A")])
def test_second_odor_window(tmp_path, first, second):
    win = op.second_odor_window_s(_frames_csv(tmp_path / "f.csv", first, second), FPS)
    assert win == pytest.approx((1648 / FPS, 2048 / FPS))      # 41.2 - 51.2 s


def test_no_second_odor_gives_none(tmp_path):
    assert op.second_odor_window_s(_frames_csv(tmp_path / "f.csv", second_frames=0), FPS) is None


def test_is_pair_label():
    assert op.is_pair_label(LABEL)
    assert op.is_pair_label("TESTING_1_IsoamylAcetate-1_then_IsoamylAcetate-1")
    assert not op.is_pair_label("testing_3_isoamylacetate-1")
    assert not op.is_pair_label(LABEL + op.ODOR2_SUFFIX)


def _wide(labels, n=N):
    rng = np.random.default_rng(0)
    rows = []
    for lab in labels:
        r = {"dataset": "IAA-PairFirst-Control-1-1", "fly": "november_13_batch_1_rig_2",
             "fly_number": 1, "trial_type": "testing", "trial_label": lab, "fps": FPS,
             "trial_odor_on_s": 1204 / FPS, "trial_odor_off_s": 1604 / FPS,
             "trial_duration_s": n / FPS, "global_max": 120.0, "trimmed_global_min": 50.0}
        r.update({f"dir_val_{i}": v for i, v in enumerate(rng.normal(60, 3, n))})
        rows.append(r)
    return pd.DataFrame(rows)


def _trace(row) -> np.ndarray:
    cols = sorted([c for c in row.index if c.startswith("dir_val_")], key=lambda c: int(c[8:]))
    return row[cols].to_numpy(dtype=float)


def test_split_pair_rows(tmp_path):
    df = _wide([LABEL, "pretest_2_linalool-1"])
    out, report = op.split_pair_rows(df, lambda row: (1648 / FPS, 2048 / FPS))
    assert list(out["trial_label"]) == [LABEL, LABEL + op.ODOR2_SUFFIX, "pretest_2_linalool-1"]
    assert report == {"pairs": 1, "split": 1, "no_second_odor": []}
    src = _trace(df.iloc[0])

    # single-odor row: untouched
    pd.testing.assert_series_equal(out.iloc[2], df.iloc[1], check_names=False)

    # odor 1: same window, trace cut at odor-2 arrival (41.2 s + 1 s = frame 1688)
    o1 = out.iloc[0]
    t1 = _trace(o1)
    assert o1["trial_odor_on_s"] == pytest.approx(30.1) and o1["trial_odor_off_s"] == pytest.approx(40.1)
    assert o1["trial_duration_s"] == pytest.approx(1688 / FPS)
    np.testing.assert_array_equal(t1[:1688], src[:1688])
    assert np.isnan(t1[1688:]).all()

    # odor 2: pre-odor block [0, 1204) then the trace from odor-2 onset (index 1648)
    o2 = out.iloc[1]
    t2 = _trace(o2)
    n2 = 1204 + (N - 1648)
    assert o2["trial_odor_on_s"] == pytest.approx(1204 / FPS)
    assert o2["trial_odor_off_s"] == pytest.approx((1204 + 400) / FPS)
    assert o2["trial_duration_s"] == pytest.approx(n2 / FPS)
    np.testing.assert_array_equal(t2[:1204], src[:1204])
    np.testing.assert_array_equal(t2[1204:n2], src[1648:])
    assert np.isnan(t2[n2:]).all()
    for k in ("dataset", "fly", "fly_number", "trial_type", "global_max", "trimmed_global_min", "fps"):
        assert o2[k] == o1[k]


def test_split_without_second_odor_keeps_odor1_only():
    df = _wide([LABEL])
    out, report = op.split_pair_rows(df, lambda row: None)
    assert list(out["trial_label"]) == [LABEL]
    assert report["no_second_odor"] == [("IAA-PairFirst-Control-1-1", "november_13_batch_1_rig_2", 1, LABEL)]
    np.testing.assert_array_equal(_trace(out.iloc[0]), _trace(df.iloc[0]))


def test_no_pairs_returns_same_frame():
    df = _wide(["testing_1_hexanol-1"])
    out, report = op.split_pair_rows(df, lambda row: pytest.fail("must not look up"))
    assert report["pairs"] == 0
    pd.testing.assert_frame_equal(out, df)


def test_frames_csv_lookup(tmp_path):
    batch = tmp_path / "IAA-PairFirst-Control-1-1" / "november_13_batch_1_rig_2"
    batch.mkdir(parents=True)
    want = batch / "output_november_13_batch_1_testing_3_IsoamylAcetate-1_then_Linalool-1_20261113_125306.csv"
    want.write_text("x\n")
    (batch / "output_november_13_batch_1_testing_13_IsoamylAcetate-1_then_Citral-1_20261113_135306.csv").write_text("x\n")
    (batch / ("sensors_" + want.name)).write_text("x\n")
    row = pd.Series({"dataset": "IAA-PairFirst-Control-1-1", "fly": "november_13_batch_1_rig_2",
                     "trial_label": LABEL})
    assert op.find_frames_csv(row, [tmp_path / "IAA-PairFirst-Control-1-1"]) == want
    row2 = row.copy(); row2["fly"] = "missing_batch"
    assert op.find_frames_csv(row2, [tmp_path / "IAA-PairFirst-Control-1-1"]) is None
