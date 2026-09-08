"""Per-trial metadata for a trial whose output dir sits under <batch>/geometry/."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fbpipe.config import Settings
from fbpipe.geometry import trial_meta as TM

STEM = "output_september_02_batch_1_testing_3_IsoamylAcetate_20260902_123456"


def _batch(tmp_path, odor_states=None, n=8, with_sensors=True):
    ds = tmp_path / "flys_Abdomen" / "Hex-Sensitivity-24-0.1"
    batch = ds / "september_02_batch_1_rig_3_copy"
    batch.mkdir(parents=True)
    (batch / "session_metadata.txt").write_text(
        "Session Metadata\n\nSubject\n-------\nFly Type: GR5a-Old\nBorn (approx): 2026-08-29\n"
        "Host: Flybehavior3 | OS: Linux\n")
    (batch / f"{STEM}.mp4").write_bytes(b"\x00")
    (batch / f"{STEM}.txt").write_text(
        "Video Metadata\n==============\nCycle Name: testing_3_IsoamylAcetate\n"
        f"Total Frames Captured: {n}\nAchieved Frame Rate: 40.02\nFrame Resolution: 1152 x 1080\n")
    states = odor_states if odor_states is not None else ["off"] * 3 + ["OFM_A"] * 3 + ["off"] * 2
    rows = []
    for k in range(n):
        rows.append({"FrameNumber": k + 1, "SensorNs": 1000 + k, "UTC_ISO": f"2026-09-02T12:34:56.{k:03d}000Z",
                     "ActiveOFM": states[k]})
    pd.DataFrame(rows).to_csv(batch / f"{STEM}.csv", index=False)
    if with_sensors:
        pd.DataFrame({"MonoNs": [0, 500_000_000, 1_000_000_000, 3_000_000_000],
                      "OFM_Event": ["RECORDING_START", "LIGHT_SOLID_ON", "", "LIGHT_OFF"]}
                     ).to_csv(batch / f"sensors_{STEM}.csv", index=False)
    return batch


def _settings(tmp_path):
    return Settings(model_path="x", main_directories=[str(tmp_path / "flys_Abdomen" / "Hex-Sensitivity-24-0.1")],
                    datasets=("Hex-Sensitivity-24-0.1",), protocol="v2")


def test_trial_name_and_sidecars_come_from_the_video_stem(tmp_path):
    batch = _batch(tmp_path)
    video = batch / f"{STEM}.mp4"
    assert TM.trial_name_from_video(video) == "september_02_batch_1_testing_3"
    txt, csv = TM.sidecar_pair_for_video(video)
    assert txt == batch / f"{STEM}.txt" and csv == batch / f"{STEM}.csv"
    assert TM.recorded_at_from_stem(STEM) == "20260902_123456"
    assert TM.recorded_at_from_stem("output_weird") is None


def test_trial_info_reads_the_sidecars_through_the_nested_out_dir(tmp_path):
    batch = _batch(tmp_path)
    video = batch / f"{STEM}.mp4"
    out_dir = batch / "geometry" / "september_02_batch_1_testing_3"
    info = TM.trial_info_for_video(video, batch, out_dir, _settings(tmp_path))
    assert info.dataset == "Hex-Sensitivity-24-0.1"
    assert Path(info.batch_dir) == batch and info.batch_name == "september_02_batch_1_rig_3_copy"
    assert info.trial_type == "testing" and info.trial_index == 3 and info.odor == "IsoamylAcetate"
    assert info.label == "testing_3_IsoamylAcetate"
    assert info.n_frames == 8 and info.fps == pytest.approx(40.02)
    assert info.has_odor and info.odor_on_frame == 4 and info.odor_off_frame == 7
    assert info.light_on_s == pytest.approx(0.5) and info.light_off_s == pytest.approx(3.0)
    assert info.fly_type == "GR5a-Old"
    assert info.stimulus_window_s == (pytest.approx(4 / 40.02), pytest.approx(7 / 40.02))
    assert not (out_dir / "_trial_meta.json").exists(), "no production cache file inside geometry dirs"
    back = TM.TrialInfo.from_dict(json.loads(json.dumps(info.to_dict())))
    assert back == info


def test_light_only_trial_has_no_odor_and_uses_the_light_window(tmp_path):
    batch = _batch(tmp_path, odor_states=["off"] * 8)
    video = batch / f"{STEM}.mp4"
    out_dir = batch / "geometry" / "september_02_batch_1_testing_3"
    info = TM.trial_info_for_video(video, batch, out_dir, _settings(tmp_path))
    assert not info.has_odor
    assert info.odor_on_frame is None and info.odor_off_frame is None
    assert info.stimulus_window_s == (pytest.approx(0.5), pytest.approx(3.0))


def test_frame_alignment_maps_video_frame_k_to_framenumber_k_plus_one(tmp_path):
    batch = _batch(tmp_path)
    ts, odor = TM.frame_alignment(batch / f"{STEM}.csv", n_frames=8, fps=40.0)
    assert ts.shape == (8,) and odor.shape == (8,)
    assert ts[0] == pytest.approx(0.0) and ts[1] == pytest.approx(0.001)
    assert odor.tolist() == [False, False, False, True, True, True, False, False]
    # a trial longer than its sidecar pads with frame/fps and odor False
    ts2, odor2 = TM.frame_alignment(batch / f"{STEM}.csv", n_frames=10, fps=40.0)
    assert ts2[9] == pytest.approx(9 / 40.0) and not odor2[9]
    ts3, odor3 = TM.frame_alignment(None, n_frames=3, fps=40.0)
    assert ts3.tolist() == [0.0, 0.025, 0.05] and not odor3.any()


def test_protocol_sort_key_orders_chronologically_then_by_phase():
    def info(rec, t, i):
        return TM.TrialInfo(video=None, batch_dir="b", dataset="d", trial_name=f"x_{t}_{i}", out_dir="o",
                            trial_type=t, trial_index=i, odor=None, label=f"{t}_{i}", n_frames=1, fps=40.0,
                            has_odor=True, odor_on_frame=1, odor_off_frame=2, light_on_s=None,
                            light_off_s=None, fly_type="GR5a-Old", sidecar_txt=None, sidecar_csv=None,
                            recorded_at=rec)
    items = [info("20260902_130000", "testing", 1), info("20260902_110000", "pretest", 2),
             info(None, "training", 1), info(None, "pretest", 1)]
    ordered = sorted(items, key=TM.protocol_sort_key)
    assert [(i.trial_type, i.trial_index) for i in ordered] == [
        ("pretest", 1), ("training", 1), ("pretest", 2), ("testing", 1)]
