"""trial_timing_qc: flag trials whose timing the pipeline would misread, and say so.

Two failures from 2026-11-05 motivate it:
* Paraffin-Sensitivity-1-100/november_05_batch_1 testing_3 (Citral): the camera
  stalled 8.4 s, the sidecar's "achieved" fps fell to 36.27 and the odor window
  was stored 3 s late (33-66 s).
* every Paraffin Oil trial: the master valve's 50 ms clicks were read as the odor
  turning off, so the stored window was 0.05 s long.
Report-only: each run upserts one row per trial into trial_timing_qc.csv,
rewrites trial_timing_qc_flagged.csv (with a suggested fix) and sends one ntfy
alert for trials *newly* flagged.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from fbpipe import pipeline
from fbpipe.steps import trial_timing_qc as qc

FPS = 40.0


def _frames_csv(path: Path, *, n=3600, on=(1196, 2396), clicks=False, stall=None,
                reopen=None) -> Path:
    """A rig per-frame log: 40 fps clock, valve open over ``on`` (frame numbers),
    optional 50 ms paraffin clicks, a ``stall=(frame, seconds)`` clock gap, or a
    second opening ``reopen=(start, end)``."""
    frames = np.arange(1, n + 1)
    t = (frames - 1) / FPS
    if stall:
        t = t + np.where(frames >= stall[0], stall[1], 0.0)
    state = np.array(["off"] * n, dtype=object)
    if on:
        state[on[0] - 1:on[1]] = "OFM_P" if clicks else "OFM_C"
        if clicks:   # ON/OFF/ON at onset, OFF/ON/OFF at offset, 2 frames apart
            state[on[0] - 1 + 1:on[0] - 1 + 3] = "off"
            state[on[1]:on[1] + 2] = "off"
            state[on[1] + 2:on[1] + 4] = "OFM_P"
    if reopen:
        state[reopen[0] - 1:reopen[1]] = "OFM_C"
    interval = np.diff(t, prepend=t[0])
    pd.DataFrame({"FrameNumber": frames, "SensorNs": (t * 1e9).astype(np.int64),
                  "UTC_ISO": "", "ActiveOFM": state, "FrameInterval_s": interval,
                  "Deviation_s": 0.0}).to_csv(path, index=False)
    return path


def test_a_clean_trial_passes(tmp_path):
    row = qc.assess_frames_csv(_frames_csv(tmp_path / "output_b_testing_1_Citral-1_x.csv"))
    assert row["status"] == "assessed" and not row["flag"]
    assert row["valve_on_s"] == pytest.approx(29.875)
    assert row["pipeline_on_s"] == pytest.approx(row["valve_on_s"])


def test_a_camera_stall_is_flagged_with_where_it_hit(tmp_path):
    row = qc.assess_frames_csv(_frames_csv(tmp_path / "output_b_testing_3_Citral-1_x.csv",
                                           n=3261, stall=(2604, 8.37)))
    assert row["flag"]
    assert row["max_gap_s"] == pytest.approx(8.395, abs=0.01)
    assert row["stall_where"] == "after odor"
    assert "camera stalled" in row["reason"] and row["suggested_fix"]


def test_a_stall_inside_the_odor_window_says_so(tmp_path):
    row = qc.assess_frames_csv(_frames_csv(tmp_path / "output_b_testing_3_Citral-1_x.csv",
                                           stall=(1800, 3.0)))
    assert row["flag"] and row["stall_where"] == "during odor"


def test_paraffin_clicks_are_read_as_one_window(tmp_path):
    row = qc.assess_frames_csv(_frames_csv(
        tmp_path / "output_b_pretest_1_ParaffinOil-100_x.csv", clicks=True))
    assert not row["flag"], row["reason"]
    assert row["pipeline_off_s"] == pytest.approx(row["valve_off_s"], abs=0.2)


def test_click_only_paraffin_is_read_as_one_window(tmp_path):
    # 2026-11-10 protocol: the master stays OFF (= paraffin flowing) and is
    # only clicked ON for 50 ms at onset and again 30 s later at offset.
    p = _frames_csv(tmp_path / "output_b_training_1_ParaffinOil-100_x.csv", on=None)
    df = pd.read_csv(p)
    df.loc[1195:1196, "ActiveOFM"] = "OFM_P"    # frames 1196-1197: onset click
    df.loc[2395:2396, "ActiveOFM"] = "OFM_P"    # frames 2396-2397: offset click
    df.to_csv(p, index=False)
    row = qc.assess_frames_csv(p)
    assert not row["flag"], row["reason"]
    assert row["valve_off_s"] - row["valve_on_s"] == pytest.approx(30.05, abs=0.03)
    assert row["pipeline_on_s"] == pytest.approx(row["valve_on_s"])
    assert row["pipeline_off_s"] == pytest.approx(row["valve_off_s"])


def test_a_valve_pattern_the_parser_misreads_is_flagged(tmp_path):
    # the valve closes for 2 s mid-odor and reopens: the parser ends the window
    # at the first close, the valve log runs to the last
    row = qc.assess_frames_csv(_frames_csv(tmp_path / "output_b_testing_2_Hexanol-0.1_x.csv",
                                           on=(1196, 1700), reopen=(1781, 2396)))
    assert row["flag"]
    assert "odor window" in row["reason"]
    assert "parse_active_ofm" in row["suggested_fix"]


def test_a_camera_clock_that_runs_backwards_is_flagged_plainly(tmp_path):
    p = _frames_csv(tmp_path / "output_b_training_4_Hexanol-0.1_x.csv")
    df = pd.read_csv(p)
    df.loc[2500:, "SensorNs"] -= int(19_100e9)      # clock reset after the odor
    df.to_csv(p, index=False)
    row = qc.assess_frames_csv(p)
    assert row["flag"]
    assert "camera clock went backwards" in row["reason"]
    assert "-19" not in row["reason"]               # no negative-time window shown


def test_a_light_only_trial_has_no_odor_and_is_not_flagged(tmp_path):
    row = qc.assess_frames_csv(_frames_csv(tmp_path / "output_b_testing_9_LightOnly_x.csv",
                                           on=None))
    assert row["status"] == "no_odor" and not row["flag"]


# ── the step: report, flagged list, one alert per newly flagged trial ─────


def _cfg(root: Path):
    return SimpleNamespace(main_directory=str(root), main_directories=[str(root)],
                           parallel=SimpleNamespace(enabled=False, n_jobs=1))


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    root = tmp_path / "Hex-Sensitivity-1-0.1"
    batch = root / "november_05_batch_1"
    batch.mkdir(parents=True)
    _frames_csv(batch / "output_november_05_batch_1_testing_1_Citral-1_20261105_1.csv")
    _frames_csv(batch / "output_november_05_batch_1_testing_3_Citral-1_20261105_2.csv",
                n=3261, stall=(2604, 8.37))
    (batch / "sensors_output_november_05_batch_1_testing_1_Citral-1_20261105_1.csv").write_text("x\n")
    monkeypatch.setattr(qc, "get_main_directories", lambda cfg: [root])
    monkeypatch.setattr(qc, "iter_live_batch_dirs", lambda cfg, r: [batch])
    sent = []
    monkeypatch.setattr(qc, "ntfy_notify", lambda title, msg, **kw: sent.append((title, msg)))
    return root, batch, sent


def test_the_step_reports_flags_and_alerts_once(dataset, tmp_path):
    root, batch, sent = dataset
    report_dir = tmp_path / "qc"
    qc.run(_cfg(root), report_dir=report_dir)
    report = pd.read_csv(report_dir / qc.REPORT_FILENAME)
    assert len(report) == 2                      # sensors_* logs are not trials
    flagged = pd.read_csv(report_dir / qc.FLAGGED_FILENAME)
    assert list(flagged["trial"]) == ["testing_3_Citral-1"]
    assert flagged["dataset"].iloc[0] == "Hex-Sensitivity-1-0.1"
    assert len(sent) == 1 and "testing_3_Citral-1" in sent[0][1]

    qc.run(_cfg(root), report_dir=report_dir)    # nothing changed: no new alert
    assert len(sent) == 1


def test_a_trial_fixed_on_disk_leaves_the_flagged_list(dataset, tmp_path):
    root, batch, sent = dataset
    report_dir = tmp_path / "qc"
    qc.run(_cfg(root), report_dir=report_dir)
    bad = batch / "output_november_05_batch_1_testing_3_Citral-1_20261105_2.csv"
    _frames_csv(bad)                                 # e.g. a corrected log
    os.utime(bad, (bad.stat().st_mtime + 10, bad.stat().st_mtime + 10))
    qc.run(_cfg(root), report_dir=report_dir)
    assert pd.read_csv(report_dir / qc.FLAGGED_FILENAME).empty


def test_the_step_never_raises(monkeypatch, capsys):
    monkeypatch.setattr(qc, "run", lambda cfg, **kw: (_ for _ in ()).throw(OSError("disk")))
    qc.main(SimpleNamespace())
    assert "[TRIAL-TIMING] ERROR" in capsys.readouterr().out


def test_registered_after_eye_drift_qc_in_both_import_branches():
    names = [s.name for s in pipeline.ORDERED_STEPS]
    assert names[names.index("eye_drift_qc") + 1] == "trial_timing_qc"
    assert pipeline.STEP_REGISTRY["trial_timing_qc"].runner is qc.main
    src = Path(pipeline.__file__).read_text()
    assert src.count("        trial_timing_qc,\n") == 2


def test_the_report_lives_with_the_pipeline_figures():
    assert "New-Opto-Fly-Figures" in str(qc.DEFAULT_REPORT_DIR)
    assert "Results/Figures" not in str(qc.DEFAULT_REPORT_DIR)
