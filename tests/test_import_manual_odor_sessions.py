"""Importer for Caleb's manual-odor Pi sessions (manual_odor_pi.py).

Pins the events.csv -> pipeline-layout translation: which trials are training
vs testing, where odor onset falls inside each video (the camera's first frame
lands ~1.2 s after the logged ``recording_start``), the rig-style file names
``trial_folder_name`` needs, and that a session without a finished test phase
is a pilot.
"""
import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import import_manual_odor_sessions as imp  # noqa: E402
from fbpipe.utils.trial_metadata import load_trial_metadata, parse_active_ofm, parse_sidecar  # noqa: E402
from fbpipe.utils.yolo_discovery import trial_folder_name  # noqa: E402


def _write_events(path: Path, rows):
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["wall_time", "elapsed_session_s", "event", "detail"])
        for t, ev, detail in rows:
            w.writerow([f"2026-11-18T15:00:00.000-06:00", f"{t:.3f}", ev, detail])


def _full_session(tmp_path: Path) -> Path:
    """1 no-odor recording, 2 IAA training trials, then a 2-odor test."""
    s = tmp_path / "Test_2026-11-18_02-53-54PM"
    s.mkdir()
    rows = [
        (0.07, "camera_preview_start", "camera_0"),
        (1.0, "session_start", "Test; live preview"),
        (100.0, "recording_start", "No odor; trial_001_camera_0.mp4"),
        (100.0, "recording_start", "No odor; trial_001_camera_1.mp4"),
        (101.0, "manual_recording_on", "No odor"),
        (200.0, "manual_recording_off", "No odor; manual stop"),
        (200.3, "recording_stop", "trial_001_camera_0.mp4"),
        (300.0, "recording_start", "Isoamyl acetate; trial_002_camera_0.mp4"),
        (300.0, "recording_start", "Isoamyl acetate; trial_002_camera_1.mp4"),
        (301.1, "odor_on", "Isoamyl acetate"),
        (331.1, "odor_off", "Isoamyl acetate; 30 s timeout"),
        (331.4, "recording_stop", "trial_002_camera_0.mp4"),
        (600.0, "recording_start", "Isoamyl acetate; trial_003_camera_0.mp4"),
        (601.1, "odor_on", "Isoamyl acetate"),
        (631.1, "odor_off", "Isoamyl acetate; 30 s timeout"),
        (631.4, "recording_stop", "trial_003_camera_0.mp4"),
        (700.0, "training_complete", "30-minute wait starts now"),
        (700.0, "testing_order", "1: Isoamyl acetate (BCM 25) | 2: 3-Octanol (BCM 12)"),
        (2500.0, "recording_start", "Isoamyl acetate; trial_004_camera_0.mp4"),
        (2503.0, "testing_baseline_start", "Isoamyl acetate; onset delay 0.000 s"),
        (2533.0, "testing_trial", "1/7; Isoamyl acetate"),
        (2533.0, "odor_on", "Isoamyl acetate"),
        (2563.0, "odor_off", "Isoamyl acetate; 30 s timeout"),
        (2593.0, "testing_post_end", "Isoamyl acetate"),
        (2593.3, "recording_stop", "trial_004_camera_0.mp4"),
        (2830.0, "recording_start", "3-Octanol; trial_005_camera_0.mp4"),
        (2863.0, "testing_trial", "2/7; 3-Octanol"),
        (2863.0, "odor_on", "3-Octanol"),
        (2893.0, "odor_off", "3-Octanol; 30 s timeout"),
        (2923.0, "testing_post_end", "3-Octanol"),
        (2923.3, "recording_stop", "trial_005_camera_0.mp4"),
        (2924.0, "testing_complete", "All 7 odors completed; closing session"),
        (2924.5, "session_end", ""),
    ]
    _write_events(s / "events.csv", rows)
    return s


def test_parse_assigns_phases_and_skips_no_odor(tmp_path):
    trials = imp.parse_events(_full_session(tmp_path) / "events.csv")
    assert [(t.number, t.phase, t.phase_index, t.odor) for t in trials] == [
        (2, "training", 1, "IsoamylAcetate"),
        (3, "training", 2, "IsoamylAcetate"),
        (4, "testing", 1, "IsoamylAcetate"),
        (5, "testing", 2, "3-Octonol"),
    ]


def test_session_status(tmp_path):
    s = _full_session(tmp_path)
    info = imp.summarize_session(s)
    assert info.complete is True
    assert info.trained_odor == "IsoamylAcetate"
    assert info.start.date().isoformat() == "2026-11-18"


def test_cancelled_test_phase_is_pilot(tmp_path):
    s = tmp_path / "Test_2026-11-09_06-00-59PM"
    s.mkdir()
    _write_events(s / "events.csv", [
        (1.0, "session_start", "Test"),
        (10.0, "recording_start", "Ethyl butyrate; trial_001_camera_0.mp4"),
        (11.1, "odor_on", "Ethyl butyrate"),
        (41.1, "odor_off", "Ethyl butyrate; 30 s timeout"),
        (50.0, "training_complete", "30-minute wait starts now"),
        (60.0, "testing_cancelled", "Esc / stop recording"),
    ])
    assert imp.summarize_session(s).complete is False


def test_odor_frames_end_aligned():
    # Video of 1840 frames @ 20 fps (91.95 s by frame count) ends at the stop
    # request (testing_post_end); onset is 30 s + latency after recording_start.
    t = imp.Trial(number=7, odor="3-Octonol", phase="testing", phase_index=1,
                  rec_start=6459.739, odor_on=6492.721, odor_off=6522.788,
                  stop_request=6552.871)
    on_f, off_f, first = imp.odor_frames(t, n_frames=1840, fps=20.0)
    assert first == pytest.approx(6552.871 - 1840 / 20.0, abs=1e-6)
    assert on_f == round((6492.721 - first) * 20.0)  # ~ frame 637 (31.9 s)
    assert off_f == round((6522.788 - first) * 20.0)
    assert 630 < on_f < 645


def test_training_onset_before_first_frame_clamps_to_zero():
    t = imp.Trial(number=2, odor="IsoamylAcetate", phase="training", phase_index=1,
                  rec_start=3258.607, odor_on=3259.712, odor_off=3289.778,
                  stop_request=3289.778)
    on_f, off_f, first = imp.odor_frames(t, n_frames=599, fps=20.0)
    assert first > t.odor_on  # the camera starts after the valve opens
    assert on_f == 0
    assert off_f == 599


def test_written_files_parse_with_pipeline_readers(tmp_path):
    t = imp.Trial(number=7, odor="3-Octonol", phase="testing", phase_index=1,
                  rec_start=100.0, odor_on=133.0, odor_off=163.0, stop_request=193.0)
    stem = imp.output_stem("november_18_batch_1", t, "20261118_164134")
    assert trial_folder_name(stem + ".mp4") == "november_18_batch_1_testing_1"
    batch = tmp_path / "november_18_batch_1"
    batch.mkdir()
    imp.write_sidecars(batch, stem, t, n_frames=1840, fps=20.0, width=1280, height=960,
                       start_utc="2026-11-18T22:41:35Z", fly_id="Test_2026-11-18_02-53-54PM",
                       source_video="/x/trial_007_camera_0.mp4")
    trial_dir = batch / "november_18_batch_1_testing_1"
    trial_dir.mkdir()
    fields = parse_sidecar(batch / f"{stem}.txt")
    assert fields["cycle_name"] == "testing_1_3-Octonol"
    assert fields["total_frames_captured"] == "1840"
    on_f, off_f, chan = parse_active_ofm(batch / f"{stem}.csv")
    exp_on, exp_off, _ = imp.odor_frames(t, n_frames=1840, fps=20.0)
    assert (on_f, off_f) == (exp_on + 1, exp_off + 1)  # FrameNumber is 1-based like the rig
    assert chan != "OFM_P"
    # End to end through the loader the pipeline steps call.
    meta = load_trial_metadata(trial_dir, use_cache=False, known_datasets=())
    assert meta.fps == 20.0 and meta.n_frames == 1840
    assert meta.trial_type == "testing" and meta.trial_index == 1
    assert meta.odor == "3-Octonol"
    assert (meta.odor_on_frame, meta.odor_off_frame) == (exp_on + 1, exp_off + 1)


def test_batch_names_number_sessions_per_day(tmp_path):
    from datetime import datetime
    starts = [datetime(2026, 11, 18, 14, 53), datetime(2026, 11, 18, 9, 0), datetime(2026, 11, 9, 18, 0)]
    assert imp.batch_names(starts) == [
        "november_18_batch_2", "november_18_batch_1", "november_09_batch_1"]


def test_v2_training_trial_ends_at_manual_post_end(tmp_path):
    # manual_odor_pi_v2: baseline -> odor -> 30 s post in one video.
    s = tmp_path / "Fly03_2026-11-20_02-00-00PM"
    s.mkdir()
    _write_events(s / "events.csv", [
        (1.0, "session_start", "Fly03"),
        (100.0, "recording_start", "Isoamyl acetate; trial_001_camera_0.mp4"),
        (101.0, "manual_baseline_start", "Isoamyl acetate; 30 s"),
        (131.0, "manual_baseline_end", "Isoamyl acetate"),
        (131.0, "odor_on", "Isoamyl acetate"),
        (161.0, "odor_off", "Isoamyl acetate; 30 s timeout"),
        (161.0, "manual_post_start", "Isoamyl acetate"),
        (191.0, "manual_post_end", "Isoamyl acetate"),
        (191.3, "recording_stop", "trial_001_camera_0.mp4"),
    ])
    (t,) = imp.parse_events(s / "events.csv")
    assert t.phase == "training" and t.stop_request == 191.0
    on_f, off_f, first = imp.odor_frames(t, n_frames=int(89.8 * 20), fps=20.0)
    assert first < t.odor_on                 # a real baseline now exists
    assert on_f == round((131.0 - first) * 20)


def test_session_still_arriving_is_not_ready(tmp_path):
    # A push from the Pi lands events.csv first; videos follow (and an
    # interrupted one waits in .rsync-partial). Importing then would freeze a
    # half batch forever (batch folders are never overwritten).
    s = _full_session(tmp_path)
    info = imp.summarize_session(s)
    assert imp.session_ready(info) is False            # no videos yet
    for t in info.trials:
        (s / f"trial_{t.number:03d}_camera_0.mp4").write_bytes(b"x")
    assert imp.session_ready(info) is True
    (s / ".rsync-partial").mkdir()
    assert imp.session_ready(info) is False


def _light_session(tmp_path: Path) -> Path:
    s = tmp_path / "Fly03_2026-11-20_02-00-00PM"
    s.mkdir()
    _write_events(s / "events.csv", [
        (1.0, "session_start", "Fly03"),
        (100.0, "recording_start", "Isoamyl acetate; trial_001_camera_0.mp4"),
        (131.0, "odor_on", "Isoamyl acetate"),
        (161.0, "odor_off", "Isoamyl acetate; 30 s timeout"),
        (191.0, "manual_post_end", "Isoamyl acetate"),
        (191.3, "recording_stop", "trial_001_camera_0.mp4"),
        (200.0, "training_complete", "30-minute wait starts now"),
        (2000.0, "recording_start", "Isoamyl acetate; trial_002_camera_0.mp4"),
        (2033.0, "odor_on", "Isoamyl acetate"),
        (2063.0, "odor_off", "Isoamyl acetate; 30 s timeout"),
        (2093.0, "testing_post_end", "Isoamyl acetate"),
        (2093.3, "recording_stop", "trial_002_camera_0.mp4"),
        (2095.0, "recording_start", "LightOnly; trial_003_camera_0.mp4"),
        (2128.0, "testing_trial", "2/2; LightOnly"),
        (2128.0, "light_on", "LightOnly; 15 % on BCM 13; 10 s"),
        (2138.0, "light_off", "LightOnly; 10 s timeout"),
        (2168.0, "testing_post_end", "LightOnly"),
        (2168.3, "recording_stop", "trial_003_camera_0.mp4"),
        (2169.0, "testing_complete", "All 2 test trials completed"),
    ])
    return s


def test_light_probe_is_a_lightonly_testing_trial(tmp_path):
    trials = imp.parse_events(_light_session(tmp_path) / "events.csv")
    assert [(t.phase, t.phase_index, t.odor) for t in trials] == [
        ("training", 1, "IsoamylAcetate"), ("testing", 1, "IsoamylAcetate"),
        ("testing", 2, "LightOnly")]
    light = trials[-1]
    assert (light.odor_on, light.odor_off, light.stop_request) == (2128.0, 2138.0, 2168.0)
    info = imp.summarize_session(tmp_path / "Fly03_2026-11-20_02-00-00PM")
    assert info.trained_odor == "IsoamylAcetate" and info.complete


def test_light_probe_window_reaches_the_pipeline_via_sensors_log(tmp_path):
    t = imp.Trial(number=3, odor="LightOnly", phase="testing", phase_index=8,
                  rec_start=2095.0, odor_on=2128.0, odor_off=2138.0, stop_request=2168.0)
    batch = tmp_path / "november_20_batch_1"
    batch.mkdir()
    stem = imp.output_stem("november_20_batch_1", t, "20261120_144000")
    imp.write_sidecars(batch, stem, t, n_frames=1460, fps=20.0, width=1280, height=960,
                       start_utc="2026-11-20T20:40:00Z", fly_id="Fly03", source_video="/x.mp4")
    assert (batch / f"sensors_{stem}.csv").exists()
    trial_dir = batch / "november_20_batch_1_testing_8"
    trial_dir.mkdir()
    meta = load_trial_metadata(trial_dir, use_cache=False, known_datasets=())
    first = 2168.0 - 1460 / 20.0
    assert meta.odor == "LightOnly" and meta.trial_type == "testing"
    assert meta.light_on_s == pytest.approx(2128.0 - first)
    assert meta.light_off_s == pytest.approx(2138.0 - first)
    assert meta.odor_on_frame == round((2128.0 - first) * 20.0)
    assert meta.odor_off_frame == round((2138.0 - first) * 20.0)
