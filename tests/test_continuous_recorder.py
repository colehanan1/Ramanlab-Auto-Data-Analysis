"""Unit tests for PiCode/continuous_recorder.py — the 20 fps between-trial
recorder. picamera2 is imported lazily by the module, so it runs off-rig; the
real encoder subclass is exercised against a stand-in ``picamera2.encoders``."""

import csv
import sys
import threading
import types
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "PiCode"))

import continuous_recorder as cr  # noqa: E402


# ── between_window ───────────────────────────────────────────────────────────

def test_between_window_guards_both_ends():
    assert cr.between_window(219.0, 2.0) == (2.0, 217.0)
    assert cr.between_window(1730.0, 2.0) == (2.0, 1728.0)


def test_between_window_exact_minimum_records():
    # 2*2 + 1 = 5 s is the shortest gap that still records 1 s
    assert cr.between_window(5.0, 2.0) == (2.0, 3.0)


@pytest.mark.parametrize("gap", [0.0, 1.0, 4.99])
def test_between_window_short_gap_is_none(gap):
    assert cr.between_window(gap, 2.0) is None


def test_between_window_min_len_is_tunable():
    assert cr.between_window(10.0, 2.0, min_len_s=7.0) is None
    assert cr.between_window(11.0, 2.0, min_len_s=7.0) == (2.0, 9.0)


# ── decimation ───────────────────────────────────────────────────────────────

def test_frame_decimator_keeps_every_other_starting_with_first():
    d = cr.FrameDecimator(2)
    assert [d.keep() for _ in range(9)] == [True, False] * 4 + [True]


def test_keep_every_for_40_to_20_is_2():
    assert cr.keep_every_for(40, 20) == 2
    assert cr.keep_every_for(40, 40) == 1
    assert cr.keep_every_for(40, 60) == 1


class _FakeRequest:
    def __init__(self, ts_ns):
        self._ts = ts_ns

    def get_metadata(self):
        return {"SensorTimestamp": self._ts}


@pytest.fixture
def fake_picamera2(monkeypatch):
    """A stand-in picamera2.encoders whose H264Encoder records _encode calls."""
    encoded = []

    class H264Encoder:
        def __init__(self, bitrate=None, repeat=True, iperiod=30, framerate=30,
                     qp=None, profile=None):
            self.bitrate = bitrate
            self.framerate = framerate

        def _encode(self, stream, request):
            encoded.append(request._ts)

    pkg = types.ModuleType("picamera2")
    enc_mod = types.ModuleType("picamera2.encoders")
    enc_mod.H264Encoder = H264Encoder
    pkg.encoders = enc_mod
    monkeypatch.setitem(sys.modules, "picamera2", pkg)
    monkeypatch.setitem(sys.modules, "picamera2.encoders", enc_mod)
    return H264Encoder, encoded


def test_decimating_encoder_drops_exactly_every_other_frame(fake_picamera2):
    base, encoded = fake_picamera2
    seen = []
    enc = cr.make_decimating_encoder(bitrate=1_000_000, framerate=20.0,
                                     keep_every=2, on_frame=seen.append)
    assert isinstance(enc, base)
    assert enc.bitrate == 1_000_000 and enc.framerate == 20.0
    stream_ts = [1_000_000_000 + i * 25_000_000 for i in range(10)]  # 40 fps
    for ts in stream_ts:
        enc._encode("main", _FakeRequest(ts))
    assert encoded == stream_ts[::2]
    assert seen == stream_ts[::2]           # one CSV row per ENCODED frame
    assert all(b - a == 50_000_000 for a, b in zip(seen, seen[1:]))


def test_decimating_encoder_survives_bad_metadata(fake_picamera2):
    _, encoded = fake_picamera2
    seen = []

    class _NoMeta(_FakeRequest):
        def get_metadata(self):
            raise RuntimeError("boom")

    enc = cr.make_decimating_encoder(bitrate=1, framerate=20.0, keep_every=1,
                                     on_frame=seen.append)
    enc._encode("main", _NoMeta(5))       # must not raise into the camera thread
    assert encoded == [5]
    assert len(seen) == 1 and isinstance(seen[0], int)


# ── record_gap ───────────────────────────────────────────────────────────────

class FakeClock:
    def __init__(self, t=1000.0):
        self.t = t
        self.slept = []

    def __call__(self):
        return self.t

    def sleep(self, s):
        assert s >= 0
        self.slept.append(s)
        self.t += s


class FakeEncoder:
    def __init__(self, on_frame, bitrate, framerate, keep_every):
        self.on_frame = on_frame
        self.bitrate = bitrate
        self.framerate = framerate
        self.keep_every = keep_every
        self.running = False


class FakePicam2:
    """Records when the encoder starts/stops; delivers 20 fps of frames."""

    def __init__(self, clock, fail_start=False, fail_stop=False):
        self.clock = clock
        self.encoders = set()
        self.lock = threading.Lock()
        self.calls = []
        self.fail_start = fail_start
        self.fail_stop = fail_stop
        self._started_at = None

    def start_encoder(self, encoder, output=None):
        self.calls.append(("start", self.clock(), output))
        if self.fail_start:
            raise RuntimeError("camera said no")
        Path(output).write_bytes(b"\x00\x00\x00\x01")
        encoder.running = True
        self.encoders.add(encoder)
        self._started_at = self.clock()

    def stop_encoder(self, encoders=None):
        self.calls.append(("stop", self.clock(), None))
        if self.fail_stop:
            raise RuntimeError("stop failed")
        enc = encoders
        n = int(round((self.clock() - self._started_at) * 20))
        for i in range(n):
            enc.on_frame(10**12 + i * 50_000_000)
        enc.running = False
        self.encoders.discard(enc)


def _make(tmp_path, clock, picam2=None, free_gb=100.0, notes=None, **kw):
    notes = [] if notes is None else notes
    picam2 = picam2 or FakePicam2(clock)
    rec = cr.ContinuousRecorder(
        picam2, tmp_path / "continuous", fps=20, bitrate=1_000_000,
        min_free_gb=2, notify=lambda t, b: notes.append((t, b)),
        clock=clock, sleep=clock.sleep, disk_free=lambda p: free_gb * 1e9,
        encoder_factory=FakeEncoder, spawn=lambda fn: fn(),
        now=lambda: datetime(2026, 11, 12, 19, 5, 30), log=lambda *a: None, **kw)
    return rec, picam2, notes


def test_record_gap_blocks_exactly_gap_and_honours_guards(tmp_path):
    clock = FakeClock()
    rec, cam, notes = _make(tmp_path, clock)
    t0 = clock()
    path = rec.record_gap(219.0, 2.0, tag="pretest_1_Paraffin")
    assert clock() - t0 == pytest.approx(219.0)
    (_, t_start, out), (_, t_stop, _) = cam.calls
    assert t_start - t0 >= 2.0
    assert (t0 + 219.0) - t_stop >= 2.0
    assert cam.encoders == set()
    assert notes == []
    assert path == tmp_path / "continuous" / \
        "between_01_after_pretest_1_Paraffin_20261112_190530.h264"
    assert out == str(path) and path.exists()


def test_record_gap_writes_frame_csv_one_row_per_encoded_frame(tmp_path):
    clock = FakeClock()
    rec, cam, _ = _make(tmp_path, clock)
    path = rec.record_gap(14.0, 2.0, tag="testing_1")   # 10 s → 200 frames
    with open(path.with_suffix(".csv"), newline="") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == ["frame", "wall_time", "sensor_ns"]
    assert len(rows) == 1 + 200
    assert [r[0] for r in rows[1:4]] == ["1", "2", "3"]
    assert rows[1][2] == str(10**12) and rows[2][2] == str(10**12 + 50_000_000)
    assert rows[1][1].endswith("Z")


def test_record_gap_counter_and_tag_sanitised(tmp_path):
    clock = FakeClock()
    rec, cam, _ = _make(tmp_path, clock)
    a = rec.record_gap(10.0, 2.0, tag="testing_1")
    b = rec.record_gap(10.0, 2.0, tag="testing 2/odd:tag")
    assert a.name.startswith("between_01_after_testing_1_")
    assert b.name.startswith("between_02_after_testing-2-odd-tag_")


def test_short_gap_records_nothing_and_still_waits(tmp_path):
    clock = FakeClock()
    rec, cam, notes = _make(tmp_path, clock)
    t0 = clock()
    assert rec.record_gap(4.5, 2.0, tag="x") is None
    assert clock() - t0 == pytest.approx(4.5)
    assert cam.calls == [] and notes == []
    assert not (tmp_path / "continuous").exists()


def test_disk_floor_skips_rest_of_session_and_notifies_once(tmp_path):
    clock = FakeClock()
    free = {"gb": 100.0}
    notes = []
    picam2 = FakePicam2(clock)
    rec = cr.ContinuousRecorder(
        picam2, tmp_path / "continuous", fps=20, bitrate=1_000_000,
        min_free_gb=2, notify=lambda t, b: notes.append((t, b)),
        clock=clock, sleep=clock.sleep, disk_free=lambda p: free["gb"] * 1e9,
        encoder_factory=FakeEncoder, spawn=lambda fn: fn(), log=lambda *a: None)
    assert rec.record_gap(30.0, 2.0, tag="a") is not None
    free["gb"] = 1.5
    t0 = clock()
    assert rec.record_gap(30.0, 2.0, tag="b") is None
    assert clock() - t0 == pytest.approx(30.0)
    free["gb"] = 100.0                       # space back: stays off this session
    assert rec.record_gap(30.0, 2.0, tag="c") is None
    assert len(notes) == 1
    assert "disk" in (notes[0][0] + notes[0][1]).lower()
    assert sum(1 for c in picam2.calls if c[0] == "start") == 1


def test_disk_floor_checks_existing_parent_when_dir_missing(tmp_path):
    seen = []
    clock = FakeClock()
    rec = cr.ContinuousRecorder(
        FakePicam2(clock), tmp_path / "continuous", fps=20, bitrate=1,
        min_free_gb=2, notify=lambda t, b: None, clock=clock, sleep=clock.sleep,
        disk_free=lambda p: (seen.append(p), 1e12)[1],
        encoder_factory=FakeEncoder, spawn=lambda fn: fn(), log=lambda *a: None)
    rec.record_gap(10.0, 2.0, tag="a")
    assert seen and seen[0] == tmp_path / "continuous"


def test_default_disk_free_walks_up_to_existing_dir(tmp_path):
    assert cr.disk_free_bytes(tmp_path / "no" / "such" / "dir") > 0


def test_exception_on_start_is_swallowed_gap_waited_notify_once(tmp_path):
    clock = FakeClock()
    cam = FakePicam2(clock, fail_start=True)
    rec, _, notes = _make(tmp_path, clock, picam2=cam)
    t0 = clock()
    assert rec.record_gap(60.0, 2.0, tag="a") is None       # no raise
    assert clock() - t0 == pytest.approx(60.0)
    assert rec.record_gap(60.0, 2.0, tag="b") is None
    assert clock() - t0 == pytest.approx(120.0)
    assert len(notes) == 1
    assert sum(1 for c in cam.calls if c[0] == "start") == 1  # disabled after
    assert cam.encoders == set()


def test_exception_on_stop_never_leaves_encoder_registered(tmp_path):
    # The trial recorder's bare picam2.stop_encoder() would call .stop() on a
    # leftover between encoder and crash the trial: it must be removed.
    clock = FakeClock()
    cam = FakePicam2(clock, fail_stop=True)
    rec, _, notes = _make(tmp_path, clock, picam2=cam)
    t0 = clock()
    assert rec.record_gap(60.0, 2.0, tag="a") is None
    assert clock() - t0 == pytest.approx(60.0)
    assert cam.encoders == set()
    assert len(notes) == 1


def test_notify_failure_does_not_escape(tmp_path):
    clock = FakeClock()

    def bad_notify(t, b):
        raise OSError("no network")

    rec = cr.ContinuousRecorder(
        FakePicam2(clock, fail_start=True), tmp_path / "c", fps=20, bitrate=1,
        min_free_gb=2, notify=bad_notify, clock=clock, sleep=clock.sleep,
        disk_free=lambda p: 1e12, encoder_factory=FakeEncoder,
        spawn=lambda fn: fn(), log=lambda *a: None)
    t0 = clock()
    assert rec.record_gap(20.0, 2.0, tag="a") is None
    assert clock() - t0 == pytest.approx(20.0)


def test_slow_stop_still_returns_at_gap_end_when_possible(tmp_path):
    # Encoder flush takes 1 s: still ≥ guard-1 before the end, and the call
    # returns exactly at gap end because waiting is deadline-based.
    clock = FakeClock()

    class SlowStop(FakePicam2):
        def stop_encoder(self, encoders=None):
            super().stop_encoder(encoders)
            clock.t += 1.0

    rec, cam, _ = _make(tmp_path, clock, picam2=SlowStop(clock))
    t0 = clock()
    rec.record_gap(30.0, 2.0, tag="a")
    assert clock() - t0 == pytest.approx(30.0)


def test_encoder_gets_requested_rate_and_decimation(tmp_path):
    clock = FakeClock()
    made = []

    def factory(**kw):
        enc = FakeEncoder(**kw)
        made.append(enc)
        return enc

    rec = cr.ContinuousRecorder(
        FakePicam2(clock), tmp_path / "c", fps=20, bitrate=1_000_000,
        min_free_gb=2, notify=lambda t, b: None, clock=clock, sleep=clock.sleep,
        disk_free=lambda p: 1e12, encoder_factory=factory,
        spawn=lambda fn: fn(), log=lambda *a: None, camera_fps=40)
    rec.record_gap(10.0, 2.0, tag="a")
    (enc,) = made
    assert enc.keep_every == 2 and enc.framerate == 20.0
    assert enc.bitrate == 1_000_000


# ── from_config ──────────────────────────────────────────────────────────────

def test_from_config_reads_expanded_continuous_block(tmp_path):
    rec = cr.from_config(object(), {"fps": 20, "bitrate": 1000000, "guard_s": 2,
                                    "min_free_gb": 2},
                         tmp_path / "continuous", notify=lambda t, b: None,
                         camera_fps=40)
    assert rec.guard_s == 2.0 and rec.keep_every == 2
    assert rec.bitrate == 1_000_000 and rec.min_free_gb == 2.0
    assert not (tmp_path / "continuous").exists()        # mkdir is lazy


@pytest.mark.parametrize("bad", [{"fps": 0}, {"fps": 20, "bitrate": -1},
                                 {"fps": 20, "guard_s": -1}])
def test_from_config_rejects_bad_values_at_startup(tmp_path, bad):
    with pytest.raises(ValueError):
        cr.from_config(object(), bad, tmp_path, notify=lambda t, b: None,
                       camera_fps=40)


# ── rig-script wiring (text-level: the scripts import Pi-only modules) ───────

RIG_SCRIPTS = ["combinedv2_1.py", "combinedv2_1_pi1.py", "combinedv2_1_pi3.py"]
PICODE = Path(__file__).resolve().parent.parent / "PiCode"


@pytest.mark.parametrize("name", RIG_SCRIPTS)
def test_rig_script_wires_recorder_only_into_flagged_gaps(name):
    src = (PICODE / name).read_text(encoding="utf-8")
    assert src.count("CONT_RECORDER = None") == 1
    assert src.count("from continuous_recorder import from_config") == 1
    assert src.count('if cfg.get("continuous"):') == 1
    assert src.count("CONT_RECORDER.record_gap(") == 1
    assert ('if (step.get("continuous_gap") and CONT_RECORDER is not None\n'
            '                    and not recording_started and duration > 0):') in src
    # the trial recorder is untouched: one 100 Mb/s encoder, bare stop
    assert src.count("H264Encoder(bitrate=100_000_000)") == 1
    assert src.count("picam2.stop_encoder()") == 1


def test_keyboard_interrupt_stops_encoder_and_is_not_swallowed(tmp_path):
    clock = FakeClock()
    cam = FakePicam2(clock)

    def interrupting_sleep(s):
        if cam.encoders:                     # mid-recording
            raise KeyboardInterrupt
        clock.sleep(s)

    rec = cr.ContinuousRecorder(
        cam, tmp_path / "c", fps=20, bitrate=1, min_free_gb=2,
        notify=lambda t, b: None, clock=clock, sleep=interrupting_sleep,
        disk_free=lambda p: 1e12, encoder_factory=FakeEncoder,
        spawn=lambda fn: fn(), log=lambda *a: None)
    with pytest.raises(KeyboardInterrupt):
        rec.record_gap(1730.0, 2.0, tag="a")
    assert cam.encoders == set()
    assert clock() - 1000.0 < 10            # did not sit out the rest
