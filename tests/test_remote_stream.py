"""Unit tests for PiCode/remote_stream.py and PiCode/stream_worker.py — the
live-stream side channel (lores tap -> shared memory -> niced PyAV worker).

Neither module needs picamera2 or PyAV at import time, so everything runs off
the rig with fakes. One test really spawns the worker (``av`` absent or
unreachable) to prove the child starts, drains slots and exits on the event.
"""

import logging
import multiprocessing as mp
import os
import queue
import sys
import threading
import time
import types
from fractions import Fraction
from multiprocessing import shared_memory
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "PiCode"))

import remote_stream as rs  # noqa: E402
import stream_worker as sw  # noqa: E402

PASSWORD = "s3cr3tpw"
URL = f"rtsp://rig:{PASSWORD}@100.1.2.3:8554/rig2"


# ── helpers ──────────────────────────────────────────────────────────────────

def make_yuv(width, height, stride):
    """A lores-style buffer: Y rows of `stride`, then U, then V (stride//2),
    every byte unique-ish so a wrong slice shows up."""
    y = bytes((r * 7 + c) % 251 for r in range(height) for c in range(stride))
    u = bytes((r * 5 + c + 1) % 241 for r in range(height // 2)
              for c in range(stride // 2))
    v = bytes((r * 3 + c + 2) % 239 for r in range(height // 2)
              for c in range(stride // 2))
    return y + u + v


def expected_i420(buf, width, height, stride):
    out = bytearray()
    for r in range(height):
        out += buf[r * stride:r * stride + width]
    base = stride * height
    cw, cs = width // 2, stride // 2
    for plane in range(2):
        for r in range(height // 2):
            off = base + plane * cs * (height // 2) + r * cs
            out += buf[off:off + cw]
    return bytes(out)


# ── pack_yuv420 ──────────────────────────────────────────────────────────────

def test_pack_strips_stride_padding():
    w, h, s = 540, 540, 576
    buf = make_yuv(w, h, s)
    out = rs.pack_yuv420(buf, w, h, s)
    assert len(out) == w * h * 3 // 2
    assert out == expected_i420(buf, w, h, s)


def test_pack_no_padding_is_identity():
    w, h = 64, 48
    buf = make_yuv(w, h, w)
    assert rs.pack_yuv420(buf, w, h, w) == buf


def test_pack_accepts_memoryview_and_bytearray():
    w, h, s = 32, 16, 64
    buf = make_yuv(w, h, s)
    exp = expected_i420(buf, w, h, s)
    assert rs.pack_yuv420(memoryview(buf), w, h, s) == exp
    assert rs.pack_yuv420(bytearray(buf), w, h, s) == exp


def test_pack_short_buffer_raises():
    w, h, s = 64, 48, 64
    buf = make_yuv(w, h, s)
    with pytest.raises(ValueError):
        rs.pack_yuv420(buf[:-1], w, h, s)


def test_pack_stride_smaller_than_width_raises():
    with pytest.raises(ValueError):
        rs.pack_yuv420(bytes(10_000), 64, 32, 32)


# ── TapCore ──────────────────────────────────────────────────────────────────

@pytest.fixture
def core():
    c = rs.TapCore(540, 540, slots=4)
    yield c
    c.close()


def test_offer_copies_frame_into_slot(core):
    s = 576
    buf = make_yuv(540, 540, s)
    assert core.offer(buf, s, 123_456) is True
    slot, ts = core.frame_q.get(timeout=2)
    assert ts == 123_456
    fb = 540 * 540 * 3 // 2
    got = bytes(core.shm.buf[slot * fb:(slot + 1) * fb])
    assert got == expected_i420(buf, 540, 540, s)
    assert core.stats()["offered"] == 1
    assert core.stats()["sent"] == 1


def test_offer_drops_without_consumer_and_never_blocks(core):
    s = 576
    buf = make_yuv(540, 540, s)
    results = [core.offer(buf, s, i) for i in range(4)]
    assert results == [True] * 4
    t0 = time.perf_counter()
    more = [core.offer(buf, s, 10 + i) for i in range(50)]
    assert time.perf_counter() - t0 < 0.5
    assert not any(more)
    st = core.stats()
    assert st["sent"] == 4
    assert st["dropped"] == 50
    assert st["errors"] == 0


def test_returned_slot_is_reusable(core):
    s = 576
    buf = make_yuv(540, 540, s)
    for i in range(4):
        assert core.offer(buf, s, i)
    slot, _ = core.frame_q.get(timeout=2)
    core.free_q.put(slot)
    deadline = time.monotonic() + 2
    ok = False
    while time.monotonic() < deadline and not ok:
        ok = core.offer(buf, s, 99)       # mp feeder thread: allow a moment
        time.sleep(0.005)
    assert ok


def test_offer_is_fast_enough_for_the_camera_thread(core):
    """Each offer runs on picamera2's request thread at 40 fps; the budget is
    well under one frame period (25 ms), we ask for < 1 ms on average."""
    s = 576
    buf = make_yuv(540, 540, s)
    stop = threading.Event()

    def recycler():
        while not stop.is_set():
            try:
                slot, _ = core.frame_q.get(timeout=0.05)
            except queue.Empty:
                continue
            core.free_q.put(slot)

    th = threading.Thread(target=recycler, daemon=True)
    th.start()
    core.offer(buf, s, 0)                 # warm-up: numpy import, feeder start
    n = 200
    t0 = time.perf_counter()
    for i in range(n):
        core.offer(buf, s, i)
    per_call = (time.perf_counter() - t0) / n
    stop.set()
    th.join(1)
    assert per_call < 1e-3, f"{per_call * 1e3:.3f} ms per offer"


def test_offer_swallows_errors(core):
    assert core.offer(b"short", 576, 1) is False
    st = core.stats()
    assert st["errors"] == 1
    # the slot it took must not leak: all four are still usable
    s = 576
    buf = make_yuv(540, 540, s)
    assert [core.offer(buf, s, i) for i in range(4)] == [True] * 4


def test_reset_restores_all_slots(core):
    s = 576
    buf = make_yuv(540, 540, s)
    for i in range(4):
        assert core.offer(buf, s, i)
    assert core.offer(buf, s, 5) is False
    old_q = core.frame_q
    core.reset()
    assert core.frame_q is not old_q
    assert [core.offer(buf, s, i) for i in range(4)] == [True] * 4


def test_close_unlinks_shm_and_is_idempotent():
    c = rs.TapCore(64, 64, slots=2)
    name = c.shm.name
    c.close()
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=name)
    c.close()                              # second close: no error
    assert c.offer(bytes(64 * 64 * 3 // 2), 64, 1) is False   # closed: no raise


# ── backoff ──────────────────────────────────────────────────────────────────

def test_backoff_delays_sequence():
    g = rs.backoff_delays()
    assert [next(g) for _ in range(8)] == [1, 2, 4, 8, 16, 32, 60, 60]


def test_backoff_delays_jitter_hook():
    g = rs.backoff_delays(start=1.0, cap=4.0, jitter=lambda d: d + 0.5)
    assert [next(g) for _ in range(4)] == [1.5, 2.5, 4.5, 4.5]


# ── URL redaction ────────────────────────────────────────────────────────────

def test_redact_hides_password_keeps_user_and_host():
    out = rs._redact(URL)
    assert PASSWORD not in out
    assert out == "rtsp://rig:***@100.1.2.3:8554/rig2"
    assert rs._redact("rtsp://host:8554/x") == "rtsp://host:8554/x"


def test_scrub_removes_password_from_arbitrary_text():
    msg = f"Connection to {URL} failed ({PASSWORD})"
    out = sw.scrub(msg, URL)
    assert PASSWORD not in out


# ── stream_worker.run with a fake PyAV ───────────────────────────────────────

class FakeCodecContext:
    pass


class FakeStream:
    def __init__(self, owner):
        self.owner = owner
        self.codec_context = FakeCodecContext()
        self.frames = []

    def encode(self, frame=None):
        if frame is None:
            return []
        self.frames.append(frame)
        return [("pkt", frame.pts)]


class FakeContainer:
    def __init__(self, av, url, options):
        self.av = av
        self.url = url
        self.options = options
        self.closed = False
        self.stream = None
        self.add_stream_args = None

    def add_stream(self, codec, rate=None, options=None, **kw):
        self.add_stream_args = (codec, rate, options)
        self.stream = FakeStream(self)
        self.av.streams.append(self.stream)
        return self.stream

    def mux(self, pkt):
        self.av.mux_calls += 1
        if self.av.mux_calls in self.av.fail_mux_at:
            raise OSError(f"broken pipe to {self.url}")
        self.av.muxed.append(pkt)

    def close(self):
        self.closed = True


class FakeVideoFrame:
    def __init__(self, arr, fmt):
        self.arr, self.format = arr, fmt
        self.pts = None
        self.time_base = None

    @classmethod
    def from_ndarray(cls, arr, format):
        return cls(arr, format)


class FakeAV:
    """Records open options; raises on the first `fail_opens` opens and on the
    mux calls numbered in `fail_mux_at`."""

    def __init__(self, fail_opens=0, fail_mux_at=()):
        self.fail_opens = fail_opens
        self.fail_mux_at = set(fail_mux_at)
        self.open_calls = []
        self.open_times = []
        self.containers = []
        self.streams = []
        self.muxed = []
        self.mux_calls = 0
        self.clock = None
        self.VideoFrame = FakeVideoFrame

    def open(self, url, mode="r", format=None, options=None, **kw):
        self.open_calls.append((url, mode, format, dict(options or {})))
        if self.clock:
            self.open_times.append(self.clock())
        if len(self.open_calls) <= self.fail_opens:
            raise ConnectionRefusedError(f"cannot connect to {url}")
        c = FakeContainer(self, url, options)
        self.containers.append(c)
        return c


class FakeShm:
    def __init__(self, width, height, slots):
        self.fb = width * height * 3 // 2
        self.buf = memoryview(bytearray(self.fb * slots))
        self.closed = False

    def close(self):
        self.closed = True


class DrainQueue(queue.Queue):
    """frame_q stand-in: once empty, flips stop_event instead of blocking so
    the loop under test ends deterministically."""

    def __init__(self, stop_event):
        super().__init__()
        self.stop_event = stop_event

    def get(self, block=True, timeout=None):
        try:
            return super().get(block=False)
        except queue.Empty:
            self.stop_event.set()
            raise


class Clock:
    """Auto-advancing fake monotonic clock: each read costs `step` seconds."""

    def __init__(self, step=0.5):
        self.t = 0.0
        self.step = step

    def __call__(self):
        self.t += self.step
        return self.t


def run_worker(n_frames, av, *, width=540, height=540, step=0.5,
               parent_pid=1234, getppid=lambda: 1234, stop_event=None,
               frame_q=None, ts_step=25_000_000):
    stop = stop_event or threading.Event()
    fq = frame_q if frame_q is not None else DrainQueue(stop)
    free_q = queue.Queue()
    shm = FakeShm(width, height, 4)
    for i in range(n_frames):
        slot = i % 4
        fq.put((slot, 1_000_000_000 + i * ts_step))
    clock = Clock(step)
    av.clock = clock
    sw.run(URL, width, height, 40, 1_500_000, 40, "main", shm, fq, free_q,
           stop, parent_pid, av_module=av, sleep=lambda s: None, now=clock,
           getppid=getppid)
    return shm, free_q, stop


def test_worker_configures_the_encoder():
    av = FakeAV()
    shm, free_q, _ = run_worker(3, av)
    (url, mode, fmt, opts), = av.open_calls
    assert url == URL and mode == "w" and fmt == "rtsp"
    assert opts["rtsp_transport"] == "tcp"
    cont = av.containers[0]
    codec, rate, enc_opts = cont.add_stream_args
    assert codec == "libx264" and rate == 40
    assert enc_opts["bframes"] == "0"
    assert enc_opts["preset"] == "ultrafast"
    assert enc_opts["tune"] == "zerolatency"
    assert enc_opts["keyint_min"] == "40"
    assert enc_opts["sc_threshold"] == "0"
    cc = cont.stream.codec_context
    assert (cc.width, cc.height) == (540, 540)
    assert cc.pix_fmt == "yuv420p"
    assert cc.bit_rate == 1_500_000
    assert cc.gop_size == 40
    assert cc.profile == "main"
    assert cc.time_base == Fraction(1, 1_000_000)
    assert len(av.muxed) == 3
    assert shm.closed


def test_worker_frames_are_i420_with_zero_based_microsecond_pts():
    av = FakeAV()
    run_worker(3, av, width=64, height=48)
    frames = av.streams[0].frames
    assert [f.pts for f in frames] == [0, 25_000, 50_000]
    assert all(f.format == "yuv420p" for f in frames)
    assert frames[0].arr.shape == (48 * 3 // 2, 64)
    assert frames[0].time_base == Fraction(1, 1_000_000)


def test_worker_reads_frame_bytes_from_the_right_slot():
    av = FakeAV()
    stop = threading.Event()
    fq = DrainQueue(stop)
    free_q = queue.Queue()
    w, h = 16, 8
    shm = FakeShm(w, h, 4)
    fb = shm.fb
    shm.buf[2 * fb:3 * fb] = bytes([7]) * fb
    fq.put((2, 5_000))
    clock = Clock()
    sw.run(URL, w, h, 40, 1_000_000, 40, "main", shm, fq, free_q, stop, 1,
           av_module=av, sleep=lambda s: None, now=clock, getppid=lambda: 1)
    assert int(av.streams[0].frames[0].arr[0, 0]) == 7
    assert free_q.get_nowait() == 2


def test_worker_returns_every_slot_even_while_disconnected():
    av = FakeAV(fail_opens=10**6)           # never connects
    _, free_q, _ = run_worker(8, av)
    slots = []
    while not free_q.empty():
        slots.append(free_q.get_nowait())
    assert len(slots) == 8                  # none leaked
    assert av.muxed == []


def test_worker_backs_off_between_failed_opens():
    av = FakeAV(fail_opens=2)
    # step 0.5 s per clock read: delays are 1 s then 2 s
    _, free_q, _ = run_worker(40, av)
    assert len(av.open_calls) == 3          # 2 failures + 1 success
    t = av.open_times
    assert t[1] - t[0] >= 1.0
    assert t[2] - t[1] >= 2.0
    assert len(av.muxed) > 0                # frames flow after the reconnect


def test_worker_reconnects_after_mux_error():
    av = FakeAV(fail_mux_at={3})
    _, free_q, _ = run_worker(30, av)
    assert len(av.containers) >= 2
    assert av.containers[0].closed
    # each connection starts its pts at zero
    assert av.streams[1].frames[0].pts == 0
    assert len(av.muxed) > 3


def test_worker_error_text_never_logged_with_password(caplog):
    av = FakeAV(fail_opens=1, fail_mux_at={2})
    with caplog.at_level(logging.DEBUG):
        run_worker(30, av)
    assert caplog.records
    assert PASSWORD not in caplog.text


def test_worker_exits_when_stop_event_set():
    av = FakeAV()
    stop = threading.Event()
    stop.set()
    _, free_q, _ = run_worker(5, av, stop_event=stop, frame_q=queue.Queue())
    assert av.open_calls == []              # exited before touching a frame


def test_worker_exits_when_orphaned():
    av = FakeAV()
    calls = {"n": 0}

    def ppid():
        calls["n"] += 1
        return 1234 if calls["n"] <= 2 else 1   # reparented to init

    fq = queue.Queue()
    for i in range(10):
        fq.put((i % 4, i * 1000))
    shm, _, _ = run_worker(0, av, getppid=ppid, frame_q=fq)
    assert len(av.muxed) <= 2
    assert shm.closed


def test_worker_flushes_and_closes_on_exit():
    av = FakeAV()
    run_worker(2, av)
    assert av.containers[0].closed


def test_worker_survives_missing_av(monkeypatch):
    """No PyAV installed -> treated as an open failure, slots keep returning."""
    monkeypatch.setitem(sys.modules, "av", None)   # import av -> ImportError
    stop = threading.Event()
    fq = DrainQueue(stop)
    free_q = queue.Queue()
    shm = FakeShm(16, 8, 4)
    for i in range(3):
        fq.put((i, i))
    clock = Clock()
    sw.run(URL, 16, 8, 40, 1_000_000, 40, "main", shm, fq, free_q, stop, 1,
           sleep=lambda s: None, now=clock, getppid=lambda: 1)
    assert free_q.qsize() == 3


# ── make_frame_tap against fake picamera2 modules ────────────────────────────

@pytest.fixture
def fake_picamera2(monkeypatch):
    pkg = types.ModuleType("picamera2")
    enc = types.ModuleType("picamera2.encoders")
    outs = types.ModuleType("picamera2.outputs")

    class Encoder:
        def __init__(self):
            self.width = self.height = self.stride = 0

    class Output:
        def __init__(self, *a, **k):
            pass

    enc.Encoder = Encoder
    outs.Output = Output
    pkg.encoders, pkg.outputs = enc, outs
    monkeypatch.setitem(sys.modules, "picamera2", pkg)
    monkeypatch.setitem(sys.modules, "picamera2.encoders", enc)
    monkeypatch.setitem(sys.modules, "picamera2.outputs", outs)
    return types.SimpleNamespace(Encoder=Encoder, Output=Output)


class FakeRequest:
    def __init__(self, buf, ts):
        self.buf, self.ts, self.asked = buf, ts, []

    def make_buffer(self, name):
        self.asked.append(name)
        return self.buf

    def get_metadata(self):
        return {"SensorTimestamp": self.ts}


def test_frame_tap_forwards_lores_buffer_and_timestamp(fake_picamera2):
    seen = []
    fake_core = types.SimpleNamespace(
        offer=lambda buf, stride, ts: seen.append((buf, stride, ts)) or True)
    tap, output = rs.make_frame_tap(fake_core)
    assert isinstance(tap, fake_picamera2.Encoder)
    assert isinstance(output, fake_picamera2.Output)
    tap.width, tap.height, tap.stride = 540, 540, 576
    req = FakeRequest(b"BUF", 777)
    tap._encode("lores", req)
    assert req.asked == ["lores"]
    assert seen == [(b"BUF", 576, 777)]
    output.outputframe(b"x", True, None, None)   # no-op, no raise


def test_frame_tap_never_raises(fake_picamera2):
    def boom(*a):
        raise RuntimeError("x")

    tap, _ = rs.make_frame_tap(types.SimpleNamespace(offer=boom))
    tap._encode("lores", FakeRequest(b"BUF", 1))          # swallowed
    tap._encode("lores", types.SimpleNamespace())         # no make_buffer: swallowed


# ── RemoteStreamer ───────────────────────────────────────────────────────────

class FakePicam:
    def __init__(self, raises=False):
        self.raises = raises
        self.started = []
        self.stopped = []

    def start_encoder(self, encoder=None, output=None, *a, **kw):
        if self.raises:
            raise RuntimeError("start boom")
        self.started.append((encoder, output, kw))

    def stop_encoder(self, encoder=None):
        self.stopped.append(encoder)
        if self.raises:
            raise RuntimeError("stop boom")


class FakeProc:
    instances = []

    def __init__(self, target=None, args=(), daemon=None, name=None, **kw):
        self.target, self.args, self.daemon = target, args, daemon
        self.alive = False
        self.started = False
        self.terminated = False
        self.killed = False
        self.exitcode = None
        self.stubborn = False       # ignore stop requests (join does nothing)
        FakeProc.instances.append(self)

    def start(self):
        self.started = True
        self.alive = True

    def is_alive(self):
        return self.alive

    def join(self, timeout=None):
        if not self.stubborn and not self.terminated and not self.killed:
            self.alive = False

    def terminate(self):
        self.terminated = True
        if not self.stubborn:
            self.alive = False

    def kill(self):
        self.killed = True
        self.alive = False


def fake_tap_factory(core):
    return types.SimpleNamespace(name="tap", core=core), types.SimpleNamespace(name="out")


@pytest.fixture
def streamer():
    FakeProc.instances = []
    made = []

    def make(**kw):
        kw.setdefault("watchdog_tick", 0.01)
        kw.setdefault("backoff_start", 0.02)
        s = rs.RemoteStreamer(URL, process_factory=FakeProc,
                              tap_factory=fake_tap_factory, **kw)
        made.append(s)
        return s

    yield make
    for s in made:
        s.stop(timeout=0.2)


def wait_for(pred, timeout=2.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.005)
    return False


def test_start_registers_tap_on_lores_and_spawns_worker(streamer):
    s = streamer()
    pc = FakePicam()
    assert s.start(pc) is True
    (tap, output, kw), = pc.started
    assert kw == {"name": "lores"}
    assert tap.name == "tap" and output.name == "out"
    proc, = FakeProc.instances
    assert proc.started and proc.daemon is True
    assert proc.target is sw.main
    (url, w, h, fps, br, gop, prof, shm_name, fq, free_q, stop_ev, ppid) = proc.args
    assert (url, w, h, fps, br, gop, prof) == (URL, 540, 540, 40, 1_500_000, 40, "main")
    assert ppid == os.getpid()
    assert shm_name == s._core.shm.name


def test_watchdog_respawns_dead_worker(streamer):
    s = streamer()
    s.start(FakePicam())
    first = FakeProc.instances[0]
    first.alive = False
    first.exitcode = 1
    assert wait_for(lambda: len(FakeProc.instances) >= 2)
    assert FakeProc.instances[1].started
    assert s.stats()["worker_restarts"] >= 1


def test_watchdog_backoff_grows(streamer):
    s = streamer(backoff_start=0.05)
    s.start(FakePicam())
    # every respawned worker dies instantly -> gaps between spawns double
    stamps = []

    def killer():
        end = time.monotonic() + 1.2
        seen = 0
        while time.monotonic() < end:
            for p in FakeProc.instances[seen:]:
                stamps.append(time.monotonic())
                p.alive = False
                seen += 1
            time.sleep(0.002)

    killer()
    assert len(stamps) >= 4
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    assert gaps[2] > gaps[0]


def test_respawn_gets_fresh_queues_and_full_slot_pool(streamer):
    s = streamer()
    s.start(FakePicam())
    q_before = FakeProc.instances[0].args[8]
    FakeProc.instances[0].alive = False
    assert wait_for(lambda: len(FakeProc.instances) >= 2)
    assert FakeProc.instances[1].args[8] is not q_before


def test_stop_is_idempotent_bounded_and_passes_the_tap(streamer):
    s = streamer()
    pc = FakePicam()
    s.start(pc)
    tap = pc.started[0][0]
    FakeProc.instances[0].stubborn = True
    t0 = time.monotonic()
    s.stop(timeout=0.2)
    s.stop(timeout=0.2)
    assert time.monotonic() - t0 < 3.0
    assert pc.stopped == [tap]               # the tap object, not bare, once
    p = FakeProc.instances[0]
    assert p.terminated and p.killed
    assert s._core is None or s._core.shm is None


def test_stop_never_raises_when_picam_raises(streamer):
    s = streamer()
    pc = FakePicam()
    s.start(pc)
    pc.raises = True
    s.stop(timeout=0.2)


def test_start_failure_returns_false_and_cleans_up(streamer):
    s = streamer()
    assert s.start(FakePicam(raises=True)) is False
    s.stop(timeout=0.2)                       # still safe


def test_start_failure_in_process_factory_returns_false():
    def bad_factory(**kw):
        raise RuntimeError("no fork for you")

    s = rs.RemoteStreamer(URL, process_factory=bad_factory,
                          tap_factory=fake_tap_factory)
    assert s.start(FakePicam()) is False
    s.stop(timeout=0.2)


def test_restart_after_reinit_registers_new_tap_on_new_camera(streamer):
    s = streamer()
    old = FakePicam()
    s.start(old)
    old_tap = old.started[0][0]
    new = FakePicam()
    assert s.restart_after_reinit(new) is True
    assert old.stopped == [old_tap]
    (new_tap, _, kw), = new.started
    assert new_tap is not old_tap and kw == {"name": "lores"}
    assert len(FakeProc.instances) == 1       # worker kept running
    s.stop(timeout=0.2)


def test_stats_merge_core_counters_and_restarts(streamer):
    s = streamer()
    s.start(FakePicam())
    st = s.stats()
    for k in ("offered", "sent", "dropped", "errors", "worker_restarts"):
        assert k in st
    assert st["worker_restarts"] == 0


def test_logs_never_contain_the_password(streamer, caplog):
    with caplog.at_level(logging.DEBUG):
        s = streamer()
        s.start(FakePicam())
        FakeProc.instances[0].alive = False
        wait_for(lambda: len(FakeProc.instances) >= 2)
        s.stop(timeout=0.2)
        s2 = streamer()
        s2.start(FakePicam(raises=True))
    assert caplog.records
    assert PASSWORD not in caplog.text
    assert "***" in caplog.text


def test_logger_injection(streamer):
    msgs = []

    class L:
        def info(self, m, *a):
            msgs.append(m % a if a else m)
        warning = error = debug = info

    s = streamer(logger=L())
    s.start(FakePicam())
    s.stop(timeout=0.2)
    assert msgs and all(PASSWORD not in m for m in msgs)


# ── one real spawn ───────────────────────────────────────────────────────────

def test_real_spawned_worker_drains_and_exits_on_event():
    """Spawn stream_worker.main for real. The URL is unreachable (and PyAV may
    be absent) so it never connects, but it must start, keep returning slots
    to the tap, and exit 0 when the stop event is set."""
    ctx = mp.get_context("spawn")
    core = rs.TapCore(64, 64, slots=4, ctx=ctx)
    stop = ctx.Event()
    proc = ctx.Process(
        target=sw.main,
        args=("rtsp://u:pw@127.0.0.1:9/x", 64, 64, 40, 500_000, 40, "main",
              core.shm.name, core.frame_q, core.free_q, stop, os.getpid()),
        daemon=True)
    proc.start()
    try:
        buf = bytes(64 * 64 * 3 // 2)
        accepted = 0
        deadline = time.monotonic() + 15
        while accepted < 12 and time.monotonic() < deadline:
            if core.offer(buf, 64, accepted):
                accepted += 1
            else:
                time.sleep(0.01)
        # 12 > 4 slots: only possible if the child returned slots
        assert accepted == 12
    finally:
        stop.set()
        proc.join(10)
        if proc.is_alive():
            proc.kill()
            proc.join(2)
        core.close()
    assert proc.exitcode == 0
