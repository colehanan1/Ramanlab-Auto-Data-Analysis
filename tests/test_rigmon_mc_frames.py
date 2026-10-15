import importlib.util
from pathlib import Path

import cv2
import numpy as np

from rigmon.mountcheck.frames import crop_square, decode, iter_jpegs, sample_preview

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("fake_rig", REPO / "scripts" / "rigmon" / "fake_rig.py")
fake_rig = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fake_rig)
GOLDEN_JPG = (Path(__file__).parent / "golden" / "mount_qc_pi2_20261014.jpg").read_bytes()


def jpeg(w=64, h=48, val=128):
    ok, buf = cv2.imencode(".jpg", np.full((h, w, 3), val, np.uint8))
    return buf.tobytes()


def test_iter_jpegs_splits_a_multipart_stream_across_chunk_boundaries():
    a, b = jpeg(val=10), jpeg(val=200)
    stream = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + a + b"\r\n--frame\r\n\r\n" + b + b"\r\n"
    chunks = [stream[i:i + 7] for i in range(0, len(stream), 7)]
    assert list(iter_jpegs(chunks)) == [a, b]


def test_crop_square_keeps_left_height_columns():
    img = np.zeros((1080, 1440, 3), np.uint8)
    assert crop_square(img).shape == (1080, 1080, 3)
    tall = np.zeros((1200, 1000, 3), np.uint8)
    assert crop_square(tall).shape == (1200, 1000, 3)


def test_decode_golden_frame_drops_the_stride_strip():
    assert decode(GOLDEN_JPG).shape == (1080, 1080, 3)


def test_sampler_keeps_about_fps_frames_for_the_duration():
    rig = fake_rig.FakeRig({}, preview_jpegs=[jpeg()], preview_interval=0.02)
    try:
        frames, complete = sample_preview(f"http://127.0.0.1:{rig.port}/preview", 1.0, 5.0)
    finally:
        rig.stop()
    assert complete and 4 <= len(frames) <= 7


def test_sampler_stops_early_when_asked():
    rig = fake_rig.FakeRig({}, preview_jpegs=[jpeg()], preview_interval=0.02)
    calls = {"n": 0}

    def stop():
        calls["n"] += 1
        return calls["n"] > 3
    try:
        frames, complete = sample_preview(f"http://127.0.0.1:{rig.port}/preview", 5.0, 50.0, stop=stop)
    finally:
        rig.stop()
    assert not complete and len(frames) <= 4


def test_sampler_returns_empty_when_unreachable():
    frames, complete = sample_preview("http://127.0.0.1:9/preview", 1.0, 2.0, timeout=1.0)
    assert frames == [] and not complete


def test_sampler_returns_empty_on_silent_stream():
    rig = fake_rig.FakeRig({}, preview_silent=True)
    try:
        frames, complete = sample_preview(f"http://127.0.0.1:{rig.port}/preview", 3.0, 2.0, timeout=1.0)
    finally:
        rig.stop()
    assert frames == [] and not complete


def test_sampler_gives_up_on_a_stream_that_never_completes_a_jpeg():
    import threading
    import time
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                self.wfile.write(b"\xff\xd8")
                for _ in range(200):
                    self.wfile.write(b"x")
                    self.wfile.flush()
                    time.sleep(0.1)
            except (BrokenPipeError, ConnectionResetError):
                pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        t0 = time.monotonic()
        frames, complete = sample_preview(f"http://127.0.0.1:{srv.server_address[1]}/preview", 1.5, 2.0,
                                          timeout=2.0)
        elapsed = time.monotonic() - t0
    finally:
        srv.shutdown()
    assert (frames, complete) == ([], False)
    assert elapsed < 1.5 + 3


def test_iter_jpegs_buffer_is_capped_for_an_endless_unterminated_jpeg():
    import tracemalloc

    from rigmon.mountcheck import frames as fr

    assert fr.MAX_BUFFER == 8 * 1024 * 1024

    def chunks():
        yield b"\xff\xd8"
        for _ in range(60):                      # 60 MB with no EOI
            yield b"\x00" * 1_000_000
        yield jpeg(val=50)                       # the stream recovers afterwards

    tracemalloc.start()
    try:
        out = list(fr.iter_jpegs(chunks()))
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert len(out) == 1
    assert peak < fr.MAX_BUFFER * 3          # an uncapped buffer would reach ~60 MB


def test_sampler_catches_invalid_url():
    assert sample_preview("http://[bad/preview", 1.0, 2.0) == ([], False)
