"""FFmpegFrameWriter latches dead after the first broken-pipe write."""

import numpy as np
import pytest

import fbpipe.utils.video_writer as vw
from fbpipe.utils.video_writer import FFmpegFrameWriter


class _DeadStdin:
    def __init__(self):
        self.write_calls = 0

    def write(self, _data):
        self.write_calls += 1
        raise BrokenPipeError("[Errno 32] Broken pipe")


class _FakeProc:
    def __init__(self):
        self.stdin = _DeadStdin()


def test_write_latches_dead_after_first_broken_pipe():
    writer = FFmpegFrameWriter.__new__(FFmpegFrameWriter)
    writer.path = "unused.mp4"
    writer.proc = _FakeProc()
    writer.encoder_used = "libx264"
    writer._dead = False

    frame = np.zeros((4, 4, 3), dtype=np.uint8)

    assert writer.ok
    assert writer.write(frame) is False
    assert writer.proc.stdin.write_calls == 1

    # Further writes must not touch the dead pipe again.
    assert writer.ok is False
    assert writer.write(frame) is False
    assert writer.write(frame) is False
    assert writer.proc.stdin.write_calls == 1


def test_init_skips_ffmpeg_missing_both_encoders(monkeypatch):
    """An ffmpeg with neither h264_nvenc nor libx264 (e.g. a bare conda base
    env shadowing the intended one on $PATH) must not be spawned at all --
    that's exactly the "-preset" unrecognized-option / broken-pipe-per-frame
    failure mode this guards against."""
    monkeypatch.setattr(vw, "_default_ffmpeg_bin", lambda: "/fake/ffmpeg")
    monkeypatch.setattr(vw, "_nvenc_available", lambda ffmpeg_bin: False)
    monkeypatch.setattr(vw, "_libx264_available", lambda ffmpeg_bin: False)

    def _boom(*args, **kwargs):
        raise AssertionError("Popen must not be called when no usable encoder was found")

    monkeypatch.setattr(vw.subprocess, "Popen", _boom)

    writer = FFmpegFrameWriter("out.mp4", fps=30, width=64, height=64)

    assert writer.ok is False
    assert writer.proc is None


def test_init_starts_when_libx264_available(monkeypatch):
    monkeypatch.setattr(vw, "_default_ffmpeg_bin", lambda: "/fake/ffmpeg")
    monkeypatch.setattr(vw, "_nvenc_available", lambda ffmpeg_bin: False)
    monkeypatch.setattr(vw, "_libx264_available", lambda ffmpeg_bin: True)

    captured_cmd = {}

    class _FakePopen:
        def __init__(self, cmd, **kwargs):
            captured_cmd["cmd"] = cmd
            self.stdin = object()

    monkeypatch.setattr(vw.subprocess, "Popen", _FakePopen)

    writer = FFmpegFrameWriter("out.mp4", fps=30, width=64, height=64)

    assert writer.ok is True
    assert "-preset" in captured_cmd["cmd"]
