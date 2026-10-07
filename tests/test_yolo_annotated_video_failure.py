"""A trial whose annotated video failed to encode must not look processed.

Reproduced in the 2026-10-05 audit: with the GPU nearly full, ffmpeg's
h264_nvenc died mid-stream ("Failed locking bitstream buffer: invalid param",
exit 234). FFmpegFrameWriter latched dead, inference carried on, and yolo_infer
wrote complete parquets beside an annotated mp4 holding 934 of 3608 frames.
The ``out_dir.exists()`` skip-guard then treated the trial as done forever, and
check_light_stimulus preferred the truncated video.

These tests pin the fix without a GPU:
  * FFmpegFrameWriter accounts for every frame it delivers and records ffmpeg's
    exit status, driven by a fake ``ffmpeg`` that dies the same way;
  * OpenCV's mp4v fallback reports nothing (``write()`` returns None even when
    the disk is full), so its output is counted on disk;
  * ``yolo_infer.main`` discards such a trial's video (only the files it
    created; the directory only when empty) and re-runs it once, right away,
    with libx264, replaying the failed attempt's track ids;
  * when the re-run's video fails too, or no writer opens at all, the trial
    keeps its tables -- inference never depended on the writer, and secure
    cleanup prunes the raw video later in the same run, so discarding them
    would lose the trial -- and only the incomplete video is dropped.
The success path is untouched.
"""
from __future__ import annotations

import logging
import os
import shutil
import stat
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("ultralytics")

import cv2  # noqa: E402

import fbpipe.utils.track as track_module  # noqa: E402
import fbpipe.utils.video_writer as vw  # noqa: E402
from fbpipe.config import Settings  # noqa: E402
from fbpipe.steps import yolo_infer  # noqa: E402
from fbpipe.utils.video_writer import FFmpegFrameWriter  # noqa: E402

_REAL_VIDEO_CAPTURE = cv2.VideoCapture
_REAL_VIDEO_WRITER = cv2.VideoWriter

NVENC_ERROR = "[h264_nvenc @ 0x0] Failed locking bitstream buffer: invalid param (8)"

needs_shell = pytest.mark.skipif(
    os.name != "posix" or not (shutil.which("bash") and shutil.which("head") and shutil.which("cat")),
    reason="needs bash + coreutils for the fake ffmpeg",
)


def _real_ffmpeg_with_libx264():
    ffmpeg_bin = vw._default_ffmpeg_bin()
    if not ffmpeg_bin or not vw._libx264_available(ffmpeg_bin):
        pytest.skip("needs an ffmpeg with libx264")
    return ffmpeg_bin


def _fake_ffmpeg(tmp_path: Path, body: str, name: str = "ffmpeg") -> Path:
    """A stand-in ffmpeg: encoder probes (lavfi test input) succeed, and a real
    encode creates its output file (the last argument) then runs ``body``."""
    script = tmp_path / "bin" / name
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(
        "#!/bin/bash\n"
        'for a in "$@"; do [ "$a" = "lavfi" ] && exit 0; done\n'
        ': > "${@: -1}"\n'
        f"{body}\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return script


def _dies_after(tmp_path: Path, nbytes: int) -> Path:
    """Reads ``nbytes`` of frames, then dies like h264_nvenc under VRAM exhaustion."""
    return _fake_ffmpeg(
        tmp_path,
        f'head -c {nbytes} > /dev/null\necho "{NVENC_ERROR}" >&2\nexit 234',
        name="ffmpeg_dies",
    )


def _reads_all_then_fails(tmp_path: Path) -> Path:
    """Consumes every frame, then exits non-zero (e.g. the muxer failing at the end)."""
    return _fake_ffmpeg(
        tmp_path,
        'cat > /dev/null\necho "[mp4 @ 0x0] moov atom write failed" >&2\nexit 1',
        name="ffmpeg_fails_late",
    )


def _frames(n: int, size: int = 64):
    return [np.full((size, size, 3), i % 251, np.uint8) for i in range(n)]


# ---------------------------------------------------------------------------
# FFmpegFrameWriter: frame accounting and exit status
# ---------------------------------------------------------------------------

class _Stdin:
    def __init__(self, fail_after=None):
        self.fail_after = fail_after
        self.writes = 0

    def write(self, data):
        if self.fail_after is not None and self.writes >= self.fail_after:
            raise BrokenPipeError(32, "Broken pipe")
        self.writes += 1


class _Proc:
    def __init__(self, returncode=0, stderr=b"", fail_after=None, hang=False):
        self.stdin = _Stdin(fail_after)
        self.returncode = None
        self._rc, self._stderr, self._hang = returncode, stderr, hang
        self.killed = False

    def communicate(self, timeout=None):
        if self._hang:
            raise subprocess.TimeoutExpired("ffmpeg", timeout)
        self.returncode = self._rc
        return None, self._stderr

    def kill(self):
        self.killed = True


def _writer_with(proc) -> FFmpegFrameWriter:
    """A writer whose ffmpeg is ``proc``, with the state __init__ would set."""
    writer = FFmpegFrameWriter.__new__(FFmpegFrameWriter)
    writer.path = "unused.mp4"
    writer.proc = proc
    writer.encoder_used = "h264_nvenc"
    writer._dead = False
    writer.frames_written = 0
    writer.returncode = None
    writer.error = None
    return writer


def test_clean_encode_counts_every_frame_and_reports_no_error():
    writer = _writer_with(_Proc(returncode=0))
    for frame in _frames(5):
        assert writer.write(frame) is True
    writer.release()
    assert writer.frames_written == 5
    assert writer.returncode == 0
    assert writer.error is None


def test_broken_pipe_stops_the_count_and_records_why():
    writer = _writer_with(_Proc(returncode=234, stderr=NVENC_ERROR.encode(), fail_after=3))
    results = [writer.write(f) for f in _frames(6)]
    assert results == [True, True, True, False, False, False]
    assert writer.frames_written == 3
    assert writer.error and "Broken pipe" in writer.error
    writer.release()
    # The exit status and ffmpeg's own message are the useful diagnosis.
    assert writer.returncode == 234
    assert "234" in writer.error and "Failed locking bitstream buffer" in writer.error


def test_nonzero_exit_without_any_write_error_is_a_failure():
    writer = _writer_with(_Proc(returncode=1, stderr=b"[mp4 @ 0x0] moov atom write failed"))
    for frame in _frames(4):
        assert writer.write(frame) is True
    writer.release()
    assert writer.frames_written == 4
    assert writer.returncode == 1
    assert writer.error and "moov atom" in writer.error


def test_ffmpeg_that_never_exits_is_killed_and_reported():
    proc = _Proc(hang=True)
    writer = _writer_with(proc)
    writer.write(_frames(1)[0])
    writer.release()
    assert proc.killed
    assert writer.error


@needs_shell
def test_real_ffmpeg_process_dying_mid_stream_is_reported(tmp_path):
    """A real subprocess reading 5 frames and exiting 234 (what h264_nvenc did)."""
    frame_bytes = 64 * 64 * 3
    script = _dies_after(tmp_path, 5 * frame_bytes)
    out = tmp_path / "out.mp4"
    writer = FFmpegFrameWriter(out, 30, 64, 64, ffmpeg=str(script))
    assert writer.ok
    for frame in _frames(200):
        writer.write(frame)
    assert writer.ok is False
    writer.release()
    assert 5 <= writer.frames_written < 200
    assert writer.returncode == 234
    assert "Failed locking bitstream buffer" in writer.error


@needs_shell
def test_real_ffmpeg_process_failing_after_reading_everything_is_reported(tmp_path):
    script = _reads_all_then_fails(tmp_path)
    writer = FFmpegFrameWriter(tmp_path / "out.mp4", 30, 64, 64, ffmpeg=str(script))
    for frame in _frames(20):
        assert writer.write(frame) is True
    writer.release()
    assert writer.frames_written == 20
    assert writer.returncode == 1
    assert writer.error and "moov atom" in writer.error


def test_real_libx264_encode_reports_success(tmp_path):
    """Control: a clean real encode has no error and every frame is in the file."""
    _real_ffmpeg_with_libx264()
    out = tmp_path / "ok.mp4"
    writer = FFmpegFrameWriter(out, 30, 64, 64, encoder="libx264")
    assert writer.ok and writer.encoder_used == "libx264"
    for frame in _frames(12):
        assert writer.write(frame) is True
    writer.release()
    assert (writer.frames_written, writer.returncode, writer.error) == (12, 0, None)
    cap = _REAL_VIDEO_CAPTURE(str(out))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 12
    cap.release()


# ---------------------------------------------------------------------------
# yolo_infer._annotated_video_failure / _discard_failed_trial_outputs /
# _drop_failed_annotated_video
# ---------------------------------------------------------------------------

class _AccountingWriter:
    def __init__(self, frames_written, error=None):
        self.frames_written = frames_written
        self.error = error


def test_failure_check_trusts_ffmpeg_writer_accounting(tmp_path):
    mp4 = tmp_path / "x.mp4"
    assert yolo_infer._annotated_video_failure(_AccountingWriter(10), mp4, 10) is None
    assert "exited 234" in yolo_infer._annotated_video_failure(
        _AccountingWriter(10, error="ffmpeg exited 234: boom"), mp4, 10
    )
    assert "7 of 10" in yolo_infer._annotated_video_failure(_AccountingWriter(7), mp4, 10)


def _write_mp4v(path: Path, n_frames: int, size=(64, 48)):
    writer = _REAL_VIDEO_WRITER(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, size)
    assert writer.isOpened()
    for i in range(n_frames):
        writer.write(np.full((size[1], size[0], 3), i * 20 % 251, np.uint8))
    writer.release()
    return writer


def test_failure_check_counts_opencv_output_on_disk(tmp_path):
    """cv2.VideoWriter.write() returns None whatever happens, so the frames
    that actually reached the mp4 are what gets checked."""
    mp4 = tmp_path / "x.mp4"
    writer = _write_mp4v(mp4, 6)
    assert yolo_infer._annotated_video_failure(writer, mp4, 6) is None
    assert "6 of 8" in yolo_infer._annotated_video_failure(writer, mp4, 8)

    unreadable = tmp_path / "no_moov.mp4"
    unreadable.write_bytes(b"\x00" * 4096)
    assert "0 of 8" in yolo_infer._annotated_video_failure(writer, unreadable, 8)


def test_discard_failed_trial_outputs_removes_video_and_empty_dir(tmp_path):
    out_dir = tmp_path / "october_05_batch_1_training_13"
    out_dir.mkdir()
    out_mp4 = out_dir / "october_05_batch_1_training_13_distance_annotated.mp4"
    out_mp4.write_bytes(b"\x00" * 1000)

    yolo_infer._discard_failed_trial_outputs(out_dir, out_mp4, "video.mp4", "ffmpeg exited 234")

    assert not out_mp4.exists()
    assert not out_dir.exists()


def test_discard_failed_trial_outputs_keeps_dir_with_other_files(tmp_path):
    out_dir = tmp_path / "october_05_batch_1_training_13"
    out_dir.mkdir()
    out_mp4 = out_dir / "october_05_batch_1_training_13_distance_annotated.mp4"
    out_mp4.write_bytes(b"\x00" * 1000)
    other = out_dir / "_trial_meta.json"
    other.write_text("{}")

    yolo_infer._discard_failed_trial_outputs(out_dir, out_mp4, "video.mp4", "ffmpeg exited 234")

    assert not out_mp4.exists()
    assert out_dir.exists() and other.exists()


def test_drop_failed_annotated_video_removes_only_the_video(tmp_path):
    """Once no complete video can be had, only the video goes: the trial folder
    stays, empty or not, because the tables are written into it next."""
    out_dir = tmp_path / "october_05_batch_1_training_13"
    out_dir.mkdir()
    out_mp4 = out_dir / "october_05_batch_1_training_13_distance_annotated.mp4"
    out_mp4.write_bytes(b"\x00" * 1000)

    yolo_infer._drop_failed_annotated_video(out_mp4, "video.mp4", "ffmpeg exited 234", retried=True)

    assert not out_mp4.exists()
    assert out_dir.is_dir()
    yolo_infer._drop_failed_annotated_video(out_mp4, "video.mp4", "no writer", retried=False)  # nothing left: no error


# ---------------------------------------------------------------------------
# yolo_infer.main end to end (no GPU, no model, no real source video)
# ---------------------------------------------------------------------------

N_FRAMES = 6
SIDE = 1080  # main() always writes 1080x1080 frames
VIDEO_NAME = "output_october_05_batch_1_training_13_Hexanol-1_20261005_111818.mp4"
TRIAL = "october_05_batch_1_training_13"
MERGED = f"{TRIAL}_distances_merged.parquet"
ANNOTATED = f"{TRIAL}_distance_annotated.mp4"
NO_VIDEO_SUMMARY = "trial(s) kept their tables without an annotated video"


class _FakeModel:
    def to(self, target):
        return self


class _SourceCap:
    """The raw trial video; frames come from the fake inference below."""

    def __init__(self, *a, **k):
        pass

    def isOpened(self):
        return True

    def get(self, prop):
        return {
            cv2.CAP_PROP_FRAME_WIDTH: SIDE,
            cv2.CAP_PROP_FRAME_HEIGHT: SIDE,
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_COUNT: N_FRAMES,
        }.get(prop, 0)

    def set(self, prop, val):
        return True

    def release(self):
        pass


def _video_capture(path, *a, **k):
    # The annotated output is read back for real; the source video is faked.
    if str(path).endswith("_distance_annotated.mp4"):
        return _REAL_VIDEO_CAPTURE(str(path), *a, **k)
    return _SourceCap()


def _fake_inference(cap, max_frame, target_wh, writer, timestamps, fps, anchor, *a, **k):
    """Every frame spawns a tracker Track, the way real inference advances the
    process-global Track id counter that the tables' track-id columns carry."""
    rows = []
    for i in range(N_FRAMES):
        writer.write(np.full((SIDE, SIDE, 3), 40 * i % 251, np.uint8))
        track = track_module.Track(np.array([100.0, 100.0, 10.0, 10.0], np.float32), 0.9)
        rows.append({"frame": i, "timestamp": i / fps, "x_anchor": anchor[0], "y_anchor": anchor[1],
                     "track_id": track.id})
    return rows


def _run_main(tmp_path, monkeypatch, *, ffmpeg_bin, videos=(VIDEO_NAME,)):
    """One yolo_infer.main run over a batch of ``videos``; repeatable, like nightly runs.
    Track ids restart at 1, as in a fresh worker process."""
    batch = tmp_path / "Some-Dataset" / "october_05_batch_1"
    batch.mkdir(parents=True, exist_ok=True)
    for name in videos:
        if not (batch / name).exists():
            (batch / name).write_bytes(b"raw")

    monkeypatch.setattr(yolo_infer, "YOLO", lambda p: _FakeModel())
    monkeypatch.setattr(yolo_infer.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(yolo_infer.cv2, "VideoCapture", _video_capture)
    monkeypatch.setattr(yolo_infer, "_scan_initial_fly_count", lambda *a, **k: 1)
    monkeypatch.setattr(yolo_infer, "_run_chunked_inference", _fake_inference)
    monkeypatch.setattr(vw, "_default_ffmpeg_bin", lambda: ffmpeg_bin)
    # Never open a real NVENC session from a test.
    monkeypatch.setattr(vw, "_nvenc_available", lambda b: False)
    monkeypatch.setattr(track_module.Track, "_next_id", 1)

    cfg = Settings(model_path=str(tmp_path / "nope.pt"), main_directories=[str(batch.parent)], allow_cpu=True)
    yolo_infer.main(cfg)
    return batch / TRIAL, batch / VIDEO_NAME


def _tables(batch: Path) -> dict:
    return {p.relative_to(batch).as_posix(): pd.read_parquet(p) for p in sorted(batch.rglob("*.parquet"))}


def _clean_run_tables(tmp_path, monkeypatch, capsys, caplog, videos=(VIDEO_NAME,)) -> dict:
    """The tables of the same batch run where every annotated video encodes fine
    (OpenCV's mp4v writer), keyed by path relative to the batch folder."""
    out_dir, _ = _run_main(tmp_path / "clean", monkeypatch, ffmpeg_bin=None, videos=videos)
    tables = _tables(out_dir.parent)
    assert tables, "the clean run must write tables"
    capsys.readouterr()
    caplog.clear()
    return tables


def _frame_count(mp4: Path) -> int:
    cap = _REAL_VIDEO_CAPTURE(str(mp4))
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()


def _assert_kept_without_video(out_dir: Path, video: Path, caplog, capsys, clean: dict, *, failure: str) -> str:
    """No complete annotated video could be made: the trial keeps exactly the
    tables a clean run writes, the incomplete video is gone, and it says so once."""
    assert out_dir.is_dir(), "the trial folder (and its tables) must survive"
    assert sorted(p.name for p in out_dir.iterdir()) == [MERGED], "only the tables: no annotated video"
    assert _tables(out_dir.parent).keys() == clean.keys()
    for name, table in clean.items():
        pd.testing.assert_frame_equal(pd.read_parquet(out_dir.parent / name), table)
    assert video.read_bytes() == b"raw", "the source video must never be touched"
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR and VIDEO_NAME in r.getMessage()]
    assert len(errors) == 1, [r.getMessage() for r in errors]
    out = capsys.readouterr().out
    assert out.count(f"[YOLO] ERROR: {VIDEO_NAME}: annotated video {failure}") == 1, out
    assert out.count("tables kept, no annotated video") == 1, out
    assert f"[YOLO] ERROR: 1 {NO_VIDEO_SUMMARY}: {VIDEO_NAME}" in out, out
    assert f"[YOLO] {VIDEO_NAME} → " in out, "the tables are reported written"
    return out


@needs_shell
def test_main_keeps_tables_when_ffmpeg_dies_mid_stream_on_both_attempts(tmp_path, monkeypatch, caplog, capsys):
    clean = _clean_run_tables(tmp_path, monkeypatch, capsys, caplog)
    script = _dies_after(tmp_path, 2 * SIDE * SIDE * 3)
    caplog.set_level(logging.WARNING, logger="fbpipe.yolo")
    out_dir, video = _run_main(tmp_path, monkeypatch, ffmpeg_bin=str(script))
    out = _assert_kept_without_video(out_dir, video, caplog, capsys, clean, failure="FAILED again (")
    assert f"[YOLO] {VIDEO_NAME}: annotated video FAILED (" in out and "re-running it with libx264" in out
    assert "Failed locking bitstream buffer" in out


@needs_shell
def test_main_keeps_tables_when_ffmpeg_fails_at_release_on_both_attempts(tmp_path, monkeypatch, caplog, capsys):
    clean = _clean_run_tables(tmp_path, monkeypatch, capsys, caplog)
    script = _reads_all_then_fails(tmp_path)
    caplog.set_level(logging.WARNING, logger="fbpipe.yolo")
    out_dir, video = _run_main(tmp_path, monkeypatch, ffmpeg_bin=str(script))
    out = _assert_kept_without_video(out_dir, video, caplog, capsys, clean, failure="FAILED again (")
    assert "moov atom write failed" in out


def test_main_keeps_trial_when_encode_succeeds(tmp_path, monkeypatch, capsys):
    """Control: the success path still writes the video and the tables, quietly."""
    ffmpeg_bin = _real_ffmpeg_with_libx264()
    out_dir, video = _run_main(tmp_path, monkeypatch, ffmpeg_bin=ffmpeg_bin)
    assert (out_dir / ANNOTATED).stat().st_size > 0
    assert (out_dir / MERGED).exists()
    assert _frame_count(out_dir / ANNOTATED) == N_FRAMES
    out = capsys.readouterr().out
    assert "FAILED" not in out and NO_VIDEO_SUMMARY not in out


class _WriterDyingAfter:
    """Pure-Python stand-in for FFmpegFrameWriter whose encoder dies after
    ``keep`` frames. Like ffmpeg it creates the output file up front, and like
    the real writer it accounts for itself: later writes fail, and release()
    reports the exit status."""

    keep = 3

    def __init__(self, path, *a, **k):
        Path(path).write_bytes(b"\x00" * 4096)  # the partial mp4 ffmpeg leaves
        self.ok = True
        self.frames_written = 0
        self.returncode = None
        self.error = None

    def write(self, frame):
        if not self.ok:
            return False
        if self.frames_written == self.keep:
            self.ok = False
            self.error = f"pipe write failed after {self.frames_written} frames: [Errno 32] Broken pipe"
            return False
        self.frames_written += 1
        return True

    def release(self):
        self.returncode = 234
        self.error = f"ffmpeg exited 234 after {self.frames_written} frames: {NVENC_ERROR}"


def test_trial_kept_without_video_is_not_reprocessed_by_the_next_run(tmp_path, monkeypatch, capsys):
    """Keeping the tables is final: the skip-guard treats the trial as processed,
    so a later run neither redoes inference for a review video nor touches the
    tables (to get the video, delete the trial folder while the raw video is there)."""
    monkeypatch.setattr(yolo_infer, "FFmpegFrameWriter", _WriterDyingAfter)
    out_dir, video = _run_main(tmp_path, monkeypatch, ffmpeg_bin=None)
    before = (out_dir / MERGED).read_bytes()
    capsys.readouterr()

    monkeypatch.setattr(yolo_infer, "FFmpegFrameWriter", FFmpegFrameWriter)
    _run_main(tmp_path, monkeypatch, ffmpeg_bin=None)
    assert f"[YOLO] Skipping {VIDEO_NAME}" in capsys.readouterr().out
    assert (out_dir / MERGED).read_bytes() == before
    assert not (out_dir / ANNOTATED).exists()


class _UnopenableWriter:
    constructed = 0

    def __init__(self, *a, **k):
        type(self).constructed += 1

    def isOpened(self):
        return False

    def release(self):
        pass


def test_main_keeps_tables_when_no_video_writer_opens(tmp_path, monkeypatch, caplog, capsys):
    """No usable ffmpeg and the OpenCV writer will not open: inference still runs
    and the tables are kept, without a video and without a re-run (nothing would
    change). This used to leave an empty folder the skip-guard honoured forever."""
    clean = _clean_run_tables(tmp_path, monkeypatch, capsys, caplog)
    monkeypatch.setattr(_UnopenableWriter, "constructed", 0)
    monkeypatch.setattr(yolo_infer.cv2, "VideoWriter", _UnopenableWriter)
    caplog.set_level(logging.WARNING, logger="fbpipe.yolo")
    out_dir, video = _run_main(tmp_path, monkeypatch, ffmpeg_bin=None)
    assert _UnopenableWriter.constructed == 1
    out = _assert_kept_without_video(out_dir, video, caplog, capsys, clean, failure="FAILED (could not open")
    assert "re-running" not in out


class _DiskFullAfter:
    """A real mp4v VideoWriter that silently stops landing frames after
    ``keep`` of them, which is what OpenCV does when the disk fills up
    (only a "Failed to write frame" warning; write() still returns None)."""

    keep = 2

    def __init__(self, *a, **k):
        self._w = _REAL_VIDEO_WRITER(*a, **k)
        self._n = 0

    def isOpened(self):
        return self._w.isOpened()

    def write(self, frame):
        self._n += 1
        if self._n <= self.keep:
            self._w.write(frame)

    def release(self):
        self._w.release()


def test_main_keeps_tables_when_mp4v_fallback_drops_frames_on_both_attempts(tmp_path, monkeypatch, caplog, capsys):
    clean = _clean_run_tables(tmp_path, monkeypatch, capsys, caplog)
    monkeypatch.setattr(yolo_infer.cv2, "VideoWriter", _DiskFullAfter)
    caplog.set_level(logging.WARNING, logger="fbpipe.yolo")
    out_dir, video = _run_main(tmp_path, monkeypatch, ffmpeg_bin=None)
    out = _assert_kept_without_video(out_dir, video, caplog, capsys, clean, failure="FAILED again (")
    assert f"holds {_DiskFullAfter.keep} of {N_FRAMES} frames" in out


def test_main_keeps_trial_when_mp4v_fallback_succeeds(tmp_path, monkeypatch):
    out_dir, video = _run_main(tmp_path, monkeypatch, ffmpeg_bin=None)
    assert (out_dir / MERGED).exists()
    assert _frame_count(out_dir / ANNOTATED) == N_FRAMES


# ---------------------------------------------------------------------------
# Re-run in the same run, with libx264
#
# Secure cleanup prunes every local raw video that has a secured copy, later
# in the same run and whether or not its trial was processed, so a retry left
# to "the next run" would usually find no video. A failed trial is therefore
# re-run once, right away, with libx264 (a CPU encode; NVENC is what dies when
# VRAM runs out); when that fails too, the trial keeps its tables (above).
# ---------------------------------------------------------------------------

def _writer_factory(constructed, *, fails=lambda trial, encoder: encoder == "auto", dying=_WriterDyingAfter):
    """Stands in for FFmpegFrameWriter: records (trial, encoder) for every writer
    main() asks for, and hands back a ``dying`` writer when ``fails`` says so,
    else the real one."""

    def make(path, *a, **k):
        trial, encoder = Path(path).parent.name, k.get("encoder", "auto")
        constructed.append((trial, encoder))
        if fails(trial, encoder):
            return dying(path)
        return FFmpegFrameWriter(path, *a, **k)

    return make


def test_main_reruns_a_failed_trial_once_with_libx264(tmp_path, monkeypatch, caplog, capsys):
    ffmpeg_bin = _real_ffmpeg_with_libx264()
    clean = _clean_run_tables(tmp_path, monkeypatch, capsys, caplog)
    constructed = []
    monkeypatch.setattr(yolo_infer, "FFmpegFrameWriter", _writer_factory(constructed))
    caplog.set_level(logging.WARNING, logger="fbpipe.yolo")
    out_dir, video = _run_main(tmp_path, monkeypatch, ffmpeg_bin=ffmpeg_bin)

    assert constructed == [(TRIAL, "auto"), (TRIAL, "libx264")]
    assert _frame_count(out_dir / ANNOTATED) == N_FRAMES
    # The re-run's tables are the clean run's, track ids included.
    pd.testing.assert_frame_equal(pd.read_parquet(out_dir / MERGED), clean[f"{TRIAL}/{MERGED}"])
    out = capsys.readouterr().out
    assert f"[YOLO] {VIDEO_NAME}: annotated video FAILED" in out and "libx264" in out
    assert f"[YOLO] {VIDEO_NAME} → " in out
    assert NO_VIDEO_SUMMARY not in out
    # Recovered, so a warning rather than an error.
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert [r for r in caplog.records if r.levelno == logging.WARNING and VIDEO_NAME in r.getMessage()]


def test_main_keeps_tables_when_the_rerun_fails_too(tmp_path, monkeypatch, caplog, capsys):
    clean = _clean_run_tables(tmp_path, monkeypatch, capsys, caplog)
    constructed = []
    monkeypatch.setattr(yolo_infer, "FFmpegFrameWriter", _writer_factory(constructed, fails=lambda t, e: True))
    caplog.set_level(logging.WARNING, logger="fbpipe.yolo")
    out_dir, video = _run_main(tmp_path, monkeypatch, ffmpeg_bin=None)

    assert constructed == [(TRIAL, "auto"), (TRIAL, "libx264")]  # one re-run, never a loop
    _assert_kept_without_video(out_dir, video, caplog, capsys, clean, failure="FAILED again (")


TRIALS = [f"october_05_batch_1_training_{n}" for n in (11, 12, 13)]
VIDEOS = tuple(f"output_{t}_Hexanol-1_20261005_111818.mp4" for t in TRIALS)


def test_rerun_replays_track_ids_so_every_table_matches_a_clean_run(tmp_path, monkeypatch, caplog, capsys):
    """The failed attempt advanced the process-global Track id counter; the
    re-run rewinds it, so the re-run's tables and every later trial's equal a
    clean run's, track-id columns included."""
    ffmpeg_bin = _real_ffmpeg_with_libx264()
    clean = _clean_run_tables(tmp_path, monkeypatch, capsys, caplog, videos=VIDEOS)
    constructed = []
    monkeypatch.setattr(
        yolo_infer, "FFmpegFrameWriter",
        _writer_factory(constructed, fails=lambda t, e: t == TRIALS[0] and e == "auto"),
    )
    out_dir, _ = _run_main(tmp_path, monkeypatch, ffmpeg_bin=ffmpeg_bin, videos=VIDEOS)

    assert constructed[:2] == [(TRIALS[0], "auto"), (TRIALS[0], "libx264")]
    got = _tables(out_dir.parent)
    assert got.keys() == clean.keys() and len(got) == len(TRIALS)
    for name, table in clean.items():
        pd.testing.assert_frame_equal(got[name], table)
    assert got[f"{TRIALS[2]}/{TRIALS[2]}_distances_merged.parquet"]["track_id"].tolist() == list(
        range(2 * N_FRAMES + 1, 3 * N_FRAMES + 1)
    )


class _WriterDyingBesideAStrayFile(_WriterDyingAfter):
    """...while something else lands in the trial folder, so it cannot be removed."""

    def __init__(self, path, *a, **k):
        super().__init__(path, *a, **k)
        Path(path).with_name("stray.json").write_text("{}")


def test_rerun_runs_even_when_the_trial_folder_could_not_be_removed(tmp_path, monkeypatch, capsys):
    """The re-run is this run's own retry: the skip-guard must not mistake the
    folder it could not remove for a processed trial."""
    ffmpeg_bin = _real_ffmpeg_with_libx264()
    constructed = []
    monkeypatch.setattr(
        yolo_infer, "FFmpegFrameWriter", _writer_factory(constructed, dying=_WriterDyingBesideAStrayFile)
    )
    out_dir, _ = _run_main(tmp_path, monkeypatch, ffmpeg_bin=ffmpeg_bin)

    assert constructed == [(TRIAL, "auto"), (TRIAL, "libx264")]
    assert "Skipping" not in capsys.readouterr().out
    assert (out_dir / MERGED).exists() and (out_dir / "stray.json").exists()
    assert _frame_count(out_dir / ANNOTATED) == N_FRAMES


def test_rerun_keeps_the_worker_slice(tmp_path, monkeypatch, capsys):
    """NUM_WORKERS/WORKER_INDEX slicing counts each video once: re-running worker
    0's first video must not shift which videos are worker 0's."""
    ffmpeg_bin = _real_ffmpeg_with_libx264()
    constructed = []
    monkeypatch.setattr(
        yolo_infer, "FFmpegFrameWriter",
        _writer_factory(constructed, fails=lambda t, e: t == TRIALS[0] and e == "auto"),
    )
    monkeypatch.setenv("NUM_WORKERS", "2")
    monkeypatch.setenv("WORKER_INDEX", "0")
    monkeypatch.setattr(yolo_infer.cv2, "setNumThreads", lambda n: None)
    monkeypatch.setattr(yolo_infer.torch, "set_num_threads", lambda n: None)
    out_dir, _ = _run_main(tmp_path, monkeypatch, ffmpeg_bin=ffmpeg_bin, videos=VIDEOS)
    batch = out_dir.parent

    assert constructed == [(TRIALS[0], "auto"), (TRIALS[0], "libx264"), (TRIALS[2], "auto")]
    assert (batch / TRIALS[0] / f"{TRIALS[0]}_distances_merged.parquet").exists()
    assert not (batch / TRIALS[1]).exists()  # worker 1's video
    assert (batch / TRIALS[2] / f"{TRIALS[2]}_distances_merged.parquet").exists()
