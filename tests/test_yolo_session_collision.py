"""A trial recorded more than once: YOLO processes the LATEST session only.

EB-Control-24-1 july_17_batch_1_rig_3 training_1 was recorded three times
(09:39, 10:20, 11:51). All three videos name one trial folder (yolo_infer
drops the odor and the stamp), so YOLO used to process the first in sorted
order -- the aborted 09:39 session, with no fly in view -- while the trial's
metadata (fbpipe.utils.trial_metadata.find_sidecar: most recent sidecar) came
from 11:51. With several workers two members of one group could even both see
the folder missing and write into it at once.

Now YOLO processes the session the trial's metadata is read from (the most
recent sidecar; without rig files, the latest stamp), and none while that
session has no video -- never the last name in sorted order, since the odor
label precedes the stamp (Hex-Control-24-0.01 april_17_batch_1 testing_3:
Linalool, then 3-Octonol). Every other member is skipped, keeping its job slot
so every other video's worker assignment is unchanged. Nothing is deleted.
"""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("ultralytics")

import cv2  # noqa: E402

import fbpipe.utils.track as track_module  # noqa: E402
import fbpipe.utils.video_writer as vw  # noqa: E402
from fbpipe.config import Settings  # noqa: E402
from fbpipe.steps import yolo_infer  # noqa: E402
from fbpipe.utils.trial_metadata import find_sidecar  # noqa: E402

SIDE = 1080
STEM = "output_july_17_batch_1_training_1_EthylButyrate_20260717_{}.mp4"
SESSIONS = ("093939", "102030", "115131")
OTHER = "output_july_17_batch_1_training_2_EthylButyrate_20260717_121000.mp4"
TRIAL = "july_17_batch_1_training_1"


class _FakeModel:
    def to(self, target):
        return self


class _SourceCap:
    def __init__(self, path, opened):
        opened.append(Path(path).name)

    def isOpened(self):
        return True

    def get(self, prop):
        return {
            cv2.CAP_PROP_FRAME_WIDTH: SIDE,
            cv2.CAP_PROP_FRAME_HEIGHT: SIDE,
            cv2.CAP_PROP_FPS: 40.0,
            cv2.CAP_PROP_FRAME_COUNT: 3,
        }.get(prop, 0)

    def set(self, prop, val):
        return True

    def release(self):
        pass


def _run(tmp_path, monkeypatch, *, workers=1, index=0, layout=None):
    """Run yolo_infer.main over one batch; *layout(batch)* writes its files
    (default: the three july_17 training_1 videos plus training_2, no rig files)."""
    batch = tmp_path / "EB-Control-24-1" / "july_17_batch_1_rig_3"
    batch.mkdir(parents=True, exist_ok=True)
    if layout is None:
        for name in [STEM.format(s) for s in SESSIONS] + [OTHER]:
            (batch / name).write_bytes(b"raw")
    else:
        layout(batch)
    opened: list[str] = []

    def _capture(path, *a, **k):
        return _SourceCap(path, opened)

    def _inference(cap, max_frame, target_wh, writer, timestamps, fps, anchor, *a, **k):
        return [{"frame": i, "timestamp": i / 40.0, "x_anchor": anchor[0], "y_anchor": anchor[1],
                 "source": opened[-1]} for i in range(3)]

    monkeypatch.setattr(yolo_infer, "YOLO", lambda p: _FakeModel())
    monkeypatch.setattr(yolo_infer.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(yolo_infer.cv2, "VideoCapture", _capture)
    monkeypatch.setattr(yolo_infer, "_scan_initial_fly_count", lambda *a, **k: 1)
    monkeypatch.setattr(yolo_infer, "_run_chunked_inference", _inference)
    monkeypatch.setattr(yolo_infer, "_annotated_video_failure", lambda *a, **k: None)
    monkeypatch.setattr(vw, "_default_ffmpeg_bin", lambda: None)
    monkeypatch.setattr(vw, "_nvenc_available", lambda b: False)
    monkeypatch.setattr(track_module.Track, "_next_id", 1)
    monkeypatch.setenv("NUM_WORKERS", str(workers))
    monkeypatch.setenv("WORKER_INDEX", str(index))
    cfg = Settings(model_path=str(tmp_path / "nope.pt"), main_directories=[str(batch.parent)], allow_cpu=True)
    yolo_infer.main(cfg)
    return batch, opened


def test_only_the_latest_session_of_a_trial_is_processed(tmp_path, monkeypatch, capsys):
    batch, opened = _run(tmp_path, monkeypatch)
    assert opened == [STEM.format("115131"), OTHER]
    merged = pd.read_parquet(batch / TRIAL / f"{TRIAL}_distances_merged.parquet")
    assert set(merged["source"]) == {STEM.format("115131")}
    out = capsys.readouterr().out
    for early in ("093939", "102030"):
        assert f"[YOLO] Skipping {STEM.format(early)}: superseded by {STEM.format('115131')}" in out
    # Nothing deleted: every session's video is still there.
    assert all((batch / STEM.format(s)).exists() for s in SESSIONS)


@pytest.mark.parametrize("index", [0, 1, 2])
def test_skipped_sessions_keep_their_job_slots(tmp_path, monkeypatch, index):
    """Sorted order: 093939 (job 0), 102030 (1), 115131 (2), training_2 (3).
    With 3 workers worker 0 owns jobs 0 and 3, worker 1 job 1, worker 2 job 2:
    the same assignment as before for every video that is processed."""
    _, opened = _run(tmp_path, monkeypatch, workers=3, index=index)
    assert opened == {0: [OTHER], 1: [], 2: [STEM.format("115131")]}[index]


def _session(batch: Path, stem: str, when: dt.datetime, *, video=True) -> None:
    """One recording session's files as the rig leaves them, written at *when*."""
    for ext in (".csv", ".txt") + ((".mp4",) if video else ()):
        path = batch / f"{stem}{ext}"
        path.write_bytes(b"raw")
        os.utime(path, (when.timestamp(), when.timestamp()))


APR_TRIAL = "april_17_batch_1_testing_3"
APR_EARLY = "output_april_17_batch_1_testing_3_Linalool_20260417_141350"
APR_LATE = "output_april_17_batch_1_testing_3_3-Octonol_20260417_143027"


def test_a_restart_under_another_odor_label_processes_the_session_its_metadata_reads(
    tmp_path, monkeypatch, capsys
):
    """Hex-Control-24-0.01 april_17_batch_1 testing_3: Linalool at 14:13:50,
    then 3-Octonol at 14:30:27, which sorts FIRST by name. The tables must come
    from the recording find_sidecar reads the trial's metadata from."""

    def layout(batch):
        _session(batch, APR_EARLY, dt.datetime(2026, 4, 17, 14, 16, 20))
        _session(batch, APR_LATE, dt.datetime(2026, 4, 17, 14, 32, 57))

    batch, opened = _run(tmp_path, monkeypatch, layout=layout)
    assert opened == [f"{APR_LATE}.mp4"]
    merged = pd.read_parquet(batch / APR_TRIAL / f"{APR_TRIAL}_distances_merged.parquet")
    assert set(merged["source"]) == {f"{APR_LATE}.mp4"}
    txt, csv = find_sidecar(batch / APR_TRIAL)
    assert (txt.name, csv.name) == (f"{APR_LATE}.txt", f"{APR_LATE}.csv")
    out = capsys.readouterr().out
    assert f"[YOLO] Skipping {APR_EARLY}.mp4: superseded by {APR_LATE}.mp4" in out
    assert (batch / f"{APR_EARLY}.mp4").exists()


def test_no_session_is_processed_while_the_metadata_session_has_no_video(
    tmp_path, monkeypatch, capsys
):
    """Only the aborted 09:39 video is local; the trial's metadata is the
    11:51 session's. Building the folder from 09:39 is what left EB-Control-24-1
    july_17_batch_1_rig_3 training_1 eyeless, so the trial waits instead: no
    folder, no video opened, the other trial processed as usual."""

    def layout(batch):
        _session(batch, STEM.format("093939")[: -len(".mp4")], dt.datetime(2026, 7, 17, 9, 41, 10))
        _session(batch, STEM.format("115131")[: -len(".mp4")], dt.datetime(2026, 7, 17, 11, 53, 1), video=False)
        (batch / OTHER).write_bytes(b"raw")

    batch, opened = _run(tmp_path, monkeypatch, layout=layout)
    assert opened == [OTHER]
    assert not (batch / TRIAL).exists()
    out = capsys.readouterr().out
    assert f"[YOLO] Skipping {STEM.format('093939')}: trial {TRIAL} was recorded more than once" in out
    assert (batch / STEM.format("093939")).exists()
