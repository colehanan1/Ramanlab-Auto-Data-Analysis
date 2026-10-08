"""A trial recorded more than once: YOLO processes the LATEST session only.

EB-Control-24-1 july_17_batch_1_rig_3 training_1 was recorded three times
(09:39, 10:20, 11:51). All three videos name one trial folder (yolo_infer
drops the odor and the stamp), so YOLO used to process the first in sorted
order -- the aborted 09:39 session, with no fly in view -- while the trial's
metadata (fbpipe.utils.trial_metadata.find_sidecar: most recent sidecar) came
from 11:51. With several workers two members of one group could even both see
the folder missing and write into it at once.

Now the latest session (last in sorted order; the stamp is in the name) is the
real one: every earlier member is skipped, keeping its job slot so every other
video's worker assignment is unchanged. Nothing is deleted.
"""

from __future__ import annotations

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


def _run(tmp_path, monkeypatch, *, workers=1, index=0):
    batch = tmp_path / "EB-Control-24-1" / "july_17_batch_1_rig_3"
    batch.mkdir(parents=True, exist_ok=True)
    for name in [STEM.format(s) for s in SESSIONS] + [OTHER]:
        (batch / name).write_bytes(b"raw")
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
