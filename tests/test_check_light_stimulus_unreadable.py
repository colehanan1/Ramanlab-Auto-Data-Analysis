"""An unreadable trial video must not crash the light-stimulus QC step.

Regression for the 2026-08-08 pipeline crash: yolo_infer left 261-byte
header-only ``*_distance_annotated.mp4`` stubs for trials whose source videos
were corrupt, and ``check_trial_light_stimulus`` raised FileNotFoundError on
them, killing the whole pipeline run inside ``parallel_map``.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from fbpipe.utils.light_stimulus import (
    check_trial_light_stimulus,
    find_trial_video,
    find_trial_video_candidates,
    sample_video,
)

# Mimics the real stubs: a valid-looking mp4 header with no decodable stream.
STUB_BYTES = bytes.fromhex("000000206674797069736f6d0000020069736f6d69736f32617663316d7034310000000866726565000000086d646174")


def _write_stub(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(STUB_BYTES)


def _write_real_video(path: Path, n_frames: int = 40, fps: float = 20.0) -> None:
    """Small decodable mp4 whose red channel steps on for the middle frames."""
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 64))
    assert writer.isOpened()
    for i in range(n_frames):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        if n_frames // 4 <= i < 3 * n_frames // 4:
            frame[..., 2] = 255
        writer.write(frame)
    writer.release()


@pytest.fixture
def trial_dir(tmp_path: Path) -> Path:
    d = tmp_path / "batch" / "august_07_batch_2_testing_9"
    d.mkdir(parents=True)
    return d


def test_stub_annotated_video_returns_unreadable_status(trial_dir: Path):
    _write_stub(trial_dir / f"{trial_dir.name}_distance_annotated.mp4")

    row = check_trial_light_stimulus(trial_dir, (0.5, 1.5), window_source="sensors")

    assert row["status"] == "unreadable_video"
    assert row["passed"] is None


def test_stub_annotated_falls_back_to_raw_sibling(trial_dir: Path):
    _write_stub(trial_dir / f"{trial_dir.name}_distance_annotated.mp4")
    raw = trial_dir.parent / f"output_{trial_dir.name}_LightOnly_20260807.mp4"
    _write_real_video(raw)

    row = check_trial_light_stimulus(trial_dir, (0.5, 1.5), window_source="sensors")

    assert row["status"] == "checked"
    assert row["video_path"] == str(raw)


def test_candidates_prefer_annotated_then_raw(trial_dir: Path):
    annotated = trial_dir / f"{trial_dir.name}_distance_annotated.mp4"
    _write_stub(annotated)
    raw = trial_dir.parent / f"output_{trial_dir.name}_LightOnly_20260807.mp4"
    _write_real_video(raw)

    assert find_trial_video_candidates(trial_dir) == [annotated, raw]
    assert find_trial_video(trial_dir) == annotated


def test_zero_sample_video_counts_as_unreadable(trial_dir: Path, monkeypatch):
    _write_stub(trial_dir / f"{trial_dir.name}_distance_annotated.mp4")
    monkeypatch.setattr("fbpipe.utils.light_stimulus.sample_video", lambda *a, **k: [])

    row = check_trial_light_stimulus(trial_dir, (0.5, 1.5), window_source="sensors")

    assert row["status"] == "unreadable_video"
    assert row["passed"] is None


def test_sample_video_still_raises_on_unopenable_file(trial_dir: Path):
    stub = trial_dir / "junk.mp4"
    _write_stub(stub)
    with pytest.raises(FileNotFoundError):
        sample_video(stub)


def test_healthy_trial_still_checked(trial_dir: Path):
    _write_real_video(trial_dir / f"{trial_dir.name}_distance_annotated.mp4")

    row = check_trial_light_stimulus(trial_dir, (0.5, 1.5), window_source="sensors")

    assert row["status"] == "checked"
    assert row["passed"] is True
