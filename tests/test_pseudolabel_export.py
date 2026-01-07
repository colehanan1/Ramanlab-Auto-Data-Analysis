from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fbpipe.utils.pseudolabel import (
    mine_top_confidence_frames,
    write_yolo_bbox_label_file,
    xyxy_to_yolo_norm,
)


def test_xyxy_to_yolo_norm_known_box() -> None:
    cx, cy, w, h = xyxy_to_yolo_norm((10, 20, 30, 60), width_px=100, height_px=200)
    assert abs(cx - 0.2) < 1e-9
    assert abs(cy - 0.2) < 1e-9
    assert abs(w - 0.2) < 1e-9
    assert abs(h - 0.2) < 1e-9


def test_mine_top_confidence_frames_deterministic(tmp_path: Path) -> None:
    fly_dir = tmp_path / "fly1"
    fly_dir.mkdir(parents=True, exist_ok=True)
    video_path = fly_dir / "toy.avi"

    w, h, n_frames = 64, 64, 20
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (w, h))
    assert writer.isOpened()
    for i in range(n_frames):
        frame = np.full((h, w, 3), i, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    class FakeBoxes:
        def __init__(self, xyxy: np.ndarray, cls: np.ndarray, conf: np.ndarray):
            self.xyxy = xyxy
            self.cls = cls
            self.conf = conf

        def __len__(self) -> int:
            return int(self.xyxy.shape[0])

    class FakeResult:
        def __init__(self, boxes: FakeBoxes):
            self.boxes = boxes
            self.obb = None

    def predict_fn(frames, conf_thres: float):
        out = []
        for frame in frames:
            idx = int(frame[0, 0, 0])
            eye_conf = 0.50 + 0.01 * idx
            prob_conf = 0.60 + 0.01 * idx
            xyxy = np.array([[0, 0, 10, 10], [5, 5, 15, 15]], dtype=np.float32)
            cls = np.array([2, 8], dtype=np.float32)
            conf = np.array([eye_conf, prob_conf], dtype=np.float32)
            out.append(FakeResult(FakeBoxes(xyxy=xyxy, cls=cls, conf=conf)))
        return out

    sel1, _ = mine_top_confidence_frames(
        [video_path],
        predict_fn,
        stride=0,
        random_sample_per_video=6,
        batch_size=3,
        target_total=4,
        per_video_cap=10,
        min_conf_keep=0.0,
        seed=1337,
    )
    sel2, _ = mine_top_confidence_frames(
        [video_path],
        predict_fn,
        stride=0,
        random_sample_per_video=6,
        batch_size=3,
        target_total=4,
        per_video_cap=10,
        min_conf_keep=0.0,
        seed=1337,
    )

    assert [c.frame_idx for c in sel1] == [c.frame_idx for c in sel2]
    assert [c.unique_key for c in sel1] == [c.unique_key for c in sel2]

    # Returned list is sorted by selection_score descending.
    scores = [c.selection_score for c in sel1]
    assert scores == sorted(scores, reverse=True)


def test_write_yolo_bbox_label_file_emits_expected_format(tmp_path: Path) -> None:
    txt_path = tmp_path / "img.txt"
    write_yolo_bbox_label_file(
        txt_path,
        [(0, (0, 0, 10, 10)), (1, (5, 5, 15, 15))],
        width_px=20,
        height_px=20,
    )

    assert txt_path.exists()
    lines = txt_path.read_text(encoding="utf-8").strip().splitlines()
    assert lines == [
        "0 0.250000 0.250000 0.500000 0.500000",
        "1 0.500000 0.500000 0.500000 0.500000",
    ]

