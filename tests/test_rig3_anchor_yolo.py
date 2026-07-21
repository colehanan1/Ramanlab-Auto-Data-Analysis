"""yolo_infer must write angles against the per-rig anchor.

rig_3 is physically mirrored, so its anchor sits on the opposite edge from
every other rig's. The anchor used to be hoisted out of the per-video loop
(``AX, AY = cfg.anchor_x, cfg.anchor_y`` once per ``main()`` call), so a
single run spanning rig_2 and rig_3 applied one rig's anchor to the other
rig's videos.

This is a hermetic, behavioural regression test: it drives ``yolo_infer.main``
end to end (no GPU, no TensorRT, no real video) over two fake batch roots --
one rig_2, one rig_3 -- and asserts the anchor recorded for each video is
correct. A source-inspection test (checking that ``resolve_anchor`` merely
*appears* in the module, or that its call site is lexically after the loop
header) cannot tell a correct implementation from one that resolves the
anchor into a throwaway variable and then still uses ``cfg.anchor_x``, one
that resolves it too late (crashing with ``UnboundLocalError``), one that
resolves from the wrong path component, one with the tuple order swapped, or
one where the call is dead code inside a never-taken branch -- all five of
which pass a substring/position check. Driving ``main()`` and recording what
``_run_chunked_inference`` actually receives catches all of them.
"""
from __future__ import annotations

import pytest

pytest.importorskip("ultralytics")

from fbpipe.config import Settings
from fbpipe.steps import yolo_infer


class _FakeModel:
    def to(self, target):
        return self


class _FakeCap:
    def __init__(self, *a, **k):
        pass

    def isOpened(self):
        return True

    def get(self, prop):
        import cv2
        return {
            cv2.CAP_PROP_FRAME_WIDTH: 1080,
            cv2.CAP_PROP_FRAME_HEIGHT: 1080,
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_COUNT: 10,
        }.get(prop, 0)

    def set(self, prop, val):
        return True

    def release(self):
        pass


class _FakeWriter:
    ok = True

    def __init__(self, *a, **k):
        pass

    def release(self):
        pass


def test_main_resolves_anchor_per_video(tmp_path, monkeypatch):
    """A single main() call spanning rig_2 and rig_3 must apply each rig's
    own anchor to its own videos, not one rig's anchor to both."""
    seen = []

    rig3 = tmp_path / "july_17_batch_2_rig_3" / "fly_1"
    rig2 = tmp_path / "july_17_batch_1_rig_2" / "fly_1"
    for d in (rig3, rig2):
        d.mkdir(parents=True)
        (d / "output_a_b_c_d_e_f_g.mp4").write_bytes(b"")

    monkeypatch.setattr(yolo_infer, "YOLO", lambda p: _FakeModel())
    monkeypatch.setattr(yolo_infer.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(yolo_infer.cv2, "VideoCapture", _FakeCap)
    monkeypatch.setattr(yolo_infer, "FFmpegFrameWriter", _FakeWriter)
    monkeypatch.setattr(yolo_infer, "_scan_initial_fly_count", lambda *a, **k: 1)
    monkeypatch.setattr(yolo_infer, "_export_per_fly_csvs", lambda *a, **k: [])
    monkeypatch.setattr(yolo_infer, "write_table", lambda *a, **k: None)

    def _fake_infer(cap, max_frame, target_wh, writer, timestamps, fps, anchor, *a, **k):
        seen.append(anchor)
        return []

    monkeypatch.setattr(yolo_infer, "_run_chunked_inference", _fake_infer)

    cfg = Settings(
        model_path=str(tmp_path / "nope.pt"),
        main_directories=[str(rig3.parent), str(rig2.parent)],
        allow_cpu=True,
    )

    yolo_infer.main(cfg)

    # sorted(roots) -> "..._batch_1_rig_2" sorts before "..._batch_2_rig_3"
    assert seen == [(1080.0, 540.0), (0.0, 540.0)], seen
