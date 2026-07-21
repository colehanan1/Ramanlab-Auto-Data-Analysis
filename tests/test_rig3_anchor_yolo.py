"""yolo_infer must write angles against the per-rig anchor.

The anchor is currently hoisted out of the per-video loop, so a single run
covering both rigs would apply one rig's anchor to the other's videos.
"""
from __future__ import annotations

import inspect

from fbpipe.steps import yolo_infer
from fbpipe.utils.rig_anchor import resolve_anchor


def test_yolo_infer_imports_resolve_anchor():
    src = inspect.getsource(yolo_infer)
    assert "resolve_anchor" in src, "yolo_infer must resolve the anchor per rig"


def test_anchor_is_resolved_inside_the_video_loop():
    """AX, AY must be assigned after the per-video loop starts."""
    src = inspect.getsource(yolo_infer.main)
    assert "for video_path in video_files:" in src
    loop_at = src.index("for video_path in video_files:")
    anchor_at = src.index("resolve_anchor(")
    assert anchor_at > loop_at, "anchor must be resolved per video, not once per run"


def test_rig_3_video_path_resolves_mirrored_anchor():
    p = "/data/EB-Training-24-1/july_17_batch_2_rig_3/output_x.mp4"
    assert resolve_anchor(p) == (0.0, 540.0)
