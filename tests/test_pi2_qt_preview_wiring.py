#!/usr/bin/env python3
"""Pi 2 ``--preview-mode qt`` wiring in ``combinedv2_1.py``.

Watching the MJPEG stream in a browser *on the Pi itself* froze the rig
during manual training (per-frame 1080² colour conversion + JPEG encode +
Chromium decode on top of software H.264). The fix is a GPU-composited
Picamera2 QTGL preview with a 1 Hz ``set_overlay`` timing layer driven by
the same ``PhaseState`` the web page reads. Verified standalone by
``PiCode/qtgl_preview_smoketest.py`` (40.04 fps sustained, 2026-08-11).

The module can't be imported off the rig (module-level argparse + lgpio +
picamera2), so these tests inspect the source with ``ast``, like
``test_pi1_light_digital_only.py``.
"""

import ast
import os
import re

PI2 = os.path.join(os.path.dirname(__file__), "..", "PiCode", "combinedv2_1.py")

with open(PI2, encoding="utf-8") as _fh:
    SOURCE = _fh.read()
TREE = ast.parse(SOURCE)


def _functions():
    return {n.name: n for n in ast.walk(TREE) if isinstance(n, ast.FunctionDef)}


def _call_name(node):
    f = node.func
    parts = []
    while isinstance(f, ast.Attribute):
        parts.append(f.attr)
        f = f.value
    if isinstance(f, ast.Name):
        parts.append(f.id)
    return ".".join(reversed(parts))


def _calls_in(node):
    return [n for n in ast.walk(node) if isinstance(n, ast.Call)]


def _preview_mode_choices():
    for call in _calls_in(TREE):
        if _call_name(call) == "parser.add_argument" and call.args \
                and isinstance(call.args[0], ast.Constant) \
                and call.args[0].value == "--preview-mode":
            for kw in call.keywords:
                if kw.arg == "choices":
                    return [c.value for c in kw.value.elts]
    raise AssertionError("--preview-mode argument not found")


# ── flag ─────────────────────────────────────────────────────────────────────

def test_preview_mode_accepts_qt():
    assert "qt" in _preview_mode_choices()


def test_preview_mode_default_is_still_mjpeg():
    """qt is opt-in: Opto runs and remote monitoring keep the web preview."""
    for call in _calls_in(TREE):
        if _call_name(call) == "parser.add_argument" and call.args \
                and isinstance(call.args[0], ast.Constant) \
                and call.args[0].value == "--preview-mode":
            defaults = [kw.value.value for kw in call.keywords if kw.arg == "default"]
            assert defaults == ["mjpeg"]
            return
    raise AssertionError("--preview-mode argument not found")


# ── start_preview: QTGL → QT → MJPEG fallback ────────────────────────────────

def test_start_preview_has_qt_branch_with_qtgl_then_qt():
    fn = _functions()["start_preview"]
    src = ast.get_source_segment(SOURCE, fn)
    assert "Preview.QTGL" in src, "qt branch must try the GPU QTGL preview first"
    assert "Preview.QT" in src.replace("Preview.QTGL", ""), \
        "qt branch must fall back to the software QT preview"
    assert src.index("Preview.QTGL") < src.index("Preview.QT,") if "Preview.QT," in src \
        else src.index("Preview.QTGL") < src.rindex("Preview.QT"), "QTGL is tried before QT"


def test_start_preview_qt_returns_qt_backend():
    fn = _functions()["start_preview"]
    returns = [n.value.value for n in ast.walk(fn)
               if isinstance(n, ast.Return) and isinstance(n.value, ast.Constant)]
    assert "qt" in returns
    assert "mjpeg" in returns  # final fallback still exists


# ── overlay thread ───────────────────────────────────────────────────────────

def test_overlay_thread_function_exists_and_uses_phase_snapshot_and_set_overlay():
    fns = _functions()
    assert "_qt_overlay_loop" in fns
    names = {_call_name(c) for c in _calls_in(fns["_qt_overlay_loop"])}
    assert "PHASE.snapshot" in names, "overlay must read the in-process PhaseState"
    assert "overlay_lines" in names, "layout comes from phase_status.overlay_lines"
    assert any(n.endswith("set_overlay") for n in names)


def test_overlay_loop_never_propagates_exceptions():
    """A rendering error must never touch the recording loop."""
    fn = _functions()["_qt_overlay_loop"]
    handlers = [h for h in ast.walk(fn) if isinstance(h, ast.ExceptHandler)]
    assert handlers, "overlay loop needs a try/except around its body"


def test_overlay_thread_is_started_only_for_qt_backend():
    m = re.search(r'if preview_backend == "qt":\s*\n\s*threading\.Thread\(target=_qt_overlay_loop',
                  SOURCE)
    assert m, "main() must start _qt_overlay_loop only when the qt backend is live"


def test_overlay_lines_is_imported_from_phase_status():
    for node in ast.walk(TREE):
        if isinstance(node, ast.ImportFrom) and node.module == "phase_status":
            assert "overlay_lines" in {a.name for a in node.names}
            return
    raise AssertionError("phase_status import not found")


# ── recording-loop relief ────────────────────────────────────────────────────

def test_recorder_throttles_preview_frame_conversion():
    """The per-frame YUV→BGR + copy only feeds the MJPEG preview; at 40 fps
    it is pure waste. It must be gated to every Nth frame."""
    fn = _functions()["video_recorder"]
    src = ast.get_source_segment(SOURCE, fn)
    assert "PREVIEW_FRAME_STRIDE" in src
    # the conversion inside the while loop is guarded by the stride
    loop = next(n for n in ast.walk(fn) if isinstance(n, ast.While))
    guarded = False
    for node in ast.walk(loop):
        if isinstance(node, ast.If):
            test_src = ast.get_source_segment(SOURCE, node.test)
            body_src = "\n".join(ast.get_source_segment(SOURCE, b) for b in node.body)
            if "PREVIEW_FRAME_STRIDE" in test_src and "COLOR_YUV2BGR_I420" in body_src:
                guarded = True
    assert guarded, "cvtColor in the recorder loop must be inside the stride guard"


def test_preview_frame_stride_is_sane():
    m = re.search(r"^PREVIEW_FRAME_STRIDE\s*=\s*(\d+)", SOURCE, re.M)
    assert m, "PREVIEW_FRAME_STRIDE must be a module-level constant"
    assert 2 <= int(m.group(1)) <= 10


# ── renderer: Hershey fonts are ASCII-only ───────────────────────────────────

def _load_renderer():
    """Exec just the pure renderer helpers out of the module (it can't be imported)."""
    import numpy as np
    import cv2
    fns = _functions()
    subs = next(n for n in TREE.body if isinstance(n, ast.Assign)
                and any(getattr(t, "id", "") == "_HERSHEY_SUBS" for t in n.targets))
    code = "\n".join(ast.get_source_segment(SOURCE, n)
                     for n in (subs, fns["_ascii_for_hershey"], fns["_render_qt_overlay"]))
    ns = {"np": np, "cv2": cv2}
    exec(code, ns)  # noqa: S102 — test-only, source is our own file
    return ns


def test_renderer_folds_non_ascii_before_drawing():
    ns = _load_renderer()
    assert ns["_ascii_for_hershey"]("Training 2 — Hexanol ON") == "Training 2 - Hexanol ON"
    assert ns["_ascii_for_hershey"]("✓") == "DONE"


def test_renderer_output_shape_and_banner_colour():
    ns = _load_renderer()
    ov = ns["_render_qt_overlay"](720, "Training 2 — Hexanol ON", "3.0 s", True)
    assert ov.shape == (720, 720, 4) and ov.dtype.name == "uint8"
    assert tuple(ov[5, 5]) == (200, 0, 0, 200)      # red banner
    assert ov[719, 719, 3] == 0                      # transparent below the band
    plain = ns["_render_qt_overlay"](720, "x", "01:30", False)
    assert tuple(plain[5, 5]) == (0, 0, 0, 150)      # translucent black band
