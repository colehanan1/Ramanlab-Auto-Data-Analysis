"""Tests for _make_batched_predict_fn: OOM backoff, CPU fallback, re-raise."""
import pytest; pytest.importorskip("ultralytics")

from src.fbpipe.steps.yolo_infer import _make_batched_predict_fn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeResult:
    """Minimal stand-in for ultralytics Results."""


def _make_results(n):
    return [_FakeResult() for _ in range(n)]


# ---------------------------------------------------------------------------
# Test 1: OOM halves sub-batch size, eventually succeeds
# ---------------------------------------------------------------------------

def test_oom_backoff_halves_then_succeeds():
    """Fake model raises on len(images) > 2; succeeds at ≤2. Full batch=4 → two sub-batches of 2."""
    call_log = []

    class FakeModel:
        def predict(self, images, conf, verbose, device, half, **kw):
            call_log.append(len(images))
            if len(images) > 2:
                raise RuntimeError("CUDA out of memory. Tried to allocate 100 MiB")
            return _make_results(len(images))

    fn = _make_batched_predict_fn(
        FakeModel(),
        get_device=lambda: "cuda",
        set_device=lambda t: None,
        allow_cpu=False,
        cuda_empty_cache=lambda: None,
    )

    images = [object() for _ in range(4)]
    results = fn(images, conf_thres=0.5)

    assert len(results) == 4, f"Expected 4 results, got {len(results)}"
    # First call was the full batch of 4
    assert call_log[0] == 4, f"First call should be full batch (4), got {call_log[0]}"
    # All subsequent calls (after backoff) were ≤ 2
    assert all(n <= 2 for n in call_log[1:]), (
        f"All retried sub-batch calls should be ≤ 2, got {call_log[1:]}"
    )


# ---------------------------------------------------------------------------
# Test 2: OOM at batch=1 falls back to CPU when allow_cpu=True
# ---------------------------------------------------------------------------

def test_oom_at_batch1_falls_back_to_cpu_when_allowed():
    """Fake model raises on CUDA (any size); succeeds on CPU. Verify device switch and half=False."""
    cpu_call_kwargs = {}
    dev = {"cur": "cuda"}

    class FakeModel:
        def predict(self, images, conf, verbose, device, half, **kw):
            if device == "cuda":
                raise RuntimeError("CUDA out of memory")
            # CPU path: record kwargs
            cpu_call_kwargs["device"] = device
            cpu_call_kwargs["half"] = half
            return _make_results(len(images))

    fn = _make_batched_predict_fn(
        FakeModel(),
        get_device=lambda: dev["cur"],
        set_device=lambda t: dev.__setitem__("cur", t),
        allow_cpu=True,
        cuda_empty_cache=lambda: None,
    )

    images = [object() for _ in range(2)]
    results = fn(images, conf_thres=0.3)

    assert dev["cur"] == "cpu", f"Device should have switched to 'cpu', got {dev['cur']}"
    assert len(results) == 2, f"Expected 2 results, got {len(results)}"
    assert cpu_call_kwargs.get("half") is False, (
        f"CPU fallback predict must use half=False, got half={cpu_call_kwargs.get('half')}"
    )


# ---------------------------------------------------------------------------
# Test 3: OOM at batch=1 re-raises when allow_cpu=False
# ---------------------------------------------------------------------------

def test_oom_reraises_at_batch1_without_cpu():
    """Fake model always raises on cuda; allow_cpu=False → RuntimeError propagates."""

    class FakeModel:
        def predict(self, images, conf, verbose, device, half, **kw):
            raise RuntimeError("CUDA out of memory")

    fn = _make_batched_predict_fn(
        FakeModel(),
        get_device=lambda: "cuda",
        set_device=lambda t: None,
        allow_cpu=False,
        cuda_empty_cache=lambda: None,
    )

    with pytest.raises(RuntimeError):
        fn([object()], conf_thres=0.5)


# ===========================================================================
# Task 5 Step 1: Parity test — chunked inference == sequential reference.
#
# Drives a sequential reference and ``_run_chunked_inference`` with the SAME
# stubbed detections and INDEPENDENTLY-seeded collaborators, then asserts the
# emitted rows are bit-identical at every batch size. This is the correctness
# proof for the whole frame-batching feature.
# ===========================================================================

import numpy as np

from fbpipe.config import Settings
from fbpipe.steps import yolo_infer
from fbpipe.steps.yolo_infer import (
    SINGLE_TRACK_CLASSES,
    _build_proboscis_tracker,
    _process_frame,
    _run_chunked_inference,
)
from fbpipe.utils.multi_fly import EyeAnchorManager, StablePairing
from fbpipe.utils.track import SingleClassTracker
from fbpipe.utils import track as track_module

# Full-size frame so detection coords (hundreds of px) and the frame-size-scaled
# pairer rebind radius are compatible. Pixel VALUE is the frame index i, kept
# small (N <= 8) so i fits in a uint8 -- that is the only size constraint.
H = W = 1080
N_FRAMES = 8
AX, AY = 540.0, 540.0


class _CannedResult:
    """Opaque per-frame stand-in for an Ultralytics Results object."""

    def __init__(self, frame_index: int):
        self.frame_index = frame_index


# One canned result per frame index. The chunked stub and the sequential
# reference both look these up by frame index, so they consume the EXACT
# same result objects.
CANNED = [_CannedResult(i) for i in range(N_FRAMES)]


def _detections_for_frame(i: int):
    """Deterministic per-frame detections so tracking evolves reproducibly.

    Eyes drift a little each frame; the proboscis sits between the two eyes.
    Same mapping is used by reference and chunked runs (keyed by frame index),
    so any divergence in output rows is purely an ordering/state bug in the
    helper -- which is exactly what this test must catch.
    """
    drift = float(i)
    eye_boxes = np.array(
        [
            [100.0 + drift, 100.0, 110.0 + drift, 110.0],
            [400.0 - drift, 100.0, 410.0 - drift, 110.0],
        ],
        dtype=np.float32,
    )
    eye_scores = np.array([0.95, 0.90], dtype=np.float32)
    prob_box = np.array([[250.0, 100.0, 260.0, 110.0]], dtype=np.float32)
    prob_scores = np.array([0.98], dtype=np.float32)
    return {
        0: {"boxes": eye_boxes, "scores": eye_scores},
        1: {"boxes": prob_box, "scores": prob_scores},
    }


def _install_collect_detections(monkeypatch):
    """Map a canned result -> deterministic detections keyed by frame index."""

    def fake_collect(result, classes):
        return _detections_for_frame(result.frame_index)

    monkeypatch.setattr(yolo_infer, "collect_detections", fake_collect)


def _make_settings() -> Settings:
    return Settings(
        model_path="model.pt",
        main_directories="/tmp/videos",
        max_flies=2,
        conf_thres=0.40,
        use_optical_flow=False,  # keep deterministic; prev_gray still threaded
    )


def _make_collaborators(cfg: Settings, active_max_flies: int):
    """Build a fresh, independently-seeded collaborator set exactly as main() does.

    Caller MUST reset ``track_module.Track._next_id = 1`` immediately before
    calling this, otherwise the process-global track id counter leaks across
    sets and ``cls8_*_track_id`` diverges between reference and chunked runs.
    """
    single_trackers = {
        cls: SingleClassTracker(
            iou_thres=cfg.iou_match_thres, max_age=cfg.max_age, ema_alpha=cfg.ema_alpha
        )
        for cls in SINGLE_TRACK_CLASSES
    }
    eye_mgr = EyeAnchorManager(max_eyes=active_max_flies, zero_iou_eps=cfg.zero_iou_epsilon)
    cls8_tracker = _build_proboscis_tracker(cfg, active_max_flies)
    pairer = StablePairing(max_pairs=active_max_flies)
    pairer.rebind_max_dist_px = cfg.pair_rebind_ratio * np.hypot(W, H)
    return single_trackers, eye_mgr, cls8_tracker, pairer


class _FakeVideoCapture:
    """Yields N frames where frame i = np.full((H, W, 3), i, uint8)."""

    def __init__(self, n_frames: int):
        self._n = n_frames
        self._i = 0

    def isOpened(self):
        return self._i < self._n

    def read(self):
        if self._i >= self._n:
            return False, None
        frame = np.full((H, W, 3), self._i, dtype=np.uint8)
        self._i += 1
        return True, frame

    def release(self):
        pass


class _CollectingWriter:
    """Captures written frames so we can ignore them; rows are what we compare."""

    def __init__(self):
        self.frames = []

    def write(self, frame):
        self.frames.append(frame)

    def release(self):
        pass


def _batched_predict_stub(frames, conf):
    """One canned result per frame, keyed off the frame's pixel value."""
    return [CANNED[int(f[0, 0, 0])] for f in frames]


def _rows_equal(a: dict, b: dict) -> bool:
    """NaN-aware per-key equality: equal iff same keys and each value matches
    (both NaN counts as equal)."""
    if a.keys() != b.keys():
        return False
    for k in a:
        va, vb = a[k], b[k]
        a_nan = isinstance(va, float) and np.isnan(va)
        b_nan = isinstance(vb, float) and np.isnan(vb)
        if a_nan and b_nan:
            continue
        if a_nan != b_nan:
            return False
        if va != vb:
            return False
    return True


def _build_reference_rows(cfg: Settings, active_max_flies: int):
    """Sequential reference: loop _process_frame over the same frames in order,
    threading prev_gray, consuming the SAME canned results."""
    track_module.Track._next_id = 1
    single_trackers, eye_mgr, cls8_tracker, pairer = _make_collaborators(cfg, active_max_flies)
    rows = []
    prev_gray = None
    fps = 30.0
    for i in range(N_FRAMES):
        frame = np.full((H, W, 3), i, dtype=np.uint8)
        ts = i / fps
        frame, row, prev_gray = _process_frame(
            frame,
            i,
            ts,
            single_trackers,
            prev_gray,
            (AX, AY),
            cfg,
            CANNED[i],
            eye_mgr,
            cls8_tracker,
            pairer,
            active_max_flies,
        )
        rows.append(row)
    return rows


@pytest.mark.parametrize("batch_size", [1, 3, 8, N_FRAMES])
def test_chunked_inference_matches_sequential_reference(monkeypatch, batch_size):
    """_run_chunked_inference must produce rows bit-identical to the sequential
    reference at every batch size: 1 (serial), 3 (partial last batch),
    8 (exact multiple == N), N (single chunk)."""
    _install_collect_detections(monkeypatch)
    cfg = _make_settings()
    active_max_flies = 2
    fps = 30.0
    timestamps = {}  # falls back to frame_idx / fps inside the helper

    # Build the single sequential reference (its own seeded collaborators).
    reference_rows = _build_reference_rows(cfg, active_max_flies)
    assert len(reference_rows) == N_FRAMES

    # Build a fresh chunked run: fresh cap, fresh writer, fresh collaborators,
    # and crucially reset the process-global track id counter first.
    track_module.Track._next_id = 1
    single_trackers, eye_mgr, cls8_tracker, pairer = _make_collaborators(cfg, active_max_flies)
    cap = _FakeVideoCapture(N_FRAMES)
    writer = _CollectingWriter()

    chunked_rows = _run_chunked_inference(
        cap,
        N_FRAMES - 1,           # max_frame: per-frame cap (last decodable index)
        (W, H),                 # target_wh == frame size -> exercises .copy() path
        writer,
        timestamps,
        fps,
        (AX, AY),
        cfg,
        _batched_predict_stub,
        single_trackers,
        None,                   # prev_gray starts None
        eye_mgr,
        cls8_tracker,
        pairer,
        active_max_flies,
        batch_size,
    )

    assert len(chunked_rows) == N_FRAMES, (
        f"batch_size={batch_size}: expected {N_FRAMES} rows, got {len(chunked_rows)}"
    )
    assert len(writer.frames) == N_FRAMES, (
        f"batch_size={batch_size}: writer received {len(writer.frames)} frames, "
        f"expected {N_FRAMES}"
    )
    for i, (ref, got) in enumerate(zip(reference_rows, chunked_rows)):
        assert _rows_equal(ref, got), (
            f"batch_size={batch_size}: row {i} diverged from sequential reference\n"
            f"  reference={ref}\n  chunked  ={got}"
        )
