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
