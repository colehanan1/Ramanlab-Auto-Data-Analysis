"""Runtime compatibility shims for external tools invoked by the pipeline."""
from __future__ import annotations

import importlib
from typing import Any

import numpy as np


def _normalise_mt19937_state(state: Any, target_name: str) -> Any:
    """Coerce legacy MT19937 payloads into a NumPy 2.x-friendly shape."""
    try:
        np_major = int(np.__version__.split(".")[0])
    except Exception:
        np_major = 0
    if np_major < 2:
        return state

    if isinstance(state, dict):
        payload = state.get("state") or state
        if isinstance(payload, dict) and {"key", "pos"}.issubset(payload):
            return {
                "bit_generator": target_name,
                "state": {
                    "key": np.asarray(payload["key"], dtype=np.uint32),
                    "pos": int(payload["pos"]),
                },
            }
    return state


def _install_numpy_joblib_shims() -> None:
    """Register MT19937 compatibility hooks when NumPy 2.x is active."""
    try:
        np_major = int(np.__version__.split(".")[0])
    except Exception:
        np_major = 0
    if np_major < 2:
        # NumPy 1.x already deserialises legacy MT19937 payloads correctly.
        return

    try:
        np_pickle = importlib.import_module("numpy.random._pickle")
    except ModuleNotFoundError:
        return

    original_ctor = getattr(np_pickle, "__bit_generator_ctor", None)
    if original_ctor is None:
        return

    class _CompatMT19937(np.random.MT19937):
        def __setstate__(self, state: Any) -> None:  # type: ignore[override]
            super().__setstate__(_normalise_mt19937_state(state, type(self).__name__))

    mapping = getattr(np_pickle, "BitGenerators", None)
    if isinstance(mapping, dict):
        mapping["MT19937"] = _CompatMT19937

    def _compat_ctor(bit_generator: Any = "MT19937") -> Any:  # noqa: ANN401 - NumPy shim
        if bit_generator in {"MT19937", np.random.MT19937, _CompatMT19937}:
            return original_ctor("MT19937")
        return original_ctor(bit_generator)

    np_pickle.__bit_generator_ctor = _compat_ctor


_install_numpy_joblib_shims()
