"""Runtime compatibility shims for external tools invoked by the pipeline."""
from __future__ import annotations

from typing import Any

import numpy as np


def _normalise_mt19937_state(state: Any, target_name: str) -> Any:
    """Convert legacy MT19937 payloads into the structure NumPy 2 expects."""

    def _build(key_obj: Any, pos_obj: Any) -> dict[str, Any]:
        key_arr = np.asarray(key_obj, dtype=np.uint32)
        return {
            "bit_generator": target_name,
            "state": {
                "key": key_arr,
                "pos": int(pos_obj),
            },
        }

    if isinstance(state, dict):
        if state.get("bit_generator") in {"MT19937", target_name}:
            inner = state.get("state")
            if isinstance(inner, dict) and {"key", "pos"}.issubset(inner):
                return _build(inner["key"], inner["pos"])
            if isinstance(inner, (list, tuple)) and len(inner) >= 2:
                return _build(inner[0], inner[1])
        if {"key", "pos"}.issubset(state):
            return _build(state["key"], state["pos"])
    elif isinstance(state, (list, tuple)):
        payload = list(state)
        if payload and payload[0] in {"MT19937", target_name}:
            if len(payload) >= 3:
                return _build(payload[1], payload[2])
        elif len(payload) >= 2:
            return _build(payload[0], payload[1])
    return state


def _install_numpy_joblib_shims() -> None:
    try:
        from numpy.random import MT19937
        from numpy.random import _pickle as np_pickle  # type: ignore
    except Exception:
        return

    original_ctor = getattr(np_pickle, "__bit_generator_ctor", None)
    if original_ctor is None:
        return

    class _CompatMT19937(MT19937):
        def __setstate__(self, state: Any) -> None:  # type: ignore[override]
            super().__setstate__(_normalise_mt19937_state(state, type(self).__name__))

    mapping = getattr(np_pickle, "BitGenerators", None)
    if isinstance(mapping, dict):
        mapping["MT19937"] = _CompatMT19937
        mapping.setdefault("numpy.random._mt19937.MT19937", _CompatMT19937)
        mapping.setdefault("<class 'numpy.random._mt19937.MT19937'>", _CompatMT19937)

    def _compat_ctor(bit_generator: Any = "MT19937") -> Any:
        candidate = bit_generator
        if isinstance(bit_generator, str):
            if bit_generator.startswith("<class '") and bit_generator.endswith("'>"):
                candidate = bit_generator[8:-2]
            if isinstance(candidate, str) and candidate.rsplit(".", 1)[-1] == "MT19937":
                candidate = "MT19937"
        elif hasattr(bit_generator, "__name__") and bit_generator.__name__ == "MT19937":
            candidate = "MT19937"
        return original_ctor(candidate)

    np_pickle.__bit_generator_ctor = _compat_ctor


_install_numpy_joblib_shims()
