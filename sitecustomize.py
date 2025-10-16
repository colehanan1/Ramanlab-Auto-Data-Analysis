"""Runtime compatibility shims for external tools invoked by the pipeline."""
from __future__ import annotations

from typing import Any, Callable


def _patch_numpy_bit_generators() -> None:
    """Allow NumPy 2.x to load joblib artifacts built on NumPy 1.x."""

    try:
        from numpy.random import _pickle  # type: ignore
    except Exception:
        return

    original_ctor: Callable[[Any], Any] | None = getattr(_pickle, "__bit_generator_ctor", None)
    mapping: dict[str, Any] | None = getattr(_pickle, "BitGenerators", None)

    if original_ctor is None:
        return

    if mapping and "MT19937" in mapping:
        target = mapping["MT19937"]
        mapping.setdefault("numpy.random._mt19937.MT19937", target)
        mapping.setdefault("<class 'numpy.random._mt19937.MT19937'>", target)

    def _normalise_name(value: str) -> str | None:
        cleaned = value.strip()
        if cleaned.startswith("<class '") and cleaned.endswith("'>"):
            cleaned = cleaned[8:-2]
        if "." in cleaned:
            cleaned = cleaned.split(".")[-1]
        if mapping and cleaned in mapping:
            return cleaned
        return None

    def _compat_ctor(bit_generator: Any = "MT19937") -> Any:
        if isinstance(bit_generator, str):
            normalised = _normalise_name(bit_generator)
            if normalised is not None:
                bit_generator = normalised
        return original_ctor(bit_generator)

    _pickle.__bit_generator_ctor = _compat_ctor


_patch_numpy_bit_generators()
