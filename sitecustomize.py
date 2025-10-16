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

    def _normalise_type(bit_generator: Any) -> str | None:
        name: str | None = None
        if hasattr(bit_generator, "__name__"):
            module = getattr(bit_generator, "__module__", "") or ""
            name = f"{module}.{bit_generator.__name__}" if module else bit_generator.__name__
        elif hasattr(bit_generator, "__qualname__"):
            name = bit_generator.__qualname__
        if name is None:
            return None
        return _normalise_name(name)

    def _compat_ctor(bit_generator: Any = "MT19937") -> Any:
        replacement: Any = bit_generator
        if isinstance(bit_generator, str):
            normalised = _normalise_name(bit_generator)
            if normalised is not None:
                replacement = normalised
        else:
            normalised = _normalise_type(bit_generator)
            if normalised is not None:
                replacement = normalised
        return original_ctor(replacement)

    _pickle.__bit_generator_ctor = _compat_ctor


_patch_numpy_bit_generators()
