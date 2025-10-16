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


def _patch_numpy_mt19937_state() -> None:
    """Relax MT19937 state validation for artifacts saved on NumPy 1.x."""

    try:
        from numpy.random import bit_generator as bitgen  # type: ignore
    except Exception:
        return

    original_setstate = getattr(bitgen.BitGenerator, "__setstate__", None)
    if original_setstate is None:
        return

    def _normalise_mt_state(state: Any) -> Any:
        if isinstance(state, (list, tuple)) and len(state) == 5 and state[0] == "MT19937":
            key, pos, has_gauss, cached_gaussian = state[1:]
            return {
                "bit_generator": "MT19937",
                "state": {
                    "key": key,
                    "pos": int(pos),
                },
            }

        if not isinstance(state, dict):
            return state
        if state.get("bit_generator") != "MT19937":
            return state

        inner = state.get("state")
        if not isinstance(inner, dict):
            if isinstance(inner, (list, tuple)):
                # Handle artifacts that store only the MT19937 payload.
                payload = _normalise_mt_state(("MT19937",) + tuple(inner))
                if isinstance(payload, dict):
                    return payload
            return state

        converted = {
            "bit_generator": "MT19937",
            "state": {
                "key": inner.get("key"),
                "pos": int(inner.get("pos", 0)),
            },
        }

        key = converted["state"].get("key")
        if key is None:
            return state

        return converted

    def _compat_setstate(self: Any, state: Any) -> None:  # type: ignore[override]
        try:
            original_setstate(self, state)
        except ValueError as exc:
            if "legacy MT19937" not in str(exc):
                raise
            normalised = _normalise_mt_state(state)
            if normalised is state:
                raise
            try:
                original_setstate(self, normalised)
            except ValueError as inner_exc:
                raise exc from inner_exc

    bitgen.BitGenerator.__setstate__ = _compat_setstate  # type: ignore[assignment]


_patch_numpy_bit_generators()
_patch_numpy_mt19937_state()
