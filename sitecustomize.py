"""Runtime compatibility shims for external tools invoked by the pipeline."""
from __future__ import annotations

import contextlib


def _patch_numpy_bit_generators() -> None:
    """Allow NumPy 2.x to load joblib artifacts built on NumPy 1.x."""
    try:
        from numpy.random import _bit_generator  # type: ignore
        from numpy.random import _pickle  # type: ignore
    except Exception:
        return

    name = "numpy.random._mt19937.MT19937"
    if name in getattr(_pickle, "BIT_GENERATOR_NAME_TO_TYPE", {}):
        return

    with contextlib.suppress(Exception):
        _pickle.BIT_GENERATOR_NAME_TO_TYPE[name] = _bit_generator.MT19937


_patch_numpy_bit_generators()
