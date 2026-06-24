"""Tests for the opt-in parallel_map helper (determinism + serial fallback)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fbpipe.utils.parallel import parallel_map, resolve_n_jobs


def _square(x: int) -> int:  # module-level so it is picklable by loky
    return x * x


def test_resolve_n_jobs_auto():
    cpu = os.cpu_count() or 1
    assert resolve_n_jobs(0) == max(1, cpu - 2)
    assert resolve_n_jobs(-5) == max(1, cpu - 2)


def test_resolve_n_jobs_explicit_capped():
    cpu = os.cpu_count() or 1
    assert resolve_n_jobs(1) == 1
    assert resolve_n_jobs(10_000) == cpu


def test_empty_returns_empty():
    assert parallel_map(_square, [], enabled=True, n_jobs=4) == []


def test_serial_when_disabled_preserves_order():
    items = list(range(10))
    assert parallel_map(_square, items, enabled=False) == [i * i for i in items]


def test_parallel_matches_serial_and_order():
    items = list(range(50))
    expected = [i * i for i in items]
    serial = parallel_map(_square, items, enabled=False)
    parallel = parallel_map(_square, items, enabled=True, n_jobs=4)
    assert serial == expected
    assert parallel == expected  # same values AND same order as serial
