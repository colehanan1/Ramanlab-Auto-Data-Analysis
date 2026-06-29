"""Shared pytest fixtures for the test suite.

The experiment ``protocol`` ("legacy" | "v2") is module-global state in
``scripts.analysis.envelope_visuals`` (``_ACTIVE_PROTOCOL``). Because the
protocol-switch tests mutate it, an autouse fixture snapshots and restores it
around every test so a test that selects a protocol cannot leak into the next
one. The restored baseline is the import-time default ("legacy"), which matches
production when ``config.yaml`` has no ``protocol:`` key.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# Allow `import _protocol_fixtures` from the tests/ directory.
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))

from scripts.analysis import envelope_visuals as _ev  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate_protocol():
    """Snapshot/restore the global experiment protocol around each test."""
    saved = _ev.get_protocol()
    try:
        yield
    finally:
        _ev.set_protocol(saved)
