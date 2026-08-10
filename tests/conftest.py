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


@pytest.fixture(autouse=True)
def _isolate_dataset_odor_remap():
    """Snapshot/restore the global per-dataset odor remap around each test.

    Same hazard as the protocol above: ``_DATASET_ODOR_REMAP`` is module-global,
    and any test that loads a real config (a figure driver whose ``--config``
    defaults to ``config_new.yaml``, say) installs that config's remap for the
    rest of the session. A later test then sees ``Linalool`` renamed to
    ``Isoamyl Acetate`` and fails for a reason that has nothing to do with it —
    and only when run after the polluter, never on its own.
    """
    saved = {ds: dict(m) for ds, m in _ev._DATASET_ODOR_REMAP.items()}
    try:
        yield
    finally:
        _ev._DATASET_ODOR_REMAP.clear()
        _ev._DATASET_ODOR_REMAP.update(saved)
