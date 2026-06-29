"""The combined-category cache must include the active protocol.

Switching legacy<->v2 changes the wide CSV / matrix rules, so a legacy run
after a v2 run (or vice versa) must NOT hash-skip and reuse the wrong figures.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.pipeline import run_workflows as rw  # noqa: E402


def _expected(protocol: str) -> dict:
    # Empty dataset_roots -> trivial (matching) combined manifest, so the only
    # thing that can change the skip decision is the protocol field.
    return {
        "non_reactive_span_px": 5.0,
        "class2_min": 70.0,
        "class2_max": 250.0,
        "dataset_roots": [],
        "protocol": protocol,
    }


def test_combined_cache_hits_when_protocol_unchanged(tmp_path):
    settings = types.SimpleNamespace(cache_dir=str(tmp_path / "cache"))
    rw._write_state(settings, "combined", "analysis", _expected("v2"))
    assert (
        rw._should_skip_with_manifest(
            settings, "combined", "analysis", _expected("v2"), force_flag=False
        )
        is True
    )


def test_combined_cache_busts_when_protocol_changes(tmp_path):
    settings = types.SimpleNamespace(cache_dir=str(tmp_path / "cache"))
    rw._write_state(settings, "combined", "analysis", _expected("v2"))
    # Same inputs but legacy protocol -> must re-run (no skip).
    assert (
        rw._should_skip_with_manifest(
            settings, "combined", "analysis", _expected("legacy"), force_flag=False
        )
        is False
    )
