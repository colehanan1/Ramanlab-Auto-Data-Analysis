import pytest

import fbpipe.utils.fly_type as ft
from fbpipe.utils.fly_type import (
    GR5A_GCAMP8,
    GR5A_NEW,
    GR5A_OLD,
    UNKNOWN_FLY_TYPE,
    canonical_fly_type,
    fly_type_for_dir,
    is_known_fly_type,
    maybe_alert_new_fly_type,
    read_fly_type_raw,
)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("GR5a-Retinol", GR5A_OLD),
        ("GR5a-OLD", GR5A_OLD),
        ("GR5a-RET-old", GR5A_OLD),
        ("GR5a-Retinol-OLD", GR5A_OLD),
        ("GR5a-Retinol-Gcamp86", GR5A_GCAMP8),
        ("GR5a-RET-GCamp86", GR5A_GCAMP8),
        ("GR5a-RET-GCAMP86", GR5A_GCAMP8),
        ("GR5a-Retinol-GCamp86", GR5A_GCAMP8),
        ("GR5a-Gcamp86", GR5A_GCAMP8),
        ("GR5a-Retinol-New", GR5A_NEW),
        ("  gr5a-retinol-new  ", GR5A_NEW),  # whitespace + case insensitive
        ("", UNKNOWN_FLY_TYPE),
        (None, UNKNOWN_FLY_TYPE),
    ],
)
def test_canonical_fly_type(raw, expected):
    assert canonical_fly_type(raw) == expected


def test_read_and_resolve_from_dir(tmp_path):
    batch = tmp_path / "Hex-Training-24-0.1" / "june_01_batch_1"
    trial = batch / "june_01_batch_1_training_1"
    trial.mkdir(parents=True)
    (batch / "session_metadata.txt").write_text(
        "Subject / Session Metadata\nFly ID: 2\nFly Type: GR5a-RET-GCAMP86\nHost: x\n"
    )
    assert read_fly_type_raw(batch / "session_metadata.txt") == "GR5a-RET-GCAMP86"
    # Resolves from the per-trial dir by walking up to the batch folder.
    assert fly_type_for_dir(trial) == GR5A_GCAMP8


def test_resolve_missing_metadata(tmp_path):
    d = tmp_path / "no_meta"
    d.mkdir()
    assert fly_type_for_dir(d) == UNKNOWN_FLY_TYPE


def test_is_known_fly_type():
    assert is_known_fly_type("GR5a-Retinol-Gcamp86")
    assert is_known_fly_type("gr5a retinol gcamp86")  # separator/case insensitive
    assert not is_known_fly_type("GR5a-CantonS")
    assert not is_known_fly_type("")


def test_maybe_alert_new_fly_type_dedup(monkeypatch):
    calls = []
    monkeypatch.setattr(ft, "ntfy_notify", lambda *a, **k: calls.append(a))
    monkeypatch.setattr(ft, "_alerted_new", set())

    # Known type -> no alert
    assert maybe_alert_new_fly_type("GR5a-Retinol") is False
    # New type -> alerts once
    assert maybe_alert_new_fly_type("GR5a-CantonS") is True
    # Same new type again -> deduped (no second alert)
    assert maybe_alert_new_fly_type("GR5a-CantonS") is False
    assert len(calls) == 1


def test_new_gcamp_spelling_groups_but_alerts(monkeypatch):
    monkeypatch.setattr(ft, "_alerted_new", set())
    monkeypatch.setattr(ft, "ntfy_notify", lambda *a, **k: None)
    # A novel spelling still groups correctly via heuristic ...
    assert canonical_fly_type("GR5a-Retinol-jGCaMP8f") == GR5A_GCAMP8
    # ... but is flagged as new so it can be catalogued.
    assert maybe_alert_new_fly_type("GR5a-Retinol-jGCaMP8f") is True
