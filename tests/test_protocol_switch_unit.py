"""Fast unit tests pinning the legacy/v2 behavior of envelope_combined.

These drive the protocol gating: each asserts that ``protocol: legacy``
reproduces the v1 rule while ``protocol: v2`` keeps current behavior. The
autouse fixture in conftest.py restores the global protocol after each test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))

from scripts.analysis import envelope_combined as ec  # noqa: E402
from scripts.analysis import envelope_visuals as ev  # noqa: E402
from _protocol_fixtures import build_envelope_fixture, MEASURE_COL, _trace  # noqa: E402

# legacy hardcoded windows (31.5 s / 61.5 s @ 40 fps) vs v2 config fallback
# (30 s / 60 s -> 1200 / 2400) for this sidecar-less fixture.
LEGACY_BEFORE, LEGACY_DURING_END = 1260, 2460
V2_BEFORE, V2_DURING_END = 1200, 2400


def _wide(tmp_path: Path, protocol: str) -> pd.DataFrame:
    dataset_root = build_envelope_fixture(tmp_path / protocol)
    ev.set_protocol(protocol)
    out = tmp_path / f"wide_{protocol}.csv"
    ec.build_wide_csv([str(dataset_root)], str(out), measure_cols=[MEASURE_COL])
    return pd.read_csv(out)


def _row(df: pd.DataFrame, fly: str, trial_prefix: str) -> pd.Series:
    sub = df[(df["fly"] == fly) & (df["trial_label"].str.startswith(trial_prefix))]
    assert len(sub) == 1, f"expected exactly one {fly}/{trial_prefix} row, got {len(sub)}"
    return sub.iloc[0]


# --- Site 1/2/4: odor-window boundary (the dominant fix) -------------------

def test_before_window_boundary_is_1260_under_legacy_and_1200_under_v2(tmp_path):
    legacy = _row(_wide(tmp_path, "legacy"), "october_01_fly1", "testing_1")
    v2 = _row(_wide(tmp_path, "v2"), "october_01_fly1", "testing_1")
    trace = _trace(0, "testing", 1)
    assert legacy["local_max_before"] == pytest.approx(float(np.nanmax(trace[:LEGACY_BEFORE])))
    assert v2["local_max_before"] == pytest.approx(float(np.nanmax(trace[:V2_BEFORE])))
    # The two boundaries genuinely differ for this ramp (sanity that the test bites).
    assert legacy["local_max_before"] != pytest.approx(v2["local_max_before"])


def test_during_window_boundary_differs_by_protocol(tmp_path):
    legacy = _row(_wide(tmp_path, "legacy"), "october_01_fly1", "testing_1")
    v2 = _row(_wide(tmp_path, "v2"), "october_01_fly1", "testing_1")
    trace = _trace(0, "testing", 1)
    assert legacy["local_max_during"] == pytest.approx(
        float(np.nanmax(trace[LEGACY_BEFORE:LEGACY_DURING_END]))
    )
    assert v2["local_max_during"] == pytest.approx(
        float(np.nanmax(trace[V2_BEFORE:V2_DURING_END]))
    )


def test_legacy_build_does_not_call_trial_metadata(tmp_path, monkeypatch):
    """Under legacy, build_wide_csv must never consult the sidecar resolver."""

    calls: list[int] = []

    def _spy(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("trial-metadata must not be consulted under legacy")

    monkeypatch.setattr(ec, "_resolve_trial_meta_for_csv", _spy)
    dataset_root = build_envelope_fixture(tmp_path)
    ev.set_protocol("legacy")
    out = tmp_path / "wide.csv"
    ec.build_wide_csv([str(dataset_root)], str(out), measure_cols=[MEASURE_COL])
    assert pd.read_csv(out).shape[0] == 6
    assert calls == [], "legacy path must not invoke _resolve_trial_meta_for_csv"


# --- Site 3: per-trial seconds schema --------------------------------------

_TRIAL_SECONDS_COLS = (
    "trial_odor_on_s",
    "trial_odor_off_s",
    "trial_duration_s",
    "trial_light_on_s",
)


def test_wide_schema_omits_trial_seconds_under_legacy(tmp_path):
    cols = list(_wide(tmp_path, "legacy").columns)
    for col in _TRIAL_SECONDS_COLS:
        assert col not in cols
    # trace_len is immediately followed by trial_type (v1 column order).
    assert cols[cols.index("trace_len") + 1] == "trial_type"


def test_wide_schema_includes_trial_seconds_under_v2(tmp_path):
    cols = list(_wide(tmp_path, "v2").columns)
    for col in _TRIAL_SECONDS_COLS:
        assert col in cols


# --- Site 4: trial-label regex ---------------------------------------------

_AMBIGUOUS_STEM = Path(
    "october_01_fly1_testing_3_Benzaldehyde_fly1_distances_fly1_angle_distance_rms_envelope.csv"
)


def test_trial_label_strips_fly_suffix_under_v2():
    ev.set_protocol("v2")
    assert ec._trial_label(_AMBIGUOUS_STEM) == "testing_3_Benzaldehyde"


def test_trial_label_keeps_greedy_suffix_under_legacy():
    ev.set_protocol("legacy")
    assert (
        ec._trial_label(_AMBIGUOUS_STEM)
        == "testing_3_Benzaldehyde_fly1_distances_fly1_angle_distance_rms_envelope"
    )


# --- Site 5: trial-type override -------------------------------------------

def test_infer_category_override_ignored_under_legacy(monkeypatch):
    from fbpipe.config import DatasetOverride

    path = Path("/data/RandomPanel/batch_1/x_training_2_fly1_distances.csv")
    monkeypatch.setattr(ec, "_RUNTIME_SETTINGS", object())  # non-None sentinel
    monkeypatch.setattr(
        ec, "get_dataset_override", lambda cfg, p: DatasetOverride(trial_type_override="testing")
    )

    ev.set_protocol("v2")
    assert ec._infer_category(path) == "testing"  # override honored

    ev.set_protocol("legacy")
    assert ec._infer_category(path) == "training"  # heuristic only; override ignored
