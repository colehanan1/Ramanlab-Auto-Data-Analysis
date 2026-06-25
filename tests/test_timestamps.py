"""Regression tests for to_seconds_series timestamp-unit handling.

Guards the pandas>=2 microsecond-resolution bug: sub-second ISO timestamps parse
as datetime64[us], and the old ``dt.astype("int64")/1e9`` undershot elapsed
seconds by 1000x (fps computed as ~40020 instead of ~40).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fbpipe.utils.timestamps import to_seconds_series


def _iso_series(base_iso: str, n: int, step_us: int, fmt: str) -> list[str]:
    """n consistently-formatted ISO timestamps, step_us apart (no second-rollover gaps)."""
    base = pd.Timestamp(base_iso)
    dt = pd.to_datetime([base + pd.Timedelta(microseconds=i * step_us) for i in range(n)], utc=True)
    return list(dt.strftime(fmt))


def test_utc_iso_microsecond_elapsed_seconds_not_milliscaled():
    # 5 frames, 25 ms apart (i.e. 40 fps), microsecond-precision ISO strings.
    times = [f"2026-05-28T22:02:43.{us:06d}Z" for us in (0, 25000, 50000, 75000, 100000)]
    secs = to_seconds_series(pd.DataFrame({"UTC_ISO": times}), "UTC_ISO")
    assert abs(float(secs.iloc[0])) < 1e-9
    # Real span is 0.1 s; the bug produced 0.0001 s.
    assert abs(float(secs.iloc[-1]) - 0.1) < 1e-6


def test_utc_iso_realistic_fps_about_40():
    # 400 frames @ 25 ms (crosses second boundaries) -> fps must be ~40, not ~40000.
    times = _iso_series("2026-05-28T22:02:43.000000Z", 400, 25_000, "%Y-%m-%dT%H:%M:%S.%fZ")
    secs = to_seconds_series(pd.DataFrame({"UTC_ISO": times}), "UTC_ISO")
    span = float(secs.iloc[-1] - secs.iloc[0])
    fps = (len(secs) - 1) / span
    assert abs(span - 9.975) < 1e-3      # 399 * 25 ms
    assert 39.0 < fps < 41.0


def test_timestamp_column_uses_same_path():
    # 'Timestamp' branch (utc=False), within one second to avoid rollover.
    times = [f"2026-05-28 22:02:43.{us:06d}" for us in (0, 250000, 500000, 750000)]
    secs = to_seconds_series(pd.DataFrame({"Timestamp": times}), "Timestamp")
    assert abs(float(secs.iloc[-1]) - 0.75) < 1e-6  # 0.75 s span, not 0.00075
