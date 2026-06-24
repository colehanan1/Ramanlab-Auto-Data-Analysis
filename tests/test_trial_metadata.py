"""Tests for ``fbpipe.utils.trial_metadata``.

Covers:
- Sidecar + ActiveOFM parsing on a synthetic RandomPanel-shaped fixture
- Sidecar parsing on a synthetic legacy Hex-Training-shaped fixture (longer trial)
- Missing-sidecar fallback to fps / odor-window defaults
- ``trial_type_override`` flipping a folder named ``..._training_N`` to testing
- Cache round-trip
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

from fbpipe.utils.trial_metadata import (
    CACHE_FILENAME,
    TrialMetadata,
    find_sensors_csv,
    find_sidecar,
    load_trial_metadata,
    parse_active_ofm,
    parse_light_on_seconds,
    parse_sidecar,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


@dataclass
class _Override:
    trial_type_override: str | None = None
    odor_on_s: float | None = None
    odor_off_s: float | None = None


def _write_sidecar(
    batch_dir: Path,
    *,
    batch: str,
    trial_type: str,
    trial_index: int,
    odor: str,
    timestamp: str,
    cycle_name: str | None = None,
    n_frames: int,
    fps: float,
) -> tuple[Path, Path]:
    """Create both sidecar siblings (.txt + .csv with ActiveOFM column).

    Returns ``(txt_path, csv_path)``.
    """
    base = f"output_{batch}_{trial_type}_{trial_index}_{odor}_{timestamp}"
    cycle = cycle_name or f"{trial_type}_{trial_index}_{odor}"
    txt = batch_dir / f"{base}.txt"
    txt.write_text(
        f"""Video Metadata
==============
Cycle Name: {cycle}
Video File: /tmp/{base}.h264
Frames CSV File: /tmp/{base}.csv
Sensors CSV File: /tmp/sensors_{base}.csv
Start Time (UTC): 2026-05-15T16:10:53Z
End Time   (UTC): 2026-05-15T16:11:33Z
Duration: {n_frames / fps:.4f} seconds
Total Frames Captured: {n_frames}
Achieved Frame Rate: {fps:.2f} FPS
Frame Resolution: 1152 x 1080
""",
        encoding="utf-8",
    )
    return txt, batch_dir / f"{base}.csv"


def _write_active_ofm_csv(
    csv_path: Path,
    *,
    n_frames: int,
    on_frame: int,
    off_frame: int,
    channel: str = "OFM_C",
) -> None:
    rows = []
    for i in range(1, n_frames + 1):
        state = channel if (on_frame <= i < off_frame) else "off"
        rows.append({
            "FrameNumber": i,
            "SensorNs": 1_000_000_000 + i * 25_000_000,
            "UTC_ISO": "2026-05-15T16:10:53.000000Z",
            "ActiveOFM": state,
            "FrameInterval_s": 0.025,
            "Deviation_s": 0.0,
        })
    pd.DataFrame(rows).to_csv(csv_path, index=False)


def _write_sensors_csv(
    csv_path: Path,
    *,
    start_ns: int = 1_000_000_000_000,
    odor_on_ns: int | None = 30_000_000_000,
    light_on_ns: int | None = 35_000_000_000,
    extra_events: list[tuple[int, str]] | None = None,
) -> None:
    """Write a sensors_output_*.csv with a RECORDING_START + light events.

    Offsets (``*_ns``) are nanoseconds relative to ``start_ns``. ``extra_events``
    is a list of ``(offset_ns, event)`` appended verbatim.
    """
    rows: list[dict] = [{"MonoNs": start_ns, "OFM_Event": "RECORDING_START"}]
    if odor_on_ns is not None:
        rows.append({"MonoNs": start_ns + odor_on_ns, "OFM_Event": "OFM_H_ON"})
    if light_on_ns is not None:
        rows.append({"MonoNs": start_ns + light_on_ns, "OFM_Event": "LIGHT_SOLID_ON"})
        # A later off + a second on, to prove we take the FIRST on.
        rows.append({"MonoNs": start_ns + light_on_ns + 5_000_000_000, "OFM_Event": "LIGHT_OFF"})
        rows.append({"MonoNs": start_ns + light_on_ns + 6_000_000_000, "OFM_Event": "LIGHT_PULSE_ON"})
    for off_ns, event in (extra_events or []):
        rows.append({"MonoNs": start_ns + off_ns, "OFM_Event": event})
    df = pd.DataFrame(rows)
    df["Humidity"] = "N/A"
    df["Pressure"] = "N/A"
    df["Temperature"] = "N/A"
    df["Gas"] = "N/A"
    df = df[["MonoNs", "Humidity", "Pressure", "Temperature", "Gas", "OFM_Event"]]
    df.to_csv(csv_path, index=False)


def _build_trial(
    base: Path,
    *,
    dataset: str,
    batch: str,
    trial_type: str,
    trial_index: int,
    odor: str,
    n_frames: int = 1598,
    fps: float = 40.03,
    on_frame: int = 597,
    off_frame: int = 997,
    timestamp: str = "20260515_111038",
    cycle_name: str | None = None,
    write_sidecar: bool = True,
    write_csv: bool = True,
    write_sensors: bool = False,
    light_on_ns: int | None = 35_000_000_000,
) -> Path:
    """Build a RandomPanel-shaped trial folder + sidecar pair on disk."""
    batch_dir = base / dataset / batch
    batch_dir.mkdir(parents=True, exist_ok=True)
    trial_dir = batch_dir / f"{batch}_{trial_type}_{trial_index}"
    trial_dir.mkdir(exist_ok=True)
    # The trial CSV the pipeline actually reads (distances). Keep it small.
    (trial_dir / f"{trial_dir.name}_fly1_distances.csv").write_text(
        "frame,eye_x,eye_y,prob_x,prob_y\n0,0,0,0,0\n", encoding="utf-8"
    )
    if write_sidecar:
        txt_path, csv_path = _write_sidecar(
            batch_dir,
            batch=batch,
            trial_type=trial_type,
            trial_index=trial_index,
            odor=odor,
            timestamp=timestamp,
            cycle_name=cycle_name,
            n_frames=n_frames,
            fps=fps,
        )
        if write_csv:
            _write_active_ofm_csv(
                csv_path,
                n_frames=n_frames,
                on_frame=on_frame,
                off_frame=off_frame,
            )
        if write_sensors:
            sensors_path = csv_path.with_name(f"sensors_{csv_path.name}")
            _write_sensors_csv(sensors_path, light_on_ns=light_on_ns)
    return trial_dir


# ---------------------------------------------------------------------------
# Parsing primitives
# ---------------------------------------------------------------------------


def test_parse_sidecar_extracts_normalised_keys(tmp_path: Path) -> None:
    trial = _build_trial(
        tmp_path,
        dataset="RandomPanel-Training-24-0.01",
        batch="may_15_batch_1",
        trial_type="training",
        trial_index=14,
        odor="Citral",
        n_frames=1598,
        fps=40.03,
    )
    txt, _ = find_sidecar(trial)
    assert txt is not None

    fields = parse_sidecar(txt)
    assert fields["cycle_name"] == "training_14_Citral"
    assert fields["total_frames_captured"] == "1598"
    assert fields["achieved_frame_rate"].startswith("40.03")


def test_parse_active_ofm_returns_first_on_and_off(tmp_path: Path) -> None:
    trial = _build_trial(
        tmp_path,
        dataset="RandomPanel-Training-24-0.01",
        batch="may_15_batch_1",
        trial_type="training",
        trial_index=14,
        odor="Citral",
        n_frames=1598,
        fps=40.03,
        on_frame=597,
        off_frame=997,
    )
    _, csv = find_sidecar(trial)
    assert csv is not None
    on_f, off_f, channel = parse_active_ofm(csv)
    assert on_f == 597
    assert off_f == 997
    assert channel == "OFM_C"


def test_parse_active_ofm_handles_never_on(tmp_path: Path) -> None:
    # Build a trial whose ActiveOFM is "off" for every frame (odor never fires)
    batch_dir = tmp_path / "ds" / "batch"
    batch_dir.mkdir(parents=True)
    csv_path = batch_dir / "x.csv"
    pd.DataFrame({
        "FrameNumber": [1, 2, 3],
        "ActiveOFM": ["off", "off", "off"],
    }).to_csv(csv_path, index=False)
    on_f, off_f, channel = parse_active_ofm(csv_path)
    assert on_f is None and off_f is None and channel is None


# ---------------------------------------------------------------------------
# load_trial_metadata: end-to-end shapes
# ---------------------------------------------------------------------------


def test_load_trial_metadata_random_panel_sidecar(tmp_path: Path) -> None:
    """Sidecar present → all values come from the rig (RandomPanel-shaped)."""
    trial = _build_trial(
        tmp_path,
        dataset="RandomPanel-Training-24-0.01",
        batch="may_15_batch_1",
        trial_type="training",
        trial_index=14,
        odor="Citral",
        n_frames=1598,
        fps=40.03,
        on_frame=597,
        off_frame=997,
    )

    meta = load_trial_metadata(
        trial,
        known_datasets=["RandomPanel-Training-24-0.01"],
        dataset_override=_Override(trial_type_override="testing"),
        use_cache=False,
    )

    assert meta.source == "sidecar"
    assert meta.dataset == "RandomPanel-Training-24-0.01"
    assert meta.n_frames == 1598
    assert meta.fps == pytest.approx(40.03, rel=1e-3)
    assert meta.cycle_name == "training_14_Citral"
    assert meta.odor == "Citral"
    assert meta.trial_index == 14
    assert meta.raw_trial_type == "training"
    # Override must flip the final label even though folder + cycle say "training"
    assert meta.trial_type == "testing"
    assert meta.odor_on_frame == 597
    assert meta.odor_off_frame == 997
    # Derived properties
    assert meta.odor_on_s == pytest.approx(597 / 40.03, rel=1e-3)
    assert meta.duration_s == pytest.approx(1598 / 40.03, rel=1e-3)


def test_load_trial_metadata_legacy_hex_training_long_trial(tmp_path: Path) -> None:
    """Sidecar present, 5998-frame legacy trial → no truncation to 3600."""
    trial = _build_trial(
        tmp_path,
        dataset="Hex-Training-24-0.01",
        batch="may_11_batch_1",
        trial_type="testing",
        trial_index=1,
        odor="Hexanol",
        n_frames=5998,
        fps=40.02,
        on_frame=1196,
        off_frame=2396,
    )

    meta = load_trial_metadata(trial, use_cache=False)
    assert meta.n_frames == 5998  # not truncated
    assert meta.odor == "Hexanol"
    assert meta.raw_trial_type == "testing"
    assert meta.trial_type == "testing"
    assert meta.odor_on_frame == 1196
    assert meta.odor_off_frame == 2396


def test_load_trial_metadata_fallback_no_sidecar(tmp_path: Path) -> None:
    """No sidecar → fall back to caller-supplied n_frames + config defaults."""
    trial = _build_trial(
        tmp_path,
        dataset="OldDataset",
        batch="batch_a",
        trial_type="testing",
        trial_index=3,
        odor="Hexanol",
        write_sidecar=False,  # leaves no sidecar siblings
    )

    meta = load_trial_metadata(
        trial,
        fps_default=40.0,
        odor_on_s_default=30.0,
        odor_off_s_default=60.0,
        n_frames_fallback=3600,
        use_cache=False,
    )

    assert meta.source == "fallback"
    assert meta.n_frames == 3600
    assert meta.fps == 40.0
    # Window comes from config defaults
    assert meta.odor_on_frame == 1200
    assert meta.odor_off_frame == 2400
    # Trial type still inferred from folder name
    assert meta.trial_type == "testing"
    # Odor token absent (no sidecar / no filename), explicit None
    assert meta.odor is None


def test_dataset_override_flips_training_to_testing(tmp_path: Path) -> None:
    trial = _build_trial(
        tmp_path,
        dataset="RandomPanel-Training-24-0.01",
        batch="may_15_batch_2",
        trial_type="training",
        trial_index=5,
        odor="Benzaldehyde",
    )
    forced = load_trial_metadata(
        trial,
        dataset_override=_Override(trial_type_override="testing"),
        use_cache=False,
    )
    untouched = load_trial_metadata(
        trial,
        dataset_override=None,
        use_cache=False,
    )
    assert forced.trial_type == "testing"
    assert untouched.trial_type == "training"


def test_cache_round_trip(tmp_path: Path) -> None:
    trial = _build_trial(
        tmp_path,
        dataset="RandomPanel-Training-24-0.01",
        batch="may_15_batch_1",
        trial_type="training",
        trial_index=7,
        odor="ACV",
        n_frames=1500,
        fps=40.0,
        on_frame=600,
        off_frame=1000,
    )

    first = load_trial_metadata(trial, use_cache=True)
    assert first.source == "sidecar"
    cache_file = trial / CACHE_FILENAME
    assert cache_file.exists()
    payload = json.loads(cache_file.read_text())
    assert payload["n_frames"] == 1500
    assert payload["odor_on_frame"] == 600

    # Second load should hit cache and yield identical primitive fields
    second = load_trial_metadata(trial, use_cache=True)
    assert second.source == "cache"
    assert second.n_frames == first.n_frames
    assert second.odor_on_frame == first.odor_on_frame
    assert second.fps == first.fps


def test_cache_invalidated_when_sidecar_changes(tmp_path: Path) -> None:
    trial = _build_trial(
        tmp_path,
        dataset="RandomPanel-Training-24-0.01",
        batch="may_15_batch_1",
        trial_type="training",
        trial_index=2,
        odor="Linalool",
        n_frames=1600,
        fps=40.0,
        on_frame=500,
        off_frame=900,
    )
    load_trial_metadata(trial, use_cache=True)

    # Rewrite the sidecar with different values; cache must be invalidated.
    batch = trial.parent
    sidecar = next(batch.glob("output_*_Linalool_*.txt"))
    sidecar.write_text(
        sidecar.read_text()
        .replace("Total Frames Captured: 1600", "Total Frames Captured: 1700")
        .replace("Achieved Frame Rate: 40.00 FPS", "Achieved Frame Rate: 42.00 FPS"),
        encoding="utf-8",
    )
    # Bump mtime to be safe
    import os
    import time

    new_time = time.time() + 1
    os.utime(sidecar, (new_time, new_time))

    refreshed = load_trial_metadata(trial, use_cache=True)
    assert refreshed.source == "sidecar"
    assert refreshed.n_frames == 1700
    assert refreshed.fps == pytest.approx(42.0)


# ---------------------------------------------------------------------------
# Light-on parsing (sensors_output_*.csv)
# ---------------------------------------------------------------------------


def test_parse_light_on_seconds_first_on_relative_to_recording_start(tmp_path: Path) -> None:
    csv_path = tmp_path / "sensors_output_x_training_2_Hexanol_20260603_110512.csv"
    _write_sensors_csv(csv_path, light_on_ns=35_000_000_000)
    light_s = parse_light_on_seconds(csv_path)
    assert light_s == pytest.approx(35.0, abs=1e-6)


def test_parse_light_on_seconds_takes_first_of_multiple(tmp_path: Path) -> None:
    # First on at 20s; later pulse-on events must be ignored.
    csv_path = tmp_path / "sensors.csv"
    _write_sensors_csv(
        csv_path,
        light_on_ns=20_000_000_000,
        extra_events=[(40_000_000_000, "LIGHT_PULSE_ON")],
    )
    assert parse_light_on_seconds(csv_path) == pytest.approx(20.0, abs=1e-6)


def test_parse_light_on_seconds_pulse_manual_start(tmp_path: Path) -> None:
    csv_path = tmp_path / "sensors.csv"
    _write_sensors_csv(
        csv_path,
        light_on_ns=None,  # no SOLID_ON
        extra_events=[(33_000_000_000, "LIGHT_PULSE_MANUAL_START")],
    )
    assert parse_light_on_seconds(csv_path) == pytest.approx(33.0, abs=1e-6)


def test_parse_light_on_seconds_returns_none_when_no_light(tmp_path: Path) -> None:
    csv_path = tmp_path / "sensors.csv"
    _write_sensors_csv(csv_path, light_on_ns=None)  # only RECORDING_START + odor
    assert parse_light_on_seconds(csv_path) is None


def test_parse_light_on_ignores_off_events(tmp_path: Path) -> None:
    csv_path = tmp_path / "sensors.csv"
    _write_sensors_csv(
        csv_path,
        light_on_ns=None,
        extra_events=[(10_000_000_000, "LIGHT_OFF"), (15_000_000_000, "LIGHT_PULSE_OFF")],
    )
    assert parse_light_on_seconds(csv_path) is None


def test_find_sensors_csv_derives_from_sidecar(tmp_path: Path) -> None:
    sidecar = tmp_path / "output_may_15_batch_1_training_2_Hexanol_20260603_110512.csv"
    sidecar.write_text("FrameNumber,ActiveOFM\n1,off\n", encoding="utf-8")
    sensors = sidecar.with_name(f"sensors_{sidecar.name}")
    _write_sensors_csv(sensors)
    found = find_sensors_csv(sidecar_csv=sidecar)
    assert found == sensors


def test_load_trial_metadata_populates_light_on_s(tmp_path: Path) -> None:
    trial = _build_trial(
        tmp_path,
        dataset="Hex-Training-48-0.1",
        batch="june_03_batch_1",
        trial_type="training",
        trial_index=2,
        odor="Hexanol",
        n_frames=3606,
        fps=40.0,
        on_frame=1200,
        off_frame=2400,
        write_sensors=True,
        light_on_ns=35_000_000_000,
    )
    meta = load_trial_metadata(trial, use_cache=False)
    assert meta.light_on_s == pytest.approx(35.0, abs=1e-6)


def test_light_on_s_survives_cache_round_trip(tmp_path: Path) -> None:
    trial = _build_trial(
        tmp_path,
        dataset="Hex-Training-48-0.1",
        batch="june_03_batch_1",
        trial_type="training",
        trial_index=3,
        odor="Hexanol",
        write_sensors=True,
        light_on_ns=40_000_000_000,
    )
    first = load_trial_metadata(trial, use_cache=True)
    assert first.light_on_s == pytest.approx(40.0, abs=1e-6)
    second = load_trial_metadata(trial, use_cache=True)
    assert second.source == "cache"
    assert second.light_on_s == pytest.approx(40.0, abs=1e-6)
