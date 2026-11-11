"""The light window light-min reads: on the video's clock, and spanning a whole pulse train.

Two bugs (2026-11-10 review):

1. Clock. A sensors log without a ``RECORDING_START`` row (older rig code) was
   timed from its own first row -- the odor or light event 5-30 s into the
   video -- so the cached light window sat ~30 s early and light-min left
   pre-odor baseline frames out of the fly's 0 % instead of the lit ones.
   Data/flys/EB-Control/october_28_batch_1 training_2 cached light_on_s 5.000
   for a light that comes on 35.0 s into the video. The trial's frame log
   (``output_<same>.csv``, ``SensorNs`` per frame on the same clock) gives
   the video's t = 0.
2. Span. The cache held the light's FIRST off, so a pulse train
   (Data/flys/Hex-Training-24-002 april_07_batch_2 training_3: 100 x 0.1 s
   pulses, 35.05-60.26 s) was masked for its first pulse only (64 frames
   instead of ~1,068). ``light_last_s`` is the end of the whole span.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fbpipe.config import Settings
from fbpipe.steps import distance_stats
from fbpipe.utils import trial_metadata as tm
from fbpipe.utils.light_frames import light_frame_mask, trial_meta_for_table
from fbpipe.utils.trial_metadata import (
    CACHE_FILENAME,
    load_trial_metadata,
    parse_light_on_seconds,
    parse_light_span_seconds,
    parse_light_window_seconds,
)

FPS = 40.0
FRAME_NS = 25_000_000            # 40 fps
T0 = 8_415_693_580_000           # first frame's SensorNs (EB-Control training_2)
N_FRAMES = 3600                  # 90 s
STEM = "october_28_batch_1_training_2_20251028_130827"   # old naming: no odor token


def _s(seconds: float) -> int:
    """Video time (s after the first frame) -> MonoNs/SensorNs."""
    return T0 + int(round(seconds * 1e9))


def _frame_log(path: Path, n_frames: int = N_FRAMES, t0: int = T0) -> Path:
    frames = np.arange(1, n_frames + 1)
    t = (frames - 1) / FPS
    pd.DataFrame({
        "FrameNumber": frames,
        "SensorNs": t0 + (frames - 1) * FRAME_NS,
        "UTC_ISO": "2025-10-28T18:08:27.764328Z",
        "ActiveOFM": np.where((t >= 30.0) & (t < 60.0), "OFM_E", "off"),
        "FrameInterval_s": 0.025,
        "Deviation_s": 0.0,
    }).to_csv(path, index=False)
    return path


def _sensors_log(path: Path, events: list[tuple[float, str]], *, marker_s: float | None = None) -> Path:
    """``events``: (video seconds, OFM_Event). ``marker_s``: write a RECORDING_START row there."""
    rows = [] if marker_s is None else [(marker_s, "RECORDING_START")]
    rows += sorted(events)
    df = pd.DataFrame({"MonoNs": [_s(t) for t, _ in rows], "OFM_Event": [e for _, e in rows]})
    for col in ("Humidity", "Pressure", "Temperature", "Gas"):
        df[col] = "N/A"
    df[["MonoNs", "Humidity", "Pressure", "Temperature", "Gas", "OFM_Event"]].to_csv(path, index=False)
    return path


def _solid(on: float, off: float) -> list[tuple[float, str]]:
    return [(30.0, "OFM_E_ON"), (on, "LIGHT_SOLID_ON"), (off, "LIGHT_OFF"), (60.0, "OFM_E_OFF")]


def _pulse_train() -> list[tuple[float, str]]:
    """100 pulses of 0.1 s in 5 bursts of 20, 35.05 s .. final LIGHT_OFF at 60.26 s
    (the shape of Hex-Training-24-002 april_07_batch_2 training_3)."""
    ev = [(30.0, "OFM_H_ON")]
    for burst, start in enumerate((35.05, 40.10, 45.15, 50.20, 55.25)):
        for k in range(20):
            t = start + 0.2 * k
            ev += [(round(t, 4), "LIGHT_PULSE_ON"), (round(t + 0.1, 4), "LIGHT_PULSE_OFF")]
        ev.append((60.26 if burst == 4 else round(start + 5.0, 4), "LIGHT_OFF"))
    ev.append((60.31, "OFM_H_OFF"))
    return ev


def _pair(tmp_path: Path, events, *, marker_s=None, frame_log=True) -> Path:
    """A sensors log and (optionally) its frame-log namesake in one folder."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    if frame_log:
        _frame_log(tmp_path / f"output_{STEM}.csv")
    return _sensors_log(tmp_path / f"sensors_output_{STEM}.csv", events, marker_s=marker_s)


# --------------------------------------------------------------------------- (a)-(c) clock


def test_no_marker_log_is_timed_from_the_frame_log(tmp_path):
    """(a) The sensors log starts at the odor (30 s into the video): the light is 35.05 s, not 5.05 s."""
    s = _pair(tmp_path, _pulse_train())
    assert parse_light_on_seconds(s) == pytest.approx(35.05, abs=1e-6)
    assert parse_light_window_seconds(s) == pytest.approx((35.05, 35.15), abs=1e-6)
    assert parse_light_span_seconds(s) == pytest.approx((35.05, 60.26), abs=1e-6)


def test_marker_log_is_unchanged(tmp_path):
    """(b) With RECORDING_START the marker stays t = 0, wherever the first frame is."""
    s = _pair(tmp_path, _solid(35.0, 55.0), marker_s=-0.17)   # the marker leads frame 1 by 0.17 s on one rig
    assert parse_light_window_seconds(s) == pytest.approx((35.17, 55.17), abs=1e-6)
    assert parse_light_span_seconds(s) == pytest.approx((35.17, 55.17), abs=1e-6)


def _no_frame_log(d: Path) -> None:
    pass


def _frame_log_without_sensorns(d: Path) -> None:
    pd.DataFrame({"FrameNumber": [1, 2], "ActiveOFM": ["off", "off"]}).to_csv(d / f"output_{STEM}.csv", index=False)


def _empty_frame_log(d: Path) -> None:
    (d / f"output_{STEM}.csv").write_bytes(b"")              # the rig's 0-byte frame CSVs


def _blank_sensorns(d: Path) -> None:
    pd.DataFrame({"FrameNumber": [1, 2, 3], "SensorNs": ["", "", ""]}).to_csv(d / f"output_{STEM}.csv", index=False)


def _unreadable_frame_log(d: Path) -> None:
    (d / f"output_{STEM}.csv").write_bytes(b"FrameNumber,SensorNs\n1,\xff\xfe\x00\n\"2,")


def _frame_log_on_another_clock(d: Path) -> None:
    _frame_log(d / f"output_{STEM}.csv", t0=1_000)           # stamps nowhere near the sensors log's MonoNs


@pytest.mark.parametrize("make_frame_log", [
    _no_frame_log, _frame_log_without_sensorns, _empty_frame_log, _blank_sensorns,
    _unreadable_frame_log, _frame_log_on_another_clock,
])
def test_without_a_usable_frame_log_the_earliest_row_stays_t0(tmp_path, make_frame_log):
    """(c) Nothing better to time it by: the earliest MonoNs (the odor, 30 s) is t = 0, as before."""
    s = _pair(tmp_path, _pulse_train(), frame_log=False)
    make_frame_log(tmp_path)
    assert parse_light_on_seconds(s) == pytest.approx(5.05, abs=1e-6)
    assert parse_light_window_seconds(s) == pytest.approx((5.05, 5.15), abs=1e-6)


def test_a_frame_log_ending_before_the_sensors_log_is_not_trusted(tmp_path):
    """A frame log that ends before the sensors log's first event (other clock, cut short) is ignored."""
    s = _pair(tmp_path, _pulse_train(), frame_log=False)
    _frame_log(tmp_path / f"output_{STEM}.csv", n_frames=400, t0=T0 - int(200e9))   # 10 s, ending 190 s before
    assert parse_light_on_seconds(s) == pytest.approx(5.05, abs=1e-6)


# --------------------------------------------------------------------------- (d)-(e) mask


def test_pulse_train_mask_covers_the_whole_span():
    """(d) 35.05 s - 0.5 s = frame 1382 .. 60.26 s + 1 s = frame 2450.4 -> frames 1382-2450."""
    meta = {"fps": FPS, "light_on_s": 35.05, "light_off_s": 35.15, "light_last_s": 60.26}
    m = light_frame_mask(np.arange(N_FRAMES), meta, before_s=0.5, after_s=1.0)
    assert np.flatnonzero(m).tolist() == list(range(1382, 2451))     # 1069 frames
    # what the first-off window masked: 1382 .. 36.15 s * 40 = 1446 -> 65 frames
    first_pulse = light_frame_mask(np.arange(N_FRAMES), {k: v for k, v in meta.items() if k != "light_last_s"},
                                   before_s=0.5, after_s=1.0)
    assert np.flatnonzero(first_pulse).tolist() == list(range(1382, 1447))


def test_solid_light_mask_is_unchanged():
    """(e) One on, one off: span end == first off, mask as before (1180-2240)."""
    for meta in ({"fps": FPS, "light_on_s": 30.0, "light_off_s": 55.0},
                 {"fps": FPS, "light_on_s": 30.0, "light_off_s": 55.0, "light_last_s": 55.0}):
        m = light_frame_mask(np.arange(N_FRAMES), meta, before_s=0.5, after_s=1.0)
        assert np.flatnonzero(m).tolist() == list(range(1180, 2241))


def test_light_never_logged_off_runs_to_the_end():
    meta = {"fps": FPS, "light_on_s": 30.0, "light_off_s": None, "light_last_s": None}
    m = light_frame_mask(np.arange(N_FRAMES), meta, before_s=0.5, after_s=1.0)
    assert np.flatnonzero(m).tolist() == list(range(1180, N_FRAMES))


# --------------------------------------------------------------------------- (f) metadata + cache


def _trial(tmp_path: Path, events, *, marker_s=None, dataset="EB-Control") -> Path:
    """An old-style trial: frame + sensors logs in the batch folder, an empty trial folder."""
    batch = tmp_path / dataset / "october_28_batch_1"
    trial = batch / "october_28_batch_1_training_2"
    trial.mkdir(parents=True)
    _frame_log(batch / f"output_{STEM}.csv")
    _sensors_log(batch / f"sensors_output_{STEM}.csv", events, marker_s=marker_s)
    return trial


def test_metadata_carries_the_clock_fixed_window_and_span(tmp_path):
    trial = _trial(tmp_path, _pulse_train())
    meta = load_trial_metadata(trial, use_cache=False)
    assert meta.light_on_s == pytest.approx(35.05, abs=1e-6)
    assert meta.light_off_s == pytest.approx(35.15, abs=1e-6)     # still the FIRST off
    assert meta.light_last_s == pytest.approx(60.26, abs=1e-6)
    assert meta.odor_on_frame == 1201                               # odor untouched (ActiveOFM FrameNumber)


def test_solid_light_span_equals_its_off(tmp_path):
    meta = load_trial_metadata(_trial(tmp_path, _solid(35.0, 55.0)), use_cache=False)
    assert (meta.light_on_s, meta.light_off_s, meta.light_last_s) == pytest.approx((35.0, 55.0, 55.0), abs=1e-6)


def test_light_last_s_round_trips_through_the_cache(tmp_path):
    trial = _trial(tmp_path, _pulse_train())
    first = load_trial_metadata(trial, use_cache=True)
    payload = json.loads((trial / CACHE_FILENAME).read_text())
    assert payload["light_last_s"] == pytest.approx(60.26, abs=1e-6)
    assert payload["_schema_version"] == tm._CACHE_SCHEMA_VERSION
    second = load_trial_metadata(trial, use_cache=True)
    assert second.source == "cache"
    assert (second.light_on_s, second.light_off_s, second.light_last_s) == pytest.approx(
        (first.light_on_s, first.light_off_s, first.light_last_s))


def _schema3_cache(trial: Path) -> dict:
    """What schema 3 cached for the EB-Control trial: the window ~30 s early, first pulse only."""
    payload = load_trial_metadata(trial, use_cache=False).to_dict()
    for k in ("light_on_s", "light_off_s", "light_last_s"):
        payload.pop(k, None)
    payload.update({"light_on_s": 5.05, "light_off_s": 5.15, "_sidecar_mtime": None, "_schema_version": 3})
    (trial / CACHE_FILENAME).write_text(json.dumps(payload, indent=2))
    return payload


def test_schema3_cache_is_re_derived(tmp_path):
    trial = _trial(tmp_path, _pulse_train())
    _schema3_cache(trial)
    assert tm._CACHE_SCHEMA_VERSION > 3
    meta = load_trial_metadata(trial, use_cache=True)
    assert meta.source != "cache"
    assert (meta.light_on_s, meta.light_last_s) == pytest.approx((35.05, 60.26), abs=1e-6)
    assert json.loads((trial / CACHE_FILENAME).read_text())["_schema_version"] == tm._CACHE_SCHEMA_VERSION


def test_a_cache_without_light_last_s_still_loads(tmp_path):
    """Back-compat: a current-schema payload missing the new key reads it as None."""
    trial = _trial(tmp_path, _solid(35.0, 55.0))
    payload = load_trial_metadata(trial, use_cache=True).to_dict()
    payload.pop("light_last_s")
    payload.update({"_sidecar_mtime": None, "_schema_version": tm._CACHE_SCHEMA_VERSION})
    (trial / CACHE_FILENAME).write_text(json.dumps(payload))
    meta = load_trial_metadata(trial, use_cache=True)
    assert meta.source == "cache" and meta.light_last_s is None and meta.light_off_s == pytest.approx(55.0)


def test_mask_reads_a_dict_without_light_last_s():
    """Back-compat: an old/hand-made meta dict falls back to light_off_s."""
    m = light_frame_mask(np.arange(100), {"fps": 10.0, "light_on_s": 2.0, "light_off_s": 3.0}, 0.0, 0.0)
    assert np.flatnonzero(m).tolist() == list(range(20, 31))


def test_table_reader_re_derives_a_stale_cache_without_writing(tmp_path):
    """trial_meta_for_table reads the JSON itself, so it must not trust a schema-3 window either."""
    trial = _trial(tmp_path, _pulse_train())
    stale = _schema3_cache(trial)
    before = (trial / CACHE_FILENAME).read_bytes()
    table = trial / f"{trial.name}_fly1_distances.parquet"
    meta = trial_meta_for_table(table)
    assert (meta["light_on_s"], meta["light_off_s"], meta["light_last_s"]) == pytest.approx(
        (35.05, 35.15, 60.26), abs=1e-6)
    assert meta["fps"] == stale["fps"] and meta["odor_on_frame"] == stale["odor_on_frame"]
    assert (trial / CACHE_FILENAME).read_bytes() == before                # read-only


def test_table_reader_keeps_a_hand_made_dict(tmp_path):
    """A JSON without _schema_version (test fixtures) is taken as it is."""
    trial = _trial(tmp_path, _pulse_train())
    (trial / CACHE_FILENAME).write_text(json.dumps({"fps": 40.0, "light_on_s": 4.5, "light_off_s": 6.5}))
    meta = trial_meta_for_table(trial / f"{trial.name}_fly1_distances.parquet")
    assert meta == {"fps": 40.0, "light_on_s": 4.5, "light_off_s": 6.5}


# --------------------------------------------------------------------------- (g) distance_stats end to end


def _distances(trial: Path) -> Path:
    d = np.full(N_FRAMES, 60.0)
    d[200] = 40.0      # 5.0 s: a real low point of the pre-odor baseline (masked by the ~30 s early window)
    d[1500] = 150.0    # 37.5 s: the light-evoked PER (sets 100 % under min-only)
    d[2000] = 15.0     # 50.0 s: junk under the 4th burst of pulses (was outside the first-pulse window)
    df = pd.DataFrame({
        "frame": np.arange(N_FRAMES),
        "x_class0": 500.0, "y_class0": 300.0,
        "x_class1": 500.0 + d, "y_class1": 300.0,
        "distance_0_1": d,
    })
    p = trial / f"{trial.name}_fly1_distances.parquet"
    df.to_parquet(p, index=False)
    return p


def _cfg(tmp_path: Path) -> Settings:
    cfg = Settings(model_path="unused", main_directories=[str(tmp_path)])
    cfg.class2_min, cfg.class2_max = 10.0, 160.0
    cfg.minmax_exclude_light = True
    cfg.minmax_light_scope = "min"
    cfg.parallel.enabled = False
    return cfg


def _stats(batch: Path) -> dict:
    return json.loads((batch / "fly1_global_distance_stats_class_0.json").read_text())


def test_distance_stats_leaves_the_whole_correctly_timed_span_out_of_the_min(tmp_path):
    """(g) As the pipeline runs it: check_light_stimulus writes the cache, distance_stats reads it."""
    trial = _trial(tmp_path, _pulse_train())
    _distances(trial)
    load_trial_metadata(trial)                                   # writes _trial_meta.json
    distance_stats._process_fly_dir(trial.parent, _cfg(tmp_path))
    s = _stats(trial.parent)
    assert s["global_min"] == 40.0       # the 50 s junk is left out, the 5 s baseline point counts
    assert s["global_max"] == 150.0      # min only: the light PER still sets 100 %


def test_distance_stats_does_not_trust_a_stale_cache(tmp_path):
    """(g) The same with the schema-3 cache still on disk (no step re-derived it first)."""
    trial = _trial(tmp_path, _pulse_train())
    _distances(trial)
    _schema3_cache(trial)
    distance_stats._process_fly_dir(trial.parent, _cfg(tmp_path))
    assert _stats(trial.parent)["global_min"] == 40.0
    assert json.loads((trial / CACHE_FILENAME).read_text())["_schema_version"] == 3   # not rewritten


# --------------------------------------------------------------------------- per-frame eye hold


def test_live_eye_hold_covers_the_whole_pulse_train():
    from fbpipe.utils.live_eye import stimulus_hold
    meta = {"fps": FPS, "light_on_s": 35.05, "light_off_s": 35.15, "light_last_s": 60.26}
    assert stimulus_hold(meta, odor=False) == (1382, 2450)
    legacy = {k: v for k, v in meta.items() if k != "light_last_s"}
    assert stimulus_hold(legacy, odor=False) == (1382, 1446)
