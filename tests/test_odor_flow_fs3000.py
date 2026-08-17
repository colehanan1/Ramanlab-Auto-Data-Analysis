"""Tests for PiCode/odor_flow_fs3000.py and PiCode/compare_flow_rigs.py.

Like odor_bme680_exp.py, the flow script must import cleanly off-Pi (no
smbus2 / lgpio at module level). These tests cover the pure logic: FS3000
packet parsing and checksum, the raw->m/s lookup tables for both sensor
variants, per-channel steady-state summaries, and the cross-rig comparison.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

PI_CODE = Path(__file__).resolve().parents[1] / "PiCode"
MODULE_PATH = PI_CODE / "odor_flow_fs3000.py"
COMPARE_PATH = PI_CODE / "compare_flow_rigs.py"
SMOKE_PATH = PI_CODE / "fs3000_smoke_test.py"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return _load(MODULE_PATH, "odor_flow_fs3000")


@pytest.fixture(scope="module")
def cmp_mod():
    return _load(COMPARE_PATH, "compare_flow_rigs")


# --------------------------------------------------------------------------
# Packet parsing / checksum
# --------------------------------------------------------------------------

def _packet(raw, junk=(0x00, 0x00)):
    """Build a valid 5-byte FS3000 packet for a 12-bit raw count."""
    body = [(raw >> 8) & 0x0F, raw & 0xFF, junk[0], junk[1]]
    checksum = (0x100 - sum(body)) & 0xFF
    return bytes([checksum] + body)


def test_parse_packet_extracts_raw_and_validates_checksum(mod):
    raw, ok = mod.parse_packet(_packet(409))
    assert raw == 409
    assert ok is True


def test_parse_packet_flags_bad_checksum(mod):
    buf = bytearray(_packet(2066))
    buf[0] = (buf[0] + 1) & 0xFF
    raw, ok = mod.parse_packet(bytes(buf))
    assert raw == 2066
    assert ok is False


def test_parse_packet_masks_high_nibble_of_data_byte(mod):
    # Bits 12-15 of the data-high byte are undefined; only 12 bits are data.
    buf = bytearray(_packet(0x199))
    buf[1] |= 0xF0
    raw, _ = mod.parse_packet(bytes(buf))
    assert raw == 0x199


def test_parse_packet_rejects_wrong_length(mod):
    with pytest.raises(ValueError):
        mod.parse_packet(b"\x00\x01\x02")


# --------------------------------------------------------------------------
# raw -> m/s conversion (datasheet lookup tables)
# --------------------------------------------------------------------------

def test_1005_table_endpoints(mod):
    assert mod.raw_to_mps(409, "1005") == 0.0
    assert mod.raw_to_mps(3686, "1005") == pytest.approx(7.23)


def test_1005_exact_table_point(mod):
    assert mod.raw_to_mps(2066, "1005") == pytest.approx(3.00)


def test_1005_interpolates_between_points(mod):
    # halfway between (409, 0.0) and (915, 1.07)
    assert mod.raw_to_mps(662, "1005") == pytest.approx(0.535)


def test_1015_table(mod):
    assert mod.raw_to_mps(1203, "1015") == pytest.approx(2.00)
    assert mod.raw_to_mps(806, "1015") == pytest.approx(1.00)
    assert mod.raw_to_mps(3686, "1015") == pytest.approx(15.00)


def test_conversion_clamps_out_of_range(mod):
    assert mod.raw_to_mps(0, "1005") == 0.0
    assert mod.raw_to_mps(4095, "1005") == pytest.approx(7.23)
    assert mod.raw_to_mps(4095, "1015") == pytest.approx(15.00)


def test_conversion_rejects_unknown_model(mod):
    with pytest.raises(ValueError):
        mod.raw_to_mps(1000, "9999")


# --------------------------------------------------------------------------
# Shared rig plumbing comes from odor_bme680_exp.py
# --------------------------------------------------------------------------

def test_pin_maps_reused_from_bme680_module(mod):
    base = _load(PI_CODE / "odor_bme680_exp.py", "odor_bme680_exp_for_flow_test")
    assert mod.RIG_PIN_MAPS == base.RIG_PIN_MAPS
    assert mod.MASTER_CHANNEL == 8


def test_sensor_address_is_fs3000_default(mod):
    assert mod.FS3000_ADDR == 0x28


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def test_parser_requires_rig_and_defaults(mod):
    parser = mod.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
    args = parser.parse_args(["--rig", "pi2"])
    assert args.model == "1005"
    assert args.channels == "1-7"
    assert args.repeats == 3
    assert args.on == pytest.approx(10.0)
    assert args.off == pytest.approx(15.0)
    assert args.hz == pytest.approx(10.0)


def test_parser_accepts_experiment_label(mod):
    parser = mod.build_parser()
    assert parser.parse_args(["--rig", "pi2"]).label is None
    args = parser.parse_args(["--rig", "pi2", "--label", "odor no filter"])
    assert args.label == "odor no filter"


def test_parser_accepts_1015_model_only(mod):
    parser = mod.build_parser()
    assert parser.parse_args(["--rig", "pi1", "--model", "1015"]).model == "1015"
    with pytest.raises(SystemExit):
        parser.parse_args(["--rig", "pi1", "--model", "1010"])


# --------------------------------------------------------------------------
# Per-channel steady-state summary
# --------------------------------------------------------------------------

def _rows(phase, channel, repeat, velocities, t0, mod, ok=True):
    out = []
    for i, v in enumerate(velocities):
        out.append(mod.make_flow_row(
            t=t0 + i * 0.1, wall="w", raw=0, mps=v, checksum_ok=ok,
            phase=phase, channel=channel,
            name=None if channel is None else f"OFM_{channel}",
            pin=None if channel is None else 10 + channel, repeat=repeat))
    return out, t0 + len(velocities) * 0.1


def test_summarize_uses_trailing_half_of_each_block(mod):
    rows, t = _rows("baseline", None, -1, [1.0, 1.0, 1.0, 1.0], 0.0, mod)
    r, t = _rows("odor_on", 1, 0, [1.0, 1.8, 2.0, 2.0], t, mod)
    rows += r
    r, t = _rows("odor_off", 1, 0, [1.2, 1.0, 1.0, 1.0], t, mod)
    rows += r
    r, t = _rows("odor_on", 1, 1, [1.0, 2.2, 2.4, 2.4], t, mod)
    rows += r
    r, t = _rows("odor_off", 1, 1, [1.0, 1.0, 1.0, 1.0], t, mod)
    rows += r

    summary = mod.summarize(rows)
    ch1 = summary[1]
    assert ch1["name"] == "OFM_1"
    assert ch1["on_mean"] == pytest.approx(2.2)      # mean of 2.0 and 2.4
    assert ch1["baseline_mean"] == pytest.approx(1.0)
    assert ch1["delta_mean"] == pytest.approx(1.2)
    assert ch1["n_repeats"] == 2
    reps = ch1["repeats"]
    assert reps[0]["on_mean"] == pytest.approx(2.0)
    assert reps[0]["delta"] == pytest.approx(1.0)    # baseline block before it
    assert reps[1]["on_mean"] == pytest.approx(2.4)
    assert reps[1]["delta"] == pytest.approx(1.4)    # preceding OFF block


def test_summarize_excludes_checksum_failures(mod):
    rows, t = _rows("baseline", None, -1, [1.0, 1.0], 0.0, mod)
    r, t = _rows("odor_on", 2, 0, [1.0, 1.0], t, mod)
    rows += r
    bad, t = _rows("odor_on", 2, 0, [99.0, 99.0], t, mod, ok=False)
    rows += bad

    summary = mod.summarize(rows)
    assert summary[2]["on_mean"] == pytest.approx(1.0)
    assert summary[2]["checksum_failures"] == 2


def test_summarize_flat_channel_has_zero_delta(mod):
    rows, t = _rows("baseline", None, -1, [1.0, 1.0, 1.0, 1.0], 0.0, mod)
    r, t = _rows("odor_on", 3, 0, [1.0, 1.0, 1.0, 1.0], t, mod)
    rows += r
    summary = mod.summarize(rows)
    assert summary[3]["delta_mean"] == pytest.approx(0.0)


# --------------------------------------------------------------------------
# Cross-rig comparison
# --------------------------------------------------------------------------

def _summary(rig, means):
    return {
        "rig": rig,
        "model": "1005",
        "channels": {
            str(ch): {"name": f"OFM_{ch}", "on_mean": m, "on_std": 0.05,
                      "baseline_mean": 0.1, "delta_mean": m - 0.1}
            for ch, m in means.items()
        },
    }


def test_compare_within_tolerance_not_flagged(cmp_mod):
    out = cmp_mod.compare_summaries(
        [_summary("pi1", {1: 2.0}), _summary("pi2", {1: 2.1})], tol_rel=0.10)
    assert len(out) == 1
    row = out[0]
    assert row["channel"] == 1
    assert row["per_rig"]["pi1"] == pytest.approx(2.0)
    assert row["per_rig"]["pi2"] == pytest.approx(2.1)
    assert row["flagged"] is False


def test_compare_flags_deviating_rig(cmp_mod):
    out = cmp_mod.compare_summaries(
        [_summary("pi1", {1: 2.0}), _summary("pi2", {1: 2.1}),
         _summary("pi3", {1: 1.0})], tol_rel=0.10)
    assert out[0]["flagged"] is True
    assert out[0]["spread_rel"] == pytest.approx((2.1 - 1.0) / ((2.0 + 2.1 + 1.0) / 3))


def test_compare_handles_channel_missing_on_one_rig(cmp_mod):
    out = cmp_mod.compare_summaries(
        [_summary("pi1", {1: 2.0, 2: 1.5}), _summary("pi2", {1: 2.0})],
        tol_rel=0.10)
    by_ch = {r["channel"]: r for r in out}
    assert set(by_ch) == {1, 2}
    assert by_ch[2]["per_rig"] == {"pi1": pytest.approx(1.5)}
    assert by_ch[2]["flagged"] is False   # nothing to compare against


# --------------------------------------------------------------------------
# ESPHome flow-wand sensor (ESP32 + FS3000 polled over web_server HTTP)
# --------------------------------------------------------------------------

ESP_PAYLOAD = '{"id":"sensor-air_velocity","value":1.23,"state":"1.23 m/s"}'


def test_parse_esp_state_extracts_velocity(mod):
    assert mod.parse_esp_state(ESP_PAYLOAD) == pytest.approx(1.23)


def test_parse_esp_state_rejects_missing_or_nonfinite_value(mod):
    with pytest.raises(ValueError):
        mod.parse_esp_state('{"id":"sensor-air_velocity","state":"unknown"}')
    with pytest.raises(ValueError):
        mod.parse_esp_state('{"id":"sensor-air_velocity","value":NaN,"state":""}')


def test_esp_sensor_reads_velocity_via_fetch(mod):
    sensor = mod.EspFlowSensor("http://flowmeter.local/", fetch=lambda: ESP_PAYLOAD)
    reading = sensor.read()
    assert reading.mps == pytest.approx(1.23)
    assert reading.checksum_ok is True


def test_esp_sensor_turns_fetch_errors_into_invalid_samples(mod):
    def boom():
        raise OSError("connection refused")
    sensor = mod.EspFlowSensor("http://flowmeter.local", fetch=boom)
    reading = sensor.read()
    assert reading.checksum_ok is False   # excluded from summaries, like a bad CRC


def test_esp_sensor_builds_url_from_base_and_sensor_id(mod):
    sensor = mod.EspFlowSensor("http://flowmeter.local/", sensor_id="air_velocity",
                               fetch=lambda: ESP_PAYLOAD)
    assert sensor.url == "http://flowmeter.local/sensor/air_velocity"


def test_parser_accepts_esp_url(mod):
    parser = mod.build_parser()
    assert parser.parse_args(["--rig", "pi1"]).esp_url is None
    args = parser.parse_args(["--rig", "pi1", "--esp-url", "http://flowmeter.local"])
    assert args.esp_url == "http://flowmeter.local"
    assert args.esp_sensor == "air_velocity"


# --------------------------------------------------------------------------
# Serial flow wand (ESP32 on USB, readings scraped from its ESPHome log)
# --------------------------------------------------------------------------

LOG_CHUNK = (
    "[V][i2c.idf:141]: Writing 0 bytes, reading 5 bytes\n"
    "[V][fs3000:058]: Got raw reading=983\n"
    "[V][sensor:125]: 'air_velocity' >> 1.23 m/s\n"
    "[V][fs3000:058]: Got raw reading=991\n"
)


def test_parse_raw_from_log_takes_the_most_recent_reading(mod):
    assert mod.parse_raw_from_log(LOG_CHUNK) == 991


def test_parse_raw_from_log_returns_none_without_a_reading(mod):
    assert mod.parse_raw_from_log("[W][fs3000:039]: Error reading data\n") is None
    assert mod.parse_raw_from_log("") is None


def test_serial_sensor_converts_raw_using_model_table(mod):
    sensor = mod.SerialFlowSensor(port="/dev/null", model="1005",
                                  drain=lambda: LOG_CHUNK)
    reading = sensor.read()
    assert reading.raw == 991
    assert reading.mps == pytest.approx(mod.raw_to_mps(991, "1005"))
    assert reading.checksum_ok is True


def test_serial_sensor_marks_sample_invalid_when_no_reading_arrives(mod):
    sensor = mod.SerialFlowSensor(port="/dev/null", model="1005", drain=lambda: "")
    reading = sensor.read()
    assert reading.checksum_ok is False
    assert reading.raw == -1


def test_serial_sensor_surfaces_sensor_read_errors_as_invalid(mod):
    chunk = "[W][fs3000:039]: Error reading data from FS3000\n"
    sensor = mod.SerialFlowSensor(port="/dev/null", model="1005",
                                  drain=lambda: chunk)
    assert sensor.read().checksum_ok is False


def test_serial_sensor_wait_ready_returns_true_once_data_flows(mod):
    calls = {"n": 0}

    def drain():
        calls["n"] += 1
        return LOG_CHUNK if calls["n"] >= 3 else ""

    sensor = mod.SerialFlowSensor(port="/dev/null", model="1005", drain=drain)
    assert sensor.wait_ready(timeout=5.0, sleep=lambda s: None) is True
    assert calls["n"] == 3


def test_serial_sensor_wait_ready_gives_up_and_reports_false(mod):
    clock = {"t": 0.0}

    def tick(_s):
        clock["t"] += 0.5

    sensor = mod.SerialFlowSensor(port="/dev/null", model="1005", drain=lambda: "")
    assert sensor.wait_ready(timeout=1.0, sleep=tick,
                             now=lambda: clock["t"]) is False


def test_parser_accepts_serial_port(mod):
    parser = mod.build_parser()
    assert parser.parse_args(["--rig", "pi2"]).serial_port is None
    args = parser.parse_args(["--rig", "pi2", "--serial-port", "/dev/ttyUSB0"])
    assert args.serial_port == "/dev/ttyUSB0"


# --------------------------------------------------------------------------
# Companion scripts import cleanly off-Pi
# --------------------------------------------------------------------------

def test_smoke_test_module_imports_without_hardware(mod):
    smoke = _load(SMOKE_PATH, "fs3000_smoke_test")
    assert hasattr(smoke, "main")
