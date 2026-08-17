"""Tests for PiCode/flow_log.py -- the valve-free FS3000 logger.

Covers the pure logic: summary statistics, the first-half/second-half drift
check, and CLI defaults. The module must import cleanly off-Pi.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

PI_CODE = Path(__file__).resolve().parents[1] / "PiCode"
MODULE_PATH = PI_CODE / "flow_log.py"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return _load(MODULE_PATH, "flow_log")


# --------------------------------------------------------------------------
# Summary statistics
# --------------------------------------------------------------------------

def test_summarize_basic_stats(mod):
    s = mod.summarize([1.0, 2.0, 3.0, 4.0], n_failed=0)
    assert s["n"] == 4
    assert s["mean"] == pytest.approx(2.5)
    assert s["median"] == pytest.approx(2.5)
    assert s["min"] == pytest.approx(1.0)
    assert s["max"] == pytest.approx(4.0)
    assert s["sd"] == pytest.approx(1.2909944, rel=1e-5)


def test_summarize_reports_failures_and_yield(mod):
    s = mod.summarize([1.0] * 90, n_failed=10)
    assert s["n_failed"] == 10
    assert s["yield_pct"] == pytest.approx(90.0)


def test_summarize_drift_compares_halves(mod):
    # first half mean 1.0, second half mean 3.0
    s = mod.summarize([1.0, 1.0, 3.0, 3.0], n_failed=0)
    assert s["first_half_mean"] == pytest.approx(1.0)
    assert s["second_half_mean"] == pytest.approx(3.0)
    assert s["drift"] == pytest.approx(2.0)


def test_summarize_drift_is_zero_for_steady_signal(mod):
    s = mod.summarize([2.0] * 50, n_failed=0)
    assert s["drift"] == pytest.approx(0.0)
    assert s["sd"] == pytest.approx(0.0)


def test_summarize_odd_count_splits_without_overlap(mod):
    s = mod.summarize([1.0, 1.0, 9.0, 3.0, 3.0], n_failed=0)
    # middle sample belongs to neither half
    assert s["first_half_mean"] == pytest.approx(1.0)
    assert s["second_half_mean"] == pytest.approx(3.0)


def test_summarize_single_sample_has_zero_sd_and_no_drift(mod):
    s = mod.summarize([1.5], n_failed=0)
    assert s["n"] == 1
    assert s["mean"] == pytest.approx(1.5)
    assert s["sd"] == pytest.approx(0.0)
    assert s["drift"] is None


def test_summarize_empty_returns_none_stats(mod):
    s = mod.summarize([], n_failed=5)
    assert s["n"] == 0
    assert s["mean"] is None
    assert s["n_failed"] == 5
    assert s["yield_pct"] == pytest.approx(0.0)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def test_parser_defaults_to_five_minutes_at_10hz(mod):
    args = mod.build_parser().parse_args([])
    assert args.minutes == pytest.approx(5.0)
    assert args.hz == pytest.approx(10.0)
    assert args.esp_url is None
    assert args.esp_sensor == "air_velocity"


def test_parser_accepts_esp_url_and_duration(mod):
    args = mod.build_parser().parse_args(
        ["--esp-url", "http://192.168.4.1", "--minutes", "1.5", "--hz", "5"])
    assert args.esp_url == "http://192.168.4.1"
    assert args.minutes == pytest.approx(1.5)
    assert args.hz == pytest.approx(5.0)


def test_parser_accepts_serial_port(mod):
    parser = mod.build_parser()
    assert parser.parse_args([]).serial_port is None
    args = parser.parse_args(["--serial-port", "/dev/ttyUSB0"])
    assert args.serial_port == "/dev/ttyUSB0"


def test_expected_sample_count(mod):
    assert mod.expected_samples(minutes=5.0, hz=10.0) == 3000
    assert mod.expected_samples(minutes=0.5, hz=2.0) == 60


# --------------------------------------------------------------------------
# Reuses the one FS3000 driver implementation
# --------------------------------------------------------------------------

def test_sensor_classes_are_defined_in_odor_flow_module_not_reimplemented(mod):
    """flow_log must borrow the driver, never carry a second copy of it."""
    import inspect

    for cls in (mod.EspFlowSensor, mod.FS3000, mod.FakeFlowSensor,
                mod.SerialFlowSensor):
        assert Path(inspect.getfile(cls)).name == "odor_flow_fs3000.py"
    assert mod.FS3000_ADDR == 0x28
    assert "class EspFlowSensor" not in MODULE_PATH.read_text()
