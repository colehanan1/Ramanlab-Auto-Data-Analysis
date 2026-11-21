"""PiCode/test_odor_pins_pi2.py drives the Pi 2 light the way the rig does:
LDD-L PWM at 500 Hz, default 50 % duty (not a digital HIGH = 100 %)."""
import importlib.util
import sys
import types
from pathlib import Path

import pytest

_P = Path(__file__).resolve().parents[1] / "PiCode" / "test_odor_pins_pi2.py"


class FakeLgpio(types.ModuleType):
    def __init__(self):
        super().__init__("lgpio")
        self.calls = []

    def __getattr__(self, name):
        def f(*a):
            self.calls.append((name, a))
            return 0
        return f


def _run(monkeypatch, argv):
    fake = FakeLgpio()
    monkeypatch.setitem(sys.modules, "lgpio", fake)
    spec = importlib.util.spec_from_file_location("pins_pi2", _P)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(sys, "argv", ["x", "--hold", "0", *argv])
    mod.main()
    return fake.calls


def _light_pwm(calls):
    return [a for n, a in calls if n == "tx_pwm" and a[1] == 13]


def test_light_defaults_to_50_percent_pwm(monkeypatch):
    calls = _run(monkeypatch, [])
    pwm = _light_pwm(calls)
    assert (0, 13, 500.0, 50.0) in pwm
    assert pwm[-1][3] == 0                       # left dark
    # the light pin is never driven fully HIGH
    assert not [a for n, a in calls if n == "gpio_write" and a[1] == 13 and a[2] == 1]


def test_light_brightness_flag(monkeypatch):
    pwm = _light_pwm(_run(monkeypatch, ["--light-brightness", "15"]))
    assert (0, 13, 500.0, 15.0) in pwm


def test_valves_still_digital(monkeypatch):
    calls = _run(monkeypatch, [])
    assert (0, 25, 1) in [a for n, a in calls if n == "gpio_write"]
    assert not [a for n, a in calls if n == "tx_pwm" and a[1] != 13]
