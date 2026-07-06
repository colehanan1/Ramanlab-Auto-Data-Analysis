#!/usr/bin/env python3
"""Tests for LDD-L driver PWM dimming limits in PiCode/light_gui.py.

The Pi 1 light is now driven through a Mean Well LDD-L constant-current
driver whose PWM DIM input only accepts 100 Hz - 1 kHz (LDD-300~700L spec).
These tests pin the clamp helpers so no code path can emit an out-of-spec
dimming frequency or duty cycle.
"""

import os
import sys
import types

# light_gui imports lgpio (Pi-only) at module level - stub it out first.
sys.modules.setdefault("lgpio", types.ModuleType("lgpio"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "PiCode"))

import light_gui


def test_freq_clamped_to_ldd_spec():
    assert light_gui.clamp_pwm_freq(2000.0) == 1000.0
    assert light_gui.clamp_pwm_freq(50.0) == 100.0
    assert light_gui.clamp_pwm_freq(500.0) == 500.0


def test_freq_bounds_match_ldd_spec():
    assert light_gui.PWM_FREQ_MIN_HZ == 100.0
    assert light_gui.PWM_FREQ_MAX_HZ == 1000.0
    assert 100.0 <= light_gui.PWM_FREQ_DEFAULT_HZ <= 1000.0


def test_duty_clamped():
    assert light_gui.clamp_duty(150.0) == 100.0
    assert light_gui.clamp_duty(-5.0) == 0.0
    assert light_gui.clamp_duty(42.5) == 42.5
