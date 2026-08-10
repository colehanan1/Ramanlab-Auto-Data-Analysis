"""Tests for PiCode/odor_bme680_exp.py.

The module must import cleanly off-Pi (no lgpio / board / adafruit_bme680),
so every hardware import lives inside main(). These tests cover the pure
logic: pin maps, channel parsing, schedule construction, sample pacing and
the gas-baseline warm-up rule.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

PI_CODE = Path(__file__).resolve().parents[1] / "PiCode"
MODULE_PATH = PI_CODE / "odor_bme680_exp.py"
VARIANT_PATH = PI_CODE / "odor_bme680_1234.py"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return _load(MODULE_PATH, "odor_bme680_exp")


@pytest.fixture(scope="module")
def variant():
    return _load(VARIANT_PATH, "odor_bme680_1234")


# --------------------------------------------------------------------------
# Pin maps
# --------------------------------------------------------------------------

def test_pi1_pins_match_combinedv2_1_pi1(mod):
    """Channel numbers 1-8 must resolve to the BCM pins the rig actually uses."""
    pins = mod.RIG_PIN_MAPS["pi1"]
    assert pins == {
        1: ("OFM_A", 23),
        2: ("OFM_B", 22),
        3: ("OFM_C", 12),
        4: ("OFM_E", 13),
        5: ("OFM_H", 27),
        6: ("OFM_L", 26),
        7: ("OFM_O", 16),
        8: ("OFM_P", 17),
    }


def test_pi2_and_pi3_pins(mod):
    assert mod.RIG_PIN_MAPS["pi2"] == {
        1: ("OFM_A", 25),
        2: ("OFM_B", 22),
        3: ("OFM_C", 27),
        4: ("OFM_E", 16),
        5: ("OFM_H", 24),
        6: ("OFM_L", 17),
        7: ("OFM_O", 12),
        8: ("OFM_P", 23),
    }
    assert mod.RIG_PIN_MAPS["pi3"] == {
        1: ("OFM_A", 26),
        2: ("OFM_B", 25),
        3: ("OFM_C", 22),
        4: ("OFM_E", 16),
        5: ("OFM_H", 24),
        6: ("OFM_L", 17),
        7: ("OFM_O", 27),
        8: ("OFM_P", 23),
    }


@pytest.mark.parametrize("rig", ["pi1", "pi2", "pi3"])
def test_every_rig_has_eight_unique_channels_and_pins(mod, rig):
    pins = mod.RIG_PIN_MAPS[rig]
    assert sorted(pins) == list(range(1, 9))
    bcm = [p for _, p in pins.values()]
    assert len(set(bcm)) == 8, f"{rig} has a duplicate BCM pin: {bcm}"
    names = [n for n, _ in pins.values()]
    assert len(set(names)) == 8


def test_channel_8_is_the_master_line(mod):
    """OFM_P gates the whole manifold; it is not a branch odor."""
    assert mod.MASTER_CHANNEL == 8
    for rig, pins in mod.RIG_PIN_MAPS.items():
        assert pins[mod.MASTER_CHANNEL][0] == "OFM_P", rig


# --------------------------------------------------------------------------
# Channel selection
# --------------------------------------------------------------------------

def test_parse_channels_range(mod):
    assert mod.parse_channels("1-7") == [1, 2, 3, 4, 5, 6, 7]


def test_parse_channels_list_is_deduped_and_ordered_as_written(mod):
    assert mod.parse_channels("5,1,3,5") == [5, 1, 3]


def test_parse_channels_mixed(mod):
    assert mod.parse_channels("1-3,7") == [1, 2, 3, 7]


def test_parse_channels_all_excludes_master(mod):
    assert mod.parse_channels("all") == [1, 2, 3, 4, 5, 6, 7]


def test_parse_channels_rejects_out_of_range(mod):
    with pytest.raises(ValueError, match="1-8"):
        mod.parse_channels("9")
    with pytest.raises(ValueError, match="1-8"):
        mod.parse_channels("0-3")


def test_parse_channels_rejects_garbage(mod):
    with pytest.raises(ValueError):
        mod.parse_channels("hexanol")


def test_parse_channels_rejects_empty(mod):
    with pytest.raises(ValueError):
        mod.parse_channels("")


def test_parse_channels_allows_master_explicitly(mod):
    """Channel 8 alone = carrier air with no branch open, a valid control."""
    assert mod.parse_channels("8") == [8]


# --------------------------------------------------------------------------
# Schedule
# --------------------------------------------------------------------------

def test_blocked_schedule_groups_repeats_per_channel(mod):
    sched = mod.build_schedule([1, 2], repeats=2, on_s=10.0, off_s=20.0, order="blocked")
    assert [(s.channel, s.repeat, s.phase) for s in sched] == [
        (1, 0, "odor_on"), (1, 0, "odor_off"),
        (1, 1, "odor_on"), (1, 1, "odor_off"),
        (2, 0, "odor_on"), (2, 0, "odor_off"),
        (2, 1, "odor_on"), (2, 1, "odor_off"),
    ]


def test_interleaved_schedule_round_robins_channels(mod):
    sched = mod.build_schedule([1, 2], repeats=2, on_s=10.0, off_s=20.0, order="interleaved")
    assert [(s.channel, s.repeat) for s in sched] == [
        (1, 0), (1, 0), (2, 0), (2, 0),
        (1, 1), (1, 1), (2, 1), (2, 1),
    ]


def test_schedule_carries_durations_and_pin_labels(mod):
    sched = mod.build_schedule([5], repeats=1, on_s=30.0, off_s=300.0, order="blocked",
                               pins=mod.RIG_PIN_MAPS["pi1"])
    on, off = sched
    assert (on.phase, on.duration, on.name, on.pin) == ("odor_on", 30.0, "OFM_H", 27)
    assert (off.phase, off.duration, off.name, off.pin) == ("odor_off", 300.0, "OFM_H", 27)


def test_schedule_rejects_zero_repeats(mod):
    with pytest.raises(ValueError):
        mod.build_schedule([1], repeats=0, on_s=10.0, off_s=10.0, order="blocked")


def test_schedule_rejects_unknown_order(mod):
    with pytest.raises(ValueError):
        mod.build_schedule([1], repeats=1, on_s=1.0, off_s=1.0, order="sideways")


def test_total_runtime_seconds(mod):
    sched = mod.build_schedule([1, 2, 3], repeats=2, on_s=30.0, off_s=210.0, order="blocked")
    assert mod.total_runtime_s(sched) == pytest.approx(3 * 2 * (30.0 + 210.0))


# --------------------------------------------------------------------------
# Sample pacing
# --------------------------------------------------------------------------

def test_pacer_holds_period_even_when_reads_are_slow(mod):
    """A read costs ~0.2 s (TPHG cycle); naive sleep(T) would drift the rate."""
    clock = [100.0]
    slept = []

    def fake_sleep(dt):
        slept.append(dt)
        clock[0] += dt

    pacer = mod.Pacer(period=0.5, now=lambda: clock[0], sleep=fake_sleep)
    pacer.start()
    for _ in range(3):
        clock[0] += 0.2  # the read itself
        pacer.wait()
    assert slept == pytest.approx([0.3, 0.3, 0.3])
    assert clock[0] == pytest.approx(101.5)


def test_pacer_does_not_sleep_negative_when_read_overruns_period(mod):
    clock = [0.0]
    slept = []

    def fake_sleep(dt):
        slept.append(dt)
        clock[0] += dt

    pacer = mod.Pacer(period=0.1, now=lambda: clock[0], sleep=fake_sleep)
    pacer.start()
    clock[0] += 0.9  # a very slow read
    pacer.wait()
    assert slept == [0.0]


# --------------------------------------------------------------------------
# Warm-up / baseline stability
# --------------------------------------------------------------------------

def test_monitor_not_stable_before_min_seconds(mod):
    mon = mod.BaselineMonitor(window_s=60.0, tol=0.01, min_s=300.0, max_s=1800.0)
    for i in range(400):
        mon.add(t=float(i), gas=50_000.0, heat_stable=True)
    assert not mon.is_stable(now=200.0)
    assert mon.is_stable(now=399.0)


def test_monitor_rejects_a_drifting_baseline(mod):
    """Fresh BME680 hotplates burn in upward for minutes; that is not stable."""
    mon = mod.BaselineMonitor(window_s=60.0, tol=0.01, min_s=300.0, max_s=1800.0)
    gas = 30_000.0
    for i in range(600):
        gas *= 1.002  # +0.2 %/sample
        mon.add(t=float(i), gas=gas, heat_stable=True)
    assert not mon.is_stable(now=599.0)


def test_monitor_accepts_a_flat_baseline_with_noise(mod):
    mon = mod.BaselineMonitor(window_s=60.0, tol=0.01, min_s=300.0, max_s=1800.0)
    for i in range(600):
        jitter = 150.0 if i % 2 else -150.0  # ±0.3 % RMS-ish noise, no trend
        mon.add(t=float(i), gas=50_000.0 + jitter, heat_stable=True)
    assert mon.is_stable(now=599.0)


def test_monitor_gives_up_at_max_seconds(mod):
    mon = mod.BaselineMonitor(window_s=60.0, tol=0.01, min_s=300.0, max_s=1800.0)
    gas = 30_000.0
    for i in range(2000):
        gas *= 1.002
        mon.add(t=float(i), gas=gas, heat_stable=True)
    assert not mon.is_stable(now=1799.0)
    assert mon.timed_out(now=1801.0)
    assert not mon.timed_out(now=1799.0)


def test_monitor_requires_heater_stability(mod):
    """heat_stab_r == 0 means the hotplate never reached target (datasheet 3.4)."""
    mon = mod.BaselineMonitor(window_s=60.0, tol=0.01, min_s=300.0, max_s=1800.0)
    for i in range(600):
        mon.add(t=float(i), gas=50_000.0, heat_stable=False)
    assert not mon.is_stable(now=599.0)


def test_monitor_tolerates_unknown_heater_stability(mod):
    """Older adafruit_bme680 does not expose the bit; None must not block."""
    mon = mod.BaselineMonitor(window_s=60.0, tol=0.01, min_s=300.0, max_s=1800.0)
    for i in range(600):
        mon.add(t=float(i), gas=50_000.0, heat_stable=None)
    assert mon.is_stable(now=599.0)


def test_monitor_needs_the_window_filled(mod):
    """Two samples 1 s apart must not pass for a 60 s window."""
    mon = mod.BaselineMonitor(window_s=60.0, tol=0.01, min_s=0.0, max_s=1800.0)
    mon.add(t=0.0, gas=50_000.0, heat_stable=True)
    mon.add(t=1.0, gas=50_000.0, heat_stable=True)
    assert not mon.is_stable(now=1.0)


def test_monitor_drift_is_reported_for_logging(mod):
    mon = mod.BaselineMonitor(window_s=60.0, tol=0.01, min_s=0.0, max_s=1800.0)
    for i in range(120):
        mon.add(t=float(i), gas=50_000.0, heat_stable=True)
    assert mon.drift(now=119.0) == pytest.approx(0.0, abs=1e-9)

    rising = mod.BaselineMonitor(window_s=60.0, tol=0.01, min_s=0.0, max_s=1800.0)
    for i in range(120):
        rising.add(t=float(i), gas=50_000.0 + 100.0 * i, heat_stable=True)
    assert rising.drift(now=119.0) > 0.05


def test_monitor_ignores_samples_outside_the_window(mod):
    """A huge early transient must not poison a now-flat window."""
    mon = mod.BaselineMonitor(window_s=60.0, tol=0.01, min_s=0.0, max_s=1800.0)
    mon.add(t=0.0, gas=1_000.0, heat_stable=True)
    for i in range(100, 200):
        mon.add(t=float(i), gas=50_000.0, heat_stable=True)
    assert mon.is_stable(now=199.0)


# --------------------------------------------------------------------------
# Warm-up defaults (Adafruit BME680 guide)
# --------------------------------------------------------------------------

def test_default_warmup_floor_is_thirty_minutes(mod):
    """Adafruit: "30 minutes in the desired mode every time the sensor is in use"."""
    assert mod.WARMUP_MIN_S == 1800.0
    args = mod.build_parser().parse_args(["--rig", "pi1"])
    assert args.warmup_min == 1800.0


def test_default_warmup_ceiling_leaves_room_past_the_floor(mod):
    """A cap at the floor would make the drift test unreachable."""
    assert mod.WARMUP_MAX_S > mod.WARMUP_MIN_S
    args = mod.build_parser().parse_args(["--rig", "pi1"])
    assert args.warmup_max == 3600.0


def test_warmup_floor_holds_even_when_the_baseline_looks_flat_early(mod):
    """A flat-looking 10 min baseline must not shortcut the 30 min soak."""
    mon = mod.BaselineMonitor(window_s=60.0, tol=0.01,
                              min_s=mod.WARMUP_MIN_S, max_s=mod.WARMUP_MAX_S)
    for i in range(2000):
        mon.add(t=float(i), gas=50_000.0, heat_stable=True)
    assert not mon.is_stable(now=600.0)
    assert not mon.is_stable(now=1799.0)
    assert mon.is_stable(now=1801.0)


def test_warmup_seconds_are_counted_in_the_runtime_estimate(mod):
    sched = mod.build_schedule([1], repeats=1, on_s=30.0, off_s=210.0, order="blocked")
    args = mod.build_parser().parse_args(["--rig", "pi1"])
    assert mod.estimated_total_s(sched, args) == pytest.approx(
        240.0 + args.warmup_min + args.stabilize)


# --------------------------------------------------------------------------
# CSV rows
# --------------------------------------------------------------------------

def test_row_keeps_legacy_column_names(mod):
    """odor_exp.py plots Timestamp/Gas/Phase/Repeat; keep them working."""
    row = mod.make_row(
        t=12.5, wall="2026-08-03T15:00:00", reading=mod.Reading(
            gas=51234.5678, temperature=24.123, pressure=1001.987,
            humidity=41.4321, heat_stable=True),
        phase="odor_on", channel=5, name="OFM_H", pin=27, repeat=1,
    )
    assert row["Timestamp"] == 12.5
    assert row["Gas"] == 51234.57
    assert row["Temp"] == 24.12
    assert row["Pres"] == 1001.99
    assert row["Hum"] == 41.43
    assert row["Phase"] == "odor_on"
    assert row["Repeat"] == 1
    assert row["Channel"] == 5
    assert row["Odor"] == "OFM_H"
    assert row["Pin"] == 27
    assert row["HeatStable"] is True
    assert row["WallClock"] == "2026-08-03T15:00:00"


def test_row_for_baseline_phase_has_no_channel(mod):
    row = mod.make_row(
        t=1.0, wall="2026-08-03T15:00:00", reading=mod.Reading(
            gas=1.0, temperature=1.0, pressure=1.0, humidity=1.0, heat_stable=None),
        phase="warmup", channel=None, name=None, pin=None, repeat=-1,
    )
    assert row["Channel"] == ""
    assert row["Odor"] == "none"
    assert row["Pin"] == ""
    assert row["Repeat"] == -1
    assert row["HeatStable"] == ""


def test_csv_fieldnames_cover_every_row_key(mod):
    row = mod.make_row(
        t=0.0, wall="w", reading=mod.Reading(1.0, 2.0, 3.0, 4.0, True),
        phase="warmup", channel=1, name="OFM_A", pin=23, repeat=0,
    )
    assert set(row) == set(mod.CSV_FIELDS)


# --------------------------------------------------------------------------
# The 1-4 variant: runnable with no arguments at all
# --------------------------------------------------------------------------

def test_variant_runs_with_no_arguments(variant, mod):
    """`python odor_bme680_1234.py` -- no flags, no required args."""
    args = mod.build_parser().parse_args(variant.DEFAULT_ARGV)
    assert args.rig == "pi1"
    assert mod.parse_channels(args.channels) == [1, 2, 3, 4]
    assert args.no_master is True


def test_variant_never_selects_the_master_channel(variant, mod):
    args = mod.build_parser().parse_args(variant.DEFAULT_ARGV)
    assert mod.MASTER_CHANNEL not in mod.parse_channels(args.channels)


def test_variant_arguments_can_still_be_overridden(variant, mod):
    """Defaults go first so a user flag of the same name wins."""
    args = mod.build_parser().parse_args(variant.DEFAULT_ARGV + ["--rig", "pi2",
                                                                "--repeats", "5"])
    assert args.rig == "pi2"
    assert args.repeats == 5
    assert args.no_master is True


def test_variant_reuses_the_main_module_rather_than_forking_it(variant, mod):
    """The variant must delegate, not carry its own copy of the pin maps."""
    assert Path(variant.exp.__file__) == MODULE_PATH
    assert variant.exp.RIG_PIN_MAPS == mod.RIG_PIN_MAPS
    assert variant.exp.WARMUP_MIN_S == mod.WARMUP_MIN_S
    assert variant.main is not mod.main


def test_variant_schedule_is_four_channels_one_at_a_time(variant, mod):
    args = mod.build_parser().parse_args(variant.DEFAULT_ARGV)
    sched = mod.build_schedule(mod.parse_channels(args.channels), args.repeats,
                               args.on, args.off, order=args.order,
                               pins=mod.RIG_PIN_MAPS[args.rig])
    opens = [s.channel for s in sched if s.phase == "odor_on"]
    assert opens == sorted(opens), "channels must be driven one at a time, in order"
    assert set(opens) == {1, 2, 3, 4}
    # every ON is followed by its own OFF: never two valves open at once
    assert [s.phase for s in sched] == ["odor_on", "odor_off"] * (len(sched) // 2)


# --------------------------------------------------------------------------
# Valve driving
# --------------------------------------------------------------------------

class FakeGPIO:
    """Records the exact write order so the master/branch stagger is checkable."""

    def __init__(self):
        self.writes = []
        self.state = {}

    def gpio_write(self, handle, pin, level):
        self.writes.append((pin, level))
        self.state[pin] = level


def test_open_odor_raises_branch_before_master(mod):
    """combinedv2_1_pi1._set_odor: branch on, 50 ms, then master on."""
    gpio = FakeGPIO()
    sleeps = []
    valves = mod.Valves(gpio=gpio, handle=0, pins=mod.RIG_PIN_MAPS["pi1"],
                        stagger_s=0.05, sleep=sleeps.append)
    valves.open(5)
    assert gpio.writes == [(27, 1), (17, 1)]
    assert sleeps == [0.05]


def test_close_odor_drops_master_before_branch(mod):
    gpio = FakeGPIO()
    sleeps = []
    valves = mod.Valves(gpio=gpio, handle=0, pins=mod.RIG_PIN_MAPS["pi1"],
                        stagger_s=0.05, sleep=sleeps.append)
    valves.open(5)
    gpio.writes.clear()
    sleeps.clear()
    valves.close(5)
    assert gpio.writes == [(17, 0), (27, 0)]
    assert sleeps == [0.05]


def test_zero_stagger_reproduces_the_old_simultaneous_behaviour(mod):
    gpio = FakeGPIO()
    sleeps = []
    valves = mod.Valves(gpio=gpio, handle=0, pins=mod.RIG_PIN_MAPS["pi1"],
                        stagger_s=0.0, sleep=sleeps.append)
    valves.open(1)
    assert gpio.writes == [(23, 1), (17, 1)]
    assert sleeps == []


def test_opening_the_master_channel_alone_drives_only_the_master(mod):
    gpio = FakeGPIO()
    valves = mod.Valves(gpio=gpio, handle=0, pins=mod.RIG_PIN_MAPS["pi1"],
                        stagger_s=0.05, sleep=lambda _: None)
    valves.open(8)
    assert gpio.writes == [(17, 1)]
    valves.close(8)
    assert gpio.writes[-1] == (17, 0)


def test_no_master_opens_only_the_branch(mod):
    """--no-master: OFM_P must never be written, in either direction."""
    gpio = FakeGPIO()
    sleeps = []
    valves = mod.Valves(gpio=gpio, handle=0, pins=mod.RIG_PIN_MAPS["pi1"],
                        stagger_s=0.05, sleep=sleeps.append, use_master=False)
    valves.open(3)
    valves.close(3)
    assert gpio.writes == [(12, 1), (12, 0)]
    assert 17 not in gpio.state
    assert sleeps == []


def test_no_master_all_off_leaves_the_master_pin_untouched(mod):
    gpio = FakeGPIO()
    valves = mod.Valves(gpio=gpio, handle=0, pins=mod.RIG_PIN_MAPS["pi1"],
                        stagger_s=0.0, sleep=lambda _: None, use_master=False)
    valves.open(1)
    valves.all_off()
    assert 17 not in [pin for pin, _ in gpio.writes]
    assert gpio.state[23] == 0


def test_no_master_rejects_opening_the_master_channel(mod):
    gpio = FakeGPIO()
    valves = mod.Valves(gpio=gpio, handle=0, pins=mod.RIG_PIN_MAPS["pi1"],
                        stagger_s=0.0, sleep=lambda _: None, use_master=False)
    with pytest.raises(ValueError, match="master"):
        valves.open(mod.MASTER_CHANNEL)


def test_pins_to_claim_skips_the_master_when_disabled(mod):
    """Claiming a pin as output drives it low, which is still driving it."""
    pins = mod.RIG_PIN_MAPS["pi1"]
    assert mod.pins_to_claim(pins, use_master=True) == [23, 22, 12, 13, 27, 26, 16, 17]
    assert mod.pins_to_claim(pins, use_master=False) == [23, 22, 12, 13, 27, 26, 16]


def test_all_off_drops_master_first_then_every_branch(mod):
    gpio = FakeGPIO()
    valves = mod.Valves(gpio=gpio, handle=0, pins=mod.RIG_PIN_MAPS["pi1"],
                        stagger_s=0.0, sleep=lambda _: None)
    valves.open(2)
    gpio.writes.clear()
    valves.all_off()
    assert gpio.writes[0] == (17, 0)
    assert set(gpio.state.values()) == {0}
    assert len(gpio.writes) == 8
