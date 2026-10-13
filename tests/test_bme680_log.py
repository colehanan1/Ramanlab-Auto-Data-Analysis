"""PiCode/bme680_log.py: 5 s countdown, then a fixed-rate BME680 log to CSV.

Run off-Pi with a fake sensor and a fake clock: ``sleep`` advances the clock,
and each sensor read can cost time too, so drift is observable.
"""

import csv
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

PICODE = Path(__file__).resolve().parent.parent / "PiCode"
sys.path.insert(0, str(PICODE))
import bme680_log  # noqa: E402
from bme680_log import record  # noqa: E402


class FakeClock:
    def __init__(self):
        self.t = 1000.0
        self.sleeps = []

    def monotonic(self):
        return self.t

    def sleep(self, s):
        assert s >= 0
        self.sleeps.append(s)
        self.t += s

    def wall(self):
        return datetime(2026, 10, 2, 10, 0, 0) + timedelta(seconds=self.t - 1000.0)


class FakeSensor:
    """Each property read costs ``read_cost`` seconds of fake time."""

    def __init__(self, clock, read_cost=0.0, fail_on=()):
        self.clock, self.read_cost, self.fail_on, self.reads = clock, read_cost, set(fail_on), 0

    def _read(self, value):
        self.clock.t += self.read_cost
        return value

    @property
    def relative_humidity(self):
        self.reads += 1
        if self.reads in self.fail_on:
            raise OSError("I2C read failed")
        return self._read(55.0 + self.reads * 0.01)

    @property
    def temperature(self):
        return self._read(24.5)

    @property
    def pressure(self):
        return self._read(1001.2)

    @property
    def gas(self):
        return self._read(120000)


def _run(tmp_path, clock, sensor, **kw):
    out = tmp_path / "log.csv"
    lines = []
    n = record(sensor, out, clock=clock.monotonic, sleep=clock.sleep, wall=clock.wall,
               echo=lambda *a, **k: lines.append(" ".join(str(x) for x in a)), **kw)
    with open(out, newline="") as fh:
        rows = list(csv.DictReader(fh))
    return n, rows, lines


def test_defaults_are_the_requested_protocol():
    assert bme680_log.DEFAULT_COUNTDOWN_S == 5
    assert bme680_log.DEFAULT_DURATION_S == 60


def test_counts_down_5_s_before_recording(tmp_path):
    clock = FakeClock()
    _, rows, lines = _run(tmp_path, clock, FakeSensor(clock), duration=60, interval=1.0)
    countdown = [l for l in lines if "Recording starts in" in l]
    assert [l.split("in ")[1] for l in countdown] == ["5 s", "4 s", "3 s", "2 s", "1 s"]
    # first sample lands 5 s after start
    assert rows[0]["time_local"] == "2026-10-02T10:00:05.000"


def test_one_minute_at_a_fixed_rate(tmp_path):
    clock = FakeClock()
    n, rows, _ = _run(tmp_path, clock, FakeSensor(clock), duration=60, interval=1.0)
    assert n == len(rows) == 60
    assert [float(r["elapsed_s"]) for r in rows] == [float(k) for k in range(60)]


def test_slow_reads_do_not_drift_the_schedule(tmp_path):
    """Deadline-based: a 0.3 s read must not push sample k to k*1.3 s."""
    clock = FakeClock()
    _, rows, _ = _run(tmp_path, clock, FakeSensor(clock, read_cost=0.075),
                      duration=10, interval=1.0)
    starts = [float(r["elapsed_s"]) for r in rows]
    assert starts == [float(k) for k in range(10)]


def test_custom_interval(tmp_path):
    clock = FakeClock()
    n, rows, _ = _run(tmp_path, clock, FakeSensor(clock), duration=60, interval=0.5)
    assert n == 120
    assert float(rows[1]["elapsed_s"]) == 0.5


def test_csv_columns_and_values(tmp_path):
    clock = FakeClock()
    _, rows, _ = _run(tmp_path, clock, FakeSensor(clock), duration=2, interval=1.0)
    assert list(rows[0]) == ["sample", "time_local", "elapsed_s", "humidity_pct",
                             "temperature_c", "pressure_hpa", "gas_ohms"]
    assert rows[0]["sample"] == "1"
    assert rows[0]["humidity_pct"] == "55.01"
    assert rows[0]["temperature_c"] == "24.5"
    assert rows[0]["pressure_hpa"] == "1001.2"
    assert rows[0]["gas_ohms"] == "120000"


def test_a_failed_read_is_logged_and_recording_continues(tmp_path):
    clock = FakeClock()
    n, rows, _ = _run(tmp_path, clock, FakeSensor(clock, fail_on={3}), duration=5, interval=1.0)
    assert n == 5
    assert rows[2]["humidity_pct"] == "NA"
    assert rows[3]["humidity_pct"] != "NA"


def test_ctrl_c_keeps_what_was_recorded(tmp_path):
    clock = FakeClock()

    class Interrupting(FakeSensor):
        @property
        def relative_humidity(self):
            if self.reads == 3:
                raise KeyboardInterrupt
            return FakeSensor.relative_humidity.fget(self)

    n, rows, lines = _run(tmp_path, clock, Interrupting(clock), duration=60, interval=1.0)
    assert n == len(rows) == 3
    assert any("Stopped early" in l for l in lines)


def test_summary_reports_humidity(tmp_path):
    clock = FakeClock()
    _, _, lines = _run(tmp_path, clock, FakeSensor(clock), duration=3, interval=1.0)
    summary = "\n".join(lines)
    assert "Humidity" in summary and "mean" in summary and "3 samples" in summary


def test_default_output_name_has_host_and_time(tmp_path):
    p = bme680_log.default_output(tmp_path, host="Flybehavior2",
                                  now=datetime(2026, 10, 2, 10, 5, 9))
    assert p == tmp_path / "bme680_Flybehavior2_20261002_100509.csv"


def test_refuses_while_a_rig_session_holds_the_sensor(monkeypatch):
    """Two readers on one BME680 corrupt each other's forced-mode reads."""
    monkeypatch.setattr(bme680_log, "_running_rig_sessions",
                        lambda: ["12345 python3 combinedv2_1.py --fly 1"])
    with pytest.raises(SystemExit, match="combinedv2_1"):
        bme680_log.main(["--out", "/tmp/x.csv"])


@pytest.mark.parametrize("argv,is_rig", [
    (["python3", "combinedv2_1.py", "--fly", "1"], True),
    (["/home/ramanlab/cam-env/bin/python", "combinedv2_1_pi1.py", "--training"], True),
    (["python3", "-u", "combinedv2_1_pi3.py"], True),
    (["bash", "-c", "python3 combinedv2_1.py --fly 1"], False),   # ssh wrapper
    (["grep", "combinedv2_1"], False),
    (["python3", "bme680_log.py"], False),
    ([], False),
])
def test_only_real_rig_processes_count(argv, is_rig):
    assert bme680_log._is_rig_cmdline(argv) is is_rig


class HeaterSensor(FakeSensor):
    def __init__(self, clock):
        super().__init__(clock)
        self.heater_calls, self.gas_reads = [], 0

    def set_gas_heater(self, temp, ms):
        self.heater_calls.append((temp, ms))
        return True

    @property
    def gas(self):
        self.gas_reads += 1
        return 120000


def test_no_gas_turns_the_heater_off():
    """The ~320 C gas hotplate warms the die and drags RH down (seen on the rigs)."""
    clock = FakeClock()
    s = HeaterSensor(clock)
    bme680_log.configure_gas(s, enabled=False)
    assert s.heater_calls == [(None, None)]


def test_gas_on_leaves_the_rig_default_alone():
    clock = FakeClock()
    s = HeaterSensor(clock)
    bme680_log.configure_gas(s, enabled=True)
    assert s.heater_calls == []


def test_no_gas_never_reads_gas_and_says_so(tmp_path):
    clock = FakeClock()
    s = HeaterSensor(clock)
    _, rows, _ = _run(tmp_path, clock, s, duration=3, interval=1.0, read_gas=False)
    assert s.gas_reads == 0
    assert {r["gas_ohms"] for r in rows} == {"off"}
