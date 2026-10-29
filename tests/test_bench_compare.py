"""Tests for scripts/tools/bench_compare.py, the OFF-vs-ON rig bench comparer.

The bench runs the same no-fly protocol OFF, ON, OFF, ON (ON = live stream +
telemetry) and the comparer decides whether ON perturbed recording or trial timing.
Hard stops: an ON frame gap beyond what the OFF runs show, or a stimulus interval
(odor / light duration, light-odor lag) more than 1 ms beyond the OFF-vs-OFF spread.
The camera-start offset (first odor-on minus RECORDING_START) is recorded, not judged.

Synthetic batch folders are built in tmp_path in the rig's real file layout (frames
CSV + ``sensors_`` CSV per trial); the script is run as a subprocess for exit codes.
"""
from __future__ import annotations

import csv
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "tools" / "bench_compare.py"

spec = importlib.util.spec_from_file_location("bench_compare", SCRIPT)
bc = importlib.util.module_from_spec(spec)
sys.modules["bench_compare"] = bc
spec.loader.exec_module(bc)

FRAME_NS = 25_000_000          # 40 fps
N_FRAMES = 200
T0 = 5_000_000_000_000         # arbitrary monotonic origin
# (label, offset from RECORDING_START in ms)
BASE_EVENTS = [
    ("OFM_A_ON", 10_000.0),
    ("LIGHT_SOLID_ON", 10_500.0),
    ("LIGHT_OFF", 15_000.0),
    ("OFM_A_OFF", 15_000.5),
]


def make_run(root: Path, *, trials=3, gap_trial=None, gap_ms=50.0, shift=None,
             skip_trial=None, extra_pulses=0, drop_event=None, cam_offset=None,
             frames=None):
    """Write a synthetic batch folder.

    gap_trial: 1-based trial that gets one frame interval of ``gap_ms``.
    shift: {(trial, label): ms} added to that event's offset.
    skip_trial: 1-based trial left out entirely.
    drop_event: (trial, label) left out of the sensors CSV.
    cam_offset: {trial: ms} added to EVERY event offset of that trial (a camera that
        starts late/early relative to the valve schedule).
    frames: {trial: n} frame count override (default N_FRAMES).
    """
    root.mkdir(parents=True, exist_ok=True)
    shift = shift or {}
    cam_offset = cam_offset or {}
    frames = frames or {}
    for i in range(1, trials + 1):
        if i == skip_trial:
            continue
        stem = f"output_oct_30_batch_1_testing_{i}_Hexanol_20251030_10{i:02d}00"
        t = T0 + i * 100_000_000_000
        with open(root / f"{stem}.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["FrameNumber", "SensorNs", "UTC_ISO", "ActiveOFM",
                        "FrameInterval_s", "Deviation_s"])
            ts = t
            for n in range(frames.get(i, N_FRAMES)):
                w.writerow([n, ts, "", "", "", ""])
                step = FRAME_NS
                if i == gap_trial and n == 100:
                    step = int(gap_ms * 1e6)
                ts += step
        with open(root / f"sensors_{stem}.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["MonoNs", "Humidity", "Pressure", "Temperature", "Gas",
                        "OFM_Event"])
            w.writerow([t, "N/A", "N/A", "N/A", "N/A", "RECORDING_START"])
            w.writerow([t + 1_000_000, 40, 1000, 22, 1, "BME_START"])
            w.writerow([t + 2_000_000_000, 40, 1000, 22, 1, "BME_PERIODIC"])
            for label, off in BASE_EVENTS:
                if drop_event == (i, label):
                    continue
                off = off + shift.get((i, label), 0.0) + cam_offset.get(i, 0.0)
                w.writerow([t + int(off * 1e6), "N/A", "N/A", "N/A", "N/A", label])
            for k in range(extra_pulses):
                w.writerow([t + int((11_000 + k * 100) * 1e6), "N/A", "N/A", "N/A",
                            "N/A", "LIGHT_PULSE_ON"])
    return root


def run_cli(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)],
                          capture_output=True, text=True)


def four_runs(tmp_path, *, on1=None, on2=None, off1=None, off2=None):
    d = {
        "off1": make_run(tmp_path / "off1", **(off1 or {})),
        "on1": make_run(tmp_path / "on1", **(on1 or {})),
        "off2": make_run(tmp_path / "off2", **(off2 or {})),
        "on2": make_run(tmp_path / "on2", **(on2 or {})),
    }
    return [f"--run=OFF={d['off1']}", f"--run=ON={d['on1']}",
            f"--run=OFF={d['off2']}", f"--run=ON={d['on2']}"]


# ── verdicts ─────────────────────────────────────────────────────────────────

def test_identical_runs_pass(tmp_path):
    r = run_cli(*four_runs(tmp_path))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "RESULT: PASS" in r.stdout
    assert "| ON1 |" in r.stdout and "| OFF2 |" in r.stdout


def test_extra_50ms_gap_in_on_fails(tmp_path):
    r = run_cli(*four_runs(tmp_path, on2={"gap_trial": 2, "gap_ms": 50.0}))
    assert r.returncode == 1, r.stdout
    assert "RESULT: FAIL" in r.stdout
    assert "ON2 trial 2" in r.stdout and "gap" in r.stdout


def test_gap_under_threshold_is_not_a_gap(tmp_path):
    # 36 ms is jitter, 37.5 ms (1.5 intervals) is the limit
    r = run_cli(*four_runs(tmp_path, on2={"gap_trial": 2, "gap_ms": 36.0}))
    assert r.returncode == 0, r.stdout


def test_gap_also_present_in_off_is_not_added_by_on(tmp_path):
    kw = {"gap_trial": 2, "gap_ms": 50.0}
    r = run_cli(*four_runs(tmp_path, off1=kw, on1=kw, off2=kw, on2=kw))
    assert r.returncode == 0, r.stdout


def test_much_longer_gap_hides_not_behind_off_gap(tmp_path):
    r = run_cli(*four_runs(tmp_path, off1={"gap_trial": 2, "gap_ms": 50.0},
                           on2={"gap_trial": 2, "gap_ms": 300.0}))
    assert r.returncode == 1, r.stdout
    assert "worst gap" in r.stdout


def test_event_shift_2ms_over_small_off_spread_fails(tmp_path):
    key = (2, "LIGHT_SOLID_ON")
    r = run_cli(*four_runs(tmp_path,
                           off2={"shift": {key: 0.3}},      # OFF spread 0.3 ms
                           on1={"shift": {key: 2.0}}))
    assert r.returncode == 1, r.stdout
    assert "LIGHT_SOLID_ON" in r.stdout
    assert any(line.startswith("- ON1 trial 2") for line in r.stdout.splitlines())


def test_event_shift_half_ms_passes(tmp_path):
    key = (2, "LIGHT_SOLID_ON")
    r = run_cli(*four_runs(tmp_path,
                           off2={"shift": {key: 0.3}},
                           on1={"shift": {key: 0.5}}))
    assert r.returncode == 0, r.stdout


def test_large_off_spread_widens_the_floor(tmp_path):
    key = (1, "OFM_A_ON")
    # OFF runs differ by 4 ms, so an ON run 3 ms out is within floor + tolerance
    r = run_cli(*four_runs(tmp_path, off2={"shift": {key: 4.0}},
                           on1={"shift": {key: 3.0}}))
    assert r.returncode == 0, r.stdout


def test_missing_trial_fails_with_reason(tmp_path):
    r = run_cli(*four_runs(tmp_path, on1={"skip_trial": 3}))
    assert r.returncode == 1, r.stdout
    assert "ON1 has 2 trial(s)" in r.stdout


def test_missing_event_key_fails(tmp_path):
    r = run_cli(*four_runs(tmp_path, on2={"drop_event": (2, "LIGHT_OFF")}))
    assert r.returncode == 1, r.stdout
    assert "LIGHT_OFF #1 missing in ON2" in r.stdout


def test_fewer_frames_in_on_fails(tmp_path):
    runs = four_runs(tmp_path)
    victim = next(tmp_path.joinpath("on1").glob("output_*_testing_1_*.csv"))
    lines = victim.read_text().splitlines()
    victim.write_text("\n".join(lines[:-5]) + "\n")
    r = run_cli(*runs)
    assert r.returncode == 1, r.stdout
    assert "ON1 trial 1" in r.stdout and "frames" in r.stdout


def test_ignore_label_drops_pulse_edges(tmp_path):
    # ON has pulse edges OFF lacks: missing keys fail unless ignored
    runs = four_runs(tmp_path, on1={"extra_pulses": 3})
    assert run_cli(*runs).returncode == 1
    r = run_cli(*runs, "--ignore-label", "LIGHT_PULSE_")
    assert r.returncode == 0, r.stdout


def test_single_off_run_warns(tmp_path):
    r = run_cli(f"--run=OFF={make_run(tmp_path / 'a')}",
                f"--run=ON={make_run(tmp_path / 'b')}")
    assert r.returncode == 0, r.stdout
    assert "only 1 OFF run" in r.stdout


def test_out_file_is_written(tmp_path):
    out = tmp_path / "report.md"
    r = run_cli(*four_runs(tmp_path), "--out", out)
    assert r.returncode == 0
    assert out.read_text().strip() == r.stdout.strip()


# ── camera-start offset (recorded) vs stimulus timing (judged) ───────────────

def test_camera_start_offset_4ms_passes_and_is_reported(tmp_path):
    r = run_cli(*four_runs(tmp_path, on1={"cam_offset": {1: -4.0}}))
    assert r.returncode == 0, r.stdout
    assert "Camera-start offset is recorded per trial and not judged" in r.stdout
    assert "## Camera start offset (recorded, not judged)" in r.stdout
    sec = r.stdout.split("## Camera start offset (recorded, not judged)")[1] \
        .split("\n## ")[0]
    row1 = next(line for line in sec.splitlines() if line.startswith("| 1 |"))
    assert "9996.000" in row1 and "-4.0" in row1        # ON1: 10000 - 4
    assert "informational" in r.stdout
    assert "## Stimulus timing (judged)" in r.stdout


def test_camera_offset_value_and_fallback(tmp_path):
    run = bc.load_run("ON1", "ON", make_run(tmp_path / "r", cam_offset={2: 2.5}))
    assert run.trials[0].camera_offset_ms == pytest.approx(10_000.0)
    assert run.trials[1].camera_offset_ms == pytest.approx(10_002.5)
    assert run.trials[0].camera_event == "OFM_A_ON"


def test_camera_offset_falls_back_to_first_event_without_odor(tmp_path):
    root = make_run(tmp_path / "r", trials=1)
    sp = next(root.glob("sensors_*.csv"))
    kept = [ln for ln in sp.read_text().splitlines() if "OFM_A_" not in ln]
    sp.write_text("\n".join(kept) + "\n")
    t = bc.load_run("OFF1", "OFF", root).trials[0]
    assert t.camera_event == "LIGHT_SOLID_ON"
    assert t.camera_offset_ms == pytest.approx(10_500.0)


def test_odor_duration_3ms_longer_fails(tmp_path):
    key = (2, "OFM_A_OFF")
    r = run_cli(*four_runs(tmp_path, off2={"shift": {key: 0.3}},
                           on1={"shift": {key: 3.0}}))
    assert r.returncode == 1, r.stdout
    assert any(line.startswith("- ON1 trial 2") and "odor A duration" in line
               for line in r.stdout.splitlines()), r.stdout


def test_light_duration_within_floor_passes(tmp_path):
    key = (2, "LIGHT_OFF")
    r = run_cli(*four_runs(tmp_path, off2={"shift": {key: 0.4}},
                           on1={"shift": {key: 0.9}}))
    assert r.returncode == 0, r.stdout


def test_light_duration_too_long_fails(tmp_path):
    key = (3, "LIGHT_OFF")
    r = run_cli(*four_runs(tmp_path, on2={"shift": {key: 2.5}}))
    assert r.returncode == 1, r.stdout
    assert "light duration" in r.stdout


def test_one_frame_short_zero_gaps_passes_with_note(tmp_path):
    r = run_cli(*four_runs(tmp_path, on1={"frames": {1: N_FRAMES - 1}}))
    assert r.returncode == 0, r.stdout
    assert "1-frame boundary difference, no gaps" in r.stdout


def test_two_frames_short_fails(tmp_path):
    r = run_cli(*four_runs(tmp_path, on1={"frames": {1: N_FRAMES - 2}}))
    assert r.returncode == 1, r.stdout
    assert any(line.startswith("- ON1 trial 1") and "frames" in line
               for line in r.stdout.splitlines())


def test_one_frame_short_with_a_gap_fails(tmp_path):
    r = run_cli(*four_runs(tmp_path, on1={"frames": {1: N_FRAMES - 1},
                                          "gap_trial": 1, "gap_ms": 50.0}))
    assert r.returncode == 1, r.stdout


def test_one_frame_short_in_off_runs_too_is_fine(tmp_path):
    # OFF runs differ by one frame among themselves: ON at either count passes
    r = run_cli(*four_runs(tmp_path, off1={"frames": {1: N_FRAMES - 1}}))
    assert r.returncode == 0, r.stdout


def test_missing_light_off_in_on_fails_with_reason(tmp_path):
    r = run_cli(*four_runs(tmp_path, on1={"drop_event": (2, "LIGHT_OFF")}))
    assert r.returncode == 1, r.stdout
    assert any(line.startswith("- ON1 trial 2") and "light duration" in line
               and "missing" in line for line in r.stdout.splitlines()), r.stdout


# ── usage / data errors (exit 2) ─────────────────────────────────────────────

def test_no_runs_is_usage_error():
    assert run_cli().returncode == 2


def test_unknown_label_is_usage_error(tmp_path):
    r = run_cli(f"--run=MAYBE={make_run(tmp_path / 'a')}")
    assert r.returncode == 2


def test_missing_dir_is_data_error(tmp_path):
    r = run_cli(f"--run=OFF={tmp_path / 'nope'}", f"--run=ON={make_run(tmp_path / 'b')}")
    assert r.returncode == 2


def test_empty_dir_is_data_error(tmp_path):
    (tmp_path / "empty").mkdir()
    r = run_cli(f"--run=OFF={tmp_path / 'empty'}", f"--run=ON={make_run(tmp_path / 'b')}")
    assert r.returncode == 2


def test_only_off_runs_is_usage_error(tmp_path):
    assert run_cli(f"--run=OFF={make_run(tmp_path / 'a')}").returncode == 2


# ── building blocks ──────────────────────────────────────────────────────────

def test_interval_stats_and_gap_count(tmp_path):
    run = bc.load_run("ON1", "ON", make_run(tmp_path / "r", gap_trial=1, gap_ms=60.0))
    t = run.trials[0]
    assert t.frames == N_FRAMES
    assert t.gap_count(37.5) == 1
    assert t.max_interval() == pytest.approx(60.0)
    s = bc.run_interval_summary(run, 37.5)
    assert s["trials"] == 3 and s["frames"] == 3 * N_FRAMES and s["gaps"] == 1
    assert s["mean"] == pytest.approx(25.0, abs=0.1)


def test_event_offsets_exclude_bme_and_start(tmp_path):
    run = bc.load_run("OFF1", "OFF", make_run(tmp_path / "r"))
    ev = run.trials[0].events
    assert set(ev) == {(label, 1) for label, _ in BASE_EVENTS}
    assert ev[("OFM_A_ON", 1)] == pytest.approx(10_000.0)


def test_repeated_label_gets_occurrence_numbers(tmp_path):
    run = bc.load_run("OFF1", "OFF", make_run(tmp_path / "r", extra_pulses=3))
    ev = run.trials[0].events
    assert {("LIGHT_PULSE_ON", n) for n in (1, 2, 3)} <= set(ev)


def test_trials_sorted_by_filename_timestamp(tmp_path):
    root = make_run(tmp_path / "r")
    # rename so alphabetical order disagrees with time order
    for p in list(root.glob("*testing_1_*")):
        p.rename(p.with_name(p.name.replace("20251030_100100", "20251030_109900")))
    run = bc.load_run("OFF1", "OFF", root)
    assert [t.tag for t in run.trials][-1] == "testing_1_Hexanol"


def test_percentile():
    assert bc.percentile([1, 2, 3, 4, 5], 50) == 3
    assert bc.percentile([0.0] * 99 + [100.0], 99) == pytest.approx(1.0)
    assert bc.percentile([], 99) == 0.0


# ── CPU CSV ──────────────────────────────────────────────────────────────────

CPU_HEADER = ("mono_ns,total_busy_pct,core0_pct,core1_pct,core2_pct,core3_pct,"
              "rig_proc_pct,stream_worker_pct,temp_c")


def write_cpu(path: Path, rows):
    path.write_text(CPU_HEADER + "\n" + "\n".join(rows) + "\n")
    return path


def test_parse_cpu_csv(tmp_path):
    p = write_cpu(tmp_path / "cpu.csv", [
        "1,20.0,10,20,30,20,55.0,0.0,60.0",
        "2,40.0,50,40,30,40,95.0,12.0,64.5",
        "3,30.0,30,30,30,30,75.0,6.0,",          # blank temp is skipped
    ])
    c = bc.parse_cpu_csv(p)
    assert c["samples"] == 3
    assert c["total_mean"] == pytest.approx(30.0)
    assert c["total_max"] == 40.0
    assert c["headroom"] == pytest.approx(70.0)
    assert c["core_max"] == {"core0": 50.0, "core1": 40.0, "core2": 30.0, "core3": 40.0}
    assert c["rig_mean"] == pytest.approx(75.0) and c["rig_max"] == 95.0
    assert c["sw_max"] == 12.0
    assert c["temp_max"] == 64.5


def test_parse_cpu_csv_rejects_other_csv(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("a,b\n1,2\n")
    with pytest.raises(bc.DataError):
        bc.parse_cpu_csv(p)


def test_cpu_shows_up_in_report(tmp_path):
    cpu = write_cpu(tmp_path / "cpu_ON1.csv", ["1,25.0,10,20,30,40,80.0,5.0,61.0"])
    r = run_cli(*four_runs(tmp_path), "--cpu", f"ON={cpu}")
    assert r.returncode == 0, r.stdout
    assert "## CPU" in r.stdout
    assert "75.0" in r.stdout          # idle headroom
    assert "61.0" in r.stdout          # max temp
    assert "| ON1 |" in r.stdout


def test_cpu_for_nonexistent_run_is_usage_error(tmp_path):
    cpu = write_cpu(tmp_path / "c.csv", ["1,25.0,10,20,30,40,80.0,5.0,61.0"])
    r = run_cli(*four_runs(tmp_path), "--cpu", f"ON={cpu}", "--cpu", f"ON={cpu}",
                "--cpu", f"ON={cpu}")
    assert r.returncode == 2
