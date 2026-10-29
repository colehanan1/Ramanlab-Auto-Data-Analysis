"""Tests for PiCode/cpu_sampler.py, the Pi-side CPU/temperature logger used by the
OFF/ON streaming bench.

The /proc parsers are pure (they take file text), so everything here runs off the rig
on fixture strings.  One end-to-end test runs the sampler for a moment against this
machine's real /proc to check the CSV it writes.
"""
from __future__ import annotations

import csv
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "PiCode" / "cpu_sampler.py"

spec = importlib.util.spec_from_file_location("cpu_sampler", SCRIPT)
cs = importlib.util.module_from_spec(spec)
sys.modules["cpu_sampler"] = cs
spec.loader.exec_module(cs)

# user nice system idle iowait irq softirq steal guest guest_nice
PROC_STAT_A = """\
cpu  1000 0 500 8000 500 0 0 0 0 0
cpu0 250 0 125 2000 125 0 0 0 0 0
cpu1 250 0 125 2000 125 0 0 0 0 0
cpu2 250 0 125 2000 125 0 0 0 0 0
cpu3 250 0 125 2000 125 0 0 0 0 0
intr 12345 1 2 3
ctxt 999
btime 1700000000
"""
# Between A and B: cpu total +500 ticks of which +300 idle/iowait (40 % busy);
# each core +125 total, +75 idle (also 40 % busy).
PROC_STAT_B = """\
cpu  1100 0 600 8260 540 0 0 0 0 0
cpu0 275 0 150 2065 135 0 0 0 0 0
cpu1 275 0 150 2065 135 0 0 0 0 0
cpu2 275 0 150 2065 135 0 0 0 0 0
cpu3 275 0 150 2065 135 0 0 0 0 0
intr 12400 1 2 3
"""


# ── /proc/stat ───────────────────────────────────────────────────────────────

def test_parse_proc_stat_splits_total_and_cores():
    st = cs.parse_proc_stat(PROC_STAT_A)
    assert list(st) == ["cpu", "cpu0", "cpu1", "cpu2", "cpu3"]
    total, idle = st["cpu"]
    assert total == 1000 + 500 + 8000 + 500
    assert idle == 8000 + 500                      # idle + iowait


def test_parse_proc_stat_does_not_double_count_guest():
    st = cs.parse_proc_stat("cpu  100 0 0 100 0 0 0 0 50 0\n")
    assert st["cpu"] == (200, 100)                 # guest is already inside user


def test_busy_pct_from_deltas():
    a, b = cs.parse_proc_stat(PROC_STAT_A), cs.parse_proc_stat(PROC_STAT_B)
    # by hand: total 10000 -> 10500, idle 8500 -> 8800, so 200 busy of 500
    assert cs.busy_pct(a["cpu"], b["cpu"]) == pytest.approx(40.0)
    per_core = cs.core_busy_pcts(a, b)
    assert len(per_core) == 4
    assert per_core == [pytest.approx(40.0)] * 4


def test_busy_pct_zero_delta_is_zero_not_error():
    a = cs.parse_proc_stat(PROC_STAT_A)
    assert cs.busy_pct(a["cpu"], a["cpu"]) == 0.0


def test_busy_pct_is_clamped():
    assert cs.busy_pct((0, 0), (100, 150)) == 0.0   # idle went up more than total
    assert cs.busy_pct((0, 0), (100, 0)) == 100.0


# ── per-process ──────────────────────────────────────────────────────────────

# pid (comm with spaces and a paren) state ppid ... utime=120 stime=30
PID_STAT = ("4242 (python3 (rig) x) S 1 4242 4242 0 -1 4194560 100 0 0 0 "
            "120 30 0 0 20 0 5 0 12345 1000000 500 18446744073709551615")


def test_parse_pid_stat_handles_parens_in_comm():
    assert cs.parse_pid_stat(PID_STAT) == 150


def test_parse_pid_stat_garbage_is_none():
    assert cs.parse_pid_stat("") is None
    assert cs.parse_pid_stat("1 (x) S 1") is None


def test_proc_cpu_pct_is_percent_of_one_core():
    # 100 ticks at 100 Hz over 1 s = one full core
    assert cs.proc_cpu_pct(100, 100, 1.0) == pytest.approx(100.0)
    assert cs.proc_cpu_pct(50, 100, 2.0) == pytest.approx(25.0)
    assert cs.proc_cpu_pct(10, 100, 0.0) == 0.0


def test_classify_cmdline_rig_worker_other():
    pat = "combinedv2_1"
    rig = b"/home/pi/myenv/bin/python\x00combinedv2_1.py\x00--odors\x00A\x00"
    worker = b"/home/pi/myenv/bin/python\x00stream_worker.py\x00--rig\x003\x00"
    other = b"/usr/bin/vim\x00notes.txt\x00"
    assert cs.classify_cmdline(rig, pat) == "rig"
    assert cs.classify_cmdline(worker, pat) == "stream_worker"
    assert cs.classify_cmdline(other, pat) is None


def test_classify_cmdline_worker_wins_over_rig_pattern():
    both = b"python\x00stream_worker.py\x00--parent\x00combinedv2_1.py\x00"
    assert cs.classify_cmdline(both, "combinedv2_1") == "stream_worker"


def test_classify_cmdline_excludes_the_sampler_itself():
    me = b"python3\x00cpu_sampler.py\x00--pid-pattern\x00combinedv2_1\x00"
    wrapper = b"bash\x00-c\x00python3 cpu_sampler.py --pid-pattern combinedv2_1\x00"
    assert cs.classify_cmdline(me, "combinedv2_1") is None
    assert cs.classify_cmdline(wrapper, "combinedv2_1") is None


def test_classify_cmdline_empty_is_none():
    assert cs.classify_cmdline(b"", "combinedv2_1") is None


# ── temperature ──────────────────────────────────────────────────────────────

def test_parse_temp_millidegrees():
    assert cs.parse_temp("54321\n") == pytest.approx(54.321)
    assert cs.parse_temp("") is None
    assert cs.parse_temp("nope") is None


# ── CSV layout ───────────────────────────────────────────────────────────────

def test_csv_header_for_4_cores():
    assert cs.csv_header(4) == [
        "mono_ns", "total_busy_pct", "core0_pct", "core1_pct", "core2_pct",
        "core3_pct", "rig_proc_pct", "stream_worker_pct", "temp_c"]


def test_format_row_blank_temp():
    row = cs.format_row(123, 12.345, [1.0, 2.0], 50.0, 7.0, None)
    assert row == ["123", "12.35", "1.00", "2.00", "50.00", "7.00", ""]
    row = cs.format_row(123, 12.345, [1.0, 2.0], 50.0, 7.0, 61.2)
    assert row[-1] == "61.2"


# ── data-dir refusal ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("path", [
    "/home/u/Documents/faststorage/Opto",
    "/home/u/Documents/FastStorage/bench",
    "/mnt/FASTSTORAGE",
    "/data/x_faststorage_y/out",
])
def test_is_data_dir_refuses_faststorage_any_case(path):
    assert cs.is_data_dir(Path(path))


def test_is_data_dir_allows_bench_dir(tmp_path):
    assert not cs.is_data_dir(tmp_path / "remote_stream_bench_cpu")


def test_is_data_dir_follows_symlinks(tmp_path):
    real = tmp_path / "faststorage"
    real.mkdir()
    link = tmp_path / "innocent"
    link.symlink_to(real)
    assert cs.is_data_dir(link / "sub")


def test_main_refuses_data_dir_with_exit_2(tmp_path):
    target = tmp_path / "faststorage" / "out"
    r = subprocess.run([sys.executable, str(SCRIPT), "--out-dir", str(target),
                        "--label", "ON1", "--duration", "1"],
                       capture_output=True, text=True)
    assert r.returncode == 2
    assert "faststorage" in r.stderr.lower()
    assert not target.exists()                      # refused before creating anything


def test_main_rejects_bad_label(tmp_path):
    r = subprocess.run([sys.executable, str(SCRIPT), "--out-dir", str(tmp_path),
                        "--label", "../evil", "--duration", "1"],
                       capture_output=True, text=True)
    assert r.returncode == 2


# ── end to end on this machine's /proc ───────────────────────────────────────

@pytest.mark.skipif(not Path("/proc/stat").exists(), reason="needs /proc")
def test_sampler_writes_csv_end_to_end(tmp_path):
    out = tmp_path / "cpu"
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--out-dir", str(out), "--label", "T1",
         "--pid-pattern", "definitely-not-running-xyz", "--interval", "0.2",
         "--duration", "1.0"],
        capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    files = list(out.glob("cpu_T1_*.csv"))
    assert len(files) == 1
    with open(files[0], newline="") as fh:
        rows = list(csv.reader(fh))
    header, data = rows[0], rows[1:]
    assert header[:2] == ["mono_ns", "total_busy_pct"]
    assert header[-3:] == ["rig_proc_pct", "stream_worker_pct", "temp_c"]
    assert len(data) >= 3
    assert all(len(d) == len(header) for d in data)
    assert all(0.0 <= float(d[1]) <= 100.0 for d in data)
    assert all(float(d[header.index("rig_proc_pct")]) == 0.0 for d in data)
    monos = [int(d[0]) for d in data]
    assert monos == sorted(monos)
