"""Tests for scripts/tools/remux_from_csv.py, the .h264 -> .mp4 batch transcoder.

The rigs launch this script over ssh after every push of a recording session, and a
second launch can land on the same folder while the first is still encoding. Before
the fix both runs listed the same .h264 files and encoded the same ``<stem>.mp4`` at
once; the ``+faststart`` rewrites interleaved and left ``ftyp | moov | moov | free |
mdat`` files that decode 0 frames (25 real recordings, always the tail of the sorted
list).

The script is run as a subprocess, the way the rigs run it. Most tests put a stub
``ffmpeg`` first on PATH (it logs start/end events, sleeps, then writes the output)
so that overlap is observable and deterministic. One test runs the real ffmpeg end to
end and walks the MP4 boxes of the result.

Every run gets a throw-away HOME and REMUX_LOCK_DIR, so the suite never touches the
real ``~/.cache/remux_from_csv``.

Set REMUX_SCRIPT_UNDER_TEST to point the suite at another copy of the script (for
example the pristine one in ~/bin) to see which tests that copy fails.
"""
from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import os
import re
import shutil
import signal
import struct
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = Path(
    os.environ.get("REMUX_SCRIPT_UNDER_TEST", REPO / "scripts" / "tools" / "remux_from_csv.py")
)

# Exactly what PiCode/combinedv2_1*.py passes after --input-dir.
CLI_ARGS = ["--crf", "22", "--preset", "slow", "--faststart", "--scale", "1.0", "--fps-cap", "50.0"]

N_FRAMES = 40
FRAME_STEP_NS = 25_000_000  # 25 ms apart -> 40 fps
BASE_NS = 1_700_000_000_000_000_000


# --------------------------------------------------------------------------
# Stub ffmpeg
# --------------------------------------------------------------------------

STUB_SOURCE = r'''#!__PYTHON__
"""Test double for ffmpeg (see tests/test_remux_from_csv.py).

Appends "start"/"end" events to $STUB_FFMPEG_LOG, sleeps, then writes bytes to its last
argument (the output path), the way the real encoder fills the file it was handed.

  STUB_FFMPEG_SLEEP             seconds to "encode" (default 1.0)
  STUB_FFMPEG_MODE              ok (4096 bytes, exit 0) | fail (100 bytes, exit 1) | tiny (10 bytes, exit 0)
  STUB_FFMPEG_EARLY_CREATE      create the (empty) output at start, like the real muxer does
  STUB_FFMPEG_REMOVE_ON_START   delete this path at start (simulates someone else taking a source)
"""
import json, os, sys, time

argv = sys.argv[1:]
out = argv[-1]
inp = argv[argv.index("-i") + 1] if "-i" in argv else ""
sleep_s = float(os.environ.get("STUB_FFMPEG_SLEEP", "1.0"))
mode = os.environ.get("STUB_FFMPEG_MODE", "ok")


def emit(kind):
    fields = [kind, out, repr(time.time()), inp, str(os.getpid()), json.dumps(argv)]
    with open(os.environ["STUB_FFMPEG_LOG"], "a") as fh:
        fh.write("\t".join(fields) + "\n")


emit("start")
if os.environ.get("STUB_FFMPEG_EARLY_CREATE"):
    open(out, "wb").close()
victim = os.environ.get("STUB_FFMPEG_REMOVE_ON_START")
if victim and os.path.exists(victim):
    os.unlink(victim)
if not os.path.exists(inp):  # like the real thing: no input, no encode
    emit("end")
    sys.stderr.write(inp + ": No such file or directory\n")
    sys.exit(1)
time.sleep(sleep_s)
with open(out, "wb") as fh:
    fh.write(b"\0" * {"ok": 4096, "fail": 100, "tiny": 10}[mode])
emit("end")
if mode == "fail":
    sys.stderr.write("stub ffmpeg: simulated encoder failure\n")
    sys.exit(1)
'''


@dataclass
class Event:
    kind: str
    output: str
    t: float
    input: str
    pid: int
    argv: list


@dataclass
class Encode:
    input: str
    output: str
    pid: int
    start: float
    end: Optional[float]
    argv: list

    def overlaps(self, other: "Encode") -> bool:
        inf = float("inf")
        a_end = inf if self.end is None else self.end
        b_end = inf if other.end is None else other.end
        return self.start < b_end and other.start < a_end


def describe(encodes) -> str:
    if not encodes:
        return "  (no encodes)"
    t0 = min(e.start for e in encodes)
    rows = []
    for e in sorted(encodes, key=lambda e: e.start):
        end = "   ?  " if e.end is None else f"{e.end - t0:6.2f}"
        rows.append(f"  {Path(e.input).name:<14} pid {e.pid:<7} {e.start - t0:6.2f}s -> {end}s")
    return "\n".join(rows)


# --------------------------------------------------------------------------
# Test rig: environment, launching runs, reading the stub's event log
# --------------------------------------------------------------------------

@dataclass
class Run:
    proc: subprocess.Popen
    out_path: Path

    def output(self) -> str:
        return self.out_path.read_text(encoding="utf-8", errors="replace")

    def wait(self, timeout: float = 90.0):
        try:
            self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.proc.wait()
            pytest.fail(f"remux run timed out after {timeout}s; output so far:\n{self.output()}")
        return self.proc.returncode, self.output()


@dataclass
class Rig:
    root: Path
    path_prefix: str
    env: dict = field(default_factory=dict)
    runs: list = field(default_factory=list)

    def __post_init__(self):
        self.log = self.root / "ffmpeg_events.log"
        (self.root / "home").mkdir(exist_ok=True)
        self.lock_dir = self.root / "locks"
        env = {k: v for k, v in os.environ.items() if not k.startswith(("STUB_FFMPEG", "REMUX_"))}
        env.update(
            HOME=str(self.root / "home"),  # never touch the real ~/.cache
            REMUX_LOCK_DIR=str(self.lock_dir),
            PATH=os.pathsep.join([self.path_prefix, "/usr/bin", "/bin"]),
            PYTHONIOENCODING="utf-8",
            STUB_FFMPEG_LOG=str(self.log),
            STUB_FFMPEG_SLEEP="0",
        )
        self.env = env

    # -- configuration ------------------------------------------------------
    def stub(self, sleep=None, mode=None, early_create=False, remove_on_start=None):
        if sleep is not None:
            self.env["STUB_FFMPEG_SLEEP"] = str(sleep)
        if mode is not None:
            self.env["STUB_FFMPEG_MODE"] = mode
        if early_create:
            self.env["STUB_FFMPEG_EARLY_CREATE"] = "1"
        if remove_on_start is not None:
            self.env["STUB_FFMPEG_REMOVE_ON_START"] = str(remove_on_start)

    # -- running ------------------------------------------------------------
    def launch(self, folder: Path, tag: str, env: Optional[dict] = None, input_dir=None) -> Run:
        cmd = [sys.executable, str(SCRIPT), "--input-dir", str(input_dir or folder), *CLI_ARGS]
        out_path = self.root / f"{tag}.out"
        with out_path.open("wb") as fh:
            proc = subprocess.Popen(
                cmd, env={**self.env, **(env or {})}, stdin=subprocess.DEVNULL,
                stdout=fh, stderr=subprocess.STDOUT, start_new_session=True,
            )
        run = Run(proc, out_path)
        self.runs.append(run)
        return run

    def close(self):
        for run in self.runs:  # also reaps an ffmpeg orphaned by a killed runner
            try:
                os.killpg(run.proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                run.proc.wait(timeout=10)
            except Exception:
                pass

    # -- the stub's event log -----------------------------------------------
    def events(self):
        if not self.log.exists():
            return []
        events = []
        for line in self.log.read_text().splitlines():
            parts = line.split("\t")
            if len(parts) != 6:
                continue
            kind, output, t, inp, pid, argv = parts
            events.append(Event(kind, output, float(t), inp, int(pid), json.loads(argv)))
        return events

    def encodes(self):
        open_, done = {}, []
        for ev in self.events():
            if ev.kind == "start":
                open_[ev.pid] = Encode(ev.input, ev.output, ev.pid, ev.t, None, ev.argv)
            elif ev.pid in open_:
                enc = open_.pop(ev.pid)
                enc.end = ev.t
                done.append(enc)
        return sorted(done + list(open_.values()), key=lambda e: e.start)

    def wait_for_encode_start(self, count: int = 1, timeout: float = 20.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if sum(e.kind == "start" for e in self.events()) >= count:
                return
            time.sleep(0.02)
        pytest.fail(f"stub ffmpeg never started {count} encode(s) within {timeout}s")


@pytest.fixture
def rig(tmp_path):
    stub_dir = tmp_path / "stubbin"
    stub_dir.mkdir()
    stub = stub_dir / "ffmpeg"
    stub.write_text(STUB_SOURCE.replace("__PYTHON__", sys.executable))
    stub.chmod(0o755)
    r = Rig(tmp_path, path_prefix=str(stub_dir))
    yield r
    r.close()


# --------------------------------------------------------------------------
# Fixture data
# --------------------------------------------------------------------------

def write_frame_csv(path: Path, n_rows: int = N_FRAMES, label: str = "off") -> None:
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["SensorNs", "ActiveOFM"])
        for k in range(n_rows):
            w.writerow([BASE_NS + k * FRAME_STEP_NS, label])


def make_session(folder: Path, stems, csv_pattern: str = "{stem}.csv") -> None:
    """Fake .h264 payloads (the stub never decodes them) plus real-looking frame CSVs."""
    folder.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        (folder / f"{stem}.h264").write_bytes(b"\x00\x00\x00\x01 not really h264")
        write_frame_csv(folder / csv_pattern.format(stem=stem))


def tree(folder: Path):
    return sorted(str(p.relative_to(folder)) for p in folder.rglob("*"))


def expected_lock_path(lock_dir: Path, input_dir: Path) -> Path:
    digest = hashlib.sha1(str(input_dir.resolve()).encode()).hexdigest()
    return lock_dir / f"{digest}.lock"


def lock_is_held(lock_path: Path) -> bool:
    """True if some other open file description holds an flock on lock_path."""
    if not lock_path.exists():
        return False
    fd = os.open(lock_path, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


# --------------------------------------------------------------------------
# Single-run behaviour (pins what the script did before the fix)
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "layout, csv_pattern",
    [
        ("flat", "{stem}.csv"),
        ("flat", "{stem}_frames.csv"),  # the script accepts either CSV name
        ("nested", "{stem}.csv"),  # rglob: sessions sit in subfolders too
    ],
)
def test_single_run_converts_and_deletes_the_source(rig, tmp_path, layout, csv_pattern):
    folder = tmp_path / "batch"
    where = folder / "fly_1" if layout == "nested" else folder
    make_session(where, ["output_1"], csv_pattern)

    rc, out = rig.launch(folder, "run").wait()

    assert rc == 0, out
    assert (where / "output_1.mp4").stat().st_size >= 1024
    assert not (where / "output_1.h264").exists()
    assert "🔄 Transcoding output_1.h264" in out
    assert "✅ output_1.mp4 (input FPS 40.000000) — deleted source output_1.h264" in out


def test_h264_without_a_csv_is_skipped_and_kept(rig, tmp_path):
    folder = tmp_path / "batch"
    folder.mkdir()
    (folder / "orphan.h264").write_bytes(b"\x00\x00\x00\x01 no csv for me")

    rc, out = rig.launch(folder, "run").wait()

    assert rc == 0, out
    # With neither <stem>.csv nor <stem>_frames.csv present, the script reports the latter.
    assert "[SKIP] orphan.h264 → no orphan_frames.csv" in out
    assert (folder / "orphan.h264").exists()
    assert not (folder / "orphan.mp4").exists()
    assert rig.encodes() == [], "ffmpeg must not be started for a file with no CSV"


def test_ffmpeg_arguments_are_unchanged_apart_from_container_and_output(rig, tmp_path):
    """The fix may add ``-f mp4`` and change the output path; nothing else about the encode."""
    folder = tmp_path / "batch"
    make_session(folder, ["output_1"])

    rc, out = rig.launch(folder, "run").wait()

    assert rc == 0, out
    (enc,) = rig.encodes()
    argv = list(enc.argv)
    assert argv[-1] == enc.output
    argv = argv[:-1]  # the output path
    if argv[-2:] == ["-f", "mp4"]:  # container flag the fix adds
        argv = argv[:-2]
    assert argv == [
        "-y", "-f", "h264", "-r", "40.000000", "-i", str(folder / "output_1.h264"),
        "-vf", "unsharp=5:5:2:5:5:2,eq=1.5:brightness=0.25,fps=40.000000",
        "-c:v", "libx264", "-preset", "slow", "-profile:v", "high", "-level:v", "4.1",
        "-pix_fmt", "yuv420p", "-g", "80", "-bf", "3", "-b_strategy", "1",
        "-movflags", "+faststart", "-crf", "22",
    ]


# --------------------------------------------------------------------------
# Odor label burned into the video
# --------------------------------------------------------------------------
# The CSV only says which valve opened (ActiveOFM = OFM_A); what was in that valve
# changes whenever vials are swapped. The rig's per-trial .txt says what was sent.

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def write_trial(folder: Path, stem: str, valves, txt_odor_line=None) -> None:
    """One trial: fake .h264, a frame CSV opening ``valves`` in turn mid-trial (10 rows
    each, off before and after), and the rig's per-trial .txt if ``txt_odor_line`` is given."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{stem}.h264").write_bytes(b"\x00\x00\x00\x01 not really h264")
    labels = ["off"] * 10 + [v for v in valves for _ in range(10)]
    labels += ["off"] * (N_FRAMES - len(labels))
    with (folder / f"{stem}.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["SensorNs", "ActiveOFM"])
        for k, label in enumerate(labels):
            w.writerow([BASE_NS + k * FRAME_STEP_NS, label])
    if txt_odor_line is not None:
        (folder / f"{stem}.txt").write_text(
            "Video Metadata\n==============\nCycle Name: testing_1_IsoamylAcetate-1\n"
            "\nOdor Vials\n----------\n"
            f"{txt_odor_line}"
            "All vial concentrations: IsoamylAcetate-1=1, Hexanol-0.1=0.1\n",
            encoding="utf-8")


def burned_labels(rig):
    """[(text, box width)] for each odor segment in the single encode's -vf."""
    (enc,) = rig.encodes()
    vf = enc.argv[enc.argv.index("-vf") + 1]
    widths = [int(w) for w in re.findall(r"drawbox=x=10:y=10:width=(\d+):height=50:", vf)]
    texts = re.findall(r"drawtext=.*?:text='((?:[^'\\]|\\.)*)'", vf)
    assert len(widths) == len(texts), vf
    return list(zip(texts, widths))


def test_label_is_the_trial_odor_from_the_rigs_txt(rig, tmp_path):
    folder = tmp_path / "batch"
    write_trial(folder, "output_1", ["OFM_A"], "Trial odor: IsoamylAcetate-1\n")

    rc, out = rig.launch(folder, "run").wait()

    assert rc == 0, out
    assert [t for t, _ in burned_labels(rig)] == ["IsoamylAcetate-1"]
    assert "odor label: IsoamylAcetate-1 (from output_1.txt)" in out


@pytest.mark.parametrize("txt_odor_line", [None, "", "Trial odor: N/A\n"],
                         ids=["no-txt", "no-trial-odor-line", "trial-odor-NA"])
def test_label_falls_back_to_the_valve_name_without_a_trial_odor(rig, tmp_path, txt_odor_line):
    folder = tmp_path / "batch"
    write_trial(folder, "output_1", ["OFM_H"], txt_odor_line)

    rc, out = rig.launch(folder, "run").wait()

    assert rc == 0, out
    assert [t for t, _ in burned_labels(rig)] == ["Hexanol"]
    assert "odor label: valve names (no Trial odor in output_1.txt)" in out


def test_trial_odor_is_not_used_when_two_valves_open(rig, tmp_path):
    """One name for the trial can't say which of two valves held it."""
    folder = tmp_path / "batch"
    write_trial(folder, "output_1", ["OFM_A", "OFM_B"], "Trial odor: IsoamylAcetate-1\n")

    rc, out = rig.launch(folder, "run").wait()

    assert rc == 0, out
    assert [t for t, _ in burned_labels(rig)] == ["ACV", "Benzaldehyde"]


@pytest.mark.parametrize("valve, txt_odor_line", [
    ("OFM_A", "Trial odor: IsoamylAcetate-1\n"),
    ("OFM_C", "Trial odor: EthylButyrate-0.1\n"),
    ("OFM_E", None),  # valve name "Ethyl Butyrate"
    ("OFM_A", None),  # valve name "ACV"
], ids=["IsoamylAcetate-1", "EthylButyrate-0.1", "Ethyl-Butyrate", "ACV"])
def test_white_box_is_long_enough_for_the_label(rig, tmp_path, valve, txt_odor_line):
    ImageFont = pytest.importorskip("PIL.ImageFont")
    folder = tmp_path / "batch"
    write_trial(folder, "output_1", [valve], txt_odor_line)

    rc, out = rig.launch(folder, "run").wait()

    assert rc == 0, out
    ((text, width),) = burned_labels(rig)
    text_right = 15 + ImageFont.truetype(FONT, 20).getlength(text)  # drawtext x=15
    assert 10 + width >= text_right + 5, f"box ends at x={10 + width}, text at x={text_right:.0f}"
    # odor_box_alignment.py samples x 20..100 inside the box; short labels keep that.
    assert width >= 100


# --------------------------------------------------------------------------
# Two overlapping runs on one folder (the bug)
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def overlapping_runs(tmp_path_factory):
    """Run A starts; once A is inside its first encode, run B starts ~0.3 s later.

    Three inputs, stub ffmpeg sleeping 1 s per encode. Shared by the tests that read
    different things off the same scenario.
    """
    root = tmp_path_factory.mktemp("overlap")
    stub_dir = root / "stubbin"
    stub_dir.mkdir()
    stub = stub_dir / "ffmpeg"
    stub.write_text(STUB_SOURCE.replace("__PYTHON__", sys.executable))
    stub.chmod(0o755)
    rig = Rig(root, path_prefix=str(stub_dir))
    rig.stub(sleep=1.0)
    stems = ["output_1", "output_2", "output_3"]
    folder = root / "batch"
    make_session(folder, stems)
    try:
        run_a = rig.launch(folder, "a")
        rig.wait_for_encode_start(1)  # A now holds the folder and is encoding
        time.sleep(0.3)
        run_b = rig.launch(folder, "b")
        rc_a, out_a = run_a.wait()
        rc_b, out_b = run_b.wait()
        yield {
            "folder": folder, "stems": stems, "encodes": rig.encodes(),
            "rc_a": rc_a, "out_a": out_a, "rc_b": rc_b, "out_b": out_b,
        }
    finally:
        rig.close()


def test_concurrent_runs_never_overlap_an_encode(overlapping_runs):
    o = overlapping_runs
    encodes = o["encodes"]

    clashes = [(a, b) for i, a in enumerate(encodes) for b in encodes[i + 1:]
               if a.input == b.input and a.overlaps(b)]
    assert not clashes, "two encodes of the same .h264 overlapped in time:\n" + describe(encodes)

    per_input = Counter(Path(e.input).name for e in encodes)
    assert per_input == {f"{s}.h264": 1 for s in o["stems"]}, (
        "each input must be encoded exactly once in total:\n" + describe(encodes)
    )

    assert (o["rc_a"], o["rc_b"]) == (0, 0), (o["out_a"], o["out_b"])
    for stem in o["stems"]:
        assert (o["folder"] / f"{stem}.mp4").stat().st_size >= 1024
        assert not (o["folder"] / f"{stem}.h264").exists()


def test_startup_line_reports_whether_the_run_waited(overlapping_runs):
    o = overlapping_runs
    lock_lines_a = re.findall(r"^\[LOCK\].*$", o["out_a"], re.M)
    lock_lines_b = re.findall(r"^\[LOCK\].*$", o["out_b"], re.M)

    assert len(lock_lines_a) == 1, o["out_a"]
    assert re.search(r"no other run", lock_lines_a[0]), lock_lines_a[0]
    assert len(lock_lines_b) == 1, o["out_b"]
    assert re.search(r"another run.*waiting", lock_lines_b[0]), lock_lines_b[0]


def test_second_run_scans_only_after_it_has_the_lock(overlapping_runs):
    """B queued behind A. If B listed the folder first it would hold stale names."""
    out_b = overlapping_runs["out_b"]

    assert "[INFO] No .h264 files in" in out_b, out_b
    assert "Transcoding" not in out_b and "[SKIP]" not in out_b, out_b


# --------------------------------------------------------------------------
# Real ffmpeg, end to end
# --------------------------------------------------------------------------

def _find_real_ffmpeg():
    """(ffmpeg, ffprobe) of a build with libx264, preferring /usr/bin (the Pis' ssh PATH)."""
    candidates = [Path("/usr/bin/ffmpeg")]
    found = shutil.which("ffmpeg")
    if found:
        candidates.append(Path(found))
    for ffmpeg in candidates:
        ffprobe = ffmpeg.with_name("ffprobe")
        if not (ffmpeg.is_file() and ffprobe.is_file()):
            continue
        try:
            encoders = subprocess.run([str(ffmpeg), "-hide_banner", "-encoders"],
                                      capture_output=True, text=True, timeout=30).stdout
        except (OSError, subprocess.SubprocessError):
            continue
        if "libx264" in encoders:
            return ffmpeg, ffprobe
    return None


def mp4_top_level_boxes(path: Path):
    """Walk the top-level boxes. Returns (box types, problem or None)."""
    data = path.read_bytes()
    boxes, off = [], 0
    while off < len(data):
        if off + 8 > len(data):
            return boxes, f"{len(data) - off} stray bytes at offset {off}"
        size, kind = struct.unpack(">I4s", data[off:off + 8])
        header = 8
        if size == 1:  # 64-bit size follows
            if off + 16 > len(data):
                return boxes, f"truncated largesize header at offset {off}"
            size, header = struct.unpack(">Q", data[off + 8:off + 16])[0], 16
        elif size == 0:  # box runs to end of file
            size = len(data) - off
        if size < header or off + size > len(data):
            return boxes, f"malformed box {kind!r} (size {size}) at offset {off}"
        boxes.append(kind.decode("latin-1"))
        off += size
    return boxes, None


def count_frames(ffprobe: Path, path: Path) -> Optional[int]:
    res = subprocess.run(
        [str(ffprobe), "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, timeout=60,
    )
    first = res.stdout.strip().splitlines()[:1]
    return int(first[0]) if first and first[0].isdigit() else None


def test_concurrent_real_ffmpeg_runs_leave_exactly_one_moov_per_file(tmp_path):
    tools = _find_real_ffmpeg()
    if tools is None:
        pytest.skip("no ffmpeg with libx264 on this machine")
    ffmpeg, ffprobe = tools

    stems = ["output_1", "output_2", "output_3"]
    folder = tmp_path / "batch"
    folder.mkdir()
    for stem in stems:
        subprocess.run(
            [str(ffmpeg), "-hide_banner", "-loglevel", "error", "-y",
             "-f", "lavfi", "-i", "testsrc=size=64x64:rate=40", "-t", "1",
             "-c:v", "libx264", "-f", "h264", str(folder / f"{stem}.h264")],
            check=True, timeout=60,
        )
        write_frame_csv(folder / f"{stem}.csv")

    rig = Rig(tmp_path, path_prefix=str(ffmpeg.parent))
    try:
        run_a = rig.launch(folder, "a")
        time.sleep(0.3)
        run_b = rig.launch(folder, "b")
        rc_a, out_a = run_a.wait()
        rc_b, out_b = run_b.wait()
    finally:
        rig.close()

    assert (rc_a, rc_b) == (0, 0), (out_a, out_b)
    for stem in stems:
        mp4 = folder / f"{stem}.mp4"
        assert mp4.exists(), f"{mp4.name} missing\n--- A ---\n{out_a}\n--- B ---\n{out_b}"
        boxes, problem = mp4_top_level_boxes(mp4)
        assert boxes.count("moov") == 1, f"{mp4.name}: top-level boxes {boxes} ({problem})"
        assert boxes[0] == "ftyp" and boxes.count("mdat") == 1, f"{mp4.name}: {boxes}"
        assert boxes.index("moov") < boxes.index("mdat"), f"{mp4.name}: faststart lost: {boxes}"
        assert problem is None, f"{mp4.name}: {problem} (boxes {boxes})"
        assert count_frames(ffprobe, mp4) == N_FRAMES, f"{mp4.name}: wrong frame count"
        assert not (folder / f"{stem}.h264").exists()
    assert not [n for n in tree(folder) if n.endswith(".part")], tree(folder)


# --------------------------------------------------------------------------
# Atomic output
# --------------------------------------------------------------------------

def test_encode_goes_to_a_hidden_part_file_and_is_published_atomically(rig, tmp_path):
    folder = tmp_path / "batch"
    make_session(folder, ["output_1"])
    rig.stub(sleep=0.8, early_create=True)  # the real muxer creates its output immediately
    final, part = folder / "output_1.mp4", folder / ".output_1.mp4.part"

    run = rig.launch(folder, "run")
    final_visible_mid_encode = part_seen = False
    while run.proc.poll() is None:
        final_now = final.exists()  # sample the file first, then the log
        ended = any(e.kind == "end" for e in rig.events())
        final_visible_mid_encode |= final_now and not ended
        part_seen |= part.exists()
        time.sleep(0.01)
    rc, out = run.wait()

    assert rc == 0, out
    (enc,) = rig.encodes()
    assert enc.output == str(part), "ffmpeg must write to the hidden temp in the input's folder"
    assert enc.argv[-3:-1] == ["-f", "mp4"], "the .part extension needs an explicit container"
    assert part_seen, "the hidden temp was never seen during the encode"
    assert not final_visible_mid_encode, "the final .mp4 name showed up before the encode finished"
    assert final.stat().st_size == 4096
    assert tree(folder) == ["output_1.csv", "output_1.mp4"]


@pytest.mark.parametrize("mode", ["fail", "tiny"])
def test_failed_encode_keeps_the_source_and_leaves_no_files(rig, tmp_path, mode):
    """A crashed ffmpeg, or one that leaves < --delete-min-bytes, must leave nothing behind."""
    folder = tmp_path / "batch"
    make_session(folder, ["output_1"])
    rig.stub(mode=mode)

    rc, out = rig.launch(folder, "run").wait()

    assert rc == 0, out
    assert "❌ Failed: output_1.h264 (input FPS 40.000000)" in out
    assert "deleted source" not in out
    assert tree(folder) == ["output_1.csv", "output_1.h264"]


# --------------------------------------------------------------------------
# The lock itself
# --------------------------------------------------------------------------

@pytest.mark.parametrize("override", [True, False], ids=["REMUX_LOCK_DIR", "default-~/.cache"])
def test_lock_file_is_outside_the_data_folder_and_named_by_sha1(rig, tmp_path, override):
    real = tmp_path / "real_batch"
    make_session(real, ["output_1"])
    link = tmp_path / "link_to_batch"
    link.symlink_to(real, target_is_directory=True)  # the key is the *resolved* path
    rig.stub(sleep=1.0)
    if override:
        lock_dir = rig.lock_dir
    else:  # no override: the lock lives under $HOME/.cache/remux_from_csv (HOME is a temp dir)
        del rig.env["REMUX_LOCK_DIR"]
        lock_dir = tmp_path / "home" / ".cache" / "remux_from_csv"
    lock_path = expected_lock_path(lock_dir, real)

    run = rig.launch(real, "run", input_dir=link)
    rig.wait_for_encode_start(1)
    held_while_encoding = lock_is_held(lock_path)
    rc, out = run.wait()

    assert rc == 0, out
    found = tree(lock_dir) if lock_dir.exists() else "no lock dir"
    assert lock_path.exists(), f"expected lock file {lock_path}; found {found}"
    assert held_while_encoding, "the lock file was not flock()ed while the encode ran"
    assert not lock_is_held(lock_path), "the lock must be released when the run exits"
    assert tree(real) == ["output_1.csv", "output_1.mp4"], "no lock or temp files in the data folder"


# --------------------------------------------------------------------------
# A source that is gone by the time its turn comes
# --------------------------------------------------------------------------

def test_source_deleted_before_its_turn_is_skipped(rig, tmp_path):
    folder = tmp_path / "batch"
    make_session(folder, ["output_1", "output_2", "output_3"])
    # While output_1 encodes, "someone else" takes output_2 away.
    rig.stub(sleep=0.2, remove_on_start=folder / "output_2.h264")

    rc, out = rig.launch(folder, "run").wait()

    assert rc == 0, out
    encoded = sorted(Path(e.input).name for e in rig.encodes())
    assert encoded == ["output_1.h264", "output_3.h264"], (
        "a .h264 that no longer exists must not be encoded:\n" + describe(rig.encodes())
    )
    assert "[SKIP] output_2.h264" in out, out
    assert (folder / "output_1.mp4").exists() and (folder / "output_3.mp4").exists()
    assert not (folder / "output_2.mp4").exists()


# --------------------------------------------------------------------------
# A killed runner must not free the folder while its ffmpeg is still writing
# --------------------------------------------------------------------------

def test_lock_is_held_while_an_orphaned_ffmpeg_is_still_encoding(rig, tmp_path):
    """kill -9 the python runner mid-encode (e.g. `pkill -f remux_from_csv`) and relaunch.

    The ffmpeg child survives, still writing the temp file. A second run that takes the
    folder at once would start a second ffmpeg on the same temp, and then publish the
    garbage and delete the only copy of the recording.
    """
    folder = tmp_path / "batch"
    make_session(folder, ["output_1"])
    rig.stub(sleep=1.5)

    run_a = rig.launch(folder, "a")
    rig.wait_for_encode_start(1)
    os.kill(run_a.proc.pid, signal.SIGKILL)  # the runner only; not its process group
    run_a.proc.wait(timeout=10)
    run_b = rig.launch(folder, "b")
    rc_b, out_b = run_b.wait()

    encodes = rig.encodes()
    assert rc_b == 0, out_b
    assert len(encodes) == 2, "expected the orphaned encode plus run B's:\n" + describe(encodes)
    assert not encodes[0].overlaps(encodes[1]), (
        "run B started encoding while the orphaned ffmpeg was still writing:\n" + describe(encodes)
    )
    assert tree(folder) == ["output_1.csv", "output_1.mp4"]
