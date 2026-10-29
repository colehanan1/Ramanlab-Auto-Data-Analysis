#!/usr/bin/env python3
"""Compare interleaved OFF/ON bench runs of a rig: does the live stream + telemetry
(ON) perturb recording or trial-event timing relative to the plain rig (OFF)?

Each ``--run LABEL=<batch_dir>`` is a batch folder the rig wrote.  Per trial the rig
writes a frames CSV ``output_<...>_<YYYYmmdd_HHMMSS>.csv`` (SensorNs = camera sensor
timestamp, monotonic ns) and a sensors CSV ``sensors_<same base>.csv`` (MonoNs +
OFM_Event label).  From them we compute

* frame intervals (SensorNs diffs): mean / p99 / max, and the number of gaps longer
  than 1.5 frame intervals (37.5 ms at 40 fps);
* event offsets: MonoNs of every non-BME event row minus the trial's RECORDING_START
  row, keyed by (trial index, event label, occurrence number) - informational only,
  because RECORDING_START is logged by the camera thread and so includes the camera
  start offset;
* camera start offset: first odor-on event (``OFM_*_ON``; else the first non-BME,
  non-RECORDING_START event) minus RECORDING_START.  Recorded and reported per trial,
  never judged (accepted 2026-10-29);
* stimulus intervals that do not depend on the camera: odor duration per channel
  (OFM_x_OFF - OFM_x_ON), light duration (LIGHT_OFF - first light-on event) and the
  light-on minus odor-on lag, paired by occurrence.

Verdict (the hard-stop criteria of the bench protocol):

* a stimulus interval FAILs when |ON - mean(OFF)| > OFF noise floor + tolerance, where
  the floor is the largest |OFF_i - OFF_j| of that interval; an interval the OFF runs
  have but a run lacks is a FAIL with a reason;
* a trial's frames FAIL when an ON run has more gaps than the worst OFF run, fewer
  than (shortest OFF run - 1) frames, or a longer worst gap than every OFF run by more
  than one frame interval (a single 200 ms stall must not hide behind an OFF run
  that also had one 40 ms gap).  Exactly one frame short of the shortest OFF run with
  no gap > 1.5 frame intervals in the run is window-boundary quantization (also seen
  in OFF runs): noted, not failed;
* missing or mismatched trials / event keys are always reported as FAIL.

CPU CSVs from PiCode/cpu_sampler.py are summarised but never decide the verdict.

Exit status: 0 PASS, 1 FAIL, 2 usage or data error.  Stdlib only.

Example:
    python3 scripts/tools/bench_compare.py \\
        --run OFF=~/b/off1 --run ON=~/b/on1 --run OFF=~/b/off2 --run ON=~/b/on2 \\
        --cpu ON=~/b/cpu_ON1.csv --out report.md
"""
from __future__ import annotations

import argparse
import csv
import re
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

EPS = 1e-9
GROUPS = ("OFF", "ON")
START_LABEL = "RECORDING_START"
ODOR_ON_RE = re.compile(r"^OFM_(.+)_ON$")
ODOR_OFF_RE = re.compile(r"^OFM_(.+)_OFF$")
LIGHT_ON_LABELS = ("LIGHT_SOLID_ON", "LIGHT_ON", "LIGHT_PULSE_ON", "LIGHT_RAMP_START")
LIGHT_OFF_LABEL = "LIGHT_OFF"
TS_RE = re.compile(r"(\d{8})_(\d{6})")
TAG_RE = re.compile(r"_batch_([^_]+)_(.*)_(\d{8})_(\d{6})$")


class DataError(Exception):
    """Unusable input (exit status 2)."""


# ── data model ───────────────────────────────────────────────────────────────

@dataclass
class Trial:
    index: int                      # 1-based position in the run, by filename time
    stem: str
    tag: str | None                 # phase tag, e.g. "testing_1_Hexanol"
    frames: int
    intervals_ms: list[float]
    events: dict[tuple[str, int], float]   # (label, occurrence) -> offset ms
    camera_offset_ms: float | None = None  # first odor-on minus RECORDING_START
    camera_event: str | None = None
    problems: list[str] = field(default_factory=list)

    def gap_count(self, thresh_ms: float) -> int:
        return sum(1 for v in self.intervals_ms if v > thresh_ms)

    def max_interval(self) -> float:
        return max(self.intervals_ms) if self.intervals_ms else 0.0


@dataclass
class Run:
    name: str                       # OFF1, ON1, ...
    group: str                      # OFF | ON
    path: Path
    trials: list[Trial]
    problems: list[str] = field(default_factory=list)
    cpu: dict | None = None


# ── CSV loading ──────────────────────────────────────────────────────────────

def _stamp(name: str) -> str:
    """Sortable YYYYmmdd_HHMMSS of a CSV file name ('' when it has none)."""
    found = TS_RE.findall(name)
    return "_".join(found[-1]) if found else ""


def _tag_of(stem: str) -> str | None:
    m = TAG_RE.search(stem)
    return m.group(2) if m else None


def _to_float(text: str) -> float | None:
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def read_frame_intervals_ms(path: Path) -> tuple[int, list[float]]:
    """(frame count, SensorNs diffs in ms) of one frames CSV."""
    stamps: list[float] = []
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            rd = csv.DictReader(fh)
            if not rd.fieldnames or "SensorNs" not in rd.fieldnames:
                raise DataError(f"{path}: no SensorNs column")
            for row in rd:
                v = _to_float(row.get("SensorNs", ""))
                if v is not None:
                    stamps.append(v)
    except OSError as exc:
        raise DataError(f"{path}: {exc}") from exc
    return len(stamps), [(b - a) / 1e6 for a, b in zip(stamps, stamps[1:])]


def read_event_offsets_ms(path: Path, ignore: re.Pattern | None
                          ) -> tuple[dict[tuple[str, int], float],
                                     tuple[float | None, str | None], list[str]]:
    """Event offsets from RECORDING_START (ms) of one sensors CSV, the camera start
    offset ``(ms, event label)`` and problems."""
    rows: list[tuple[float, str]] = []
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            rd = csv.DictReader(fh)
            if not rd.fieldnames or "MonoNs" not in rd.fieldnames \
                    or "OFM_Event" not in rd.fieldnames:
                raise DataError(f"{path}: needs MonoNs and OFM_Event columns")
            for row in rd:
                v = _to_float(row.get("MonoNs", ""))
                label = (row.get("OFM_Event") or "").strip()
                if v is not None and label:
                    rows.append((v, label))
    except OSError as exc:
        raise DataError(f"{path}: {exc}") from exc
    start = next((t for t, label in rows if label == START_LABEL), None)
    if start is None:
        return {}, (None, None), [
            f"{path.name}: no {START_LABEL} row, event offsets unavailable"]
    stim = [(t, label) for t, label in rows
            if label != START_LABEL and not label.startswith("BME")]
    cam = next(((t, label) for t, label in stim if ODOR_ON_RE.match(label)),
               stim[0] if stim else None)
    cam_off = ((cam[0] - start) / 1e6, cam[1]) if cam else (None, None)
    seen: dict[str, int] = {}
    out: dict[tuple[str, int], float] = {}
    for t, label in rows:
        if label == START_LABEL or label.startswith("BME"):
            continue
        if ignore is not None and ignore.search(label):
            continue
        seen[label] = seen.get(label, 0) + 1
        out[(label, seen[label])] = (t - start) / 1e6
    return out, cam_off, []


def load_run(name: str, group: str, path: Path, ignore: re.Pattern | None = None) -> Run:
    if not path.is_dir():
        raise DataError(f"{name}: {path} is not a directory")
    frames = {p.stem: p for p in path.rglob("output_*.csv")}
    sensors = {p.stem[len("sensors_"):]: p for p in path.rglob("sensors_output_*.csv")}
    run = Run(name, group, path, [])
    if not frames and not sensors:
        raise DataError(f"{name}: no output_*.csv / sensors_output_*.csv under {path}")
    for stem in sorted(frames.keys() - sensors.keys()):
        run.problems.append(f"frames CSV without sensors CSV: {stem}")
    for stem in sorted(sensors.keys() - frames.keys()):
        run.problems.append(f"sensors CSV without frames CSV: {stem}")
    paired = sorted(frames.keys() & sensors.keys(), key=lambda s: (_stamp(s), s))
    for i, stem in enumerate(paired, 1):
        n, intervals = read_frame_intervals_ms(frames[stem])
        events, (cam_ms, cam_label), probs = read_event_offsets_ms(sensors[stem], ignore)
        run.trials.append(Trial(i, stem, _tag_of(stem), n, intervals, events,
                                cam_ms, cam_label, probs))
    if not run.trials:
        raise DataError(f"{name}: no frames/sensors CSV pairs under {path}")
    return run


# ── statistics ───────────────────────────────────────────────────────────────

def percentile(values: list[float], q: float) -> float:
    """Linear-interpolation percentile (numpy's default), q in [0, 100]."""
    if not values:
        return 0.0
    v = sorted(values)
    pos = (len(v) - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


def run_interval_summary(run: Run, thresh_ms: float) -> dict:
    allv = [v for t in run.trials for v in t.intervals_ms]
    return {
        "trials": len(run.trials),
        "frames": sum(t.frames for t in run.trials),
        "mean": statistics.fmean(allv) if allv else 0.0,
        "p99": percentile(allv, 99),
        "max": max(allv) if allv else 0.0,
        "gaps": sum(1 for v in allv if v > thresh_ms),
    }


def parse_cpu_csv(path: Path) -> dict:
    """Summarise a cpu_sampler.py CSV."""
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            rd = csv.DictReader(fh)
            cols = rd.fieldnames or []
            rows = list(rd)
    except OSError as exc:
        raise DataError(f"{path}: {exc}") from exc
    if "total_busy_pct" not in cols:
        raise DataError(f"{path}: not a cpu_sampler CSV (no total_busy_pct column)")
    if not rows:
        raise DataError(f"{path}: no samples")

    def col(name: str) -> list[float]:
        vals = (_to_float(r.get(name, "")) for r in rows)
        return [v for v in vals if v is not None]

    core_cols = sorted((c for c in cols if re.fullmatch(r"core\d+_pct", c)),
                       key=lambda c: int(c[4:-4]))
    total = col("total_busy_pct")
    rig = col("rig_proc_pct")
    sw = col("stream_worker_pct")
    temp = col("temp_c")
    mean_busy = statistics.fmean(total) if total else 0.0
    return {
        "samples": len(rows),
        "total_mean": mean_busy,
        "total_max": max(total) if total else 0.0,
        "core_max": {c[:-4]: (max(col(c)) if col(c) else 0.0) for c in core_cols},
        "rig_mean": statistics.fmean(rig) if rig else 0.0,
        "rig_max": max(rig) if rig else 0.0,
        "sw_mean": statistics.fmean(sw) if sw else 0.0,
        "sw_max": max(sw) if sw else 0.0,
        "temp_max": max(temp) if temp else None,
        "headroom": 100.0 - mean_busy,
    }


def stimulus_intervals(events: dict[tuple[str, int], float]) -> dict[str, float]:
    """Camera-independent intervals (ms) of one trial, keyed by a readable name:
    ``odor X duration #k``, ``light duration #k`` and ``light-odor lag #k``
    (light-on k minus the first odor channel's odor-on k)."""
    def times(label: str) -> list[float]:
        return [events[(label, n)] for n in range(1, 1000) if (label, n) in events]

    out: dict[str, float] = {}
    channels = sorted({m.group(1) for lab, _ in events if (m := ODOR_ON_RE.match(lab))})
    odor_on: dict[str, list[float]] = {}
    for ch in channels:
        ons, offs = times(f"OFM_{ch}_ON"), times(f"OFM_{ch}_OFF")
        odor_on[ch] = ons
        for k, t_on in enumerate(ons, 1):
            if k <= len(offs):
                out[f"odor {ch} duration #{k}"] = offs[k - 1] - t_on
    light_label = next((lab for lab in LIGHT_ON_LABELS if times(lab)), None)
    if light_label:
        l_on = times(light_label)
        l_off = times(LIGHT_OFF_LABEL)
        for k in range(1, min(len(l_on), len(l_off)) + 1):
            out[f"light duration #{k}"] = l_off[k - 1] - l_on[k - 1]
        first_ch = next((ch for ch in channels if odor_on.get(ch)), None)
        if first_ch:
            for k in range(1, min(len(l_on), len(odor_on[first_ch])) + 1):
                out[f"light-odor lag #{k}"] = l_on[k - 1] - odor_on[first_ch][k - 1]
    return out


# ── comparison ───────────────────────────────────────────────────────────────

@dataclass
class EventResult:
    run: str
    trial: int
    label: str
    occ: int
    off_mean: float
    on_value: float
    shift: float
    floor: float
    ok: bool


@dataclass
class IntervalResult:
    run: str
    trial: int
    name: str
    off_mean: float
    on_value: float
    shift: float
    floor: float
    ok: bool


@dataclass
class Verdict:
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    events: list[EventResult] = field(default_factory=list)      # informational
    intervals: list[IntervalResult] = field(default_factory=list)  # judged

    @property
    def passed(self) -> bool:
        return not self.reasons


def compare(runs: list[Run], fps: float, tolerance_ms: float) -> Verdict:
    v = Verdict()
    offs = [r for r in runs if r.group == "OFF"]
    ons = [r for r in runs if r.group == "ON"]
    thresh = 1.5 * 1000.0 / fps
    frame_ms = 1000.0 / fps
    if len(offs) < 2:
        v.warnings.append(
            f"only {len(offs)} OFF run(s): there is no OFF-vs-OFF noise floor, "
            "so every stimulus interval is judged against the tolerance alone")

    for r in runs:
        for p in r.problems:
            v.reasons.append(f"{r.name}: {p}")
        for t in r.trials:
            for p in t.problems:
                v.reasons.append(f"{r.name} trial {t.index}: {p}")

    # trial-set agreement: every run must have the same trials as the first OFF run
    ref = offs[0]
    for r in runs:
        if r is ref:
            continue
        if len(r.trials) != len(ref.trials):
            v.reasons.append(f"{r.name} has {len(r.trials)} trial(s), "
                             f"{ref.name} has {len(ref.trials)}")
    common = min(len(r.trials) for r in runs)
    for i in range(common):
        tags = {r.name: r.trials[i].tag for r in runs if r.trials[i].tag is not None}
        if len(set(tags.values())) > 1:
            v.reasons.append(f"trial {i + 1} phase tag differs between runs: "
                             + ", ".join(f"{k}={t}" for k, t in tags.items()))

    # frames
    for i in range(common):
        off_t = [r.trials[i] for r in offs]
        max_off_gaps = max(t.gap_count(thresh) for t in off_t)
        min_off_frames = min(t.frames for t in off_t)
        max_off_gap_ms = max(t.max_interval() for t in off_t)
        for r in ons:
            t = r.trials[i]
            g = t.gap_count(thresh)
            if g > max_off_gaps:
                v.reasons.append(
                    f"{r.name} trial {i + 1}: {g} gap(s) > {thresh:.1f} ms, "
                    f"worst OFF run has {max_off_gaps} (ON max interval "
                    f"{t.max_interval():.1f} ms)")
            elif t.max_interval() > thresh and t.max_interval() > max_off_gap_ms + frame_ms:
                v.reasons.append(
                    f"{r.name} trial {i + 1}: worst gap {t.max_interval():.1f} ms vs "
                    f"{max_off_gap_ms:.1f} ms in the worst OFF run (> 1 frame longer)")
            if t.frames < min_off_frames - 1:
                v.reasons.append(
                    f"{r.name} trial {i + 1}: {t.frames} frames, shortest OFF run "
                    f"has {min_off_frames} (more than 1 frame short)")
            elif t.frames == min_off_frames - 1:
                if any(x.gap_count(thresh) for x in r.trials):
                    v.reasons.append(
                        f"{r.name} trial {i + 1}: {t.frames} frames, one short of "
                        f"the shortest OFF run ({min_off_frames}), and the run has "
                        f"gap(s) > {thresh:.1f} ms")
                else:
                    v.notes.append(
                        f"{r.name} trial {i + 1}: {t.frames} vs {min_off_frames} "
                        "frames: 1-frame boundary difference, no gaps")

    # events
    missing: list[str] = []
    for i in range(common):
        keys: set[tuple[str, int]] = set()
        for r in runs:
            keys |= r.trials[i].events.keys()
        for key in sorted(keys):
            lacking = [r.name for r in runs if key not in r.trials[i].events]
            if lacking:
                missing.append(f"trial {i + 1} event {key[0]} #{key[1]} missing in "
                               + ", ".join(lacking))
                continue
            off_vals = [r.trials[i].events[key] for r in offs]
            floor = max(off_vals) - min(off_vals)
            off_mean = statistics.fmean(off_vals)
            for r in ons:
                on_val = r.trials[i].events[key]
                shift = abs(on_val - off_mean)
                ok = shift <= floor + tolerance_ms + EPS
                v.events.append(EventResult(r.name, i + 1, key[0], key[1],
                                            off_mean, on_val, shift, floor, ok))
    v.reasons.extend(missing)

    # stimulus intervals (judged)
    for i in range(common):
        per_run = {r.name: stimulus_intervals(r.trials[i].events) for r in runs}
        off_names = [r.name for r in offs]
        names: set[str] = set()
        for n in off_names:
            names |= per_run[n].keys()
        for r in runs:
            for n in sorted(names - per_run[r.name].keys()):
                v.reasons.append(
                    f"{r.name} trial {i + 1}: stimulus interval '{n}' missing "
                    "(present in the OFF runs)")
            if r.group == "ON":
                for n in sorted(per_run[r.name].keys() - names):
                    v.reasons.append(
                        f"{r.name} trial {i + 1}: stimulus interval '{n}' present "
                        "in ON but in no OFF run")
        for n in sorted(names):
            off_vals = [per_run[o][n] for o in off_names if n in per_run[o]]
            floor = max(off_vals) - min(off_vals)
            off_mean = statistics.fmean(off_vals)
            for r in ons:
                if n not in per_run[r.name]:
                    continue
                on_val = per_run[r.name][n]
                shift = abs(on_val - off_mean)
                ok = shift <= floor + tolerance_ms + EPS
                v.intervals.append(IntervalResult(r.name, i + 1, n, off_mean, on_val,
                                                  shift, floor, ok))
                if not ok:
                    v.reasons.append(
                        f"{r.name} trial {i + 1} {n}: shift {shift:.3f} ms > OFF "
                        f"floor {floor:.3f} ms + tolerance {tolerance_ms:g} ms")
    return v


# ── report ───────────────────────────────────────────────────────────────────

def _fmt_cpu(cpu: dict | None) -> str:
    if not cpu:
        return "n/a"
    return f"{cpu['total_mean']:.1f} / {cpu['total_max']:.1f}"


def render_report(runs: list[Run], verdict: Verdict, fps: float,
                  tolerance_ms: float) -> str:
    thresh = 1.5 * 1000.0 / fps
    L: list[str] = []
    L.append("# Bench comparison: OFF vs ON")
    L.append("")
    L.append(f"fps {fps:g}, gap threshold {thresh:.1f} ms (1.5 frame intervals), "
             f"stimulus tolerance {tolerance_ms:g} ms beyond the OFF-vs-OFF floor.")
    L.append("")
    L.append("Camera-start offset is recorded per trial and not judged (accepted "
             "2026-10-29); stimulus timing is judged on event-to-event intervals.")
    L.append("")
    L.append("## Per-run summary")
    L.append("")
    L.append("| run | dir | trials | frames | mean ms | p99 ms | max ms | gaps | "
             "CPU busy % mean / max |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for r in runs:
        s = run_interval_summary(r, thresh)
        L.append(f"| {r.name} | {r.path.name} | {s['trials']} | {s['frames']} | "
                 f"{s['mean']:.3f} | {s['p99']:.3f} | {s['max']:.3f} | {s['gaps']} | "
                 f"{_fmt_cpu(r.cpu)} |")
    L.append("")

    L.append("## Per-trial frames (count / gaps / max interval ms)")
    L.append("")
    n_trials = max(len(r.trials) for r in runs)
    L.append("| trial | " + " | ".join(r.name for r in runs) + " |")
    L.append("|---|" + "---|" * len(runs))
    for i in range(n_trials):
        cells = []
        for r in runs:
            if i < len(r.trials):
                t = r.trials[i]
                cells.append(f"{t.frames} / {t.gap_count(thresh)} / {t.max_interval():.1f}")
            else:
                cells.append("MISSING")
        L.append(f"| {i + 1} | " + " | ".join(cells) + " |")
    L.append("")

    cpu_runs = [r for r in runs if r.cpu]
    if cpu_runs:
        L.append("## CPU")
        L.append("")
        L.append("| run | samples | busy mean % | busy max % | idle headroom % | "
                 "rig proc mean / max % | stream_worker mean / max % | max temp C |")
        L.append("|---|---|---|---|---|---|---|---|")
        for r in cpu_runs:
            c = r.cpu
            temp = "n/a" if c["temp_max"] is None else f"{c['temp_max']:.1f}"
            L.append(f"| {r.name} | {c['samples']} | {c['total_mean']:.1f} | "
                     f"{c['total_max']:.1f} | {c['headroom']:.1f} | "
                     f"{c['rig_mean']:.1f} / {c['rig_max']:.1f} | "
                     f"{c['sw_mean']:.1f} / {c['sw_max']:.1f} | {temp} |")
        L.append("")
        L.append("Per-core max busy %:")
        L.append("")
        for r in cpu_runs:
            cores = ", ".join(f"{k} {v:.0f}" for k, v in r.cpu["core_max"].items())
            L.append(f"- {r.name}: {cores or 'no per-core columns'}")
        L.append("")

    L.append("## Camera start offset (recorded, not judged)")
    L.append("")
    L.append("First odor-on event minus RECORDING_START, ms (the event used is the "
             "first `OFM_*_ON`, else the first non-BME event). Cell: value (deviation "
             "from the OFF mean of that trial).")
    L.append("")
    L.append("| trial | " + " | ".join(r.name for r in runs) + " |")
    L.append("|---|" + "---|" * len(runs))
    for i in range(n_trials):
        offv = [r.trials[i].camera_offset_ms for r in runs
                if r.group == "OFF" and i < len(r.trials)
                and r.trials[i].camera_offset_ms is not None]
        mean = statistics.fmean(offv) if offv else None
        cells = []
        for r in runs:
            c = r.trials[i].camera_offset_ms if i < len(r.trials) else None
            if c is None:
                cells.append("n/a")
            elif mean is None:
                cells.append(f"{c:.3f}")
            else:
                cells.append(f"{c:.3f} ({c - mean:+.1f})")
        L.append(f"| {i + 1} | " + " | ".join(cells) + " |")
    L.append("")

    L.append("## Stimulus timing (judged)")
    L.append("")
    L.append("Odor duration, light duration and light-on minus odor-on lag; floor = "
             "largest |OFF_i - OFF_j|; worst 20 by shift minus floor.")
    L.append("")
    worst_i = sorted(verdict.intervals, key=lambda e: e.shift - e.floor, reverse=True)[:20]
    if worst_i:
        L.append("| ON run | trial | interval | OFF mean ms | ON ms | shift ms | "
                 "floor ms | verdict |")
        L.append("|---|---|---|---|---|---|---|---|")
        for e in worst_i:
            L.append(f"| {e.run} | {e.trial} | {e.name} | {e.off_mean:.3f} | "
                     f"{e.on_value:.3f} | {e.shift:.3f} | {e.floor:.3f} | "
                     f"{'ok' if e.ok else 'FAIL'} |")
        n_bad = sum(1 for e in verdict.intervals if not e.ok)
        L.append("")
        L.append(f"{len(verdict.intervals)} interval comparison(s), {n_bad} over the limit.")
    else:
        L.append("No comparable stimulus intervals.")
    L.append("")

    L.append("## Event shifts relative to camera start (informational, worst 20)")
    L.append("")
    L.append("Relative to camera start - includes the recorded camera-start offset; "
             "informational, never a FAIL reason.")
    L.append("")
    worst = sorted(verdict.events, key=lambda e: e.shift - e.floor, reverse=True)[:20]
    if worst:
        L.append("| ON run | trial | event | # | OFF mean ms | ON ms | shift ms | "
                 "floor ms | within floor+tol |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for e in worst:
            L.append(f"| {e.run} | {e.trial} | {e.label} | {e.occ} | {e.off_mean:.3f} | "
                     f"{e.on_value:.3f} | {e.shift:.3f} | {e.floor:.3f} | "
                     f"{'yes' if e.ok else 'no'} |")
        n_bad = sum(1 for e in verdict.events if not e.ok)
        L.append("")
        L.append(f"{len(verdict.events)} event comparison(s), {n_bad} beyond floor + "
                 "tolerance (informational).")
    else:
        L.append("No comparable events.")
    L.append("")

    for note in verdict.notes:
        L.append(f"NOTE: {note}")
    if verdict.notes:
        L.append("")
    for w in verdict.warnings:
        L.append(f"WARNING: {w}")
    if verdict.warnings:
        L.append("")
    if verdict.passed:
        L.append("**RESULT: PASS** - ON adds no frame gap beyond the OFF runs and no "
                 "stimulus-interval shifts beyond the OFF-vs-OFF spread plus tolerance.")
    else:
        L.append(f"**RESULT: FAIL** - {len(verdict.reasons)} reason(s):")
        L.append("")
        for reason in verdict.reasons:
            L.append(f"- {reason}")
    L.append("")
    return "\n".join(L)


# ── CLI ──────────────────────────────────────────────────────────────────────

def _split_labelled(text: str, what: str) -> tuple[str, str]:
    label, sep, rest = text.partition("=")
    label = label.strip().upper()
    if not sep or label not in GROUPS or not rest:
        raise DataError(f"{what} must look like OFF=<path> or ON=<path>, got {text!r}")
    return label, rest


def build_runs(run_args: list[str], cpu_args: list[str],
               ignore: re.Pattern | None) -> list[Run]:
    runs: list[Run] = []
    counts = {g: 0 for g in GROUPS}
    for text in run_args:
        group, rest = _split_labelled(text, "--run")
        counts[group] += 1
        runs.append(load_run(f"{group}{counts[group]}", group,
                             Path(rest).expanduser(), ignore))
    if counts["OFF"] < 1 or counts["ON"] < 1:
        raise DataError("need at least one OFF and one ON --run")
    # CPU CSVs pair with runs of the same group in order of appearance.
    used = {g: 0 for g in GROUPS}
    for text in cpu_args:
        group, rest = _split_labelled(text, "--cpu")
        used[group] += 1
        if used[group] > counts[group]:
            raise DataError(f"--cpu {group}= given {used[group]} time(s) but only "
                            f"{counts[group]} {group} run(s)")
        target = next(r for r in runs if r.name == f"{group}{used[group]}")
        target.cpu = parse_cpu_csv(Path(rest).expanduser())
    return runs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Compare interleaved OFF/ON bench runs (frame gaps, event timing, CPU).")
    ap.add_argument("--run", action="append", default=[], metavar="OFF|ON=DIR",
                    help="batch folder of one run; repeat, in run order")
    ap.add_argument("--cpu", action="append", default=[], metavar="OFF|ON=CSV",
                    help="cpu_sampler.py CSV; the n-th OFF/ON pairs with the n-th OFF/ON run")
    ap.add_argument("--fps", type=float, default=40.0)
    ap.add_argument("--tolerance-ms", type=float, default=1.0)
    ap.add_argument("--ignore-label", default=None, metavar="REGEX",
                    help="skip event labels matching this regex (e.g. LIGHT_PULSE_)")
    ap.add_argument("--out", default=None, help="also write the Markdown report here")
    try:
        args = ap.parse_args(argv)
    except SystemExit as exc:
        return 2 if exc.code not in (0, None) else 0
    try:
        if args.fps <= 0:
            raise DataError("--fps must be positive")
        if args.tolerance_ms < 0:
            raise DataError("--tolerance-ms must not be negative")
        try:
            ignore = re.compile(args.ignore_label) if args.ignore_label else None
        except re.error as exc:
            raise DataError(f"--ignore-label is not a valid regex: {exc}") from exc
        runs = build_runs(args.run, args.cpu, ignore)
        verdict = compare(runs, args.fps, args.tolerance_ms)
        report = render_report(runs, verdict, args.fps, args.tolerance_ms)
    except DataError as exc:
        print(f"bench_compare: {exc}", file=sys.stderr)
        return 2
    print(report)
    if args.out:
        try:
            Path(args.out).expanduser().write_text(report, encoding="utf-8")
        except OSError as exc:
            print(f"bench_compare: cannot write {args.out}: {exc}", file=sys.stderr)
            return 2
    return 0 if verdict.passed else 1


if __name__ == "__main__":
    sys.exit(main())
