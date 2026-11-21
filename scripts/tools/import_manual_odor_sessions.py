#!/usr/bin/env python3
"""Import manual-odor Pi sessions (Caleb's ``manual_odor_pi.py``) into the
rig layout ``fbpipe`` reads, so ``config/config_caleb.yaml`` can process them.

Source session folder (one fly)::

    Test_2026-11-18_02-53-54PM/
        events.csv                    # the script's event log
        trial_007_camera_0.mp4        # side view  -> imported (YOLO model fits it)
        trial_007_camera_1.mp4        # top view   -> left in place

Output (one batch folder per session)::

    <dest>/<Dataset>/november_18_batch_1/
        output_november_18_batch_1_testing_1_3-Octonol_20261118_164134.mp4
        output_..._testing_1_3-Octonol_20261118_164134.txt   # sidecar (cycle name, fps, frames)
        output_..._testing_1_3-Octonol_20261118_164134.csv   # FrameNumber + ActiveOFM odor window
        session_metadata.txt, source_events.csv, import_manifest.csv

Odor timing. The script logs no per-frame timestamps. The camera's first frame
arrives ~1.2 s after the logged ``recording_start`` (measured from every video:
duration = logged span - 1.15..1.5 s), so the first frame is placed by
END-alignment: first_frame = stop_request - n_frames / fps, where stop_request
is ``testing_post_end`` for test trials and ``odor_off`` for training trials
(the recorder is stopped the moment the odor times out). Estimated error
±0.15 s (3 frames at 20 fps).

Training / manual trials in the current script start recording only 1.1 s
before the valve opens, i.e. AFTER the odor is already on: their videos have
no pre-odor baseline and the ActiveOFM window starts at frame 1.

Sessions are COMPLETE when ``training_complete`` was followed by at least one
finished test trial and no ``testing_cancelled``. Only complete sessions are
imported unless ``--include-pilot`` is given; pilots go to ``<Dataset>-Pilot``.

Usage::

    python3 scripts/tools/import_manual_odor_sessions.py \
        --src "/securedstorage/DATAsec/Caleb/Caleb/Experiment 2 Manual Odor Testing" \
        --dest /home/ramanlab/Documents/cole/Data/flys_Caleb [--include-pilot] [--dry-run]

Never overwrites: a batch folder that already exists is skipped (YOLO outputs
live there). Source files are only read and copied, never moved or deleted.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

# Script odor names -> the rig's cycle labels (exact spellings the pipeline
# already handles, including the rig's "3-Octonol").
ODOR_LABELS = {
    "Isoamyl acetate": "IsoamylAcetate",
    "3-Octanol": "3-Octonol",
    "Ethyl butyrate": "EthylButyrate",
    "Benzaldehyde": "Benzaldehyde",
    "Citral": "Citral",
    "Hexanol": "Hexanol",
    "Linalool": "Linalool",
}
# v2's end-of-test light probe (no odor). The pipeline recognises the label and
# reads the stimulus window from the trial's sensors log (written below).
LIGHT_ONLY = "LightOnly"
# ActiveOFM channel per odor (any non-"off", non-OFM_P value marks odor on).
OFM_CHANNEL = {
    "IsoamylAcetate": "OFM_A", "3-Octonol": "OFM_O", "EthylButyrate": "OFM_E",
    "Benzaldehyde": "OFM_B", "Citral": "OFM_C", "Hexanol": "OFM_H", "Linalool": "OFM_L",
}
# Trained odor -> dataset prefix used across the repo (IAA-, Hex-, 3Oct-, ...).
DATASET_PREFIX = {
    "IsoamylAcetate": "IAA", "3-Octonol": "3Oct", "EthylButyrate": "EB",
    "Benzaldehyde": "Benz", "Citral": "Cit", "Hexanol": "Hex", "Linalool": "Lin",
}
DATASET_SUFFIX = "CalebManual"
HOST = "CalebManualPi"
FPS = 20.0  # manual_odor_pi.py records at --framerate 20

_TRIAL_RE = re.compile(r"trial_(\d+)_camera_0\.mp4")
_SESSION_RE = re.compile(r"_(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2}[AP]M)$")


@dataclass
class Trial:
    number: int            # the script's trial_NNN
    odor: str              # rig label
    phase: str             # "training" | "testing"
    phase_index: int       # 1-based within the phase
    rec_start: float       # elapsed_session_s of camera_0 recording_start
    odor_on: float
    odor_off: Optional[float]
    stop_request: Optional[float]


@dataclass
class SessionInfo:
    path: Path
    start: datetime
    complete: bool
    trained_odor: Optional[str]
    trials: list


def _odor_label(raw: str) -> str:
    name = raw.split(";")[0].strip()
    if name not in ODOR_LABELS:
        raise ValueError(f"Unknown odor {name!r} (add it to ODOR_LABELS)")
    return ODOR_LABELS[name]


def _read_rows(events_csv: Path) -> list[dict]:
    with Path(events_csv).open(newline="") as f:
        return list(csv.DictReader(f))


def parse_events(events_csv: Path) -> list[Trial]:
    """Odor trials in recording order; no-odor recordings are skipped."""
    rows = _read_rows(events_csv)
    trials: list[Trial] = []
    cur: Optional[dict] = None
    in_testing = False
    counts = Counter()

    def _close():
        nonlocal cur
        if cur and cur.get("odor_on") is not None:
            phase = "testing" if cur["testing"] else "training"
            counts[phase] += 1
            # The recorder stops at the end of the post-odor tail (tests; v2
            # training) or the moment the odor times out (v1 training).
            stop = cur.get("post_end", cur.get("odor_off"))
            trials.append(Trial(cur["number"], cur["odor"], phase, counts[phase],
                                cur["rec_start"], cur["odor_on"], cur.get("odor_off"),
                                stop if stop is not None else cur.get("stop")))
        cur = None

    for r in rows:
        ev, detail, t = r["event"], r.get("detail") or "", float(r["elapsed_session_s"])
        if ev == "training_complete":
            in_testing = True
        elif ev == "testing_cancelled":
            in_testing = False
        if ev == "recording_start":
            m = _TRIAL_RE.search(detail)
            if m:
                _close()
                cur = {"number": int(m.group(1)), "rec_start": t, "testing": in_testing,
                       "odor": None, "odor_on": None}
            continue
        if cur is None:
            continue
        if ev == "odor_on" and cur["odor_on"] is None:
            cur["odor"], cur["odor_on"] = _odor_label(detail), t
        elif ev == "light_on" and cur["odor_on"] is None:
            cur["odor"], cur["odor_on"] = LIGHT_ONLY, t
        elif ev in ("odor_off", "light_off") and cur.get("odor_off") is None:
            cur["odor_off"] = t
        elif ev in ("testing_post_end", "manual_post_end"):
            cur["post_end"] = t  # v2 training trials record a post-odor tail too
        elif ev == "recording_stop" and "camera_0" in detail:
            cur["stop"] = t
            _close()
    _close()
    return trials


def _session_start(session_dir: Path, rows: list[dict]) -> datetime:
    if rows:
        return datetime.fromisoformat(rows[0]["wall_time"])
    m = _SESSION_RE.search(session_dir.name)
    return datetime.strptime(f"{m.group(1)} {m.group(2)}", "%Y-%m-%d %I-%M-%S%p")


def summarize_session(session_dir: Path) -> SessionInfo:
    rows = _read_rows(session_dir / "events.csv")
    events = [r["event"] for r in rows]
    trials = parse_events(session_dir / "events.csv")
    complete = ("training_complete" in events and "testing_cancelled" not in events
                and any(t.phase == "testing" for t in trials))
    training = Counter(t.odor for t in trials if t.phase == "training")
    trained = training.most_common(1)[0][0] if training else None
    return SessionInfo(session_dir, _session_start(session_dir, rows), complete, trained, trials)


def session_ready(info: SessionInfo) -> bool:
    """Every trial's camera_0 video is present and no rsync transfer is pending.

    Sessions pushed from the Pi (manual_odor_pi_v2 --push-to) arrive file by
    file; a batch folder is never re-imported, so importing early would keep a
    half session forever. Skipped sessions are picked up on the next run.
    """
    if (info.path / ".rsync-partial").exists():
        return False
    return all((info.path / f"trial_{t.number:03d}_camera_0.mp4").is_file() for t in info.trials)


def odor_frames(t: Trial, n_frames: int, fps: float) -> tuple[int, int, float]:
    """(on_frame, off_frame, first_frame_elapsed_s), 0-based, end-aligned."""
    end = t.stop_request if t.stop_request is not None else t.odor_off
    first = end - n_frames / fps
    on_f = max(0, min(n_frames - 1, round((t.odor_on - first) * fps)))
    off_t = t.odor_off if t.odor_off is not None else end
    off_f = max(on_f + 1, min(n_frames, round((off_t - first) * fps)))
    return on_f, off_f, first


def batch_names(starts: list[datetime]) -> list[str]:
    """``<month>_<dd>_batch_<k>``, k = order of the session within its day."""
    by_day = defaultdict(list)
    for i, s in enumerate(starts):
        by_day[s.date()].append((s, i))
    out = [""] * len(starts)
    for day, items in by_day.items():
        for k, (_, i) in enumerate(sorted(items), start=1):
            out[i] = f"{day.strftime('%B').lower()}_{day.day:02d}_batch_{k}"
    return out


def output_stem(batch: str, t: Trial, stamp: str) -> str:
    return f"output_{batch}_{t.phase}_{t.phase_index}_{t.odor}_{stamp}"


def write_sidecars(batch_dir: Path, stem: str, t: Trial, *, n_frames: int, fps: float,
                   width: int, height: int, start_utc: str, fly_id: str,
                   source_video: str) -> tuple[int, int]:
    on_f, off_f, first = odor_frames(t, n_frames, fps)
    (batch_dir / f"{stem}.txt").write_text(
        "Video Metadata\n==============\n"
        f"Cycle Name: {t.phase}_{t.phase_index}_{t.odor}\n"
        f"Video File: {source_video}\n"
        f"Start Time (UTC): {start_utc}\n"
        f"Duration: {n_frames / fps:.6f} seconds\n"
        f"Total Frames Captured: {n_frames}\n"
        f"Achieved Frame Rate: {fps:.2f} FPS\n"
        f"Frame Resolution: {width} x {height}\n\n"
        "Subject / Session Metadata\n--------------------------\n"
        f"Fly ID: {fly_id}\n"
        f"Host: {HOST} | imported by scripts/tools/import_manual_odor_sessions.py\n\n"
        "Odor Vials\n----------\n"
        f"Trial odor: {t.odor}\n\n"
        "Odor Timing (estimated)\n-----------------------\n"
        f"Source trial: trial_{t.number:03d}\n"
        f"Odor on frame (0-based): {on_f}\nOdor off frame (0-based): {off_f}\n"
        f"First frame after recording_start (s): {first - t.rec_start:.3f}\n"
        f"Odor onset before first frame: {'YES - no pre-odor baseline' if t.odor_on < first else 'no'}\n",
        encoding="utf-8",
    )
    light = t.odor == LIGHT_ONLY
    chan = OFM_CHANNEL.get(t.odor, "off")
    with (batch_dir / f"{stem}.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["FrameNumber", "ActiveOFM"])
        for i in range(n_frames):
            # No valve opens in the light probe: its window lives in the sensors log.
            w.writerow([i + 1, chan if (not light and on_f <= i < off_f) else "off"])
    if light:
        # Same layout the rigs write (MonoNs + OFM_Event); t = 0 is the first frame.
        off_t = t.odor_off if t.odor_off is not None else t.stop_request
        with (batch_dir / f"sensors_{stem}.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["MonoNs", "OFM_Event"])
            w.writerow([int(round(first * 1e9)), "RECORDING_START"])
            w.writerow([int(round(t.odor_on * 1e9)), "LIGHT_SOLID_ON"])
            w.writerow([int(round(off_t * 1e9)), "LIGHT_OFF"])
    return on_f, off_f


def _probe(video: Path) -> tuple[int, int, int]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,nb_read_packets", "-of", "json", str(video)],
        capture_output=True, text=True, check=True).stdout
    s = json.loads(out)["streams"][0]
    return int(s["nb_read_packets"]), int(s["width"]), int(s["height"])


def import_session(info: SessionInfo, batch: str, dataset_dir: Path, *, dry_run: bool) -> list[dict]:
    batch_dir = dataset_dir / batch
    if batch_dir.exists():
        print(f"[skip] {batch_dir} exists (never overwritten)")
        return []
    rows = []
    zero_wall = info.start  # wall time of the first event row ~ elapsed 0
    first_elapsed = float(_read_rows(info.path / "events.csv")[0]["elapsed_session_s"])
    if not dry_run:
        batch_dir.mkdir(parents=True)
    for t in info.trials:
        src = info.path / f"trial_{t.number:03d}_camera_0.mp4"
        if not src.exists():
            print(f"[warn] missing {src}")
            continue
        n_frames, width, height = _probe(src)
        on_f, off_f, first = odor_frames(t, n_frames, FPS)
        start_local = zero_wall + timedelta(seconds=first - first_elapsed)
        stamp = start_local.strftime("%Y%m%d_%H%M%S")
        stem = output_stem(batch, t, stamp)
        row = {"batch": batch, "source": str(src), "dest": f"{stem}.mp4", "phase": t.phase,
               "phase_index": t.phase_index, "odor": t.odor, "n_frames": n_frames,
               "odor_on_frame": on_f, "odor_off_frame": off_f,
               "first_frame_after_rec_start_s": round(first - t.rec_start, 3),
               "has_pre_odor_baseline": t.odor_on >= first}
        rows.append(row)
        print(f"  {stem}.mp4  odor frames {on_f}-{off_f} of {n_frames}"
              + ("" if row["has_pre_odor_baseline"] else "  [NO BASELINE]"))
        if dry_run:
            continue
        shutil.copy2(src, batch_dir / f"{stem}.mp4")
        start_utc = start_local.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        write_sidecars(batch_dir, stem, t, n_frames=n_frames, fps=FPS, width=width,
                       height=height, start_utc=start_utc, fly_id=info.path.name,
                       source_video=str(src))
    if dry_run:
        return rows
    shutil.copy2(info.path / "events.csv", batch_dir / "source_events.csv")
    (batch_dir / "session_metadata.txt").write_text(
        "Session Metadata\n===============\n\n"
        f"Fly ID: {info.path.name}\n"
        f"Source session: {info.path}\n"
        f"Trained odor: {info.trained_odor}\n"
        f"Protocol complete: {info.complete}\n"
        f"Host: {HOST}\n"
        "Camera: camera_0 (side view) imported; camera_1 (top view) left in the source folder\n",
        encoding="utf-8")
    with (batch_dir / "import_manifest.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["batch"])
        w.writeheader()
        w.writerows(rows)
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", required=True, type=Path, help="folder holding the session folders")
    ap.add_argument("--dest", required=True, type=Path, help="dataset base (config dataset_bases.data)")
    ap.add_argument("--include-pilot", action="store_true",
                    help="also import incomplete/ad-hoc sessions into <Dataset>-Pilot")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    # Number batches over EVERY session with odor trials (ready or not, pilot or
    # not), by start time, so a session's batch name never depends on which
    # other sessions happened to be importable on a given run.
    every = [summarize_session(d) for d in sorted(p for p in args.src.iterdir()
                                                   if (p / "events.csv").exists())]
    every = [info for info in every if info.trials]
    names = batch_names([info.start for info in every])
    for info, batch in zip(every, names):
        d = info.path
        if not session_ready(info):
            print(f"[not ready, retry next run] {d.name}: videos still arriving")
            continue
        if not info.complete and not args.include_pilot:
            print(f"[pilot, not imported] {d.name}: {len(info.trials)} odor trial(s)")
            continue
        prefix = DATASET_PREFIX.get(info.trained_odor or "", "Mixed")
        dataset = f"{prefix}-{DATASET_SUFFIX}" + ("" if info.complete else "-Pilot")
        print(f"{info.path.name} -> {dataset}/{batch} (trained: {info.trained_odor})")
        import_session(info, batch, args.dest / dataset, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
