#!/usr/bin/env python3
"""Per-experiment Pi -> secured-storage backup (runs every 10 min from a timer).

For every Pi, finds finished batch folders, copies them to
/securedstorage/DATAsec/cole/<week_tag>/<pi>/<batch>/, verifies md5 of every
file on both sides, and only then deletes exactly those files from the Pi.
Continuous *.h264 recordings are remuxed to .mp4 (the .h264 is kept).

A batch is eligible when the rigmon hub says its session is "landed", or the
hub has no record and nothing in it changed for 12 h.  Never a batch with a file
touched in the last 30 min.  A Pi with a running/waiting session is not touched
at all (rsync + md5 load competes with its software trial encoder); the hub is
re-checked before every batch so a session starting mid-backup stops it.  If
the hub is unreachable nothing is backed up.  No rsync delete flag is ever used.

Session states come from the hub's sqlite store (read-only): the hub's
/api/sessions/.../status endpoint only answers to the rig's own address.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import logging
import shlex
import sqlite3
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

DEST_BASE = Path("/securedstorage/DATAsec/cole")
DEFAULT_LOG = DEST_BASE / "_weekly_logs" / "session_backup.log"
HUB_URL = "http://localhost:8765"
HUB_DB = Path("/home/ramanlab/Documents/cole/Data/rigmon_staging/rigmon.sqlite")
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]
FFMPEG = "/usr/bin/ffmpeg"
FFPROBE = "/usr/bin/ffprobe"
RECENT_S = 30 * 60
IDLE_S = 12 * 3600
ACTIVE_RUN_STATES = {"running", "waiting"}

log = logging.getLogger("pi_session_backup")


@dataclass(frozen=True)
class Pi:
    name: str      # ssh host and secured-storage folder name
    rig: str       # rig name in the hub
    root: str      # data root on the Pi


PIS = [
    Pi("behaviorlocust", "BehaviorLocust", "/home/ramanlab/FastStorage/fly_videos"),
    Pi("flybehavior2", "Flybehavior2", "/home/ramanlab/Documents/faststorage/Opto"),
    Pi("flybehavior3", "Flybehavior3", "/home/ramanlab/Documents/faststorage/Opto"),
]


def week_tag(d: date) -> str:
    """ISO year + week + the Monday of that ISO week, e.g. 2026-W46-2026-11-09."""
    iso = d.isocalendar()
    monday = d - timedelta(days=d.weekday())
    return f"{iso[0]}-W{iso[1]:02d}-{monday.isoformat()}"


def eligible_batches(pi_batches: dict[str, float], hub_sessions: dict[str, str],
                     current_batch: str | None, now: float) -> list[str]:
    out = []
    for batch, newest in sorted(pi_batches.items()):
        if batch == current_batch:
            continue
        if now - newest < RECENT_S:
            continue
        state = hub_sessions.get(batch)
        # landed: safely in flys_New. aborted / no record: never will land, so
        # copy once idle. failed / pulling / pending: the hub may still need
        # the Pi copy to retry landing - leave it.
        if state == "landed" or (state in (None, "aborted") and now - newest > IDLE_S):
            out.append(batch)
    return out


# ── process plumbing ─────────────────────────────────────────────────────
def real_runner(argv, input=None):
    return subprocess.run(argv, input=input, capture_output=True, text=True)


def real_http_get(url: str, timeout: float = 10.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode()


def ssh_argv(host: str, remote_cmd: str) -> list[str]:
    return ["ssh", *SSH_OPTS, host, remote_cmd]


def _parse_md5(out: str, prefix: str) -> dict[str, str]:
    res = {}
    for line in out.splitlines():
        if len(line) > 34 and line[32:34] == "  ":
            path = line[34:]
            if path.startswith(prefix):
                path = path[len(prefix):]
            res[path] = line[:32]
    return res


# ── hub ──────────────────────────────────────────────────────────────────
def hub_running_batches(http_get) -> dict[str, str | None] | None:
    """rig name -> batch_dir of its running/waiting run (None when idle);
    None overall when the hub cannot be read."""
    try:
        state = json.loads(http_get(f"{HUB_URL}/api/state"))
        rigs = state["rigs"]
    except Exception as e:  # noqa: BLE001
        log.error("hub unreachable (%s) - backing up nothing", e)
        return None
    out = {}
    for rig, v in rigs.items():
        run = ((v or {}).get("status") or {}).get("run") or {}
        out[rig] = run.get("batch_dir") if run.get("state") in ACTIVE_RUN_STATES else None
        if (v or {}).get("status") is None:
            out[rig] = "?unknown"   # no status -> cannot rule out a running batch
    return out


def hub_session_states(rig: str, db: Path = HUB_DB) -> dict[str, str]:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=10)
    try:
        rows = con.execute("SELECT batch_dir, state FROM sessions WHERE rig=?", (rig,)).fetchall()
    finally:
        con.close()
    return {b: s for b, s in rows}


# ── Pi inventory ─────────────────────────────────────────────────────────
LIST_CMD = (
    "cd {root} 2>/dev/null || exit 0; for d in */; do d=${{d%/}}; [ -d \"$d\" ] || continue; "
    "n=$(find \"$d\" -type f -printf '%T@\\n' | sort -nr | head -1); "
    "[ -n \"$n\" ] || n=$(stat -c %Y \"$d\"); "
    "s=$(du -sb \"$d\" | cut -f1); echo \"$n|$s|$d\"; done"
)


def list_pi_batches(pi: Pi, runner) -> tuple[dict[str, float], dict[str, int]]:
    r = runner(ssh_argv(pi.name, LIST_CMD.format(root=shlex.quote(pi.root))))
    if r.returncode != 0:
        raise RuntimeError(f"{pi.name}: listing failed rc={r.returncode} {getattr(r, 'stderr', '')}")
    mt, sz = {}, {}
    for line in r.stdout.splitlines():
        try:
            n, s, d = line.split("|", 2)
            mt[d], sz[d] = float(n), int(s)
        except ValueError:
            continue
    return mt, sz


# ── backup of one batch ──────────────────────────────────────────────────
def rsync_argv(pi: Pi, batch: str, dest: Path) -> list[str]:
    return ["rsync", "-a", "--partial", "-e", "ssh " + " ".join(SSH_OPTS),
            f"{pi.name}:{pi.root}/{batch}/", f"{dest}/"]


def _csv_info(csv_path: Path) -> tuple[int, float | None]:
    """(data rows, fps from a time column or None)."""
    try:
        with open(csv_path, newline="") as f:
            rd = csv.reader(f)
            header = next(rd, [])
            rows = list(rd)
    except OSError:
        return 0, None
    fps = None
    for i, h in enumerate(header):
        if h.strip().lower() in ("t", "time", "time_s", "timestamp", "timestamp_s", "elapsed_s"):
            try:
                v = [float(r[i]) for r in rows if len(r) > i and r[i] != ""]
                if len(v) > 10 and v[-1] > v[0]:
                    cand = (len(v) - 1) / (v[-1] - v[0])
                    if 5 <= cand <= 120:
                        fps = cand
            except ValueError:
                pass
            break
    return len(rows), fps


def remux_continuous(dest: Path, runner) -> bool:
    ok = True
    for h264 in sorted((dest / "continuous").glob("*.h264")):
        mp4 = h264.with_suffix(".mp4")
        rows, fps = _csv_info(h264.with_suffix(".csv"))
        fps = fps or 20.0
        if not mp4.exists() or mp4.stat().st_size == 0:
            r = runner([FFMPEG, "-y", "-f", "h264", "-r", f"{fps:g}", "-i", str(h264),
                        "-c", "copy", str(mp4)])
            if r.returncode != 0:
                log.error("remux failed: %s rc=%s", h264, r.returncode)
                ok = False
                continue
        p = runner([FFPROBE, "-v", "error", "-count_packets", "-select_streams", "v:0",
                    "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(mp4)])
        try:
            frames = int(p.stdout.strip().split(",")[0])
        except (ValueError, IndexError):
            log.error("ffprobe unreadable for %s", mp4)
            ok = False
            continue
        if frames <= 0 or (rows and abs(frames - rows) > max(2, 0.01 * rows)):
            log.error("frame count mismatch %s: mp4=%d csv_rows=%d", mp4, frames, rows)
            ok = False
        else:
            log.info("remuxed %s: %d frames @%.2f fps (csv rows %d)", mp4.name, frames, fps, rows)
    return ok


def backup_batch(pi: Pi, batch: str, dest_root: Path, runner, dry_run: bool = False) -> bool:
    dest = Path(dest_root) / pi.name / batch
    remote = f"{pi.root}/{batch}"
    if dry_run:
        log.info("[dry-run] would rsync %s:%s/ -> %s/, md5-verify, delete verified Pi files, "
                 "remux continuous/*.h264", pi.name, remote, dest)
        return True
    dest.mkdir(parents=True, exist_ok=True)
    r = runner(rsync_argv(pi, batch, dest))
    if r.returncode != 0:
        log.error("%s/%s: rsync failed rc=%s - Pi untouched", pi.name, batch, r.returncode)
        return False
    pr = runner(ssh_argv(pi.name, f"find {shlex.quote(remote)} -type f -exec md5sum {{}} +"))
    if pr.returncode != 0:
        log.error("%s/%s: Pi md5 failed - Pi untouched", pi.name, batch)
        return False
    lr = runner(["find", str(dest), "-type", "f", "-exec", "md5sum", "{}", "+"])
    if lr.returncode != 0:
        log.error("%s/%s: local md5 failed - Pi untouched", pi.name, batch)
        return False
    pi_md5 = _parse_md5(pr.stdout, remote + "/")
    loc_md5 = _parse_md5(lr.stdout, str(dest) + "/")
    bad = [p for p, h in pi_md5.items() if loc_md5.get(p) != h]
    if bad:
        log.error("%s/%s: %d file(s) missing/mismatched locally (e.g. %s) - Pi untouched",
                  pi.name, batch, len(bad), bad[0])
        return False
    # remux before the Pi delete, so a failed remux leaves the batch eligible
    # and a rerun retries it
    if not remux_continuous(dest, runner):
        log.error("%s/%s: continuous remux failed - Pi untouched", pi.name, batch)
        return False
    if pi_md5:
        listing = "".join(f"{remote}/{p}\n" for p in sorted(pi_md5))
        d = runner(ssh_argv(pi.name, "xargs -d '\\n' -r rm -f --"), input=listing)
        if d.returncode != 0:
            log.error("%s/%s: Pi file removal returned rc=%s", pi.name, batch, d.returncode)
            return False
        runner(ssh_argv(pi.name, f"find {shlex.quote(remote)} -depth -type d -empty -delete"))
        log.info("%s/%s: %d files verified and removed from Pi", pi.name, batch, len(pi_md5))
    return True


# ── driver ───────────────────────────────────────────────────────────────
def run(pis, runner, http_get, dry_run: bool, now: float | None = None,
        dest_base: Path = DEST_BASE, sessions_loader=hub_session_states) -> int:
    now = time.time() if now is None else now
    running = hub_running_batches(http_get)
    if running is None:
        return 1
    dest_root = dest_base / week_tag(date.fromtimestamp(now))
    errors, total = 0, 0
    for pi in pis:
        if pi.rig not in running or running[pi.rig] == "?unknown":
            log.warning("%s: rig state unknown in hub - skipping", pi.name)
            continue
        if running[pi.rig] is not None:
            # rsync/md5 load competes with the rig's software trial encoder
            log.info("%s: session %s is running - not touching this Pi", pi.name, running[pi.rig])
            continue
        try:
            mtimes, sizes = list_pi_batches(pi, runner)
            sessions = sessions_loader(pi.rig)
        except Exception as e:  # noqa: BLE001
            log.error("%s: %s", pi.name, e)
            errors += 1
            continue
        todo = eligible_batches(mtimes, sessions, running[pi.rig], now)
        log.info("%s: %d batches on Pi, running=%s, eligible=%s", pi.name, len(mtimes),
                 running[pi.rig], todo)
        for b in todo:
            now_running = hub_running_batches(http_get)
            if now_running is None or now_running.get(pi.rig) is not None:
                log.info("%s: a session started (or hub lost) - stopping before %s", pi.name, b)
                break
            total += sizes.get(b, 0)
            if not backup_batch(pi, b, dest_root, runner, dry_run):
                errors += 1
    log.info("done: errors=%d bytes_in_scope=%d%s", errors, total, " (dry-run)" if dry_run else "")
    return 1 if errors else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--pi", choices=[p.name for p in PIS])
    ap.add_argument("--all-eligible", action="store_true")
    ap.add_argument("--log", default=str(DEFAULT_LOG))
    a = ap.parse_args(argv)
    if not (a.all_eligible or a.pi):
        ap.error("give --all-eligible or --pi NAME")
    handlers = [logging.StreamHandler()]
    try:
        Path(a.log).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(a.log))
    except OSError:
        pass
    logging.basicConfig(level=logging.INFO, handlers=handlers,
                        format="[%(asctime)s] %(levelname)s %(message)s")
    lock = open("/tmp/pi_session_backup.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        log.info("another run holds the lock - exiting")
        return 0
    pis = [p for p in PIS if not a.pi or p.name == a.pi]
    return run(pis, real_runner, real_http_get, a.dry_run)


if __name__ == "__main__":
    sys.exit(main())
