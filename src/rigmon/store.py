"""SQLite persistence: events, trials, sessions, mutes. Thread-safe."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT, at REAL, rig TEXT, kind TEXT, data TEXT);
CREATE TABLE IF NOT EXISTS trials (
  rig TEXT, batch_dir TEXT, idx INTEGER, phase_tag TEXT, label TEXT, odor TEXT,
  video TEXT, frames_csv TEXT, video_bytes INTEGER, csv_bytes INTEGER, ended_at REAL,
  state TEXT DEFAULT 'queued', error TEXT DEFAULT '', mp4 TEXT DEFAULT '',
  trace TEXT DEFAULT '', trace_error TEXT DEFAULT '',
  PRIMARY KEY (rig, batch_dir, idx));
CREATE TABLE IF NOT EXISTS sessions (
  rig TEXT, batch_dir TEXT, state TEXT, dest TEXT DEFAULT '', error TEXT DEFAULT '',
  updated_at REAL, PRIMARY KEY (rig, batch_dir));
CREATE TABLE IF NOT EXISTS mutes (rig TEXT PRIMARY KEY, muted INTEGER);
CREATE TABLE IF NOT EXISTS mount_checks (
  id INTEGER PRIMARY KEY AUTOINCREMENT, rig TEXT, batch_dir TEXT, at REAL, trigger TEXT,
  state TEXT, flags TEXT DEFAULT '[]', slots TEXT DEFAULT '[]', sharpness REAL,
  frames INTEGER DEFAULT 0, json_path TEXT DEFAULT '', jpg_path TEXT DEFAULT '',
  error TEXT DEFAULT '');
"""
IN_FLIGHT = ("pending", "pulling", "remuxing")


class Store:
    def __init__(self, path, clock=time.time):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._clock = clock
        with self._lock:
            self._db.executescript(SCHEMA)
            cols = {r[1] for r in self._db.execute("PRAGMA table_info(trials)")}
            for col in ("trace", "trace_error"):
                if col not in cols:
                    self._db.execute(f"ALTER TABLE trials ADD COLUMN {col} TEXT DEFAULT ''")
            self._db.commit()

    def _exec(self, sql: str, args=()) -> int:
        with self._lock:
            cur = self._db.execute(sql, args)
            self._db.commit()
            return cur.rowcount

    def _all(self, sql: str, args=()) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._db.execute(sql, args).fetchall()]

    # events
    def add_event(self, ev) -> None:
        self._exec("INSERT INTO events (at, rig, kind, data) VALUES (?,?,?,?)",
                   (ev.at, ev.rig, ev.kind, json.dumps(ev.data, default=str)))

    def recent_events(self, limit: int = 100) -> list[dict]:
        rows = self._all("SELECT at, rig, kind, data FROM events ORDER BY id DESC LIMIT ?", (limit,))
        for r in rows:
            r["data"] = json.loads(r["data"] or "{}")
        return rows

    # trials
    def upsert_trial(self, rig: str, batch_dir: str, t: dict) -> None:
        self._exec(
            "INSERT OR IGNORE INTO trials (rig, batch_dir, idx, phase_tag, label, odor, video,"
            " frames_csv, video_bytes, csv_bytes, ended_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (rig, batch_dir, int(t["index"]), t.get("phase_tag", ""), t.get("label", ""),
             t.get("odor", ""), t.get("video", ""), t.get("frames_csv", ""),
             int(t.get("video_bytes") or 0), int(t.get("csv_bytes") or 0),
             float(t.get("ended_at") or 0)))

    def set_trial_state(self, rig, batch_dir, idx, state, *, error="", mp4="") -> None:
        self._exec("UPDATE trials SET state=?, error=?, mp4=COALESCE(NULLIF(?, ''), mp4)"
                   " WHERE rig=? AND batch_dir=? AND idx=?",
                   (state, error, mp4, rig, batch_dir, int(idx)))

    def set_trial_trace(self, rig, batch_dir, idx, *, trace: str = "", error: str = "") -> None:
        self._exec("UPDATE trials SET trace=?, trace_error=? WHERE rig=? AND batch_dir=? AND idx=?",
                   (trace, error, rig, batch_dir, int(idx)))

    def clear_traces(self, rig: str, batch_dir: str) -> int:
        return self._exec("UPDATE trials SET trace='', trace_error='' WHERE rig=? AND batch_dir=? AND (trace<>'' OR trace_error<>'')",
                          (rig, batch_dir))

    def batch_last_ended(self, rig: str, batch_dir: str) -> float | None:
        rows = self._all("SELECT MAX(ended_at) AS m FROM trials WHERE rig=? AND batch_dir=?", (rig, batch_dir))
        return float(rows[0]["m"]) if rows and rows[0]["m"] is not None else None

    def get_trial(self, rig, batch_dir, idx) -> dict | None:
        rows = self._all("SELECT * FROM trials WHERE rig=? AND batch_dir=? AND idx=?",
                         (rig, batch_dir, int(idx)))
        return rows[0] if rows else None

    def trials(self, limit: int = 200) -> list[dict]:
        return self._all("SELECT * FROM trials ORDER BY ended_at DESC, idx DESC LIMIT ?", (limit,))

    # sessions
    def set_session(self, rig, batch_dir, state, *, dest="", error="") -> None:
        self._exec(
            "INSERT INTO sessions (rig, batch_dir, state, dest, error, updated_at)"
            " VALUES (?,?,?,?,?,?) ON CONFLICT(rig, batch_dir) DO UPDATE SET"
            " state=excluded.state, dest=COALESCE(NULLIF(excluded.dest, ''), sessions.dest),"
            " error=excluded.error, updated_at=excluded.updated_at",
            (rig, batch_dir, state, dest, error, self._clock()))

    def set_session_unless(self, rig, batch_dir, state, *, not_in=("aborted", "landed"),
                           dest="", error="") -> bool:
        """Compare-and-set: write `state` unless the session is already in one
        of `not_in`. One statement under the store lock, so an abort that lands
        between a caller's check and its write can never be overwritten."""
        marks = ",".join("?" * len(not_in))
        n = self._exec(
            "INSERT INTO sessions (rig, batch_dir, state, dest, error, updated_at)"
            " VALUES (?,?,?,?,?,?) ON CONFLICT(rig, batch_dir) DO UPDATE SET"
            " state=excluded.state, dest=COALESCE(NULLIF(excluded.dest, ''), sessions.dest),"
            " error=excluded.error, updated_at=excluded.updated_at"
            f" WHERE sessions.state NOT IN ({marks})",
            (rig, batch_dir, state, dest, error, self._clock(), *not_in))
        return n > 0

    def finish_batch_trials(self, rig, batch_dir, existing: set) -> int:
        """The session landed: every non-done trial of the batch is now in
        flys_New, so close them out. `existing` = file names present in dest."""
        n = 0
        for r in self._all("SELECT idx, video FROM trials WHERE rig=? AND batch_dir=?"
                           " AND state!='done'", (rig, batch_dir)):
            mp4 = Path(r["video"] or "").with_suffix(".mp4").name
            self.set_trial_state(rig, batch_dir, r["idx"], "done",
                                 mp4=mp4 if r["video"] and mp4 in existing else "")
            n += 1
        return n

    def get_session(self, rig, batch_dir) -> dict | None:
        rows = self._all("SELECT * FROM sessions WHERE rig=? AND batch_dir=?", (rig, batch_dir))
        return rows[0] if rows else None

    def fail_interrupted_sessions(self) -> int:
        self._exec("UPDATE trials SET state='failed', error='hub restarted mid-transfer'"
                   " WHERE state IN ('pulling','remuxing')")
        return self._exec(
            "UPDATE sessions SET state='failed', error='hub restarted during finalize', updated_at=?"
            f" WHERE state IN ({','.join('?' * len(IN_FLIGHT))})", (self._clock(), *IN_FLIGHT))

    # mutes
    def set_muted(self, rig: str, muted: bool) -> None:
        self._exec("INSERT OR REPLACE INTO mutes (rig, muted) VALUES (?,?)", (rig, int(bool(muted))))

    def is_muted(self, rig: str) -> bool:
        return bool(self.mutes().get(rig, False))

    def mutes(self) -> dict[str, bool]:
        return {r["rig"]: bool(r["muted"]) for r in self._all("SELECT rig, muted FROM mutes")}

    # mount checks
    def add_mount_check(self, rig: str, batch_dir: str, trigger: str) -> int:
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO mount_checks (rig, batch_dir, at, trigger, state) VALUES (?,?,?,?,?)",
                (rig, batch_dir, self._clock(), trigger, "running"))
            self._db.commit()
            return int(cur.lastrowid)

    def update_mount_check(self, check_id: int, *, state: str, flags=None, slots=None,
                           sharpness=None, frames=None, json_path=None, jpg_path=None,
                           error=None) -> None:
        sets, args = ["state=?"], [state]
        for col, val in (("flags", flags), ("slots", slots)):
            if val is not None:
                sets.append(f"{col}=?")
                args.append(json.dumps(val, default=str))
        for col, val in (("sharpness", sharpness), ("frames", frames), ("json_path", json_path),
                         ("jpg_path", jpg_path), ("error", error)):
            if val is not None:
                sets.append(f"{col}=?")
                args.append(val)
        self._exec(f"UPDATE mount_checks SET {', '.join(sets)} WHERE id=?", (*args, int(check_id)))

    def latest_mount_check(self, rig: str) -> dict | None:
        rows = self._all("SELECT * FROM mount_checks WHERE rig=? ORDER BY id DESC LIMIT 1", (rig,))
        if not rows:
            return None
        row = rows[0]
        row["flags"] = json.loads(row["flags"] or "[]")
        row["slots"] = json.loads(row["slots"] or "[]")
        return row

    def mount_check_done(self, rig: str, batch_dir: str, trigger: str) -> bool:
        return bool(self._all("SELECT 1 FROM mount_checks WHERE rig=? AND batch_dir=? AND trigger=? LIMIT 1",
                              (rig, batch_dir, trigger)))

    def mount_check_running(self, rig: str) -> bool:
        return bool(self._all("SELECT 1 FROM mount_checks WHERE rig=? AND state='running' LIMIT 1", (rig,)))

    def sharpness_reference(self, rig: str) -> float | None:
        rows = self._all("SELECT sharpness FROM mount_checks WHERE rig=? AND state='ok' "
                         "AND sharpness IS NOT NULL ORDER BY id DESC LIMIT 20", (rig,))
        vals = sorted(r["sharpness"] for r in rows)
        if not vals:
            return None
        mid = len(vals) // 2
        return float(vals[mid]) if len(vals) % 2 else float((vals[mid - 1] + vals[mid]) / 2)

    def fail_interrupted_mount_checks(self) -> int:
        return self._exec("UPDATE mount_checks SET state='error', error='hub restarted during the check' "
                          "WHERE state='running'")
