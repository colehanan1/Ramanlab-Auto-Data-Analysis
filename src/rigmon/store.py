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
  PRIMARY KEY (rig, batch_dir, idx));
CREATE TABLE IF NOT EXISTS sessions (
  rig TEXT, batch_dir TEXT, state TEXT, dest TEXT DEFAULT '', error TEXT DEFAULT '',
  updated_at REAL, PRIMARY KEY (rig, batch_dir));
CREATE TABLE IF NOT EXISTS mutes (rig TEXT PRIMARY KEY, muted INTEGER);
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
