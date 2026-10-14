"""Turn successive /status.json snapshots of one rig into events. Pure."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

log = logging.getLogger("rigmon.events")


@dataclass(frozen=True)
class Event:
    kind: str
    rig: str
    at: float
    data: dict = field(default_factory=dict)


def _run(snap: dict | None) -> dict:
    return (snap or {}).get("run") or {}


class RigTracker:
    def __init__(self, rig: str, offline_after_s: float = 60.0):
        self.rig = rig
        self.offline_after_s = offline_after_s
        self.last: dict | None = None
        self.last_seen: float | None = None
        self.offline = False
        self._first = True
        self._warned = False
        self._seen: set[tuple[str, int]] = set()

    def update(self, snap: dict | None, now: float) -> list[Event]:
        if snap is None:
            if (not self.offline and self.last_seen is not None
                    and now - self.last_seen >= self.offline_after_s):
                self.offline = True
                return [Event("rig_offline", self.rig, now, {
                    "last_seen": self.last_seen,
                    "was_running": _run(self.last).get("state") == "running"})]
            return []

        out: list[Event] = []
        ev = lambda kind, data: out.append(Event(kind, self.rig, now, data))  # noqa: E731
        if self.offline:
            self.offline = False
            ev("rig_online", {})
        prev, cur = _run(self.last), _run(snap)
        batch = cur.get("batch_dir", "")
        replay = self._first

        if cur.get("state") == "running" and (
                prev.get("state") != "running" or prev.get("batch_dir") != batch):
            ev("session_started", {"batch_dir": batch,
                                   "expected_end_at": cur.get("expected_end_at"),
                                   "replay": replay})
        if self.last is not None and (
                (self.last.get("phase"), self.last.get("trial_label"))
                != (snap.get("phase"), snap.get("trial_label"))):
            ev("phase_changed", {"phase": snap.get("phase"),
                                 "trial_label": snap.get("trial_label"),
                                 "label": snap.get("label")})
        for t in cur.get("trials") or []:
            try:
                key = (batch, int(t["index"]))
            except (KeyError, TypeError, ValueError):
                if not self._warned:
                    self._warned = True
                    log.warning("%s: skipping malformed trial entry %r", self.rig, t)
                continue
            if key in self._seen:
                continue
            self._seen.add(key)
            ev("trial_finished", {"batch_dir": batch, "output_dir": cur.get("output_dir", ""),
                                  "trial": t, "replay": replay})
        if not replay:
            for state, kind in (("complete", "session_complete"), ("failed", "session_failed")):
                if cur.get("state") == state and prev.get("state") != state:
                    ev(kind, {"batch_dir": batch, "error": cur.get("error", "")})

        self.last, self.last_seen, self._first = snap, now, False
        return out
