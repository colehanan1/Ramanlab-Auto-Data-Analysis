"""Desktop pop-ups (notify-send) and hub ntfy alerts. Muting a rig silences
its routine pop-ups only; failures, offline-mid-run and alerts always show."""

from __future__ import annotations

import logging
import threading
import urllib.request
from datetime import datetime

from .transfer import subprocess_runner

log = logging.getLogger("rigmon.notify")
ALWAYS = {"session_failed", "rig_offline"}


def notify_send_argv(title: str, body: str, urgency: str = "normal") -> list[str]:
    return ["notify-send", "-a", "Rig Monitor", "-u", urgency, title, body]


def ntfy_post(topic: str, title: str, body: str) -> None:
    req = urllib.request.Request(
        f"https://ntfy.sh/{topic}", data=body.encode("utf-8"), method="POST",
        headers={"Title": title.encode("ascii", "replace").decode("ascii"), "Priority": "high"})
    urllib.request.urlopen(req, timeout=10)


def _hhmm(epoch) -> str:
    return datetime.fromtimestamp(float(epoch)).strftime("%H:%M")


def describe(ev, label: str):
    d = ev.data
    if ev.kind == "trial_finished":
        t = d.get("trial", {})
        return (f"{label} · {t.get('label') or t.get('phase_tag', 'trial')} done",
                t.get("odor") or "", "normal")
    if ev.kind == "session_started":
        end = d.get("expected_end_at")
        body = d.get("batch_dir", "")
        if end:
            try:
                body += f" · done ≈ {_hhmm(end)}"
            except (TypeError, ValueError, OverflowError, OSError):
                pass
        return (f"{label} · session started", body, "normal")
    if ev.kind == "session_complete":
        return (f"{label} · experiment complete", d.get("batch_dir", ""), "normal")
    if ev.kind == "session_failed":
        return (f"{label} · experiment FAILED", d.get("error") or d.get("batch_dir", ""), "critical")
    if ev.kind == "rig_offline" and d.get("was_running"):
        return (f"{label} · offline", "No answer from the rig for 60 s mid-run", "critical")
    return None


def _thread(fn) -> None:
    threading.Thread(target=fn, daemon=True).start()


class Notifier:
    def __init__(self, cfg, store, runner=subprocess_runner, ntfy=ntfy_post, spawn=_thread):
        self.cfg, self.store, self.runner, self.ntfy, self.spawn = cfg, store, runner, ntfy, spawn

    def handle(self, ev, label: str) -> None:
        if ev.data.get("replay"):
            return
        msg = describe(ev, label)
        if msg is None:
            return
        if ev.kind not in ALWAYS and self.store.is_muted(ev.rig):
            return
        title, body, urgency = msg
        self._popup(title, body, urgency)
        if ev.kind == "rig_offline":
            self._ntfy(title, body)

    def alert(self, title: str, body: str) -> None:
        self._popup(title, body, "critical")
        self._ntfy(title, body)

    def _popup(self, title, body, urgency) -> None:
        def go():
            rc, out = self.runner(notify_send_argv(title, body, urgency), 10.0, None)
            if rc != 0:
                log.warning("notify-send failed (%s): %s", rc, out)
        self.spawn(go)

    def _ntfy(self, title, body) -> None:
        if not self.cfg.ntfy_topic:
            return
        def go():
            try:
                self.ntfy(self.cfg.ntfy_topic, title, body)
            except Exception as e:
                log.warning("ntfy failed: %s", e)
        self.spawn(go)
