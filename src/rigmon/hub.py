"""The hub: poll every rig, turn snapshots into events, and fan them out to
the store, desktop notifications and the per-rig transfer queue."""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

import httpx

from .events import RigTracker

log = logging.getLogger("rigmon.hub")


async def fetch_status(client: httpx.AsyncClient, rig, timeout: float = 2.0) -> dict | None:
    try:
        r = await client.get(rig.status_url, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        return data if isinstance(data, dict) else None
    except (httpx.HTTPError, ValueError):
        return None


def _public(snap: dict | None) -> dict | None:
    """Dashboard copy of a Pi snapshot: no rsync destination or Pi paths."""
    if not isinstance(snap, dict):
        return snap
    out = dict(snap)
    if isinstance(out.get("run"), dict):
        out["run"] = {k: v for k, v in out["run"].items() if k not in ("push_to", "output_dir")}
    return out


def _public_mc(row: dict | None) -> dict | None:
    """Dashboard copy of a mount-check row: filenames only, no hub filesystem paths."""
    if not isinstance(row, dict):
        return row
    out = {k: v for k, v in row.items() if k not in ("json_path", "jpg_path")}
    out["jpg_name"] = Path(row["jpg_path"]).name if row.get("jpg_path") else ""
    return out


RECORDING_PHASES = {"BASELINE", "STIMULUS", "POST"}


class Hub:
    def __init__(self, cfg, store, notifier, transfer, finalizer, workers, clock=time.time, mountcheck=None,
                 mount_check_error=None):
        self.cfg, self.store, self.notifier = cfg, store, notifier
        self.transfer, self.finalizer, self.workers, self.clock = transfer, finalizer, workers, clock
        self.trackers = {r.name: RigTracker(r.name, cfg.offline_after_s) for r in cfg.rigs}
        self.mountcheck, self.mount_check_error = mountcheck, mount_check_error
        self.latest: dict[str, dict | None] = {r.name: None for r in cfg.rigs}
        self.reachable: dict[str, bool] = {r.name: False for r in cfg.rigs}
        self.fetched_at: dict[str, float | None] = {r.name: None for r in cfg.rigs}

    def ingest(self, rig_name: str, snap: dict | None, now: float | None = None):
        now = self.clock() if now is None else now
        rig = self.cfg.rig(rig_name)
        self.reachable[rig_name] = snap is not None
        if snap is not None:
            self.latest[rig_name], self.fetched_at[rig_name] = snap, now
        events = self.trackers[rig_name].update(snap, now)
        for ev in events:
            try:
                if ev.kind != "phase_changed":
                    self.store.add_event(ev)
                if ev.kind == "trial_finished":
                    if self._already_landed(rig_name, ev.data["batch_dir"]):
                        continue  # replay of a finished run the hub already filed
                    self.store.upsert_trial(rig_name, ev.data["batch_dir"], ev.data["trial"])
                    if self.cfg.transfer_enabled:
                        self._queue_trial(rig, ev.data)
                self.notifier.handle(ev, rig.label)
            except Exception:
                log.exception("event %s for %s failed", ev.kind, rig_name)
        if snap is not None and self.mountcheck is not None:
            try:
                self._maybe_mount_check(rig, snap)
            except Exception:
                log.exception("mount-check trigger for %s failed", rig_name)
        return events

    def is_recording(self, rig_name: str) -> bool:
        snap = self.latest.get(rig_name) or {}
        return snap.get("phase") in RECORDING_PHASES

    def _already_landed(self, rig_name: str, batch: str) -> bool:
        s = self.store.get_session(rig_name, batch)
        return bool(s and s.get("state") == "landed")

    def _queue_trial(self, rig, d: dict) -> None:
        t, batch = d["trial"], d["batch_dir"]
        row = self.store.get_trial(rig.name, batch, int(t["index"]))
        if row and row.get("state") == "done":
            return  # replayed after a hub restart; staging was already moved

        def job():
            if self.transfer.process(rig, batch, d["output_dir"], t) == "failed":
                row = self.store.get_trial(rig.name, batch, int(t["index"])) or {}
                self.notifier.alert(f"{rig.label} transfer failed",
                                    f"{t.get('label') or t.get('phase_tag')}: {row.get('error', '')}")

        self.workers.submit(rig.name, ("trial", rig.name, batch, int(t["index"])), job)

    def request_finalize(self, rig, batch_dir, output_dir, push_to, manifest) -> None:
        cur = self.store.get_session(rig.name, batch_dir)
        if cur and cur.get("state") == "landed":
            return
        run = ((self.latest.get(rig.name) or {}).get("run") or {})
        if not run:
            raise ValueError(f"no status seen from {rig.name} yet; cannot confirm the session")
        if run.get("batch_dir") != batch_dir or run.get("output_dir") != output_dir:
            raise ValueError(f"{rig.name} is not reporting batch {batch_dir!r} at {output_dir!r}")
        self.store.set_session(rig.name, batch_dir, "pending")
        self.workers.submit(rig.name, ("finalize", rig.name, batch_dir),
                            lambda: self.finalizer.run(rig, batch_dir, output_dir, push_to, manifest))

    def abort(self, rig, batch_dir) -> str:
        return self.finalizer.abort(rig.name, batch_dir)

    def resolve_rig(self, host: str, client_ip: str):
        for r in self.cfg.rigs:
            if r.name == host:
                return r
        for r in self.cfg.rigs:
            if client_ip and r.host == client_ip:
                return r
        return None

    def _maybe_mount_check(self, rig, snap: dict) -> None:
        mc = self.cfg.mount_check
        run = snap.get("run") or {}
        if (not mc.enabled or snap.get("phase") != "WAITING" or run.get("state") != "running"
                or int(run.get("trials_done") or 0) > 0):
            return
        started, ends, pi_now = snap.get("started_at"), snap.get("ends_at"), snap.get("server_time")
        if started is None or ends is None or pi_now is None:
            return
        if pi_now - started < mc.delay_after_wait_start_s or ends - pi_now < mc.min_remaining_s:
            return
        batch = run.get("batch_dir") or ""
        if not batch or self.store.mount_check_done(rig.name, batch, "auto"):
            return
        self.mountcheck.start(rig, batch, "auto", stop=lambda: self._trials_started(rig.name))

    def _trials_started(self, rig_name: str) -> bool:
        snap = self.latest.get(rig_name) or {}
        run = snap.get("run") or {}
        return snap.get("phase") != "WAITING" or int(run.get("trials_done") or 0) > 0

    def start_mount_check(self, rig_name: str) -> int | None:
        rig = self.cfg.rig(rig_name)
        batch = ((self.latest.get(rig_name) or {}).get("run") or {}).get("batch_dir") or "no-batch"
        return self.mountcheck.start(rig, batch, "manual")

    def state(self) -> dict:
        mutes = self.store.mutes()
        rigs = {}
        for r in self.cfg.rigs:
            tr = self.trackers[r.name]
            rigs[r.name] = {
                "label": r.label, "live_url": r.live_url,
                "reachable": self.reachable[r.name], "offline": tr.offline,
                "last_seen": tr.last_seen, "fetched_at": self.fetched_at[r.name],
                "status": _public(self.latest[r.name]), "muted": mutes.get(r.name, False),
                "mount_check": _public_mc(self.store.latest_mount_check(r.name)),
            }
        return {"server_time": self.clock(), "rigs": rigs,
                "mount_check_enabled": self.mountcheck is not None,
                "mount_check_error": self.mount_check_error,
                "trials": self.store.trials(limit=150),
                "events": self.store.recent_events(limit=50)}

    async def poll_once(self, client: httpx.AsyncClient) -> None:
        snaps = await asyncio.gather(*(fetch_status(client, r) for r in self.cfg.rigs))
        now = self.clock()
        for r, snap in zip(self.cfg.rigs, snaps):
            try:
                self.ingest(r.name, snap, now)
            except Exception:
                log.exception("ingest %s failed", r.name)

    async def poll_forever(self) -> None:
        async with httpx.AsyncClient() as client:
            while True:
                await self.poll_once(client)
                await asyncio.sleep(self.cfg.poll_interval_s)
