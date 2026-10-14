"""The hub: poll every rig, turn snapshots into events, and fan them out to
the store, desktop notifications and the per-rig transfer queue."""

from __future__ import annotations

import asyncio
import logging
import time

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


class Hub:
    def __init__(self, cfg, store, notifier, transfer, finalizer, workers, clock=time.time):
        self.cfg, self.store, self.notifier = cfg, store, notifier
        self.transfer, self.finalizer, self.workers, self.clock = transfer, finalizer, workers, clock
        self.trackers = {r.name: RigTracker(r.name, cfg.offline_after_s) for r in cfg.rigs}
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
            if ev.kind != "phase_changed":
                self.store.add_event(ev)
            self.notifier.handle(ev, rig.label)
            if ev.kind == "trial_finished":
                self.store.upsert_trial(rig_name, ev.data["batch_dir"], ev.data["trial"])
                if self.cfg.transfer_enabled:
                    self._queue_trial(rig, ev.data)
        return events

    def _queue_trial(self, rig, d: dict) -> None:
        t, batch = d["trial"], d["batch_dir"]

        def job():
            if self.transfer.process(rig, batch, d["output_dir"], t) == "failed":
                row = self.store.get_trial(rig.name, batch, int(t["index"])) or {}
                self.notifier.alert(f"{rig.label} transfer failed",
                                    f"{t.get('label') or t.get('phase_tag')}: {row.get('error', '')}")

        self.workers.submit(rig.name, ("trial", rig.name, batch, int(t["index"])), job)

    def request_finalize(self, rig, batch_dir, output_dir, push_to, manifest) -> None:
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

    def state(self) -> dict:
        mutes = self.store.mutes()
        rigs = {}
        for r in self.cfg.rigs:
            tr = self.trackers[r.name]
            rigs[r.name] = {
                "label": r.label, "live_url": r.live_url,
                "reachable": self.reachable[r.name], "offline": tr.offline,
                "last_seen": tr.last_seen, "fetched_at": self.fetched_at[r.name],
                "status": self.latest[r.name], "muted": mutes.get(r.name, False),
            }
        return {"server_time": self.clock(), "rigs": rigs,
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
