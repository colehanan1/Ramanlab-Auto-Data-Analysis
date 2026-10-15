"""One mount check: preview frames → detector worker → mount_qc → snapshot → store → notify."""

from __future__ import annotations

import json
import logging
import tempfile
import threading
import time
from datetime import date
from pathlib import Path

import cv2
import numpy as np

from fbpipe.utils.mount_qc import Detection, analyze, region_sharpness
from fbpipe.utils.rig_gates import gate_for_host

from .detect import run_worker
from .frames import decode, sample_preview
from .snapshot import render

log = logging.getLogger("rigmon.mountcheck")


def _thread(fn) -> None:
    threading.Thread(target=fn, daemon=True, name="rigmon-mountcheck").start()


def _plain(name: str) -> bool:
    return bool(name) and "/" not in name and "\\" not in name and name not in (".", "..")


class MountCheckRunner:
    def __init__(self, cfg, store, notifier, settings, *, sampler=sample_preview, worker=run_worker,
                 today=date.today, spawn=_thread):
        self.cfg, self.store, self.notifier, self.settings = cfg, store, notifier, settings
        self.sampler, self.worker, self.today, self.spawn = sampler, worker, today, spawn
        self._lock = threading.Lock()

    def start(self, rig, batch_dir: str, trigger: str, stop=lambda: False) -> int | None:
        batch_dir = batch_dir if _plain(batch_dir) else "no-batch"
        with self._lock:
            if self.store.mount_check_running(rig.name):
                return None
            check_id = self.store.add_mount_check(rig.name, batch_dir, trigger)
        self.spawn(lambda: self.run(rig, batch_dir, trigger, check_id, stop))
        return check_id

    def run(self, rig, batch_dir, trigger, check_id, stop=lambda: False) -> dict:
        mc = self.cfg.mount_check
        try:
            jpgs, complete = self.sampler(rig.live_url + "preview", mc.sample_s, mc.sample_fps, stop=stop)
            gate = gate_for_host(self.settings, rig.name, self.today())
            up = float(getattr(self.settings.proboscis_filter, "up_divisor", 4.0))
            if not jpgs:
                report, imgs, dets = analyze([], 0, 0, gate_px=gate, up_divisor=up, th=mc.th()), [], []
            else:
                with tempfile.TemporaryDirectory(prefix="rigmon-mc-") as tmp:
                    for i, j in enumerate(jpgs):
                        (Path(tmp) / f"{i:05d}.jpg").write_bytes(j)
                    res = self.worker(self.settings.model_path, tmp, Path(tmp) / "dets.json",
                                      timeout=mc.worker_timeout_s, conf=mc.conf)
                dets = [[Detection.from_dict(d) for d in f["dets"]] for f in res["frames"]]
                w, h = int(res["width"]), int(res["height"])
                first = analyze(dets, w, h, gate_px=gate, up_divisor=up, th=mc.th())
                imgs = [decode(jpgs[i]) for i in np.linspace(0, len(jpgs) - 1, min(20, len(jpgs))).astype(int)]
                grays = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in imgs]
                sharp = {s.slot: float(np.median([region_sharpness(g, *s.eye_xy, half=mc.sharpness_half_px)
                                                  for g in grays])) for s in first.slots}
                report = analyze(dets, w, h, gate_px=gate, up_divisor=up, th=mc.th(), sharpness=sharp,
                                 sharpness_reference=self.store.sharpness_reference(rig.name))
            state = "issues" if report.flags else ("ok" if complete else "incomplete")
            json_path, jpg_path = self._write(rig, batch_dir, trigger, report, jpgs, dets)
            sharpness = (float(np.median([s.sharpness for s in report.slots if s.sharpness is not None]))
                         if any(s.sharpness is not None for s in report.slots) else None)
            self.store.update_mount_check(
                check_id, state=state, flags=[f.__dict__ for f in report.flags],
                slots=[s.__dict__ for s in report.slots], sharpness=sharpness, frames=report.frames,
                json_path=str(json_path), jpg_path=str(jpg_path) if jpg_path else "")
            if state == "issues":
                self.notifier.mount_check(rig.label, state, [f.message for f in report.flags])
        except Exception as e:  # a check must never stay "running": every failure becomes an error row
            log.warning("mount check %s/%s failed: %r", rig.name, batch_dir, e)
            self.store.update_mount_check(check_id, state="error", error=str(e)[-600:])
            self.notifier.mount_check(rig.label, "error", [str(e)[-300:]])
        return self.store.latest_mount_check(rig.name)

    def _write(self, rig, batch_dir, trigger, report, jpgs, dets):
        out = self.cfg.qc_root / rig.name / batch_dir
        out.mkdir(parents=True, exist_ok=True)
        stem = "mount_check_" + time.strftime("%Y%m%d_%H%M%S")
        json_path = out / f"{stem}.json"
        payload = {"rig": rig.name, "label": rig.label, "batch_dir": batch_dir, "trigger": trigger,
                   "written_at": time.time(), "model": str(self.settings.model_path),
                   "thresholds": self.cfg.mount_check.th().__dict__, **report.to_dict()}
        json_path.write_text(json.dumps(payload, indent=1, default=str))
        if not jpgs:
            return json_path, None
        mid = len(jpgs) // 2
        jpg_path = out / f"{stem}.jpg"
        cv2.imwrite(str(jpg_path), render(decode(jpgs[mid]), dets[mid], report))
        return json_path, jpg_path
