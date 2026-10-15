#!/usr/bin/env python3
"""A pretend Pi for the rig monitor: serves /status.json on localhost.

Library use (tests): FakeRig(status_dict) → .port, .stop(); mutate .status.
Demo: python scripts/rigmon/fake_rig.py --port 18008 — plays a looping
3-trial session so the dashboard can be eyeballed without a rig. Add
{name: fakepi, label: "Fake", ip: local, port: 18008} to config/rigmon.yaml.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class FakeRig:
    def __init__(self, status: dict, port: int = 0, preview_jpegs=None,
                 preview_interval: float = 0.05, preview_silent: bool = False):
        self.status = status
        self.preview_jpegs = preview_jpegs
        self.preview_interval = preview_interval
        self.preview_silent = preview_silent
        self._stopped = False
        rig = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                if self.path == "/preview" and (rig.preview_jpegs or rig.preview_silent):
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.end_headers()
                    try:
                        if rig.preview_silent:
                            time.sleep(30)   # connected, never sends a frame
                            return
                        i = 0
                        while not rig._stopped:
                            jpg = rig.preview_jpegs[i % len(rig.preview_jpegs)]
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
                            i += 1
                            time.sleep(rig.preview_interval)
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    return
                if self.path != "/status.json":
                    self.send_response(404)
                    self.end_headers()
                    return
                body = json.dumps({**rig.status, "server_time": time.time()}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self._httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self._httpd.daemon_threads = True
        self.port = self._httpd.server_address[1]
        threading.Thread(target=self._httpd.serve_forever, daemon=True).start()

    def stop(self) -> None:
        self._stopped = True
        self._httpd.shutdown()
        self._httpd.server_close()


def _demo(port: int) -> None:
    t0 = time.time()
    rig = FakeRig({}, port)
    phases = [("BASELINE", 30), ("STIMULUS", 10), ("POST", 20), ("GAP", 60)]
    period = sum(d for _, d in phases)
    print(f"fake rig on http://127.0.0.1:{rig.port}/status.json — Ctrl-C to stop")
    while True:
        el = (time.time() - t0) % (3 * period)
        n, into = int(el // period), el % period
        acc = 0
        for phase, dur in phases:
            if into < acc + dur:
                break
            acc += dur
        start = time.time() - (into - acc)
        rig.status = {
            "phase": phase, "label": "", "trial_label": f"Training {n + 1}/3", "odor": "ACV",
            "stim": "ODOR" if phase == "STIMULUS" else "", "started_at": start, "ends_at": start + dur,
            "session": {}, "run": {
                "state": "running", "batch_dir": "demo_batch_1", "output_dir": "/nonexistent",
                "push_mode": "legacy", "planned_total_s": 3 * period,
                "expected_end_at": t0 + 3 * period * (1 + (time.time() - t0) // (3 * period)),
                "trials_done": n, "trials_total": 3, "trials": []}}
        time.sleep(0.5)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18008)
    _demo(ap.parse_args().port)
