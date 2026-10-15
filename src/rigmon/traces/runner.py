"""One trial trace: detector on the staged mp4 → per-fly px series → CSV + PNG; never raises."""

from __future__ import annotations

import csv
import logging
import math
import tempfile
from datetime import date
from pathlib import Path

from fbpipe.utils.rig_gates import gate_for_host

from ..gpu import GPU
from .detect_video import run_video_worker
from .plot import render_trace
from .series import build_series, odor_segments

log = logging.getLogger("rigmon.traces")


def _plain(name) -> bool:
    return isinstance(name, str) and bool(name) and "/" not in name and "\\" not in name and name not in (".", "..")


class TraceRunner:
    def __init__(self, cfg, store, settings, transfer, *, worker=run_video_worker, today=date.today, gpu=GPU):
        self.cfg, self.store, self.settings, self.transfer = cfg, store, settings, transfer
        self.worker, self.today, self.gpu = worker, today, gpu

    def run(self, rig, batch_dir: str, trial: dict):
        idx = int(trial["index"])
        try:
            video, frames_csv = trial.get("video"), trial.get("frames_csv")
            if not (_plain(batch_dir) and _plain(video) and _plain(frames_csv)):
                raise ValueError("refusing a non-plain trial name")
            staging = self.transfer.staging_dir(rig.name, batch_dir)
            stem = Path(video).stem
            mp4 = staging / f"{stem}.mp4"
            if not mp4.is_file():
                raise FileNotFoundError(f"no mp4 for {stem} in staging")
            tc = self.cfg.traces
            with tempfile.TemporaryDirectory(prefix="rigmon-trace-") as tmp:
                with self.gpu:   # one detector process hub-wide
                    res = self.worker(self.settings.model_path, mp4, Path(tmp) / "dets.json",
                                      timeout=tc.worker_timeout_s, stride=tc.frame_stride, conf=tc.conf)
            gate = gate_for_host(self.settings, rig.name, self.today())
            up = float(getattr(self.settings.proboscis_filter, "up_divisor", 4.0))
            series = build_series(res["frames"], float(res.get("fps") or 0.0), gate_px=gate, up_divisor=up)
            segments = odor_segments(staging / frames_csv)
            out = self.cfg.qc_root / rig.name / batch_dir / "traces"
            out.mkdir(parents=True, exist_ok=True)
            self._write_csv(out / f"{stem}.csv", series, segments)
            title = f"{rig.label} · {batch_dir} · {trial.get('label') or trial.get('phase_tag') or stem}"
            render_trace(series, segments, out / f"{stem}.png", title=title)
            self.store.set_trial_trace(rig.name, batch_dir, idx, trace=stem)
            return stem
        except Exception as e:  # a trace must never disturb the copy queue
            log.warning("trace %s/%s #%s failed: %r", rig.name, batch_dir, idx, e)
            try:
                self.store.set_trial_trace(rig.name, batch_dir, idx, error=str(e)[-300:] or repr(e))
            except Exception:
                log.exception("could not record trace failure")
            return None

    @staticmethod
    def _write_csv(path: Path, series, segments) -> None:
        flies = sorted(series.flies)
        with path.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["t_s", "odor", *[f"fly{f}_px" for f in flies]])
            for k, t in enumerate(series.t_s):
                odor = next((lab for on, off, lab in segments if on <= t < off), "")
                vals = [series.flies[f][k] for f in flies]
                w.writerow([f"{t:.3f}", odor, *["" if math.isnan(v) else f"{v:.1f}" for v in vals]])
