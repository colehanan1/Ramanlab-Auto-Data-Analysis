"""Per-trial metadata for the geometry pipeline.

The trial's output folder sits under ``<batch>/geometry/<trial>/`` -- one level
deeper than production -- so the sidecar pair is derived from the VIDEO path and
handed to :func:`fbpipe.utils.trial_metadata.load_trial_metadata` explicitly.
Two production behaviours are deliberately not inherited:

* a trial whose ``ActiveOFM`` never turns on (LightOnly) gets **no** odor window
  instead of the 30-60 s config default, and its stimulus window is the light;
* video frame ``k`` is aligned with sidecar ``FrameNumber k+1`` (the sidecar
  counts from 1; production indexes it from 0 and is one frame late).
"""
from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ..config import Settings, get_dataset_override
from ..utils.fly_type import fly_type_for_dir
from ..utils.timestamps import pick_frame_column, pick_timestamp_column, to_seconds_series
from ..utils.trial_metadata import load_trial_metadata, parse_active_ofm

_STEM_TS_RE = re.compile(r"_(\d{8})_(\d{6})$")
_PHASE_ORDER = {"pretest": 0, "training": 1, "testing": 2}


def trial_name_from_video(video: Path) -> str:
    """``output_september_02_batch_1_testing_3_Odor_20260902_123456`` ->
    ``september_02_batch_1_testing_3`` (fields 2..7, exactly as ``yolo_infer``)."""
    parts = Path(video).stem.split("_")
    return "_".join(parts[1:7]) if len(parts) >= 7 else Path(video).stem


def sidecar_pair_for_video(video: Path) -> Tuple[Optional[Path], Optional[Path]]:
    video = Path(video)
    txt, csv = video.with_suffix(".txt"), video.with_suffix(".csv")
    return (txt if txt.is_file() else None, csv if csv.is_file() else None)


def recorded_at_from_stem(stem: str) -> Optional[str]:
    m = _STEM_TS_RE.search(str(stem))
    return f"{m.group(1)}_{m.group(2)}" if m else None


@dataclass(frozen=True)
class TrialInfo:
    video: Optional[str]
    batch_dir: str
    dataset: str
    trial_name: str
    out_dir: str
    trial_type: str
    trial_index: int
    odor: Optional[str]
    label: str
    n_frames: int
    fps: float
    has_odor: bool
    odor_on_frame: Optional[int]
    odor_off_frame: Optional[int]
    light_on_s: Optional[float]
    light_off_s: Optional[float]
    fly_type: str
    sidecar_txt: Optional[str]
    sidecar_csv: Optional[str]
    recorded_at: Optional[str]

    @property
    def batch_name(self) -> str:
        return Path(self.batch_dir).name

    @property
    def odor_on_s(self) -> Optional[float]:
        return None if self.odor_on_frame is None else self.odor_on_frame / max(self.fps, 1e-6)

    @property
    def odor_off_s(self) -> Optional[float]:
        return None if self.odor_off_frame is None else self.odor_off_frame / max(self.fps, 1e-6)

    @property
    def duration_s(self) -> float:
        return self.n_frames / max(self.fps, 1e-6)

    @property
    def stimulus_window_s(self) -> Optional[Tuple[float, float]]:
        """(on, off) of the trial's stimulus: the odor when there is one, else the light."""
        if self.has_odor and self.odor_on_s is not None and self.odor_off_s is not None:
            return (self.odor_on_s, self.odor_off_s)
        if self.light_on_s is not None:
            off = self.light_off_s if self.light_off_s is not None else self.duration_s
            return (self.light_on_s, off)
        return None

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrialInfo":
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


def trial_info_for_video(video: Path, batch_dir: Path, out_dir: Path, settings: Settings, *,
                         n_frames_hint: Optional[int] = None) -> TrialInfo:
    video, batch_dir, out_dir = Path(video), Path(batch_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    txt, csv = sidecar_pair_for_video(video)
    meta = load_trial_metadata(
        out_dir,
        fps_default=settings.fps_default,
        odor_on_s_default=settings.odor_on_s,
        odor_off_s_default=settings.odor_off_s,
        n_frames_fallback=n_frames_hint,
        dataset_override=get_dataset_override(settings, batch_dir),
        known_datasets=tuple(settings.datasets),
        use_cache=False,
        sidecar_pair=(txt, csv),
    )
    has_odor = False
    on_f = off_f = None
    if csv is not None:
        on_f, off_f, channel = parse_active_ofm(csv)
        has_odor = channel is not None
    if not has_odor:
        on_f = off_f = None
    odor = meta.odor
    label = f"{meta.trial_type}_{meta.trial_index}" + (f"_{odor}" if odor else "")
    return TrialInfo(
        video=str(video), batch_dir=str(batch_dir), dataset=meta.dataset or "", trial_name=trial_name_from_video(video),
        out_dir=str(out_dir), trial_type=meta.trial_type, trial_index=int(meta.trial_index), odor=odor, label=label,
        n_frames=int(meta.n_frames), fps=float(meta.fps), has_odor=has_odor,
        odor_on_frame=None if on_f is None else int(on_f), odor_off_frame=None if off_f is None else int(off_f),
        light_on_s=meta.light_on_s, light_off_s=meta.light_off_s,
        fly_type=fly_type_for_dir(batch_dir, alert_new=False),
        sidecar_txt=None if txt is None else str(txt), sidecar_csv=None if csv is None else str(csv),
        recorded_at=recorded_at_from_stem(video.stem),
    )


def frame_alignment(sidecar_csv: Optional[Path], n_frames: int, fps: float) -> Tuple[np.ndarray, np.ndarray]:
    """Per video frame ``k``: seconds since trial start and whether odor was on,
    read from sidecar row ``FrameNumber == k+1``. Missing rows fall back to
    ``k / fps`` and ``False``."""
    n = int(n_frames)
    ts = np.arange(n, dtype=float) / max(float(fps), 1e-6)
    odor = np.zeros(n, dtype=bool)
    if sidecar_csv is None or not Path(sidecar_csv).is_file():
        return ts, odor
    try:
        df = pd.read_csv(sidecar_csv)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError):
        return ts, odor
    frame_col = pick_frame_column(df)
    if frame_col is None:
        return ts, odor
    frames = pd.to_numeric(df[frame_col], errors="coerce")
    ts_col = pick_timestamp_column(df)
    secs = to_seconds_series(df, ts_col) if ts_col is not None else None
    ofm_col = next((c for c in df.columns if c.lower() == "activeofm"), None)
    states = df[ofm_col].astype(str).str.strip().str.lower() if ofm_col else None
    for row, fn in enumerate(frames):
        if not np.isfinite(fn):
            continue
        k = int(fn) - 1
        if 0 <= k < n:
            if secs is not None and np.isfinite(secs.iloc[row]):
                ts[k] = float(secs.iloc[row])
            if states is not None:
                odor[k] = states.iloc[row] not in ("off", "", "nan", "none")
    return ts, odor


def protocol_sort_key(info: TrialInfo) -> tuple:
    """Chronological order (sidecar timestamp), then phase order, then index."""
    return (info.recorded_at or "", _PHASE_ORDER.get(info.trial_type, 9), int(info.trial_index))
