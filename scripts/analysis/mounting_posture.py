#!/usr/bin/env python3
"""Mounting-posture and abdomen-movement tables for the new-glue flies.

    python scripts/analysis/mounting_posture.py --config config/config_new_abdomen.yaml \\
        --predictions-csv .../model_predictions.csv [--csv-out-dir DIR] [--out-dir DIR] [--abdomen-out-dir DIR]

Walks the live batches of the abdomen overlay on or after its cut-off date and joins, per fly-trial, the abdomen
geometry tables (``<abdomen out_base>/<dataset>/<batch>/geometry``) to the production track of the same fly
number. Writes to ``--csv-out-dir`` (default ``mounting.csv_out_dir``):

* ``fly_trial_posture.csv`` -- one row per fly-trial, every trial type: mounting pitch, proboscis rest, held-out
  state, spontaneous PER and pull-ins in eye units (pixels / ``eye_scale_px``), abdomen-movement metrics (the last
  ten columns, ``pb_window_min`` ... ``abd_straighten_events_during``, are the placebo design of the stimulus
  comparison: events in the last L s before the stimulus and in the first L s of it, both against the resting bend
  ``abd_rest_bend_loo_deg``, the median of the early baseline windows of the fly's OTHER trials), the score;
* ``fly_posture.csv``       -- one row per identity-consistent fly. Event rates divide pooled events by pooled
  window-minutes and proportions pool trials, so nothing is a mean of per-trial rates; the continuous abdomen
  measures are medians over the fly's pooled frames (they ride on every row of the fly's trial table as ``fly_*``
  columns, because only the frames can give them);
* ``exclusions.json``       -- what was left out and why (including ``pending_trials``: videos the abdomen stage has
  not tracked yet), under ``scoring`` how many pretest / testing fly-trials could carry a score and how many do, and
  under ``run_info`` what produced the tables: the time (UTC), every mounting setting used, and per
  rig the retracted reference used, how many flies it came from, whether the config set it, and a warning when most of
  those flies never rest below it.

Then, into ``--out-dir`` (default ``mounting.out_dir``), the six ``Mounting-Posture`` figures and ``report.md``
(``mounting_posture_figures``): exclusions, rho / p of every per-fly scatter, every permutation p, all labelled uncorrected.
Into ``--abdomen-out-dir`` (default ``mounting.abdomen_out_dir``) go the six ``Abdomen-Movement`` figures
(``abdomen_movement_figures``); their "Abdomen movement" section is part of the same ``report.md``.

Read-only on production, the secured mirror and the geometry tree: an output directory (flag or config) that is, or lies
inside, the base config's ``dataset_bases.data`` (flys_New), its secured mirror, ``/securedstorage`` or the hand-run
``Results/Figures`` is refused (exit 2) before anything is read or written. The definitions live in
``fbpipe.analysis.mounting``; the stimulus windows come from the geometry meta, never from
``load_trial_metadata`` (which writes a ``_trial_meta.json`` beside production on a cache miss).

A fly-trial is joined to production only when both are of ONE recording. A restarted session records a trial again under
the same name, so the sidecar the geometry meta names (``trial.sidecar_txt``, ``_preprocessed`` dropped) must be the one
production's ``_trial_meta.json`` names (read as plain JSON); a trial whose sessions differ is dropped
(``session_mismatch``), one with nothing to compare is kept and counted (``session_unverified``). A trial whose tracking
stopped before 99 % of its video (``n_frames_processed`` against ``n_frames_video`` in the geometry meta) is dropped
(``truncated_video``); production's table is never length-checked, it is routinely a few dozen frames shorter.
"""
import argparse
import dataclasses
import json
import sys
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (str(REPO_ROOT / "src"), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from fbpipe.analysis import mounting as M  # noqa: E402
from fbpipe.config import canon_fly_number, load_flagged_fly_exclusions, resolve_config_path  # noqa: E402
from fbpipe.config_layers import resolve_layers  # noqa: E402
from fbpipe.geometry import columns as C  # noqa: E402
from fbpipe.geometry.config import assert_out_tree_safe, load_geometry_config, read_only_roots  # noqa: E402
from fbpipe.geometry.steps import geometry_dir, iter_batch_jobs, pending_trials  # noqa: E402
from fbpipe.utils.rig_anchor import rig_token  # noqa: E402
from fbpipe.utils.rig_gates import read_batch_host  # noqa: E402
from fbpipe.utils.tables import read_schema_columns, read_table  # noqa: E402
from scripts.analysis import abdomen_movement_figures as AM  # noqa: E402
from scripts.analysis import mounting_posture_figures as FIG  # noqa: E402

GEOMETRY = {"eye_x": C.EYE_X, "eye_y": C.EYE_Y, "abd_x": C.ABD_X, "abd_y": C.ABD_Y, "tip_x": C.TIP_X, "tip_y": C.TIP_Y,
            "pro_x": C.PRO_X, "pro_y": C.PRO_Y, "bend": C.ABD_BEND}
PRODUCTION = ["frame", "x_class0", "y_class0", "eye_live_x", "eye_live_y", "x_class1", "y_class1", "x_anchor"]
# The fly-level continuous measures: medians / fractions over the fly's POOLED frames. They sit on every row of the
# fly's trial table (prefix ``fly_``) because only the frames themselves can produce them.
POOLED = ["abd_rest_bend_deg", "tip_pitch_rest_deg", "tip_speed_med_pre_eu_s", "tip_speed_p95_pre_eu_s",
          "tip_moving_frac_pre", "tip_speed_med_stim_eu_s", "tip_speed_p95_stim_eu_s", "tip_moving_frac_stim"]
ABDOMEN = ["abd_tracked_frac_pre", "abd_tracked_frac_stim", "abd_quality_ok", "abd_quality_ok_stim", "abd_rest_bend_deg",
           "abd_bend_sd_pre_deg", "abd_bend_sd_stim_deg", "abd_curl_events_pre", "abd_curl_events_stim",
           "abd_straighten_events_pre", "abd_straighten_events_stim", "abd_curl_delta_deg", "tip_pitch_rest_deg",
           "tip_speed_med_pre_eu_s", "tip_speed_p95_pre_eu_s", "tip_moving_frac_pre", "tip_speed_med_stim_eu_s",
           "tip_speed_p95_stim_eu_s", "tip_moving_frac_stim", "per_abdomen_coupling_rho"]
# The placebo design (``fbpipe.analysis.mounting.placebo_fly``): baseline / placebo / during windows cut from the onset,
# scored against the resting bend of the fly's other trials.
PLACEBO = list(M.PLACEBO_COLUMNS)
FLY_TRIAL_COLUMNS = [
    "dataset", "batch", "fly_number", "fly_id", "rig", "trial", "trial_type", "trial_index", "odor", "label", "stim_kind",
    "identity_consistent", "anchor_x", "anchor_y",
    "fps", "n_frames", "eye_scale_px", "onset_frame", "offset_frame", "pre_window_known", "pre_min", "stim_min",
    "trial_body_pitch_deg",
    "rest_eu", "rest_theta_deg", "retracted_ref_eu", "held_out", "held_subtype", "spont_per_events", "spont_per_any",
    "pullin_events_pre", "pullin_events_stim", "proboscis_rest_vs_body_deg", "score", "responded",
    *ABDOMEN, *(f"fly_{c}" for c in POOLED), *PLACEBO]
# Per-fly columns: (events, window-minutes) pairs for the pooled rates, (known, positive, fraction) for the proportions.
RATES = {"spont_per_rate_per_min": ("spont_per_events", "pre_min"),
         "pullin_rate_pre_per_min": ("pullin_events_pre", "pre_min"),
         "pullin_rate_stim_per_min": ("pullin_events_stim", "stim_min"),
         "abd_curl_rate_pre_per_min": ("abd_curl_events_pre", "pre_min"),
         "abd_straighten_rate_pre_per_min": ("abd_straighten_events_pre", "pre_min"),
         "abd_curl_rate_stim_per_min": ("abd_curl_events_stim", "stim_min"),
         "abd_straighten_rate_stim_per_min": ("abd_straighten_events_stim", "stim_min")}
PROPORTIONS = {"spont_per_any": ("n_spont_known", "n_spont_per", "spont_per_frac"),
               "held_out": ("n_held_known", "n_held_out", "held_out_frac"),
               "responded": ("n_scored", "n_responded", "response_rate")}
TRIAL_MEDIANS = ["abd_bend_sd_pre_deg", "abd_bend_sd_stim_deg", "abd_curl_delta_deg", "per_abdomen_coupling_rho"]
FLY_COLUMNS = ["dataset", "batch", "fly_number", "fly_id", "rig", "n_trials", "mount_pitch_deg", "mount_pitch_centred_deg",
               "rest_eu_min", "rest_eu_median", "spont_per_events", *(c for cols in PROPORTIONS.values() for c in cols),
               "held_subtype_dominant", "proboscis_state_dominant", *RATES, *TRIAL_MEDIANS, "n_coupling_trials", *POOLED]
STATE_ORDER = ("retracted", "held_down", "held_back", "held_forward")        # also the tie-break for the dominant state
COUNTS = ("n_batches", "n_trials", "n_fly_trials", "eye_mismatch", "non_square_frame", "warmup_slots",
          "no_production_table", "unreadable_geometry", "flagged_flies", "flagged_fly_trials", "truncated_video",
          "session_mismatch", "session_unverified")
PRODUCTION_META = "_trial_meta.json"      # production's per-trial metadata cache: only ever READ, and as plain JSON
SCOREABLE_TYPES = ("pretest", "testing")  # the trial types the model scores (training trials get no score)
TRUNCATED_BELOW = 0.99                    # a trial whose tracking stopped before this fraction of its video is dropped


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _finite(a) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    return a[np.isfinite(a)]


def _median(a) -> float:
    a = _finite(a)
    return float(np.median(a)) if a.size else np.nan


def _speed_stats(speed, moving_eu_s: float) -> tuple:
    """(median, 95th percentile, fraction above ``moving_eu_s``) of the finite speeds; NaNs without any."""
    s = _finite(speed)
    return (float(np.median(s)), float(np.quantile(s, 0.95)), float(np.mean(s > moving_eu_s))) if s.size else (np.nan,) * 3


@dataclasses.dataclass(frozen=True)
class Window:
    """A trial's stimulus window: ``pre`` / ``stim`` are slices clipped to the trial, ``onset`` / ``offset`` the meta's."""
    fps: float
    onset: Optional[int]
    offset: Optional[int]
    pre: slice
    stim: slice

    @property
    def known(self) -> bool:
        """A non-empty pre-stimulus window exists (a stimulus onset after the first frame)."""
        return self.pre.stop > self.pre.start

    def minutes(self, window: slice) -> float:
        return (window.stop - window.start) / self.fps / 60.0 if self.known else np.nan


def _window(info: dict, n: int, fps: float) -> Window:
    onset, offset = M.stimulus_frames(info, fps=fps)
    pre, stim = (slice(*w.indices(n)[:2]) for w in M.window_slice(n, onset, offset))
    return Window(fps, onset, offset, pre, stim)


def _table(path: Path, wanted) -> pd.DataFrame:
    """Only the *wanted* columns that the Parquet / CSV table has are read."""
    have = set(read_schema_columns(path))
    keep = [c for c in wanted if c in have]
    return read_table(path, **({"columns": keep} if path.suffix.lower() == ".parquet" else {"usecols": keep}))


def _read_meta(path: Path) -> Optional[dict]:
    """A trial's geometry meta with a positive ``fps``, or None when it cannot be used."""
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
        meta["fps"] = float(meta.get("fps") or meta["trial"]["fps"])
        return meta if isinstance(meta["trial"], dict) and meta["fps"] > 0 else None
    except (OSError, ValueError, KeyError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Production tracks
# ---------------------------------------------------------------------------

def _production_folders(job, gcfg, trial: str) -> list:
    """The folders production's output for *trial* can be in, best first: its own trial folder, then the same folder in
    each secured mirror."""
    return [job.prod_dir / trial, *(Path(b) / job.root_name / job.name / trial for b in gcfg.video_bases)]


def _session_stem(path) -> str:
    """The recording session a sidecar (or video) path names: its stem with any ``_preprocessed`` dropped, the way
    ``fbpipe.utils.yolo_discovery`` reads a session."""
    return Path(str(path)).stem.replace("_preprocessed", "")


def _production_session(job, gcfg, trial: str) -> Optional[str]:
    """The session production processed *trial* from: the stem of ``sidecar_txt`` in the first ``_trial_meta.json`` of
    ``_production_folders`` that names one; None when none does. A plain ``json.load``: ``load_trial_metadata`` would
    WRITE a ``_trial_meta.json`` beside production on a cache miss, and this analysis writes nothing there."""
    for folder in _production_folders(job, gcfg, trial):
        try:
            with open(folder / PRODUCTION_META, encoding="utf-8") as fh:
                sidecar = json.load(fh).get("sidecar_txt")
        except (OSError, ValueError, AttributeError):
            continue                                  # absent, unreadable, not JSON, not an object
        if isinstance(sidecar, str) and sidecar.strip():
            return _session_stem(sidecar)
    return None


def _session_state(job, gcfg, trial: str, meta: dict, known: dict) -> str:
    """``"same"``, ``"different"`` or ``"unknown"``: whether the video the geometry tracked is the recording production
    processed. A restarted session records a trial again under one trial name; the video tracked here and the tables and
    score production holds are then of one recording only if the sidecar named in the geometry meta is the one named in
    production's ``_trial_meta.json``. ``"unknown"`` when either side names none. *known* memoises per trial (a trial's
    flies share it)."""
    key = (job.root_name, job.name, trial)
    if key not in known:
        sidecar = (meta.get("trial") or {}).get("sidecar_txt")
        geometry = _session_stem(sidecar) if isinstance(sidecar, str) and sidecar.strip() else None
        production = _production_session(job, gcfg, trial)
        if geometry is None or production is None:
            known[key] = "unknown"
        else:
            known[key] = "same" if geometry == production else "different"
    return known[key]


def _truncated(meta: dict) -> bool:
    """Whether tracking stopped short: the geometry meta's ``n_frames_processed`` is under ``TRUNCATED_BELOW`` of its
    ``n_frames_video``, both known and positive. (Production's table is not compared: it is routinely a few dozen frames
    shorter than the video, which is no truncation.)"""
    try:
        processed, video = float(meta["n_frames_processed"]), float(meta["n_frames_video"])
    except (KeyError, TypeError, ValueError):
        return False
    return processed > 0 and video > 0 and processed < TRUNCATED_BELOW * video


def _production_table(job, gcfg, trial: str, slot: int) -> Optional[pd.DataFrame]:
    """The primary ``<trial>_fly{N}_distances`` table (never ``RMS_calculations/updated_*``, which lack ``x_anchor`` and
    ``eye_live_*``): the production trial folder first, then the same folder in each secured mirror."""
    for folder in _production_folders(job, gcfg, trial):
        for suffix in (".parquet", ".csv"):
            try:
                path = folder / f"{trial}_fly{slot}_distances{suffix}"
                if path.is_file():
                    table = _table(path, PRODUCTION)
                    if {"x_class0", "y_class0", "x_class1", "y_class1", "x_anchor"} <= set(table.columns):
                        return table
            except (OSError, ValueError):
                continue
    return None


def _aligned(prod: pd.DataFrame, n: int) -> dict:
    """Each production column as a length-``n`` array indexed by video frame (its ``frame`` column, else row order)."""
    frame = prod["frame"].to_numpy(float) if "frame" in prod else np.arange(len(prod), dtype=float)
    keep = np.isfinite(frame) & (frame >= 0) & (frame < n)
    out = {}
    for name in PRODUCTION[1:]:
        out[name] = np.full(n, np.nan)
        if name in prod:
            out[name][frame[keep].astype(int)] = prod[name].to_numpy(float)[keep]
    return out


def _head_frame(p: dict, scale: float) -> tuple:
    """(eye -> proboscis distance in eye units, theta in degrees); the live eye where it was measured, else the anchor."""
    live = np.isfinite(p["eye_live_x"]) & np.isfinite(p["eye_live_y"])
    ex, ey = np.where(live, p["eye_live_x"], p["x_class0"]), np.where(live, p["eye_live_y"], p["y_class0"])
    dist_px, theta = M.head_frame_vector(ex, ey, p["x_class1"], p["y_class1"], p["x_anchor"])
    return dist_px / scale, theta


# ---------------------------------------------------------------------------
# One fly-trial
# ---------------------------------------------------------------------------

def _rest_vs_body(g: dict, rest: np.ndarray) -> float:
    """Angle between the body axis (abdomen -> eye) and eye -> proboscis, from the median vectors over the rest frames."""
    use = rest & np.all([np.isfinite(g[k]) for k in ("eye_x", "eye_y", "abd_x", "abd_y", "pro_x", "pro_y")], axis=0)
    if not use.any():
        return np.nan
    body = [np.median(g["eye_x"][use] - g["abd_x"][use]), np.median(g["eye_y"][use] - g["abd_y"][use])]
    head = [np.median(g["pro_x"][use] - g["eye_x"][use]), np.median(g["pro_y"][use] - g["eye_y"][use])]
    return M.angle_between_deg(body, head)


def _proboscis(dist: np.ndarray, theta: np.ndarray, g: dict, win: Window, mcfg) -> tuple:
    """Rest, spontaneous PER and pull-ins of one trial (NaN without a pre-window or a rest) and its pre-stimulus distance."""
    level = M.rest_level(dist, q=mcfg.rest_quantile)
    rest = np.isfinite(dist) & (dist <= level)                          # all False when the level is unknown
    out = {"rest_eu": _median(dist[rest]), "rest_theta_deg": _median(theta[rest]),
           "proboscis_rest_vs_body_deg": _rest_vs_body(g, rest),
           "spont_per_events": np.nan, "pullin_events_pre": np.nan, "pullin_events_stim": np.nan}
    if win.known and np.isfinite(out["rest_eu"]):
        frames = max(1, round(mcfg.event_min_s * win.fps))
        up, down = out["rest_eu"] + mcfg.event_rise_eu, out["rest_eu"] - mcfg.event_drop_eu
        out["spont_per_events"] = len(M.runs_above(dist, up, min_frames=frames, end=win.pre.stop))
        out["pullin_events_pre"] = len(M.runs_below(dist, down, min_frames=frames, end=win.pre.stop))
        out["pullin_events_stim"] = len(M.runs_below(dist, down, min_frames=frames, start=win.stim.start, end=win.stim.stop))
    out["spont_per_any"] = np.nan if np.isnan(out["spont_per_events"]) else float(out["spont_per_events"] > 0)
    return out, dist[win.pre].copy()                                      # a copy: the whole trial is not kept alive


def _abdomen(g: dict, dist: np.ndarray, scale: float, win: Window, mcfg) -> tuple:
    """Abdomen-movement columns of one trial, and the finite frames the fly-level pools take.

    A window whose bend is tracked in fewer than ``abd_min_tracked_frac`` of its frames is unusable, and so is every
    window when the resting bend (the pre-window median) is: those columns stay NaN, never zero. (The placebo columns need
    the fly's other trials and are filled in a second pass, ``_placebo_columns``.)
    """
    bend, out, pooled, ok = g["bend"], dict.fromkeys(ABDOMEN, np.nan), {}, {}
    for name, sl in (("pre", win.pre), ("stim", win.stim)):
        out[f"abd_tracked_frac_{name}"] = float(np.isfinite(bend[sl]).mean()) if bend[sl].size else np.nan
        ok[name] = bool(bend[sl].size) and out[f"abd_tracked_frac_{name}"] >= mcfg.abd_min_tracked_frac
    rest = _median(bend[win.pre]) if ok["pre"] else np.nan
    ok["pre"] = ok["pre"] and bool(np.isfinite(rest))
    ok["stim"] = ok["stim"] and ok["pre"]
    out.update(abd_quality_ok=ok["pre"], abd_quality_ok_stim=ok["stim"])
    if not ok["pre"]:
        return out, pooled
    speed = M.point_speed_eu_s(g["tip_x"], g["tip_y"], eye_scale_px=scale, fps=win.fps,
                               smooth_frames=mcfg.speed_smooth_frames)
    pitch = M.tip_pitch_deg(g["abd_x"], g["abd_y"], g["tip_x"], g["tip_y"])
    out.update(abd_rest_bend_deg=rest, tip_pitch_rest_deg=_median(pitch[win.pre]),
               per_abdomen_coupling_rho=M.coupling_rho(dist, bend))
    pooled.update(abd_rest_bend_deg=_finite(bend[win.pre]), tip_pitch_rest_deg=_finite(pitch[win.pre]))
    frames = max(1, round(mcfg.curl_min_s * win.fps))
    for name, sl in (("pre", win.pre), ("stim", win.stim)):
        if not ok[name]:
            continue
        curls, straightens = M.curl_events(bend, rest, rise_deg=mcfg.curl_rise_deg, min_frames=frames,
                                           start=sl.start, end=sl.stop)
        med, p95, moving = _speed_stats(speed[sl], mcfg.tip_move_eu_s)
        out.update({f"abd_bend_sd_{name}_deg": float(np.std(_finite(bend[sl]))), f"abd_curl_events_{name}": curls,
                    f"abd_straighten_events_{name}": straightens, f"tip_speed_med_{name}_eu_s": med,
                    f"tip_speed_p95_{name}_eu_s": p95, f"tip_moving_frac_{name}": moving})
        pooled[f"tip_speed_{name}"] = _finite(speed[sl])
    if ok["stim"]:
        out["abd_curl_delta_deg"] = _median(bend[win.stim]) - rest
    return out, pooled


def _first_finite_pair(x: np.ndarray, y: np.ndarray) -> tuple:
    ok = np.isfinite(x) & np.isfinite(y)
    return (float(x[ok][0]), float(y[ok][0])) if ok.any() else (np.nan, np.nan)


def _fly_trial(job, rig: str, trial: str, meta: dict, slot: int, path: Path, gcfg, mcfg, exc: dict,
               sessions: Optional[dict] = None) -> Optional[tuple]:
    """One geometry slot of one trial joined to its production track: ``(row, pre-stimulus distance, pooled frames,
    placebo trace)``.

    None when the slot drops out (counted in ``exc``): tracking that stopped short of its video (``truncated_video``), a
    video of another recording than the one production processed (``session_mismatch``: a restarted session records the
    trial again under the same name), unreadable geometry, no production table, or a geometry eye further than
    ``eye_match_px`` from the production anchor. A trial with no session to compare (production has no
    ``_trial_meta.json`` for it, or either side names no sidecar) is kept and counted in ``session_unverified``.
    Only a 1080-square frame (``sx == sy == 1``) has eye units; any other keeps its angle metrics and gets NaN for every
    eye-unit one. *sessions* memoises the session check per trial.
    """
    if _truncated(meta):
        exc["truncated_video"] += 1
        return None
    session = _session_state(job, gcfg, trial, meta, {} if sessions is None else sessions)
    if session == "different":
        exc["session_mismatch"] += 1
        return None
    try:
        table = _table(path, [C.FRAME, C.EYE_SCALE, *GEOMETRY.values()])
    except (OSError, ValueError):
        table = None
    if table is None or table.empty:
        exc["unreadable_geometry"] += 1
        return None
    n, info, frame = len(table), meta["trial"], meta.get("frame") or {}
    g = {k: table[c].to_numpy(float) if c in table else np.full(n, np.nan) for k, c in GEOMETRY.items()}
    square = frame.get("sx") == 1 and frame.get("sy") == 1
    if not square:
        exc["non_square_frame"] += 1
    geometry_scale = _median(table[C.EYE_SCALE]) if C.EYE_SCALE in table else np.nan
    scale = geometry_scale if square and geometry_scale > 0 else np.nan
    dist, theta, anchor = np.full(n, np.nan), np.full(n, np.nan), (np.nan, np.nan)
    if square:
        prod = _production_table(job, gcfg, trial, slot)
        if prod is None:
            exc["no_production_table"] += 1
            return None
        prod = _aligned(prod, n)
        anchor = _first_finite_pair(prod["x_class0"], prod["y_class0"])
        eye = (_median(g["eye_x"]), _median(g["eye_y"]))
        if not np.hypot(eye[0] - anchor[0], eye[1] - anchor[1]) <= mcfg.eye_match_px:         # also True for NaN
            exc["eye_mismatch"] += 1
            return None
        dist, theta = _head_frame(prod, scale)
    win = _window(info, n, meta["fps"])
    state, pre_dist = _proboscis(dist, theta, g, win, mcfg)
    abdomen, pooled = _abdomen(g, dist, scale, win, mcfg)
    has_odor, has_light = bool(info.get("has_odor")), _finite([info.get("light_on_s")]).size > 0
    row = {"dataset": job.root_name, "batch": job.name, "fly_number": int(slot),
           "fly_id": f"{job.root_name}/{job.name}/fly{slot}", "rig": rig, "trial": trial,
           "trial_type": info.get("trial_type"), "trial_index": int(info.get("trial_index", -1)),
           "odor": info.get("odor"), "label": info.get("label"),
           "stim_kind": "odor+light" if has_odor and has_light else "odor" if has_odor else "light" if has_light else "none",
           "anchor_x": anchor[0], "anchor_y": anchor[1], "fps": win.fps, "n_frames": n, "eye_scale_px": geometry_scale,
           "onset_frame": np.nan if win.onset is None else win.onset,
           "offset_frame": np.nan if win.offset is None else win.offset,
           "pre_window_known": win.known, "pre_min": win.minutes(win.pre), "stim_min": win.minutes(win.stim),
           "trial_body_pitch_deg": _median(M.body_pitch_deg(g["eye_x"], g["eye_y"], g["abd_x"], g["abd_y"])[win.pre]),
           **state, **abdomen}
    trace = (M.PlaceboTrial(g["bend"].copy(), win.onset, win.offset, win.fps) if win.known
             else M.PlaceboTrial(np.empty(0), None, None, win.fps))       # a copy: the table is not kept alive for it
    if session == "unknown":
        exc["session_unverified"] += 1                                    # counted once the fly-trial is kept
    return row, pre_dist, pooled, trace


# ---------------------------------------------------------------------------
# The fly-trial table
# ---------------------------------------------------------------------------

def _mark_identity(df: pd.DataFrame, mcfg, exc: dict) -> None:
    """``identity_consistent``: the fly's production anchors stay within ``identity_tol_px`` of each other across trials."""
    spread = {fid: M.anchor_spread_px(g[["anchor_x", "anchor_y"]].to_numpy()) for fid, g in df.groupby("fly_id", sort=False)}
    df["identity_consistent"] = df["fly_id"].map(spread) <= mcfg.identity_tol_px
    exc["identity_drift"] = [
        {"fly_id": fid, "dataset": g["dataset"].iloc[0], "batch": g["batch"].iloc[0],
         "fly_number": int(g["fly_number"].iloc[0]), "spread_px": round(spread[fid], 1), "n_trials": len(g)}
        for fid, g in df.groupby("fly_id", sort=False) if spread[fid] > mcfg.identity_tol_px]


def _placebo_columns(df: pd.DataFrame, traces: list, mcfg) -> None:
    """The ten placebo columns (``fbpipe.analysis.mounting.PLACEBO_COLUMNS``) in two passes per fly: every trial lends the
    early baseline of its pre-stimulus window, then each trial is scored against the pooled baselines of the OTHER trials
    of the same fly (``placebo_fly``: at least two of them, each tracked in ``abd_min_tracked_frac`` of its frames; never
    the trial itself, never a placebo or a during frame). A fly whose identity is not consistent pools nothing: all ten
    columns stay NaN (``abd_base_ok`` False)."""
    columns = {name: np.full(len(df), np.nan) for name in M.PLACEBO_COLUMNS}
    columns["abd_base_ok"] = np.zeros(len(df), dtype=bool)
    consistent = df["identity_consistent"].to_numpy(dtype=bool)
    for rows in df.groupby("fly_id", sort=False).indices.values():
        if not consistent[rows[0]]:
            continue
        scored = M.placebo_fly([traces[i] for i in rows], rise_deg=mcfg.curl_rise_deg, min_s=mcfg.curl_min_s,
                               min_tracked_frac=mcfg.abd_min_tracked_frac)
        for i, out in zip(rows, scored):
            for name in M.PLACEBO_COLUMNS:
                columns[name][i] = out[name]
    for name, values in columns.items():
        df[name] = values


def _fly_lowest_rest(df: pd.DataFrame) -> pd.Series:
    """Each identity-consistent fly's smallest trial ``rest_eu`` (NaN where it has none), indexed by (rig, fly_id)."""
    return df[df["identity_consistent"]].groupby(["rig", "fly_id"], sort=False)["rest_eu"].min()


def _rig_reference(df: pd.DataFrame, mcfg) -> pd.Series:
    """The retracted reference per row: the configured value, else per rig the median over its identity-consistent flies
    of each fly's smallest trial rest (NaN for a rig without any)."""
    if mcfg.retracted_ref_eu is not None:
        return pd.Series(float(mcfg.retracted_ref_eu), index=df.index)
    fly_min = _fly_lowest_rest(df)
    reference = {rig: M.retracted_reference(g.to_numpy()) for rig, g in fly_min.groupby(level="rig", sort=False)}
    return df["rig"].map(reference).astype(float)


def reference_info(df: pd.DataFrame, mcfg) -> dict:
    """The retracted reference of every rig as the held-out call used it, and a warning where it looks wrong.

    ``{"retracted_ref": {rig: {...}}, "warnings": [text, ...]}``; per rig: ``retracted_ref_eu`` (None without any),
    ``n_flies`` (identity-consistent flies with a known rest: the ones a derived reference is the median of), ``from_config``
    (``mounting.retracted_ref_eu`` was set), ``n_flies_at_held_level`` (flies whose smallest trial rest is at or above the
    reference plus ``held_margin_eu``) and ``warning``. The warning is set when MORE than half of the rig's flies are at that
    level even in their most retracted trial: the reference may then sit at the held-out level, and held-out flies would be
    labelled retracted. (A reference derived as the median cannot trip it with a positive margin; a configured one can.)
    """
    rigs, warnings = {}, []
    if len(df):
        lowest = _fly_lowest_rest(df)
        for rig in sorted(df["rig"].dropna().unique(), key=str):
            mins = lowest.xs(rig, level="rig").to_numpy(float) if rig in lowest.index.get_level_values("rig") else np.empty(0)
            mins = mins[np.isfinite(mins)]
            from_config = mcfg.retracted_ref_eu is not None
            ref = float(mcfg.retracted_ref_eu) if from_config else M.retracted_reference(mins)
            cutoff = ref + mcfg.held_margin_eu
            at_level = int(np.count_nonzero(mins >= cutoff)) if np.isfinite(cutoff) else 0
            warn = bool(mins.size and at_level > mins.size / 2)
            rigs[str(rig)] = {"retracted_ref_eu": ref if np.isfinite(ref) else None, "n_flies": int(mins.size),
                              "from_config": from_config, "n_flies_at_held_level": at_level, "warning": warn}
            if warn:
                warnings.append(
                    f"{rig}: {at_level} of {mins.size} identity-consistent flies have their lowest trial rest at or above "
                    f"the retracted reference plus held_margin_eu ({ref:.2f} + {mcfg.held_margin_eu:g} = {cutoff:.2f} eu); "
                    "the reference may sit at the held-out level, and held-out flies would be labelled retracted")
    return {"retracted_ref": rigs, "warnings": warnings}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def settings_record(mcfg) -> dict:
    """Every ``MountingSettings`` value as plain JSON data (a date as its ISO string)."""
    values = {f.name: getattr(mcfg, f.name) for f in dataclasses.fields(mcfg)}
    return {name: value.isoformat() if isinstance(value, date) else value for name, value in values.items()}


def run_info(df: pd.DataFrame, mcfg) -> dict:
    """What produced these tables, for ``exclusions.json`` and the head of ``report.md``: the time (UTC), every mounting
    setting used, and the retracted reference of each rig (:func:`reference_info`)."""
    return {"generated_utc": _utc_now().astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "settings": settings_record(mcfg), **reference_info(df, mcfg)}


def _classify_held(df: pd.DataFrame, pre_dist: list, mcfg) -> None:
    """``held_out`` (1 / 0) and ``held_subtype``; both stay unknown without a pre-stimulus frame, rest or rig reference."""
    held, subtype = np.full(len(df), np.nan), [None] * len(df)
    columns = (df["rest_eu"].to_numpy(), df["retracted_ref_eu"].to_numpy(), df["rest_theta_deg"].to_numpy())
    for i, (dist, rest, ref, theta) in enumerate(zip(pre_dist, *columns)):
        if not (np.isfinite(rest) and np.isfinite(ref) and np.isfinite(dist).any()):    # dist is empty without a pre-window
            continue
        held[i] = float(M.held_out(dist, ref_eu=ref, margin_eu=mcfg.held_margin_eu, frac_min=mcfg.held_frac_min))
        if not held[i]:
            subtype[i] = "retracted"
        elif np.isfinite(theta):                        # held_subtype(NaN) would say "held_down": not measured, so unknown
            subtype[i] = M.held_subtype(theta, back_deg=mcfg.held_back_deg, forward_deg=mcfg.held_forward_deg)
    df["held_out"], df["held_subtype"] = held, pd.Series(subtype, index=df.index, dtype=object)


def _attach_scores(df: pd.DataFrame, predictions_csv, mcfg, exc: dict) -> None:
    """``score`` and ``responded`` (``score >= binary_threshold``) from model_predictions.csv; NaN where there is none.

    The trial is parsed from the label (``pretest_3_...`` / ``testing_2_...``): the file's own ``trial_type`` column
    says "testing" for ``training_*`` rows too, and those must never reach a testing trial.
    """
    df["score"], df["responded"] = np.nan, np.nan
    exc["predictions_missing"] = not (predictions_csv and Path(predictions_csv).is_file())
    if exc["predictions_missing"]:
        return
    pred, counts = M.dedupe_predictions(pd.read_csv(predictions_csv))
    exc.update({f"prediction_{k}": v for k, v in counts.items()})
    parsed = pred["trial_label"].astype(str).str.extract(r"^(pretest|testing)_(\d+)")
    key = ["dataset", "fly", "fly_number", "trial_type", "trial_index"]
    scores = pred.assign(trial_type=parsed[0], trial_index=pd.to_numeric(parsed[1]),
                         fly_number=pred["fly_number"].map(canon_fly_number)).dropna(subset=["trial_type", "trial_index"])
    scores = scores.astype({"trial_index": int})[key + ["score"]].drop_duplicates()
    clash = scores.duplicated(key, keep=False)                        # two labels of one trial with different scores
    exc["prediction_key_conflicts"] = int(scores.loc[clash, key].drop_duplicates().shape[0])
    left = df[["dataset", "batch", "fly_number", "trial_type", "trial_index"]].rename(columns={"batch": "fly"})
    left = left.assign(fly_number=left["fly_number"].map(canon_fly_number), trial_index=left["trial_index"].astype(int))
    df["score"] = left.merge(scores[~clash], how="left", on=key)["score"].to_numpy(float)
    df["responded"] = np.where(np.isfinite(df["score"]), (df["score"] >= mcfg.binary_threshold).astype(float), np.nan)


def scoring_counts(df: pd.DataFrame) -> dict:
    """How many fly-trials of the table could carry a score and how many do, per trial type: ``{trial type: {"scoreable":
    pretest / testing fly-trials in scope (training trials get none), "scored": those with a score,
    "scored_identity_consistent": those of identity-consistent flies}}``. The last is what the pooled bars of the report
    can count; the first two cover every row of the table."""
    out = {}
    for kind in SCOREABLE_TYPES:
        rows = df[df["trial_type"] == kind] if len(df) else df
        scored = rows[rows["score"].notna()] if len(rows) else rows
        consistent = scored[scored["identity_consistent"].astype(bool)] if len(scored) else scored
        out[kind] = {"scoreable": int(len(rows)), "scored": int(len(scored)),
                     "scored_identity_consistent": int(len(consistent))}
    return out


def _pooled_fly_values(chunks: dict, moving_eu_s: float) -> dict:
    cat = {k: np.concatenate(v) for k, v in chunks.items() if v}
    out = {"abd_rest_bend_deg": _median(cat.get("abd_rest_bend_deg", [])),
           "tip_pitch_rest_deg": _median(cat.get("tip_pitch_rest_deg", []))}
    for name in ("pre", "stim"):
        speeds = _speed_stats(cat.get(f"tip_speed_{name}", []), moving_eu_s)
        out.update(zip((f"tip_speed_med_{name}_eu_s", f"tip_speed_p95_{name}_eu_s", f"tip_moving_frac_{name}"), speeds))
    return out


def _tracked_slots(job, gcfg, exc: dict):
    """``(trial, meta, slot, table path)`` of every tracked fly-trial of a batch. A trial folder without a geometry meta is
    not tracked yet and is skipped; a meta that cannot be used is counted."""
    gdir = geometry_dir(job.out_dir, gcfg)
    for tdir in sorted(p for p in gdir.iterdir() if p.is_dir()) if gdir.is_dir() else ():
        if not C.meta_path(tdir, tdir.name).is_file():
            continue
        meta = _read_meta(C.meta_path(tdir, tdir.name))
        if meta is None:
            exc["unreadable_geometry"] += 1
            continue
        exc["n_trials"] += 1
        for slot, path in C.iter_geometry_tables(tdir):
            yield tdir.name, meta, slot, path


def _scope(gcfg, mcfg):
    """*gcfg* with the later of its own and the mounting block's cut-off date: the batches this analysis reads."""
    cutoffs = [d for d in (gcfg.batch_date_from, mcfg.batch_date_from) if d is not None]
    return dataclasses.replace(gcfg, batch_date_from=max(cutoffs)) if cutoffs else gcfg


def build_fly_trial_table(settings, gcfg, mcfg, predictions_csv) -> tuple:
    """``(fly_trial, exclusions)``: one row per tracked fly-trial of the live batches on or after the cut-off date.

    ``settings`` / ``gcfg`` come from ``load_geometry_config`` of the abdomen overlay, ``mcfg`` from
    ``fbpipe.analysis.mounting.parse_mounting_block``. Only trials with a geometry meta are read; a slot is joined to
    production fly N of the same trial, and counted in ``exclusions`` when it drops out (or, for ``identity_drift``,
    is kept but flagged ``identity_consistent=False``).
    """
    exc = {**dict.fromkeys(COUNTS, 0), "identity_drift": [], "predictions_csv": str(predictions_csv or ""),
           "predictions_missing": not (predictions_csv and Path(predictions_csv).is_file()),
           "prediction_exact_duplicates": 0, "prediction_conflicts": 0, "prediction_same_score_duplicates": 0,
           "prediction_key_conflicts": 0}
    flagged = load_flagged_fly_exclusions(settings.flagged_flies_csv) if settings.flagged_flies_csv else set()
    scope = _scope(gcfg, mcfg)
    rows, pre_dist, traces, chunks, flagged_seen = [], [], [], defaultdict(lambda: defaultdict(list)), set()
    sessions: dict = {}                                                   # per trial: do geometry and production agree?
    for job in iter_batch_jobs(settings, scope, live_only=True):
        exc["n_batches"] += 1
        rig = read_batch_host(job.prod_dir) or rig_token(job.name) or "unknown"
        for trial, meta, slot, path in _tracked_slots(job, gcfg, exc):
            key = (job.root_name, job.name, canon_fly_number(slot))
            if key in flagged:
                flagged_seen.add(key)
                exc["flagged_fly_trials"] += 1
            elif gcfg.slot_seeds == "production" and meta.get("slot_source") != "production":
                exc["warmup_slots"] += 1                                  # numbered by warm-up order: not production's fly N
            elif (joined := _fly_trial(job, rig, trial, meta, slot, path, gcfg, mcfg, exc, sessions)) is not None:
                row, pre, pooled, trace = joined
                rows.append(row)
                pre_dist.append(pre)
                traces.append(trace)
                for name, frames in pooled.items():
                    chunks[row["fly_id"]][name].append(frames)
    exc["flagged_flies"], exc["n_fly_trials"] = len(flagged_seen), len(rows)
    if not rows:
        empty = pd.DataFrame(columns=FLY_TRIAL_COLUMNS)
        exc["run_info"] = run_info(empty, mcfg)
        exc["scoring"] = scoring_counts(empty)
        return empty, exc
    df = pd.DataFrame(rows)
    _mark_identity(df, mcfg, exc)
    _placebo_columns(df, traces, mcfg)
    traces.clear()
    df["retracted_ref_eu"] = _rig_reference(df, mcfg)
    exc["run_info"] = run_info(df, mcfg)
    _classify_held(df, pre_dist, mcfg)
    _attach_scores(df, predictions_csv, mcfg, exc)
    exc["scoring"] = scoring_counts(df)
    per_fly = {fid: _pooled_fly_values(c, mcfg.tip_move_eu_s) for fid, c in chunks.items()}
    for name in POOLED:
        df[f"fly_{name}"] = df["fly_id"].map({fid: v[name] for fid, v in per_fly.items()})
    return df.reindex(columns=FLY_TRIAL_COLUMNS), exc


# ---------------------------------------------------------------------------
# The fly table
# ---------------------------------------------------------------------------

def _dominant(states: pd.Series):
    """The most common state; a tie goes to the earlier one in ``STATE_ORDER``. None without any."""
    counts = states.value_counts()
    ranked = [s for s in STATE_ORDER if s in counts.index]
    return max(ranked, key=counts.get) if ranked else None


def _fly_row(g: pd.DataFrame) -> dict:
    first = g.iloc[0]
    row = {c: first[c] for c in ("dataset", "batch", "fly_number", "fly_id", "rig")}
    events = g["spont_per_events"].dropna()
    row.update(n_trials=len(g), mount_pitch_deg=_median(g["trial_body_pitch_deg"]), rest_eu_min=g["rest_eu"].min(),
               rest_eu_median=_median(g["rest_eu"]), spont_per_events=float(events.sum()) if len(events) else np.nan)
    for state, (n_known, n_positive, fraction) in PROPORTIONS.items():      # proportions pool the fly's trials
        known = g[state].dropna()
        row.update({n_known: len(known), n_positive: int(known.sum()),
                    fraction: float(known.mean()) if len(known) else np.nan})
    states = g["held_subtype"].dropna()
    row.update(proboscis_state_dominant=_dominant(states), held_subtype_dominant=_dominant(states[states != "retracted"]))
    for name, (events_col, minutes_col) in RATES.items():              # rates pool events over the minutes they came from
        used = g[events_col].notna() & g[minutes_col].notna()
        minutes = g.loc[used, minutes_col].sum()
        row[name] = float(g.loc[used, events_col].sum() / minutes) if minutes > 0 else np.nan
    row.update({c: _median(g[c]) for c in TRIAL_MEDIANS})
    row["n_coupling_trials"] = int(g["per_abdomen_coupling_rho"].notna().sum())
    row.update({c: first[f"fly_{c}"] if f"fly_{c}" in g else np.nan for c in POOLED})
    return row


def build_fly_table(fly_trial: pd.DataFrame, mcfg) -> pd.DataFrame:
    """One row per identity-consistent fly (``mcfg`` is accepted for symmetry: every setting is already in the rows).

    Rates pool events over window-minutes (only windows with a count), proportions pool trials, ``mount_pitch_deg`` is
    the median of the fly's ``trial_body_pitch_deg`` and ``mount_pitch_centred_deg`` that minus the rig's median over flies.
    """
    keep = fly_trial[fly_trial["identity_consistent"].astype(bool)] if len(fly_trial) else fly_trial
    if keep.empty:
        return pd.DataFrame(columns=FLY_COLUMNS)
    fly = pd.DataFrame([_fly_row(g) for _, g in keep.groupby("fly_id", sort=False)], columns=FLY_COLUMNS)
    fly["mount_pitch_centred_deg"] = fly["mount_pitch_deg"] - fly.groupby("rig")["mount_pitch_deg"].transform("median")
    return fly


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def _refused_out_dirs(base_path, requested: dict) -> list:
    """Why each of the output directories in *requested* (``{"--out-dir": (path, "mounting.out_dir"), ...}``; a path that
    is empty is not an output) may not be written: it is, or lies inside, the base config's production tree
    (``dataset_bases.data``, flys_New), its secured mirror or ``/securedstorage`` (``assert_out_tree_safe``), or it lies
    inside the hand-run ``Results/Figures``. Messages name the flag, or the config key it came from."""
    roots, reasons = read_only_roots(base_path), []
    for flag, (path, source) in requested.items():
        if not path:
            continue
        name = f"{flag} ({source})" if source else flag
        try:
            assert_out_tree_safe(Path(path), roots, what=f"[mounting] {name}", who="the mounting analysis",
                                 advice="Give its outputs a directory of their own: figures under "
                                        "Results/New-Opto-Fly-Figures, tables under Data/CSVs-New-Opto-Flys.")
        except ValueError as err:
            reasons.append(str(err))
            continue
        parts = Path(path).expanduser().resolve().parts
        if any(pair == ("Results", "Figures") for pair in zip(parts, parts[1:])):
            reasons.append(f"[mounting] {name} {path} is inside Results/Figures, which only hand-run figure sets may use")
    return reasons


def _count_pending(settings, scope) -> dict:
    """``{"pending_trials": n}``: trial videos of the in-scope batches the abdomen stage would still track. A walk that
    fails is recorded (``pending_trials`` None + ``pending_trials_error``) and never stops the tables, figures or report."""
    try:
        return {"pending_trials": int(pending_trials(settings, scope))}
    except Exception as err:                      # noqa: BLE001  (whatever the walk hit, the analysis goes on)
        print(f"[MOUNTING] WARNING: could not count the pending trials: {type(err).__name__}: {err}")
        return {"pending_trials": None, "pending_trials_error": f"{type(err).__name__}: {err}"}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="the abdomen overlay (an extends-overlay of config_new.yaml)")
    p.add_argument("--predictions-csv", default=None, help="model_predictions.csv (default: the base config's "
                   "reaction_prediction.output_csv)")
    p.add_argument("--out-dir", default=None, help="figure directory (default: mounting.out_dir): the six "
                   "Mounting-Posture figures and report.md are written here")
    p.add_argument("--abdomen-out-dir", default=None, help="figure directory for the six Abdomen-Movement figures "
                   "(default: mounting.abdomen_out_dir); their section of report.md is written to --out-dir")
    p.add_argument("--csv-out-dir", default=None, help="where the tables go (default: mounting.csv_out_dir)")
    a = p.parse_args(argv)
    path = resolve_config_path(a.config)
    overlay = resolve_layers(path).overlay
    settings, gcfg = load_geometry_config(path)
    if not gcfg.out_base:
        p.error(f"{path} has no abdomen: block; this analysis reads the abdomen overlay's output tree")
    mcfg = M.parse_mounting_block(overlay["mounting"] if "mounting" in overlay else {},
                                  default_batch_date_from=gcfg.batch_date_from)
    csv_dir = a.csv_out_dir or mcfg.csv_out_dir
    if not csv_dir:
        p.error("no output directory: pass --csv-out-dir or set mounting.csv_out_dir")
    fig_dir, abdomen_dir = a.out_dir or mcfg.out_dir, a.abdomen_out_dir or mcfg.abdomen_out_dir
    refused = _refused_out_dirs(resolve_layers(path).base_path, {
        "--csv-out-dir": (csv_dir, "" if a.csv_out_dir else "mounting.csv_out_dir"),
        "--out-dir": (fig_dir, "" if a.out_dir else "mounting.out_dir"),
        "--abdomen-out-dir": (abdomen_dir, "" if a.abdomen_out_dir else "mounting.abdomen_out_dir")})
    if refused:
        p.error("; ".join(refused))                  # exit 2, before anything is read or written
    predictions = a.predictions_csv or settings.reaction_prediction.output_csv
    fly_trial, exclusions = build_fly_trial_table(settings, gcfg, mcfg, predictions)
    fly = build_fly_table(fly_trial, mcfg)
    exclusions.update(_count_pending(settings, _scope(gcfg, mcfg)))
    out = Path(csv_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    fly_trial.to_csv(out / "fly_trial_posture.csv", index=False)
    fly.to_csv(out / "fly_posture.csv", index=False)
    (out / "exclusions.json").write_text(json.dumps(exclusions, indent=2, default=str), encoding="utf-8")
    if exclusions["predictions_missing"]:
        print(f"[MOUNTING] WARNING: no predictions file at {predictions!r}; the scores are left empty")
    print(f"[MOUNTING] {len(fly_trial)} fly-trial(s), {len(fly)} fly/flies -> {out}; excluded: "
          f"{exclusions['eye_mismatch']} eye mismatch, {len(exclusions['identity_drift'])} identity drift, "
          f"{exclusions['flagged_flies']} flagged fly/flies, {exclusions['no_production_table']} without a production table")
    if not fig_dir and not abdomen_dir:
        print("[MOUNTING] no figure directory (--out-dir / mounting.out_dir, --abdomen-out-dir / mounting.abdomen_out_dir): "
              "figures and report.md skipped")
        return 0
    section, abdomen = AM.abdomen_outputs(fly_trial, fly, abdomen_dir, n_perm=mcfg.n_permutations,
                                          min_tracked_frac=mcfg.abd_min_tracked_frac)
    if abdomen is not None:
        print(f"[MOUNTING] {len(abdomen)} abdomen figure(s) -> {abdomen_dir}")
    if not fig_dir:
        print("[MOUNTING] no report directory (--out-dir / mounting.out_dir): report.md skipped")
        return 0
    fig_dir = Path(fig_dir).expanduser()
    stats = FIG.compute_stats(fly_trial, fly, n_perm=mcfg.n_permutations)
    figures = FIG.render_figures(fly_trial, fly, fig_dir, n_perm=mcfg.n_permutations, stats=stats)
    report = FIG.write_report(fly_trial, fly, exclusions, stats, fig_dir, figures=figures, extra_sections=[section])
    print(f"[MOUNTING] {len(figures)} figure(s) and {report.name} -> {fig_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
