"""Verify a red LED light stimulus is visible on-screen for its sensor-commanded duration.

The Pi rig's sensors log (``sensors_output_*.csv``) records when the LED
*driver* was told to turn on/off (see ``trial_metadata.parse_light_window_seconds``).
That is a software-side record: it says nothing about whether the LED
*physically* lit up for the whole commanded window (loose wiring, a bad LDD-L
driver, etc. can make it drop out early). This module decodes the trial video
and tracks the red channel to confirm the physical stimulus actually matches
what the sensors say was commanded.

Detection method: for each sampled frame, ``redness = R - (G+B)/2`` isolates a
red-shifted light source from general (white/room) brightness. Turning the LED
on produces a sharp, single-frame step in this signal, so a simple auto
threshold (midpoint of the 5th/95th percentile) cleanly separates on/off.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

LOGGER = logging.getLogger("fbpipe.light_stimulus")

CACHE_FILENAME = "_light_check.json"


@dataclass
class FrameSample:
    t: float
    redness: float  # R - (G+B)/2


def sample_video(
    video_path: Path, roi: Optional[tuple[int, int, int, int]] = None, stride: int = 1
) -> list[FrameSample]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        cap.release()
        raise ValueError(f"Could not read a valid fps from {video_path}")

    samples: list[FrameSample] = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % stride == 0:
            crop = frame[roi[1]:roi[3], roi[0]:roi[2]] if roi is not None else frame
            b_mean, g_mean, r_mean = crop[..., 0].mean(), crop[..., 1].mean(), crop[..., 2].mean()
            samples.append(FrameSample(t=frame_idx / fps, redness=r_mean - (g_mean + b_mean) / 2.0))
        frame_idx += 1
    cap.release()
    return samples


def detect_on_intervals(
    samples: list[FrameSample], threshold: Optional[float] = None
) -> tuple[list[tuple[float, float]], float]:
    """Return (contiguous (start_t, end_t) on-intervals, threshold used)."""
    if not samples:
        return [], 0.0
    times = np.array([s.t for s in samples])
    vals = np.array([s.redness for s in samples])
    if threshold is None:
        lo, hi = np.percentile(vals, [5, 95])
        threshold = (lo + hi) / 2.0

    on_mask = vals > threshold
    intervals = []
    start = None
    for i, on in enumerate(on_mask):
        if on and start is None:
            start = times[i]
        elif not on and start is not None:
            intervals.append((start, times[i - 1]))
            start = None
    if start is not None:
        intervals.append((start, times[-1]))
    return intervals, threshold


def check_expected_window(
    samples: list[FrameSample],
    expected_on: tuple[float, float],
    threshold: Optional[float] = None,
    min_fraction_on: float = 0.9,
) -> dict:
    """Check that redness reads ON for (at least) ``min_fraction_on`` of ``expected_on``.

    A genuine dropout is the light going OFF before the window ends (an
    ON->OFF transition followed by staying off through the window's end) — not
    the LED's brief onset ramp, which shows up as a few OFF frames right at
    the start of the window and is expected/harmless.
    """
    intervals, threshold = detect_on_intervals(samples, threshold)
    on_start, on_end = expected_on

    times = np.array([s.t for s in samples])
    vals = np.array([s.redness for s in samples])
    in_window = (times >= on_start) & (times <= on_end)
    window_vals = vals[in_window]
    window_times = times[in_window]
    on_in_window = window_vals > threshold

    fraction_on = float(on_in_window.mean()) if len(on_in_window) else 0.0

    dropout_t = None
    never_on = False
    if len(on_in_window) and not on_in_window[-1]:
        idx = len(on_in_window) - 1
        while idx >= 0 and not on_in_window[idx]:
            idx -= 1
        if idx >= 0:
            dropout_t = float(window_times[idx + 1])
        else:
            never_on = True

    return {
        "threshold": float(threshold),
        "detected_on_intervals": intervals,
        "expected_on": expected_on,
        "fraction_on_in_window": fraction_on,
        "passed": fraction_on >= min_fraction_on,
        "dropout_time_s": dropout_t,
        "never_on": never_on,
    }


def grab_frame(video_path: Path, t: float, roi: Optional[tuple[int, int, int, int]] = None):
    """Decode the frame nearest time t (seconds) and return it as RGB (or None if out of range)."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 40.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = int(round(t * fps))
    if frame_idx < 0 or frame_idx >= n_total:
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    if roi is not None:
        cv2.rectangle(frame, (roi[0], roi[1]), (roi[2], roi[3]), (0, 255, 0), 3)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _apply_style():
    try:
        from ..plot_style import apply_lab_style
        apply_lab_style()
    except Exception:
        pass


def plot_timeseries(samples: list[FrameSample], result: dict, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _apply_style()

    times = [s.t for s in samples]
    redness = [s.redness for s in samples]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(times, redness, color="firebrick", lw=0.8, label="redness (R - (G+B)/2)")
    ax.axhline(result["threshold"], color="gray", ls="--", lw=0.8, label="on/off threshold")
    on_start, on_end = result["expected_on"]
    ax.axvspan(on_start, on_end, color="#ffd24d", alpha=0.30, label="expected ON window (sensors)")
    if result["dropout_time_s"] is not None:
        ax.axvline(result["dropout_time_s"], color="black", ls=":", lw=1.2,
                    label=f"dropout @ {result['dropout_time_s']:.1f}s")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("redness")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"Light stimulus check — {'PASS' if result['passed'] else 'FAIL'} "
                 f"({result['fraction_on_in_window']*100:.1f}% on in window)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_transitions(samples: list[FrameSample], result: dict, out_path: Path, window_s: float = 4.0) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _apply_style()

    times = np.array([s.t for s in samples])
    redness = np.array([s.redness for s in samples])
    on_start, on_end = result["expected_on"]
    actual_off_t = result["dropout_time_s"] if result["dropout_time_s"] is not None else on_end

    edges = [("onset", on_start), ("offset (actual)", actual_off_t)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (label, edge_t) in zip(axes, edges):
        mask = (times >= edge_t - window_s) & (times <= edge_t + window_s)
        ax.plot(times[mask], redness[mask], color="firebrick", lw=1.0, marker=".", ms=3)
        ax.axhline(result["threshold"], color="gray", ls="--", lw=0.8)
        ax.axvline(edge_t, color="black", ls=":", lw=1.2)
        ax.set_title(f"{label} @ {edge_t:.2f}s")
        ax.set_xlabel("time (s)")
    axes[0].set_ylabel("redness")
    fig.suptitle(f"Zoomed transitions — {'PASS' if result['passed'] else 'FAIL'}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_frame_montage(video_path: Path, result: dict, roi: Optional[tuple[int, int, int, int]], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _apply_style()

    on_start, on_end = result["expected_on"]
    dur = on_end - on_start
    candidate_times = [on_start - 5.0, on_start + 1.0, on_start + dur * 0.5, on_end - 1.0, on_end + 1.0, on_end + 10.0]

    fig, axes = plt.subplots(2, 3, figsize=(13, 9))
    for ax, t in zip(axes.flat, candidate_times):
        t = max(0.0, t)
        frame = grab_frame(video_path, t, roi=roi)
        if frame is None:
            ax.axis("off")
            ax.set_title(f"t={t:.1f}s (out of range)")
            continue
        ax.imshow(frame)
        ax.axis("off")
        state = "expected ON" if on_start <= t <= on_end else "expected OFF"
        ax.set_title(f"t={t:.1f}s ({state})", fontsize=10)
    fig.suptitle(f"Frame montage — {'PASS' if result['passed'] else 'FAIL'} "
                 f"(threshold redness={result['threshold']:.1f})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# High-level per-trial check with caching
# ---------------------------------------------------------------------------

def find_trial_video_candidates(trial_dir: Path) -> list[Path]:
    """All existing videos for a trial, in preference order.

    Preference order: the YOLO-annotated video inside the trial folder, then
    (if ``move_videos`` already relocated it) the ``videos_with_rms`` staging
    copy, then the raw sibling video in the batch folder. Callers should fall
    through the list when a candidate turns out to be undecodable (e.g. a
    header-only stub left behind by a failed yolo_infer run).
    """
    trial_dir = Path(trial_dir)
    candidates: list[Path] = []
    annotated = trial_dir / f"{trial_dir.name}_distance_annotated.mp4"
    if annotated.exists():
        candidates.append(annotated)

    batch = trial_dir.parent
    import re
    m = re.search(r"(pretest|training|testing)_(\d+)$", trial_dir.name, re.IGNORECASE)
    if m:
        phase = m.group(1).lower()
        staged = batch / "videos_with_rms" / phase / annotated.name
        if staged.exists():
            candidates.append(staged)

    raw = sorted(batch.glob(f"output_{trial_dir.name}_*.mp4"))
    if raw:
        candidates.append(max(raw, key=lambda p: p.stat().st_mtime))
    return candidates


def find_trial_video(trial_dir: Path) -> Optional[Path]:
    """Resolve the best available video for a trial (first candidate)."""
    candidates = find_trial_video_candidates(trial_dir)
    return candidates[0] if candidates else None


def _cache_path(trial_dir: Path) -> Path:
    return trial_dir / CACHE_FILENAME


def check_trial_light_stimulus(
    trial_dir: Path,
    expected_on: tuple[float, float],
    *,
    window_source: str,
    stride: int = 2,
    min_fraction_on: float = 0.9,
    roi: Optional[tuple[int, int, int, int]] = None,
    force: bool = False,
    plots_dir: Optional[Path] = None,
    extra_columns: Optional[dict] = None,
) -> dict:
    """Run (or load cached) light-stimulus check for one trial.

    Returns a flat dict suitable for a CSV row. ``status`` is one of
    ``"checked"`` (produced a pass/fail verdict), ``"no_video"`` (video could
    not be resolved) or ``"unreadable_video"`` (every candidate video exists
    but none could be decoded) — the latter two are recorded but not flagged
    as light failures.
    """
    trial_dir = Path(trial_dir)
    candidates = find_trial_video_candidates(trial_dir)
    base_row = {
        **(extra_columns or {}),
        "trial_dir": str(trial_dir),
        "window_source": window_source,
        "expected_on_s": expected_on[0],
        "expected_off_s": expected_on[1],
    }
    if not candidates:
        return {**base_row, "status": "no_video", "video_path": None, "passed": None, "from_cache": False}

    def _key(path: Path) -> dict:
        return {
            "video_path": str(path),
            "video_mtime": path.stat().st_mtime,
            "expected_on": list(expected_on),
            "stride": stride,
        }

    cache_path = _cache_path(trial_dir)
    if not force and cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if any(cached.get("_cache_key") == _key(c) for c in candidates):
                row = dict(cached["row"])
                row["from_cache"] = True
                return row
        except (json.JSONDecodeError, OSError, KeyError):
            pass

    samples, video_path = None, None
    for candidate in candidates:
        try:
            decoded = sample_video(candidate, roi=roi, stride=stride)
        except (FileNotFoundError, ValueError):
            continue
        if decoded:
            samples, video_path = decoded, candidate
            break
    if samples is None:
        LOGGER.warning("No decodable video among %d candidate(s) for %s", len(candidates), trial_dir)
        return {
            **base_row,
            "status": "unreadable_video",
            "video_path": str(candidates[0]),
            "passed": None,
            "from_cache": False,
        }

    cache_key = _key(video_path)
    result = check_expected_window(samples, expected_on, min_fraction_on=min_fraction_on)

    row = {
        **base_row,
        "status": "checked",
        "video_path": str(video_path),
        "fps_stride": stride,
        "threshold": result["threshold"],
        "fraction_on_in_window": result["fraction_on_in_window"],
        "dropout_time_s": result["dropout_time_s"],
        "never_on": result["never_on"],
        "passed": result["passed"],
        "from_cache": False,
    }

    if not result["passed"] and plots_dir is not None:
        try:
            plots_dir.mkdir(parents=True, exist_ok=True)
            stem = trial_dir.name
            plot_timeseries(samples, result, plots_dir / f"{stem}_timeseries.png")
            plot_transitions(samples, result, plots_dir / f"{stem}_transitions.png")
            plot_frame_montage(video_path, result, roi, plots_dir / f"{stem}_frames.png")
            row["diagnostic_plots_dir"] = str(plots_dir)
        except Exception as exc:  # pragma: no cover - plotting is best-effort
            LOGGER.warning("Failed to write diagnostic plots for %s: %s", trial_dir, exc)

    try:
        cache_path.write_text(
            json.dumps({"_cache_key": cache_key, "row": row}, indent=2), encoding="utf-8"
        )
    except OSError as exc:
        LOGGER.debug("Failed to write light-check cache for %s: %s", trial_dir, exc)

    return row


# ---------------------------------------------------------------------------
# Aggregated CSV of all checked trials (upserted by trial_dir)
# ---------------------------------------------------------------------------

def update_light_check_csv(csv_path: Path, rows: list[dict]) -> list[dict]:
    """Upsert ``rows`` (keyed by ``trial_dir``) into ``csv_path``.

    Returns the subset of ``rows`` that are *newly flagged failures*: trials
    that failed this run and either weren't in the CSV before or previously
    passed. Callers use this to decide what to ntfy about, so re-running over
    unchanged videos never re-notifies.
    """
    import pandas as pd

    checked_rows = [r for r in rows if r.get("status") == "checked"]
    if not checked_rows:
        return []

    new_df = pd.DataFrame(checked_rows).set_index("trial_dir", drop=False)

    prior_passed: dict[str, bool] = {}
    if csv_path.exists():
        old_df = pd.read_csv(csv_path)
        if "trial_dir" in old_df.columns:
            prior_passed = dict(zip(old_df["trial_dir"], old_df["passed"]))
            old_df = old_df.set_index("trial_dir", drop=False)
            combined = pd.concat([old_df, new_df])
            combined = combined[~combined.index.duplicated(keep="last")]
        else:
            combined = new_df
    else:
        combined = new_df

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    combined.sort_index().to_csv(csv_path, index=False)

    newly_flagged = [
        r for r in checked_rows
        if r.get("passed") is False and prior_passed.get(r["trial_dir"]) is not False
    ]
    return newly_flagged


__all__ = [
    "FrameSample",
    "sample_video",
    "detect_on_intervals",
    "check_expected_window",
    "grab_frame",
    "plot_timeseries",
    "plot_transitions",
    "plot_frame_montage",
    "find_trial_video",
    "find_trial_video_candidates",
    "check_trial_light_stimulus",
    "update_light_check_csv",
]
