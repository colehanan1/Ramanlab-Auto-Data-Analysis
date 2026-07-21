"""Verify a red LED light stimulus is visible on-screen for its expected duration.

Samples frame pixel values from a recorded trial video and checks that the red
channel is elevated (LED on) for the full expected on-window, rather than trusting
sensor/GPIO logs alone. Useful for catching cases where the LED was commanded on
but physically failed partway through (e.g. loose wiring, driver dropout).

Saves 3 diagnostic plots per video into figures/light_stimulus_check/<video_stem>_*.png:
  - timeseries: full-trial redness trace with expected ON window and any detected dropout
  - transitions: zoomed views around the onset and offset edges
  - frames: a montage of actual decoded video frames at key timestamps, so the
    redness signal can be visually cross-checked against what the LED actually looks
    like on screen (not just trusted as an abstract signal)

Usage:
    python scripts/check_light_stimulus.py VIDEO --on 35 60
    python scripts/check_light_stimulus.py VIDEO --on 35 60 --roi 300 300 480 480
    python scripts/check_light_stimulus.py VIDEO --on 35 60 --no-plot
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTDIR = REPO_ROOT / "figures" / "light_stimulus_check"


@dataclass
class FrameSample:
    t: float
    red_mean: float
    redness: float  # R - (G+B)/2, isolates a red-shifted light from general brightness


def sample_video(video_path: str, roi: tuple[int, int, int, int] | None, stride: int = 1) -> list[FrameSample]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        raise ValueError(f"Could not read a valid fps from {video_path}")

    samples: list[FrameSample] = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % stride == 0:
            if roi is not None:
                x0, y0, x1, y1 = roi
                crop = frame[y0:y1, x0:x1]
            else:
                crop = frame
            # OpenCV loads BGR
            b_mean, g_mean, r_mean = crop[..., 0].mean(), crop[..., 1].mean(), crop[..., 2].mean()
            redness = r_mean - (g_mean + b_mean) / 2.0
            samples.append(FrameSample(t=frame_idx / fps, red_mean=r_mean, redness=redness))
        frame_idx += 1
    cap.release()
    return samples


def detect_on_intervals(samples: list[FrameSample], threshold: float | None = None) -> tuple[list[tuple[float, float]], float]:
    """Return (contiguous (start_t, end_t) on-intervals, threshold used).

    threshold: absolute redness cutoff. If None, auto-picked as the midpoint between
    the 5th and 95th percentile of the redness signal (assumes a clearly bimodal
    on/off signal, which a directly-lit red LED against a room-lit background produces).
    """
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


def check_expected_window(samples: list[FrameSample], expected_on: tuple[float, float], threshold: float | None = None,
                           min_fraction_on: float = 0.9) -> dict:
    intervals, threshold = detect_on_intervals(samples, threshold)
    on_start, on_end = expected_on

    times = np.array([s.t for s in samples])
    vals = np.array([s.redness for s in samples])
    in_window = (times >= on_start) & (times <= on_end)
    window_vals = vals[in_window]
    window_times = times[in_window]
    on_in_window = window_vals > threshold

    fraction_on = on_in_window.mean() if len(on_in_window) else 0.0

    # A genuine dropout is the light going OFF before the window ends (an ON->OFF
    # transition followed by staying off through on_end) — not the LED's brief onset
    # ramp, which shows up as a few OFF frames right at on_start and is expected.
    dropout_t = None
    never_on = False
    if len(on_in_window) and not on_in_window[-1]:
        idx = len(on_in_window) - 1
        while idx >= 0 and not on_in_window[idx]:
            idx -= 1
        if idx >= 0:
            dropout_t = window_times[idx + 1]
        else:
            never_on = True

    return {
        "threshold": float(threshold),
        "detected_on_intervals": intervals,
        "expected_on": expected_on,
        "fraction_on_in_window": float(fraction_on),
        "passed": fraction_on >= min_fraction_on,
        "dropout_time_s": float(dropout_t) if dropout_t is not None else None,
        "never_on": never_on,
    }


def _apply_style():
    try:
        from fbpipe.plot_style import apply_lab_style
        apply_lab_style()
    except Exception:
        pass


def grab_frame(video_path: str, t: float, roi: tuple[int, int, int, int] | None = None):
    """Decode the frame nearest time t (seconds) and return it as RGB (or None if out of range)."""
    cap = cv2.VideoCapture(video_path)
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
        x0, y0, x1, y1 = roi
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 3)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


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
    ax.axvspan(on_start, on_end, color="#ffd24d", alpha=0.30, label="expected ON window")
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


def plot_frame_montage(video_path: str, result: dict, roi: tuple[int, int, int, int] | None, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _apply_style()

    on_start, on_end = result["expected_on"]
    dur = on_end - on_start
    candidate_times = [
        on_start - 5.0,
        on_start + 1.0,
        on_start + dur * 0.5,
        on_end - 1.0,
        on_end + 1.0,
        on_end + 10.0,
    ]

    fig, axes = plt.subplots(2, 3, figsize=(13, 9))
    threshold = result["threshold"]
    for ax, t in zip(axes.flat, candidate_times):
        t = max(0.0, t)
        frame = grab_frame(video_path, t, roi=roi)
        if frame is None:
            ax.axis("off")
            ax.set_title(f"t={t:.1f}s (out of range)")
            continue
        ax.imshow(frame)
        ax.axis("off")
        in_window = on_start <= t <= on_end
        state = "expected ON" if in_window else "expected OFF"
        ax.set_title(f"t={t:.1f}s ({state})", fontsize=10)
    fig.suptitle(f"Frame montage — {'PASS' if result['passed'] else 'FAIL'} "
                 f"(threshold redness={threshold:.1f})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video", help="Path to the (annotated or raw) trial video")
    p.add_argument("--on", nargs=2, type=float, metavar=("START_S", "END_S"), required=True,
                    help="Expected light-on window in seconds, e.g. --on 35 60")
    p.add_argument("--roi", nargs=4, type=int, metavar=("X0", "Y0", "X1", "Y1"), default=None,
                    help="Pixel ROI to sample (default: whole frame)")
    p.add_argument("--stride", type=int, default=1, help="Sample every Nth frame (default: 1, all frames)")
    p.add_argument("--threshold", type=float, default=None, help="Manual redness threshold (default: auto)")
    p.add_argument("--min-fraction-on", type=float, default=0.9,
                    help="Minimum fraction of the expected window that must read as ON to pass (default: 0.9)")
    p.add_argument("--outdir", default=str(DEFAULT_OUTDIR),
                    help=f"Directory to save the 3 diagnostic plots into (default: {DEFAULT_OUTDIR})")
    p.add_argument("--no-plot", action="store_true", help="Skip generating plots")
    args = p.parse_args()

    roi = tuple(args.roi) if args.roi else None
    samples = sample_video(args.video, roi=roi, stride=args.stride)
    result = check_expected_window(samples, tuple(args.on), threshold=args.threshold,
                                    min_fraction_on=args.min_fraction_on)

    print(f"video: {args.video}")
    print(f"frames sampled: {len(samples)}")
    print(f"auto/used threshold: {result['threshold']:.2f}")
    print(f"expected ON window: {result['expected_on']}")
    print(f"detected ON intervals (redness > threshold): "
          f"{[(round(a,1), round(b,1)) for a, b in result['detected_on_intervals']]}")
    print(f"fraction of expected window reading ON: {result['fraction_on_in_window']*100:.1f}%")
    if result["never_on"]:
        print("*** LIGHT NEVER TURNED ON during the expected window ***")
    elif result["dropout_time_s"] is not None:
        print(f"*** LIGHT DROPOUT DETECTED at t={result['dropout_time_s']:.2f}s "
              f"(turned off early, inside expected {result['expected_on']} window) ***")
    print(f"RESULT: {'PASS' if result['passed'] else 'FAIL'}")

    if not args.no_plot:
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        stem = Path(args.video).stem
        timeseries_path = outdir / f"{stem}_timeseries.png"
        transitions_path = outdir / f"{stem}_transitions.png"
        frames_path = outdir / f"{stem}_frames.png"

        plot_timeseries(samples, result, timeseries_path)
        plot_transitions(samples, result, transitions_path)
        plot_frame_montage(args.video, result, roi, frames_path)

        print(f"plots saved to {timeseries_path}, {transitions_path}, {frames_path}")

    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
