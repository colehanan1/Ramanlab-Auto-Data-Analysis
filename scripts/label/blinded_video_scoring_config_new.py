#!/usr/bin/env python3
"""Blinded fly behavior scoring GUI — video + trace side-by-side.

Displays randomised, blinded testing trials with the video (black-box
overlay hiding the odor label) on the left and the envelope trace on the
right.  The user scores each trial -1 to 5 and optionally adds a comment.
Results are saved to CSV with resume capability.

Review mode:
    python scripts/label/blinded_video_scoring.py --review-score 0
    # review and re-score all previously scored 0s (works for -1..5)

Usage:
    python scripts/label/blinded_video_scoring.py
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import cv2
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd
from PIL import Image, ImageTk

# Make the repo importable so the pure-logic helper resolves when run as a script.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.label import scoring_filters as sf  # noqa: E402
from fbpipe.utils.tables import read_table  # noqa: E402

# ---------------------------------------------------------------------------
# Paths & constants — edit these if your layout changes
# ---------------------------------------------------------------------------
INPUT_CSV = Path(
    "/home/ramanlab/Documents/cole/Data/CSVs-New-Opto-Flys/"
    "all_envelope_rows_wide_combined_base.csv"
)
# Second envelope overlaid on the trace (same rows/keys, different dir_val metric)
INPUT_CSV_DISTANCE = Path(
    "/home/ramanlab/Documents/cole/Data/CSVs-New-Opto-Flys/"
    "all_envelope_rows_wide_distance.csv"
)
FLAGGED_CSV = Path(
    "/home/ramanlab/Documents/cole/Data/CSVs-New-Opto-Flys/flagged-flys-truth.csv"
)
VIDEOS_ROOT = Path("/securedstorage/DATAsec/cole/Data-secured-New/")
OUTPUT_CSV = Path(
    "/home/ramanlab/Documents/cole/Data/CSVs-New-Opto-Flys/"
    "blinded_video_scores.csv"
)
SEED_FILE = Path(
    "/home/ramanlab/Documents/cole/Data/CSVs-New-Opto-Flys/"
    "blinded_video_scoring_seed.json"
)
SKIPPED_FILE = Path(
    "/home/ramanlab/Documents/cole/Data/CSVs-New-Opto-Flys/"
    "blinded_video_scoring_skipped.json"
)

# Black-box overlay (x, y, width, height) in raw video pixel coords
BLACK_BOX_X, BLACK_BOX_Y, BLACK_BOX_W, BLACK_BOX_H = 0, 0, 300, 60

# Playback / plot
MAX_SECONDS = 90
DEFAULT_FPS = 40.0
RANDOM_SEED = 42

# Trace-plot styling
ODOR_ON_S = 30.0
ODOR_OFF_S = 60.0
THRESHOLD_STD_MULT = 2.0
FIXED_Y_MAX = 100.0
ODOR_SHADE_COLOR = "#9e9e9e"
ODOR_SHADE_ALPHA = 0.20
THRESHOLD_COLOR = "tab:red"
THRESHOLD_ALPHA = 0.9
THRESHOLD_LW = 1.0
TRACE_COLOR = "black"
TRACE_LW = 1.2
TRACE2_COLOR = "gray"      # distance-metric overlay trace
THRESHOLD2_COLOR = "gray"  # its threshold line
ODOR_LINE_LW = 1.0
MAX_FRAMES = 3600  # dir_val columns to use (~90 s at 40 fps)
NO_SCORE = -999

# Display sizes
VIDEO_W, VIDEO_H = 1080, 1080

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(dataset: str | None = None) -> pd.DataFrame:
    """Eligible testing trials, optionally restricted to one dataset.

    Delegates filtering to ``scoring_filters.filter_trials``: keeps testing_<N>
    trials, drops light-only datasets, and (when ``dataset`` is given) keeps only
    exact-matching dataset rows. Training is already absent from the combined_base
    CSV and ``testing_11`` is intentionally retained.
    """
    print("Loading envelope data …")
    df = read_table(INPUT_CSV)
    before = len(df)
    df = sf.filter_trials(df, dataset=dataset)
    suffix = f" for dataset {dataset!r}" if dataset else ""
    print(f"  Eligible testing trials{suffix}: {before} → {len(df)} rows")
    return df


def load_exclusion_set() -> set[tuple[str, int]]:
    if not FLAGGED_CSV.exists():
        print("  No flagged CSV found, skipping exclusions.")
        return set()
    flagged = pd.read_csv(FLAGGED_CSV)
    score_col = [c for c in flagged.columns if "State" in c][0]
    flagged[score_col] = pd.to_numeric(flagged[score_col], errors="coerce")
    bad = flagged[flagged[score_col] <= 0]
    exclude: set[tuple[str, int]] = set()
    for _, row in bad.iterrows():
        exclude.add((str(row["fly"]).strip(), int(row["fly_number"])))
    exclude.add(("october_14_batch_1", 1))
    return exclude


def apply_exclusions(df: pd.DataFrame) -> pd.DataFrame:
    exclude = load_exclusion_set()
    before = len(df)
    df = sf.apply_exclusions(df, exclude)
    print(f"  Exclusion: {before} → {len(df)} rows ({before - len(df)} removed)")
    return df


# ---------------------------------------------------------------------------
# Video path resolution
# ---------------------------------------------------------------------------

def _strip_leading_zeros_in_dates(name: str) -> str:
    """september_09_fly_1 -> september_9_fly_1 (strip leading zeros after month)."""
    import re
    return re.sub(r'(?<=_)0+(\d)', r'\1', name)


def _try_video_patterns(dataset_path: Path, fly: str, base_fly: str, trial_label: str) -> Path | None:
    """Try Pattern A and B with a given base_fly name for the video filename."""
    stem = f"{base_fly}_{trial_label}"
    filename = f"{stem}_distance_annotated.mp4"

    # Try with original fly name AND base_fly name (in case fly dir was stripped of leading zeros too)
    for fly_dir_name in [fly, base_fly]:
        # Pattern A: {fly_dir}/{base_fly}_{trial}/{base_fly}_{trial}_distance_annotated.mp4
        path_a = dataset_path / fly_dir_name / stem / filename
        if path_a.exists():
            return path_a

        # Pattern B: {fly_dir}/videos_with_rms/testing/{filename}
        path_b = dataset_path / fly_dir_name / "videos_with_rms" / "testing" / filename
        if path_b.exists():
            return path_b

    return None


def resolve_video_path(dataset: str, fly: str, trial_label: str) -> Path | None:
    """Find the _distance_annotated.mp4 for a given trial.

    Handles: _rig_N suffixes, flagged/ subfolder, leading-zero date mismatches,
    and trial_labels with suffixes like _fly1_angle_distance_rms_envelope.
    """
    import re

    # Extract the core trial ID from trial_label
    # E.g., "testing_1_fly1_angle_distance_rms_envelope" -> "testing_1"
    match = re.match(r'(testing_\d+)', trial_label)
    core_trial_label = match.group(1) if match else trial_label

    # Dataset roots to search: normal + flagged/
    dataset_roots = [VIDEOS_ROOT / dataset]
    if "flagged" in dataset.lower():
        dataset_roots.append(VIDEOS_ROOT / "flagged" / dataset)

    # Fly name variants: original, stripped _rig_N, stripped leading zeros
    base_flies = [fly]
    stripped = re.sub(r'_rig_\d+$', '', fly)
    if stripped != fly:
        base_flies.append(stripped)
    # Also try stripping leading zeros in date portions
    for bf in list(base_flies):
        no_zeros = _strip_leading_zeros_in_dates(bf)
        if no_zeros != bf:
            base_flies.append(no_zeros)

    for ds_root in dataset_roots:
        for base_fly in base_flies:
            # Try with core trial label first
            result = _try_video_patterns(ds_root, fly, base_fly, core_trial_label)
            if result:
                return result
            # Also try with stripped zeros in trial label
            tl_no_zeros = _strip_leading_zeros_in_dates(core_trial_label)
            if tl_no_zeros != core_trial_label:
                result = _try_video_patterns(ds_root, fly, base_fly, tl_no_zeros)
                if result:
                    return result

    return None


# ---------------------------------------------------------------------------
# Randomisation helpers
# ---------------------------------------------------------------------------

def randomize_order(df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, list[int]]:
    rng = np.random.RandomState(seed)
    indices = df.index.tolist()
    rng.shuffle(indices)
    shuffled = df.loc[indices].reset_index(drop=True)
    return shuffled, indices


def save_seed_info(seed: int, order: list[int]) -> None:
    info = {"random_seed": seed, "order": order}
    with SEED_FILE.open("w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2)


def load_seed_info() -> dict | None:
    if not SEED_FILE.exists():
        return None
    with SEED_FILE.open("r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Envelope extraction & threshold
# ---------------------------------------------------------------------------

def get_dir_val_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.startswith("dir_val_")]
    cols.sort(key=lambda c: int(c.split("_")[-1]))
    return cols


def extract_envelope(row: pd.Series, dir_val_cols: list[str]) -> np.ndarray:
    vals = row[dir_val_cols].to_numpy(dtype=float)
    vals = vals[:MAX_FRAMES]
    finite_mask = np.isfinite(vals)
    if not finite_mask.any():
        return np.empty(0, dtype=float)
    last_finite = int(np.max(np.where(finite_mask)[0]))
    return vals[: last_finite + 1]


def compute_theta(env: np.ndarray, fps: float) -> float:
    """One-sided (upper) robust threshold from the pre-odor baseline (0–ODOR_ON_S).

    Only upward deviations from the baseline median feed the spread, so dips
    below resting cannot raise the line; with no upward jitter the line sits
    right at resting. Mirrors _baseline_theta in envelope_visuals.py.
    """
    return sf.compute_theta(env, fps, odor_on_s=ODOR_ON_S, std_mult=THRESHOLD_STD_MULT)


def load_distance_envelope_map(
    dir_val_cols: list[str],
) -> dict[tuple[str, str, int, str], np.ndarray]:
    """Load the distance-metric envelope per trial, keyed by full trial key.

    Used to overlay a second trace alongside the primary (combined_base) one.
    The two CSVs share identical rows/keys, differing only in dir_val values.
    """
    if not INPUT_CSV_DISTANCE.exists():
        print("  No distance CSV found; trace overlay disabled.")
        return {}
    print("Loading distance-metric envelope for overlay …")
    d = read_table(INPUT_CSV_DISTANCE)
    out: dict[tuple[str, str, int, str], np.ndarray] = {}
    for _, r in d.iterrows():
        try:
            key = (
                str(r["dataset"]).strip(),
                str(r["fly"]).strip(),
                int(r["fly_number"]),
                str(r["trial_label"]).strip(),
            )
        except Exception:
            continue
        out[key] = extract_envelope(r, dir_val_cols)
    print(f"  Distance envelopes loaded: {len(out)}")
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_trace(
    fig: plt.Figure,
    env: np.ndarray,
    fps: float,
    env2: np.ndarray | None = None,
    fps2: float | None = None,
    label1: str = "combined_base",
    label2: str = "distance",
) -> plt.Line2D | None:
    """Plot the envelope trace(s). Returns the playback cursor line (or None).

    If ``env2`` is given it is overlaid on the same axes (same 0–100 scale) in a
    contrasting colour, and each trace gets its own threshold line.
    """
    fig.clear()
    ax = fig.add_subplot(111)
    has_env2 = env2 is not None and len(env2) > 0
    f2 = fps2 if (fps2 and fps2 > 0) else fps

    t = np.arange(len(env), dtype=float) / fps
    ax.plot(t, env, linewidth=TRACE_LW, color=TRACE_COLOR, label=label1)
    if has_env2:
        t2 = np.arange(len(env2), dtype=float) / f2
        ax.plot(t2, env2, linewidth=TRACE_LW, color=TRACE2_COLOR, alpha=0.9, label=label2)

    ax.axvline(ODOR_ON_S, linestyle="--", linewidth=ODOR_LINE_LW, color="black")
    ax.axvline(ODOR_OFF_S, linestyle="--", linewidth=ODOR_LINE_LW, color="black")
    ax.axvspan(ODOR_ON_S, ODOR_OFF_S, alpha=ODOR_SHADE_ALPHA, color=ODOR_SHADE_COLOR)

    theta = compute_theta(env, fps)
    if math.isfinite(theta):
        ax.axhline(theta, linestyle="--", linewidth=THRESHOLD_LW,
                    color=THRESHOLD_COLOR, alpha=THRESHOLD_ALPHA)
    if has_env2:
        theta2 = compute_theta(env2, f2)
        if math.isfinite(theta2):
            ax.axhline(theta2, linestyle="--", linewidth=THRESHOLD_LW,
                        color=THRESHOLD2_COLOR, alpha=THRESHOLD_ALPHA)

    # Playback cursor — vertical line that tracks the current video frame
    cursor_line = ax.axvline(0.0, linestyle="-", linewidth=1.5, color="tab:blue", alpha=0.8)

    ax.set_ylim(0, FIXED_Y_MAX)
    x_max = t[-1] if len(t) > 0 else 120.0
    if has_env2:
        x_max = max(x_max, (len(env2) - 1) / f2)
    ax.set_xlim(0, x_max if x_max > 0 else 120.0)
    ax.margins(x=0, y=0.02)
    ax.set_ylabel("Max Distance x Angle %", fontsize=10)
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if has_env2:
        ax.legend(loc="upper right", fontsize=9, framealpha=0.85)
    fig.tight_layout()
    return cursor_line


# ---------------------------------------------------------------------------
# Resume / save helpers
# ---------------------------------------------------------------------------

def trial_key(row: pd.Series) -> tuple[str, str, int, str]:
    """Create a unique key for resume matching.
    Uses core trial label (testing_1) for matching, even if full label has suffixes.
    """
    full_trial_label = str(row["trial_label"]).strip()
    core_trial_label = _extract_core_trial_label(full_trial_label)
    return (
        str(row["dataset"]).strip(),
        str(row["fly"]).strip(),
        int(row["fly_number"]),
        core_trial_label,
    )


def load_existing_scores_map() -> dict[tuple[str, str, int, str], dict[str, object]]:
    """Load existing scores keyed by trial key.

    If duplicates exist, later rows in the CSV win.
    """
    if not OUTPUT_CSV.exists():
        return {}
    try:
        df = pd.read_csv(OUTPUT_CSV)
    except Exception:
        return {}

    out: dict[tuple[str, str, int, str], dict[str, object]] = {}
    for _, row in df.iterrows():
        try:
            key = (
                str(row["dataset"]).strip(),
                str(row["fly"]).strip(),
                int(row["fly_number"]),
                _extract_core_trial_label(str(row["trial_label"]).strip()),
            )
        except Exception:
            continue
        out[key] = {
            "user_score": row.get("user_score", np.nan),
            "comment": "" if pd.isna(row.get("comment", "")) else str(row.get("comment", "")),
        }
    return out


def load_existing_scores() -> set[tuple[str, str, int, str]]:
    if not OUTPUT_CSV.exists():
        return set()
    try:
        df = pd.read_csv(OUTPUT_CSV)
    except Exception:
        return set()
    scored: set[tuple[str, str, int, str]] = set()
    for _, row in df.iterrows():
        scored.add((
            str(row["dataset"]).strip(),
            str(row["fly"]).strip(),
            int(row["fly_number"]),
            str(row["trial_label"]).strip(),
        ))
    return scored


def load_skipped_trials() -> set[tuple[str, str, int, str]]:
    """Load set of skipped trial keys from persistent storage."""
    if not SKIPPED_FILE.exists():
        return set()
    try:
        with SKIPPED_FILE.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
            return set(tuple(k) for k in data.get("skipped", []))
    except Exception:
        return set()


def load_skipped_comments() -> dict[tuple[str, str, int, str], str]:
    """Load comments saved alongside skipped trials (key -> comment)."""
    if not SKIPPED_FILE.exists():
        return {}
    try:
        with SKIPPED_FILE.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return {}
    out: dict[tuple[str, str, int, str], str] = {}
    for item in data.get("comments", []):
        if isinstance(item, list) and len(item) == 5:
            try:
                key = (str(item[0]).strip(), str(item[1]).strip(), int(item[2]), str(item[3]).strip())
            except Exception:
                continue
            out[key] = str(item[4])
    return out


def save_skipped_trials(
    skipped: set[tuple[str, str, int, str]],
    comments: dict[tuple[str, str, int, str], str] | None = None,
) -> None:
    """Save skipped trial keys (and any per-trial comments) to persistent storage."""
    data: dict[str, object] = {"skipped": [list(k) for k in skipped]}
    if comments:
        data["comments"] = [
            [k[0], k[1], k[2], k[3], comments[k]]
            for k in skipped
            if comments.get(k)
        ]
    with SKIPPED_FILE.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def cleanup_score_and_skipped_files() -> None:
    """Normalize artifacts:

    - scoring CSV: one row per trial key, keeping the highest user_score
      (tie-breaker: latest row)
    - skipped JSON: unique keys only and remove keys already scored
    """
    scored_keys: set[tuple[str, str, int, str]] = set()

    if OUTPUT_CSV.exists() and OUTPUT_CSV.stat().st_size > 0:
        try:
            df = pd.read_csv(OUTPUT_CSV)
            key_cols = ["dataset", "fly", "fly_number", "trial_label"]
            required = key_cols + ["user_score"]
            if all(c in df.columns for c in required):
                for c in ["dataset", "fly", "trial_label"]:
                    df[c] = df[c].astype(str).str.strip()
                df["fly_number"] = pd.to_numeric(df["fly_number"], errors="coerce")
                df["user_score"] = pd.to_numeric(df["user_score"], errors="coerce")

                before = len(df)
                df = (
                    df.reset_index(names="__rowid__")
                    .sort_values(key_cols + ["user_score", "__rowid__"])
                    .drop_duplicates(subset=key_cols, keep="last")
                    .sort_values("__rowid__")
                    .drop(columns="__rowid__")
                    .reset_index(drop=True)
                )
                if len(df) != before:
                    print(f"  Cleaned scoring duplicates: {before} -> {len(df)}")
                df.to_csv(OUTPUT_CSV, index=False)

                for _, r in df.iterrows():
                    if pd.notna(r["fly_number"]):
                        scored_keys.add(
                            (
                                str(r["dataset"]).strip(),
                                str(r["fly"]).strip(),
                                int(r["fly_number"]),
                                str(r["trial_label"]).strip(),
                            )
                        )
        except Exception as exc:
            print(f"  Warning: could not normalize scoring CSV: {exc}")

    if SKIPPED_FILE.exists():
        try:
            payload = json.loads(SKIPPED_FILE.read_text(encoding="utf-8"))
            raw = payload.get("skipped", [])
            # Index any saved comments by key so we can carry them forward.
            comment_map: dict[tuple[str, str, int, str], str] = {}
            for c in payload.get("comments", []):
                if isinstance(c, list) and len(c) == 5:
                    try:
                        comment_map[(str(c[0]).strip(), str(c[1]).strip(), int(c[2]), str(c[3]).strip())] = str(c[4])
                    except Exception:
                        continue
            cleaned: list[list[object]] = []
            cleaned_comments: list[list[object]] = []
            seen: set[tuple[str, str, int, str]] = set()
            for item in raw:
                if not (isinstance(item, list) and len(item) == 4):
                    continue
                key = (
                    str(item[0]).strip(),
                    str(item[1]).strip(),
                    int(item[2]),
                    str(item[3]).strip(),
                )
                if key in seen or key in scored_keys:
                    continue
                seen.add(key)
                cleaned.append([key[0], key[1], key[2], key[3]])
                if comment_map.get(key):
                    cleaned_comments.append([key[0], key[1], key[2], key[3], comment_map[key]])

            if len(cleaned) != len(raw):
                print(f"  Cleaned skipped entries: {len(raw)} -> {len(cleaned)}")
            out_payload: dict[str, object] = {"skipped": cleaned}
            if cleaned_comments:
                out_payload["comments"] = cleaned_comments
            SKIPPED_FILE.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")
        except Exception as exc:
            print(f"  Warning: could not normalize skipped JSON: {exc}")


def _extract_core_trial_label(full_label: str) -> str:
    """Extract core trial ID: testing_1_fly1_distances_... -> testing_1"""
    import re
    match = re.match(r'(testing_\d+)', full_label)
    return match.group(1) if match else full_label


def save_score(row: pd.Series, score: int, comment: str) -> None:
    full_trial_label = str(row["trial_label"]).strip()
    core_trial_label = _extract_core_trial_label(full_trial_label)

    row_data = {
        "dataset": str(row["dataset"]).strip(),
        "fly": str(row["fly"]).strip(),
        "fly_number": int(row["fly_number"]),
        "trial_type": str(row["trial_type"]).strip(),
        "trial_label": core_trial_label,
        "user_score": score,
        "comment": comment,
    }
    # Upsert behavior: keep one row per (dataset, fly, fly_number, core trial_label).
    # This allows review mode to re-score without creating duplicate entries.
    if OUTPUT_CSV.exists() and OUTPUT_CSV.stat().st_size > 0:
        try:
            existing = pd.read_csv(OUTPUT_CSV)
            required = {"dataset", "fly", "fly_number", "trial_label"}
            if required.issubset(existing.columns):
                existing["dataset"] = existing["dataset"].astype(str).str.strip()
                existing["fly"] = existing["fly"].astype(str).str.strip()
                existing["fly_number"] = pd.to_numeric(existing["fly_number"], errors="coerce").astype("Int64")
                existing["trial_label"] = existing["trial_label"].astype(str).str.strip().map(_extract_core_trial_label)

                keep_mask = ~(
                    (existing["dataset"] == row_data["dataset"]) &
                    (existing["fly"] == row_data["fly"]) &
                    (existing["fly_number"] == row_data["fly_number"]) &
                    (existing["trial_label"] == row_data["trial_label"])
                )
                existing = existing[keep_mask].copy()
                updated = pd.concat([existing, pd.DataFrame([row_data])], ignore_index=True)
                updated.to_csv(OUTPUT_CSV, index=False)
                return
        except Exception:
            # Fall back to append if parsing/upsert fails for any reason.
            pass

    file_exists = OUTPUT_CSV.exists() and OUTPUT_CSV.stat().st_size > 0
    with OUTPUT_CSV.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row_data.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)


def delete_score(row: pd.Series) -> None:
    """Delete any saved score rows for a specific trial key."""
    if not (OUTPUT_CSV.exists() and OUTPUT_CSV.stat().st_size > 0):
        return

    try:
        df = pd.read_csv(OUTPUT_CSV)
    except Exception:
        return

    required = {"dataset", "fly", "fly_number", "trial_label"}
    if not required.issubset(df.columns):
        return

    dataset = str(row["dataset"]).strip()
    fly = str(row["fly"]).strip()
    fly_number = int(row["fly_number"])
    core_trial_label = _extract_core_trial_label(str(row["trial_label"]).strip())

    df["dataset"] = df["dataset"].astype(str).str.strip()
    df["fly"] = df["fly"].astype(str).str.strip()
    df["fly_number"] = pd.to_numeric(df["fly_number"], errors="coerce").astype("Int64")
    df["trial_label"] = df["trial_label"].astype(str).str.strip().map(_extract_core_trial_label)

    keep_mask = ~(
        (df["dataset"] == dataset)
        & (df["fly"] == fly)
        & (df["fly_number"] == fly_number)
        & (df["trial_label"] == core_trial_label)
    )
    df = df[keep_mask].copy()
    df.to_csv(OUTPUT_CSV, index=False)


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class BlindedVideoScoringApp:
    def __init__(
        self,
        master: tk.Tk,
        df: pd.DataFrame,
        dir_val_cols: list[str],
        video_paths: list[Path],
        scored_keys: set[tuple[str, str, int, str]],
        existing_scores_map: dict[tuple[str, str, int, str], dict[str, object]],
        show_skipped: bool = False,
        review_score: int | None = None,
        distance_env_map: dict[tuple[str, str, int, str], np.ndarray] | None = None,
    ) -> None:
        self.master = master
        self.df = df
        self.dir_val_cols = dir_val_cols
        self.video_paths = video_paths
        self.distance_env_map = distance_env_map or {}
        self.total = len(df)
        self.show_skipped = show_skipped
        self.review_score = review_score
        self.existing_scores_map = existing_scores_map

        # Load persistent skipped trials (and any comments saved with them)
        self.skipped_keys: set[tuple[str, str, int, str]] = load_skipped_trials()
        self.skipped_comments: dict[tuple[str, str, int, str], str] = load_skipped_comments()

        if self.review_score is not None:
            # Review only trials currently scored with this label.
            self.pending_indices = [
                idx for idx in range(self.total)
                if self._existing_score_for_idx(idx) == self.review_score
            ]
        elif self.show_skipped:
            # Show ONLY previously skipped trials (that haven't been scored yet)
            self.pending_indices: list[int] = [
                idx for idx in range(self.total)
                if trial_key(self.df.iloc[idx]) in self.skipped_keys
                and trial_key(self.df.iloc[idx]) not in scored_keys
            ]
        else:
            # Normal mode: skip scored and skipped trials
            self.pending_indices: list[int] = [
                idx for idx in range(self.total)
                if trial_key(self.df.iloc[idx]) not in scored_keys
                and trial_key(self.df.iloc[idx]) not in self.skipped_keys
            ]
        self.current_pending_pos = 0
        self.already_scored = self.total - len(self.pending_indices)

        # Playback state
        self.cap: cv2.VideoCapture | None = None
        self.fps = DEFAULT_FPS
        self.playing = False
        self.frame_counter = 0
        self.max_playback_frames = 0
        self.slider_active = False
        self.slider_resume_playback = False
        self.slider_updating = False

        # matplotlib rcParams
        plt.rcParams.update({
            "figure.dpi": 150,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 10,
        })

        # --- Window ---
        title = "Blinded Fly Scoring — SKIPPED TRIALS" if self.show_skipped else "Blinded Fly Scoring — Video + Trace"
        self.master.title(title)
        self.master.protocol("WM_DELETE_WINDOW", self._on_close)
        self.master.configure(bg="#f8f8fb")

        style = ttk.Style(self.master)
        try:
            style.theme_use("clam")
        except Exception:
            pass

        # --- Top frame: video (left) + trace (right) — fixed height ---
        top = tk.Frame(self.master, bg="#f8f8fb", height=VIDEO_H)
        top.pack(fill=tk.X, padx=8, pady=(8, 0))
        top.pack_propagate(False)

        self.video_canvas = tk.Canvas(
            top, width=VIDEO_W, height=VIDEO_H, bg="black", highlightthickness=0
        )
        self.video_canvas.pack(side=tk.LEFT, padx=(0, 4))

        # Trace plot: match video height, fill remaining width
        fig_h_in = VIDEO_H / 100.0  # match video pixel height at 100 dpi
        self.fig = plt.Figure(figsize=(12, fig_h_in), dpi=100)
        self.mpl_canvas = FigureCanvasTkAgg(self.fig, master=top)
        self.mpl_canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Playback cursor line on the trace plot (set per-trial)
        self.cursor_line: plt.Line2D | None = None

        # --- Slider ---
        slider_frame = tk.Frame(self.master, bg="#f8f8fb")
        slider_frame.pack(fill=tk.X, padx=12, pady=(6, 0))

        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_scale = ttk.Scale(
            slider_frame, from_=0.0, to=1.0, orient="horizontal",
            variable=self.progress_var,
        )
        self.progress_scale.pack(fill="x")
        self.progress_scale.configure(command=self._on_slider_move)
        self.progress_scale.bind("<ButtonPress-1>", self._on_slider_press)
        self.progress_scale.bind("<ButtonRelease-1>", self._on_slider_release)

        self.time_label = tk.Label(
            self.master, text="0.00 s / 0.00 s", bg="#f8f8fb",
            font=("Helvetica", 16),
        )
        self.time_label.pack(anchor="e", padx=16)

        # --- Progress label ---
        self.progress_text = tk.Label(
            self.master, text="", bg="#f8f8fb", font=("Helvetica", 18, "bold"),
        )
        self.progress_text.pack(anchor="w", padx=16, pady=(6, 0))

        # --- Score controls ---
        score_frame = tk.Frame(self.master, bg="#f8f8fb")
        score_frame.pack(fill=tk.X, padx=16, pady=(8, 4))

        tk.Label(score_frame, text="Score (-1 to 5):", font=("Helvetica", 22, "bold"),
                 bg="#f8f8fb").pack(side=tk.LEFT, padx=(0, 16))

        self.score_var = tk.IntVar(value=NO_SCORE)
        self._score_buttons: dict[int, tk.Button] = {}
        for val in [-1, 0, 1, 2, 3, 4, 5]:
            btn = tk.Button(
                score_frame, text=str(val), font=("Helvetica", 28, "bold"),
                width=3, height=1, relief="raised", bd=3,
                command=lambda v=val: self._select_score(v),
            )
            btn.pack(side=tk.LEFT, padx=6)
            self._score_buttons[val] = btn

        # Or type a score directly
        tk.Label(score_frame, text="  or type:", font=("Helvetica", 20),
                 bg="#f8f8fb").pack(side=tk.LEFT, padx=(20, 6))
        self.score_entry_var = tk.StringVar()
        self.score_entry = tk.Entry(
            score_frame, textvariable=self.score_entry_var, width=3,
            font=("Helvetica", 28, "bold"), justify="center",
        )
        self.score_entry.pack(side=tk.LEFT, padx=(0, 8))
        self.score_entry_var.trace_add("write", self._on_score_entry_changed)

        # --- Comment ---
        comment_frame = tk.Frame(self.master, bg="#f8f8fb")
        comment_frame.pack(fill=tk.X, padx=16, pady=(4, 6))

        tk.Label(comment_frame, text="Comment:", font=("Helvetica", 20),
                 bg="#f8f8fb").pack(side=tk.LEFT, padx=(0, 8))
        self.comment_var = tk.StringVar()
        self.comment_entry = tk.Entry(
            comment_frame, textvariable=self.comment_var, width=40,
            font=("Helvetica", 20),
        )
        self.comment_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 16))

        # --- Buttons ---
        btn_frame = tk.Frame(self.master, bg="#f8f8fb")
        btn_frame.pack(pady=(4, 14))

        self.submit_btn = tk.Button(
            btn_frame, text="Submit", font=("Helvetica", 22, "bold"),
            width=10, height=1, bg="#4CAF50", fg="white", relief="raised", bd=3,
            command=self._on_submit,
        )
        self.submit_btn.grid(row=0, column=0, padx=12)
        self.replay_btn = tk.Button(
            btn_frame, text="Replay", font=("Helvetica", 22, "bold"),
            width=10, height=1, bg="#2196F3", fg="white", relief="raised", bd=3,
            command=self._on_replay,
        )
        self.replay_btn.grid(row=0, column=1, padx=12)
        self.back_btn = tk.Button(
            btn_frame, text="Back", font=("Helvetica", 22, "bold"),
            width=10, height=1, bg="#7E57C2", fg="white", relief="raised", bd=3,
            command=self._on_back,
        )
        self.back_btn.grid(row=0, column=2, padx=12)
        self.skip_btn = tk.Button(
            btn_frame, text="Skip", font=("Helvetica", 22, "bold"),
            width=10, height=1, bg="#FF9800", fg="white", relief="raised", bd=3,
            command=self._on_skip,
        )
        self.skip_btn.grid(row=0, column=3, padx=12)
        self.clear_btn = tk.Button(
            btn_frame, text="Clear Score", font=("Helvetica", 22, "bold"),
            width=12, height=1, bg="#795548", fg="white", relief="raised", bd=3,
            command=self._on_clear_score,
        )
        self.clear_btn.grid(row=0, column=4, padx=12)
        self.exit_btn = tk.Button(
            btn_frame, text="Save & Exit", font=("Helvetica", 22, "bold"),
            width=12, height=1, bg="#f44336", fg="white", relief="raised", bd=3,
            command=self._on_exit,
        )
        self.exit_btn.grid(row=0, column=5, padx=12)

        # Clear is only meaningful in review mode.
        if self.review_score is None:
            self.clear_btn.config(state=tk.DISABLED)

        # Safer keyboard shortcut: Ctrl+Enter submits; plain Enter does not.
        self.master.bind("<Control-Return>", lambda _: self._on_submit())

        self.submit_hint = tk.Label(
            self.master,
            text="Tip: Click Submit, or press Ctrl+Enter (Enter alone will not submit).",
            bg="#f8f8fb",
            font=("Helvetica", 12),
        )
        self.submit_hint.pack(anchor="w", padx=16, pady=(0, 6))

        # Guard against rapid double-submit / re-entrancy when users click quickly.
        self.busy = False

        # Focus the score entry so user can just type a number right away
        self.score_entry.focus_set()

        # Resume message
        if self.show_skipped:
            messagebox.showinfo(
                "Skipped Trials Mode",
                f"Showing {len(self.pending_indices)} previously skipped trials.\n"
                f"Score them now — they'll be removed from the skipped list once scored.",
            )
        elif self.review_score is not None:
            messagebox.showinfo(
                "Review Mode",
                f"Showing {len(self.pending_indices)} previously scored trial(s) with score {self.review_score}.\n"
                "You can confirm labels or re-score; updates overwrite existing entries.",
            )
        elif self.already_scored > 0:
            messagebox.showinfo(
                "Resuming",
                f"Resuming: {self.already_scored} of {self.total} already scored.\n"
                f"{len(self.pending_indices)} remaining.",
            )

        # Show first trial
        if self.pending_indices:
            self._show_current_trial()
        else:
            self._show_completion()

    # ---- Score helpers ----

    def _existing_score_for_idx(self, idx: int) -> int | None:
        info = self.existing_scores_map.get(trial_key(self.df.iloc[idx]))
        if info is None:
            return None
        try:
            return int(info.get("user_score"))
        except Exception:
            return None

    def _select_score(self, val: int) -> None:
        self.score_var.set(val)
        self.score_entry_var.set(str(val))
        self._highlight_score_button(val)

    def _highlight_score_button(self, val: int) -> None:
        for score_value, btn in self._score_buttons.items():
            if score_value == val:
                btn.configure(bg="#2196F3", fg="white", relief="sunken")
            else:
                btn.configure(bg="#d9d9d9", fg="black", relief="raised")

    def _on_score_entry_changed(self, *_args: object) -> None:
        raw = self.score_entry_var.get().strip()
        if raw in ("-1", "0", "1", "2", "3", "4", "5"):
            val = int(raw)
            self.score_var.set(val)
            self._highlight_score_button(val)

    # ---- Video helpers ----

    def _draw_frame(self, frame: np.ndarray) -> None:
        # Black box overlay
        cv2.rectangle(
            frame,
            (BLACK_BOX_X, BLACK_BOX_Y),
            (BLACK_BOX_X + BLACK_BOX_W, BLACK_BOX_Y + BLACK_BOX_H),
            (0, 0, 0),
            -1,
        )
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(
            frame_rgb, (VIDEO_W, VIDEO_H), interpolation=cv2.INTER_AREA
        )
        im = Image.fromarray(frame_resized)
        imgtk = ImageTk.PhotoImage(image=im)
        self.video_canvas.imgtk = imgtk  # prevent GC
        self.video_canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)

    def _update_time_readout(self, frame_index: int) -> None:
        current_s = frame_index / self.fps if self.fps > 0 else 0.0
        total_s = self.max_playback_frames / self.fps if self.fps > 0 else 0.0
        self.time_label.config(text=f"{current_s:0.2f} s / {total_s:0.2f} s")

    def _update_cursor(self, frame_index: int) -> None:
        if self.cursor_line is not None:
            current_s = frame_index / self.fps if self.fps > 0 else 0.0
            self.cursor_line.set_xdata([current_s, current_s])
            self.mpl_canvas.draw_idle()

    def _set_progress(self, frame_index: int) -> None:
        if not self.slider_active:
            self.slider_updating = True
            self.progress_var.set(frame_index)
            self.slider_updating = False
        self._update_time_readout(frame_index)
        self._update_cursor(frame_index)

    def _advance(self) -> None:
        if not self.playing or self.cap is None:
            return
        if self.max_playback_frames and self.frame_counter >= self.max_playback_frames:
            self.playing = False
            return
        ok, frame = self.cap.read()
        if not ok:
            self.playing = False
            return
        self.frame_counter += 1
        self._draw_frame(frame)
        self._set_progress(max(0, self.frame_counter - 1))
        if self.max_playback_frames and self.frame_counter >= self.max_playback_frames:
            self.playing = False
            return
        delay = int(1000 / max(1.0, self.fps))
        self.master.after(delay, self._advance)

    def _seek_to_frame(self, target: int, resume: bool) -> None:
        if self.cap is None:
            return
        target = max(0, min(int(round(target)), max(0, self.max_playback_frames - 1)))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok, frame = self.cap.read()
        if ok:
            self.frame_counter = target + 1
            self._draw_frame(frame)
            self._set_progress(target)
        else:
            self.frame_counter = target
        if resume and (not self.max_playback_frames or self.frame_counter < self.max_playback_frames):
            self.playing = True
            delay = int(1000 / max(1.0, self.fps))
            self.master.after(delay, self._advance)
        else:
            self.playing = False

    # ---- Slider callbacks ----

    def _on_slider_move(self, value: str) -> None:
        if self.slider_updating:
            return
        try:
            self._update_time_readout(int(float(value)))
        except (TypeError, ValueError):
            pass

    def _on_slider_press(self, _event: tk.Event) -> None:
        if self.cap is None:
            return
        self.slider_active = True
        self.slider_resume_playback = self.playing
        self.playing = False

    def _on_slider_release(self, _event: tk.Event) -> None:
        if self.cap is None:
            self.slider_active = False
            return
        self.slider_active = False
        self._seek_to_frame(int(self.progress_var.get()), self.slider_resume_playback)

    # ---- Trial display ----

    def _show_current_trial(self) -> None:
        if self.current_pending_pos >= len(self.pending_indices):
            self._show_completion()
            return

        idx = self.pending_indices[self.current_pending_pos]
        row = self.df.iloc[idx]

        # Trace — primary (combined_base) plus optional distance overlay
        env = extract_envelope(row, self.dir_val_cols)
        fps = float(row.get("fps", DEFAULT_FPS))
        if not math.isfinite(fps) or fps <= 0:
            fps = DEFAULT_FPS
        env2 = None
        if self.distance_env_map:
            k = (
                str(row["dataset"]).strip(),
                str(row["fly"]).strip(),
                int(row["fly_number"]),
                str(row["trial_label"]).strip(),
            )
            env2 = self.distance_env_map.get(k)
        self.cursor_line = plot_trace(self.fig, env, fps, env2=env2, fps2=fps)
        self.mpl_canvas.draw()

        # Video
        if self.cap is not None:
            self.cap.release()
        vp = self.video_paths[idx]
        self.cap = cv2.VideoCapture(str(vp))
        if not self.cap.isOpened():
            self.cap = None

        # Force fixed 40 fps playback for every video, ignoring the container's
        # native/reported fps so all trials play at the same speed.
        self.fps = DEFAULT_FPS
        self.max_playback_frames = int(round(self.fps * MAX_SECONDS))
        if self.cap is not None:
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames > 0:
                self.max_playback_frames = min(self.max_playback_frames, total_frames)

        slider_max = max(1.0, float(self.max_playback_frames - 1))
        self.progress_scale.configure(to=slider_max)
        self._set_progress(0)
        self.frame_counter = 0

        # Reset controls
        self.score_var.set(NO_SCORE)
        self.score_entry_var.set("")
        self._highlight_score_button(NO_SCORE)  # unhighlight all
        self.comment_var.set("")

        # In skipped-review mode, restore any comment typed when the trial was skipped.
        if self.show_skipped:
            saved_comment = self.skipped_comments.get(trial_key(row))
            if saved_comment:
                self.comment_var.set(saved_comment)

        # In review mode, prefill existing score/comment for quick confirmation.
        if self.review_score is not None:
            info = self.existing_scores_map.get(trial_key(row))
            if info is not None:
                try:
                    prev_score = int(info.get("user_score"))
                except Exception:
                    prev_score = NO_SCORE
                if prev_score in {-1, 0, 1, 2, 3, 4, 5}:
                    self.score_var.set(prev_score)
                    self.score_entry_var.set(str(prev_score))
                    self._highlight_score_button(prev_score)
                self.comment_var.set(str(info.get("comment", "")))
        self.submit_btn.config(state=tk.NORMAL)
        self.score_entry.focus_set()

        fly_number = int(row.get("fly_number", 0))
        scored_so_far = self.already_scored + self.current_pending_pos
        self.progress_text.config(
            text=f"Trial {scored_so_far + 1} of {self.total}  —  Fly #{fly_number}"
        )
        if self.review_score is not None:
            self.progress_text.config(
                text=(
                    f"Review score {self.review_score}: {self.current_pending_pos + 1}"
                    f" of {len(self.pending_indices)}  —  Fly #{fly_number}"
                )
            )

        # Start playback
        if self.cap is not None:
            self.playing = True
            self.master.after(0, self._advance)

    def _on_submit(self) -> None:
        if self.busy:
            return

        score = self.score_var.get()
        if score == NO_SCORE:
            messagebox.showwarning("Select a score", "Please select a score (-1 to 5).")
            return

        self.busy = True
        self.playing = False
        self.submit_btn.config(state=tk.DISABLED)
        self.replay_btn.config(state=tk.DISABLED)
        self.back_btn.config(state=tk.DISABLED)
        self.skip_btn.config(state=tk.DISABLED)
        self.clear_btn.config(state=tk.DISABLED)
        self.progress_text.config(text="Saving score and loading next trial …")

        # Queue save/advance so Tk can repaint and remain responsive.
        self.master.after(1, lambda: self._save_and_advance(score))

    def _save_and_advance(self, score: int) -> None:
        try:
            idx = self.pending_indices[self.current_pending_pos]
            row = self.df.iloc[idx]
            save_score(row, score, self.comment_var.get().strip())
            self.existing_scores_map[trial_key(row)] = {
                "user_score": score,
                "comment": self.comment_var.get().strip(),
            }

            # Remove from skipped set once scored
            key = trial_key(row)
            if key in self.skipped_keys:
                self.skipped_keys.discard(key)
                self.skipped_comments.pop(key, None)
                save_skipped_trials(self.skipped_keys, self.skipped_comments)

            self.current_pending_pos += 1
            if self.current_pending_pos >= len(self.pending_indices):
                self._show_completion()
            else:
                self._show_current_trial()
        finally:
            self.busy = False
            if self.current_pending_pos < len(self.pending_indices):
                self.submit_btn.config(state=tk.NORMAL)
                self.replay_btn.config(state=tk.NORMAL)
                self.back_btn.config(state=tk.NORMAL)
                self.skip_btn.config(state=tk.NORMAL)
                if self.review_score is not None:
                    self.clear_btn.config(state=tk.NORMAL)

    def _on_back(self) -> None:
        """Go to the previous trial in the current session order."""
        if self.busy:
            return
        if self.current_pending_pos <= 0:
            messagebox.showinfo("At first trial", "You are already at the first trial in this session.")
            return

        self.playing = False
        self.current_pending_pos -= 1
        self._show_current_trial()

    def _on_replay(self) -> None:
        if self.cap is None:
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.frame_counter = 0
        self._set_progress(0)
        if not self.playing:
            self.playing = True
            self.master.after(0, self._advance)

    def _on_skip(self) -> None:
        """Skip this trial, persist it (with any comment), and move to the next one."""
        self.playing = False
        idx = self.pending_indices[self.current_pending_pos]
        row = self.df.iloc[idx]
        key = trial_key(row)
        self.skipped_keys.add(key)
        comment = self.comment_var.get().strip()
        if comment:
            self.skipped_comments[key] = comment
        else:
            self.skipped_comments.pop(key, None)
        save_skipped_trials(self.skipped_keys, self.skipped_comments)

        self.current_pending_pos += 1
        if self.current_pending_pos >= len(self.pending_indices):
            self._show_completion()
        else:
            self._show_current_trial()

    def _on_clear_score(self) -> None:
        """In review mode, clear current trial score from CSV and move forward."""
        if self.busy:
            return
        if self.review_score is None:
            return

        idx = self.pending_indices[self.current_pending_pos]
        row = self.df.iloc[idx]
        if not messagebox.askyesno(
            "Clear score",
            "Clear saved score for this trial and move to next?",
        ):
            return

        self.busy = True
        self.playing = False
        self.submit_btn.config(state=tk.DISABLED)
        self.replay_btn.config(state=tk.DISABLED)
        self.back_btn.config(state=tk.DISABLED)
        self.skip_btn.config(state=tk.DISABLED)
        self.clear_btn.config(state=tk.DISABLED)
        self.progress_text.config(text="Clearing score and loading next trial …")

        try:
            delete_score(row)
            key = trial_key(row)
            if key in self.existing_scores_map:
                del self.existing_scores_map[key]

            self.current_pending_pos += 1
            if self.current_pending_pos >= len(self.pending_indices):
                self._show_completion()
            else:
                self._show_current_trial()
        finally:
            self.busy = False
            if self.current_pending_pos < len(self.pending_indices):
                self.submit_btn.config(state=tk.NORMAL)
                self.replay_btn.config(state=tk.NORMAL)
                self.back_btn.config(state=tk.NORMAL)
                self.skip_btn.config(state=tk.NORMAL)
                self.clear_btn.config(state=tk.NORMAL)

    def _show_completion(self) -> None:
        self.playing = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.text(0.5, 0.5, "All trials scored!\nThank you.",
                ha="center", va="center", fontsize=20, transform=ax.transAxes)
        ax.set_axis_off()
        self.mpl_canvas.draw()

        self.video_canvas.delete("all")
        self.progress_text.config(text=f"Complete: {self.total} of {self.total}")
        self.submit_btn.config(state=tk.DISABLED)
        messagebox.showinfo("Complete", f"All {self.total} trials have been scored.")

    def _on_exit(self) -> None:
        self.playing = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.master.destroy()

    def _on_close(self) -> None:
        self._on_exit()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Blinded video + trace scoring GUI")
    parser.add_argument(
        "--show-skipped", action="store_true",
        help="Show only previously skipped trials so you can score them",
    )
    parser.add_argument(
        "--review-score",
        type=int,
        choices=[-1, 0, 1, 2, 3, 4, 5],
        default=None,
        help="Show only trials previously scored with this value, so you can confirm/re-score them",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Only score trials whose dataset column EXACTLY equals this "
             "(e.g. RandomPanel-Training-24-0.01).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List datasets with trial/fly counts and scoring progress, then exit.",
    )
    args = parser.parse_args()

    # Keep persisted artifacts consistent before loading state.
    cleanup_score_and_skipped_files()

    if args.list:
        df_all = apply_exclusions(load_data(dataset=None))
        scored_keys = set(load_existing_scores_map().keys())
        print("\nDataset                                   trials  flies  scored")
        print("-" * 64)
        for r in sf.summarize_datasets(df_all, scored_keys):
            print(f"{r['dataset']:<40} {r['trials']:>6} {r['flies']:>6} {r['scored']:>7}")
        sys.exit(0)

    # 1. Load & filter (optionally to one dataset)
    df = apply_exclusions(load_data(dataset=args.dataset))
    if args.dataset is not None and df.empty:
        avail = sf.available_datasets(read_table(INPUT_CSV))
        print(f"\nNo testing trials for dataset {args.dataset!r}. Available datasets:")
        for name in avail:
            print(f"  {name}")
        sys.exit(2)
    dir_val_cols = get_dir_val_cols(df)
    print(f"  dir_val columns: {len(dir_val_cols)}")
    print(f"  Trial count after exclusions: {len(df)}")

    # Second envelope (distance metric) to overlay on each trace
    distance_env_map = load_distance_envelope_map(dir_val_cols)

    # 2. Resolve video paths and keep only rows with a video
    print("Resolving video paths …")
    video_paths: list[Path | None] = []
    for _, row in df.iterrows():
        vp = resolve_video_path(
            str(row["dataset"]).strip(),
            str(row["fly"]).strip(),
            str(row["trial_label"]).strip(),
        )
        video_paths.append(vp)

    has_video = [vp is not None for vp in video_paths]
    n_found = sum(has_video)
    print(f"  Videos found: {n_found} / {len(df)}")
    if n_found == 0:
        print("No videos found. Check VIDEOS_ROOT path.")
        sys.exit(1)

    # Filter to only rows with a video
    keep_mask = pd.Series(has_video, index=df.index)
    df = df[keep_mask].reset_index(drop=True)
    video_paths_clean: list[Path] = [vp for vp in video_paths if vp is not None]
    print(f"  Trials with video: {len(df)}")

    # 3. Seed & randomise
    seed_info = load_seed_info()
    seed = seed_info["random_seed"] if seed_info else RANDOM_SEED
    print(f"  Seed: {seed}")

    df_shuffled, order = randomize_order(df, seed)
    # Reorder video paths to match shuffled df
    video_paths_shuffled = [video_paths_clean[i] for i in order]
    save_seed_info(seed, order)

    # 4. Resume
    existing_scores_map = load_existing_scores_map()
    scored_keys = set(existing_scores_map.keys())
    print(f"  Already scored: {len(scored_keys)}")

    # 5. Launch
    root = tk.Tk()
    BlindedVideoScoringApp(
        root,
        df_shuffled,
        dir_val_cols,
        video_paths_shuffled,
        scored_keys,
        existing_scores_map,
        show_skipped=args.show_skipped,
        review_score=args.review_score,
        distance_env_map=distance_env_map,
    )
    root.mainloop()


if __name__ == "__main__":
    main()
