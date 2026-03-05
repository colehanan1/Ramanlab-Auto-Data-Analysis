#!/usr/bin/env python3
"""Blinded fruit fly behavioral trace scoring GUI.

Displays randomised, blinded combinedBase envelope traces one at a time.
The user scores each trace 0-5 and optionally adds a comment.
Results are saved to CSV with resume capability.

Usage:
    python scripts/blinded_scoring_gui.py
"""
from __future__ import annotations

import csv
import json
import math
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
INPUT_CSV = Path(
    "/home/ramanlab/Documents/cole/Data/Opto/Combined/"
    "all_envelope_rows_wide_combined_base.csv"
)
FLAGGED_CSV = Path(
    "/home/ramanlab/Documents/cole/Data/Opto/Combined/flagged-flys-truth.csv"
)
OUTPUT_CSV = Path(
    "/home/ramanlab/Documents/cole/Data/Opto/Combined/blinded_scoring_results.csv"
)
SEED_FILE = Path(
    "/home/ramanlab/Documents/cole/Data/Opto/Combined/blinded_scoring_seed.json"
)

# ---------------------------------------------------------------------------
# Plot / timing constants  (match existing combinedBase envelope style)
# ---------------------------------------------------------------------------
ODOR_ON_S = 30.0
ODOR_OFF_S = 60.0
THRESHOLD_STD_MULT = 3.0
FIXED_Y_MAX = 100.0
ODOR_SHADE_COLOR = "#9e9e9e"
ODOR_SHADE_ALPHA = 0.20
THRESHOLD_COLOR = "tab:red"
THRESHOLD_ALPHA = 0.9
THRESHOLD_LW = 1.0
TRACE_COLOR = "black"
TRACE_LW = 1.2
ODOR_LINE_LW = 1.0
DEFAULT_FPS = 40.0
RANDOM_SEED = 42
MAX_FRAMES = 3600  # Only show up to frame 3600 (~90 s at 40 fps)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    """Load envelope CSV and filter to testing trials only."""
    print("Loading envelope data …")
    df = pd.read_csv(INPUT_CSV)
    df = df[df["trial_type"].str.strip().str.lower() == "testing"].copy()
    print(f"  Testing rows: {len(df)}")
    return df


def load_exclusion_set() -> set[tuple[str, int]]:
    """Build (fly, fly_number) pairs to exclude (FLY-State <= 0).

    Matches by (fly, fly_number) only—ignores dataset because several
    entries in the flagged CSV carry the wrong dataset name.
    Force-excludes october_14_batch_1 fly 1 (conflicting entries).
    """
    flagged = pd.read_csv(FLAGGED_CSV)
    score_col = [c for c in flagged.columns if "State" in c][0]
    flagged[score_col] = pd.to_numeric(flagged[score_col], errors="coerce")

    bad = flagged[flagged[score_col] <= 0]
    exclude: set[tuple[str, int]] = set()
    for _, row in bad.iterrows():
        exclude.add((str(row["fly"]).strip(), int(row["fly_number"])))

    # Conflicting entry – treat as excluded
    exclude.add(("october_14_batch_1", 1))
    return exclude


def apply_exclusions(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows whose (fly, fly_number) is in the exclusion set."""
    exclude = load_exclusion_set()
    before = len(df)
    mask = pd.Series(
        [
            (str(r["fly"]).strip(), int(r["fly_number"])) in exclude
            for _, r in df.iterrows()
        ],
        index=df.index,
    )
    df = df[~mask].copy()
    print(f"  Exclusion: {before} → {len(df)} rows ({before - len(df)} removed)")
    return df


# ---------------------------------------------------------------------------
# Randomisation helpers
# ---------------------------------------------------------------------------

def randomize_order(df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, list[int]]:
    """Shuffle rows deterministically. Returns (shuffled_df, original_indices)."""
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
    """Get trace values, capped at MAX_FRAMES and trimmed of trailing NaN."""
    vals = row[dir_val_cols].to_numpy(dtype=float)
    # Cap at MAX_FRAMES
    vals = vals[:MAX_FRAMES]
    finite_mask = np.isfinite(vals)
    if not finite_mask.any():
        return np.empty(0, dtype=float)
    last_finite = int(np.max(np.where(finite_mask)[0]))
    return vals[: last_finite + 1]


def compute_theta(env: np.ndarray, fps: float) -> float:
    """theta = median(baseline) + 3 * 1.4826 * MAD(baseline), baseline = first 30 s."""
    n_before = int(round(ODOR_ON_S * fps))
    before = env[:n_before]
    before = before[np.isfinite(before)]
    if before.size < 3:
        return float("nan")
    med = float(np.nanmedian(before))
    mad = float(np.nanmedian(np.abs(before - med)))
    return med + THRESHOLD_STD_MULT * 1.4826 * mad


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_trace(fig: plt.Figure, env: np.ndarray, fps: float) -> None:
    """Render a single blinded trace into *fig* (combinedBase style)."""
    fig.clear()
    ax = fig.add_subplot(111)

    t = np.arange(len(env), dtype=float) / fps

    # Trace
    ax.plot(t, env, linewidth=TRACE_LW, color=TRACE_COLOR)

    # Odor on / off dashed lines
    ax.axvline(ODOR_ON_S, linestyle="--", linewidth=ODOR_LINE_LW, color="black")
    ax.axvline(ODOR_OFF_S, linestyle="--", linewidth=ODOR_LINE_LW, color="black")

    # Odor shading
    ax.axvspan(ODOR_ON_S, ODOR_OFF_S, alpha=ODOR_SHADE_ALPHA, color=ODOR_SHADE_COLOR)

    # Threshold
    theta = compute_theta(env, fps)
    if math.isfinite(theta):
        ax.axhline(theta, linestyle="-", linewidth=THRESHOLD_LW,
                    color=THRESHOLD_COLOR, alpha=THRESHOLD_ALPHA)

    # Axes
    ax.set_ylim(0, FIXED_Y_MAX)
    x_max = t[-1] if len(t) > 0 else 120.0
    ax.set_xlim(0, x_max)
    ax.margins(x=0, y=0.02)
    ax.set_ylabel("Max Distance x Angle %", fontsize=10)
    ax.set_xlabel("Time (s)", fontsize=11)

    # Remove top/right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()


# ---------------------------------------------------------------------------
# Resume / save helpers
# ---------------------------------------------------------------------------

def trial_key(row: pd.Series) -> tuple[str, str, int, str]:
    return (
        str(row["dataset"]).strip(),
        str(row["fly"]).strip(),
        int(row["fly_number"]),
        str(row["trial_label"]).strip(),
    )


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


def save_score(row: pd.Series, score: int, comment: str) -> None:
    """Append a single scored row to the output CSV."""
    file_exists = OUTPUT_CSV.exists() and OUTPUT_CSV.stat().st_size > 0
    row_data = {
        "dataset": str(row["dataset"]).strip(),
        "fly": str(row["fly"]).strip(),
        "fly_number": int(row["fly_number"]),
        "trial_type": str(row["trial_type"]).strip(),
        "trial_label": str(row["trial_label"]).strip(),
        "user_score_odor": score,
        "comment": comment,
    }
    with OUTPUT_CSV.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row_data.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class BlindedScoringApp:
    """Tkinter GUI for blinded fly trace scoring."""

    def __init__(
        self,
        master: tk.Tk,
        df: pd.DataFrame,
        dir_val_cols: list[str],
        scored_keys: set[tuple[str, str, int, str]],
    ) -> None:
        self.master = master
        self.df = df
        self.dir_val_cols = dir_val_cols
        self.total = len(df)

        # Determine pending (un-scored) trials
        self.pending_indices: list[int] = [
            idx for idx in range(self.total)
            if trial_key(self.df.iloc[idx]) not in scored_keys
        ]
        self.current_pending_pos = 0
        self.already_scored = self.total - len(self.pending_indices)

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

        # Window
        self.master.title("Blinded Fly Trace Scoring")
        self.master.geometry("1050x750")
        self.master.protocol("WM_DELETE_WINDOW", self._on_close)

        # -- Matplotlib canvas --
        self.fig = plt.Figure(figsize=(9.5, 4.5), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=(10, 0))

        # -- Controls --
        controls = tk.Frame(self.master)
        controls.pack(fill=tk.X, padx=10, pady=10)

        tk.Label(controls, text="Score (0-5):", font=("Arial", 12)).grid(
            row=0, column=0, padx=(0, 5), sticky="e")
        self.score_var = tk.StringVar()
        self.score_entry = tk.Entry(controls, textvariable=self.score_var,
                                    width=5, font=("Arial", 14))
        self.score_entry.grid(row=0, column=1, padx=(0, 20), sticky="w")

        tk.Label(controls, text="Comment (optional):", font=("Arial", 12)).grid(
            row=0, column=2, padx=(0, 5), sticky="e")
        self.comment_var = tk.StringVar()
        self.comment_entry = tk.Entry(controls, textvariable=self.comment_var,
                                      width=30, font=("Arial", 11))
        self.comment_entry.grid(row=0, column=3, padx=(0, 20), sticky="w")

        self.submit_btn = tk.Button(controls, text="Submit", font=("Arial", 12, "bold"),
                                    command=self._on_submit, width=10)
        self.submit_btn.grid(row=0, column=4, padx=(0, 20))

        self.progress_var = tk.StringVar()
        tk.Label(controls, textvariable=self.progress_var,
                 font=("Arial", 11)).grid(row=0, column=5, sticky="e")
        controls.columnconfigure(5, weight=1)

        # Bind Enter to submit
        self.master.bind("<Return>", lambda _: self._on_submit())

        # Resume message
        if self.already_scored > 0:
            messagebox.showinfo(
                "Resuming",
                f"Resuming: {self.already_scored} of {self.total} already scored.\n"
                f"{len(self.pending_indices)} remaining.",
            )

        # Show first trial (or completion)
        if self.pending_indices:
            self._show_current_trial()
        else:
            self._show_completion()

    # ---- internal methods ----

    def _show_current_trial(self) -> None:
        if self.current_pending_pos >= len(self.pending_indices):
            self._show_completion()
            return

        idx = self.pending_indices[self.current_pending_pos]
        row = self.df.iloc[idx]

        env = extract_envelope(row, self.dir_val_cols)
        fps = float(row.get("fps", DEFAULT_FPS))
        if not math.isfinite(fps) or fps <= 0:
            fps = DEFAULT_FPS

        plot_trace(self.fig, env, fps)
        self.canvas.draw()

        scored_so_far = self.already_scored + self.current_pending_pos
        self.progress_var.set(f"Trial {scored_so_far + 1} of {self.total}")

        self.score_var.set("")
        self.comment_var.set("")
        self.score_entry.focus_set()

    def _on_submit(self) -> None:
        raw = self.score_var.get().strip()
        try:
            score = int(raw)
        except ValueError:
            messagebox.showerror("Invalid Score", "Score must be an integer 0–5.")
            return
        if score < 0 or score > 5:
            messagebox.showerror("Invalid Score", "Score must be between 0 and 5.")
            return

        idx = self.pending_indices[self.current_pending_pos]
        row = self.df.iloc[idx]
        save_score(row, score, self.comment_var.get().strip())

        self.current_pending_pos += 1
        if self.current_pending_pos >= len(self.pending_indices):
            self._show_completion()
        else:
            self._show_current_trial()

    def _show_completion(self) -> None:
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.text(0.5, 0.5, "All trials scored!\nThank you.",
                ha="center", va="center", fontsize=20, transform=ax.transAxes)
        ax.set_axis_off()
        self.canvas.draw()
        self.progress_var.set(f"Complete: {self.total} of {self.total}")
        self.submit_btn.config(state=tk.DISABLED)
        messagebox.showinfo("Complete", f"All {self.total} trials have been scored.")

    def _on_close(self) -> None:
        self.master.destroy()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # 1. Load & filter
    df = load_data()
    df = apply_exclusions(df)
    dir_val_cols = get_dir_val_cols(df)
    print(f"  dir_val columns: {len(dir_val_cols)}")
    print(f"  Final trial count: {len(df)}")

    # 2. Seed & randomise
    seed_info = load_seed_info()
    seed = seed_info["random_seed"] if seed_info else RANDOM_SEED
    print(f"  Seed: {seed}")

    df_shuffled, order = randomize_order(df, seed)
    save_seed_info(seed, order)

    # 3. Resume
    scored_keys = load_existing_scores()
    print(f"  Already scored: {len(scored_keys)}")

    # 4. Launch
    root = tk.Tk()
    BlindedScoringApp(root, df_shuffled, dir_val_cols, scored_keys)
    root.mainloop()


if __name__ == "__main__":
    main()
