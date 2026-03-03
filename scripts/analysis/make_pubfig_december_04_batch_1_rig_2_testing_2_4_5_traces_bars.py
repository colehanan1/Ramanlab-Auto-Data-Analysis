#!/usr/bin/env python3
"""Standalone figure: testing 2/4/5 traces + PER% bars for december_04_batch_1_rig_2 fly 1.

Uses combined-base envelope matrix (Distance x Angle %) and renders:
1) Traces for testing_2, testing_4, testing_5 with shared axes + odor span.
2) PER% bar plot by odor during testing (ordered: 3-Octonol, ACV, Linalool).
"""

from __future__ import annotations

from pathlib import Path

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path("/home/ramanlab/Documents/cole")

MATRIX_PATH = ROOT / "Data/Opto/Combined/matrix/combined_base/envelope_matrix_float16.npy"
CODES_PATH = ROOT / "Data/Opto/Combined/matrix/combined_base/code_maps.json"

OUT_PATH = ROOT / "Data/imagesForFigures/imagesForOptoHEXTesting/december_04_batch_1_rig_2_testing_2_4_5_traces_bars.png"

FLY_NAME = "december_04_batch_1_rig_2"
FLY_NUMBER = "1"

TRACE_TRIALS = [2, 4, 5]
TRACE_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]
ODOR_LABEL = "Apple Cider Vinegar"

BAR_ORDER = [
    ("3-Octonol", "3-Octanol"),
    ("Apple Cider Vinegar", "ACV"),
    ("Benzaldehyde", "Benzaldehyde"),
    ("Citral", "Citral"),
    ("Ethyl Butyrate", "Ethyl Butyrate"),
    ("Hexanol", "Hexanol"),
    ("Linalool", "Linalool"),
]
MANUAL_BAR_VALUES = {
    "Hexanol": 19.0,
    "Apple Cider Vinegar": 8.0,
    "3-Octonol": 19.0,
    "Ethyl Butyrate": 56.0,
    "Benzaldehyde": 8.0,
    "Citral": 8.0,
    "Linalool": 0.0,
}

ODOR_ON_S = 32.15
ODOR_OFF_S = 62.15
AFTER_SHOW_S = 30.0
Y_MAX = 100.0


def _trial_num(trial_label: str) -> int:
    for token in str(trial_label).split("_"):
        if token.isdigit():
            return int(token)
    digits = "".join([c for c in str(trial_label) if c.isdigit()])
    return int(digits) if digits else -1


def _display_odor(dataset_canon: str, trial_label: str) -> str:
    """Minimal mapping for opto_ACV testing trials."""
    number = _trial_num(trial_label)
    dataset = str(dataset_canon)
    if dataset == "opto_ACV":
        if number in (1, 3):
            return "Hexanol"
        if number in (2, 4, 5):
            return "Apple Cider Vinegar"
        mapping = {
            6: "3-Octonol",
            7: "Ethyl Butyrate",
            8: "Benzaldehyde",
            9: "Citral",
            10: "Linalool",
        }
        return mapping.get(number, str(trial_label))
    return str(trial_label)


def _load_matrix() -> tuple[pd.DataFrame, list[str]]:
    with CODES_PATH.open("r", encoding="utf-8") as fh:
        meta = json.load(fh)
    matrix = np.load(MATRIX_PATH, allow_pickle=False)
    df = pd.DataFrame(matrix, columns=meta["column_order"])

    code_maps = meta["code_maps"]
    rev_maps = {
        col: {int(code): label for label, code in mapping.items()}
        for col, mapping in code_maps.items()
    }
    for col in ["dataset", "fly", "fly_number", "trial_type", "trial_label"]:
        if col in df.columns:
            vals = np.rint(df[col].to_numpy(np.float32, copy=False)).astype(np.int32, copy=False)
            df[col] = pd.Series(vals).map(rev_maps.get(col, {})).fillna("UNKNOWN")

    if "fps" in df.columns:
        fps_codes = np.rint(df["fps"].to_numpy(np.float32, copy=False)).astype(np.int32, copy=False)
        if "fps" in rev_maps:
            fps_strings = pd.Series(fps_codes).map(rev_maps["fps"]).astype(str)
            df["fps"] = pd.to_numeric(fps_strings, errors="coerce")
        else:
            df["fps"] = pd.to_numeric(df["fps"], errors="coerce")
    else:
        df["fps"] = np.nan

    env_cols = [c for c in meta["env_columns"] if str(c).startswith("dir_val_")]
    return df, env_cols


def _extract_env(row: pd.Series, env_cols: list[str]) -> np.ndarray:
    env = row[env_cols].to_numpy(float, copy=False)
    env = env[np.isfinite(env) & (env > 0)]
    return env * 2.0  # correct scaling


def _max_during(env: np.ndarray, fps: float) -> float:
    if env.size == 0 or fps <= 0:
        return float("nan")
    t = np.arange(env.size, dtype=float) / max(fps, 1e-9)
    mask = (t >= ODOR_ON_S) & (t <= ODOR_OFF_S)
    if not np.any(mask):
        return float("nan")
    return float(np.nanmax(env[mask]))


def main() -> None:
    # Ensure Arial is registered if installed
    for font_path in [
        "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/Arial_Italic.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/Arial_Bold_Italic.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/arialbd.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/ariali.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/arialbi.ttf",
    ]:
        path = Path(font_path)
        if path.exists():
            font_manager.fontManager.addfont(str(path))

    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "font.family": "Arial",
            "font.size": 14,
            "svg.fonttype": "none",
        }
    )

    df, env_cols = _load_matrix()
    df = df[
        (df["fly"] == FLY_NAME)
        & (df["fly_number"] == FLY_NUMBER)
        & (df["trial_type"] == "testing")
    ].copy()

    if df.empty:
        raise RuntimeError("No testing trials found for december_04_batch_1_rig_2 fly 1.")

    df["trial_num"] = df["trial_label"].map(_trial_num)
    df["odor"] = df.apply(
        lambda r: _display_odor(str(r.get("dataset", "UNKNOWN")), str(r.get("trial_label", ""))),
        axis=1,
    )

    # Trace rows for testing 2/4/5
    trace_rows = df[df["trial_num"].isin(TRACE_TRIALS)].sort_values("trial_num")
    if trace_rows.empty:
        raise RuntimeError("No testing_2/4/5 rows found for december_04_batch_1_rig_2 fly 1.")

    # PER% bars by odor across all testing trials for this fly
    odor_values: dict[str, list[float]] = {}
    for _, row in df.iterrows():
        env = _extract_env(row, env_cols)
        fps = float(row.get("fps", 40.0)) if np.isfinite(row.get("fps", 40.0)) else 40.0
        odor = str(row.get("odor", "UNKNOWN"))
        max_val = _max_during(env, fps)
        if np.isfinite(max_val):
            odor_values.setdefault(odor, []).append(max_val)

    bar_values = []
    bar_labels = []
    for odor_key, odor_label in BAR_ORDER:
        if MANUAL_BAR_VALUES:
            bar_values.append(float(MANUAL_BAR_VALUES.get(odor_key, 0.0)))
        else:
            vals = odor_values.get(odor_key, [])
            bar_values.append(float(np.nanmean(vals)) if vals else 0.0)
        bar_labels.append(odor_label)

    # Build figure
    fig = plt.figure(figsize=(11, 9.5))
    gs = fig.add_gridspec(
        3, 1, height_ratios=[1.0, 0.02, 1.0], hspace=0.20
    )

    ax = fig.add_subplot(gs[0, 0])
    for color, (_, row) in zip(TRACE_COLORS, trace_rows.iterrows()):
        env = _extract_env(row, env_cols)
        fps = float(row.get("fps", 40.0)) if np.isfinite(row.get("fps", 40.0)) else 40.0
        t = np.arange(env.size, dtype=float) / max(fps, 1e-9)
        mask = t <= ODOR_OFF_S + AFTER_SHOW_S + 1e-9
        t = t[mask]
        env = env[mask]
        ax.plot(t, env, linewidth=2.0, color=color)

    ax.axvspan(ODOR_ON_S, ODOR_OFF_S, alpha=0.20, color="#9e9e9e")
    ax.set_ylim(0, Y_MAX)
    ax.set_xlim(0, ODOR_OFF_S + AFTER_SHOW_S)
    ax.set_ylabel("Max Distance x Angle (%)", fontsize=12)
    ax.set_xlabel("Time (s)", fontsize=12)
    trace_handles = [
        Line2D([0], [0], color=TRACE_COLORS[0], lw=2.0, label="1st presentation"),
        Line2D([0], [0], color=TRACE_COLORS[1], lw=2.0, label="2nd presentation"),
        Line2D([0], [0], color=TRACE_COLORS[2], lw=2.0, label="3rd presentation"),
        Patch(facecolor="#9e9e9e", edgecolor="none", alpha=0.20, label=ODOR_LABEL),
    ]
    ax.legend(
        handles=trace_handles,
        loc="upper right",
        frameon=True,
        fontsize=14,
    )

    # Bar plot
    ax_bar = fig.add_subplot(gs[2, 0])
    x_pos = np.arange(len(bar_labels))
    bar_colors = ["#6c6c6c"] * len(bar_labels)
    bar_edges = ["black"] * len(bar_labels)
    bar_linewidths = [0.8] * len(bar_labels)
    for i, (odor_key, _) in enumerate(BAR_ORDER):
        if odor_key.lower() == "apple cider vinegar":
            bar_colors[i] = "#1f77b4"
            bar_linewidths[i] = 1.2

    bars = ax_bar.bar(
        x_pos,
        bar_values,
        color=bar_colors,
        edgecolor=bar_edges,
        linewidth=bar_linewidths,
    )
    ax_bar.set_ylim(0, 75.0)
    ax_bar.set_ylabel("PER%", fontsize=12)
    ax_bar.set_title("PER% During Testing by Odor", fontsize=14, weight="bold", loc="center", pad=6)
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(bar_labels, fontsize=12)
    ax_bar.tick_params(axis="y", labelsize=14)

    for label, (odor_key, _) in zip(ax_bar.get_xticklabels(), BAR_ORDER):
        if odor_key.lower() == "apple cider vinegar":
            label.set_color("#1f77b4")
            label.set_fontweight("bold")
        else:
            label.set_color("black")
            label.set_fontweight("normal")

    ax_bar.text(
        0.98,
        0.95,
        "n = 16",
        transform=ax_bar.transAxes,
        ha="right",
        va="top",
        fontsize=14,
        color="black",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="black", linewidth=0.8),
    )

    for rect, val in zip(bars, bar_values):
        ax_bar.text(
            rect.get_x() + rect.get_width() / 2.0,
            min(val + 1.5, 73.0),
            f"{val:.0f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            color="black",
        )

    fig.suptitle(
        "Conditioned Fruit Fly — Testing 2, 4, 5 Traces + PER% Bars (ACV Trained)",
        fontsize=14,
        weight="bold",
        y=0.96,
    )

    _save_all(fig, OUT_PATH)
    plt.close(fig)

    print(f"[OK] Wrote {OUT_PATH}")


def _save_all(fig: plt.Figure, out_path: Path) -> None:
    out_path = Path(out_path)
    fig.savefig(out_path, bbox_inches="tight")

    svg_path = out_path.with_suffix(".svg")
    fig.savefig(svg_path, bbox_inches="tight")


if __name__ == "__main__":
    main()
