"""Visualisation utilities for Hilbert envelope outputs.

This script extends the notebook-style plotting snippets shared by the lab
into efficient, configurable CLI commands.  It can generate reaction matrix
figures (with optional alternate trial ordering) and produce per-fly envelope
plots using the float16 matrix emitted by :mod:`scripts.envelope_exports`.

Usage examples::

    # Build both matrix variants (testing order + trained-first)
    python scripts/envelope_visuals.py matrices \
        --matrix-npy /path/to/envelope_matrix_float16.npy \
        --codes-json /path/to/code_maps.json \
        --latency-sec 2.75

    # Produce envelope traces for every fly, grouped by odor
    python scripts/envelope_visuals.py envelopes \
        --matrix-npy /path/to/envelope_matrix_float16.npy \
        --codes-json /path/to/code_maps.json \
        --latency-sec 2.75

"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch


# ---------------------------------------------------------------------------
# Canonical mappings shared across visualisations


ODOR_CANON: Mapping[str, str] = {
    "acv": "ACV",
    "apple cider vinegar": "ACV",
    "apple-cider-vinegar": "ACV",
    "3-octonol": "3-octonol",
    "3 octonol": "3-octonol",
    "3-octanol": "3-octonol",
    "3 octanol": "3-octonol",
    "benz": "Benz",
    "benzaldehyde": "Benz",
    "benz-ald": "Benz",
    "benzadhyde": "Benz",
    "ethyl butyrate": "EB",
    "optogenetics benzaldehyde": "opto_benz",
    "optogenetics benzaldehyde 1": "opto_benz_1",
    "optogenetics ethyl butyrate": "opto_EB",
    "optogenetics hexanol": "opto_hex",
    "optogenetics hex": "opto_hex",
    "hexanol": "opto_hex",
    "opto_hex": "opto_hex",
    "10s_odor_benz": "10s_Odor_Benz",
}

DISPLAY_LABEL = {
    "ACV": "ACV",
    "3-octonol": "3-Octonol",
    "Benz": "Benzaldehyde",
    "10s_Odor_Benz": "Benzaldehyde",
    "EB": "Ethyl Butyrate",
    "opto_benz": "Benzaldehyde",
    "opto_EB": "Ethyl Butyrate",
    "opto_benz_1": "Benzaldehyde",
    "opto_hex": "Optogenetics Hexanol",
}

ODOR_ORDER = [
    "ACV",
    "3-octonol",
    "Benz",
    "EB",
    "10s_Odor_Benz",
    "opto_benz",
    "opto_EB",
    "opto_benz_1",
    "opto_hex",
]

TRAINED_FIRST_ORDER = (2, 4, 5, 1, 3, 6, 7, 8, 9)
HEXANOL_LABEL = "Optogenetics Hexanol"


# ---------------------------------------------------------------------------
# Utility helpers


def _canon_dataset(value: str) -> str:
    if not isinstance(value, str):
        return "UNKNOWN"
    key = value.strip().lower()
    return ODOR_CANON.get(key, value.strip())


def _safe_dirname(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_") or "export"


def canonical_dataset(value: str) -> str:
    """Return the canonical ODOR identifier for *value*."""

    return _canon_dataset(value)


def resolve_dataset_label(values: Sequence[str] | str) -> str:
    """Return a human-readable label for one or more dataset identifiers."""

    if isinstance(values, str):
        candidates = {_canon_dataset(values)} if values else set()
    else:
        candidates = {_canon_dataset(val) for val in values if isinstance(val, str) and val}

    candidates = {val for val in candidates if val}
    if not candidates:
        return "UNKNOWN"

    if len(candidates) == 1:
        key = next(iter(candidates))
        return DISPLAY_LABEL.get(key, key)

    pretty = [DISPLAY_LABEL.get(key, key) for key in sorted(candidates)]
    return f"Mixed ({'+'.join(pretty)})"


def resolve_dataset_output_dir(base: Path, values: Sequence[str] | str) -> Path:
    """Return the output directory for the provided dataset identifiers."""

    label = resolve_dataset_label(values)
    return base / _safe_dirname(label)


def should_write(path: Path, overwrite: bool) -> bool:
    """Return ``True`` if *path* should be written, honouring overwrite policy."""

    if path.exists() and not overwrite:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    return True


def _trial_num(label: str) -> int:
    match = re.search(r"(\d+)", str(label))
    return int(match.group(1)) if match else -1


def _display_odor(dataset_canon: str, trial_label: str) -> str:
    number = _trial_num(trial_label)
    label_lower = str(trial_label).lower()
    if (
        dataset_canon == "opto_hex"
        and "testing" in label_lower
        and number in (1, 3)
    ):
        return "Apple Cider Vinegar"
    if number in (1, 3):
        return HEXANOL_LABEL
    if number in (2, 4, 5):
        return DISPLAY_LABEL.get(dataset_canon, dataset_canon)

    mapping = {
        "ACV": {6: "3-Octonol", 7: "Benzaldehyde", 8: "Citral", 9: "Linalool"},
        "3-octonol": {6: "Benzaldehyde", 7: "Citral", 8: "Linalool"},
        "Benz": {6: "Citral", 7: "Linalool"},
        "EB": {
            6: "Apple Cider Vinegar",
            7: "3-Octonol",
            8: "Benzaldehyde",
            9: "Citral",
            10: "Linalool",
        },
        "10s_Odor_Benz": {6: "Benzaldehyde", 7: "Benzaldehyde"},
        "opto_EB": {
            6: "Apple Cider Vinegar",
            7: "3-Octonol",
            8: "Benzaldehyde",
            9: "Citral",
            10: "Linalool",
        },
        "opto_benz": {6: "3-Octonol", 7: "Benzaldehyde", 8: "Citral", 9: "Linalool"},
        "opto_benz_1": {
            6: "Apple Cider Vinegar",
            7: "3-Octonol",
            8: "Ethyl Butyrate",
            9: "Citral",
            10: "Linalool",
        },
        "opto_hex": {
            6: "Benzaldehyde",
            7: "3-Octonol",
            8: "Ethyl Butyrate",
            9: "Citral",
            10: "Linalool",
        },
    }

    return mapping.get(dataset_canon, {}).get(number, trial_label)


def _is_trained_odor(dataset_canon: str, odor_name: str) -> bool:
    trained = DISPLAY_LABEL.get(dataset_canon, dataset_canon)
    return str(odor_name).strip().lower() == str(trained).strip().lower()


def _extract_env_row(env_row: np.ndarray) -> np.ndarray:
    env = env_row.astype(float, copy=False)
    mask = np.isfinite(env) & (env > 0)
    if not np.any(mask):
        return np.empty(0, dtype=float)
    return env[mask]


def _compute_theta(
    env: np.ndarray, fps: float, baseline_until_s: float, std_mult: float
) -> float:
    """Compute the response threshold using only the pre-command baseline."""

    if env.size == 0 or fps <= 0:
        return math.nan

    before_end = min(int(round(baseline_until_s * fps)), env.size)
    if before_end <= 0:
        return math.nan

    window = env[:before_end]
    return float(np.nanmean(window) + std_mult * np.nanstd(window))


def _load_matrix(matrix_path: Path, codes_json: Path) -> tuple[pd.DataFrame, list[str]]:
    matrix = np.load(matrix_path, allow_pickle=False)
    with codes_json.open("r", encoding="utf-8") as fh:
        meta = json.load(fh)

    ordered_cols: list[str] = list(meta["column_order"])
    code_maps: Mapping[str, Mapping[str, int]] = meta["code_maps"]
    df = pd.DataFrame(matrix, columns=ordered_cols)

    decode_candidates = ["dataset", "fly", "fly_number", "trial_type", "trial_label"]
    rev_maps = {
        col: {int(code): label for label, code in mapping.items()}
        for col, mapping in code_maps.items()
        if col in decode_candidates or col == "fps"
    }

    for col in decode_candidates:
        if col not in df.columns:
            continue
        values = np.rint(df[col].to_numpy(np.float32, copy=False)).astype(np.int32, copy=False)
        mapped = pd.Series(values).map(rev_maps.get(col, {})).fillna("UNKNOWN")
        df[col] = mapped

    if "fps" in df.columns:
        fps_codes = np.rint(df["fps"].to_numpy(np.float32, copy=False)).astype(np.int32, copy=False)
        if "fps" in rev_maps:
            fps_strings = pd.Series(fps_codes).map(rev_maps["fps"]).astype(str)
            df["fps"] = pd.to_numeric(fps_strings, errors="coerce")
        else:
            df["fps"] = pd.to_numeric(df["fps"], errors="coerce")
    else:
        df["fps"] = np.nan

    env_cols = [
        col
        for col in ordered_cols
        if col
        not in {"fps", "dataset", "fly", "fly_number", "trial_type", "trial_label"}
    ]
    return df, env_cols


# ---------------------------------------------------------------------------
# Reaction matrix generation


@dataclass
class MatrixPlotConfig:
    matrix_npy: Path
    codes_json: Path
    out_dir: Path
    latency_sec: float
    fps_default: float = 40.0
    before_sec: float = 30.0
    during_sec: float = 30.0
    after_window_sec: float = 30.0
    threshold_std_mult: float = 4.0
    min_samples_over: int = 20
    row_gap: float = 0.6
    height_per_gap_in: float = 3.0
    bottom_shift_in: float = 0.5
    trial_orders: Sequence[str] = field(default_factory=lambda: ("observed", "trained-first"))
    include_hexanol: bool = True
    overwrite: bool = False


def _score_trial(env: np.ndarray, fps: float, cfg: MatrixPlotConfig) -> tuple[int, int]:
    if env.size == 0:
        return (0, 0)

    fps = fps if math.isfinite(fps) and fps > 0 else cfg.fps_default
    before_end = int(round(cfg.before_sec * fps))
    shift = int(round(cfg.latency_sec * fps))
    during_start = before_end + shift
    during_end = during_start + int(round(cfg.during_sec * fps))
    after_end = during_end + int(round(cfg.after_window_sec * fps))

    total = env.size
    before_end = max(0, min(before_end, total))
    during_start = max(before_end, min(during_start, total))
    during_end = max(during_start, min(during_end, total))
    after_end = max(during_end, min(after_end, total))

    before = env[:before_end]
    during = env[during_start:during_end]
    after = env[during_end:after_end]

    if before.size == 0:
        return (0, 0)

    theta = float(np.nanmean(before) + cfg.threshold_std_mult * np.nanstd(before))
    during_hit = int(np.sum(during > theta) >= cfg.min_samples_over) if during.size else 0
    after_hit = int(np.sum(after > theta) >= cfg.min_samples_over) if after.size else 0
    return during_hit, after_hit


def _compute_category_counts(
    matrix: np.ndarray,
    labels: Sequence[str],
    trained_display: str,
    include_hexanol: bool,
) -> dict[str, int]:
    if matrix.size == 0:
        return {"Trained only": 0, "Trained + Others": 0, "Others only": 0}

    trained_idx = [idx for idx, lab in enumerate(labels) if lab.strip().lower() == trained_display.lower()]
    other_idx = [
        idx
        for idx, lab in enumerate(labels)
        if lab.strip().lower() != trained_display.lower()
        and (include_hexanol or lab.strip().lower() != HEXANOL_LABEL.lower())
    ]

    if not trained_idx:
        return {"Trained only": 0, "Trained + Others": 0, "Others only": 0}

    counts = {"Trained only": 0, "Trained + Others": 0, "Others only": 0}
    hits = np.where(matrix < 0, 0, matrix)
    for row in hits:
        trained_hit = np.any(row[trained_idx] == 1)
        other_hit = np.any(row[other_idx] == 1) if other_idx else False
        if trained_hit and not other_hit:
            counts["Trained only"] += 1
        elif trained_hit and other_hit:
            counts["Trained + Others"] += 1
        elif other_hit:
            counts["Others only"] += 1
    return counts


def _plot_category_counts(ax, counts: Mapping[str, int], total: int, title: str) -> None:
    labels = ["Trained only", "Trained + Others", "Others only"]
    values = np.array([counts.get(label, 0) for label in labels], dtype=float)
    percentages = 100.0 * values / total if total > 0 else np.zeros_like(values)
    x = np.arange(len(labels))
    bars = ax.bar(x, percentages, width=0.75, edgecolor="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("% of flies")
    ax.set_title(title, fontsize=12, weight="bold")
    for bar, pct in zip(bars, percentages):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{pct:.0f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def _style_trained_xticks(ax, labels: Sequence[str], trained_display: str, fontsize: int) -> None:
    ax.set_xticks(np.arange(len(labels)))
    styled = []
    for label in labels:
        if label.strip().lower() == trained_display.lower():
            styled.append(label.upper())
        else:
            styled.append(label)
    ax.set_xticklabels(styled, rotation=90, ha="center", va="top", fontsize=fontsize)
    for tick, label in zip(ax.get_xticklabels(), styled):
        if label.upper() == trained_display.upper():
            tick.set_color("tab:blue")
    ax.tick_params(axis="x", pad=2)


def _trial_order_for(dataset_trials: Sequence[str], order: str) -> list[str]:
    if order == "observed":
        return sorted(dataset_trials, key=_trial_num)
    if order == "trained-first":
        mapping = {trial: _trial_num(trial) for trial in dataset_trials}
        ordered: list[str] = []
        for number in TRAINED_FIRST_ORDER:
            for trial, tnum in mapping.items():
                if tnum == number and trial not in ordered:
                    ordered.append(trial)
        extras = [trial for trial in dataset_trials if trial not in ordered]
        ordered.extend(sorted(extras, key=_trial_num))
        return ordered
    raise ValueError(f"Unsupported trial order: {order}")


def _order_suffix(order: str) -> str:
    return "unordered" if order == "trained-first" else order.replace("_", "-")


def generate_reaction_matrices(cfg: MatrixPlotConfig) -> None:
    df, env_cols = _load_matrix(cfg.matrix_npy, cfg.codes_json)
    df = df[df["trial_type"].str.lower() == "testing"].copy()
    if df.empty:
        raise RuntimeError("No testing trials found in matrix; cannot build reaction matrices.")

    df["fps"] = df["fps"].replace([np.inf, -np.inf], np.nan).fillna(cfg.fps_default)
    df["dataset_canon"] = df["dataset"].map(_canon_dataset)

    env_data = df[env_cols].to_numpy(np.float32, copy=False)
    dataset_vals = df["dataset_canon"].to_numpy(str)
    fly_vals = df["fly"].to_numpy(str)
    trial_vals = df["trial_label"].to_numpy(str)
    fps_vals = df["fps"].to_numpy(float)

    scores = []
    for env_row, dataset_val, fly_val, trial_val, fps_val in zip(
        env_data, dataset_vals, fly_vals, trial_vals, fps_vals, strict=False
    ):
        env = _extract_env_row(env_row)
        during_hit, after_hit = _score_trial(env, float(fps_val), cfg)
        scores.append(
            {
                "dataset": dataset_val,
                "fly": fly_val,
                "trial": trial_val,
                "trial_num": _trial_num(trial_val),
                "during_hit": during_hit,
                "after_hit": after_hit,
            }
        )

    scores_df = pd.DataFrame(scores)
    if scores_df.empty:
        raise RuntimeError("Scoring yielded no results; verify the matrix inputs.")

    cmap = ListedColormap(["0.7", "1.0", "0.0"])
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)

    present = scores_df["dataset"].unique().tolist()
    ordered_present = [odor for odor in ODOR_ORDER if odor in present]
    extras = sorted(odor for odor in present if odor not in ODOR_ORDER)

    for order in cfg.trial_orders:
        order_suffix = _order_suffix(order)
        for odor in ordered_present + extras:
            subset = scores_df[scores_df["dataset"] == odor]
            if subset.empty:
                continue

            flies = sorted(subset["fly"].unique())
            trial_list = _trial_order_for(list(subset["trial"].unique()), order)
            pretty_labels = [_display_odor(odor, trial) for trial in trial_list]

            during_matrix = np.full((len(flies), len(trial_list)), -1, dtype=int)
            after_matrix = np.full_like(during_matrix, -1)

            fly_map = {fly: idx for idx, fly in enumerate(flies)}
            trial_map = {trial: idx for idx, trial in enumerate(trial_list)}
            for _, row in subset.iterrows():
                i = fly_map[row["fly"]]
                j = trial_map[row["trial"]]
                during_matrix[i, j] = int(row["during_hit"])
                after_matrix[i, j] = int(row["after_hit"])

            odor_label = DISPLAY_LABEL.get(odor, odor)
            trained_display = DISPLAY_LABEL.get(odor, odor)
            n_flies = len(flies)
            n_trials = len(trial_list)

            base_w = max(10.0, 0.70 * n_trials + 6.0)
            base_h = max(5.0, n_flies * 0.26 + 3.8)
            fig_w = base_w
            fig_h = base_h + cfg.row_gap * cfg.height_per_gap_in + cfg.bottom_shift_in

            xtick_fs = 9 if n_trials <= 10 else (8 if n_trials <= 16 else 7)

            during_counts = _compute_category_counts(during_matrix, pretty_labels, trained_display, cfg.include_hexanol)
            after_counts = _compute_category_counts(after_matrix, pretty_labels, trained_display, cfg.include_hexanol)

            plt.rcParams.update(
                {
                    "figure.dpi": 300,
                    "savefig.dpi": 300,
                    "axes.spines.top": False,
                    "axes.spines.right": False,
                }
            )
            fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=False)
            gs = gridspec.GridSpec(
                2,
                2,
                height_ratios=[3.0, 1.25],
                width_ratios=[1, 1],
                hspace=cfg.row_gap,
                wspace=0.10,
            )

            ax_during = fig.add_subplot(gs[0, 0])
            ax_after = fig.add_subplot(gs[0, 1])
            ax_dc = fig.add_subplot(gs[1, 0])
            ax_ac = fig.add_subplot(gs[1, 1])

            ax_during.imshow(during_matrix, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
            ax_during.set_title(
                f"{odor_label} — During\n(DURING shifted by +{cfg.latency_sec:.2f} s)",
                fontsize=14,
                weight="bold",
                linespacing=1.1,
            )
            _style_trained_xticks(ax_during, pretty_labels, trained_display, xtick_fs)
            ax_during.set_yticks([])
            ax_during.set_ylabel(f"{n_flies} Flies", fontsize=11)

            ax_after.imshow(after_matrix, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
            ax_after.set_title(
                f"{odor_label} — After (first {int(cfg.after_window_sec)} s)",
                fontsize=14,
                weight="bold",
            )
            _style_trained_xticks(ax_after, pretty_labels, trained_display, xtick_fs)
            ax_after.set_yticks([])
            ax_after.set_ylabel(f"{n_flies} Flies", fontsize=11)

            _plot_category_counts(ax_dc, during_counts, n_flies, "During — Fly Reaction Categories")
            _plot_category_counts(
                ax_ac,
                after_counts,
                n_flies,
                f"After (first {int(cfg.after_window_sec)} s) — Fly Reaction Categories",
            )

            red_patch = Patch(
                facecolor="red",
                edgecolor="red",
                alpha=0.30,
                label=f"Odor transit {cfg.latency_sec:.2f} s (pre-DURING)",
            )
            ax_during.legend(handles=[red_patch], loc="upper left", frameon=True, fontsize=9)

            shift_frac = cfg.bottom_shift_in / fig_h if fig_h else 0.0
            for axis in (ax_dc, ax_ac):
                pos = axis.get_position()
                new_y0 = max(0.05, pos.y0 - shift_frac)
                axis.set_position([pos.x0, new_y0, pos.width, pos.height])

            odor_dir = resolve_dataset_output_dir(cfg.out_dir, odor)

            png_name = (
                f"reaction_matrix_{odor.replace(' ', '_')}_{int(cfg.after_window_sec)}"
                f"_latency_{cfg.latency_sec:.3f}s"
            )
            if order_suffix != "observed":
                png_name += f"_{order_suffix}"
            png_path = odor_dir / f"{png_name}.png"
            if should_write(png_path, cfg.overwrite):
                fig.savefig(png_path, dpi=300, bbox_inches="tight")

            row_key_name = f"row_key_{odor.replace(' ', '_')}_{int(cfg.after_window_sec)}"
            if order_suffix != "observed":
                row_key_name += f"_{order_suffix}"
            row_key_path = odor_dir / f"{row_key_name}.txt"
            if should_write(row_key_path, cfg.overwrite):
                with row_key_path.open("w", encoding="utf-8") as fh:
                    for idx, fly in enumerate(flies):
                        fh.write(f"Row {idx}: {fly}\n")

            if order == "trained-first":
                export = subset.copy()
                export["odor_sent"] = export["trial"].apply(lambda t: _display_odor(odor, t))
                order_map = {trial: idx for idx, trial in enumerate(trial_list)}
                export["trial_ord"] = export["trial"].map(order_map).fillna(10**9).astype(int)
                export = export.sort_values(["fly", "trial_ord", "trial_num", "trial"])
                export_cols = ["dataset", "fly", "trial_num", "odor_sent", "during_hit", "after_hit"]
                export_path = odor_dir / f"binary_reactions_{odor.replace(' ', '_')}_{order_suffix}.csv"
                if should_write(export_path, cfg.overwrite):
                    export.to_csv(export_path, columns=export_cols, index=False)

            plt.close(fig)


# ---------------------------------------------------------------------------
# Envelope traces per fly


@dataclass
class EnvelopePlotConfig:
    matrix_npy: Path
    codes_json: Path
    out_dir: Path
    latency_sec: float
    fps_default: float = 40.0
    odor_on_s: float = 30.0
    odor_off_s: float = 60.0
    odor_latency_s: float = 0.0
    after_show_sec: float = 30.0
    threshold_std_mult: float = 4.0
    trial_type: str = "testing"
    overwrite: bool = False


def generate_envelope_plots(cfg: EnvelopePlotConfig) -> None:
    df, env_cols = _load_matrix(cfg.matrix_npy, cfg.codes_json)
    trial_type = cfg.trial_type.strip().lower()
    if trial_type not in {"testing", "training"}:
        raise ValueError(f"Unsupported trial type: {cfg.trial_type!r}")

    df = df[df["trial_type"].str.lower() == trial_type].copy()
    if df.empty:
        raise RuntimeError(
            f"No {trial_type} trials found in matrix; cannot build envelope plots."
        )

    if "fly_number" not in df.columns:
        df["fly_number"] = "UNKNOWN"
    else:
        df["fly_number"] = (
            df["fly_number"]
            .astype(str)
            .str.strip()
            .replace({"": "UNKNOWN", "nan": "UNKNOWN", "None": "UNKNOWN"})
        )

    df["fps"] = df["fps"].replace([np.inf, -np.inf], np.nan).fillna(cfg.fps_default)
    df["dataset_canon"] = df["dataset"].map(_canon_dataset)

    env_data = df[env_cols].to_numpy(np.float32, copy=False)
    fps_values = df["fps"].to_numpy(float)
    dataset_values = df["dataset_canon"].to_numpy(str)
    trial_values = df["trial_label"].to_numpy(str)

    env_lookup = {idx: env_row for idx, env_row in zip(df.index, env_data, strict=False)}
    fps_lookup = {idx: fps for idx, fps in zip(df.index, fps_values, strict=False)}
    dataset_lookup = {idx: ds for idx, ds in zip(df.index, dataset_values, strict=False)}
    trial_lookup = {idx: tr for idx, tr in zip(df.index, trial_values, strict=False)}

    odor_latency = max(cfg.odor_latency_s, 0.0)
    odor_on_cmd = cfg.odor_on_s
    odor_off_cmd = cfg.odor_off_s
    odor_on_effective = odor_on_cmd + odor_latency
    odor_off_effective = odor_off_cmd + odor_latency
    linger = max(cfg.latency_sec, 0.0)
    x_max_limit = odor_off_effective + linger + cfg.after_show_sec

    for (fly, fly_number), fly_df in df.groupby(["fly", "fly_number"], sort=False):
        fly_df = fly_df.sort_values("trial_label", key=lambda s: s.map(_trial_num))
        indices = fly_df.index.to_numpy()
        trial_curves: list[tuple[str, np.ndarray, np.ndarray, float, bool]] = []
        y_max = 0.0

        dataset_candidates = [dataset_lookup[idx] for idx in indices if dataset_lookup[idx]]
        folder_dir = resolve_dataset_output_dir(cfg.out_dir, dataset_candidates or ("UNKNOWN",))
        fly_number_label = str(fly_number)
        suffix = "" if fly_number_label.upper() == "UNKNOWN" else f"_fly{fly_number_label}"
        out_path = folder_dir / (
            f"{fly}{suffix}_{trial_type}_envelope_trials_by_odor_"
            f"{int(cfg.after_show_sec)}_shifted.png"
        )
        print(
            "[DEBUG] envelope_plots: generating",
            f"fly={fly}",
            f"fly_number={fly_number_label}",
            f"trials={len(indices)}",
            f"output={out_path}",
        )
        if out_path.exists() and not cfg.overwrite:
            continue

        for idx in indices:
            env = _extract_env_row(env_lookup[idx])
            if env.size == 0:
                continue

            fps = float(fps_lookup[idx]) if math.isfinite(fps_lookup[idx]) else cfg.fps_default
            t_full = np.arange(env.size, dtype=float) / max(fps, 1e-9)
            mask = t_full <= x_max_limit + 1e-9
            env = env[mask]
            t_full = t_full[mask]
            if env.size == 0:
                continue

            theta = _compute_theta(env, fps, odor_on_cmd, cfg.threshold_std_mult)
            dataset_canon = dataset_lookup[idx]
            odor_name = _display_odor(dataset_canon, trial_lookup[idx])
            is_trained = _is_trained_odor(dataset_canon, odor_name)

            max_local = float(np.nanmax(env)) if np.isfinite(env).any() else 0.0
            if math.isfinite(theta):
                max_local = max(max_local, theta)
            y_max = max(y_max, max_local)

            trial_curves.append((odor_name, t_full, env, theta, is_trained))

        if not trial_curves:
            continue

        plt.rcParams.update(
            {
                "figure.dpi": 300,
                "savefig.dpi": 300,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.linewidth": 0.8,
                "xtick.direction": "out",
                "ytick.direction": "out",
                "font.size": 10,
            }
        )

        n_rows = len(trial_curves)
        fig_h = max(3.0, n_rows * 1.6 + 1.5)
        fig, axes = plt.subplots(n_rows, 1, figsize=(10, fig_h), sharex=True)
        if n_rows == 1:
            axes = [axes]

        for ax, (odor_name, t, env, theta, is_trained) in zip(axes, trial_curves):
            ax.plot(t, env, linewidth=1.2, color="black")
            ax.axvline(odor_on_effective, linestyle="--", linewidth=1.0, color="black")
            ax.axvline(odor_off_effective, linestyle="--", linewidth=1.0, color="black")

            transit_on_end = min(odor_on_effective, x_max_limit)
            transit_off_end = min(odor_off_effective, x_max_limit)
            if odor_latency > 0:
                ax.axvspan(odor_on_cmd, transit_on_end, alpha=0.25, color="red")
                ax.axvspan(odor_off_cmd, transit_off_end, alpha=0.25, color="red")

            effective_off_end = min(odor_off_effective + linger, x_max_limit)
            if effective_off_end > transit_on_end:
                ax.axvspan(transit_on_end, effective_off_end, alpha=0.15, color="gray")

            if math.isfinite(theta):
                ax.axhline(theta, linestyle="-", linewidth=1.0, color="tab:red", alpha=0.9)

            ax.set_ylim(0, y_max * 1.02 if y_max > 0 else 1.0)
            ax.set_xlim(0, x_max_limit)
            ax.margins(x=0, y=0.02)
            ax.set_ylabel("RMS (a.u.)", fontsize=10)

            if is_trained:
                ax.set_title(odor_name.upper(), loc="left", fontsize=11, weight="bold", pad=2, color="tab:blue")
            else:
                ax.set_title(odor_name, loc="left", fontsize=11, weight="bold", pad=2, color="black")

        axes[-1].set_xlabel("Time (s)", fontsize=11)

        legend_handles = [
            plt.Line2D([0], [0], linestyle="--", linewidth=1.0, color="black", label="Odor at fly"),
            plt.Rectangle((0, 0), 1, 1, alpha=0.25, color="red", label=f"Valve→fly transit (~{odor_latency:.2f}s)"),
            plt.Rectangle((0, 0), 1, 1, alpha=0.15, color="gray", label="Odor present / linger"),
            plt.Line2D([0], [0], linestyle="-", linewidth=1.0, color="tab:red", label=r"$\theta = \mu_{before} + k\sigma_{before}$"),
        ]
        fig.legend(
            handles=legend_handles,
            loc="upper right",
            bbox_to_anchor=(0.98, 0.97),
            frameon=True,
            fontsize=9,
            title=f"Threshold: k = {cfg.threshold_std_mult:g}",
            title_fontsize=9,
        )

        fly_caption = fly
        if fly_number_label.upper() != "UNKNOWN":
            fly_caption = f"{fly} — Fly {fly_number_label}"
        fig.suptitle(
            f"{fly_caption} {trial_type.title()} Trials — RMS of Proboscis vs Eye Distance",
            y=0.995,
            fontsize=14,
            weight="bold",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        if should_write(out_path, cfg.overwrite):
            fig.savefig(out_path, dpi=300, bbox_inches="tight")

        plt.close(fig)


# ---------------------------------------------------------------------------
# CLI plumbing


def _parse_matrices_args(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument("--matrix-npy", type=Path, required=True, help="Float16 matrix produced by the convert step.")
    subparser.add_argument("--codes-json", type=Path, required=True, help="JSON metadata file from the convert step.")
    subparser.add_argument("--out-dir", type=Path, required=True, help="Directory for exported figures.")
    subparser.add_argument("--latency-sec", type=float, default=0.0, help="Mean odor transit latency in seconds.")
    subparser.add_argument("--fps-default", type=float, default=40.0, help="Fallback FPS when decoding rows.")
    subparser.add_argument("--before-sec", type=float, default=30.0, help="Duration of the baseline window (seconds).")
    subparser.add_argument("--during-sec", type=float, default=30.0, help="Duration of the DURING window (seconds).")
    subparser.add_argument("--after-window-sec", type=float, default=30.0, help="Duration of the AFTER window (seconds).")
    subparser.add_argument("--threshold-std-mult", type=float, default=4.0, help="Threshold multiplier applied to baseline std dev.")
    subparser.add_argument("--min-samples-over", type=int, default=20, help="Minimum samples over threshold to count a hit.")
    subparser.add_argument("--row-gap", type=float, default=0.6, help="Vertical gap between matrix and bar charts.")
    subparser.add_argument("--height-per-gap-in", type=float, default=3.0, help="Figure height added per 1.0 of row gap (inches).")
    subparser.add_argument("--bottom-shift-in", type=float, default=0.5, help="Downward shift applied to bar charts (inches).")
    subparser.add_argument(
        "--trial-order",
        action="append",
        choices=("observed", "trained-first"),
        help="Trial ordering strategy. Repeat to request multiple variants.",
    )
    subparser.add_argument(
        "--exclude-hexanol",
        action="store_true",
        help="Exclude Optogenetics Hexanol from 'other' reaction counts.",
    )
    subparser.add_argument("--overwrite", action="store_true", help="Rebuild plots even if the target files exist.")


def _parse_envelopes_args(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument("--matrix-npy", type=Path, required=True, help="Float16 matrix produced by the convert step.")
    subparser.add_argument("--codes-json", type=Path, required=True, help="JSON metadata file from the convert step.")
    subparser.add_argument("--out-dir", type=Path, required=True, help="Directory for per-fly envelope plots.")
    subparser.add_argument("--latency-sec", type=float, default=0.0, help="Mean odor transit latency in seconds.")
    subparser.add_argument("--fps-default", type=float, default=40.0, help="Fallback FPS when decoding rows.")
    subparser.add_argument("--odor-on-s", type=float, default=30.0, help="Commanded odor ON timestamp (seconds).")
    subparser.add_argument("--odor-off-s", type=float, default=60.0, help="Commanded odor OFF timestamp (seconds).")
    subparser.add_argument(
        "--odor-latency-s",
        type=float,
        default=0.0,
        help="Transit delay between valve command and odor at the fly (seconds).",
    )
    subparser.add_argument("--after-show-sec", type=float, default=30.0, help="Duration to display after odor off (seconds).")
    subparser.add_argument("--threshold-std-mult", type=float, default=4.0, help="Threshold multiplier applied to baseline std dev.")
    subparser.add_argument(
        "--trial-type",
        choices=("testing", "training"),
        default="testing",
        help="Trial type to visualise.",
    )
    subparser.add_argument("--overwrite", action="store_true", help="Rebuild plots even if the target files exist.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    matrices_parser = subparsers.add_parser("matrices", help="Generate reaction matrix figures.")
    _parse_matrices_args(matrices_parser)

    envelopes_parser = subparsers.add_parser("envelopes", help="Generate per-fly envelope plots.")
    _parse_envelopes_args(envelopes_parser)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.command == "matrices":
        trial_orders: Sequence[str] = args.trial_order or ("observed", "trained-first")
        cfg = MatrixPlotConfig(
            matrix_npy=args.matrix_npy,
            codes_json=args.codes_json,
            out_dir=args.out_dir,
            latency_sec=args.latency_sec,
            fps_default=args.fps_default,
            before_sec=args.before_sec,
            during_sec=args.during_sec,
            after_window_sec=args.after_window_sec,
            threshold_std_mult=args.threshold_std_mult,
            min_samples_over=args.min_samples_over,
            row_gap=args.row_gap,
            height_per_gap_in=args.height_per_gap_in,
            bottom_shift_in=args.bottom_shift_in,
            trial_orders=trial_orders,
            include_hexanol=not args.exclude_hexanol,
            overwrite=args.overwrite,
        )
        generate_reaction_matrices(cfg)
        return

    if args.command == "envelopes":
        cfg = EnvelopePlotConfig(
            matrix_npy=args.matrix_npy,
            codes_json=args.codes_json,
            out_dir=args.out_dir,
            latency_sec=args.latency_sec,
            fps_default=args.fps_default,
            odor_on_s=args.odor_on_s,
            odor_off_s=args.odor_off_s,
            odor_latency_s=args.odor_latency_s,
            after_show_sec=args.after_show_sec,
            threshold_std_mult=args.threshold_std_mult,
            trial_type=args.trial_type,
            overwrite=args.overwrite,
        )
        generate_envelope_plots(cfg)
        return

    parser.error(f"Unknown command: {args.command!r}")


if __name__ == "__main__":  # pragma: no cover
    main()

