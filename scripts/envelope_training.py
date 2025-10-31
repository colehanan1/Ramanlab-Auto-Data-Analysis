#!/usr/bin/env python3
"""Training-focused envelope utilities backed by the float16 matrix exports."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd

from scripts.envelope_visuals import EnvelopePlotConfig, generate_envelope_plots

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
    "eb_control": "EB_control",
    "eb control": "EB_control",
    "hex_control": "hex_control",
    "hex control": "hex_control",
    "benz_control": "benz_control",
    "benz control": "benz_control",
    "optogenetics benzaldehyde": "opto_benz",
    "optogenetics benzaldehyde 1": "opto_benz_1",
    "optogenetics ethyl butyrate": "opto_EB",
    "10s_odor_benz": "10s_Odor_Benz",
    "optogenetics hexanol": "opto_hex",
    "optogenetics hex": "opto_hex",
    "hexanol": "opto_hex",
    "opto_hex": "opto_hex",
}

DISPLAY_LABEL = {
    "ACV": "ACV",
    "3-octonol": "3-Octonol",
    "Benz": "Benzaldehyde",
    "10s_Odor_Benz": "Benzaldehyde",
    "EB": "Ethyl Butyrate",
    "EB_control": "EB Control",
    "hex_control": "Hexanol Control",
    "benz_control": "Benzaldehyde Control",
    "opto_benz": "Benzaldehyde",
    "opto_benz_1": "Benzaldehyde",
    "opto_EB": "Ethyl Butyrate",
    "opto_hex": "Hexanol",
}

HEXANOL_LABEL = "Hexanol"

PRIMARY_ODOR_LABEL = {
    "EB_control": "Ethyl Butyrate",
    "hex_control": HEXANOL_LABEL,
    "benz_control": "Benzaldehyde",
}

TRAINING_PRIMARY_TRIALS = {1, 2, 3, 4, 6, 8}

TRAINING_SPECIAL_CASES = {
    "EB_control": {5: HEXANOL_LABEL, 7: HEXANOL_LABEL},
    "hex_control": {5: "Apple Cider Vinegar", 7: "Apple Cider Vinegar"},
}


# ---------------------------------------------------------------------------
# Matrix loading helpers
# ---------------------------------------------------------------------------


def _canon_dataset(value: str) -> str:
    if not isinstance(value, str):
        return "UNKNOWN"
    return ODOR_CANON.get(value.strip().lower(), value.strip())


def _safe_dirname(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_") or "export"


def _dataset_label(dataset: str) -> str:
    canon = _canon_dataset(dataset)
    return DISPLAY_LABEL.get(canon, canon)


def _target_dir(base: Path, datasets: Sequence[str] | str) -> Path:
    if isinstance(datasets, str):
        values = {_canon_dataset(datasets)} if datasets else set()
    else:
        values = {_canon_dataset(val) for val in datasets if isinstance(val, str) and val}

    values = {val for val in values if val}
    if not values:
        label = "UNKNOWN"
    elif len(values) == 1:
        key = next(iter(values))
        label = DISPLAY_LABEL.get(key, key)
    else:
        pretty = [DISPLAY_LABEL.get(key, key) for key in sorted(values)]
        label = f"Mixed ({'+'.join(pretty)})"
    return base / _safe_dirname(label)


def _should_write(path: Path, overwrite: bool) -> bool:
    if path.exists() and not overwrite:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    return True


def _trial_num(label: str) -> int:
    match = re.search(r"(\d+)", str(label))
    return int(match.group(1)) if match else -1


def _trained_label(dataset_canon: str) -> str:
    return PRIMARY_ODOR_LABEL.get(
        dataset_canon, DISPLAY_LABEL.get(dataset_canon, dataset_canon)
    )


def _display_odor(dataset_canon: str, trial_label: str) -> str:
    number = _trial_num(trial_label)
    label_lower = str(trial_label).lower()
    if "training" in label_lower:
        special = TRAINING_SPECIAL_CASES.get(dataset_canon, {})
        if number in special:
            return special[number]
        if number in TRAINING_PRIMARY_TRIALS:
            return _trained_label(dataset_canon)

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
        "EB": {6: "Apple Cider Vinegar", 7: "3-Octonol", 8: "Benzaldehyde", 9: "Citral", 10: "Linalool"},
        "EB_control": {
            6: "Apple Cider Vinegar",
            7: "3-Octonol",
            8: "Benzaldehyde",
            9: "Citral",
            10: "Linalool",
        },
        "10s_Odor_Benz": {6: "Benzaldehyde", 7: "Benzaldehyde"},
        "opto_EB": {6: "Apple Cider Vinegar", 7: "3-Octonol", 8: "Benzaldehyde", 9: "Citral", 10: "Linalool"},
        "opto_benz": {6: "3-Octonol", 7: "Benzaldehyde", 8: "Citral", 9: "Linalool"},
        "opto_benz_1": {6: "Apple Cider Vinegar", 7: "3-Octonol", 8: "Ethyl Butyrate", 9: "Citral", 10: "Linalool"},
        "opto_hex": {
            6: "Benzaldehyde",
            7: "3-Octonol",
            8: "Ethyl Butyrate",
            9: "Citral",
            10: "Linalool",
        },
    }
    return mapping.get(dataset_canon, {}).get(number, trial_label)


def _load_envelope_matrix(matrix_path: Path, codes_json: Path) -> tuple[pd.DataFrame, list[str]]:
    matrix = np.load(matrix_path, allow_pickle=False)
    with codes_json.open("r", encoding="utf-8") as fh:
        meta = json.load(fh)

    ordered_cols: list[str] = meta["column_order"]
    code_maps: Mapping[str, Mapping[str, int]] = meta["code_maps"]
    df = pd.DataFrame(matrix, columns=ordered_cols)

    decode_cols = [c for c in ("dataset", "fly", "trial_type", "trial_label", "fps") if c in ordered_cols]
    for col in decode_cols:
        if col == "fps":
            df[col] = pd.to_numeric(df[col], errors="coerce")
            continue
        mapping = code_maps.get(col)
        if not mapping:
            continue
        inverse = {code: label for label, code in mapping.items()}
        df[col] = df[col].astype(int).map(inverse).fillna("UNKNOWN")

    if "fps" not in df.columns:
        df["fps"] = np.nan

    env_cols = [c for c in ordered_cols if c not in {"dataset", "fly", "trial_type", "trial_label", "fps"}]
    return df, env_cols


def _extract_env(row: pd.Series, env_cols: Sequence[str]) -> np.ndarray:
    env = row[env_cols].to_numpy(dtype=float)
    if env.ndim == 0:
        return np.empty(0, dtype=float)
    mask = np.isfinite(env) & (env > 0)
    return env[mask]


def _latency_to_cross(
    env: np.ndarray,
    fps: float,
    before_sec: float,
    during_sec: float,
    threshold_mult: float,
) -> Optional[float]:
    if env.size == 0 or not np.isfinite(fps) or fps <= 0:
        return None

    b_end = min(int(round(before_sec * fps)), env.size)
    d_end = min(b_end + int(round(during_sec * fps)), env.size)
    before = env[:b_end]
    during = env[b_end:d_end]
    if before.size == 0 or during.size == 0:
        return None

    mu = float(np.nanmean(before))
    sd = float(np.nanstd(before))
    theta = mu + threshold_mult * sd
    idx = np.where(during > theta)[0]
    if idx.size == 0:
        return None
    return float(idx[0]) / fps


# ---------------------------------------------------------------------------
# Latency visualisations
# ---------------------------------------------------------------------------


def _plot_latency_per_fly(
    lat_df: pd.DataFrame,
    out_dir: Path,
    *,
    latency_ceiling: float,
    trials_of_interest: Sequence[int],
    overwrite: bool,
) -> None:
    for fly in sorted(lat_df["fly"].unique()):
        subset = lat_df[lat_df["fly"] == fly]
        if subset.empty:
            continue

        datasets = subset["dataset_canon"].dropna().unique().tolist()
        target_dir = _target_dir(out_dir, datasets or ("UNKNOWN",))
        out_png = target_dir / f"{fly}_training_{'_'.join(map(str, trials_of_interest))}_latency.png"
        if out_png.exists() and not overwrite:
            continue

        latencies = []
        labels = []
        for trial_num in trials_of_interest:
            labels.append(f"Training {trial_num}")
            row = subset[subset["trial_num"] == trial_num]
            latencies.append(row["latency"].iloc[0] if not row.empty else None)

        any_response = any(lat is not None and lat <= latency_ceiling for lat in latencies)

        if not any_response:
            fig, ax = plt.subplots(figsize=(6.5, 3.2))
            ax.set_title(f"{fly} — Time to PER", pad=10, fontsize=14, weight="bold")
            ax.set_xticks(np.arange(len(labels)))
            ax.set_xticklabels(labels)
            ax.set_ylim(0, latency_ceiling + 2.0)
            ax.text(0.5, 0.55, "NR", transform=ax.transAxes, ha="center", va="center", fontsize=18, color="#666666", weight="bold")
            ax.set_ylabel("Time After Odor Sent (s)")
            ax.axhline(latency_ceiling, linestyle="--", linewidth=1.1, color="#444444")
            trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
            ax.text(0.995, latency_ceiling + 0.12, f"NR if > {latency_ceiling:.1f} s", transform=trans, ha="right", va="bottom", fontsize=10, color="#444444", clip_on=False)
            fig.tight_layout()
            if _should_write(out_png, overwrite):
                fig.savefig(out_png, dpi=300)
            plt.close(fig)
            continue

        values = []
        annotations = []
        colors = []
        for lat in latencies:
            if lat is None or lat > latency_ceiling:
                values.append(latency_ceiling)
                annotations.append("NR")
                colors.append("#BDBDBD")
            else:
                values.append(lat)
                annotations.append(f"{lat:.2f}s")
                colors.append("#1A1A1A")

        fig, ax = plt.subplots(figsize=(6.5, 3.6))
        x = np.arange(len(labels))
        bars = ax.bar(x, values, width=0.6, color=colors, edgecolor="black", linewidth=1.0)

        for bar, text in zip(bars, annotations):
            ypos = max(bar.get_height() * 0.5, 0.35)
            ax.text(bar.get_x() + bar.get_width() / 2, ypos, text, ha="center", va="center", fontsize=10, color="white" if text != "NR" else "#444444")

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Time After Odor Sent (s)")
        ax.set_ylim(0, latency_ceiling + 2.5)
        ax.axhline(latency_ceiling, linestyle="--", linewidth=1.1, color="#444444")
        trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
        ax.text(0.995, latency_ceiling + 0.12, f"NR if > {latency_ceiling:.1f} s", transform=trans, ha="right", va="bottom", fontsize=10, color="#444444", clip_on=False)
        ax.set_title(f"{fly} — Time to PER", pad=10, fontsize=14, weight="bold")

        fig.tight_layout()
        if _should_write(out_png, overwrite):
            fig.savefig(out_png, dpi=300)
        plt.close(fig)


def _plot_latency_by_odor(
    lat_df: pd.DataFrame,
    out_dir: Path,
    *,
    latency_ceiling: float,
    trials_of_interest: Sequence[int],
    overwrite: bool,
) -> None:
    for odor in sorted(lat_df["dataset_canon"].unique()):
        target_dir = _target_dir(out_dir, odor)
        filename = f"{odor}_training_{'_'.join(map(str, trials_of_interest))}_mean_latency.png"
        out_png = target_dir / filename
        if out_png.exists() and not overwrite:
            continue

        subset = lat_df[lat_df["dataset_canon"] == odor]
        labels = [f"Training {n}" for n in trials_of_interest]
        odor_label = _dataset_label(odor)

        if subset.empty:
            fig, ax = plt.subplots(figsize=(6.8, 3.2))
            ax.set_title(f"{odor_label} — Mean Time to PER", pad=10, fontsize=14, weight="bold")
            ax.set_xticks(np.arange(len(labels)))
            ax.set_xticklabels(labels)
            ax.set_ylim(0, latency_ceiling + 2.0)
            ax.text(0.5, 0.55, "NR", transform=ax.transAxes, ha="center", va="center", fontsize=18, color="#666666", weight="bold")
            ax.set_ylabel("Time After Odor Sent (s)")
            fig.tight_layout()
            if _should_write(out_png, overwrite):
                fig.savefig(out_png, dpi=300)
            plt.close(fig)
            continue

        means = []
        sems = []
        counts = []
        for trial_num in trials_of_interest:
            values = subset[subset["trial_num"] == trial_num]["lat_for_mean"].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            counts.append(finite.size)
            if finite.size == 0:
                means.append(math.nan)
                sems.append(math.nan)
                continue
            means.append(float(finite.mean()))
            if finite.size > 1:
                sems.append(float(finite.std(ddof=1)) / math.sqrt(finite.size))
            else:
                sems.append(0.0)

        y = np.nan_to_num(np.array(means, dtype=float), nan=0.0)
        yerr_up = np.nan_to_num(np.array(sems, dtype=float), nan=0.0)
        yerr = np.vstack([np.zeros_like(yerr_up), yerr_up])

        fig, ax = plt.subplots(figsize=(6.8, 3.8))
        x = np.arange(len(labels))
        bars = ax.bar(x, y, width=0.6, color="#1A1A1A", edgecolor="black", linewidth=1.0)
        ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor="black", elinewidth=1.2, capsize=4)

        for idx, bar in enumerate(bars):
            n_resp = counts[idx]
            sem_val = yerr_up[idx]
            if n_resp == 0:
                label_y = max(0.5, y[idx] + 0.08)
                ax.text(bar.get_x() + bar.get_width() / 2, label_y, "NR", ha="center", va="bottom", fontsize=9, color="#444444")
                continue
            inside = max(y[idx] * 0.5, min(y[idx] - 0.10, y[idx] * 0.90))
            ax.text(bar.get_x() + bar.get_width() / 2, inside, f"{y[idx]:.2f}s", ha="center", va="top", fontsize=10, color="white")
            top = y[idx] + (sem_val if np.isfinite(sem_val) else 0.0) + 0.06
            ax.text(bar.get_x() + bar.get_width() / 2, top, f"SEM={sem_val:.2f}s\nn={n_resp}", ha="center", va="bottom", fontsize=9, color="#333333")

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Time After Odor Sent (s)")
        ymax = max(latency_ceiling + 2.0, float((y + yerr_up).max()) + 1.2)
        ax.set_ylim(0, ymax)
        ax.axhline(latency_ceiling, linestyle="--", linewidth=1.0, color="#6f6f6f")
        trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
        ax.text(0.995, latency_ceiling + 0.08, f"NR if > {latency_ceiling:.1f} s", transform=trans, ha="right", va="bottom", fontsize=9, color="#6f6f6f")
        ax.set_title(f"{odor_label} — Mean Time to PER", pad=10, fontsize=14, weight="bold")

        fig.tight_layout()
        if _should_write(out_png, overwrite):
            fig.savefig(out_png, dpi=300)
        plt.close(fig)


def _plot_latency_grand_means(
    lat_df: pd.DataFrame,
    out_dir: Path,
    latency_ceiling: float,
    overwrite: bool,
) -> None:
    odors = sorted(lat_df["dataset_canon"].dropna().unique())
    rows = []
    means = []
    sems = []
    counts = []

    for odor in odors:
        values = lat_df[lat_df["dataset_canon"] == odor]["lat_for_mean"].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        n_resp = finite.size
        counts.append(n_resp)
        if n_resp == 0:
            means.append(math.nan)
            sems.append(math.nan)
        else:
            mu = float(finite.mean())
            if n_resp > 1:
                sd = float(finite.std(ddof=1))
                sem = sd / math.sqrt(n_resp)
            else:
                sem = 0.0
            means.append(mu)
            sems.append(sem)
        rows.append({"odor": odor, "n_resp": n_resp, "mean_s": means[-1], "sem_s": sems[-1]})

    csv_path = out_dir / "grand_mean_by_odor_latency.csv"
    if _should_write(csv_path, overwrite):
        pd.DataFrame(rows).to_csv(csv_path, index=False)

    out_png = out_dir / "grand_mean_by_odor_latency.png"
    if out_png.exists() and not overwrite:
        return

    if sum(counts) == 0:
        fig, ax = plt.subplots(figsize=(7.2, 3.2))
        ax.set_title("Grand Mean Time to Reaction by Trained Odor", pad=10, fontsize=14, weight="bold")
        ax.set_xticks(np.arange(len(odors)))
        ax.set_xticklabels(odors)
        ax.set_ylim(0, latency_ceiling + 2.0)
        ax.text(0.5, 0.55, "NR", transform=ax.transAxes, ha="center", va="center", fontsize=18, color="#666666", weight="bold")
        ax.set_ylabel("Time After Odor Sent (s)")
        ax.axhline(latency_ceiling, linestyle="--", linewidth=1.0, color="#6f6f6f")
        fig.tight_layout()
        if _should_write(out_png, overwrite):
            fig.savefig(out_png, dpi=300)
        plt.close(fig)
        return

    y = np.nan_to_num(np.array(means, dtype=float), nan=0.0)
    yerr_up = np.nan_to_num(np.array(sems, dtype=float), nan=0.0)
    yerr = np.vstack([np.zeros_like(yerr_up), yerr_up])

    fig, ax = plt.subplots(figsize=(max(7.2, 1.8 * len(odors)), 3.8))
    x = np.arange(len(odors))
    bars = ax.bar(x, y, width=0.6, color="#1A1A1A", edgecolor="black", linewidth=1.0)
    ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor="black", elinewidth=1.2, capsize=4)

    for idx, bar in enumerate(bars):
        n_resp = counts[idx]
        sem_val = yerr_up[idx]
        if n_resp == 0:
            label_y = max(0.5, y[idx] + 0.08)
            ax.text(bar.get_x() + bar.get_width() / 2, label_y, "NR", ha="center", va="bottom", fontsize=9, color="#444444")
            continue
        label_y = y[idx] + (sem_val if np.isfinite(sem_val) else 0.0) + 0.06
        ax.text(bar.get_x() + bar.get_width() / 2, label_y, f"SEM={sem_val:.2f} s\nn={n_resp}", ha="center", va="bottom", fontsize=9, color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels(odors)
    ax.set_ylabel("Time After Odor Sent (s)")
    ymax = max(latency_ceiling + 2.0, float((y + yerr_up).max()) + 1.2)
    ax.set_ylim(0, ymax)
    ax.axhline(latency_ceiling, linestyle="--", linewidth=1.0, color="#6f6f6f")
    ax.set_title("Grand Mean Time to Reaction by Trained Odor", pad=10, fontsize=14, weight="bold")

    fig.tight_layout()
    if _should_write(out_png, overwrite):
        fig.savefig(out_png, dpi=300)
    plt.close(fig)


def latency_reports(
    matrix_path: Path,
    codes_json: Path,
    out_dir: Path,
    *,
    before_sec: float,
    during_sec: float,
    threshold_mult: float | None = None,
    threshold_std_mult: float | None = None,
    latency_ceiling: float,
    trials_of_interest: Sequence[int],
    fps_default: float,
    overwrite: bool,
) -> None:
    """Generate latency plots and summaries for the requested trials.

    Historically this function only accepted ``threshold_std_mult`` while recent
    workflow updates switched to ``threshold_mult``.  Accept both names (with
    the former taking precedence when both supplied) so that existing configs
    and cached CLI invocations continue to function without raising
    ``TypeError`` when mixing versions of the scripts.
    """
    if threshold_mult is None and threshold_std_mult is None:
        # Maintain the previous implicit default of 4.0 when neither keyword is
        # provided.  This mirrors the CLI default and the behaviour prior to the
        # signature change.
        threshold_mult = 4.0

    # Allow either keyword while favouring the legacy ``threshold_std_mult`` if
    # both are present.  This keeps behaviour consistent for older configs that
    # may still pass the legacy name indirectly (e.g. via cached YAML renders).
    if threshold_std_mult is not None:
        threshold_mult = threshold_std_mult

    if threshold_mult is None:
        raise ValueError("Latency threshold multiplier must be provided.")

    threshold_mult = float(threshold_mult)

    df, env_cols = _load_envelope_matrix(matrix_path, codes_json)
    df = df[df["trial_type"].str.lower() == "training"].copy()
    if df.empty:
        raise RuntimeError("No training trials present in matrix; cannot compute latency.")

    df["fps"] = df["fps"].replace([np.inf, -np.inf], np.nan).fillna(fps_default)
    df["dataset_canon"] = df["dataset"].map(_canon_dataset)

    records = []
    for _, row in df.iterrows():
        trial_num = _trial_num(row["trial_label"])
        if trial_num not in trials_of_interest:
            continue
        env = _extract_env(row, env_cols)
        fps = float(row.get("fps", fps_default))
        latency = _latency_to_cross(env, fps, before_sec, during_sec, threshold_mult)
        lat_for_mean = latency if (latency is not None and latency <= latency_ceiling) else math.nan
        records.append(
            {
                "dataset": row["dataset"],
                "dataset_canon": row["dataset_canon"],
                "fly": row["fly"],
                "trial_num": trial_num,
                "latency": latency,
                "lat_for_mean": lat_for_mean,
            }
        )

    if not records:
        raise RuntimeError("No training trials matched the requested trial numbers.")

    lat_df = pd.DataFrame(records)

    _plot_latency_per_fly(
        lat_df,
        out_dir,
        latency_ceiling=latency_ceiling,
        trials_of_interest=trials_of_interest,
        overwrite=overwrite,
    )
    _plot_latency_by_odor(
        lat_df,
        out_dir,
        latency_ceiling=latency_ceiling,
        trials_of_interest=trials_of_interest,
        overwrite=overwrite,
    )
    _plot_latency_grand_means(lat_df, out_dir, latency_ceiling, overwrite)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    env_parser = sub.add_parser("envelopes", help="Render per-fly training envelopes from the matrix outputs.")
    env_parser.add_argument("--matrix-npy", type=Path, required=True, help="Float16 matrix produced by the convert step.")
    env_parser.add_argument("--codes-json", type=Path, required=True, help="JSON metadata file from the convert step.")
    env_parser.add_argument("--out-dir", type=Path, required=True, help="Destination directory for envelope figures.")
    env_parser.add_argument("--latency-sec", type=float, default=0.0, help="Mean odor transit latency in seconds.")
    env_parser.add_argument("--fps-default", type=float, default=40.0, help="Fallback FPS when decoding rows.")
    env_parser.add_argument("--odor-on-s", type=float, default=30.0, help="Commanded odor ON timestamp (seconds).")
    env_parser.add_argument("--odor-off-s", type=float, default=60.0, help="Commanded odor OFF timestamp (seconds).")
    env_parser.add_argument(
        "--odor-latency-s",
        type=float,
        default=0.0,
        help="Transit delay between valve command and odor at the fly (seconds).",
    )
    env_parser.add_argument("--after-show-sec", type=float, default=30.0, help="Duration to display after odor OFF (seconds).")
    env_parser.add_argument("--threshold-std-mult", type=float, default=4.0, help="Threshold multiplier for baseline std dev.")
    env_parser.add_argument("--overwrite", action="store_true", help="Rebuild plots even if the target files exist.")

    lat_parser = sub.add_parser("latency", help="Compute latency-to-threshold metrics from the matrix outputs.")
    lat_parser.add_argument("--matrix-npy", type=Path, required=True, help="Float16 matrix produced by the convert step.")
    lat_parser.add_argument("--codes-json", type=Path, required=True, help="JSON metadata file from the convert step.")
    lat_parser.add_argument("--out-dir", type=Path, required=True, help="Directory for figures and CSV summaries.")
    lat_parser.add_argument("--before-sec", type=float, default=30.0, help="Baseline window length in seconds.")
    lat_parser.add_argument("--during-sec", type=float, default=35.0, help="During window length in seconds.")
    lat_parser.add_argument("--threshold-mult", type=float, default=4.0, help="Threshold multiplier (mu + k*std).")
    lat_parser.add_argument(
        "--threshold-std-mult",
        type=float,
        default=None,
        help="Deprecated alias for --threshold-mult maintained for backward compatibility.",
    )
    lat_parser.add_argument("--latency-ceiling", type=float, default=9.5, help="Cap for marking NR trials in seconds.")
    lat_parser.add_argument("--trials", nargs="+", type=int, default=[4, 5, 6], help="Training trial numbers to analyse.")
    lat_parser.add_argument("--fps-default", type=float, default=40.0, help="Fallback FPS when metadata missing.")
    lat_parser.add_argument("--overwrite", action="store_true", help="Rebuild plots even if the target files exist.")

    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "envelopes":
        cfg = EnvelopePlotConfig(
            matrix_npy=args.matrix_npy.expanduser().resolve(),
            codes_json=args.codes_json.expanduser().resolve(),
            out_dir=args.out_dir.expanduser().resolve(),
            latency_sec=args.latency_sec,
            fps_default=args.fps_default,
            odor_on_s=args.odor_on_s,
            odor_off_s=args.odor_off_s,
            odor_latency_s=args.odor_latency_s,
            after_show_sec=args.after_show_sec,
            threshold_std_mult=args.threshold_std_mult,
            trial_type="training",
            overwrite=args.overwrite,
        )
        generate_envelope_plots(cfg)
        return

    if args.command == "latency":
        latency_reports(
            args.matrix_npy.expanduser().resolve(),
            args.codes_json.expanduser().resolve(),
            args.out_dir.expanduser().resolve(),
            before_sec=args.before_sec,
            during_sec=args.during_sec,
            threshold_mult=args.threshold_mult,
            threshold_std_mult=args.threshold_std_mult,
            latency_ceiling=args.latency_ceiling,
            trials_of_interest=tuple(args.trials),
            fps_default=args.fps_default,
            overwrite=args.overwrite,
        )
        return

    parser.error(f"Unhandled command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    main()
