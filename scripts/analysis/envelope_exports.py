"""Utilities for exporting per-frame envelopes and matrices.

This module packages the two notebook-style utilities requested by the
lab into a reusable command-line script with explicit configuration
options.  It can (1) scan multiple experiment roots to build a wide CSV
containing the Hilbert envelope of distance metrics and (2) convert that
CSV into a compact float16 matrix accompanied by code maps for metadata.

Example usage:

    # Build the combined wide CSV (defaults match the notebook snippet)
    python scripts/analysis/envelope_exports.py collect

    # Convert the wide CSV to matrix + key artefacts
    python scripts/analysis/envelope_exports.py convert

"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.signal import hilbert

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for path in (str(REPO_ROOT), str(SRC_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from fbpipe.config import load_raw_config
from fbpipe.utils.columns import PROBOSCIS_DISTANCE_PCT_COL


# ---------------------------------------------------------------------------
# Constants for per-trial summaries


AUC_COLUMNS = (
    "AUC-Before",
    "AUC-During",
    "AUC-After",
    "AUC-During-Before-Ratio",
    "AUC-After-Before-Ratio",
    "TimeToPeak-During",
    "Peak-Value",
)

# Frame windows follow existing reaction-matrix conventions (≈30 s at 40 FPS)
BEFORE_FRAMES = 1260
DURING_FRAMES = 1200
AFTER_FRAMES = 1200


# ---------------------------------------------------------------------------
# Shared helpers


TIMESTAMP_CANDIDATES = ("UTC_ISO", "Timestamp", "Number", "MonoNs")
FRAME_CANDIDATES = ("Frame", "FrameNumber", "Frame Number")
TRIAL_REGEX = re.compile(r"(testing|training)_(\d+)", re.IGNORECASE)
FLY_SLOT_REGEX = re.compile(r"(fly\d+)_distances", re.IGNORECASE)
FLY_NUMBER_REGEX = re.compile(r"fly\s*[_-]?\s*(\d+)", re.IGNORECASE)


def _nanmin(values: np.ndarray) -> float:
    mask = np.isfinite(values)
    if not np.any(mask):
        return 0.0
    return float(np.min(values[mask]))


def _effective_fps(value: float, *, fallback: float, default: float) -> float:
    for candidate in (value, fallback, default):
        if np.isfinite(candidate) and candidate > 0:
            return float(candidate)
    return 0.0


def _segment_bounds(total_len: int) -> tuple[int, int, int, int]:
    before_end = max(0, min(BEFORE_FRAMES, total_len))
    during_start = before_end
    during_end = max(during_start, min(during_start + DURING_FRAMES, total_len))
    after_end = max(during_end, min(during_end + AFTER_FRAMES, total_len))
    return before_end, during_start, during_end, after_end


def _segment_auc(values: np.ndarray, threshold: float, dt: float) -> float:
    if values.size == 0 or dt <= 0:
        return 0.0
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    diff = finite - threshold
    positive = diff[diff > 0]
    if positive.size == 0:
        return 0.0
    return float(np.sum(positive) * dt)


def _compute_trial_metrics(
    env: np.ndarray,
    fps: float,
    *,
    fallback_fps: float,
    default_fps: float,
    fly_before_mean: float,
    use_per_trial_baseline: bool = False,
) -> dict[str, float]:
    total_len = env.size
    before_end, during_start, during_end, after_end = _segment_bounds(total_len)

    before = env[:before_end]
    during = env[during_start:during_end]
    after = env[during_end:after_end]

    before_mean = float(np.nanmean(before)) if before.size else np.nan
    before_std = float(np.nanstd(before)) if before.size else 0.0

    # Select baseline mode
    if use_per_trial_baseline:
        # Use only this trial's before period
        baseline = before_mean
    else:
        # Use the fly-level mean (current behavior)
        baseline = fly_before_mean
        if not np.isfinite(baseline):
            baseline = before_mean

    if not np.isfinite(baseline):
        baseline = 0.0
    if not np.isfinite(before_std):
        before_std = 0.0

    threshold = baseline + 3.0 * before_std

    fps_eff = _effective_fps(fps, fallback=fallback_fps, default=default_fps)
    dt = 1.0 / fps_eff if fps_eff > 0 else 0.0

    auc_before = _segment_auc(before, threshold, dt)
    auc_during = _segment_auc(during, threshold, dt)
    auc_after = _segment_auc(after, threshold, dt)

    during_ratio = float(auc_during / auc_before) if auc_before > 0 else float("nan")
    after_ratio = float(auc_after / auc_before) if auc_before > 0 else float("nan")

    peak_value = float("nan")
    time_to_peak = float("nan")
    if during.size:
        finite_during = during[np.isfinite(during)]
        if finite_during.size:
            peak_value = float(np.max(finite_during))

    if total_len and fps_eff > 0:
        finite_env = env[np.isfinite(env)]
        if finite_env.size:
            try:
                global_peak_idx = int(np.nanargmax(env))
            except ValueError:
                global_peak_idx = -1
            if during_start <= global_peak_idx < during_end:
                time_to_peak = (global_peak_idx - during_start) / fps_eff

    return {
        "AUC-Before": float(auc_before),
        "AUC-During": float(auc_during),
        "AUC-After": float(auc_after),
        "AUC-During-Before-Ratio": during_ratio,
        "AUC-After-Before-Ratio": after_ratio,
        "TimeToPeak-During": time_to_peak,
        "Peak-Value": peak_value,
    }


def _pick_column(candidates: Sequence[str], df: pd.DataFrame) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _to_seconds_series(df: pd.DataFrame, ts_col: str) -> pd.Series:
    series = df[ts_col]
    if ts_col in ("UTC_ISO", "Timestamp"):
        dt = pd.to_datetime(series, errors="coerce", utc=(ts_col == "UTC_ISO"))
        secs = dt.astype("int64", copy=False) / 1e9
        t0 = _nanmin(secs.to_numpy(np.float64, copy=False))
        return (secs - t0).astype(float)

    if ts_col == "Number":
        vals = pd.to_numeric(series, errors="coerce").astype(float)
        t0 = _nanmin(vals.to_numpy(np.float64, copy=False))
        return vals - t0

    if ts_col == "MonoNs":
        vals = pd.to_numeric(series, errors="coerce").astype(float)
        secs = vals / 1e9
        t0 = _nanmin(secs.to_numpy(np.float64, copy=False))
        return secs - t0

    raise ValueError(f"Unsupported timestamp column: {ts_col}")


def _estimate_fps_from_seconds(seconds_series: pd.Series) -> Optional[float]:
    mask = seconds_series.notna()
    if mask.sum() < 2:
        return None
    duration = seconds_series[mask].iloc[-1] - seconds_series[mask].iloc[0]
    if duration <= 0:
        return None
    return mask.sum() / duration


def _compute_envelope(series: pd.Series, win_frames: int) -> np.ndarray:
    series = pd.to_numeric(series, errors="coerce").fillna(0.0).clip(lower=0, upper=100)
    analytic = hilbert(series.to_numpy())
    env = np.abs(analytic)
    return (
        pd.Series(env, index=series.index)
        .rolling(window=win_frames, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )


def _infer_trial_type(path: Path) -> str:
    composite = (path.stem + "/" + "/".join(parent.name for parent in path.parents)).lower()
    if "testing" in composite:
        return "testing"
    if "training" in composite:
        return "training"
    return "unknown"


def _trial_label(path: Path) -> str:
    match = TRIAL_REGEX.search(path.stem)
    if not match:
        chain = (path.stem + "/" + "/".join(parent.name for parent in path.parents)).lower()
        match = TRIAL_REGEX.search(chain)
    if match:
        kind, num = match.group(1).lower(), match.group(2)
        return f"{kind}_{num}"

    trailing = re.search(r"(\d+)$", path.stem)
    if trailing:
        return f"{_infer_trial_type(path)}_{trailing.group(1)}"
    return path.stem


def _find_trial_csvs(fly_dir: Path) -> Iterator[Path]:
    search_root = fly_dir / "RMS_calculations"
    if not search_root.is_dir():
        search_root = fly_dir
    print(
        f"[DEBUG]    Searching for trial CSVs under {search_root}"
    )
    patterns = ("**/*testing*.csv", "**/*training*.csv")
    seen: set[Path] = set()
    yielded = False
    for pattern in patterns:
        for csv_path in search_root.glob(pattern):
            if csv_path.is_file():
                resolved = csv_path.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    yielded = True
                    yield resolved
    if not yielded:
        print(f"[DEBUG]    No CSVs matched testing/training glob in {search_root}")


# ---------------------------------------------------------------------------
# Wide CSV builder


@dataclass
class CollectConfig:
    roots: List[Path]
    measure_cols: Sequence[str]
    fps_default: float
    window_sec: float
    fallback_fps: float
    out_csv: Path
    use_per_trial_baseline: bool = False

    @property
    def window_frames(self) -> int:
        return max(int(self.window_sec * self.fps_default), 1)


def _resolve_measure_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def collect_envelopes(cfg: CollectConfig) -> None:
    print(
        "[DEBUG] collect_envelopes → roots=%s, measure_cols=%s, window_frames=%s"
        % (
            [str(r) for r in cfg.roots],
            list(cfg.measure_cols),
            cfg.window_frames,
        )
    )

    items: List[dict] = []
    max_len = 0

    for root in cfg.roots:
        root = root.expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Not a directory: {root}")
        print(f"[DEBUG] Scanning dataset root: {root}")
        dataset = root.name

        for fly_dir in sorted((p for p in root.iterdir() if p.is_dir())):
            print(f"[DEBUG]  ↳ Fly directory: {fly_dir}")
            fly = fly_dir.name
            found_any = False
            for csv_path in _find_trial_csvs(fly_dir):
                found_any = True
                print(
                    f"[DEBUG]   ↳ Candidate CSV: {csv_path}"
                )
                try:
                    header_df = pd.read_csv(csv_path, nrows=0)
                except Exception as exc:  # pragma: no cover - purely defensive
                    print(f"[WARN] Skip {csv_path.name}: header read error: {exc}")
                    continue

                print(
                    "[DEBUG]     Available columns=%s"
                    % list(header_df.columns)
                )
                measure_col = _resolve_measure_column(header_df, cfg.measure_cols)
                if measure_col is None:
                    print(f"[SKIP] {csv_path.name}: none of {cfg.measure_cols} present.")
                    continue
                print(f"[DEBUG]     Selected measure column: {measure_col}")

                slot_token = _fly_slot_from_name(csv_path.name)
                fly_number = _fly_number_from_name(csv_path.name)
                if slot_token:
                    slot_label = slot_token.replace("_distances", "")
                    fly_id = f"{fly}_{slot_label}"
                    if fly_number is None:
                        fly_number = _fly_number_from_name(slot_label)
                else:
                    fly_id = fly
                    if fly_number is None:
                        fly_number = _fly_number_from_name(fly)

                fly_number_label = str(fly_number) if fly_number is not None else "UNKNOWN"
                if fly_number is None:
                    print(
                        f"[WARN] {csv_path.name}: fly number not detected from filename; using 'UNKNOWN'."
                    )
                else:
                    print(
                        f"[DEBUG]     Parsed fly_number={fly_number_label} from name tokens"
                    )

                slot_token = _fly_slot_from_name(csv_path.name)
                if slot_token:
                    slot_label = slot_token.replace("_distances", "")
                    fly_id = f"{fly}_{slot_label}"
                else:
                    fly_id = fly

                try:
                    n_frames = pd.read_csv(csv_path, usecols=[measure_col]).shape[0]
                except Exception as exc:  # pragma: no cover - purely defensive
                    print(f"[WARN] Skip {csv_path.name}: count error: {exc}")
                    continue

                print(
                    f"[DEBUG]     Frame count={n_frames} for column={measure_col}"
                )

                items.append(
                    {
                        "dataset": dataset,
                        "fly": fly_id,
                        "csv_path": csv_path,
                        "fly_number": fly_number_label,
                        "trial_type": _infer_trial_type(csv_path),
                        "trial_label": _trial_label(csv_path),
                        "measure_col": measure_col,
                        "n_frames": n_frames,
                    }
                )
                max_len = max(max_len, n_frames)

            if not found_any:
                print(
                    f"[DEBUG]   ↳ No testing/training CSVs detected under {fly_dir}"
                )

    if not items:
        raise RuntimeError("No eligible testing/training CSVs found in provided roots.")

    print(f"[INFO] Datasets: {[root.name for root in cfg.roots]}")
    print(f"[INFO] Discovered {len(items)} videos. Max frames = {max_len}")

    env_cols = [f"env_{i}" for i in range(max_len)]
    cols = [
        "dataset",
        "fly",
        "fly_number",
        "trial_type",
        "trial_label",
        "fps",
        *AUC_COLUMNS,
        *env_cols,
    ]

    trial_rows: List[dict] = []
    fly_before_totals: dict[str, tuple[float, int]] = {}
    for item in items:
        csv_path = item["csv_path"]
        measure_col = item["measure_col"]

        try:
            hdr2 = pd.read_csv(csv_path, nrows=0)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[WARN] {csv_path.name}: failed to read header for FPS columns: {exc}")
            hdr2 = pd.DataFrame()

        frame_col = _pick_column(FRAME_CANDIDATES, hdr2)
        ts_col = _pick_column(TIMESTAMP_CANDIDATES, hdr2)

        print(
            f"[DEBUG] {csv_path.name}: frame_col={frame_col}, timestamp_col={ts_col}"
        )

        fps = float("nan")
        if frame_col is not None and ts_col is not None:
            try:
                df_ts = pd.read_csv(csv_path, usecols=[frame_col, ts_col])
                secs = _to_seconds_series(df_ts, ts_col)
                fps_from_csv = _estimate_fps_from_seconds(secs)
                if fps_from_csv and np.isfinite(fps_from_csv) and fps_from_csv > 0:
                    fps = float(fps_from_csv)
                else:
                    fps = float(cfg.fallback_fps)
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[WARN] FPS inference failed for {csv_path.name}: {exc}")
                fps = float(cfg.fallback_fps)
        else:
            fps = float(cfg.fallback_fps)

        try:
            df = pd.read_csv(csv_path, usecols=[measure_col])
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[WARN] Read failed {csv_path}: {exc}")
            continue

        env = _compute_envelope(df[measure_col], cfg.window_frames).astype(float)

        print(
            f"[DEBUG] {csv_path.name}: envelope_length={len(env)}, fly_number={item['fly_number']}"
        )

        trial_rows.append(
            {
                "dataset": item["dataset"],
                "fly": item["fly"],
                "fly_number": item["fly_number"],
                "trial_type": item["trial_type"],
                "trial_label": item["trial_label"],
                "fps": fps,
                "env": env,
            }
        )

        if item["trial_type"].lower() == "testing":
            before_len = min(BEFORE_FRAMES, env.size)
            if before_len > 0:
                before_segment = env[:before_len]
                finite_mask = np.isfinite(before_segment)
                if np.any(finite_mask):
                    sum_before = float(np.nansum(before_segment[finite_mask]))
                    count_before = int(np.sum(finite_mask))
                    total, count = fly_before_totals.get(item["fly"], (0.0, 0))
                    fly_before_totals[item["fly"]] = (
                        total + sum_before,
                        count + count_before,
                    )

        print(
            f"[DEBUG] Buffered row for fly={item['fly']} fly_number={item['fly_number']} frames={len(env)}"
        )

    fly_before_means = {
        fly: (total / count if count > 0 else float("nan"))
        for fly, (total, count) in fly_before_totals.items()
    }

    records: List[dict] = []
    for row in trial_rows:
        env = row["env"]
        metrics = _compute_trial_metrics(
            env,
            row["fps"],
            fallback_fps=cfg.fallback_fps,
            default_fps=cfg.fps_default,
            fly_before_mean=fly_before_means.get(row["fly"], float("nan")),
            use_per_trial_baseline=cfg.use_per_trial_baseline,
        )

        padded = np.full(max_len, np.nan, dtype=float)
        valid = min(env.size, max_len)
        if valid:
            padded[:valid] = env[:valid]

        record = {key: row[key] for key in ("dataset", "fly", "fly_number", "trial_type", "trial_label", "fps")}
        record.update(metrics)
        record.update({col: padded[idx] for idx, col in enumerate(env_cols)})
        records.append(record)

    df_out = pd.DataFrame.from_records(records, columns=cols)
    cfg.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(cfg.out_csv, index=False)

    print(f"[OK] Wrote combined envelope table: {cfg.out_csv}")


# ---------------------------------------------------------------------------
# Matrix converter


@dataclass
class ConvertConfig:
    input_csv: Path
    out_dir: Path
    matrix_npy: Optional[Path]
    code_key: Optional[Path]
    codes_json: Optional[Path]


def convert_wide_csv(cfg: ConvertConfig) -> None:
    print(
        f"[DEBUG] convert_wide_csv → input={cfg.input_csv}, output_dir={cfg.out_dir}"
    )
    df = pd.read_csv(cfg.input_csv)
    print(
        f"[DEBUG] Loaded wide CSV with shape={df.shape} columns={list(df.columns)}"
    )

    meta_candidates = ["dataset", "fly", "fly_number", "trial_type", "trial_label", "fps"]
    meta_cols = [col for col in meta_candidates if col in df.columns]
    if not meta_cols:
        raise RuntimeError("No metadata columns found. Expected at least one of: dataset, fly, trial_type, trial_label, fps.")
    print(f"[DEBUG] Metadata columns detected: {meta_cols}")

    metric_cols = [col for col in AUC_COLUMNS if col in df.columns]
    env_cols = [
        col
        for col in df.columns
        if col not in meta_cols and col not in metric_cols
    ]
    if not env_cols:
        raise RuntimeError("No envelope columns found.")
    print(
        f"[DEBUG] Envelope columns detected: {env_cols[:5]}{'...' if len(env_cols) > 5 else ''}"
    )

    code_maps: dict[str, dict[str, int]] = {}
    for col in meta_cols:
        uniques = pd.Series(df[col].astype(str).fillna("UNKNOWN")).unique().tolist()
        mapping: dict[str, int] = {"UNKNOWN": 0}
        next_code = 1
        for value in uniques:
            if value not in mapping:
                mapping[value] = next_code
                next_code += 1
        code_maps[col] = mapping
        print(f"[DEBUG] Encoded {col}: {mapping}")

    df_num = df.copy()
    for col, mapping in code_maps.items():
        df_num[col] = df_num[col].astype(str).map(mapping).fillna(0).astype(np.int32)
        print(f"[DEBUG] Column {col} mapped to numeric codes")

    numeric_cols = metric_cols + env_cols
    df_num[numeric_cols] = (
        df_num[numeric_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .astype(np.float32, copy=False)
    )
    print("[DEBUG] Envelope + metric values coerced to numeric with NaNs filled as 0.0")

    ordered_cols = meta_cols + metric_cols + env_cols
    matrix_f32 = df_num[ordered_cols].to_numpy(dtype=np.float32, copy=False)
    finite_info = np.finfo(np.float16)
    np.clip(matrix_f32, finite_info.min, finite_info.max, out=matrix_f32)
    matrix_f16 = matrix_f32.astype(np.float16)
    print(f"[DEBUG] Float16 matrix shape={matrix_f16.shape}")

    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    matrix_path = cfg.matrix_npy or (cfg.out_dir / "envelope_matrix_float16.npy")
    np.save(matrix_path, matrix_f16)

    key_path = cfg.code_key or (cfg.out_dir / "code_key.txt")
    with key_path.open("w", encoding="utf-8") as fh:
        fh.write("# Envelope matrix schema (float16), row-wise\n")
        fh.write("# Columns (in order):\n")
        for idx, col in enumerate(ordered_cols):
            fh.write(f"{idx:>5}: {col}\n")
        fh.write("\n# Metadata code maps (string → integer code)\n")
        for col in meta_cols:
            fh.write(f"\n[{col}]\n")
            for code, name in sorted(((code, name) for name, code in code_maps[col].items()), key=lambda x: x[0]):
                fh.write(f"{code:>5} : {name}\n")
        fh.write("\nNotes:\n")
        fh.write("- Matrix dtype is float16 (16-bit). Metadata codes are stored as float16 numbers in the matrix.\n")
        fh.write("- Envelope NaNs (shorter videos) were replaced with 0.0.\n")
        fh.write("- Code '0' means UNKNOWN for the metadata fields.\n")

    json_path = cfg.codes_json or (cfg.out_dir / "code_maps.json")
    with json_path.open("w", encoding="utf-8") as jf:
        json.dump(
            {
                "column_order": ordered_cols,
                "code_maps": code_maps,
                "metric_columns": metric_cols,
                "env_columns": env_cols,
            },
            jf,
            indent=2,
        )

    print(f"[OK] Saved 16-bit matrix: {matrix_path}  (shape={matrix_f16.shape}, dtype={matrix_f16.dtype})")
    print(f"[OK] Saved key:           {key_path}")
    print(f"[OK] Saved JSON maps:     {json_path}")


# ---------------------------------------------------------------------------
# CLI entry point


def _parse_collect_args(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument(
        "--roots",
        nargs="*",
        type=Path,
        default=None,
        help="Experiment roots to scan for testing/training CSV files.",
    )
    subparser.add_argument(
        "--measure-cols",
        nargs="+",
        default=[PROBOSCIS_DISTANCE_PCT_COL, "distance_percentage_2_8", "distance_percentage", "distance_percentage_2_6"],
        help="Ordered list of candidate measurement columns.",
    )
    subparser.add_argument(
        "--fps-default",
        type=float,
        default=40.0,
        help="Default FPS used for rolling window sizing.",
    )
    subparser.add_argument(
        "--window-sec",
        type=float,
        default=0.25,
        help="Window size in seconds for the centred rolling mean applied to the envelope.",
    )
    subparser.add_argument(
        "--fallback-fps",
        type=float,
        default=40.0,
        help="Fallback FPS when timestamps are unavailable.",
    )
    subparser.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Output CSV path for the combined envelope table.",
    )
    subparser.add_argument(
        "--use-per-trial-baseline",
        action="store_true",
        help="Use each trial's own before-period for baseline instead of fly-level mean.",
    )


def _parse_convert_args(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help="Wide CSV file produced by the collect step.",
    )
    subparser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for generated matrix and key artefacts.",
    )
    subparser.add_argument("--matrix-npy", type=Path, default=None, help="Explicit output path for the float16 matrix.")
    subparser.add_argument("--code-key", type=Path, default=None, help="Explicit output path for the human-readable key.")
    subparser.add_argument("--codes-json", type=Path, default=None, help="Explicit output path for the JSON code maps.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(Path("config") / "config.yaml"),
        help="Path to pipeline configuration YAML.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect_parser = subparsers.add_parser("collect", help="Build the combined envelope wide CSV.")
    _parse_collect_args(collect_parser)

    convert_parser = subparsers.add_parser("convert", help="Convert the wide CSV into float16 artefacts.")
    _parse_convert_args(convert_parser)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config_data = load_raw_config(args.config)
    analysis_cfg = config_data.get("analysis", {})
    combined_cfg = analysis_cfg.get("combined", {}) if isinstance(analysis_cfg, dict) else {}
    combine_cfg = combined_cfg.get("combine", {}) if isinstance(combined_cfg, dict) else {}
    wide_cfg = combined_cfg.get("wide", {}) if isinstance(combined_cfg, dict) else {}
    matrix_cfg = combined_cfg.get("matrix", {}) if isinstance(combined_cfg, dict) else {}

    if args.command == "collect":
        roots = args.roots
        if not roots:
            roots = [Path(root) for root in combine_cfg.get("roots", [])]
        if not roots:
            raise SystemExit("No roots configured. Provide --roots or set analysis.combined.combine.roots.")

        out_csv = args.out_csv or wide_cfg.get("output_csv")
        if not out_csv:
            raise SystemExit("No output CSV configured. Provide --out-csv or set analysis.combined.wide.output_csv.")

        use_per_trial = args.use_per_trial_baseline or wide_cfg.get("use_per_trial_baseline", False)
        cfg = CollectConfig(
            roots=list(roots),
            measure_cols=args.measure_cols,
            fps_default=args.fps_default,
            window_sec=args.window_sec,
            fallback_fps=args.fallback_fps,
            out_csv=Path(out_csv),
            use_per_trial_baseline=use_per_trial,
        )
        collect_envelopes(cfg)
        return

    if args.command == "convert":
        input_csv = args.input_csv or matrix_cfg.get("input_csv") or wide_cfg.get("output_csv")
        out_dir = args.out_dir or matrix_cfg.get("out_dir")
        if not input_csv:
            raise SystemExit("No input CSV configured. Provide --input-csv or set analysis.combined.matrix.input_csv.")
        if not out_dir:
            raise SystemExit("No output directory configured. Provide --out-dir or set analysis.combined.matrix.out_dir.")

        cfg = ConvertConfig(
            input_csv=Path(input_csv),
            out_dir=Path(out_dir),
            matrix_npy=args.matrix_npy,
            code_key=args.code_key,
            codes_json=args.codes_json,
        )
        convert_wide_csv(cfg)
        return

    parser.error(f"Unknown command {args.command!r}")


if __name__ == "__main__":  # pragma: no cover
    main()

def _fly_slot_from_name(name: str) -> Optional[str]:
    match = FLY_SLOT_REGEX.search(name.lower())
    if match:
        return match.group(1)
    return None


def _fly_number_from_name(name: str) -> Optional[int]:
    match = FLY_NUMBER_REGEX.search(name)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None
