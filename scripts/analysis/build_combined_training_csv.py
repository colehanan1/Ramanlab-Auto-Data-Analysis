"""
Build combined training-only wide CSV from multiple dataset sources.

Groups:
  "New Flys Training" — EB-Training-24-0.1 + Hex-Training-24-0.005
                         (only batches AFTER April 11)
  "Old Flys" datasets  — Hex-Training, Benz-Training, 3OCT-Training,
                          ACV-Training, AIR-Training, EB-Training
                          (only batches BEFORE January, flagged flies excluded)

Reads angle_distance_rms_envelope CSVs (combined_pct column),
writes a wide CSV matching the existing format.

Output: /home/ramanlab/Documents/cole/Data/CSVs-New-Opto-Flys/
        all_envelope_rows_wide_combined_base.csv
"""

from __future__ import annotations

import glob
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────
OLD_DATA = Path("/home/ramanlab/Documents/cole/Data/flys")
NEW_DATA = Path("/home/ramanlab/Documents/cole/Data/flys_New")
FLAGGED_CSV = Path(
    "/home/ramanlab/Documents/cole/Data/CSVs-ALL-Opto-Flys/flagged-flys-truth.csv"
)
OUTPUT_CSV = Path(
    "/home/ramanlab/Documents/cole/Data/CSVs-New-Opto-Flys/"
    "all_envelope_rows_wide_combined_base.csv"
)

# ── Constants ─────────────────────────────────────────────────────────────
FPS = 40.0
MEASURE_COL = "combined_pct"
BEFORE_FRAMES = int(30.0 * FPS)   # 0-30 s
DURING_START = int(30.0 * FPS)    # 30 s
DURING_END = int(60.0 * FPS)      # 60 s
AFTER_START = int(60.0 * FPS)
AFTER_END = int(90.0 * FPS)

MONTH_NUM = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5,
    "june": 6, "july": 7, "august": 8, "september": 9, "october": 10,
    "november": 11, "december": 12,
}
BEFORE_JAN_MONTHS = {"september", "october", "november", "december"}


def _parse_month(folder: str) -> str | None:
    m = re.match(r"([a-z]+)_", folder)
    return m.group(1) if m else None


def _parse_day(folder: str) -> int | None:
    m = re.match(r"[a-z]+_(\d+)", folder)
    return int(m.group(1)) if m else None


def _extract_fly_number(fname: str) -> int | None:
    m = re.search(r"_fly(\d+)_", fname)
    return int(m.group(1)) if m else None


def _infer_trial_label(fname: str) -> str:
    """Extract trial label like 'training_3' from filename."""
    m = re.match(r"(training_\d+)_", fname)
    return m.group(1) if m else "unknown"


def _load_flagged() -> dict[str, set[tuple[str, int]]]:
    """Return {dataset: set of (fly_folder, fly_number)} for state < 1."""
    df = pd.read_csv(FLAGGED_CSV)
    df = df[df["FLY-State(1, 0, -1)"] < 1]
    result: dict[str, set[tuple[str, int]]] = {}
    for _, row in df.iterrows():
        ds = row["dataset"]
        result.setdefault(ds, set()).add((row["fly"], int(row["fly_number"])))
    return result


def _compute_auc(values: np.ndarray, start: int, end: int) -> float:
    seg = values[start:end]
    seg = seg[np.isfinite(seg)]
    if seg.size == 0:
        return 0.0
    return float(np.trapz(seg) / FPS)


def _compute_row_stats(values: np.ndarray) -> dict:
    """Compute per-trial statistics matching the wide CSV format."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {k: np.nan for k in [
            "global_min", "global_max", "trimmed_global_min", "trimmed_global_max",
            "local_min", "local_max", "local_min_before", "local_max_before",
            "before_std_from_median", "local_min_during", "local_max_during",
            "local_max_over_global_min", "local_max_during_over_global_min",
            "non_reactive_flag", "low_max_flag",
            "tracking_missing_frames", "tracking_pct_missing", "tracking_flagged",
        ]}

    before = values[:BEFORE_FRAMES]
    during = values[DURING_START:DURING_END]
    before_f = before[np.isfinite(before)] if before.size else np.array([])
    during_f = during[np.isfinite(during)] if during.size else np.array([])

    gmin, gmax = float(np.nanmin(finite)), float(np.nanmax(finite))
    p5, p95 = np.nanpercentile(finite, [5, 95]) if finite.size else (np.nan, np.nan)

    lmin_b = float(np.nanmin(before_f)) if before_f.size else np.nan
    lmax_b = float(np.nanmax(before_f)) if before_f.size else np.nan
    lmin_d = float(np.nanmin(during_f)) if during_f.size else np.nan
    lmax_d = float(np.nanmax(during_f)) if during_f.size else np.nan

    med_b = float(np.nanmedian(before_f)) if before_f.size else 0.0
    mad_b = float(np.nanmedian(np.abs(before_f - med_b))) if before_f.size else 0.0
    std_from_med = (lmax_b - med_b) / max(mad_b, 1e-9) if before_f.size else 0.0

    span = gmax - gmin
    non_reactive = 1.0 if span < 7.5 else 0.0

    return {
        "global_min": gmin, "global_max": gmax,
        "trimmed_global_min": float(p5), "trimmed_global_max": float(p95),
        "local_min": float(np.nanmin(finite)), "local_max": float(np.nanmax(finite)),
        "local_min_before": lmin_b, "local_max_before": lmax_b,
        "before_std_from_median": std_from_med,
        "local_min_during": lmin_d, "local_max_during": lmax_d,
        "local_max_over_global_min": lmax_b / max(gmin, 1e-9) if np.isfinite(lmax_b) else np.nan,
        "local_max_during_over_global_min": lmax_d / max(gmin, 1e-9) if np.isfinite(lmax_d) else np.nan,
        "non_reactive_flag": non_reactive,
        "low_max_flag": 0.0,
        "tracking_missing_frames": 0,
        "tracking_pct_missing": 0.0,
        "tracking_flagged": False,
    }


@dataclass
class Source:
    dataset_label: str        # what appears in the 'dataset' column
    dataset_dir_name: str     # actual folder name under data root
    data_root: Path
    is_training_only: bool = True
    # Filtering
    only_months: set[str] | None = None        # e.g. {"october", "november", ...}
    after_day: tuple[int, int] | None = None   # (month_num, day) exclusive lower bound
    exclude_flagged: bool = False
    flagged_dataset_key: str | None = None      # key in flagged CSV


SOURCES = [
    # ── New Flys Training (after April 11) ──
    Source(
        dataset_label="Hex-Training-24-0.005",
        dataset_dir_name="Hex-Training-24-0.005",
        data_root=NEW_DATA,
        after_day=(4, 11),  # after April 11
    ),
    Source(
        dataset_label="EB-Training-24-0.1",
        dataset_dir_name="EB-Training-24-0.1",
        data_root=NEW_DATA,
        after_day=(4, 11),
    ),
    # ── Old Flys (before January, non-flagged) ──
    Source(
        dataset_label="Hex-Training",
        dataset_dir_name="Hex-Training",
        data_root=OLD_DATA,
        only_months=BEFORE_JAN_MONTHS,
        exclude_flagged=True,
        flagged_dataset_key="Hex-Training",
    ),
    Source(
        dataset_label="Benz-Training",
        dataset_dir_name="Benz-Training",
        data_root=OLD_DATA,
        only_months=BEFORE_JAN_MONTHS,
        exclude_flagged=True,
        flagged_dataset_key="Benz-Training",
    ),
    Source(
        dataset_label="3OCT-Training",
        dataset_dir_name="3OCT-Training",
        data_root=OLD_DATA,
        only_months=BEFORE_JAN_MONTHS,
        exclude_flagged=True,
        flagged_dataset_key="3OCT-Training",
    ),
    Source(
        dataset_label="ACV-Training",
        dataset_dir_name="ACV-Training",
        data_root=OLD_DATA,
        only_months=BEFORE_JAN_MONTHS,
        exclude_flagged=True,
        flagged_dataset_key="ACV-Training",
    ),
    Source(
        dataset_label="AIR-Training",
        dataset_dir_name="AIR-Training",
        data_root=OLD_DATA,
        only_months=BEFORE_JAN_MONTHS,
        exclude_flagged=True,
        flagged_dataset_key="AIR-Training",
    ),
    Source(
        dataset_label="EB-Training",
        dataset_dir_name="EB-Training",
        data_root=OLD_DATA,
        only_months=BEFORE_JAN_MONTHS,
        exclude_flagged=True,
        flagged_dataset_key="EB-Training",
    ),
]


def _folder_passes_filter(folder: str, src: Source) -> bool:
    month = _parse_month(folder)
    if month is None:
        return False

    if src.only_months is not None and month not in src.only_months:
        return False

    if src.after_day is not None:
        month_num = MONTH_NUM.get(month)
        day = _parse_day(folder)
        if month_num is None or day is None:
            return False
        if (month_num, day) <= src.after_day:
            return False

    return True


def main() -> None:
    flagged_map = _load_flagged()
    all_rows: list[dict] = []
    max_trace_len = 0

    for src in SOURCES:
        ds_path = src.data_root / src.dataset_dir_name
        if not ds_path.is_dir():
            print(f"[WARN] Dataset not found: {ds_path}")
            continue

        flagged_set = set()
        if src.exclude_flagged and src.flagged_dataset_key:
            flagged_set = flagged_map.get(src.flagged_dataset_key, set())

        pattern = str(
            ds_path / "*" / "angle_distance_rms_envelope"
            / "training_*_fly*_angle_distance_rms_envelope.csv"
        )
        files = sorted(glob.glob(pattern))
        print(f"[{src.dataset_label}] Found {len(files)} training envelope files")

        accepted = 0
        skipped_date = 0
        skipped_flag = 0

        for fpath in files:
            parts = Path(fpath).parts
            batch_folder = parts[-3]
            fname = Path(fpath).name

            # Date filter
            if not _folder_passes_filter(batch_folder, src):
                skipped_date += 1
                continue

            # Fly number
            fly_num = _extract_fly_number(fname)
            if fly_num is None:
                continue

            # Flagged filter
            if flagged_set and (batch_folder, fly_num) in flagged_set:
                skipped_flag += 1
                continue

            # Read data
            try:
                df = pd.read_csv(fpath)
            except Exception as exc:
                print(f"  SKIP {fpath}: {exc}")
                continue

            if MEASURE_COL not in df.columns:
                print(f"  SKIP {fname}: no {MEASURE_COL} column")
                continue

            values = df[MEASURE_COL].to_numpy(dtype=np.float64)
            trace_len = len(values)
            max_trace_len = max(max_trace_len, trace_len)

            trial_label = _infer_trial_label(fname)
            stats = _compute_row_stats(values)

            auc_before = _compute_auc(values, 0, BEFORE_FRAMES)
            auc_during = _compute_auc(values, DURING_START, DURING_END)
            auc_after = _compute_auc(values, AFTER_START, AFTER_END)

            row = {
                "dataset": src.dataset_label,
                "fly": batch_folder,
                "fly_number": fly_num,
                **stats,
                "trace_len": trace_len,
                "trial_type": "training",
                "trial_label": trial_label,
                "fps": FPS,
                "AUC-Before": auc_before,
                "AUC-During": auc_during,
                "AUC-After": auc_after,
                "AUC-During-Before-Ratio": auc_during / max(auc_before, 1e-9),
                "AUC-After-Before-Ratio": auc_after / max(auc_before, 1e-9),
                "TimeToPeak-During": np.nan,
                "Peak-Value": float(np.nanmax(values[DURING_START:DURING_END]))
                    if trace_len > DURING_START else np.nan,
                "_values": values,
            }
            all_rows.append(row)
            accepted += 1

        print(f"  Accepted: {accepted}, skipped (date): {skipped_date}, "
              f"skipped (flagged): {skipped_flag}")

    if not all_rows:
        print("ERROR: No rows collected!")
        return

    print(f"\nTotal rows: {len(all_rows)}, max trace length: {max_trace_len}")

    # Build wide DataFrame
    meta_prefix = ["dataset", "fly", "fly_number"]
    stat_cols = [
        "global_min", "global_max", "trimmed_global_min", "trimmed_global_max",
        "local_min", "local_max", "local_min_before", "local_max_before",
        "before_std_from_median", "local_min_during", "local_max_during",
        "local_max_over_global_min", "local_max_during_over_global_min",
        "non_reactive_flag", "low_max_flag",
        "tracking_missing_frames", "tracking_pct_missing", "tracking_flagged",
        "trace_len",
    ]
    meta_suffix = ["trial_type", "trial_label", "fps"]
    auc_cols = [
        "AUC-Before", "AUC-During", "AUC-After",
        "AUC-During-Before-Ratio", "AUC-After-Before-Ratio",
        "TimeToPeak-During", "Peak-Value",
    ]
    dir_val_cols = [f"dir_val_{i}" for i in range(max_trace_len)]

    output_rows = []
    for row in all_rows:
        vals = row.pop("_values")
        # Pad to max_trace_len
        padded = np.full(max_trace_len, np.nan, dtype=np.float64)
        padded[:len(vals)] = vals
        for i, v in enumerate(padded):
            row[f"dir_val_{i}"] = v
        output_rows.append(row)

    all_columns = meta_prefix + stat_cols + meta_suffix + auc_cols + dir_val_cols
    out_df = pd.DataFrame(output_rows, columns=all_columns)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nWrote {len(out_df)} rows to {OUTPUT_CSV}")

    # Summary
    print("\n=== Dataset Summary ===")
    for ds, grp in out_df.groupby("dataset"):
        flies = grp.groupby(["fly", "fly_number"]).ngroups
        print(f"  {ds}: {len(grp)} rows, {flies} unique flies")


if __name__ == "__main__":
    main()
