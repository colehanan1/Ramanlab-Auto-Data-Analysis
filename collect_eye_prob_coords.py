"""collect_eye_prob_coords.py

Discover and aggregate testing CSV files containing eye/proboscis coordinates
into a single wide CSV suitable for downstream analysis.

Usage examples:
    python collect_eye_prob_coords.py
    python collect_eye_prob_coords.py --verbose --out /home/ramanlab/Documents/cole/Data/Opto/all_eye_prob_coords_wide.csv
    python collect_eye_prob_coords.py --sources /path/a /path/b --dry-run
    python collect_eye_prob_coords.py --out results.csv --npy-out results.npy --json-out results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import re

import numpy as np
import pandas as pd


LOGGER_NAME = "collect_eye_prob_coords"
DEFAULT_SOURCES = [
    "/securedstorage/DATAsec/cole/Data-secured/opto_EB/",
    "/securedstorage/DATAsec/cole/Data-secured/opto_benz_1/",
    "/securedstorage/DATAsec/cole/Data-secured/opto_hex/",
    "/securedstorage/DATAsec/cole/Data-secured/hex_control/",
]
DEFAULT_OUTPUT = "/home/ramanlab/Documents/cole/Data/Opto/all_eye_prob_coords_wide.csv"

CSV_PATTERN = re.compile(
    r"^(?P<date>[a-z]+_\d{2})_fly_(?P<flynum>\d+)_testing_(?P<trial>\d+)_fly(?P<slot>[1-4])_distances\.csv$"
)

REQUIRED_COLUMNS = [
    "x_class2",
    "y_class2",
    "x_class8",
    "y_class8",
]

FPS_VALUE = 40


@dataclass(frozen=True, order=True)
class FlyKey:
    """Unique identifier for a fly across trials."""

    date_tag: str
    fly_num: int
    slot: int
    source_root: Path

    def dataset_name(self) -> str:
        return self.source_root.name

    def fly_label(self) -> str:
        return f"{self.date_tag}_fly_{self.fly_num}"


@dataclass
class TrialAggregation:
    """Aggregated data for a single trial."""

    key: FlyKey
    trial: int
    n_trials_for_fly: int
    total_frames: int
    frames: List[Tuple[float, float, float, float]]


@dataclass
class DiscoveryResult:
    """Container describing discovered trial files for each fly."""

    files_by_fly: Dict[FlyKey, Dict[int, List[Path]]]
    skipped_files: List[str]
    total_candidates: int


@dataclass
class ProcessingResult:
    """Processing outcome summary."""

    fly_results: List[TrialAggregation]
    total_trials: int
    skipped_files: List[str]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate testing CSV eye/proboscis coordinates into a wide format."
    )
    parser.add_argument(
        "--sources",
        nargs="*",
        default=DEFAULT_SOURCES,
        help="Source directories to scan recursively for testing CSV files.",
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_OUTPUT,
        help="Output CSV path (directories will be created if absent).",
    )
    parser.add_argument(
        "--npy-out",
        default=None,
        help="Optional NumPy .npy output path. Defaults to the CSV path with a .npy suffix.",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional JSON output path. Defaults to the CSV path with a .json suffix.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging (DEBUG level).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover and process files without writing the output CSV.",
    )
    return parser.parse_args(argv)


def configure_logging(verbose: bool) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False
    return logger


def discover_testing_files(sources: Sequence[str], logger: logging.Logger) -> DiscoveryResult:
    files_by_fly: Dict[FlyKey, Dict[int, List[Path]]] = {}
    skipped: List[str] = []
    total_candidates = 0

    for src in sources:
        root = Path(src).expanduser().resolve()
        if not root.exists():
            logger.warning("Source root does not exist: %s", root)
            continue
        if not root.is_dir():
            logger.warning("Source root is not a directory: %s", root)
            continue

        for dirpath, _, filenames in os.walk(root):
            for filename in filenames:
                if "testing" not in filename:
                    continue
                match = CSV_PATTERN.match(filename)
                if not match:
                    continue
                total_candidates += 1
                date_tag = match.group("date")
                fly_num = int(match.group("flynum"))
                trial = int(match.group("trial"))
                slot = int(match.group("slot"))
                file_path = Path(dirpath) / filename
                logger.debug(
                    "Matched file: %s (date=%s, fly=%d, trial=%d, slot=%d)",
                    file_path,
                    date_tag,
                    fly_num,
                    trial,
                    slot,
                )
                key = FlyKey(date_tag=date_tag, fly_num=fly_num, slot=slot, source_root=root)
                if key not in files_by_fly:
                    files_by_fly[key] = {}
                files_by_fly[key].setdefault(trial, []).append(file_path)

    return DiscoveryResult(files_by_fly=files_by_fly, skipped_files=skipped, total_candidates=total_candidates)


def _deduplicate_trial_paths(
    paths: List[Path],
    key: FlyKey,
    trial: int,
    logger: logging.Logger,
) -> Path:
    if len(paths) == 1:
        return paths[0]
    sorted_paths = sorted(paths, key=lambda p: p.as_posix())
    logger.warning(
        "Duplicate files for %s trial %d found. Using first path: %s",
        key,
        trial,
        sorted_paths[0],
    )
    return sorted_paths[0]


def process_files(
    discovery: DiscoveryResult,
    logger: logging.Logger,
) -> ProcessingResult:
    fly_results: List[TrialAggregation] = []
    skipped: List[str] = list(discovery.skipped_files)
    total_trials = 0

    for key in sorted(
        discovery.files_by_fly.keys(),
        key=lambda k: (k.date_tag, k.fly_num, k.slot, k.source_root.as_posix()),
    ):
        trial_map = discovery.files_by_fly[key]
        processed_trials: List[Tuple[int, List[Tuple[float, float, float, float]]]] = []

        for trial in sorted(trial_map.keys()):
            path = _deduplicate_trial_paths(trial_map[trial], key, trial, logger)
            try:
                df = pd.read_csv(path)
            except Exception as exc:  # pylint: disable=broad-except
                message = f"Failed to read {path}: {exc}"
                logger.error(message)
                skipped.append(message)
                continue

            missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
            if missing:
                message = f"Missing required columns {missing} in {path}; skipping"
                logger.warning(message)
                skipped.append(message)
                continue

            numeric_df = df[REQUIRED_COLUMNS].apply(pd.to_numeric, errors="coerce")
            frame_count = len(numeric_df)
            logger.debug(
                "Processed file %s with %d frames for %s trial %d",
                path,
                frame_count,
                key,
                trial,
            )

            trial_frames = list(numeric_df.itertuples(index=False, name=None))
            processed_trials.append((trial, trial_frames))
            total_trials += 1

        n_trials_for_fly = len(processed_trials)
        for trial, frames in processed_trials:
            fly_results.append(
                TrialAggregation(
                    key=key,
                    trial=trial,
                    n_trials_for_fly=n_trials_for_fly,
                    total_frames=len(frames),
                    frames=frames,
                )
            )
        logger.info(
            "Fly %s -> %d trial(s) processed",
            key,
            n_trials_for_fly,
        )

    return ProcessingResult(fly_results=fly_results, total_trials=total_trials, skipped_files=skipped)


def _build_frame_columns(max_frames: int) -> List[str]:
    columns: List[str] = []
    for frame in range(max_frames):
        columns.extend(
            [
                f"eye_x_f{frame}",
                f"eye_y_f{frame}",
                f"prob_x_f{frame}",
                f"prob_y_f{frame}",
            ]
        )
    return columns
def build_output_dataframe(results: List[TrialAggregation]) -> pd.DataFrame:
    metadata_columns = [
        "dataset",
        "fly",
        "fly_number",
        "trial_type",
        "trial_label",
        "fps",
    ]

    if not results:
        return pd.DataFrame(columns=metadata_columns)

    max_frames = max(result.total_frames for result in results)
    frame_columns = _build_frame_columns(max_frames)
    columns = metadata_columns + frame_columns

    rows: List[Dict[str, object]] = []
    for result in results:
        row: Dict[str, object] = {col: "" for col in columns}
        dataset_name = result.key.dataset_name()
        fly_label = result.key.fly_label()
        trial_label = f"testing_{int(result.trial)}"
        row.update(
            {
                "dataset": dataset_name,
                "fly": fly_label,
                "fly_number": int(result.key.fly_num),
                "trial_type": "testing",
                "trial_label": trial_label,
                "fps": FPS_VALUE,
            }
        )
        row["__trial_index__"] = int(result.trial)

        for frame_index, values in enumerate(result.frames):
            base = frame_index * 4
            try:
                eye_x, eye_y, prob_x, prob_y = values
            except ValueError:
                eye_x = eye_y = prob_x = prob_y = np.nan
            frame_cols = frame_columns[base : base + 4]
            row[frame_cols[0]] = eye_x
            row[frame_cols[1]] = eye_y
            row[frame_cols[2]] = prob_x
            row[frame_cols[3]] = prob_y

        rows.append(row)

    df = pd.DataFrame(rows)
    df.sort_values(
        by=["dataset", "fly", "__trial_index__"],
        inplace=True,
        kind="mergesort",
    )
    df.drop(columns=["__trial_index__"], inplace=True)
    df = df.reindex(columns=columns, fill_value="")
    df.reset_index(drop=True, inplace=True)
    return df


def _run_self_check(logger: logging.Logger) -> None:
    logger.info("Running self-check (synthetic data)")
    key1 = FlyKey(date_tag="october_01", fly_num=1, slot=1, source_root=Path("/tmp/src1"))
    key2 = FlyKey(date_tag="october_02", fly_num=2, slot=2, source_root=Path("/tmp/src2"))

    trial1 = TrialAggregation(
        key=key1,
        trial=1,
        n_trials_for_fly=2,
        total_frames=2,
        frames=[
            (0.1, 0.2, 0.3, 0.4),
            (1.1, 1.2, 1.3, 1.4),
        ],
    )
    trial2 = TrialAggregation(
        key=key1,
        trial=2,
        n_trials_for_fly=2,
        total_frames=1,
        frames=[(2.1, 2.2, 2.3, 2.4)],
    )
    trial3 = TrialAggregation(
        key=key2,
        trial=1,
        n_trials_for_fly=1,
        total_frames=1,
        frames=[(9.9, 9.8, 9.7, 9.6)],
    )

    df = build_output_dataframe([trial1, trial2, trial3])
    expected_columns = [
        "dataset",
        "fly",
        "fly_number",
        "trial_type",
        "trial_label",
        "fps",
        "eye_x_f0",
        "eye_y_f0",
        "prob_x_f0",
        "prob_y_f0",
        "eye_x_f1",
        "eye_y_f1",
        "prob_x_f1",
        "prob_y_f1",
    ]
    assert list(df.columns) == expected_columns, "Unexpected column layout in self-check"
    assert df.loc[0, "dataset"] == "src1"
    assert df.loc[0, "trial_label"] == "testing_1"
    assert df.loc[1, "eye_x_f1"] == "", "Padding failed for missing frames"
    logger.info("Self-check completed successfully")


def ensure_output_directory(out_path: Path, logger: logging.Logger) -> None:
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Failed to create directories for %s: %s", out_path, exc)
        raise


def write_output(df: pd.DataFrame, out_path: Path, logger: logging.Logger) -> None:
    ensure_output_directory(out_path, logger)
    df.to_csv(out_path, index=False, encoding="utf-8", lineterminator="\n")
    logger.info("Wrote output CSV: %s", out_path)


def write_numpy_output(df: pd.DataFrame, out_path: Path, logger: logging.Logger) -> None:
    ensure_output_directory(out_path, logger)
    array_data = df.to_numpy(dtype=object)
    np.save(out_path, array_data, allow_pickle=True)
    logger.info("Wrote output NPY: %s", out_path)


def write_json_output(df: pd.DataFrame, out_path: Path, logger: logging.Logger) -> None:
    ensure_output_directory(out_path, logger)
    payload = {
        "columns": list(df.columns),
        "data": df.fillna("").values.tolist(),
    }
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    logger.info("Wrote output JSON: %s", out_path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logger = configure_logging(args.verbose)
    logger.info("Starting collection from %d source(s)", len(args.sources))

    discovery = discover_testing_files(args.sources, logger)
    logger.info(
        "Discovered %d candidate file(s) across %d fly key(s)",
        discovery.total_candidates,
        len(discovery.files_by_fly),
    )

    processing = process_files(discovery, logger)
    if not processing.fly_results:
        logger.info("No fly data processed.")
        if args.dry_run:
            _run_self_check(logger)
        df = pd.DataFrame(
            columns=[
                "dataset",
                "fly",
                "fly_number",
                "trial_type",
                "trial_label",
                "fps",
            ]
        )
    else:
        df = build_output_dataframe(processing.fly_results)

    if not args.dry_run:
        out_path = Path(args.out).expanduser().resolve()
        write_output(df, out_path, logger)

        npy_path = (
            Path(args.npy_out).expanduser().resolve() if args.npy_out else out_path.with_suffix(".npy")
        )
        write_numpy_output(df, npy_path, logger)

        json_path = (
            Path(args.json_out).expanduser().resolve()
            if args.json_out
            else out_path.with_suffix(".json")
        )
        write_json_output(df, json_path, logger)

        output_location = str(out_path)
        npy_location = str(npy_path)
        json_location = str(json_path)
    else:
        output_location = "dry-run (no output)"
        npy_location = "dry-run (no output)"
        json_location = "dry-run (no output)"
        logger.info("Dry-run enabled; outputs not written")

    global_max_frames = max((result.total_frames for result in processing.fly_results), default=0)
    logger.info(
        "Summary: %d trial row(s), %d processed trial(s), max frames per trial %d | Outputs -> CSV: %s | NPY: %s | JSON: %s",
        len(processing.fly_results),
        processing.total_trials,
        global_max_frames,
        output_location,
        npy_location,
        json_location,
    )
    if processing.skipped_files:
        logger.info("Skipped %d file(s).", len(processing.skipped_files))
        for message in processing.skipped_files:
            logger.debug("Skipped: %s", message)

    return 0


if __name__ == "__main__":
    sys.exit(main())
