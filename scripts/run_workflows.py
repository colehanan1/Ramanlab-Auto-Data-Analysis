#!/usr/bin/env python3
"""Run the full pipeline plus optional analysis workflows defined in config.yaml."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
LOGGER = logging.getLogger(__name__)

# Ensure local source tree takes precedence over any installed package.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for path in (str(SRC_ROOT), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

STATE_VERSION = 1

from fbpipe.config import Settings, load_settings
from fbpipe.pipeline import ORDERED_STEPS
from fbpipe.steps import predict_reactions

from scripts.envelope_visuals import (
    EnvelopePlotConfig,
    MatrixPlotConfig,
    generate_envelope_plots,
    generate_reaction_matrices,
)
from scripts.envelope_training import latency_reports
from scripts.envelope_combined import (
    CombineConfig,
    build_wide_csv,
    combine_distance_angle,
    mirror_directory,
    overlay_sources,
    secure_copy_and_cleanup,
    wide_to_matrix,
)


def _resolve_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    return Path(value).expanduser().resolve()


def _ensure_path(value: str | Path, field: str) -> Path:
    path = _resolve_path(value)
    if path is None:
        raise ValueError(f"Missing required path for '{field}'.")
    return path


def _cache_root(settings: Settings) -> Path:
    path = Path(settings.cache_dir or (REPO_ROOT / "cache"))
    path = path.expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_key(value: str) -> str:
    text = str(Path(value).expanduser()) if value else ""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return cleaned.strip("_") or "default"


def _state_path(settings: Settings, category: str, key: str) -> Path:
    return _cache_root(settings) / category / _safe_key(key) / "state.json"


def _load_state(settings: Settings, category: str, key: str) -> dict[str, Any] | None:
    state_path = _state_path(settings, category, key)
    if not state_path.exists():
        return None
    try:
        with state_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None


def _write_state(settings: Settings, category: str, key: str, payload: dict[str, Any]) -> None:
    """Write state with file manifest for relevant categories."""
    state_path = _state_path(settings, category, key)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    data = dict(payload)
    data.setdefault("version", STATE_VERSION)

    # Add file manifest for pipeline category
    if category == "pipeline":
        # Key is the dataset root path
        dataset_root = Path(key)
        if dataset_root.exists():
            LOGGER.debug(f"Building file manifest for state persistence: {dataset_root.name}")
            data["file_manifest"] = _build_file_manifest(dataset_root)

    # Add file manifest for combined category
    if category == "combined":
        dataset_roots = payload.get("dataset_roots", [])
        combined_manifest = {}
        for root_str in dataset_roots:
            root_path = Path(root_str)
            if root_path.exists():
                combined_manifest.update(_build_file_manifest(root_path))
        data["file_manifest"] = combined_manifest

    # Write to disk
    with state_path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)

    LOGGER.debug(f"Wrote state: {state_path}")


# ============================================================================
# File Manifest Tracking Functions
# ============================================================================


def _should_track_file(file_path: Path, dataset_root: Path) -> bool:
    """
    Determine if a CSV file should be tracked in the manifest.

    Tracks all CSV files in dataset subdirectories, excluding metadata files.

    Args:
        file_path: Path to the file
        dataset_root: Root of the dataset

    Returns:
        True if file should be tracked
    """
    # Must be CSV
    if file_path.suffix.lower() != '.csv':
        return False

    # Exclude metadata and temp files
    if file_path.name.startswith(('sensors_', '.', '~')):
        return False

    # Must be in nested structure (not directly in dataset root)
    try:
        relative = file_path.relative_to(dataset_root)
        # Need at least fly_dir/trial_dir/file.csv (2+ parts)
        if len(relative.parts) < 2:
            return False
        return True
    except ValueError:
        # File not in dataset_root
        return False


def _build_file_manifest(dataset_root: Path) -> Dict[str, Dict[str, Any]]:
    """
    Build manifest of all trackable CSV files in dataset.

    Recursively scans dataset directory for CSV files and captures their
    modification time and size for change detection.

    Args:
        dataset_root: Root directory of dataset (e.g., /Data/flys/opto_EB/)

    Returns:
        Dict mapping absolute file path to {mtime: float, size: int}

    Performance: ~0.2ms per file (180ms for 900 files)
    """
    manifest = {}
    file_count = 0

    start_time = time.time()

    # Recursive walk through dataset
    for csv_file in dataset_root.rglob("*.csv"):
        if not _should_track_file(csv_file, dataset_root):
            continue

        try:
            stat = csv_file.stat()
            manifest[str(csv_file.absolute())] = {
                "mtime": stat.st_mtime,
                "size": stat.st_size
            }
            file_count += 1
        except (OSError, PermissionError) as e:
            LOGGER.warning(f"Failed to stat {csv_file}: {e}")
            continue

    elapsed = time.time() - start_time
    LOGGER.info(
        f"Built file manifest for {dataset_root.name}: "
        f"{file_count} files in {elapsed:.2f}s"
    )

    if file_count > 5000:
        LOGGER.warning(
            f"Large manifest: {file_count} files in {dataset_root.name}. "
            f"Consider dataset splitting for better performance."
        )

    return manifest


def _compare_manifests(
    current: Dict[str, Dict],
    cached: Dict[str, Dict]
) -> Tuple[bool, List[str]]:
    """
    Compare current manifest with cached manifest.

    Detects new files, modified files (mtime or size changed), and deleted files.

    Args:
        current: Current file manifest
        cached: Cached file manifest from state file

    Returns:
        (is_valid, change_descriptions)
        - is_valid: True if manifests match (cache valid)
        - change_descriptions: List of human-readable changes (empty if valid)
    """
    changes = []

    # Track counts for summary
    new_count = 0
    modified_count = 0
    deleted_count = 0

    # Check for new and modified files
    for file_path, current_info in current.items():
        if file_path not in cached:
            new_count += 1
            if LOGGER.level <= logging.DEBUG:
                changes.append(f"New file: {file_path}")
        else:
            cached_info = cached[file_path]
            # Check if modified (mtime or size changed)
            if (current_info["mtime"] != cached_info["mtime"] or
                current_info["size"] != cached_info["size"]):
                modified_count += 1
                if LOGGER.level <= logging.DEBUG:
                    changes.append(
                        f"Modified file: {file_path} "
                        f"(mtime: {cached_info['mtime']:.0f} → {current_info['mtime']:.0f}, "
                        f"size: {cached_info['size']} → {current_info['size']})"
                    )

    # Check for deleted files
    for file_path in cached:
        if file_path not in current:
            deleted_count += 1
            if LOGGER.level <= logging.DEBUG:
                changes.append(f"Deleted file: {file_path}")

    # Summary message (always at INFO level)
    if new_count + modified_count + deleted_count > 0:
        summary = (
            f"File changes detected: "
            f"{new_count} new, {modified_count} modified, {deleted_count} deleted"
        )
        changes.insert(0, summary)  # Prepend summary
        is_valid = False
    else:
        is_valid = True

    return is_valid, changes


def _should_skip_with_manifest(
    settings: Settings,
    category: str,
    key: str,
    expected: Dict[str, Any],
    *,
    force_flag: bool,
    dataset_root: Optional[Path] = None
) -> bool:
    """
    Enhanced skip check with file manifest tracking.

    For 'pipeline' category: Checks dataset CSV file manifest
    For 'combined' category: Checks all dataset root manifests
    For other categories: Uses original expectation-only logic

    Args:
        settings: Pipeline settings
        category: State category (pipeline, combined, etc.)
        key: State key (dataset path, etc.)
        expected: Expected state values
        force_flag: If True, always bypass cache
        dataset_root: Dataset root path (for pipeline category)

    Returns:
        True if processing can be skipped (cache valid)
        False if processing required (cache invalid or miss)
    """
    # Force flag always bypasses cache
    if force_flag:
        LOGGER.debug(f"[{category}] force flag set, bypassing cache")
        return False

    # Load cached state
    state = _load_state(settings, category, key)
    if not state:
        LOGGER.debug(f"[{category}] No cached state found")
        return False

    # Check version compatibility
    if state.get("version") != STATE_VERSION:
        LOGGER.info(f"[{category}] State version mismatch, invalidating cache")
        return False

    # Check expectation fields (non_reactive_span_px, etc.)
    for field, value in expected.items():
        if field == "file_manifest":
            continue  # Handle separately below
        if state.get(field) != value:
            LOGGER.info(
                f"[{category}] Config changed: {field} "
                f"({state.get(field)} → {value})"
            )
            return False

    # NEW: File manifest checking for pipeline
    if category == "pipeline" and dataset_root:
        cached_manifest = state.get("file_manifest", {})

        # Build current manifest
        current_manifest = _build_file_manifest(dataset_root)

        # Compare manifests
        is_valid, changes = _compare_manifests(current_manifest, cached_manifest)

        if not is_valid:
            LOGGER.info(f"[CACHE MISS] {changes[0]}")  # Log summary
            for change in changes[1:]:  # Log details at DEBUG
                LOGGER.debug(f"[CACHE MISS] {change}")
            return False

        LOGGER.info(
            f"[CACHE HIT] No file changes in {dataset_root.name} "
            f"({len(current_manifest)} files checked)"
        )
        return True

    # NEW: File manifest checking for combined
    if category == "combined":
        dataset_roots = expected.get("dataset_roots", [])
        cached_manifest = state.get("file_manifest", {})

        # Build combined manifest from all datasets
        current_manifest = {}
        for root_str in dataset_roots:
            root_path = Path(root_str)
            if root_path.exists():
                current_manifest.update(_build_file_manifest(root_path))

        # Compare manifests
        is_valid, changes = _compare_manifests(current_manifest, cached_manifest)

        if not is_valid:
            LOGGER.info(f"[CACHE MISS] {changes[0]}")
            for change in changes[1:]:
                LOGGER.debug(f"[CACHE MISS] {change}")
            return False

        LOGGER.info(
            f"[CACHE HIT] No file changes across {len(dataset_roots)} datasets "
            f"({len(current_manifest)} files checked)"
        )
        return True

    # For other categories (reactions, etc.), use original logic
    LOGGER.info(f"[CACHE HIT] {category} state valid (no file tracking)")
    return True


def _should_skip(settings: Settings, category: str, key: str, expected: dict[str, Any], *, force_flag: bool) -> bool:
    if force_flag:
        return False
    state = _load_state(settings, category, key)
    if not state:
        return False
    if state.get("version") != STATE_VERSION:
        return False
    for field, value in expected.items():
        if state.get(field) != value:
            return False
    return True


def _matrix_plot_config(data: Mapping[str, Any]) -> MatrixPlotConfig:
    opts = dict(data)
    for key in ("matrix_npy", "codes_json", "out_dir"):
        opts[key] = _ensure_path(opts.get(key), key)
    if "trial_orders" in opts and opts["trial_orders"] is not None:
        opts["trial_orders"] = tuple(opts["trial_orders"])
    return MatrixPlotConfig(**opts)  # type: ignore[arg-type]


def _envelope_plot_config(data: Mapping[str, Any]) -> EnvelopePlotConfig:
    opts = dict(data)
    for key in ("matrix_npy", "codes_json", "out_dir"):
        opts[key] = _ensure_path(opts.get(key), key)
    return EnvelopePlotConfig(**opts)  # type: ignore[arg-type]


def _run_envelope_visuals(cfg: Mapping[str, Any] | None) -> None:
    if not cfg:
        return

    matrices_cfg = cfg.get("matrices")
    if matrices_cfg:
        config = _matrix_plot_config(matrices_cfg)
        print(f"[analysis] envelope_visuals.matrices → {config.out_dir}")
        generate_reaction_matrices(config)

    envelopes_cfg = cfg.get("envelopes")
    if envelopes_cfg:
        config = _envelope_plot_config(envelopes_cfg)
        print(f"[analysis] envelope_visuals.envelopes → {config.out_dir}")
        generate_envelope_plots(config)


def _run_training(cfg: Mapping[str, Any] | None) -> None:
    if not cfg:
        return

    envelopes_cfg = cfg.get("envelopes")
    if envelopes_cfg:
        opts = dict(envelopes_cfg)
        opts.setdefault("trial_type", "training")
        config = _envelope_plot_config(opts)
        print(f"[analysis] training.envelopes → {config.out_dir}")
        generate_envelope_plots(config)

    latency_cfg = cfg.get("latency")
    if latency_cfg:
        opts = dict(latency_cfg)
        matrix = _ensure_path(opts.pop("matrix_npy"), "matrix_npy")
        codes = _ensure_path(opts.pop("codes_json"), "codes_json")
        out_dir = _ensure_path(opts.pop("out_dir"), "out_dir")
        trials = tuple(int(t) for t in opts.pop("trials"))
        print(f"[analysis] training.latency → {out_dir}")
        latency_reports(
            matrix,
            codes,
            out_dir,
            trials_of_interest=trials,
            **opts,
        )


def _auto_sync_wide_roots(
    combine_roots: Mapping[str, Path],
    wide_roots: Sequence[Path],
    handled_datasets: Sequence[str] | None = None,
) -> None:
    """Mirror freshly generated combine outputs into their wide-root peers."""

    if not combine_roots or not wide_roots:
        return

    handled = {entry.lower() for entry in (handled_datasets or ())}
    dest_map = {path.name.lower(): path for path in wide_roots}

    for dataset, source in combine_roots.items():
        key = dataset.lower()
        if key in handled:
            continue
        destination = dest_map.get(key)
        if destination is None:
            continue
        if destination == source:
            continue

        print(f"[analysis] combined.wide.auto_mirror[{key}] → {destination}")
        copied, bytes_copied = mirror_directory(str(source), str(destination))
        size_mb = bytes_copied / (1024 * 1024) if bytes_copied else 0.0
        print(
            "[analysis] combined.wide.auto_mirror[{}] copied {} file(s) ({:.1f} MiB).".format(
                key, copied, size_mb
            )
        )


def _run_combined(cfg: Mapping[str, Any] | None, settings: Settings | None) -> None:
    if not cfg:
        return

    combine_roots: dict[str, Path] = {}
    combine_cfg = cfg.get("combine")
    if combine_cfg:
        opts = dict(combine_cfg)
        roots_cfg = opts.pop("roots", None)
        root_value = opts.pop("root", None)
        if roots_cfg:
            if not isinstance(roots_cfg, Sequence) or isinstance(roots_cfg, (str, bytes, Path)):
                roots_iter = [roots_cfg]
            else:
                roots_iter = list(roots_cfg)
        elif root_value is not None:
            roots_iter = [root_value]
        else:
            raise ValueError("combined.combine requires 'root' or 'roots'.")

        for entry in roots_iter:
            entry_path = _ensure_path(entry, "combine.root")
            run_opts = dict(opts)
            run_opts["root"] = entry_path
            if "odor_on" in run_opts and "odor_on_s" not in run_opts:
                run_opts["odor_on_s"] = float(run_opts.pop("odor_on"))
            if "odor_off" in run_opts and "odor_off_s" not in run_opts:
                run_opts["odor_off_s"] = float(run_opts.pop("odor_off"))
            if "odor_latency" in run_opts and "odor_latency_s" not in run_opts:
                run_opts["odor_latency_s"] = float(run_opts.pop("odor_latency"))
            config = CombineConfig(**run_opts)  # type: ignore[arg-type]
            print(f"[analysis] combined.combine → {config.root}")
            combine_distance_angle(config)
            combine_roots[config.root.name.lower()] = config.root

    wide_cfg = cfg.get("wide")
    if wide_cfg:
        roots_cfg = wide_cfg.get("roots", [])
        if not roots_cfg:
            raise ValueError("combined.wide.roots must list at least one directory.")
        roots_iter = (
            roots_cfg
            if isinstance(roots_cfg, Sequence) and not isinstance(roots_cfg, (str, bytes, Path))
            else [roots_cfg]
        )
        wide_root_paths = [_ensure_path(root, "roots") for root in roots_iter]

        mirror_cfg = wide_cfg.get("mirror")
        handled_datasets: list[str] = []
        if mirror_cfg:
            entries = mirror_cfg if isinstance(mirror_cfg, Sequence) else [mirror_cfg]
            for entry in entries:
                src = _ensure_path(entry.get("source"), "mirror.source")
                dest = _ensure_path(entry.get("destination"), "mirror.destination")
                print(f"[analysis] combined.mirror → {dest}")
                copied, bytes_copied = mirror_directory(str(src), str(dest))
                size_mb = bytes_copied / (1024 * 1024) if bytes_copied else 0.0
                print(
                    f"[analysis] combined.mirror copied {copied} file(s) "
                    f"({size_mb:.1f} MiB)."
                )
                handled_datasets.append(src.name.lower())

        auto_sync = wide_cfg.get("auto_sync_roots", True)
        if auto_sync:
            _auto_sync_wide_roots(combine_roots, wide_root_paths, handled_datasets)

        roots = [str(path) for path in wide_root_paths]
        output_csv = _ensure_path(wide_cfg.get("output_csv"), "output_csv")
        measure_cols = wide_cfg.get("measure_cols") or ["envelope_of_rms"]
        fps_fallback = float(wide_cfg.get("fps_fallback", 40.0))
        exclude_cfg = [
            str(_ensure_path(path, "exclude_roots"))
            for path in wide_cfg.get("exclude_roots", [])
        ]
        trial_type_filter = wide_cfg.get("trial_type_filter")
        extra_exports_cfg = wide_cfg.get("trial_type_exports", [])
        extra_exports: dict[str, str] = {}
        extra_matrix_dirs: dict[str, str] = {}
        if extra_exports_cfg:
            entries = (
                extra_exports_cfg
                if isinstance(extra_exports_cfg, Sequence)
                and not isinstance(extra_exports_cfg, (str, bytes))
                else [extra_exports_cfg]
            )
            for entry in entries:
                if not isinstance(entry, Mapping):
                    raise ValueError("trial_type_exports entries must be mappings.")
                trial_type = entry.get("trial_type")
                export_csv = entry.get("output_csv")
                if not trial_type or not export_csv:
                    raise ValueError(
                        "trial_type_exports entries require 'trial_type' and 'output_csv'."
                    )
                export_path = _ensure_path(export_csv, "trial_type_exports.output_csv")
                trial_key = str(trial_type).strip().lower()
                extra_exports[trial_key] = str(export_path)
                matrix_dir = entry.get("matrix_out_dir")
                if matrix_dir:
                    matrix_path = _ensure_path(
                        matrix_dir, "trial_type_exports.matrix_out_dir"
                    )
                    extra_matrix_dirs[trial_key] = str(matrix_path)
        print(f"[analysis] combined.wide → {output_csv}")
        limits = None
        if settings is not None:
            limits = (settings.class2_min, settings.class2_max)
        build_wide_csv(
            roots,
            str(output_csv),
            measure_cols=measure_cols,
            fps_fallback=fps_fallback,
            exclude_roots=exclude_cfg,
            distance_limits=limits,
            trial_type_filter=trial_type_filter,
            extra_trial_exports=extra_exports or None,
            non_reactive_threshold=settings.non_reactive_span_px,
        )

        for trial_key, matrix_dir in extra_matrix_dirs.items():
            export_csv = extra_exports.get(trial_key)
            if not export_csv:
                continue
            print(
                "[analysis] combined.wide.trial_type_matrix[{}] → {}".format(
                    trial_key, matrix_dir
                )
            )
            wide_to_matrix(export_csv, matrix_dir)

    matrix_cfg = cfg.get("matrix")
    if matrix_cfg:
        input_csv = _ensure_path(matrix_cfg.get("input_csv"), "input_csv")
        out_dir = _ensure_path(matrix_cfg.get("out_dir"), "out_dir")
        print(f"[analysis] combined.matrix → {out_dir}")
        wide_to_matrix(str(input_csv), str(out_dir))

    matrices_cfg = cfg.get("matrices")
    if matrices_cfg:
        config = _matrix_plot_config(matrices_cfg)
        print(f"[analysis] combined.matrices → {config.out_dir}")
        generate_reaction_matrices(config)

    envelopes_cfg = cfg.get("envelopes")
    if envelopes_cfg:
        entries = (
            envelopes_cfg
            if isinstance(envelopes_cfg, Sequence)
            and not isinstance(envelopes_cfg, (str, bytes))
            else [envelopes_cfg]
        )
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise ValueError("combined.envelopes entries must be mappings.")
            config = _envelope_plot_config(entry)
            print(f"[analysis] combined.envelopes → {config.out_dir}")
            generate_envelope_plots(config)

    overlay_cfg = cfg.get("overlay")
    if overlay_cfg:
        combined_matrix = _ensure_path(overlay_cfg.get("combined_matrix"), "combined_matrix")
        combined_codes = _ensure_path(overlay_cfg.get("combined_codes"), "combined_codes")
        distance_matrix = _ensure_path(overlay_cfg.get("distance_matrix"), "distance_matrix")
        distance_codes = _ensure_path(overlay_cfg.get("distance_codes"), "distance_codes")
        out_dir = _ensure_path(overlay_cfg.get("out_dir"), "out_dir")
        latency_sec = float(overlay_cfg.get("latency_sec", 0.0))
        after_show = float(overlay_cfg.get("after_show_sec", 30.0))
        thresh = float(overlay_cfg.get("threshold_std_mult", 4.0))
        odor_on = float(overlay_cfg.get("odor_on_s", 30.0))
        odor_off = float(overlay_cfg.get("odor_off_s", 60.0))
        odor_latency = float(overlay_cfg.get("odor_latency_s", overlay_cfg.get("odor_latency", 0.0)))
        overwrite = bool(overlay_cfg.get("overwrite", False))
        print(f"[analysis] combined.overlay → {out_dir}")
        overlay_sources(
            {
                "RMS × Angle": {
                    "MATRIX_NPY": str(combined_matrix),
                    "CODES_JSON": str(combined_codes),
                },
                "RMS": {
                    "MATRIX_NPY": str(distance_matrix),
                    "CODES_JSON": str(distance_codes),
                },
            },
            latency_sec=latency_sec,
            after_show_sec=after_show,
            output_dir=str(out_dir),
            threshold_mult=thresh,
            odor_on_s=odor_on,
            odor_off_s=odor_off,
            odor_latency_s=odor_latency,
            overwrite=overwrite,
            non_reactive_threshold=settings.non_reactive_span_px if settings is not None else None,
        )

    secure_cfg = cfg.get("secure_cleanup")
    if secure_cfg:
        sources = secure_cfg.get("sources") or []
        if not sources:
            raise ValueError("combined.secure_cleanup.sources must list at least one directory.")
        dest = _ensure_path(secure_cfg.get("destination"), "secure_cleanup.destination")
        resolved_sources = [str(_ensure_path(src, "secure_cleanup.sources")) for src in sources]
        perform_cleanup = bool(secure_cfg.get("perform_cleanup", True))
        print(
            f"[analysis] combined.secure_cleanup → dest={dest} sources={resolved_sources} perform_cleanup={perform_cleanup}"
        )
        secure_copy_and_cleanup(
            resolved_sources,
            str(dest),
            perform_cleanup=perform_cleanup,
        )


def _run_reactions(settings: Settings) -> None:
    reaction_cfg = settings.reaction_prediction

    required_pairs: list[tuple[str, str]] = []
    if reaction_cfg.run_prediction:
        required_pairs.extend(
            [
                ("reaction_prediction.data_csv", reaction_cfg.data_csv),
                ("reaction_prediction.model_path", reaction_cfg.model_path),
            ]
        )
    required_pairs.append(("reaction_prediction.output_csv", reaction_cfg.output_csv))

    missing = [name for name, value in required_pairs if not value]

    if missing:
        print(
            "[REACTION] Skipping reaction prediction; missing configuration values: "
            + ", ".join(missing)
        )
        return

    output_csv_path = Path(reaction_cfg.output_csv).expanduser()

    if reaction_cfg.run_prediction:
        prediction_expected = {
            "data_csv": str(Path(reaction_cfg.data_csv).expanduser()),
            "model_path": str(Path(reaction_cfg.model_path).expanduser()),
        }
        data_path = Path(reaction_cfg.data_csv).expanduser()
        model_path = Path(reaction_cfg.model_path).expanduser()
        if data_path.exists():
            prediction_expected["data_mtime"] = data_path.stat().st_mtime
        if model_path.exists():
            prediction_expected["model_mtime"] = model_path.stat().st_mtime
        prediction_expected["output_exists"] = output_csv_path.exists()
        if output_csv_path.exists():
            prediction_expected["output_mtime"] = output_csv_path.stat().st_mtime

        skip_prediction = output_csv_path.exists() and _should_skip(
            settings,
            category="reaction_prediction",
            key="predict",
            expected=prediction_expected,
            force_flag=settings.force.reaction_prediction,
        )

        if skip_prediction:
            print(
                "[analysis] reactions.predict cached → skipping. Set force.reaction_prediction=true to recompute."
            )
        else:
            print("[analysis] reactions.predict → flybehavior-response predict")
            predict_reactions.main(settings)
            prediction_expected["output_exists"] = output_csv_path.exists()
            if output_csv_path.exists():
                prediction_expected["output_mtime"] = output_csv_path.stat().st_mtime
            prediction_expected["version"] = STATE_VERSION
            _write_state(settings, "reaction_prediction", "predict", prediction_expected)
    else:
        print("[analysis] reactions.predict → skipped (run_prediction = False)")

    if not reaction_cfg.matrix.out_dir:
        print("[REACTION] Matrix generation skipped; reaction_prediction.matrix.out_dir is empty.")
        return

    python_exec = reaction_cfg.python or sys.executable
    script_path = Path(__file__).resolve().parent / "reaction_matrix_from_spreadsheet.py"
    csv_path = output_csv_path
    if not csv_path.exists():
        raise FileNotFoundError(f"Reaction prediction CSV not found: {csv_path}")

    matrix_cfg = reaction_cfg.matrix
    out_dir = Path(matrix_cfg.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    matrix_expected = {
        "non_reactive_span_px": settings.non_reactive_span_px,
        "csv_mtime": csv_path.stat().st_mtime,
        "latency_sec": matrix_cfg.latency_sec,
        "after_window_sec": matrix_cfg.after_window_sec,
        "row_gap": matrix_cfg.row_gap,
        "height_per_gap_in": matrix_cfg.height_per_gap_in,
        "bottom_shift_in": matrix_cfg.bottom_shift_in,
        "trial_orders": list(matrix_cfg.trial_orders),
        "include_hexanol": matrix_cfg.include_hexanol,
    }

    skip_matrix = _should_skip(
        settings,
        category="reaction_matrix",
        key="reaction_prediction",
        expected=matrix_expected,
        force_flag=settings.force.reaction_matrix,
    )

    if skip_matrix:
        print(
            "[analysis] reactions.matrix cached → skipping. Set force.reaction_matrix=true to recompute."
        )
        return

    cmd = [
        str(Path(python_exec).expanduser()),
        str(script_path),
        "--csv-path",
        str(csv_path.resolve()),
        "--out-dir",
        str(out_dir.resolve()),
        "--latency-sec",
        str(matrix_cfg.latency_sec),
        "--after-window-sec",
        str(matrix_cfg.after_window_sec),
        "--row-gap",
        str(matrix_cfg.row_gap),
        "--height-per-gap-in",
        str(matrix_cfg.height_per_gap_in),
        "--bottom-shift-in",
        str(matrix_cfg.bottom_shift_in),
        "--non-reactive-threshold",
        str(settings.non_reactive_span_px),
    ]

    for trial_order in matrix_cfg.trial_orders:
        cmd.extend(["--trial-order", trial_order])

    if not matrix_cfg.include_hexanol:
        cmd.append("--exclude-hexanol")
    if matrix_cfg.overwrite:
        cmd.append("--overwrite")

    print("[analysis] reactions.matrix →", " ".join(cmd))
    env = os.environ.copy()
    repo_root = Path(__file__).resolve().parent.parent
    extra_path = str(repo_root / "src")
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [extra_path, pythonpath]))
    subprocess.run(cmd, check=True, env=env)

    matrix_expected["version"] = STATE_VERSION
    _write_state(settings, "reaction_matrix", "reaction_prediction", matrix_expected)


def _run_pipeline(
    config_path: Path,
    *,
    main_directory: str | None = None,
    steps: Sequence[str] | None = None,
) -> None:
    cmd = [sys.executable, "-m", "fbpipe.pipeline", "--config", str(config_path)]
    if steps:
        cmd.extend(steps)
    else:
        cmd.append("all")
    suffix = f" (MAIN_DIRECTORY={main_directory})" if main_directory else ""
    print(f"[analysis] pipeline{suffix} → {' '.join(cmd)}")
    env = os.environ.copy()
    if main_directory:
        env["MAIN_DIRECTORY"] = main_directory
    pythonpath = env.get("PYTHONPATH")
    extra_paths = [str(SRC_ROOT)]
    if pythonpath:
        extra_paths.append(pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(extra_paths)
    try:
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Pipeline failed with exit code {e.returncode}")
        if e.stderr:
            print(e.stderr.decode() if hasattr(e.stderr, "decode") else e.stderr)
        raise


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml", help="Path to pipeline configuration YAML.")
    args = parser.parse_args(argv)

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    settings = load_settings(config_path)

    dataset_cfg = data.get("main_directories")
    dataset_roots: Sequence[str] | None = None
    if isinstance(dataset_cfg, str):
        dataset_roots = (str(Path(dataset_cfg).expanduser()),)
    elif isinstance(dataset_cfg, Sequence):
        dataset_roots = tuple(str(Path(root).expanduser()) for root in dataset_cfg)

    if not dataset_roots:
        combined_cfg = data.get("analysis", {}).get("combined")
        combine_cfg: Mapping[str, Any] | None = None
        if isinstance(combined_cfg, Mapping):
            combine_cfg = combined_cfg.get("combine") if isinstance(combined_cfg.get("combine"), Mapping) else None
        if combine_cfg and isinstance(combine_cfg.get("roots"), Sequence):
            roots_seq = combine_cfg.get("roots")
            if roots_seq:
                dataset_roots = tuple(str(Path(root).expanduser()) for root in roots_seq)

    dataset_roots = tuple(dataset_roots) if dataset_roots else None

    if dataset_roots:
        remaining_steps = [step.name for step in ORDERED_STEPS if step.name != "yolo"]

        pipeline_expectation = {"non_reactive_span_px": settings.non_reactive_span_px}

        yolo_targets: list[str] = []
        pipeline_targets: list[tuple[str, dict[str, Any]]] = []

        for root in dataset_roots:
            resolved = str(Path(root).expanduser())
            skip_pipeline = _should_skip_with_manifest(
                settings,
                category="pipeline",
                key=resolved,
                expected=pipeline_expectation,
                force_flag=settings.force.pipeline,
                dataset_root=Path(resolved),
            )
            if skip_pipeline and not settings.force.yolo:
                print(
                    f"[analysis] pipeline cached → skipping full run for {resolved}. Set force.pipeline=true to recompute."
                )
            else:
                yolo_targets.append(resolved)

            if not skip_pipeline:
                pipeline_targets.append((resolved, dict(pipeline_expectation)))

        if yolo_targets:
            for resolved in yolo_targets:
                _run_pipeline(config_path, main_directory=resolved, steps=("yolo",))

        if pipeline_targets:
            for resolved, expected in pipeline_targets:
                _run_pipeline(config_path, main_directory=resolved, steps=remaining_steps)
                payload = dict(expected)
                payload["version"] = STATE_VERSION
                _write_state(settings, "pipeline", resolved, payload)
        elif not yolo_targets:
            print("[analysis] pipeline skipped for all datasets (cached).")
    else:
        _run_pipeline(config_path)

    analysis_cfg = data.get("analysis") or {}

    combined_cfg_input = analysis_cfg.get("combined")
    if combined_cfg_input:
        combined_expected = {
            "non_reactive_span_px": settings.non_reactive_span_px,
            "dataset_roots": sorted(dataset_roots) if dataset_roots else [],
        }
        skip_combined = _should_skip_with_manifest(
            settings,
            category="combined",
            key="analysis",
            expected=combined_expected,
            force_flag=settings.force.combined,
        )
        if skip_combined:
            print(
                "[analysis] combined analysis cached → skipping. Set force.combined=true to recompute."
            )
        else:
            _run_combined(combined_cfg_input, settings)
            payload = dict(combined_expected)
            payload["version"] = STATE_VERSION
            _write_state(settings, "combined", "analysis", payload)
    else:
        _run_combined(None, settings)
    _run_envelope_visuals(analysis_cfg.get("envelope_visuals"))
    _run_training(analysis_cfg.get("training"))
    _run_reactions(settings)


if __name__ == "__main__":  # pragma: no cover
    main()
