#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml


def _find_repo_root(start: Path) -> Path:
    for parent in (start, *start.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    return start.parents[1]


REPO_ROOT = _find_repo_root(Path(__file__).resolve())
SRC_ROOT = REPO_ROOT / "src"
for path in (str(SRC_ROOT), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from fbpipe.config import resolve_config_path
from fbpipe.utils.data_secured_sync import (
    collect_relative_file_sample,
    load_data_secured_sync_config,
    sync_data_secured,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Incrementally back up Data-secured to the file server and Box.")
    parser.add_argument(
        "--config",
        default=str(Path("config") / "config.yaml"),
        help="Path to pipeline configuration YAML.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview transfers without copying files.")
    parser.add_argument(
        "--sample-files",
        type=int,
        default=0,
        help="Only copy the first N source files for verification.",
    )
    parser.add_argument(
        "--destination",
        action="append",
        choices=("file_server", "box"),
        help="Limit the sync to one or more destinations.",
    )
    args = parser.parse_args(argv)

    config_path = resolve_config_path(args.config)
    with config_path.open("r", encoding="utf-8") as fh:
        raw_cfg = yaml.safe_load(fh) or {}

    sample_files: list[str] | None = None
    if args.sample_files:
        cfg = load_data_secured_sync_config(raw_cfg)
        sample_files = collect_relative_file_sample(cfg.source, args.sample_files)
        LOGGER.info("Verification sample: %s", ", ".join(sample_files) if sample_files else "(none found)")

    results = sync_data_secured(
        raw_cfg,
        dry_run=args.dry_run,
        sample_files=sample_files,
        destinations=args.destination,
    )
    if not results:
        return 0
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
