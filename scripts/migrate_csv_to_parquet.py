"""Migrate pipeline-produced CSV files to Parquet.

Usage
-----
    python scripts/migrate_csv_to_parquet.py --root /path/to/data [options]

Flags
-----
--root PATH        Root directory to search (repeatable).
--delete-csv       Remove the source CSV after a successful conversion
                   (default: keep CSV).
--glob PATTERN     Glob pattern to match CSV files (default: ``*.csv``).
--dry-run          Print what would be done without writing or deleting.
--force            Re-convert even when an up-to-date .parquet already exists.

Behaviour
---------
Idempotent by default: if a ``.parquet`` file with the same stem already exists
AND its mtime is newer than the source CSV, the file is skipped.  Pass
``--force`` to always overwrite.

A summary line is printed at the end:
    Converted: N  Skipped: M  Bytes saved: X
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a script without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fbpipe.utils.tables import write_table  # noqa: E402

try:
    import pandas as pd
except ImportError as exc:
    sys.exit(f"[MIGRATE] Required package missing: {exc}")


def _should_skip(csv_path: Path, parquet_path: Path, *, force: bool) -> bool:
    """Return True when conversion can be skipped."""
    if force or not parquet_path.exists():
        return False
    return parquet_path.stat().st_mtime >= csv_path.stat().st_mtime


def migrate(
    roots: list[Path],
    *,
    glob: str = "*.csv",
    delete_csv: bool = False,
    dry_run: bool = False,
    force: bool = False,
) -> None:
    converted = 0
    skipped = 0
    bytes_saved = 0

    for root in roots:
        root = root.expanduser().resolve()
        if not root.is_dir():
            print(f"[MIGRATE] WARNING: '{root}' is not a directory — skipping.")
            continue
        for csv_path in sorted(root.rglob(glob)):
            if not csv_path.is_file() or csv_path.suffix.lower() != ".csv":
                continue
            parquet_path = csv_path.with_suffix(".parquet")
            if _should_skip(csv_path, parquet_path, force=force):
                skipped += 1
                print(f"[MIGRATE] SKIP (up-to-date): {csv_path.relative_to(root)}")
                continue

            if dry_run:
                print(f"[MIGRATE] DRY-RUN: would convert {csv_path.relative_to(root)}")
                converted += 1
                continue

            try:
                df = pd.read_csv(csv_path)
                actual_out = write_table(df, parquet_path)
                csv_bytes = csv_path.stat().st_size
                pq_bytes = actual_out.stat().st_size
                saved = csv_bytes - pq_bytes
                bytes_saved += saved
                print(
                    f"[MIGRATE] OK: {csv_path.relative_to(root)} -> "
                    f"{actual_out.name} "
                    f"({csv_bytes:,} -> {pq_bytes:,} bytes, saved {saved:,})"
                )
                converted += 1
                if delete_csv:
                    csv_path.unlink()
                    print(f"[MIGRATE] DELETED: {csv_path.relative_to(root)}")
            except Exception as exc:
                print(f"[MIGRATE] ERROR converting '{csv_path}': {exc}")

    action = "Would convert" if dry_run else "Converted"
    print(
        f"\n[MIGRATE] {action}: {converted}  Skipped: {skipped}  "
        f"Bytes saved: {bytes_saved:,}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate pipeline CSV files to Parquet.",
    )
    parser.add_argument(
        "--root",
        dest="roots",
        metavar="PATH",
        action="append",
        required=True,
        help="Root directory to search (repeatable).",
    )
    parser.add_argument(
        "--delete-csv",
        action="store_true",
        default=False,
        help="Delete the source CSV after successful conversion.",
    )
    parser.add_argument(
        "--glob",
        default="*.csv",
        help="Glob pattern to match files (default: '*.csv').",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print what would happen without writing or deleting files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Re-convert even if an up-to-date .parquet already exists.",
    )
    args = parser.parse_args()

    migrate(
        [Path(r) for r in args.roots],
        glob=args.glob,
        delete_csv=args.delete_csv,
        dry_run=args.dry_run,
        force=args.force,
    )


if __name__ == "__main__":
    main()
