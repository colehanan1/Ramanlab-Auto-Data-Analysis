"""Export Parquet files to CSV for humans/Excel/external tools.

Usage
-----
    python scripts/export_csv.py --path /path/to/file_or_dir [options]

Flags
-----
--path PATH        File or directory to export (repeatable).
--out-dir DIR      Write CSV files here instead of alongside the source
                   Parquet file (optional).
--recursive        When a directory is given, search recursively for
                   .parquet files (default: top-level only).

Behaviour
---------
For each Parquet file found, reads it via ``read_table`` and writes a
``.csv`` sibling (or a file under ``--out-dir`` with the same name).  Any
existing CSV at the target path is overwritten.

A summary line is printed at the end:
    Exported: N  Errors: M
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a script without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fbpipe.utils.tables import read_table  # noqa: E402


def _find_parquets(path: Path, *, recursive: bool) -> list[Path]:
    """Return all .parquet files under *path* (or just *path* itself)."""
    if path.is_file():
        if path.suffix.lower() == ".parquet":
            return [path]
        print(f"[EXPORT] Skipping non-parquet file: '{path}'")
        return []
    if path.is_dir():
        if recursive:
            return sorted(path.rglob("*.parquet"))
        return sorted(path.glob("*.parquet"))
    print(f"[EXPORT] WARNING: '{path}' does not exist — skipping.")
    return []


def export(
    paths: list[Path],
    *,
    out_dir: Path | None = None,
    recursive: bool = False,
) -> None:
    exported = 0
    errors = 0

    parquet_files: list[Path] = []
    for p in paths:
        parquet_files.extend(_find_parquets(p.expanduser().resolve(), recursive=recursive))

    if not parquet_files:
        print("[EXPORT] No .parquet files found.")
        return

    for pq_path in parquet_files:
        if out_dir is not None:
            csv_path = out_dir / (pq_path.stem + ".csv")
        else:
            csv_path = pq_path.with_suffix(".csv")

        try:
            df = read_table(pq_path)
            if out_dir is not None:
                out_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(csv_path, index=False)
            print(
                f"[EXPORT] OK: '{pq_path.name}' -> '{csv_path}' "
                f"({len(df):,} rows x {len(df.columns)} cols)"
            )
            exported += 1
        except Exception as exc:
            print(f"[EXPORT] ERROR exporting '{pq_path}': {exc}")
            errors += 1

    print(f"\n[EXPORT] Exported: {exported}  Errors: {errors}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export Parquet files to CSV for humans/Excel/external tools.",
    )
    parser.add_argument(
        "--path",
        dest="paths",
        metavar="PATH",
        action="append",
        required=True,
        help="File or directory to export (repeatable).",
    )
    parser.add_argument(
        "--out-dir",
        metavar="DIR",
        default=None,
        help="Write CSV files here instead of alongside the source file.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=False,
        help="Search recursively for .parquet files when a directory is given.",
    )
    args = parser.parse_args()

    export(
        [Path(p) for p in args.paths],
        out_dir=Path(args.out_dir) if args.out_dir else None,
        recursive=args.recursive,
    )


if __name__ == "__main__":
    main()
