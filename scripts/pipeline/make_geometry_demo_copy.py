#!/usr/bin/env python3
"""Copy ONE batch's raw inputs from the secured mirror into the demo tree.

    python3 scripts/pipeline/make_geometry_demo_copy.py --dataset Hex-Sensitivity-24-0.1 \
        --batch september_02_batch_1_rig_3            # dry run: prints what would be copied
    python3 scripts/pipeline/make_geometry_demo_copy.py --dataset ... --batch ... --run

Creates <dest-base>/<dataset>/<batch>_copy/ holding only the raw input FILES
(output_*.mp4/.csv/.txt, sensors_output_*.csv, session_metadata.txt,
odor_vials.csv). No derived folders, no delete flag ever, existing files are
kept unless --overwrite, and the destination may never lie inside
/securedstorage or Data/flys_New (the production tree would treat the copy as a
new live batch).
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence

INCLUDE_PATTERNS = ("output_*.mp4", "output_*.csv", "output_*.txt", "sensors_output_*.csv",
                    "session_metadata.txt", "odor_vials.csv")
DEFAULT_SOURCE_BASE = "/securedstorage/DATAsec/cole/Data-secured-New"
DEFAULT_DEST_BASE = "/home/ramanlab/Documents/cole/Data/flys_Abdomen"
FORBIDDEN_DEST_PREFIXES = ("/securedstorage", "/home/ramanlab/Documents/cole/Data/flys_New")


def plan_files(src_dir: Path) -> List[Path]:
    """Top-level FILES of the batch that match the include patterns (directories never)."""
    out: List[Path] = []
    for pattern in INCLUDE_PATTERNS:
        out.extend(p for p in Path(src_dir).glob(pattern) if p.is_file())
    return sorted(set(out))


def build_rsync_command(src_dir: Path, dst_dir: Path, *, dry_run: bool, overwrite: bool) -> List[str]:
    cmd = ["rsync", "-a", "--itemize-changes", "--human-readable", "--exclude=*/"]
    cmd += [f"--include={p}" for p in INCLUDE_PATTERNS]
    cmd.append("--exclude=*")
    if not overwrite:
        cmd.append("--ignore-existing")
    if dry_run:
        cmd.append("--dry-run")
    cmd += [str(src_dir).rstrip("/") + "/", str(dst_dir).rstrip("/") + "/"]
    assert not any(a.startswith("--delete") for a in cmd)
    return cmd


def validate_paths(src_dir: Path, dst_dir: Path) -> None:
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    if not src_dir.is_dir():
        raise FileNotFoundError(f"source batch does not exist: {src_dir}")
    if not any(src_dir.glob("output_*.mp4")):
        raise FileNotFoundError(f"source batch holds no output_*.mp4: {src_dir}")
    dst_str = str(dst_dir.resolve() if dst_dir.exists() else dst_dir)
    for prefix in FORBIDDEN_DEST_PREFIXES:
        if dst_str == prefix or dst_str.startswith(prefix.rstrip("/") + "/"):
            raise ValueError(f"refusing destination under {prefix}: {dst_dir}")
    src_dataset = src_dir.resolve().parent
    if dst_str.startswith(str(src_dataset) + "/"):
        raise ValueError(f"refusing destination inside the source dataset {src_dataset}: {dst_dir}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--batch", required=True)
    p.add_argument("--source-base", default=DEFAULT_SOURCE_BASE)
    p.add_argument("--dest-base", default=DEFAULT_DEST_BASE)
    p.add_argument("--suffix", default="_copy")
    p.add_argument("--overwrite", action="store_true", help="let rsync update files that already exist")
    p.add_argument("--run", action="store_true", help="actually copy (default: dry run)")
    a = p.parse_args(argv)
    src = Path(a.source_base) / a.dataset / a.batch
    dst = Path(a.dest_base) / a.dataset / f"{a.batch}{a.suffix}"
    validate_paths(src, dst)
    if shutil.which("rsync") is None:
        print("rsync not found on PATH", file=sys.stderr)
        return 2
    files = plan_files(src)
    total = sum(f.stat().st_size for f in files)
    print(f"{'COPY' if a.run else 'DRY RUN'}: {src} -> {dst}")
    print(f"{len(files)} file(s), {total / 1e6:.1f} MB:")
    for f in files:
        print(f"   {f.name}")
    if a.run:
        dst.mkdir(parents=True, exist_ok=True)
    cmd = build_rsync_command(src, dst, dry_run=not a.run, overwrite=a.overwrite)
    print("$ " + " ".join(cmd))
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        print(f"rsync exited {rc}", file=sys.stderr)
        return rc
    if a.run:
        print(f"done. Point config_abdomen.yaml at dataset_bases.data={a.dest_base} and datasets=[{a.dataset}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
