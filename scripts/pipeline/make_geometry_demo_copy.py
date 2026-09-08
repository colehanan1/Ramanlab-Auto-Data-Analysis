#!/usr/bin/env python3
"""Copy raw trial inputs from the secured mirror into the abdomen-pipeline tree.

    # one batch, as <batch>_copy (demo)
    python3 scripts/pipeline/make_geometry_demo_copy.py --dataset Hex-Sensitivity-24-0.1 \
        --batch september_02_batch_1_rig_3            # dry run: prints what would be copied
    python3 scripts/pipeline/make_geometry_demo_copy.py --dataset ... --batch ... --run
    # EVERY batch of every dataset, with the ORIGINAL folder names (so results join with
    # the scoring / prediction tables): --all [--datasets A B ...]
    python3 scripts/pipeline/make_geometry_demo_copy.py --all --run

Creates <dest-base>/<dataset>/<batch><suffix>/ holding only the raw input FILES
(output_*.mp4/.csv/.txt, sensors_output_*.csv, session_metadata.txt,
odor_vials.csv). No derived folders, no delete flag ever, existing files are
kept unless --overwrite, and the destination may never lie inside
/securedstorage or Data/flys_New (the production tree would treat the copy as a
new live batch).
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

INCLUDE_PATTERNS = ("output_*.mp4", "output_*.csv", "output_*.txt", "sensors_output_*.csv",
                    "session_metadata.txt", "odor_vials.csv")
DEFAULT_SOURCE_BASE = "/securedstorage/DATAsec/cole/Data-secured-New"
DEFAULT_DEST_BASE = "/home/ramanlab/Documents/cole/Data/flys_Abdomen"
FORBIDDEN_DEST_PREFIXES = ("/securedstorage", "/home/ramanlab/Documents/cole/Data/flys_New")


@dataclass(frozen=True)
class CopyJob:
    src: Path
    dst: Path


def plan_all(source_base: Path, dest_base: Path, *, suffix: str = "",
             datasets: Optional[Sequence[str]] = None) -> List[CopyJob]:
    """Every ``<dataset>/<batch>`` under ``source_base`` that holds ``output_*.mp4``,
    mapped to ``<dest_base>/<dataset>/<batch><suffix>``. Sorted by dataset then batch."""
    source_base, dest_base = Path(source_base), Path(dest_base)
    wanted = set(datasets) if datasets else None
    jobs: List[CopyJob] = []
    for ds in sorted(p for p in source_base.iterdir() if p.is_dir()):
        if wanted is not None and ds.name not in wanted:
            continue
        for batch in sorted(p for p in ds.iterdir() if p.is_dir()):
            if any(batch.glob("output_*.mp4")):
                jobs.append(CopyJob(batch, dest_base / ds.name / f"{batch.name}{suffix}"))
    return jobs


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


def _canonical(path: Path) -> str:
    """Absolute path with ``~``, ``..`` and symlinks in every EXISTING prefix resolved,
    so the check sees the same location rsync will write to (a not-yet-existing
    tail cannot hide a traversal)."""
    return os.path.realpath(os.path.expanduser(str(path)))


def _under(path: str, prefix: str) -> bool:
    prefix = prefix.rstrip("/")
    return path == prefix or path.startswith(prefix + "/")


def validate_paths(src_dir: Path, dst_dir: Path) -> None:
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    if not src_dir.is_dir():
        raise FileNotFoundError(f"source batch does not exist: {src_dir}")
    if not any(src_dir.glob("output_*.mp4")):
        raise FileNotFoundError(f"source batch holds no output_*.mp4: {src_dir}")
    dst_str = _canonical(dst_dir)
    for prefix in FORBIDDEN_DEST_PREFIXES:
        if _under(dst_str, prefix) or _under(dst_str, _canonical(Path(prefix))):
            raise ValueError(f"refusing destination under {prefix}: {dst_dir}")
    src_dataset = _canonical(src_dir.parent)
    if _under(dst_str, src_dataset):
        raise ValueError(f"refusing destination inside the source dataset {src_dataset}: {dst_dir}")


def _copy_one(src: Path, dst: Path, *, run: bool, overwrite: bool, verbose: bool) -> int:
    validate_paths(src, dst)
    files = plan_files(src)
    total = sum(f.stat().st_size for f in files)
    print(f"{'COPY' if run else 'DRY RUN'}: {src} -> {dst}  ({len(files)} file(s), {total / 1e6:.1f} MB)")
    if verbose:
        for f in files:
            print(f"   {f.name}")
    if run:
        dst.mkdir(parents=True, exist_ok=True)
    cmd = build_rsync_command(src, dst, dry_run=not run, overwrite=overwrite)
    if verbose:
        print("$ " + " ".join(cmd))
    rc = subprocess.run(cmd, stdout=None if verbose else subprocess.DEVNULL).returncode
    if rc != 0:
        print(f"rsync exited {rc} for {src}", file=sys.stderr)
    return rc


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=None)
    p.add_argument("--batch", default=None)
    p.add_argument("--all", action="store_true", help="every batch of every dataset (original names)")
    p.add_argument("--datasets", nargs="*", default=None, help="with --all: restrict to these datasets")
    p.add_argument("--source-base", default=DEFAULT_SOURCE_BASE)
    p.add_argument("--dest-base", default=DEFAULT_DEST_BASE)
    p.add_argument("--suffix", default=None, help="folder suffix (default: _copy for one batch, none for --all)")
    p.add_argument("--overwrite", action="store_true", help="let rsync update files that already exist")
    p.add_argument("--run", action="store_true", help="actually copy (default: dry run)")
    a = p.parse_args(argv)
    if shutil.which("rsync") is None:
        print("rsync not found on PATH", file=sys.stderr)
        return 2
    if a.all:
        jobs = plan_all(Path(a.source_base), Path(a.dest_base), suffix=a.suffix or "", datasets=a.datasets)
        grand = sum(sum(f.stat().st_size for f in plan_files(j.src)) for j in jobs)
        print(f"{'COPY' if a.run else 'DRY RUN'}: {len(jobs)} batch(es), {grand / 1e9:.1f} GB -> {a.dest_base}")
        failed = 0
        for j in jobs:
            if _copy_one(j.src, j.dst, run=a.run, overwrite=a.overwrite, verbose=False) != 0:
                failed += 1
        print(f"{'copied' if a.run else 'would copy'} {len(jobs) - failed}/{len(jobs)} batch(es); failures: {failed}")
        return 1 if failed else 0
    if not a.dataset or not a.batch:
        p.error("--dataset and --batch are required unless --all is given")
    src = Path(a.source_base) / a.dataset / a.batch
    dst = Path(a.dest_base) / a.dataset / f"{a.batch}{'_copy' if a.suffix is None else a.suffix}"
    rc = _copy_one(src, dst, run=a.run, overwrite=a.overwrite, verbose=True)
    if rc == 0 and a.run:
        print(f"done. Point config_abdomen.yaml at dataset_bases.data={a.dest_base} and datasets=[{a.dataset}]")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
