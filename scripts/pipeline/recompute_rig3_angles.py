"""Diagnostic / belt-and-braces tool for cached rig_3 angle columns.

rig_3 is a physically mirrored rig and its angles were computed against the
wrong (right-edge) anchor. Rerunning the analysis alone already corrects
rig_3: `envelope_combined._ensure_angle_percentages` has no short-circuit
guard on the cached angle columns -- it recomputes them unconditionally and
rewrites whenever `_series_matches` finds the recomputed values differ. On
the real rig_3 tables (474 of them, as of this writing) none carry a stale
`angle_multiplier` column, so today there is nothing for this script to find
or drop.

This script exists for two situations where an invalidation pass would
matter:

  1. If a table ever does carry a stale `angle_multiplier` (or the other
     `ANGLE_COLUMNS`) written by some path that predates or bypasses
     `_ensure_angle_percentages`'s unconditional recompute.
  2. If `compose_videos_rms._process_fly_angles` is ever run manually --
     it is not wired into `ORDERED_STEPS` and is not called by `pipeline.py`
     or `run_workflows.py`, so it is unreachable from the live pipeline, but
     if invoked directly it does short-circuit on `"angle_multiplier" in df`
     and would need the cached columns dropped first.

`--dry-run` is the primary intended use: it reports which tables (if any)
carry a stale angle column, without writing anything. This is a pure
recalculation from the stored eye/proboscis coordinates: no video is
re-encoded and YOLO is not re-run.

Usage:
    python -m scripts.pipeline.recompute_rig3_angles --dry-run ROOT [ROOT ...]
    python -m scripts.pipeline.recompute_rig3_angles ROOT [ROOT ...]
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from fbpipe.utils.rig_anchor import MIRRORED_RIGS, rig_token  # noqa: E402

ANGLE_COLUMNS: tuple[str, ...] = (
    "angle_ARB_deg",
    "angle_centered_deg",
    "angle_centered_pct",
    "angle_multiplier",
)


def find_rig3_tables(roots: Sequence[Path]) -> list[Path]:
    """Return every per-fly distance table living under a mirrored rig."""
    out: list[Path] = []
    for root in roots:
        for path in Path(root).rglob("*_distances.parquet"):
            if rig_token(path) in MIRRORED_RIGS:
                out.append(path)
    return sorted(out)


def invalidate_angle_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *df* with every cached angle column removed."""
    present = [c for c in ANGLE_COLUMNS if c in df.columns]
    return df.drop(columns=present)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("roots", nargs="+", type=Path)
    ap.add_argument("--dry-run", action="store_true",
                    help="List what would change without writing.")
    args = ap.parse_args(argv)

    tables = find_rig3_tables(args.roots)
    print(f"found {len(tables)} rig_3 tables")

    changed = 0
    for path in tables:
        try:
            df = pd.read_parquet(path)
        except Exception as exc:  # a partial/corrupt table must not abort the run
            print(f"  SKIP (unreadable) {path}: {exc}")
            continue
        present = [c for c in ANGLE_COLUMNS if c in df.columns]
        if not present:
            continue
        changed += 1
        if args.dry_run:
            print(f"  would drop {present} from {path}")
            continue
        invalidate_angle_columns(df).to_parquet(path, index=False)
        print(f"  dropped {present} from {path}")

    verb = "would update" if args.dry_run else "updated"
    print(f"{verb} {changed} of {len(tables)} tables")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
