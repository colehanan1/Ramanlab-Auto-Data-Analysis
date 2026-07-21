"""Drop cached angle columns from rig_3 tables so they recompute correctly.

rig_3 is a physically mirrored rig and its angles were computed against the
wrong (right-edge) anchor. Both producers of the angle columns short-circuit
when the columns already exist:

  compose_videos_rms._process_fly_angles      -- "if 'angle_multiplier' in df"
  envelope_combined._ensure_angle_percentages -- via _series_matches

so the stale values must be removed before rerunning the analysis. This is a
pure recalculation from the stored eye/proboscis coordinates: no video is
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
