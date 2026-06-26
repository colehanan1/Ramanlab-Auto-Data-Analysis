#!/usr/bin/env python3
"""Delete raw ``.h264`` elementary-stream files to reclaim disk space.

The recording rigs leave a raw ``<name>.h264`` alongside the usable
``<name>.mp4``. The ``.h264`` is a redundant raw stream once the ``.mp4``
exists, so this removes ``.h264`` files from the working ``flys_New`` tree and
the ``/securedstorage`` backup.

SAFETY: by default it only deletes a ``.h264`` when a matching ``.mp4`` exists
somewhere (sibling, or same-stem anywhere in the working/secured trees), so no
recording is ever lost. ``.h264`` files with no matching ``.mp4`` ("orphans")
are KEPT unless you pass --allow-orphans.

⚠️  DRY-RUN BY DEFAULT. Pass --delete to actually remove files (irreversible).

Examples:
  # safe: show what would be deleted, remove nothing
  python scripts/pipeline/cleanup_h264_streams.py --config config/config_new.yaml

  # delete covered .h264 from both working + secured (keeps orphans)
  python scripts/pipeline/cleanup_h264_streams.py --config config/config_new.yaml --delete

  # working folder only
  python scripts/pipeline/cleanup_h264_streams.py --config config/config_new.yaml --locations working --delete
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fbpipe.config import load_settings, get_main_directories  # noqa: E402


def _human(n: int) -> str:
    return f"{n / 1e9:.2f} GB" if n >= 1e9 else f"{n / 1e6:.1f} MB"


def _gather(root: Path, pattern: str) -> list[Path]:
    return sorted(root.rglob(pattern)) if root.is_dir() else []


def _report(label: str, h264s: list[Path], mp4_stems: set[str], allow_orphans: bool) -> list[Path]:
    to_delete, orphans = [], []
    for h in h264s:
        covered = (h.with_suffix(".mp4").exists() or h.stem in mp4_stems)
        if covered or allow_orphans:
            to_delete.append(h)
        if not covered:
            orphans.append(h)
    del_bytes = sum(h.stat().st_size for h in to_delete if h.exists())
    print(f"\n[{label}] {len(h264s)} .h264 files")
    print(f"  -> would delete : {len(to_delete)}  ({_human(del_bytes)})")
    print(f"     orphans (NO matching .mp4): {len(orphans)}"
          + ("  [INCLUDED via --allow-orphans -> raw lost!]" if allow_orphans and orphans else "  [KEPT]"))
    for h in orphans[:5]:
        print(f"     orphan kept: {h}")
    return to_delete


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--locations", choices=["working", "secured", "both"], default="both")
    ap.add_argument("--secured-root", default="/securedstorage/DATAsec/cole/Data-secured-New")
    ap.add_argument("--allow-orphans", action="store_true",
                    help="Also delete .h264 with no matching .mp4 (loses the raw recording).")
    ap.add_argument("--delete", action="store_true",
                    help="Actually delete. Without this it is a dry-run (default).")
    args = ap.parse_args()

    cfg = load_settings(args.config)
    working_roots = [Path(r) for r in get_main_directories(cfg)]
    secured_root = Path(args.secured_root)

    # Index every .mp4 stem across BOTH trees so coverage is location-agnostic.
    mp4_stems: set[str] = set()
    for r in working_roots:
        mp4_stems.update(p.stem for p in _gather(r, "*.mp4"))
    mp4_stems.update(p.stem for p in _gather(secured_root, "*.mp4"))
    print(f"indexed {len(mp4_stems)} distinct .mp4 stems (working + secured)")

    targets: list[Path] = []
    if args.locations in ("working", "both"):
        wv = []
        for r in working_roots:
            wv += _gather(r, "*.h264")
        targets += _report("WORKING flys_New", wv, mp4_stems, args.allow_orphans)
    if args.locations in ("secured", "both"):
        targets += _report("SECURED storage", _gather(secured_root, "*.h264"), mp4_stems, args.allow_orphans)

    total = sum(h.stat().st_size for h in targets if h.exists())
    print(f"\n==== TOTAL to delete: {len(targets)} .h264 files, {_human(total)} ====")

    if not args.delete:
        print("DRY-RUN — nothing deleted. Re-run with --delete to remove these files.")
        return 0

    print("DELETING .h264 streams (irreversible)...")
    removed = freed = failed = 0
    for h in targets:
        try:
            sz = h.stat().st_size
            h.unlink()
            removed += 1
            freed += sz
        except OSError as exc:
            failed += 1
            print(f"  FAILED {h}: {exc}")
    print(f"DONE: deleted {removed} .h264 files, freed {_human(freed)}, {failed} failures.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
