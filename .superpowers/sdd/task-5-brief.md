### Task 5: Recompute existing rig_3 data and validate

**Files:**

- Create: `scripts/pipeline/recompute_rig3_angles.py`
- Test: `tests/test_recompute_rig3_angles.py`

**Interfaces:**

- Consumes: `resolve_anchor`, `MIRRORED_RIGS` from Task 1.
- Produces:
  - `find_rig3_tables(roots: Sequence[Path]) -> list[Path]`
  - `invalidate_angle_columns(df: pd.DataFrame) -> pd.DataFrame`
  - `ANGLE_COLUMNS: tuple[str, ...]` = `("angle_ARB_deg", "angle_centered_deg", "angle_centered_pct", "angle_multiplier")`

- [ ] **Step 1: Write the failing test**

Create `tests/test_recompute_rig3_angles.py`:

```python
"""Invalidation pass for cached rig_3 angle columns.

_process_fly_angles and _ensure_angle_percentages both short-circuit when the
cached columns are already present, so they must be dropped before a recompute.
"""
from __future__ import annotations

import pandas as pd

from scripts.pipeline.recompute_rig3_angles import (
    ANGLE_COLUMNS,
    find_rig3_tables,
    invalidate_angle_columns,
)


def test_angle_columns_cover_both_producers():
    assert set(ANGLE_COLUMNS) == {
        "angle_ARB_deg",
        "angle_centered_deg",
        "angle_centered_pct",
        "angle_multiplier",
    }


def test_invalidate_drops_only_angle_columns():
    df = pd.DataFrame(
        {
            "frame": [0, 1],
            "x_class0": [1.0, 2.0],
            "angle_ARB_deg": [10.0, 20.0],
            "angle_centered_deg": [1.0, 2.0],
            "angle_centered_pct": [5.0, 6.0],
            "angle_multiplier": [1.1, 1.2],
        }
    )
    out = invalidate_angle_columns(df)
    assert list(out.columns) == ["frame", "x_class0"]
    # input must not be mutated
    assert "angle_ARB_deg" in df.columns


def test_invalidate_is_safe_when_columns_absent():
    df = pd.DataFrame({"frame": [0], "x_class0": [1.0]})
    assert list(invalidate_angle_columns(df).columns) == ["frame", "x_class0"]


def test_find_rig3_tables_selects_only_rig3(tmp_path):
    r3 = tmp_path / "july_17_batch_2_rig_3" / "trial_1"
    r2 = tmp_path / "july_17_batch_2_rig_2" / "trial_1"
    r3.mkdir(parents=True)
    r2.mkdir(parents=True)
    (r3 / "updated_t_fly1_distances.parquet").write_bytes(b"")
    (r2 / "updated_t_fly1_distances.parquet").write_bytes(b"")

    found = find_rig3_tables([tmp_path])
    assert len(found) == 1
    assert "rig_3" in str(found[0])
```

- [ ] **Step 2: Run test to verify it fails**

```bash
conda run -n yolo-env python -m pytest tests/test_recompute_rig3_angles.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.pipeline.recompute_rig3_angles'`

- [ ] **Step 3: Write the implementation**

Create `scripts/pipeline/recompute_rig3_angles.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
conda run -n yolo-env python -m pytest tests/test_recompute_rig3_angles.py -v
```

Expected: PASS (4 tests)

- [ ] **Step 5: Dry-run against the real trees**

```bash
conda run -n yolo-env python -m scripts.pipeline.recompute_rig3_angles --dry-run \
  /home/ramanlab/Documents/cole/Data/flys_New \
  /securedstorage/DATAsec/cole/Data-secured-New
```

Expected: every reported path contains `rig_3` and sits under `EB-Training-24-1`
or `EB-Control-24-1` (the `-0.11` datasets hold no `*_distances.parquet`).
**No `rig_2` path may appear.** If one does, stop and fix `find_rig3_tables`
before writing anything.

- [ ] **Step 6: Capture the rig_2 regression baseline BEFORE writing anything**

```bash
conda run -n yolo-env python - <<'PY'
import hashlib, json
from pathlib import Path
root = Path("/home/ramanlab/Documents/cole/Data/flys_New")
sums = {}
for p in sorted(root.rglob("*_distances.parquet")):
    if "rig_2" in str(p):
        sums[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
Path("/tmp/rig2_baseline.json").write_text(json.dumps(sums, indent=0))
print(f"hashed {len(sums)} rig_2 tables -> /tmp/rig2_baseline.json")
PY
```

- [ ] **Step 7: Apply the invalidation, then rerun the analysis**

```bash
conda run -n yolo-env python -m scripts.pipeline.recompute_rig3_angles \
  /home/ramanlab/Documents/cole/Data/flys_New \
  /securedstorage/DATAsec/cole/Data-secured-New
```

Then rerun the analysis for the datasets holding rig_3 data. Only two qualify:

```bash
export MPLBACKEND=Agg ORT_LOGGING_LEVEL=3 PYTHONNOUSERSITE=1
for DS in EB-Training-24-1 EB-Control-24-1; do
  conda run -n yolo-env python scripts/pipeline/run_workflows.py \
    --config config/config_new.yaml --folder "$DS"
done
```

**Why only these two.** Eight rig_3 directories exist, but they split as:

| dataset | rig_3 dirs | distance parquets | action |
| --- | --- | --- | --- |
| EB-Training-24-1 | 5 | present | recompute now |
| EB-Control-24-1 | 1 | present | recompute now |
| 3Oct-Training-24-0.11 | 1 | none | nothing to fix |
| 3Oct-Control-24-0.11 | 1 | none | nothing to fix |

The two `-0.11` datasets have been recorded but never processed — zero
`*_distances.parquet` files — so there is no stale angle data to correct. They
are also absent from `config_new.yaml`'s `datasets:` list (which contains
`3Oct-Training-24-0.1` / `3Oct-Control-24-0.1`, different datasets), so
`--folder` on them would be a no-op. When they are eventually processed, Task 4
writes their angles correctly on the first pass — **provided they are added to
`datasets:` first.** Flag this to the user rather than editing the list here;
it is outside this change.

Both `EB-Training-24-1` and `EB-Control-24-1` are deliberately left LIVE (not
frozen) per the comment at `config_new.yaml:194-196`, so the frozen-dataset
skip added to `_run_combined` does not suppress this recompute. Confirm the run
log does **not** print `[FROZEN] combined.combine → skipping recompute` for
either dataset; if it does, the recompute silently did nothing.

Each dataset also contains rig_2 directories, which is expected and safe: their
cached angle columns are still present and their anchor is unchanged, so
`_process_fly_angles` short-circuits on `"angle_multiplier" in df.columns` and
`_ensure_angle_percentages` writes nothing because `_series_matches` finds the
recomputed values identical. That is exactly what Step 8 verifies.

YOLO does not re-run: `config_new.yaml` sets `force.yolo: false`, and
`yolo_infer` skips any video whose output directory already exists
(`yolo_infer.py:588`). This step recomputes from stored coordinates only.

- [ ] **Step 8: Verify rig_2 is byte-identical**

```bash
conda run -n yolo-env python - <<'PY'
import hashlib, json
from pathlib import Path
sums = json.loads(Path("/tmp/rig2_baseline.json").read_text())
bad = [p for p, h in sums.items()
       if not Path(p).exists() or hashlib.sha256(Path(p).read_bytes()).hexdigest() != h]
print("rig_2 tables changed:", len(bad))
for p in bad[:10]:
    print("  ", p)
assert not bad, "rig_2 output changed -- the fix leaked outside rig_3"
print("OK: rig_2 byte-identical")
PY
```

Expected: `rig_2 tables changed: 0` and `OK: rig_2 byte-identical`.

- [ ] **Step 9: Verify rig_3 is corrected**

```bash
conda run -n yolo-env python - <<'PY'
import numpy as np, pandas as pd
from pathlib import Path
base = Path("/home/ramanlab/Documents/cole/Data/flys_New/EB-Training-24-1")
for rig in ("rig_2", "rig_3"):
    cs = []
    for f in sorted((base / f"july_17_batch_2_{rig}").glob("*/*_fly*_distances.parquet"))[:20]:
        df = pd.read_parquet(f)
        if not {"angle_ARB_deg", "distance_percentage_0_1"} <= set(df.columns):
            continue
        a = pd.to_numeric(df["angle_ARB_deg"], errors="coerce").to_numpy(float)
        d = pd.to_numeric(df["distance_percentage_0_1"], errors="coerce").to_numpy(float)
        m = np.isfinite(a) & np.isfinite(d)
        if m.sum() > 50:
            cs.append(np.corrcoef(d[m], a[m])[0, 1])
    print(f"{rig}: corr(extension, angle) = {np.mean(cs):+.3f}  (n={len(cs)})")
PY
```

Expected: **both rigs report the same sign** (rig_2 near `+0.054`, rig_3 flipped from `-0.220` to roughly `+0.220`). A negative rig_3 value means the recompute did not take effect.

- [ ] **Step 10: Commit**

```bash
git add scripts/pipeline/recompute_rig3_angles.py tests/test_recompute_rig3_angles.py
git commit -m "feat(recompute_rig3_angles): invalidate cached rig_3 angle columns

Both angle producers short-circuit on the cached columns, so stale rig_3
values must be dropped before the corrected anchor can take effect. Pure
recalculation from stored coordinates: no re-encode, no YOLO re-run."
```

---

