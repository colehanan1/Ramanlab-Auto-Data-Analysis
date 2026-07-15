# Dataset Freeze Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a finished dataset keep its rows in the wide CSV without re-derivation, and stop redrawing its figures, via two independent per-dataset config flags.

**Architecture:** A new `src/fbpipe/freeze.py` owns the cache (fingerprint, save/load of a dataset's row slice as parquet + `meta.json` under `<cache_dir>/frozen/<block>/<dataset>/`). `build_wide_csv` gains one new parameter, `frozen_slices: Mapping[str, tuple[DataFrame, int]]`, and stays config-free: it skips walking those roots, folds each slice's own max trace length into the global `max_len`, and splices the rows back via `DataFrame.reindex(columns=...)` — which is exactly pad-and-truncate. `run_workflows` resolves which datasets are frozen, loads slices, and writes the cache after any live derivation. Auto-rebuild is implicit: a cache miss or fingerprint drift returns `None`, the dataset is simply absent from `frozen_slices`, and the root gets walked normally.

**Tech Stack:** Python 3, pandas, pyarrow (parquet), PyYAML, pytest.

**Spec:** `docs/superpowers/specs/2026-07-15-dataset-freeze-design.md`

## Global Constraints

- **Legacy protocol output must stay byte-for-byte identical.** Freeze is v2-and-legacy agnostic but must never alter existing output when no dataset is frozen. Verify with a worktree at the branch point using the SAME data — never `git stash` (a no-op diff produces a vacuous pass), never a stale baseline (the pipeline regenerates `model_predictions.csv`).
- **`freeze.data` must never walk the frozen root.** No `iterdir`, no `rglob`, no `stat`, no hashing. This is the entire performance claim.
- **Default off.** Absent or empty `freeze:` block means both flags `False` and behavior identical to today.
- **Pure trust applies to raw data only.** Config IS checked (it is already in memory); raw data is NOT.
- **Config drift and cache miss both auto-rebuild that dataset once**, then re-cache.
- **A figure is skipped only when EVERY dataset contributing to it is frozen for figures.**
- `dataset` column values are the root directory **basename** (`envelope_combined.py:2668`). Cache keys use the same basename, unmodified.
- Existing test suite is 538 passing. Do not regress it.

---

### Task 1: Config surface — `freeze.data` / `freeze.figures`

**Files:**
- Modify: `src/fbpipe/config.py:456-483` (DatasetOverride dataclass), `:881-895` (override parsing)
- Test: `tests/test_freeze_config.py` (create)

**Interfaces:**
- Consumes: nothing (first task)
- Produces:
  - `DatasetOverride.freeze_data: bool = False`
  - `DatasetOverride.freeze_figures: bool = False`

- [ ] **Step 1: Write the failing test**

Create `tests/test_freeze_config.py`:

```python
"""Per-dataset freeze flags parse off the dataset_overrides block."""

import textwrap

from fbpipe.config import load_settings


def _write_cfg(tmp_path, overrides_yaml: str):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        textwrap.dedent(
            f"""
            protocol: v2
            dataset_bases:
              data: {tmp_path}/data
              secured: {tmp_path}/secured
            dataset_overrides:
            {overrides_yaml}
            """
        )
    )
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "secured").mkdir(exist_ok=True)
    return cfg


def test_freeze_flags_parse_both_true(tmp_path):
    cfg = _write_cfg(
        tmp_path,
        """
              EB-Control-24-1:
                freeze:
                  data: true
                  figures: true
        """,
    )
    s = load_settings(str(cfg))
    ov = s.dataset_overrides["EB-Control-24-1"]
    assert ov.freeze_data is True
    assert ov.freeze_figures is True


def test_freeze_flags_are_independent(tmp_path):
    """data:true + figures:false must be representable -- the styling-iteration mode."""
    cfg = _write_cfg(
        tmp_path,
        """
              EB-Control-24-1:
                freeze:
                  data: true
                  figures: false
              EB-Training-24-1:
                freeze:
                  figures: true
        """,
    )
    s = load_settings(str(cfg))
    a = s.dataset_overrides["EB-Control-24-1"]
    b = s.dataset_overrides["EB-Training-24-1"]
    assert (a.freeze_data, a.freeze_figures) == (True, False)
    assert (b.freeze_data, b.freeze_figures) == (False, True)


def test_absent_freeze_block_defaults_false(tmp_path):
    """A dataset with other overrides but no freeze: block is not frozen."""
    cfg = _write_cfg(
        tmp_path,
        """
              EB-Control-24-1:
                trial_type_override: testing
        """,
    )
    s = load_settings(str(cfg))
    ov = s.dataset_overrides["EB-Control-24-1"]
    assert ov.freeze_data is False
    assert ov.freeze_figures is False
    # The sibling override must still parse -- freeze must not disturb it.
    assert ov.trial_type_override == "testing"


def test_empty_freeze_block_defaults_false(tmp_path):
    cfg = _write_cfg(
        tmp_path,
        """
              EB-Control-24-1:
                freeze: {}
        """,
    )
    s = load_settings(str(cfg))
    ov = s.dataset_overrides["EB-Control-24-1"]
    assert ov.freeze_data is False
    assert ov.freeze_figures is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_freeze_config.py -v`
Expected: FAIL with `AttributeError: 'DatasetOverride' object has no attribute 'freeze_data'`

- [ ] **Step 3: Add the fields to the dataclass**

In `src/fbpipe/config.py`, inside the `DatasetOverride` dataclass, after the
`odor_remap` field (currently the last field, ~`:483`):

```python
    # Per-dataset freeze. Two INDEPENDENT flags; either may be set alone.
    # ``freeze_data``: do not re-derive this dataset's wide rows -- splice them
    # from the freeze cache and never walk its root. ``freeze_figures``: do not
    # regenerate figures that belong solely to this dataset. A figure drawn from
    # several datasets is skipped only when EVERY contributor is frozen.
    # Both default False, so an absent ``freeze:`` block is today's behavior.
    freeze_data: bool = False
    freeze_figures: bool = False
```

- [ ] **Step 4: Parse the nested `freeze:` block**

In `src/fbpipe/config.py`, in the `for ds_name, block in raw_overrides.items():`
loop (~`:886`), add before the closing `)` of the `DatasetOverride(...)` call —
i.e. as two further keyword arguments after `odor_remap=...`:

```python
            freeze_data=bool((block.get("freeze") or {}).get("data", False)),
            freeze_figures=bool((block.get("freeze") or {}).get("figures", False)),
```

Note `(block.get("freeze") or {})` handles all three of: key absent, `freeze:`
with an empty/None body, and `freeze: {}`.

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_freeze_config.py -v`
Expected: PASS (4 passed)

- [ ] **Step 6: Verify no regression**

Run: `python -m pytest tests/test_config_settings.py tests/test_randompanel_odor_remap.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add tests/test_freeze_config.py src/fbpipe/config.py
git commit -m "feat(config): add independent freeze.data / freeze.figures per-dataset flags"
```

---

### Task 2: Freeze cache module

**Files:**
- Create: `src/fbpipe/freeze.py`
- Test: `tests/test_freeze_cache.py` (create)

**Interfaces:**
- Consumes: `DatasetOverride` (Task 1)
- Produces:
  - `FREEZE_SCHEMA_VERSION: int = 1`
  - `class FrozenSlice(NamedTuple): rows: pd.DataFrame; own_max_len: int`
  - `freeze_flags(cfg, dataset: str, *, thawed=(), thaw_all=False) -> tuple[bool, bool]`
  - `build_fingerprint(*, protocol, measure_cols, fps_fallback, distance_limits, non_reactive_threshold, low_max_threshold_px, use_per_trial_baseline, override) -> dict`
  - `slice_dir(cache_dir, wide_block: str, dataset: str) -> Path`
  - `own_max_len(rows: pd.DataFrame) -> int`
  - `save_slice(cache_dir, wide_block, dataset, rows, fingerprint) -> None`
  - `load_slice(cache_dir, wide_block, dataset, fingerprint) -> FrozenSlice | None`

`load_slice` returns `None` on cache miss OR fingerprint drift OR unreadable
cache. That single return value is how "auto-rebuild once" is implemented: the
caller simply omits the dataset from `frozen_slices` and the root is walked.

- [ ] **Step 1: Write the failing test**

Create `tests/test_freeze_cache.py`:

```python
"""Freeze cache: fingerprint, own_max_len, save/load round-trip."""

import numpy as np
import pandas as pd
import pytest

from fbpipe import freeze
from fbpipe.config import DatasetOverride


def _rows(trace_lens, n_val_cols):
    """Build a wide-CSV-shaped frame: trace_len + NaN-padded dir_val_* columns."""
    recs = []
    for tl in trace_lens:
        rec = {"dataset": "DS", "trial_type": "testing", "trace_len": tl}
        for i in range(n_val_cols):
            rec[f"dir_val_{i}"] = float(i) if i < tl else np.nan
        recs.append(rec)
    return pd.DataFrame(recs)


def _fp(**kw):
    base = dict(
        protocol="v2",
        measure_cols=["envelope_of_rms"],
        fps_fallback=40.0,
        distance_limits=None,
        non_reactive_threshold=12.5,
        low_max_threshold_px=5.0,
        use_per_trial_baseline=False,
        override=DatasetOverride(),
    )
    base.update(kw)
    return freeze.build_fingerprint(**base)


def test_own_max_len_is_max_trace_len(tmp_path):
    assert freeze.own_max_len(_rows([10, 7, 3], 10)) == 10


def test_own_max_len_clamped_to_present_columns():
    """A truncated row can carry trace_len > the dir_val columns it actually has.
    own_max_len must not over-report, or reload pads to a width with no data."""
    rows = _rows([10], 10)
    rows["trace_len"] = 99  # claims 99 samples, only 10 dir_val columns exist
    assert freeze.own_max_len(rows) == 10


def test_save_load_round_trip(tmp_path):
    rows = _rows([10, 7], 10)
    fp = _fp()
    freeze.save_slice(tmp_path, "combined_base", "DS", rows, fp)
    got = freeze.load_slice(tmp_path, "combined_base", "DS", fp)
    assert got is not None
    assert got.own_max_len == 10
    pd.testing.assert_frame_equal(
        got.rows.reset_index(drop=True), rows.reset_index(drop=True)
    )


def test_load_miss_returns_none(tmp_path):
    assert freeze.load_slice(tmp_path, "combined_base", "NOPE", _fp()) is None


def test_fingerprint_drift_returns_none(tmp_path):
    """A changed analysis parameter must invalidate -- else one CSV mixes two
    parameterizations."""
    freeze.save_slice(tmp_path, "combined_base", "DS", _rows([5], 5), _fp())
    drifted = _fp(non_reactive_threshold=10.0)
    assert freeze.load_slice(tmp_path, "combined_base", "DS", drifted) is None


def test_protocol_drift_returns_none(tmp_path):
    """legacy and v2 have different column sets -- a schema mismatch, not a value
    drift."""
    freeze.save_slice(tmp_path, "combined_base", "DS", _rows([5], 5), _fp())
    assert freeze.load_slice(tmp_path, "combined_base", "DS", _fp(protocol="legacy")) is None


def test_odor_remap_drift_returns_none(tmp_path):
    freeze.save_slice(tmp_path, "combined_base", "DS", _rows([5], 5), _fp())
    drifted = _fp(override=DatasetOverride(odor_remap={"Citral": "Yeast"}))
    assert freeze.load_slice(tmp_path, "combined_base", "DS", drifted) is None


def test_figure_output_subdir_does_not_invalidate(tmp_path):
    """figure_output_subdir routes figures; it cannot change a row value, so it
    must NOT invalidate a data cache."""
    freeze.save_slice(tmp_path, "combined_base", "DS", _rows([5], 5), _fp())
    same = _fp(override=DatasetOverride(figure_output_subdir="RandomPanel"))
    assert freeze.load_slice(tmp_path, "combined_base", "DS", same) is not None


def test_blocks_do_not_collide(tmp_path):
    """combined_base and distance_base have different measure_cols and so
    different rows. Serving one for the other is a silent data corruption."""
    cb = _rows([5], 5)
    db = _rows([9], 9)
    freeze.save_slice(tmp_path, "combined_base", "DS", cb, _fp(measure_cols=["combined_pct"]))
    freeze.save_slice(tmp_path, "distance_base", "DS", db, _fp(measure_cols=["distance_percentage"]))
    got_cb = freeze.load_slice(tmp_path, "combined_base", "DS", _fp(measure_cols=["combined_pct"]))
    got_db = freeze.load_slice(tmp_path, "distance_base", "DS", _fp(measure_cols=["distance_percentage"]))
    assert got_cb.own_max_len == 5
    assert got_db.own_max_len == 9


def test_corrupt_cache_returns_none_not_raise(tmp_path):
    freeze.save_slice(tmp_path, "combined_base", "DS", _rows([5], 5), _fp())
    (freeze.slice_dir(tmp_path, "combined_base", "DS") / "meta.json").write_text("{not json")
    assert freeze.load_slice(tmp_path, "combined_base", "DS", _fp()) is None


class _Cfg:
    def __init__(self, overrides):
        self.dataset_overrides = overrides


def test_freeze_flags_reads_override():
    cfg = _Cfg({"DS": DatasetOverride(freeze_data=True, freeze_figures=False)})
    assert freeze.freeze_flags(cfg, "DS") == (True, False)


def test_freeze_flags_unknown_dataset_is_unfrozen():
    assert freeze.freeze_flags(_Cfg({}), "NOPE") == (False, False)


def test_thaw_overrides_named_dataset():
    cfg = _Cfg({"DS": DatasetOverride(freeze_data=True, freeze_figures=True)})
    assert freeze.freeze_flags(cfg, "DS", thawed=["DS"]) == (False, False)


def test_thaw_does_not_affect_other_datasets():
    cfg = _Cfg({
        "DS": DatasetOverride(freeze_data=True, freeze_figures=True),
        "OTHER": DatasetOverride(freeze_data=True, freeze_figures=True),
    })
    assert freeze.freeze_flags(cfg, "OTHER", thawed=["DS"]) == (True, True)


def test_thaw_all_overrides_everything():
    cfg = _Cfg({"DS": DatasetOverride(freeze_data=True, freeze_figures=True)})
    assert freeze.freeze_flags(cfg, "DS", thaw_all=True) == (False, False)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_freeze_cache.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fbpipe.freeze'`

- [ ] **Step 3: Write the module**

Create `src/fbpipe/freeze.py`:

```python
"""Per-dataset freeze cache.

A frozen dataset keeps its rows in the wide CSV without being re-derived. This
module owns the cache only: fingerprinting, and saving/loading a dataset's row
slice. It deliberately knows nothing about ``build_wide_csv`` -- the splice
lives there, and the policy (which datasets are frozen) lives in the caller.

Layout, under the configured ``cache_dir``::

    <cache_dir>/frozen/<wide_block>/<dataset>/
        rows.parquet    # the dataset's slice of that block's wide CSV
        meta.json       # fingerprint + own_max_len

The cache is a DERIVED artifact. Deleting it is always safe: every frozen
dataset's raw data stays on disk, so a miss simply re-derives.

The ``wide_block`` component is a partitioning LABEL, not the correctness
mechanism. Correctness rests on the fingerprint (which includes
``measure_cols``). Keying by block is a safe over-partition: it may duplicate an
identical slice across two blocks that share parameters, but it can never serve
the wrong rows.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, NamedTuple, Optional

import pandas as pd

FREEZE_SCHEMA_VERSION = 1


class FrozenSlice(NamedTuple):
    """A dataset's cached wide rows plus its OWN maximum trace length.

    ``own_max_len`` is load-bearing. ``build_wide_csv`` pads *and truncates*
    every row to one global ``max_len``; if a frozen dataset's own length is not
    folded into that global max, its cached rows are silently chopped.
    """

    rows: pd.DataFrame
    own_max_len: int


def freeze_flags(
    cfg: Any,
    dataset: str,
    *,
    thawed: Iterable[str] = (),
    thaw_all: bool = False,
) -> tuple[bool, bool]:
    """Return ``(freeze_data, freeze_figures)`` for *dataset*, honoring thaw.

    ``thawed`` names datasets to treat as unfrozen for this run only;
    ``thaw_all`` unfreezes everything. Neither edits config.
    """
    if thaw_all or dataset in set(thawed):
        return (False, False)
    override = (getattr(cfg, "dataset_overrides", None) or {}).get(dataset)
    if override is None:
        return (False, False)
    return (bool(override.freeze_data), bool(override.freeze_figures))


def build_fingerprint(
    *,
    protocol: str,
    measure_cols: Iterable[str],
    fps_fallback: float,
    distance_limits: Optional[tuple[float, float]],
    non_reactive_threshold: Optional[float],
    low_max_threshold_px: float,
    use_per_trial_baseline: bool,
    override: Any,
) -> dict:
    """Everything that determines a dataset's rows OTHER than its raw data.

    Raw data is deliberately absent: ``freeze.data`` never walks the root. Config
    IS checked, because it is already in memory and free -- without this, changing
    an analysis parameter while a dataset is frozen would leave one CSV silently
    mixing two parameterizations.

    ``figure_output_subdir`` is deliberately EXCLUDED: it routes figures and
    cannot change a row value, so it must not invalidate a data cache.
    """
    return {
        "protocol": str(protocol),
        "measure_cols": [str(c) for c in measure_cols],
        "fps_fallback": float(fps_fallback),
        "distance_limits": (
            None if distance_limits is None else [float(x) for x in distance_limits]
        ),
        "non_reactive_threshold": (
            None if non_reactive_threshold is None else float(non_reactive_threshold)
        ),
        "low_max_threshold_px": float(low_max_threshold_px),
        "use_per_trial_baseline": bool(use_per_trial_baseline),
        "override": {
            "trial_type_override": getattr(override, "trial_type_override", None),
            "odor_on_s": getattr(override, "odor_on_s", None),
            "odor_off_s": getattr(override, "odor_off_s", None),
            "light_only": bool(getattr(override, "light_only", False)),
            "light_start_s": getattr(override, "light_start_s", None),
            "light_duration_s": getattr(override, "light_duration_s", None),
            "odor_remap": dict(getattr(override, "odor_remap", {}) or {}),
        },
    }


def _safe(name: str) -> str:
    """Filesystem-safe cache component. Dataset names carry dots and dashes
    (``Hex-Control-24-0.1``); only separators are unsafe."""
    return str(name).replace("/", "_").replace("\\", "_").strip() or "_"


def slice_dir(cache_dir: str | Path, wide_block: str, dataset: str) -> Path:
    return Path(cache_dir).expanduser() / "frozen" / _safe(wide_block) / _safe(dataset)


def own_max_len(rows: pd.DataFrame) -> int:
    """This dataset's own maximum trace length, in samples.

    ``trace_len`` is written as ``int(len(values))`` -- the UNPADDED length
    (``envelope_combined.py:3114``). But a row whose values exceeded the run's
    global max was truncated on write while ``trace_len`` kept the original
    figure, so clamp to the dir_val columns actually present. Over-reporting
    would pad the global max out to a width holding no data.
    """
    val_cols = [c for c in rows.columns if str(c).startswith("dir_val_")]
    present = len(val_cols)
    if "trace_len" not in rows.columns or rows.empty:
        return present
    tl = pd.to_numeric(rows["trace_len"], errors="coerce").max()
    if pd.isna(tl):
        return present
    return max(0, min(int(tl), present))


def save_slice(
    cache_dir: str | Path,
    wide_block: str,
    dataset: str,
    rows: pd.DataFrame,
    fingerprint: Mapping[str, Any],
) -> None:
    """Persist *rows* as *dataset*'s slice of *wide_block*.

    Stores the rows exactly as written, across ALL trial types -- the splice
    routes them to the main output or an extra trial export the same way live
    rows are routed.
    """
    target = slice_dir(cache_dir, wide_block, dataset)
    target.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(target / "rows.parquet", index=False)
    meta = {
        "schema_version": FREEZE_SCHEMA_VERSION,
        "dataset": str(dataset),
        "wide_block": str(wide_block),
        "own_max_len": own_max_len(rows),
        "row_count": int(len(rows)),
        "columns": [str(c) for c in rows.columns],
        "fingerprint": dict(fingerprint),
    }
    (target / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))


def load_slice(
    cache_dir: str | Path,
    wide_block: str,
    dataset: str,
    fingerprint: Mapping[str, Any],
) -> Optional[FrozenSlice]:
    """Return the cached slice, or ``None`` if it cannot be trusted.

    ``None`` means "re-derive this dataset": cache miss, fingerprint drift,
    schema-version bump, or an unreadable/corrupt cache. Callers implement
    auto-rebuild by simply omitting the dataset from ``frozen_slices``, which
    lets its root be walked normally.
    """
    target = slice_dir(cache_dir, wide_block, dataset)
    meta_path = target / "meta.json"
    rows_path = target / "rows.parquet"
    if not meta_path.is_file() or not rows_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text())
    except Exception:
        return None
    if meta.get("schema_version") != FREEZE_SCHEMA_VERSION:
        return None
    if meta.get("fingerprint") != json.loads(json.dumps(dict(fingerprint))):
        return None
    try:
        rows = pd.read_parquet(rows_path)
    except Exception:
        return None
    return FrozenSlice(rows=rows, own_max_len=int(meta.get("own_max_len", 0)))


def drift_reason(
    cached: Mapping[str, Any], current: Mapping[str, Any]
) -> Optional[str]:
    """Name the first fingerprint field that differs, for a one-line log."""
    for key in sorted(set(cached) | set(current)):
        if cached.get(key) != current.get(key):
            return f"{key}: {cached.get(key)!r} -> {current.get(key)!r}"
    return None
```

Note the fingerprint comparison round-trips `current` through JSON before
comparing. The cached side has already been through JSON (tuples became lists,
int keys became strings); comparing a live dict directly against it would report
spurious drift on every run.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_freeze_cache.py -v`
Expected: PASS (15 passed)

- [ ] **Step 5: Prove the fingerprint tests are not vacuous**

Temporarily change `build_fingerprint` to drop `non_reactive_threshold` from the
returned dict. Re-run. `test_fingerprint_drift_returns_none` MUST fail. Revert.

This is a mutation placed INSIDE the unit under test — do not skip it and do not
trust a passing report without it.

Run: `python -m pytest tests/test_freeze_cache.py::test_fingerprint_drift_returns_none -v`
Expected (with mutation): FAIL. Then revert and confirm PASS.

- [ ] **Step 6: Commit**

```bash
git add src/fbpipe/freeze.py tests/test_freeze_cache.py
git commit -m "feat(freeze): add per-dataset freeze cache (fingerprint, slice save/load)"
```

---

### Task 3: Splice frozen slices into `build_wide_csv`

**Files:**
- Modify: `scripts/analysis/envelope_combined.py:2600-2614` (signature), `:2664-2668` (root walk), `:2733` (empty-items guard), `:2736-2743` (max_len), and the row-write region ending ~`:3298`
- Test: `tests/test_freeze_wide.py` (create)

**Interfaces:**
- Consumes: `FrozenSlice` from Task 2 (as a plain `(rows, own_max_len)` tuple — no import needed)
- Produces: `build_wide_csv(..., frozen_slices: Mapping[str, tuple[pd.DataFrame, int]] | None = None)`

`build_wide_csv` stays config-free: it receives already-loaded slices. Policy
(which datasets are frozen, cache load) lives in Task 4's caller.

- [ ] **Step 1: Write the failing test**

Create `tests/test_freeze_wide.py`:

```python
"""build_wide_csv splices frozen slices without walking their roots."""

import numpy as np
import pandas as pd
import pytest

import scripts.analysis.envelope_combined as ec
import scripts.analysis.envelope_visuals as ev


def _make_dataset(root, n_samples, fly="october_01_fly1"):
    """One testing trial of *n_samples* points, in the layout build_wide_csv walks."""
    out = root / fly / "angle_distance_rms_envelope"
    out.mkdir(parents=True, exist_ok=True)
    values = np.linspace(0, 100, n_samples, dtype=float)
    pd.DataFrame({"envelope_of_rms": values}).to_csv(
        out / f"{fly}_testing_1_angle_distance_rms_envelope.csv", index=False
    )
    return root


def _build(roots, out_csv, **kw):
    ec.build_wide_csv(
        [str(r) for r in roots], str(out_csv), measure_cols=["envelope_of_rms"], **kw
    )
    return pd.read_csv(out_csv)


@pytest.fixture(autouse=True)
def _v2():
    ev.set_protocol("v2")


def test_frozen_slice_round_trips_identically_when_frozen_is_longer(tmp_path):
    """THE load-bearing test.

    The frozen dataset is given a LONGER trace than the live one on purpose. With
    equal lengths this test would pass even if the own_max_len fold were missing
    entirely -- the global max would be correct by accident. Unequal lengths are
    what make it bite.
    """
    live = _make_dataset(tmp_path / "LIVE", 8)
    frozen = _make_dataset(tmp_path / "FROZEN", 20)  # LONGER than live

    baseline_csv = tmp_path / "baseline.csv"
    baseline = _build([live, frozen], baseline_csv)

    # Seed the cache from the baseline, exactly as Task 4's caller will.
    frozen_rows = baseline[baseline["dataset"] == "FROZEN"].reset_index(drop=True)
    own = int(frozen_rows["trace_len"].max())
    assert own == 20

    spliced_csv = tmp_path / "spliced.csv"
    spliced = _build(
        [live, frozen], spliced_csv, frozen_slices={"FROZEN": (frozen_rows, own)}
    )

    # Byte-identical output, frozen or not.
    assert baseline_csv.read_bytes() != b""
    pd.testing.assert_frame_equal(
        baseline.sort_values(["dataset", "fly"]).reset_index(drop=True),
        spliced.sort_values(["dataset", "fly"]).reset_index(drop=True),
    )
    # The global max grew to the FROZEN dataset's length, not the live one's.
    assert "dir_val_19" in spliced.columns
    assert "dir_val_20" not in spliced.columns


def test_frozen_rows_not_truncated_when_no_live_dataset_is_as_long(tmp_path):
    """Directly pins the failure the own_max_len fold prevents: silent chopping."""
    live = _make_dataset(tmp_path / "LIVE", 5)
    frozen = _make_dataset(tmp_path / "FROZEN", 30)

    baseline = _build([live, frozen], tmp_path / "b.csv")
    frozen_rows = baseline[baseline["dataset"] == "FROZEN"].reset_index(drop=True)

    spliced = _build(
        [live, frozen], tmp_path / "s.csv", frozen_slices={"FROZEN": (frozen_rows, 30)}
    )
    row = spliced[spliced["dataset"] == "FROZEN"].iloc[0]
    # Sample 29 is real data, not padding, and must survive.
    assert not pd.isna(row["dir_val_29"])
    assert int(row["trace_len"]) == 30


def test_frozen_shorter_than_live_is_nan_padded(tmp_path):
    live = _make_dataset(tmp_path / "LIVE", 25)
    frozen = _make_dataset(tmp_path / "FROZEN", 6)

    baseline = _build([live, frozen], tmp_path / "b.csv")
    frozen_rows = baseline[baseline["dataset"] == "FROZEN"].reset_index(drop=True)

    spliced = _build(
        [live, frozen], tmp_path / "s.csv", frozen_slices={"FROZEN": (frozen_rows, 6)}
    )
    row = spliced[spliced["dataset"] == "FROZEN"].iloc[0]
    assert not pd.isna(row["dir_val_5"])   # last real sample
    assert pd.isna(row["dir_val_24"])      # padding out to the live max
    assert int(row["trace_len"]) == 6


def test_frozen_root_is_never_walked(tmp_path):
    """The entire performance claim. If this passes vacuously the feature is a lie."""
    live = _make_dataset(tmp_path / "LIVE", 8)
    frozen = _make_dataset(tmp_path / "FROZEN", 8)

    baseline = _build([live, frozen], tmp_path / "b.csv")
    frozen_rows = baseline[baseline["dataset"] == "FROZEN"].reset_index(drop=True)

    walked = []
    real_iterdir = ec.Path.iterdir

    def _spy(self):
        walked.append(str(self))
        return real_iterdir(self)

    ec.Path.iterdir = _spy
    try:
        _build(
            [live, frozen], tmp_path / "s.csv",
            frozen_slices={"FROZEN": (frozen_rows, 8)},
        )
    finally:
        ec.Path.iterdir = real_iterdir

    assert not any("FROZEN" in w for w in walked), f"frozen root was walked: {walked}"
    # Guard against the spy simply never firing -- the live root MUST be walked.
    assert any("LIVE" in w for w in walked), "spy never fired; test proves nothing"


def test_all_datasets_frozen_does_not_raise(tmp_path):
    """items is empty when every root is frozen. The pre-existing guard at
    envelope_combined.py:2733 raises RuntimeError on empty items -- it must not
    fire when frozen rows are present."""
    a = _make_dataset(tmp_path / "A", 8)
    b = _make_dataset(tmp_path / "B", 12)

    baseline = _build([a, b], tmp_path / "b.csv")
    rows_a = baseline[baseline["dataset"] == "A"].reset_index(drop=True)
    rows_b = baseline[baseline["dataset"] == "B"].reset_index(drop=True)

    out = _build(
        [a, b], tmp_path / "s.csv",
        frozen_slices={"A": (rows_a, 8), "B": (rows_b, 12)},
    )
    assert set(out["dataset"]) == {"A", "B"}
    assert len(out) == len(baseline)


def test_no_eligible_data_and_no_frozen_still_raises(tmp_path):
    """The empty-items guard must survive for the case it was written for."""
    empty = tmp_path / "EMPTY"
    empty.mkdir()
    with pytest.raises(RuntimeError, match="No eligible"):
        _build([empty], tmp_path / "s.csv")


def test_frozen_slices_none_is_todays_behavior(tmp_path):
    """Default off: absent frozen_slices must not perturb output."""
    live = _make_dataset(tmp_path / "LIVE", 8)
    a = _build([live], tmp_path / "a.csv")
    b = _build([live], tmp_path / "b.csv", frozen_slices=None)
    pd.testing.assert_frame_equal(a, b)


def test_frozen_training_rows_route_to_extra_export(tmp_path):
    """A cached slice holds ALL trial types; the splice must route them the same
    way live rows are routed (testing -> main, training -> extra export)."""
    root = tmp_path / "DS"
    fly = "october_01_fly1"
    out = root / fly / "angle_distance_rms_envelope"
    out.mkdir(parents=True, exist_ok=True)
    for tt in ("testing", "training"):
        pd.DataFrame({"envelope_of_rms": np.linspace(0, 100, 9)}).to_csv(
            out / f"{fly}_{tt}_1_angle_distance_rms_envelope.csv", index=False
        )

    main_csv = tmp_path / "main.csv"
    train_csv = tmp_path / "train.csv"
    ec.build_wide_csv(
        [str(root)], str(main_csv),
        measure_cols=["envelope_of_rms"],
        extra_trial_exports={"training": str(train_csv)},
    )
    all_rows = pd.concat(
        [pd.read_csv(main_csv), pd.read_csv(train_csv)], ignore_index=True
    )
    assert set(all_rows["trial_type"].str.lower()) == {"testing", "training"}

    m2 = tmp_path / "main2.csv"
    t2 = tmp_path / "train2.csv"
    ec.build_wide_csv(
        [str(root)], str(m2),
        measure_cols=["envelope_of_rms"],
        extra_trial_exports={"training": str(t2)},
        frozen_slices={"DS": (all_rows, 9)},
    )
    assert set(pd.read_csv(m2)["trial_type"].str.lower()) == {"testing"}
    assert set(pd.read_csv(t2)["trial_type"].str.lower()) == {"training"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_freeze_wide.py -v`
Expected: FAIL with `TypeError: build_wide_csv() got an unexpected keyword argument 'frozen_slices'`

- [ ] **Step 3: Add the parameter**

In `scripts/analysis/envelope_combined.py`, add to `build_wide_csv`'s keyword-only
parameters (after `use_per_trial_baseline: bool = False,` at `:2613`):

```python
    frozen_slices: Mapping[str, tuple[pd.DataFrame, int]] | None = None,
```

Then immediately after `out_path.parent.mkdir(parents=True, exist_ok=True)` (~`:2620`):

```python
    # Frozen datasets: rows come from the freeze cache, their roots are NEVER
    # walked. Keyed by the `dataset` column value == root basename (:2668).
    # Values are (rows, own_max_len). This function stays config-free -- the
    # caller decides what is frozen and loads the slices.
    frozen: dict[str, tuple[pd.DataFrame, int]] = {
        str(k): v for k, v in (frozen_slices or {}).items()
    }
    if frozen:
        print(f"[FROZEN] Splicing cached rows for: {', '.join(sorted(frozen))}")
```

- [ ] **Step 4: Skip frozen roots before any filesystem access**

In the root loop at `:2664-2668`, after `dataset = root.name`, insert the skip
BEFORE the `for fly_dir in sorted(p for p in root.iterdir() ...)` line:

```python
    for root in _normalise_roots(roots):
        if root in exclude:
            print(f"[SKIP] Excluding dataset root: {root}")
            continue
        dataset = root.name
        if dataset in frozen:
            # freeze.data is a promise, not a check: never stat/iterdir/hash a
            # frozen root. This skip IS the performance win.
            print(f"[FROZEN] Not walking root (cached rows will be spliced): {root}")
            continue
        for fly_dir in sorted(p for p in root.iterdir() if p.is_dir()):
```

- [ ] **Step 5: Let the empty-items guard tolerate an all-frozen run**

Replace the guard at `:2733`:

```python
    if not items:
        raise RuntimeError("No eligible testing/training CSVs found in provided roots.")
```

with:

```python
    if not items and not frozen:
        raise RuntimeError("No eligible testing/training CSVs found in provided roots.")
```

Every root being frozen is a legitimate run: there are no live items, but there
are cached rows to write.

- [ ] **Step 6: Fold each frozen dataset's own max into the global max_len**

Immediately after the max_len loop ending `max_len = max(max_len, n_rows)` (`:2743`):

```python
    # Fold each frozen dataset's OWN max trace length into the global max.
    # Without this, a frozen dataset whose traces are longer than any live
    # dataset's has its cached rows silently CHOPPED by the truncation branch
    # below (:3283-3284) -- data loss with no error.
    for _frozen_rows, _own in frozen.values():
        max_len = max(max_len, int(_own))
```

- [ ] **Step 7: Splice the frozen rows**

In the row-write region, immediately AFTER the `for item in items:` write loop
closes and BEFORE `flagged_path = out_path.with_name(...)` (~`:3298`):

```python
    # Splice frozen rows. reindex(columns=...) IS pad-and-truncate: it inserts
    # NaN for dir_val_* columns the cache lacks and drops any beyond max_len --
    # exactly the live semantics at :3281-3284, in one call.
    target_cols = metadata + list(AUC_COLUMNS) + value_cols
    for _ds in sorted(frozen):
        _rows, _ = frozen[_ds]
        aligned = _rows.reindex(columns=target_cols)
        trial_keys = aligned["trial_type"].astype(str).str.strip().str.lower()
        for trial_key, group in aligned.groupby(trial_keys):
            if trial_key in main_trial_allow:
                group.to_csv(out_path, index=False, mode="a", header=False)
                main_rows_written += len(group)
            extra_target = extra_paths.get(trial_key)
            if extra_target is not None:
                group.to_csv(extra_target, index=False, mode="a", header=False)
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `python -m pytest tests/test_freeze_wide.py -v`
Expected: PASS (8 passed)

- [ ] **Step 9: Prove the max_len fold test is not vacuous**

Comment out the Step 6 fold loop. Re-run. BOTH
`test_frozen_slice_round_trips_identically_when_frozen_is_longer` and
`test_frozen_rows_not_truncated_when_no_live_dataset_is_as_long` MUST fail.
Restore the loop.

If they still pass, the test fixtures' trace lengths are not actually unequal —
fix the fixture, not the assertion.

Run: `python -m pytest tests/test_freeze_wide.py -v`
Expected (with fold removed): 2 FAILED. Then restore and confirm PASS.

- [ ] **Step 10: Verify no regression**

Run: `python -m pytest tests/test_envelope_combined.py -v`
Expected: PASS

- [ ] **Step 11: Commit**

```bash
git add scripts/analysis/envelope_combined.py tests/test_freeze_wide.py
git commit -m "feat(build_wide_csv): splice frozen dataset slices, folding own_max_len into the global max"
```

---

### Task 4: Wire freeze into `run_workflows` (load + write cache)

**Files:**
- Modify: `scripts/pipeline/run_workflows.py` — the three `build_wide_csv` call sites (`:1135`, `:1257`, `:1403`)
- Test: `tests/test_freeze_pipeline.py` (create)

**Interfaces:**
- Consumes: `freeze.freeze_flags`, `freeze.build_fingerprint`, `freeze.load_slice`, `freeze.save_slice`, `freeze.FrozenSlice` (Task 2); `build_wide_csv(..., frozen_slices=...)` (Task 3)
- Produces:
  - `_resolve_frozen_slices(settings, roots, wide_block, *, measure_cols, fps_fallback, distance_limits, non_reactive_threshold, low_max_threshold_px, use_per_trial_baseline, thawed, thaw_all) -> dict[str, FrozenSlice]`
  - `_write_freeze_cache(settings, wide_block, output_csv, extra_export_paths, *, fingerprint_for) -> None`

- [ ] **Step 1: Write the failing test**

Create `tests/test_freeze_pipeline.py`:

```python
"""Freeze resolution + cache writing in run_workflows."""

import numpy as np
import pandas as pd
import pytest

import scripts.pipeline.run_workflows as rw
from fbpipe import freeze
from fbpipe.config import DatasetOverride


class _Settings:
    def __init__(self, tmp_path, overrides, datasets):
        self.cache_dir = str(tmp_path / "cache")
        self.dataset_overrides = overrides
        self.datasets = tuple(datasets)
        self.protocol = "v2"


def _rows(dataset, tl=5):
    rec = {"dataset": dataset, "fly": "f1", "trial_type": "testing", "trace_len": tl}
    for i in range(tl):
        rec[f"dir_val_{i}"] = float(i)
    return pd.DataFrame([rec])


def _fp_kw():
    return dict(
        measure_cols=["envelope_of_rms"],
        fps_fallback=40.0,
        distance_limits=None,
        non_reactive_threshold=12.5,
        low_max_threshold_px=5.0,
        use_per_trial_baseline=False,
    )


def test_frozen_dataset_with_cache_resolves(tmp_path):
    s = _Settings(tmp_path, {"DS": DatasetOverride(freeze_data=True)}, ["DS"])
    fp = freeze.build_fingerprint(
        protocol="v2", override=s.dataset_overrides["DS"], **_fp_kw()
    )
    freeze.save_slice(s.cache_dir, "combined_base", "DS", _rows("DS"), fp)

    got = rw._resolve_frozen_slices(
        s, [str(tmp_path / "DS")], "combined_base", thawed=(), thaw_all=False, **_fp_kw()
    )
    assert "DS" in got
    assert got["DS"].own_max_len == 5


def test_unfrozen_dataset_is_not_resolved_even_with_a_cache(tmp_path):
    """A cache exists but the dataset is not frozen -- it must be derived live."""
    s = _Settings(tmp_path, {"DS": DatasetOverride(freeze_data=False)}, ["DS"])
    fp = freeze.build_fingerprint(
        protocol="v2", override=s.dataset_overrides["DS"], **_fp_kw()
    )
    freeze.save_slice(s.cache_dir, "combined_base", "DS", _rows("DS"), fp)

    got = rw._resolve_frozen_slices(
        s, [str(tmp_path / "DS")], "combined_base", thawed=(), thaw_all=False, **_fp_kw()
    )
    assert got == {}


def test_frozen_but_cache_miss_auto_rebuilds(tmp_path):
    """No cache -> absent from frozen_slices -> root gets walked. That IS the
    auto-rebuild."""
    s = _Settings(tmp_path, {"DS": DatasetOverride(freeze_data=True)}, ["DS"])
    got = rw._resolve_frozen_slices(
        s, [str(tmp_path / "DS")], "combined_base", thawed=(), thaw_all=False, **_fp_kw()
    )
    assert got == {}


def test_frozen_but_config_drift_auto_rebuilds(tmp_path):
    s = _Settings(tmp_path, {"DS": DatasetOverride(freeze_data=True)}, ["DS"])
    fp = freeze.build_fingerprint(
        protocol="v2", override=s.dataset_overrides["DS"], **_fp_kw()
    )
    freeze.save_slice(s.cache_dir, "combined_base", "DS", _rows("DS"), fp)

    kw = _fp_kw()
    kw["non_reactive_threshold"] = 10.0  # drift
    got = rw._resolve_frozen_slices(
        s, [str(tmp_path / "DS")], "combined_base", thawed=(), thaw_all=False, **kw
    )
    assert got == {}


def test_thaw_ignores_freeze(tmp_path):
    s = _Settings(tmp_path, {"DS": DatasetOverride(freeze_data=True)}, ["DS"])
    fp = freeze.build_fingerprint(
        protocol="v2", override=s.dataset_overrides["DS"], **_fp_kw()
    )
    freeze.save_slice(s.cache_dir, "combined_base", "DS", _rows("DS"), fp)

    got = rw._resolve_frozen_slices(
        s, [str(tmp_path / "DS")], "combined_base", thawed=["DS"], thaw_all=False, **_fp_kw()
    )
    assert got == {}


def test_write_freeze_cache_stores_each_dataset_slice(tmp_path):
    s = _Settings(tmp_path, {}, ["A", "B"])
    out = tmp_path / "wide.csv"
    pd.concat([_rows("A", 4), _rows("B", 6)], ignore_index=True).to_csv(out, index=False)

    rw._write_freeze_cache(
        s, "combined_base", str(out), [],
        fingerprint_for=lambda ds: freeze.build_fingerprint(
            protocol="v2", override=DatasetOverride(), **_fp_kw()
        ),
    )
    fp = freeze.build_fingerprint(protocol="v2", override=DatasetOverride(), **_fp_kw())
    a = freeze.load_slice(s.cache_dir, "combined_base", "A", fp)
    b = freeze.load_slice(s.cache_dir, "combined_base", "B", fp)
    assert a is not None and b is not None
    assert a.own_max_len == 4
    assert b.own_max_len == 6
    assert set(a.rows["dataset"]) == {"A"}   # slices must not bleed into each other
    assert set(b.rows["dataset"]) == {"B"}


def test_write_freeze_cache_includes_extra_trial_exports(tmp_path):
    """A slice must hold ALL trial types, or freezing loses the training rows."""
    s = _Settings(tmp_path, {}, ["A"])
    main = tmp_path / "wide.csv"
    train = tmp_path / "wide_training.csv"
    _rows("A", 4).to_csv(main, index=False)
    tr = _rows("A", 4)
    tr["trial_type"] = "training"
    tr.to_csv(train, index=False)

    rw._write_freeze_cache(
        s, "combined_base", str(main), [str(train)],
        fingerprint_for=lambda ds: freeze.build_fingerprint(
            protocol="v2", override=DatasetOverride(), **_fp_kw()
        ),
    )
    fp = freeze.build_fingerprint(protocol="v2", override=DatasetOverride(), **_fp_kw())
    got = freeze.load_slice(s.cache_dir, "combined_base", "A", fp)
    assert set(got.rows["trial_type"].str.lower()) == {"testing", "training"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_freeze_pipeline.py -v`
Expected: FAIL with `AttributeError: module 'scripts.pipeline.run_workflows' has no attribute '_resolve_frozen_slices'`

- [ ] **Step 3: Add the two helpers**

In `scripts/pipeline/run_workflows.py`, near the other module-level helpers
(after `_build_file_manifest` / `_compare_manifests`, ~`:414`), add:

```python
def _freeze_fingerprint(
    settings,
    dataset: str,
    *,
    measure_cols,
    fps_fallback: float,
    distance_limits,
    non_reactive_threshold,
    low_max_threshold_px: float,
    use_per_trial_baseline: bool,
) -> dict:
    """Fingerprint for one dataset under one set of build_wide_csv parameters.

    Shared by the load and save paths so they can never disagree -- if they did,
    every run would see spurious drift and re-derive.
    """
    from fbpipe import freeze as _freeze

    return _freeze.build_fingerprint(
        protocol=settings.protocol,
        measure_cols=measure_cols,
        fps_fallback=fps_fallback,
        distance_limits=distance_limits,
        non_reactive_threshold=non_reactive_threshold,
        low_max_threshold_px=low_max_threshold_px,
        use_per_trial_baseline=use_per_trial_baseline,
        override=(settings.dataset_overrides or {}).get(dataset),
    )


def _resolve_frozen_slices(
    settings,
    roots,
    wide_block: str,
    *,
    measure_cols,
    fps_fallback: float,
    distance_limits,
    non_reactive_threshold,
    low_max_threshold_px: float,
    use_per_trial_baseline: bool,
    thawed=(),
    thaw_all: bool = False,
) -> dict:
    """Return {dataset: FrozenSlice} for every root frozen for DATA with a
    trustworthy cache.

    A dataset is omitted when it is not frozen, is thawed, has no cache, or its
    fingerprint drifted. Omission IS the auto-rebuild: build_wide_csv then walks
    the root normally.
    """
    from fbpipe import freeze as _freeze

    out: dict = {}
    for root in roots:
        dataset = Path(str(root)).name
        if not dataset:
            continue
        freeze_data, _ = _freeze.freeze_flags(
            settings, dataset, thawed=thawed, thaw_all=thaw_all
        )
        if not freeze_data:
            continue
        fingerprint = _freeze_fingerprint(
            settings,
            dataset,
            measure_cols=measure_cols,
            fps_fallback=fps_fallback,
            distance_limits=distance_limits,
            non_reactive_threshold=non_reactive_threshold,
            low_max_threshold_px=low_max_threshold_px,
            use_per_trial_baseline=use_per_trial_baseline,
        )
        slice_ = _freeze.load_slice(settings.cache_dir, wide_block, dataset, fingerprint)
        if slice_ is None:
            print(
                f"[FREEZE] {dataset} is frozen but its {wide_block} cache is "
                f"missing or stale -- rebuilding it once, then re-caching."
            )
            continue
        out[dataset] = slice_
    return out


def _write_freeze_cache(
    settings,
    wide_block: str,
    output_csv: str,
    extra_export_paths,
    *,
    fingerprint_for,
) -> None:
    """Cache each dataset's slice of a freshly written wide CSV.

    Runs after EVERY live derivation, frozen or not, so that freezing a dataset
    later finds a cache already waiting rather than needing a priming run.

    Reads the main output plus every extra trial export, so a slice holds ALL
    trial types -- the splice routes them back the same way live rows are.
    """
    from fbpipe import freeze as _freeze

    frames = []
    for path in [output_csv, *(extra_export_paths or [])]:
        p = Path(str(path))
        if not p.is_file():
            continue
        try:
            frames.append(pd.read_csv(p))
        except Exception as exc:
            print(f"[FREEZE] Skipping cache read of {p}: {exc}")
    if not frames:
        return
    allrows = pd.concat(frames, ignore_index=True)
    if "dataset" not in allrows.columns:
        return
    for dataset, group in allrows.groupby(allrows["dataset"].astype(str)):
        try:
            _freeze.save_slice(
                settings.cache_dir,
                wide_block,
                dataset,
                group.reset_index(drop=True),
                fingerprint_for(dataset),
            )
        except Exception as exc:  # a cache write must never fail the run
            print(f"[FREEZE] Could not cache {dataset}/{wide_block}: {exc}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_freeze_pipeline.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Import the threshold constant**

`_freeze_fingerprint` must record the SAME `low_max_threshold_px` that
`build_wide_csv` actually used. The `combined_base` / `distance_base` call site
does not pass that argument, so it takes the default,
`LOW_MAX_FLAG_THRESHOLD_PX` — which `run_workflows.py` does not currently import.

Add it to the existing import block at `run_workflows.py:67-75`:

```python
from scripts.analysis.envelope_combined import (
    CombineConfig,
    LOW_MAX_FLAG_THRESHOLD_PX,
    build_wide_csv,
    combine_distance_angle,
    mirror_directory,
    overlay_sources,
    secure_copy_and_cleanup,
    wide_to_matrix,
)
```

If the fingerprint records a different value than the build used, every run sees
drift and re-derives — freeze would silently never engage.

- [ ] **Step 6: Wire the `combined_base` / `distance_base` call site**

`_process_base_block(combined_base_cfg, *, label, default_measure_cols)` is
defined at `:1159` and called at `:1323` from the loop at `:1318-1322` over
`("combined_base", ["combined_pct", "combined_base"])` and
`("distance_base", ["distance_percentage"])`. So **`label` is already exactly the
block key** — use it directly.

Replace the call at `:1252-1264`:

```python
            print(f"[analysis] {label}.wide → {output_csv}")
            build_wide_csv(
                roots,
                str(output_csv),
                measure_cols=base_measure_cols,
                fps_fallback=base_fps_fallback,
                exclude_roots=base_exclude_cfg,
                distance_limits=limits,
                trial_type_filter=trial_type_filter,
                extra_trial_exports=extra_exports or None,
                non_reactive_threshold=non_reactive_threshold,
                use_per_trial_baseline=base_use_per_trial_baseline,
            )
```

with:

```python
            _block = str(label)
            _fp_kw = dict(
                measure_cols=base_measure_cols,
                fps_fallback=base_fps_fallback,
                distance_limits=limits,
                non_reactive_threshold=non_reactive_threshold,
                # build_wide_csv is not passed low_max_threshold_px at this call
                # site, so it uses the default. The fingerprint must record the
                # same value or every run reports drift.
                low_max_threshold_px=LOW_MAX_FLAG_THRESHOLD_PX,
                use_per_trial_baseline=base_use_per_trial_baseline,
            )
            _frozen = _resolve_frozen_slices(
                settings,
                roots,
                _block,
                thawed=getattr(settings, "_thawed", ()),
                thaw_all=getattr(settings, "_thaw_all", False),
                **_fp_kw,
            )
            print(f"[analysis] {label}.wide → {output_csv}")
            build_wide_csv(
                roots,
                str(output_csv),
                measure_cols=base_measure_cols,
                fps_fallback=base_fps_fallback,
                exclude_roots=base_exclude_cfg,
                distance_limits=limits,
                trial_type_filter=trial_type_filter,
                extra_trial_exports=extra_exports or None,
                non_reactive_threshold=non_reactive_threshold,
                use_per_trial_baseline=base_use_per_trial_baseline,
                frozen_slices=_frozen or None,
            )
            _write_freeze_cache(
                settings,
                _block,
                str(output_csv),
                list(extra_exports.values()),
                fingerprint_for=lambda ds: _freeze_fingerprint(settings, ds, **_fp_kw),
            )
```

`settings` is in scope here — `_process_base_block` is a closure inside
`_run_combined` and already reads `settings.flagged_secured_root` at `:1185`.

- [ ] **Step 7: Wire the `wide` call site (`:1135`)**

Same shape, with `_block = "wide"`, `wide_root_paths` as roots,
`wide_measure_cols`, `wide_cfg.output_csv`, and that block's extra exports.

Note `config/config_new.yaml` has no `analysis.combined.wide` block (only
`combined_base.wide`), so this site does not fire under the v2 config. Wire it
anyway for `config/config.yaml`, which does use it.

- [ ] **Step 8: Wire the `pair_groups` call site (`:1403`)**

Same shape, with `_block = "pair_groups"` and `resolved_roots`.

This call site references `wide_measure_cols`, which is bound only inside the
`if wide_cfg:` block at `:1097-1103`. Under `config_new.yaml` it would therefore
`NameError` if `pair_groups` were configured. It is not configured, so the site is
unreachable. **Do not fix that pre-existing latent bug here** — wire freeze in the
same shape as the others and leave it untouched and out of scope.

- [ ] **Step 9: Run the full suite**

Run: `python -m pytest tests/ -q`
Expected: PASS, ≥538 passed

- [ ] **Step 10: Commit**

```bash
git add scripts/pipeline/run_workflows.py tests/test_freeze_pipeline.py
git commit -m "feat(run_workflows): resolve frozen slices and write the freeze cache after live derivations"
```

---

### Task 5: `--thaw` / `--thaw-all` CLI

**Files:**
- Modify: `scripts/pipeline/run_workflows.py:1908-1934` (argparse), and `main()` where settings are loaded
- Test: `tests/test_freeze_cli.py` (create)

**Interfaces:**
- Consumes: `freeze.freeze_flags` (Task 2), `_resolve_frozen_slices` (Task 4)
- Produces: `_build_arg_parser() -> argparse.ArgumentParser`; `settings._thawed: tuple[str, ...]`, `settings._thaw_all: bool`

Extracting `_build_arg_parser` is required: `main()`'s argparse is currently
inline and untested, so there is no way to assert on flag parsing without it.

- [ ] **Step 1: Write the failing test**

Create `tests/test_freeze_cli.py`:

```python
"""--thaw / --thaw-all parsing and validation."""

import pytest

import scripts.pipeline.run_workflows as rw


def test_thaw_is_repeatable():
    p = rw._build_arg_parser()
    a = p.parse_args(["--thaw", "EB-Control-24-1", "--thaw", "Hex-Control-24-0.1"])
    assert a.thaw == ["EB-Control-24-1", "Hex-Control-24-0.1"]


def test_thaw_defaults_empty():
    a = rw._build_arg_parser().parse_args([])
    assert not a.thaw
    assert a.thaw_all is False


def test_thaw_all_is_a_flag():
    a = rw._build_arg_parser().parse_args(["--thaw-all"])
    assert a.thaw_all is True


def test_existing_flags_survive():
    """The parser extraction must not drop any existing flag."""
    a = rw._build_arg_parser().parse_args(
        ["--config", "c.yaml", "--folder", "F", "--figures-only", "--svg"]
    )
    assert a.config == "c.yaml"
    assert a.folder == "F"
    assert a.figures_only is True
    assert a.svg is True


def test_unknown_thaw_name_raises_listing_valid_names():
    """A silently ignored typo looks exactly like a successful thaw."""
    with pytest.raises(SystemExit):
        rw._validate_thaw(["Nope-24-1"], datasets=("EB-Control-24-1",))


def test_known_thaw_name_passes():
    rw._validate_thaw(["EB-Control-24-1"], datasets=("EB-Control-24-1",))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_freeze_cli.py -v`
Expected: FAIL with `AttributeError: ... has no attribute '_build_arg_parser'`

- [ ] **Step 3: Extract the parser and add the flags**

In `run_workflows.py`, move the existing argparse setup out of `main()` into:

```python
def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(...)          # keep the existing description
    # ... keep --config, --folder, --figures-only, --svg EXACTLY as they are ...
    parser.add_argument(
        "--thaw",
        action="append",
        default=[],
        metavar="DATASET",
        help=(
            "Ignore freeze for DATASET this run (repeatable). Re-derives and "
            "re-caches it. Does not edit config."
        ),
    )
    parser.add_argument(
        "--thaw-all",
        action="store_true",
        help="Ignore every dataset freeze this run.",
    )
    return parser
```

`main()` then calls `parser = _build_arg_parser()`.

- [ ] **Step 4: Add thaw validation**

```python
def _validate_thaw(thaw, *, datasets) -> None:
    """Exit on an unknown --thaw name rather than silently ignoring it.

    A typo'd name would otherwise look exactly like a successful thaw: the run
    proceeds, the dataset stays frozen, and nothing says so.
    """
    known = set(datasets or ())
    unknown = [t for t in (thaw or []) if t not in known]
    if unknown:
        raise SystemExit(
            f"[FREEZE] Unknown --thaw dataset(s): {', '.join(sorted(unknown))}\n"
            f"         Known datasets: {', '.join(sorted(known)) or '(none)'}"
        )
```

- [ ] **Step 5: Stash the thaw selection on settings**

In `main()`, after settings are loaded and before `_run_combined` is called:

```python
    _validate_thaw(args.thaw, datasets=getattr(settings, "datasets", ()))
    settings._thawed = tuple(args.thaw or ())
    settings._thaw_all = bool(args.thaw_all)
    if settings._thaw_all:
        print("[FREEZE] --thaw-all: every dataset freeze ignored this run.")
    elif settings._thawed:
        print(f"[FREEZE] Thawed this run: {', '.join(settings._thawed)}")
```

`Settings` is a plain `@dataclass` (`src/fbpipe/config.py:486`) — not
`frozen=True` and without `__slots__` — so direct attribute assignment works.
These are per-run CLI state, not config, which is why they are set here rather
than added as `Settings` fields.

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_freeze_cli.py -v`
Expected: PASS (6 passed)

- [ ] **Step 7: Verify the CLI still runs**

Run: `python scripts/pipeline/run_workflows.py --help`
Expected: exit 0; `--thaw` and `--thaw-all` listed alongside `--config`, `--folder`, `--figures-only`, `--svg`.

- [ ] **Step 8: Commit**

```bash
git add scripts/pipeline/run_workflows.py tests/test_freeze_cli.py
git commit -m "feat(cli): add --thaw / --thaw-all escape hatch for frozen datasets"
```

---

### Task 6: Figure freeze

**Files:**
- Modify: `scripts/analysis/envelope_visuals.py:303-331` (`resolve_dataset_output_dir`), and the figure emit sites
- Test: `tests/test_freeze_figures.py` (create)

**Interfaces:**
- Consumes: `freeze.freeze_flags` (Task 2)
- Produces: `should_skip_frozen_figure(cfg, datasets, *, thawed=(), thaw_all=False) -> bool`

- [ ] **Step 1: Write the failing test**

Create `tests/test_freeze_figures.py`:

```python
"""A figure is skipped only when EVERY contributing dataset is frozen."""

import scripts.analysis.envelope_visuals as ev
from fbpipe.config import DatasetOverride


class _Cfg:
    def __init__(self, overrides):
        self.dataset_overrides = overrides


F = DatasetOverride(freeze_figures=True)
U = DatasetOverride(freeze_figures=False)


def test_single_frozen_dataset_figure_is_skipped():
    assert ev.should_skip_frozen_figure(_Cfg({"A": F}), ["A"]) is True


def test_single_live_dataset_figure_is_drawn():
    assert ev.should_skip_frozen_figure(_Cfg({"A": U}), ["A"]) is False


def test_all_frozen_aggregate_is_skipped():
    assert ev.should_skip_frozen_figure(_Cfg({"A": F, "B": F}), ["A", "B"]) is True


def test_mixed_frozen_and_live_is_DRAWN():
    """The correctness case: frozen Control beside live Training must redraw, or
    adding flies to Training silently fails to appear."""
    assert ev.should_skip_frozen_figure(_Cfg({"A": F, "B": U}), ["A", "B"]) is False


def test_unknown_dataset_counts_as_live():
    assert ev.should_skip_frozen_figure(_Cfg({"A": F}), ["A", "UNKNOWN"]) is False


def test_empty_dataset_set_is_drawn():
    """Never skip on an empty contributor set -- that is 'unknown', not 'all frozen'.
    all([]) is True, which would silently skip every such figure."""
    assert ev.should_skip_frozen_figure(_Cfg({"A": F}), []) is False


def test_freeze_data_alone_does_not_skip_figures():
    """The flags are independent: data:true + figures:false still draws."""
    cfg = _Cfg({"A": DatasetOverride(freeze_data=True, freeze_figures=False)})
    assert ev.should_skip_frozen_figure(cfg, ["A"]) is False


def test_thaw_all_draws_everything():
    assert ev.should_skip_frozen_figure(_Cfg({"A": F}), ["A"], thaw_all=True) is False


def test_thaw_named_dataset_draws():
    assert ev.should_skip_frozen_figure(_Cfg({"A": F}), ["A"], thawed=["A"]) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_freeze_figures.py -v`
Expected: FAIL with `AttributeError: ... has no attribute 'should_skip_frozen_figure'`

- [ ] **Step 3: Implement the guard**

In `scripts/analysis/envelope_visuals.py`, next to `resolve_dataset_output_dir`
(~`:331`):

```python
def should_skip_frozen_figure(cfg, datasets, *, thawed=(), thaw_all=False) -> bool:
    """True when EVERY dataset contributing to a figure is frozen for figures.

    Deliberately NOT "any contributor is frozen". A figure drawn from a frozen
    Control and a live Training must still redraw, or adding flies to Training
    would silently fail to appear in it.

    An empty or unknown contributor set counts as LIVE. Note all([]) is True, so
    an empty set would otherwise skip every such figure -- an empty set means
    "we do not know", not "all frozen".
    """
    from fbpipe.freeze import freeze_flags

    names = [str(d) for d in (datasets or []) if str(d).strip()]
    if not names:
        return False
    for name in names:
        _, freeze_figures = freeze_flags(cfg, name, thawed=thawed, thaw_all=thaw_all)
        if not freeze_figures:
            return False
    return True
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_freeze_figures.py -v`
Expected: PASS (9 passed)

- [ ] **Step 5: Apply the guard at the figure emit sites**

Add an early return at each site, using the dataset set that site already
resolves. Do NOT reuse `should_write` (`:334-343`) — it force-returns `True` for
any `reaction_matrix` / `reaction_prediction` path regardless of `overwrite`, so
folding freeze into it would be ignored for exactly those figures.

- `generate_envelope_plots` (`:2024`, savefig `:2677`) — guard per fly, on that fly's dataset.
- `generate_reaction_matrices` (`:1410`, savefig `:1753`) — guard on the contributing set.
- `plot_reaction_rate_bars` (`:1242`) — guard on the contributing set.
- `scripts/analysis/score_summary.py`: `_plot_bar_charts` (`:565`), `_plot_training_vs_control_bars` (`:728`), `_plot_heatmap` (`:859`), `_plot_score_pair` (`:928`).

Each guard logs one line so a skip is visible:

```python
        if should_skip_frozen_figure(cfg, contributing_datasets,
                                     thawed=thawed, thaw_all=thaw_all):
            print(f"[FROZEN] Skipping figure (all contributors frozen): {out_path}")
            return
```

`score_summary.py` runs as a SUBPROCESS (`run_workflows.py:1807-1824`), so it
does not share `settings`. It must load config itself via its existing
`--config` argument and read `dataset_overrides` from there.

- [ ] **Step 6: Honor freeze under `--figures-only`**

`--figures-only` force-sets every figure step to `True` (`:2024-2046`). Do NOT
let that bypass `freeze.figures` — it is the most common way figures are run, so
bypassing there would make the feature nearly inert. `--thaw` remains the way to
force a redraw.

Add to `tests/test_freeze_figures.py`:

```python
def test_figures_only_does_not_bypass_freeze():
    """--figures-only forces figure steps on; it must not un-freeze them."""
    cfg = _Cfg({"A": F})
    assert ev.should_skip_frozen_figure(cfg, ["A"]) is True
```

- [ ] **Step 7: Run the full suite**

Run: `python -m pytest tests/ -q`
Expected: PASS, ≥538 passed

- [ ] **Step 8: Commit**

```bash
git add scripts/analysis/envelope_visuals.py scripts/analysis/score_summary.py tests/test_freeze_figures.py
git commit -m "feat(figures): skip a figure only when every contributing dataset is frozen"
```

---

### Task 7: Frozen dataset with a missing folder is a hard error

**Files:**
- Modify: `src/fbpipe/config.py:114-152` (`_expand_datasets`)
- Test: `tests/test_freeze_missing_folder.py` (create)

**Interfaces:**
- Consumes: `DatasetOverride.freeze_data` (Task 1)
- Produces: `_expand_datasets` raises `RuntimeError` for a frozen dataset absent from disk

`_expand_datasets` silently drops datasets not on disk (`:141-152`). For a frozen
dataset that is a mistake, not a workflow: the decision is that raw data stays on
disk, and we cannot auto-rebuild what is not there. Unfrozen datasets keep the
existing silent-skip semantics.

- [ ] **Step 1: Write the failing test**

Create `tests/test_freeze_missing_folder.py`:

```python
"""A frozen dataset that is not on disk must fail loudly, not vanish."""

import textwrap

import pytest

from fbpipe.config import load_raw_config


def _cfg(tmp_path, body):
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "secured").mkdir(exist_ok=True)
    p = tmp_path / "cfg.yaml"
    p.write_text(
        textwrap.dedent(
            f"""
            protocol: v2
            dataset_bases:
              data: {tmp_path}/data
              secured: {tmp_path}/secured
            {body}
            """
        )
    )
    return p


def test_frozen_dataset_missing_from_disk_raises(tmp_path):
    cfg = _cfg(
        tmp_path,
        """
            datasets:
              - GONE-24-1
            dataset_overrides:
              GONE-24-1:
                freeze:
                  data: true
        """,
    )
    with pytest.raises(RuntimeError, match="GONE-24-1"):
        load_raw_config(str(cfg))


def test_unfrozen_dataset_missing_from_disk_still_skips_silently(tmp_path):
    """Pre-existing behavior must not change for unfrozen datasets."""
    cfg = _cfg(
        tmp_path,
        """
            datasets:
              - GONE-24-1
        """,
    )
    data = load_raw_config(str(cfg))
    assert data["main_directories"] == []


def test_frozen_dataset_present_on_disk_is_fine(tmp_path):
    (tmp_path / "data" / "HERE-24-1").mkdir(parents=True)
    cfg = _cfg(
        tmp_path,
        """
            datasets:
              - HERE-24-1
            dataset_overrides:
              HERE-24-1:
                freeze:
                  data: true
        """,
    )
    data = load_raw_config(str(cfg))
    assert any("HERE-24-1" in d for d in data["main_directories"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_freeze_missing_folder.py -v`
Expected: FAIL — `test_frozen_dataset_missing_from_disk_raises` does not raise.

- [ ] **Step 3: Raise for frozen datasets missing from disk**

In `src/fbpipe/config.py`, in `_expand_datasets`, replace the `skipped` block at
`:146-149`:

```python
    skipped = [ds for ds in datasets
               if not Path(data_base, ds).is_dir() and not Path(secured_base, ds).is_dir()]
    if skipped:
        print(f"[config] Skipping datasets not yet on disk: {', '.join(skipped)}")
```

with:

```python
    skipped = [ds for ds in datasets
               if not Path(data_base, ds).is_dir() and not Path(secured_base, ds).is_dir()]
    # A frozen dataset that is not on disk cannot be silently dropped: freeze
    # assumes the raw data stays put, and we cannot auto-rebuild what is absent.
    # Dropping it would quietly delete its rows from the wide CSV.
    raw_overrides = data.get("dataset_overrides") or {}
    frozen_missing = [
        ds for ds in skipped
        if isinstance(raw_overrides.get(ds), dict)
        and (raw_overrides[ds].get("freeze") or {}).get("data", False)
    ]
    if frozen_missing:
        raise RuntimeError(
            f"[config] Frozen dataset(s) not found on disk: {', '.join(sorted(frozen_missing))}\n"
            f"         freeze.data assumes the raw data stays on disk. Restore the "
            f"folder(s), or remove freeze.data from the dataset_overrides block."
        )
    if skipped:
        print(f"[config] Skipping datasets not yet on disk: {', '.join(skipped)}")
```

This reads the RAW yaml block, not a parsed `DatasetOverride`, because
`_expand_datasets` runs inside `load_raw_config` before `Settings` exists.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_freeze_missing_folder.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest tests/ -q`
Expected: PASS, ≥538 passed

- [ ] **Step 6: Commit**

```bash
git add src/fbpipe/config.py tests/test_freeze_missing_folder.py
git commit -m "feat(config): frozen dataset missing from disk is a hard error, not a silent drop"
```

---

### Task 8: End-to-end verification and legacy regression

**Files:**
- Test: `tests/test_freeze_e2e.py` (create)

**Interfaces:**
- Consumes: everything from Tasks 1-7

- [ ] **Step 1: Write the end-to-end test**

Create `tests/test_freeze_e2e.py`:

```python
"""Freeze end-to-end: derive live, freeze, re-run, output is unchanged."""

import numpy as np
import pandas as pd
import pytest

import scripts.analysis.envelope_combined as ec
import scripts.analysis.envelope_visuals as ev
from fbpipe import freeze
from fbpipe.config import DatasetOverride


def _make(root, n, fly="october_01_fly1"):
    out = root / fly / "angle_distance_rms_envelope"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"envelope_of_rms": np.linspace(0, 100, n)}).to_csv(
        out / f"{fly}_testing_1_angle_distance_rms_envelope.csv", index=False
    )
    return root


@pytest.fixture(autouse=True)
def _v2():
    ev.set_protocol("v2")


def test_freeze_then_rerun_is_byte_identical(tmp_path):
    """The user-facing promise, exercised through the real cache.

    FROZEN is deliberately longer than LIVE so the own_max_len fold is genuinely
    exercised rather than correct by accident.
    """
    live = _make(tmp_path / "LIVE", 9)
    frozen = _make(tmp_path / "FROZEN", 21)
    cache = tmp_path / "cache"

    fp_kw = dict(
        protocol="v2",
        measure_cols=["envelope_of_rms"],
        fps_fallback=40.0,
        distance_limits=None,
        non_reactive_threshold=None,
        low_max_threshold_px=ec.LOW_MAX_FLAG_THRESHOLD_PX,
        use_per_trial_baseline=False,
        override=DatasetOverride(),
    )
    fp = freeze.build_fingerprint(**fp_kw)

    # Pass 1: fully live.
    first = tmp_path / "first.csv"
    ec.build_wide_csv(
        [str(live), str(frozen)], str(first), measure_cols=["envelope_of_rms"]
    )
    baseline = pd.read_csv(first)

    # Cache FROZEN's slice, as the pipeline does after a live derivation.
    freeze.save_slice(
        cache, "wide", "FROZEN",
        baseline[baseline["dataset"] == "FROZEN"].reset_index(drop=True), fp,
    )

    # Pass 2: FROZEN spliced from cache.
    got = freeze.load_slice(cache, "wide", "FROZEN", fp)
    assert got is not None and got.own_max_len == 21

    second = tmp_path / "second.csv"
    ec.build_wide_csv(
        [str(live), str(frozen)], str(second),
        measure_cols=["envelope_of_rms"],
        frozen_slices={"FROZEN": (got.rows, got.own_max_len)},
    )

    pd.testing.assert_frame_equal(
        baseline.sort_values(["dataset", "fly"]).reset_index(drop=True),
        pd.read_csv(second).sort_values(["dataset", "fly"]).reset_index(drop=True),
    )


def test_deleting_the_cache_is_safe(tmp_path):
    """The cache is a derived artifact: deleting it must only cost time."""
    import shutil

    live = _make(tmp_path / "LIVE", 9)
    frozen = _make(tmp_path / "FROZEN", 21)
    cache = tmp_path / "cache"
    fp = freeze.build_fingerprint(
        protocol="v2", measure_cols=["envelope_of_rms"], fps_fallback=40.0,
        distance_limits=None, non_reactive_threshold=None,
        low_max_threshold_px=ec.LOW_MAX_FLAG_THRESHOLD_PX,
        use_per_trial_baseline=False, override=DatasetOverride(),
    )
    out1 = tmp_path / "a.csv"
    ec.build_wide_csv([str(live), str(frozen)], str(out1), measure_cols=["envelope_of_rms"])
    base = pd.read_csv(out1)
    freeze.save_slice(
        cache, "wide", "FROZEN", base[base["dataset"] == "FROZEN"].reset_index(drop=True), fp
    )
    shutil.rmtree(cache)
    assert freeze.load_slice(cache, "wide", "FROZEN", fp) is None

    out2 = tmp_path / "b.csv"
    ec.build_wide_csv([str(live), str(frozen)], str(out2), measure_cols=["envelope_of_rms"])
    pd.testing.assert_frame_equal(base, pd.read_csv(out2))
```

- [ ] **Step 2: Run the test**

Run: `python -m pytest tests/test_freeze_e2e.py -v`
Expected: PASS (2 passed)

- [ ] **Step 3: Run the full suite**

Run: `python -m pytest tests/ -q`
Expected: PASS, ≥538 passed plus the new freeze tests.

- [ ] **Step 4: Verify legacy is byte-for-byte identical**

Legacy output must not move. Create a worktree at the branch point and compare
against the SAME data:

```bash
git worktree add /tmp/freeze-baseline 47d6fe3
```

Run the legacy protocol suite in both trees and diff the emitted wide CSVs.

Do NOT verify this with `git stash` — a no-op diff produces a vacuous pass. Do
NOT compare against a stale baseline — the pipeline regenerates
`model_predictions.csv`.

Run: `python -m pytest tests/test_protocol_legacy_regression.py -v`
Expected: PASS

```bash
git worktree remove /tmp/freeze-baseline
```

- [ ] **Step 5: Verify the real pipeline runs**

Run: `python scripts/pipeline/run_workflows.py --config config/config_new.yaml --figures-only`
Expected: exit 0, no freeze warnings (no dataset is frozen yet, so behavior is unchanged).

Then, on a dataset that has been derived at least once, add to
`config/config_new.yaml`:

```yaml
dataset_overrides:
  EB-Control-24-1:
    freeze:
      data: true
      figures: true
```

Re-run and confirm the log shows `[FROZEN] Not walking root ...` for that dataset
and that its rows are still present in the wide CSV.

- [ ] **Step 6: Commit**

```bash
git add tests/test_freeze_e2e.py
git commit -m "test(freeze): end-to-end round-trip and cache-deletion safety"
```

---

## Notes for the implementer

**The one thing that will silently break this:** `build_wide_csv` truncates rows
longer than the global `max_len` (`:3283-3284`). If a frozen dataset's
`own_max_len` is not folded into that global max (Task 3, Step 6), its cached
rows are chopped with no error. Every round-trip test therefore uses a frozen
dataset LONGER than the live one — with equal lengths the tests pass even with
the fold deleted. If you find yourself simplifying a fixture to equal lengths,
stop: you are deleting the only thing the test proves.

**This branch's recorded failure modes** — check your own tests against them:
- An assertion that reads its expectation from the constant it validates.
- An assertion over an empty collection (empty for the wrong reason).
- An assertion on a quantity that is equal by construction.
- Place mutants INSIDE the unit under test and re-run. Do not trust a passing
  report you have not tried to break.

**Out of scope; do not fix here:**
- `wide_measure_cols` is bound only inside `if wide_cfg:` (`run_workflows.py:1097-1103`), so the `pair_groups` call site (`:1403`) would `NameError` if configured. Latent, pre-existing, currently unreachable.
- `_style_trained_xticks` (`envelope_visuals.py:1309`) compares odor names against a dataset display label, so sibling figures disagree about the trained odor. Pre-existing.
- `docs/` and `config/` are gitignored (`.gitignore:116`), so `config/config_new.yaml` — and any `freeze:` block in it — is not tracked. Flagged in the spec.
