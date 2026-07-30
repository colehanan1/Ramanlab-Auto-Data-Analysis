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

