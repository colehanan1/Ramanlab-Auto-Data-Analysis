"""Resolve which experiment folders inside a dataset are frozen.

The whole-dataset ``freeze:`` block is all-or-nothing, but a dataset can outlive
its own rig plumbing: ``Hex-Training-24-0.1``'s may/june batches ran a different
odor panel than its august ones. Folder freeze retires the old batches without
retiring the dataset.

Two sources, unioned:

* ``freeze_folders_before: 2026-06-26`` (top level) — freeze every batch
  recorded strictly BEFORE that date. The date comes from the batch itself via
  :func:`fbpipe.utils.rig_gates.read_batch_date` (sidecar filename stamp, then
  labeled metadata lines) — never from the folder name, which carries no year.
* ``dataset_overrides.<ds>.freeze.folders:`` — :class:`FolderRule` entries
  scoped to that one dataset. Each is an exact folder name, an fnmatch glob
  (``*_rig_3``), or a glob bounded by recording date
  (``{match: "*batch_2*", before: 2026-07-27}``).

A frozen folder is skipped by the heavy per-trial steps and excluded from every
figure, but its already-derived rows still land in the wide CSV marked
``frozen`` — the CSV stays a complete record, the plots do not.

Two deliberate asymmetries:

* **A data-frozen dataset is never folder-frozen.** ``freeze.data`` means its
  root is never walked and its rows splice in whole from the cache; folder
  freezing one of its batches would silently delete rows from that slice.
* **An undated batch stays live**, with a printed warning. The date rule can
  only ever be reached by a positive match, so a failed lookup keeps data
  rather than dropping it. Same fail-open stance as ``rig_gates``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from fnmatch import fnmatchcase
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Optional

from .rig_gates import read_batch_date

if TYPE_CHECKING:  # pandas is imported lazily; the filters are the only users
    import pandas as pd

__all__ = [
    "FROZEN_COLUMN",
    "FolderRule",
    "drop_frozen",
    "folder_freeze_rules",
    "frozen_folders_for_root",
    "frozen_mask",
    "is_frozen_folder",
    "iter_live_batch_dirs",
    "warn_if_unmarked",
]

#: Column ``build_wide_csv`` writes to mark a row's experiment folder as frozen.
FROZEN_COLUMN = "frozen"


@dataclass(frozen=True)
class FolderRule:
    """One entry of a dataset's ``freeze.folders`` list.

    ``pattern`` is an fnmatch glob (``*_rig_3``); a wildcard-free pattern is an
    exact folder name. ``before`` optionally bounds the rule to folders recorded
    strictly before that date, so "every batch 2 before july 27" is one line
    instead of fifteen folder names that go stale.

    A bounded rule needs the folder's recording date, so an undated folder never
    matches it -- same fail-open stance as the global cutoff. An UNBOUNDED rule
    needs no date and matches on the name alone.
    """

    pattern: str
    before: Optional[date] = None

    def matches(self, folder_name: str, batch_date: Optional[date]) -> bool:
        if not fnmatchcase(folder_name, self.pattern):
            return False
        if self.before is None:
            return True
        return batch_date is not None and batch_date < self.before

    @property
    def needs_date(self) -> bool:
        return self.before is not None

# Batches already reported as undated, so a run with 100+ batch dirs prints each
# one once instead of once per step. Keyed by resolved path.
_warned_undated: set[str] = set()


def _thawed(dataset: str, thawed: Iterable[str], thaw_all: bool) -> bool:
    return bool(thaw_all) or dataset in {str(t) for t in (thawed or ())}


def folder_freeze_rules(
    cfg: Any,
    dataset: str,
    *,
    thawed: Iterable[str] = (),
    thaw_all: bool = False,
) -> tuple[Optional[date], tuple[FolderRule, ...]]:
    """The ``(cutoff, rules)`` in force for *dataset*.

    ``(None, ())`` means nothing is folder-frozen here — because the dataset is
    thawed, is frozen for DATA, or simply has no rules. Callers use that as a
    cheap "skip the whole check" signal.

    Recorded in the freeze fingerprint, so changing either input rebuilds a
    dataset's cached slice instead of serving rows filtered under the old rules.
    """
    if _thawed(dataset, thawed, thaw_all):
        return (None, ())

    override = (getattr(cfg, "dataset_overrides", None) or {}).get(dataset)
    # "Live datasets only": a data-frozen dataset's root is never walked, so
    # there are no folders to judge and its cached rows must splice in whole.
    if override is not None and bool(getattr(override, "freeze_data", False)):
        return (None, ())

    cutoff = getattr(cfg, "freeze_folders_before", None)
    rules: list[FolderRule] = []
    for entry in getattr(override, "freeze_folders", ()) or ():
        if isinstance(entry, FolderRule):
            rules.append(entry)
        elif isinstance(entry, str):
            # Convenience for hand-built Settings; config parsing already
            # produces FolderRule.
            if entry.strip():
                rules.append(FolderRule(entry.strip()))
        else:
            # Never str()-coerce: a mapping would stringify to a pattern that
            # matches nothing, silently freezing nothing at all.
            raise TypeError(
                f"dataset_overrides.{dataset}.freeze_folders entries must be "
                f"FolderRule or str, got {type(entry).__name__}: {entry!r}"
            )
    return (cutoff, tuple(rules))


def is_frozen_folder(
    cfg: Any,
    dataset: str,
    batch_dir: str | Path,
    *,
    thawed: Iterable[str] = (),
    thaw_all: bool = False,
) -> bool:
    """True when this experiment folder is frozen out of the figures."""
    cutoff, rules = folder_freeze_rules(
        cfg, dataset, thawed=thawed, thaw_all=thaw_all
    )
    if cutoff is None and not rules:
        return False

    path = Path(batch_dir)
    # Name-only rules first: a direct instruction, and they cost no I/O.
    unbounded = [r for r in rules if not r.needs_date]
    if any(r.matches(path.name, None) for r in unbounded):
        return True

    bounded = [r for r in rules if r.needs_date]
    if cutoff is None and not bounded:
        return False

    batch_date = read_batch_date(path)
    if batch_date is None:
        key = str(path)
        if key not in _warned_undated:
            _warned_undated.add(key)
            print(
                f"[FREEZE] Cannot date batch {path.name} ({path}); keeping it LIVE. "
                f"Add it to dataset_overrides.{dataset}.freeze.folders to freeze it."
            )
        return False
    if cutoff is not None and batch_date < cutoff:
        return True
    return any(r.matches(path.name, batch_date) for r in bounded)


def iter_live_batch_dirs(
    cfg: Any,
    root: str | Path,
    *,
    dataset: str | None = None,
    thawed: Iterable[str] = (),
    thaw_all: bool = False,
) -> list[Path]:
    """The batch directories under *root* the heavy steps should process.

    Drop-in for the ``sorted(p for p in root.iterdir() if p.is_dir())`` every
    step open-codes, minus the frozen folders. Sorted, because yolo_infer shards
    videos across workers by enumeration order.

    A skipped folder keeps whatever per-trial CSVs it already has -- that is
    what lets build_wide_csv still emit its rows for the wide CSV.
    """
    root_path = Path(root)
    if not root_path.is_dir():
        return []
    name = dataset if dataset is not None else root_path.name
    dirs = sorted((p for p in root_path.iterdir() if p.is_dir()), key=lambda p: p.name)
    cutoff, names = folder_freeze_rules(cfg, name, thawed=thawed, thaw_all=thaw_all)
    if cutoff is None and not names:
        return dirs

    live, skipped = [], []
    for d in dirs:
        if is_frozen_folder(cfg, name, d, thawed=thawed, thaw_all=thaw_all):
            skipped.append(d.name)
        else:
            live.append(d)
    if skipped:
        print(f"[FROZEN] {name}: skipping {len(skipped)} frozen folder(s): "
              + ", ".join(skipped[:4])
              + (f", +{len(skipped) - 4} more" if len(skipped) > 4 else ""))
    return live


def frozen_mask(df: "pd.DataFrame"):
    """Boolean Series: True where the row's experiment folder is frozen.

    Only an explicit truthy mark counts. NaN (a cached slice written before the
    column existed) and a missing column both read as LIVE -- the marker can
    only ever REMOVE a row, so an unreadable value keeps the data.

    CSV round-trips turn booleans into the strings ``"True"``/``"False"``, and
    ``bool("False")`` is True, so strings are matched by value rather than cast.
    """
    import pandas as pd

    if FROZEN_COLUMN not in df.columns:
        return pd.Series(False, index=df.index)
    raw = df[FROZEN_COLUMN]
    if raw.dtype == bool:
        return raw
    as_str = raw.astype(str).str.strip().str.lower()
    return as_str.isin({"true", "1", "1.0", "yes"})


def drop_frozen(df: "pd.DataFrame", *, include_frozen: bool = False) -> "pd.DataFrame":
    """*df* without its frozen rows (a copy; the caller's frame is untouched).

    This is the "in the CSV, not in the plots" cut. ``include_frozen=True``
    returns everything, for the rare consumer that wants the complete record.
    """
    if include_frozen or FROZEN_COLUMN not in df.columns:
        return df
    return df.loc[~frozen_mask(df)].copy()


def any_folder_rules(cfg: Any) -> bool:
    """Whether ANY folder freeze is configured, anywhere."""
    if getattr(cfg, "freeze_folders_before", None) is not None:
        return True
    return any(
        getattr(ov, "freeze_folders", ())
        for ov in (getattr(cfg, "dataset_overrides", None) or {}).values()
    )


def warn_if_unmarked(df: "pd.DataFrame", cfg: Any, *, source: str) -> bool:
    """Warn when folder rules exist but *df* carries no ``frozen`` column.

    ``--figures-only`` reuses whatever wide table is already on disk. One built
    before folder freeze existed marks nothing, so every retired fly silently
    returns to the figures — a wrong result that looks exactly like a right one.
    Returns True when the warning fired.
    """
    if FROZEN_COLUMN in df.columns or not any_folder_rules(cfg):
        return False
    print(
        f"[FREEZE] WARNING: {source} has no '{FROZEN_COLUMN}' column, but folder "
        f"freeze rules are configured. It predates folder freeze, so NOTHING is "
        f"being excluded. Re-run without --figures-only to rebuild the wide table."
    )
    return True


def frozen_folders_for_root(
    cfg: Any,
    root: str | Path,
    *,
    dataset: str | None = None,
    thawed: Iterable[str] = (),
    thaw_all: bool = False,
) -> set[str]:
    """Names of the frozen experiment folders directly under *root*.

    *dataset* defaults to the root's own directory name — roots are named for
    their dataset on both the live and the secured mirror.
    """
    root_path = Path(root)
    name = dataset if dataset is not None else root_path.name
    cutoff, names = folder_freeze_rules(cfg, name, thawed=thawed, thaw_all=thaw_all)
    if cutoff is None and not names:
        return set()
    if not root_path.is_dir():
        return set()
    return {
        d.name
        for d in sorted(p for p in root_path.iterdir() if p.is_dir())
        if is_frozen_folder(cfg, name, d, thawed=thawed, thaw_all=thaw_all)
    }
