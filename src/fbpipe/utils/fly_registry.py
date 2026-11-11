"""Dataset-wide fly numbers that never change.

Today a fly is ``<batch folder> fly<slot>``, and slots are numbered top to
bottom inside each batch, so a re-run that finds a fly above fly 1 (as
november_07_batch_1 did on brightened video) renumbers every fly below it.

The registry gives each fly of a dataset one number:

* the first time, by recording time -- **Fly 1 is the earliest fly** (its
  batch's first recording), then top-to-bottom within the batch;
* every fly seen later -- a new batch, or a fly a re-run found in an old
  batch -- **goes on top** with the next free number. No existing fly is
  ever renumbered, so a number in a figure from last month still means the
  same fly;
* a **flagged fly keeps its number**: ``flagged`` is a column, never a reason
  to renumber;
* a fly that is no longer found keeps its number, marked ``present=False``;
  numbers are never reused.

Inside a batch a fly is recognised by its eye position (nearest within
``match_px``), not its slot number.

Pure functions; the caller reads and writes the CSV.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

COLUMNS = ["dataset_fly", "batch", "fly_number", "eye_x", "eye_y", "first_recorded", "present", "flagged"]
MATCH_PX = 60.0


def label(dataset_fly: int) -> str:
    return f"Fly {int(dataset_fly)}"


def _match(old: pd.DataFrame, new: pd.DataFrame, max_px: float) -> dict:
    """Greedy nearest-eye matching: {new index: old index} within ``max_px``."""
    pairs = []
    for i, n in new.iterrows():
        for j, o in old.iterrows():
            d = float(np.hypot(n.eye_x - o.eye_x, n.eye_y - o.eye_y))
            if d <= max_px:
                pairs.append((d, i, j))
    out, used_new, used_old = {}, set(), set()
    for _, i, j in sorted(pairs):
        if i not in used_new and j not in used_old:
            out[i] = j
            used_new.add(i)
            used_old.add(j)
    return out


def update_registry(
    flies: pd.DataFrame, existing: Optional[pd.DataFrame], match_px: float = MATCH_PX
) -> pd.DataFrame:
    """The registry after seeing ``flies`` (one row per fly found now).

    ``flies`` needs batch, fly_number, eye_x, eye_y, first_recorded
    (sortable), and optionally flagged. ``existing`` is the previous registry
    or None. Returns the full registry, one row per number ever given.
    """
    flies = flies.copy()
    if "flagged" not in flies.columns:
        flies["flagged"] = False
    flies["flagged"] = flies["flagged"].fillna(False).astype(bool)
    reg = (existing.copy() if existing is not None and len(existing)
           else pd.DataFrame(columns=COLUMNS))
    reg["present"] = False
    matched_new: set = set()
    for batch, now in flies.groupby("batch", sort=False):
        before = reg[reg["batch"] == batch]
        for i, j in _match(before, now, match_px).items():
            reg.loc[j, ["fly_number", "eye_x", "eye_y", "first_recorded", "flagged"]] = (
                now.loc[i, ["fly_number", "eye_x", "eye_y", "first_recorded", "flagged"]].to_numpy()
            )
            reg.loc[j, "present"] = True
            matched_new.add(i)
    new = flies.loc[[i for i in flies.index if i not in matched_new]]
    new = new.sort_values(["first_recorded", "batch", "fly_number"], kind="stable")
    start = int(pd.to_numeric(reg["dataset_fly"]).max()) + 1 if len(reg) else 1
    added = pd.DataFrame({
        "dataset_fly": np.arange(start, start + len(new)),
        "batch": new["batch"].to_numpy(),
        "fly_number": new["fly_number"].to_numpy(),
        "eye_x": new["eye_x"].to_numpy(),
        "eye_y": new["eye_y"].to_numpy(),
        "first_recorded": new["first_recorded"].to_numpy(),
        "present": True,
        "flagged": new["flagged"].to_numpy(),
    })
    out = pd.concat([reg[COLUMNS], added[COLUMNS]], ignore_index=True) if len(reg) else added[COLUMNS]
    out = out.astype({"dataset_fly": int, "fly_number": int, "eye_x": float, "eye_y": float,
                      "present": bool, "flagged": bool})
    for col in ("batch", "first_recorded"):
        out[col] = out[col].astype(str).astype(object)
    return out.sort_values("dataset_fly", kind="stable").reset_index(drop=True)
