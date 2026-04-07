#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis import reaction_matrix_from_spreadsheet as _impl
from scripts.analysis.reaction_matrix_from_spreadsheet import *  # noqa: F401,F403

_filter_trial_types = _impl._filter_trial_types


def _load_predictions(csv_path, *, threshold=None, flagged_flies_csv=""):
    if threshold is None:
        threshold = _impl.NON_REACTIVE_SPAN_PX
    return _impl._load_predictions(
        csv_path,
        threshold=threshold,
        flagged_flies_csv=flagged_flies_csv,
    )

if __name__ == "__main__":
    main = getattr(_impl, "main", None)
    if callable(main):
        raise SystemExit(main())
