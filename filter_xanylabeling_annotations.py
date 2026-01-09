#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.curation import filter_xanylabeling_annotations as _impl
from scripts.curation.filter_xanylabeling_annotations import *  # noqa: F401,F403

if __name__ == "__main__":
    main = getattr(_impl, "main", None)
    if callable(main):
        raise SystemExit(main())
