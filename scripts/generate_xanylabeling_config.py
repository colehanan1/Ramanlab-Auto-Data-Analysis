#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.convert import generate_xanylabeling_config as _impl
from scripts.convert.generate_xanylabeling_config import *  # noqa: F401,F403

if __name__ == "__main__":
    main = getattr(_impl, "main", None)
    if callable(main):
        raise SystemExit(main())
