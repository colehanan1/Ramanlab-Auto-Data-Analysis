"""Generate dataset-level mean plots of Distance % for training odors.

Thin wrapper around ``dataset_means.py`` that defaults ``--trial-type`` to
``training`` and uses the training-specific output directory from the config.

Example
-------
    python scripts/analysis/dataset_means_training.py \
        --config config/config.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
_root_str = str(ROOT)
if _root_str not in sys.path:
    sys.path.insert(0, _root_str)
_src_str = str(ROOT / "src")
if _src_str not in sys.path:
    sys.path.insert(0, _src_str)

from scripts.analysis.dataset_means import build_parser, main as _dataset_means_main


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Run dataset_means with ``--trial-type training`` by default."""

    if argv is None:
        argv = sys.argv[1:]

    argv = list(argv)

    # Inject --trial-type training unless the caller already specified one.
    has_trial_type = any(
        arg.startswith("--trial-type") for arg in argv
    )
    if not has_trial_type:
        argv.extend(["--trial-type", "training"])

    _dataset_means_main(argv)


if __name__ == "__main__":
    main()
