#!/usr/bin/env python3
"""Run the abdomen geometry pipeline (adjacent to run_workflows.py / config_new.yaml).

    conda run -n yolo-env python scripts/pipeline/run_abdomen_workflows.py --config config/config_abdomen.yaml
    ... --only-trial pretest_1 --max-frames 200 track      # smoke test on one trial
    ... --force                                            # re-track everything

The model is a YOLO26-OBB checkpoint: the base python3 (ultralytics 8.3) cannot load it.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (str(REPO_ROOT / "src"), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from fbpipe.geometry_pipeline import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
