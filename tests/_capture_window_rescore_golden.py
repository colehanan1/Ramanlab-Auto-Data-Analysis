"""Regenerate tests/golden/window_rescore_delay_free/golden.json.

Run ONLY when a change to the delay-free window-rescore path is intentional:
    python tests/_capture_window_rescore_golden.py

The golden was first captured from window_rescore.py as it stood BEFORE the
odor-delay support was added (2026-11-10), so test_window_rescore_delay.py
proves that delay-free models are rescored exactly as before: every window
bound, every raw feature value (to the last bit) and, where the delay-free
production model is on this machine, every score.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "src", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import _window_rescore_fixtures as fx  # noqa: E402
from scripts.analysis import window_rescore as wr  # noqa: E402

GOLDEN = ROOT / "tests" / "golden" / "window_rescore_delay_free" / "golden.json"
AFTER_S = 5.0
WINDOWS = ("odor", "after", "odor_plus_after")
#: The delay-free production model (recipe v2-auc-logratio, no fbr_odor_delay_s).
V2_MODEL = Path("/home/ramanlab/Documents/cole/VSCode/FlyBehaviorScoring/outputs/"
                "ordinal_scorer/model_ordinal_xgb_v2_both.json")


def capture() -> dict:
    df = fx.wide_rows()
    rules = {"none": None, "pipeline": fx.pipeline_rule()}
    cols = [c for c in df.columns if c.startswith("dir_val_")]
    traces = df[cols].to_numpy(dtype=float)
    out: dict = {"after_s": AFTER_S, "rows": fx.ROW_NAMES, "windows": {}, "raw": {},
                 "scores_v2": None}
    for name in WINDOWS:
        spec = wr.window_spec(name, AFTER_S)
        bounds = []
        for i in range(len(df)):
            w, fps = wr.trial_windows(df.iloc[i], traces.shape[1], spec)
            bounds.append([w.baseline_end, w.odor_start, w.odor_end, w.n_frames, fps])
        out["windows"][name] = bounds
        for tag, rule in rules.items():
            raw = wr.raw_features(df, spec, rule=rule)
            out["columns"] = list(raw.columns)
            out["raw"][f"{name}|{tag}"] = raw.to_numpy(dtype=float).tolist()
    if V2_MODEL.exists():
        out["scores_v2"] = {
            name: [int(s) for s in wr.rescore(df, wr.window_spec(name, AFTER_S), V2_MODEL,
                                              rule=rules["pipeline"])]
            for name in WINDOWS
        }
    return out


def main() -> None:
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    data = capture()
    GOLDEN.write_text(json.dumps(data, indent=1))   # json writes floats by repr: exact
    n = sum(len(v) for v in data["raw"].values())
    print(f"Wrote {GOLDEN} ({n} feature rows; v2 scores: {data['scores_v2'] is not None})")


if __name__ == "__main__":
    main()
