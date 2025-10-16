"""Build reaction matrices from the prediction spreadsheet."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from ..config import Settings


def _script_path() -> Path:
    return Path(__file__).resolve().parents[2] / "scripts" / "reaction_matrix_from_spreadsheet.py"


def main(cfg: Settings) -> None:
    settings = cfg.reaction_prediction
    matrix_cfg = settings.matrix

    if not settings.output_csv:
        raise SystemExit("reaction_prediction.output_csv is not configured")

    predictions_csv = Path(settings.output_csv).expanduser().resolve()
    if not predictions_csv.exists():
        raise FileNotFoundError(
            f"Predictions spreadsheet not found: {predictions_csv}. Run predict_reactions first."
        )

    if not matrix_cfg.out_dir:
        raise SystemExit("reaction_prediction.matrix.out_dir is not configured")
    out_dir = Path(matrix_cfg.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    script = _script_path()
    if not script.exists():
        raise FileNotFoundError(f"Reaction matrix script is missing: {script}")

    cmd = [
        sys.executable,
        str(script),
        "--csv-path",
        str(predictions_csv),
        "--out-dir",
        str(out_dir),
        "--latency-sec",
        str(matrix_cfg.latency_sec),
        "--after-window-sec",
        str(matrix_cfg.after_window_sec),
        "--row-gap",
        str(matrix_cfg.row_gap),
        "--height-per-gap-in",
        str(matrix_cfg.height_per_gap_in),
        "--bottom-shift-in",
        str(matrix_cfg.bottom_shift_in),
    ]

    for trial_order in matrix_cfg.trial_orders:
        cmd.extend(["--trial-order", trial_order])

    if not matrix_cfg.include_hexanol:
        cmd.append("--exclude-hexanol")
    if matrix_cfg.overwrite:
        cmd.append("--overwrite")

    print("[REACTION] Generating reaction matrices →", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"[REACTION] Reaction matrices written under {out_dir}")
