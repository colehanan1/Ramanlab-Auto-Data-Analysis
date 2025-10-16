"""Invoke the flybehavior-response CLI to score reactions."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from ..config import Settings


def main(cfg: Settings) -> None:
    settings = cfg.reaction_prediction
    if not settings.data_csv:
        raise SystemExit("reaction_prediction.data_csv is not configured")
    if not settings.model_path:
        raise SystemExit("reaction_prediction.model_path is not configured")
    if not settings.output_csv:
        raise SystemExit("reaction_prediction.output_csv is not configured")

    data_csv = Path(settings.data_csv).expanduser().resolve()
    model_path = Path(settings.model_path).expanduser().resolve()
    output_csv = Path(settings.output_csv).expanduser().resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if not data_csv.exists():
        raise FileNotFoundError(f"Data CSV not found: {data_csv}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    cmd = [
        "flybehavior-response",
        "predict",
        "--data-csv",
        str(data_csv),
        "--model-path",
        str(model_path),
        "--output-csv",
        str(output_csv),
    ]

    env = os.environ.copy()
    repo_root = Path(__file__).resolve().parents[3]
    existing_path = env.get("PYTHONPATH")
    path_parts = [str(repo_root)]
    if existing_path:
        path_parts.append(existing_path)
    env["PYTHONPATH"] = os.pathsep.join(path_parts)

    print("[REACTION] Running flybehavior-response →", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)
    print(f"[REACTION] Wrote predictions to {output_csv}")
