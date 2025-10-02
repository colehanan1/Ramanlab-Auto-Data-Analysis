#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${CONDA_DEFAULT_ENV:-}" && -z "${VIRTUAL_ENV:-}" ]]; then
  echo "[fbpipe] No Python environment detected. Activate your Conda environment (e.g., yolo-env)." >&2
  exit 1
fi

python -m fbpipe.pipeline --config config.yaml all
