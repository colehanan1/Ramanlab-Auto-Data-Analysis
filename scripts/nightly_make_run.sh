#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
YOLO_ENV_NAME="${YOLO_ENV_NAME:-yolo-env}"

if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    CONDA_BASE="$(conda info --base 2>/dev/null || true)"
    if [ -z "${CONDA_BASE}" ] || [ ! -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
        echo "Unable to locate conda.sh. Ensure Conda is initialised for non-interactive shells." >&2
        exit 1
    fi

    # shellcheck disable=SC1091
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${YOLO_ENV_NAME}"
else
    echo "'conda' command not found in PATH. Nightly run requires the ${YOLO_ENV_NAME} environment." >&2
    exit 1
fi

cd "${PROJECT_ROOT}"
exec make run
