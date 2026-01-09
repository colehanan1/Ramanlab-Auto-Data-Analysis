#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
YOLO_ENV_NAME="${YOLO_ENV_NAME:-yolo-env}"

discover_conda_base() {
    if [ -n "${CONDA_BASE:-}" ] && [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
        printf '%s\n' "${CONDA_BASE}"
        return 0
    fi

    if [ -n "${CONDA_EXE:-}" ] && [ -x "${CONDA_EXE}" ]; then
        local base
        base="$(cd "$(dirname "${CONDA_EXE}")/.." && pwd)"
        if [ -f "${base}/etc/profile.d/conda.sh" ]; then
            printf '%s\n' "${base}"
            return 0
        fi
    fi

    if [ -n "${MAMBA_EXE:-}" ] && [ -x "${MAMBA_EXE}" ]; then
        local base
        base="$(cd "$(dirname "${MAMBA_EXE}")/.." && pwd)"
        if [ -f "${base}/etc/profile.d/conda.sh" ]; then
            printf '%s\n' "${base}"
            return 0
        fi
    fi

    if command -v conda >/dev/null 2>&1; then
        local base
        base="$(conda info --base 2>/dev/null || true)"
        if [ -n "${base}" ] && [ -f "${base}/etc/profile.d/conda.sh" ]; then
            printf '%s\n' "${base}"
            return 0
        fi
    fi

    for candidate in "${HOME}/miniconda3" "${HOME}/anaconda3" "${HOME}/conda" \
        "/opt/conda"; do
        if [ -f "${candidate}/etc/profile.d/conda.sh" ]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done

    return 1
}

if ! CONDA_BASE="$(discover_conda_base)"; then
    echo "Unable to locate Conda. Set CONDA_BASE or ensure CONDA_EXE/conda is available to cron." >&2
    exit 1
fi

# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if ! conda activate "${YOLO_ENV_NAME}" >/dev/null 2>&1; then
    echo "Failed to activate Conda environment '${YOLO_ENV_NAME}'." >&2
    exit 1
fi

cd "${PROJECT_ROOT}"
exec make run
