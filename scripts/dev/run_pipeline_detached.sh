#!/usr/bin/env bash
# Run the pipeline as its own systemd user service so it outlives VS Code.
# A run started from a VS Code terminal lives in VS Code's cgroup and dies with it.
# Usage: scripts/dev/run_pipeline_detached.sh [extra run_workflows.py args]
#   e.g. scripts/dev/run_pipeline_detached.sh --figures-only
# Unlike `make run`: no `make backup` afterwards (it currently backs up nothing),
# config/config_new.yaml unless --config is passed, and variables exported in the
# calling shell (FBPIPE_*, CACHE_DIR, ...) do not reach the service.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
YOLO_ENV_NAME="${YOLO_ENV_NAME:-yolo-env}"

# Same conda discovery as nightly_make_run.sh (CONDA_BASE, CONDA_EXE, MAMBA_EXE, PATH, common dirs).
discover_conda_base() {
    local base candidate
    if [ -n "${CONDA_BASE:-}" ] && [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
        printf '%s\n' "${CONDA_BASE}"; return 0
    fi
    for exe in "${CONDA_EXE:-}" "${MAMBA_EXE:-}"; do
        if [ -n "${exe}" ] && [ -x "${exe}" ]; then
            base="$(cd "$(dirname "${exe}")/.." && pwd)"
            if [ -f "${base}/etc/profile.d/conda.sh" ]; then
                printf '%s\n' "${base}"; return 0
            fi
        fi
    done
    if command -v conda >/dev/null 2>&1; then
        base="$(conda info --base 2>/dev/null || true)"
        if [ -n "${base}" ] && [ -f "${base}/etc/profile.d/conda.sh" ]; then
            printf '%s\n' "${base}"; return 0
        fi
    fi
    for candidate in "${HOME}/miniconda3" "${HOME}/anaconda3" "${HOME}/conda" "/opt/conda"; do
        if [ -f "${candidate}/etc/profile.d/conda.sh" ]; then
            printf '%s\n' "${candidate}"; return 0
        fi
    done
    return 1
}

if ! CONDA_BASE="$(discover_conda_base)"; then
    echo "Unable to locate Conda. Set CONDA_BASE or ensure CONDA_EXE/conda is available." >&2
    exit 1
fi

STAMP="$(date +%Y%m%d-%H%M%S)"
UNIT="fbpipe-run-${STAMP}"
LOG="${PROJECT_ROOT}/logs/pipeline_detached_${STAMP}.log"
mkdir -p "${PROJECT_ROOT}/logs"

# Runs inside the service: activate the env, then exec the driver with the caller's args.
INNER='source "$1/etc/profile.d/conda.sh"
conda activate "$2" >/dev/null 2>&1 || { echo "Failed to activate Conda environment $2." >&2; exit 1; }
shift 2
exec python3 scripts/pipeline/run_workflows.py --config config/config_new.yaml "$@"'

# Type=exec: a unit that cannot even start fails here, not only in the journal.
# --expand-environment=no: an argument containing "$" reaches the driver as typed.
# No --collect, so a failed run stays listed (systemctl --user --failed), and
# ExecStopPost writes how the unit ended into the log, however it ended.
# OOMScoreAdjust=100 matches a terminal run; a service would otherwise get 200.
if ! systemd-run --user --unit="${UNIT}" --expand-environment=no \
    -p Type=exec \
    -p OOMPolicy=continue \
    -p OOMScoreAdjust=100 \
    -p "WorkingDirectory=${PROJECT_ROOT}" \
    -p "StandardOutput=append:${LOG}" \
    -p "StandardError=append:${LOG}" \
    -p "ExecStopPost=/bin/sh -c 'echo \"[detached] unit ended: result=\$SERVICE_RESULT code=\$EXIT_CODE status=\$EXIT_STATUS\" >> \"\$0\"' ${LOG}" \
    --setenv=MPLBACKEND=Agg \
    --setenv=ORT_LOGGING_LEVEL=3 \
    --setenv=PYTHONNOUSERSITE=1 \
    --setenv=PYTHONUNBUFFERED=1 \
    bash -c "${INNER}" _ "${CONDA_BASE}" "${YOLO_ENV_NAME}" "$@"; then
    echo "systemd-run failed; the pipeline was NOT started." >&2
    exit 1
fi

echo "Started unit: ${UNIT}"
echo "Log:          ${LOG}"
echo "Follow:       tail -f ${LOG}   (the last line says how the run ended)"
echo "Status:       systemctl --user status ${UNIT}   (a failed run stays listed until: systemctl --user reset-failed ${UNIT})"
echo "Stop:         systemctl --user stop ${UNIT}"
