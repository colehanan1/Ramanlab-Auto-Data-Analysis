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

# Cron appends to logs/nightly_make_run.log forever. When our stdout is that
# file and it is over the limit, move it into logs/archive/, point stdout and
# stderr at a fresh file, and gzip the archive in the background. Never deletes
# content, never overwrites an archive, and never stops the run: every step
# bails out with `return 0` (set -e is off inside a function called with ||).
NIGHTLY_LOG_MAX_BYTES="${NIGHTLY_LOG_MAX_BYTES:-1073741824}"
NIGHTLY_LOG_GZIP_POLL_S="${NIGHTLY_LOG_GZIP_POLL_S:-60}"

rotate_log() {
    local log="${PROJECT_ROOT}/logs/nightly_make_run.log"
    local archive_dir="${PROJECT_ROOT}/logs/archive"
    local archive size

    [ -f "${log}" ] && [ /dev/stdout -ef "${log}" ] || return 0
    # A root-owned fresh log would lock the user's cron redirect out of it.
    if [ -L "${log}" ] || [ ! -O "${log}" ]; then
        echo "[nightly] not rotating ${log}: it is a symlink or not owned by $(id -un)"
        return 0
    fi
    size="$(stat -c %s "${log}" 2>/dev/null)" || return 0
    [ "${size}" -gt "${NIGHTLY_LOG_MAX_BYTES}" ] 2>/dev/null || return 0

    archive="${archive_dir}/nightly_make_run.$(date +%F).log"
    mkdir -p "${archive_dir}" || return 0
    if [ -e "${archive}" ] || [ -e "${archive}.gz" ]; then
        return 0
    fi
    mv -n "${log}" "${archive}" || return 0
    if [ -e "${log}" ] || ! touch "${log}" 2>/dev/null; then
        mv -n "${archive}" "${log}" || true
        return 0
    fi
    exec >> "${log}" 2>&1
    echo "[nightly] $(date '+%F %T') moved the previous ${size}-byte log to ${archive}.gz"

    # gzip only once nothing writes to the archive: cron's /bin/sh keeps its
    # redirect on the moved file until the run ends, and a run still going from
    # the night before would too. Detached from cron's stream; errors go to the log.
    ( { while fuser -s "${archive}" 2>/dev/null; do sleep "${NIGHTLY_LOG_GZIP_POLL_S}"; done
        nice -n 19 gzip "${archive}"; } < /dev/null > /dev/null 2>> "${log}" & ) || true
}
rotate_log || true

cd "${PROJECT_ROOT}"
exec make run CONFIG=config/config_new.yaml
