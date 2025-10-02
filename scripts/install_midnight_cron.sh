#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_PATH="${PROJECT_ROOT}/scripts/nightly_make_run.sh"
LOG_DIR="${PROJECT_ROOT}/logs"
LOG_FILE="${LOG_DIR}/nightly_make_run.log"
CRON_MARK="# Ramanlab Auto Data Analysis nightly make run"
CRON_JOB="0 0 * * * /bin/bash ${SCRIPT_PATH} >> ${LOG_FILE} 2>&1"

if [ ! -x "${SCRIPT_PATH}" ]; then
    echo "Expected executable ${SCRIPT_PATH} not found." >&2
    exit 1
fi

mkdir -p "${LOG_DIR}"

TEMP_FILE="$(mktemp)"
if crontab -l 2>/dev/null > "${TEMP_FILE}"; then
    :
else
    : > "${TEMP_FILE}"
fi

grep -vF "${SCRIPT_PATH}" "${TEMP_FILE}" | grep -vF "${CRON_MARK}" > "${TEMP_FILE}.filtered" || true
{
    echo "${CRON_MARK}"
    echo "${CRON_JOB}"
} >> "${TEMP_FILE}.filtered"

crontab "${TEMP_FILE}.filtered"
rm -f "${TEMP_FILE}" "${TEMP_FILE}.filtered"

echo "Nightly cron job installed. Review with 'crontab -l'. Logs will accumulate in ${LOG_FILE}."
