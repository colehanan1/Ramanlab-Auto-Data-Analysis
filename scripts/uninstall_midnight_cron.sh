#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_BASENAME="nightly_make_run.sh"
SCRIPT_PATH="${PROJECT_ROOT}/scripts/${SCRIPT_BASENAME}"
CRON_MARK="# Ramanlab Auto Data Analysis nightly make run"

TEMP_FILE="$(mktemp)"
if crontab -l 2>/dev/null > "${TEMP_FILE}"; then
    :
else
    : > "${TEMP_FILE}"
fi

awk -v mark="${CRON_MARK}" -v script="${SCRIPT_BASENAME}" -v path="${SCRIPT_PATH}" '
    $0 == mark { skip_next = 1; next }
    skip_next {
        if (index($0, script) > 0 || index($0, path) > 0) { skip_next = 0; next }
        skip_next = 0
    }
    index($0, script) > 0 { next }
    index($0, path) > 0 { next }
    { print }
' "${TEMP_FILE}" > "${TEMP_FILE}.filtered"

crontab "${TEMP_FILE}.filtered"
rm -f "${TEMP_FILE}" "${TEMP_FILE}.filtered"

echo "Nightly cron job removed. Review with 'crontab -l'."
