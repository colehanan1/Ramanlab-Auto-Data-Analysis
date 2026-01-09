#!/usr/bin/env bash
set -euo pipefail
exec "$(dirname "${BASH_SOURCE[0]}")/dev/uninstall_midnight_cron.sh" "$@"
