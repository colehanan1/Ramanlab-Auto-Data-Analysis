#!/usr/bin/env bash
set -euo pipefail
exec "$(dirname "${BASH_SOURCE[0]}")/train/monitor_pseudolabel.sh" "$@"
