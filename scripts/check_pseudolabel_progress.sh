#!/usr/bin/env bash
set -euo pipefail
exec "$(dirname "${BASH_SOURCE[0]}")/train/check_pseudolabel_progress.sh" "$@"
