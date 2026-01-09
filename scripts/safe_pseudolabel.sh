#!/usr/bin/env bash
set -euo pipefail
exec "$(dirname "${BASH_SOURCE[0]}")/train/safe_pseudolabel.sh" "$@"
