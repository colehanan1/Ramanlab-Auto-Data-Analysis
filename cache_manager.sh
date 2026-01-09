#!/usr/bin/env bash
set -euo pipefail
exec "$(dirname "${BASH_SOURCE[0]}")/scripts/dev/cache_manager.sh" "$@"
