#!/usr/bin/env bash
# Install (or refresh after editing) the Saturday 12:00 GitHub sync timer.
# Usage: scripts/dev/install_weekly_github_sync.sh
set -euo pipefail
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/systemd"
DEST="${HOME}/.config/systemd/user"
mkdir -p "${DEST}"
cp "${SRC}/weekly-github-sync.service" "${SRC}/weekly-github-sync.timer" "${DEST}/"
systemctl --user daemon-reload
systemctl --user enable --now weekly-github-sync.timer
systemctl --user list-timers weekly-github-sync.timer --no-pager
