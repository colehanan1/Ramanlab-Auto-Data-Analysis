#!/usr/bin/env bash
# Weekly Pi → secured-storage migration.
#
# Runs on the desktop. For each Pi listed in PI_SOURCES below:
#   1. rsync the Pi's data folder → /securedstorage/DATAsec/cole/<week-tag>/<pi-name>/
#   2. If rsync succeeds AND the destination size is sane, ssh-delete the source
#      files (only those untouched for >30 min, so we don't nuke a mid-write).
#
# Run from cron (Monday 03:00 by default — see install instructions at bottom).
# Idempotent: rerunning a week just resyncs whatever's left.

set -uo pipefail

# ─── config ───────────────────────────────────────────────────────────
# SSH connection details (HostName, User, IdentityFile, IdentityAgent=none)
# live in ~/.ssh/config under Host behaviorlocust / Host flybehavior2 — keep
# those entries in sync if the Pis are re-imaged.
DEST_BASE="/securedstorage/DATAsec/cole"
LOG_DIR="$DEST_BASE/_weekly_logs"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=15)

# pi_name|ssh_host|remote_path
PI_SOURCES=(
    "behaviorlocust|behaviorlocust|/home/ramanlab/FastStorage/fly_videos"
    "flybehavior2|flybehavior2|/home/ramanlab/Documents/faststorage/Opto"
)

# Skip files modified within this many minutes (protects mid-write recordings).
SAFETY_MINUTES=30

# Tag the backup folder with the ISO week the script ran in.
# Format: 2026-W20-2026-05-18  (week tag + the Monday it ran on)
WEEK_TAG="$(date +%Y-W%V)-$(date +%Y-%m-%d)"
DEST_WEEK="$DEST_BASE/$WEEK_TAG"

mkdir -p "$DEST_WEEK" "$LOG_DIR"

LOG="$LOG_DIR/$WEEK_TAG.log"
exec >> "$LOG" 2>&1

log() { echo "[$(date '+%F %T')] $*"; }
fail() { log "FATAL: $*"; exit 1; }

log "======================================="
log "weekly_pi_backup starting"
log "WEEK_TAG=$WEEK_TAG"
log "DEST_WEEK=$DEST_WEEK"
log "SAFETY_MINUTES=$SAFETY_MINUTES"
log "host=$(hostname) user=$(whoami)"

# ─── sanity ──────────────────────────────────────────────────────────
[ -d "$DEST_BASE" ] || fail "destination $DEST_BASE missing"
[ -d "$DEST_WEEK" ] || fail "could not create $DEST_WEEK"

# ─── per-Pi loop ─────────────────────────────────────────────────────
TOTAL_BYTES_COPIED=0
ERRORS=0

for entry in "${PI_SOURCES[@]}"; do
    IFS='|' read -r PI_NAME PI_HOST PI_PATH <<< "$entry"
    log ""
    log "─── $PI_NAME ──────────────────────────────"
    log "host=$PI_HOST  path=$PI_PATH"

    DEST_PI="$DEST_WEEK/$PI_NAME"
    mkdir -p "$DEST_PI"

    # Reachability check
    if ! ssh "${SSH_OPTS[@]}" "$PI_HOST" 'echo ok' > /dev/null 2>&1; then
        log "ERROR: $PI_NAME unreachable via SSH — skipping"
        ERRORS=$((ERRORS + 1))
        continue
    fi

    # Source size before copy
    SRC_SIZE_BEFORE=$(ssh "${SSH_OPTS[@]}" "$PI_HOST" "du -sb '$PI_PATH' 2>/dev/null | awk '{print \$1}'" || echo 0)
    log "source size before: $SRC_SIZE_BEFORE bytes ($(numfmt --to=iec --suffix=B $SRC_SIZE_BEFORE 2>/dev/null || echo n/a))"

    if [ "$SRC_SIZE_BEFORE" -eq 0 ] 2>/dev/null; then
        log "source empty or missing — nothing to do"
        continue
    fi

    # rsync (preserve mtime + perms; skip files modified in last $SAFETY_MINUTES)
    # We can't use rsync --exclude based on mtime directly, so we generate a
    # transfer list via SSH `find` and feed it to rsync via --files-from.
    LIST_FILE=$(mktemp)
    ssh "${SSH_OPTS[@]}" "$PI_HOST" \
        "find '$PI_PATH' -type f -mmin +$SAFETY_MINUTES -printf '%P\n'" \
        > "$LIST_FILE"
    FILE_COUNT=$(wc -l < "$LIST_FILE" | tr -d ' ')
    log "transfer-list size: $FILE_COUNT files (older than $SAFETY_MINUTES min)"

    if [ "$FILE_COUNT" -eq 0 ]; then
        log "no files eligible (all newer than $SAFETY_MINUTES min) — skipping"
        rm -f "$LIST_FILE"
        continue
    fi

    log "rsync starting…"
    if rsync -avh --partial --info=stats2 \
        -e "ssh ${SSH_OPTS[*]}" \
        --files-from="$LIST_FILE" \
        "$PI_HOST:$PI_PATH/" \
        "$DEST_PI/"; then
        log "rsync succeeded"
    else
        RC=$?
        log "ERROR: rsync failed exit=$RC — NOT deleting source"
        ERRORS=$((ERRORS + 1))
        rm -f "$LIST_FILE"
        continue
    fi

    DEST_SIZE=$(du -sb "$DEST_PI" 2>/dev/null | awk '{print $1}')
    log "destination size after: $DEST_SIZE bytes ($(numfmt --to=iec --suffix=B $DEST_SIZE 2>/dev/null || echo n/a))"
    TOTAL_BYTES_COPIED=$((TOTAL_BYTES_COPIED + DEST_SIZE))

    # Delete the source files we just copied (paths in $LIST_FILE)
    log "deleting transferred files on $PI_NAME…"
    REMOTE_LIST=$(mktemp)
    awk -v p="$PI_PATH/" '{print p $0}' "$LIST_FILE" > "$REMOTE_LIST"

    # ship list to Pi, delete by name, then drop empty parent dirs
    REMOTE_TMP="/tmp/weekly_backup_delete_$$.list"
    scp "${SSH_OPTS[@]}" "$REMOTE_LIST" "$PI_HOST:$REMOTE_TMP" > /dev/null
    DELETE_CMD="xargs -d '\\n' -r rm -f -- < '$REMOTE_TMP' && rm -f '$REMOTE_TMP' && find '$PI_PATH' -mindepth 1 -type d -empty -delete"
    if ssh "${SSH_OPTS[@]}" "$PI_HOST" "$DELETE_CMD"; then
        log "source delete OK"
    else
        log "WARN: source delete returned non-zero (files may remain)"
    fi
    rm -f "$LIST_FILE" "$REMOTE_LIST"
done

log ""
log "======================================="
log "Summary"
log "  Total bytes copied: $TOTAL_BYTES_COPIED ($(numfmt --to=iec --suffix=B $TOTAL_BYTES_COPIED 2>/dev/null || echo n/a))"
log "  Errors: $ERRORS"
log "Done."

exit "$ERRORS"
