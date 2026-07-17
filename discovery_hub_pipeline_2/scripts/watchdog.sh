#!/usr/bin/env bash
# watchdog.sh -- Drew-side unattended watchdog. Polls the H100 for the DONE sentinel,
# auto-pulls results, then optionally DESTROYS the Vast instance (from Drew, where your
# Vast API key lives -- the box never holds it). This is the "leave it unattended" piece.
#
# Usage (on Drew, in tmux/nohup):
#   nohup ./watchdog.sh <host> <remote_root> [vast_instance_id] > watchdog.log 2>&1 &
# If vast_instance_id is given and `vastai` CLI is configured, the box is destroyed on
# success. Otherwise it just prints "safe to destroy".
set -euo pipefail
HOST="${1:?usage: watchdog.sh <host> <remote_root> [vast_instance_id]}"
RROOT="${2:-/workspace}"
VAST_ID="${3:-}"
RDATA="$RROOT/dh_data/pipeline2"
POLL="${POLL_SECONDS:-120}"
SCRIPTS="$(cd "$(dirname "$0")" && pwd)"

echo "[watchdog] polling $HOST:$RDATA for DONE/FAILED every ${POLL}s ..."
while true; do
  if ssh -o ConnectTimeout=15 "$HOST" "test -f $RDATA/DONE" 2>/dev/null; then
    echo "[watchdog] DONE detected @ $(date -u). Pulling results ..."
    bash "$SCRIPTS/pull_pipeline2_results.sh" "$HOST" "$RROOT"
    if [[ -n "$VAST_ID" ]] && command -v vastai >/dev/null 2>&1; then
      echo "[watchdog] destroying Vast instance $VAST_ID ..."
      vastai destroy instance "$VAST_ID" || echo "[watchdog] destroy failed; destroy manually!"
    else
      echo "[watchdog] SAFE TO DESTROY the box now (no vast id / CLI given)."
    fi
    exit 0
  fi
  if ssh -o ConnectTimeout=15 "$HOST" "test -f $RDATA/FAILED" 2>/dev/null; then
    echo "[watchdog] FAILED sentinel detected @ $(date -u). Pulling logs for diagnosis ..."
    rsync -az "$HOST:$RDATA/logs/" "/home/alex/discovery_hub/data/pipeline2/logs/" 2>/dev/null || true
    ssh "$HOST" "cat $RDATA/FAILED" 2>/dev/null || true
    echo "[watchdog] NOT destroying (left up for debugging). Exiting."
    exit 2
  fi
  sleep "$POLL"
done
