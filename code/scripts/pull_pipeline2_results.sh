#!/usr/bin/env bash
# pull_pipeline2_results.sh -- pull pipeline_2 results back from the H100 to Drew.
# Run ON DREW. Pulls reports, labels, qrels, and the winning merged model.
set -euo pipefail
HOST="${1:?usage: pull_pipeline2_results.sh <host> [remote_root] [local_dest]}"
RROOT="${2:-/workspace}"
DEST="${3:-/home/alex/discovery_hub/data/pipeline2}"
RDATA="$RROOT/dh_data/pipeline2"

mkdir -p "$DEST"
echo "==> pulling reports, qrels, labels, manifests"
rsync -az "$HOST:$RDATA/reports/"   "$DEST/reports/"   2>/dev/null || true
rsync -az "$HOST:$RDATA/qrels/"     "$DEST/qrels/"     2>/dev/null || true
rsync -az "$HOST:$RDATA/train_labels/" "$DEST/train_labels/" 2>/dev/null || true
rsync -az "$HOST:$RDATA/manifests/" "$DEST/manifests/" 2>/dev/null || true
rsync -az "$HOST:$RDATA/logs/"      "$DEST/logs/"      2>/dev/null || true

echo "==> pulling the winning 4B model (if the run produced one)"
WIN_JSON="$DEST/bakeoff/final_model.json"
mkdir -p "$DEST/bakeoff"
rsync -az "$HOST:$RDATA/bakeoff/final_model.json" "$DEST/bakeoff/" 2>/dev/null || true
rsync -az "$HOST:$RDATA/bakeoff/winner.json"      "$DEST/bakeoff/" 2>/dev/null || true
if [[ -f "$WIN_JSON" ]]; then
  REMOTE_MODEL="$(python3 -c "import json;print(json.load(open('$WIN_JSON'))['model'])")"
  echo "    winning model on box: $REMOTE_MODEL"
  mkdir -p "$DEST/final_model"
  rsync -az "$HOST:$REMOTE_MODEL/" "$DEST/final_model/"
  echo "    pulled -> $DEST/final_model"
fi
echo "==> pull complete. Reports in $DEST/reports/"
ls -la "$DEST/reports/" 2>/dev/null || true
