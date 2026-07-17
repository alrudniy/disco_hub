#!/usr/bin/env bash
# pull_pipeline2_results.sh -- pull pipeline_2 results back from the H100 to Drew.
# Run ON DREW.
#
# BUG #9 FIX (2026-07-15). The previous version had two failures that compounded:
#
#   1. It pulled reports/ qrels/ train_labels/ manifests/ logs/ and NOTHING ELSE --
#      omitting adjudication/ and teacher_scores/, which are the two most expensive
#      artifacts on the box (52,709 LLM verdicts and 329,761 dual cross-encoder scores,
#      many GPU-hours and real API spend). The watchdog then destroyed the box.
#
#   2. Every rsync ended `2>/dev/null || true`, so with SSH broken it printed
#      "==> pull complete" and exited 0 having transferred zero bytes. A silent success
#      that triggers an irreversible destroy is the worst possible failure mode.
#
# Now: an explicit tarball of a named manifest, verified after transfer, with a real
# non-zero exit on any failure. The watchdog MUST check this exit code before destroying.
set -euo pipefail

HOST="${1:?usage: pull_pipeline2_results.sh <host> [remote_root] [local_dest]}"
RROOT="${2:-/workspace}"
DEST="${3:-/home/alex/discovery_hub/data/pipeline2}"
RDATA="$RROOT/dh_data/pipeline2"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
TARBALL="pipeline2_results_${STAMP}.tar.gz"

fail () { echo "ERROR: $*" >&2; exit 1; }

mkdir -p "$DEST"

# Artifacts in rough order of cost to regenerate. adjudication/ and teacher_scores/ are
# FIRST because they are the ones you cannot cheaply rebuild.
MANIFEST=(
  adjudication          # llm_cache.jsonl + judgments -- 52,709 verdicts, REAL MONEY
  teacher_scores        # 329,761 dual cross-encoder scores -- GPU-hours
  qrels                 # incl. held_out_query_ids.json
  train_labels          # multi_positive_labels_v1.jsonl
  reports               # eval_*.json/md
  manifests             # §9.2 version metadata
  bakeoff               # winner.json / final_model.json / screen_models.json
  logs
)

echo "==> [1/4] scrubbing secrets before archiving"
# dh2_env.sh holds the z.ai key. It must never ride back in a tarball.
ssh "$HOST" "cd '$RROOT/discovery_hub_pipeline_2' 2>/dev/null && \
    if [[ -f dh2_env.sh ]]; then \
      sed -i -E 's/(API_KEY=)[^ ]*/\1REDACTED/g' dh2_env.sh; \
      echo '    scrubbed dh2_env.sh'; \
    fi" || echo "    (no dh2_env.sh to scrub)"

echo "==> [2/4] building tarball on $HOST"
PRESENT=()
for d in "${MANIFEST[@]}"; do
  if ssh "$HOST" "test -d '$RDATA/$d'"; then
    PRESENT+=("$d")
  else
    echo "    WARNING: $d absent on box"
  fi
done
[[ ${#PRESENT[@]} -gt 0 ]] || fail "no artifact directories found under $RDATA -- wrong path, or the run never started"

ssh "$HOST" "cd '$RDATA' && tar -czf '/tmp/$TARBALL' ${PRESENT[*]}" \
  || fail "tar failed on $HOST"

REMOTE_SIZE="$(ssh "$HOST" "stat -c %s '/tmp/$TARBALL'")" || fail "cannot stat remote tarball"
echo "    $TARBALL: $((REMOTE_SIZE / 1024 / 1024)) MB (${PRESENT[*]})"
[[ "$REMOTE_SIZE" -gt 1024 ]] || fail "tarball is $REMOTE_SIZE bytes -- refusing to treat that as a successful run"

echo "==> [3/4] transferring"
# NO `|| true`. A failed transfer must fail this script so the watchdog does not destroy.
scp -q "$HOST:/tmp/$TARBALL" "$DEST/$TARBALL" || fail "scp failed"
LOCAL_SIZE="$(stat -c %s "$DEST/$TARBALL")"
[[ "$LOCAL_SIZE" -eq "$REMOTE_SIZE" ]] \
  || fail "size mismatch: remote $REMOTE_SIZE != local $LOCAL_SIZE (truncated transfer)"
tar -tzf "$DEST/$TARBALL" > /dev/null || fail "tarball is corrupt after transfer"
tar -xzf "$DEST/$TARBALL" -C "$DEST" || fail "extract failed"
echo "    extracted -> $DEST"

echo "==> [4/4] pulling the winning 4B model (if the run produced one)"
WIN_JSON="$DEST/bakeoff/final_model.json"
if [[ -f "$WIN_JSON" ]]; then
  REMOTE_MODEL="$(python3 -c "import json;print(json.load(open('$WIN_JSON'))['model'])")"
  echo "    winning model on box: $REMOTE_MODEL"
  mkdir -p "$DEST/final_model"
  rsync -az "$HOST:$REMOTE_MODEL/" "$DEST/final_model/" || fail "model rsync failed"
  echo "    pulled -> $DEST/final_model"
else
  echo "    no final_model.json -- no arm was promoted (this is a valid outcome)"
fi

# Loud, checkable proof of what actually landed. The honest progress meter is the
# artifact, not the log.
echo
echo "==> pull complete. Contents:"
for d in "${PRESENT[@]}"; do
  n="$(find "$DEST/$d" -type f 2>/dev/null | wc -l)"
  sz="$(du -sh "$DEST/$d" 2>/dev/null | cut -f1)"
  printf '    %-16s %5s files  %s\n' "$d" "$n" "$sz"
done
CACHE="$DEST/adjudication/llm_cache.jsonl"
if [[ -f "$CACHE" ]]; then
  echo "    llm_cache.jsonl: $(wc -l < "$CACHE") verdicts  <-- BACK THIS UP"
else
  echo "    WARNING: no adjudication/llm_cache.jsonl in the pull. Do NOT destroy the box."
  exit 2
fi
echo
echo "Safe to destroy the box (this script exited 0)."
