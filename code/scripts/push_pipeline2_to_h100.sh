#!/usr/bin/env bash
# push_pipeline2_to_h100.sh -- sync pipeline_2 code + required data from Drew to the
# rented Vast H100, and write the remote env file. Run this ON DREW.
#
# Usage:
#   ./push_pipeline2_to_h100.sh <ssh_alias_or_host> [remote_root]
# Example (Vast alias in ~/.ssh/config):
#   ./push_pipeline2_to_h100.sh vast_h100 /workspace
set -euo pipefail

HOST="${1:?usage: push_pipeline2_to_h100.sh <host> [remote_root]}"
RROOT="${2:-/workspace}"
LOCAL_PIPE="$(cd "$(dirname "$0")/.." && pwd)"
DH_DATA_ROOT="${DH_DATA_ROOT:-/home/alex/discovery_hub/data}"
MODELS_DIR="${DH_MODELS_DIR:-/home/alex/discovery_hub/models}"

RCODE="$RROOT/discovery_hub_pipeline_2"
RDATA="$RROOT/dh_data"
RMODELS="$RROOT/models"

echo "==> creating remote dirs on $HOST"
ssh "$HOST" "mkdir -p $RCODE $RDATA/normalized $RDATA/embeddings $RDATA/index $RDATA/finetune $RMODELS"

echo "==> syncing pipeline_2 code (+ the base discovery_hub package it imports)"
rsync -az --delete --exclude '__pycache__' --exclude '*.pyc' \
    "$LOCAL_PIPE"/ "$HOST:$RCODE"/
# the base package must be importable on the box:
rsync -az --exclude '__pycache__' \
    /home/alex/discovery_hub_pipeline/discovery_hub "$HOST:$RCODE/"

echo "==> syncing required data (docs.jsonl, 4B vectors+ids, train triples/queries)"
rsync -az "$DH_DATA_ROOT/normalized/docs.jsonl"        "$HOST:$RDATA/normalized/"
rsync -az "$DH_DATA_ROOT/embeddings/doc_vectors.npy"   "$HOST:$RDATA/embeddings/"
rsync -az "$DH_DATA_ROOT/index/doc_ids.json"           "$HOST:$RDATA/index/"
rsync -az "$DH_DATA_ROOT/finetune/train_triples.jsonl" "$HOST:$RDATA/finetune/" 2>/dev/null || true
rsync -az "$DH_DATA_ROOT/finetune/synthetic_queries.jsonl" "$HOST:$RDATA/finetune/" 2>/dev/null || true

echo "==> syncing models (4B required; 8B optional; 0.6B for screening)"
rsync -az "$MODELS_DIR/qwen3-dh-ft-4b" "$HOST:$RMODELS/"
rsync -az "$MODELS_DIR/qwen3-dh-ft"    "$HOST:$RMODELS/" 2>/dev/null || true
if [[ "${PUSH_8B:-0}" == "1" ]]; then
  rsync -az "$MODELS_DIR/qwen3-dh-ft-8b" "$HOST:$RMODELS/"
  rsync -az "$DH_DATA_ROOT/embeddings_8b/doc_vectors.npy" "$HOST:$RDATA/embeddings_8b/" 2>/dev/null || true
  rsync -az "$DH_DATA_ROOT/embeddings_8b/doc_ids.json"    "$HOST:$RDATA/embeddings_8b/" 2>/dev/null || true
fi

echo "==> writing remote env file"
# NOTE: the teacher LLM key is passed via your shell env; it is written to the box so
# the unattended run can call 1min.ai. The box never holds your Vast API key.
ssh "$HOST" "cat > $RCODE/dh2_env.sh" <<ENVEOF
export PYTHON=/venv/main/bin/python
export DH_DATA_ROOT=$RDATA
export DH2_ROOT=$RDATA/pipeline2
export DH_EMBED_MODEL=$RMODELS/qwen3-dh-ft-4b
export DH2_MODEL_4B=$RMODELS/qwen3-dh-ft-4b
export DH2_MODEL_8B=$RMODELS/qwen3-dh-ft-8b
export DH2_MODEL_06B=$RMODELS/qwen3-dh-ft
export DH2_TEACHER_BGE=BAAI/bge-reranker-v2-m3
export DH2_TEACHER_QWEN=Qwen/Qwen3-Reranker-8B
export DH2_TEACHER_LLM_API_KEY=${DH2_TEACHER_LLM_API_KEY:-PUT_KEY_HERE}
export DH2_TEACHER_LLM_MODEL=${DH2_TEACHER_LLM_MODEL:-claude-sonnet-4-6}
export DH2_VECTORS_8B=$RDATA/embeddings_8b/doc_vectors.npy
export DH2_IDS_8B=$RDATA/embeddings_8b/doc_ids.json
export PYTHONPATH=$RCODE
unset HF_HUB_OFFLINE HF_HOME
ENVEOF

echo "==> installing python deps on the box"
ssh "$HOST" "/venv/main/bin/pip install -q -r $RCODE/requirements2.txt || true"

echo "==> DONE. To launch the unattended run:"
echo "    ssh $HOST 'cd $RCODE && source dh2_env.sh && nohup bash scripts/run_pipeline2_h100.sh > \$DH2_ROOT/logs/run.log 2>&1 &'"
