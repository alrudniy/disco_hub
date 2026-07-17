#!/usr/bin/env bash
set -uo pipefail
cd /workspace/discovery_hub_pipeline_2
source dh2_env.sh
unset HF_HUB_OFFLINE HF_HOME
export DH2_GR_TP_SIZE=1
mkdir -p "$DH2_ROOT/logs"
echo "START $(date -u +%FT%TZ)" > "$DH2_ROOT/logs/settle.log"
"$PYTHON" stages/p1_4_eval_cascade.py \
    --utility /workspace/dh2_takehome/data/qrels/qrels_scout_utility_v1.jsonl \
    --labels  /workspace/dh2_takehome/data/multi_positive_labels_v1.jsonl \
    --vectors-base-06b "$DH2_VECTORS_BASE_06B" \
    --ids-base-06b     "$DH2_IDS_BASE_06B" \
    --vectors-ft-06b   "$DH2_ROOT/vectors/ft06b/doc_vectors.npy" \
    --ids-ft-06b       "$DH2_ROOT/vectors/ft06b/doc_ids.json" \
    --vectors-8b       "$DH2_VECTORS_8B" \
    --ids-8b           "$DH2_IDS_8B" \
    --arms C0,C0ft,C1,C1b,C2 \
    --llm-judged-only \
    --name settle \
    >> "$DH2_ROOT/logs/settle.log" 2>&1
echo "EXIT=$?" >> "$DH2_ROOT/logs/settle.log"
