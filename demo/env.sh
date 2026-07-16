# demo/env.sh -- source this before the demo.  `source demo/env.sh`
#
# This is spec section 5's setup block, CORRECTED against the real filesystem. The
# spec's version does not run: it exports DH_GRAPH_DIR (which config.py did not
# read until this build added it), points at a data root that cannot address both
# the graph and the embeddings, and leaves the embedding model at a default whose
# dimensionality does not match the shipped index. Every export below carries the
# measurement or the file fact that justifies it.
#
# Every value here is also the DEFAULT in config.py, or is checked at load time --
# nothing is silently required. What this file buys is a demo that runs the
# MEASURED configuration rather than the historical one.

# --------------------------------------------------------------------------- #
# Data roots. TWO of them, which is the whole reason per-directory overrides exist.
# --------------------------------------------------------------------------- #
# The corpus, the embeddings and the FAISS index live under data/. The graph and
# its R-GCN embeddings live under data_merged/. config.py derives every directory
# from ONE root (DH_DATA_ROOT), so a single root physically cannot address both --
# hence DH_GRAPH_DIR / DH_ARTIFACT_DIR, which override just those two.
export DH_DATA_ROOT=/home/alex/discovery_hub/data
export DH_GRAPH_DIR=/home/alex/discovery_hub/data_merged/graph
export DH_ARTIFACT_DIR=/home/alex/discovery_hub/data_merged/artifacts

# WHY data_merged AND NOT data/graph: data/graph is the OLD, UNMERGED graph
# (~1.6M nodes) and its node embeddings are NOT row-aligned with data_merged's.
# Pointing at it yields an index whose rows silently describe different nodes --
# wrong answers, no error. The merged graph is 1,489,785 nodes / 4,520,029 edges,
# and artifacts/node_ids.json is the verified row authority for rgcn_node_emb.npy
# (checked: 0 order mismatches across all 1,489,785 rows).

# --------------------------------------------------------------------------- #
# Embedding model. THE ONE THAT CRASHES IF YOU GET IT WRONG.
# --------------------------------------------------------------------------- #
# doc_vectors.npy is (603369, 2560) and faiss.index is 2560-dim, both produced by
# the fine-tuned local model below (config.json hidden_size=2560). config.py's
# default is Qwen/Qwen3-Embedding-0.6B at 1024-dim, so a real run with the default
# config dies on a dimension mismatch at query time.
#
# NOTE embed.log claims "(603369, 1024)" -- it is STALE. Those 1024-dim vectors are
# doc_vectors_06B.npy, and no FAISS index was ever built for them. Ignore the log,
# trust the array.
export DH_EMBED_MODEL=/home/alex/discovery_hub/models/qwen3-dh-ft-4b
export DH_EMBED_DIM=2560

# --------------------------------------------------------------------------- #
# Retrieval channels: ship the MEASURED config, not the historical one.
# --------------------------------------------------------------------------- #
# KEYWORD OFF. Measured on the 123 independently LLM-adjudicated utility-qrels
# queries (relevant = grade >= 2, pool budget 100):
#     dense top-100 only ......... recall@100 0.7736
#     RRF(dense, keyword) ........ recall@100 0.6798   <- was production
# RRF discarded 578 relevant docs dense had already found in order to seat
# keyword's 24 unique ones: dense-only wins by +0.0937 recall@100. BM25 is NOT
# useless (+0.0520 unique reach; union hits 0.8256) -- but only at a 200-doc pool,
# i.e. double the reranker bill. Re-open by unioning at a larger pool, NOT by
# re-enabling RRF.
#
# HONESTY NOTE: spec section 5 justifies this flag with "+0.30 nDCG, 6x clear of
# any power concern". No measurement in this repo produces that number -- the
# measured effect is +0.0937 recall@100, a different metric and a different
# magnitude. The flag is right; the spec's number for it is not. Quote config.py.
export DH_USE_KEYWORD=0

# GRAPH OFF. The graph channel's UNIQUE REACH is exactly zero: across the 62
# firing queries (668 relevant docs, 6,200 graph candidates) it surfaced 0
# relevant documents that dense's top-100 missed. Not "few" -- zero. In its full
# 600,738-doc ranking the median best rank of any relevant doc is 195,898, so no
# depth and no fusion rescues it. Root cause is upstream of the R-GCN:
# entity_link fires on junk ("CXCR4 gene therapy approach" ->
# organization:science approach), so the channel averages unrelated entities.
#
# THIS IS NOT AN ARGUMENT AGAINST THE GRAPH. It is an argument against the graph
# as a RETRIEVAL channel. The expertise-gap agent is the same graph doing the job
# it is actually shaped for -- absence, ownership and traversal -- and it is on.
export DH_USE_GRAPH=0

# --------------------------------------------------------------------------- #
# LLM: glm-4.6 via z.ai (OpenAI-compatible: POST {base}/chat/completions,
# Authorization: Bearer). Verified live -- see agents/LLM_CONTRACT.md.
# --------------------------------------------------------------------------- #
export DH_LLM_BASE_URL='https://api.z.ai/api/paas/v4'
export DH_LLM_MODEL='glm-4.6'

# NEVER put the key in this file -- it is committed. Provide it one of two ways:
#   1. put it in /home/alex/discovery_hub_pipeline/.env (gitignored, mode 600),
#      which the block below sources; or
#   2. export it in your shell before sourcing this file.
# export DH_LLM_API_KEY='...'          # <- do NOT uncomment with a real key
_dh_env_file="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)/.env"
if [ -f "$_dh_env_file" ]; then
    set -a; . "$_dh_env_file"; set +a
    echo "demo/env.sh: sourced $_dh_env_file"
fi
unset _dh_env_file

# WITH NO KEY THE DEMO STILL RUNS, and says so on every screen: synthesis falls
# back to verbatim quotation of the retrieved evidence, and the verifier falls
# back to a lexical grounding check that is strictly weaker (it cannot detect
# contradiction). Both label themselves in their output. A deterministic fallback
# that reads as model output is the exact overclaim this repo exists to avoid.
if [ -n "${DH_LLM_API_KEY:-}" ]; then
    echo "demo/env.sh: DH_LLM_API_KEY is set -- LLM path ACTIVE (${DH_LLM_MODEL})"
else
    echo "demo/env.sh: DH_LLM_API_KEY is NOT set -- deterministic fallbacks will run"
fi

echo "demo/env.sh: embed=${DH_EMBED_MODEL} dim=${DH_EMBED_DIM} keyword=${DH_USE_KEYWORD} graph=${DH_USE_GRAPH}"
