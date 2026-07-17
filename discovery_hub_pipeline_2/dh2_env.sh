export PYTHON=/venv/main/bin/python
export DH_DATA_ROOT=/workspace/dh_data
export DH2_ROOT=/workspace/dh_data/pipeline2
export DH_EMBED_MODEL=/workspace/models/qwen3-dh-ft-4b
export DH2_MODEL_4B=/workspace/models/qwen3-dh-ft-4b
export DH2_MODEL_8B=/workspace/models/qwen3-dh-ft-8b
export DH2_MODEL_06B=/workspace/models/qwen3-dh-ft
export DH2_MODEL_BASE_06B=Qwen/Qwen3-Embedding-0.6B
export DH2_TEACHER_BGE=BAAI/bge-reranker-v2-m3
export DH2_TEACHER_QWEN=Qwen/Qwen3-Reranker-8B
export DH2_TEACHER_LLM_API_KEY=REDACTED
# CRITICAL: this default used to be 'claude-sonnet-4-6' on 1min.ai, which is DEAD. The
# cache key is derived from the model string, so a wrong value silently misses all 52,709
# cached verdicts and re-pays for every one of them.
export DH2_TEACHER_LLM_MODEL=glm-4.6
export DH2_TEACHER_LLM_BASE_URL=https://api.z.ai/api/paas/v4
export DH2_VECTORS_8B=/workspace/dh_data/embeddings_8b/doc_vectors.npy
export DH2_IDS_8B=/workspace/dh_data/embeddings_8b/doc_ids.json
# untouched-0.6B vectors (semantic hedge). Empty until the corpus is embedded with the
# stock checkpoint (~45 min); p0_1/p1_4 warn and fall back to the old pool without it.
export DH2_VECTORS_BASE_06B=/workspace/dh_data/embeddings_base06b/doc_vectors.npy
export DH2_IDS_BASE_06B=/workspace/dh_data/embeddings_base06b/doc_ids.json
# RASC flags: cascade off in the R&D orchestrator; GroupRank off until it passes its gates
export DH_REGISTER_AWARE_CASCADE=0
export DH_ENABLE_GROUPRANK=0
export PYTHONPATH=/workspace/discovery_hub_pipeline_2
unset HF_HUB_OFFLINE HF_HOME
