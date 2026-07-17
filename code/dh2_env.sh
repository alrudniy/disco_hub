export PYTHON=/venv/main/bin/python
export DH_DATA_ROOT=/workspace/dh_data
export DH2_ROOT=/workspace/dh_data/pipeline2
export DH_EMBED_MODEL=/workspace/models/qwen3-dh-ft-4b
export DH2_MODEL_4B=/workspace/models/qwen3-dh-ft-4b
export DH2_MODEL_8B=/workspace/models/qwen3-dh-ft-8b
export DH2_MODEL_06B=/workspace/models/qwen3-dh-ft
export DH2_TEACHER_BGE=BAAI/bge-reranker-v2-m3
export DH2_TEACHER_QWEN=Qwen/Qwen3-Reranker-8B
export DH2_TEACHER_LLM_API_KEY=PUT_KEY_HERE
export DH2_TEACHER_LLM_MODEL=glm-4.6
export DH2_VECTORS_8B=/workspace/dh_data/embeddings_8b/doc_vectors.npy
export DH2_IDS_8B=/workspace/dh_data/embeddings_8b/doc_ids.json
export PYTHONPATH=/workspace/discovery_hub_pipeline_2
unset HF_HUB_OFFLINE HF_HOME
export DH2_TEACHER_LLM_BASE_URL=https://api.z.ai/api/paas/v4
