# Discovery Hub pipeline -- chains the 9 numbered steps.
#
# Reproducibility env vars are exported up front so CUBLAS_WORKSPACE_CONFIG and
# PYTHONHASHSEED are set before any CUDA/interpreter initialization.
export CUBLAS_WORKSPACE_CONFIG := :4096:8
export PYTHONHASHSEED := 0
export DH_DATA_ROOT ?= ./data

PY := python3

.PHONY: help mock real clean smoke \
        01 02 03 04 05 06 07 08 09 10

help:
	@echo "Targets:"
	@echo "  make mock     - run the full pipeline end-to-end with deterministic mocks"
	@echo "  make real     - run the full pipeline against real APIs + GPU models"
	@echo "  make smoke    - run the pytest smoke test (mock, tiny)"
	@echo "  make clean    - remove the data/ tree"
	@echo "  make 01..10   - run an individual stage (mock)"
	@echo "  make 10       - retrieval-quality eval (Recall@k/MRR/nDCG + A/B)"

# ----- full mock pipeline (no GPU, no network) -----
mock: 01 02 03 04 05 06 07 08 09 10
	@echo "\n=== MOCK PIPELINE COMPLETE ==="

01:
	$(PY) 01_download_data.py --mock --mvp
02:
	$(PY) 02_parse_normalize.py
03:
	$(PY) 03_build_graph.py --validate
04:
	$(PY) 04_generate_embeddings.py --mock
05:
	$(PY) 05_build_index.py
06:
	$(PY) 06_train_rgcn.py --mock --steps 2
07:
	$(PY) 07_retrieve_rank.py --mock --query "EGFR inhibitor for oncology"
08:
	$(PY) 08_multiagent_rag.py --mock --query "EGFR inhibitor for oncology"
09:
	$(PY) 09_stability_harness.py --mock --runs 5
10:
	$(PY) 10_eval_retrieval.py --mock --num-queries 200

# ----- full real pipeline (GPU + network). Run 01-06 on Anvil, 07-09 on Drew. -----
real:
	$(PY) 01_download_data.py --mvp
	$(PY) 02_parse_normalize.py
	$(PY) 03_build_graph.py --validate
	$(PY) 04_generate_embeddings.py --batch-size 64
	$(PY) 05_build_index.py
	$(PY) 06_train_rgcn.py --epochs 20
	$(PY) 07_retrieve_rank.py --query "EGFR inhibitor for oncology"
	$(PY) 08_multiagent_rag.py --query "EGFR inhibitor for oncology"
	$(PY) 09_stability_harness.py --runs 5
	$(PY) 10_eval_retrieval.py --num-queries 200

smoke:
	$(PY) -m pytest tests/ -q

clean:
	rm -rf $(DH_DATA_ROOT)
	@echo "removed $(DH_DATA_ROOT)"
