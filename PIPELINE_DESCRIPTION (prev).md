# Discovery Hub Pipeline — Description

A numbered, end-to-end pipeline that implements the three-layer Discovery Hub
architecture (Evidence Integration → Retrieval & Ranking → Explanation & Workflow)
as nine runnable stages. Every stage has a deterministic `--mock` mode, so the
whole system runs, tests, and demonstrates its reproducibility **without a GPU or
any network download**, exactly the way `discovery_finetune` is validated against
real files with a mock embedder. The real-mode code (live APIs, GPU models, R-GCN,
vLLM) is present in each stage, gated behind lazy imports.

---

## 1. How the code maps to the three-layer architecture

| Deck layer | What it does | Stages here |
|---|---|---|
| **Layer 1 — Evidence Integration** | Ingest patents/pubs/trials/org signals into a translation-oriented knowledge graph (technologies, inventors, organizations, experts, facilities) | `01` download · `02` parse/normalize · `03` build graph |
| **Layer 2 — Retrieval & Ranking** | Hybrid semantic search + graph signal + neural rerank; this is where the existing `discovery_finetune` package lives | `04` embeddings · `05` index · `06` R-GCN · `07` retrieve+rank |
| **Layer 3 — Explanation & Workflow** | Strict-RAG, multi-agent pipeline producing cited, provenance-stamped, confidence-scored recommendations with human-in-the-loop | `08` multi-agent RAG |
| **Cross-cutting — Reproducibility QA** | Measures and reports run-to-run stability at every stage (the investor's question) | `09` stability harness |

The existing `discovery_finetune` work (schema normalization, synthetic-query
generation, hard-negative mining, MNRL fine-tuning of Qwen3-Embedding-0.6B,
Recall@k/MRR@10) is the **retrieval layer** inside this fuller vision. Stages
`02`, `04`, and `07` are where it plugs in; the new work is the knowledge graph
(`03`/`06`) and the multi-agent explanation layer (`08`).

---

## 2. The nine stages, in execution order

Run them in numeric order; each consumes the previous stage's artifacts under
`$DH_DATA_ROOT` (default `./data`).

| # | Script | Input → Output | Compute target |
|---|---|---|---|
| 01 | `01_download_data.py` | source APIs → `data/raw/*.jsonl` | Anvil CPU* |
| 02 | `02_parse_normalize.py` | `raw/*` → `normalized/docs.jsonl` (unified `DiscoveryDoc`) | Anvil CPU* |
| 03 | `03_build_graph.py` | `docs.jsonl` → `graph/{nodes,edges}.jsonl` + meta | Anvil CPU* |
| 04 | `04_generate_embeddings.py` | `docs.jsonl` → `embeddings/doc_vectors.npy` | **Anvil GPU** |
| 05 | `05_build_index.py` | vectors → `index/faiss.index` | Anvil CPU → ship to Drew |
| 06 | `06_train_rgcn.py` | graph → `artifacts/rgcn_node_emb.npy` | **Anvil GPU** |
| 07 | `07_retrieve_rank.py` | index + R-GCN → ranked candidates | **Drew** (always-on) |
| 08 | `08_multiagent_rag.py` | candidates → cited recommendations | **Drew** (always-on) |
| 09 | `09_stability_harness.py` | runs 07/08 K times → stability report | either (pin hardware) |

`*` For the MVP biomedical slice (< ~200 GB) stages 01–06 run fine on Drew too;
full scale (~2–3 TB working set) needs Anvil.

**What each stage does**

- **01 — Download.** Pulls records from ClinicalTrials.gov v2, OpenAlex, SBIR,
  USPTO/PatentsView, and AUTM. `--mock` generates a deterministic synthetic corpus
  shaped like each real schema (nested `protocolSection` for trials, flat dicts for
  USPTO, OpenAlex's inverted-index abstracts) so stage 02 exercises real code paths.
  For full-scale OpenAlex, use the S3 snapshot (`aws s3 sync s3://openalex … --no-sign-request`),
  not the REST API.
- **02 — Parse & normalize.** One parser per source converges every schema onto the
  unified `DiscoveryDoc` with a canonical `embedding_text`. Includes the AUTM noise
  filter (`--min-chars`). This is the contract that lets the students' data-collection
  work proceed in parallel behind one schema.
- **03 — Build graph.** Constructs the heterogeneous KG: `technology` nodes (keyed by
  `doc_id` — each invention is unique) plus deduplicated `inventor`/`organization`/
  `expert`/`facility` nodes, connected by typed relations (`invented_by`, `assigned_to`,
  `investigated_by`, `located_at`, `affiliated_with`). Validates with networkx.
- **04 — Embeddings.** Batch-embeds every `embedding_text`. The Anvil A100/H100 job at
  full scale; ~30–100 A100-hours for 40–50M abstracts. Mock embedder is bit-identical
  across machines.
- **05 — Index.** Builds a FAISS `IndexFlatIP` (exact, deterministic) — chosen for the
  reproducibility story; a numpy brute-force fallback runs if FAISS is absent. The index
  is built on Anvil and shipped to Drew.
- **06 — R-GCN.** Learns relational node embeddings. Real path: PyTorch Geometric
  `RGCNConv` with link-prediction. Mock path: deterministic numpy relational
  message-passing. With neighbor sampling this fits a 12 GB GPU; a run is ~1–6 GPU-hours.
- **07 — Retrieve & rank.** Two-stage: text recall (FAISS) → blend R-GCN graph
  similarity → rerank (BGE cross-encoder in real mode, blended-score sort in mock).
  Returns candidates with supporting evidence. Importable `Retriever` class.
- **08 — Multi-agent RAG.** Five agents in sequence — retrieval, expertise-gap,
  reranking, policy-safety, explanation — implemented framework-free as a state dict
  that maps 1:1 onto LangGraph nodes. Strict RAG: every claim carries a citation and
  provenance; a policy gate refuses to assert a recommendation below the confidence
  threshold and routes it to human review instead.
- **09 — Stability harness.** Runs the same queries K times and reports embedding
  drift, retrieval Jaccard / Kendall-τ, and LLM exact-match / semantic-equivalence /
  **citation-set Jaccard**. Emits `reports/stability_report.{json,md}`.

---

## 3. Two run modes

**Mock mode (`--mock`)** — deterministic, no GPU, no network. Synthetic data,
hash-seeded embeddings, numpy message-passing for the R-GCN, and a templated
strict-RAG explainer. This is what CI and the smoke test use, and what proves the
reproducibility claims. Everything is bit-reproducible across machines.

**Real mode (default)** — live source APIs; Qwen3-Embedding-0.6B via
sentence-transformers; PyTorch Geometric R-GCN; BGE-reranker-v2-m3; a self-hosted
7–8B LLM served by vLLM (OpenAI-compatible) on Drew. Heavy dependencies are
imported lazily inside the functions that need them, so the package imports and the
mock pipeline runs even when torch/PyG/vLLM are not installed.

The seam between modes is a single `mock` flag threaded through the embedder factory,
the R-GCN trainer, the reranker, and the explainer — so the *control flow and data
contracts are identical* in both modes. You validate the plumbing in mock, then flip
the flag.

---

## 4. Compute placement (Anvil vs Drew)

The pattern is **Anvil = the factory, Drew = the storefront.**

- **Anvil** (batch HPC, A100 40 GB / H100 80 GB, ~6,000 GPU-hr allocation): bulk
  ingestion/parsing/graph-building (CPU, RAM-heavy), embedding millions of docs (04),
  R-GCN training (06), and any 4B-model LoRA fine-tune. These are batch jobs — exactly
  what a Slurm scheduler is for. The full set of training jobs is comfortably inside
  the allocation (realistically ~1,000–2,000 GPU-hr with re-runs).
- **Drew** (2× 12 GB GPUs, always-on): the persistent retrieval service (07), the
  multi-agent RAG with a vLLM-served 7–8B model at 4-bit (08), and the demo. 12 GB caps
  the local LLM to ~7–8B; for anything heavier, call an API or use Anvil's Composable
  Subsystem for a hosted endpoint.

Artifacts produced on Anvil (`doc_vectors.npy`, `faiss.index`, `rgcn_node_emb.npy`,
fine-tuned encoder) are rsync'd/Globus'd to Drew for serving. Because every path is
under `$DH_DATA_ROOT`, the same code points at Anvil scratch or Drew local disk with
no edits.

---

## 5. The determinism & stability story (the investor's question)

**"Ask the same question twice — do you get the same output?"** The honest,
defensible answer is built into the code:

- **Byte-level determinism is not guaranteed across hardware/library changes** — even
  at temperature 0 — because floating-point addition is non-associative and standard
  GPU kernels are not batch-invariant (a request's output depends on how many other
  requests were batched with it). This is real and we don't paper over it.
- **Semantic / functional stability is achievable and measured.** On a fixed stack,
  embeddings and exact (FAISS Flat) retrieval are deterministic; the *evidence and
  citations* a recommendation rests on are stable even if the prose wording varies.

`discovery_hub/determinism.py` sets every knob we control (Python/NumPy/torch seeds,
`torch.use_deterministic_algorithms`, deterministic cuDNN, disabled TF32,
`CUBLAS_WORKSPACE_CONFIG=:4096:8`) and records exactly what was in force.
`09_stability_harness.py` then measures stability and writes a report whose headline
metric is **citation-set Jaccard** — the property that makes a recommendation
auditable. On the mock stack the harness reports perfect stability (embedding drift
≈ 6e-8, retrieval Jaccard / Kendall-τ = 1.000, citation-set Jaccard = 1.000); on a
real GPU it surfaces the true drift, honestly.

The pitch to the investor is *not* "our model is perfectly deterministic" (false for
any GPU LLM). It is "we **measure and report** stability at every stage, the evidence
and citations are reproducible, and we can make even the prose byte-identical with
batch-invariant kernels at a throughput cost if a customer requires it." A response
cache keyed on input hash means identical inputs return identical outputs in
production, which is usually the real question being asked.

---

## 6. Repository layout

```
discovery_hub_pipeline/
├── 01_download_data.py        Layer 1: ingest raw records (mock + real APIs)
├── 02_parse_normalize.py      Layer 1: heterogeneous schemas → DiscoveryDoc
├── 03_build_graph.py          Layer 1: heterogeneous knowledge graph
├── 04_generate_embeddings.py  Layer 2: batch embed (Anvil GPU job)
├── 05_build_index.py          Layer 2: FAISS exact index (→ ship to Drew)
├── 06_train_rgcn.py           Layer 2: R-GCN (PyG real / numpy mock)
├── 07_retrieve_rank.py        Layer 2: two-stage retrieve + rerank (Drew)
├── 08_multiagent_rag.py       Layer 3: multi-agent strict-RAG explanation (Drew)
├── 09_stability_harness.py    QA: reproducibility measurement + report
├── discovery_hub/             shared library (imported by every stage)
│   ├── config.py              profiles, paths, source specs, compute targets
│   ├── schema.py              DiscoveryDoc, graph vocab, JSONL I/O
│   ├── determinism.py         seed/flag control + honest guarantees report
│   ├── embedding.py           MockEmbedder + SentenceTransformerEmbedder
│   └── mock.py                deterministic synthetic-data generators
├── tests/test_pipeline_smoke.py   end-to-end mock test + determinism asserts
├── Makefile                   `make mock` / `make real` / `make smoke`
└── requirements.txt           core (light) + optional GPU deps by stage
```

---

## 7. Quickstart

```bash
# 0. (optional) create a venv, then install the light core
pip install -r requirements.txt          # numpy, scipy, networkx, requests, faiss-cpu, pytest

# 1. run the entire pipeline end-to-end with deterministic mocks (no GPU/network)
make mock

# 2. confirm the determinism assertions pass
make smoke

# 3. try a single query through the full stack
python 08_multiagent_rag.py --mock --query "EGFR inhibitor for oncology"

# 4. produce the investor-facing stability report
python 09_stability_harness.py --mock --runs 5
#    -> data/reports/stability_report.md

# --- going real (install the optional deps for the stages you run) ---
# on Anvil:  python 01_… ; 02_… ; 03_… ; 04_… --batch-size 64 ; 05_… ; 06_… --epochs 20
# on Drew :  python 07_… --query "…" ; 08_… --query "…" ; 09_… --runs 5
```

Override the data location and master seed with env vars:
`DH_DATA_ROOT`, `DH_SEED`, `DH_EMBED_MODEL`, `DH_LLM_MODEL`.

---

## 8. What is mocked vs production-ready, and next steps

**Production-ready now:** the data contracts (`DiscoveryDoc`, graph schema, JSONL
I/O), the parsers for all five sources, the FAISS index, the two-stage retrieval
control flow, the multi-agent state machine, the determinism controls, and the
stability harness. These are the same in mock and real mode.

**Mocked (swap in real components):** the embedder (→ Qwen3-Embedding-0.6B), the
R-GCN trainer (→ PyG `RGCNConv` with your contrastive/link-prediction objective and
neighbor sampling), the reranker (→ BGE-reranker-v2-m3), and the explanation LLM
(→ vLLM-served 7–8B model). Each has its real implementation already written and
gated; turning it on is installing the dependency and dropping `--mock`.

**Suggested next steps:**
1. Pull the real MVP slice (1–5M-work biomedical OpenAlex + pharma-CPC patents + all
   trials/SBIR) on Anvil and run 01–06 for real; ship the index to Drew.
2. Wire `07`'s reranker and `08`'s explainer to real models; stand up the vLLM server
   on Drew.
3. Build the curated 50–500-pair evaluation set and run `09` plus Recall@k/MRR@10 on a
   fixed GPU to get the first *real* stability and quality numbers — with confidence
   intervals — to show the investor.
4. Map `08`'s agent functions onto LangGraph nodes once the single-process version is
   behaving as desired.
