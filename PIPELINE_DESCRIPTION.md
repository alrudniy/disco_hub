# Discovery Hub Pipeline — Description

A numbered, end-to-end pipeline that implements the three-layer Discovery Hub
architecture (Evidence Integration → Retrieval & Ranking → Explanation & Workflow)
as **ten** runnable stages. Every stage has a deterministic `--mock` mode, so the
whole system runs, tests, and demonstrates its reproducibility **without a GPU or
any network download**, exactly the way `discovery_finetune` is validated against
real files with a mock embedder. The real-mode code (live APIs, GPU models, R-GCN,
vLLM) is present in each stage, gated behind lazy imports.

Retrieval is **hybrid** — dense vectors + BM25 keyword scoring fused with
Reciprocal Rank Fusion, then neural reranking — and the system ships with a
**retrieval-evaluation harness** (Recall@k / MRR / nDCG with bootstrap confidence
intervals and paired significance tests) and a **citation-faithfulness verifier**,
so quality and trust claims are measured rather than asserted.

---

## 1. How the code maps to the three-layer architecture

| Deck layer | What it does | Stages here |
|---|---|---|
| **Layer 1 — Evidence Integration** | Ingest patents/pubs/trials/org signals into a translation-oriented knowledge graph (technologies, inventors, organizations, experts, facilities) | `01` download · `02` parse/normalize · `03` build graph |
| **Layer 2 — Retrieval & Ranking** | Hybrid semantic + keyword search, graph signal, and neural rerank; this is where the existing `discovery_finetune` package lives | `04` embeddings · `05` index (FAISS + BM25) · `06` R-GCN · `07` hybrid retrieve+rank |
| **Layer 3 — Explanation & Workflow** | Strict-RAG, multi-agent pipeline producing cited, provenance-stamped, confidence-scored recommendations with human-in-the-loop | `08` multi-agent RAG |
| **Cross-cutting — QA & Evaluation** | Measures run-to-run stability and citation faithfulness, and scores retrieval quality with confidence intervals | `09` stability + faithfulness harness · `10` retrieval eval |

The existing `discovery_finetune` work (schema normalization, synthetic-query
generation, hard-negative mining, MNRL fine-tuning of Qwen3-Embedding-0.6B,
Recall@k/MRR@10) is the **retrieval layer** inside this fuller vision. Stages
`02`, `04`, `05`, and `07` are where it plugs in; the new work is the knowledge
graph (`03`/`06`), the hybrid retrieval + evaluation machinery (`07`/`10`), and the
multi-agent explanation layer (`08`).

---

## 2. The ten stages, in execution order

Run them in numeric order; each consumes the previous stage's artifacts under
`$DH_DATA_ROOT` (default `./data`).

| # | Script | Input → Output | Compute target |
|---|---|---|---|
| 01 | `01_download_data.py` | source APIs → `data/raw/*.jsonl` | Anvil CPU* |
| 02 | `02_parse_normalize.py` | `raw/*` → `normalized/docs.jsonl` (unified `DiscoveryDoc`) | Anvil CPU* |
| 03 | `03_build_graph.py` | `docs.jsonl` → `graph/{nodes,edges}.jsonl` + meta | Anvil CPU* |
| 04 | `04_generate_embeddings.py` | `docs.jsonl` → `embeddings/doc_vectors.npy` | **Anvil GPU** |
| 05 | `05_build_index.py` | vectors → `index/faiss.index` + `index/bm25.json` | Anvil CPU → ship to Drew |
| 06 | `06_train_rgcn.py` | graph → `artifacts/rgcn_node_emb.npy` + link-pred eval | **Anvil GPU** |
| 07 | `07_retrieve_rank.py` | index + BM25 + R-GCN → ranked candidates | **Drew** (always-on) |
| 08 | `08_multiagent_rag.py` | candidates → cited recommendations | **Drew** (always-on) |
| 09 | `09_stability_harness.py` | runs 07/08 K times → stability + faithfulness report | either (pin hardware) |
| 10 | `10_eval_retrieval.py` | ranked results + qrels → Recall@k / MRR / nDCG with CIs | either |

`*` For the MVP biomedical slice (< ~200 GB) stages 01–06 run fine on Drew too;
full scale (~2–3 TB working set) needs Anvil.

**What each stage does**

- **01 — Download.** Pulls records from ClinicalTrials.gov v2, OpenAlex, SBIR,
  USPTO, and AUTM. `--mock` generates a deterministic synthetic corpus shaped like
  each real schema (nested `protocolSection` for trials, flat dicts for USPTO,
  OpenAlex's inverted-index abstracts) so stage 02 exercises real code paths. The
  USPTO fetcher uses the PatentSearch API (keyed via `PATENTSVIEW_API_KEY`, filtered
  to pharma CPC subclasses A61K/A61P with cursor pagination); a `--uspto-backend
  odp-bulk` fallback targets the USPTO Open Data Portal (`api.uspto.gov`) for the
  ongoing PatentSearch→ODP migration. For full-scale OpenAlex, use the S3 snapshot,
  not the REST API.
- **02 — Parse & normalize.** One parser per source converges every schema onto the
  unified `DiscoveryDoc` with a canonical `embedding_text`. Includes the AUTM noise
  filter (`--min-chars`). This is the contract that lets the students' data-collection
  work proceed in parallel behind one schema.
- **03 — Build graph.** Constructs the heterogeneous KG: `technology` nodes (keyed by
  `doc_id` — each invention is unique) plus deduplicated `inventor`/`organization`/
  `expert`/`facility` nodes, connected by typed relations (`invented_by`, `assigned_to`,
  `investigated_by`, `located_at`, `affiliated_with`). Validates with networkx.
- **04 — Embeddings.** Batch-embeds every `embedding_text`. **Asymmetric by design**:
  documents are embedded raw (`encode_documents`), while queries get the Qwen3-Embedding
  instruction prefix `Instruct: {task}\nQuery: {q}` (`encode_queries`) — the documented
  contract for that model, and the bridge across the query/document register gap.
  The Anvil A100/H100 job at full scale (~30–100 A100-hours for 40–50M abstracts).
  Mock embedder is bit-identical across machines.
- **05 — Index.** Builds a FAISS `IndexFlatIP` (exact, deterministic) — chosen for the
  reproducibility story; a numpy brute-force fallback runs if FAISS is absent. **Also
  builds a BM25 inverted index** (`bm25.json`) over the same corpus, so stage 07 can
  fuse lexical and semantic retrieval. Built on Anvil, shipped to Drew.
- **06 — R-GCN.** Learns relational node embeddings. Real path: PyTorch Geometric
  `RGCNConv` trained with link prediction over a no-leakage edge split, scored with
  ROC-AUC (tie-aware Mann-Whitney) / Hits@k / MRR. Mock path: deterministic numpy
  relational mean-aggregation. With neighbor sampling this fits a 12 GB GPU; a run is
  ~1–6 GPU-hours.
- **07 — Hybrid retrieve & rank.** Three retrieval signals fused with **Reciprocal
  Rank Fusion**: dense vectors (FAISS), **BM25 keyword** scoring, and an
  **entity-linked graph signal** (a query is linked to graph nodes by surface form;
  it abstains cleanly when nothing links). The fused shortlist is reranked (BGE
  cross-encoder in real mode, fused-score order in mock). Returns candidates with
  supporting evidence and per-signal scores. Importable `Retriever` class. See the
  evidence note below for why the graph signal is *off by default* as a retrieval
  feature.
- **08 — Multi-agent RAG.** Five agents in sequence — retrieval, expertise-gap,
  reranking, policy-safety, explanation — implemented framework-free as a state dict
  that maps 1:1 onto LangGraph nodes. Strict RAG: every claim carries a citation and
  provenance. Confidence is read off the **dense semantic similarity** of the top
  evidence (a calibrated [0,1] signal), and a policy gate refuses to assert a
  recommendation below the confidence threshold, routing it to human review instead.
- **09 — Stability & faithfulness harness.** Runs the same queries K times and reports
  embedding drift, retrieval Jaccard / Kendall-τ, and LLM exact-match /
  semantic-equivalence / **citation-set Jaccard**. It additionally measures
  **batch-size and device invariance** of the embedder (the dominant real-mode
  nondeterminism source, invisible to same-condition repeats) and runs a
  **citation-faithfulness verifier** — citation validity plus claim groundedness —
  over the generated answers. Emits `reports/stability_report.{json,md}`.
- **10 — Retrieval eval.** Scores retrieval quality with **Recall@k, MRR@10, nDCG@10,
  Precision@k, and MAP**, each reported as a measured lift with a **95% bootstrap
  confidence interval and a paired Wilcoxon p-value** — the honest form of "+20%
  Precision". Runs on synthetic labels out of the box (one positive per query) and on
  curated relevance judgments via `--qrels`. Emits `reports/eval_report.{json,md}`.

> **What the retrieval evidence shows (and a recalibrated claim).** On a 200-query
> synthetic benchmark over the mock corpus, **hybrid retrieval (dense + BM25) plus
> reranking beats dense-only decisively** — Recall@50 rises from ~0.61 to ~0.78, with
> every metric's improvement significant at p < 0.05. Adding the **graph signal as a
> retrieval feature did *not* help** (it slightly hurt Recall@50, p ≈ 0.005), so it is
> disabled by default in retrieval (`use_graph=False`). This is a deliberate,
> evidence-based recalibration: the knowledge graph's value is as **provenance and
> evidence infrastructure** (the expertise/organization backing behind each
> recommendation in stage 08), not as a proven retrieval-ranking signal — and the gains
> are credited to hybrid search + reranking, which the numbers support.

---

## 3. Two run modes

**Mock mode (`--mock`)** — deterministic, no GPU, no network. Synthetic data,
hash-seeded embeddings, BM25 over the synthetic corpus, numpy message-passing for
the R-GCN, and a templated strict-RAG explainer. This is what CI and the smoke test
use, and what proves the reproducibility claims. Everything is bit-reproducible
across machines.

**Real mode (default)** — live source APIs; Qwen3-Embedding-0.6B via
sentence-transformers (with the query instruction prefix, left-padding, and
flash-attention hooks); PyTorch Geometric R-GCN; BGE-reranker-v2-m3; a self-hosted
7–8B LLM served by vLLM (OpenAI-compatible) on Drew. The faithfulness verifier's
default lexical scorer is a deterministic CI floor; a real NLI / LLM-judge
(RAGAS / HHEM-style) is a one-argument swap. Heavy dependencies are imported lazily
inside the functions that need them, so the package imports and the mock pipeline
runs even when torch/PyG/vLLM are not installed.

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

Artifacts produced on Anvil (`doc_vectors.npy`, `faiss.index`, `bm25.json`,
`rgcn_node_emb.npy`, fine-tuned encoder) are rsync'd/Globus'd to Drew for serving.
Because every path is under `$DH_DATA_ROOT`, the same code points at Anvil scratch or
Drew local disk with no edits.

---

## 5. The determinism & stability story (the investor's question)

**"Ask the same question twice — do you get the same output?"** The honest,
defensible answer is built into the code:

- **Byte-level determinism is not guaranteed across hardware/library changes** — even
  at temperature 0 — because floating-point addition is non-associative and standard
  GPU kernels are not batch-invariant (a request's output depends on how many other
  requests were batched with it). This is real and we don't paper over it; the harness
  **measures** the batch-size/device drift instead of hiding it.
- **Semantic / functional stability is achievable and measured.** On a fixed stack,
  embeddings and exact (FAISS Flat) retrieval are deterministic; the *evidence and
  citations* a recommendation rests on are stable even if the prose wording varies —
  and the citations are now verified *correct*, not merely stable.

`discovery_hub/determinism.py` sets every knob we control (Python/NumPy/torch seeds,
`torch.use_deterministic_algorithms`, deterministic cuDNN, disabled TF32,
`CUBLAS_WORKSPACE_CONFIG=:4096:8`) and records exactly what was in force.
`09_stability_harness.py` then measures stability and writes a report. On the mock
stack the harness reports:

| Stage | Metric | Mock result | Ideal |
|---|---|---|---|
| Embedding | max cosine drift (same conditions) | ≈ 6e-8 | ~0 |
| Embedding | drift across batch sizes {1, 8, 32} | ≈ 6e-8 (batch-invariant) | ~0 on fixed stack |
| Retrieval | top-k set Jaccard / Kendall-τ | 1.000 | 1.000 |
| Explanation | citation-set Jaccard (stability) | 1.000 | 1.000 |
| Explanation | citation validity / groundedness | 1.000 / 1.000 | high |
| Explanation | hallucination rate | 0.000 | low |

The mock embedder is batch- and device-invariant *by construction*, so those rows are
0; on a real GPU the same rows surface the true production drift, honestly. The
faithfulness numbers are optimistic in mock (the templated explainer cites by
construction) — the verifier's job is to score the *real* LLM, and its unit tests prove
it catches injected hallucinations and fabricated citations. The policy gate's
`refusal_rate` is high in mock because generic demo queries match a random synthetic
corpus weakly; that is the gate working, not a fault.

The pitch to the investor is *not* "our model is perfectly deterministic" (false for
any GPU LLM). It is "we **measure and report** stability *and citation faithfulness* at
every stage, the evidence and citations are reproducible and verified, and we can make
even the prose byte-identical with batch-invariant kernels at a throughput cost if a
customer requires it." A response cache keyed on input hash means identical inputs
return identical outputs in production, which is usually the real question being asked.

---

## 6. Repository layout

```
discovery_hub_pipeline/
├── 01_download_data.py        Layer 1: ingest raw records (mock + real APIs)
├── 02_parse_normalize.py      Layer 1: heterogeneous schemas → DiscoveryDoc
├── 03_build_graph.py          Layer 1: heterogeneous knowledge graph
├── 04_generate_embeddings.py  Layer 2: batch embed, query/doc-asymmetric (Anvil GPU)
├── 05_build_index.py          Layer 2: FAISS exact index + BM25 (→ ship to Drew)
├── 06_train_rgcn.py           Layer 2: R-GCN + link-prediction eval (PyG / numpy mock)
├── 07_retrieve_rank.py        Layer 2: hybrid RRF retrieve + rerank (Drew)
├── 08_multiagent_rag.py       Layer 3: multi-agent strict-RAG explanation (Drew)
├── 09_stability_harness.py    QA: stability + batch-invariance + faithfulness report
├── 10_eval_retrieval.py       QA: Recall@k / MRR / nDCG with bootstrap CIs
├── discovery_hub/             shared library (imported by every stage)
│   ├── config.py              profiles, paths, source specs, retrieval config
│   ├── schema.py              DiscoveryDoc, graph vocab, JSONL I/O
│   ├── determinism.py         seed/flag control + honest guarantees report
│   ├── embedding.py           Mock + SentenceTransformer (encode_documents/queries)
│   ├── keyword.py             BM25 inverted index
│   ├── fusion.py              Reciprocal Rank Fusion
│   ├── entity_link.py         query → graph-node surface linking
│   ├── graph_eval.py          link-prediction split + AUC / Hits@k / MRR
│   ├── evaluate.py            IR metrics + bootstrap CI + paired Wilcoxon
│   ├── faithfulness.py        citation validity + claim groundedness verifier
│   └── mock.py                deterministic synthetic-data generators
├── tests/                     44 tests across 7 files (smoke, eval, graph eval,
│                              hybrid, embedding instruction, R-GCN agg, faithfulness)
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

# 2. confirm the determinism assertions pass (full suite: 44 tests)
make smoke

# 3. try a single query through the full stack
python 08_multiagent_rag.py --mock --query "EGFR inhibitor for oncology"

# 4. produce the investor-facing stability + faithfulness report
python 09_stability_harness.py --mock --runs 5      # -> data/reports/stability_report.md

# 5. score retrieval quality with confidence intervals
python 10_eval_retrieval.py --mock                  # -> data/reports/eval_report.md

# --- going real (install the optional deps for the stages you run) ---
# on Anvil:  01 ; 02 ; 03 ; 04 --batch-size 64 ; 05 ; 06 --epochs 20
# on Drew :  07 --query "..." ; 08 --query "..." ; 09 --runs 5 ; 10 --qrels curated.jsonl
```

Override the data location and master seed with env vars:
`DH_DATA_ROOT`, `DH_SEED`, `DH_EMBED_MODEL`, `DH_LLM_MODEL`, `DH_QUERY_INSTRUCTION`.

---

## 8. What is mocked vs production-ready, and next steps

**Production-ready now:** the data contracts (`DiscoveryDoc`, graph schema, JSONL
I/O), the parsers for all five sources (including the keyed USPTO PatentSearch
fetcher), the FAISS + BM25 indexes, the **hybrid RRF retrieval** control flow with
entity-linked graph signal, the multi-agent state machine with its calibrated
confidence gate, the determinism controls, the **stability + batch-invariance +
faithfulness harness**, and the **retrieval-evaluation harness with bootstrap CIs and
significance tests**. These are the same in mock and real mode.

**Mocked (swap in real components):** the embedder (→ Qwen3-Embedding-0.6B), the
R-GCN trainer (→ PyG `RGCNConv` with your contrastive/link-prediction objective and
neighbor sampling), the reranker (→ BGE-reranker-v2-m3), the explanation LLM
(→ vLLM-served 7–8B model), and the faithfulness scorer (→ NLI / LLM judge). Each has
its real implementation already written and gated; turning it on is installing the
dependency and dropping `--mock`.

**Known gap:** the USPTO ODP-bulk backend lists and downloads the bulk file-wrapper
archives, but a parser converting that JSON shape into `DiscoveryDoc`s is not yet
written (it fails honestly with a note rather than emitting bad data); the default
PatentSearch backend is unaffected.

**Suggested next steps:**
1. Pull the real MVP slice (1–5M-work biomedical OpenAlex + pharma-CPC patents + all
   trials/SBIR) on Anvil and run 01–06 for real; ship the FAISS + BM25 indexes to Drew.
2. Wire `07`'s reranker and `08`'s explainer to real models; stand up the vLLM server
   on Drew, and swap the faithfulness scorer to a real NLI/LLM judge.
3. Build the curated 50–500-pair evaluation set and re-run `10` (the harness already
   exists) plus `09` on a fixed GPU to get the first *real* quality, stability, and
   faithfulness numbers — with confidence intervals — to show the investor.
4. Map `08`'s agent functions onto LangGraph nodes once the single-process version is
   behaving as desired.
