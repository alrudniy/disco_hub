"""
Central configuration for the Discovery Hub pipeline.

This file is the single place that encodes the two scaling profiles (MVP vs FULL),
the two compute targets (Anvil batch HPC vs the Drew always-on server), all on-disk
paths, and the per-source data specs. Every numbered script imports from here so the
pipeline has one source of truth.

COMPUTE PLACEMENT (see the engineering memo for the full rationale):

    Stage                          Target   Why
    -----------------------------  -------  ---------------------------------------
    01 download / 02 parse         Anvil*   I/O + RAM heavy, not GPU heavy
    03 build graph                 Anvil*   RAM heavy (entity resolution)
    04 generate embeddings         Anvil    A100/H100 batch job (millions of docs)
    05 build index                 Anvil    CPU; ships the index to Drew
    06 train R-GCN                 Anvil    single A100 w/ neighbor sampling
    07 retrieve + rank             Drew     always-on retrieval service
    08 multi-agent RAG             Drew     always-on LLM serving (vLLM)
    09 stability harness           either   runs anywhere; pin hardware when measuring

    (*) The MVP biomedical slice is < ~200 GB and fits on Drew, so for the MVP you
        can run 01-06 on Drew too. Full scale (~2-3 TB working set) needs Anvil.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Global knobs
# --------------------------------------------------------------------------- #
SEED = int(os.environ.get("DH_SEED", "20240611"))

# Embedding model. Qwen3-Embedding-0.6B is the production target (fits in 12 GB).
# The mock embedder produces deterministic vectors of EMBED_DIM so the whole
# pipeline runs without a GPU and is bit-reproducible across machines.
EMBED_MODEL = os.environ.get("DH_EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B")
EMBED_DIM = int(os.environ.get("DH_EMBED_DIM", "1024"))

# Qwen3-Embedding is instruction-aware and ASYMMETRIC: queries are prefixed with
# "Instruct: {task}\nQuery: " while documents are embedded raw (no instruction).
# This task description is what tells the model to bridge the register gap between
# a clinically/commercially phrased research interest (query) and the legal/
# technical language of a patent or trial record (document) -- the core matching
# challenge of this project. Override via DH_QUERY_INSTRUCTION.
QUERY_INSTRUCTION = os.environ.get(
    "DH_QUERY_INSTRUCTION",
    "Given a pharmaceutical or biotech company's research interest, retrieve "
    "patents, clinical trials, and university inventions that are technically "
    "relevant and potentially available for licensing or partnership.")

# Reranker used in the second retrieval stage (real mode only).
RERANK_MODEL = os.environ.get("DH_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")

# Self-hosted explanation LLM for Layer 3 (Drew, 12 GB => ~7-8B at 4-bit).
LLM_MODEL = os.environ.get("DH_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")

# --------------------------------------------------------------------------- #
# Paths -- everything lives under DH_DATA_ROOT (override with an env var so the
# same code points at Drew local disk or Anvil scratch without edits).
# --------------------------------------------------------------------------- #
DATA_ROOT = Path(os.environ.get("DH_DATA_ROOT", "./data")).resolve()

RAW_DIR = DATA_ROOT / "raw"            # 01 -> heterogeneous source records (JSONL)
NORM_DIR = DATA_ROOT / "normalized"   # 02 -> unified DiscoveryDoc records (JSONL)
GRAPH_DIR = DATA_ROOT / "graph"       # 03 -> nodes.jsonl, edges.jsonl, graph_meta.json
EMB_DIR = DATA_ROOT / "embeddings"    # 04 -> doc_vectors.npy + doc_ids.json
INDEX_DIR = DATA_ROOT / "index"       # 05 -> faiss.index (or mock_index.npz)
ARTIFACT_DIR = DATA_ROOT / "artifacts"  # 06 -> rgcn_node_emb.npy + node_ids.json
REPORT_DIR = DATA_ROOT / "reports"    # 09 -> stability_report.json / .md

ALL_DIRS = [RAW_DIR, NORM_DIR, GRAPH_DIR, EMB_DIR, INDEX_DIR, ARTIFACT_DIR, REPORT_DIR]


def ensure_dirs() -> None:
    """Create every output directory. Safe to call repeatedly."""
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Data sources. `mvp_records` is what 01 pulls in --mvp mode; `full_note`
# documents the real full-scale volume so the numbers from the memo stay
# attached to the code.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SourceSpec:
    name: str
    license: str
    base_url: str
    mvp_records: int
    full_note: str


SOURCES: dict[str, SourceSpec] = {
    "clinicaltrials": SourceSpec(
        name="ClinicalTrials.gov v2",
        license="public domain",
        base_url="https://clinicaltrials.gov/api/v2/studies",
        mvp_records=2000,
        full_note="~590k studies total; small (low single-digit GB).",
    ),
    "sbir": SourceSpec(
        name="SBIR.gov awards",
        license="public domain",
        base_url="https://api.www.sbir.gov/public/api/awards",
        mvp_records=2000,
        full_note="Full bulk file is 290 MB with abstracts; just download it all.",
    ),
    "uspto": SourceSpec(
        name="USPTO / PatentsView",
        license="CC-BY 4.0",
        base_url="https://search.patentsview.org/api/v1/patent/",
        mvp_records=3000,
        full_note="Granted ~100 GB + pre-grant ~26 GB; ~500 GB unzipped full text.",
    ),
    "openalex": SourceSpec(
        name="OpenAlex works",
        license="CC0",
        base_url="https://api.openalex.org/works",
        mvp_records=5000,
        full_note="Snapshot ~330 GB gzip -> ~1.6 TB; ~250M works. Filter to the "
        "~43M medicine+biology subset, or 1-5M for the MVP.",
    ),
    "autm": SourceSpec(
        name="AUTM Innovation Marketplace",
        license="portal terms",
        base_url="",  # no clean bulk API; provided as pre-scraped JSONL by students
        mvp_records=1000,
        full_note="31k+ listings; respect portal terms before redistributing.",
    ),
}

# Biomedical concept filter for OpenAlex (Topic/Concept ids). These are the
# top-level fields used to cut the 250M-work corpus down to the ~43M slice.
OPENALEX_BIOMED_CONCEPTS = {
    "C71924100": "Medicine",
    "C86803240": "Biology",
    "C185592680": "Chemistry",
    "C70721500": "Pharmacology",
}

# --------------------------------------------------------------------------- #
# Retrieval / RAG knobs
# --------------------------------------------------------------------------- #
@dataclass
class RetrievalConfig:
    top_k_recall: int = 50      # first-stage retrieval depth per signal
    top_k_rerank: int = 10      # after cross-encoder rerank
    min_confidence: float = 0.35  # Layer-3 policy gate
    # Hybrid retrieval: dense + BM25 keyword, fused by Reciprocal Rank Fusion,
    # with an entity-linked graph signal that abstains when nothing links.
    use_keyword: bool = True    # include the BM25 keyword half (the "hybrid")
    # Graph signal OFF by default: on the only evidence we have (stage 10 over the
    # synthetic eval), the entity-linked R-GCN signal does NOT improve retrieval
    # and slightly hurts it -- anchoring on a coarse entity (e.g. an org with many
    # patents) pulls structurally-adjacent-but-irrelevant docs into the fused pool.
    # It remains fully implemented and one flag away; re-enable and re-measure on a
    # richer real graph (fine-grained entities, concept nodes, a learned query->
    # graph encoder) before trusting it. See sample_outputs/DECK_CLAIM_RECALIBRATION.md.
    use_graph: bool = False
    rrf_k: int = 60             # RRF damping constant
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    graph_weight: float = 0.25  # (deprecated: legacy linear blend; RRF is used now)


RETRIEVAL = RetrievalConfig()
