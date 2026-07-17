"""
dh2.parallel_tracks_stubs -- INTERFACE STUBS for the spec's parallel/expansion tracks
(P2-P8). These are intentionally NOT implemented: the delivered pipeline_2 builds the
P0+P1 critical path (the highest-leverage, clearest go/no-go work). Each stub documents
the intended interface so the tracks can be filled in later without redesigning.

Everything here raises NotImplementedError with a pointer to the spec section. This keeps
the package import-clean and makes the roadmap explicit in code.
"""
from __future__ import annotations


def truncation_audit(*_a, **_k):
    """P2 (S7): measure per-source token length + evidence position; decide if the 512
    limit loses material evidence before committing to multi-vector field/chunk indexing."""
    raise NotImplementedError("P2 truncation audit — see spec S7 / Priority 2. Not in the "
                              "P0+P1 critical-path build.")


def source_query_rewrite(*_a, **_k):
    """P3 (S7): generate source-style query variants (patent/trial/paper/SBIR) + HyDE as
    ADDITIONAL channels; RRF-fuse with the original query (never replace it)."""
    raise NotImplementedError("P3 query rewrite/HyDE — see spec Priority 3.")


def build_alias_index(*_a, **_k):
    """P3 (S7): offline scout-language aliases per doc, encoded query-side, collapsed by
    document_id. Must be isolated from held-out eval queries (anti-leakage)."""
    raise NotImplementedError("P3 alias index — see spec Priority 3.")


def reranker_ablation(*_a, **_k):
    """P4 (S7): Qwen3-Reranker-4B (online) vs BGE vs ensembles on the same top-100 lists.
    Gate on Recall@100 first (reranker can't fix candidates that never enter the pool)."""
    raise NotImplementedError("P4 reranker A/B — see spec Priority 4.")


def conditional_splade(*_a, **_k):
    """P6 (S7): learned sparse expansion routed ONLY for rare-entity queries (drug/gene/
    NCT/patent numbers). Candidate channel only — NOT a fixed dense-sparse score blend."""
    raise NotImplementedError("P6 conditional SPLADE — see spec Priority 6.")


def matryoshka_dim_sweep(*_a, **_k):
    """P7 (S7): evaluate the 4B model at 2560/1024/768/512/256 dims for vector/index
    efficiency (NOT transformer-weight compression). Confirm LoRA preserved nesting."""
    raise NotImplementedError("P7 Matryoshka dims — see spec Priority 7.")


def kg_typed_expansion(*_a, **_k):
    """P8 (S7): typed one-hop graph expansion + structural reranking on a relational-query
    slice. Must prove value before any R-GCN (the BM25 lesson)."""
    raise NotImplementedError("P8 KG expansion — see spec Priority 8.")
