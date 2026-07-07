"""
Reciprocal Rank Fusion (RRF) -- how the dense, keyword, and graph retrievers are
combined.

Why RRF instead of a weighted sum of scores: BM25 scores, cosine similarities,
and graph dot products live on completely different scales, so a convex
combination needs fragile per-signal normalization and a tuned weight. RRF uses
only the RANK each retriever assigns, so it is scale-free and robust, and it
handles a missing retriever for free -- when the graph signal abstains on a
query, its list simply isn't in the fusion. This is the standard,
well-established hybrid-retrieval fusion (Cormack et al., 2009).

    rrf_score(d) = sum over retrievers r of 1 / (k + rank_r(d))

with rank_r 1-based and k a damping constant (60 is the common default). Pure and
deterministic; unit-tested in tests/test_hybrid.py.
"""
from __future__ import annotations


def reciprocal_rank_fusion(ranked_lists: dict[str, list[str]],
                           k: int = 60) -> dict[str, float]:
    """
    ranked_lists: name -> list of doc_ids (best first). Returns doc_id ->
    fused score. A document absent from a list contributes nothing from it.
    """
    fused: dict[str, float] = {}
    for lst in ranked_lists.values():
        for rank, doc_id in enumerate(lst, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return fused


def fuse_to_pool(ranked_lists: dict[str, list[str]], top_k: int,
                 k: int = 60) -> list[tuple[str, float]]:
    """RRF then take the top_k by (score desc, doc_id asc) for deterministic order."""
    fused = reciprocal_rank_fusion(ranked_lists, k=k)
    ranked = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
    return ranked[:top_k]
