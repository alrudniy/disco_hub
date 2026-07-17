"""
dh2.cascade -- the Register-Aware Semantic Cascade (RASC).

    scout query
        |
        +-- untouched Qwen3-Embedding-0.6B  top 100   semantic hedge
        +-- fine-tuned 8B                   top 100   domain/lexical specialist
        +-- fine-tuned 4B                   top 50    optional diversity
        +-- ReasonEmbed-8B                  top 100   optional, gated
        +-- exact-identifier channel                  NCT/patent/CAS/compound only
                    |
        deduplicate, preserve channel provenance
                    |
        Qwen3-Reranker-8B, official prompt, 8K context  -> top 40
                    |
        Diver-GroupRank-32B, random groupwise           -> top 10   (flagged, gated)

INFERENCE ONLY. Nothing here trains anything, and that is the point: the 2026-07-15 eval
found the best trained arm gained +0.165 R@10 on high-overlap queries while LOSING 0.089
on low-overlap ones, because the synthetic queries were generated from their origin
documents and inherited their vocabulary. Until the query generator produces scout-register
queries, another fine-tune optimizes the same artifact. The union + corrected reranking is
the part that can be defended on measurements this project already has.

The one structural guarantee: an uncapped candidate union cannot have lower
relevant-document recall than either constituent retriever. Everything after the union --
the cap, the reranking, the groupwise pass -- must be promoted by measured utility on
independently judged data, particularly low-overlap queries. See CascadeGates.

Failure is always downward, never sideways:
    GroupRank failure -> Qwen ordering
    Qwen failure      -> union ordering by best retrieval rank
    union failure     -> whatever channels did return
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from dh2 import config2 as C
from dh2.candidate_pool import (CH_EXACT, CH_FT_4B, CH_FT_8B, CH_REASON, CH_SEMANTIC,
                                Candidate, build_union)


@dataclass
class Result:
    """One ranked document, carrying the provenance that produced it."""
    doc_id: str
    rank: int = 0
    channels: list[str] = field(default_factory=list)
    ranks: dict[str, int] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)
    qwen_logit_margin: float | None = None
    qwen_probability: float | None = None
    group_score: float | None = None

    def to_dict(self) -> dict:
        return {
            "document_id": self.doc_id,
            "rank": self.rank,
            "channels": self.channels,
            "ranks": self.ranks,
            "scores": {k: round(v, 5) for k, v in self.scores.items()},
            "qwen_logit_margin": self.qwen_logit_margin,
            "qwen_probability": self.qwen_probability,
            "group_score": self.group_score,
        }


@dataclass
class CascadeTrace:
    """Per-query diagnostics. This is how the gates get measured, so it is not optional."""
    union_size: int = 0
    per_channel_counts: dict[str, int] = field(default_factory=dict)
    qwen_ran: bool = False
    grouprank_ran: bool = False
    grouprank_parse_success: float | None = None
    latency_ms: dict[str, float] = field(default_factory=dict)
    fell_back_to: str | None = None


class RegisterAwareCascade:
    """Union -> Qwen rerank -> optional GroupRank. Inference only.

    retrievers: {channel_name: DenseRetriever|None}. CH_SEMANTIC (the UNTOUCHED 0.6B) is
                the load-bearing one; omitting it reproduces the low-overlap regression
                this class exists to fix, so it warns loudly rather than failing silently.
    doc_lookup: doc_id -> document object/dict (for source-aware views).
    """

    def __init__(self,
                 retrievers: dict[str, Any],
                 doc_lookup: dict[str, Any] | Callable[[str], Any] | None = None,
                 *,
                 cfg: C.CascadeConfig = C.CASCADE,
                 qwen_teacher: Any = None,
                 group_ranker: Any = None,
                 identifier_index: dict[str, list[str]] | None = None,
                 enable_grouprank: bool | None = None):
        self.retrievers = retrievers
        self.doc_lookup = doc_lookup
        self.cfg = cfg
        self._qwen = qwen_teacher
        self._grouprank = group_ranker
        self.identifier_index = identifier_index
        self.enable_grouprank = (cfg.enable_grouprank if enable_grouprank is None
                                 else enable_grouprank)
        if retrievers.get(CH_SEMANTIC) is None:
            print("[cascade] WARNING: no untouched-semantic channel (base_06b). The union "
                  "is fine-tuned-only, which is the configuration measured to lose 0.089 "
                  "R@10 on low-overlap queries. Low-overlap results will be missing.")

    # ---- lazy heavy deps ---------------------------------------------------- #
    @property
    def qwen(self):
        if self._qwen is None:
            from dh2.teachers import QwenTeacher
            self._qwen = QwenTeacher()
        return self._qwen

    @property
    def group_ranker(self):
        if self._grouprank is None:
            from dh2.grouprank import GroupRanker
            self._grouprank = GroupRanker(cfg=self.cfg)
        return self._grouprank

    def _depths(self) -> dict[str, int]:
        return {
            CH_SEMANTIC: self.cfg.depth_base_06b,
            CH_FT_8B: self.cfg.depth_ft_8b,
            CH_FT_4B: self.cfg.depth_ft_4b,
            CH_REASON: (self.cfg.depth_reasonembed
                        if self.retrievers.get(CH_REASON) is not None else 0),
        }

    def _doc(self, doc_id: str):
        if self.doc_lookup is None:
            return None
        if callable(self.doc_lookup):
            return self.doc_lookup(doc_id)
        return self.doc_lookup.get(doc_id)

    # ---- Stage 1 ------------------------------------------------------------ #
    def build_candidate_union(self, query: str) -> dict[str, Candidate]:
        """Retrieve every channel and union. No cross-model score comparison."""
        cand = build_union(query, self.retrievers, self._depths(),
                           identifier_index=(self.identifier_index
                                             if self.cfg.exact_identifier_channel else None),
                           deep_window=None)   # deep window is a training-pool device
        from dh2.candidate_pool import cap_union
        return cap_union(cand, self.cfg.max_union)

    # ---- Stage 2 ------------------------------------------------------------ #
    def _views(self, doc_ids: Sequence[str], max_tokens: int) -> dict[str, str]:
        from dh2.doc_views import render_view
        views: dict[str, str] = {}
        counter = None
        tok = getattr(self._qwen, "_tok", None)
        if tok is not None:
            def counter(t: str) -> int:                       # noqa: E306
                return len(tok.encode(t, add_special_tokens=False))
        for did in doc_ids:
            doc = self._doc(did)
            views[did] = (render_view(doc, max_tokens=max_tokens, token_counter=counter)
                          if doc is not None else "")
        return views

    # ---- Stage 3 ------------------------------------------------------------ #
    def qwen_rerank(self, query: str, pool: dict[str, Candidate]) -> list[Result]:
        """Rerank the union with Qwen. Ranks on the LOGIT MARGIN, not the probability.

        Probability saturates: grade-3 pairs average 0.802 (low-overlap) and 0.921
        (high-overlap), so the top of the list compresses into a few thousandths and the
        ordering that decides a top-10 becomes numerical noise. The margin does not
        saturate. Both are stored; only the margin ranks.
        """
        ids = list(pool)
        results = [self._result_from(pool[d]) for d in ids]
        if not ids:
            return results
        try:
            views = self._views(ids, self.cfg.qwen_view_tokens)
            pairs = [(query, views.get(d, "")) for d in ids]
            detail = self.qwen.score_detailed(pairs)
        except Exception as e:                                  # noqa: BLE001
            print(f"[cascade] Qwen rerank failed ({e}); falling back to dense ordering")
            results.sort(key=lambda r: min(r.ranks.values()) if r.ranks else 10**9)
            return self._renumber(results)
        for r, d in zip(results, detail):
            r.qwen_logit_margin = d["logit_margin"]
            r.qwen_probability = d["probability"]
        results.sort(key=lambda r: (-(r.qwen_logit_margin if r.qwen_logit_margin
                                      is not None else -1e9),
                                    min(r.ranks.values()) if r.ranks else 10**9))
        return self._renumber(results)

    # ---- Stage 4 ------------------------------------------------------------ #
    def group_rerank(self, query: str, shortlist: list[Result]) -> list[Result]:
        """Groupwise rescoring of the Qwen shortlist. Fails closed to `shortlist`."""
        if not shortlist:
            return shortlist
        ids = [r.doc_id for r in shortlist]
        try:
            views = self._views(ids, self.cfg.group_view_tokens)
            gr = self.group_ranker.rank(query, ids, views)
        except Exception as e:                                  # noqa: BLE001
            print(f"[cascade] GroupRank failed ({e}); keeping Qwen order")
            return shortlist
        if not gr.ok:
            return shortlist                                    # already logged its reason
        by_id = {r.doc_id: r for r in shortlist}
        for did, s in gr.scores.items():
            by_id[did].group_score = s
        ordered = sorted(shortlist,
                         key=lambda r: (-(r.group_score if r.group_score is not None
                                          else -1e9),
                                        -(r.qwen_logit_margin if r.qwen_logit_margin
                                          is not None else -1e9),
                                        min(r.ranks.values()) if r.ranks else 10**9))
        return self._renumber(ordered)

    # ---- top level ---------------------------------------------------------- #
    def retrieve(self, query: str, top_k: int = 10,
                 trace: CascadeTrace | None = None) -> list[Result]:
        t = trace if trace is not None else CascadeTrace()

        t0 = time.perf_counter()
        pool = self.build_candidate_union(query)
        t.union_size = len(pool)
        for c in pool.values():
            for ch in c.channels:
                t.per_channel_counts[ch] = t.per_channel_counts.get(ch, 0) + 1
        t.latency_ms["union"] = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        ranked = self.qwen_rerank(query, pool)
        t.qwen_ran = any(r.qwen_logit_margin is not None for r in ranked)
        t.latency_ms["qwen"] = (time.perf_counter() - t1) * 1000
        if not t.qwen_ran and ranked:
            t.fell_back_to = "dense"

        shortlist = ranked[: self.cfg.qwen_shortlist]

        if self.enable_grouprank and shortlist:
            t2 = time.perf_counter()
            regrouped = self.group_rerank(query, shortlist)
            t.grouprank_ran = any(r.group_score is not None for r in regrouped)
            t.latency_ms["grouprank"] = (time.perf_counter() - t2) * 1000
            if not t.grouprank_ran:
                t.fell_back_to = t.fell_back_to or "qwen"
            shortlist = regrouped

        t.latency_ms["total"] = sum(v for k, v in t.latency_ms.items() if k != "total")
        return shortlist[:top_k]

    def rank_ids(self, query: str, top_k: int = 10) -> list[str]:
        """Convenience for the eval harness: ordered doc_ids only."""
        return [r.doc_id for r in self.retrieve(query, top_k=top_k)]

    # ---- helpers ------------------------------------------------------------ #
    @staticmethod
    def _result_from(c: Candidate) -> Result:
        return Result(doc_id=c.doc_id, channels=sorted(c.channels),
                      ranks=dict(c.ranks), scores=dict(c.scores))

    @staticmethod
    def _renumber(results: list[Result]) -> list[Result]:
        for i, r in enumerate(results, start=1):
            r.rank = i
        return results


def build_default_cascade(*, vectors: dict[str, tuple[str, str, str]],
                          doc_lookup: dict[str, Any] | None = None,
                          device: str | None = None,
                          identifier_index: dict[str, list[str]] | None = None,
                          enable_grouprank: bool | None = None) -> RegisterAwareCascade:
    """Construct the cascade from {channel: (model_dir, vectors_path, ids_path)}.

    Each channel needs its OWN doc vectors embedded with its OWN model -- vectors are not
    interchangeable across models (different dims, different spaces), and the untouched
    0.6B's vectors do not exist until someone embeds the corpus with it (~45 min).
    """
    from dh2.retriever2 import DenseRetriever
    retrievers: dict[str, Any] = {}
    for ch, triple in vectors.items():
        if not triple:
            continue
        model_dir, vpath, ipath = triple
        retrievers[ch] = DenseRetriever(model_dir, vpath, ipath, device=device)
    return RegisterAwareCascade(retrievers, doc_lookup,
                                identifier_index=identifier_index,
                                enable_grouprank=enable_grouprank)
