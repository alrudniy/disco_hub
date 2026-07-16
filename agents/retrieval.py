"""
The retrieval agent: an Agent-protocol wrapper around the EXISTING retriever.

This module deliberately adds no retrieval logic. `07_retrieve_rank.py::Retriever`
is the shipped, measured hybrid retrieve-then-rerank service (dense + BM25 + graph,
fused with RRF); this file only adapts its output to the `AgentResult` contract so
the orchestrator can fan it out alongside the other agents. If retrieval quality is
wrong, it is wrong in 07 -- not here.

HONEST LIMITS, so nobody reads more into this than is there:

  - The confidence is the reranker's score for the TOP candidate, squashed to [0,1]
    by the same rule as 08. It is NOT calibrated against labeled relevance. It is a
    relevance score being used as a confidence, which is a defensible proxy and
    nothing more. It must never be averaged with another agent's confidence
    (agents/base.py invariant 3) -- the scales are unrelated.

  - In mock mode NO reranker runs, so there is no rerank_score to read. We fall back
    to the RRF fused score, which is a rank-fusion artifact (~1/60-ish), NOT a
    relevance probability. The payload says so explicitly via `confidence_basis`;
    a mock-mode confidence is not comparable to a real-mode one and is not evidence
    of anything.

  - `signals_fired` is derived from the returned candidates (which score fields are
    non-zero), not from the retriever's internal state. A signal that ran but scored
    every returned candidate 0.0 will read as "not fired". That is a reporting
    approximation, not a measurement of the retriever's plumbing.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from typing import Any

from agents.base import AgentResult, make_evidence, timed

# A quote_span must be VERBATIM -- the verifier checks synthesized claims against these
# spans, so anything we return has to be findable in the source abstract character for
# character. We therefore truncate on a word boundary and append nothing (no ellipsis).
#
# THIS WINDOW MUST BE STRICTLY WIDER THAN synthesis._EVIDENCE_CHARS (1200), AND IT WAS
# NOT. It was 240. That is not a tuning preference -- it silently broke the gate, and in
# the exact shape this layer exists to prevent:
#
#   synthesis._call_llm   shows the model    abstract[:1200]  <- what the claim is built on
#   retrieval._quote_span gives the verifier abstract[:240]   <- what the claim is judged on
#
# The orchestrator hands r.evidence (these spans) to the verifier while synthesis reads
# the candidates' abstracts directly, so THE TWO AGENTS WERE JUDGING DIFFERENT EVIDENCE.
# Any claim resting on characters 241-1200 is invisible to the verifier, which then
# correctly calls it unsupported -- it genuinely cannot see the sentence that grounds it
# -- the gate fails, and the orchestrator degrades a perfectly grounded answer to
# evidence-only. The gate did not misjudge; it was starved.
#
# MEASURED, on the first query of the strict-RAG counterfactual harness
# (11_counterfactual.py) against the real corpus and live glm-4.6:
#   uspto:US10034823B2, abstract 829 chars -> _quote_span returned 232.
#   Claim: "The direct dye compounds enter the hair shaft ..."
#     present in the 1200-char window synthesis read : True
#     present in the 232-char span the verifier read : False   -> unsupported -> fail
#   The control arm answered 0 of 3 queries. Patent abstracts routinely exceed 240 chars
#   (829 here), so this was the COMMON CASE, not an edge case: the gate rejected almost
#   every well-grounded answer, and would have taken the demo's money-shot query with it.
#
# Why the existing tests were green: tests/test_synthesis.py::TestTheAnswerSurvivesItsOwn
# Gate does compose the two agents -- but only through the DETERMINISTIC path, where
# synthesis quotes via _quotable(abstract, limit=240) and the two 240s coincidentally
# agree. In LLM mode -- the demo path -- they do not. tests/test_evidence_window.py now
# pins the invariant directly, in both modes.
#
# 1400 rather than exactly 1200 because _quote_span trims back to a word boundary: at
# equality the verifier would still see a few characters LESS than synthesis on any
# abstract longer than the window -- the same bug with a smaller blast radius. The margin
# puts the trim strictly beyond anything the model was shown.
_QUOTE_CHARS = 1400


def _load_retriever_module() -> Any:
    """
    Import 07_retrieve_rank.py by path. Its module name starts with a digit, so a
    normal `import` is syntactically impossible. Idiom copied from
    08_multiagent_rag.py::_load_retriever_module -- kept identical on purpose.
    """
    path = Path(__file__).resolve().parent.parent / "07_retrieve_rank.py"
    spec = importlib.util.spec_from_file_location("retrieve_rank", path)
    if spec is None or spec.loader is None:  # pragma: no cover -- file is in-repo
        raise ImportError(f"cannot load retriever module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _confidence(score: float) -> float:
    """
    Squash a score to [0,1]. Logic reused verbatim from 08_multiagent_rag.py::_confidence
    -- deliberately NOT reimplemented, so the two entry points cannot drift apart.

    Why the RERANK score and not text_score, per 08's measured rationale: the
    cross-encoder is the pipeline's final and most accurate relevance judge. Reading
    the dense cosine instead badly under-scores candidates surfaced by the keyword or
    graph arms -- e.g. a clinical trial BM25 correctly retrieves for clinical-language
    phrasing can sit at dense cosine ~0.04 while the cross-encoder rightly scores it
    ~0.9 -- which made the policy gate refuse genuine matches. BGE reranker scores are
    already ~[0,1]; the sigmoid only guards out-of-range values.
    """
    if 0.0 <= score <= 1.0:
        return round(score, 4)
    return round(1.0 / (1.0 + math.exp(-score)), 4)  # squash raw logits to [0,1]


def _quote_span(abstract: str) -> str:
    """Shortest useful VERBATIM prefix of the abstract, cut on a word boundary."""
    text = (abstract or "").strip()
    if len(text) <= _QUOTE_CHARS:
        return text
    cut = text[:_QUOTE_CHARS]
    boundary = cut.rfind(" ")
    return (cut[:boundary] if boundary > 0 else cut).rstrip()


def _signals_fired(candidates: list[dict]) -> list[str]:
    """
    Which retrieval arms contributed a non-zero score to the returned set. Sorted for
    determinism (stage 09 verifies run-to-run identity).
    """
    fired = {
        name
        for name, key in (("dense", "text_score"), ("keyword", "keyword_score"),
                          ("graph", "graph_score"))
        if any(c.get(key, 0.0) for c in candidates)
    }
    return sorted(fired)


class RetrievalAgent:
    """Agent-protocol front end for 07's Retriever. Always fires; abstains only on zero hits."""

    name = "retrieval"

    def __init__(self, retriever: Any = None, mock: bool = True,
                 device: str | None = None, k: int | None = None) -> None:
        # The retriever is constructed LAZILY on first run(). A real (mock=False) load is
        # ~20 GB RSS (faiss 6.17 GB + doc_vectors 6.17 GB + 603k docs + bm25) on a 30 GB
        # box, so merely constructing this agent -- which the orchestrator and its tests do
        # freely -- must not pay that cost. Tests inject a fake and never touch the real one.
        self._retriever = retriever
        self._mock = mock
        self._device = device
        self._k = k

    def _get_retriever(self) -> Any:
        if self._retriever is None:
            self._retriever = _load_retriever_module().Retriever(mock=self._mock,
                                                                 device=self._device)
        return self._retriever

    @timed
    def run(self, query: str, ctx: dict) -> AgentResult:
        # ctx may widen the pool for this call only: the expertise-gap agent needs a
        # deeper slice of the field (GAP_TARGET_K) than the ~10 a reader wants shown.
        # Same retriever, same weights -- only the cut-off differs.
        #
        # top_k MUST be raised alongside rerank_k. 07 applies top_k (default
        # top_k_recall=50) to FIRST-STAGE recall and rerank_k only slices the reranked
        # list, so rerank_k=100 against a 50-deep pool silently returns 50 -- measured:
        # the gap agent asked for 100 targets and scored 50, with nothing to say it had
        # been truncated. A cut-off that quietly saturates is worse than a small one.
        # top_k comes ONLY from ctx -- never derived from rerank_k. Deriving it
        # (top_k = rerank_k) shrinks first-stage recall from config's 50 to the
        # display k of 10 on every ordinary query, quietly making normal retrieval
        # worse to fix a problem the gap agent has. None => 07 uses top_k_recall.
        rerank_k = ctx.get("rerank_k") or self._k
        candidates = self._get_retriever().retrieve(query, top_k=ctx.get("top_k"),
                                                    rerank_k=rerank_k)
        if not candidates:
            return AgentResult.abstain(
                self.name, f"retriever returned zero candidates for {query!r}")

        # Mock mode is detected from the DATA, not from self._mock: an injected retriever
        # may be neither. No rerank_score => no cross-encoder ran => the fused RRF score is
        # all we have, and it is a rank artifact rather than a relevance judgement.
        reranked = "rerank_score" in candidates[0]
        if reranked:
            basis = "rerank_score"
            top = max((c.get("rerank_score", 0.0) for c in candidates), default=0.0)
        else:
            basis = ("fused_rrf_score (NO reranker ran -- mock mode; this is a rank-fusion "
                     "artifact, not a relevance score, and is not comparable to real mode)")
            top = max((c.get("score", 0.0) for c in candidates), default=0.0)

        evidence = [
            make_evidence(c["doc_id"], _quote_span(c.get("abstract", "")),
                          c.get("source_url", ""))
            for c in candidates
        ]
        payload = {
            "candidates": candidates,
            # graph_linked is uniform across a retrieve() call (07 sets it from the single
            # query-side entity link), so the top candidate's value speaks for the set.
            "linked": bool(candidates[0].get("graph_linked", False)),
            "signals_fired": _signals_fired(candidates),
            "confidence_basis": basis,
            "reranked": reranked,
        }
        return AgentResult(agent=self.name, ok=True, payload=payload, evidence=evidence,
                           confidence=_confidence(top), latency_ms=0.0)
