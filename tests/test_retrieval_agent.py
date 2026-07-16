"""
Tests for agents/retrieval.py.

Every test injects a FAKE retriever. A real (mock=False) Retriever is ~20 GB RSS on a
30 GB box; nothing in this file may construct one, and the laziness test below pins that
constructing the agent does not construct a retriever at all.
"""
from __future__ import annotations

import pytest

from agents.base import Agent, AgentResult, run_agent
from agents.retrieval import RetrievalAgent, _QUOTE_CHARS, _confidence, _quote_span

ABSTRACT = (
    "B7-H1, a novel immunoregulatory molecule, costimulates T-cell proliferation and "
    "interleukin-10 secretion. The molecule is expressed on activated dendritic cells "
    "and mediates peripheral tolerance through engagement of an unidentified receptor "
    "on activated T cells, which has consequences for tumour immune evasion in vivo."
)


def _cand(doc_id: str, rerank: float | None = 0.9, **over) -> dict:
    c = {
        "doc_id": doc_id, "score": 0.031, "text_score": 0.62, "keyword_score": 0.0,
        "graph_score": 0.0, "graph_linked": False, "title": f"T {doc_id}",
        "abstract": ABSTRACT, "source": "uspto",
        "source_url": f"https://patents.google.com/patent/{doc_id}",
        "organizations": ["mayo foundation"],
    }
    if rerank is not None:
        c["rerank_score"] = rerank
    c.update(over)
    return c


class FakeRetriever:
    """Canned candidates. Records the call so we can assert what the agent asked for."""

    def __init__(self, candidates: list[dict]) -> None:
        self.candidates = candidates
        self.calls: list[tuple] = []

    def retrieve(self, query, top_k=None, rerank_k=None, **kw):
        self.calls.append((query, top_k, rerank_k))
        return list(self.candidates)


class ExplodingRetriever:
    def retrieve(self, *a, **k):
        raise RuntimeError("faiss index went missing")


def test_implements_agent_protocol_and_name():
    agent = RetrievalAgent(retriever=FakeRetriever([_cand("US6803192B1")]))
    assert agent.name == "retrieval"
    assert isinstance(agent, Agent)


def test_constructing_agent_does_not_construct_a_retriever():
    # Memory discipline: a real load is ~20 GB. Passing no retriever must stay free
    # until run() is actually called.
    agent = RetrievalAgent(mock=False)
    assert agent._retriever is None


def test_payload_shape_and_evidence_is_verbatim():
    fake = FakeRetriever([_cand("US6803192B1"), _cand("US9012409B2", rerank=0.7)])
    result = RetrievalAgent(retriever=fake).run("b7-h1 checkpoint", {})

    assert result.ok and not result.abstained
    assert [c["doc_id"] for c in result.payload["candidates"]] == ["US6803192B1", "US9012409B2"]
    assert result.payload["linked"] is False
    assert result.payload["reranked"] is True
    assert result.payload["confidence_basis"] == "rerank_score"

    assert len(result.evidence) == 2
    for ev, cand in zip(result.evidence, fake.candidates):
        assert set(ev) == {"doc_id", "quote_span", "source_url"}
        assert ev["doc_id"] == cand["doc_id"]
        assert ev["source_url"] == cand["source_url"]
        # The verifier checks claims against these spans -- they must be findable in the
        # source character for character, so no ellipsis and no paraphrase.
        assert ev["quote_span"] in cand["abstract"]
        assert "..." not in ev["quote_span"]


def test_confidence_is_top_rerank_score_not_top_text_score():
    # The whole point of 08's rationale: a keyword-surfaced hit with a low dense cosine
    # must still get its high cross-encoder confidence.
    fake = FakeRetriever([_cand("NCT04789655", rerank=0.93, text_score=0.04,
                                keyword_score=8.2, source="clinicaltrials")])
    result = RetrievalAgent(retriever=fake).run("phase 2 nivolumab trial", {})
    assert result.confidence == 0.93


def test_confidence_reads_the_maximum_not_the_first_candidate():
    fake = FakeRetriever([_cand("A", rerank=0.4), _cand("B", rerank=0.88)])
    assert RetrievalAgent(retriever=fake).run("q", {}).confidence == 0.88


def test_out_of_range_rerank_score_is_sigmoid_squashed():
    fake = FakeRetriever([_cand("A", rerank=4.2)])
    result = RetrievalAgent(retriever=fake).run("q", {})
    assert 0.0 <= result.confidence <= 1.0
    assert result.confidence == _confidence(4.2)


def test_mock_mode_falls_back_to_fused_score_and_says_so():
    # No rerank_score => no cross-encoder ran. The agent must never silently present an
    # RRF rank artifact as if it were a relevance score.
    fake = FakeRetriever([_cand("US6803192B1", rerank=None, score=0.031)])
    result = RetrievalAgent(retriever=fake, mock=True).run("q", {})
    assert result.payload["reranked"] is False
    assert "NO reranker ran" in result.payload["confidence_basis"]
    assert result.confidence == 0.031


def test_zero_candidates_abstains_with_a_reason_and_ok_true():
    result = RetrievalAgent(retriever=FakeRetriever([])).run("kjhgfdsa", {})
    assert result.abstained is True
    assert result.ok is True           # abstention is success, not failure
    assert result.confidence == 0.0
    assert result.evidence == []
    assert "zero candidates" in result.payload["reason"]


def test_signals_fired_reports_only_arms_that_scored():
    fake = FakeRetriever([_cand("A", text_score=0.6, keyword_score=0.0, graph_score=0.0)])
    assert RetrievalAgent(retriever=fake).run("q", {}).payload["signals_fired"] == ["dense"]

    fake = FakeRetriever([_cand("A", text_score=0.6, keyword_score=1.2, graph_score=0.3)])
    # sorted for determinism
    assert RetrievalAgent(retriever=fake).run("q", {}).payload["signals_fired"] == [
        "dense", "graph", "keyword"]


def test_linked_is_true_when_the_query_linked_to_a_graph_entity():
    fake = FakeRetriever([_cand("A", graph_linked=True, graph_score=0.4)])
    assert RetrievalAgent(retriever=fake).run("bristol myers squibb", {}).payload["linked"] is True


def test_k_is_passed_through_as_rerank_k():
    fake = FakeRetriever([_cand("A")])
    RetrievalAgent(retriever=fake, k=3).run("q", {})
    assert fake.calls == [("q", None, 3)]


def test_run_is_timed():
    result = RetrievalAgent(retriever=FakeRetriever([_cand("A")])).run("q", {})
    assert result.latency_ms > 0.0


def test_determinism_same_input_same_output():
    a = RetrievalAgent(retriever=FakeRetriever([_cand("A"), _cand("B", rerank=0.5)])).run("q", {})
    b = RetrievalAgent(retriever=FakeRetriever([_cand("A"), _cand("B", rerank=0.5)])).run("q", {})
    assert a.payload == b.payload
    assert a.evidence == b.evidence
    assert a.confidence == b.confidence


def test_agent_does_not_swallow_its_own_exception_but_run_agent_contains_it():
    agent = RetrievalAgent(retriever=ExplodingRetriever())
    with pytest.raises(RuntimeError):
        agent.run("q", {})                       # raises: broken != empty
    result = run_agent(agent, "q", {})           # contained by the orchestrator's entry point
    assert isinstance(result, AgentResult)
    assert result.ok is False and result.abstained is False
    assert "faiss index went missing" in result.error


def test_quote_span_truncates_on_a_word_boundary_and_stays_verbatim():
    # Reads _QUOTE_CHARS rather than hardcoding it. The literal 240 here outlived the
    # constant: the window had to widen because the verifier was handed 240 chars of
    # evidence to judge claims synthesis built from 1200, which starved the gate into
    # rejecting well-grounded answers (see tests/test_evidence_window.py). A test that
    # pins a number the code is meant to tune tests the number, not the behaviour --
    # this assertion would have gone red for the fix that repaired the bug.
    long = "word " * _QUOTE_CHARS               # comfortably longer than the window
    span = _quote_span(long)
    assert span in long
    assert len(span) <= _QUOTE_CHARS
    assert not span.endswith(" ")
    assert _quote_span("") == ""
    assert _quote_span("short abstract") == "short abstract"


def test_missing_abstract_yields_empty_span_not_a_fabricated_one():
    fake = FakeRetriever([_cand("A", abstract="")])
    result = RetrievalAgent(retriever=fake).run("q", {})
    assert result.evidence[0]["quote_span"] == ""
