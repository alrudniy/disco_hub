"""
Tests for the synthesis agent.

The load-bearing tests here are the INTEGRATION ones against the real verifier
(`TestTheAnswerSurvivesItsOwnGate`). Synthesis and the verifier are correct in
isolation and were nearly incompatible when composed -- an inline citation makes
every sentence fail the verifier's risky-token check -- so the contract that
matters is the composed one, and a unit test of either half alone would not see it.

No network in any test: the LLM is always a fake or absent.
"""
from __future__ import annotations

import pytest

from agents.synthesis import SynthesisAgent, _foreground_gaps, _quotable, _quote_claims
from agents.verifier import VerifierAgent

ABSTRACT = ("The invention provides B7-H1, a novel immunoregulatory molecule expressed "
            "on antigen presenting cells and tumor cells that costimulates T cell "
            "responses and modulates interleukin production.")
ABSTRACT2 = ("Compositions comprising anti-B7-H1 antibodies are described for the "
             "treatment of solid tumors in mammals, including methods of administering "
             "such antibodies to a subject in need thereof.")


def _cand(doc_id="uspto:US6803192B1", abstract=ABSTRACT, rerank=0.9, **kw):
    c = {"doc_id": doc_id, "title": "B7-H1, a novel immunoregulatory molecule",
         "abstract": abstract, "source_url": f"https://patents.google.com/patent/{doc_id}",
         "source": "uspto", "score": 0.016, "rerank_score": rerank,
         "organizations": ["mayo foundation"]}
    c.update(kw)
    return c


class _FakeLLM:
    """Scripted stand-in. `available` and the returned object are the only surface."""

    def __init__(self, response, available=True):
        self.response = response
        self.available = available
        self.calls = []

    def chat_json(self, messages, schema_hint, **kw):
        self.calls.append({"messages": messages, "schema_hint": schema_hint, **kw})
        return self.response


def _no_llm():
    """
    An explicitly UNAVAILABLE client, for the tests that exercise the no-key path.

    NOT `llm=None`: both SynthesisAgent and VerifierAgent read `llm=None` as
    "construct the default", and the default LLMClient reads DH_LLM_API_KEY from
    the environment. So on a developer's machine with the demo env sourced --
    which is exactly where these tests get run before a demo -- `llm=None` builds a
    LIVE client, and the "no model ran" tests would fire real HTTP requests at
    z.ai and then fail on the response. Caught in exactly that way: the suite went
    from 0.12 s to 8.5 s and one test failed only when a key was present. Tests
    must be hermetic and must not depend on the ambient environment.
    """
    return _FakeLLM(None, available=False)


class TestTestsAreHermetic:
    def test_no_test_here_depends_on_the_ambient_api_key(self):
        # Pins the fix above: constructing an agent with a None llm in this file
        # would silently mean "use whatever key the developer happens to have
        # exported". Matches construction sites only -- the prose above discusses
        # the anti-pattern by name and must stay readable.
        import pathlib
        import re
        src = pathlib.Path(__file__).read_text()
        assert not re.search(r"Agent\(\s*llm=None", src)

    def test_the_stub_is_actually_unavailable(self):
        assert _no_llm().available is False


# --------------------------------------------------------------------------- #
# Abstention -- a first-class outcome
# --------------------------------------------------------------------------- #
class TestAbstention:
    def test_abstains_with_no_candidates(self):
        r = SynthesisAgent(llm=_no_llm()).run("q", {"candidates": []})
        assert r.abstained and r.ok          # abstention travels with ok=True
        assert "no candidates" in r.payload["reason"]

    def test_abstains_when_no_candidate_carries_evidence(self):
        # Strict RAG: a candidate with no abstract grounds nothing, whatever its rank.
        cands = [_cand(abstract=""), _cand(doc_id="uspto:US2B1", abstract="   ")]
        r = SynthesisAgent(llm=_no_llm()).run("q", {"candidates": cands})
        assert r.abstained
        assert "lack an abstract" in r.payload["reason"]

    def test_abstains_below_the_confidence_threshold(self):
        r = SynthesisAgent(llm=_no_llm(), min_confidence=0.35).run(
            "q", {"candidates": [_cand(rerank=0.10)]})
        assert r.abstained
        assert "below" in r.payload["reason"]

    def test_an_abstention_is_not_an_error(self):
        r = SynthesisAgent(llm=_no_llm()).run("q", {"candidates": []})
        assert r.ok is True and r.error is None and r.confidence == 0.0


# --------------------------------------------------------------------------- #
# The confidence gate is only applied where it means something
# --------------------------------------------------------------------------- #
class TestConfidenceGate:
    def test_gate_not_applied_in_mock_mode(self):
        # No rerank_score => no cross-encoder ran => the score is an RRF rank
        # artifact (~0.016). Comparing THAT to 0.35 would refuse every mock query
        # for a reason that is arithmetic, not evidence.
        cand = _cand()
        del cand["rerank_score"]
        r = SynthesisAgent(llm=_no_llm()).run("q", {"candidates": [cand]})
        assert not r.abstained
        assert "NOT APPLIED" in r.payload["confidence_gate"]
        assert "mock" in r.payload["confidence_basis"]

    def test_gate_applied_when_a_reranker_ran(self):
        r = SynthesisAgent(llm=_no_llm()).run("q", {"candidates": [_cand(rerank=0.9)]})
        assert r.payload["confidence_basis"] == "rerank_score"
        assert r.confidence == pytest.approx(0.9)


# --------------------------------------------------------------------------- #
# Deterministic path -- quotation, not synthesis
# --------------------------------------------------------------------------- #
class TestDeterministicPath:
    def test_says_plainly_that_no_model_ran(self):
        r = SynthesisAgent(llm=_no_llm()).run("q", {"candidates": [_cand()]})
        assert r.payload["mode"] == "deterministic"
        assert r.payload["synthesized"] is False
        assert r.payload["fallback_reason"] == "no_llm_api_key"
        assert "no LLM ran" in r.payload["caveat"]

    def test_claims_are_verbatim_spans_of_the_abstract(self):
        r = SynthesisAgent(llm=_no_llm()).run("q", {"candidates": [_cand()]})
        text = r.payload["claims"][0]["text"]
        # Verbatim: the claim minus the boundary-safety period IS the source text.
        assert text.rstrip(".") in ABSTRACT

    def test_no_inline_citation_appears_in_the_prose(self):
        # THE regression guard. An inline URL or doc_id here fails the downstream
        # verifier's risky-token check on every sentence -- see the module docstring.
        r = SynthesisAgent(llm=_no_llm()).run("q", {"candidates": [_cand()]})
        answer = r.payload["answer"]
        assert "http" not in answer
        assert "[source:" not in answer
        assert "US6803192B1" not in answer

    def test_every_claim_still_carries_a_citation(self):
        # The spec's requirement is met structurally, and is now machine-checkable.
        r = SynthesisAgent(llm=_no_llm()).run("q", {"candidates": [_cand(), _cand("uspto:US7B2", ABSTRACT2)]})
        assert all(c["doc_id"] for c in r.payload["claims"])
        assert r.payload["cited_doc_ids"] == ["uspto:US6803192B1", "uspto:US7B2"]

    def test_quotable_terminates_the_sentence(self):
        # Two quotes joined by a space must not fuse into one cross-document
        # sentence -- that fused sentence matches neither document and fails.
        assert _quotable("a truncated fragment with no end").endswith(".")
        assert _quotable("Already ends properly.") == "Already ends properly."

    def test_quotable_does_not_append_an_ellipsis(self):
        long = "word " * 100
        assert "..." not in _quotable(long)

    def test_no_claim_from_an_empty_abstract(self):
        claims, _ = _quote_claims([_cand(abstract="")])
        assert claims == []


# --------------------------------------------------------------------------- #
# LLM path
# --------------------------------------------------------------------------- #
class TestLLMPath:
    def test_uses_the_models_claims_and_labels_the_mode(self):
        llm = _FakeLLM({"claims": [{"text": "B7-H1 costimulates T cell responses.",
                                    "doc_id": "uspto:US6803192B1"}]})
        r = SynthesisAgent(llm=llm).run("q", {"candidates": [_cand()]})
        assert r.payload["mode"] == "llm"
        assert r.payload["answer"] == "B7-H1 costimulates T cell responses."

    def test_a_fabricated_doc_id_is_dropped_not_shipped(self):
        # The model cites a document it was never shown. Strict RAG: that claim
        # rests on a document it invented or remembered, so it does not ship.
        llm = _FakeLLM({"claims": [
            {"text": "Real claim.", "doc_id": "uspto:US6803192B1"},
            {"text": "Invented claim.", "doc_id": "uspto:US0000000B9"}]})
        r = SynthesisAgent(llm=llm).run("q", {"candidates": [_cand()]})
        assert r.payload["cited_doc_ids"] == ["uspto:US6803192B1"]
        assert "Invented claim." not in r.payload["answer"]

    def test_empty_llm_response_falls_back_and_says_so(self):
        # glm-4.6 returns 200 with content:"" when reasoning eats max_tokens
        # (agents/LLM_CONTRACT.md) -> chat_json gives None. Never pretend it ran.
        llm = _FakeLLM(None)
        r = SynthesisAgent(llm=llm).run("q", {"candidates": [_cand()]})
        assert r.payload["mode"] == "deterministic"
        assert r.payload["fallback_reason"] == "llm_returned_no_usable_claims"
        assert "returned nothing usable" in r.payload["caveat"]

    def test_all_claims_fabricated_falls_back_rather_than_shipping_nothing(self):
        llm = _FakeLLM({"claims": [{"text": "x", "doc_id": "nope:1"}]})
        r = SynthesisAgent(llm=llm).run("q", {"candidates": [_cand()]})
        assert r.payload["mode"] == "deterministic"

    def test_unavailable_llm_never_gets_called(self):
        llm = _FakeLLM({"claims": []}, available=False)
        SynthesisAgent(llm=llm).run("q", {"candidates": [_cand()]})
        assert llm.calls == []

    def test_framing_is_constructive_and_differs_from_the_verifier(self):
        # The verifier is a "falsification engine" that assumes this agent is lying.
        # Converging the two prompts would give them the same blind spots and turn
        # the second pass into agreement theatre. Pin the divergence.
        from agents.synthesis import _SYSTEM_PROMPT as SYN
        from agents.verifier import _SYSTEM_PROMPT as VER
        assert "falsification engine" in VER and "falsification engine" not in SYN
        assert "scouting analyst" in SYN
        assert "INADMISSIBLE" in SYN  # both refuse world knowledge, for opposite jobs


# --------------------------------------------------------------------------- #
# gaps -> ordering (spec 4.2 passes gaps into synthesis)
# --------------------------------------------------------------------------- #
class TestGapsContext:
    def test_gap_targets_are_foregrounded(self):
        cands = [_cand("uspto:A"), _cand("uspto:B"), _cand("uspto:C")]
        out = _foreground_gaps(cands, {"uspto:C"})
        assert [c["doc_id"] for c in out] == ["uspto:C", "uspto:A", "uspto:B"]

    def test_ordering_within_a_group_is_retrievals_own_rank(self):
        cands = [_cand("uspto:A"), _cand("uspto:B"), _cand("uspto:C")]
        out = _foreground_gaps(cands, {"uspto:C", "uspto:B"})
        assert [c["doc_id"] for c in out] == ["uspto:B", "uspto:C", "uspto:A"]

    def test_accepts_an_agent_result_or_a_payload(self):
        class _R:
            payload = {"gaps": [{"doc_id": "uspto:US7B2"}]}
        cands = [_cand(), _cand("uspto:US7B2", ABSTRACT2)]
        r = SynthesisAgent(llm=_no_llm()).run("q", {"candidates": cands, "gaps": _R()})
        assert r.payload["gap_targets_foregrounded"] == ["uspto:US7B2"]
        assert r.payload["claims"][0]["doc_id"] == "uspto:US7B2"

    def test_gap_narration_is_never_merged_into_the_gated_prose(self):
        # A gap is a claim about ABSENCE. No retrieved abstract can support it, so
        # folding it in would fail the gate on every run and degrade the money shot.
        class _R:
            payload = {"gaps": [{"doc_id": "uspto:US6803192B1"}],
                       "narration": "BRISTOL MYERS SQUIBB has no coverage of B7-H3."}
        r = SynthesisAgent(llm=_no_llm()).run("q", {"candidates": [_cand()], "gaps": _R()})
        assert "no coverage" not in r.payload["answer"]

    def test_no_gaps_is_not_an_error(self):
        r = SynthesisAgent(llm=_no_llm()).run("q", {"candidates": [_cand()], "gaps": None})
        assert r.ok and not r.abstained


# --------------------------------------------------------------------------- #
# THE COMPOSED CONTRACT -- the reason this file exists
# --------------------------------------------------------------------------- #
class TestTheAnswerSurvivesItsOwnGate:
    """
    Synthesis feeds the verifier. Both are individually correct and were nearly
    incompatible: the verifier flags any digit-bearing token absent from the
    evidence, and does not strip URLs first, so 08's inline `[source: <url>]`
    citation makes EVERY sentence unsupported -> fail -> degrade. That would have
    degraded all three demo queries, including the two meant to succeed.
    """

    def _evidence_from(self, cands):
        # The shape retrieval actually emits (agents/retrieval.py::make_evidence).
        from agents.retrieval import _quote_span
        return [{"doc_id": c["doc_id"], "quote_span": _quote_span(c["abstract"]),
                 "source_url": c["source_url"]} for c in cands]

    def test_deterministic_answer_passes_the_deterministic_verifier(self):
        cands = [_cand(), _cand("uspto:US7B2", ABSTRACT2)]
        s = SynthesisAgent(llm=_no_llm()).run("q", {"candidates": cands})
        v = VerifierAgent(llm=_no_llm()).run("q", {"answer": s.payload["answer"],
                                              "evidence": self._evidence_from(cands)})
        assert v.payload["verdict"] == "pass", v.payload["claims"]
        assert v.payload["unsupported_count"] == 0
        assert v.ok is True

    def test_an_inline_citation_would_have_failed_the_gate(self):
        # Pins the finding itself: this is what copying 08's _explain_mock does.
        # If a future edit reintroduces inline citations, the test above goes red
        # and this one explains why.
        cands = [_cand()]
        answer = (f'{ABSTRACT} [source: {cands[0]["source_url"]}]')
        v = VerifierAgent(llm=_no_llm()).run("q", {"answer": answer,
                                              "evidence": self._evidence_from(cands)})
        assert v.payload["verdict"] == "fail"

    def test_meta_commentary_in_the_prose_would_fail_the_gate(self):
        # Why every caveat lives in a sibling payload field and never in "answer":
        # the token "2" is absent from the evidence, so the aside fails the answer.
        cands = [_cand()]
        v = VerifierAgent(llm=_no_llm()).run(
            "q", {"answer": f"{ABSTRACT} Retrieval returned 2 candidate documents.",
                  "evidence": self._evidence_from(cands)})
        assert v.payload["verdict"] == "fail"

    def test_two_quotes_do_not_fuse_into_one_cross_document_sentence(self):
        # The abstract is deliberately cut mid-sentence, so without _quotable's
        # terminator the two spans fuse and match neither document.
        truncated = "word " * 60 + "final fragment with no terminator"
        cands = [_cand(abstract=truncated), _cand("uspto:US7B2", ABSTRACT2)]
        s = SynthesisAgent(llm=_no_llm()).run("q", {"candidates": cands})
        v = VerifierAgent(llm=_no_llm()).run("q", {"answer": s.payload["answer"],
                                              "evidence": self._evidence_from(cands)})
        assert v.payload["verdict"] == "pass", v.payload["claims"]
