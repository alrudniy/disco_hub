"""
Known-answer tests for the evidence-gating agent.

The properties worth locking down are the ones whose failure is silent in a demo:
a fabricated claim sailing through the gate, a fabricated CITATION being taken at
face value, the model's own verdict being trusted over our recomputation, and the
no-key path quietly pretending an LLM ran.

No network. The LLM is a fake with a scripted .chat_json; the deterministic path
runs with no key at all.

Run: venv/bin/python -m pytest tests/test_verifier.py -q
"""
import pytest

from agents.verifier import VerifierAgent, _SUPPORT_THRESHOLD
from discovery_hub.faithfulness import lexical_support

# Real-shaped evidence: the B7-H1 patent that agents/LLM_CONTRACT.md used for its
# live verified call, and a second doc so citation-picking has something to choose
# between.
EVIDENCE = [
    {"doc_id": "uspto:US6803192B1",
     "quote_span": "B7-H1, a novel immunoregulatory molecule expressed on antigen "
                   "presenting cells, costimulates T cell proliferation.",
     "source_url": "https://patents.google.com/patent/US6803192B1"},
    {"doc_id": "clinicaltrials:NCT04789655",
     "quote_span": "A study of an anti-PD-L1 antibody in participants with advanced "
                   "solid tumors.",
     "source_url": "https://clinicaltrials.gov/study/NCT04789655"},
]


class _FakeLLM:
    """Scripted stand-in. `available` is True; .chat_json returns whatever it was given."""

    def __init__(self, response):
        self._response = response
        self.available = True
        self.calls = []

    def chat_json(self, messages, schema_hint, **kwargs):
        self.calls.append(messages)
        return self._response


def _agent(response):
    return VerifierAgent(llm=_FakeLLM(response))


# --------------------------------------------------------------------------- #
# LLM path: the gate
# --------------------------------------------------------------------------- #
def test_fully_grounded_answer_passes():
    agent = _agent({
        "claims": [{"text": "B7-H1 costimulates T cell proliferation.",
                    "status": "supported", "doc_id": "uspto:US6803192B1"}],
        "verdict": "pass", "unsupported_count": 0})
    r = agent.run("what does B7-H1 do?",
                  {"answer": "B7-H1 costimulates T cell proliferation.",
                   "evidence": EVIDENCE})

    assert r.ok is True
    assert r.payload["verdict"] == "pass"
    assert r.payload["unsupported_count"] == 0
    assert r.payload["mode"] == "llm"
    assert r.confidence == 1.0
    # Only the evidence actually cited travels onward.
    assert [e["doc_id"] for e in r.evidence] == ["uspto:US6803192B1"]


def test_fabricated_claim_fails_the_gate():
    # The exact shape glm-4.6 returned live (LLM_CONTRACT.md): doc_id null on the
    # unsupported claim.
    agent = _agent({
        "claims": [{"text": "B7-H1 is an immunoregulatory molecule.",
                    "status": "supported", "doc_id": "uspto:US6803192B1"},
                   {"text": "B7-H1 was approved by the FDA in 2019.",
                    "status": "unsupported", "doc_id": None}],
        "verdict": "fail", "unsupported_count": 1})
    r = agent.run("tell me about B7-H1",
                  {"answer": "B7-H1 is an immunoregulatory molecule. B7-H1 was "
                             "approved by the FDA in 2019.",
                   "evidence": EVIDENCE})

    assert r.ok is False           # drives the orchestrator's degrade-to-evidence-only
    assert r.error is None         # a gate-fail is NOT a crash; both are ok=False
    assert r.payload["verdict"] == "fail"
    assert r.payload["unsupported_count"] == 1


def test_contradicted_claim_also_fails_the_gate():
    # The spec's wording gates only on `unsupported`; a contradicted claim is
    # strictly worse and must not sail through on a technicality.
    agent = _agent({
        "claims": [{"text": "B7-H1 suppresses T cell proliferation.",
                    "status": "contradicted", "doc_id": "uspto:US6803192B1"}],
        "verdict": "pass", "unsupported_count": 0})
    r = agent.run("q", {"answer": "B7-H1 suppresses T cell proliferation.",
                        "evidence": EVIDENCE})

    assert r.ok is False
    assert r.payload["verdict"] == "fail"
    assert r.payload["contradicted_count"] == 1
    assert r.payload["unsupported_count"] == 0   # the spec's field still means what it says


# --------------------------------------------------------------------------- #
# LLM path: post-validation. The model is not trusted.
# --------------------------------------------------------------------------- #
def test_fabricated_doc_id_citation_becomes_unsupported():
    # A doc_id that was never retrieved cannot support anything -- and the model
    # citing it is itself the finding worth surfacing.
    agent = _agent({
        "claims": [{"text": "Pembrolizumab is approved for melanoma.",
                    "status": "supported", "doc_id": "uspto:US9999999B9"}],
        "verdict": "pass", "unsupported_count": 0})
    r = agent.run("q", {"answer": "Pembrolizumab is approved for melanoma.",
                        "evidence": EVIDENCE})

    assert r.ok is False
    assert r.payload["claims"][0]["status"] == "unsupported"
    assert r.payload["claims"][0]["doc_id"] is None
    assert r.payload["unsupported_count"] == 1
    assert r.payload["invalid_citations"][0]["cited_doc_id"] == "uspto:US9999999B9"
    assert "uspto:US9999999B9" in r.payload["caveat"] or "1 claim" in r.payload["caveat"]


def test_model_verdict_is_recomputed_not_trusted():
    # Model says pass with unsupported_count 0 while listing an unsupported claim.
    # The gate is ours; its self-grade is discarded.
    agent = _agent({
        "claims": [{"text": "A fabricated thing.", "status": "unsupported",
                    "doc_id": None}],
        "verdict": "pass", "unsupported_count": 0})
    r = agent.run("q", {"answer": "A fabricated thing.", "evidence": EVIDENCE})

    assert r.payload["verdict"] == "fail"
    assert r.payload["unsupported_count"] == 1
    assert r.ok is False


def test_unknown_status_degrades_to_unsupported():
    agent = _agent({"claims": [{"text": "Something.", "status": "probably fine",
                                "doc_id": "uspto:US6803192B1"}]})
    r = agent.run("q", {"answer": "Something.", "evidence": EVIDENCE})
    assert r.payload["claims"][0]["status"] == "unsupported"
    assert r.ok is False


def test_prompt_framing_is_adversarial_and_hides_no_evidence():
    # Agreement theatre is the failure mode this framing exists to prevent.
    agent = _agent({"claims": [{"text": "x", "status": "supported",
                                "doc_id": "uspto:US6803192B1"}]})
    agent.run("what does B7-H1 do?", {"answer": "x", "evidence": EVIDENCE})

    system = agent.llm.calls[0][0]["content"].lower()
    assert "falsif" in system
    assert "unsupported" in system and "default" in system
    # The evidence, with doc_ids, must actually reach the model.
    user = agent.llm.calls[0][1]["content"]
    assert "uspto:US6803192B1" in user and "immunoregulatory" in user


# --------------------------------------------------------------------------- #
# Deterministic path: must work, and must admit it is weaker
# --------------------------------------------------------------------------- #
def test_deterministic_path_runs_with_no_key(monkeypatch):
    monkeypatch.delenv("DH_LLM_API_KEY", raising=False)
    agent = VerifierAgent()            # real LLMClient, no key -> unavailable
    assert agent.llm.available is False

    r = agent.run("what does B7-H1 do?",
                  {"answer": "B7-H1 is a novel immunoregulatory molecule that "
                             "costimulates T cell proliferation.",
                   "evidence": EVIDENCE})

    assert r.payload["mode"] == "deterministic"
    assert r.payload["verdict"] == "pass"
    assert r.ok is True
    assert r.payload["claims"][0]["doc_id"] == "uspto:US6803192B1"


def test_deterministic_path_never_implies_an_llm_ran(monkeypatch):
    monkeypatch.delenv("DH_LLM_API_KEY", raising=False)
    r = VerifierAgent().run("q", {"answer": "B7-H1 costimulates T cell proliferation.",
                                  "evidence": EVIDENCE})
    caveat = r.payload["caveat"].lower()
    assert "no llm" in caveat and "weaker" in caveat
    assert r.payload["detects_contradiction"] is False
    assert r.payload["thresholds_calibrated"] is False


def test_deterministic_path_flags_fabricated_number(monkeypatch):
    # Isolates the risky-token check, and this isolation is load-bearing: the
    # sentence copies the evidence almost verbatim, so its overlap score is 0.909
    # (measured) -- comfortably ABOVE _SUPPORT_THRESHOLD. Check 1 passes it. Only
    # the ungrounded '87%' catches it. Written this way on purpose: an earlier
    # draft used "...approved in 2019", which scores 0.6 and is caught by overlap
    # alone, so it would have gone green with the risky-token check deleted.
    monkeypatch.delenv("DH_LLM_API_KEY", raising=False)
    fabricated = ("B7-H1, a novel immunoregulatory molecule expressed on antigen "
                  "presenting cells, costimulates T cell proliferation in 87% of "
                  "patients.")
    assert lexical_support(fabricated, EVIDENCE[0]["quote_span"]) > _SUPPORT_THRESHOLD

    r = VerifierAgent().run("q", {"answer": fabricated, "evidence": EVIDENCE})
    assert r.payload["verdict"] == "fail"
    assert r.payload["unsupported_count"] == 1
    assert r.ok is False


def test_deterministic_path_grounds_real_identifiers(monkeypatch):
    # The mirror image: 'B7-H1' is a risky-looking token that IS in the evidence,
    # so it must not be flagged. Otherwise every real identifier fails the gate and
    # the check is useless in exactly the queries this corpus is built for.
    monkeypatch.delenv("DH_LLM_API_KEY", raising=False)
    r = VerifierAgent().run("q", {
        "answer": "B7-H1 costimulates T cell proliferation.", "evidence": EVIDENCE})
    assert r.payload["verdict"] == "pass"


def test_deterministic_path_flags_off_topic_sentence(monkeypatch):
    monkeypatch.delenv("DH_LLM_API_KEY", raising=False)
    r = VerifierAgent().run("q", {
        "answer": "This compound cures Alzheimer's disease and reverses aging.",
        "evidence": EVIDENCE})
    assert r.payload["verdict"] == "fail"
    assert r.ok is False


def test_deterministic_is_deterministic(monkeypatch):
    # Same input, reversed evidence order -> identical claims/citations.
    monkeypatch.delenv("DH_LLM_API_KEY", raising=False)
    answer = "B7-H1 costimulates T cell proliferation."
    a = VerifierAgent().run("q", {"answer": answer, "evidence": EVIDENCE})
    b = VerifierAgent().run("q", {"answer": answer, "evidence": list(reversed(EVIDENCE))})
    assert a.payload["claims"] == b.payload["claims"]


def test_llm_returning_nothing_falls_back_and_says_so():
    # Unparseable JSON / empty content (the glm-4.6 reasoning-budget trap) must
    # degrade to the lexical check, labelled -- never silently report mode=llm.
    agent = _agent(None)
    r = agent.run("q", {"answer": "B7-H1 costimulates T cell proliferation.",
                        "evidence": EVIDENCE})
    assert r.payload["mode"] == "deterministic"
    assert "returned no usable verdict" in r.payload["caveat"]


# --------------------------------------------------------------------------- #
# Edges
# --------------------------------------------------------------------------- #
def test_empty_answer_abstains_rather_than_failing():
    # Nothing synthesized -> nothing to gate. "fail" would tell the orchestrator to
    # degrade an answer that does not exist. Abstention is ok=True (base.py inv. #1).
    r = VerifierAgent(llm=_FakeLLM(None)).run("q", {"answer": "", "evidence": EVIDENCE})
    assert r.ok is True and r.abstained is True
    assert r.payload["reason"] == "no answer text to check"


def test_answer_with_no_evidence_fails_closed(monkeypatch):
    # An answer grounded in zero retrieved passages is ungrounded by definition.
    monkeypatch.delenv("DH_LLM_API_KEY", raising=False)
    r = VerifierAgent().run("q", {"answer": "B7-H1 costimulates T cells.",
                                  "evidence": []})
    assert r.ok is False
    assert r.payload["verdict"] == "fail"


def test_dict_answer_is_accepted():
    agent = _agent({"claims": [{"text": "B7-H1 costimulates T cell proliferation.",
                                "status": "supported", "doc_id": "uspto:US6803192B1"}]})
    r = agent.run("q", {"answer": {"answer": "B7-H1 costimulates T cell proliferation.",
                                   "citations": ["uspto:US6803192B1"]},
                        "evidence": EVIDENCE})
    assert r.ok is True and r.payload["verdict"] == "pass"


def test_malformed_evidence_entries_are_dropped():
    agent = _agent({"claims": [{"text": "x", "status": "supported",
                                "doc_id": "uspto:US6803192B1"}]})
    r = agent.run("q", {"answer": "x", "evidence": EVIDENCE + [
        "not a dict", {"doc_id": "", "quote_span": "empty id"},
        {"doc_id": "d", "quote_span": "   "}]})
    assert r.ok is True


def test_agent_conforms_to_protocol():
    from agents.base import Agent
    assert isinstance(VerifierAgent(llm=_FakeLLM(None)), Agent)


def test_threshold_is_conservative():
    # Documented as hand-set and stricter than faithfulness.py's 0.5 default.
    assert _SUPPORT_THRESHOLD > 0.5


@pytest.mark.parametrize("bad", [{"claims": []}, {"claims": "nonsense"}, {}])
def test_llm_returning_no_claims_falls_back(bad, monkeypatch):
    monkeypatch.delenv("DH_LLM_API_KEY", raising=False)
    r = _agent(bad).run("q", {"answer": "B7-H1 costimulates T cell proliferation.",
                              "evidence": EVIDENCE})
    # No claims extracted is not a pass -- it degrades to the lexical check.
    assert r.payload["mode"] == "deterministic"
