"""
Known-answer tests for the policy/answer-safety agent.

The load-bearing test in this file is the FALSE-POSITIVE pair (see
TestDosingSpeechAct): a patent/trial abstract that describes how an invention is
administered must NOT flag, while the demo's "what dose should I give this
patient" MUST. Everything else in this agent is tunable lexicon; that distinction
is the design. If it regresses, the agent is either useless (flags nothing) or
worse than useless (flags every patent in the corpus and gets switched off).

The descriptive-text fixtures are REAL corpus text, quoted from
/home/alex/discovery_hub/data/normalized/docs.jsonl (clinicaltrials:NCT04789655)
and from patent-abstract register, not invented to be easy to pass. The whole
false-positive risk is that this corpus talks about dosing constantly.

No network, no artifacts, no DH_LLM_API_KEY needed. The LLM layer is exercised
with stubs.

Run: venv/bin/python -m pytest tests/test_policy.py -q
"""
import pytest

from agents.base import run_agent, to_trace_row
from agents.policy import PolicyAgent, _clip, _dedupe_and_sort, _sentences

# --------------------------------------------------------------------------- #
# Fixtures: real corpus register vs. real clinical advice
# --------------------------------------------------------------------------- #

# Verbatim from docs.jsonl, doc_id clinicaltrials:NCT04789655. Saturated with
# dosing vocabulary -- "escalating intravenous doses", "administered at or below
# the MTD" -- and contains zero clinical advice. This is what the corpus IS.
TRIAL_ABSTRACT = (
    "This Phase 1, clinical study of CC-96191 will explore the safety, "
    "tolerability and preliminary biological and clinical activity of CC-96191 "
    "as a single-agent in the setting of Relapsed or refractory acute myeloid "
    "leukemia (R/R AML). The dose escalation (Part A) of the study will explore "
    "escalating intravenous doses of CC-96191 to estimate the MTD and/or RP2D of "
    "CC-96191 as monotherapy. The expansion (Part B), will further evaluate the "
    "safety and efficacy of CC-96191 administered at or below the MTD in one or "
    "more expansion cohorts in order to determine the RP2D."
)

# Patent-abstract register. "for the treatment of" is describing what an invention
# is FOR. A composition claim reciting a dose range is claim scope, not a
# prescription. The task brief calls this out by name as the thing not to flag.
PATENT_ABSTRACT = (
    "A pharmaceutical composition for the treatment of non-small cell lung "
    "cancer, comprising an anti-PD-L1 monoclonal antibody, wherein the antibody "
    "is administered intravenously at a dose of about 10 mg/kg. The composition "
    "should be administered in a pharmaceutically acceptable carrier."
)

# The demo's query 3 (spec section 5). This MUST block.
DOSING_QUERY = ("what dose of pembrolizumab should I give a stage IV NSCLC patient")

# The demo's query 1 (spec section 5). Policy must ABSTAIN -- the demo points at it.
SCOUTING_QUERY = "monoclonal antibody targeting PD-L1 for oncology"


def _kinds(result):
    # .get: an abstention's payload carries only `reason`, and "abstained" is a
    # legitimate way for a text to have no flags of a given kind.
    return [f["kind"] for f in result.payload.get("flags", [])]


# --------------------------------------------------------------------------- #
# THE distinction: speech act, not vocabulary
# --------------------------------------------------------------------------- #
class TestDosingSpeechAct:
    def test_pembrolizumab_dosing_question_blocks(self):
        """Demo query 3. Fires on the QUERY alone, before anything is drafted."""
        r = run_agent(PolicyAgent(use_llm=False), DOSING_QUERY, {})
        assert not r.abstained
        assert "dosing" in _kinds(r)
        flag = next(f for f in r.payload["flags"] if f["kind"] == "dosing")
        assert flag["severity"] == "block"
        assert r.ok is False, "a block must set ok=False so the orchestrator degrades"
        assert "pembrolizumab" in flag["span"].lower()

    def test_trial_abstract_does_not_trigger_dosing(self):
        """
        Real corpus text, dense with dosing vocabulary, zero advice. If this flags,
        the agent flags the corpus and is unusable.
        """
        r = run_agent(PolicyAgent(use_llm=False), SCOUTING_QUERY,
                      {"answer": TRIAL_ABSTRACT})
        assert "dosing" not in _kinds(r)

    def test_patent_abstract_phrase_does_not_trigger_dosing(self):
        """
        'for the treatment of' + 'administered ... at a dose of about 10 mg/kg' +
        'should be administered' -- description of an invention, NOT guidance to a
        clinician. The deontic ('should be administered') is present and must not
        be sufficient on its own: there is no care recipient.
        """
        r = run_agent(PolicyAgent(use_llm=False), SCOUTING_QUERY,
                      {"answer": PATENT_ABSTRACT})
        assert "dosing" not in _kinds(r)

    @pytest.mark.parametrize("answer", [
        "You should administer 200 mg every three weeks to the patient.",
        "I recommend starting your patient on 2 mg/kg pembrolizumab.",
        "For this patient, the dose should be reduced to 100 mg.",
    ])
    def test_advice_in_the_answer_blocks(self, answer):
        r = run_agent(PolicyAgent(use_llm=False), "pembrolizumab NSCLC dosing",
                      {"answer": answer})
        assert "dosing" in _kinds(r)
        assert r.ok is False

    @pytest.mark.parametrize("answer", [
        # Trial-registration register: patients as a studied cohort, not a recipient.
        "Patients with R/R AML received escalating intravenous doses of CC-96191.",
        # Portfolio/scouting register -- the tool's actual job.
        "Bristol Myers Squibb holds 826 technologies assigned in the graph, "
        "including several anti-PD-1 antibody compositions.",
        # Describes a regimen studied by someone else, in the past tense.
        "The Phase 2 trial evaluated a 200 mg dose administered every 3 weeks.",
    ])
    def test_descriptive_dosing_language_does_not_block(self, answer):
        r = run_agent(PolicyAgent(use_llm=False), "PD-1 antibody dosing studies",
                      {"answer": answer})
        assert "dosing" not in _kinds(r)
        assert r.ok is True


# --------------------------------------------------------------------------- #
# Abstention -- spec 4.2 routing, and the demo's query 1
# --------------------------------------------------------------------------- #
class TestAbstention:
    def test_scouting_query_abstains(self):
        """Demo query 1. 'Two agents declined. That's the design.'"""
        agent = PolicyAgent(use_llm=False)
        assert agent.should_fire(SCOUTING_QUERY, "") is False
        r = run_agent(agent, SCOUTING_QUERY, {})
        assert r.abstained is True
        assert r.ok is True, "abstention travels with ok=True (base.py invariant 1)"
        assert r.payload["flags"] == [] if "flags" in r.payload else True
        assert "reason" in r.payload

    def test_abstention_renders_as_abstained_in_the_trace(self):
        row = to_trace_row(run_agent(PolicyAgent(use_llm=False), SCOUTING_QUERY, {}))
        assert row["agent"] == "policy"
        assert row["status"] == "abstained"
        assert row["conf"] is None      # not 0.0 -- that would read as a low score

    def test_answer_can_trigger_a_query_that_would_not(self):
        agent = PolicyAgent(use_llm=False)
        assert agent.should_fire(SCOUTING_QUERY, "") is False
        assert agent.should_fire(SCOUTING_QUERY,
                                 "This antibody is FDA-approved for melanoma.") is True

    def test_fired_but_clean_is_not_abstention(self):
        """
        'Read it, found nothing' and 'nothing here to read' are different results
        and the trace must not conflate them.
        """
        r = run_agent(PolicyAgent(use_llm=False), SCOUTING_QUERY,
                      {"answer": TRIAL_ABSTRACT})
        assert r.abstained is False
        assert r.payload["flags"] == []
        assert r.ok is True
        assert to_trace_row(r)["status"] == "fired"

    def test_pure_ip_query_abstains_rather_than_certifying(self):
        """
        Scope (b) is not built. A freedom-to-operate query must not fire and come
        back clean -- that reads as 'checked, no IP issues', which is a lie of
        omission. It abstains instead.
        """
        agent = PolicyAgent(use_llm=False)
        assert agent.should_fire("when does the B7-H1 patent family expire?", "") is False


# --------------------------------------------------------------------------- #
# Efficacy claims
# --------------------------------------------------------------------------- #
class TestEfficacyClaims:
    @pytest.mark.parametrize("answer", [
        "Pembrolizumab cures stage IV NSCLC.",
        "This antibody is highly effective for melanoma.",
        "The compound has been proven to eliminate tumors.",
    ])
    def test_unhedged_efficacy_claims_warn(self, answer):
        r = run_agent(PolicyAgent(use_llm=False), "PD-L1 antibody efficacy",
                      {"answer": answer})
        assert "efficacy_claim" in _kinds(r)
        flag = next(f for f in r.payload["flags"] if f["kind"] == "efficacy_claim")
        assert flag["severity"] == "warn"
        assert r.ok is True, "efficacy claims warn; they do not block"

    @pytest.mark.parametrize("answer", [
        "US6803192B1 reports that B7-H1 blockade may be effective in murine models.",
        "The patent claims the antibody is effective against solid tumors.",
        "A Phase 2 trial reported the combination was effective in 40% of patients.",
        "Preclinical data suggests the antibody eradicates tumors in mice.",
    ])
    def test_hedged_or_attributed_efficacy_does_not_flag(self, answer):
        """This is the register the tool SHOULD write in. Punishing it is backwards."""
        r = run_agent(PolicyAgent(use_llm=False), "PD-L1 antibody efficacy",
                      {"answer": answer})
        assert "efficacy_claim" not in _kinds(r)

    def test_a_question_about_efficacy_is_not_an_efficacy_claim(self):
        """The user asking is not the tool claiming. Flags target the ANSWER."""
        r = run_agent(PolicyAgent(use_llm=False),
                      "is pembrolizumab effective for NSCLC?", {"answer": ""})
        assert "efficacy_claim" not in _kinds(r)


# --------------------------------------------------------------------------- #
# Stage overstatement
# --------------------------------------------------------------------------- #
class TestStageOverstatement:
    def test_approval_claim_warns(self):
        r = run_agent(PolicyAgent(use_llm=False), "CC-96191 status",
                      {"answer": "CC-96191 is approved for relapsed AML."})
        assert "stage_overstatement" in _kinds(r)
        flag = next(f for f in r.payload["flags"]
                    if f["kind"] == "stage_overstatement")
        assert flag["severity"] == "warn"
        assert r.ok is True

    def test_evidence_phase_language_enriches_but_does_not_escalate(self):
        ctx = {"answer": "CC-96191 is FDA-approved for relapsed AML.",
               "candidates": [{"doc_id": "clinicaltrials:NCT04789655",
                               "title": "Study of CC-96191",
                               "abstract": TRIAL_ABSTRACT,
                               "source_url": "https://clinicaltrials.gov/study/NCT04789655"}]}
        r = run_agent(PolicyAgent(use_llm=False), "CC-96191 status", ctx)
        flag = next(f for f in r.payload["flags"]
                    if f["kind"] == "stage_overstatement")
        assert flag["severity"] == "warn", "contradiction is noticed, not adjudicated"
        assert r.ok is True
        assert "NCT04789655" in flag["rationale"]
        assert r.evidence and r.evidence[0]["doc_id"] == "clinicaltrials:NCT04789655"
        assert r.evidence[0]["source_url"].startswith("https://")

    def test_stage_flag_without_evidence_says_nothing_was_checked(self):
        r = run_agent(PolicyAgent(use_llm=False), "CC-96191 status",
                      {"answer": "CC-96191 is approved for relapsed AML."})
        flag = next(f for f in r.payload["flags"]
                    if f["kind"] == "stage_overstatement")
        assert "No retrieved document was checked" in flag["rationale"]
        assert r.evidence == []

    def test_evidence_key_shape_also_works(self):
        """base.make_evidence emits quote_span; the retriever emits abstract."""
        ctx = {"answer": "CC-96191 has been approved.",
               "evidence": [{"doc_id": "clinicaltrials:NCT04789655",
                             "quote_span": "This Phase 1, clinical study of CC-96191",
                             "source_url": "https://example.org/x"}]}
        r = run_agent(PolicyAgent(use_llm=False), "CC-96191 status", ctx)
        assert r.evidence[0]["doc_id"] == "clinicaltrials:NCT04789655"


# --------------------------------------------------------------------------- #
# The LLM layer is optional and may never touch the gate
# --------------------------------------------------------------------------- #
class _StubLLM:
    """Minimal LLMClient stand-in. Never touches a socket."""

    def __init__(self, response, available=True):
        self._response = response
        self.available = available
        self.calls = []

    def chat_json(self, messages, schema_hint, **kw):
        self.calls.append(messages)
        return self._response


class TestLLMLayer:
    def test_no_api_key_means_deterministic_mode(self, monkeypatch):
        """Must hold in CI, where DH_LLM_API_KEY is absent."""
        monkeypatch.delenv("DH_LLM_API_KEY", raising=False)
        r = run_agent(PolicyAgent(), DOSING_QUERY, {})   # use_llm defaults True
        assert r.payload["mode"] == "deterministic"
        assert "dosing" in _kinds(r), "rules are the floor and run regardless"

    def test_llm_adds_a_flag_and_mode_becomes_hybrid(self):
        stub = _StubLLM({"flags": [{"kind": "efficacy_claim",
                                    "span": "it knocks the tumor right out",
                                    "rationale": "idiomatic efficacy assertion"}]})
        r = run_agent(PolicyAgent(llm=stub),
                      "pembrolizumab efficacy",
                      {"answer": "The drug is approved and it knocks the tumor "
                                 "right out."})
        assert r.payload["mode"] == "hybrid"
        assert "efficacy_claim" in _kinds(r)
        llm_flag = next(f for f in r.payload["flags"] if f["source"] == "llm")
        assert llm_flag["rationale"] == "idiomatic efficacy assertion"

    def test_llm_proposed_dosing_is_clamped_to_warn_and_cannot_block(self):
        """
        The gate is a property of the answer, not of the network. An LLM-only flag
        never flips ok=False, or a flaky API would change what the tool blocks.
        """
        stub = _StubLLM({"flags": [{"kind": "dosing", "span": "give it a go at 200mg",
                                    "rationale": "informal dosing suggestion"}]})
        r = run_agent(PolicyAgent(llm=stub), "pembrolizumab dose in NSCLC",
                      {"answer": "Just give it a go at 200mg."})
        llm_flag = next(f for f in r.payload["flags"] if f["source"] == "llm")
        assert llm_flag["kind"] == "dosing"
        assert llm_flag["severity"] == "warn"
        assert r.ok is True, "no rule flag blocked, so the LLM must not block either"

    def test_llm_failure_falls_back_and_says_deterministic(self):
        """chat_json returns None on transport failure or empty content."""
        r = run_agent(PolicyAgent(llm=_StubLLM(None)), DOSING_QUERY, {})
        assert r.payload["mode"] == "deterministic", \
            "must never claim a review that did not happen"
        assert r.ok is False, "the rule-based block is unaffected by the LLM failing"

    def test_llm_unavailable_is_not_called(self):
        stub = _StubLLM({"flags": []}, available=False)
        r = run_agent(PolicyAgent(llm=stub), DOSING_QUERY, {})
        assert stub.calls == []
        assert r.payload["mode"] == "deterministic"

    @pytest.mark.parametrize("response", [
        {"flags": "not a list"},
        {"no_flags_key": []},
        {"flags": [{"kind": "made_up_kind", "span": "x", "rationale": "y"}]},
        {"flags": [{"kind": "dosing", "rationale": "no span at all"}]},
        {"flags": ["a bare string", None]},
    ])
    def test_off_contract_llm_output_is_dropped_not_coerced(self, response):
        r = run_agent(PolicyAgent(llm=_StubLLM(response)), DOSING_QUERY, {})
        assert all(f["source"] == "rules" for f in r.payload["flags"])
        assert "dosing" in _kinds(r)

    def test_llm_does_not_duplicate_a_rule_flag(self):
        stub = _StubLLM({"flags": [{"kind": "dosing", "span": DOSING_QUERY,
                                    "rationale": "same thing the rules found"}]})
        r = run_agent(PolicyAgent(llm=stub), DOSING_QUERY, {})
        assert len(_kinds(r)) == 1
        assert r.payload["flags"][0]["source"] == "rules"
        assert r.payload["flags"][0]["severity"] == "block"


# --------------------------------------------------------------------------- #
# Contract, determinism, honesty
# --------------------------------------------------------------------------- #
class TestContract:
    def test_flag_schema(self):
        r = run_agent(PolicyAgent(use_llm=False), DOSING_QUERY, {})
        for flag in r.payload["flags"]:
            assert set(flag) == {"kind", "severity", "span", "rationale", "source"}
            assert flag["kind"] in {"dosing", "efficacy_claim", "stage_overstatement"}
            assert flag["severity"] in {"block", "warn"}
            assert isinstance(flag["rationale"], str) and flag["rationale"]

    def test_payload_declares_scope_b_is_absent(self):
        """The deck must not be able to read 'no IP flags' as 'IP is clear'."""
        r = run_agent(PolicyAgent(use_llm=False), DOSING_QUERY, {})
        assert "priority_date" in r.payload["scope"]
        assert r.payload["rules_unmeasured"] is True

    def test_deterministic_across_runs(self):
        """Stage 09 verifies reproducibility; a reordering flag list is a diff."""
        answer = ("Pembrolizumab is FDA-approved and cures NSCLC. You should give "
                  "the patient 200 mg every 3 weeks.")
        agent = PolicyAgent(use_llm=False)
        first = agent.run(DOSING_QUERY, {"answer": answer}).payload["flags"]
        for _ in range(3):
            assert agent.run(DOSING_QUERY, {"answer": answer}).payload["flags"] == first

    def test_blocks_sort_first(self):
        answer = ("Pembrolizumab is FDA-approved and cures NSCLC. You should give "
                  "the patient 200 mg every 3 weeks.")
        r = run_agent(PolicyAgent(use_llm=False), DOSING_QUERY, {"answer": answer})
        assert _kinds(r)[0] == "dosing"
        assert set(_kinds(r)) == {"dosing", "efficacy_claim", "stage_overstatement"}

    def test_confidence_is_ordinal_and_clean_is_zero(self):
        blocked = run_agent(PolicyAgent(use_llm=False), DOSING_QUERY, {})
        warned = run_agent(PolicyAgent(use_llm=False), "efficacy",
                           {"answer": "The drug cures cancer."})
        clean = run_agent(PolicyAgent(use_llm=False), SCOUTING_QUERY,
                          {"answer": TRIAL_ABSTRACT})
        assert blocked.confidence > warned.confidence > clean.confidence == 0.0

    def test_latency_is_measured(self):
        r = run_agent(PolicyAgent(use_llm=False), SCOUTING_QUERY,
                      {"answer": TRIAL_ABSTRACT})
        assert r.latency_ms > 0
        assert to_trace_row(r)["status"] == "fired"

    def test_a_deliberate_block_is_distinguishable_from_a_crash(self):
        """
        PINS A KNOWN WART. base.to_trace_row maps ok=False to status="error", so a
        correct policy block renders as if the agent crashed. base.py is shared and
        not ours to change; the payload carries the discriminator instead. If the
        orchestrator renders policy's row from ok alone, the refusal slide says
        "policy: error", which undercuts the point it is making.
        """
        r = run_agent(PolicyAgent(use_llm=False), DOSING_QUERY, {})
        assert r.ok is False
        assert to_trace_row(r)["status"] == "error"      # the wart, pinned
        assert r.error is None                            # ...but nothing crashed
        assert r.payload["blocking_flags"] == ["dosing"]  # ...and this says why

    @pytest.mark.parametrize("answer", [
        None, "", {"answer": "The drug cures cancer."}, {"text": "The drug cures cancer."},
    ])
    def test_answer_coercion_shapes(self, answer):
        """The orchestrator has passed str, dict and AgentResult at various points."""
        r = run_agent(PolicyAgent(use_llm=False), "efficacy of the drug",
                      {"answer": answer})
        assert r.ok is True

    def test_answer_coercion_from_agent_result(self):
        from agents.base import AgentResult
        drafted = AgentResult(agent="synthesis", ok=True,
                              payload={"answer": "The drug cures cancer."},
                              evidence=[], confidence=0.5, latency_ms=1.0)
        r = run_agent(PolicyAgent(use_llm=False), "efficacy", {"answer": drafted})
        assert "efficacy_claim" in _kinds(r)

    def test_missing_ctx_keys_do_not_crash(self):
        r = run_agent(PolicyAgent(use_llm=False), DOSING_QUERY,
                      {"candidates": None, "evidence": [{"no_doc_id": 1}, "junk"]})
        assert r.payload["flags"]


CORPUS = "/home/alex/discovery_hub/data/normalized/docs.jsonl"
PROBE_N = 20000
# Ceilings, not targets. Measured at 4 / 244 / 331 when the lexicons were tuned;
# these leave headroom for honest lexicon growth while catching a blow-up.
PROBE_CEILINGS = {"dosing": 15, "efficacy_claim": 400, "stage_overstatement": 500}


@pytest.fixture(scope="module")
def probe_counts():
    """
    Run the rules over real corpus abstracts, fed in as if the tool had emitted
    them. Skips when the corpus is absent -- the data lives outside the repo, so
    CI and a fresh clone do not have it.
    """
    import collections
    import itertools
    import json
    import os

    if not os.path.exists(CORPUS):
        pytest.skip(f"corpus not present at {CORPUS} (lives outside the repo)")
    agent = PolicyAgent(use_llm=False)
    found = collections.Counter()
    with open(CORPUS) as fh:
        for line in itertools.islice(fh, PROBE_N):
            abstract = (json.loads(line).get("abstract") or "").strip()
            if not abstract:
                continue
            r = agent.run("technology scouting", {"answer": abstract})
            for flag in r.payload.get("flags", []):
                found[flag["kind"]] += 1
    return found


class TestCorpusFalsePositiveProbe:
    """
    The measurement that drove the lexicon tuning, pinned so it cannot silently
    regress. EVERY flag the probe finds is by construction a false positive: a
    trial registration is descriptive text, not advice, and not the tool's own
    claim.

    This is NOT an accuracy test. It measures only what the rules wrongly catch,
    never what they correctly catch, because no labelled set of unsafe answers
    exists. Do not derive a precision number from it.
    """

    @pytest.mark.parametrize("kind", ["dosing", "efficacy_claim", "stage_overstatement"])
    def test_false_positives_stay_under_ceiling(self, probe_counts, kind):
        assert probe_counts[kind] <= PROBE_CEILINGS[kind], (
            f"{kind} false positives on {PROBE_N} real abstracts rose to "
            f"{probe_counts[kind]} (ceiling {PROBE_CEILINGS[kind]}). A lexicon "
            f"change made this agent noisier on ordinary corpus text."
        )

    def test_the_blocking_flag_is_rare_on_descriptive_text(self, probe_counts):
        """
        Dosing is the only flag that sets ok=False, so its false-positive rate is
        the one that decides whether this agent is usable or gets switched off.
        """
        assert probe_counts["dosing"] / PROBE_N < 0.001


class TestHelpers:
    def test_clip_truncates_long_spans(self):
        assert len(_clip("word " * 200)) <= 240
        assert _clip("  a   b  ") == "a b"

    def test_sentences_ignores_empty(self):
        assert _sentences("") == []
        assert _sentences("A. B!  C?") == ["A.", "B!", "C?"]

    def test_dedupe_keeps_first_and_is_case_insensitive(self):
        flags = [{"kind": "dosing", "severity": "block", "span": "Give 200 mg",
                  "rationale": "a", "source": "rules"},
                 {"kind": "dosing", "severity": "warn", "span": "give 200 mg",
                  "rationale": "b", "source": "llm"}]
        out = _dedupe_and_sort(flags)
        assert len(out) == 1 and out[0]["source"] == "rules"
