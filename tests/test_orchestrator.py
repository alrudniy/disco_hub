"""
Tests for the orchestrator: routing, the gate, failure containment, the trace.

Everything is a fake. Constructing an Orchestrator must never load an index, a
graph or a model, and these tests would be unrunnable on a 30 GB box if it did.

The tests that matter most are the ones about HONESTY OF THE TRACE, not the ones
about the happy path: a correct refusal that renders as "error" and a sequential
pipeline that renders as concurrent are both failures of this module even though
the answer would be identical.
"""
from __future__ import annotations

from agents.base import AgentResult
from agents.orchestrator import Orchestrator

EVIDENCE = [{"doc_id": "uspto:US6803192B1", "quote_span": "B7-H1 is an immunoregulatory molecule.",
             "source_url": "https://patents.google.com/patent/US6803192B1"}]


class _Stub:
    """An agent that returns whatever it was handed."""

    def __init__(self, name, result, fires=True):
        self.name = name
        self._result = result
        self._fires = fires
        self.ctx_seen = None
        self.ran = False

    def run(self, query, ctx):
        self.ran = True
        self.ctx_seen = ctx
        return self._result

    def should_fire(self, query, answer=""):
        return self._fires


class _Boom:
    name = "boom"

    def __init__(self, name="boom"):
        self.name = name

    def run(self, query, ctx):
        raise RuntimeError("kaboom")

    def should_fire(self, query, answer=""):
        return True


def _retrieval(name="retrieval"):
    return _Stub(name, AgentResult(
        agent=name, ok=True,
        payload={"candidates": [{"doc_id": "uspto:US6803192B1", "title": "B7-H1",
                                 "abstract": "B7-H1 is an immunoregulatory molecule.",
                                 "source_url": EVIDENCE[0]["source_url"],
                                 "rerank_score": 0.81}]},
        evidence=list(EVIDENCE), confidence=0.81, latency_ms=0.0))


def _synthesis(text="B7-H1 is an immunoregulatory molecule.", mode="deterministic"):
    return _Stub("synthesis", AgentResult(
        agent="synthesis", ok=True,
        payload={"answer": text, "claims": [{"text": text, "doc_id": EVIDENCE[0]["doc_id"]}],
                 "mode": mode},
        evidence=list(EVIDENCE), confidence=0.81, latency_ms=0.0))


def _verifier(verdict="pass", unsupported=0):
    return _Stub("verifier", AgentResult(
        agent="verifier", ok=(verdict == "pass"),
        payload={"verdict": verdict, "unsupported_count": unsupported,
                 "contradicted_count": 0, "claims": [], "mode": "deterministic"},
        evidence=[], confidence=0.92, latency_ms=0.0))


def _policy(blocking=(), fires=True):
    flags = [{"kind": k, "severity": "block"} for k in blocking]
    return _Stub("policy", AgentResult(
        agent="policy", ok=not blocking,
        payload={"flags": flags, "blocking_flags": list(blocking), "mode": "deterministic"},
        evidence=[], confidence=0.85 if blocking else 0.0, latency_ms=0.0), fires=fires)


def _row(out, agent):
    return next(r for r in out["trace"] if r["agent"] == agent)


# --------------------------------------------------------------------------- #
# 1. ROUTE
# --------------------------------------------------------------------------- #
class TestRouting:
    def test_gap_agent_that_declines_still_appears_in_the_trace(self):
        # The demo points AT the abstentions. An agent that silently never appears
        # cannot be pointed at, so a router decline gets a real row.
        gap = _Stub("expertise-gap", None, fires=False)
        out = Orchestrator(retrieval=_retrieval(), synthesis=_synthesis(),
                           verifier=_verifier(), expertise_gap=gap).run("q")
        assert gap.ran is False
        row = _row(out, "expertise-gap")
        assert row["status"] == "abstained"
        assert "names no organization" in row["reason"]

    def test_router_decline_reports_its_own_measured_latency(self):
        gap = _Stub("expertise-gap", None, fires=False)
        out = Orchestrator(retrieval=_retrieval(), synthesis=_synthesis(),
                           expertise_gap=gap).run("q")
        assert _row(out, "expertise-gap")["ms"] >= 0.0

    def test_gap_agent_receives_retrievals_candidates(self):
        gap = _Stub("expertise-gap", AgentResult.abstain("expertise-gap", "no org"))
        Orchestrator(retrieval=_retrieval(), synthesis=_synthesis(),
                     expertise_gap=gap).run("q")
        assert gap.ctx_seen["candidates"][0]["doc_id"] == "uspto:US6803192B1"

    def test_synthesis_receives_the_gap_result(self):
        gap_result = AgentResult(agent="expertise-gap", ok=True,
                                 payload={"gaps": [{"doc_id": "uspto:US6803192B1"}]},
                                 evidence=[], confidence=0.6, latency_ms=0.0)
        syn = _synthesis()
        Orchestrator(retrieval=_retrieval(), synthesis=syn,
                     expertise_gap=_Stub("expertise-gap", gap_result)).run("q")
        assert syn.ctx_seen["gaps"] is gap_result

    def test_an_unwired_agent_is_not_reported_as_an_abstention(self):
        # Abstention is a judgment an agent made. Claiming one for an agent that was
        # never constructed puts a decision in the trace that nothing ever took.
        out = Orchestrator(retrieval=_retrieval(), synthesis=_synthesis()).run("q")
        for name in ("expertise-gap", "policy", "verifier"):
            assert _row(out, name)["status"] == "not wired"

    def test_policy_sees_the_drafted_answer_not_just_the_query(self):
        pol = _policy()
        Orchestrator(retrieval=_retrieval(), synthesis=_synthesis("drafted text."),
                     verifier=_verifier(), policy=pol).run("q")
        assert pol.ctx_seen["answer"] == "drafted text."


# --------------------------------------------------------------------------- #
# 2. FAN OUT -- honesty about sequencing
# --------------------------------------------------------------------------- #
class TestFanOut:
    def test_the_pipeline_reports_itself_as_sequential(self):
        # Spec 4.2 says "run concurrently" and its own sketch makes them sequentially
        # dependent. We run sequentially and SAY so rather than wrapping a dependency
        # in a ThreadPoolExecutor for show.
        out = Orchestrator(retrieval=_retrieval(), synthesis=_synthesis()).run("q")
        assert out["concurrency"] == "sequential"

    def test_no_thread_pool_is_used(self):
        import agents.orchestrator as mod
        src = open(mod.__file__).read()
        # The docstring discusses ThreadPoolExecutor; the code must not import one.
        assert "from concurrent" not in src and "import concurrent" not in src


# --------------------------------------------------------------------------- #
# 3. AGGREGATE -- never average
# --------------------------------------------------------------------------- #
class TestAggregation:
    def test_confidences_are_carried_separately_per_agent(self):
        out = Orchestrator(retrieval=_retrieval(), synthesis=_synthesis(),
                           verifier=_verifier()).run("q")
        assert out["confidences"]["retrieval"] == 0.81
        assert out["confidences"]["verifier"] == 0.92
        # No combined/overall/mean confidence exists anywhere in the output.
        assert not {"overall_confidence", "mean_confidence", "confidence"} & set(out)

    def test_an_abstaining_agents_confidence_is_none_not_zero(self):
        # 0.0 would read as a measured low score in the trace table.
        gap = _Stub("expertise-gap", None, fires=False)
        out = Orchestrator(retrieval=_retrieval(), synthesis=_synthesis(),
                           expertise_gap=gap).run("q")
        assert _row(out, "expertise-gap")["conf"] is None


# --------------------------------------------------------------------------- #
# 4. GATE
# --------------------------------------------------------------------------- #
class TestTheGate:
    def test_verifier_fail_degrades_to_evidence_only(self):
        out = Orchestrator(retrieval=_retrieval(), synthesis=_synthesis(),
                           verifier=_verifier("fail", unsupported=2)).run("q")
        assert out["degraded"] is True
        assert out["verified"] is False
        assert out["answer"]["mode"] == "evidence_only"
        assert out["answer"]["text"] == ""        # the synthesis is GONE, not caveated
        assert out["answer"]["documents"][0]["doc_id"] == "uspto:US6803192B1"
        assert "2 claim(s) not supported" in out["answer"]["reason"]

    def test_policy_block_degrades_even_when_the_gate_passes(self):
        out = Orchestrator(retrieval=_retrieval(), synthesis=_synthesis(),
                           verifier=_verifier("pass"), policy=_policy(("dosing",))).run("q")
        assert out["degraded"] is True
        assert out["answer"]["mode"] == "evidence_only"
        assert "policy blocked" in out["answer"]["reason"]
        assert "dosing" in out["answer"]["reason"]

    def test_evidence_survives_degradation(self):
        # The synthesis is withheld; the documents the reader can open are not.
        out = Orchestrator(retrieval=_retrieval(), synthesis=_synthesis(),
                           verifier=_verifier("fail")).run("q")
        assert out["evidence"] == EVIDENCE

    def test_clean_run_ships_the_synthesis(self):
        out = Orchestrator(retrieval=_retrieval(), synthesis=_synthesis(),
                           verifier=_verifier("pass"), policy=_policy()).run("q")
        assert out["degraded"] is False and out["verified"] is True
        assert out["answer"]["text"] == "B7-H1 is an immunoregulatory molecule."
        assert out["verdict"] == "pass"

    def test_a_crashed_verifier_fails_closed(self):
        # If the check did not run, the claim is unchecked -- and an unchecked claim
        # is exactly what the gate exists to stop.
        out = Orchestrator(retrieval=_retrieval(), synthesis=_synthesis(),
                           verifier=_Boom("verifier")).run("q")
        assert out["degraded"] is True
        assert out["verified"] is False
        assert "could not run" in out["answer"]["reason"]

    def test_a_crashed_verifier_carrying_a_stale_pass_is_not_trusted(self):
        # ISOLATES the fail-closed guard. The test above passes even without it:
        # run_agent's failure path leaves payload={}, so `verdict != "pass"` already
        # returns False and the guard is never reached. Mutation-testing caught that
        # -- deleting the guard killed no test. This is the case that needs it: an
        # agent that raised AFTER writing a verdict, so ok=False and error is set
        # while payload still says "pass". A check that did not complete cannot
        # certify anything, whatever it left behind in the payload.
        v = _Stub("verifier", AgentResult(
            agent="verifier", ok=False, payload={"verdict": "pass"}, evidence=[],
            confidence=0.0, latency_ms=0.0, error="RuntimeError: died mid-check"))
        out = Orchestrator(retrieval=_retrieval(), synthesis=_synthesis(),
                           verifier=v).run("q")
        assert out["degraded"] is True
        assert out["verified"] is False
        assert "could not run" in out["answer"]["reason"]
        # ...and it must render as a crash, not as the pass it claims to be.
        assert _row(out, "verifier")["status"] == "error"

    def test_verifier_abstention_does_not_degrade(self):
        # Nothing was synthesized => nothing to gate => no claim to withhold.
        v = _Stub("verifier", AgentResult.abstain("verifier", "no answer text to check"))
        out = Orchestrator(retrieval=_retrieval(), synthesis=_synthesis(""),
                           verifier=v).run("q")
        assert out["degraded"] is False

    def test_no_verifier_wired_means_not_verified(self):
        out = Orchestrator(retrieval=_retrieval(), synthesis=_synthesis()).run("q")
        assert out["verified"] is False   # never claim a gate that does not exist


# --------------------------------------------------------------------------- #
# 5. TRACE -- a correct refusal must not read as a crash
# --------------------------------------------------------------------------- #
class TestTrace:
    def test_verifier_fail_renders_as_a_verdict_not_an_error(self):
        # base.to_trace_row maps any ok=False to "error"; the verifier returns
        # ok=False for a considered FAIL. Spec line 310 shows `verifier | pass`.
        out = Orchestrator(retrieval=_retrieval(), synthesis=_synthesis(),
                           verifier=_verifier("fail")).run("q")
        row = _row(out, "verifier")
        assert row["status"] == "fail"
        assert row["error"] is None
        assert row["conf"] == 0.92

    def test_policy_block_renders_as_blocked_not_error(self):
        # On the refusal slide, "policy: error" undercuts the exact point the slide
        # makes -- that the refusal was deliberate.
        out = Orchestrator(retrieval=_retrieval(), synthesis=_synthesis(),
                           verifier=_verifier(), policy=_policy(("dosing",))).run("q")
        row = _row(out, "policy")
        assert row["status"] == "blocked"
        assert row["error"] is None
        assert "dosing" in row["reason"]

    def test_a_clean_policy_review_shows_no_score_rather_than_zero(self):
        # policy's confidence is 0.0 when it flags nothing ("no verdict to be
        # confident about"). Rendering that literally reads as "0% confident",
        # i.e. a measured low score, when it means the review came back clean.
        out = Orchestrator(retrieval=_retrieval(), synthesis=_synthesis(),
                           verifier=_verifier(), policy=_policy()).run("q")
        row = _row(out, "policy")
        assert row["status"] == "fired"
        assert row["conf"] is None
        assert "no policy flags raised" in row["reason"]

    def test_a_real_crash_still_renders_as_an_error(self):
        # The discriminator must cut both ways, or it is not a discriminator.
        out = Orchestrator(retrieval=_retrieval(), synthesis=_synthesis(),
                           verifier=_verifier(), policy=_Boom("policy")).run("q")
        row = _row(out, "policy")
        assert row["status"] == "error"
        assert "kaboom" in row["error"]

    def test_trace_row_order_follows_execution_order(self):
        out = Orchestrator(retrieval=_retrieval(), synthesis=_synthesis(),
                           verifier=_verifier(), policy=_policy(),
                           expertise_gap=_Stub("expertise-gap", None, fires=False)).run("q")
        assert [r["agent"] for r in out["trace"]] == [
            "retrieval", "expertise-gap", "synthesis", "policy", "verifier"]

    def test_every_row_carries_a_latency(self):
        out = Orchestrator(retrieval=_retrieval(), synthesis=_synthesis(),
                           verifier=_verifier(), policy=_policy()).run("q")
        assert all(isinstance(r["ms"], float) for r in out["trace"])


# --------------------------------------------------------------------------- #
# Failure containment (base.py invariant 2)
# --------------------------------------------------------------------------- #
class TestOneAgentFailingDoesNotKillTheRun:
    def test_broken_gap_agent_is_recorded_and_the_run_continues(self):
        out = Orchestrator(retrieval=_retrieval(), synthesis=_synthesis(),
                           verifier=_verifier(), expertise_gap=_Boom("expertise-gap")).run("q")
        assert _row(out, "expertise-gap")["status"] == "error"
        assert out["answer"]["text"]          # the run still produced an answer
        assert out["gaps"] is None

    def test_broken_router_is_contained(self):
        class _BadRouter:
            name = "expertise-gap"
            def should_fire(self, q):
                raise ValueError("router exploded")
            def run(self, q, ctx):
                raise AssertionError("must not be reached")
        out = Orchestrator(retrieval=_retrieval(), synthesis=_synthesis(),
                           verifier=_verifier(), expertise_gap=_BadRouter()).run("q")
        row = _row(out, "expertise-gap")
        assert row["status"] == "error" and "router exploded" in row["error"]
        assert out["verified"] is True

    def test_broken_retrieval_still_produces_a_trace(self):
        out = Orchestrator(retrieval=_Boom("retrieval"), synthesis=_synthesis(),
                           verifier=_verifier()).run("q")
        assert _row(out, "retrieval")["status"] == "error"
        assert out["evidence"] == []

    def test_broken_synthesis_does_not_prevent_the_gate(self):
        out = Orchestrator(retrieval=_retrieval(), synthesis=_Boom("synthesis"),
                           verifier=_verifier("fail")).run("q")
        assert _row(out, "synthesis")["status"] == "error"
        assert out["degraded"] is True


# --------------------------------------------------------------------------- #
# Output shape (spec 4.2 / the architecture diagram)
# --------------------------------------------------------------------------- #
class TestOutputShape:
    def test_has_every_key_the_spec_names(self):
        out = Orchestrator(retrieval=_retrieval(), synthesis=_synthesis(),
                           verifier=_verifier(), policy=_policy()).run("q")
        for key in ("answer", "evidence", "gaps", "flags", "trace", "verified",
                    "degraded", "verdict"):
            assert key in out, key

    def test_gaps_travel_in_their_own_channel(self):
        gap_result = AgentResult(agent="expertise-gap", ok=True,
                                 payload={"gaps": [{"doc_id": "x"}], "unmeasured": True},
                                 evidence=[], confidence=0.6, latency_ms=0.0)
        out = Orchestrator(retrieval=_retrieval(), synthesis=_synthesis(),
                           verifier=_verifier(),
                           expertise_gap=_Stub("expertise-gap", gap_result)).run("q")
        assert out["gaps"]["gaps"] == [{"doc_id": "x"}]
        assert out["gaps"]["unmeasured"] is True

    def test_an_abstaining_gap_agent_exposes_its_reason_not_a_payload(self):
        gap = _Stub("expertise-gap", AgentResult.abstain("expertise-gap", "owns only 1"))
        out = Orchestrator(retrieval=_retrieval(), synthesis=_synthesis(),
                           verifier=_verifier(), expertise_gap=gap).run("q")
        assert out["gaps"] is None
        assert out["gaps_abstained_reason"] == "owns only 1"

    def test_constructing_an_orchestrator_loads_nothing_heavy(self):
        # A real Retriever is ~20 GB. Construction must stay free.
        o = Orchestrator(retrieval=_retrieval(), synthesis=_synthesis())
        assert o.verifier is None
