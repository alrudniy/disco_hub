"""
The orchestrator: routing, aggregation, the gate, and the trace.

Spec 4.2's five responsibilities, in order: route -> fan out -> aggregate -> gate
-> trace. This module owns the composition and nothing else -- it contains no
retrieval, no graph traversal and no prompt. Every agent is called through
`base.run_agent`, which times it and converts an exception into ok=False, so one
broken agent degrades the run instead of killing it.

--------------------------------------------------------------------------- #
FAN-OUT: THE SPEC CONTRADICTS ITSELF, AND THIS PIPELINE RUNS SEQUENTIALLY
--------------------------------------------------------------------------- #
Spec 4.2 step 2 says retrieval and expertise-gap "are independent -- run
concurrently (ThreadPoolExecutor, both are I/O-bound)". Spec 4.2's own code sketch,
eleven lines later, says:

    gaps = self.expertise_gap.run(query, {"candidates": r.payload["candidates"]})

Those cannot both be true. The sketch feeds retrieval's OUTPUT into expertise-gap's
INPUT, which is a sequential dependency by construction, and the implemented agent
agrees: `ExpertiseGapAgent.run` treats the candidates as the query's therapeutic
area and abstains outright when they are missing ("the retrieval agent supplied no
candidate technologies to compare the portfolio against"). It cannot score a gap
without knowing what to score against, and it deliberately does not re-retrieve --
a second Retriever is ~20 GB of RAM on a 30 GB box.

DECISION: option (b) -- run them sequentially and SAY SO. No ThreadPoolExecutor.

The honest reasoning, since the spec offers a real alternative:

  * Option (a) -- split expertise-gap into prefetch (link the org, resolve the
    portfolio, cluster it) and score (compare candidates to the centroids) -- is
    genuinely available. That first half really is independent of retrieval and
    really could overlap it. It is the better design at scale and it is what this
    should become if the gap agent is ever put on a latency budget.

  * It is not worth it here, for reasons that are measured rather than felt. The
    independent half is ~2.9 s warm, of which ~1 s is KMeans -- and the CPU-bound
    parts (KMeans, numpy) hold the GIL, so "both are I/O-bound" is not true of this
    half either. Overlapping it against retrieval's ~840 ms saves under a second on
    a demo that runs three queries. Buying that second costs a refactor of a
    751-line sibling module whose `run()` is pinned by 40 tests and which was hand-
    validated against real portfolios (BMS 3,155 techs, Chen Lieping's 52 patents)
    -- i.e. it puts the money-shot agent at regression risk to shave 1 s off a
    rehearsed demo. Wrong trade, today.

  * The one-time cost that ACTUALLY dominates is the entity-link surface index:
    ~3.3 s and ~0.5 GB, built once. So `from_config` builds it once and injects it
    into the gap agent, which is where the real second went. That is a fix, not a
    workaround.

What is explicitly NOT done: wrapping a sequential dependency in a ThreadPoolExecutor
so the trace shows two agents "running concurrently". That would be a lie told by a
diagram, and this spec exists because someone overclaimed. The trace reports the
pipeline as sequential (`concurrency: "sequential"`) and the latencies add up,
because they really do add up.

--------------------------------------------------------------------------- #
THE GATE, AND WHY IT IS NOT `if not v.ok`
--------------------------------------------------------------------------- #
Spec 4.2 branches on `if not v.ok`. That is right for the verifier and wrong for
policy, and both agents' authors flagged it independently:

  * `base.to_trace_row` maps ANY ok=False to status="error", so a verifier that
    correctly refuses an answer, or a policy agent that correctly blocks one, would
    both render in the demo's trace as if they had CRASHED -- on the refusal slide,
    which is the slide whose entire point is that the refusal was deliberate. The
    trace rows for those two agents are therefore rendered from their payloads
    (`verdict`, `blocking_flags`), not from `ok` alone.
  * A deliberate block is (ok=False, error=None); a crash is (ok=False, error=...).
    That discriminator is what `_status_of` reads.

--------------------------------------------------------------------------- #
ON THE WORD "verified"
--------------------------------------------------------------------------- #
The output carries `verified: bool` because spec 4.2 fixes that key. It means "the
evidence gate passed" -- an LLM (or a lexical proxy) found every claim tied to a
retrieved passage. It does NOT mean the answer is true, and spec section 6 is
explicit that the word to use out loud is EVIDENCE-GATED. `verdict` and
`gate_reason` carry the honest detail; the demo prints the honest words.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from agents.base import AgentResult, run_agent, to_trace_row
from discovery_hub import config

# Documents listed when the answer is degraded to evidence-only. The user still
# gets the retrieval result -- the synthesis is what is withheld, not the evidence.
_MAX_EVIDENCE_DOCS = 10

# How deep the gap agent's target pool goes (see _gap_targets). The answer shows ~10
# candidates because that is what a person reads; a gap search needs the field, not
# the shortlist -- at k=10 the whole B7 family is a handful of documents and "no gap"
# means "nothing in ten documents", which is not a finding about coverage.
# HAND-SET AND UNMEASURED: there is no judged gap set to tune k against, and a bigger
# k trades recall of real gaps against more noise for the ownership/centroid filters
# to reject. Raise it via DH_GAP_TARGET_K before concluding an org has no gaps.
GAP_TARGET_K = config.env_int("DH_GAP_TARGET_K", 100)


@dataclass
class Orchestrator:
    """
    Composes the agents. Construct with whatever you have: every agent except
    retrieval and synthesis is optional, and a missing one is reported in the trace
    as "not wired" rather than silently skipped.

    Nothing here is heavy -- constructing an Orchestrator loads no model, no index
    and no graph. `from_config` is where the real wiring (and the real memory) is.
    """

    retrieval: Any
    synthesis: Any
    verifier: Any | None = None
    expertise_gap: Any | None = None
    policy: Any | None = None

    # -- the five responsibilities, in spec order --------------------------- #
    def run(self, query: str) -> dict:
        trace: list[dict] = []

        # 1/2. ROUTE + FAN OUT. Sequential, on purpose -- see the module docstring.
        r = run_agent(self.retrieval, query, {})
        trace.append(to_trace_row(r))
        candidates = r.payload.get("candidates", []) if r.ok else []

        gaps = self._maybe_gaps(query, candidates, trace)

        # Synthesis needs both. Spec 4.2's sketch passes gaps in; the agent uses
        # them to order what it talks about, never to assert (see synthesis.py).
        s = run_agent(self.synthesis, query,
                      {"candidates": candidates, "gaps": gaps})
        trace.append(to_trace_row(s))
        answer_text = s.payload.get("answer", "") if s.ok else ""

        # Policy routes on the ANSWER as well as the query, so it can only run once
        # the answer exists. Its own should_fire is the router (spec 4.2 step 1).
        policy_result = self._maybe_policy(query, answer_text, r, candidates, trace)

        # The verifier gates the prose against retrieval's evidence. Gap findings
        # are NOT gated here: a claim about absence cannot be supported by any
        # retrieved document (synthesis.py explains), so they travel with their own
        # graph provenance and the reader is told they are unmeasured.
        v = self._maybe_verify(query, answer_text, r, trace)

        # 3. AGGREGATE. Confidences are carried SEPARATELY, per agent, and are
        # never averaged (base.py invariant 3): retrieval's is a reranker score,
        # expertise-gap's is an evidence-availability heuristic with no eval, the
        # verifier's is a supported-claim fraction. Averaging three numbers on three
        # unrelated scales produces a fourth number that means nothing -- the same
        # discipline as never averaging BGE and Qwen scores.
        confidences = {row["agent"]: row["conf"] for row in trace}

        # 4. GATE.
        blocked_by = _blocking_flags(policy_result)
        gate_failed = v is not None and not _verdict_passed(v)
        degraded = bool(blocked_by) or gate_failed
        if degraded:
            answer: Any = self._degrade_to_evidence_only(
                r, _gate_reason(blocked_by, gate_failed, v))
        else:
            answer = {"mode": s.payload.get("mode", "unknown"),
                      "text": answer_text,
                      "claims": s.payload.get("claims", []),
                      "caveat": s.payload.get("caveat")}

        return {
            "query": query,
            "answer": answer,
            "evidence": r.evidence,
            "gaps": gaps.payload if (gaps is not None and not gaps.abstained
                                     and gaps.ok) else None,
            "gaps_abstained_reason": (gaps.payload.get("reason")
                                      if gaps is not None and gaps.abstained else None),
            "flags": (policy_result.payload.get("flags", [])
                      if policy_result is not None else []),
            "trace": trace,          # 5. TRACE -- the demo's primary artifact
            # "verified" is the spec's key. It means EVIDENCE-GATED: PASSED.
            # It does not mean true. See the module docstring.
            "verified": bool(v is not None and _verdict_passed(v)),
            "degraded": degraded,
            "verdict": (v.payload.get("verdict") if v is not None else None),
            "confidences": confidences,   # separate, never combined
            "concurrency": "sequential",  # measured reality, not a diagram
        }

    # -- routing helpers ---------------------------------------------------- #
    def _maybe_gaps(self, query: str, candidates: list[dict],
                    trace: list[dict]) -> AgentResult | None:
        """
        Expertise-gap fires only when the query names an organization (~45% of
        queries, per entity_link's measured fire rate). A router decline is a REAL
        outcome and gets a real trace row -- the demo's most persuasive moment is an
        agent that correctly declines, and an agent that silently never appears
        cannot be pointed at. The latency is the router's own measured cost, not a
        fabricated zero.
        """
        if self.expertise_gap is None:
            trace.append(_not_wired("expertise-gap"))
            return None

        t0 = time.perf_counter()
        try:
            fire = self.expertise_gap.should_fire(query)
        except Exception as exc:  # a broken router must not kill the run
            trace.append(_router_error("expertise-gap", exc,
                                       (time.perf_counter() - t0) * 1000.0))
            return None
        elapsed = (time.perf_counter() - t0) * 1000.0

        if not fire:
            declined = AgentResult.abstain(
                self.expertise_gap.name,
                "router: the query names no organization, so there is no portfolio "
                "to analyse the gaps of", elapsed)
            trace.append(to_trace_row(declined))
            return declined

        g = run_agent(self.expertise_gap, query,
                      {"candidates": candidates,
                       "gap_targets": self._gap_targets(query, candidates)})
        trace.append(to_trace_row(g))
        return g

    def _gap_targets(self, query: str, candidates: list[dict]) -> list[dict] | None:
        """
        A SECOND retrieval, on the query with the org's name stripped, widened to
        GAP_TARGET_K. This is the pool the gap agent scores against.

        WHY A SECOND RETRIEVAL. The first one answers the user's literal question
        ("Bristol Myers Squibb gaps in B7 family immunotherapy") and is what the
        answer must cite. But its candidates are selected partly BY the org name, so
        handing them to the gap agent asks what BMS lacks using documents retrieved
        for "BMS" -- measured on the first full demo run: 7 of 10 already BMS-owned,
        zero gaps found. Absence has to be measured against the field, not against
        the company. Spec 4.3 step 4 says the target is the therapeutic AREA's top-k;
        this is that, literally.

        The cost is one extra retrieve (~8 s warm) and no extra memory: the same
        Retriever instance serves both. Returns None when there is nothing to do --
        no topic left after stripping, or the retrieval agent is unavailable -- and
        the gap agent then falls back to `candidates` and says so in target_pool.
        """
        if self.retrieval is None or not hasattr(self.expertise_gap, "topic_query"):
            return None
        topic = self.expertise_gap.topic_query(query)
        if not topic or topic.strip() == query.strip():
            return None      # nothing was stripped; a second identical call is waste
        # top_k as well as rerank_k: 07's first-stage recall defaults to 50 and would
        # otherwise cap this pool below GAP_TARGET_K without saying so.
        r = run_agent(self.retrieval, topic,
                      {"rerank_k": GAP_TARGET_K, "top_k": GAP_TARGET_K})
        if not r.ok or r.abstained:
            return None
        return r.payload.get("candidates") or None

    def _maybe_policy(self, query: str, answer_text: str, r: AgentResult,
                      candidates: list[dict], trace: list[dict]) -> AgentResult | None:
        """Policy fires only when the exchange makes a clinical/regulatory claim."""
        if self.policy is None:
            trace.append(_not_wired("policy"))
            return None

        ctx = {"answer": answer_text, "evidence": r.evidence, "candidates": candidates}
        p = run_agent(self.policy, query, ctx)
        # A deliberate block is (ok=False, error=None) and must NOT read as a crash.
        row = to_trace_row(p)
        if _blocking_flags(p):
            row["status"] = "blocked"
            row["reason"] = "policy block: " + ", ".join(_blocking_flags(p))
            row["conf"] = round(p.confidence, 4)
        elif row["status"] == "fired" and not p.payload.get("flags"):
            # Policy read the answer and raised nothing. Its confidence is 0.0 by
            # design ("no flags => no verdict to be confident about"), but a literal
            # 0.00 in the trace table reads as "0% confident" -- a measured low
            # score -- when it means the opposite: a clean review. Same reason
            # base.to_trace_row renders an abstention's conf as "--" rather than 0.0.
            row["conf"] = None
            row["reason"] = "reviewed the answer; no policy flags raised"
        trace.append(row)
        return p

    def _maybe_verify(self, query: str, answer_text: str, r: AgentResult,
                      trace: list[dict]) -> AgentResult | None:
        if self.verifier is None:
            trace.append(_not_wired("verifier"))
            return None

        v = run_agent(self.verifier, query,
                      {"answer": answer_text, "evidence": r.evidence})
        # The verifier returns ok=False for a FAIL verdict, which to_trace_row would
        # render "error". Spec line 310's table shows `verifier | pass | 0.92`, so
        # the row is rendered from the verdict; a genuine crash (error is not None)
        # still renders as an error.
        row = to_trace_row(v)
        if v.error is None and "verdict" in v.payload:
            row["status"] = v.payload["verdict"]           # "pass" | "fail"
            row["conf"] = round(v.confidence, 4)
        trace.append(row)
        return v

    # -- the gate ----------------------------------------------------------- #
    def _degrade_to_evidence_only(self, r: AgentResult, reason: str) -> dict:
        """
        Spec 4.2 step 4: retrieved documents, no synthesis. NEVER ship an
        unverified claim.

        The synthesis is dropped ENTIRELY rather than shown with a warning attached.
        A drafted answer next to a caveat is still a drafted answer, and it is the
        sentence the reader remembers -- the caveat is not what gets quoted in the
        meeting. What survives is what retrieval actually returned: real documents,
        with real URLs, that the reader can open. That is a smaller output and an
        honest one.
        """
        docs = [{"doc_id": e["doc_id"], "quote_span": e["quote_span"],
                 "source_url": e.get("source_url", "")}
                for e in r.evidence[:_MAX_EVIDENCE_DOCS]]
        return {
            "mode": "evidence_only",
            "text": "",             # deliberately empty: there is no answer to show
            "claims": [],
            "reason": reason,
            "documents": docs,
        }

    # -- wiring ------------------------------------------------------------- #
    @classmethod
    def from_config(cls, mock: bool = True, k: int | None = None,
                    device: str | None = None) -> Orchestrator:
        """
        Build the real thing from config/env.

        MEMORY: a real (mock=False) run loads faiss + doc_vectors + bm25 + 603k docs
        (~20 GB) lazily on retrieval's first run(), plus the graph index (~1.6 GB
        peak) here. Do not construct two of these.

        The surface index is built ONCE and injected into the gap agent -- it is the
        ~3.3 s / ~0.5 GB cost the gap agent's author flagged, and the constructor
        accepts it precisely so the orchestrator can own it.
        """
        from agents.expertise_gap import ExpertiseGapAgent
        from agents.llm import LLMClient
        from agents.policy import PolicyAgent
        from agents.retrieval import RetrievalAgent
        from agents.synthesis import SynthesisAgent
        from agents.verifier import VerifierAgent

        llm = LLMClient()          # inert without DH_LLM_API_KEY; agents fall back
        gap: Any | None
        try:
            gap = ExpertiseGapAgent.from_config(llm=llm)
        except (FileNotFoundError, OSError) as exc:
            # The graph artifact is a 286 MB build product living outside the repo.
            # Without it the gap agent cannot exist -- but the other four still can,
            # and a demo that reports "expertise-gap: not wired" is more useful than
            # an import error. The trace will say so rather than imply an abstention.
            gap = None
            print(f"[orchestrator] expertise-gap NOT wired: {type(exc).__name__}: {exc}")

        return cls(
            retrieval=RetrievalAgent(mock=mock, device=device, k=k),
            synthesis=SynthesisAgent(llm=llm),
            verifier=VerifierAgent(llm=llm),
            expertise_gap=gap,
            policy=PolicyAgent(llm=llm),
        )


# --------------------------------------------------------------------------- #
# Trace / gate helpers
# --------------------------------------------------------------------------- #
def _not_wired(name: str) -> dict:
    """
    An agent that was never constructed. Distinct from abstained: abstention is a
    judgment the agent made, and claiming one on behalf of an agent that does not
    exist would put a decision in the trace that nothing ever took.
    """
    return {"agent": name, "status": "not wired", "conf": None, "ms": 0.0,
            "reason": "agent not constructed for this run", "error": None}


def _router_error(name: str, exc: Exception, ms: float) -> dict:
    return {"agent": name, "status": "error", "conf": None, "ms": round(ms, 1),
            "reason": None, "error": f"router raised {type(exc).__name__}: {exc}"}


def _blocking_flags(p: AgentResult | None) -> list[str]:
    """
    Policy's blocking flags, read from the PAYLOAD.

    Not from `ok`: the policy agent returns ok=False for a deliberate block, and
    ok=False also means "crashed". `blocking_flags` is non-empty exactly when the
    block was intentional, which is the discriminator its author pinned in
    test_a_deliberate_block_is_distinguishable_from_a_crash.
    """
    if p is None or not p.ok and p.error is not None:
        return []
    return list(p.payload.get("blocking_flags") or [])


def _verdict_passed(v: AgentResult) -> bool:
    """
    Did the evidence gate pass?

    A verifier CRASH (error is not None) is not a pass. It is also not a considered
    fail -- but it must not silently ship the answer either: if the check did not
    run, the claim is unchecked, and an unchecked claim is exactly what the gate
    exists to stop. Fail closed.
    """
    if v.error is not None:
        return False
    if v.abstained:
        # Nothing was synthesized, so there was nothing to gate. Degrading an answer
        # that does not exist is meaningless; there is no claim to withhold.
        return True
    return v.payload.get("verdict") == "pass"


def _gate_reason(blocked_by: list[str], gate_failed: bool,
                 v: AgentResult | None) -> str:
    parts = []
    if blocked_by:
        parts.append("policy blocked this answer (" + ", ".join(blocked_by) + ")")
    if gate_failed and v is not None:
        if v.error is not None:
            parts.append(f"the evidence gate could not run ({v.error}), so the "
                         f"answer is unchecked and is withheld")
        else:
            n_unsupported = v.payload.get("unsupported_count", 0)
            n_contradicted = v.payload.get("contradicted_count", 0)
            parts.append(f"the evidence gate failed: {n_unsupported} claim(s) not "
                         f"supported by any retrieved document, {n_contradicted} "
                         f"contradicted by one")
    return "; ".join(parts) or "gate failed"
