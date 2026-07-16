"""
The agent contract: one result type, one protocol, one trace row.

Every agent in the orchestrator implements `Agent` and returns an `AgentResult`.
Nothing else crosses an agent boundary. This module is deliberately tiny and has
no dependencies beyond the stdlib -- it is imported by every agent, so anything
heavy here is paid for on every import path.

THREE INVARIANTS, in order of how easy they are to break:

  1. ABSTENTION IS SUCCESS. `abstained=True` travels with `ok=True`. An agent
     that has no signal for a query (expertise-gap on a query naming no org --
     roughly 55% of queries, per entity_link's measured fire rate) must abstain,
     not guess. `ok=False` is reserved for an agent that actually broke.

  2. ONE AGENT FAILING NEVER KILLS THE ORCHESTRATOR. `run_agent()` converts an
     unexpected exception into ok=False with the exception text, so the fan-out
     degrades to the agents that did work instead of raising.

  3. CONFIDENCES ARE NEVER AVERAGED ACROSS AGENTS. Each is calibrated (or not)
     on its own scale; they are carried separately all the way to the trace. This
     type does not offer a way to combine them, on purpose.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable


@dataclass
class AgentResult:
    agent: str
    ok: bool
    payload: dict
    evidence: list[dict]        # [{doc_id, quote_span, source_url}]
    confidence: float           # 0-1, calibrated per agent -- NEVER averaged across agents
    latency_ms: float
    abstained: bool = False     # abstention is a FIRST-CLASS outcome, not a failure
    error: str | None = None

    @classmethod
    def abstain(cls, agent: str, reason: str, latency_ms: float = 0.0) -> AgentResult:
        """
        The abstain constructor. ok=True because declining to answer is a correct
        outcome, not an error -- the caller must not treat this as a failure. The
        reason is carried in the payload so the trace can say *why* it declined,
        which is the part that reads as trustworthy in a demo.
        """
        return cls(agent=agent, ok=True, payload={"reason": reason}, evidence=[],
                   confidence=0.0, latency_ms=latency_ms, abstained=True)

    @classmethod
    def failure(cls, agent: str, error: str, latency_ms: float = 0.0) -> AgentResult:
        """An agent that actually broke. Distinct from abstention, on purpose."""
        return cls(agent=agent, ok=False, payload={}, evidence=[], confidence=0.0,
                   latency_ms=latency_ms, error=error)


@runtime_checkable
class Agent(Protocol):
    name: str

    def run(self, query: str, ctx: dict) -> AgentResult: ...


def make_evidence(doc_id: str, quote_span: str, source_url: str = "") -> dict:
    """
    Build one evidence entry. Every claim an agent makes should be traceable to
    one of these; an agent returning evidence=[] is telling the verifier it has
    nothing to check against, which is itself informative.
    """
    return {"doc_id": doc_id, "quote_span": quote_span, "source_url": source_url}


def run_agent(agent: Agent, query: str, ctx: dict | None = None) -> AgentResult:
    """
    Invoke an agent with timing and failure containment. This is the ONLY way the
    orchestrator should call an agent.

    - Fills latency_ms from a monotonic clock (the agent need not time itself; any
      value it set is overwritten with the measured wall time, which includes the
      agent's own overhead).
    - Converts an unexpected exception into AgentResult(ok=False, error=...) so a
      single broken agent degrades the fan-out instead of killing the run.
    """
    t0 = time.perf_counter()
    try:
        result = agent.run(query, ctx or {})
    except Exception as exc:  # noqa: BLE001 -- containment is the whole point
        elapsed = (time.perf_counter() - t0) * 1000.0
        return AgentResult.failure(getattr(agent, "name", agent.__class__.__name__),
                                   f"{type(exc).__name__}: {exc}", elapsed)
    result.latency_ms = (time.perf_counter() - t0) * 1000.0
    return result


def timed(fn: Callable[..., AgentResult]) -> Callable[..., AgentResult]:
    """
    Decorator form of the timing half of `run_agent`, for an agent's own `run`.
    Failure containment stays in `run_agent` -- an agent should not swallow its
    own exceptions, because then the orchestrator cannot tell broken from empty.
    """
    def wrapper(*args: Any, **kwargs: Any) -> AgentResult:
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        result.latency_ms = (time.perf_counter() - t0) * 1000.0
        return result
    wrapper.__name__ = getattr(fn, "__name__", "wrapper")
    wrapper.__doc__ = fn.__doc__
    return wrapper


def to_trace_row(result: AgentResult) -> dict:
    """
    One row of the demo's trace table: agent / status / conf / ms. The trace is
    the most persuasive artifact the architecture has -- it is the only place the
    orchestration is visible, including the agents that correctly decline. Status
    distinguishes all three outcomes; confidence is None (rendered "--") when the
    agent abstained or failed, because 0.0 would read as a measured low score.
    """
    if not result.ok:
        status = "error"
    elif result.abstained:
        status = "abstained"
    else:
        status = "fired"
    return {
        "agent": result.agent,
        "status": status,
        "conf": None if status != "fired" else round(result.confidence, 4),
        "ms": round(result.latency_ms, 1),
        "reason": result.payload.get("reason") if result.abstained else None,
        "error": result.error,
    }
