"""
Known-answer tests for the agent contract and the optional LLM client.

The contract's three invariants are what these lock down: abstention is success,
one agent failing does not kill the orchestrator, and the trace distinguishes
fired / abstained / error. Plus the property that matters for CI: with no
DH_LLM_API_KEY set, the LLM client degrades to None instead of raising or
reaching the network. Nothing here touches a real artifact or a socket.

Run: python -m pytest tests/ -q
"""
import pytest

from agents.base import (AgentResult, make_evidence, run_agent, timed,
                         to_trace_row)
from agents.llm import LLMClient, _parse_json_loose


# --------------------------------------------------------------------------- #
# Test doubles
# --------------------------------------------------------------------------- #
class _OkAgent:
    name = "ok"

    def run(self, query, ctx):
        return AgentResult(agent=self.name, ok=True, payload={"q": query},
                           evidence=[], confidence=0.81, latency_ms=0.0)


class _AbstainAgent:
    name = "abstainer"

    def run(self, query, ctx):
        return AgentResult.abstain(self.name, "no org named in query")


class _BoomAgent:
    name = "boom"

    def run(self, query, ctx):
        raise RuntimeError("kaboom")


# --------------------------------------------------------------------------- #
# AgentResult
# --------------------------------------------------------------------------- #
def test_abstain_is_success_not_failure():
    # The central invariant: an agent with no signal reports ok=True.
    r = AgentResult.abstain("expertise_gap", "query names no organization")
    assert r.ok is True and r.abstained is True
    assert r.confidence == 0.0
    assert r.payload["reason"] == "query names no organization"
    assert r.error is None


def test_failure_is_distinct_from_abstention():
    r = AgentResult.failure("retrieval", "ValueError: bad index")
    assert r.ok is False and r.abstained is False
    assert "ValueError" in r.error


def test_make_evidence_shape():
    e = make_evidence("uspto:US6803192B1", "B7-H1, a novel immunoregulatory molecule",
                      "https://patents.google.com/patent/US6803192B1")
    assert set(e) == {"doc_id", "quote_span", "source_url"}
    assert e["doc_id"] == "uspto:US6803192B1"


# --------------------------------------------------------------------------- #
# run_agent: timing + failure containment
# --------------------------------------------------------------------------- #
def test_run_agent_fills_latency():
    r = run_agent(_OkAgent(), "pd-l1 antibody", {})
    assert r.ok and r.latency_ms >= 0.0
    assert r.payload["q"] == "pd-l1 antibody"


def test_run_agent_contains_exception():
    # One broken agent must degrade to ok=False, never propagate.
    r = run_agent(_BoomAgent(), "q", {})
    assert r.ok is False
    assert r.abstained is False
    assert "RuntimeError: kaboom" in r.error
    assert r.agent == "boom"
    assert r.latency_ms >= 0.0


def test_run_agent_preserves_abstention():
    r = run_agent(_AbstainAgent(), "q", {})
    assert r.ok is True and r.abstained is True


def test_timed_decorator_overwrites_latency():
    @timed
    def run():
        return AgentResult(agent="x", ok=True, payload={}, evidence=[],
                           confidence=1.0, latency_ms=-999.0)

    assert run().latency_ms >= 0.0


# --------------------------------------------------------------------------- #
# Trace rows
# --------------------------------------------------------------------------- #
def test_trace_row_distinguishes_three_outcomes():
    fired = to_trace_row(run_agent(_OkAgent(), "q", {}))
    abst = to_trace_row(run_agent(_AbstainAgent(), "q", {}))
    err = to_trace_row(run_agent(_BoomAgent(), "q", {}))

    assert fired["status"] == "fired" and fired["conf"] == 0.81
    assert abst["status"] == "abstained"
    assert err["status"] == "error"


def test_trace_row_hides_confidence_when_not_fired():
    # 0.0 would read as a measured low score; None renders as "--".
    row = to_trace_row(AgentResult.abstain("expertise_gap", "no org"))
    assert row["conf"] is None
    assert row["reason"] == "no org"


# --------------------------------------------------------------------------- #
# LLM client: must be inert with no key (this is the CI property)
# --------------------------------------------------------------------------- #
def test_llm_unavailable_without_key(monkeypatch):
    monkeypatch.delenv("DH_LLM_API_KEY", raising=False)
    c = LLMClient()
    assert c.available is False
    # Returns None rather than raising or hitting the network.
    assert c.chat([{"role": "user", "content": "hi"}]) is None
    assert c.chat_json([{"role": "user", "content": "hi"}], schema_hint="{}") is None


def test_llm_defaults_and_no_key_leak(monkeypatch):
    monkeypatch.delenv("DH_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("DH_LLM_MODEL", raising=False)
    monkeypatch.setenv("DH_LLM_API_KEY", "super-secret-value")
    c = LLMClient()
    assert c.available is True
    assert c.base_url == "https://api.z.ai/api/paas/v4"
    assert c.model == "glm-4.6"
    assert "super-secret-value" not in repr(c)   # secrets never reach a log line


@pytest.mark.parametrize("text, expected", [
    ('{"verdict": "pass"}', {"verdict": "pass"}),
    ('```json\n{"verdict": "fail"}\n```', {"verdict": "fail"}),
    ('Sure! Here it is:\n{"a": 1}\nHope that helps.', {"a": 1}),
    ("not json at all", None),
    ('["a", "b"]', None),      # valid JSON, wrong shape -> None, never coerced
])
def test_parse_json_loose(text, expected):
    assert _parse_json_loose(text) == expected
