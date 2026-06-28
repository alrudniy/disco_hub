#!/usr/bin/env python3
"""
08_multiagent_rag.py  --  LAYER 3 (Explanation & Workflow).

A strict-RAG, multi-agent pipeline that turns retrieved candidates into a cited,
provenance-stamped, confidence-scored recommendation. Implemented framework-free
as a state dict flowing through agent functions (the same shape maps 1:1 onto
LangGraph nodes for the real build).

  AGENTS
    1. retrieval        -> pull candidates (delegates to 07's Retriever)
    2. expertise_gap    -> annotate which experts/orgs/facilities back each tech;
                           flag candidates lacking provenance
    3. reranking        -> final policy-aware ordering (completeness + score)
    4. policy_safety    -> strict-RAG gates: drop evidence-free candidates, refuse
                           if top confidence < threshold, require a citation per claim
    5. explanation      -> the LLM (mock template or vLLM/Gemini) writes the prose,
                           citing ONLY retrieved sources

  TARGET: Drew (always-on). With a self-hosted LLM you control batching/seeds and
          can approach reproducibility; an API model trades that control for power.

  HUMAN-IN-THE-LOOP: output is a shortlist + explanation. People decide.

Usage:
  python 08_multiagent_rag.py --mock --query "kinase inhibitor for oncology"
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from discovery_hub import config
from discovery_hub.determinism import set_global_determinism


def _load_retriever_module():
    """Import 07_retrieve_rank.py by path (module name starts with a digit)."""
    path = Path(__file__).with_name("07_retrieve_rank.py")
    spec = importlib.util.spec_from_file_location("retrieve_rank", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _confidence(score: float) -> float:
    """
    Confidence = strength of the top semantic match (dense cosine in [0,1]),
    clamped. Previously a steep sigmoid over the *blended* retrieval score; after
    Fix #3 made hybrid the default, that score is the RRF fused rank score (~0.02),
    which is not a calibrated relevance and drove confidence to ~0.05 -- so the
    policy gate refused every query. We read confidence off the dense cosine
    instead (carried on each candidate as ``text_score``). Real deployments should
    calibrate this against labeled relevance (e.g. Platt scaling on the eval set).
    """
    return round(max(0.0, min(1.0, score)), 4)


# --------------------------------------------------------------------------- #
# Agents: each takes and returns the shared state dict.
# --------------------------------------------------------------------------- #
def agent_retrieval(state: dict) -> dict:
    cands = state["_retriever"].retrieve(state["query"], rerank_k=state["k"])
    state["candidates"] = cands
    return state


def agent_expertise_gap(state: dict) -> dict:
    for c in state["candidates"]:
        backers = list(c.get("organizations", []))
        c["expertise"] = backers
        c["expertise_gap"] = (len(backers) == 0)  # no org/expert linked => gap
    state["num_gaps"] = sum(c["expertise_gap"] for c in state["candidates"])
    return state


def agent_reranking(state: dict) -> dict:
    # Policy-aware final order: prefer complete provenance, then score.
    state["candidates"].sort(
        key=lambda c: (not c["expertise_gap"], c["score"]), reverse=True)
    return state


def agent_policy_safety(state: dict) -> dict:
    # Strict RAG: keep only candidates that actually carry evidence text.
    kept = [c for c in state["candidates"]
            if (c.get("abstract") or "").strip() and c.get("source_url")]
    state["candidates"] = kept
    # Confidence reads off the dense semantic similarity, not the RRF fused rank
    # score (see _confidence). Use the strongest available evidence match.
    top = max((c.get("text_score", 0.0) for c in kept), default=0.0)
    state["overall_confidence"] = _confidence(top)
    # Refuse to assert a recommendation we cannot ground / are not confident in.
    if not kept:
        state["refusal"] = "No evidence-backed candidates found; nothing to recommend."
    elif state["overall_confidence"] < config.RETRIEVAL.min_confidence:
        state["refusal"] = (f"Top confidence {state['overall_confidence']:.2f} below "
                            f"threshold {config.RETRIEVAL.min_confidence}; flagged for "
                            f"human review rather than asserted.")
    else:
        state["refusal"] = None
    return state


def _explain_mock(query: str, c: dict) -> str:
    """Deterministic stand-in for the LLM. Cites the real source; no free text
    beyond the retrieved evidence (strict RAG)."""
    snippet = (c.get("abstract") or "")[:160].rstrip()
    org = (c.get("organizations") or ["an unnamed organization"])[0]
    return (f"Relevant to '{query}': {c.get('title','(untitled)')} from {org}. "
            f"Supporting evidence: \"{snippet}...\" [source: {c['source_url']}]")


def agent_explanation(state: dict, explain_fn) -> dict:
    recs = []
    for c in state["candidates"]:
        recs.append({
            "doc_id": c["doc_id"],
            "title": c.get("title", ""),
            "why": explain_fn(state["query"], c),
            "citation": c["source_url"],          # every claim carries a citation
            "confidence": _confidence(c.get("text_score", 0.0)),
            "provenance": {"source": c.get("source"),
                           "retrieved_via": "DiscoveryHub/retrieve_rank"},
            "expertise_gap": c["expertise_gap"],
        })
    state["recommendations"] = recs
    return state


def run_pipeline(query: str, mock: bool = True, k: int | None = None,
                 retriever=None, explain_fn=None) -> dict:
    """Execute the agent graph and return the structured, cited result."""
    k = k or config.RETRIEVAL.top_k_rerank
    if retriever is None:
        rr = _load_retriever_module()
        retriever = rr.Retriever(mock=mock)
    explain_fn = explain_fn or (_explain_mock if mock else _explain_real_factory())

    state = {"query": query, "k": k, "_retriever": retriever}
    for agent in (agent_retrieval, agent_expertise_gap,
                  agent_reranking, agent_policy_safety):
        state = agent(state)
    if state["refusal"] is None:
        state = agent_explanation(state, explain_fn)
    else:
        state["recommendations"] = []

    state.pop("_retriever", None)
    state.pop("candidates", None)
    return {
        "query": query,
        "overall_confidence": state["overall_confidence"],
        "refusal": state["refusal"],
        "num_expertise_gaps": state.get("num_gaps", 0),
        "recommendations": state["recommendations"],
        "human_in_the_loop": "Shortlist only. A person reviews and decides.",
    }


def _explain_real_factory():
    """Returns an explain_fn backed by a self-hosted LLM (vLLM) in strict RAG."""
    from openai import OpenAI  # vLLM exposes an OpenAI-compatible server
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

    def explain(query: str, c: dict) -> str:
        prompt = (
            "You are a strict-RAG assistant. Using ONLY the evidence below, write "
            "one sentence on why this technology matches the interest, then cite the "
            f"source URL in brackets. Do not add facts.\n\nInterest: {query}\n"
            f"Title: {c.get('title')}\nEvidence: {c.get('abstract')}\n"
            f"Source: {c['source_url']}")
        resp = client.chat.completions.create(
            model=config.LLM_MODEL, temperature=0.0, seed=config.SEED,
            messages=[{"role": "user", "content": prompt}])
        return resp.choices[0].message.content.strip()
    return explain


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--query", required=True)
    ap.add_argument("--k", type=int, default=config.RETRIEVAL.top_k_rerank)
    args = ap.parse_args()

    set_global_determinism(config.SEED)
    result = run_pipeline(args.query, mock=args.mock, k=args.k)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
