"""
The synthesis agent: turn retrieved candidates into a cited answer, strict-RAG.

This is `08_multiagent_rag.py`'s answer step ("08, largely as-is", per spec 4.2),
re-homed behind the Agent protocol and re-pointed at agents/llm.py (z.ai/glm-4.6).
08's own LLM client talks to the 1min.ai shape (`API-KEY` header,
`/api/chat-with-ai`); z.ai is OpenAI-compatible. Copying 08's client would send the
right prompt to the wrong protocol, so only the DISCIPLINE is reused, not the wire:

  * cite ONLY retrieved sources -- never the model's own knowledge
  * every claim carries a citation to a doc_id that was actually retrieved
  * evidence-free candidates are dropped before anything is written (08's
    agent_policy_safety) -- a candidate with no abstract grounds nothing
  * refusing is a legitimate outcome, not an error

TWO PATHS, and the payload ALWAYS says which one ran:
  mode="llm"           -- glm-4.6 wrote the prose from the evidence.
  mode="deterministic" -- no DH_LLM_API_KEY (or the call failed). The answer is
                          then QUOTATION, NOT SYNTHESIS: verbatim evidence spans,
                          each tagged with the doc_id it came from. Nothing is
                          paraphrased, because paraphrasing without a model would
                          mean inventing text and attributing it to a document.
                          The demo prints this distinction; a template must never
                          read as model output.

--------------------------------------------------------------------------- #
WHY THERE ARE NO INLINE "[source: ...]" CITATIONS IN THE PROSE
--------------------------------------------------------------------------- #
08's `_explain_mock` writes the citation INTO the sentence:

    ... Supporting evidence: "<abstract>..." [source: https://patents.google.com/...]

That is unsafe here, and it is not a style preference -- it is MEASURED against
the verifier that now sits downstream. `agents/verifier.py`'s deterministic path
flags any "risky token" (anything containing a digit, plus acronyms) that does not
appear in the evidence spans, and it does NOT strip URLs before tokenizing. So a
citation renders as the token `patents.google.com/patent/US6803192B1`, which by
construction never appears inside an abstract -> every cited sentence scores
`unsupported` -> verdict=fail -> the orchestrator degrades to evidence-only.
Measured on one real B7-H1 abstract:

    08-style sentence with [source: <url>]  -> risky tokens
                                               {B7-H1, PD-L1, patents.google.com/patent/US6803192B1}
                                            -> BOTH sentences unsupported -> fail
    same claim, citation carried structurally -> risky tokens {B7-H1}
                                               (B7-H1 IS in the abstract) -> supported

Copying 08 verbatim would therefore degrade every query in the demo, including the
two that are supposed to succeed. (Note `discovery_hub.faithfulness.lexical_support`
DOES strip URLs -- the two checks disagree, which is why this only shows up when
they are composed.) So citations travel STRUCTURALLY: one doc_id per claim in
`payload["claims"]`, rendered next to the prose by the caller. Every claim still
carries a citation -- the spec's requirement is met, and it is now machine-checkable
rather than a substring in prose. The prose itself carries only assertions, which
is exactly what the verifier should be grading.

THE HARD RULE THIS IMPLIES: `payload["answer"]` contains ONLY sentences that are
grounded in the retrieved evidence. Every piece of meta-commentary -- mode,
caveats, counts, "no LLM ran" -- lives in sibling payload fields, NEVER in the
answer text. A single ungrounded aside like "Retrieval returned 5 documents"
introduces the token "5", which is absent from the evidence, and fails the whole
answer (measured: that exact sentence -> unsupported). Meta-commentary in gated
prose is a self-inflicted gate failure.

--------------------------------------------------------------------------- #
WHAT `gaps` IS USED FOR, AND WHAT IT IS DELIBERATELY NOT USED FOR
--------------------------------------------------------------------------- #
Spec 4.2's sketch passes the expertise-gap result into synthesis, so this agent
accepts it. It uses it to FOREGROUND the candidates the gap agent flagged -- a real
use of the payload that changes what the answer talks about first.

It does NOT merge the gap narration into the answer prose, on purpose. A gap claim
is a claim about ABSENCE ("BMS has no coverage of B7-H3"), derived by graph
traversal. No retrieved abstract can support it -- the document that says it does
not exist. The verifier gates the prose against retrieval's evidence only, so
folding gap narration in would make the money-shot query fail verification and
degrade on every single run. The spec's own architecture diagram agrees: its output
is "answer + evidence + gaps + flags + verdict" -- `gaps` is a SIBLING of `answer`,
not a part of it. The orchestrator carries the gap narration in its own channel,
with its own graph provenance and its own (unmeasured) confidence.
"""
from __future__ import annotations

from typing import Any

from agents.base import AgentResult, make_evidence, timed
from agents.llm import LLMClient
# Imported rather than re-implemented so the [0,1] squash cannot drift between the
# retrieval agent, 08 and here. It is private to that module by name only; the
# alternative is a fourth copy of the same six lines.
from agents.retrieval import _confidence
from discovery_hub import config

# How many evidence-bearing candidates the answer is allowed to rest on. Beyond
# this the prose stops being an answer and becomes a list. Presentation only.
_MAX_CLAIMS = 5

# Evidence handed to the LLM, per candidate. Long enough to carry a real abstract,
# bounded so the prompt stays inside a sane budget.
_EVIDENCE_CHARS = 1200

# --------------------------------------------------------------------------- #
# THE SYNTHESIS FRAMING.
#
# Deliberately NOT the verifier's framing, and they must never be converged --
# that divergence is the entire point of cross-agent verification. This agent is
# constructive: it is a scouting analyst reporting what the documents say. The
# verifier is a falsification engine that assumes this agent is lying. If both
# shared a framing they would share their blind spots, and the second pass would
# be agreement theatre rather than a check.
#
# The rules below are the strict-RAG contract from 08, tightened: no inline
# citations (see the module docstring -- they break the downstream gate), and no
# world knowledge (the model knows B7-H1 is PD-L1; the corpus may not say so, and
# an answer that asserts it is asserting something the evidence cannot back).
# --------------------------------------------------------------------------- #
_SYSTEM_PROMPT = (
    "You are a technology-scouting analyst. You report ONLY what the supplied "
    "EVIDENCE passages state, for a professional who will check every word against "
    "the sources.\n\n"
    "Write the answer as a list of atomic claims. Each claim is one self-contained "
    "sentence, and each names the doc_id of the ONE passage that states it.\n\n"
    "RULES YOU MUST NOT BREAK:\n"
    "1. Every claim must be stated by one of the EVIDENCE passages. If the evidence "
    "does not answer the question, say so in a claim and cite the closest passage, "
    "or return fewer claims. Returning nothing is better than inventing anything.\n"
    "2. Your own knowledge is INADMISSIBLE, even when you are certain. If you know "
    "two terms are synonyms and the evidence does not say so, you may not say so.\n"
    "3. Do NOT put citations, doc_ids, URLs or bracketed references inside the claim "
    "text. The doc_id belongs in its own field. A claim's text must read as plain "
    "prose.\n"
    "4. Numbers, doses, percentages, dates and identifiers must appear in the cited "
    "passage exactly as you write them. Never round, never approximate.\n"
    "5. Only cite a doc_id that appears in the EVIDENCE block."
)

_SCHEMA_HINT = '{"claims": [{"text": str, "doc_id": str}]}'


class SynthesisAgent:
    """
    Writes the cited answer from retrieval's candidates.

    ctx = {"candidates": [...],            # from the retrieval agent
           "gaps": AgentResult | dict | None}   # from the expertise-gap agent

    Abstains -- never errors -- when there is nothing it can honestly answer from.
    """

    name = "synthesis"

    def __init__(self, llm: LLMClient | None = None, max_claims: int = _MAX_CLAIMS,
                 min_confidence: float | None = None) -> None:
        # Injectable so tests script .chat_json and never touch a socket. Default
        # construction is inert without a key (LLMClient does no I/O in __init__).
        self.llm = llm if llm is not None else LLMClient()
        self.max_claims = max_claims
        self.min_confidence = (config.RETRIEVAL.min_confidence
                               if min_confidence is None else min_confidence)

    @timed
    def run(self, query: str, ctx: dict) -> AgentResult:
        candidates = list(ctx.get("candidates") or [])
        if not candidates:
            return AgentResult.abstain(
                self.name, "retrieval supplied no candidates, so there is no evidence "
                           "to write an answer from")

        # 08's agent_policy_safety, kept: strict RAG means a candidate that carries
        # no text grounds no claim, whatever its rank.
        kept = [c for c in candidates
                if (c.get("abstract") or "").strip() and c.get("source_url")]
        if not kept:
            return AgentResult.abstain(
                self.name,
                f"all {len(candidates)} candidates lack an abstract or a source URL; "
                f"nothing here can ground a claim")

        # The confidence gate is 08's, and it is applied ONLY when it is meaningful.
        # In mock mode no cross-encoder runs, so the number is a rank-fusion artifact
        # (~1/60-ish) rather than a relevance score -- comparing it to the 0.35
        # threshold would refuse every mock query for a reason that is an artifact of
        # the arithmetic, not of the evidence. See agents/retrieval.py on the basis.
        reranked = "rerank_score" in kept[0]
        if reranked:
            confidence = _confidence(max(c.get("rerank_score", 0.0) for c in kept))
            gate_note = f"rerank_score >= {self.min_confidence} (08's gate)"
            if confidence < self.min_confidence:
                return AgentResult.abstain(
                    self.name,
                    f"top reranker confidence {confidence:.2f} is below the "
                    f"{self.min_confidence} threshold; flagged for human review "
                    f"rather than asserted")
        else:
            confidence = _confidence(max(c.get("score", 0.0) for c in kept))
            gate_note = ("NOT APPLIED: no reranker ran (mock mode), so the confidence "
                         "is a fused-RRF rank artifact and is not comparable to the "
                         "threshold")

        # Spec 4.2 passes gaps in. Used to reorder, never to assert -- module docstring.
        gap_doc_ids = _gap_doc_ids(ctx.get("gaps"))
        ordered = _foreground_gaps(kept, gap_doc_ids)[:self.max_claims]

        claims, mode, notes = self._write(query, ordered)
        if not claims:
            return AgentResult.abstain(
                self.name, "no claim could be grounded in the retrieved evidence")

        answer = " ".join(c["text"] for c in claims)
        by_id = {c["doc_id"]: c for c in ordered}
        payload = {
            # ONLY grounded sentences. Meta-commentary lives in the fields below --
            # putting it here fails the verifier's gate. See module docstring.
            "answer": answer,
            "claims": claims,                      # every claim carries its doc_id
            "mode": mode,
            "cited_doc_ids": sorted({c["doc_id"] for c in claims}),
            "candidates_considered": len(candidates),
            "candidates_with_evidence": len(kept),
            "confidence_basis": "rerank_score" if reranked else "fused_rrf_score (mock)",
            "confidence_gate": gate_note,
            "gap_targets_foregrounded": sorted(
                d for d in gap_doc_ids if d in {c["doc_id"] for c in ordered}),
        }
        payload.update(notes)

        evidence = [
            make_evidence(c["doc_id"],
                          (by_id[c["doc_id"]].get("abstract") or "").strip(),
                          by_id[c["doc_id"]].get("source_url", ""))
            for c in claims if c["doc_id"] in by_id
        ]
        return AgentResult(agent=self.name, ok=True, payload=payload,
                           evidence=evidence, confidence=confidence, latency_ms=0.0)

    # ----------------------------------------------------------------- #
    def _write(self, query: str, cands: list[dict]) -> tuple[list[dict], str, dict]:
        """Try the LLM; fall back to quotation, saying so, when it cannot run."""
        if self.llm is not None and getattr(self.llm, "available", False):
            raw = self._call_llm(query, cands)
            claims = _validate_claims(raw, cands) if raw is not None else []
            if claims:
                return claims, "llm", {}
            # No key, transport failure, unparseable JSON, or -- per
            # agents/LLM_CONTRACT.md -- a 200 with content:"" because reasoning
            # tokens ate max_tokens. All mean "the model did not write this".
            claims, notes = _quote_claims(cands)
            notes["fallback_reason"] = "llm_returned_no_usable_claims"
            notes["caveat"] = ("LLM was configured but returned nothing usable; "
                               + notes["caveat"])
            return claims, "deterministic", notes

        claims, notes = _quote_claims(cands)
        notes["fallback_reason"] = "no_llm_api_key"
        return claims, "deterministic", notes

    def _call_llm(self, query: str, cands: list[dict]) -> dict | None:
        block = "\n\n".join(
            f"doc_id: {c['doc_id']}\ntitle: {c.get('title', '')}\n"
            f"{(c.get('abstract') or '')[:_EVIDENCE_CHARS]}"
            for c in cands
        )
        user = f"QUESTION:\n{query}\n\nEVIDENCE:\n{block}"
        # max_tokens is generous deliberately: glm-4.6 bills reasoning tokens
        # against the budget and returns HTTP 200 with content:"" if it runs out
        # mid-think (agents/LLM_CONTRACT.md). The contract's real fix is
        # thinking:{"type":"disabled"}, which the shared client does not expose;
        # headroom is the mitigation available from inside this module.
        return self.llm.chat_json(
            [{"role": "system", "content": _SYSTEM_PROMPT},
             {"role": "user", "content": user}],
            schema_hint=_SCHEMA_HINT, temperature=0.0, max_tokens=1600)


# --------------------------------------------------------------------------- #
# LLM post-validation -- trust the prose, verify every citation
# --------------------------------------------------------------------------- #
def _validate_claims(raw: dict, cands: list[dict]) -> list[dict]:
    """
    Keep only claims that cite a doc_id we actually retrieved.

    A claim citing anything else is resting on a document the model invented or
    remembered, which is precisely the failure strict RAG exists to prevent. We
    DROP it rather than downgrade it: unlike the verifier -- whose job is to report
    on bad claims -- this agent's output IS the answer, and an answer is not the
    place to carry a claim we already know is ungrounded.
    """
    known = {c["doc_id"] for c in cands}
    out: list[dict] = []
    for item in raw.get("claims") or []:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        doc_id = item.get("doc_id")
        doc_id = doc_id.strip() if isinstance(doc_id, str) else ""
        if text and doc_id in known:
            out.append({"text": text, "doc_id": doc_id})
    return out


# --------------------------------------------------------------------------- #
# Deterministic path -- quotation, not synthesis
# --------------------------------------------------------------------------- #
def _quote_claims(cands: list[dict]) -> tuple[list[dict], dict]:
    """
    With no model, the honest answer is the evidence itself: each claim is a
    VERBATIM span of one candidate's abstract, tagged with that candidate's doc_id.

    This is 08's `_explain_mock` in spirit -- a deterministic, cited stand-in --
    with its inline `[source: ...]` removed, because that citation is what breaks
    the downstream gate (module docstring). Nothing is paraphrased and nothing is
    added: without a model, generating prose ABOUT a document and attributing it to
    that document would be inventing text, which is the one thing a template must
    never do.
    """
    claims = []
    for c in cands:
        span = _quotable(c.get("abstract") or "")
        if span:
            claims.append({"text": span, "doc_id": c["doc_id"]})
    notes = {
        "caveat": ("DETERMINISTIC MODE: no LLM ran. The answer is VERBATIM QUOTATION "
                   "of the retrieved evidence, not synthesis -- no text was written, "
                   "summarized or paraphrased by a model. Claims are ordered by "
                   "retrieval rank, which is not an argument."),
        "synthesized": False,
    }
    return claims, notes


def _quotable(abstract: str, limit: int = 240) -> str:
    """
    A verbatim, sentence-terminated span of the abstract.

    The terminator matters and is not cosmetic: the verifier splits the answer on
    sentence boundaries, so two quotes joined by a space where the first does not
    end in punctuation fuse into ONE sentence spanning two documents. That fused
    sentence then matches neither document's span well enough to clear the support
    threshold, and a perfectly good answer fails its own gate. Appending "." when
    the cut lands mid-sentence changes no token, so it cannot affect either the
    overlap score or the risky-token check -- it only keeps the boundary findable.
    """
    text = (abstract or "").strip()
    if not text:
        return ""
    if len(text) > limit:
        cut = text[:limit]
        boundary = cut.rfind(" ")
        text = (cut[:boundary] if boundary > 0 else cut).rstrip()
    return text if text.endswith((".", "!", "?")) else text + "."


# --------------------------------------------------------------------------- #
# gaps -> ordering
# --------------------------------------------------------------------------- #
def _gap_doc_ids(gaps: Any) -> set[str]:
    """doc_ids the expertise-gap agent flagged. Accepts an AgentResult or its payload."""
    payload = getattr(gaps, "payload", gaps)
    if not isinstance(payload, dict):
        return set()
    return {g["doc_id"] for g in payload.get("gaps") or []
            if isinstance(g, dict) and g.get("doc_id")}


def _foreground_gaps(cands: list[dict], gap_doc_ids: set[str]) -> list[dict]:
    """
    Stable-sort gap targets to the front, preserving retrieval's order within each
    group. If the gap agent says the organization has no coverage of a document,
    that document is the interesting one and the answer should reach it before the
    budget runs out. The sort is STABLE, so within each group retrieval's own rank
    order survives untouched -- this reorders across the gap/non-gap boundary and
    nowhere else, and it inherits retrieval's determinism rather than imposing a
    second ordering of its own.
    """
    if not gap_doc_ids:
        return list(cands)
    return sorted(cands, key=lambda c: (c["doc_id"] not in gap_doc_ids,))
