"""
Faithfulness / citation verification for RAG answers (fix #6).

The mock explainer cites by construction, so stage 09's citation-set Jaccard says
nothing about whether a REAL LLM cites correctly or invents facts. This adds the
missing verifier. Two checks, both runnable in CI without a GPU, each with a
real-model upgrade path:

  1. CITATION VALIDITY  -- every source the prose cites must be the evidence the
     claim was actually given (catches an LLM citing a different or fabricated
     URL than the passage it was shown). Deterministic, exact.
  2. CLAIM GROUNDEDNESS -- the claim's substantive content must be supported by
     its cited evidence text. The DEFAULT scorer is a lexical-overlap proxy
     (deterministic, cheap, catches egregious hallucination). The REAL, accurate
     scorer is a pluggable NLI / LLM-judge entailment model (RAGAS / Vectara
     HHEM style): pass `support_fn=` in real mode.

PROXY HONESTY: lexical overlap cannot catch a fluent semantic hallucination that
reuses the source's vocabulary, nor credit a faithful paraphrase that doesn't
share words. It is a floor, not the real metric -- a cheap regression guard and a
detector of blatant ungrounded text. The accurate metric needs an entailment
model; this module's interface makes that a one-argument swap.
"""
from __future__ import annotations

import re

_URL = re.compile(r"https?://[^\s\]\)\"]+")
_TOK = re.compile(r"[a-z0-9]+")

# Function words plus explanation boilerplate that carry no factual content and
# would otherwise dilute the groundedness signal.
_STOP = {
    "the", "a", "an", "and", "or", "of", "for", "to", "in", "on", "with", "is",
    "are", "this", "that", "as", "by", "at", "from", "it", "its", "be", "can",
    "relevant", "supporting", "evidence", "source", "interest", "given", "using",
    "why", "matches", "match", "technology", "one", "sentence", "do", "not", "add",
}


def extract_citations(text: str) -> list[str]:
    """URLs the prose actually cites (in citation order, de-duplicated)."""
    seen, out = set(), []
    for u in _URL.findall(text or ""):
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _content_tokens(text: str) -> set[str]:
    """Substantive tokens: alphanumerics >= 3 chars, minus URLs and stopwords."""
    stripped = _URL.sub(" ", (text or "").lower())
    return {t for t in _TOK.findall(stripped) if len(t) >= 3 and t not in _STOP}


def lexical_support(claim: str, evidence_text: str, context: str = "") -> float:
    """
    Proxy groundedness in [0,1]: the fraction of the claim's content tokens that
    appear in the cited evidence (the question text is also allowed, so restating
    the query is not counted as hallucination). 1.0 if the claim has no
    substantive tokens. This is the DEFAULT support_fn; swap an NLI/LLM judge in
    for real scoring.
    """
    claim_c = _content_tokens(claim)
    if not claim_c:
        return 1.0
    grounded = _content_tokens(evidence_text) | _content_tokens(context)
    return len(claim_c & grounded) / len(claim_c)


def evaluate_recommendations(recs, text_by_doc_id, *, support_fn=None,
                             threshold: float = 0.5, context: str = "") -> dict:
    """
    Score a list of stage-08 recommendations for citation validity and
    groundedness against the evidence each one was built from.

    recs            : [{doc_id, why, citation}, ...]
    text_by_doc_id  : doc_id -> evidence text (title + abstract)
    support_fn      : (claim, evidence, context) -> [0,1]; default lexical_support
    threshold       : groundedness score at/above which a claim counts as grounded
    context         : the query text (restating it is not a hallucination)
    """
    support_fn = support_fn or lexical_support
    n = len(recs)
    invalid, grounded, flagged = 0, 0, []
    for r in recs:
        evidence = text_by_doc_id.get(r.get("doc_id"), "")
        cited = extract_citations(r.get("why", ""))
        allowed = {r.get("citation")}
        bad = [c for c in cited if c not in allowed]   # cited something it wasn't given
        score = support_fn(r.get("why", ""), evidence, context)
        is_grounded = score >= threshold               # content support, independent
        if bad:
            invalid += 1
        if is_grounded:
            grounded += 1
        if bad or not is_grounded:
            flagged.append({"doc_id": r.get("doc_id"), "support": round(score, 3),
                            "grounded": is_grounded, "bad_citations": bad})
    return {
        "num_claims": n,
        "citation_validity_rate": (n - invalid) / n if n else 1.0,
        "grounded_rate": grounded / n if n else 1.0,
        "hallucination_rate": (1.0 - grounded / n) if n else 0.0,
        "flagged": flagged,
    }
