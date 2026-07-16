"""
Evidence-gating: a second LLM pass whose only job is to attack the first one's answer.

WHAT THIS IS NOT: verification of truth. This is an LLM checking an LLM against a
handful of retrieved passages. It can only ever answer "does the retrieved text
support this sentence?" -- never "is this sentence true?". A claim can be true and
land as `unsupported` (the corpus simply lacks it); a claim can be false and land as
`supported` (the corpus is wrong). Both are correct behaviour for this module and
neither is a truth judgment. The term for what it does is EVIDENCE-GATING, and that
is the only term that should appear in a deck, a trace, or a payload. Nothing here
emits, or is named, "verified".

What it buys, honestly: it reduces the rate at which an ungrounded sentence reaches
the user, by refusing the whole answer when any sentence fails to tie back to a
retrieved doc_id. That reduction is UNMEASURED in this repo -- there is no labelled
set of grounded/ungrounded answers to measure it against. Do not put a number on it.

TWO PATHS, and the output always says which one ran:
  - `mode="llm"`           -- glm-4.6, adversarially framed (see _SYSTEM_PROMPT).
  - `mode="deterministic"` -- no DH_LLM_API_KEY, or the LLM call came back empty.
    A real lexical grounding check, and a STRICTLY WEAKER one. It never runs silently:
    payload carries mode, a `caveat`, and `detects_contradiction=False`.

THE GATE IS COMPUTED HERE, NOT BY THE MODEL. The model's own `verdict` /
`unsupported_count` fields are read and discarded -- recomputed from the
post-validated claim list. A model that fabricates a doc_id must not also get to
grade itself on whether that matters.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from agents.base import AgentResult, timed
from agents.llm import LLMClient
from discovery_hub.faithfulness import lexical_support

# --------------------------------------------------------------------------- #
# Deterministic-path thresholds.
#
# HAND-SET, NOT TUNED. There is no labelled grounding set in this repo to tune
# against, so these are judgment calls, deliberately biased toward flagging: a
# false "fail" costs the demo a synthesis and degrades to evidence-only, while a
# false "pass" ships an ungrounded claim. Asymmetric cost -> asymmetric threshold.
# `faithfulness.lexical_support` defaults to 0.5; 0.65 here is the stricter choice.
# --------------------------------------------------------------------------- #
_SUPPORT_THRESHOLD = 0.65

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-./%]*")

# Tokens whose fabrication is both most likely and most damaging in a scouting
# answer: any number, dose, percentage, year, or identifier (US6803192B1, PD-L1,
# NCT04789655), plus ALL-CAPS acronyms. A sentence can share ~all its vocabulary
# with the evidence and still invent the one number that matters -- overlap
# scoring alone gives that 0.9+ and waves it through, which is exactly the
# failure this second check exists to catch.
_HAS_DIGIT = re.compile(r"\d")
_ACRONYM = re.compile(r"^[A-Z][A-Z0-9\-]{1,}$")

_VALID_STATUS = ("supported", "contradicted", "unsupported")

# Never let a stray uppercase word (sentence-initial, or a common word) count as a
# risky entity; that would flag every sentence and make the gate useless.
_NOT_ENTITIES = {"A", "AN", "THE", "IT", "THIS", "THAT", "AND", "OR", "BUT", "IF",
                 "NO", "NOT", "US", "IS", "ARE", "WAS", "WERE", "BE", "I"}


@dataclass
class _Claim:
    text: str
    status: str
    doc_id: str | None

    def as_dict(self) -> dict:
        # The spec fixes this shape EXACTLY: {"text","status","doc_id"}. Anything
        # else a check learns travels in the payload, not smuggled into a claim.
        return {"text": self.text, "status": self.status, "doc_id": self.doc_id}


# --------------------------------------------------------------------------- #
# THE ADVERSARIAL FRAMING (spec 4.4).
#
# A verifier handed the synthesis agent's framing ("you are a helpful biotech
# scouting assistant; is this answer good?") shares the generator's priors and its
# blind spots, and returns agreement theatre -- it rubber-stamps fluent text
# because fluent text is what its framing taught it to like. So this prompt
# inverts the job: the model is not asked to assess the answer, it is asked to
# FALSIFY each sentence, with "unsupported" as the DEFAULT that a claim must earn
# its way out of by naming the doc_id that carries it. Burden of proof sits on the
# claim, not on the doubt. It is also told it is not being helpful and not writing
# prose -- both are how a chat model drifts back toward agreement.
# --------------------------------------------------------------------------- #
_SYSTEM_PROMPT = (
    "You are a falsification engine. You are not an assistant, you are not helpful, "
    "and you do not write prose. You are given EVIDENCE (numbered passages, each with "
    "a doc_id) and an ANSWER written by a different system that you have every reason "
    "to distrust.\n\n"
    "Split the ANSWER into atomic factual claims -- one verifiable assertion each. For "
    "every claim, actively TRY TO FALSIFY IT against the EVIDENCE, then assign:\n"
    "  supported    -- a specific passage states this. You MUST cite its exact doc_id.\n"
    "  contradicted -- a specific passage states the opposite. Cite its exact doc_id.\n"
    "  unsupported  -- the evidence does not settle it. doc_id must be null.\n\n"
    "RULES YOU MUST NOT BREAK:\n"
    "1. 'unsupported' is the DEFAULT. If you are uncertain, if the passage is merely "
    "adjacent or topically similar, or if the claim is plausible-sounding but not "
    "actually stated -- it is unsupported. Do not be generous.\n"
    "2. Your own world knowledge is INADMISSIBLE. A claim you know to be true is still "
    "'unsupported' if these passages do not state it.\n"
    "3. Only cite a doc_id that appears verbatim in the EVIDENCE. Never invent one.\n"
    "4. Numbers, doses, percentages, dates and identifiers must match the evidence "
    "exactly. An approximated or rounded number is NOT supported."
)

_SCHEMA_HINT = (
    '{"claims": [{"text": str, "status": "supported"|"contradicted"|"unsupported", '
    '"doc_id": str|null}], "verdict": "pass"|"fail", "unsupported_count": int}'
)


class VerifierAgent:
    """
    Evidence-gates a synthesized answer against the evidence it was built from.

    ctx = {"answer": str | dict, "evidence": [{doc_id, quote_span, source_url}, ...]}
    """

    name = "verifier"

    def __init__(self, llm: LLMClient | None = None) -> None:
        # Injectable so tests can hand in a fake with a scripted .chat_json and
        # never touch a socket. Default construction is inert without a key.
        self.llm = llm if llm is not None else LLMClient()

    @timed
    def run(self, query: str, ctx: dict) -> AgentResult:
        answer_text = _answer_text(ctx.get("answer"))
        evidence = _clean_evidence(ctx.get("evidence"))

        if not answer_text.strip():
            # Nothing was synthesized, so there is nothing to gate. Abstaining is
            # correct: reporting "fail" would tell the orchestrator to degrade an
            # answer that does not exist.
            return AgentResult.abstain(self.name, "no answer text to check")

        claims, mode, notes = self._verify(query, answer_text, evidence)

        # THE GATE. Recomputed here from post-validated claims -- never taken from
        # the model. Note the spec's wording only gates on `unsupported`, which
        # would let a CONTRADICTED claim (strictly worse: the evidence says the
        # opposite) sail through. Gating on both; unsupported_count stays the
        # spec's field and counts only unsupported, so the number means what the
        # spec says it means.
        unsupported = [c for c in claims if c.status == "unsupported"]
        contradicted = [c for c in claims if c.status == "contradicted"]
        passed = not unsupported and not contradicted

        supported = [c for c in claims if c.status == "supported"]
        payload = {
            "claims": [c.as_dict() for c in claims],
            "verdict": "pass" if passed else "fail",
            "unsupported_count": len(unsupported),
            "contradicted_count": len(contradicted),
            "mode": mode,
        }
        payload.update(notes)

        # Confidence = the fraction of claims that tied back to a retrieved doc.
        # It is an UNCALIBRATED proportion, not a probability that the answer is
        # right, and base.py forbids averaging it with any other agent's number.
        confidence = len(supported) / len(claims) if claims else 0.0

        return AgentResult(
            agent=self.name,
            # ok=False on a fail verdict is what drives the orchestrator's
            # degrade-to-evidence-only branch (spec 4.2 step 4: `if not v.ok`).
            # This deliberately bends base.py invariant #1 (ok=False means "the
            # agent broke"): the agent worked perfectly, it just refused the
            # answer. error stays None so the two remain distinguishable -- a
            # gate-fail is (ok=False, error=None), a crash is (ok=False, error=...).
            ok=passed,
            payload=payload,
            # Only the evidence actually leaned on, so the trace shows what carried
            # the answer rather than everything retrieval happened to return.
            evidence=[e for e in evidence
                      if e["doc_id"] in {c.doc_id for c in supported}],
            confidence=confidence,
            latency_ms=0.0,
        )

    # ----------------------------------------------------------------- #
    def _verify(self, query: str, answer: str,
                evidence: list[dict]) -> tuple[list[_Claim], str, dict]:
        """Try the LLM; fall back to the lexical check, saying so, if it can't run."""
        if self.llm is not None and getattr(self.llm, "available", False):
            raw = self._call_llm(query, answer, evidence)
            if raw is not None:
                claims, invalid = _validate_claims(raw, evidence)
                if claims:
                    notes: dict = {}
                    if invalid:
                        # A verifier citing a doc it was never shown is itself a
                        # finding -- surface it rather than just silently
                        # downgrading the claim.
                        notes["invalid_citations"] = invalid
                        notes["caveat"] = (
                            f"{len(invalid)} claim(s) cited a doc_id absent from the "
                            "retrieved evidence; each was forced to 'unsupported'.")
                    return claims, "llm", notes
            # Unavailable key, transport failure, unparseable JSON, or (per
            # agents/LLM_CONTRACT.md) a 200 with empty content because reasoning
            # tokens ate max_tokens. All of them mean "no LLM verdict" -> fall back
            # and label it. Never pretend the model ran.
            claims, notes = _deterministic_claims(answer, evidence)
            notes["caveat"] = ("LLM was configured but returned no usable verdict; "
                               "fell back to the weaker lexical check. " + notes["caveat"])
            return claims, "deterministic", notes

        claims, notes = _deterministic_claims(answer, evidence)
        return claims, "deterministic", notes

    def _call_llm(self, query: str, answer: str, evidence: list[dict]) -> dict | None:
        block = "\n\n".join(
            f"[{i}] doc_id: {e['doc_id']}\n{e['quote_span']}"
            for i, e in enumerate(evidence, 1)
        ) or "(no evidence was retrieved)"
        user = (f"QUESTION ASKED:\n{query}\n\nEVIDENCE:\n{block}\n\n"
                f"ANSWER TO FALSIFY:\n{answer}")
        # max_tokens is generous on purpose: glm-4.6 bills reasoning tokens against
        # the budget and returns HTTP 200 with content:"" if it runs out mid-think
        # (agents/LLM_CONTRACT.md). The contract's fix is thinking:{"type":"disabled"},
        # which the shared client does not currently expose; headroom is the
        # mitigation available from here.
        return self.llm.chat_json(
            [{"role": "system", "content": _SYSTEM_PROMPT},
             {"role": "user", "content": user}],
            schema_hint=_SCHEMA_HINT, temperature=0.0, max_tokens=2048)


# --------------------------------------------------------------------------- #
# LLM post-validation
# --------------------------------------------------------------------------- #
def _validate_claims(raw: dict, evidence: list[dict]) -> tuple[list[_Claim], list[dict]]:
    """
    Trust nothing the model returned except the claim texts.

    Every cited doc_id must exist in the evidence actually shown. A citation to
    anything else is a fabricated citation, and the claim resting on it is by
    definition unsupported -- the model has demonstrated it is not reading the
    evidence for that claim. Unknown statuses degrade to 'unsupported' too, per
    the module's default-to-doubt rule.
    """
    known = {e["doc_id"] for e in evidence}
    claims: list[_Claim] = []
    invalid: list[dict] = []

    for item in raw.get("claims") or []:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        status = str(item.get("status") or "").strip().lower()
        if status not in _VALID_STATUS:
            status = "unsupported"

        doc_id = item.get("doc_id")
        # doc_id is nullable and the model does return null for unsupported claims
        # (measured -- agents/LLM_CONTRACT.md). Normalize before membership testing.
        doc_id = str(doc_id).strip() if isinstance(doc_id, str) and doc_id.strip() else None

        if status in ("supported", "contradicted"):
            if doc_id is None or doc_id not in known:
                if doc_id is not None:
                    invalid.append({"text": text, "cited_doc_id": doc_id,
                                    "claimed_status": status})
                status, doc_id = "unsupported", None
        else:
            doc_id = None  # an unsupported claim cites nothing, by construction

        claims.append(_Claim(text=text, status=status, doc_id=doc_id))
    return claims, invalid


# --------------------------------------------------------------------------- #
# Deterministic path -- a real check, and an honestly weaker one
# --------------------------------------------------------------------------- #
def _deterministic_claims(answer: str, evidence: list[dict]) -> tuple[list[_Claim], dict]:
    """
    No LLM. Two independent lexical checks per sentence, both must pass:

      1. CONTENT OVERLAP -- reuses discovery_hub.faithfulness.lexical_support (the
         repo's existing proxy, already honest in its own docstring about what it
         cannot see). Best-scoring evidence span wins and is cited.
      2. RISKY-TOKEN GROUNDING -- every number, dose, percentage, date, identifier
         and acronym in the sentence must appear SOMEWHERE in the evidence. Check 1
         alone scores a sentence with one invented number at ~0.9 and passes it;
         this is what catches "approved by the FDA in 2019".

    WEAKER THAN THE LLM PATH, and the payload says so. It cannot detect
    contradiction at all (that needs entailment, not word counting), so it never
    emits 'contradicted' -- an answer that flatly reverses the evidence while
    reusing its vocabulary PASSES this check. It also cannot credit a faithful
    paraphrase that shares no vocabulary; those fail closed as unsupported.
    """
    spans = [e["quote_span"] for e in evidence]
    blob = " ".join(spans).lower()
    claims: list[_Claim] = []

    for sentence in _sentences(answer):
        best_score, best_doc = 0.0, None
        for ev in evidence:
            score = lexical_support(sentence, ev["quote_span"])
            # Ties break on doc_id so the citation is deterministic across runs,
            # not dependent on evidence ordering (stage 09 checks determinism).
            if score > best_score or (score == best_score and best_doc is not None
                                      and ev["doc_id"] < best_doc):
                best_score, best_doc = score, ev["doc_id"]

        ungrounded = sorted(t for t in _risky_tokens(sentence)
                            if t.lower() not in blob)
        grounded = best_score >= _SUPPORT_THRESHOLD and not ungrounded and best_doc
        claims.append(_Claim(text=sentence,
                             status="supported" if grounded else "unsupported",
                             doc_id=best_doc if grounded else None))

    notes = {
        "caveat": ("DETERMINISTIC MODE: no LLM ran. This is lexical overlap plus a "
                   "number/entity grounding check -- a WEAKER check than the LLM "
                   "verifier. It cannot detect contradiction, cannot credit a "
                   "paraphrase, and cannot catch a fluent hallucination that reuses "
                   "the evidence's vocabulary."),
        "detects_contradiction": False,
        "support_threshold": _SUPPORT_THRESHOLD,
        "thresholds_calibrated": False,  # hand-set; there is no labelled set here
    }
    return claims, notes


def _risky_tokens(sentence: str) -> set[str]:
    """Tokens whose invention is most damaging: anything with a digit, plus acronyms."""
    out = set()
    for tok in _TOKEN.findall(sentence):
        cleaned = tok.strip(".,;:")
        if not cleaned or cleaned.upper() in _NOT_ENTITIES:
            continue
        if _HAS_DIGIT.search(cleaned) or _ACRONYM.match(cleaned):
            out.add(cleaned)
    return out


def _sentences(text: str) -> list[str]:
    """
    Cheap sentence split. Deliberately not a full segmenter: over-splitting an
    abbreviation ('Inc. of Delaware') yields two fragments that are each checked
    for grounding, which is conservative in the flagging direction -- the failure
    mode is a spurious flag, not a missed one.
    """
    return [s.strip() for s in _SENTENCE_SPLIT.split((text or "").strip()) if s.strip()]


def _answer_text(answer) -> str:
    """
    The synthesis agent is being built concurrently and may hand back a str or a
    dict. Accept both; for a dict take the first populated prose field rather than
    dumping JSON, so that key names and doc_ids in the structure aren't checked for
    grounding as if they were claims.
    """
    if isinstance(answer, str):
        return answer
    if isinstance(answer, dict):
        for key in ("answer", "text", "summary", "synthesis"):
            value = answer.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return ""


def _clean_evidence(evidence) -> list[dict]:
    """Keep only well-formed, non-empty spans -- an empty span grounds nothing."""
    if not isinstance(evidence, list):
        return []
    out = []
    for e in evidence:
        if not isinstance(e, dict):
            continue
        doc_id, span = e.get("doc_id"), e.get("quote_span")
        if isinstance(doc_id, str) and doc_id and isinstance(span, str) and span.strip():
            out.append({"doc_id": doc_id, "quote_span": span,
                        "source_url": e.get("source_url", "")})
    return out
