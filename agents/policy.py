"""
Policy / answer-safety. Scope (a) of spec 4.5, and ONLY scope (a).

WHAT THIS AGENT IS FOR: this is a technology-scouting tool. It reads patents and
trial registrations and tells you who owns what. It is not a clinical decision
support system, and the failure mode that would end the conversation with a
pharma partner's legal team is this tool answering "what dose should I give this
patient". This agent's job is to notice when an answer has drifted out of the
scouting lane and into clinical advice, unhedged efficacy assertion, or an
overstated regulatory stage.

WHY SCOPE (b) IS NOT HERE: spec 4.5(b) is IP / freedom-to-operate, whose central
computation is patent expiry = priority_date + 20 years. The spec itself says to
verify `priority_date` exists before designing around it. It was verified, by
inspection of the corpus: `DiscoveryDoc` (discovery_hub/schema.py) has no
`priority_date`, no filing date and no grant date, and `extra{}` is empty across
all 603,369 docs. Expiry is therefore not computable from this data at any level
of effort, and the other two (b) items -- family members across jurisdictions,
assignee-vs-affiliation drift -- need bibliographic fields the corpus also lacks.
Scope (b) is not stubbed, not partially implemented, and not TODO'd here. It is
absent, because a half-built freedom-to-operate check is worse than none: it
would read as a clearance opinion. Building it means going back for new data.

TWO LAYERS, AND WHICH ONE IS LOAD-BEARING:

  RULES (always run) -- a hand-written lexicon/regex layer. Fast (sub-millisecond),
  auditable line-by-line, and works with no DH_LLM_API_KEY. It is the FLOOR: the
  ok=False gate is computed from rules alone.

  LLM (optional) -- a glm-4.6 pass that catches phrasings the rules miss. It is
  the CEILING: it may ADD warnings, never remove one and never flip the gate. This
  is deliberate. If an LLM-proposed flag could set ok=False, then a rate-limited
  API or an absent key would silently change what this tool blocks, and the gate
  would stop being a property of the answer and start being a property of the
  network. When the LLM does not run, `mode` says "deterministic" -- out loud, in
  the payload. It never pretends a model reviewed the text.

`mode` is typed "llm" | "deterministic" | "hybrid" per spec, but "llm" is never
emitted: the rules are never skipped, so an LLM-assisted run is always "hybrid".

KNOWN WART, FOR WHOEVER BUILDS THE ORCHESTRATOR: a block sets ok=False, per the
contract that lets the orchestrator degrade to evidence-only. But base.to_trace_row
maps ok=False to status="error", so a correct, deliberate policy block renders in
the demo's trace table as if this agent crashed. base.py is shared and not this
module's to edit, so the signal is carried in the payload instead:
`payload["blocking_flags"]` is non-empty exactly when the block was intentional,
and `error` is None. Render policy's row from those, not from ok alone -- "policy:
error" on the refusal slide would undercut the exact point that slide makes.

HONESTY ABOUT WHAT THE RULES ARE: the lexicons below are hand-built from reading
the corpus and thinking about the demo. There is NO labelled set of unsafe answers
for this domain, so this agent HAS NO PRECISION OR RECALL NUMBER and must not be
given one in any deck. It is a tripwire, not a classifier.

THE ONE THING THAT WAS MEASURED, AND EXACTLY WHAT IT MEANS: the rules were run
over the first 20,000 real clinicaltrials abstracts in docs.jsonl, fed in as if
the tool had emitted them. Every flag there is BY CONSTRUCTION a false positive --
a trial registration is descriptive text, not advice, not the tool's own claim.
Flags per 20,000 abstracts, before and after the tuning that measurement drove:

    dosing               81  ->   4     (0.02%)   <- the blocking flag
    efficacy_claim      928  -> 244     (1.2%)
    stage_overstatement 331  -> 331     (1.7%)

That is a FALSE-POSITIVE PROBE ON CORPUS TEXT. It is NOT accuracy: it says nothing
about what the rules CATCH, because no labelled unsafe answers exist to test
against. Recall here is entirely unknown and no number should be attached to it.
The probe is also worst-case-ish in one direction and lenient in another: trial
protocols are far denser in dosing language than a scouting answer will be, but
they are also not adversarial. The residual efficacy/stage flags are mostly true
statements about approved comparator drugs ("Docetaxel is approved for ...") --
correct to notice in the tool's own voice, which is why they warn rather than
block. Reproduce the probe before touching a lexicon; it is ~2 s over 20k lines.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

from agents.base import AgentResult, make_evidence, timed
from agents.llm import LLMClient

# --------------------------------------------------------------------------- #
# Lexicons and patterns.
#
# ALL OF THIS IS HAND-BUILT AND UNMEASURED -- see the module docstring. These are
# module constants precisely so they are tunable without touching the logic. Each
# group carries the reasoning that produced it, because in six months the reason a
# term is present is the only thing that makes it safe to remove.
# --------------------------------------------------------------------------- #

# Vocabulary of dose/administration. NOTE: on its own this fires constantly on the
# corpus -- clinicaltrials abstracts are full of "escalating intravenous doses",
# "administered at or below the MTD". Presence of a dosing WORD is not evidence of
# dosing ADVICE. This list is only ever used in conjunction with _ADVICE_FRAMES.
_DOSING_TERMS = re.compile(
    r"\b(?:dose|doses|dosed|dosing|dosage|posology|regimen|titrat\w+|"
    r"administer\w*|infusion|mg/kg|mg/m2|q\dw|bid|tid|qd)\b",
    re.IGNORECASE,
)

# A number with a unit -- "200 mg", "2 mg/kg every 3 weeks". Also only ever used
# with an advice frame; a patent claim reciting "about 10 mg/kg" is a description
# of an invention's scope, not a prescription.
_DOSE_QUANTITY = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:mg/kg|mg/m2|mcg|µg|ug|mg|ml|units?|iu)\b",
    re.IGNORECASE,
)

# THE CRUX OF THIS MODULE. A dosing flag must target the SPEECH ACT -- advice
# addressed to a clinician about a patient -- not the corpus's descriptive
# language. "A composition for the treatment of NSCLC, administered intravenously"
# is an invention description and must NOT flag. "What dose should I give this
# patient" is advice-seeking and MUST flag. The discriminator is the frame, and
# these are the two frames that carry it:
#
#   1. clinician-first-person / imperative: "what dose ... should I give"
#   2. deontic + an explicit care recipient: "the patient should receive 200 mg"
#
# Frames are graded STRONG vs WEAK, and the grading is empirical, not aesthetic.
# Scanning 20,000 real clinicaltrials abstracts as if the tool had emitted them
# (the harshest honest probe: none of that text is advice, so every flag is a false
# positive) an earlier flat frame list produced 81 dosing false positives. Two
# patterns caused nearly all of them:
#   "select the recommended dose for ..."          -- RP2D trial-speak
#   "what dose of the TA-CIN vaccine is safe ..."  -- a study objective
# Both are interrogative//selective about a dose without advising anyone. So they
# moved to WEAK, which requires _PATIENT_REF to corroborate. The demo query keeps
# firing on the STRONG "should I give" and is corroborated on both counts anyway.
_STRONG_ADVICE_FRAMES = re.compile(
    # NB: the verb alternation here excludes "dose" itself. It used to include it,
    # and matched "how much of the treatment dose would go to the lungs"
    # (NCT07476651) -- a quantity question about biodistribution, not an advice
    # frame. _DOSING_TERMS already supplies the dose substance; this must supply an
    # ADVICE verb or it supplies nothing.
    r"(?:\bhow\s+(?:much|often)\b[^.?!]{0,40}\b(?:give|administer|prescribe|take)\b"
    r"|\bshould\s+i\b[^.?!]{0,40}\b(?:give|administer|prescribe|start|dose|use)\b"
    r"|\bdo\s+i\b[^.?!]{0,40}\b(?:give|administer|prescribe|dose)\b"
    r"|\b(?:i|we|you)\s+(?:would|should|could|can|must)\s+"
    r"(?:give|administer|prescribe|start|titrate)\b"
    r"|\bi\s+recommend\b"
    r"|\btreat\s+(?:my|your|this)\s+patient\b)",
    re.IGNORECASE,
)

# Dose-selection language. Ubiquitous in trial protocols, so it only counts as
# advice when the sentence also names a care recipient.
_WEAK_ADVICE_FRAMES = re.compile(
    r"(?:\bwhat\s+(?:dose|dosage|regimen)\b"
    r"|\bwhich\s+(?:dose|dosage|regimen)\b"
    r"|\brecommended\s+(?:dose|dosage|regimen|starting\s+dose)\b"
    r"|\bprescribe\b)",
    re.IGNORECASE,
)

_DEONTIC = re.compile(
    r"\b(?:should|must|ought\s+to|needs?\s+to|is\s+advised\s+to|"
    r"is\s+recommended\s+to)\b",
    re.IGNORECASE,
)

# An explicit care recipient. "patients" (plural, generic) is deliberately ABSENT:
# "patients with R/R AML were enrolled" is trial description, not advice.
_PATIENT_REF = re.compile(
    r"\b(?:my|your|the|this|that|a|an)\s+(?:\w+\s+){0,4}?patient\b"
    r"|\b(?:him|her|the\s+subject)\b",
    re.IGNORECASE,
)

# Efficacy asserted as accomplished fact. Note what is NOT here: "for the
# treatment of", "therapeutic", "indicated for" -- all standard patent-abstract
# register describing what an invention is FOR, which is not a claim that it works.
_EFFICACY_CLAIMS = re.compile(
    # "cure" as a VERB only. The lookbehinds kill the noun sense, which the corpus
    # uses constantly and innocently -- "offered as the best chance of a cure"
    # (NCT02349776) was a measured false positive of the bare \bcures?\b pattern.
    r"(?:(?<!a\s)(?<!the\s)(?<!of\s)(?<!for\s)\bcures?\b|\bcured\b|\bcuring\b"
    # Past tense included on purpose: "the drug was effective" asserts a settled
    # fact just as hard as "is effective". Only the hedge/attribution saves it.
    r"|\b(?:is|are|was|were)\s+(?:highly\s+|clinically\s+|very\s+)?"
    r"(?:effective|efficacious)\b"
    r"|\b(?:proven|shown|demonstrated)\s+to\s+"
    r"(?:cure|treat|eliminate|eradicate|prevent|reverse)\b"
    r"|\bworks\s+(?:for|against|in)\b"
    r"|\beradicates?\b|\beliminates?\s+(?:the\s+)?(?:tumou?rs?|cancer|disease)\b"
    r"|\bguarantees?\b"
    r"|\bwill\s+(?:cure|eliminate|eradicate|prevent)\b)",
    re.IGNORECASE,
)

# Hedges/attributions that make an efficacy sentence a report rather than an
# assertion. "A 2019 trial reported the antibody was effective in mice" is a
# citation of a finding; "the antibody is effective" is a claim. Sentence-scoped.
_HEDGES = re.compile(
    r"\b(?:may|might|could|can\s+potentially|suggests?|suggesting|suggested"
    r"|appears?\s+to|seems?\s+to|potential(?:ly)?|preclinical|in\s+vitro"
    r"|in\s+vivo|in\s+mice|in\s+animals?|murine|investigational|hypothes\w+"
    # "claims"/"claimed" bare, not just "claims that": "the patent claims the
    # antibody is effective" is attribution to a patent's own assertion, which is
    # precisely what this corpus contains and what the tool should be relaying.
    r"|reported(?:ly)?|according\s+to|claims?|claimed|claiming"
    # Negation and open questions. "It is not yet known whether bexarotene is
    # effective" (NCT00055991) and "evaluate whether X are effective"
    # (NCT07081789) are the OPPOSITE of an efficacy claim, and both were measured
    # false positives. A negated or interrogative efficacy sentence asserts nothing.
    r"|whether|\bnot\s+yet\b|unknown|unclear"
    r"|(?:is|are|was|were|has|have|had|do|does|did|would|will)\s+not"
    r"|\bno\s+(?:evidence|benefit|improvement)\b"
    r"|is\s+being\s+(?:studied|investigated|evaluated|tested)"
    # "studying how well X works in treating patients" (NCT01222676) is a study
    # OBJECTIVE -- the whole point is that nobody knows yet whether it works.
    # Measured false positive of the "works in/for/against" pattern.
    r"|how\s+well\b|to\s+(?:see|determine|find\s+out)\s+(?:if|whether|how)"
    r"|trial\s+(?:reported|found|showed)|study\s+(?:reported|found|showed))\b",
    re.IGNORECASE,
)

# Regulatory stage asserted at the top of the ladder.
_APPROVAL_CLAIMS = re.compile(
    r"(?:\b(?:fda|ema|mhra)[-\s]?approved\b"
    # Bare "is approved" with no complement, not just "is approved FOR x": "the
    # drug is approved and widely used" asserts the same thing and was missed by
    # an earlier version of this pattern that required the preposition.
    r"|\b(?:is|are|was|were)\s+approved\b"
    r"|\bhas\s+been\s+approved\b|\bhave\s+been\s+approved\b"
    r"|\breceived\s+(?:fda|ema|regulatory|marketing)\s+approval\b"
    r"|\bgained\s+approval\b"
    r"|\bon\s+the\s+market\b|\bcommercially\s+available\b"
    r"|\bcleared\s+by\s+the\s+fda\b|\bmarketing\s+authorou?isation\b)",
    re.IGNORECASE,
)

# Evidence-side stage language, used ONLY to enrich a stage flag's rationale --
# never to escalate it. See _rule_stage for why.
_PHASE_LANGUAGE = re.compile(
    r"\b(?:phase\s*(?:1|2|3|i|ii|iii|i/ii|ii/iii)\b|preclinical|investigational"
    r"|first-in-human|dose\s+escalation|recruiting|not\s+yet\s+recruiting)\b",
    re.IGNORECASE,
)

# Router lexicon for should_fire (spec 4.2: policy only fires when the query or
# answer makes a regulatory, clinical or IP claim). Broader than the rules -- it
# decides whether the text is worth READING, not whether it is unsafe.
#
# IP TRIGGERS ARE DELIBERATELY ABSENT. Scope (b) is not built (see module
# docstring), so an IP-triggered run would fire, find zero flags, and render in the
# trace as "policy: fired, clean" -- which reads as "the IP was checked and it is
# fine". It was not checked. An agent that cannot inspect a claim must not appear
# to have inspected it, so policy stays out of the way and abstains on pure IP
# queries rather than certifying them by silence.
_ROUTER_TRIGGERS = (
    _DOSING_TERMS,
    _DOSE_QUANTITY,
    _STRONG_ADVICE_FRAMES,
    _WEAK_ADVICE_FRAMES,
    _EFFICACY_CLAIMS,
    _APPROVAL_CLAIMS,
    re.compile(r"\b(?:fda|ema|regulatory|approval|contraindicat\w+|"
               r"adverse\s+event|side\s+effects?|safety\s+profile|"
               r"standard\s+of\s+care|off-label)\b", re.IGNORECASE),
)

# Ordinal, hand-set, UNMEASURED. These are NOT probabilities and NOT calibrated --
# there is no labelled set to calibrate against. They exist so the trace can rank
# a dosing block above a stage warning, and mean nothing finer than that ordering.
_CONFIDENCE_BY_KIND = {"dosing": 0.85, "efficacy_claim": 0.60,
                       "stage_overstatement": 0.50}

# Only dosing blocks. Efficacy/stage flags warn: they are judgement calls on
# register, and a warn lets the orchestrator surface them without nuking an answer.
_BLOCKING_KINDS = frozenset({"dosing"})

_MAX_SPAN_CHARS = 240        # keep a flag's span quotable in a trace table
_SENTENCE_SPLIT = re.compile(r"(?<=[.?!;\n])\s+")

_LLM_SYSTEM = (
    "You review draft answers from a TECHNOLOGY SCOUTING tool that searches "
    "patents and clinical trial registrations. The tool must never give clinical "
    "advice. Flag ONLY these three problems in the ANSWER or the QUESTION:\n"
    "  dosing            - dosing or treatment guidance directed at a clinician "
    "or a specific patient. Descriptive text about how an invention is "
    "administered is NOT a dosing flag.\n"
    "  efficacy_claim    - a claim that a drug works, stated as established fact "
    "rather than as a hedged or attributed finding.\n"
    "  stage_overstatement - a development stage stated higher than the evidence "
    "supports, e.g. calling an investigational agent 'approved'.\n"
    "Quote the offending text verbatim in 'span'. If nothing qualifies, return an "
    "empty flags list. Do not invent problems."
)

_LLM_SCHEMA = ('{"flags": [{"kind": "dosing|efficacy_claim|stage_overstatement", '
               '"span": "<verbatim quote from the text>", "rationale": "<one sentence>"}]}')

_LLM_KINDS = frozenset(_CONFIDENCE_BY_KIND)


@dataclass(frozen=True)
class _Segment:
    """One sentence, tagged with which side of the exchange it came from."""
    origin: str      # "query" | "answer"
    text: str


@dataclass
class PolicyAgent:
    """
    Answer-safety review. Implements the `Agent` protocol (agents/base.py).

    ctx keys, all optional:
      answer     : str | dict | AgentResult-like -- the drafted answer under review
      evidence   : list[dict] with doc_id / quote_span / source_url
      candidates : list[dict] from the retrieval agent (doc_id/title/abstract/...);
                   used interchangeably with `evidence` to look up stage language

    Abstains when neither the query nor the answer makes a regulatory, clinical or
    efficacy claim -- which is the common case for a scouting query, and is the
    abstention the demo points at ("two agents declined; that's the design").
    """

    name: str = "policy"
    llm: LLMClient | None = None
    use_llm: bool = True

    def __post_init__(self) -> None:
        # Construct lazily-but-eagerly: LLMClient() is cheap and never touches the
        # network in __init__, and .available then tells us the truth up front.
        if self.llm is None and self.use_llm:
            self.llm = LLMClient()

    # ----------------------------------------------------------------- routing
    def should_fire(self, query: str, answer: Any = "") -> bool:
        """
        Spec 4.2's router. True iff the exchange contains regulatory, clinical or
        efficacy vocabulary worth reviewing.

        Cheap and intentionally over-inclusive relative to the rules: a text that
        trips this may still produce zero flags, and that is a real result ("read
        it, found nothing") as opposed to an abstention ("nothing here to read").
        The demo's query 1 -- "monoclonal antibody targeting PD-L1 for oncology" --
        trips none of these, so policy abstains, which is the point.
        """
        blob = f"{query}\n{_answer_text(answer)}"
        return any(p.search(blob) for p in _ROUTER_TRIGGERS)

    # -------------------------------------------------------------------- run
    @timed
    def run(self, query: str, ctx: dict) -> AgentResult:
        answer = _answer_text(ctx.get("answer", ""))
        if not self.should_fire(query, answer):
            return AgentResult.abstain(
                self.name,
                "query and answer make no clinical, efficacy or regulatory claim -- "
                "nothing in policy scope to review",
            )

        segments = _segment(query, answer)
        evidence_texts = _evidence_texts(ctx)

        flags: list[dict] = []
        flags += _rule_dosing(segments)
        flags += _rule_efficacy(segments)
        stage_flags, stage_evidence = _rule_stage(segments, evidence_texts)
        flags += stage_flags

        # The gate is computed from rules ONLY, before the LLM is consulted, so it
        # cannot depend on whether the network was up. See module docstring.
        rule_flags = list(flags)
        blocked = any(f["severity"] == "block" for f in rule_flags)

        mode = "deterministic"
        if self.use_llm and self.llm is not None and self.llm.available:
            extra = self._llm_flags(query, answer, rule_flags)
            if extra is not None:
                mode = "hybrid"
                flags += extra
            # extra is None => the call failed or returned unparseable text. mode
            # stays "deterministic": we do not claim a review that did not happen.

        flags = _dedupe_and_sort(flags)

        payload = {
            "flags": flags,
            "mode": mode,
            "scope": "answer-safety (spec 4.5a). IP/freedom-to-operate (4.5b) is "
                     "NOT implemented: docs.jsonl has no priority_date, so patent "
                     "expiry is not computable. No IP claim here is checked.",
            "rules_unmeasured": True,   # loud on purpose -- see module docstring
            "blocking_flags": [f["kind"] for f in flags if f["severity"] == "block"],
        }
        confidence = max((_CONFIDENCE_BY_KIND.get(f["kind"], 0.5) for f in flags),
                         default=0.0)

        return AgentResult(
            agent=self.name,
            ok=not blocked,          # block => ok=False => orchestrator degrades
            payload=payload,
            evidence=stage_evidence,
            confidence=confidence,   # 0.0 when clean: nothing was flagged, so there
                                     # is no verdict to be confident about
            latency_ms=0.0,          # overwritten by @timed
        )

    # -------------------------------------------------------------- LLM layer
    def _llm_flags(self, query: str, answer: str,
                   rule_flags: list[dict]) -> list[dict] | None:
        """
        Optional second pass. Returns [] (LLM saw nothing new), a list of warn-only
        flags, or None if the LLM did not usefully respond.

        Every flag from here is clamped to severity="warn" regardless of what the
        model thinks, including dosing. The model is allowed to widen the review,
        never to change the gate -- an API hiccup must not alter what this tool
        blocks. If the rules miss a dosing case the LLM catches, that is a bug in
        the rules to be fixed in the lexicon above, not papered over at runtime.
        """
        assert self.llm is not None
        obj = self.llm.chat_json(
            [{"role": "system", "content": _LLM_SYSTEM},
             {"role": "user",
              "content": f"QUESTION:\n{query}\n\nANSWER:\n{answer or '(no answer drafted)'}"}],
            schema_hint=_LLM_SCHEMA,
            temperature=0.0,
            # Generous on purpose: glm-4.6 bills reasoning tokens against
            # max_tokens and returns HTTP 200 with content="" if the budget runs
            # out mid-reasoning (see agents/LLM_CONTRACT.md). An empty body parses
            # to None here, which falls back cleanly -- but a larger budget avoids
            # burning the call in the first place.
            max_tokens=1024,
        )
        if obj is None:
            return None
        raw = obj.get("flags")
        if not isinstance(raw, list):
            return None      # a response without the contracted key is not a verdict

        seen = {(f["kind"], f["span"].lower()) for f in rule_flags}
        out: list[dict] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            kind = item.get("kind")
            span = item.get("span")
            if kind not in _LLM_KINDS or not isinstance(span, str) or not span.strip():
                continue     # drop anything off-contract rather than coercing it
            span = _clip(span)
            if (kind, span.lower()) in seen:
                continue     # the rules already have it; do not double-report
            seen.add((kind, span.lower()))
            rationale = item.get("rationale")
            out.append({
                "kind": kind,
                "severity": "warn",   # clamped -- see docstring
                "span": span,
                "rationale": (str(rationale).strip() if rationale
                              else "flagged by the LLM pass"),
                "source": "llm",
            })
        return out


# --------------------------------------------------------------------------- #
# Rules. Each returns flag dicts; each is pure and independently testable.
# --------------------------------------------------------------------------- #
def _rule_dosing(segments: Iterable[_Segment]) -> list[dict]:
    """
    Dosing / treatment guidance. BLOCK: a scouting tool has no business here, and
    this is the one flag that should stop an answer from shipping.

    Fires on ADVICE FRAME AND DOSING SUBSTANCE, never on either alone. That
    conjunction is the entire false-positive defence: the corpus is saturated with
    dosing substance ("escalating intravenous doses", "administered at or below the
    MTD", "about 10 mg/kg") and none of it is advice. The frame -- a clinician
    asking what to give, or a deontic aimed at a named patient -- is what makes it
    a speech act rather than a description.

    Scans the QUERY as well as the answer: "what dose should I give this patient"
    is out of scope on arrival, before anything is drafted.
    """
    flags = []
    for seg in segments:
        substance = _DOSING_TERMS.search(seg.text) or _DOSE_QUANTITY.search(seg.text)
        if not substance:
            continue

        strong = _STRONG_ADVICE_FRAMES.search(seg.text)
        weak = _WEAK_ADVICE_FRAMES.search(seg.text)
        patient = _PATIENT_REF.search(seg.text)
        if strong:
            why = (f"asks/advises what to administer ({strong.group(0).strip()!r}) "
                   f"in the {seg.origin}")
        elif weak and patient:
            why = (f"selects a dose ({weak.group(0).strip()!r}) for a specific "
                   f"patient, in the {seg.origin}")
        elif _DEONTIC.search(seg.text) and patient:
            why = f"states what a specific patient should receive, in the {seg.origin}"
        else:
            continue      # dosing words with no advice frame: descriptive, not advice

        flags.append({
            "kind": "dosing",
            "severity": "block",
            "span": _clip(seg.text),
            "rationale": ("Dosing/treatment guidance is out of scope for a "
                          f"technology-scouting tool: this {why}. Refer to the "
                          "product label and a treating clinician."),
            "source": "rules",
        })
    return flags


def _rule_efficacy(segments: Iterable[_Segment]) -> list[dict]:
    """
    Clinical/efficacy claims stated as fact. WARN.

    ANSWER ONLY. A user ASKING "is pembrolizumab effective in NSCLC?" is not making
    a claim -- flagging the question would punish the user for the tool's risk.
    Only the tool's own assertions are the tool's problem.

    A sentence carrying a hedge or an attribution ("a trial reported...", "may be
    effective in mice") is a report of a finding, not an assertion of fact, and is
    exactly the register this tool SHOULD use. Hedges are scoped to the sentence.
    """
    flags = []
    for seg in segments:
        if seg.origin != "answer":
            continue
        hit = _EFFICACY_CLAIMS.search(seg.text)
        if not hit:
            continue
        hedge = _HEDGES.search(seg.text)
        if hedge:
            continue
        flags.append({
            "kind": "efficacy_claim",
            "severity": "warn",
            "span": _clip(seg.text),
            "rationale": (f"States efficacy as established fact ({hit.group(0).strip()!r}) "
                          "with no hedge or attribution. This corpus is patents and "
                          "trial registrations -- it records what was claimed and "
                          "studied, not what was proven. Attribute it or hedge it."),
            "source": "rules",
        })
    return flags


def _rule_stage(segments: Iterable[_Segment],
                evidence_texts: list[tuple[str, str, str]]
                ) -> tuple[list[dict], list[dict]]:
    """
    Overstated development stage -- "approved" where the evidence says Phase 2. WARN.

    Returns (flags, evidence). ANSWER ONLY, same reasoning as _rule_efficacy.

    WHY THIS WARNS AND NEVER BLOCKS, even when the evidence contradicts it: the
    obvious escalation is "answer says approved + retrieved doc says Phase 1 =>
    contradiction => block". It does not survive contact with this corpus. The
    documents are patents and trial registrations; phase/investigational language
    is near-ubiquitous in them, so a doc-mentions-a-phase test would escalate
    almost every approval sentence, including the true ones (a drug can be both
    approved for one indication and in Phase 2 for another -- pembrolizumab is).
    The phase language is reported in the rationale and carried as evidence so a
    human can adjudicate; it does not decide. A tripwire that always trips is a
    broken tripwire.
    """
    flags: list[dict] = []
    evidence: list[dict] = []
    for seg in segments:
        if seg.origin != "answer":
            continue
        hit = _APPROVAL_CLAIMS.search(seg.text)
        if not hit:
            continue

        staged = [(doc_id, url, _PHASE_LANGUAGE.search(text))
                  for doc_id, url, text in evidence_texts]
        staged = [(doc_id, url, m) for doc_id, url, m in staged if m]
        if staged:
            doc_id, url, match = sorted(staged, key=lambda t: t[0])[0]  # tie-break by id
            note = (f" The retrieved evidence describes an earlier stage "
                    f"({match.group(0).strip()!r} in {doc_id}); this was not "
                    f"adjudicated, only noticed.")
            evidence.append(make_evidence(doc_id, match.group(0).strip(), url))
        else:
            note = (" No retrieved document was checked against this -- the claim is "
                    "flagged on its own register, not on a contradiction.")

        flags.append({
            "kind": "stage_overstatement",
            "severity": "warn",
            "span": _clip(seg.text),
            "rationale": (f"Asserts regulatory approval ({hit.group(0).strip()!r}). "
                          "Nothing in this corpus establishes approval status: it "
                          "holds patents and trial registrations, which record "
                          "filings and studies." + note),
            "source": "rules",
        })
    return flags, _dedupe_evidence(evidence)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _answer_text(answer: Any) -> str:
    """
    Coerce whatever the orchestrator has in hand into text. Spec 4.2 passes the
    synthesis agent's return value straight through, and that has been a str, a
    payload dict and an AgentResult at different points in the design -- accept all
    three rather than making policy's correctness depend on that choice.
    """
    if answer is None:
        return ""
    if isinstance(answer, str):
        return answer
    payload = getattr(answer, "payload", None)      # AgentResult-like
    if isinstance(payload, dict):
        return _answer_text(payload)
    if isinstance(answer, dict):
        for key in ("answer", "text", "summary", "content"):
            value = answer.get(key)
            if isinstance(value, str):
                return value
        return ""
    return str(answer)


def _segment(query: str, answer: str) -> list[_Segment]:
    """Sentence-split both sides, tagging origin. Hedge scoping needs sentences."""
    segments = [_Segment("query", s) for s in _sentences(query)]
    segments += [_Segment("answer", s) for s in _sentences(answer)]
    return segments


def _sentences(text: str) -> list[str]:
    if not text or not text.strip():
        return []
    return [s.strip() for s in _SENTENCE_SPLIT.split(text.strip()) if s.strip()]


def _evidence_texts(ctx: dict) -> list[tuple[str, str, str]]:
    """
    (doc_id, source_url, text) from ctx["evidence"] or ctx["candidates"]. The two
    carry different keys -- base.make_evidence emits quote_span, the retriever emits
    title/abstract -- so read both shapes and let the caller stay ignorant.
    """
    rows: list[tuple[str, str, str]] = []
    for key in ("evidence", "candidates"):
        for item in ctx.get(key) or []:
            if not isinstance(item, dict):
                continue
            doc_id = str(item.get("doc_id") or "")
            if not doc_id:
                continue
            text = " ".join(str(item.get(f) or "") for f in
                            ("quote_span", "title", "abstract", "text"))
            rows.append((doc_id, str(item.get("source_url") or ""), text))
    return rows


def _dedupe_and_sort(flags: list[dict]) -> list[dict]:
    """
    Deterministic order, ties broken by span text -- stage 09 verifies this repo's
    runs are reproducible, and a flag list that reorders between runs would show up
    there as a diff. Blocks sort first so a reader sees the worst thing first.
    """
    seen: set[tuple[str, str]] = set()
    unique = []
    for f in flags:
        key = (f["kind"], f["span"].lower())
        if key in seen:
            continue
        seen.add(key)
        unique.append(f)
    return sorted(unique, key=lambda f: (f["severity"] != "block", f["kind"], f["span"]))


def _dedupe_evidence(evidence: list[dict]) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    out = []
    for e in evidence:
        key = (e["doc_id"], e["quote_span"])
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return sorted(out, key=lambda e: (e["doc_id"], e["quote_span"]))


def _clip(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= _MAX_SPAN_CHARS else text[:_MAX_SPAN_CHARS - 1] + "…"
