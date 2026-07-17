"""
dh2.doc_views -- source-aware compact document views (RASC Stage 2).

The reranker sees a *rendered view* of a document, not the raw embedding_text. Why this
matters here: `embedding_text` is one undifferentiated blob, so a 512-token truncation
keeps whatever happens to be first (usually the title + the front of the abstract) and
silently drops claims, phase, population, and biomarkers -- exactly the fields that
separate "matches the words" from "matches the scientific requirement".

Two budgets, per the recommendation:
  * Qwen pointwise rerank: ~2,000-4,000 tokens/doc inside an 8,192-token pair context.
  * GroupRank: ~250-500 tokens/doc so ~10 documents fit in one group prompt.

DESIGN CONSTRAINT (recommendation S2): "For the July 17 implementation, use only fields
that are already accessible." The base `DiscoveryDoc` carries doc_id/title/abstract/
source/embedding_text; richer fields (claims, CPC, phase, biomarkers, sponsor) are NOT
guaranteed. So each renderer asks for its ideal fields, emits the ones that exist, and
degrades to title+abstract when they don't. No field is fabricated, and a missing field
is simply omitted rather than rendered as an empty header -- an empty "Phase:" line
teaches the reranker that the trial has no phase, which is a lie the model will act on.

Token budget is enforced by an optional tokenizer; without one it falls back to a
~4-chars-per-token heuristic. Truncation is per-field and proportional, so the abstract
gets squeezed before the title disappears.
"""
from __future__ import annotations

from typing import Any, Callable, Sequence

# Ideal field order per source (recommendation S2). Rendered only where present.
#   (attribute_or_key, display_label)
_SCHEMAS: dict[str, list[tuple[str, str]]] = {
    "uspto": [
        ("title", "Title"),
        ("abstract", "Abstract"),
        ("claims", "Independent claims or claim summary"),
        ("cpc", "CPC/IPC"),
        ("assignee", "Assignee"),
        ("priority_date", "Priority/publication date"),
    ],
    "clinicaltrials": [
        ("title", "Title"),
        ("brief_summary", "Brief summary"),
        ("conditions", "Conditions"),
        ("interventions", "Interventions"),
        ("phase", "Phase/status"),
        ("population", "Population/biomarkers"),
        ("primary_outcomes", "Primary outcomes"),
        ("sponsor", "Sponsor"),
    ],
    "pubmed": [
        ("title", "Title"),
        ("abstract", "Abstract"),
        ("concepts", "Concepts/keywords"),
        ("study_type", "Study type"),
        ("conclusion", "Conclusion"),
    ],
    "sbir": [
        ("title", "Title"),
        ("abstract", "Abstract"),
        ("technical_objective", "Technical objective"),
        ("application", "Application"),
        ("agency", "Agency"),
        ("awardee", "Awardee"),
        ("phase", "Phase"),
    ],
}

# Aliases: normalized `source` values seen in this corpus -> schema key above.
_SOURCE_ALIASES = {
    "patent": "uspto", "patents": "uspto", "patentsview": "uspto", "us": "uspto",
    "ct": "clinicaltrials", "ctgov": "clinicaltrials", "clinical_trials": "clinicaltrials",
    "trials": "clinicaltrials", "nct": "clinicaltrials",
    "publication": "pubmed", "publications": "pubmed", "pmid": "pubmed",
    "medline": "pubmed", "openalex": "pubmed",
    "sttr": "sbir", "sbir_sttr": "sbir", "awards": "sbir", "award": "sbir",
}

_HEADERS = {
    "uspto": "PATENT",
    "clinicaltrials": "CLINICAL TRIAL",
    "pubmed": "PUBLICATION",
    "sbir": "SBIR/STTR",
}


def normalize_source(source: str | None) -> str:
    """Map a raw source string onto a schema key. Unknown -> 'generic'."""
    if not source:
        return "generic"
    s = str(source).strip().lower()
    if s in _SCHEMAS:
        return s
    if s in _SOURCE_ALIASES:
        return _SOURCE_ALIASES[s]
    # doc_ids look like "us:123" / "ct:NCT01" -- accept a bare prefix
    head = s.split(":", 1)[0]
    if head in _SCHEMAS:
        return head
    return _SOURCE_ALIASES.get(head, "generic")


def _get(doc: Any, key: str) -> Any:
    """Field access that works for dicts, dataclasses, and DiscoveryDoc alike."""
    if isinstance(doc, dict):
        return doc.get(key)
    return getattr(doc, key, None)


def _stringify(value: Any) -> str:
    """Render a field value. Lists become '; '-joined; None/empty become ''."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        parts = [_stringify(v) for v in value]
        return "; ".join(p for p in parts if p)
    if isinstance(value, dict):
        parts = [f"{k}: {_stringify(v)}" for k, v in value.items()]
        return "; ".join(p for p in parts if p)
    return str(value).strip()


def _approx_tokens(text: str) -> int:
    """~4 chars/token. Only used when no tokenizer is supplied."""
    return max(1, len(text) // 4)


def render_view(doc: Any, *, max_tokens: int = 3000,
                token_counter: Callable[[str], int] | None = None,
                source: str | None = None) -> str:
    """Render one document as a compact, source-aware view within a token budget.

    doc may be a DiscoveryDoc, a dataclass, or a plain dict. Missing fields are omitted
    (never emitted as empty headers). Always degrades to title + abstract/embedding_text.
    """
    count = token_counter or _approx_tokens
    src = normalize_source(source if source is not None else _get(doc, "source"))
    schema = _SCHEMAS.get(src)

    fields: list[tuple[str, str]] = []
    if schema:
        for key, label in schema:
            val = _stringify(_get(doc, key))
            if val:
                fields.append((label, val))
    # Fallback / backfill: every source needs at least a title and a body.
    if not any(lbl == "Title" for lbl, _ in fields):
        t = _stringify(_get(doc, "title"))
        if t:
            fields.insert(0, ("Title", t))
    if len(fields) <= 1:
        body = _stringify(_get(doc, "abstract")) or _stringify(_get(doc, "embedding_text"))
        if body:
            fields.append(("Abstract", body))

    if not fields:
        return ""

    header = _HEADERS.get(src, "DOCUMENT")
    lines = [header]

    # Budget: header + labels are fixed cost; squeeze the long free-text fields
    # proportionally rather than hard-cutting the tail (which drops the claims/outcomes
    # the whole exercise depends on).
    fixed = count(header) + sum(count(f"{lbl}: ") for lbl, _ in fields)
    remaining = max(max_tokens - fixed, 0)
    costs = [count(v) for _, v in fields]
    total = sum(costs)

    if total <= remaining:
        keep = [v for _, v in fields]
    else:
        keep = []
        # Short fields (metadata: phase, assignee, dates) are cheap and highly
        # discriminative -- pay for them in full first, then split what's left.
        share = remaining / total if total else 0.0
        for (_, v), c in zip(fields, costs):
            allow = max(int(c * share), 24)
            keep.append(_truncate_to_tokens(v, allow, count))

    for (lbl, _), val in zip(fields, keep):
        if val:
            lines.append(f"{lbl}: {val}")
    return "\n".join(lines)


def _truncate_to_tokens(text: str, max_tokens: int, count: Callable[[str], int]) -> str:
    """Truncate to a token budget, cutting on a word boundary."""
    if count(text) <= max_tokens:
        return text
    # binary search on characters -- token_counter may be a real tokenizer (non-linear)
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count(text[:mid]) <= max_tokens:
            lo = mid
        else:
            hi = mid - 1
    cut = text[:lo].rstrip()
    if " " in cut:
        cut = cut[: cut.rfind(" ")]
    return cut.rstrip(" ,;.") + " ..."


def build_view_map(docs: Sequence[Any], *, max_tokens: int = 3000,
                   token_counter: Callable[[str], int] | None = None) -> dict[str, str]:
    """doc_id -> rendered view, for a corpus or a candidate shortlist."""
    out: dict[str, str] = {}
    for d in docs:
        did = _get(d, "doc_id") or _get(d, "document_id")
        if did is None:
            continue
        out[str(did)] = render_view(d, max_tokens=max_tokens, token_counter=token_counter)
    return out
