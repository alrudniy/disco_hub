"""
dh2.identifiers -- the routed exact-identifier channel (RASC Stage 1).

The recommendation is explicit: **do not restore global BM25** (measured cross-register
regression), but a scout who types "NCT04381936" or "US10485802" wants that exact record,
and no dense model reliably returns a literal ID string. So lexical retrieval survives
only as a *routed* channel: it fires when the query contains a recognizable identifier and
is a no-op otherwise.

Recognized:
  * NCT numbers        NCT04381936
  * US patents         US10485802B2, US 10,485,802, USPP12345P2
  * WO/EP/other        WO2020123456A1, EP3456789B1
  * CAS registry       50-00-0  (with checksum validation -- see _cas_checksum_ok)
  * Compound codes     AZD9291, LY3009120, BMS-986165, PF-07321332, GDC-0994

Precision matters more than recall here. A false positive doesn't just add a bad
candidate: it hands a lexical match a guaranteed slot in the union, which is the exact
failure mode the cascade exists to correct. The compound-code pattern is therefore
deliberately narrow (a known-sponsor prefix, or 2-4 letters + 3-6 digits) and rejects the
things that look like codes but aren't -- gene names (CD274, PD-1), HLA alleles, and
dosages (100mg).
"""
from __future__ import annotations

import re
from typing import Iterable

# --------------------------------------------------------------------------- #
# Patterns
# --------------------------------------------------------------------------- #
_NCT = re.compile(r"\bNCT\s?0?\d{7,8}\b", re.I)
_US_PATENT = re.compile(
    r"\bUS\s?(?:RE|PP|D)?\s?\d{1,2}[,\s]?\d{3}[,\s]?\d{3}\s?(?:[A-Z]\d?)?\b", re.I)
_WO_EP_PATENT = re.compile(r"\b(?:WO|EP|CN|JP|KR|CA|AU)\s?\d{4}[/\s]?\d{5,7}\s?(?:[A-Z]\d?)?\b",
                           re.I)
_CAS = re.compile(r"\b\d{2,7}-\d{2}-\d\b")
# Compound codes: sponsor prefix + digits (AZD9291, BMS-986165, PF-07321332, GDC-0994).
_COMPOUND = re.compile(r"\b([A-Z]{2,4})[-\s]?(\d{3,7})\b")

# Tokens that match _COMPOUND's shape but are never drug codes.
_COMPOUND_STOPWORDS = {
    "CD", "PD", "IL", "TNF", "HLA", "IGG", "IGA", "IGM", "HER", "EGFR", "KRAS", "BRAF",
    "TP", "BRCA", "ALK", "ROS", "MET", "RET", "NTRK", "PIK", "AKT", "MTOR", "JAK",
    "STAT", "VEGF", "PDGF", "FGF", "TGF", "CTLA", "LAG", "TIM", "TIGIT", "CAR", "TCR",
    "MHC", "APC", "NK", "DNA", "RNA", "MRNA", "SIRNA", "PCR", "ELISA", "IC", "EC",
    "MG", "ML", "KG", "MM", "NM", "UM", "PH", "PART", "FIG", "NO", "US", "WO", "EP",
    "PHASE", "TYPE", "GRADE", "DAY", "WEEK", "YEAR", "COVID", "SARS", "MERS",
}
# Known sponsor/development-code prefixes: always accepted (they're unambiguous).
_KNOWN_PREFIXES = {
    "AZD", "BMS", "PF", "GDC", "LY", "MK", "GSK", "AMG", "ABT", "ABBV", "BI", "BAY",
    "JNJ", "RO", "RG", "SAR", "TAK", "NVP", "CC", "INCB", "ARQ", "AG", "APG", "BGB",
    "CFI", "DS", "E", "EPZ", "GNE", "GS", "HM", "IPI", "KRT", "MRTX", "NUV", "OTX",
    "PLX", "PRT", "REGN", "RMC", "SHP", "SNDX", "SRA", "TNO", "TPX", "VX", "XL", "ZN",
}


def _cas_checksum_ok(cas: str) -> bool:
    """CAS numbers carry a check digit: sum(digit_i * position_from_right) % 10.
    Without this, any date-like or dosage-like 'NN-NN-N' string becomes an identifier."""
    try:
        body, check = cas.rsplit("-", 1)
        digits = body.replace("-", "")
        total = sum(int(d) * (i + 1) for i, d in enumerate(reversed(digits)))
        return total % 10 == int(check)
    except (ValueError, IndexError):
        return False


def _normalize(kind: str, raw: str) -> str:
    """Canonical form for matching against doc ids/fields: uppercase, no separators."""
    s = raw.upper().replace(" ", "").replace(",", "")
    if kind == "nct":
        digits = re.sub(r"\D", "", s)
        return f"NCT{digits.zfill(8)}"
    if kind in ("patent", "wo_ep"):
        return re.sub(r"[^A-Z0-9]", "", s)
    if kind == "compound":
        return re.sub(r"[-\s]", "", s)
    return s


def extract_identifiers(query: str) -> list[dict]:
    """Return [{kind, raw, normalized}] for every identifier in the query. Empty if none."""
    if not query:
        return []
    found: list[dict] = []
    seen: set[tuple[str, str]] = set()

    def _add(kind: str, raw: str):
        norm = _normalize(kind, raw)
        key = (kind, norm)
        if key not in seen:
            seen.add(key)
            found.append({"kind": kind, "raw": raw.strip(), "normalized": norm})

    for m in _NCT.finditer(query):
        _add("nct", m.group(0))
    for m in _US_PATENT.finditer(query):
        _add("patent", m.group(0))
    for m in _WO_EP_PATENT.finditer(query):
        _add("wo_ep", m.group(0))
    for m in _CAS.finditer(query):
        if _cas_checksum_ok(m.group(0)):
            _add("cas", m.group(0))
    for m in _COMPOUND.finditer(query):
        prefix = m.group(1).upper()
        if prefix in _KNOWN_PREFIXES:
            _add("compound", m.group(0))
        elif prefix not in _COMPOUND_STOPWORDS and len(m.group(2)) >= 4:
            # Unknown prefix: require 4+ digits. "CD274" (3 digits) stays a gene.
            _add("compound", m.group(0))
    return found


def contains_exact_identifier(query: str) -> bool:
    """True iff the exact-identifier channel should fire for this query."""
    return bool(extract_identifiers(query))


# --------------------------------------------------------------------------- #
# Lookup
# --------------------------------------------------------------------------- #
def _haystacks(doc_id: str, fields: dict | None) -> Iterable[str]:
    yield doc_id
    if fields:
        for v in fields.values():
            if isinstance(v, str):
                yield v


def build_identifier_index(docs: Iterable) -> dict[str, list[str]]:
    """normalized identifier -> [doc_id]. Built once over the corpus; ~O(n) and cheap.

    Scans doc_id + title + abstract. This is a *lookup table for literal IDs*, not a
    lexical retriever: it can only ever return documents whose text contains the exact
    identifier the scout typed, so it cannot reintroduce the BM25 cross-register
    regression.
    """
    index: dict[str, list[str]] = {}
    for d in docs:
        did = d.get("doc_id") if isinstance(d, dict) else getattr(d, "doc_id", None)
        if not did:
            continue
        title = (d.get("title") if isinstance(d, dict) else getattr(d, "title", "")) or ""
        abstract = (d.get("abstract") if isinstance(d, dict)
                    else getattr(d, "abstract", "")) or ""
        blob = f"{did} {title} {abstract[:2000]}"
        for ident in extract_identifiers(blob):
            index.setdefault(ident["normalized"], []).append(str(did))
        # the doc_id itself is often the bare identifier (e.g. "ct:NCT04381936")
        tail = str(did).split(":", 1)[-1]
        norm = re.sub(r"[^A-Z0-9]", "", tail.upper())
        if norm:
            index.setdefault(norm, []).append(str(did))
    return {k: sorted(set(v)) for k, v in index.items()}


def exact_identifier_search(query: str, index: dict[str, list[str]],
                            max_results: int = 20) -> list[tuple[str, float]]:
    """Return [(doc_id, 1.0)] for exact identifier hits. Empty when the query has none.

    Score is a constant 1.0 and is NEVER compared against a dense score -- this channel
    contributes candidates to the union, and Qwen decides the ordering.
    """
    idents = extract_identifiers(query)
    if not idents or not index:
        return []
    hits: list[str] = []
    for ident in idents:
        for did in index.get(ident["normalized"], []):
            if did not in hits:
                hits.append(did)
    return [(d, 1.0) for d in hits[:max_results]]
