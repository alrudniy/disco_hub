"""
dh2.pharma_tok -- the tokenizer that query<->document overlap is measured with.

REPLACES the rule in graded_metrics._toks / analysis.register_gap_analysis, which was:

    {w for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w) > 2 and w not in STOP}

Two independent defects, both measured:

1. PHARMA IDENTIFIERS VANISHED. `re.findall(r"[a-z0-9]+", "pd-l1")` -> ["pd", "l1"], and
   the len>2 filter drops both. So PD-L1, PD-1, IL-2 and B7-H1 tokenized to the EMPTY SET.
   The consequence is not a small bias -- it inverts the metric. "PD-L1 inhibitor" reduces
   to {"inhibitor"}, so an unrelated anti-CD20 patent scored overlap 1.000 while the
   correct B7-H1 patent scored 0.000.

2. NO STEMMING. "antibodies" and "antibody" are different strings, so a query about
   antibodies matching a document that says antibody counted as a MISS. This depresses
   overlap for exactly the queries a scout writes in their own words -- i.e. the ones the
   metric exists to identify -- so fixing (1) without fixing (2) trades one broken number
   for another.

Scope: this measures word overlap as a paraphrase proxy. It is NOT the retrieval tokenizer
(discovery_hub.keyword.tokenize) and NOT the exact-identifier router (dh2.identifiers).

Synonymy is handled, but ONLY when you ask for it -- pass `synonyms=` (see
dh2/build_gene_synonyms.py). The default is RAW, because the two callers want opposite
answers: eval bucketing must stay raw (a PD-L1/B7-H1 pair genuinely IS hard for a lexical
matcher, and that is the phenomenon being measured), while the §5 example check must
normalize (an alias swap is still a restatement of the document).

KNOWN LIMITS, stated rather than papered over:
  * PD-1 does not normalize. HGNC lists "PD1" as an alias of PDCD1, SNCA *and* SPATA2, so
    the builder drops it as ambiguous rather than guess. In a pharma corpus PD-1 is almost
    always PDCD1, but HGNC will not say so -- that needs a hand-curated overlay.
  * Hyphenated non-numeric entities split. CAR-T -> {"car"} because there is no way to
    tell CAR-T from "well-known" without a vocabulary. Digit-bearing forms are safe.
  * The stemmer is Porter steps 1a/1b/1c only (plurals, -ed, -ing, terminal y). It does
    not stem -ion/-ation/-ity, so "administration"/"administer" still miss.
  * Modifier-prefix peeling is a fixed list. "anti-PD-L1" resolves; a prefix not in
    _MODIFIER_PREFIX still glues to its target and silently misses.
  * Overlap remains a PROXY. Low overlap does not prove a query reads like a scout wrote
    it -- it only proves absent shared vocabulary. The human read is the only real check.
"""
from __future__ import annotations

import re

_STOP = set(
    "the a an of for and or to in with as by from thereof use uses using method methods "
    "composition compositions available licensing novel new therapy treatment".split()
)

# Units that appear glued to a dose ("100mg"). Digit-bearing, so the identifier rule below
# would otherwise admit them as content.
_DOSE = re.compile(r"\d+(mg|ml|kg|mm|nm|um|mcg|iu|ng|pg|mol|hr|min|day|wk)$")

# A token is a run of alphanumerics that may contain internal hyphens: pd-l1, b7-h1,
# bms-986165, covid-19, well-known.
_TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")

# Modifier prefixes that glue onto a target name. "Anti-PD-L1" is ONE hyphenated token, so
# without stripping it collapses to "antipdl1" -- which matches neither a query saying
# "PD-L1" nor the "pdl1" key in the synonym table. anti-X is the most common construction
# in this corpus, so leaving it glued silently defeats both the metric and normalization.
_MODIFIER_PREFIX = {"anti", "non", "pre", "post", "pro", "neo", "multi", "semi",
                    "sub", "super", "co", "re", "un", "mono", "bi", "tri"}

_VOWELS = "aeiou"


def _is_consonant(w: str, i: int) -> bool:
    c = w[i]
    if c in _VOWELS:
        return False
    if c == "y":
        return i == 0 or not _is_consonant(w, i - 1)
    return True


def _measure(stem: str) -> int:
    """Porter's m: the number of VC sequences in the stem."""
    n = 0
    i = 0
    L = len(stem)
    while True:                                   # skip an initial C
        if i >= L:
            return n
        if not _is_consonant(stem, i):
            break
        i += 1
    i += 1
    while True:
        while True:                               # V ...
            if i >= L:
                return n
            if _is_consonant(stem, i):
                break
            i += 1
        i += 1
        n += 1
        while True:                               # ... C
            if i >= L:
                return n
            if not _is_consonant(stem, i):
                break
            i += 1
        i += 1


def _has_vowel(stem: str) -> bool:
    return any(not _is_consonant(stem, i) for i in range(len(stem)))


def _double_consonant_suffix(w: str) -> bool:
    return len(w) >= 2 and w[-1] == w[-2] and _is_consonant(w, len(w) - 1)


def _cvc(w: str) -> bool:
    """C V C where the final C is not w, x or y -- Porter's *o condition."""
    if len(w) < 3:
        return False
    if not (_is_consonant(w, len(w) - 3) and not _is_consonant(w, len(w) - 2)
            and _is_consonant(w, len(w) - 1)):
        return False
    return w[-1] not in "wxy"


def stem(w: str) -> str:
    """Porter steps 1a/1b/1c: plurals and verb inflections.

    Deliberately stops there. Later Porter steps (-ational -> -ate, -iveness -> -ive, ...)
    buy little for query/document overlap and each one is a chance to conflate two
    biomedical terms that should stay apart.
    """
    if len(w) <= 2:
        return w

    # --- 1a: plurals
    if w.endswith("sses"):
        w = w[:-2]
    elif w.endswith("ies"):
        w = w[:-2]
    elif w.endswith("ss"):
        pass
    elif w.endswith("s") and len(w) > 4:
        # Porter strips a terminal "s" unconditionally. That is wrong here: gene and
        # protein symbols ending in S are not plurals. Unguarded, KRAS -> "kra",
        # NRAS -> "nra", HRAS -> "hra" -- the metric then scores a KRAS query against a
        # KRAS patent as a MISS. The length guard keeps every 4-letter symbol intact
        # while still stemming real plurals (cells -> cell, genes -> gene, drugs -> drug),
        # which are all >= 5 characters.
        w = w[:-1]

    # --- 1b: -eed / -ed / -ing
    step1b_flag = False
    if w.endswith("eed"):
        if _measure(w[:-1]) > 0:
            w = w[:-1]
    elif w.endswith("ed") and _has_vowel(w[:-2]):
        w = w[:-2]
        step1b_flag = True
    elif w.endswith("ing") and _has_vowel(w[:-3]):
        w = w[:-3]
        step1b_flag = True
    if step1b_flag:
        if w.endswith(("at", "bl", "iz")):
            w += "e"
        elif _double_consonant_suffix(w) and not w.endswith(("l", "s", "z")):
            w = w[:-1]
        elif _measure(w) == 1 and _cvc(w):
            w += "e"

    # --- 1c: terminal y -> i
    if w.endswith("y") and _has_vowel(w[:-1]):
        w = w[:-1] + "i"
    return w


def _emit_identifier(raw: str, out: set[str], synonyms: dict | None = None) -> None:
    """A digit-bearing token is an identifier. Emit ONE canonical form: hyphens collapsed.

    Emitting both "pd-l1" and "pdl1" was wrong. word_overlap divides by |query tokens|, so
    a two-form query token could never fully match a document using the other surface form:
    query "PD-L1" -> {pd-l1, pdl1} against a doc saying PDL1 -> {pdl1} scored 0.5, not 1.0.
    Collapsing to a single canonical form is what actually makes the variants equal.
    """
    if _DOSE.fullmatch(raw):
        return                                    # "100mg" is not content
    if not any(c.isalpha() for c in raw):
        return                                    # bare numbers: "10", "485", "2020"
    collapsed = raw.replace("-", "")
    if len(collapsed) >= 3:                       # drops figure refs like "3a"
        if synonyms:
            collapsed = synonyms.get(collapsed, collapsed).lower()
        out.add(collapsed)


def load_synonyms(path: str = "/workspace/dh_data_v2/gene_synonyms.json") -> dict[str, str]:
    """HGNC alias -> canonical symbol (see dh2/build_gene_synonyms.py)."""
    import json
    from pathlib import Path
    return json.loads(Path(path).read_text())["alias_to_symbol"]


def pharma_toks(s: str, synonyms: dict[str, str] | None = None) -> set[str]:
    """Content tokens of `s`, pharma-aware and stemmed.

    `synonyms` (HGNC alias -> canonical symbol) folds PD-L1 / B7-H1 / PDL1 onto CD274.

    PASS IT DELIBERATELY. The two jobs want OPPOSITE answers:

      * §5 example checking -> normalize=ON. An alias swap is still a restatement of the
        document. A hand-written example saying "PD-L1" where the source says "B7-H1"
        looks like a good low-overlap example to the raw metric and is not one -- it is a
        paraphrase wearing another name. Without normalization the §5 gate rewards
        alias-swapping.
      * Eval bucketing -> normalize=OFF (the default). A PD-L1/B7-H1 pair genuinely IS
        hard for a lexical matcher, and that difficulty is the phenomenon the register
        split exists to measure. Normalizing there erases it.
    """
    out: set[str] = set()
    for raw in _TOKEN.findall((s or "").lower()):
        # Peel modifier prefixes so the target name underneath is reachable:
        # anti-b7-h1 -> b7-h1 -> "b7h1" -> (synonyms) -> cd274.
        parts = raw.split("-")
        while len(parts) > 1 and parts[0] in _MODIFIER_PREFIX:
            parts = parts[1:]
        raw = "-".join(parts)
        if any(c.isdigit() for c in raw):
            _emit_identifier(raw, out, synonyms)
            continue
        if "-" in raw:
            # No digit and hyphenated: a compound word ("well-known", "tnf-alpha"). Split.
            # CAR-T loses its "t" here; see KNOWN LIMITS.
            parts = raw.split("-")
        else:
            parts = [raw]
        for p in parts:
            t = stem(p)
            if len(t) > 2 and t not in _STOP and p not in _STOP:
                out.add(t)
    return out


def word_overlap(query: str, doc_text: str, synonyms: dict[str, str] | None = None) -> float:
    """Fraction of the query's content words present in the doc. Same contract as the
    function it replaces: 0.0 when the query has no content tokens.

    synonyms=None (default) = RAW, for eval bucketing. Pass a table for §5 example checking.
    """
    q = pharma_toks(query, synonyms)
    return len(q & pharma_toks(doc_text, synonyms)) / len(q) if q else 0.0
