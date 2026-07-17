"""
Query -> graph entity linking: the part that makes the graph signal *real*.

The old graph signal scored candidates against the centroid of the top TEXT hits
-- so it was downstream of text retrieval and largely re-encoded the same signal,
which is why the eval (stage 10) found it didn't help. This module builds the
graph-space query from the query's OWN named entities instead: if a query names
an organization, inventor, or facility that exists in the graph, we anchor on
that node and let the R-GCN's learned structure (who invented/assigned/affiliated
with what) surface connected technologies -- including ones whose text does not
lexically match the query. That is the actual promise of the knowledge graph.

Honesty by construction: if a query links to NO graph entity (common for novel,
register-crossing research interests that don't name known players), the signal
ABSTAINS rather than guessing. It can only help or stay silent, never inject
noise. That also scopes the deck claim honestly -- the graph helps when the query
names entities connected to the answer, not on every query.

Pure, deterministic, unit-tested (tests/test_hybrid.py).
"""
from __future__ import annotations

import re

# Entity node types worth linking a free-text query to. Technologies are the
# retrieval targets, not query anchors, so they are excluded.
LINKABLE_NTYPES = ("organization", "inventor", "expert", "facility")

_NORM = re.compile(r"[^a-z0-9 ]+")
_WS = re.compile(r"\s+")
# Tokens too generic to alias a multi-word entity on (would cause false links).
_STOP_ALIAS = {"university", "medical", "center", "institute", "inc", "corp",
               "llc", "ltd", "company", "co", "the", "of", "and", "school",
               "college", "hospital", "labs", "laboratory", "laboratories"}


def normalize(s: str) -> str:
    return _WS.sub(" ", _NORM.sub(" ", (s or "").lower())).strip()


def build_surface_index(nodes) -> dict[str, set[str]]:
    """
    surface form -> set(node_id) for entity nodes. Indexes the full normalized
    label, plus single-token aliases for multi-word labels when the token is
    globally unique among surfaces and not a generic stopword (so "Eli Lilly" is
    reachable as "lilly", but "Pfizer Medical Center" is not reachable as
    "medical"). Precision is favored over recall: a wrong entity link is worse
    than a missing one.
    """
    full: dict[str, set[str]] = {}
    token_owners: dict[str, set[str]] = {}   # token -> node_ids it could alias

    for nd in nodes:
        if nd.get("ntype") not in LINKABLE_NTYPES:
            continue
        nid = nd["node_id"]
        surf = normalize(nd.get("label") or nid.split(":", 1)[-1])
        if not surf:
            continue
        full.setdefault(surf, set()).add(nid)
        toks = surf.split()
        if len(toks) > 1:
            for t in toks:
                if t not in _STOP_ALIAS and len(t) >= 3:
                    token_owners.setdefault(t, set()).add(nid)

    index = dict(full)
    for tok, owners in token_owners.items():
        if tok in index:            # already a full surface; don't override
            continue
        if len(owners) == 1:        # unique alias only
            index[tok] = set(owners)
    return index


def link_query(query: str, surface_index: dict[str, set[str]],
               max_gram: int = 4) -> list[str]:
    """
    Greedy longest-match n-gram linking. Scans left to right; at each position
    tries the longest n-gram (up to max_gram) that is a known surface form, links
    it, and advances past it. Returns the sorted unique list of linked node_ids.
    """
    toks = normalize(query).split()
    linked: set[str] = set()
    i = 0
    while i < len(toks):
        matched = False
        for g in range(min(max_gram, len(toks) - i), 0, -1):
            span = " ".join(toks[i:i + g])
            if span in surface_index:
                linked.update(surface_index[span])
                i += g
                matched = True
                break
        if not matched:
            i += 1
    return sorted(linked)
