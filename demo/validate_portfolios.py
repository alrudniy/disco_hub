#!/usr/bin/env python3
"""
Hand-check the expertise-gap agent's portfolio extraction, before it is demoed.

Spec 4.3: "Pick 5 orgs whose portfolios you can check by hand (Mayo, Yale, BMS are
in the graph and verifiable on Google Patents). If the portfolio extraction is
wrong for those five, it is wrong everywhere, and a partner who knows pharma will
spot it instantly."

This is that check. For each org it prints the canonical variants that were
merged, the portfolio size before and after the merge, and a deterministic sample
of titles WITH CLICKABLE SOURCE URLS, so the claim "BMS owns this" can be settled
in a browser in ten seconds rather than believed.

It also shouts about the two things that are genuinely wrong, because a validator
that only prints reassuring numbers is not a validator:

  * UNMERGED FRAGMENTS. Canonicalization strips legal-form suffixes, so
    "bristol myers squibb co" folds into "bristol myers squibb". It CANNOT merge
    word-order or abbreviation variants: "univ yale" (622 technologies) and
    "yale university" (340) stay separate nodes with separate portfolios. Any
    query naming Yale therefore sees a fraction of Yale's real coverage. The
    heuristic below finds these siblings and reports them; it deliberately does
    NOT merge them, because automatic merging on token overlap would fold
    "univ bristol" into Bristol Myers Squibb.
  * ORGS THE QUERY CANNOT REACH. The agent only fires when entity_link resolves a
    name. "Genentech" does not link, because the node is labelled "Genentech Inc"
    and entity_link indexes full labels plus only globally-unique single tokens.

Memory: GraphIndex + doc_ids only (~1.3 GB). doc_vectors is mmapped and never
read here; no faiss.index, no bm25.json, no Retriever, no docs.jsonl. Titles come
from the graph's technology labels, verified byte-identical to docs.jsonl titles.

Run:
    DH_GRAPH_DIR=/home/alex/discovery_hub/data_merged/graph \
    DH_ARTIFACT_DIR=/home/alex/discovery_hub/data_merged/artifacts \
    DH_EMB_DIR=/home/alex/discovery_hub/data/embeddings \
    DH_INDEX_DIR=/home/alex/discovery_hub/data/index \
    venv/bin/python demo/validate_portfolios.py
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.expertise_gap import ExpertiseGapAgent
from agents.graph_index import canonical_org_key
from discovery_hub import config

# The five to check by hand. BMS/Mayo/Yale are the spec's picks; Celgene and
# Genentech are added because each exposes a different failure: Celgene has real
# subsidiaries that must NOT merge, Genentech is unreachable from a natural query.
ORGS_TO_VALIDATE = ["bristol myers squibb", "mayo clinic", "yale university",
                    "celgene", "genentech"]

N_SAMPLE_TITLES = 10

# A token appearing in more canonical keys than this is generic ("university",
# "medical") and is useless for spotting an unmerged sibling.
_GENERIC_KEY_COUNT = 200
_MIN_TOKEN_LEN = 4


def _source_url(doc_id: str) -> str:
    """
    A URL a human can actually click to check the assignment. docs.jsonl carries
    source_url, but loading 1.5 GB of corpus to print ten links is not a trade
    worth making -- both forms are derivable from the doc_id.
    """
    source, _, ident = doc_id.partition(":")
    if source == "uspto":
        return f"https://patents.google.com/patent/{ident}"
    if source == "clinicaltrials":
        return f"https://clinicaltrials.gov/study/{ident}"
    return f"({source} record {ident})"


def _token_to_key_count(graph) -> Counter:
    """
    How many canonical org keys use each token -- to tell 'yale' from 'medical'.

    Walks org rows through GraphIndex's public accessors rather than reading its
    packed key table directly: that table is another agent's internal, and this
    script must not break when it is refactored.
    """
    keys = {graph.canonical_key(int(r)) for r in np.flatnonzero(graph.ntype_codes == 2)}
    counts = Counter()
    for key in keys:
        if key:                     # each canonical key counted ONCE, not per fragment
            counts.update(set(key.split()))
    return counts


def _unmerged_siblings(graph, key: str, token_counts: Counter) -> list[tuple[int, str, str]]:
    """
    Org nodes that a human would probably call the same institution but that
    canonicalization left separate. HEURISTIC: shares a distinctive token with
    `key`, different canonical key, owns something.

    Returns (portfolio size, node_id, canonical key), largest first. This is an
    AUDIT HINT FOR A HUMAN, not a merge rule -- 'univ bristol' shares "bristol"
    with Bristol Myers Squibb and is a different institution entirely.
    """
    distinctive = {t for t in key.split()
                   if len(t) >= _MIN_TOKEN_LEN and token_counts[t] <= _GENERIC_KEY_COUNT}
    if not distinctive:
        return []
    out = []
    for row in np.flatnonzero(graph.ntype_codes == 2):
        row = int(row)
        other = graph.canonical_key(row)
        if not other or other == key:
            continue
        if distinctive & set(other.split()):
            n = len(graph.portfolio(row))
            if n:
                out.append((n, graph.node_id(row), other))
    out.sort(key=lambda x: (-x[0], x[1]))
    return out


def validate(agent: ExpertiseGapAgent, name: str, token_counts: Counter) -> dict:
    graph = agent.graph
    key = canonical_org_key(name)
    rows = [int(r) for r in np.flatnonzero(graph.ntype_codes == 2)
            if graph.canonical_key(int(r)) == key]

    print("=" * 78)
    print(f"ORG: {name!r}   ->   canonical key: {key!r}")
    print("=" * 78)

    if not rows:
        print(f"  !! NO ORG NODE resolves to canonical key {key!r}. NOT VALIDATED.")
        return {"name": name, "resolved": False}

    # -- What the agent would actually see for a natural query ----------------
    # The portfolio can be perfect and still unreachable: the agent only fires
    # when entity_link resolves the name. Probed with the BARE name -- an earlier
    # version of this script probed "<name> oncology" and reported Mayo Clinic as
    # unreachable, which was this script's bug, not the agent's: the node
    # `facility:mayo clinic oncology` exists, and entity_link matches the longest
    # n-gram first, so the 3-gram swallowed the org. That artifact is now a real
    # check of its own (`hazard` below) instead of a false alarm.
    linked = agent._link_orgs(name)
    fires = agent.should_fire(name)
    phrase = f"{name} oncology"
    linked_phrase = agent._link_orgs(phrase)
    hazard = fires and not linked_phrase

    # -- Variants merged -------------------------------------------------------
    portfolio = graph.canonical_portfolio(rows[0])
    variants = sorted(((len(graph.portfolio(int(r))), graph.node_id(int(r)))
                       for r in graph.canonical_org_rows(rows[0])), reverse=True)
    direct_max = max((n for n, _ in variants), default=0)

    print(f"  entity_link on {name!r}: fires={fires} linked={linked or '[]'}")
    print(f"  entity_link on {phrase!r}: linked={linked_phrase or '[]'}")
    print(f"  canonical variants merged ({len(variants)}):")
    for n, nid in variants:
        print(f"      {n:6d} techs   {nid}")
    print(f"  PORTFOLIO SIZE: {len(portfolio)}   "
          f"(largest single variant: {direct_max}; merging added "
          f"{len(portfolio) - direct_max})")
    srcs = Counter(graph.doc_id_of_tech(int(t)).split(":", 1)[0] for t in portfolio)
    print(f"  by source: {dict(sorted(srcs.items()))}")

    # -- Sample titles for the human ------------------------------------------
    rng = np.random.default_rng(config.SEED)
    take = min(N_SAMPLE_TITLES, len(portfolio))
    sample = sorted(rng.choice(portfolio, size=take, replace=False).tolist())
    print(f"  {take} sample technologies (check these against the assignee):")
    for t in sample:
        doc_id = graph.doc_id_of_tech(int(t))
        owners = ", ".join(graph.label(int(o)) for o in graph.owners_of(int(t)))
        print(f"      {doc_id}")
        print(f"        title    : {graph.label(int(t))[:96]}")
        print(f"        assignees: {owners[:96]}")
        print(f"        url      : {_source_url(doc_id)}")

    # -- The loud part ---------------------------------------------------------
    warnings = []
    if not fires:
        warnings.append(
            f"UNREACHABLE: entity_link does not link {name!r} to any org node, so "
            f"the agent NEVER FIRES on a query naming it. The {len(portfolio)}-"
            f"technology portfolio below is CORRECT AND UNREACHABLE. Cause: the "
            f"node is labelled {graph.label(rows[0])!r}, so the indexed surface is "
            f"its full normalized label; entity_link only adds a single-token alias "
            f"when that token is globally unique, and another node here shares it. "
            f"Fixing this belongs in entity_link/aliasing, not in this agent.")
    elif not any(graph.canonical_key(graph.org_row(n)) == key for n in linked
                 if graph.org_row(n) is not None):
        warnings.append(
            f"MISLINK: {name!r} links {linked}, none of which canonicalize to "
            f"{key!r}. The agent would analyse a different company.")
    if hazard:
        warnings.append(
            f"GREEDY-MATCH HAZARD: {name!r} links an org, but {phrase!r} links "
            f"{linked_phrase or '[]'} -- adding a word DESTROYED the org link, "
            f"because entity_link matches the longest n-gram first and a "
            f"facility/org node is named after the longer phrase. The agent will "
            f"abstain on the longer query. Phrasing-dependent, so rehearse the "
            f"exact demo query, not a paraphrase of it.")

    siblings = _unmerged_siblings(graph, key, token_counts)
    if siblings:
        unseen = sum(n for n, _, _ in siblings)
        print(f"\n  !! POSSIBLE UNMERGED FRAGMENTS (heuristic; a human must judge -- "
              f"this list contains genuinely different institutions too):")
        for n, nid, other_key in siblings[:8]:
            print(f"      {n:6d} techs   {nid:58s} key={other_key!r}")
        if unseen > len(portfolio) * 0.25:
            warnings.append(
                f"FRAGMENTED: {unseen} technologies sit on {len(siblings)} name-"
                f"variant node(s) that canonicalization did NOT merge, vs "
                f"{len(portfolio)} in the portfolio itself. Suffix-stripping cannot "
                f"merge word-order/abbreviation variants. If any of those variants "
                f"are the same institution, this portfolio UNDERSTATES coverage and "
                f"every gap derived from it inherits the error.")

    for w in warnings:
        print(f"\n  *** WARNING: {w}")
    if not warnings:
        print("\n  OK: links, merges, and portfolio look self-consistent.")
    return {"name": name, "resolved": True, "key": key,
            "portfolio": len(portfolio), "variants": len(variants),
            "fires": fires, "hazard": hazard, "warnings": warnings}


def main() -> int:
    print(f"graph dir   : {config.GRAPH_DIR}")
    print(f"artifact dir: {config.ARTIFACT_DIR}")
    print(f"emb dir     : {config.EMB_DIR}\n")

    agent = ExpertiseGapAgent.from_config()
    token_counts = _token_to_key_count(agent.graph)

    results = [validate(agent, name, token_counts) for name in ORGS_TO_VALIDATE]

    print("\n" + "=" * 78)
    print("SUMMARY  (portfolio extraction is only trustworthy where warnings are 0)")
    print("=" * 78)
    print(f"{'org':<26}{'key':<24}{'techs':>7}{'vars':>6}{'fires':>7}{'warn':>6}")
    for r in results:
        if not r["resolved"]:
            print(f"{r['name']:<26}{'-- UNRESOLVED --':<24}")
            continue
        print(f"{r['name']:<26}{r['key']:<24}{r['portfolio']:>7}{r['variants']:>6}"
              f"{str(r['fires']):>7}{len(r['warnings']):>6}")
    print("\nfires=False means the portfolio is right but no query can reach it.")
    total_warnings = sum(len(r.get("warnings", [])) for r in results)
    print(f"\n{total_warnings} warning(s) across {len(results)} organizations.")
    print("Portfolios are extracted from `technology --assigned_to--> organization`,")
    print("which is an assertion in the source data, not a verified ownership record.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
