"""
Known-answer tests for the compact graph index, on a hand-built 12-node fixture.

The fixture is a miniature of the real graph's awkward parts, because those are
what the index exists to get right:
  * assigned_to runs tech -> org, so a portfolio is a REVERSE lookup
  * affiliated_with mixes inventor->inventor (co-invention noise) with
    inventor->org (real affiliation); only the latter is an affiliation
  * experts have NO affiliated_with edge, so expert -> org is a two-hop
  * org names fragment ("acme" / "acme co"), and a subsidiary is not the parent

The 826-tech BMS portfolio and the 52 Chen Lieping patents are verified against
the real 4.52M-edge graph at build time, not here -- these tests must stay small
and fast, and must not touch the real artifacts.

Run: python -m pytest tests/ -q
"""
import json

import numpy as np
import pytest

from agents.graph_index import GraphIndex, build, canonical_org_key

# --------------------------------------------------------------------------- #
# Fixture graph
# --------------------------------------------------------------------------- #
_NODES = [
    {"node_id": "technology:uspto:T1", "ntype": "technology", "label": "Widget A", "doc_id": "uspto:T1"},
    {"node_id": "technology:uspto:T2", "ntype": "technology", "label": "Widget B", "doc_id": "uspto:T2"},
    {"node_id": "technology:uspto:T3", "ntype": "technology", "label": "Widget C", "doc_id": "uspto:T3"},
    {"node_id": "organization:acme", "ntype": "organization", "label": "Acme"},
    {"node_id": "organization:acme co", "ntype": "organization", "label": "Acme Co"},
    {"node_id": "organization:zeta labs a acme company", "ntype": "organization",
     "label": "Zeta Labs, a Acme Company"},
    {"node_id": "inventor:ada lovelace", "ntype": "inventor", "label": "Ada Lovelace"},
    {"node_id": "inventor:bob smith", "ntype": "inventor", "label": "Bob Smith"},
    {"node_id": "expert:carol jones", "ntype": "expert", "label": "Carol Jones"},
    {"node_id": "facility:site 1", "ntype": "facility", "label": "Site 1"},
]

_EDGES = [
    # tech -> org (the direction the spec gets backwards)
    {"src": "technology:uspto:T1", "dst": "organization:acme", "rel": "assigned_to"},
    {"src": "technology:uspto:T2", "dst": "organization:acme", "rel": "assigned_to"},
    {"src": "technology:uspto:T3", "dst": "organization:acme co", "rel": "assigned_to"},
    {"src": "technology:uspto:T1", "dst": "organization:acme", "rel": "assigned_to"},  # exact dup
    # an assignee that is a person: noise, must be dropped
    {"src": "technology:uspto:T2", "dst": "inventor:bob smith", "rel": "assigned_to"},
    # tech -> inventor
    {"src": "technology:uspto:T1", "dst": "inventor:ada lovelace", "rel": "invented_by"},
    {"src": "technology:uspto:T2", "dst": "inventor:ada lovelace", "rel": "invented_by"},
    {"src": "technology:uspto:T3", "dst": "inventor:bob smith", "rel": "invented_by"},
    # inventor -> org: a real affiliation
    {"src": "inventor:ada lovelace", "dst": "organization:acme", "rel": "affiliated_with"},
    # inventor -> inventor: co-invention, NOT an affiliation
    {"src": "inventor:ada lovelace", "dst": "inventor:bob smith", "rel": "affiliated_with"},
    # tech -> expert
    {"src": "technology:uspto:T1", "dst": "expert:carol jones", "rel": "investigated_by"},
    # a relation this index does not traverse
    {"src": "technology:uspto:T1", "dst": "facility:site 1", "rel": "located_at"},
]


@pytest.fixture(scope="module")
def gi(tmp_path_factory):
    d = tmp_path_factory.mktemp("graph")
    with (d / "nodes.jsonl").open("w") as fh:
        for n in _NODES:
            fh.write(json.dumps(n) + "\n")
    with (d / "edges.jsonl").open("w") as fh:
        for e in _EDGES:
            fh.write(json.dumps(e) + "\n")
    # node_ids.json is the row-order authority (aligned to rgcn_node_emb.npy).
    # Deliberately NOT in nodes.jsonl order, to prove rows follow this file.
    ids = [n["node_id"] for n in _NODES][::-1]
    (d / "node_ids.json").write_text(json.dumps(ids))
    summary = build(graph_dir=d, artifact_dir=d, out=d / "graph_index.npz")
    idx = GraphIndex.load(d / "graph_index.npz")
    idx._summary = summary
    return idx


# --------------------------------------------------------------------------- #
# Row order / identity
# --------------------------------------------------------------------------- #
def test_rows_follow_node_ids_json_not_nodes_jsonl(gi):
    # node_ids.json was reversed; row 0 must be the LAST node in nodes.jsonl.
    assert gi.node_id(0) == "facility:site 1"
    assert gi.n_nodes == len(_NODES)
    assert gi._summary["node_order_mismatches_vs_nodes_jsonl"] > 0


def test_ntype_and_label(gi):
    r = gi.row("inventor:ada lovelace")
    assert gi.ntype(r) == "inventor"
    assert gi.label(r) == "Ada Lovelace"


def test_doc_id_roundtrip(gi):
    r = gi.tech_row_of_doc_id("uspto:T1")
    assert gi.doc_id_of_tech(r) == "uspto:T1"
    assert gi.ntype(r) == "technology"
    assert gi.tech_row_of_doc_id("uspto:NOPE") is None


def test_org_row_rejects_non_org(gi):
    assert gi.org_row("organization:acme") is not None
    assert gi.org_row("inventor:ada lovelace") is None   # exists, but not an org
    assert gi.org_row("organization:nonexistent") is None


# --------------------------------------------------------------------------- #
# The portfolio: a REVERSE lookup over assigned_to
# --------------------------------------------------------------------------- #
def test_portfolio_is_reverse_of_assigned_to(gi):
    p = gi.portfolio(gi.org_row("organization:acme"))
    assert {gi.doc_id_of_tech(int(r)) for r in p} == {"uspto:T1", "uspto:T2"}


def test_portfolio_deduplicates(gi):
    # T1 -> acme appears twice in the fixture; a dup would double-count a tech.
    p = gi.portfolio(gi.org_row("organization:acme"))
    assert len(p) == len(set(p.tolist())) == 2


def test_person_assignee_is_dropped(gi):
    # tech --assigned_to--> inventor is noise; it must not become a portfolio.
    assert len(gi.portfolio(gi.row("inventor:bob smith"))) == 0
    assert gi._summary["dropped_assigned_to_non_org"] == 1


def test_owners_of(gi):
    owners = gi.owners_of(gi.tech_row_of_doc_id("uspto:T1"))
    assert {gi.node_id(int(r)) for r in owners} == {"organization:acme"}


def test_neighbours_are_sorted_for_determinism(gi):
    p = gi.portfolio(gi.org_row("organization:acme"))
    assert (np.diff(p) > 0).all()


# --------------------------------------------------------------------------- #
# affiliated_with: only the inventor->org half is an affiliation
# --------------------------------------------------------------------------- #
def test_inventor_to_orgs_excludes_co_inventor_edges(gi):
    orgs = gi.orgs_of_inventor(gi.row("inventor:ada lovelace"))
    assert {gi.node_id(int(r)) for r in orgs} == {"organization:acme"}
    # Bob is a co-inventor, not an employer -- he must not appear.
    assert gi.row("inventor:bob smith") not in orgs.tolist()
    assert gi._summary["dropped_affiliated_with_not_inventor_to_org"] == 1


def test_org_to_inventors_is_the_reverse(gi):
    invs = gi.inventors_of_org(gi.org_row("organization:acme"))
    assert {gi.node_id(int(r)) for r in invs} == {"inventor:ada lovelace"}


def test_inventors_and_techs_of_inventor(gi):
    ada = gi.row("inventor:ada lovelace")
    assert {gi.doc_id_of_tech(int(r)) for r in gi.techs_of_inventor(ada)} == {"uspto:T1", "uspto:T2"}
    t1 = gi.tech_row_of_doc_id("uspto:T1")
    assert {gi.node_id(int(r)) for r in gi.inventors_of(t1)} == {"inventor:ada lovelace"}


# --------------------------------------------------------------------------- #
# expert -> org: the two-hop, because the spec's direct edge does not exist
# --------------------------------------------------------------------------- #
def test_expert_to_org_two_hop(gi):
    carol = gi.row("expert:carol jones")
    assert {gi.doc_id_of_tech(int(r)) for r in gi.techs_of_expert(carol)} == {"uspto:T1"}
    # expert <-investigated_by- T1 -assigned_to-> acme
    assert {gi.node_id(int(r)) for r in gi.orgs_of_expert(carol)} == {"organization:acme"}


def test_expert_with_no_techs_returns_empty_not_error(gi):
    # Abstention-shaped: an empty result is a normal answer, never an exception.
    assert len(gi.orgs_of_expert(gi.row("inventor:bob smith"))) == 0


# --------------------------------------------------------------------------- #
# Org canonicalization: merge legal-form fragments, never subsidiaries
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("label, expected", [
    ("Acme", "acme"),
    ("Acme Co", "acme"),
    ("Acme, Inc.", "acme"),
    ("The Acme Corporation", "acme"),
    ("Bristol Myers Squibb Co", "bristol myers squibb"),
    # Subsidiaries: only the trailing legal form goes. The parent's name being a
    # SUBSTRING must never collapse a distinct legal entity into the parent.
    ("Juno Therapeutics, Inc., a Bristol-Myers Squibb Company",
     "juno therapeutics inc a bristol myers squibb"),
    ("Adnexus, A Bristol-Myers Squibb R&D Company",
     "adnexus a bristol myers squibb r d"),
    ("Co", "co"),          # never strip the last remaining token
])
def test_canonical_org_key(label, expected):
    assert canonical_org_key(label) == expected


def test_canonical_rows_merge_fragments(gi):
    acme = gi.org_row("organization:acme")
    assert gi.canonical_key(acme) == "acme"
    rows = {gi.node_id(int(r)) for r in gi.canonical_org_rows(acme)}
    assert rows == {"organization:acme", "organization:acme co"}


def test_canonical_rows_exclude_subsidiary(gi):
    acme = gi.org_row("organization:acme")
    sub = gi.org_row("organization:zeta labs a acme company")
    assert sub not in gi.canonical_org_rows(acme).tolist()
    assert gi.canonical_key(sub) == "zeta labs a acme"


def test_canonical_portfolio_unions_fragments(gi):
    # "acme" owns T1+T2, "acme co" owns T3 -> the real coverage is all three.
    p = gi.canonical_portfolio(gi.org_row("organization:acme"))
    assert {gi.doc_id_of_tech(int(r)) for r in p} == {"uspto:T1", "uspto:T2", "uspto:T3"}


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def test_build_is_deterministic(gi, tmp_path):
    d = tmp_path / "rebuild"
    d.mkdir()
    with (d / "nodes.jsonl").open("w") as fh:
        for n in _NODES:
            fh.write(json.dumps(n) + "\n")
    with (d / "edges.jsonl").open("w") as fh:
        for e in _EDGES:
            fh.write(json.dumps(e) + "\n")
    (d / "node_ids.json").write_text(json.dumps([n["node_id"] for n in _NODES][::-1]))
    build(graph_dir=d, artifact_dir=d, out=d / "graph_index.npz")
    again = GraphIndex.load(d / "graph_index.npz")
    acme = again.org_row("organization:acme")
    assert again.portfolio(acme).tolist() == gi.portfolio(gi.org_row("organization:acme")).tolist()
