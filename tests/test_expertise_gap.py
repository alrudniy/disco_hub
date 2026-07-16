"""
Known-answer tests for the expertise-gap agent, on a synthetic 25-node graph.

The fixture is small but it is not arbitrary -- it is shaped to reproduce the
exact failure modes the real 4.52M-edge graph exhibits, because those are what
this agent exists to get right:

  * a technology the org OWNS but which sits far from every capability centroid
    (the real case: uspto:US6183734B1 scores 0.358 against BMS's centroids while
    being assigned to BRISTOL MYERS SQUIBB CO). TWIN carries the identical vector
    but is owned by someone else, which isolates the ownership filter from the
    geometry -- if the filter regresses, PFAR becomes a gap and TWIN stays one.
  * org name fragments that must merge ("acme" / "acme co") and a subsidiary that
    must NOT ("zeta labs, a acme company")
  * an inventor node collapsed from namesakes, carrying more employers than any
    real person has (the real case: inventor:wang jun with 45)
  * an expert, which in this graph can only reach an org via the two-hop, because
    expert --affiliated_with--> org does not exist for any of the 213,687 experts

Nothing here touches the real artifacts: no faiss.index, no doc_vectors.npy, no
Retriever. Tests must pass with NO DH_LLM_API_KEY set, which is why every
narration assertion below expects the deterministic path.

Run: venv/bin/python -m pytest tests/test_expertise_gap.py -q
"""
import json

import numpy as np
import pytest

from agents import expertise_gap as eg
from agents.base import Agent, AgentResult, run_agent
from agents.graph_index import GraphIndex, build
from agents.expertise_gap import ExpertiseGapAgent
from discovery_hub import config

DIM = 8

# --------------------------------------------------------------------------- #
# Fixture graph.
#
# Acme's portfolio: 5 technologies on axis e0, 4 on e1, and PFAR out on e7.
# PFAR is the outlier: with k=2 it cannot own a centroid, so the centroid rule
# calls it a gap -- and Acme owns it. TWIN is PFAR's vector under a different
# assignee, and is a genuine gap.
# --------------------------------------------------------------------------- #
_PORTFOLIO = ([(f"uspto:PA{i}", 0) for i in range(5)] +      # cluster on e0
              [(f"uspto:PB{i}", 1) for i in range(4)] +      # cluster on e1
              [("uspto:PFAR", 7)])                           # owned, far, outlier

_TARGETS = [("uspto:TNEAR", 0),      # covered by the e0 cluster
            ("uspto:TFAR", 6),       # a real gap
            ("uspto:TWIN", 7),       # PFAR's vector, NOT owned by Acme -> a gap
            ("uspto:PFAR", 7)]       # owned by Acme -> must never be a gap

_COLLIDED_ORGS = [f"organization:filler {i}" for i in range(11)]  # > MAX_ORGS_PER_INVENTOR


def _nodes() -> list[dict]:
    nodes = [{"node_id": "organization:acme", "ntype": "organization", "label": "Acme"},
             {"node_id": "organization:acme co", "ntype": "organization", "label": "Acme Co"},
             {"node_id": "organization:zeta labs a acme company", "ntype": "organization",
              "label": "Zeta Labs, a Acme Company"},
             {"node_id": "organization:beta corp", "ntype": "organization", "label": "Beta Corp"},
             {"node_id": "organization:tinyshop", "ntype": "organization", "label": "Tinyshop"},
             {"node_id": "inventor:grace hopper", "ntype": "inventor", "label": "Grace Hopper"},
             {"node_id": "inventor:common name", "ntype": "inventor", "label": "Common Name"},
             {"node_id": "expert:carol jones", "ntype": "expert", "label": "Carol Jones"}]
    nodes += [{"node_id": o, "ntype": "organization", "label": o.split(":", 1)[1].title()}
              for o in _COLLIDED_ORGS]
    seen = set()
    for doc_id, _ in _PORTFOLIO + _TARGETS:
        if doc_id in seen:
            continue
        seen.add(doc_id)
        nodes.append({"node_id": f"technology:{doc_id}", "ntype": "technology",
                      "label": f"Title of {doc_id}", "doc_id": doc_id})
    # Tinyshop's single technology: an org too small to analyse.
    nodes.append({"node_id": "technology:uspto:TINY1", "ntype": "technology",
                  "label": "Title of uspto:TINY1", "doc_id": "uspto:TINY1"})
    return nodes


def _edges() -> list[dict]:
    edges = []
    # Acme's portfolio, split across two name fragments -- the union is the truth.
    for i, (doc_id, _) in enumerate(_PORTFOLIO):
        org = "organization:acme" if i % 2 == 0 else "organization:acme co"
        edges.append({"src": f"technology:{doc_id}", "dst": org, "rel": "assigned_to"})
    # The subsidiary owns something. It must NOT land in Acme's portfolio.
    edges.append({"src": "technology:uspto:TFAR",
                  "dst": "organization:zeta labs a acme company", "rel": "assigned_to"})
    edges.append({"src": "technology:uspto:TFAR", "dst": "organization:beta corp",
                  "rel": "assigned_to"})
    edges.append({"src": "technology:uspto:TWIN", "dst": "organization:beta corp",
                  "rel": "assigned_to"})
    edges.append({"src": "technology:uspto:TNEAR", "dst": "organization:beta corp",
                  "rel": "assigned_to"})
    edges.append({"src": "technology:uspto:TINY1", "dst": "organization:tinyshop",
                  "rel": "assigned_to"})
    # Inventors.
    edges.append({"src": "technology:uspto:TFAR", "dst": "inventor:grace hopper",
                  "rel": "invented_by"})
    edges.append({"src": "technology:uspto:TWIN", "dst": "inventor:common name",
                  "rel": "invented_by"})
    edges.append({"src": "inventor:grace hopper", "dst": "organization:beta corp",
                  "rel": "affiliated_with"})
    # Grace also "affiliated_with" Acme itself: Acme must be excluded from fillers.
    edges.append({"src": "inventor:grace hopper", "dst": "organization:acme",
                  "rel": "affiliated_with"})
    # The collided namesake: more employers than any real person has.
    for o in _COLLIDED_ORGS:
        edges.append({"src": "inventor:common name", "dst": o, "rel": "affiliated_with"})
    # An expert. Reachable to an org only via TFAR --assigned_to--> beta/zeta.
    edges.append({"src": "technology:uspto:TFAR", "dst": "expert:carol jones",
                  "rel": "investigated_by"})
    return edges


@pytest.fixture(scope="module")
def graph(tmp_path_factory) -> GraphIndex:
    d = tmp_path_factory.mktemp("eg_graph")
    nodes = _nodes()
    with (d / "nodes.jsonl").open("w") as fh:
        for n in nodes:
            fh.write(json.dumps(n) + "\n")
    with (d / "edges.jsonl").open("w") as fh:
        for e in _edges():
            fh.write(json.dumps(e) + "\n")
    (d / "node_ids.json").write_text(json.dumps([n["node_id"] for n in nodes]))
    build(graph_dir=d, artifact_dir=d, out=d / "graph_index.npz")
    return GraphIndex.load(d / "graph_index.npz")


@pytest.fixture(scope="module")
def vectors() -> tuple[np.ndarray, list[str]]:
    """
    One-hot-ish vectors on orthogonal axes + a little seeded noise, so cosines
    between different axes are ~0 and the covered/gap split is unambiguous.
    """
    rng = np.random.default_rng(0)
    doc_ids, rows = [], []
    for doc_id, axis in _PORTFOLIO + _TARGETS + [("uspto:TINY1", 3)]:
        if doc_id in doc_ids:
            continue
        v = np.zeros(DIM, dtype=np.float32)
        v[axis] = 1.0
        v += rng.normal(0, 0.01, DIM).astype(np.float32)
        doc_ids.append(doc_id)
        rows.append(v / np.linalg.norm(v))
    return np.asarray(rows, dtype=np.float32), doc_ids


@pytest.fixture(scope="module")
def agent(graph, vectors) -> ExpertiseGapAgent:
    vecs, doc_ids = vectors
    # llm=None is the point: with no DH_LLM_API_KEY these tests must still pass.
    return ExpertiseGapAgent(graph=graph, doc_vectors=vecs, doc_ids=doc_ids, llm=None)


def _candidates(*doc_ids: str) -> dict:
    return {"candidates": [{"doc_id": d, "title": f"Title of {d}",
                            "source_url": f"https://example.test/{d}"} for d in doc_ids]}


# --------------------------------------------------------------------------- #
# Contract
# --------------------------------------------------------------------------- #
def test_conforms_to_agent_protocol(agent):
    assert isinstance(agent, Agent)
    assert agent.name == "expertise-gap"


# --------------------------------------------------------------------------- #
# Routing: fires on an org, declines otherwise
# --------------------------------------------------------------------------- #
def test_should_fire_on_linked_org(agent):
    assert agent.should_fire("Acme gaps in oncology") is True


# --------------------------------------------------------------------------- #
# Target pool: the gap question must not be asked in a circle
# --------------------------------------------------------------------------- #
def test_topic_query_strips_the_linked_org(agent):
    # The org is what we measure gaps OF; it must not also select the field we
    # measure them IN. See ExpertiseGapAgent.topic_query.
    assert agent.topic_query("Acme gaps in oncology") == "gaps in oncology"


def test_topic_query_strips_a_canonical_variant_too(agent):
    assert agent.topic_query("Acme Co gaps in oncology") == "gaps in oncology"


def test_topic_query_is_unchanged_when_no_org_links(agent):
    q = "monoclonal antibody targeting PD-L1"
    assert agent.topic_query(q) == q


def test_topic_query_keeps_the_query_when_stripping_leaves_nothing(agent):
    # A bare org name has no topic to retrieve on. Returning "" would send an empty
    # query to the retriever; the caller reuses the original candidates instead.
    assert agent.topic_query("Acme") == "Acme"


def test_gap_targets_are_preferred_over_query_candidates(agent):
    # gap_targets (retrieved on the topic alone) and candidates (retrieved on the
    # literal org-bearing query) answer different questions. When both are present
    # the topic pool wins, and the payload says which ran.
    ctx = _candidates("uspto:P0")                       # an OWNED doc: no gap
    ctx["gap_targets"] = [{"doc_id": "uspto:TFAR", "title": "Title of uspto:TFAR",
                           "source_url": "https://example.test/TFAR"}]
    res = agent.run("Acme gaps in oncology", ctx)
    assert res.payload["target_pool"] == "topic"
    assert [g["doc_id"] for g in res.payload["gaps"]] == ["uspto:TFAR"]


def test_falls_back_to_query_candidates_and_says_so(agent):
    res = agent.run("Acme gaps in oncology", _candidates("uspto:TFAR"))
    assert res.payload["target_pool"] == "query"


def test_should_fire_false_when_no_org_named(agent):
    assert agent.should_fire("monoclonal antibody targeting PD-L1") is False


def test_should_fire_ignores_non_org_entities(agent):
    # An inventor links, but this agent analyses ORGS. A person has no portfolio.
    assert agent.should_fire("work by Grace Hopper") is False


# --------------------------------------------------------------------------- #
# Abstention is a first-class outcome, and it carries a readable reason
# --------------------------------------------------------------------------- #
def test_abstains_when_no_org_in_query(agent):
    r = agent.run("monoclonal antibody targeting PD-L1", _candidates("uspto:TFAR"))
    assert r.ok and r.abstained and r.confidence == 0.0
    assert "no organization named" in r.payload["reason"]


def test_abstains_when_portfolio_too_small(agent):
    # Tinyshop owns 1 technology. This is the guard that stops the real agent
    # analysing "What 3 Things Joint Venture, LLC" (1 tech) on the demo's own
    # refusal query, where entity_link resolves the token "what" to it.
    r = agent.run("Tinyshop oncology", _candidates("uspto:TFAR"))
    assert r.ok and r.abstained
    assert "only 1 technologies" in r.payload["reason"]
    assert str(eg.MIN_PORTFOLIO_FOR_GAP) in r.payload["reason"]


def test_abstains_when_retrieval_supplied_no_candidates(agent):
    r = agent.run("Acme oncology", {"candidates": []})
    assert r.ok and r.abstained
    assert "no candidate technologies" in r.payload["reason"]


def test_abstains_when_no_candidate_is_analysable(agent):
    r = agent.run("Acme oncology", _candidates("uspto:NOT_IN_GRAPH"))
    assert r.ok and r.abstained
    assert "absent from the" in r.payload["reason"]


# --------------------------------------------------------------------------- #
# The portfolio: canonical variants merge, subsidiaries do not
# --------------------------------------------------------------------------- #
def test_portfolio_unions_canonical_variants(agent):
    r = agent.run("Acme oncology", _candidates("uspto:TFAR"))
    assert not r.abstained
    # 5 assigned to "acme" + 5 to "acme co" -- the union is the real coverage.
    assert r.payload["portfolio_size"] == len(_PORTFOLIO)
    assert set(r.payload["canonical_variants_merged"]) == {"organization:acme",
                                                           "organization:acme co"}


def test_canonicalization_is_auditable(agent):
    r = agent.run("Acme oncology", _candidates("uspto:TFAR"))
    audit = r.payload["canonicalization_audit"]
    assert len(audit) == 1
    assert audit[0]["canonical_key"] == "acme"
    assert audit[0]["n_variants"] == 2
    # The merge must visibly change the number, or it is not worth auditing.
    assert audit[0]["techs_direct"] < audit[0]["techs_after_merge"] == len(_PORTFOLIO)


def test_subsidiary_is_not_absorbed(agent):
    r = agent.run("Acme oncology", _candidates("uspto:TFAR"))
    assert "organization:zeta labs a acme company" not in r.payload["canonical_variants_merged"]


# --------------------------------------------------------------------------- #
# THE REGRESSION: a technology the org owns is never a gap
# --------------------------------------------------------------------------- #
def test_owned_technology_is_never_a_gap(agent):
    r = agent.run("Acme oncology", _candidates("uspto:PFAR"))
    assert r.ok and r.abstained          # the only candidate was owned -> nothing left
    assert r.payload["reason"].count("1 already in the portfolio") == 1


def test_ownership_filter_not_geometry_is_what_saves_the_owned_tech(agent):
    # PFAR and TWIN carry the SAME vector. TWIN (assigned to Beta) is a gap;
    # PFAR (assigned to Acme) is not. That isolates the filter from the geometry:
    # if the ownership check regresses, PFAR appears here too.
    r = agent.run("Acme oncology", _candidates("uspto:PFAR", "uspto:TWIN"))
    gap_ids = [g["doc_id"] for g in r.payload["gaps"]]
    assert "uspto:TWIN" in gap_ids
    assert "uspto:PFAR" not in gap_ids
    assert r.payload["targets_already_owned"] == 1


# --------------------------------------------------------------------------- #
# The gap test itself
# --------------------------------------------------------------------------- #
def test_semantic_proximity_ranks_gaps_but_does_not_gate_them(agent):
    """
    THIS TEST ASSERTED THE OPPOSITE, and the real data refuted it.

    It used to require that TNEAR -- unowned, but sitting on Acme's e0 cluster -- was
    NOT a gap, i.e. that similarity to the footprint proves coverage. Measured over
    all 294 B7-titled technologies in the real graph, it does not: owned average 0.637
    cosine to their nearest centroid, unowned 0.526, and at the old 0.40 gate the rule
    found 16.7% of true gaps while calling 7.7% of owned tech missing. The synthetic
    fixture hid this because its clusters are orthogonal by construction; real patents
    in one protein family are not.

    So ownership gates (an assigned_to edge is a fact) and cosine ranks (it is a guess
    about text). Both unowned targets are gaps; the far one outranks the near one.
    """
    r = agent.run("Acme oncology", _candidates("uspto:TNEAR", "uspto:TFAR"))
    gap_ids = [g["doc_id"] for g in r.payload["gaps"]]
    assert "uspto:TFAR" in gap_ids
    assert "uspto:TNEAR" in gap_ids           # unowned => a gap, however familiar
    assert gap_ids.index("uspto:TFAR") < gap_ids.index("uspto:TNEAR")   # ranked, not gated


def test_gap_distance_is_cosine_distance_to_nearest_centroid(agent):
    r = agent.run("Acme oncology", _candidates("uspto:TFAR"))
    gap = r.payload["gaps"][0]
    assert gap["distance"] == pytest.approx(1.0 - gap["nearest_centroid_cos"], abs=1e-3)
    # The auditable second opinion travels with every gap.
    assert 0.0 <= gap["nearest_portfolio_doc_cos"] <= 1.0


def test_gaps_sorted_by_distance_then_doc_id(agent):
    r = agent.run("Acme oncology", _candidates("uspto:TFAR", "uspto:TWIN"))
    keys = [(-g["distance"], g["doc_id"]) for g in r.payload["gaps"]]
    assert keys == sorted(keys)


def test_threshold_is_off_by_default_and_is_opt_in(agent, monkeypatch):
    # Default 0.0 == disabled: a measured-weak signal must not silently discard gaps.
    assert eg.GAP_COSINE_THRESHOLD == 0.0
    r = agent.run("Acme oncology", _candidates("uspto:TNEAR"))
    assert r.payload["threshold"] == 0.0
    assert [g["doc_id"] for g in r.payload["gaps"]] == ["uspto:TNEAR"]

    # Opt in to the old filter and TNEAR (close to the e0 cluster) is discarded --
    # which is the behaviour the real-data measurement argues against, hence opt-in.
    monkeypatch.setattr(eg, "GAP_COSINE_THRESHOLD", 0.9)
    assert agent.run("Acme oncology", _candidates("uspto:TNEAR")).payload["gaps"] == []


def test_clustering_k_follows_spec_and_clamps(agent):
    r = agent.run("Acme oncology", _candidates("uspto:TFAR"))
    # k = min(8, max(2, n // 25)); n=10 -> 2. Never more than n.
    assert r.payload["n_clusters"] == 2
    assert r.payload["n_clusters"] <= r.payload["portfolio_size"]


def test_centroids_are_normalized_so_diffuse_clusters_are_not_penalised(agent):
    """
    A centroid is a MEAN of unit vectors, so it is never unit norm -- measured on
    the real graph, BMS's 8 centroids have norms 0.218 to 0.785. Comparing a raw
    dot product against a fixed threshold therefore penalises a cluster for being
    DIFFUSE rather than for being unrelated to the query, and reports a false gap.

    Built here rather than asserted on the module fixture because the fixture's
    clusters are tight (near-unit centroids), which hides the bug entirely.
    """
    from sklearn.cluster import KMeans

    dim = 14
    rows = []
    for i in range(1, 13):      # a weak shared direction e0, each dominated by e_i
        v = np.zeros(dim, dtype=np.float32)
        v[0], v[i] = 0.3, 1.0
        rows.append(v / np.linalg.norm(v))
    portfolio = np.asarray(rows, dtype=np.float32)
    target = np.zeros((1, dim), dtype=np.float32)
    target[0, 0] = 1.0          # sits squarely on the portfolio's shared direction

    raw = KMeans(n_clusters=2, random_state=config.SEED,
                 n_init=10).fit(portfolio).cluster_centers_.astype(np.float32)
    assert np.linalg.norm(raw, axis=1).min() < 0.5      # the diffuse cluster is short

    centroids = agent._cluster(portfolio)
    assert np.allclose(np.linalg.norm(centroids, axis=1), 1.0, atol=1e-5)

    # The consequence, which is the actual point: unnormalized, a target sitting on
    # the portfolio's own axis scores as if it were unrelated. Normalized, it scores
    # as the neighbour it is. Pinned against a fixed reference rather than
    # GAP_COSINE_THRESHOLD, which is now 0.0/disabled -- the cosine ranks gaps rather
    # than gating them, and a rank built on unnormalized centroids would order by how
    # DIFFUSE each cluster is instead of how far the target sits from the portfolio.
    reference = 0.4
    assert float((target @ raw.T).max()) < reference
    assert float((target @ centroids.T).max()) > reference


# --------------------------------------------------------------------------- #
# Fillers: who covers what the org does not
# --------------------------------------------------------------------------- #
def test_fillers_name_the_owner_and_exclude_the_queried_org(agent):
    r = agent.run("Acme oncology", _candidates("uspto:TFAR"))
    fillers = r.payload["gaps"][0]["fillers"]
    owners = {o["node_id"] for o in fillers["orgs"]}
    assert "organization:beta corp" in owners
    # The subsidiary is a distinct entity and IS a legitimate filler here.
    assert "organization:zeta labs a acme company" in owners
    assert "organization:acme" not in owners and "organization:acme co" not in owners


def test_inventor_route_excludes_the_queried_org(agent):
    # Grace is affiliated with Beta AND Acme; Acme must not be its own lead.
    r = agent.run("Acme oncology", _candidates("uspto:TFAR"))
    inventors = r.payload["gaps"][0]["fillers"]["inventors"]
    grace = next(i for i in inventors if i["node_id"] == "inventor:grace hopper")
    assert grace["orgs"] == ["Beta Corp"]
    assert grace["affiliations_suppressed_name_collision"] is False


def test_collided_inventor_affiliations_are_suppressed_and_flagged(agent):
    r = agent.run("Acme oncology", _candidates("uspto:TWIN"))
    inventors = r.payload["gaps"][0]["fillers"]["inventors"]
    collided = next(i for i in inventors if i["node_id"] == "inventor:common name")
    assert collided["affiliations_suppressed_name_collision"] is True
    assert collided["orgs"] == []        # 11 employers is a collision, not evidence


def test_expert_org_uses_the_two_hop(agent):
    # expert --affiliated_with--> org does not exist in this graph, for anyone.
    r = agent.run("Acme oncology", _candidates("uspto:TFAR"))
    experts = r.payload["gaps"][0]["fillers"]["experts"]
    carol = next(e for e in experts if e["node_id"] == "expert:carol jones")
    # Reached via: carol <-investigated_by- TFAR -assigned_to-> beta/zeta.
    assert "Beta Corp" in carol["orgs_via_investigated_tech"]
    assert "Acme" not in carol["orgs_via_investigated_tech"]


# --------------------------------------------------------------------------- #
# Narration: never silently pretend an LLM ran
# --------------------------------------------------------------------------- #
def test_narration_is_deterministic_without_a_key(agent):
    r = agent.run("Acme oncology", _candidates("uspto:TFAR"))
    assert r.payload["narration_mode"] == "deterministic"
    assert r.payload["narration_fallback_reason"] == "no_llm_api_key"
    assert "WITHOUT an LLM" in r.payload["narration"]


def test_narration_cites_only_real_doc_ids(agent):
    r = agent.run("Acme oncology", _candidates("uspto:TFAR"))
    assert "uspto:TFAR" in r.payload["narration"]


def test_llm_failure_falls_back_and_says_so(agent, graph, vectors):
    class _EmptyLLM:
        available = True
        def chat(self, *a, **k):
            return ""      # glm-4.6's silent reasoning-budget failure

    vecs, doc_ids = vectors
    a = ExpertiseGapAgent(graph=graph, doc_vectors=vecs, doc_ids=doc_ids, llm=_EmptyLLM())
    r = a.run("Acme oncology", _candidates("uspto:TFAR"))
    assert r.payload["narration_mode"] == "deterministic"
    assert r.payload["narration_fallback_reason"] == "llm_call_failed_or_returned_empty"


def test_llm_narration_is_used_when_available(agent, graph, vectors):
    class _LLM:
        available = True
        def chat(self, *a, **k):
            return "Acme lacks coverage here (uspto:TFAR)."

    vecs, doc_ids = vectors
    a = ExpertiseGapAgent(graph=graph, doc_vectors=vecs, doc_ids=doc_ids, llm=_LLM())
    r = a.run("Acme oncology", _candidates("uspto:TFAR"))
    assert r.payload["narration_mode"] == "llm"
    assert r.payload["narration_invalid_citations"] == []


def test_fabricated_doc_id_is_stripped_and_flagged(agent, graph, vectors):
    class _LiarLLM:
        available = True
        def chat(self, *a, **k):
            return "See uspto:TFAR and also uspto:US9999999B9 (invented)."

    vecs, doc_ids = vectors
    a = ExpertiseGapAgent(graph=graph, doc_vectors=vecs, doc_ids=doc_ids, llm=_LiarLLM())
    r = a.run("Acme oncology", _candidates("uspto:TFAR"))
    assert r.payload["narration_invalid_citations"] == ["uspto:US9999999B9"]
    assert "uspto:US9999999B9" not in r.payload["narration"]
    assert "[unverified citation removed]" in r.payload["narration"]
    assert "uspto:TFAR" in r.payload["narration"]      # the real one survives


@pytest.mark.parametrize("text, allowed, expect_invalid", [
    ("cites uspto:A1", {"uspto:A1"}, []),
    ("cites uspto:B2", {"uspto:A1"}, ["uspto:B2"]),
    ("no citations at all here", {"uspto:A1"}, []),
    # Ordinary prose with a colon must not be mistaken for a doc_id.
    ("Note: this is prose.", {"uspto:A1"}, []),
    ("clinicaltrials:NCT1 and uspto:A1", {"uspto:A1"}, ["clinicaltrials:NCT1"]),
])
def test_strip_invalid_citations(text, allowed, expect_invalid):
    out, invalid = eg._strip_invalid_citations(text, allowed)
    assert invalid == expect_invalid
    for bad in expect_invalid:
        assert bad not in out


# --------------------------------------------------------------------------- #
# Honesty of the payload
# --------------------------------------------------------------------------- #
def test_unmeasured_flag_is_always_set(agent):
    r = agent.run("Acme oncology", _candidates("uspto:TFAR"))
    assert r.payload["unmeasured"] is True
    assert "NO evaluation" in r.payload["method_note"]


def test_confidence_is_bounded_and_reflects_evidence(agent):
    r = agent.run("Acme oncology", _candidates("uspto:TFAR"))
    assert 0.0 <= r.confidence <= 1.0
    # A 10-technology portfolio is thin evidence; the score must not read as high.
    assert r.confidence < 0.7


def test_confidence_can_never_reach_certainty(agent):
    """
    Every evidence term saturates, so without the ceiling a large portfolio scores
    a flat 1.00 -- measured: BMS's real 3,155-technology portfolio does exactly
    that. An agent carrying unmeasured=True must not render as certain in the
    demo's trace table.
    """
    assert agent._confidence(10_000, 1.0, 1.0) == eg._CONFIDENCE_CEILING
    assert eg._CONFIDENCE_CEILING < 1.0
    assert agent._confidence(0, 0.0, 0.0) == 0.0


def test_evidence_entries_carry_doc_id_and_url(agent):
    r = agent.run("Acme oncology", _candidates("uspto:TFAR"))
    assert [e["doc_id"] for e in r.evidence] == [g["doc_id"] for g in r.payload["gaps"]]
    assert r.evidence[0]["source_url"] == "https://example.test/uspto:TFAR"


# --------------------------------------------------------------------------- #
# Determinism (stage 09 cares) and failure containment
# --------------------------------------------------------------------------- #
def test_run_is_deterministic(agent):
    ctx = _candidates("uspto:TNEAR", "uspto:TFAR", "uspto:TWIN", "uspto:PFAR")
    a = agent.run("Acme oncology", ctx)
    b = agent.run("Acme oncology", ctx)
    assert json.dumps(a.payload, sort_keys=True) == json.dumps(b.payload, sort_keys=True)
    assert a.confidence == b.confidence


def test_run_agent_wrapper_times_and_contains(agent):
    r = run_agent(agent, "Acme oncology", _candidates("uspto:TFAR"))
    assert isinstance(r, AgentResult) and r.ok and r.latency_ms >= 0.0


def test_a_broken_dependency_becomes_ok_false_not_a_raise(graph, vectors):
    class _Boom:
        available = True
        def chat(self, *a, **k):
            raise RuntimeError("z.ai exploded")

    vecs, doc_ids = vectors
    a = ExpertiseGapAgent(graph=graph, doc_vectors=vecs, doc_ids=doc_ids, llm=_Boom())
    r = run_agent(a, "Acme oncology", _candidates("uspto:TFAR"))
    assert r.ok is False and "z.ai exploded" in r.error
