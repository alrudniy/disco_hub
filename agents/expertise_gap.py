#!/usr/bin/env python3
"""
The expertise-gap agent: "Company X wants to move into indication Y. What do they
already cover, what are they missing, and who owns the invention that fills it?"

This is the one question in this system that no keyword search answers, because it
is a question about ABSENCE. You cannot retrieve a document that says "Bristol
Myers Squibb has no B7-H3 coverage" -- that document does not exist. It has to be
derived by traversing `assigned_to` and comparing what is owned against what is
being asked about. That derivation is this module.

HOW IT WORKS (and where it departs from the spec, which is wrong about the graph):

  1. link the query to org nodes            entity_link.link_query()
  2. portfolio = union over the org's CANONICAL VARIANTS of org_to_techs
     The spec writes `org --assigned_to--> tech`; the real edge runs
     tech --assigned_to--> org, so a portfolio is a REVERSE lookup. GraphIndex
     already stores it that way (`org_to_techs`).
  3. embed the portfolio from the EXISTING doc vectors and KMeans it -> the
     company's capability footprint
  4. targets = candidates for the TOPIC (the query with the org's name stripped --
     see topic_query), retrieved by the orchestrator and handed over in ctx. This
     agent never constructs a Retriever: a second one would cost ~20 GB of RAM.
  5. gap = a target the org DOES NOT OWN (exact, from assigned_to). The cosine to
     the capability footprint only ORDERS the gaps; it does not decide them. That
     order was the gate until it was measured -- see GAP_COSINE_THRESHOLD.
  6. for each gap, walk the graph for who DOES cover it (owners / inventors'
     employers / investigating experts) -- the licensing lead
  7. narrate. With no LLM key this is a deterministic summary, and it says so.

WHAT THIS AGENT IS NOT. There is no eval for it. Spec section 6 is explicit: a
judged set of gap-analyses does not exist, so this is a CAPABILITY DEMO, NOT A
MEASURED RESULT. `payload["unmeasured"]` is True on every single result, and the
confidence below expresses HOW MUCH EVIDENCE WAS AVAILABLE, not how likely the
analysis is to be right. Nothing here should ever be quoted as an accuracy number.

THREE THINGS THE REAL DATA FORCED, EACH OF WHICH WOULD OTHERWISE SHIP A BUG:

  * OWNED TARGETS ARE EXCLUDED BEFORE THE CENTROID TEST. Measured: on the demo's
    own money-shot query, `uspto:US6183734B1` ("Inhibition of tumor cell growth by
    administration of B7-transfected cells") scores 0.358 against BMS's centroids
    -- under any usable threshold, i.e. "BMS has no coverage here" -- and it is
    ASSIGNED TO BRISTOL MYERS SQUIBB CO. It is in their own portfolio. An
    `assigned_to` edge is a hard fact and a centroid is a lossy summary, so the
    fact wins: anything in the portfolio can never be reported as a gap. Without
    this filter the demo tells a BMS partner they don't own a patent they own.

  * THE CENTROID RULE DOES NOT CARRY THE FACT, so it no longer gates. Measured over
    all 294 B7-titled technologies: the ones BMS owns average 0.637 cosine to their
    nearest centroid, the ones they do NOT own average 0.526, and no cut separates
    them (at 0.40 the rule finds 16.7% of true gaps and calls 7.7% of owned tech
    missing). A B7-H3 patent reads like a B7-H1 patent. Ownership is a fact on an
    edge; similarity is a guess about text. The full table is in
    GAP_COSINE_THRESHOLD's comment. Portfolio centroids are diffuse (measured
    norms 0.22-0.79 for BMS's 8 clusters) because a 3,155-document portfolio
    spanning oncology trials and chemistry patents has no tight centre. The
    "covered" and "unrelated" distributions overlap heavily. `nearest_doc_cos` is
    therefore reported alongside every gap as an auditable second opinion.

  * INVENTOR AFFILIATIONS ARE DEGRADED BY NAME COLLISION. `inventor:wang jun` has
    45 affiliated orgs (Merck + BC Cancer + UPenn + Hansoh + ...) because entity
    resolution collapsed many different people into one node. See
    MAX_ORGS_PER_INVENTOR.

Memory: this agent mmaps doc_vectors.npy and materializes ONLY the portfolio's
rows (the largest org portfolio in the graph is 3,548 technologies = ~35 MB at
2560-dim float32). It never loads faiss.index, bm25.json, docs.jsonl, or a
Retriever. Titles come from the graph's technology labels, which were verified
byte-identical to docs.jsonl `title` -- so the 1.5 GB corpus stays on disk.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.base import AgentResult, make_evidence, timed
from agents.graph_index import NTYPES, GraphIndex
from agents.llm import LLMClient
from discovery_hub import config, entity_link

AGENT_NAME = "expertise-gap"

# --------------------------------------------------------------------------- #
# Tunable constants. Every one of these is HAND-SET. None is a learned or
# validated optimum, because there is no ground truth to validate against.
# --------------------------------------------------------------------------- #

# OWNERSHIP IS THE GATE. THE COSINE IS A RANKING SIGNAL. THIS ORDER IS MEASURED.
#
# A gap is a technology in the queried field that the organization DOES NOT OWN.
# That is read off `assigned_to` edges: exact, graph-derived, no threshold. The
# distance-to-capability-footprint is kept, but only to ORDER the gaps (farthest
# from what they do today = most novel to them), never to decide whether one exists.
#
# It was the gate, and the measurement is why it no longer is. Against BMS's 8
# centroids (seed 20240611, canonical portfolio n=3155), over the 294 B7-titled
# technologies in the graph, split by whether BMS actually owns them:
#
#                            n    mean cos to nearest centroid
#     BMS OWNS              13              0.637
#     BMS does NOT own     281              0.526
#
#     threshold   % of OWNED flagged "gap"   % of NOT-OWNED flagged "gap"
#     0.30                 0.0                          7.5
#     0.40                 7.7                         16.7
#     0.50                 7.7                         29.9
#     0.70                76.9                         97.5
#
# The two distributions overlap almost entirely. At 0.40 the rule found 16.7% of
# the technologies BMS demonstrably does not own -- it MISSED 83% of real gaps --
# while calling 7.7% of the ones it does own missing. No cut in the sweep separates
# them. That is not a threshold needing tuning; it is a signal that does not carry
# the fact being asked for. A B7-H3 patent READS like a B7-H1 patent, so cosine to a
# footprint cannot tell whether the company owns that family member. The graph can:
# it says BMS owns 0 of 68 B7-H3, 0 of 32 B7-H4, 0 of 26 B7-H1.
#
# This restores the spec's OWN thesis (section 0): "what does this company cover,
# what does it not cover, and who covers the gap" is a GRAPH TRAVERSAL question, not
# a NEAREST-NEIGHBOUR question. Section 4.3's algorithm then prescribed nearest
# neighbours anyway. The measurement above is what that contradiction costs.
#
# Kept as an OPTIONAL filter, OFF by default, because on a broader query than a
# single protein family the ordering may not be enough and a caller may want a
# distance cut. Setting it re-enables a rule measured to be weak -- do not set it
# and then report the result as coverage.
GAP_COSINE_THRESHOLD = config.env_float("DH_GAP_THRESHOLD", 0.0)

# Below this many technologies, we abstain instead of doing gap analysis.
#
# Not a nicety -- it is load-bearing twice over. (1) A capability footprint built
# from three documents is noise, and "they have no coverage of X" inferred from it
# is an artifact of sparsity, not evidence of absence. (2) It kills a real false
# positive: entity_link resolves the token "what" in the demo's own refusal query
# ("what dose of pembrolizumab...") to `organization:what 3 things joint venture
# llc`, which owns exactly 1 technology. Without this guard the agent solemnly
# analyses the portfolio of What 3 Things Joint Venture LLC on stage.
# Measured context: 53.6% of the 83,320 org nodes own exactly 1 technology and
# only 18.1% own 5 or more, so most org nodes in this graph cannot support the
# question at all. HAND-SET.
MIN_PORTFOLIO_FOR_GAP = 5

# An inventor affiliated with more orgs than this is treated as an entity-
# resolution collision and dropped from the filler walk.
#
# Measured over all 323,242 inventor nodes: mean 1.97 orgs, p50=1, p90=4, p99=10,
# max=83. The tail is not prolific consultants, it is collapsed namesakes --
# `inventor:wang jun` carries 45 unrelated employers, `inventor:li jian` 83. The
# demo's real case, `inventor:chen lieping`, has 8 (Mayo / Yale / Johns Hopkins /
# BMS / Amplimmune / MedImmune / Tayu Huaxia), all genuine, and sits under the
# cap. Set at the measured p99. HAND-SET; a real fix is entity resolution, not a
# threshold.
MAX_ORGS_PER_INVENTOR = 10

# Fillers reported per gap, per kind. Presentation only -- a licensing lead with
# 40 candidate owners is not a lead. Ranked deterministically before truncation.
MAX_FILLERS_PER_KIND = 5

# Clustering: k = min(8, max(2, n // 25)), per spec, clamped to n.
_MAX_CLUSTERS = 8
_MIN_CLUSTERS = 2
_TECHS_PER_CLUSTER = 25

# Portfolio size at which the "we have enough of a footprint to judge" evidence
# term saturates. HAND-SET.
_PORTFOLIO_EVIDENCE_SATURATION = 100.0

# Hard ceiling on the reported confidence.
#
# This agent has NO eval. A score of 1.00 rendered in the demo's trace table next
# to "expertise-gap | fired" reads as certainty to every person in the room, and
# certainty is precisely the thing an unevaluated heuristic cannot earn -- the
# evidence terms below saturate easily (a 3,155-technology portfolio maxes the
# first one on its own), so without a ceiling the agent reports 1.00 for BMS. The
# cap makes "this number can never mean certain" structural rather than a lucky
# consequence of the weights. HAND-SET.
_CONFIDENCE_CEILING = 0.85

# A doc_id looks like "<source>:<id>" -- "uspto:US6803192B1",
# "clinicaltrials:NCT04789655". Used ONLY to catch citations the LLM invented.
_DOC_ID_RE = re.compile(r"\b[a-z][a-z0-9_]*:[A-Za-z0-9][A-Za-z0-9._/-]*")

_LINKABLE_CODES = frozenset(
    i for i, t in enumerate(NTYPES) if t in entity_link.LINKABLE_NTYPES
)


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    """
    Row-wise L2 normalization.

    NOT redundant, twice over. doc_vectors.npy is documented as L2-normalized but
    measured row norms are 0.998-1.003, and -- far more importantly -- a KMeans
    centroid is a MEAN of unit vectors and is never unit norm (BMS's 8 centroids
    measured 0.218 to 0.785). Comparing raw dot products against a fixed threshold
    would therefore rank a tight cluster above a diffuse one for reasons that have
    nothing to do with the query. Cosine requires normalizing BOTH sides.
    """
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-12)


class ExpertiseGapAgent:
    """
    Agent protocol implementation. Construct once and reuse: the surface index
    costs ~3 s and ~0.5 GB to build, and the orchestrator should share one across
    agents rather than paying for it twice (pass `surface_index=`).
    """

    name = AGENT_NAME

    def __init__(self, graph: GraphIndex, doc_vectors: np.ndarray,
                 doc_ids: list[str], llm: LLMClient | None = None,
                 surface_index: dict[str, set[str]] | None = None) -> None:
        self.graph = graph
        self.doc_vectors = doc_vectors          # expected mmap_mode='r'
        self.doc_ids = doc_ids
        self.llm = llm
        self._surface_index = surface_index     # lazily built if not injected
        self._row_of_doc_id = {d: i for i, d in enumerate(doc_ids)}

    # -- construction ------------------------------------------------------- #
    @classmethod
    def from_config(cls, llm: LLMClient | None = None,
                    surface_index: dict[str, set[str]] | None = None
                    ) -> ExpertiseGapAgent:
        """
        Wire the agent from config paths. doc_vectors is MMAPPED, never read into
        RAM: the array is 603,369 x 2560 float32 = 6.17 GB on disk and this agent
        touches only a portfolio's worth of rows.

        doc_ids is read from EMB_DIR, which is the row-order authority for
        doc_vectors.npy. (INDEX_DIR/doc_ids.json was verified byte-identical to
        it, so either would do; EMB_DIR is the one that is definitionally aligned.)
        """
        graph = GraphIndex.load()
        doc_ids = json.loads((config.EMB_DIR / "doc_ids.json").read_text())
        vectors = np.load(config.EMB_DIR / "doc_vectors.npy", mmap_mode="r")
        return cls(graph=graph, doc_vectors=vectors, doc_ids=doc_ids, llm=llm,
                   surface_index=surface_index)

    @property
    def surface_index(self) -> dict[str, set[str]]:
        """Built on first use over every LINKABLE node type -- see _link_orgs."""
        if self._surface_index is None:
            self._surface_index = entity_link.build_surface_index(
                self._iter_linkable_nodes())
        return self._surface_index

    def _iter_linkable_nodes(self):
        """
        Generator over linkable nodes, straight from the loaded GraphIndex, so we
        never re-read the 224 MB nodes.jsonl. Yields rather than materializing:
        889,047 nodes are linkable and none of the dicts need to coexist.
        """
        codes = self.graph.ntype_codes
        for row in np.flatnonzero(np.isin(codes, list(_LINKABLE_CODES))):
            row = int(row)
            yield {"node_id": self.graph.node_id(row),
                   "ntype": NTYPES[int(codes[row])],
                   "label": self.graph.label(row)}

    # -- routing ------------------------------------------------------------ #
    def _link_orgs(self, query: str) -> list[str]:
        """
        Linked ORGANIZATION node_ids for a query.

        The surface index is built over ALL of entity_link.LINKABLE_NTYPES and the
        org filter is applied AFTER linking, deliberately. build_surface_index()
        only creates a single-token alias when that token is unique among the
        nodes it was given, so an org-only index would invent aliases that the
        shared index refuses (a token owned by both an org and an inventor is
        ambiguous and must not link). Filtering after keeps this agent's linking
        semantics identical to every other consumer's, and keeps entity_link's
        precision-over-recall contract intact.
        """
        return [n for n in entity_link.link_query(query, self.surface_index)
                if n.startswith("organization:")]

    def topic_query(self, query: str) -> str:
        """
        The query with the linked org's name removed: "Bristol Myers Squibb gaps in
        B7 family immunotherapy" -> "gaps in B7 family immunotherapy".

        WHY THIS EXISTS -- the target pool must not be selected by the org whose
        gaps are being measured. Asking "what does BMS lack in B7?" while retrieving
        for a query that CONTAINS "Bristol Myers Squibb" hands back BMS's own B7
        patents, and an agent comparing BMS's portfolio against BMS's portfolio
        correctly finds nothing missing. That is not a threshold being too strict;
        it is the question being asked in a circle. MEASURED on the first full demo
        run: 7 of 10 candidates for the org-bearing query were already BMS-owned,
        the remaining 3 sat inside the footprint, and the agent reported zero gaps.

        Spec 4.3 step 4 says the target is "the query's THERAPEUTIC AREA; retrieve
        its top-k technologies" -- the area, not the area-plus-the-company. This
        method is what makes that literal.

        Returns the query unchanged when nothing links or when stripping would
        leave nothing to retrieve on (a bare "Bristol Myers Squibb" has no topic;
        the caller then reuses the original candidates rather than retrieving on "").

        Cuts are made on the ORIGINAL string, never on a normalized copy. The result
        goes to the encoder, and normalize() folds case and drops punctuation --
        "PD-L1" would arrive as "pd l1", degrading the very retrieval this exists to
        improve. Only the org's own characters are removed; the rest survives byte
        for byte.
        """
        surfaces = []
        for nid in self._link_orgs(query):
            # The node_id suffix is itself a normalized surface, so it stands alone
            # if the row lookup misses (a linked id absent from the index).
            surfaces.append(entity_link.normalize(nid.split(":", 1)[1]))
            row = self.graph.org_row(nid) if self.graph else None
            if row is not None:
                surfaces.append(entity_link.normalize(self.graph.label(row)))
        # Longest first, so "bristol myers squibb" is consumed before "bristol".
        surfaces = sorted({s for s in surfaces if s}, key=len, reverse=True)
        if not surfaces:
            return query

        out = query
        for surf in surfaces:
            # The surface is normalized but the target is not, so match its tokens
            # across whatever separators the original used: "Bristol-Myers Squibb",
            # "Bristol_Myers Squibb" and "BRISTOL=MYERS SQUIBB" all appear in this
            # graph's own labels, and all must cut.
            pattern = r"[^A-Za-z0-9]+".join(re.escape(t) for t in surf.split())
            out = re.sub(rf"(?<![A-Za-z0-9]){pattern}(?![A-Za-z0-9])", " ", out,
                         flags=re.IGNORECASE)
        out = re.sub(r"\s+", " ", out).strip(" ,;:-")
        return out if out.strip() else query

    def should_fire(self, query: str) -> bool:
        """
        Routing predicate for the orchestrator: does this query name an org we
        know? entity_link's measured fire rate is ~45%, and the other ~55% is the
        agent correctly declining.

        Deliberately cheap and graph-free: it does not check whether the org has a
        usable portfolio. That check needs a traversal and belongs in run(), which
        abstains with a readable reason -- "fired, then abstained because the
        portfolio is too small" is a more honest trace than silently never firing.
        """
        return bool(self._link_orgs(query))

    # -- the portfolio ------------------------------------------------------ #
    def _resolve_portfolio(self, org_ids: list[str]) -> tuple[np.ndarray, list[dict], list[str]]:
        """
        Linked org node_ids -> (portfolio tech rows, per-org audit records,
        every org node_id absorbed).

        The union runs over CANONICAL VARIANTS, which is the whole point: the
        query "Bristol Myers Squibb" links one node owning 826 technologies, but
        the company's actual coverage is 3,155 across 5 name fragments
        ("bristol myers squibb", "... co", "... company", "bristol_myers squibb
        company", "... inc"). Reporting 826 would understate the portfolio by 74%
        and every gap downstream would inherit the error. Subsidiaries are NOT
        merged -- Juno and Adnexus are separate legal entities.
        """
        tech_rows: list[np.ndarray] = []
        audit: list[dict] = []
        absorbed: set[str] = set()
        for oid in sorted(org_ids):                     # deterministic
            row = self.graph.org_row(oid)
            if row is None:
                continue
            variants = [self.graph.node_id(int(r))
                        for r in self.graph.canonical_org_rows(row)]
            portfolio = self.graph.canonical_portfolio(row)
            absorbed.update(variants)
            tech_rows.append(portfolio)
            audit.append({
                "queried_node": oid,
                "canonical_key": self.graph.canonical_key(row),
                "variants_merged": sorted(variants),
                "n_variants": len(variants),
                "techs_direct": int(len(self.graph.portfolio(row))),
                "techs_after_merge": int(len(portfolio)),
            })
        if not tech_rows:
            return np.zeros(0, dtype=np.int32), audit, sorted(absorbed)
        return np.unique(np.concatenate(tech_rows)), audit, sorted(absorbed)

    def _vectors_for_techs(self, tech_rows: np.ndarray) -> tuple[np.ndarray, list[int]]:
        """
        Technology rows -> (normalized vectors, the rows that had one).

        Rows are gathered sorted so the mmap read is forward-only, and only these
        rows are ever materialized. Measured: 100% of the graph's 600,738
        technology nodes have a doc vector, so the miss branch is defensive rather
        than load-bearing -- but a silent misalignment here would corrupt every
        downstream number, so it is checked rather than assumed.
        """
        pairs = []
        for t in tech_rows:
            doc_id = self.graph.doc_id_of_tech(int(t))
            emb_row = self._row_of_doc_id.get(doc_id) if doc_id else None
            if emb_row is not None:
                pairs.append((emb_row, int(t)))
        if not pairs:
            return np.zeros((0, self.doc_vectors.shape[1]), dtype=np.float32), []
        pairs.sort()
        emb_rows = [p[0] for p in pairs]
        vectors = np.asarray(self.doc_vectors[emb_rows], dtype=np.float32)
        return _l2_normalize(vectors), [p[1] for p in pairs]

    def _cluster(self, vectors: np.ndarray) -> np.ndarray:
        """
        KMeans the portfolio -> normalized centroids (the capability footprint).

        k follows the spec: min(8, max(2, n // 25)), clamped to n so that n < k
        cannot raise. random_state is config.SEED and n_init is pinned, because
        stage 09 verifies this pipeline is reproducible and an unpinned n_init is
        a silent source of run-to-run drift. Cost is bounded: the largest
        portfolio in the graph is 3,548 technologies and fits in ~1 s.
        """
        from sklearn.cluster import KMeans      # imported late: ~0.4 s of import

        n = len(vectors)
        k = min(_MAX_CLUSTERS, max(_MIN_CLUSTERS, n // _TECHS_PER_CLUSTER))
        k = max(1, min(k, n))
        km = KMeans(n_clusters=k, random_state=config.SEED, n_init=10).fit(vectors)
        return _l2_normalize(km.cluster_centers_.astype(np.float32))

    # -- who fills the gap -------------------------------------------------- #
    def _fillers(self, tech_row: int, excluded_org_rows: set[int]) -> dict:
        """
        Who covers a technology this org does not: the licensing lead.

        Three routes, in descending order of how much they can be trusted:

          owners     tech --assigned_to--> organization
                     The assignee. A hard fact and the only authoritative route.
          inventors  tech --invented_by--> inventor --affiliated_with--> org
                     Real, but the org half is degraded by name collision, so
                     collided inventor nodes are dropped (MAX_ORGS_PER_INVENTOR).
          experts    tech --investigated_by--> expert, then the expert's orgs via
                     expert <--investigated_by-- tech --assigned_to--> org.
                     SPEC BUG: spec 4.3 step 6 says `expert --affiliated_with-->
                     org`. That edge does not exist for any of the 213,687 expert
                     nodes, so the two-hop is the only route (GraphIndex.
                     orgs_of_expert implements it and documents that it resolves
                     for only ~13.4% of experts). It is also a WEAKER claim: it
                     means "investigated something that org owns" -- sponsorship
                     or collaboration -- NOT employment, and it is never rendered
                     as "works for".

        The queried org and all its canonical variants are excluded throughout: a
        company is not its own licensing lead.
        """
        g = self.graph

        owners = [int(o) for o in g.owners_of(tech_row) if int(o) not in excluded_org_rows]

        inventors: list[dict] = []
        for inv in g.inventors_of(tech_row):
            inv = int(inv)
            orgs = g.orgs_of_inventor(inv)
            collided = len(orgs) > MAX_ORGS_PER_INVENTOR
            employers = ([] if collided else
                         [int(o) for o in orgs if int(o) not in excluded_org_rows])
            inventors.append({
                "node_id": g.node_id(inv),
                "name": g.label(inv),
                "orgs": [g.label(o) for o in employers[:MAX_FILLERS_PER_KIND]],
                # Surfaced, not hidden: a dropped inventor is a data-quality fact
                # the reader is entitled to see.
                "affiliations_suppressed_name_collision": collided,
            })

        experts: list[dict] = []
        for exp in g.experts_of(tech_row):
            exp = int(exp)
            orgs = [int(o) for o in g.orgs_of_expert(exp) if int(o) not in excluded_org_rows]
            experts.append({
                "node_id": g.node_id(exp),
                "name": g.label(exp),
                # "investigated a technology this org owns", NOT "is employed by".
                "orgs_via_investigated_tech": [g.label(o) for o in orgs[:MAX_FILLERS_PER_KIND]],
            })

        return {
            "orgs": [{"node_id": g.node_id(o), "name": g.label(o)}
                     for o in owners[:MAX_FILLERS_PER_KIND]],
            "inventors": inventors[:MAX_FILLERS_PER_KIND],
            "experts": experts[:MAX_FILLERS_PER_KIND],
        }

    # -- confidence --------------------------------------------------------- #
    def _confidence(self, portfolio_size: int, target_coverage: float,
                    fillable: float) -> float:
        """
        A HEURISTIC AND UNCALIBRATED score in [0,1]. Read it as "how much evidence
        did the agent have", NOT as "how likely is this analysis correct". There is
        no eval for this agent (spec section 6), so a calibrated number is not
        available and inventing one would be the exact overclaim this repo exists
        to avoid. It is deliberately built only from things that were COUNTED:

          portfolio_evidence  min(1, n / 100)  -- a 5-technology footprint supports
                              a far weaker absence claim than a 500-technology one.
          target_coverage     fraction of target candidates that are technology
                              nodes in the graph. Targets outside the graph cannot
                              be checked for ownership or walked for fillers, so
                              the analysis is blind to them.
          fillable            fraction of gaps with >=1 filler. A gap nobody fills
                              is not a licensing lead. Vacuously 1.0 when there are
                              no gaps, since then there is nothing left to fill and
                              the conclusion rests on the two evidence terms.

        The 0.5/0.3/0.2 weights are HAND-SET and express a preference (portfolio
        size matters most), not a fit to any data. The result is capped at
        _CONFIDENCE_CEILING so that it can never render as certainty.
        """
        portfolio_evidence = min(1.0, portfolio_size / _PORTFOLIO_EVIDENCE_SATURATION)
        score = 0.5 * portfolio_evidence + 0.3 * target_coverage + 0.2 * fillable
        return round(float(min(_CONFIDENCE_CEILING, max(0.0, score))), 4)

    # -- narration ---------------------------------------------------------- #
    def _deterministic_narration(self, org_names: list[str], portfolio_size: int,
                                 n_clusters: int, gaps: list[dict],
                                 n_targets: int, n_owned: int) -> str:
        """
        The no-LLM path. A template, and the payload says so via
        narration_mode="deterministic" -- letting this read as model output is
        precisely the overclaim the LLM client's docstring warns about.
        """
        who = ", ".join(org_names) if org_names else "the linked organization"
        lines = [
            f"{who}: {portfolio_size} technologies in the merged graph, grouped "
            f"into {n_clusters} capability clusters.",
            f"Of {n_targets} candidate technologies in this field, {n_owned} are "
            f"already in the portfolio (assigned_to) and {len(gaps)} are not owned "
            f"by them -- those are the gaps, ordered by distance from the "
            f"capabilities they do have.",
        ]
        if not gaps:
            lines.append("Every candidate in this field is already in the portfolio, "
                         "so this analysis surfaces no gap. That is not proof of full "
                         "coverage: it is bounded by the candidates supplied.")
        for gap in gaps[:5]:
            fillers = gap["fillers"]
            owners = [o["name"] for o in fillers["orgs"]]
            who_fills = (f" Owned by: {', '.join(owners)}." if owners else
                         " No assignee in the graph covers it.")
            lines.append(
                f"- {gap['doc_id']} ({gap['distance']:.3f} cosine distance to the "
                f"nearest cluster): {gap['title'][:90]}.{who_fills}")
        lines.append("Generated WITHOUT an LLM (no DH_LLM_API_KEY set): this is a "
                     "deterministic template over graph traversal, not model prose.")
        return "\n".join(lines)

    def _llm_narration(self, query: str, org_names: list[str], portfolio_size: int,
                       gaps: list[dict]) -> tuple[str | None, list[str]]:
        """
        Narrate via glm-4.6. Returns (text, invalid_citations) or (None, []) when
        the LLM did not produce usable output -- the caller then falls back to the
        deterministic path and records that it did.

        Two hard rules, both enforced here rather than trusted to the prompt:

          * The model is handed the EXACT gap doc_ids and told to cite only those.
            Post-validation then strips any doc_id-shaped token that is not in the
            allowed set, because "cite only these" is a request, not a guarantee,
            and a fabricated doc_id in a scouting tool is the worst possible bug.
          * Empty content is a FAILURE, not an empty answer. glm-4.6 is a reasoning
            model and bills reasoning tokens against max_tokens: when the budget
            runs out mid-reasoning it returns HTTP 200, finish_reason="length" and
            content="" (see agents/LLM_CONTRACT.md). agents/llm.py exposes no way
            to pass thinking={"type":"disabled"}, so we budget max_tokens generously
            and treat empty as failure instead of narrating nothing.
        """
        allowed = [g["doc_id"] for g in gaps]
        facts = [
            f"Organization: {', '.join(org_names)}",
            f"Portfolio size (technologies assigned in the knowledge graph): {portfolio_size}",
            f"Query: {query}",
            "Coverage gaps (technologies far from every portfolio cluster):",
        ]
        for gap in gaps:
            owners = ", ".join(o["name"] for o in gap["fillers"]["orgs"]) or "none in graph"
            inv = ", ".join(i["name"] for i in gap["fillers"]["inventors"][:3]) or "none"
            facts.append(
                f"  - doc_id={gap['doc_id']} | title={gap['title'][:120]} | "
                f"cosine_distance_to_nearest_cluster={gap['distance']:.3f} | "
                f"assignees={owners} | inventors={inv}")
        messages = [
            {"role": "system",
             "content": ("You are a technology-scouting analyst. Write ONE short "
                         "paragraph (<=120 words) for a licensing team. Use ONLY "
                         "the facts given. Cite doc_ids in parentheses, and cite "
                         f"ONLY these exact doc_ids: {', '.join(allowed)}. Never "
                         "invent a doc_id, an organization, or a date. Do not give "
                         "clinical or dosing advice. State that a gap means 'no "
                         "similar technology is assigned to this organization in "
                         "our graph', not that the company lacks the capability.")},
            {"role": "user", "content": "\n".join(facts)},
        ]
        # max_tokens generous on purpose -- see the reasoning-budget trap above.
        text = self.llm.chat(messages, temperature=0.0, max_tokens=1200)
        if text is None or not text.strip():
            return None, []
        return _strip_invalid_citations(text.strip(), set(allowed))

    # -- the agent ---------------------------------------------------------- #
    @timed
    def run(self, query: str, ctx: dict) -> AgentResult:
        """
        ctx: {"candidates": [...], "gap_targets": [...] | None} from the orchestrator.

        TARGETS come from `gap_targets` when supplied, falling back to `candidates`.
        They differ, and the difference is the whole analysis: `candidates` answer
        the user's literal query (org included), while `gap_targets` are retrieved
        on the TOPIC ALONE (see topic_query) and are what spec 4.3 step 4 actually
        asks for. Scoring against `candidates` asks what BMS lacks using documents
        selected by the word "BMS" -- a circle that returns zero gaps by
        construction. The fallback is kept for callers that supply no target pool
        (and for the org-only query, which has no topic to retrieve on); it is the
        weaker mode and payload["target_pool"] says which one ran.

        This agent still never constructs a Retriever (a second one is ~20 GB) --
        the orchestrator owns retrieval and hands the pool in.
        """
        g = self.graph

        # 1. Link. No org -> abstain. This is ~55% of queries and it is the feature.
        org_ids = self._link_orgs(query)
        if not org_ids:
            return AgentResult.abstain(
                self.name,
                "no organization named in the query, so there is no portfolio to "
                "analyse; gap analysis needs a company to analyse the gaps OF")

        # 2. Portfolio = union over canonical variants.
        portfolio_rows, audit, absorbed = self._resolve_portfolio(org_ids)
        org_names = [g.label(int(g.org_row(o))) for o in org_ids if g.org_row(o) is not None]

        if len(portfolio_rows) < MIN_PORTFOLIO_FOR_GAP:
            return AgentResult.abstain(
                self.name,
                f"linked {', '.join(org_ids)} but it owns only "
                f"{len(portfolio_rows)} technologies in the graph (minimum "
                f"{MIN_PORTFOLIO_FOR_GAP}); a capability footprint built from that "
                f"is sparsity, not evidence, and any 'gap' found against it would "
                f"be an artifact")

        # 3. Capability footprint.
        portfolio_vecs, kept_rows = self._vectors_for_techs(portfolio_rows)
        if len(portfolio_vecs) < MIN_PORTFOLIO_FOR_GAP:
            return AgentResult.abstain(
                self.name,
                f"only {len(portfolio_vecs)} of {len(portfolio_rows)} portfolio "
                f"technologies have document embeddings; too few to cluster")
        centroids = self._cluster(portfolio_vecs)

        # 4. Targets: the topic-only pool when the orchestrator retrieved one,
        #    else the literal query's candidates (weaker -- see run's docstring).
        targets = ctx.get("gap_targets")
        target_pool = "topic" if targets else "query"
        candidates = targets or ctx.get("candidates") or []
        if not candidates:
            return AgentResult.abstain(
                self.name,
                f"linked {', '.join(org_names)} ({len(portfolio_rows)} technologies) "
                f"but the retrieval agent supplied no candidate technologies to "
                f"compare the portfolio against")

        # 5. Gap test -- but ownership is checked FIRST and wins.
        portfolio_set = {int(r) for r in portfolio_rows}
        excluded_org_rows = {int(r) for oid in org_ids
                             if (r0 := g.org_row(oid)) is not None
                             for r in g.canonical_org_rows(r0)}

        scored, n_owned, n_no_vector, n_off_graph = [], 0, 0, 0
        for cand in candidates:
            doc_id = cand.get("doc_id")
            if not doc_id:
                continue
            tech_row = g.tech_row_of_doc_id(doc_id)
            if tech_row is None:
                # In the corpus but not a graph technology node (2,631 such docs).
                n_off_graph += 1
                continue
            if tech_row in portfolio_set:
                # THE FIX: an assigned_to edge is a fact; a centroid is a lossy
                # summary. Measured, this rescues uspto:US6183734B1 (0.358 vs BMS's
                # centroids, i.e. "gap") which is assigned to BRISTOL MYERS SQUIBB CO.
                n_owned += 1
                continue
            emb_row = self._row_of_doc_id.get(doc_id)
            if emb_row is None:
                n_no_vector += 1
                continue
            scored.append((doc_id, tech_row, emb_row, cand))

        if not scored:
            return AgentResult.abstain(
                self.name,
                f"linked {', '.join(org_names)} ({len(portfolio_rows)} technologies) "
                f"but none of the {len(candidates)} candidates are analysable: "
                f"{n_owned} already in the portfolio, {n_off_graph} absent from the "
                f"graph, {n_no_vector} without an embedding")

        target_vecs = _l2_normalize(
            np.asarray(self.doc_vectors[[s[2] for s in scored]], dtype=np.float32))
        sims_to_centroids = target_vecs @ centroids.T            # (n_targets, k)
        best_centroid = sims_to_centroids.max(axis=1)
        # Second opinion: the single most similar document they actually own. A
        # centroid is a lossy summary of a diffuse portfolio; this is not, and it
        # lets a reader sanity-check any gap claim the centroid rule makes.
        nearest_doc = (target_vecs @ portfolio_vecs.T).max(axis=1)

        # 6. Walk the graph for whoever does cover each gap.
        #
        # Every `scored` target is ALREADY a gap: the owned ones were removed above by
        # exact assigned_to lookup, which is the gate. The cosine only ORDERS what
        # comes back -- see GAP_COSINE_THRESHOLD for the measurement that demoted it.
        # The threshold survives as an opt-in filter (default 0.0 = keep everything);
        # at any non-zero value it discards real gaps at the rate measured there.
        gaps = []
        for i, (doc_id, tech_row, _emb_row, cand) in enumerate(scored):
            if GAP_COSINE_THRESHOLD > 0.0 and best_centroid[i] >= GAP_COSINE_THRESHOLD:
                continue
            gaps.append({
                "doc_id": doc_id,
                "title": cand.get("title") or g.label(tech_row),
                # cosine DISTANCE to the nearest capability cluster: 1 - cosine
                # similarity. Larger = further from anything they own = more of a
                # gap. Named explicitly because "distance" is ambiguous.
                "distance": round(float(1.0 - best_centroid[i]), 4),
                "nearest_centroid_cos": round(float(best_centroid[i]), 4),
                "nearest_portfolio_doc_cos": round(float(nearest_doc[i]), 4),
                "source_url": cand.get("source_url", ""),
                "fillers": self._fillers(tech_row, excluded_org_rows),
            })
        # Biggest gap first; ties broken by doc_id so the order is reproducible.
        gaps.sort(key=lambda x: (-x["distance"], x["doc_id"]))

        # 7. Narrate.
        n_clusters = int(len(centroids))
        narration_mode, invalid_citations, fallback_reason = "deterministic", [], None
        narration = None
        if self.llm is not None and self.llm.available and gaps:
            narration, invalid_citations = self._llm_narration(
                query, org_names, len(portfolio_rows), gaps[:MAX_FILLERS_PER_KIND])
            if narration is None:
                fallback_reason = "llm_call_failed_or_returned_empty"
            else:
                narration_mode = "llm"
        elif self.llm is None or not self.llm.available:
            fallback_reason = "no_llm_api_key"
        elif not gaps:
            fallback_reason = "no_gaps_to_narrate"
        if narration is None:
            narration = self._deterministic_narration(
                org_names, len(portfolio_rows), n_clusters, gaps, len(candidates), n_owned)

        fillable = (sum(1 for x in gaps if x["fillers"]["orgs"]
                        or x["fillers"]["inventors"] or x["fillers"]["experts"]) / len(gaps)
                    if gaps else 1.0)
        target_coverage = len(scored) / max(1, len(candidates) - n_owned)

        payload = {
            "org_linked": org_ids,
            "org_labels": org_names,
            "canonical_variants_merged": absorbed,
            "canonicalization_audit": audit,
            "portfolio_size": int(len(portfolio_rows)),
            "portfolio_embedded": int(len(portfolio_vecs)),
            "n_clusters": n_clusters,
            "gaps": gaps,
            "narration": narration,
            "narration_mode": narration_mode,
            "narration_fallback_reason": fallback_reason,
            "narration_invalid_citations": invalid_citations,
            "threshold": float(GAP_COSINE_THRESHOLD),
            # "topic" = targets retrieved on the query minus the org name (correct);
            # "query" = the literal query's candidates, which are biased toward the
            # org's own documents and understate gaps. Reported because the two
            # answer measurably different questions.
            "target_pool": target_pool,
            "n_targets": len(candidates),
            "n_targets_analysed": len(scored),
            "targets_already_owned": n_owned,
            "targets_not_in_graph": n_off_graph,
            "targets_without_embedding": n_no_vector,
            # Never removed, never conditional. There is no eval for this agent.
            "unmeasured": True,
            "method_note": (
                "Gap = cosine to the nearest KMeans centroid of the organization's "
                "assigned technologies < threshold. Technologies the organization "
                "already owns are excluded before this test, because an assigned_to "
                "edge is a fact and a centroid is a lossy summary. This method has "
                "NO evaluation: no judged gap-analysis ground truth exists. Treat "
                "gaps as leads to check, not findings."),
        }
        evidence = [make_evidence(x["doc_id"], x["title"], x.get("source_url", ""))
                    for x in gaps]
        return AgentResult(
            agent=self.name, ok=True, payload=payload, evidence=evidence,
            confidence=self._confidence(len(portfolio_rows), target_coverage, fillable),
            latency_ms=0.0,   # overwritten by @timed / run_agent
        )


def _strip_invalid_citations(text: str, allowed: set[str]) -> tuple[str, list[str]]:
    """
    Remove any doc_id-shaped token the model emitted that is not in the allowed
    set, and return what was removed.

    An LLM citing a doc_id that does not exist is the most damaging failure this
    tool has: it looks exactly like a real citation. The prompt asks for restraint;
    this enforces it. Tokens are matched conservatively -- the pattern is
    "<lowercase source>:<id>", which is the doc_id shape and is unlikely to occur
    in ordinary prose.
    """
    invalid = sorted({m for m in _DOC_ID_RE.findall(text) if m not in allowed})
    for bad in invalid:
        text = text.replace(bad, "[unverified citation removed]")
    return text, invalid


# --------------------------------------------------------------------------- #
# CLI: a smoke run against the real graph. Needs the graph index + doc vectors,
# and takes candidates from a JSON file so it never constructs a Retriever.
# --------------------------------------------------------------------------- #
def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Expertise-gap agent smoke run.")
    ap.add_argument("--query", required=True)
    ap.add_argument("--candidates", type=Path, default=None,
                    help="JSON list of {doc_id,title,source_url} (retrieval agent output)")
    args = ap.parse_args()

    t0 = time.perf_counter()
    agent = ExpertiseGapAgent.from_config(llm=LLMClient())
    print(f"loaded in {time.perf_counter() - t0:.1f}s", flush=True)
    print(f"should_fire({args.query!r}) = {agent.should_fire(args.query)}")

    candidates = json.loads(args.candidates.read_text()) if args.candidates else []
    result = agent.run(args.query, {"candidates": candidates})
    print(json.dumps({"ok": result.ok, "abstained": result.abstained,
                      "confidence": result.confidence,
                      "latency_ms": round(result.latency_ms, 1),
                      "payload": result.payload}, indent=2)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
