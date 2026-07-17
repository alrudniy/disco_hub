#!/usr/bin/env python3
"""
07_retrieve_rank.py  --  LAYER 2 serving (and the seam to Layer 3).

The always-on retrieval service. WHAT ACTUALLY RUNS (verified 2026-07-17, and this
docstring previously described a system that does not exist -- it advertised "three
signals fused by RRF" while two of the three were switched off in config):

  LIVE:
  dense    : query embedding -> top_k by text cosine (FAISS exact)
  alias    : routed HGNC gene-alias OR-search (use_alias=True). Fires only when the
             query names a gene; no-op otherwise. See _alias_search.

  OFF, each for a measured reason -- read config.RetrievalConfig before re-enabling:
  keyword  : BM25. use_keyword=False. NOT because it is useless: on
             "HER2 targeted therapy for breast cancer" BM25 ranks
             clinicaltrials:NCT01779050 at 55 -- inside the pool -- while dense ranks
             it 152,142 despite the document containing the literal token "HER2". It
             is off because RRF(dense,keyword) evicted 578 relevant documents dense
             had already found to seat keyword's 24 unique ones at a 100-doc budget.
             config says re-open by RAISING THE POOL AND UNIONING, not by re-enabling
             RRF. The pool went to 100 on 2026-07-17, so that condition is now half
             met and this is a live question, not a closed one.
  graph    : entity-linked R-GCN. use_graph=False; measured unique reach of exactly
             zero relevant documents across 62 firing queries.

  fusion   : RRF over whichever signals fired (scale-free; a missing signal just
             isn't fused) -> candidate pool
  rerank   : cross-encoder (real) or fused-score sort (mock) -> top_k_rerank,
             each candidate carrying supporting evidence for Layer 3.

  TARGET: Drew (always-on). Every signal is deterministic (FAISS Flat, BM25, and
          a fixed graph dot product), so the retrieved set and its order are
          reproducible run-to-run -- which 09 verifies (Jaccard = 1.0, tau = 1.0).

Exposes a `Retriever` class imported by 08_multiagent_rag.py and 09_stability_harness.py.

Usage:
  python 07_retrieve_rank.py --mock --query "GLP-1 agonist for metabolic disease"
  python 07_retrieve_rank.py --mock --query "..." --no-keyword --no-graph
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys

import numpy as np

# Bound GPU memory fragmentation on small (12 GB) cards -- the reranker/embedder
# "reserved but unallocated" OOM. Set before torch/CUDA initializes; harmless else.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from discovery_hub import config, entity_link
from discovery_hub.determinism import set_global_determinism
from discovery_hub.embedding import get_embedder
from discovery_hub.fusion import fuse_to_pool
from discovery_hub.keyword import BM25Index
from discovery_hub.schema import read_docs


def _assert_serving_compat() -> None:
    """Startup check (spec S10.4), via dh2.validate -- which existed and was never called.

    Fails fast on the mismatches it CAN see: DH_EMBED_MODEL unset (an ambiguous encoder
    silently mismatching the indexed vectors), doc_ids vs vectors row-count drift, and
    docs.jsonl regenerated out of sync with the embeddings.

    BE CLEAR ABOUT ITS LIMIT. This would NOT have caught the stale-shard corruption that
    made 60,000 rows unretrievable: that bug preserves the row count, the row order and the
    dim exactly, and swaps only row CONTENT -- so every assertion here passes. Content
    provenance is enforced upstream, where it belongs: 04_generate_embeddings.py now binds
    each shard to the model that wrote it and refuses to reuse a foreign one. This check is
    the cheap startup net, not the fix; do not let its passing imply the index is sound.
    Mock mode is exempt -- it has no real encoder and no DH_EMBED_MODEL to check.
    """
    try:
        from dh2.validate import assert_serving_compat
    except ImportError:
        print("[07] dh2.validate unavailable; skipping startup compat check "
              "(PYTHONPATH should include the pipeline tree)")
        return
    facts = assert_serving_compat(
        embed_model=os.environ.get("DH_EMBED_MODEL") or getattr(config, "EMBED_MODEL", ""),
        doc_vectors_path=str(config.EMB_DIR / "doc_vectors.npy"),
        doc_ids_path=str(config.INDEX_DIR / "doc_ids.json"),
        docs_jsonl_path=str(config.NORM_DIR / "docs.jsonl"))
    print(f"[07] serving compat OK: {facts['vector_count']} vectors x "
          f"{facts['vector_dim']}d, model={facts['embed_model']}")


# Any title hit outranks any body-only hit. Sized to dominate IDF (max ~13 here), not tuned:
# the claim is ordinal ("titled after the alias beats mentions it"), not a weighting.
TITLE_BONUS = 100.0


class Retriever:
    def __init__(self, mock: bool = True, device: str | None = None):
        self.mock = mock
        self.device = device
        self._cross_encoder = None          # lazy-loaded once, not per query
        self._alias = None                  # lazy (AliasTable, postings)
        self.embedder = get_embedder(mock=mock, device=device)
        # mmap_mode="r": doc_vectors.npy is 6.17 GB (603369 x 2560 float32). Loading it
        # resident put a real run at ~21 GB anon-rss on a 30 GB box and it was OOM-killed
        # mid-demo (verified: kernel oom-kill of this process at anon-rss 20.9 GB, during
        # the third scripted query). When faiss.index is present -- it is, 6.17 GB, and
        # faiss-cpu is installed -- self._faiss serves every dense search, and this array
        # is only ever indexed ONE ROW AT A TIME to backfill text_score (see _rerank).
        # Single-row reads are exactly what mmap is good at, so this costs nothing here
        # and returns ~6 GB of RAM. The brute-force fallback at _text_search (used only if
        # faiss is unavailable) still works against the mmap -- it just streams from disk.
        self.doc_vectors = np.load(config.EMB_DIR / "doc_vectors.npy", mmap_mode="r")
        self.doc_ids = json.loads((config.INDEX_DIR / "doc_ids.json").read_text())
        if not mock:
            _assert_serving_compat()
        # doc_id -> row in doc_vectors, so any candidate's true dense cosine can be read
        # back (used to backfill text_score for keyword/graph-sourced candidates).
        self._docid_to_row = {did: i for i, did in enumerate(self.doc_ids)}
        self.docs = {d.doc_id: d for d in read_docs(config.NORM_DIR / "docs.jsonl")}

        # Dense index (FAISS exact if available, else numpy brute force).
        self._faiss = None
        faiss_path = config.INDEX_DIR / "faiss.index"
        if faiss_path.exists():
            try:
                import faiss
                self._faiss = faiss.read_index(str(faiss_path))
            except ImportError:
                self._faiss = None

        # BM25 keyword index: load the one built by stage 05, else build from docs
        # (defensive so 07 works even if 05 wasn't re-run).
        bm25_path = config.INDEX_DIR / "bm25.json"
        if bm25_path.exists():
            self.bm25 = BM25Index.load(bm25_path)
        else:
            ordered = [self.docs[i] for i in self.doc_ids if i in self.docs]
            self.bm25 = BM25Index.build(ordered, k1=config.RETRIEVAL.bm25_k1,
                                        b=config.RETRIEVAL.bm25_b)

        # Graph: R-GCN node embeddings + node bookkeeping for the entity-linked
        # query signal. We need (a) row of every node, (b) a surface-form index of
        # entity nodes for linking, (c) the technology rows aligned to doc_ids so
        # the graph can be searched.
        self.graph_emb = None
        self.node_grow: dict[str, int] = {}      # node_id -> row
        self.surface_index: dict[str, set[str]] = {}
        self.docid_to_grow: dict[str, int] = {}   # doc_id -> tech node row
        self._tech_rows = None
        self._tech_docids: list[str] = []
        emb_path = config.ARTIFACT_DIR / "rgcn_node_emb.npy"
        if emb_path.exists():
            self.graph_emb = np.load(emb_path)
            node_ids = json.loads((config.ARTIFACT_DIR / "node_ids.json").read_text())
            self.node_grow = {nid: i for i, nid in enumerate(node_ids)}
            nodes = [json.loads(l) for l in
                     (config.GRAPH_DIR / "nodes.jsonl").read_text().splitlines() if l]
            self.surface_index = entity_link.build_surface_index(nodes)
            for nd in nodes:
                did = nd.get("doc_id")
                if did and nd["node_id"] in self.node_grow:
                    self.docid_to_grow[did] = self.node_grow[nd["node_id"]]
            if self.docid_to_grow:
                self._tech_docids = list(self.docid_to_grow.keys())
                self._tech_rows = np.array(
                    [self.docid_to_grow[d] for d in self._tech_docids], dtype=np.int64)

    # -- individual signals --------------------------------------------------
    def _text_search(self, qvec: np.ndarray, k: int):
        if self._faiss is not None:
            scores, idxs = self._faiss.search(qvec.reshape(1, -1).astype(np.float32), k)
            return idxs[0], scores[0]
        sims = self.doc_vectors @ qvec
        idxs = np.argsort(-sims)[:k]
        return idxs, sims[idxs]

    def _keyword_search(self, query: str, k: int) -> list[tuple[str, float]]:
        return self.bm25.score(query, top_k=k)

    def _graph_query_vector(self, query: str):
        """Build a graph-space query vector from the query's named entities.
        Returns (unit_vector | None, linked_node_ids). None == abstain."""
        linked = entity_link.link_query(query, self.surface_index)
        rows = [self.node_grow[nid] for nid in linked if nid in self.node_grow]
        if not rows:
            return None, linked
        v = self.graph_emb[rows].mean(0)
        return (v / (np.linalg.norm(v) + 1e-9)).astype(np.float32), linked

    def _graph_search(self, qgvec: np.ndarray, k: int) -> list[tuple[str, float]]:
        """Rank technologies by structural proximity to the query's entities.
        O(num_docs) dot product -- fine at MVP scale; at production scale this
        wants an ANN index over the graph embeddings."""
        scores = self.graph_emb[self._tech_rows] @ qgvec
        order = sorted(range(len(scores)),
                       key=lambda j: (-scores[j], self._tech_docids[j]))[:k]
        return [(self._tech_docids[j], float(scores[j])) for j in order]

    # -- routed gene-alias channel (RASC stage 1's missing sibling) ----------
    #
    # WHY. Measured on the deployed index: for "Monoclonal antibody targeting PD-L1",
    # every patent that SAYS "PD-L1" ranks 1-21, while US6803192B1 -- "B7-H1, a novel
    # immunoregulatory molecule", the same molecule under its older name -- ranks 11,324
    # of 603,369. The corpus is not missing the art; it is missing the ALIAS. This is
    # upstream of the confidence gate: the gate never gets a chance to suppress the patent
    # because retrieval never surfaces it.
    #
    # dh2.identifiers routes exact identifiers because "no dense model reliably returns a
    # literal ID string", and it EXPLICITLY refuses gene names ("rejects the things that
    # look like codes but aren't -- gene names (CD274, PD-1)"). That refusal is right for
    # an identifier channel and leaves gene aliases handled by nothing. This is the
    # sibling channel for that gap: routed, lexical, deterministic, and a no-op on a query
    # that names no gene.
    #
    # PD-L1 = CD274 = B7-H1 = PDCD1LG1 is a published HGNC fact, not a learned one. An
    # analyst can read the table and check it; they cannot audit a learned alias. Every
    # candidate this channel returns carries the HGNC id that licensed it.
    def _alias_index(self):
        """Lazy (AliasTable, postings). Returns (None, None) if unavailable -> no-op."""
        if self._alias is not None:
            return self._alias
        hgnc = os.environ.get("DH_HGNC_TSV", "")
        post = os.environ.get("DH_ALIAS_POSTINGS", "")
        if not hgnc and not post:
            self._alias = (None, None)          # not configured: silent, correct no-op
            return self._alias
        # CONFIGURED BUT BROKEN != NOT CONFIGURED. Silence here cost a whole freeze run:
        # the channel was switched on, its artifacts were on another host, `_alias_index`
        # returned (None, None), and the run looked exactly like "no gene in the query" --
        # right down to producing a clean, plausible, wrong table. If someone asked for
        # this channel, say so when it cannot start.
        missing = [n for n, v in (("DH_HGNC_TSV", hgnc), ("DH_ALIAS_POSTINGS", post))
                   if not v or not os.path.exists(v)]
        if missing:
            print(f"[07] WARNING: alias channel REQUESTED but cannot start -- missing/"
                  f"unreadable: {missing}. Retrieval continues WITHOUT it; any result that "
                  f"depends on gene-alias expansion is invalid.", flush=True)
            self._alias = (None, None)
            return self._alias
        try:
            import gzip
            sys.path.insert(0, os.environ.get("DH_CITEPROV", "/home/alex/citation_provenance"))
            from citeprov.aliases import AliasTable
            tbl = AliasTable.from_hgnc_tsv(hgnc)
            with gzip.open(post, "rt") as fh:
                postings = json.load(fh)
            print(f"[07] alias channel ACTIVE: {len(tbl.genes)} genes, "
                  f"{len(postings.get('title_postings', {}))} title / "
                  f"{len(postings.get('body_postings', {}))} body surface forms")
            self._alias = (tbl, postings)
        except Exception as e:                                   # noqa: BLE001
            print(f"[07] alias channel unavailable ({e}); continuing without it")
            self._alias = (None, None)
        return self._alias

    def _alias_search(self, query: str, k: int):
        """Routed OR-search over the aliases the scout did NOT type.

        Searching the alias the query already contains would flood the channel with the
        documents dense already ranks 1-21 (measured: "PDL1" has document frequency 1,687).
        The channel exists for what the encoder MISSES -- the vocabulary the scout did not
        type -- so that is all it searches. Ranked by summed IDF of the distinct rare
        aliases a document contains; ties by doc_id, so the order is deterministic.
        """
        tbl, postings = self._alias_index()
        if tbl is None:
            return [], []
        exps = tbl.expand(query)
        if not exps:
            return [], []                        # no gene named -> channel abstains
        from citeprov.aliases import _norm
        typed = {_norm(m) for m in re.findall(r"[A-Za-z][A-Za-z0-9]*(?:[-/][A-Za-z0-9]+)*",
                                              query)}
        terms = {_norm(s) for e in exps for s in e.gene.all_symbols} - typed
        n_docs = len(self.doc_ids)
        title_p = postings.get("title_postings", {})
        body_p = postings.get("body_postings", {})
        # TITLE POSITION IS THE SIGNAL, and it is the honest one for this channel.
        #
        # Ranking purely by summed IDF put documents that ENUMERATE synonyms -- "Expression
        # of CD274 (PD-L1) and CD276 in B-cell Malignancies" -- above US6803192B1, whose
        # 2004 text only ever says "B7-H1". The discovery patent came 36th in its own
        # channel and never reached the context (measured).
        #
        # A document TITLED after the alias is ABOUT that molecule: "B7-H1, a novel
        # immunoregulatory molecule" is the subject, not a passing mention. Only 22 of
        # 603,369 documents are titled with B7-H1 at all. Title position is deterministic,
        # classic, and legible to an analyst -- no tuning, no learning.
        #
        # WHAT THIS CHANNEL IS, EXACTLY: a FILTER, not a ranker. It scores the ALIAS MATCH,
        # not the document -- so all 22 B7-H1-titled patents score identically (109.75,
        # measured) and their order is an alphabetical tiebreak on doc_id. That is not a
        # defect to tune away; it is the design. The channel's job is to establish
        # CANDIDACY for documents the encoder's vocabulary cannot reach. Deciding which of
        # 22 genuine B7-H1 patents a scout should read is a RANKING problem, and this
        # channel has no opinion on it by construction. Something else has to rank -- and
        # the incumbent something (BGE) is the component measured to swing 4.12x on wording,
        # which is why that is named as the next work item rather than papered over here.
        scores: dict[str, float] = {}
        for t in terms:
            t_docs, b_docs = title_p.get(t) or [], body_p.get(t) or []
            df = len(t_docs) + len(b_docs)
            if not df:
                continue
            idf = math.log(1 + n_docs / (1 + df))
            for d in t_docs:
                scores[d] = scores.get(d, 0.0) + idf + TITLE_BONUS
            for d in b_docs:
                scores[d] = scores.get(d, 0.0) + idf
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        prov = [{"matched": e.matched, "symbol": e.gene.symbol, "hgnc_id": e.gene.hgnc_id,
                 "searched": sorted(terms)} for e in exps]
        return ranked, prov

    # -- hybrid retrieve -----------------------------------------------------
    def retrieve(self, query: str, top_k: int | None = None,
                 rerank_k: int | None = None, *,
                 use_keyword: bool | None = None,
                 use_graph: bool | None = None,
                 use_alias: bool | None = None) -> list[dict]:
        top_k = top_k or config.RETRIEVAL.top_k_recall
        rerank_k = rerank_k or config.RETRIEVAL.top_k_rerank
        use_keyword = config.RETRIEVAL.use_keyword if use_keyword is None else use_keyword
        use_graph = config.RETRIEVAL.use_graph if use_graph is None else use_graph
        use_alias = config.RETRIEVAL.use_alias if use_alias is None else use_alias

        qvec = self.embedder.encode_queries([query])[0]
        lists: dict[str, list[str]] = {}

        d_idx, d_sc = self._text_search(qvec, top_k)
        lists["dense"] = [self.doc_ids[i] for i in d_idx]
        dense_map = {self.doc_ids[i]: float(d_sc[r]) for r, i in enumerate(d_idx)}

        kw_map: dict[str, float] = {}
        if use_keyword and self.bm25 is not None:
            kw = self._keyword_search(query, top_k)
            lists["keyword"] = [d for d, _ in kw]
            kw_map = dict(kw)

        graph_map: dict[str, float] = {}
        linked: list[str] = []
        if use_graph and self.graph_emb is not None and self._tech_rows is not None:
            qg, linked = self._graph_query_vector(query)
            if qg is not None:
                gl = self._graph_search(qg, top_k)
                lists["graph"] = [d for d, _ in gl]
                graph_map = dict(gl)
            # else: graph abstains (its list is simply absent from the fusion)

        alias_map: dict[str, float] = {}
        alias_prov: list[dict] = []
        if use_alias:
            al, alias_prov = self._alias_search(query, top_k)
            if al:
                lists["alias"] = [d for d, _ in al]
                alias_map = dict(al)
            # else: the channel abstains -- its list is simply absent from the fusion.

        pool = fuse_to_pool(lists, top_k, k=config.RETRIEVAL.rrf_k)
        # Tag every candidate with the channel(s) that actually surfaced it. Provenance is
        # per-candidate and observed, never a configured list (spec section 3, section 7).
        chan_of = {d: [n for n, lst in lists.items() if d in set(lst)] for d, _ in pool}
        cands = [{"doc_id": did, "score": fscore,
                  "text_score": dense_map.get(did, 0.0),
                  "keyword_score": kw_map.get(did, 0.0),
                  "graph_score": graph_map.get(did, 0.0),
                  "alias_score": alias_map.get(did, 0.0),
                  "channels": chan_of.get(did, []),
                  "alias_provenance": alias_prov if "alias" in chan_of.get(did, []) else [],
                  "graph_linked": bool(linked)}
                 for did, fscore in pool]

        cands = self._rerank(query, cands)
        cands = self._apply_routed_policy(cands, rerank_k)
        # Backfill the true dense cosine for candidates that surfaced via keyword/graph
        # (and so were absent from the dense top-k `dense_map`, leaving text_score=0.0).
        # text_score is the calibrated [0,1] semantic-relevance signal that downstream
        # confidence reads, so every returned candidate must carry its real cosine, not
        # 0.0 just because dense wasn't the arm that retrieved it. Vectors are unit-norm
        # (verified), so dot product == cosine.
        for c in cands:
            if not c.get("text_score"):
                row = self._docid_to_row.get(c["doc_id"])
                if row is not None:
                    c["text_score"] = float(self.doc_vectors[row] @ qvec)
        for c in cands:
            d = self.docs.get(c["doc_id"])
            if d:
                c["title"] = d.title
                c["abstract"] = d.abstract
                c["source"] = d.source
                c["source_url"] = d.source_url
                c["organizations"] = d.organizations
        return cands

    def _apply_routed_policy(self, cands: list[dict], rerank_k: int) -> list[dict]:
        """Guarantee routed-alias hits a place in the context, and exempt them from the gate.

        WHY A ROUTED HIT MUST NOT BE RANKED BY THE CROSS-ENCODER. The reranker is BGE, and
        BGE is measured, over 55,605 adjudicated pairs, to correlate +0.468 with query/
        document WORD OVERLAP and only +0.211 with usefulness -- at judged grade 3 its score
        swings 4.12x on wording alone. The routed alias channel exists precisely to surface
        documents whose WORDING DIFFERS from the query. So handing its hits to BGE asks the
        one component that penalises vocabulary distance to judge the one channel built to
        cross it. It reliably says no: for US6803192B1 against "Human PD-L1 antibodies" BGE
        scores 0.0004, and 0.032 even when the alias is written into the document's own text
        ("B7-H1" -> "B7-H1 (CD274 / PD-L1)") -- both far under the 0.35 gate, and NOT a
        truncation artifact (the record is 166 tokens against a 512 window).
        Two independent measurements, no new experiment needed.

        This is not special pleading; it is the precedent dh2.identifiers already sets. A
        scout who types NCT04381936 gets that trial, and no one would let a cross-encoder
        veto it. A scout who types PD-L1 is, per HGNC, also typing B7-H1.

        WHAT THIS DOES NOT DO. It does not reorder BGE's judgement of the dense pool, and it
        does not invent a score. Reserved hits keep their real rerank_score for display; they
        are marked `routed_protected` so Layer 3 gates them on evidence (the HGNC record)
        rather than on a similarity. At most `alias_reserved_slots` positions are displaced,
        and only the WEAKEST BGE candidates are displaced.
        """
        ranked = sorted(cands,
                        key=lambda c: (-(c.get("rerank_score") if c.get("rerank_score")
                                         is not None else c["score"]), c["doc_id"]))
        n_slots = getattr(config.RETRIEVAL, "alias_reserved_slots", 3)
        routed = [c for c in ranked if "alias" in (c.get("channels") or [])]
        reserved = sorted(routed, key=lambda c: (-c.get("alias_score", 0.0),
                                                 c["doc_id"]))[:n_slots]
        for c in reserved:
            c["routed_protected"] = True
            # Layer 3 reads this INSTEAD of the cross-encoder score. The basis is the HGNC
            # record that surfaced it -- a fact an analyst can check, not a similarity.
            c["confidence_basis"] = "routed_alias_hgnc"

        final = ranked[:rerank_k]
        have = {c["doc_id"] for c in final}
        missing = [c for c in reserved if c["doc_id"] not in have]
        if missing:
            # Displace the weakest BGE candidates that are not themselves reserved.
            keep = [c for c in final if c.get("routed_protected")]
            rest = [c for c in final if not c.get("routed_protected")]
            rest = rest[:max(0, rerank_k - len(keep) - len(missing))]
            final = ranked[:0] + keep + rest + missing
            # Re-order by BGE for display, protected hits included at their true score.
            final = sorted(final,
                           key=lambda c: (-(c.get("rerank_score") if c.get("rerank_score")
                                            is not None else c["score"]), c["doc_id"]))
        return final[:rerank_k]

    def _get_cross_encoder(self):
        """Load the cross-encoder ONCE and cache it. Previously it was constructed
        per query, reloading ~560M weights every call (slow + fragmented GPU mem).
        max_length caps input so a long patent/trial abstract can't OOM a batch."""
        if self._cross_encoder is None:
            from sentence_transformers import CrossEncoder
            max_len = getattr(config.RETRIEVAL, "rerank_max_length", 512)
            self._cross_encoder = CrossEncoder(
                config.RERANK_MODEL, max_length=max_len, device=self.device)
        return self._cross_encoder

    def _rerank(self, query: str, cands: list[dict]) -> list[dict]:
        if self.mock:
            # Deterministic: fused score, ties broken by doc_id.
            return sorted(cands, key=lambda c: (-c["score"], c["doc_id"]))
        ce = self._get_cross_encoder()
        batch_size = getattr(config.RETRIEVAL, "rerank_batch_size", 16)
        pairs = [(query, self.docs[c["doc_id"]].embedding_text) for c in cands]
        rr = ce.predict(pairs, batch_size=batch_size)
        for c, s in zip(cands, rr):
            c["rerank_score"] = float(s)
        return sorted(cands, key=lambda c: (-c["rerank_score"], c["doc_id"]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--query", required=True)
    ap.add_argument("--rerank-k", type=int, default=config.RETRIEVAL.top_k_rerank)
    ap.add_argument("--no-keyword", action="store_true", help="disable BM25 half")
    ap.add_argument("--no-graph", action="store_true", help="disable graph signal")
    args = ap.parse_args()

    set_global_determinism(config.SEED)
    r = Retriever(mock=args.mock)
    results = r.retrieve(args.query, rerank_k=args.rerank_k,
                         use_keyword=not args.no_keyword,
                         use_graph=not args.no_graph)
    linked = results[0].get("graph_linked") if results else False
    print(f"\nQuery: {args.query}")
    print(f"(graph signal: {'fired' if linked else 'abstained -- no entity linked'})")
    print(f"Top {len(results)} candidates:\n")
    for i, c in enumerate(results, 1):
        print(f"{i:2d}. [{c['score']:.4f}] {c.get('title','')[:68]}")
        print(f"     {c['doc_id']}  (text={c['text_score']:.3f} "
              f"bm25={c['keyword_score']:.2f} graph={c['graph_score']:.3f})  "
              f"{c.get('source_url','')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
