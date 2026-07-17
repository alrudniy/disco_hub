"""
dh2.candidate_pool -- P0/P1 candidate pooling.

For each query, pool unique candidate documents from multiple retrieval channels so the
teacher cascade can grade a rich set (designated positive + likely additional positives
+ medium-hard + easy + cross-source negatives). Spec: 64-128 unique candidates/query.

Channels (all dense, no static BM25 fusion per the BM25-hurts finding):
  * 4B top-k              (the deployed model)
  * 8B top-k              (the ceiling model, if available)
  * 4B deep rank window   [deep_window_start, deep_window_end)  -> medium-hard negatives
The designated positive (from the source query record) is always injected.

Output record (candidate_pool_v*.jsonl), one per (query, candidate):
  {"query_id","query","document_id","channels":[...],"is_designated_positive":bool,
   "dense_4b":float|null,"dense_8b":float|null,"source":str}
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from dh2 import config2 as C
from dh2.retriever2 import DenseRetriever


def _load_query_records(path: Path, max_queries: int = 0) -> list[dict]:
    """Read synthetic-query records. Accepts the pipeline_1 formats:
       {"query","positive"} (train triples) or {"query_id","query","relevant_doc_ids"}
       (eval qrels). Normalizes to {"query_id","query","positive_ids":[...]}.
    """
    rows = []
    for i, line in enumerate(path.open()):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        qid = r.get("query_id") or f"q-{i:07d}"
        q = r.get("query") or r.get("q") or ""
        pos = r.get("positive_ids")
        if pos is None:
            if "positive" in r:
                pos = [r["positive"]]
            elif "relevant_doc_ids" in r:
                pos = list(r["relevant_doc_ids"])
            else:
                pos = []
        rows.append({"query_id": qid, "query": q, "positive_ids": pos})
        if max_queries and len(rows) >= max_queries:
            break
    return rows


def build_pool(query_records: list[dict],
               retr_4b: DenseRetriever,
               retr_8b: DenseRetriever | None,
               docs_source: dict[str, str] | None = None,
               pool_cfg: C.PoolConfig = C.POOL) -> Iterable[dict]:
    """Yield candidate-pool records. docs_source maps doc_id->source for the source field."""
    for rec in query_records:
        q = rec["query"]
        pos_ids = set(rec["positive_ids"])
        cand: dict[str, dict] = {}

        def _add(doc_id: str, channel: str, score: float | None, which: str):
            c = cand.setdefault(doc_id, {"channels": [], "dense_4b": None, "dense_8b": None})
            if channel not in c["channels"]:
                c["channels"].append(channel)
            if which == "4b" and score is not None:
                c["dense_4b"] = score
            if which == "8b" and score is not None:
                c["dense_8b"] = score

        for did, sc in retr_4b.search(q, top_k=pool_cfg.per_source_topk):
            _add(did, "dense_4b_top", sc, "4b")
        for did, sc in retr_4b.search(q, top_k=pool_cfg.deep_window_end,
                                      window=(pool_cfg.deep_window_start,
                                              pool_cfg.deep_window_end)):
            _add(did, "dense_4b_deep", sc, "4b")
        if retr_8b is not None:
            for did, sc in retr_8b.search(q, top_k=pool_cfg.per_source_topk):
                _add(did, "dense_8b_top", sc, "8b")

        # always include the designated positive(s)
        for pid in pos_ids:
            _add(pid, "designated_positive", None, "pos")
        # backfill the positive's dense score if the model has the vector
        need = [pid for pid in pos_ids if cand[pid]["dense_4b"] is None]
        if need:
            for d, s in retr_4b.score_pairs(q, need).items():
                cand[d]["dense_4b"] = s

        # cap to max_candidates, but never drop a designated positive
        items = list(cand.items())
        if len(items) > pool_cfg.max_candidates:
            keep = [(d, c) for d, c in items if d in pos_ids]
            rest = [(d, c) for d, c in items if d not in pos_ids]
            rest.sort(key=lambda kv: (kv[1]["dense_4b"] if kv[1]["dense_4b"] is not None
                                      else kv[1]["dense_8b"] or -1.0), reverse=True)
            items = keep + rest[: pool_cfg.max_candidates - len(keep)]

        for did, c in items:
            yield {
                "query_id": rec["query_id"],
                "query": q,
                "document_id": did,
                "channels": c["channels"],
                "is_designated_positive": did in pos_ids,
                "dense_4b": c["dense_4b"],
                "dense_8b": c["dense_8b"],
                "source": (docs_source or {}).get(did, did.split(":", 1)[0]),
            }
