"""
dh2.candidate_pool -- P0/P1 candidate pooling + the RASC semantic-hedge union.

For each query, pool unique candidate documents from multiple retrieval channels so the
teacher cascade (or the inference cascade) can grade a rich set.

WHAT CHANGED (2026-07-15, RASC recommendation S1):

  1. The untouched 0.6B is now a FIRST-CLASS CHANNEL. The old pool drew from the
     fine-tuned 4B (top + deep window) and the fine-tuned 8B only -- i.e. it omitted the
     single model measured to be BEST on the low-overlap queries that are 76% of this
     corpus's true matches (LOW R@10: untouched 0.6B 0.2556 vs best trained arm 0.1667).
     Documents only that model could find were never in the pool, so no reranker could
     recover them and no judge could grade them.

  2. The cap is ROUND-ROBIN, not a global sort by dense_4b. The old cap sorted the
     non-positive remainder by `dense_4b if not None else dense_8b or -1.0`, which meant
     (a) 8B-only and semantic-only candidates were evicted first, and (b) uncalibrated
     dense scores from different models were compared directly. That is a structural
     thumb on the scale for the model family that learned the lexical shortcut. Retention
     now walks the channels in turn, so every channel keeps its own best candidates.

  3. Scores are NEVER averaged or compared across models. Per-channel rank and score are
     preserved as provenance, so the union's composition is auditable and "unique
     relevant contribution by channel" is measurable at eval time.

Channels (dense; no global BM25 -- lexical survives only as the routed exact-ID channel):
  * base_06b     untouched semantic model  -> protects low-overlap recall
  * ft_8b        fine-tuned 8B             -> strongest domain/capacity branch
  * ft_4b        fine-tuned 4B             -> optional diversity branch
  * ft_4b_deep   4B deep rank window       -> medium-hard negatives (TRAINING pools only)
  * reasonembed  optional reasoning branch
  * exact        routed identifier lookup  -> NCT/patent/CAS/compound only
The designated positive (from the source query record) is always injected and never
evicted.

Output record (candidate_pool_v*.jsonl), one per (query, candidate) -- a SUPERSET of the
old schema, so p0_2/p0_3/p1_1 and analysis/ keep working unchanged:
  {"query_id","query","document_id","channels":[...],"is_designated_positive":bool,
   "dense_4b":float|null,"dense_8b":float|null,"dense_semantic":float|null,
   "rank_4b":int|null,"rank_8b":int|null,"rank_semantic":int|null,
   "ranks":{channel:rank},"scores":{channel:score},"source":str}
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from dh2 import config2 as C
from dh2.retriever2 import DenseRetriever

# Canonical channel names. RETENTION_ORDER is the round-robin order, so the semantic
# hedge gets first refusal on every cap slot.
CH_SEMANTIC = "base_06b"
CH_FT_8B = "ft_8b"
CH_FT_4B = "ft_4b"
CH_FT_4B_DEEP = "ft_4b_deep"
CH_REASON = "reasonembed"
CH_EXACT = "exact"
CH_POSITIVE = "designated_positive"

RETENTION_ORDER = [CH_EXACT, CH_SEMANTIC, CH_FT_8B, CH_FT_4B, CH_REASON, CH_FT_4B_DEEP]

# Back-compat: the old writer used these channel labels in the `channels` list.
_LEGACY_ALIAS = {
    CH_FT_4B: "dense_4b_top",
    CH_FT_4B_DEEP: "dense_4b_deep",
    CH_FT_8B: "dense_8b_top",
}


@dataclass
class Candidate:
    """One pooled document with per-channel provenance. Scores are per-model and are
    never averaged across models -- see module docstring."""
    doc_id: str
    channels: set[str] = field(default_factory=set)
    ranks: dict[str, int] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)

    def add(self, channel: str, rank: int | None = None,
            score: float | None = None) -> None:
        self.channels.add(channel)
        if rank is not None:
            prev = self.ranks.get(channel)          # keep the best (lowest) rank
            self.ranks[channel] = rank if prev is None else min(prev, rank)
        if score is not None:
            self.scores[channel] = float(score)

    def best_rank(self) -> int:
        """Best rank across all channels -- the final ordering tiebreak."""
        return min(self.ranks.values()) if self.ranks else 10**9


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


def build_union(query: str,
                retrievers: dict[str, DenseRetriever | None],
                depths: dict[str, int],
                *,
                identifier_index: dict[str, list[str]] | None = None,
                deep_window: tuple[int, int] | None = None) -> dict[str, Candidate]:
    """Retrieve every channel and union the results, preserving channel provenance.

    An UNCAPPED union cannot have lower relevant-document recall than any constituent
    retriever -- that is the one guarantee this design rests on, and it holds only as
    long as nothing here silently drops candidates. Capping happens later, in cap_union(),
    and is round-robin so the guarantee degrades evenly instead of collapsing onto one
    model.
    """
    candidates: dict[str, Candidate] = {}

    def _add(doc_id: str, channel: str, rank: int | None, score: float | None):
        c = candidates.get(doc_id)
        if c is None:
            c = candidates[doc_id] = Candidate(doc_id)
        c.add(channel, rank, score)

    for channel, retr in retrievers.items():
        depth = depths.get(channel, 0)
        if retr is None or depth <= 0:
            continue
        for rank, (did, score) in enumerate(retr.search(query, top_k=depth), start=1):
            _add(did, channel, rank, score)

    # Deep rank window off the 4B: medium-hard negatives for TRAINING pools only. Not
    # part of the inference cascade -- it exists to give the teachers something to grade,
    # not to serve a scout.
    if deep_window and retrievers.get(CH_FT_4B) is not None:
        start, end = deep_window
        for offset, (did, score) in enumerate(
                retrievers[CH_FT_4B].search(query, top_k=end, window=(start, end))):
            _add(did, CH_FT_4B_DEEP, start + offset + 1, score)

    # Routed exact-identifier channel (no-op unless the query carries an identifier).
    if identifier_index:
        from dh2.identifiers import exact_identifier_search
        for rank, (did, score) in enumerate(
                exact_identifier_search(query, identifier_index), start=1):
            _add(did, CH_EXACT, rank, score)

    return candidates


def cap_union(candidates: dict[str, Candidate], max_union: int,
              protected: set[str] | None = None,
              order: list[str] | None = None) -> dict[str, Candidate]:
    """Round-robin retention across channels (replaces the 4B-score-biased global sort).

    Walks the channels in RETENTION_ORDER, taking each channel's next-best candidate in
    turn until the cap is hit. Each channel is ranked by ITS OWN rank -- no cross-model
    score comparison anywhere. `protected` (the designated positives) is always kept.
    """
    if max_union <= 0 or len(candidates) <= max_union:
        return candidates
    protected = protected or set()
    order = list(order or RETENTION_ORDER)

    kept: dict[str, Candidate] = {d: c for d, c in candidates.items() if d in protected}

    # Any channel not named in `order` (forward-compat) cycles last.
    seen_channels = {ch for c in candidates.values() for ch in c.channels}
    extra = [ch for ch in sorted(seen_channels)
             if ch not in order and ch != CH_POSITIVE]
    cycle = order + extra

    # Per-channel queues, each sorted by that channel's own rank.
    queues: dict[str, list[str]] = {}
    for ch in cycle:
        members = [d for d, c in candidates.items()
                   if ch in c.channels and d not in kept]
        members.sort(key=lambda d: candidates[d].ranks.get(ch, 10**9))
        queues[ch] = members

    cursors = {ch: 0 for ch in cycle}
    progressed = True
    while len(kept) < max_union and progressed:
        progressed = False
        for ch in cycle:
            if len(kept) >= max_union:
                break
            q = queues[ch]
            i = cursors[ch]
            while i < len(q) and q[i] in kept:   # already taken by an earlier channel
                i += 1
            if i < len(q):
                kept[q[i]] = candidates[q[i]]
                cursors[ch] = i + 1
                progressed = True
            else:
                cursors[ch] = i
    return kept


def _to_record(query_id: str, query: str, cand: Candidate, is_pos: bool,
               source: str) -> dict:
    """Flatten a Candidate to the output schema (superset of the legacy schema)."""
    legacy_channels = [_LEGACY_ALIAS.get(ch, ch) for ch in RETENTION_ORDER
                       if ch in cand.channels]
    legacy_channels += [ch for ch in sorted(cand.channels)
                        if ch not in RETENTION_ORDER and ch != CH_POSITIVE]
    if is_pos:
        legacy_channels.append(CH_POSITIVE)
    r4 = cand.ranks.get(CH_FT_4B, cand.ranks.get(CH_FT_4B_DEEP))
    s4 = cand.scores.get(CH_FT_4B, cand.scores.get(CH_FT_4B_DEEP))
    return {
        "query_id": query_id,
        "query": query,
        "document_id": cand.doc_id,
        "channels": legacy_channels,
        "is_designated_positive": is_pos,
        # flat per-model fields (legacy consumers: p0_2, p0_3, p1_1, analysis/)
        "dense_4b": s4,
        "dense_8b": cand.scores.get(CH_FT_8B),
        "dense_semantic": cand.scores.get(CH_SEMANTIC),
        "rank_4b": r4,
        "rank_8b": cand.ranks.get(CH_FT_8B),
        "rank_semantic": cand.ranks.get(CH_SEMANTIC),
        # full provenance
        "ranks": dict(cand.ranks),
        "scores": {k: round(v, 5) for k, v in cand.scores.items()},
        "source": source,
    }


def build_pool(query_records: list[dict],
               retr_4b: DenseRetriever,
               retr_8b: DenseRetriever | None,
               docs_source: dict[str, str] | None = None,
               pool_cfg: C.PoolConfig = C.POOL,
               retr_base_06b: DenseRetriever | None = None,
               retr_reasonembed: DenseRetriever | None = None,
               identifier_index: dict[str, list[str]] | None = None) -> Iterable[dict]:
    """Yield candidate-pool records. docs_source maps doc_id->source for the source field.

    Signature is backward compatible: the first five arguments are unchanged, so existing
    callers keep working. Pass `retr_base_06b` to enable the semantic hedge -- without it
    the pool reproduces the old, 4B-favoring composition and the low-overlap documents
    stay invisible.
    """
    retrievers: dict[str, DenseRetriever | None] = {
        CH_SEMANTIC: retr_base_06b,
        CH_FT_8B: retr_8b,
        CH_FT_4B: retr_4b,
        CH_REASON: retr_reasonembed,
    }
    depths = {
        CH_SEMANTIC: pool_cfg.per_source_topk if retr_base_06b is not None else 0,
        CH_FT_8B: pool_cfg.per_source_topk if retr_8b is not None else 0,
        CH_FT_4B: pool_cfg.per_source_topk,
        CH_REASON: pool_cfg.per_source_topk if retr_reasonembed is not None else 0,
    }

    for rec in query_records:
        q = rec["query"]
        pos_ids = set(rec["positive_ids"])

        cand = build_union(q, retrievers, depths,
                           identifier_index=identifier_index,
                           deep_window=(pool_cfg.deep_window_start,
                                        pool_cfg.deep_window_end))

        # always include the designated positive(s), even if no channel retrieved them
        for pid in pos_ids:
            if pid not in cand:
                cand[pid] = Candidate(pid)
            cand[pid].channels.add(CH_POSITIVE)

        # backfill the positive's 4B score so MarginMSE has a teacher-anchor score
        need = [pid for pid in pos_ids if CH_FT_4B not in cand[pid].scores]
        if need:
            for d, s in retr_4b.score_pairs(q, need).items():
                cand[d].scores[CH_FT_4B] = float(s)

        cand = cap_union(cand, pool_cfg.max_candidates, protected=pos_ids)

        for did, c in cand.items():
            yield _to_record(rec["query_id"], q, c, did in pos_ids,
                             (docs_source or {}).get(did, did.split(":", 1)[0]))
