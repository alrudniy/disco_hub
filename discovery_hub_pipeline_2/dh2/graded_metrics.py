"""
dh2.graded_metrics -- graded scout-utility metrics (spec P0/S5.1) with bootstrap CIs.

Consumes graded qrels ({query_id, grades:{doc_id:0..3}}) and a ranking function that,
given a query, returns an ordered list of doc_ids. Reports:
  * graded nDCG@10
  * utility Recall@10 (grades 2-3 count as relevant)
  * grade-3 Recall@10
  * MRR to first grade-3 document
plus exact-origin Recall@1/10 + MRR@10 when an exact-origin qrels is supplied.

Bootstrap 95% CIs over queries. Pure numpy; no torch.
"""
from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np


def _dcg(gains: list[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranked_ids: list[str], grades: dict[str, int], k: int = 10) -> float:
    gains = [grades.get(d, 0) for d in ranked_ids[:k]]
    dcg = _dcg([float(g) for g in gains])
    ideal = sorted(grades.values(), reverse=True)[:k]
    idcg = _dcg([float(g) for g in ideal])
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranked_ids: list[str], relevant: set[str], k: int = 10) -> float:
    if not relevant:
        return 0.0
    hit = sum(1 for d in ranked_ids[:k] if d in relevant)
    return hit / len(relevant)


def mrr_to_first(ranked_ids: list[str], targets: set[str], k: int = 10) -> float:
    for i, d in enumerate(ranked_ids[:k]):
        if d in targets:
            return 1.0 / (i + 1)
    return 0.0


def success_at_k(ranked_ids: list[str], relevant: set[str], k: int = 10) -> float:
    """1.0 if AT LEAST ONE relevant doc appears in the top k, else 0.0.

    This is the metric that matches what a scout experiences. Recall@k with
    len(relevant)==1 is a trap on this benchmark: surface ten genuinely relevant patents,
    rank the designated origin doc 11th, and Recall@10 reports 0.0 -- a total failure for
    a result page the scout would have been delighted with.
    """
    if not relevant:
        return 0.0
    return 1.0 if any(d in relevant for d in ranked_ids[:k]) else 0.0


def _bootstrap_ci(per_query: list[float], n_boot: int = 1000, seed: int = 20240611):
    if not per_query:
        return (0.0, 0.0, 0.0)
    arr = np.asarray(per_query)
    rng = np.random.default_rng(seed)
    means = [arr[rng.integers(0, len(arr), len(arr))].mean() for _ in range(n_boot)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return (float(arr.mean()), float(lo), float(hi))


# --------------------------------------------------------------------------- #
# Register (word-overlap) bucketing -- the aggregate hides the trade, so never
# report an aggregate without these two buckets beside it.
# --------------------------------------------------------------------------- #
_STOP = set(
    "the a an of for and or to in with as by from thereof use uses using method methods "
    "composition compositions available licensing novel new therapy treatment".split()
)


def _toks(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", (s or "").lower())
            if len(w) > 2 and w not in _STOP}


def word_overlap(query: str, doc_text: str) -> float:
    """Fraction of the query's content words present in the doc. Identical definition to
    analysis/register_gap_analysis.py -- keep them in sync or the buckets stop comparing."""
    q = _toks(query)
    return len(q & _toks(doc_text)) / len(q) if q else 0.0


def bucket_queries(queries: dict[str, str], relevant_text: dict[str, str],
                   split: float | None = None) -> tuple[dict[str, float], float]:
    """query_id -> overlap(query, its best-known relevant doc); plus the split point.

    Default split is the MEDIAN overlap over the evaluated queries (0.333 on the 440
    held-out set), which is what the 2026-07-15 analysis used. Passing a fixed split lets
    you compare across eval sets.
    """
    ov = {qid: word_overlap(q, relevant_text.get(qid, "")) for qid, q in queries.items()}
    if split is None:
        split = float(np.median(list(ov.values()))) if ov else 0.0
    return ov, split


def evaluate_graded(utility_qrels_path: str | Path, rank_fn: Callable[[str], list[str]],
                    k: int = 10) -> dict:
    """rank_fn(query) -> ordered doc_ids. Returns graded metrics with CIs."""
    ndcgs, urec, g3rec, g3mrr = [], [], [], []
    for line in Path(utility_qrels_path).open():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        grades = {d: int(g) for d, g in r["grades"].items()}
        ranked = rank_fn(r["query"])
        rel_23 = {d for d, g in grades.items() if g >= 2}
        rel_3 = {d for d, g in grades.items() if g >= 3}
        ndcgs.append(ndcg_at_k(ranked, grades, k))
        urec.append(recall_at_k(ranked, rel_23, k))
        g3rec.append(recall_at_k(ranked, rel_3, k))
        g3mrr.append(mrr_to_first(ranked, rel_3, k))

    def ci(name, xs):
        m, lo, hi = _bootstrap_ci(xs)
        return {name: {"mean": round(m, 4), "ci": [round(lo, 4), round(hi, 4)], "n": len(xs)}}

    out = {}
    out.update(ci(f"ndcg@{k}", ndcgs))
    out.update(ci(f"utility_recall@{k}", urec))
    out.update(ci(f"grade3_recall@{k}", g3rec))
    out.update(ci(f"grade3_mrr@{k}", g3mrr))
    return out


def evaluate_cascade(utility_qrels_path: str | Path,
                     rank_fn: Callable[[str], list[str]],
                     *,
                     doc_text: dict[str, str] | None = None,
                     doc_source: dict[str, str] | None = None,
                     candidate_fn: Callable[[str], list[str]] | None = None,
                     llm_judged_query_ids: set[str] | None = None,
                     overlap_split: float | None = None,
                     k: int = 10,
                     recall_k: int = 100,
                     latency_ms: dict[str, float] | None = None) -> dict:
    """The RASC evaluation (recommendation "Evaluation and promotion gates").

    Deliberately NOT built around exact-origin R@10. That benchmark measures paraphrase
    retrieval: designated positives carry ~+0.100 more query-word overlap than equally
    relevant alternatives found by the judge (paired within query, [+0.085, +0.115] SIG),
    so optimizing it rewards the lexical shortcut the cascade exists to undo.

    Reports, per the gate table:
      * candidate Recall@100        did ANY retriever find a useful result?
      * Success@5 / Success@10      did the scout get >=1 grade-2/3 result?
      * nDCG@10                     are the useful results ordered first?
      * grade-3 MRR                 how fast does the best actionable result appear?
      * LOW/HIGH overlap slices     did the register gap improve, and did direct-match
                                    ability survive? NEVER report the aggregate alone --
                                    +0.061 aggregate concealed +0.165/-0.089.
      * per-source                  patents / trials / publications / SBIR
      * p50/p95 latency             is the cascade usable?

    `llm_judged_query_ids` restricts scoring to independently adjudicated rows (excluding
    automatic teacher-consensus labels), which is slice 1 of the anti-circularity design.
    """
    rows = []
    for line in Path(utility_qrels_path).open():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if llm_judged_query_ids is not None and r["query_id"] not in llm_judged_query_ids:
            continue
        rows.append(r)
    if not rows:
        return {"error": "no evaluable queries", "n": 0}

    # register bucketing: overlap between the query and its best-graded relevant doc
    queries = {r["query_id"]: r["query"] for r in rows}
    best_rel_text: dict[str, str] = {}
    if doc_text:
        for r in rows:
            grades = {d: int(g) for d, g in r["grades"].items()}
            best = max(grades, key=lambda d: grades[d], default=None)
            best_rel_text[r["query_id"]] = doc_text.get(best, "") if best else ""
    ov, split = bucket_queries(queries, best_rel_text, overlap_split)

    per: dict[str, list[float]] = defaultdict(list)
    by_bucket: dict[str, dict[str, list[float]]] = {"low": defaultdict(list),
                                                    "high": defaultdict(list)}
    by_source: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    channel_unique: dict[str, int] = defaultdict(int)
    per_query: list[dict] = []

    for r in rows:
        qid, q = r["query_id"], r["query"]
        grades = {d: int(g) for d, g in r["grades"].items()}
        rel_23 = {d for d, g in grades.items() if g >= 2}
        rel_3 = {d for d, g in grades.items() if g >= 3}
        ranked = rank_fn(q)

        m = {
            f"ndcg@{k}": ndcg_at_k(ranked, grades, k),
            "success@5": success_at_k(ranked, rel_23, 5),
            f"success@{k}": success_at_k(ranked, rel_23, k),
            f"utility_recall@{k}": recall_at_k(ranked, rel_23, k),
            f"grade3_recall@{k}": recall_at_k(ranked, rel_3, k),
            f"grade3_mrr@{k}": mrr_to_first(ranked, rel_3, k),
        }
        if candidate_fn is not None:
            pool = candidate_fn(q)
            # BUG A FIX: measure candidate recall over the FULL union depth, not a fixed
            # rank-100 cutoff. candidate_fn returns the union in insertion order (base-0.6B
            # channel first), so a fixed @100 window sees only the first channel's 100 docs
            # and reports its recall -- not the union's. The RASC guarantee ("an uncapped
            # union cannot have lower recall than any constituent") is about union
            # membership, so we score the whole returned pool. For single-retriever arms
            # (C0/C1) pool is already the top-100, so this is a no-op there.
            depth = max(recall_k, len(pool))
            m[f"candidate_recall@{recall_k}"] = recall_at_k(pool, rel_23, depth)
            m[f"candidate_success@{recall_k}"] = success_at_k(pool, rel_23, depth)
            # Reranker ceiling: best nDCG@k achievable if the pool were ordered perfectly.
            # oracle - actual = the headroom the ranking stage is leaving on the table.
            m[f"oracle_ndcg@{k}"] = oracle_utility(pool, grades, k)

        for name, v in m.items():
            per[name].append(v)
        bucket = "low" if ov.get(qid, 0.0) < split else "high"
        for name, v in m.items():
            by_bucket[bucket][name].append(v)

        # Keep the per-query row. Without it, `ov[qid]` and this query's metrics die here
        # and ANY re-derivation -- a different split, a regression of nDCG on overlap, a
        # per-source breakdown -- costs a full GPU re-run of retrieval. The bucketing above
        # is a THRESHOLD on a continuum: 9.2% of queries change side when the tokenizer is
        # fixed, and moving the threshold only moves which ones. The honest instrument is
        # a regression (nDCG_i ~ a + b*overlap_i; b = how much the arm leans on lexical
        # overlap), and it needs exactly these two columns. They cost bytes.
        per_query.append({
            "query_id": qid,
            "overlap": round(float(ov.get(qid, 0.0)), 6),
            "bucket": bucket,
            "n_relevant": len(rel_23),
            "n_grade3": len(rel_3),
            **{name: round(float(v), 6) for name, v in m.items()},
        })

        if doc_source:
            for src in {doc_source.get(d, "unknown") for d in rel_23}:
                src_rel = {d for d in rel_23 if doc_source.get(d, "unknown") == src}
                by_source[src][f"success@{k}"].append(success_at_k(ranked, src_rel, k))
                by_source[src][f"ndcg@{k}"].append(
                    ndcg_at_k(ranked, {d: grades[d] for d in src_rel}, k))

    def _ci(xs):
        m_, lo, hi = _bootstrap_ci(xs)
        return {"mean": round(m_, 4), "ci": [round(lo, 4), round(hi, 4)], "n": len(xs)}

    out: dict = {
        "n_queries": len(rows),
        "overlap_split": round(split, 4),
        # Provenance, so a reader never has to reverse-engineer the split from a coincidence:
        # `split` is np.median(overlap) over the EVALUATED queries unless one is passed in.
        # It is a library default, not a derived constant. The pinned 0.375 is simply the
        # median of the 123-query llm-judged slice under the old tokenizer.
        "overlap_split_source": "caller-pinned" if overlap_split is not None else "np.median(evaluated)",
        "overall": {name: _ci(xs) for name, xs in per.items()},
        "low_overlap": {name: _ci(xs) for name, xs in by_bucket["low"].items()},
        "high_overlap": {name: _ci(xs) for name, xs in by_bucket["high"].items()},
        # The raw rows. Every re-derivation (re-split, regression, per-source cut) is now
        # free and needs no GPU. This is the difference between one re-run and every
        # future re-run.
        "per_query": per_query,
    }
    out["bucket_sizes"] = {"low": len(by_bucket["low"].get(f"ndcg@{k}", [])),
                           "high": len(by_bucket["high"].get(f"ndcg@{k}", []))}
    if doc_source:
        out["per_source"] = {src: {n: _ci(xs) for n, xs in d.items()}
                             for src, d in by_source.items()}
    if channel_unique:
        out["unique_relevant_by_channel"] = dict(channel_unique)
    if latency_ms:
        vals = sorted(latency_ms.values())
        if vals:
            out["latency_ms"] = {
                "p50": round(float(np.percentile(vals, 50)), 1),
                "p95": round(float(np.percentile(vals, 95)), 1),
                "n": len(vals),
            }
    return out


def unique_relevant_contribution(candidates_by_channel: dict[str, list[str]],
                                 relevant: set[str]) -> dict[str, int]:
    """Per channel: how many relevant docs did ONLY that channel contribute?

    This is the question that decides whether a channel earns its GPU. A channel that
    finds 80 relevant documents but shares all 80 with another channel is redundant; a
    channel that finds 3 nobody else found is why the union exists.
    """
    out: dict[str, int] = {}
    for ch, docs in candidates_by_channel.items():
        others = {d for c, ds in candidates_by_channel.items() if c != ch for d in ds}
        out[ch] = len({d for d in docs if d in relevant and d not in others})
    return out


def oracle_utility(candidate_ids: list[str], grades: dict[str, int], k: int = 10) -> float:
    """Best achievable nDCG@k if the reranker ordered the candidate pool perfectly.

    The ceiling on everything downstream of retrieval. If oracle nDCG is 0.55, no
    reranker -- Qwen, GroupRank, or otherwise -- can do better, and the work belongs in
    the retrieval union, not the ranking stage.
    """
    present = {d: g for d, g in grades.items() if d in set(candidate_ids)}
    ranked = sorted(present, key=lambda d: -present[d])
    return ndcg_at_k(ranked, grades, k)


def evaluate_exact(exact_qrels_path: str | Path, rank_fn: Callable[[str], list[str]],
                   ks=(1, 10)) -> dict:
    """Exact-origin Recall@1/10 + MRR@10 (continuity benchmark)."""
    per = {f"recall@{k}": [] for k in ks}
    mrr10 = []
    for line in Path(exact_qrels_path).open():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        rel = set(r["relevant_doc_ids"])
        ranked = rank_fn(r["query"])
        for k in ks:
            per[f"recall@{k}"].append(recall_at_k(ranked, rel, k))
        mrr10.append(mrr_to_first(ranked, rel, 10))
    out = {}
    for k in ks:
        m, lo, hi = _bootstrap_ci(per[f"recall@{k}"])
        out[f"recall@{k}"] = {"mean": round(m, 4), "ci": [round(lo, 4), round(hi, 4)]}
    m, lo, hi = _bootstrap_ci(mrr10)
    out["mrr@10"] = {"mean": round(m, 4), "ci": [round(lo, 4), round(hi, 4)]}
    return out
