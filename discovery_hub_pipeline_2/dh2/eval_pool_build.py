"""
dh2.eval_pool_build -- P0: turn the graded teacher/LLM labels into the two evaluation
qrels the spec requires, plus leakage guards.

Outputs:
  qrels_exact_origin_v1.jsonl   -- {query_id, query, relevant_doc_ids:[designated positive(s)]}
                                   (continuity with the pipeline_1 benchmark)
  qrels_scout_utility_v1.jsonl  -- graded: {query_id, query, grades:{doc_id:0..3}}
                                   used for graded nDCG@10, utility-Recall (2-3 relevant),
                                   grade-3 Recall, MRR-to-first-grade-3.

Leakage guards (spec S10.3):
  * eval queries are a HELD-OUT set (disjoint query_ids from training labels)
  * family/cluster collapse: if a family_or_cluster_id is provided, only one member per
    cluster is kept as relevant to prevent duplicate-inflation.

This module consumes the merged-label JSONL produced by stage p0_3 and the query text
from the pool, and is pure I/O + grouping (no models) so it runs anywhere.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def _family_of(doc_id: str, families: dict[str, str] | None) -> str:
    if families and doc_id in families:
        return families[doc_id]
    return doc_id  # its own cluster


def build_qrels(labels_path: str | Path, out_exact: str | Path, out_utility: str | Path,
                query_text: dict[str, str] | None = None,
                families: dict[str, str] | None = None,
                held_out_query_ids: set[str] | None = None) -> dict:
    """Read merged labels; write exact-origin + graded utility qrels. Returns a summary."""
    labels_path = Path(labels_path)
    by_q_exact: dict[str, dict] = {}
    by_q_grades: dict[str, dict[str, int]] = defaultdict(dict)
    q_text: dict[str, str] = dict(query_text or {})
    seen_clusters: dict[str, set] = defaultdict(set)  # query_id -> set(cluster) already counted

    n_labels = 0
    for line in labels_path.open():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        n_labels += 1
        qid = r["query_id"]
        if held_out_query_ids is not None and qid not in held_out_query_ids:
            continue
        q_text.setdefault(qid, r.get("query", ""))
        did = r["document_id"]
        grade = int(r.get("grade", 0))
        if r.get("masked"):
            continue  # masked labels don't count in evaluation

        # family/cluster collapse for relevant grades
        cluster = _family_of(did, families)
        if grade >= 1 and cluster in seen_clusters[qid]:
            # keep the higher grade for the cluster, skip duplicates
            continue
        if grade >= 1:
            seen_clusters[qid].add(cluster)

        by_q_grades[qid][did] = max(by_q_grades[qid].get(did, 0), grade)
        if r.get("is_designated_positive"):
            e = by_q_exact.setdefault(qid, {"query_id": qid, "query": q_text.get(qid, ""),
                                            "relevant_doc_ids": []})
            if did not in e["relevant_doc_ids"]:
                e["relevant_doc_ids"].append(did)

    # write exact-origin
    Path(out_exact).parent.mkdir(parents=True, exist_ok=True)
    with Path(out_exact).open("w") as f:
        for qid in sorted(by_q_exact):
            f.write(json.dumps(by_q_exact[qid]) + "\n")

    # write graded utility
    n_util = 0
    with Path(out_utility).open("w") as f:
        for qid in sorted(by_q_grades):
            grades = {d: g for d, g in by_q_grades[qid].items() if g > 0}
            if not grades:
                continue
            rec = {"query_id": qid, "query": q_text.get(qid, ""), "grades": grades}
            f.write(json.dumps(rec) + "\n")
            n_util += 1

    # BUG #8 FIX. This used to be:
    #     sum(len(v) for v in by_q_grades.values()) / len(by_q_grades)
    # which counts every CANDIDATE, grade-0 included -- the `if g > 0` filter is applied
    # only when writing. It reported ~50 and that number became the previous handoff's
    # "~50 relevant docs/query". The real median is 1. A single stale mean sent the whole
    # project's mental model of its own label density off by ~50x, so both numbers are
    # now reported, explicitly named, alongside the median (means are useless on this
    # distribution: 71.6% of queries have no verdict at all, so it is two datasets
    # stacked, not one dense one).
    rel_counts = [sum(1 for g in v.values() if g > 0) for v in by_q_grades.values()]
    cand_counts = [len(v) for v in by_q_grades.values()]

    def _median(xs: list[int]) -> float:
        if not xs:
            return 0.0
        s = sorted(xs)
        mid = len(s) // 2
        return float(s[mid]) if len(s) % 2 else (s[mid - 1] + s[mid]) / 2.0

    n_q = max(len(by_q_grades), 1)
    return {
        "labels_read": n_labels,
        "exact_origin_queries": len(by_q_exact),
        "utility_queries": n_util,
        # relevant == grade >= 1, matching what is actually written to the utility qrels
        "mean_relevant_per_utility_query": round(sum(rel_counts) / n_q, 2),
        "median_relevant_per_utility_query": _median(rel_counts),
        # candidates == every graded row, grade-0 included (the old, mislabeled number)
        "mean_candidates_per_query": round(sum(cand_counts) / n_q, 2),
        "median_candidates_per_query": _median(cand_counts),
        "queries_with_zero_relevant": sum(1 for c in rel_counts if c == 0),
    }


def split_held_out(query_ids: list[str], eval_frac: float = 0.15,
                   seed: int = 20240611) -> tuple[set, set]:
    """Deterministic held-out split by hashing query_id (stable across runs)."""
    import hashlib
    held = set()
    for qid in query_ids:
        h = int(hashlib.sha256(f"{seed}:{qid}".encode()).hexdigest(), 16)
        if (h % 10000) / 10000.0 < eval_frac:
            held.add(qid)
    train = set(query_ids) - held
    return train, held
