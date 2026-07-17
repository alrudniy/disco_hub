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


def _bootstrap_ci(per_query: list[float], n_boot: int = 1000, seed: int = 20240611):
    if not per_query:
        return (0.0, 0.0, 0.0)
    arr = np.asarray(per_query)
    rng = np.random.default_rng(seed)
    means = [arr[rng.integers(0, len(arr), len(arr))].mean() for _ in range(n_boot)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return (float(arr.mean()), float(lo), float(hi))


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
