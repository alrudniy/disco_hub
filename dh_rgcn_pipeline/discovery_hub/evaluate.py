"""
Retrieval-quality evaluation: the metrics the deck actually sells (Recall@k,
MRR@10, nDCG@10, Precision@10, MAP), plus the statistics that make a number
defensible (95% bootstrap confidence intervals and a paired significance test
for A/B comparisons).

WHY THIS EXISTS
    Without this, the system is blind on its headline claim ("+20% Precision@10")
    and there is no way to tell whether the R-GCN graph signal or a fine-tuned
    encoder actually *helps* versus plain text retrieval. This module turns "we
    think it's better" into "it's +X% [CI a, b], p=...".

GROUND TRUTH WITHOUT HUMAN LABELS
    There are no labeled query->document pairs. The standard fix (InPars /
    Promptagator) is to generate each evaluation query *from* a known source
    document; that document is then the gold positive for the query. It is a
    proxy for true relevance -- the gold standard is human-judged qrels, which
    this module also supports via load_qrels(). build_synthetic_eval() is the
    deterministic, testable default so the harness runs in CI with no labels.

    Honesty caveat baked into the report: synthetic eval has ONE positive per
    query, so Precision@10 is capped at 0.1 and Recall@k / MRR are the metrics to
    read. Curated qrels with several positives per query make Precision@10
    meaningful.

Everything here is pure (numpy, with scipy only for the optional Wilcoxon test)
and deterministic given a seed, matching the rest of the pipeline.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np

from .config import SEED

# Default cutoffs. Recall is reported at several k; the "main" cutoff (10) drives
# Precision/nDCG/MRR to match the deck's Precision@10 / NDCG framing.
DEFAULT_KS = (1, 5, 10, 20, 50)
MAIN_K = 10

_TOKEN_RE = re.compile(r"[a-z]{4,}")  # content tokens: alphabetic, length >= 4


# --------------------------------------------------------------------------- #
# Per-query metrics (binary relevance)
# --------------------------------------------------------------------------- #
def evaluate_query(ranked_ids: list[str], rel_ids, ks=DEFAULT_KS,
                   main_k: int = MAIN_K) -> dict[str, float]:
    """
    Score one query. `ranked_ids` is the system's ranked doc_id list (best first);
    `rel_ids` is the set/list of relevant doc_ids. Returns a flat metric dict.
    """
    rel = set(rel_ids)
    R = len(rel)
    out: dict[str, float] = {}

    for k in ks:
        hits = sum(1 for d in ranked_ids[:k] if d in rel)
        out[f"recall@{k}"] = hits / R if R else 0.0

    topm = ranked_ids[:main_k]
    hits_m = sum(1 for d in topm if d in rel)
    out[f"precision@{main_k}"] = hits_m / main_k if main_k else 0.0

    # Reciprocal rank of the first relevant hit within main_k.
    rr = 0.0
    for i, d in enumerate(ranked_ids[:main_k], start=1):
        if d in rel:
            rr = 1.0 / i
            break
    out[f"mrr@{main_k}"] = rr

    # nDCG@main_k with binary gains.
    dcg = sum(1.0 / math.log2(i + 1)
              for i, d in enumerate(ranked_ids[:main_k], start=1) if d in rel)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(R, main_k) + 1))
    out[f"ndcg@{main_k}"] = (dcg / idcg) if idcg > 0 else 0.0

    # Truncated mean average precision over the returned list.
    if R:
        num, h = 0.0, 0
        for i, d in enumerate(ranked_ids, start=1):
            if d in rel:
                h += 1
                num += h / i
        out["map"] = num / R
    else:
        out["map"] = 0.0
    return out


def metric_names(ks=DEFAULT_KS, main_k: int = MAIN_K) -> list[str]:
    return ([f"recall@{k}" for k in ks]
            + [f"precision@{main_k}", f"mrr@{main_k}", f"ndcg@{main_k}", "map"])


def aggregate(per_query: list[dict[str, float]], ks=DEFAULT_KS,
              main_k: int = MAIN_K) -> dict[str, np.ndarray]:
    """Transpose a list of per-query metric dicts into metric -> value array."""
    names = metric_names(ks, main_k)
    return {m: np.array([q[m] for q in per_query], dtype=float) for m in names}


# --------------------------------------------------------------------------- #
# Statistics: bootstrap CIs + paired comparison
# --------------------------------------------------------------------------- #
def bootstrap_ci(values: np.ndarray, n_boot: int = 1000, ci: float = 0.95,
                 seed: int = SEED) -> tuple[float, float, float]:
    """Mean and percentile bootstrap CI over queries. Deterministic given seed."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    means = values[idx].mean(axis=1)
    lo = float(np.percentile(means, (1 - ci) / 2 * 100))
    hi = float(np.percentile(means, (1 + ci) / 2 * 100))
    return float(values.mean()), lo, hi


def paired_delta(a: np.ndarray, b: np.ndarray, n_boot: int = 1000,
                 ci: float = 0.95, seed: int = SEED) -> dict:
    """
    Compare two systems on the SAME queries (a = baseline, b = candidate).
    Returns the mean per-query delta (b - a), a bootstrap CI of that delta, and a
    Wilcoxon signed-rank p-value (paired, two-sided) when computable.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = b - a
    mean_d, lo, hi = bootstrap_ci(diff, n_boot=n_boot, ci=ci, seed=seed)

    p_value = None
    if diff.size >= 1 and np.any(diff != 0):
        try:
            from scipy.stats import wilcoxon
            p_value = float(wilcoxon(b, a, zero_method="wilcox").pvalue)
        except Exception:
            p_value = None
    elif diff.size and not np.any(diff != 0):
        p_value = 1.0  # identical on every query

    return {"mean_delta": mean_d, "ci_low": lo, "ci_high": hi,
            "p_value": p_value, "n": int(diff.size)}


# --------------------------------------------------------------------------- #
# Qrels I/O (the labeled set)
# --------------------------------------------------------------------------- #
@dataclass
class EvalQuery:
    query_id: str
    query: str
    relevant_doc_ids: list[str]


def load_qrels(path) -> list[EvalQuery]:
    """
    Load human-judged or pre-generated relevance judgments from JSONL. Each line:
      {"query_id": "...", "query": "...", "relevant_doc_ids": ["uspto:US...", ...]}
    This is the path you switch to once you have a curated evaluation set.
    """
    import json
    out: list[EvalQuery] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            out.append(EvalQuery(
                query_id=str(o["query_id"]),
                query=o["query"],
                relevant_doc_ids=list(o["relevant_doc_ids"]),
            ))
    return out


def write_qrels(queries: list[EvalQuery], path) -> int:
    """Persist an eval set to JSONL so it can be inspected or hand-corrected."""
    import json
    from dataclasses import asdict
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for q in queries:
            fh.write(json.dumps(asdict(q), ensure_ascii=False, sort_keys=True) + "\n")
    return len(queries)


# --------------------------------------------------------------------------- #
# Synthetic eval-set generation (the no-labels default)
# --------------------------------------------------------------------------- #
def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


_FRAMES = (
    "therapeutic candidate for",
    "novel approach to",
    "partner sought for",
    "platform addressing",
    "research interest in",
)


def build_synthetic_eval(docs, num_queries: int, seed: int = SEED) -> list[EvalQuery]:
    """
    Generate `num_queries` evaluation queries, each derived from one source
    document (its gold positive). To avoid a degenerate test, queries are built
    from each document's *discriminative-but-shared* terms -- tokens that occur in
    a few documents (2 <= df <= 30% of the corpus), not unique-to-the-source
    tokens (which would make every query trivially rank its source first). A
    generic "research interest" frame is prepended to cross register (the way a
    real pharma interest is phrased differently from a patent), exactly the
    asymmetry the encoder must bridge.

    Deterministic: source sampling and per-doc term selection are seeded.
    """
    docs = list(docs)
    n = len(docs)
    if n == 0:
        return []

    # Corpus document frequencies over content tokens.
    df: dict[str, int] = {}
    doc_tokens: list[set[str]] = []
    for d in docs:
        toks = set(_tokens(getattr(d, "embedding_text", "") or d.abstract or d.title))
        doc_tokens.append(toks)
        for t in toks:
            df[t] = df.get(t, 0) + 1

    hi_df = max(2, int(0.30 * n))   # exclude near-stopword tokens
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)[:min(num_queries, n)]

    queries: list[EvalQuery] = []
    for qi, di in enumerate(order):
        d = docs[di]
        toks = doc_tokens[di]
        # discriminative-but-shared candidates, then fallbacks
        cand = [t for t in toks if 2 <= df[t] <= hi_df]
        if len(cand) < 3:
            cand = [t for t in toks if df[t] <= hi_df] or sorted(toks)
        # rank by ascending df (most distinctive first), tie-break lexicographically
        cand = sorted(set(cand), key=lambda t: (df[t], t))[:5]
        frame = _FRAMES[int(di) % len(_FRAMES)]
        query = f"{frame} {' '.join(cand)}".strip()
        queries.append(EvalQuery(
            query_id=f"q{qi:04d}",
            query=query,
            relevant_doc_ids=[d.doc_id],
        ))
    return queries
