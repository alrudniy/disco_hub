"""
Link-prediction evaluation for the knowledge graph (the metric that tells you
whether the R-GCN learned anything).

A graph embedding is only useful if structurally-connected nodes end up closer
than random pairs. This module measures exactly that on HELD-OUT edges:

    roc_auc   : P(score(real edge) > score(random non-edge)). 0.5 = no signal.
    hits@k    : for each held-out edge (u,v), rank the true v against k_neg
                sampled candidates; fraction where v lands in the top k.
    mrr       : mean reciprocal rank of the true v in that candidate list.

The scoring function is the dot product of the two endpoint embeddings, matching
how 07_retrieve_rank.py uses the graph embeddings. Everything is pure numpy
(scipy only for tie-aware ranking) and deterministic given a seed, so it is
unit-tested in tests/test_graph_eval.py and runs in CI with no GPU.

split_edges() produces the train/val/test partition. The held-out edges are
removed from message passing so the evaluation has no leakage.
"""
from __future__ import annotations

import numpy as np

from .config import SEED


# --------------------------------------------------------------------------- #
# Edge splitting
# --------------------------------------------------------------------------- #
def split_edges(edges, val_frac: float = 0.1, test_frac: float = 0.1,
                seed: int = SEED):
    """
    Deterministically partition edges into (message_passing, val, test).

    The val/test edges are held out from message passing entirely, so node
    embeddings never see them -- that is what makes link-prediction on them an
    honest generalization test rather than memorization.
    """
    edges = list(edges)
    m = len(edges)
    if m == 0:
        return [], [], []
    rng = np.random.default_rng(seed)
    perm = rng.permutation(m)
    n_test = int(m * test_frac)
    n_val = int(m * val_frac)
    test_idx = set(perm[:n_test].tolist())
    val_idx = set(perm[n_test:n_test + n_val].tolist())
    mp, val, test = [], [], []
    for i, e in enumerate(edges):
        if i in test_idx:
            test.append(e)
        elif i in val_idx:
            val.append(e)
        else:
            mp.append(e)
    return mp, val, test


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def roc_auc(pos_scores, neg_scores) -> float:
    """
    Rank-based ROC-AUC (Mann-Whitney U), tie-aware. Equivalent to the probability
    that a random positive scores above a random negative.
    """
    pos = np.asarray(pos_scores, dtype=float)
    neg = np.asarray(neg_scores, dtype=float)
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    from scipy.stats import rankdata
    ranks = rankdata(np.concatenate([pos, neg]))  # average ranks for ties
    n_pos, n_neg = pos.size, neg.size
    return float((ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def sample_negatives(num: int, n_nodes: int, pos_set: set, rng,
                     max_tries_factor: int = 20) -> list[tuple[int, int]]:
    """Uniformly sample `num` directed non-edges (u != v, (u,v) not a real edge)."""
    negs: list[tuple[int, int]] = []
    cap = num * max_tries_factor + 100
    tries = 0
    while len(negs) < num and tries < cap:
        u = int(rng.integers(0, n_nodes))
        v = int(rng.integers(0, n_nodes))
        tries += 1
        if u != v and (u, v) not in pos_set:
            negs.append((u, v))
    return negs


def evaluate_link_prediction(Z: np.ndarray, pos_pairs, pos_set: set,
                             seed: int = SEED, k: int = 10,
                             ranking_negs: int = 50) -> dict:
    """
    Score held-out positive edges and report AUC + ranking metrics.

    Z          : [num_nodes, dim] node embeddings (rows aligned to node index).
    pos_pairs  : iterable of (u, v) integer node indices -- the held-out edges.
    pos_set    : set of ALL real (u, v) edges, so negatives avoid true edges.
    """
    pos_pairs = [(int(u), int(v)) for u, v in pos_pairs]
    if not pos_pairs:
        return {"num_eval": 0, "roc_auc": float("nan"),
                f"hits@{k}": float("nan"), "mrr": float("nan"),
                "ranking_negatives": ranking_negs}
    rng = np.random.default_rng(seed)
    n = Z.shape[0]

    su = np.array([u for u, _ in pos_pairs])
    sv = np.array([v for _, v in pos_pairs])
    pos_scores = np.sum(Z[su] * Z[sv], axis=1)

    neg = sample_negatives(len(pos_pairs), n, pos_set, rng)
    nu = np.array([u for u, _ in neg]) if neg else np.array([], dtype=int)
    nv = np.array([v for _, v in neg]) if neg else np.array([], dtype=int)
    neg_scores = np.sum(Z[nu] * Z[nv], axis=1) if neg else np.array([])
    auc = roc_auc(pos_scores, neg_scores)

    hits, rr, cnt = 0, 0.0, 0
    for (u, v) in pos_pairs:
        cand = [v]
        t = 0
        while len(cand) < ranking_negs + 1 and t < (ranking_negs + 1) * 20:
            w = int(rng.integers(0, n))
            t += 1
            if w != u and (u, w) not in pos_set:
                cand.append(w)
        cand_arr = np.array(cand)
        scores = Z[u] @ Z[cand_arr].T
        # rank of the true tail (index 0); strictly-greater => optimistic on ties
        rank = 1 + int(np.sum(scores[1:] > scores[0]))
        hits += 1 if rank <= k else 0
        rr += 1.0 / rank
        cnt += 1

    return {"num_eval": len(pos_pairs), "roc_auc": auc,
            f"hits@{k}": hits / cnt if cnt else 0.0,
            "mrr": rr / cnt if cnt else 0.0,
            "ranking_negatives": ranking_negs}


def edges_to_pairs(edge_dicts, idx) -> list[tuple[int, int]]:
    """Map {src,dst,rel} edge dicts to (row, row) index pairs via the node index."""
    return [(idx[e["src"]], idx[e["dst"]]) for e in edge_dicts
            if e["src"] in idx and e["dst"] in idx]
