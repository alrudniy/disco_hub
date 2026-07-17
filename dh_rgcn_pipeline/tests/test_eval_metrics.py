"""
Known-answer tests for the evaluation metrics. If these pass, the numbers the
harness reports mean what they say. Run: python -m pytest tests/ -q
"""
import math

import numpy as np

from discovery_hub import evaluate as ev


def test_perfect_rank1():
    m = ev.evaluate_query(["A", "B", "C", "D"], ["A"], ks=(1, 5, 10), main_k=10)
    assert m["recall@1"] == 1.0
    assert m["mrr@10"] == 1.0
    assert m["ndcg@10"] == 1.0
    assert m["map"] == 1.0
    # one positive in a depth-10 list => precision@10 capped at 0.1
    assert abs(m["precision@10"] - 0.1) < 1e-9


def test_relevant_at_rank2():
    m = ev.evaluate_query(["B", "A", "C"], ["A"], ks=(1, 2, 5), main_k=10)
    assert m["recall@1"] == 0.0
    assert m["recall@2"] == 1.0
    assert m["mrr@10"] == 0.5
    # nDCG with the single positive at rank 2 = (1/log2(3)) / (1/log2(2))
    assert abs(m["ndcg@10"] - (1.0 / math.log2(3))) < 1e-9
    assert abs(m["map"] - 0.5) < 1e-9  # AP = (1/1) * (1/2)


def test_no_relevant_retrieved():
    m = ev.evaluate_query(["B", "C", "D"], ["A"], ks=(1, 5, 10), main_k=10)
    for k in (1, 5, 10):
        assert m[f"recall@{k}"] == 0.0
    assert m["mrr@10"] == 0.0
    assert m["ndcg@10"] == 0.0
    assert m["map"] == 0.0


def test_multi_positive_recall_and_precision():
    # 4 relevant; top-10 contains 2 of them => recall@10 = 0.5, precision@10 = 0.2
    ranked = ["A", "x", "B", "y", "z"] + [f"d{i}" for i in range(5)]
    m = ev.evaluate_query(ranked, ["A", "B", "C", "D"], ks=(1, 5, 10), main_k=10)
    assert abs(m["recall@10"] - 0.5) < 1e-9
    assert abs(m["precision@10"] - 0.2) < 1e-9
    assert m["recall@1"] == 0.25  # only A in top-1, of 4 relevant


def test_ndcg_two_positives_ideal():
    # both positives at the very top => nDCG@10 == 1.0
    ranked = ["A", "B", "C", "D"]
    m = ev.evaluate_query(ranked, ["A", "B"], ks=(1, 5), main_k=10)
    assert abs(m["ndcg@10"] - 1.0) < 1e-9


def test_bootstrap_ci_is_deterministic_and_brackets_mean():
    vals = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
    mean1, lo1, hi1 = ev.bootstrap_ci(vals, n_boot=500, seed=42)
    mean2, lo2, hi2 = ev.bootstrap_ci(vals, n_boot=500, seed=42)
    assert (mean1, lo1, hi1) == (mean2, lo2, hi2)   # deterministic
    assert lo1 <= mean1 <= hi1
    assert abs(mean1 - vals.mean()) < 1e-9


def test_paired_delta_detects_uniform_improvement():
    base = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    cand = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    d = ev.paired_delta(base, cand, n_boot=500)
    assert abs(d["mean_delta"] - 1.0) < 1e-9
    assert d["ci_low"] > 0.0           # CI does not cross zero
    assert d["p_value"] is not None and d["p_value"] < 0.05


def test_paired_delta_identical_systems():
    a = np.array([0.3, 0.7, 0.5, 1.0])
    d = ev.paired_delta(a, a.copy(), n_boot=200)
    assert d["mean_delta"] == 0.0
    assert d["p_value"] == 1.0          # no difference


def test_synthetic_eval_is_deterministic():
    class Doc:
        def __init__(self, i):
            self.doc_id = f"src:{i}"
            self.title = f"compound {i} inhibitor"
            self.abstract = (f"a potent selective inhibitor targeting kinase "
                             f"pathway {i} for oncology treatment")
            self.embedding_text = self.title + "\n" + self.abstract

    docs = [Doc(i) for i in range(20)]
    q1 = ev.build_synthetic_eval(docs, num_queries=10, seed=7)
    q2 = ev.build_synthetic_eval(docs, num_queries=10, seed=7)
    assert [(q.query_id, q.query, tuple(q.relevant_doc_ids)) for q in q1] == \
           [(q.query_id, q.query, tuple(q.relevant_doc_ids)) for q in q2]
    # every query's gold positive is a real doc id
    ids = {d.doc_id for d in docs}
    for q in q1:
        assert len(q.relevant_doc_ids) == 1 and q.relevant_doc_ids[0] in ids
