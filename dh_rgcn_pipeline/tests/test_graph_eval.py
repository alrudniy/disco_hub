"""
Known-answer tests for the link-prediction metrics and edge split in
discovery_hub.graph_eval. Run: python -m pytest tests/ -q
"""
import numpy as np

from discovery_hub import graph_eval as ge


def test_roc_auc_extremes():
    # perfect separation -> 1.0, reversed -> 0.0, identical -> 0.5
    assert ge.roc_auc([3.0, 2.0], [1.0, 0.0]) == 1.0
    assert ge.roc_auc([0.0, 1.0], [2.0, 3.0]) == 0.0
    assert abs(ge.roc_auc([1.0, 1.0], [1.0, 1.0]) - 0.5) < 1e-9


def test_split_edges_partitions_without_overlap():
    edges = [{"src": f"a{i}", "dst": f"b{i}", "rel": "r"} for i in range(100)]
    mp, val, test = ge.split_edges(edges, val_frac=0.1, test_frac=0.2, seed=1)
    assert len(mp) == 70 and len(val) == 10 and len(test) == 20
    # disjoint and complete
    keyed = lambda L: {(e["src"], e["dst"]) for e in L}
    assert keyed(mp) & keyed(val) == set()
    assert keyed(mp) & keyed(test) == set()
    assert keyed(val) & keyed(test) == set()
    assert len(keyed(mp) | keyed(val) | keyed(test)) == 100


def test_split_edges_is_deterministic():
    edges = [{"src": f"a{i}", "dst": f"b{i}", "rel": "r"} for i in range(50)]
    a = ge.split_edges(edges, seed=7)
    b = ge.split_edges(edges, seed=7)
    assert [e["src"] for e in a[2]] == [e["src"] for e in b[2]]


def test_link_prediction_separates_signal_from_noise():
    # 6 nodes; make the held-out edge (0,1) have identical endpoint vectors
    # (dot = 1) while everything else is orthogonal => AUC = 1, perfect ranking.
    dim = 8
    Z = np.zeros((6, dim), dtype=np.float32)
    Z[0] = Z[1] = np.eye(dim)[0]       # endpoints of the true edge coincide
    Z[2] = np.eye(dim)[1]
    Z[3] = np.eye(dim)[2]
    Z[4] = np.eye(dim)[3]
    Z[5] = np.eye(dim)[4]
    pos_pairs = [(0, 1)]
    pos_set = {(0, 1)}
    m = ge.evaluate_link_prediction(Z, pos_pairs, pos_set, seed=3, k=3,
                                    ranking_negs=4)
    assert m["num_eval"] == 1
    assert m["roc_auc"] == 1.0
    assert m["hits@3"] == 1.0
    assert m["mrr"] == 1.0


def test_link_prediction_empty_is_safe():
    Z = np.random.default_rng(0).standard_normal((5, 4)).astype(np.float32)
    m = ge.evaluate_link_prediction(Z, [], set(), seed=0)
    assert m["num_eval"] == 0


def test_edges_to_pairs_maps_via_index():
    idx = {"a": 0, "b": 1, "c": 2}
    edges = [{"src": "a", "dst": "b", "rel": "r"},
             {"src": "b", "dst": "c", "rel": "r"},
             {"src": "a", "dst": "zzz", "rel": "r"}]  # zzz not in idx -> dropped
    pairs = ge.edges_to_pairs(edges, idx)
    assert pairs == [(0, 1), (1, 2)]
