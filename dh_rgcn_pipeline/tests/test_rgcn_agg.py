"""
Known-answer test for the R-GCN mock relational mean aggregation (fix #5).

Pins the corrected "per-relation mean, then sum across relations" behavior so the
original double-dividing bug -- a node receiving [1,1] from a degree-1 relation
and a [1,1] mean from a degree-2 relation yielding [1.5,1.5] instead of [2,2] --
cannot silently return. Run: python -m pytest tests/ -q
"""
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _load_stage06():
    spec = importlib.util.spec_from_file_location("train_rgcn", ROOT / "06_train_rgcn.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = _load_stage06()


def test_per_relation_mean_then_sum_reviewer_scenario():
    # node 0 receives [1,1] from relation A (1 neighbor) and a [1,1] MEAN from
    # relation B (2 neighbors, each [1,1]). Correct sum-of-means = [2,2].
    h = np.zeros((4, 2), np.float32)
    h[1] = [1.0, 1.0]   # A-neighbor of node 0
    h[2] = [1.0, 1.0]   # B-neighbor of node 0
    h[3] = [1.0, 1.0]   # B-neighbor of node 0
    by_rel = {"A": [(0, 1)], "B": [(0, 2), (0, 3)]}
    agg = M.relational_mean_aggregate(h, by_rel)
    np.testing.assert_allclose(agg[0], [2.0, 2.0], atol=1e-6)


def test_discriminates_against_the_old_double_dividing_bug():
    # Reproduce the OLD logic (divide the running accumulator per relation) and
    # confirm it yields the WRONG [1.5,1.5] -- so this test actually guards the
    # fix rather than passing vacuously, then confirm the real function is right.
    h = np.zeros((4, 2), np.float32)
    h[1] = h[2] = h[3] = [1.0, 1.0]
    by_rel = {"A": [(0, 1)], "B": [(0, 2), (0, 3)]}

    def old_buggy(h, by_rel):
        n = h.shape[0]
        agg = np.zeros_like(h)
        for pairs in by_rel.values():
            src = np.fromiter((p[0] for p in pairs), dtype=np.int64)
            dst = np.fromiter((p[1] for p in pairs), dtype=np.int64)
            add = np.zeros_like(h)
            np.add.at(add, src, h[dst])
            agg += add
            deg = np.zeros(n, np.float32)
            np.add.at(deg, src, 1.0)
            deg[deg == 0] = 1.0
            agg /= deg[:, None]      # the bug: divides the WHOLE accumulator
        return agg

    assert np.allclose(old_buggy(h, by_rel)[0], [1.5, 1.5])                     # bug
    assert np.allclose(M.relational_mean_aggregate(h, by_rel)[0], [2.0, 2.0])   # fixed


def test_single_relation_is_plain_mean():
    h = np.zeros((3, 2), np.float32)
    h[1] = [2.0, 0.0]
    h[2] = [0.0, 4.0]
    by_rel = {"A": [(0, 1), (0, 2)]}     # mean of [2,0] and [0,4] = [1,2]
    agg = M.relational_mean_aggregate(h, by_rel)
    np.testing.assert_allclose(agg[0], [1.0, 2.0], atol=1e-6)


def test_empty_relations_contribute_nothing():
    h = np.ones((3, 2), np.float32)
    by_rel = {"A": [], "B": []}
    agg = M.relational_mean_aggregate(h, by_rel)
    np.testing.assert_allclose(agg, np.zeros((3, 2)), atol=1e-6)
