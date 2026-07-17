#!/usr/bin/env python3
"""Flag-safety tests for 07_retrieve_rank.py's register-aware cascade.

The July 17 demo runs on the CURRENT path. So the single most important property of this
change is that DH_REGISTER_AWARE_CASCADE=0 is behaviorally identical to the file before
the edit -- the cascade must be unreachable, not merely unused. These tests stub
discovery_hub and assert exactly that, plus every fallback rung.

CPU-only, no models, no network. Run: python tests/test_07_cascade_flags.py
"""
import importlib.util
import json
import os
import sys
import tempfile
import types
from pathlib import Path

import numpy as np


def _stub_discovery_hub(root: Path):
    """Minimal discovery_hub so 07 imports without the real package."""
    dh = types.ModuleType("discovery_hub")
    retrieval = types.SimpleNamespace(
        top_k_recall=10, top_k_rerank=5, rrf_k=60, bm25_k1=1.2, bm25_b=0.75,
        use_keyword=True, use_graph=True, rerank_max_length=512, rerank_batch_size=16)
    cfg = types.SimpleNamespace(
        DATA_ROOT=root, NORM_DIR=root / "normalized", EMB_DIR=root / "embeddings",
        INDEX_DIR=root / "index", ARTIFACT_DIR=root / "artifacts",
        GRAPH_DIR=root / "graph", RERANK_MODEL="BAAI/bge-reranker-v2-m3",
        RETRIEVAL=retrieval, SEED=42,
        QUERY_INSTRUCTION="Given a pharmaceutical research interest, retrieve documents")
    dh.config = cfg

    el = types.ModuleType("discovery_hub.entity_link")
    el.build_surface_index = lambda nodes: {}
    el.link_query = lambda q, idx: []
    dh.entity_link = el

    det = types.ModuleType("discovery_hub.determinism")
    det.set_global_determinism = lambda seed: None

    emb = types.ModuleType("discovery_hub.embedding")

    class _Emb:
        def encode_queries(self, qs):
            # deterministic unit vector
            v = np.ones(4, dtype=np.float32)
            return np.array([v / np.linalg.norm(v)])
    emb.get_embedder = lambda mock=True, device=None: _Emb()
    # dh2.retriever2 imports these at module scope; without them _build_cascade fails
    # with an ImportError and the test silently checks the wrong fallback rung.
    emb.format_query = lambda q, task: f"Instruct: {task}\nQuery: {q}"
    cfg_query_instruction = "Given a pharmaceutical research interest, retrieve documents"

    fus = types.ModuleType("discovery_hub.fusion")

    def fuse_to_pool(lists, top_k, k=60):
        scores = {}
        for _, ids in lists.items():
            for rank, did in enumerate(ids, start=1):
                scores[did] = scores.get(did, 0.0) + 1.0 / (k + rank)
        return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
    fus.fuse_to_pool = fuse_to_pool

    kw = types.ModuleType("discovery_hub.keyword")

    class BM25Index:
        @staticmethod
        def load(p): return BM25Index()
        @staticmethod
        def build(docs, k1=1.2, b=0.75): return BM25Index()
        def score(self, q, top_k=10): return [("d2", 3.0), ("d1", 1.0)]
    kw.BM25Index = BM25Index

    sch = types.ModuleType("discovery_hub.schema")

    class Doc:
        def __init__(s, i, t, a, src):
            s.doc_id, s.title, s.abstract, s.source = i, t, a, src
            s.source_url, s.organizations = f"http://x/{i}", []
            s.embedding_text = f"{t} {a}"

    def read_docs(p):
        for line in Path(p).open():
            if line.strip():
                r = json.loads(line)
                yield Doc(r["doc_id"], r["title"], r["abstract"], r["source"])
    sch.read_docs, sch.DiscoveryDoc = read_docs, Doc

    for name, mod in [("discovery_hub", dh), ("discovery_hub.config", cfg),
                      ("discovery_hub.entity_link", el),
                      ("discovery_hub.determinism", det),
                      ("discovery_hub.embedding", emb), ("discovery_hub.fusion", fus),
                      ("discovery_hub.keyword", kw), ("discovery_hub.schema", sch)]:
        sys.modules[name] = mod
    return cfg


def _fixture(root: Path):
    (root / "normalized").mkdir(parents=True, exist_ok=True)
    (root / "embeddings").mkdir(parents=True, exist_ok=True)
    (root / "index").mkdir(parents=True, exist_ok=True)
    docs = [{"doc_id": "d1", "title": "Anti-B7-H1 antibodies for tumors",
             "abstract": "antibody", "source": "uspto"},
            {"doc_id": "d2", "title": "PD-L1 monoclonal antibody",
             "abstract": "oncology", "source": "uspto"}]
    (root / "normalized" / "docs.jsonl").write_text(
        "\n".join(json.dumps(d) for d in docs) + "\n")
    v = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    np.save(root / "embeddings" / "doc_vectors.npy", v)
    (root / "index" / "doc_ids.json").write_text(json.dumps(["d1", "d2"]))


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location("mod07", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    root = Path(tempfile.mkdtemp()) / "data"
    root.mkdir(parents=True)
    _stub_discovery_hub(root)
    _fixture(root)

    here = Path(__file__).resolve().parent
    for cand in (here / "07_retrieve_rank.py",
                 here.parent / "prod" / "07_retrieve_rank.py",
                 here.parent / "07_retrieve_rank.py"):
        if cand.exists():
            path = cand
            break
    else:
        raise SystemExit("07_retrieve_rank.py not found (looked in tests/, prod/, root)")
    for k in ("DH_REGISTER_AWARE_CASCADE", "DH_ENABLE_GROUPRANK",
              "DH_CASCADE_VECTORS_BASE_06B"):
        os.environ.pop(k, None)

    # 1) flag OFF: the cascade must be UNREACHABLE, not just unused
    m = _load_module(path)
    assert m.REGISTER_AWARE_CASCADE is False
    r = m.Retriever(mock=True)
    assert r.register_aware is False

    def _boom(*a, **k):
        raise AssertionError("cascade must NOT be reached when the flag is off")
    r._build_cascade = _boom
    base = r.retrieve("PD-L1 monoclonal antibody for oncology")
    assert base and all("title" in c for c in base), "existing path must still work"
    assert all("text_score" in c and "keyword_score" in c for c in base), \
        "the 08 contract fields must be present"
    assert all(c.get("qwen_logit_margin") is None for c in base), \
        "flag-off output must carry no cascade fields"
    print(f"  [1] flag OFF: cascade unreachable, {len(base)} results, contract intact OK")

    # 2) flag ON + mock: refuses (mock has no real embeddings for a reranker to judge)
    os.environ["DH_REGISTER_AWARE_CASCADE"] = "1"
    m = _load_module(path)
    assert m.REGISTER_AWARE_CASCADE is True
    r = m.Retriever(mock=True)
    assert r.register_aware is False, "cascade must self-disable under --mock"
    print("  [2] flag ON + mock: self-disables OK")

    # 3) flag ON, real mode, NO untouched-0.6B vectors -> falls back, does not crash.
    # dh2 must be importable here, or this silently tests the ImportError rung instead
    # of the missing-vectors rung -- same outcome, different code path, weaker test.
    # Locate the tree root by looking for dh2/, NOT by directory name: this tree gets
    # renamed (pipeline_2 -> pipeline_3 -> ...), and a hardcoded name degrades this test
    # to the wrong rung while still reporting PASS.
    for cand in (Path(__file__).resolve().parents[1],          # tests/ -> tree root
                 Path(__file__).resolve().parent):
        if (cand / "dh2").is_dir():
            sys.path.insert(0, str(cand))
            break
    try:
        import dh2  # noqa: F401
        have_dh2 = True
    except ImportError:
        have_dh2 = False
    assert have_dh2, ("dh2 not importable from the tree root -- this test would silently "
                      "check the ImportError rung instead of the missing-vectors rung")
    r = m.Retriever(mock=False)
    assert r.register_aware is True
    assert r._build_cascade() is None, "must refuse without the semantic-hedge vectors"
    out = r._retrieve_cascade("q", np.ones(4, dtype=np.float32), 5, False)
    assert out is None, "missing prereqs must return None so retrieve() falls back"
    rung = "missing-vectors" if have_dh2 else "ImportError (dh2 not on path)"
    print(f"  [3] flag ON, no base-06B vectors: refuses via the {rung} rung + "
          f"falls back (no regression shipped) OK")

    # 4) GroupRank without the cascade is ignored, not silently half-enabled
    os.environ["DH_REGISTER_AWARE_CASCADE"] = "0"
    os.environ["DH_ENABLE_GROUPRANK"] = "1"
    m = _load_module(path)
    r = m.Retriever(mock=False)
    assert r.enable_grouprank is False, "grouprank requires the cascade"
    print("  [4] grouprank without cascade: ignored OK")

    # 5) the fallback rungs are wired: Qwen failure -> dense/RRF
    os.environ["DH_REGISTER_AWARE_CASCADE"] = "1"
    os.environ["DH_ENABLE_GROUPRANK"] = "0"
    m = _load_module(path)
    r = m.Retriever(mock=False)

    class _DeadCascade:
        cfg = types.SimpleNamespace(qwen_shortlist=40)
        def build_candidate_union(self, q): return {}
        def qwen_rerank(self, q, pool): raise RuntimeError("qwen OOM")
    r._cascade = _DeadCascade()
    r._build_cascade = lambda: r._cascade
    assert r._retrieve_cascade("q", np.ones(4, dtype=np.float32), 5, False) is None, \
        "a Qwen exception must fall back, not propagate"
    print("  [5] Qwen failure -> dense/RRF fallback OK")

    for k in ("DH_REGISTER_AWARE_CASCADE", "DH_ENABLE_GROUPRANK"):
        os.environ.pop(k, None)
    print("\n07 CASCADE FLAG TESTS PASSED (flag-off is the untouched demo path)")


if __name__ == "__main__":
    main()
