#!/usr/bin/env python3
"""Mini-corpus end-to-end smoke test for pipeline_2 -- mocks all heavy deps (no GPU,
no network, no torch). Exercises the data-flow contracts between stages so a broken
interface is caught before renting an H100. Run: python tests/test_dh2_smoke.py
"""
import sys, types, json, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ---- stub discovery_hub so dh2 imports without the real package ----
def _install_stub_discovery_hub(data_root: Path):
    dh = types.ModuleType("discovery_hub")
    cfg = types.SimpleNamespace(
        DATA_ROOT=data_root, NORM_DIR=data_root / "normalized",
        EMB_DIR=data_root / "embeddings", INDEX_DIR=data_root / "index",
        RERANK_MODEL="BAAI/bge-reranker-v2-m3",
        QUERY_INSTRUCTION="retrieve relevant docs")
    dh.config = cfg
    emb = types.ModuleType("discovery_hub.embedding")
    emb.format_query = lambda q, task: f"Instruct: {task}\nQuery: {q}"
    sch = types.ModuleType("discovery_hub.schema")
    class Doc:
        def __init__(s, i, t, a, src):
            s.doc_id=i; s.title=t; s.abstract=a; s.source=src; s.embedding_text=f"{t} {a}"
    def read_docs(p):
        for line in Path(p).open():
            if line.strip():
                r=json.loads(line); yield Doc(r["doc_id"],r["title"],r["abstract"],r["source"])
    sch.read_docs=read_docs; sch.DiscoveryDoc=Doc
    sys.modules["discovery_hub"]=dh; sys.modules["discovery_hub.config"]=cfg
    sys.modules["discovery_hub.embedding"]=emb; sys.modules["discovery_hub.schema"]=sch


def main():
    tmp = Path(tempfile.mkdtemp())
    data = tmp / "data"; (data/"normalized").mkdir(parents=True)
    _install_stub_discovery_hub(data)
    import os; os.environ["DH_DATA_ROOT"]=str(data); os.environ["DH2_ROOT"]=str(data/"pipeline2")
    os.environ["DH2_TEACHER_LLM_API_KEY"]="test"

    from dh2 import config2 as C
    from dh2 import candidate_pool as CP
    from dh2 import teacher_merge as TM
    from dh2 import eval_pool_build as EP
    from dh2 import graded_metrics as GM
    from dh2 import train_bakeoff as TB
    C.ensure_dirs()

    # mini corpus + queries
    docs=[{"doc_id":"ct:1","title":"Oral drug in relapsed CLL","abstract":"oral therapy relapsed chronic lymphocytic leukemia","source":"clinicaltrials"},
          {"doc_id":"ct:2","title":"Venetoclax relapsed CLL","abstract":"oral venetoclax relapsed CLL","source":"clinicaltrials"},
          {"doc_id":"us:9","title":"Unrelated widget","abstract":"a mechanical widget","source":"uspto"}]
    (data/"normalized"/"docs.jsonl").write_text("\n".join(json.dumps(d) for d in docs)+"\n")

    # 1) candidate pool via a fake retriever
    class FakeRetr:
        def search(self, q, top_k=100, window=None):
            r=[("ct:1",0.9),("ct:2",0.85),("us:9",0.1)]
            return r[window[0]:window[1]] if window else r[:top_k]
        def score_pairs(self, q, ids): 
            m={"ct:1":0.9,"ct:2":0.85,"us:9":0.1}; return {d:m.get(d,0.05) for d in ids}
    qrecs=[{"query_id":"q1","query":"oral therapy for relapsed CLL","positive_ids":["ct:1"]}]
    pool=list(CP.build_pool(qrecs, FakeRetr(), None, pool_cfg=C.PoolConfig(per_source_topk=3, deep_window_start=1, deep_window_end=3, max_candidates=10)))
    assert any(p["is_designated_positive"] for p in pool)
    print(f"  [1] candidate pool: {len(pool)} rows OK")

    # 2) merged labels (BGE only, no LLM -> masking for gray-zone)
    labels=[]
    for p in pool:
        scores={"ct:1":0.95,"ct:2":0.9,"us:9":0.08}
        lab=TM.merge_one(query_id=p["query_id"],query=p["query"],document_id=p["document_id"],
            source=p["source"],doc_title="",doc_text="",is_designated_positive=p["is_designated_positive"],
            bge_score=scores.get(p["document_id"],0.1),qwen_score=scores.get(p["document_id"],0.1),
            positive_teacher_score=0.95,llm=None)
        labels.append(lab.__dict__)
    lp=data/"pipeline2"/"labels.jsonl"; lp.write_text("\n".join(json.dumps(x) for x in labels)+"\n")
    print(f"  [2] merged labels: {len(labels)} OK ({sum(l['grade']>=2 for l in labels)} positives)")

    # 3) qrels
    summ=EP.build_qrels(lp, data/"pipeline2"/"exact.jsonl", data/"pipeline2"/"util.jsonl")
    assert summ["exact_origin_queries"]==1
    print(f"  [3] qrels: {summ} OK")

    # 4) graded metrics with a perfect ranker
    def rank_fn(q): return ["ct:1","ct:2","us:9"]
    graded=GM.evaluate_graded(data/"pipeline2"/"util.jsonl", rank_fn)
    assert graded["ndcg@10"]["mean"] > 0.9
    print(f"  [4] graded eval nDCG@10={graded['ndcg@10']['mean']} OK")

    # 5) bakeoff data prep
    grouped=TB.load_labels_grouped(lp)
    assert "q1" in grouped and sum(grouped["q1"].is_positive)>=1
    trips=TB.make_triples_for_marginmse(grouped["q1"], n_neg=2, seed=1)
    packed=TB.pack_listwise(grouped["q1"], max_list=8)
    assert len(packed["doc_ids"])==8
    print(f"  [5] bakeoff prep: {len(trips)} triples, listwise packed OK")

    print("\nSMOKE TEST PASSED (all stage interfaces wired correctly)")

if __name__ == "__main__":
    main()
