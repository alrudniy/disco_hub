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

    # ---------------- RASC + bug-fix coverage ---------------- #
    from dh2 import identifiers as ID
    from dh2 import doc_views as DV
    from dh2 import candidate_pool as CP2
    from dh2 import grouprank as GR
    from dh2.cascade import RegisterAwareCascade

    # 6) exact-identifier channel: fires on real ids, silent on lookalikes
    assert ID.contains_exact_identifier("results for NCT04381936 please")
    assert ID.contains_exact_identifier("see US10485802B2")
    assert ID.contains_exact_identifier("AZD9291 resistance")
    assert not ID.contains_exact_identifier("PD-L1 antibody for oncology")
    assert not ID.contains_exact_identifier("CD274 expression at 100mg")   # gene + dose
    assert not ID.contains_exact_identifier("CAS 12-34-5")                 # bad checksum
    assert ID._cas_checksum_ok("50-00-0")                                  # formaldehyde
    idx = ID.build_identifier_index(docs)
    print(f"  [6] identifiers: {len(idx)} indexed, precision guards OK")

    # 7) source-aware views: header by source, no fabricated empty fields
    v = DV.render_view({"doc_id":"us:9","title":"Anti-B7-H1 antibodies","abstract":"x"*4000,
                        "source":"uspto"}, max_tokens=100)
    assert v.startswith("PATENT") and "Title:" in v and "Phase" not in v
    assert DV._approx_tokens(v) <= 130           # budget honored (+ label overhead)
    vt = DV.render_view({"doc_id":"ct:1","title":"T","brief_summary":"S","phase":"2",
                         "source":"clinicaltrials"}, max_tokens=500)
    assert vt.startswith("CLINICAL TRIAL") and "Phase/status: 2" in vt
    print("  [7] doc views: source headers, budget, no empty fields OK")

    # 8) union + round-robin cap (BUG: 4B-score-biased cap dropped 8B-only candidates)
    class R:
        def __init__(s, res): s.res=res
        def search(s, q, top_k=100, window=None):
            return s.res[window[0]:window[1]] if window else s.res[:top_k]
        def score_pairs(s, q, ids): return {d: 0.5 for d in ids}
    semantic = R([("only_semantic",0.9),("shared",0.4)])
    ft8b     = R([("only_8b",0.3),("shared",0.2)])
    union = CP2.build_union("q", {CP2.CH_SEMANTIC: semantic, CP2.CH_FT_8B: ft8b},
                            {CP2.CH_SEMANTIC: 10, CP2.CH_FT_8B: 10})
    assert set(union) == {"only_semantic","shared","only_8b"}, "union must not lose docs"
    assert union["shared"].channels == {CP2.CH_SEMANTIC, CP2.CH_FT_8B}
    assert union["shared"].ranks[CP2.CH_SEMANTIC] == 2
    capped = CP2.cap_union(union, 2)
    # round-robin keeps ONE per channel; a global sort by the 4B/8B score would have
    # evicted the semantic-only doc, which is the whole bug.
    assert "only_semantic" in capped, "round-robin must not starve the semantic channel"
    assert len(capped) == 2
    print(f"  [8] union: no doc lost, provenance kept, round-robin cap OK -> {sorted(capped)}")

    # 9) GroupRank: partitions cover every doc; strict parse; fail-closed
    groups = GR.make_groups([f"d{i}" for i in range(25)], group_size=10, repeats=2, seed=1)
    seen = [d for g in groups for d in g]
    assert len(seen) == 50 and len(set(seen)) == 25, "each doc once per repeat"
    assert GR.parse_answer_json('{"D1": 7, "D2": 0}', 2) == {"D1":7,"D2":0}
    assert GR.parse_answer_json('```json\n{"D1":3}\n```', 1) == {"D1":3}
    for bad, n in [('{"D1":7}', 2), ('{"D1":7,"D3":1}', 2), ('{"D1":99}', 1), ('sorry', 1)]:
        try:
            GR.parse_answer_json(bad, n); raise AssertionError(f"should reject {bad!r}")
        except AssertionError: raise
        except Exception: pass
    print(f"  [9] grouprank: {len(groups)} groups, strict parse rejects malformed OK")

    # 10) cascade end-to-end. The stub INVERTS the dense order, so this only passes if the
    # Qwen logit margin actually drives the ranking. This is the PD-L1/B7-H1 case in
    # miniature: the dense retriever ranks the lexical match first, Qwen knows better.
    class StubQwen:
        _tok = None
        def score_detailed(s, pairs):
            # reward the doc the dense channel ranked LAST
            return [{"probability": 0.9,
                     "logit_margin": 5.0 if "Venetoclax" in d else -5.0}
                    for _, d in pairs]
    dense = R([("ct:1",0.9),("ct:2",0.5),("us:9",0.1)])   # ct:2 is dense-rank 2
    cas = RegisterAwareCascade({CP2.CH_SEMANTIC: dense},
                               {d["doc_id"]: d for d in docs},
                               qwen_teacher=StubQwen(), enable_grouprank=False)
    order = cas.rank_ids("oral therapy for relapsed CLL", top_k=3)
    assert order[0] == "ct:2", f"qwen margin must OVERRIDE dense order, got {order}"
    assert set(order) == {"ct:1","ct:2","us:9"}, "rerank must not drop candidates"
    # ties below the winner fall back to best retrieval rank
    assert order[1] == "ct:1", f"tie-break must be best retrieval rank, got {order}"
    print(f"  [10] cascade: qwen margin overrides dense order OK -> {order}")

    # 11) cascade fails CLOSED to dense ordering when Qwen raises
    class BoomQwen:
        _tok = None
        def score_detailed(s, pairs): raise RuntimeError("OOM")
    cas2 = RegisterAwareCascade({CP2.CH_SEMANTIC: R([("ct:1",0.9),("us:9",0.1)])},
                                {d["doc_id"]: d for d in docs},
                                qwen_teacher=BoomQwen(), enable_grouprank=False)
    assert cas2.rank_ids("q", top_k=2) == ["ct:1","us:9"], "must fall back to dense order"
    print("  [11] cascade: Qwen failure -> dense fallback OK")

    # 12) BUG #6: pack_listwise stratifies (was: first 8 in pool order, 32.7% zero-pos)
    from dh2.train_bakeoff import PackedQuery
    skewed = PackedQuery("q","q", [f"n{i}" for i in range(20)]+["p1","p2"],
                         [0.1]*20+[0.9,0.9], [False]*20+[True,True],
                         [1.0]*22, [False]*22)
    old = TB.pack_listwise(skewed, max_list=8, stratify=False)
    new = TB.pack_listwise(skewed, max_list=8, seed=0)
    assert old["n_positive"] == 0, "reproduces the bug: no positives in the window"
    assert new["n_positive"] > 0, "stratified packing must include positives"
    assert sum(new["mask"]) == 8
    print(f"  [12] bug#6 pack_listwise: unstratified={old['n_positive']} pos, "
          f"stratified={new['n_positive']} pos OK")

    # 13) BUG #7: fn_flag is per-(query, doc), and the legacy shape still resolves
    assert TB.is_fn_flagged({"qA": {"d1": True}}, "qA", "d1") is True
    assert TB.is_fn_flagged({"qA": {"d1": True}}, "qB", "d1") is False   # the fix
    assert TB.is_fn_flagged({"d1": True}, "qB", "d1") is True            # legacy honored
    assert TB.fn_flag_is_legacy({"d1": True}) and not TB.fn_flag_is_legacy({"qA":{"d1":True}})
    print("  [13] bug#7 fn_flag: per-query scoping OK, legacy shape detected")

    # 14) BUG #8: relevant vs candidate counts are no longer conflated
    assert "median_relevant_per_utility_query" in summ and "mean_candidates_per_query" in summ
    assert summ["mean_relevant_per_utility_query"] <= summ["mean_candidates_per_query"]
    print(f"  [14] bug#8 qrels counts: relevant={summ['mean_relevant_per_utility_query']} "
          f"candidates={summ['mean_candidates_per_query']} OK")

    # 15) BUG #10: an unparseable verdict is recognized, incl. the legacy poisoned rows
    from dh2.llm_client import TeacherLLM
    assert TeacherLLM.is_unparseable({"grade": None, "unparseable": True})
    assert TeacherLLM.is_unparseable({"grade": 1, "relevance": 0.5,
                                      "evidence": "[unparseable LLM reply masked] junk"})
    assert not TeacherLLM.is_unparseable({"grade": 3, "relevance": 0.9, "evidence": "ok"})
    print("  [15] bug#10 unparseable: detected, legacy cache rows caught retroactively")

    # 16) graded metrics: Success@k and the LOW/HIGH register buckets
    assert GM.success_at_k(["a","b"], {"b"}, 2) == 1.0
    assert GM.success_at_k(["a","b"], {"z"}, 2) == 0.0
    assert GM.word_overlap("PD-L1 antibody oncology", "anti-B7-H1 antibodies tumors") < 0.5
    assert GM.oracle_utility(["ct:1","ct:2"], {"ct:1":3,"ct:2":2}, 10) > 0.9
    print("  [16] metrics: success@k, overlap buckets, oracle OK")

    print("\nSMOKE TEST PASSED (all stage interfaces wired correctly)")

if __name__ == "__main__":
    main()
