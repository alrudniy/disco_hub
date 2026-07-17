#!/usr/bin/env python3
"""p1_5 -- evaluate the PRODUCTION path on the same slice as p1_4.

WHY THIS EXISTS
eval_settle scored C0 / C0ft / C1 / C1b / C2. None of them is the system that
actually serves scouts. Production (07_retrieve_rank.py, DH_REGISTER_AWARE_CASCADE=0)
is:

    4B dense  +  BM25  +  graph(R-GCN, abstains)  --RRF-->  pool  --BGE rerank--> top-k

So C1b's headline "+0.1242 nDCG over C1" is measured against 8B-dense-no-rerank, a
clean scientific control that nobody deploys and nobody proposed deploying. Until the
production path is on the same 853 judged queries, with the same metric and the same
LOW/HIGH split, there is no shipping baseline and "should we change production" is
unanswerable.

The outcomes DIVERGE, which is what makes this worth GPU time:
    production ~= 0.50  -> C1b buys ~nothing. The cascade line closes. Ship nothing.
    production ~= 0.30  -> C1b is a large win and the 4.5 s p95 becomes a real debate.

COMPARABILITY IS THE WHOLE POINT. This reuses graded_metrics.evaluate_cascade, the
same --llm-judged-only slice, and takes --overlap-split so the LOW/HIGH buckets are
IDENTICAL to eval_settle's (0.375). A median-recomputed split would silently rebucket
the queries and every delta in the comparison table would be meaningless.

Usage (Drew or the box -- wherever the production artifacts live):

    python stages/p1_5_eval_production.py \
        --prod-script /home/alex/discovery_hub/07_retrieve_rank.py \
        --overlap-split 0.375 \
        --llm-judged-only \
        --name production
"""
from __future__ import annotations
import argparse, importlib.util, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dh2 import config2 as C
from dh2 import graded_metrics as GM


def _default_prod_script() -> str:
    """The cascade-aware 07 lives in this tree's prod/ (same search order as
    tests/test_07_cascade_flags.py). NOTE: a second, OLDER copy of 07_retrieve_rank.py
    exists at ~/discovery_hub_pipeline/ whose Retriever has NO register_aware parameter --
    pointing at it raises TypeError rather than silently measuring the wrong path."""
    here = Path(__file__).resolve().parent
    for cand in (here.parent / "prod" / "07_retrieve_rank.py",
                 here.parent / "07_retrieve_rank.py",
                 Path("/home/alex/discovery_hub/07_retrieve_rank.py")):
        if cand.exists():
            return str(cand)
    return str(here.parent / "prod" / "07_retrieve_rank.py")


def _load_prod(path: Path):
    """Import 07_retrieve_rank.py by path -- the leading digit blocks a normal import."""
    if not path.exists():
        sys.exit(f"ERROR: {path} not found. Pass --prod-script explicitly.")
    spec = importlib.util.spec_from_file_location("prod07", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _judged_query_ids(labels_path: Path) -> set[str]:
    qids: set[str] = set()
    for line in labels_path.open():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("llm_grade") is not None and not r.get("masked"):
            qids.add(r["query_id"])
    return qids


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prod-script", default=_default_prod_script())
    ap.add_argument("--docs", default=str(C.PROD_DOCS))
    ap.add_argument("--utility", default=str(C.QRELS_DIR / "qrels_scout_utility_v1.jsonl"))
    ap.add_argument("--labels", default=str(C.LABELS_DIR / "multi_positive_labels_v1.jsonl"))
    ap.add_argument("--llm-judged-only", action="store_true")
    ap.add_argument("--overlap-split", type=float, default=0.375,
                    help="MUST match eval_settle's split (0.375) or the buckets differ")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--pool-k", type=int, default=100)
    ap.add_argument("--no-keyword", action="store_true", help="ablate BM25")
    ap.add_argument("--no-graph", action="store_true", help="ablate the graph channel")
    ap.add_argument("--device", default=None)
    ap.add_argument("--name", default="production")
    args = ap.parse_args()

    C.ensure_dirs()

    prod = _load_prod(Path(args.prod_script))
    # Explicitly OFF. This arm measures what ships TODAY, not the cascade.
    r = prod.Retriever(mock=False, device=args.device, register_aware=False)
    if getattr(r, "register_aware", False):
        sys.exit("ERROR: cascade is ON. This stage must measure the production path.")

    # A MISSING ARTIFACT IS NOT AN ABLATION. 07 loads the graph channel only if
    # ARTIFACT_DIR/rgcn_node_emb.npy exists and BM25 only from INDEX_DIR/bm25.json (else it
    # rebuilds). If those are absent, retrieve() silently skips the channel and we would
    # publish a degraded system AS production -- and --no-graph would measure a zero
    # contribution that is really "never loaded". Fail loudly instead.
    if not args.no_graph and getattr(r, "graph_emb", None) is None:
        sys.exit("ERROR: graph channel requested but rgcn_node_emb.npy did not load "
                 f"(looked in {C.base.ARTIFACT_DIR}). Push the graph artifacts, or pass "
                 "--no-graph to ablate it DELIBERATELY. Refusing to report a silently "
                 "graph-less run as production.")
    if not args.no_graph and getattr(r, "_tech_rows", None) is None:
        sys.exit("ERROR: graph embeddings loaded but no technology rows aligned to doc_ids "
                 f"(needs {C.base.GRAPH_DIR}/nodes.jsonl). The graph channel would no-op.")
    if not args.no_keyword and getattr(r, "bm25", None) is None:
        sys.exit("ERROR: keyword channel requested but BM25 index is None.")
    _faiss_state = ("yes" if getattr(r, "_faiss", None) is not None
                    else "NO -- numpy brute force; p95 latency NOT representative")
    print(f"[p1_5] production retriever ready "
          f"(keyword={not args.no_keyword} graph={not args.no_graph} faiss={_faiss_state})")

    print("[p1_5] loading corpus for source/text maps ...")
    from discovery_hub.schema import read_docs
    docs = list(read_docs(Path(args.docs)))
    doc_source = {d.doc_id: getattr(d, "source", "unknown") for d in docs}
    doc_text = {d.doc_id: ((getattr(d, "title", "") or "") + " " +
                           (getattr(d, "embedding_text", None)
                            or getattr(d, "abstract", "") or "")[:1500])
                for d in docs}

    judged = _judged_query_ids(Path(args.labels)) if args.llm_judged_only else None
    if judged is not None:
        print(f"[p1_5] slice: {len(judged)} independently adjudicated queries")

    # retrieve() is expensive (dense + BM25 + graph + BGE). evaluate_cascade calls
    # rank_fn and candidate_fn separately, so cache per query or we pay twice.
    cache: dict[str, list[str]] = {}
    lat: dict[str, float] = {}

    def _run(q: str) -> list[str]:
        if q not in cache:
            t0 = time.perf_counter()
            res = r.retrieve(q, top_k=args.pool_k, rerank_k=args.pool_k,
                             use_keyword=not args.no_keyword,
                             use_graph=not args.no_graph)
            lat[q] = (time.perf_counter() - t0) * 1000
            cache[q] = [c["doc_id"] for c in res]
        return cache[q]

    rank_fn = lambda q: _run(q)[: args.top_k]      # noqa: E731
    cand_fn = lambda q: _run(q)                    # noqa: E731

    print("[p1_5] scoring ...")
    rep = GM.evaluate_cascade(
        args.utility, rank_fn,
        doc_text=doc_text, doc_source=doc_source, candidate_fn=cand_fn,
        llm_judged_query_ids=judged, overlap_split=args.overlap_split,
        k=args.top_k, recall_k=args.pool_k, latency_ms=lat)

    out_json = C.REPORTS2_DIR / f"eval_{args.name}.json"
    out_json.write_text(json.dumps({"PROD": rep}, indent=2))

    k = args.top_k
    o = rep.get("overall", {}); lo = rep.get("low_overlap", {}); hi = rep.get("high_overlap", {})

    def g(d, key):
        return d.get(key, {}).get("mean", 0.0)

    # eval_settle, for a side-by-side. Hardcoded so the report is self-contained.
    SETTLE = {
        "C1  FT-8B dense":      (0.3903, 0.3578, 0.4212, 0.6098, 0.3559, 0.7478, 363),
        "C1b 8B -> Qwen":       (0.5145, 0.4382, 0.5872, 0.7154, 0.5271, 0.7478, 4458),
        "C2  cascade":          (0.3736, 0.2474, 0.4939, 0.6585, 0.3857, 0.8940, 8582),
    }
    p = (g(o, f"ndcg@{k}"), g(lo, f"ndcg@{k}"), g(hi, f"ndcg@{k}"),
         g(o, f"success@{k}"), g(o, f"grade3_mrr@{k}"),
         g(o, f"candidate_recall@{args.pool_k}"), rep.get("latency_ms", {}).get("p95", 0))

    md = [f"# Production baseline vs eval_settle ({args.name})", "",
          f"Same harness, same slice, overlap split pinned to {args.overlap_split} "
          f"so LOW/HIGH buckets match eval_settle exactly.", "",
          f"Production = 07_retrieve_rank.py, DH_REGISTER_AWARE_CASCADE=0 "
          f"(4B dense + {'BM25 + ' if not args.no_keyword else ''}"
          f"{'graph + ' if not args.no_graph else ''}RRF + BGE rerank).", "",
          f"| arm | nDCG@{k} | LOW | HIGH | Success@{k} | grade-3 MRR | cand R@{args.pool_k} | p95 ms |",
          "|---|---:|---:|---:|---:|---:|---:|---:|",
          f"| **PROD (ships today)** | {p[0]:.4f} | {p[1]:.4f} | {p[2]:.4f} | {p[3]:.4f} | "
          f"{p[4]:.4f} | {p[5]:.4f} | {p[6]:.0f} |"]
    for name, v in SETTLE.items():
        md.append(f"| {name} | {v[0]:.4f} | {v[1]:.4f} | {v[2]:.4f} | {v[3]:.4f} | "
                  f"{v[4]:.4f} | {v[5]:.4f} | {v[6]:.0f} |")

    # The decision: C1b vs what ships today.
    c1b = SETTLE["C1b 8B -> Qwen"]
    md += ["", "## C1b vs PRODUCTION -- the only comparison that decides a ship", "",
           "| gate | value | result |", "|---|---:|---|"]
    checks = [("low-overlap does not regress", c1b[1] - p[1]),
              ("grade-3 MRR does not regress", c1b[4] - p[4]),
              (f"graded nDCG@{k} improves", c1b[0] - p[0]),
              (f"candidate Recall@{args.pool_k} >= production", c1b[5] - p[5])]
    for name, v in checks:
        md.append(f"| {name} | {v:+.4f} | {'PASS' if v >= 0 else '**FAIL**'} |")
    md.append(f"| p95 latency vs production | {c1b[6]:.0f} vs {p[6]:.0f} ms "
              f"({c1b[6]/max(p[6],1):.1f}x) | **product call, not a gate** |")
    verdict = "CLEARS THE QUALITY GATES" if all(v >= 0 for _, v in checks) \
        else "DOES NOT CLEAR"
    md += ["", f"**C1b {verdict} against the deployed system.**", "",
           "Latency is deliberately not scored: no gate bounds the Qwen path "
           "(CascadeGates only bounds GroupRank). That is a decision for a human.", ""]

    out_md = C.REPORTS2_DIR / f"eval_{args.name}.md"
    out_md.write_text("\n".join(md))
    print("\n" + "\n".join(md))
    print(f"\n[p1_5] -> {out_json}\n[p1_5] -> {out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
