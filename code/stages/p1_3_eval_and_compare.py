#!/usr/bin/env python3
"""p1_3 -- embed a model's corpus, build index, eval on both qrels, compare vs baselines.

Given a merged model dir, this: (1) embeds the corpus, (2) evaluates exact-origin and
graded-utility metrics via an in-process dense ranker, (3) writes an eval report, and
(4) checks the spec promotion gates against the 4B baseline.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from dh2 import config2 as C
from dh2 import graded_metrics as GM
from dh2.retriever2 import DenseRetriever


def _embed_corpus(model_dir, docs_path, out_vectors, out_ids, device=None, batch_size=256):
    from sentence_transformers import SentenceTransformer
    from discovery_hub.schema import read_docs
    docs = list(read_docs(Path(docs_path)))
    texts = [d.embedding_text or f"{d.title}\n{d.abstract}" for d in docs]
    ids = [d.doc_id for d in docs]
    m = SentenceTransformer(model_dir, device=device)
    vecs = m.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                    show_progress_bar=True, convert_to_numpy=True)
    np.save(out_vectors, vecs.astype("float32"))
    Path(out_ids).write_text(json.dumps(ids))
    return out_vectors, out_ids


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="merged model dir to evaluate")
    ap.add_argument("--name", required=True, help="short arm/label name for the report")
    ap.add_argument("--docs", default=str(C.PROD_DOCS))
    ap.add_argument("--exact", default=str(C.QRELS_DIR / "qrels_exact_origin_v1.jsonl"))
    ap.add_argument("--utility", default=str(C.QRELS_DIR / "qrels_scout_utility_v1.jsonl"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--reuse-embeddings", default="", help="skip embed; use these vectors")
    ap.add_argument("--baseline-r10", type=float, default=0.445, help="4B baseline exact R@10")
    args = ap.parse_args()

    C.ensure_dirs()
    work = C.BAKEOFF_DIR / f"eval_{args.name}"
    work.mkdir(parents=True, exist_ok=True)
    if args.reuse_embeddings:
        vpath = args.reuse_embeddings
        ipath = str(Path(args.reuse_embeddings).with_name("doc_ids.json"))
    else:
        vpath = str(work / "doc_vectors.npy"); ipath = str(work / "doc_ids.json")
        print(f"[p1_3] embedding corpus with {args.model} ...")
        _embed_corpus(args.model, args.docs, vpath, ipath, device=args.device)

    retr = DenseRetriever(args.model, vpath, ipath, device=args.device)
    def rank_fn(q): return [d for d, _ in retr.search(q, top_k=10)]

    print("[p1_3] evaluating exact-origin ...")
    exact = GM.evaluate_exact(args.exact, rank_fn)
    print("[p1_3] evaluating graded utility ...")
    graded = GM.evaluate_graded(args.utility, rank_fn)

    report = {"name": args.name, "model": args.model, "exact_origin": exact, "graded_utility": graded}
    rpath = C.REPORTS2_DIR / f"eval_{args.name}.json"
    rpath.write_text(json.dumps(report, indent=2))

    r10 = exact.get("recall@10", {}).get("mean", 0.0)
    ndcg = graded.get("ndcg@10", {}).get("mean", 0.0)
    gate_r10 = (r10 - args.baseline_r10) >= C.GATES.min_delta_r10_exact
    md = [f"# Bakeoff eval: {args.name}", "",
          f"- exact-origin R@10: **{r10:.4f}** (baseline {args.baseline_r10:.3f}, "
          f"delta {r10-args.baseline_r10:+.4f})",
          f"- exact-origin R@1: {exact.get('recall@1',{}).get('mean',0):.4f}",
          f"- exact-origin MRR@10: {exact.get('mrr@10',{}).get('mean',0):.4f}",
          f"- graded utility nDCG@10: **{ndcg:.4f}**",
          f"- utility Recall@10 (grade 2-3): {graded.get('utility_recall@10',{}).get('mean',0):.4f}",
          f"- grade-3 Recall@10: {graded.get('grade3_recall@10',{}).get('mean',0):.4f}",
          "",
          f"- promotion gate (+{C.GATES.min_delta_r10_exact} R@10 exact): "
          f"{'PASS' if gate_r10 else 'FAIL'}"]
    (C.REPORTS2_DIR / f"eval_{args.name}.md").write_text("\n".join(md))
    print("\n".join(md))
    print(f"[p1_3] report -> {rpath}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
