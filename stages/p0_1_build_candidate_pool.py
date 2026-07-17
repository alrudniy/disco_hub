#!/usr/bin/env python3
"""p0_1 -- build the candidate pool.

Channels: untouched 0.6B (semantic hedge) + fine-tuned 8B + fine-tuned 4B (top + deep
window) + routed exact-identifier lookup.

The untouched 0.6B is the important addition (RASC S1). Without --vectors-base-06b this
stage reproduces the OLD, fine-tuned-only pool -- the one that omitted the model measured
to be best on the low-overlap queries that are 76% of this corpus's true matches. Those
documents then never reach the teachers, never get graded, and never appear in the qrels,
so nothing downstream can recover them.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dh2 import config2 as C
from dh2.retriever2 import DenseRetriever
from dh2 import candidate_pool as CP


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", default=str(C.PROD_TRAIN_TRIPLES),
                    help="query records (train triples or eval qrels)")
    ap.add_argument("--vectors-4b", default=str(C.base.EMB_DIR / "doc_vectors.npy"))
    ap.add_argument("--ids-4b", default=str(C.base.INDEX_DIR / "doc_ids.json"))
    ap.add_argument("--vectors-8b", default="", help="optional 8B vectors")
    ap.add_argument("--ids-8b", default="")
    ap.add_argument("--vectors-base-06b", default="",
                    help="UNTOUCHED 0.6B vectors (the semantic hedge; strongly recommended)")
    ap.add_argument("--ids-base-06b", default="")
    ap.add_argument("--docs", default=str(C.PROD_DOCS))
    ap.add_argument("--exact-identifiers", action="store_true",
                    help="add the routed NCT/patent/CAS/compound channel")
    ap.add_argument("--out", default=str(C.POOL_DIR / "candidate_pool_v1.jsonl"))
    ap.add_argument("--max-queries", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    C.ensure_dirs()
    qrecs = CP._load_query_records(Path(args.queries), args.max_queries)
    r4 = DenseRetriever(C.MODEL_4B, args.vectors_4b, args.ids_4b, device=args.device)
    r8 = None
    if args.vectors_8b and Path(args.vectors_8b).exists():
        r8 = DenseRetriever(C.MODEL_8B, args.vectors_8b, args.ids_8b, device=args.device)
    r06 = None
    if args.vectors_base_06b and Path(args.vectors_base_06b).exists():
        r06 = DenseRetriever(C.MODEL_BASE_06B, args.vectors_base_06b, args.ids_base_06b,
                             device=args.device)
    else:
        print("[p0_1] WARNING: no --vectors-base-06b. Building the OLD fine-tuned-only "
              "pool; low-overlap candidates will be missing (see module docstring).")

    ident_index = None
    if args.exact_identifiers:
        from dh2.identifiers import build_identifier_index
        from discovery_hub.schema import read_docs
        print("[p0_1] building exact-identifier index ...")
        ident_index = build_identifier_index(read_docs(Path(args.docs)))
        print(f"[p0_1]   {len(ident_index)} identifiers indexed")

    n = 0
    chan_counts: dict[str, int] = {}
    with Path(args.out).open("w") as f:
        for rec in CP.build_pool(qrecs, r4, r8, retr_base_06b=r06,
                                 identifier_index=ident_index):
            f.write(json.dumps(rec) + "\n"); n += 1
            for ch in rec.get("ranks", {}):
                chan_counts[ch] = chan_counts.get(ch, 0) + 1
    print(f"[p0_1] wrote {n} candidate rows for {len(qrecs)} queries -> {args.out}")
    print(f"[p0_1] rows contributed per channel: {json.dumps(chan_counts, indent=2)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
