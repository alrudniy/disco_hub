#!/usr/bin/env python3
"""p0_1 -- build the candidate pool from 4B/8B/deep-dense channels."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dh2 import config2 as C
from dh2.retriever2 import DenseRetriever
from dh2 import candidate_pool as CP


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--queries", default=str(C.PROD_TRAIN_TRIPLES),
                    help="query records (train triples or eval qrels)")
    ap.add_argument("--vectors-4b", default=str(C.base.EMB_DIR / "doc_vectors.npy"))
    ap.add_argument("--ids-4b", default=str(C.base.INDEX_DIR / "doc_ids.json"))
    ap.add_argument("--vectors-8b", default="", help="optional 8B vectors")
    ap.add_argument("--ids-8b", default="")
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

    n = 0
    with Path(args.out).open("w") as f:
        for rec in CP.build_pool(qrecs, r4, r8):
            f.write(json.dumps(rec) + "\n"); n += 1
    print(f"[p0_1] wrote {n} candidate rows for {len(qrecs)} queries -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
