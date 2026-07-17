#!/usr/bin/env python3
"""p1_1 -- prepare the training-label store (train split only; masks/held-out excluded).

The merged labels from p0_3 are already the multi-positive training labels. This stage
filters to the TRAIN query split (excluding the held-out eval queries from p0_4) and
emits a fn_flag sidecar for the hard-filter arm.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dh2 import config2 as C


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", default=str(C.LABELS_DIR / "multi_positive_labels_v1.jsonl"))
    ap.add_argument("--held-out", default=str(C.QRELS_DIR / "held_out_query_ids.json"))
    ap.add_argument("--out", default=str(C.LABELS_DIR / "train_labels_v1.jsonl"))
    ap.add_argument("--fn-flag-out", default=str(C.LABELS_DIR / "fn_flag_v1.json"))
    args = ap.parse_args()

    C.ensure_dirs()
    held = set()
    if Path(args.held_out).exists():
        held = set(json.loads(Path(args.held_out).read_text()))

    # BUG #7 FIX: fn_flag is keyed by QUERY, then document.
    # Was: `fn_flag[r["document_id"]] = True` -- a flat, global {doc_id: bool} map. The
    # flag means "this candidate scored ~as high as THIS query's designated positive",
    # which is a fact about a (query, document) PAIR. Collapsing it to the document
    # banned 38,861 docs as negatives for all 3,000 queries. Arm B then deleted its
    # hardest negatives globally, which is a large part of why "B beats A by 2.2x" did
    # not replicate and in fact reversed.
    fn_flag: dict[str, dict[str, bool]] = {}
    n_in, n_out, n_flags = 0, 0, 0
    with Path(args.out).open("w") as f:
        for l in Path(args.labels).open():
            if not l.strip():
                continue
            n_in += 1
            r = json.loads(l)
            if r["query_id"] in held:
                continue  # keep eval queries out of training (anti-leakage)
            f.write(json.dumps(r) + "\n"); n_out += 1
            if r.get("false_negative_flag"):
                fn_flag.setdefault(r["query_id"], {})[r["document_id"]] = True
                n_flags += 1
    Path(args.fn_flag_out).write_text(json.dumps(fn_flag))
    n_docs = len({d for v in fn_flag.values() for d in v})
    print(f"[p1_1] train labels: {n_out}/{n_in} rows (excluded {len(held)} held-out queries)")
    print(f"[p1_1] fn_flag: {n_flags} (query,doc) pairs over {len(fn_flag)} queries "
          f"({n_docs} distinct docs) -> {args.fn_flag_out}")
    print(f"[p1_1] NOTE: sidecar is now per-query (bug #7). Arm B's hard-filter semantics "
          f"changed: it removes a doc only from the query it was flagged for.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
