#!/usr/bin/env python3
"""p0_4 -- build exact-origin + graded scout-utility qrels from merged labels."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dh2 import config2 as C
from dh2 import eval_pool_build as EP


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", default=str(C.LABELS_DIR / "multi_positive_labels_v1.jsonl"))
    ap.add_argument("--out-exact", default=str(C.QRELS_DIR / "qrels_exact_origin_v1.jsonl"))
    ap.add_argument("--out-utility", default=str(C.QRELS_DIR / "qrels_scout_utility_v1.jsonl"))
    ap.add_argument("--eval-frac", type=float, default=0.15)
    ap.add_argument("--held-out-only", action="store_true",
                    help="restrict qrels to a deterministic held-out query split (anti-leakage)")
    args = ap.parse_args()

    C.ensure_dirs()
    qids = []
    seen = set()
    for l in Path(args.labels).open():
        if l.strip():
            qid = json.loads(l)["query_id"]
            if qid not in seen:
                seen.add(qid); qids.append(qid)
    held = None
    if args.held_out_only:
        _, held = EP.split_held_out(qids, args.eval_frac)
        # persist the split so training excludes these query_ids
        (C.QRELS_DIR / "held_out_query_ids.json").write_text(json.dumps(sorted(held)))
        print(f"[p0_4] held-out eval queries: {len(held)} / {len(qids)}")

    summary = EP.build_qrels(args.labels, args.out_exact, args.out_utility,
                             held_out_query_ids=held)
    print(f"[p0_4] {summary}")
    print(f"[p0_4] wrote {args.out_exact} + {args.out_utility}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
