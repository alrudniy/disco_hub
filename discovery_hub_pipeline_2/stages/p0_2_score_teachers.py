#!/usr/bin/env python3
"""p0_2 -- score every (query, candidate) pair with BGE + Qwen3-Reranker-8B."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dh2 import config2 as C
from dh2.teachers import BgeTeacher, QwenTeacher
from discovery_hub.schema import read_docs


def _doc_text_map(docs_path: Path) -> dict:
    return {d.doc_id: (d.embedding_text or f"{d.title}\n{d.abstract}") for d in read_docs(docs_path)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", default=str(C.POOL_DIR / "candidate_pool_v1.jsonl"))
    ap.add_argument("--docs", default=str(C.PROD_DOCS))
    ap.add_argument("--out", default=str(C.TEACHER_DIR / "teacher_scores_v1.jsonl"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--skip-qwen", action="store_true", help="BGE only (skip 8B teacher)")
    args = ap.parse_args()

    C.ensure_dirs()
    rows = [json.loads(l) for l in Path(args.pool).open() if l.strip()]
    texts = _doc_text_map(Path(args.docs))
    pairs = [(r["query"], texts.get(r["document_id"], "")) for r in rows]

    bge = BgeTeacher(device=args.device)
    print(f"[p0_2] scoring {len(pairs)} pairs with BGE ...")
    bge_scores = bge.score(pairs)
    qwen_scores = [None] * len(pairs)
    if not args.skip_qwen:
        qwen = QwenTeacher(device=args.device)
        print(f"[p0_2] scoring {len(pairs)} pairs with Qwen3-Reranker-8B ...")
        qwen_scores = qwen.score(pairs)

    n = 0
    with Path(args.out).open("w") as f:
        for r, b, q in zip(rows, bge_scores, qwen_scores):
            r = dict(r); r["bge_score"] = round(float(b), 5)
            r["qwen_score"] = None if q is None else round(float(q), 5)
            f.write(json.dumps(r) + "\n"); n += 1
    print(f"[p0_2] wrote {n} scored rows -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
