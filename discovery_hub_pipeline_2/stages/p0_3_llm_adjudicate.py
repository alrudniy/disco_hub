#!/usr/bin/env python3
"""p0_3 -- merge two cross-encoder teachers + GLM-4.6 (z.ai) gray-zone judge -> merged
graded labels.

Two-phase design for SPEED (z.ai has a fat latency tail, ~2-20s/call):
  Phase 1 (PARALLEL): identify the gray-zone pairs that need an LLM verdict and fetch
    them CONCURRENTLY with a thread pool, filling the on-disk cache. A budget bounds the
    number of NEW calls for a proof run; pairs beyond the budget stay uncached.
  Phase 2 (SERIAL, fast): run the normal merge over every pair. LLM-bound pairs are now
    cache hits (instant); pairs left uncached by the budget get masked by merge_one.

This keeps merge_one's decision logic race-free (all concurrency is confined to the pure
"fetch verdicts" phase, synchronized through the thread-safe cache).
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dh2 import config2 as C
from dh2 import teacher_merge as TM
from dh2.llm_client import TeacherLLM
from discovery_hub.schema import read_docs


def _doc_maps(docs_path: Path):
    title, text, src = {}, {}, {}
    for d in read_docs(docs_path):
        title[d.doc_id] = d.title
        text[d.doc_id] = d.embedding_text or f"{d.title}\n{d.abstract}"
        src[d.doc_id] = d.source
    return title, text, src


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scored", default=str(C.TEACHER_DIR / "teacher_scores_v1.jsonl"))
    ap.add_argument("--docs", default=str(C.PROD_DOCS))
    ap.add_argument("--out", default=str(C.LABELS_DIR / "multi_positive_labels_v1.jsonl"))
    ap.add_argument("--cache", default=str(C.ADJUD_DIR / "llm_cache.jsonl"))
    ap.add_argument("--skip-llm", action="store_true", help="mask gray-zone instead of LLM")
    ap.add_argument("--max-llm-calls", type=int, default=0,
                    help="cap NEW LLM judge calls (0=unlimited). Beyond the cap, gray-zone "
                         "pairs are masked. Use for bounded proof runs.")
    ap.add_argument("--workers", type=int, default=16,
                    help="parallel LLM requests in the prefetch phase (z.ai fat tail -> "
                         "overlap many; 16-32 is a good range).")
    args = ap.parse_args()

    C.ensure_dirs()
    rows = [json.loads(l) for l in Path(args.scored).open() if l.strip()]
    title, text, src = _doc_maps(Path(args.docs))
    print(f"[p0_3] {len(rows)} scored pairs, {len(title)} docs loaded")

    # positive teacher score per query (max over designated positives), for FN flagging
    pos_score = {}
    for r in rows:
        if r.get("is_designated_positive"):
            s = max(r.get("bge_score", 0.0), r.get("qwen_score") or 0.0)
            pos_score[r["query_id"]] = max(pos_score.get(r["query_id"], 0.0), s)

    llm = None
    if not args.skip_llm:
        llm = TeacherLLM(cache_path=Path(args.cache))
        if not llm.health_check():
            print("[p0_3] WARNING: LLM health check failed; masking gray-zone instead.")
            llm = None

    # -------- Phase 1: PARALLEL prefetch of gray-zone verdicts -------- #
    if llm is not None:
        jobs = []
        for r in rows:
            if TM.needs_llm(float(r.get("bge_score", 0.0)), r.get("qwen_score"),
                            bool(r.get("is_designated_positive"))):
                did = r["document_id"]
                jobs.append({"query": r["query"],
                             "doc_title": title.get(did, ""),
                             "doc_text": text.get(did, ""),
                             "source": r.get("source", src.get(did, ""))})
        print(f"[p0_3] gray-zone pairs needing LLM: {len(jobs)}")
        if args.max_llm_calls > 0:
            print(f"[p0_3] NEW-call budget: {args.max_llm_calls} "
                  f"(gray-zone beyond this is masked)")
        stats = llm.prefetch_parallel(jobs, workers=args.workers,
                                      budget=args.max_llm_calls)
        print(f"[p0_3] prefetch done: {stats}")

    # -------- Phase 2: SERIAL merge (LLM-bound pairs read cache, NO new calls) -------- #
    # Critical: use a CACHE-ONLY view so the merge never re-fetches the ~budget-skipped
    # gray-zone pairs one-at-a-time (that would silently redo all the work serially).
    # Anything the parallel prefetch didn't cache (beyond budget, or errored) gets masked.
    merge_llm = llm.as_cache_only() if llm is not None else None
    n, n_llm, n_masked, n_fn = 0, 0, 0, 0
    with Path(args.out).open("w") as f:
        for r in rows:
            lab = TM.merge_one(
                query_id=r["query_id"], query=r["query"], document_id=r["document_id"],
                source=r.get("source", src.get(r["document_id"], "")),
                doc_title=title.get(r["document_id"], ""),
                doc_text=text.get(r["document_id"], ""),
                is_designated_positive=bool(r.get("is_designated_positive")),
                bge_score=float(r.get("bge_score", 0.0)),
                qwen_score=r.get("qwen_score"),
                positive_teacher_score=pos_score.get(r["query_id"]),
                llm=merge_llm)
            rec = lab.__dict__
            f.write(json.dumps(rec) + "\n"); n += 1
            n_llm += int(lab.llm_grade is not None)
            n_masked += int(lab.masked)
            n_fn += int(lab.false_negative_flag)
    print(f"[p0_3] wrote {n} labels ({n_llm} LLM-judged, {n_masked} masked, "
          f"{n_fn} false-neg-flagged) -> {args.out}")
    # HONEST provenance: only claim LLM adjudication if the LLM actually judged pairs.
    if n_llm > 0:
        provenance = "LLM-adjudicated (gray-zone) + cross-encoder, not human-calibrated"
    else:
        provenance = ("CROSS-ENCODER-ONLY (LLM judge did NOT run -- gray-zone masked); "
                      "NOT LLM-adjudicated")
    print(f"[p0_3] false-negative flag rate: {100*n_fn/max(n,1):.1f}%  [{provenance}]")
    if n_llm == 0 and llm is not None:
        print("[p0_3] ERROR: LLM was configured but judged 0 pairs -- check the health "
              "check / key. Re-run this stage after fixing the LLM before trusting labels.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
