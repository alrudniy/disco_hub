#!/usr/bin/env python3
"""p1_4 -- evaluate the Register-Aware Semantic Cascade against its constituents.

Arms (recommendation "New stages/p1_4_eval_cascade.py"):

  C0  Untouched 0.6B alone                     the semantic baseline that beat every
                                               trained arm on low-overlap queries
  C1  Fine-tuned 8B alone                      the domain/lexical specialist
  C2  Union + corrected Qwen                   the ship-critical arm
  C3  Union + corrected Qwen + GroupRank-32B   the gated upside experiment

Also reports ORACLE utility over the full candidate union: the ceiling any reranker could
reach on this pool. If C2 is already near the oracle, further reranking work is wasted and
the remaining headroom is in retrieval.

This stage does NOT train and does NOT promote anything on exact-origin R@10 -- see
dh2.graded_metrics.evaluate_cascade for why that benchmark scores paraphrase retrieval.

Usage (each channel needs vectors embedded with ITS OWN model):

    python stages/p1_4_eval_cascade.py \
        --vectors-base-06b  $DH2_ROOT/vectors/base06b/doc_vectors.npy \
        --ids-base-06b      $DH2_ROOT/vectors/base06b/doc_ids.json \
        --vectors-8b        $DH2_ROOT/vectors/ft8b/doc_vectors.npy \
        --ids-8b            $DH2_ROOT/vectors/ft8b/doc_ids.json \
        --arms C0,C1,C2 \
        --llm-judged-only
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dh2 import config2 as C
from dh2 import graded_metrics as GM
from dh2.candidate_pool import CH_FT_4B, CH_FT_8B, CH_REASON, CH_SEMANTIC
from dh2.cascade import RegisterAwareCascade
from dh2.retriever2 import DenseRetriever


def _load_docs(docs_path: Path):
    from discovery_hub.schema import read_docs
    return list(read_docs(docs_path))


def _judged_query_ids(labels_path: Path) -> set[str]:
    """Query ids with >=1 INDEPENDENT LLM verdict.

    Slice 1 of the anti-circularity design: teacher-consensus labels are 96.6% grade-0 and
    were produced by the same cross-encoders the cascade is built on, so scoring the
    cascade against them measures agreement with itself.
    """
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
    ap.add_argument("--docs", default=str(C.PROD_DOCS))
    ap.add_argument("--utility", default=str(C.QRELS_DIR / "qrels_scout_utility_v1.jsonl"))
    ap.add_argument("--labels", default=str(C.LABELS_DIR / "multi_positive_labels_v1.jsonl"))
    ap.add_argument("--arms", default="C0,C1,C2",
                    help="comma list of C0,C0ft,C1,C1b,C2,C3. C0=stock 0.6B dense; "
                         "C0ft=FINE-TUNED 0.6B dense (same backbone as C0 -- the "
                         "backbone-controlled replication of the bakeoff premise); "
                         "C1=FT-8B dense; C1b=FT-8B top-100 -> shared Qwen rerank "
                         "(isolates union vs reranker); C2=full cascade")
    # per-channel vectors (each embedded with its own model)
    ap.add_argument("--vectors-base-06b", default="", help="UNTOUCHED 0.6B vectors")
    ap.add_argument("--ids-base-06b", default="")
    # FINE-TUNED 0.6B (C0ft). Same backbone as the stock 0.6B, so C0 vs C0ft isolates
    # fine-tuning from model size -- which is exactly what the bakeoff held fixed.
    ap.add_argument("--vectors-ft-06b", default="", help="FINE-TUNED 0.6B vectors (C0ft)")
    ap.add_argument("--ids-ft-06b", default="")
    ap.add_argument("--vectors-8b", default="")
    ap.add_argument("--ids-8b", default="")
    ap.add_argument("--vectors-4b", default=str(C.base.EMB_DIR / "doc_vectors.npy"))
    ap.add_argument("--ids-4b", default=str(C.base.INDEX_DIR / "doc_ids.json"))
    ap.add_argument("--vectors-reasonembed", default="")
    ap.add_argument("--ids-reasonembed", default="")
    ap.add_argument("--device", default=None)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--max-queries", type=int, default=0)
    ap.add_argument("--llm-judged-only", action="store_true",
                    help="score only independently adjudicated queries (recommended)")
    ap.add_argument("--overlap-split", type=float, default=None,
                    help="fixed LOW/HIGH word-overlap split (default: median)")
    ap.add_argument("--name", default="cascade")
    args = ap.parse_args()

    C.ensure_dirs()

    print("[p1_4] loading corpus ...")
    docs = _load_docs(Path(args.docs))
    doc_lookup = {d.doc_id: d for d in docs}
    doc_source = {d.doc_id: getattr(d, "source", "unknown") for d in docs}
    doc_text = {d.doc_id: ((getattr(d, "title", "") or "") + " " +
                           (getattr(d, "embedding_text", None)
                            or getattr(d, "abstract", "") or "")[:1500])
                for d in docs}

    from dh2.identifiers import build_identifier_index
    print("[p1_4] building exact-identifier index ...")
    ident_index = build_identifier_index(docs)
    print(f"[p1_4]   {len(ident_index)} identifiers indexed")

    def _retr(model, vec, ids, query_instruction=None):
        if not vec or not Path(vec).exists():
            return None
        return DenseRetriever(model, vec, ids, device=args.device,
                              query_instruction=query_instruction)

    retrievers = {
        # BUG B: the stock 0.6B carries its OWN query instruction, not the FT models'.
        CH_SEMANTIC: _retr(C.MODEL_BASE_06B, args.vectors_base_06b, args.ids_base_06b,
                           query_instruction=C.QWEN3_BASE_QUERY_INSTRUCTION),
        CH_FT_8B: _retr(C.MODEL_8B, args.vectors_8b, args.ids_8b),
        CH_FT_4B: _retr(C.MODEL_4B, args.vectors_4b, args.ids_4b),
        CH_REASON: _retr(C.MODEL_REASONEMBED, args.vectors_reasonembed,
                         args.ids_reasonembed) if C.MODEL_REASONEMBED else None,
    }
    for ch, r in retrievers.items():
        print(f"[p1_4] channel {ch:12s}: {'ready' if r is not None else 'ABSENT'}")

    # C0ft channel: the FINE-TUNED 0.6B. Deliberately kept OUT of `retrievers` so it never
    # enters C2's union -- C2 must stay byte-comparable to the previous run. It exists to
    # answer exactly one question: with the BACKBONE HELD FIXED at 0.6B (what the bakeoff
    # actually tested), does fine-tuning cost low-overlap utility on independent judgments?
    # C0 vs C0ft is that replication; a 0.6B-vs-8B comparison would confound size with it.
    retr_ft06b = _retr(C.MODEL_06B, args.vectors_ft_06b, args.ids_ft_06b)
    print(f"[p1_4] channel {'ft_06b':12s}: "
          f"{'ready' if retr_ft06b is not None else 'ABSENT'}  (C0ft only; not in union)")
    if retrievers[CH_SEMANTIC] is None:
        print("[p1_4] WARNING: no untouched-0.6B vectors given (--vectors-base-06b). C0 "
              "and the semantic hedge in C2/C3 cannot run. Embed the corpus with "
              f"{C.MODEL_BASE_06B} first (~45 min).")

    judged = _judged_query_ids(Path(args.labels)) if args.llm_judged_only else None
    if judged is not None:
        print(f"[p1_4] restricting to {len(judged)} independently adjudicated queries")

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    reports: dict[str, dict] = {}

    # ONE reranker for every arm. Each RegisterAwareCascade lazily builds its own
    # QwenTeacher if none is passed, so C2 and C3 would otherwise load Qwen3-Reranker-8B
    # TWICE -- 32 GB of BF16 weights instead of 16. On a multi-GPU box device_map="auto"
    # hides that; on a single card it is 16 GB you cannot spare.
    shared_qwen = None
    shared_grouprank = None
    if any(a in ("C1b", "C2", "C3") for a in arms):
        from dh2.teachers import QwenTeacher
        shared_qwen = QwenTeacher()
    if "C3" in arms:
        from dh2.grouprank import GroupRanker
        shared_grouprank = GroupRanker()
        # vLLM reserves gpu_memory_utilization x TOTAL VRAM, not x FREE VRAM. With the
        # encoders + reranker already resident, the default 0.90 will OOM on a single
        # card. Budget it against what is actually left.
        need = (0.6 + 4 + 8 + 8) * 2          # encoders + reranker, GB of BF16 weights
        weights = 32 * 2                       # GroupRank-32B
        print(f"[p1_4] single-process C3: ~{need:.0f} GB already resident + {weights:.0f} GB "
              f"GroupRank weights. If vLLM OOMs, lower DH2_GR_GPU_UTIL "
              f"(gpu_memory_utilization={C.CASCADE.group_rank_gpu_util}) or run C3 in a "
              f"separate process. TP={C.CASCADE.group_rank_tp_size}.")

    for arm in arms:
        print(f"\n[p1_4] ===== arm {arm} =====")
        latencies: dict[str, float] = {}

        if arm in ("C0", "C0ft", "C1"):
            # Dense-only arms: one retriever, no rerank. C0ft shares C0's backbone.
            if arm == "C0ft":
                ch, retr = "ft_06b", retr_ft06b
            else:
                ch = CH_SEMANTIC if arm == "C0" else CH_FT_8B
                retr = retrievers.get(ch)
            if retr is None:
                print(f"[p1_4] {arm}: channel {ch} unavailable, skipping")
                continue

            def rank_fn(q, _r=retr):
                t0 = time.perf_counter()
                out = [d for d, _ in _r.search(q, top_k=args.top_k)]
                latencies[q] = (time.perf_counter() - t0) * 1000
                return out

            def cand_fn(q, _r=retr):
                return [d for d, _ in _r.search(q, top_k=100)]
        elif arm == "C1b":
            # C1b = C1's candidates, C2's reranker. Same 8B top-100 as C1, but ordered by
            # the shared Qwen reranker instead of dense score. C1b vs C1 isolates the
            # RERANKER; C2 vs C1b isolates the UNION. Built as a single-channel cascade so
            # it reuses the exact rerank path C2 uses (same shared_qwen, same margin
            # ordering). identifier_index=None keeps it a pure 8B->rerank arm.
            if retrievers.get(CH_FT_8B) is None:
                print(f"[p1_4] {arm}: channel {CH_FT_8B} unavailable, skipping")
                continue
            cas_1b = RegisterAwareCascade(
                {CH_FT_8B: retrievers[CH_FT_8B]}, doc_lookup,
                identifier_index=None,
                qwen_teacher=shared_qwen,
                group_ranker=None,
                enable_grouprank=False)

            def rank_fn(q, _c=cas_1b):
                t0 = time.perf_counter()
                out = _c.rank_ids(q, top_k=args.top_k)
                latencies[q] = (time.perf_counter() - t0) * 1000
                return out

            def cand_fn(q, _c=cas_1b):
                return list(_c.build_candidate_union(q))
        else:
            cas = RegisterAwareCascade(
                retrievers, doc_lookup,
                identifier_index=ident_index,
                qwen_teacher=shared_qwen,
                group_ranker=shared_grouprank,
                enable_grouprank=(arm == "C3"))

            def rank_fn(q, _c=cas):
                t0 = time.perf_counter()
                out = _c.rank_ids(q, top_k=args.top_k)
                latencies[q] = (time.perf_counter() - t0) * 1000
                return out

            def cand_fn(q, _c=cas):
                return list(_c.build_candidate_union(q))

        rep = GM.evaluate_cascade(
            args.utility, rank_fn,
            doc_text=doc_text, doc_source=doc_source, candidate_fn=cand_fn,
            llm_judged_query_ids=judged, overlap_split=args.overlap_split,
            k=args.top_k, latency_ms=latencies)
        reports[arm] = rep
        low = rep.get("low_overlap", {}).get(f"ndcg@{args.top_k}", {}).get("mean", 0)
        high = rep.get("high_overlap", {}).get(f"ndcg@{args.top_k}", {}).get("mean", 0)
        print(f"[p1_4] {arm}: nDCG@{args.top_k} "
              f"overall={rep.get('overall', {}).get(f'ndcg@{args.top_k}', {}).get('mean', 0):.4f} "
              f"LOW={low:.4f} HIGH={high:.4f}")

    out_json = C.REPORTS2_DIR / f"eval_{args.name}.json"
    out_json.write_text(json.dumps(reports, indent=2))

    # ---- report ---------------------------------------------------------- #
    k = args.top_k
    md = [f"# Cascade eval: {args.name}", "",
          "Scored on graded utility, bucketed by query/document word overlap. "
          "Exact-origin R@10 is deliberately NOT the promotion criterion: it measures "
          "paraphrase retrieval.", ""]
    if judged is not None:
        md.append(f"Slice: independently LLM-adjudicated queries only (n={len(judged)}).")
        md.append("")
    md += ["| arm | nDCG@%d | LOW nDCG | HIGH nDCG | Success@%d | grade-3 MRR | cand R@100 "
           "| oracle nDCG@%d | headroom | p95 ms |" % (k, k, k),
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for arm, rep in reports.items():
        o = rep.get("overall", {})
        lo = rep.get("low_overlap", {})
        hi = rep.get("high_overlap", {})
        ndcg = o.get(f'ndcg@{k}', {}).get('mean', 0)
        oracle = o.get(f'oracle_ndcg@{k}', {}).get('mean', 0)
        # headroom = what a PERFECT reranker would add on top of this arm's own pool.
        # Large headroom => the docs are already retrieved and the ranking stage is the
        # bottleneck. Small headroom => the work belongs back in the union.
        md.append(
            f"| {arm} "
            f"| {ndcg:.4f} "
            f"| {lo.get(f'ndcg@{k}', {}).get('mean', 0):.4f} "
            f"| {hi.get(f'ndcg@{k}', {}).get('mean', 0):.4f} "
            f"| {o.get(f'success@{k}', {}).get('mean', 0):.4f} "
            f"| {o.get(f'grade3_mrr@{k}', {}).get('mean', 0):.4f} "
            f"| {o.get('candidate_recall@100', {}).get('mean', 0):.4f} "
            f"| {oracle:.4f} "
            f"| {oracle - ndcg:+.4f} "
            f"| {rep.get('latency_ms', {}).get('p95', 0):.0f} |")

    # ---- promotion gates -------------------------------------------------- #
    md += ["", "## Promotion gates", ""]
    base_arm = "C1" if "C1" in reports else ("C0" if "C0" in reports else None)
    if "C2" in reports and base_arm:
        md += _gate_block("C2", reports["C2"], base_arm, reports[base_arm], k)
    if "C3" in reports and "C2" in reports:
        md += _gate_block("C3", reports["C3"], "C2", reports["C2"], k)
    if not base_arm:
        md.append("No constituent arm evaluated; cannot check gates.")

    out_md = C.REPORTS2_DIR / f"eval_{args.name}.md"
    out_md.write_text("\n".join(md))
    print("\n" + "\n".join(md))
    print(f"\n[p1_4] reports -> {out_json} , {out_md}")
    return 0


def _gate_block(arm: str, rep: dict, base_name: str, base: dict, k: int) -> list[str]:
    """Render the gate table for `arm` vs `base_name`. Gates from config2.CascadeGates."""
    g = C.CASCADE_GATES

    def _m(r, slice_, name):
        return r.get(slice_, {}).get(name, {}).get("mean", 0.0)

    d_low = _m(rep, "low_overlap", f"ndcg@{k}") - _m(base, "low_overlap", f"ndcg@{k}")
    d_mrr = _m(rep, "overall", f"grade3_mrr@{k}") - _m(base, "overall", f"grade3_mrr@{k}")
    d_ndcg = _m(rep, "overall", f"ndcg@{k}") - _m(base, "overall", f"ndcg@{k}")
    d_rec = (_m(rep, "overall", "candidate_recall@100")
             - _m(base, "overall", "candidate_recall@100"))

    worst_src, worst_delta = None, 0.0
    for src, vals in rep.get("per_source", {}).items():
        b = base.get("per_source", {}).get(src, {})
        if not b:
            continue
        delta = (vals.get(f"success@{k}", {}).get("mean", 0)
                 - b.get(f"success@{k}", {}).get("mean", 0)) * 100
        if delta < worst_delta:
            worst_src, worst_delta = src, delta

    checks = [
        ("low-overlap utility does not regress",
         d_low >= -g.max_low_overlap_utility_regression, f"{d_low:+.4f}"),
        ("grade-3 MRR does not regress",
         d_mrr >= -g.max_grade3_mrr_regression, f"{d_mrr:+.4f}"),
        ("graded nDCG@%d improves" % k,
         d_ndcg > g.min_delta_ndcg10_utility, f"{d_ndcg:+.4f}"),
        ("no source loses > %.0f pp" % g.max_source_regression_pp,
         worst_delta >= -g.max_source_regression_pp,
         f"{worst_src or 'n/a'} {worst_delta:+.1f} pp"),
        ("candidate Recall@100 >= constituent",
         d_rec >= 0 if g.require_union_recall_at_least_constituents else True,
         f"{d_rec:+.4f}"),
    ]
    if arm == "C3":
        p95 = rep.get("latency_ms", {}).get("p95", 0)
        checks.append(("p95 latency within budget",
                       p95 <= g.grouprank_max_p95_latency_s * 1000,
                       f"{p95:.0f} ms"))

    lines = [f"### {arm} vs {base_name}", "",
             "| gate | value | result |", "|---|---:|---|"]
    for name, ok, val in checks:
        lines.append(f"| {name} | {val} | {'PASS' if ok else '**FAIL**'} |")
    verdict = "PROMOTE" if all(ok for _, ok, _ in checks) else "DO NOT PROMOTE"
    lines += ["", f"**{arm}: {verdict}**", ""]
    return lines


if __name__ == "__main__":
    sys.exit(main())
