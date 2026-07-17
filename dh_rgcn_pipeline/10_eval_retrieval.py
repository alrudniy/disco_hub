#!/usr/bin/env python3
"""
10_eval_retrieval.py  --  LAYER 2 QUALITY EVALUATION (the metric the deck sells).

Measures retrieval quality (Recall@k, MRR@10, nDCG@10, Precision@10, MAP) and --
crucially -- A/Bs two systems on the SAME queries so you can see whether a change
actually helps, with a 95% bootstrap confidence interval and a paired Wilcoxon
p-value on every delta. The default A/B is:

    text_only   : dense retrieval, graph_weight = 0
    text+graph  : dense retrieval blended with the R-GCN graph signal

so the first question this answers is "does the graph layer earn its keep?".
(The same machinery compares a fine-tuned encoder vs. the base model: run 04 into
two embedding dirs, build two indexes, and pass two retrievers to run_eval().)

  TARGET: Drew or Anvil; CPU is fine. Logically runs any time after stage 06
          (it needs the index from 05 and, for the graph arm, the R-GCN from 06).
          Numbered 10 so the existing stages keep their numbers.

GROUND TRUTH
  --qrels PATH   evaluate against a curated/judged set (JSONL: query_id, query,
                 relevant_doc_ids). This is the trustworthy path.
  (default)      build a deterministic synthetic set: each query is generated from
                 a known source document (Promptagator/InPars), which is its gold
                 positive. Honest caveat: ONE positive per query, so Precision@10
                 is capped at 0.1 -- read Recall@k and MRR. Curated qrels with
                 several positives per query make Precision@10 meaningful.

Outputs:
  data/reports/eval_report.json   full per-system + delta results
  data/reports/eval_report.md     a readable scorecard
  data/reports/eval_qrels.jsonl   the eval set used (synthetic mode)

Usage:
  python 10_eval_retrieval.py --mock --num-queries 200
  python 10_eval_retrieval.py --qrels data/curated_qrels.jsonl     # real labels
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from discovery_hub import config
from discovery_hub.determinism import set_global_determinism
from discovery_hub import evaluate as ev
from discovery_hub.schema import read_docs


def _load(name_py: str, mod_name: str):
    """Import a digit-prefixed sibling stage by path."""
    path = Path(__file__).with_name(name_py)
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_eval(systems, eval_queries, ks=ev.DEFAULT_KS, main_k: int = ev.MAIN_K,
             depth: int | None = None) -> dict:
    """
    systems: list of (name, retriever, retrieve_kwargs).
    eval_queries: list of ev.EvalQuery.
    Returns a structured results dict (per-system aggregates + pairwise deltas).
    Retrieval depth defaults to max(ks) so Recall@max_k is well defined.
    """
    depth = depth or max(ks)
    top_k = max(depth, config.RETRIEVAL.top_k_recall)

    # Collect per-query metric arrays for each system, aligned by query order.
    per_system: dict[str, list[dict]] = {name: [] for name, _, _ in systems}
    for q in eval_queries:
        for name, retr, kwargs in systems:
            ranked = [c["doc_id"] for c in
                      retr.retrieve(q.query, top_k=top_k, rerank_k=depth, **kwargs)]
            per_system[name].append(
                ev.evaluate_query(ranked, q.relevant_doc_ids, ks=ks, main_k=main_k))

    names = ev.metric_names(ks, main_k)
    aggregates = {}
    for name, _, _ in systems:
        arrs = ev.aggregate(per_system[name], ks=ks, main_k=main_k)
        aggregates[name] = {
            m: dict(zip(("mean", "ci_low", "ci_high"), ev.bootstrap_ci(arrs[m])))
            for m in names
        }

    # Pairwise deltas vs. the first system as baseline.
    comparisons = []
    # Marginal comparisons: each system vs. the PREVIOUS one, so each row isolates
    # the contribution of one added signal (hybrid over dense; +graph over hybrid).
    for i in range(1, len(systems)):
        base_name = systems[i - 1][0]
        cand_name = systems[i][0]
        base_arrs = ev.aggregate(per_system[base_name], ks=ks, main_k=main_k)
        cand_arrs = ev.aggregate(per_system[cand_name], ks=ks, main_k=main_k)
        deltas = {m: ev.paired_delta(base_arrs[m], cand_arrs[m]) for m in names}
        comparisons.append({"baseline": base_name, "candidate": cand_name,
                            "deltas": deltas})

    return {"systems": [n for n, _, _ in systems], "metrics": names,
            "num_queries": len(eval_queries), "main_k": main_k,
            "aggregates": aggregates, "comparisons": comparisons}


def to_markdown(report: dict, source: str) -> str:
    names = report["metrics"]
    L = ["# Discovery Hub -- Retrieval Quality Report",
         f"_Generated {report['generated']} | mode={'mock' if report['mock'] else 'real'} "
         f"| queries={report['num_queries']} | labels={source}_", ""]
    L += ["## What this measures",
          "Whether retrieval returns the right documents, and whether a change "
          "**helps** -- every comparison carries a 95% bootstrap confidence "
          "interval and a paired Wilcoxon p-value, so a lift is reported as "
          "`+X.XXX [lo, hi], p=...` rather than asserted.", ""]
    if "one positive per query" in source:
        L += ["> Note: synthetic labels have **one positive per query**, so "
              "Precision@10 is capped at 0.10 -- read Recall@k and MRR@10 here. "
              "Curated qrels with multiple positives make Precision@10 meaningful.", ""]

    # Per-system scorecard.
    L += ["## Scorecard (mean [95% CI])", ""]
    header = "| Metric | " + " | ".join(report["systems"]) + " |"
    L += [header, "|" + "---|" * (len(report["systems"]) + 1)]
    for m in names:
        cells = []
        for s in report["systems"]:
            a = report["aggregates"][s][m]
            cells.append(f"{a['mean']:.3f} [{a['ci_low']:.3f}, {a['ci_high']:.3f}]")
        L.append(f"| {m} | " + " | ".join(cells) + " |")
    L.append("")

    # Deltas.
    for comp in report["comparisons"]:
        L += [f"## Does it help? {comp['candidate']} vs {comp['baseline']}", ""]
        if comp["candidate"] == "hybrid+graph" and "graph_fire_rate" in report:
            L += [f"_The entity-linked graph signal fired on "
                  f"{report['graph_fired']}/{report['num_queries']} queries "
                  f"({report['graph_fire_rate']*100:.0f}%) and abstained on the "
                  f"rest; on abstaining queries hybrid+graph == hybrid by "
                  f"construction, which dilutes any average effect._", ""]
        L += ["| Metric | Δ (cand − base) | 95% CI | p | verdict |",
              "|---|---|---|---|---|"]
        for m in names:
            d = comp["deltas"][m]
            p = "n/a" if d["p_value"] is None else f"{d['p_value']:.3f}"
            sig = (d["p_value"] is not None and d["p_value"] < 0.05)
            if d["mean_delta"] > 0 and sig:
                verdict = "helps ✓"
            elif d["mean_delta"] < 0 and sig:
                verdict = "hurts ✗"
            else:
                verdict = "no sig. diff"
            L.append(f"| {m} | {d['mean_delta']:+.3f} | "
                     f"[{d['ci_low']:+.3f}, {d['ci_high']:+.3f}] | {p} | {verdict} |")
        L.append("")

    L += ["## How to read this",
          "- **Recall@k**: did the right document make the top-k. The headline "
          "quality number for a retrieve-then-rerank system.",
          "- **MRR@10 / nDCG@10**: how high the right document ranked.",
          "- **Δ with CI + p**: the honest form of \"+20% Precision@10\" -- a "
          "measured lift with uncertainty, on the same queries, paired. A wide CI "
          "that crosses 0 means \"not enough evidence yet\", not \"better\".",
          "- Swap in **curated qrels** (`--qrels`) and re-run to report numbers "
          "you can put in front of an investor.", ""]
    L += ["## Reproducibility configuration in force", "```json",
          json.dumps(report["determinism_config"], indent=2), "```"]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--num-queries", type=int, default=200,
                    help="synthetic eval-set size (ignored when --qrels is given)")
    ap.add_argument("--qrels", default=None,
                    help="path to a curated qrels JSONL (real relevance labels)")
    ap.add_argument("--depth", type=int, default=max(ev.DEFAULT_KS),
                    help="retrieval depth (>= max recall cutoff)")
    args = ap.parse_args()

    cfg = set_global_determinism(config.SEED)
    config.ensure_dirs()

    rr = _load("07_retrieve_rank.py", "retrieve_rank")
    retriever = rr.Retriever(mock=args.mock)

    # Build or load the evaluation set.
    if args.qrels:
        eval_queries = ev.load_qrels(args.qrels)
        source = f"curated ({Path(args.qrels).name})"
    else:
        docs = list(read_docs(config.NORM_DIR / "docs.jsonl"))
        eval_queries = ev.build_synthetic_eval(docs, args.num_queries, seed=config.SEED)
        ev.write_qrels(eval_queries, config.REPORT_DIR / "eval_qrels.jsonl")
        source = "synthetic (one positive per query)"
    if not eval_queries:
        print("No evaluation queries (empty corpus / qrels). Run stages 01-05 first.")
        return 1

    # Systems compared marginally (each adds one signal):
    #   dense        -> dense embeddings only
    #   hybrid       -> dense + BM25 keyword (the "hybrid" claim)
    #   hybrid+graph -> + entity-linked graph signal (abstains per-query)
    # The graph arm needs the R-GCN artifacts; drop it if absent and say so.
    systems = [("dense", retriever, {"use_keyword": False, "use_graph": False}),
               ("hybrid", retriever, {"use_keyword": True, "use_graph": False})]
    if retriever.graph_emb is not None:
        systems.append(("hybrid+graph", retriever,
                        {"use_keyword": True, "use_graph": True}))
    else:
        print("  [warn] no R-GCN node embeddings found (run stage 06) -- "
              "comparing dense vs hybrid only, no graph arm.")

    print(f"Evaluating {len(systems)} system(s) on {len(eval_queries)} queries "
          f"[{source}] ...")
    report = run_eval(systems, eval_queries, depth=args.depth)

    # How often did the entity-linked graph signal actually fire vs. abstain?
    # This is the honest scope of the graph: it only contributes when the query
    # names an entity present in the graph.
    graph_fired = 0
    if retriever.graph_emb is not None:
        from discovery_hub import entity_link
        for q in eval_queries:
            if entity_link.link_query(q.query, retriever.surface_index):
                graph_fired += 1
    fire_rate = graph_fired / len(eval_queries) if eval_queries else 0.0

    report.update({"generated": datetime.now().isoformat(timespec="seconds"),
                   "mock": args.mock, "labels": source,
                   "graph_fire_rate": fire_rate, "graph_fired": graph_fired,
                   "determinism_config": cfg})

    (config.REPORT_DIR / "eval_report.json").write_text(json.dumps(report, indent=2))
    md = to_markdown(report, source)
    (config.REPORT_DIR / "eval_report.md").write_text(md)
    print("\n" + md)
    print(f"\nWrote {config.REPORT_DIR / 'eval_report.json'}, .md, and eval_qrels.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
