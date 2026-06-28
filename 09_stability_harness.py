#!/usr/bin/env python3
"""
09_stability_harness.py  --  REPRODUCIBILITY / STABILITY QA (the investor demo).

Runs the same queries K times and measures how stable the system is at each stage,
then writes a report. This operationalizes the honest answer to "ask the same
question twice -- do you get the same output?":

  * BYTE-LEVEL determinism: measured and reported, but NOT promised across
    hardware/library changes (float non-associativity, batch-variant kernels).
  * SEMANTIC / FUNCTIONAL stability: the evidence, the retrieved set, and the
    citations are stable -- and that is what makes the recommendation trustworthy.

  METRICS
    embedding:  max cosine drift across runs; exact-match rate
    retrieval:  top-k set Jaccard overlap; Kendall's tau (rank correlation);
                exact ordered-match rate
    explanation:exact-match rate of prose; semantic-equivalence (output-embedding
                cosine); citation-set Jaccard (the one that matters most)

  TARGET: run anywhere, but PIN the hardware + library versions when you quote a
          number -- the determinism_report() block records what was in force.

Usage:
  python 09_stability_harness.py --mock --runs 5
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

from discovery_hub import config
from discovery_hub import faithfulness as F
from discovery_hub.determinism import set_global_determinism
from discovery_hub.embedding import get_embedder

FAITH_THRESHOLD = 0.5   # lexical-proxy groundedness cutoff (see faithfulness.py)

DEFAULT_QUERIES = [
    "EGFR inhibitor for non-small-cell lung cancer",
    "GLP-1 receptor agonist for metabolic disease",
    "monoclonal antibody targeting PD-L1",
    "blood-brain-barrier penetrant for neurodegeneration",
    "broad-spectrum antiviral protease inhibitor",
]


def _load(name_py: str, mod_name: str):
    path = Path(__file__).with_name(name_py)
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def jaccard(a, b) -> float:
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a | b) else 1.0


def measure_embeddings(queries, mock, runs) -> dict:
    emb = get_embedder(mock=mock)
    # These are queries, so exercise the query path (instructed in real mode).
    mats = [emb.encode_queries(queries) for _ in range(runs)]
    ref = mats[0]
    max_drift = 0.0
    exact = True
    for m in mats[1:]:
        # cosine drift per row = 1 - dot (vectors are normalized)
        drift = float(np.max(1.0 - np.sum(ref * m, axis=1)))
        max_drift = max(max_drift, abs(drift))
        exact = exact and np.array_equal(ref, m)
    return {"runs": runs, "max_cosine_drift": max_drift,
            "exact_match_all_runs": bool(exact)}


def measure_encoding_robustness(texts, mock, batch_sizes, devices) -> dict:
    """
    The HARD case the same-condition repeat loop above cannot see: vary the BATCH
    SIZE (and, if given, the DEVICE) and measure embedding drift against a
    full-batch reference. The memo identifies batch-variant GPU kernels as the
    dominant source of run-to-run variation in production, where requests are
    batched under fluctuating load -- so a stability number measured at a single
    fixed batch size is optimistic.

    In mock the embedder is batch- and device-invariant by construction, so drift
    is exactly 0 -- reported honestly. In real mode this is the test that surfaces
    the production drift; nonzero values here are expected and are the number to
    quote (alongside the fixed-condition near-zero), not hidden.
    """
    base = get_embedder(mock=mock, device=devices[0])
    ref = base.encode_queries(texts, batch_size=len(texts))   # full batch, device[0]

    def drift_vs_ref(m):
        return float(np.max(1.0 - np.sum(ref * m, axis=1)))

    bs_rows = []
    for bs in batch_sizes:
        m = base.encode_queries(texts, batch_size=bs)
        bs_rows.append({"batch_size": bs, "max_cosine_drift_vs_full": drift_vs_ref(m),
                        "exact": bool(np.array_equal(ref, m))})

    dev_rows = []
    for dev in devices:
        emb = base if dev == devices[0] else get_embedder(mock=mock, device=dev)
        m = emb.encode_queries(texts, batch_size=len(texts))
        dev_rows.append({"device": dev, "max_cosine_drift_vs_ref": drift_vs_ref(m),
                         "exact": bool(np.array_equal(ref, m))})

    return {
        "reference": f"full batch (size {len(texts)}) on {devices[0]}",
        "batch_sizes_tested": list(batch_sizes),
        "per_batch_size": bs_rows,
        "max_drift_across_batch_sizes": max((r["max_cosine_drift_vs_full"]
                                             for r in bs_rows), default=0.0),
        "batch_invariant": all(r["exact"] for r in bs_rows),
        "devices_tested": list(devices),
        "per_device": dev_rows,
        "device_invariant": all(r["exact"] for r in dev_rows),
        "note": ("mock is batch/device-invariant by construction (drift 0); in real "
                 "mode nonzero drift here is expected and is the honest production "
                 "signal that fixed-condition repeats hide."),
    }


def measure_retrieval(retriever, queries, runs, k) -> dict:
    per_query = []
    for q in queries:
        results = [[c["doc_id"] for c in retriever.retrieve(q, rerank_k=k)]
                   for _ in range(runs)]
        # pairwise Jaccard of the result sets
        jacs, taus, exacts = [], [], []
        for a, b in combinations(results, 2):
            jacs.append(jaccard(a, b))
            exacts.append(1.0 if a == b else 0.0)
            # Kendall tau over the rank of shared items.
            common = [x for x in a if x in b]
            if len(common) > 1:
                ra = [a.index(x) for x in common]
                rb = [b.index(x) for x in common]
                t = kendalltau(ra, rb).correlation
                if t == t:  # not NaN
                    taus.append(t)
        per_query.append({
            "query": q,
            "mean_jaccard": float(np.mean(jacs)) if jacs else 1.0,
            "mean_kendall_tau": float(np.mean(taus)) if taus else 1.0,
            "exact_order_match_rate": float(np.mean(exacts)) if exacts else 1.0,
        })
    return {
        "runs": runs, "k": k, "per_query": per_query,
        "mean_jaccard": float(np.mean([p["mean_jaccard"] for p in per_query])),
        "mean_kendall_tau": float(np.mean([p["mean_kendall_tau"] for p in per_query])),
        "exact_order_match_rate": float(np.mean([p["exact_order_match_rate"]
                                                 for p in per_query])),
    }


def measure_explanation(rag_mod, retriever, queries, mock, runs) -> dict:
    emb = get_embedder(mock=mock)  # to score semantic equivalence of outputs
    text_by_doc = {d.doc_id: f"{d.title} {d.abstract}"
                   for d in retriever.docs.values()}   # evidence for faithfulness
    per_query = []
    for q in queries:
        outs = [rag_mod.run_pipeline(q, mock=mock, retriever=retriever)
                for _ in range(runs)]
        proses = ["\n".join(r["why"] for r in o["recommendations"]) for o in outs]
        cites = [frozenset(r["citation"] for r in o["recommendations"]) for o in outs]
        exacts, sems, cjac = [], [], []
        # embed prose once per run for semantic-equivalence comparison
        prose_vecs = emb.encode([p if p else " " for p in proses])
        for (i, a), (j, b) in combinations(enumerate(proses), 2):
            exacts.append(1.0 if a == b else 0.0)
            sems.append(float(np.dot(prose_vecs[i], prose_vecs[j])))
            cjac.append(jaccard(cites[i], cites[j]))
        # Faithfulness of the generated answers: do the claims stay grounded in
        # cited evidence, and are the citations valid? (Stability above asks if the
        # SAME sources are cited run-to-run; this asks if they are CORRECT.)
        fa = [F.evaluate_recommendations(o["recommendations"], text_by_doc,
                                         context=q, threshold=FAITH_THRESHOLD)
              for o in outs]
        answered = any(o["recommendations"] for o in outs)
        per_query.append({
            "query": q,
            "answered": answered,
            "exact_match_rate": float(np.mean(exacts)) if exacts else 1.0,
            "mean_semantic_equivalence": float(np.mean(sems)) if sems else 1.0,
            "citation_set_jaccard": float(np.mean(cjac)) if cjac else 1.0,
            "faithfulness": {
                "num_claims": fa[0]["num_claims"],
                "grounded_rate": float(np.mean([f["grounded_rate"] for f in fa])),
                "citation_validity_rate":
                    float(np.mean([f["citation_validity_rate"] for f in fa])),
                "hallucination_rate":
                    float(np.mean([f["hallucination_rate"] for f in fa])),
            },
        })
    answered = [p for p in per_query if p["answered"]]
    return {
        "runs": runs, "per_query": per_query,
        "exact_match_rate": float(np.mean([p["exact_match_rate"] for p in per_query])),
        "mean_semantic_equivalence": float(np.mean([p["mean_semantic_equivalence"]
                                                    for p in per_query])),
        "citation_set_jaccard": float(np.mean([p["citation_set_jaccard"]
                                               for p in per_query])),
        "answered_queries": len(answered),
        "refusal_rate": 1.0 - len(answered) / len(per_query) if per_query else 0.0,
        "faithfulness": {
            "scored_over_answered_queries": len(answered),
            "mean_grounded_rate": float(np.mean([p["faithfulness"]["grounded_rate"]
                                                 for p in answered])) if answered else None,
            "mean_citation_validity_rate":
                float(np.mean([p["faithfulness"]["citation_validity_rate"]
                               for p in answered])) if answered else None,
            "mean_hallucination_rate":
                float(np.mean([p["faithfulness"]["hallucination_rate"]
                               for p in answered])) if answered else None,
        },
    }


def to_markdown(report: dict) -> str:
    e, r, x = report["embedding"], report["retrieval"], report["explanation"]
    er = report["encoding_robustness"]
    fa = x["faithfulness"]
    cfg = report["determinism_config"]

    def fv(key):
        v = fa.get(key)
        return f"{v:.3f}" if isinstance(v, (int, float)) else "n/a (all refused)"

    lines = [
        "# Discovery Hub -- Stability & Reproducibility Report",
        f"_Generated {report['generated']} | mode={'mock' if report['mock'] else 'real'} "
        f"| runs={report['runs']} | queries={report['num_queries']}_",
        "",
        "## What this measures",
        "Whether asking the same question repeatedly yields the same output, at "
        "three stages -- plus the two things naive repeats miss: drift when the "
        "**batch size / device** changes, and whether the explanation's **citations "
        "are correct**, not merely stable. We distinguish **byte-level** determinism "
        "(hard to guarantee across hardware/library changes) from **semantic/"
        "functional stability** (achievable and what actually matters for trust).",
        "",
        "## Results",
        "",
        "| Stage | Metric | Value | Ideal |",
        "|---|---|---|---|",
        f"| Embedding | max cosine drift (same conditions) | {e['max_cosine_drift']:.2e} | ~0 |",
        f"| Embedding | exact match across runs | {e['exact_match_all_runs']} | true |",
        f"| Embedding | drift across batch sizes {er['batch_sizes_tested']} | "
        f"{er['max_drift_across_batch_sizes']:.2e} | ~0 on fixed stack |",
        f"| Embedding | batch-invariant | {er['batch_invariant']} | true (mock) / measure (real) |",
        f"| Retrieval | top-k set Jaccard | {r['mean_jaccard']:.3f} | 1.000 |",
        f"| Retrieval | Kendall's tau (rank) | {r['mean_kendall_tau']:.3f} | 1.000 |",
        f"| Retrieval | exact ordered-match rate | {r['exact_order_match_rate']:.3f} | 1.000 |",
        f"| Explanation | citation-set Jaccard (stability) | {x['citation_set_jaccard']:.3f} | 1.000 |",
        f"| Explanation | **citation validity** (answered) | {fv('mean_citation_validity_rate')} | 1.000 |",
        f"| Explanation | **claim groundedness** (answered) | {fv('mean_grounded_rate')} | high |",
        f"| Explanation | **hallucination rate** (answered) | {fv('mean_hallucination_rate')} | low |",
        f"| Explanation | refusal rate (policy gate) | {x['refusal_rate']:.3f} | -- |",
        f"| Explanation | semantic equivalence | {x['mean_semantic_equivalence']:.3f} | high |",
        f"| Explanation | exact prose match | {x['exact_match_rate']:.3f} | (stretch) |",
        "",
        "## How to read this for the investor",
        "- **Embedding and exact (FAISS Flat) retrieval are deterministic on a "
        "fixed stack** -- identical inputs return identical vectors and identical "
        "retrieved sets. The mock embedder is bit-identical across machines.",
        "- **Batch-size / device drift is measured, not assumed.** In mock it is 0 "
        "by construction; in real mode this row is where production batching shows "
        "up, and we report it honestly rather than quoting only the fixed-condition "
        "number. This is the memo's dominant nondeterminism source.",
        "- **Citation validity and groundedness check correctness, not just "
        "stability.** A faithfulness verifier scores whether each claim is supported "
        "by its cited evidence and whether the citation is real; the policy gate "
        "refuses low-confidence queries rather than guessing. (Mock numbers are "
        "optimistic -- the mock explainer cites by construction; the verifier's job "
        "is to measure the real LLM, and its unit tests prove it catches injected "
        "hallucinations and fabricated citations.)",
        "- **Citation-set stability remains the headline**: even if wording varies, "
        "the evidence pointed to is stable AND now verified correct.",
        "- **Byte-identical prose is a stretch goal**, reachable with a self-hosted "
        "LLM + batch-invariant kernels at a throughput cost; we do not claim it for "
        "API-served models, where batching and hardware are outside our control.",
        "",
        "## Reproducibility configuration in force",
        "```json",
        json.dumps(cfg, indent=2),
        "```",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--k", type=int, default=config.RETRIEVAL.top_k_rerank)
    ap.add_argument("--batch-sizes", default="1,8,32",
                    help="comma-separated batch sizes for the invariance test")
    ap.add_argument("--devices", default="cpu",
                    help="comma-separated devices to test, e.g. cuda:0,cuda:1 (real)")
    args = ap.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x.strip()]
    devices = [d.strip() for d in args.devices.split(",") if d.strip()] or ["cpu"]

    cfg = set_global_determinism(config.SEED)
    config.ensure_dirs()

    rr = _load("07_retrieve_rank.py", "retrieve_rank")
    rag = _load("08_multiagent_rag.py", "multiagent_rag")
    retriever = rr.Retriever(mock=args.mock)

    print(f"Measuring stability over {args.runs} runs x {len(DEFAULT_QUERIES)} queries...")
    report = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "mock": args.mock,
        "runs": args.runs,
        "num_queries": len(DEFAULT_QUERIES),
        "determinism_config": cfg,
        "embedding": measure_embeddings(DEFAULT_QUERIES, args.mock, args.runs),
        "encoding_robustness": measure_encoding_robustness(
            DEFAULT_QUERIES, args.mock, batch_sizes, devices),
        "retrieval": measure_retrieval(retriever, DEFAULT_QUERIES, args.runs, args.k),
        "explanation": measure_explanation(rag, retriever, DEFAULT_QUERIES,
                                           args.mock, args.runs),
    }

    (config.REPORT_DIR / "stability_report.json").write_text(json.dumps(report, indent=2))
    md = to_markdown(report)
    (config.REPORT_DIR / "stability_report.md").write_text(md)
    print("\n" + md)
    print(f"\nWrote {config.REPORT_DIR / 'stability_report.json'} and .md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
