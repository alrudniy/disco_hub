#!/usr/bin/env python3
"""register_gap_analysis.py -- the headline finding of the 2026-07-15 run.

Reproduces, from artifacts only (no GPU, no model loads, ~60s on CPU):

  1. BGE-reranker-v2-m3 correlates with query<->doc WORD OVERLAP far more than with
     actual relevance. Qwen3-Reranker-8B does not.
  2. At FIXED relevance (LLM grade 3 = "directly actionable"), BGE's score swings 4.1x
     on wording alone; Qwen swings 1.15x. 76% of judged true matches are low-overlap.
  3. Synthetic queries are generated FROM their origin document (InPars/Promptagator),
     so origin docs carry ~+0.10 more query-word overlap than equally-relevant
     alternatives. qrels_exact_origin therefore rewards paraphrase retrieval.
  4. Consequence: fine-tuning on those pairs taught LEXICAL matching. Against the
     deployed baseline, the best arm gained +0.165 R@10 on lexically-similar queries
     and LOST 0.089 on semantically-distant ones. The aggregate +0.061 hid the trade.
     Difference-in-differences +0.2545 [+0.1530, +0.3556], z~5.

Usage (Drew):
    python3 analysis/register_gap_analysis.py \
        /home/alex/discovery_hub/data/normalized/docs.jsonl \
        data/qrels/qrels_exact_origin_v1.jsonl \
        data/per_query_r10.json \
        data/multi_positive_labels_v1.jsonl

Arg 4 is optional; omit it to skip sections 1-3 and run only the bucket/DiD analysis.
"""
from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

import numpy as np

# Words that carry no discriminative signal in this corpus: function words plus the
# boilerplate that appears in nearly every patent title AND nearly every scout query.
STOP = set(
    "the a an of for and or to in with as by from thereof use uses using method methods "
    "composition compositions available licensing novel new therapy treatment".split()
)


def toks(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w) > 2 and w not in STOP}


def overlap(query: str, doc_text: str) -> float:
    """Fraction of the query's content words that appear anywhere in the doc."""
    q = toks(query)
    return len(q & toks(doc_text)) / len(q) if q else 0.0


def load_doc_text(docs_path: Path) -> dict[str, str]:
    text = {}
    for line in docs_path.open():
        if not line.strip():
            continue
        d = json.loads(line)
        body = d.get("embedding_text") or d.get("abstract") or ""
        text[d["doc_id"]] = ((d.get("title") or "") + " " + body[:1500])
    return text


def boot_mean(x: np.ndarray, rng, n: int = 20000) -> np.ndarray:
    return x[rng.integers(0, len(x), (n, len(x)))].mean(axis=1)


def section_teacher_bias(labels_path: Path, text: dict[str, str]) -> None:
    """Sections 1-3: what the teachers actually measure, and how the qrels inherit it."""
    llm, des = [], []
    origin_ov: dict[str, float] = {}
    g3_ov: dict[str, list[float]] = collections.defaultdict(list)

    for line in labels_path.open():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("masked"):
            continue
        ov = overlap(r["query"], text.get(r["document_id"], ""))
        if not toks(r["query"]):
            continue
        rec = (ov, float(r["bge_score"]), float(r["qwen_score"] or 0.0), int(r.get("grade", 0)))
        if r.get("is_designated_positive"):
            des.append(rec)
            origin_ov[r["query_id"]] = ov
        elif r.get("llm_grade") is not None:
            llm.append(rec)
            if r["llm_grade"] == 3:
                g3_ov[r["query_id"]].append(ov)

    a, d = np.array(llm), np.array(des)
    print("=" * 78)
    print(f"[1] WHAT THE TEACHERS MEASURE   (LLM-judged pairs, n={len(a)})")
    print("=" * 78)
    print(f"  corr(word_overlap, bge)  = {np.corrcoef(a[:, 0], a[:, 1])[0, 1]:+.3f}"
          f"     corr(llm_grade, bge)  = {np.corrcoef(a[:, 3], a[:, 1])[0, 1]:+.3f}")
    print(f"  corr(word_overlap, qwen) = {np.corrcoef(a[:, 0], a[:, 2])[0, 1]:+.3f}"
          f"     corr(llm_grade, qwen) = {np.corrcoef(a[:, 3], a[:, 2])[0, 1]:+.3f}")
    print("  -> BGE tracks strings ~2.2x more than meaning. Qwen slightly favors meaning.")

    g3 = a[a[:, 3] == 3]
    lo, hi = g3[g3[:, 0] < 0.34], g3[g3[:, 0] >= 0.34]
    print()
    print("=" * 78)
    print(f"[2] RELEVANCE HELD CONSTANT (LLM grade 3 only, n={len(g3)}) -- only wording varies")
    print("=" * 78)
    print(f"  {'bucket':28s} {'n':>7s} {'BGE':>8s} {'Qwen':>8s}")
    print(f"  {'LOW  overlap (<34%)':28s} {len(lo):7d} {lo[:, 1].mean():8.3f} {lo[:, 2].mean():8.3f}")
    print(f"  {'HIGH overlap (>=34%)':28s} {len(hi):7d} {hi[:, 1].mean():8.3f} {hi[:, 2].mean():8.3f}")
    print(f"  {'swing':28s} {'':7s} {hi[:, 1].mean()/lo[:, 1].mean():7.1f}x {hi[:, 2].mean()/lo[:, 2].mean():7.2f}x")
    print(f"  -> {len(lo)/len(g3):.0%} of judged true matches are LOW overlap. This corpus IS the gap.")

    common = [q for q in origin_ov if q in g3_ov]
    diff = np.array([origin_ov[q] - float(np.mean(g3_ov[q])) for q in common])
    rng = np.random.default_rng(0)
    lo95, hi95 = np.percentile(boot_mean(diff, rng), [2.5, 97.5])
    print()
    print("=" * 78)
    print("[3] THE QRELS INHERITED THE BIAS  (paired within query)")
    print("=" * 78)
    print(f"  queries with an origin doc AND >=1 LLM grade-3 alternative: {len(common)}")
    print(f"  origin_overlap - alternative_overlap = {diff.mean():+.3f} "
          f"[{lo95:+.3f}, {hi95:+.3f}]  {'SIG' if lo95 > 0 or hi95 < 0 else 'ns'}")
    print("  -> Synthetic queries inherit their source document's vocabulary. The one doc")
    print("     qrels_exact_origin credits is the one BGE was always going to find.")
    print("  CAVEAT: origin docs are grade-3 BY CONSTRUCTION (never judged); alternatives")
    print("     are grade-3 BY THE JUDGE. Some of this gap could be a relevance gap.")


def section_buckets(qrels_path: Path, pq_path: Path, text: dict[str, str]) -> None:
    """Section 4: the consequence, measured on held-out eval."""
    qr = [json.loads(l) for l in qrels_path.open() if l.strip()]
    h = {k: np.array(v) for k, v in json.load(open(pq_path)).items()}
    ov = np.array([overlap(q["query"], text.get(q["relevant_doc_ids"][0], "")) for q in qr])
    med = np.median(ov)
    LO, HI = ov < med, ov >= med
    rng = np.random.default_rng(0)

    print()
    print("=" * 78)
    print(f"[4] WHAT FINE-TUNING ACTUALLY LEARNED  (n={len(ov)}, median overlap {med:.3f},"
          f" LOW n={LO.sum()}, HIGH n={HI.sum()})")
    print("=" * 78)
    print(f"  {'model':26s} {'LOW':>8s} {'HIGH':>8s} {'ratio':>7s}")
    for m in ["Z_baseline_untrained", "A_mnrl_control", "A_fixed"]:
        if m not in h:
            continue
        l_, h_ = h[m][LO].mean(), h[m][HI].mean()
        print(f"  {m:26s} {l_:8.4f} {h_:8.4f} {h_/l_:7.2f}")
    print("  -> Z (deployed, never touched by this pipeline) is the BEST semantic matcher.")

    print()
    print("  Difference-in-differences vs Z -- ONE test, the actual claim:")
    print("  (does training trade semantic recall for lexical recall?)")
    base = "Z_baseline_untrained"
    for x in ["A_mnrl_control", "A_fixed"]:
        if x not in h:
            continue
        dh_, dl_ = h[x][HI] - h[base][HI], h[x][LO] - h[base][LO]
        did = boot_mean(dh_, rng) - boot_mean(dl_, rng)
        lo95, hi95 = np.percentile(did, [2.5, 97.5])
        print(f"    {x:22s} HIGH gain - LOW gain = {did.mean():+.4f} "
              f"[{lo95:+.4f}, {hi95:+.4f}]  {'SIG' if lo95 > 0 or hi95 < 0 else 'ns'}")


def main() -> int:
    docs = Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/dh_data/normalized/docs.jsonl")
    qrels = Path(sys.argv[2] if len(sys.argv) > 2
                 else "/workspace/dh_data/pipeline2/qrels/qrels_exact_origin_v1.jsonl")
    pq = Path(sys.argv[3] if len(sys.argv) > 3 else "/workspace/per_query_r10.json")
    labels = Path(sys.argv[4]) if len(sys.argv) > 4 else None

    print(f"loading docs from {docs} ...", flush=True)
    text = load_doc_text(docs)
    print(f"  {len(text)} docs\n", flush=True)

    if labels and labels.exists():
        section_teacher_bias(labels, text)
    else:
        print("(labels not given -- skipping sections 1-3)\n")

    section_buckets(qrels, pq, text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
