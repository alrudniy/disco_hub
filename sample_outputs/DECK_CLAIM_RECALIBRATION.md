# Deck Claim Recalibration — Retrieval Architecture

_What the pipeline now measures, and what the deck should say as a result._

## TL;DR

The deck attributes a **+20% Precision@10 / NDCG** retrieval gain to a stack of
"hybrid semantic search + R-GCN graph + contrastive embeddings + neural rerank."
With the retrieval-quality harness (stage 10) now in place, we can attribute the
gain to specific components instead of asserting it. Two findings:

1. **The hybrid (dense + BM25) half is real and earns the claim.** Adding the BM25
   keyword signal improves every retrieval metric over dense-only, with
   statistical significance.
2. **The R-GCN graph signal does not improve retrieval and slightly hurts it** —
   even after replacing the old text-derived proxy with a genuine entity-linked
   query→graph signal. It is therefore **off by default in retrieval ranking**.

The honest recalibration: credit retrieval gains to **hybrid lexical+dense
retrieval and cross-encoder reranking**; describe the knowledge graph as
**evidence/provenance infrastructure and an unproven retrieval signal**, not a
driver of Precision@10.

## How this was measured

Stage 10 builds a labeled evaluation set (each query generated from a known source
document, which is its gold positive — Promptagator/InPars), runs three retrieval
configurations on the *same* queries, and reports each metric with a 95% bootstrap
confidence interval and a paired Wilcoxon p-value. The three configurations are
marginal — each adds one signal:

- `dense` — embedding cosine only
- `hybrid` — dense **+ BM25 keyword**, fused by Reciprocal Rank Fusion
- `hybrid+graph` — **+ entity-linked R-GCN signal** (abstains when the query names
  no graph entity)

> These specific numbers come from the synthetic eval over the mock corpus, so
> read them as **relative component contributions**, not absolute real-world
> quality. The mechanism and the measurement are what transfer; absolute values
> require curated qrels on real data (`stage 10 --qrels`).

## Finding 1 — BM25 hybrid helps (keep it)

`hybrid` vs `dense`, 200 queries — every metric improves, all p < 0.05:

| Metric | dense | hybrid | Δ | p |
|---|---|---|---|---|
| Recall@10 | 0.340 | 0.440 | **+0.100** | <0.001 |
| Recall@50 | 0.605 | 0.780 | **+0.175** | <0.001 |
| MRR@10 | 0.108 | 0.165 | **+0.057** | <0.001 |
| nDCG@10 | 0.162 | 0.229 | **+0.067** | <0.001 |
| MAP | 0.123 | 0.182 | **+0.060** | <0.001 |

BM25 contributes exact-term matching (gene symbols, acronyms, assay names) that a
dense bag-of-tokens encoder under-weights. On real data the dense (semantic) and
BM25 (lexical) signals are even *more* complementary than in this lexical-only mock,
so this is a conservative read. **Action: "hybrid semantic search" is now accurate
and beneficial — keep the claim, attribute the lift here.**

## Finding 2 — the graph signal does not help retrieval (downgrade it)

The previous graph signal scored candidates against the centroid of the top *text*
hits — circular, and the harness confirmed it added nothing. We replaced it with a
genuine query→graph signal: link the query to graph entities (org / inventor /
facility), anchor on those nodes, retrieve technologies by structural proximity,
and **abstain** when nothing links. It fired on **76%** of eval queries — yet:

`hybrid+graph` vs `hybrid` — small, mostly negative deltas, several significant:

| Metric | hybrid | hybrid+graph | Δ | p |
|---|---|---|---|---|
| Recall@20 | 0.600 | 0.580 | −0.020 | 0.046 |
| Recall@50 | 0.780 | 0.740 | **−0.040** | 0.005 |
| MRR@10 | 0.165 | 0.150 | −0.015 | <0.001 |
| nDCG@10 | 0.229 | 0.217 | −0.012 | <0.001 |
| MAP | 0.182 | 0.165 | −0.017 | <0.001 |

**Why it hurts:** anchoring on a coarse entity (in this corpus, an organization
tied to ~1,300 documents) pulls many structurally-adjacent-but-irrelevant
documents into the fused pool, and RRF gives them rank credit that displaces
stronger text/keyword hits. The graph knows "same org," which is too blunt to
locate *the specific* relevant technology.

**Action: the graph signal is OFF by default in retrieval ranking** (one config
flag re-enables it). The deck should not claim the R-GCN drives retrieval
precision.

## What the deck should say instead

- **Before:** "Hybrid semantic search + R-GCN + contrastive + neural rerank →
  +20% Precision@10 / NDCG."
- **After:** "Retrieval combines dense semantic search with BM25 lexical matching
  (measured to improve Recall@10 by ~0.10 and nDCG@10 by ~0.07 over dense alone in
  internal eval) and a cross-encoder reranker. A knowledge graph provides
  provenance and inventor/organization evidence for every match; using it as a
  retrieval *ranking* signal is an open research direction that did not yet beat
  the hybrid baseline in our evaluation." Report any headline number as a
  measured value with a confidence interval, not a fixed promise.

## The graph still has value — just not here

Turning the graph off for *ranking* does not make it useless:

- **Evidence & provenance (Layer 3):** the KG is how a match is explained — which
  inventors, which organization, which facility, which collaborations. That is a
  product differentiator independent of whether it moves Recall@10.
- **It may yet help ranking** with (a) **finer entities** (specific inventors,
  concept/MeSH nodes, CPC subgroups) so the anchor is selective rather than
  "everything this big company filed"; (b) a **learned query→graph encoder**
  instead of string entity-linking; (c) using the graph **surgically** (a
  tie-breaker or a light rerank feature) rather than as a co-equal RRF list. Each
  is testable with the existing harness — re-enable `use_graph` and re-run stage 10.

## Reproduce / re-test

```
make 10                                   # dense vs hybrid vs hybrid+graph (mock)
python 10_eval_retrieval.py --qrels data/curated_qrels.jsonl   # on real labels
```
Re-enabling the graph for a fresh measurement: set `RetrievalConfig.use_graph =
True` (or call `retrieve(..., use_graph=True)`) and re-run stage 10.
