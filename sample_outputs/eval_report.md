# Discovery Hub -- Retrieval Quality Report
_Generated 2026-06-26T21:34:20 | mode=mock | queries=200 | labels=synthetic (one positive per query)_

## What this measures
Whether retrieval returns the right documents, and whether a change **helps** -- every comparison carries a 95% bootstrap confidence interval and a paired Wilcoxon p-value, so a lift is reported as `+X.XXX [lo, hi], p=...` rather than asserted.

> Note: synthetic labels have **one positive per query**, so Precision@10 is capped at 0.10 -- read Recall@k and MRR@10 here. Curated qrels with multiple positives make Precision@10 meaningful.

## Scorecard (mean [95% CI])

| Metric | dense | hybrid | hybrid+graph |
|---|---|---|---|
| recall@1 | 0.025 [0.005, 0.050] | 0.065 [0.035, 0.100] | 0.050 [0.025, 0.080] |
| recall@5 | 0.205 [0.155, 0.260] | 0.270 [0.215, 0.330] | 0.260 [0.205, 0.320] |
| recall@10 | 0.340 [0.275, 0.410] | 0.440 [0.380, 0.510] | 0.440 [0.380, 0.510] |
| recall@20 | 0.505 [0.440, 0.575] | 0.600 [0.535, 0.665] | 0.580 [0.520, 0.645] |
| recall@50 | 0.605 [0.540, 0.675] | 0.780 [0.720, 0.840] | 0.740 [0.680, 0.800] |
| precision@10 | 0.034 [0.028, 0.041] | 0.044 [0.038, 0.051] | 0.044 [0.038, 0.051] |
| mrr@10 | 0.108 [0.083, 0.138] | 0.165 [0.132, 0.201] | 0.150 [0.118, 0.181] |
| ndcg@10 | 0.162 [0.131, 0.198] | 0.229 [0.193, 0.271] | 0.217 [0.182, 0.255] |
| map | 0.123 [0.099, 0.151] | 0.182 [0.150, 0.217] | 0.165 [0.135, 0.196] |

## Does it help? hybrid vs dense

| Metric | Δ (cand − base) | 95% CI | p | verdict |
|---|---|---|---|---|
| recall@1 | +0.040 | [+0.010, +0.075] | 0.021 | helps ✓ |
| recall@5 | +0.065 | [+0.015, +0.115] | 0.020 | helps ✓ |
| recall@10 | +0.100 | [+0.055, +0.145] | 0.000 | helps ✓ |
| recall@20 | +0.095 | [+0.045, +0.145] | 0.000 | helps ✓ |
| recall@50 | +0.175 | [+0.125, +0.225] | 0.000 | helps ✓ |
| precision@10 | +0.010 | [+0.006, +0.015] | 0.000 | helps ✓ |
| mrr@10 | +0.057 | [+0.028, +0.086] | 0.000 | helps ✓ |
| ndcg@10 | +0.067 | [+0.039, +0.094] | 0.000 | helps ✓ |
| map | +0.060 | [+0.031, +0.088] | 0.000 | helps ✓ |

## Does it help? hybrid+graph vs hybrid

_The entity-linked graph signal fired on 152/200 queries (76%) and abstained on the rest; on abstaining queries hybrid+graph == hybrid by construction, which dilutes any average effect._

| Metric | Δ (cand − base) | 95% CI | p | verdict |
|---|---|---|---|---|
| recall@1 | -0.015 | [-0.040, +0.005] | 0.180 | no sig. diff |
| recall@5 | -0.010 | [-0.030, +0.010] | 0.317 | no sig. diff |
| recall@10 | +0.000 | [-0.015, +0.015] | 1.000 | no sig. diff |
| recall@20 | -0.020 | [-0.040, -0.005] | 0.046 | hurts ✗ |
| recall@50 | -0.040 | [-0.070, -0.015] | 0.005 | hurts ✗ |
| precision@10 | +0.000 | [-0.002, +0.002] | 1.000 | no sig. diff |
| mrr@10 | -0.015 | [-0.031, +0.001] | 0.000 | hurts ✗ |
| ndcg@10 | -0.012 | [-0.025, +0.003] | 0.000 | hurts ✗ |
| map | -0.017 | [-0.033, -0.001] | 0.000 | hurts ✗ |

## How to read this
- **Recall@k**: did the right document make the top-k. The headline quality number for a retrieve-then-rerank system.
- **MRR@10 / nDCG@10**: how high the right document ranked.
- **Δ with CI + p**: the honest form of "+20% Precision@10" -- a measured lift with uncertainty, on the same queries, paired. A wide CI that crosses 0 means "not enough evidence yet", not "better".
- Swap in **curated qrels** (`--qrels`) and re-run to report numbers you can put in front of an investor.

## Reproducibility configuration in force
```json
{
  "seed": 20240611,
  "strict_deterministic": true,
  "PYTHONHASHSEED": "20240611",
  "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
  "numpy_version": "2.4.4",
  "torch_configured": false,
  "guarantees": {
    "byte_identical_same_stack": "embeddings & exact (FAISS Flat) retrieval: yes. LLM greedy decode: only with batch-invariant kernels + fixed batch size.",
    "byte_identical_cross_hardware": "no (float non-associativity across GPU arch / driver / library versions).",
    "semantic_stability": "yes, and measured in 09_stability_harness.py."
  },
  "torch_version": null,
  "cuda_available": false
}
```