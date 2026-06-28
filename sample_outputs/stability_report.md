# Discovery Hub -- Stability & Reproducibility Report
_Generated 2026-06-27T01:02:40 | mode=mock | runs=5 | queries=5_

## What this measures
Whether asking the same question repeatedly yields the same output, at three stages -- plus the two things naive repeats miss: drift when the **batch size / device** changes, and whether the explanation's **citations are correct**, not merely stable. We distinguish **byte-level** determinism (hard to guarantee across hardware/library changes) from **semantic/functional stability** (achievable and what actually matters for trust).

## Results

| Stage | Metric | Value | Ideal |
|---|---|---|---|
| Embedding | max cosine drift (same conditions) | 5.96e-08 | ~0 |
| Embedding | exact match across runs | True | true |
| Embedding | drift across batch sizes [1, 8, 32] | 5.96e-08 | ~0 on fixed stack |
| Embedding | batch-invariant | True | true (mock) / measure (real) |
| Retrieval | top-k set Jaccard | 1.000 | 1.000 |
| Retrieval | Kendall's tau (rank) | 1.000 | 1.000 |
| Retrieval | exact ordered-match rate | 1.000 | 1.000 |
| Explanation | citation-set Jaccard (stability) | 1.000 | 1.000 |
| Explanation | **citation validity** (answered) | 1.000 | 1.000 |
| Explanation | **claim groundedness** (answered) | 1.000 | high |
| Explanation | **hallucination rate** (answered) | 0.000 | low |
| Explanation | refusal rate (policy gate) | 0.800 | -- |
| Explanation | semantic equivalence | 1.000 | high |
| Explanation | exact prose match | 1.000 | (stretch) |

## How to read this for the investor
- **Embedding and exact (FAISS Flat) retrieval are deterministic on a fixed stack** -- identical inputs return identical vectors and identical retrieved sets. The mock embedder is bit-identical across machines.
- **Batch-size / device drift is measured, not assumed.** In mock it is 0 by construction; in real mode this row is where production batching shows up, and we report it honestly rather than quoting only the fixed-condition number. This is the memo's dominant nondeterminism source.
- **Citation validity and groundedness check correctness, not just stability.** A faithfulness verifier scores whether each claim is supported by its cited evidence and whether the citation is real; the policy gate refuses low-confidence queries rather than guessing. (Mock numbers are optimistic -- the mock explainer cites by construction; the verifier's job is to measure the real LLM, and its unit tests prove it catches injected hallucinations and fabricated citations.)
- **Citation-set stability remains the headline**: even if wording varies, the evidence pointed to is stable AND now verified correct.
- **Byte-identical prose is a stretch goal**, reachable with a self-hosted LLM + batch-invariant kernels at a throughput cost; we do not claim it for API-served models, where batching and hardware are outside our control.

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