# Rerank lexical dependence — raw evidence

Everything behind README section 4's rerank claim. Pulled off the rented H200
(vast_h100) because that box gets reclaimed and this was the only copy.

Query throughout: `HER2 targeted therapy for breast cancer`.
Config, unchanged from shipped: BAAI/bge-reranker-v2-m3, max_length=512,
batch_size=16, Sigmoid head, num_labels=1; encoder qwen3-dh-ft-4b, dim 2560,
max_seq_length=512. top_k=50, rerank_k=50.

| file | what it is |
|---|---|
| `t1_t2_forensics.py` / `t1.log` | T1 score forensics + T2 text parity. All 50 with rank_dense, rank_rerank, unrounded score, literal-HER2 flag, token counts. |
| `t1_rows.json` | the 50 rows, machine-readable. Source for the T0 re-analyses. |
| `t4_causal.py` / `t4.log` | T4 single-term causal probe, both directions. |
| `bridge_hunt.py` / `bridge_hunt2.log` | the 28-pair synonym sweep, separator-blind control. |
| `diag_gap.py` | the B7 ownership-vs-cosine measurement that demoted the gap threshold. |
| `demo_h200.log` / `demo_v3.log` | full demo runs, LLM live. |

## What is settled

- The token is CAUSAL, not correlated. Rank-1 doc, HER2 -> ErbB2 (same protein,
  every other byte constant, 331 tokens on-topic): **-5.47 logits, rank 1 -> 38**.
- Magnitude is enough to produce the observed 45/5 partition: the boundary is
  0.8951 logits, the 2nd-largest of the 49 adjacent gaps in logit space (19.5x
  the 0.0459 median).
- It is NOT a constant bonus: -5.47 vs +1.76 is a 3.1x disagreement.
- It is NOT the whole story: +1.76 lifts the bridge to rank 38, not into top-10.

## Ruled out

truncation (0/50 pairs truncated; identical 538-char source string both stages) *
degenerate scores (5 distinct bridged values; no 0/None/NaN/inf/floor) *
non-determinism (bitwise identical across two runs) *
length confound at the partition (bridged median 201 tok vs literal 213, inside
the literal range -- though spearman(n_tok, z)=+0.55 overall, so length is not
inert generally) * swallowed exceptions (no try/except in 07's scoring path).

## Open

Whether the shape is DOSE-SATURATING (the rank-1 doc carried 6 HER2 mentions, the
bridge 1 -- so per-mention saturation could reconcile -5.47 and +1.76 with no
context term at all) or a genuine CONTEXT INTERACTION. T4c is the dose-response
probe that separates them. Until it runs, claim only what survives both shapes:

> the reranker carries a lexical dependence large enough, on this query, to bury
> a semantically identical document 37 places.

n=2 documents, 1 query. This is a causal probe on two instances, not a population
effect. Do not quote -5.47 as "the" lexical penalty.

## A methods warning worth keeping

The first synonym sweep used a plain substring exclusion and reported PD-1/CD279
at rank 3 and B7-H3/CD276 at rank 17 as bridges. Both were leaks -- the documents
spell the modern term without the hyphen ("Anti-PD1 antibodies", "Anti-CD276
antibodies (B7H3)"). Separator-blind, they collapse to 2,434 and 5,466. And the
first T1 write-up characterised the partition in SIGMOID space, where the
boundary looked like an unremarkable 3.7x-median tail; in logit space it is the
2nd-largest gap of 49. Both errors pointed the same way: toward a more flattering
story. Measure in the model's own space, and normalise punctuation.
