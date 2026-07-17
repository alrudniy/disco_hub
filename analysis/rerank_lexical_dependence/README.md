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
| `t4c_dose.py` / `t4c.log` | T4c dose-response: z(k) over k=0..6 HER2 mentions, first-k and last-k. |
| `demo_h200.log` / `demo_v3.log` / `dryrun.log` | full demo runs, LLM live. dryrun.log is the pre-demo dry run (peak RSS 21.69 GB). |

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

## T4c: PRESENCE DOMINATES COUNT ~2:1, and dose does NOT explain the asymmetry

z(k), replacing k of the rank-1 doc's 6 HER2 mentions with ErbB2:

    k         0       1       2       3       4       5       6
    z    +0.531  +0.128  -0.323  +0.021  -0.476  -1.271  -4.941
    marg     --   -0.40   -0.45   +0.34   -0.50   -0.79   -3.67

Removing FIVE of six mentions costs -1.80 logits. Removing the SIXTH costs -3.67
-- so the last-token transition is worth ~2x the other five put together. Same
shape in last-k order (final marginal -4.23).

PRESENCE DOMINATES COUNT ~2:1. Not "presence, not count" -- that was an
overclaim, and two numbers here refute it. The count component is -1.80, which is
2x the 0.8951 boundary: on its own it would still be a large effect, so count is
not negligible, it is merely the smaller half. And the curve is not monotone
(k=3 is +0.34), which at n=1 means the per-step structure is noise-limited and
the marginals should not be read individually at all. What survives is the ratio
of the two halves, not the shape of either.

None of the three pre-registered branches fit: not dose-saturating (z(1)-z(0) is
-0.40, not -1.76), not linear-additive (marginals range -3.67..+0.34), not a
first-mention effect (the cliff is the LAST removal).

And the last-token transition's HEIGHT is context-dependent: 1->0 costs -3.67 in
the rank-1 doc but 0->1 buys only +1.76 in the bridge doc, 2.1x apart on the same
transition. Dose does not reconcile them. A context term survives, now localised
to the presence transition rather than smeared across the count.

The wording holds unchanged, which is why it was chosen before the run:

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

---

## V1: query 2's verifier flip is NOT a fabricated abstention

The risk: "verifier fail 0.00" might be an empty/unparseable LLM response defaulting
to 0.00 -- a crash wearing a verdict's clothes, on the one query whose story is
"declines instead of confabulating". This codebase already shipped that bug's mirror
image once (reasoning tokens ate max_tokens, content came back "", empty was treated
as an answer). Three candidates: H1 tiny denominator, H2 verifier non-determinism,
H3 empty-defaults-to-zero.

**EVERY CRASH PATH IS LABELLED, so a crash cannot silently wear a verdict's
clothes.** That is what V1 establishes, and it holds regardless of what the failing
run did.

An earlier draft of this section claimed something stronger and did not earn it:
that fail+0.00 was structurally UNREACHABLE from the empty/unparseable path. It is
reachable. `_verify` routes an empty or unparseable response to
`_deterministic_claims` and returns `(claims, "deterministic", notes)`
(verifier.py:210-217); `run()` has exactly ONE AgentResult return, so BOTH modes
compute confidence through the same line:

    confidence = len(supported) / len(claims) if claims else 0.0     verifier.py:171

so 0 supported of 1 deterministic claim IS 0.00 + fail. The `else 0.0` branch does
carry verdict=pass (claims == [] forces passed=True, verifier.py:154-156, 161), but
that only rules out the EMPTY-CLAIMS default -- not the fallback.

What actually rules out an unlabelled crash is the labelling, not the structure:

  * empty / unparseable -> mode="deterministic" + caveat="LLM was configured but
    returned no usable verdict; fell back to the weaker lexical check"
    (verifier.py:197, 210-217). A real lexical verdict, honestly named.
  * exception -> unwinds through @timed to (ok=False, error=...), distinguishable
    from a gate-fail (ok=False, error=None) by construction (verifier.py:175-181).

So the fallback firing would still be an honest abstention with a real (weaker)
verdict behind it. What it would NOT be is well-explained on screen: the decline
reason would read "1 claim(s) not supported" when the truer reason is "the LLM
returned no usable verdict, so a weaker check ran". The payload knows; the screen
does not print `mode`.

**And it did not fire.** 10 live LLM verifier calls, 0 empty, 0 unparseable, 0
deterministic fallbacks.

V1b -- 5 orchestrator runs, query 2, LLM live:

    run  draft_sha1     chars  num  den  score  verdict  mode  raw_empty  raw_len
      1  b398eea45331    1353    9    9   1.00     pass   llm      False     1954
      2  b398eea45331    1353    9    9   1.00     pass   llm      False     1954
      3  67dcd6d33bc3    1545   10   10   1.00     pass   llm      False     2207
      4  67dcd6d33bc3    1545   10   10   1.00     pass   llm      False     2207
      5  b398eea45331    1353    9    9   1.00     pass   llm      False     1954

V1c -- ONE captured draft, re-verified 5x on FIXED input: scores [1.0]*5, verdicts
all pass, denominators [9,9,9,9,9], and the raw response byte-identical all five
times. **The verifier is stable on fixed input: H2 refuted.** Drafts varied (2
distinct sha1 in 5 runs): the variance is UPSTREAM, in synthesis. H1.

**What V1 did NOT establish, and cannot.** The fail did not reproduce in 5 runs, so
its raw response was never captured. Worse, the demo does not print `mode`, so the
original run's log CANNOT say whether the LLM verdict or the lexical fallback
produced it. Verbatim, the whole record of that run is:

    verifier       fail          0.00    4010.5
    WHY   : the evidence gate failed: 1 claim(s) not supported by any retrieved
            document, 0 contradicted by one

No mode, no caveat, no denominator, no timestamp. So whether that particular fail
was an LLM verdict or a labelled fallback is UNRESOLVED and will stay unresolved --
the evidence to settle it was never written down. What is settled is that either
way it was a real verdict from a named checker, not a default.

That the run had **denominator 1** is an INFERENCE from the logged numbers, not a
measurement: unsupported_count=1, contradicted_count=0, confidence 0.00; statuses
are exactly ("supported","contradicted","unsupported") (verifier.py:61); so
supported=0 and claims = 0+1+0 = 1.

**Next week's problem, not tonight's:** every run tonight extracted 9-10 claims; the
failing run extracted 1. Synthesis has an uncharacterised short-output mode. That is
verifier-ADJACENT, not verifier -- the gate behaved correctly on whatever it was
handed. Printing `mode` on the trace row would also have made this a ten-second
question instead of an unresolvable one.

With denominator 1, 0.00 and 1.00 are the only attainable scores -- "flaky" is the
wrong word for a two-valued statistic. The suspicious cell (0.00 with denominator
>= 2) was not observed.

Nothing was fixed. The pre-authorized fix (distinguish FAILED-TO-VERIFY from
VERIFIED-UNSUPPORTED) did not trigger, because that distinction already exists and
is already labelled.
