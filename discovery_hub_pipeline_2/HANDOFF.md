# DiscoveryHub pipeline_2 — handoff (2026-07-15, end of day)

**Supersedes `HANDOFF_pipeline2_bakeoff.md`. Several claims in that document are now
known to be wrong; see "Corrections to the previous handoff" below.**

---

## TL;DR

The bakeoff was going to tell us which training objective wins. It didn't — the answer is
**none of them, and the eval is underpowered anyway**. But instrumenting it produced a
much better finding:

> **Our fine-tuning was teaching the model to match words, not meaning — and our benchmark
> was rewarding it for that.** Against the deployed 0.6B, our best arm gained **+0.165 R@10
> on lexically-similar queries and lost 0.089 on semantically-distant ones**. The reported
> aggregate of +0.061 hid the trade. Difference-in-differences +0.2545 [+0.1530, +0.3556],
> z≈5, n=440.

Root cause: synthetic queries are generated **from** their origin document (InPars/
Promptagator), so they inherit its vocabulary. The training pairs don't contain the
register gap. Neither does the eval.

**The H100 has been pulled and destroyed. Everything below is on Drew.**

> **UPDATE (RASC pass, same day).** Acting on that finding: the tree now ships an
> inference-only **Register-Aware Semantic Cascade** — union the untouched 0.6B with the
> fine-tuned 8B, drop the 4B-score-biased candidate cap, and rerank with a *correctly
> prompted* Qwen3-Reranker-8B (official chat framing, 8K context, pharma instruction,
> ranked on the logit margin). No retraining. GroupRank-32B is built, feature-flagged OFF,
> and gated. Bugs **#3–#10 are fixed and verified by running them**, not by reading the
> source. See "Bugs" and "Open decisions" below. Nothing is promoted on exact-origin R@10.

---

## The finding, in four measurements

All reproducible on Drew, CPU-only, ~60s:

```bash
cd ~/dh2_takehome
python3 code/analysis/register_gap_analysis.py \
    /home/alex/discovery_hub/data/normalized/docs.jsonl \
    data/qrels/qrels_exact_origin_v1.jsonl \
    data/per_query_r10.json \
    data/multi_positive_labels_v1.jsonl
```

### 1. BGE is a lexical matcher; Qwen is a semantic one

n = 55,605 LLM-judged pairs. "Overlap" = fraction of the query's content words present in
the doc (title + 1500 chars), minus stopwords and patent/scout boilerplate.

|  | corr w/ word overlap | corr w/ LLM grade |
|---|---:|---:|
| **BGE-reranker-v2-m3** | **+0.468** | **+0.211** |
| **Qwen3-Reranker-8B** | +0.371 | +0.410 |

BGE tracks strings ~2.2× more than it tracks meaning.

### 2. Relevance held constant — only wording varies

LLM grade 3 = "disease/stage, mechanism/target, and modality/route all clearly match."
Same judge, same rubric, same grade:

| bucket | n | BGE | Qwen |
|---|---:|---:|---:|
| LOW overlap (<34%) | 13,749 | 0.053 | 0.802 |
| HIGH overlap (≥34%) | 4,310 | 0.218 | 0.921 |
| **swing** | | **4.1×** | **1.15×** |

**76% of judged true matches are low-overlap.** The register gap is the modal case in this
corpus, not an edge case.

Canonical example, hand-verified: query *"Monoclonal antibody targeting PD-L1 for oncology"*
→ patent *"Anti-B7-H1 antibodies for treating tumors"*. B7-H1 **is** PD-L1 (older name).
BGE: **0.002**. Qwen: 0.859. LLM: grade 3. BGE ranks it in the bottom half of the pool.

### 3. The qrels inherited the bias

| | word overlap | BGE |
|---|---:|---:|
| designated positives (the synthetic query's origin doc) | 0.373 | 0.273 |
| LLM grade-3 alternatives (found by the judge) | 0.236 | 0.092 |

Paired **within query** (n=792 queries having both): **+0.100 [+0.085, +0.115] SIG**.

`recall_at_k` with `len(relevant)==1` makes this worse than a bias: surface ten genuinely
relevant patents and rank the origin doc 11th → **R@10 = 0.0**.

*Caveat to state out loud:* origin docs are grade-3 **by construction** (`if
is_designated_positive: grade = 3`, never judged); alternatives are grade-3 **by the judge**.
Part of the +0.100 could be a relevance gap rather than a generation artifact. Findings 1
and 2 don't depend on it.

### 4. The consequence — measured on held-out eval

440 held-out queries, split at median overlap 0.333 (LOW n=180, HIGH n=260):

| model | LOW (semantic needed) | HIGH (lexical works) | ratio |
|---|---:|---:|---:|
| Z_baseline_untrained (deployed 0.6B) | **0.2556** | 0.3346 | 1.31 |
| A_mnrl_control | 0.1667 | **0.5000** | 3.00 |
| A_fixed | 0.1611 | 0.4115 | 2.55 |

**Z — which this pipeline never touched — is the best semantic matcher of the three.**
Training bought +49% relative on lexical queries and cost −35% on semantic ones.

Difference-in-differences vs Z (one test, 20k paired bootstrap):
- `A_mnrl_control`: **+0.2545 [+0.1530, +0.3556] SIG**
- `A_fixed`: **+0.1713 [+0.0722, +0.2688] SIG**

No floors/ceilings — every cell is between 0.16 and 0.50. The decomposition reconciles to
4 decimals with every aggregate: `(260×0.1654 + 180×−0.0889)/440 = +0.0614` ✓

---

## Bakeoff result (secondary — it's a null)

440 held-out queries, `qrels_exact_origin_v1`, all arms scored against **identical** qrels.

| arm | R@10 | vs Z | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|
| A_mnrl_control | 0.3636 | +0.0614 | 0.1938 | 0.2838 |
| F_rand1lh | 0.3591 | +0.0568 | 0.1825 | 0.2730 |
| B_mnrl_hardfilter | 0.3364 | +0.0341 | 0.1754 | 0.2614 |
| D_listwise_kl | 0.3114 | +0.0091 | 0.1551 | 0.2409 |
| **Z_baseline_untrained** | **0.3023** | — | 0.1516 | 0.2341 |
| A_fixed | 0.3091 | +0.0068 | 0.1484 | 0.2278 |
| C_gpl_marginmse | 0.2045 | −0.0977 | 0.1008 | 0.1508 |
| E_lsepair | 0.0545 | −0.2477 | 0.0231 | 0.0416 |

Paired bootstrap vs Z, and with Bonferroni ×6 (α=0.0083 → 99.17% CI):

| arm | delta | 95% CI | | 99.17% CI | |
|---|---:|---|---|---|---|
| A_mnrl_control | +0.0614 | [+0.0091, +0.1136] | SIG | [−0.0091, +0.1318] | **ns** |
| F_rand1lh | +0.0568 | [+0.0068, +0.1068] | SIG | [−0.0091, +0.1250] | **ns** |
| B_mnrl_hardfilter | +0.0341 | [−0.0182, +0.0864] | ns | [−0.0356, +0.1045] | ns |
| D_listwise_kl | +0.0091 | [−0.0386, +0.0568] | ns | [−0.0545, +0.0750] | ns |
| C_gpl_marginmse | −0.0977 | [−0.1523, −0.0432] | SIG | [−0.1705, −0.0273] | **SIG** |
| E_lsepair | −0.2477 | [−0.2909, −0.2045] | SIG | [−0.3068, −0.1886] | **SIG** |

Head-to-head, all **ns**: A−B +0.0273 [−0.0068, +0.0614] · A−F +0.0045 [−0.0295, +0.0386] ·
F−B +0.0227 [−0.0159, +0.0614]

**Three findings:**
1. **No objective significantly beats any other.** Underpowered: SE(A−B) ≈ 0.0174; detecting
   a true 0.027 gap at 80% power needs SE ≤ 0.0098 → **~1,400 held-out queries, not 440.**
2. **Two arms significantly degrade retrieval**, both survive correction, and both were
   predicted from reading the loss code (see "Diagnosed, NOT fixed").
3. **The old "B beats A by 2.2×" headline did not replicate — it reversed.** A > B now (ns).
   `fn_flag` fires on 21.3% of pairs (negatives scoring ≥95% of the positive — the hardest
   ones) and B deletes them globally. Removing your hardest negatives should hurt. It does.

### A_fixed (prefix fix + 4× label coverage)

- vs Z: **+0.0068 [−0.0432, +0.0568] ns** — indistinguishable from the deployed model.
- vs A_old: **−0.0545 [−0.0909, −0.0205] SIG** — the "fixes" significantly hurt.
- nDCG also down (0.2278 vs A_old 0.2838, below Z's 0.2341).

Two changes at once, no ablation → **the −0.0545 is unattributed.** The missing 2×2 cell is
`revert prefix + keep new labels` (~25 min train + ~60 min eval). Only worth running if
someone asks.

---

## Coverage: the re-merge (free, 4×, do this pattern again)

`multi_positive_labels_v1.jsonl` was **17 hours older than `llm_cache.jsonl`**. Phase 2 had
gone rogue re-fetching serially and was killed; `as_cache_only()` fixed it but the labels
were never regenerated. Re-running `p0_3 --max-llm-calls 1` re-read the whole cache:

| | before | after |
|---|---:|---:|
| LLM-judged label rows | 13,987 | **55,605** |
| queries with ≥1 verdict | 213 (7.1%) | **853 (28.4%)** |
| masked | 181,001 | 139,383 |
| positives/query (mean) | 5.7 | **15.8** |
| queries with ≥5 positives | 17.8% | **36.0%** |

**+41,617 adjudicated pairs for one API call.** Median positives/query is still 1 — 71.6% of
queries have no verdict at all, so it's two datasets stacked, not one dense one.

Cache state: **52,709 verdicts** (52,707 + 1 per re-merge run × 2), 52,706 applicable,
**1 orphan**, dup factor 1.00. Gray zone: 194,988 pairs → 184,689 unique keys → **28.5%**
covered. **131,983 keys still uncached.**

Judge is **not** a yes-machine (checked, then hand-verified 12 grade-3 samples: 9 clearly
defensible, 3 marginal):

| bucket | n | g0 | g1 | g2 | g3 |
|---|---:|---:|---:|---:|---:|
| LLM-judged | 55,605 | 12.4% | 15.3% | 39.8% | 32.5% |
| teacher-consensus | 131,773 | 96.6% | 0.1% | 0.3% | 3.0% |
| designated | 3,000 | — | — | — | 100% |

12.4% g0 is low because the gray zone is enriched by construction (≈ `qwen > 0.25` over
~110 top-retrieved docs/query). Unparseable replies: 42 (0.08%) — `max_tokens=40` is fine.

---

## Why the gray zone is 60% (it's the same root cause)

| | p1 | p25 | p50 | p75 | p99 | mean | sd | AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bge | 0.000 | 0.001 | 0.003 | 0.013 | 0.818 | 0.044 | 0.139 | 0.821 |
| qwen | 0.002 | 0.113 | 0.408 | 0.816 | 1.000 | 0.462 | 0.355 | 0.872 |

pearson(bge, qwen) = 0.387. Both teachers rank well; **neither is broken.**

`_in_grayzone` fires when `|bge − qwen| > 0.25` — and that accounts for **59.8% of the
59.9% "disagree"** (the `same_side` midpoint test contributes **0.0%**; that theory was
wrong). BGE says ~0.05 on a relevant doc, Qwen says ~0.80. **Every true positive trips the
condition.** The 60% gray zone isn't disagreement; it's one teacher measuring strings and
the other measuring meaning, on a corpus where those diverge 76% of the time.

BGE's AUC of 0.821 is **inflated** — it's scored against designated positives, whose defining
property (lexical overlap) is the thing BGE measures.

---

## Bugs

### Fixed and deployed (in this tree)

1. **Train/inference prefix asymmetry** — `train_arm` encoded `pq.query` bare; `DenseRetriever.
   encode_query` applies `format_query()` → `"Instruct: Given a pharmaceutical or biotech
   company's research interest, ... \nQuery: {q}"`. 4-word input at train, 44-word at eval.
   Violated the README's own invariant. Now `q = format_query(pq.query, base.QUERY_INSTRUCTION)`.
   Verify: `grep -c format_query dh2/train_bakeoff.py` → **3**.
   *Note:* fixing it made things **worse** (A_fixed − A_old = −0.0545 SIG). Unexplained.
   Do not assume this fix is a win.
2. **Stale labels** — see re-merge above.

### FIXED 2026-07-15 (RASC pass) — every one verified by a RUN, not a reading

Numerical proof in `tests/test_losses_bugfix.py` (CPU, ~2s) and `tests/test_dh2_smoke.py`.
The handoff's own lesson held: the *diagnoses* from reading the source were all correct,
and two of the *predicted numbers* were confirmed to 3 significant figures.

| # | fix | verified |
|---|---|---|
| 3 | `E_lsepair` zero-positive guard, matching F | old loss reproduces at **10,008.8** (predicted ≈10,010); guarded loss 0.0 with zero gradient; zero-pos rows now **excluded from the mean**, not averaged in as 0 |
| 4 | `marginmse_scale` applied, via an explicit `marginmse_mode` knob | broken error floor measured at **0.9025** (unclosable); both modes drive it to <1e-12 |
| 5 | `D_listwise_kl` teacher enters softmax in the student's units | raw teacher target ratio measured at **2.72** (= *e*, exactly as predicted); flattening confirmed — gradient on the teacher's BEST doc was **+9.83** (pushed down), now **−0.32** (pulled up) |
| 6 | `pack_listwise` stratified + seeded + shuffled | skewed fixture: unstratified window = **0 positives**, stratified = 2 |
| 7 | `fn_flag` keyed per `(query, doc)`; legacy flat shape still honored with a loud warning | per-query scoping asserted both ways |
| 8 | `build_qrels` reports relevant *and* candidate counts, plus medians | both emitted; `relevant <= candidates` asserted |
| 9 | `pull_pipeline2_results.sh` rewritten: explicit tarball, **includes `adjudication/` + `teacher_scores/`**, size+integrity verified, **exits non-zero on failure**, scrubs the z.ai key first | exits 2 if `llm_cache.jsonl` is missing — the watchdog must check this before destroying |
| 10 | unparseable LLM replies now actually mask | flagged `unparseable=True`; the **42 legacy poisoned rows already in the cache are caught retroactively** by their evidence string, so the 52,709-verdict cache is not re-paid for |
| — | `push_pipeline2_to_h100.sh` wrote `DH2_TEACHER_LLM_MODEL=claude-sonnet-4-6` (dead 1min.ai) with no base URL | now `glm-4.6` + z.ai base URL, so cache keys match |

**Bug #4 needs a decision from you — see "Open decisions" below.**

### The original diagnoses (all now fixed; retained verbatim as the record)

These are the readings that predicted each failure before the eval confirmed it. They are
kept because the *reasoning* is the reusable artifact — and because anyone reverting a fix
should have to argue with the diagnosis first.

3. **`E_lsepair`: no zero-positive guard.** `pos = s.masked_fill(is_positive == 0, -1e4)` →
   with no positives in the packed window, every entry is −1e4 → `logsumexp(pos) ≈ −9998` vs
   `logsum_all ≈ 12` → **loss ≈ +10,010** on the 32.7% of queries with zero positives in the
   first 8 candidates. `masked_fill` detaches those positions, so the gradient is `+softmax(s)`:
   push every score down, hardest on the top-ranked doc, nothing pulling up. `F_rand1lh` has
   the guard (`if len(pos_pos) == 0: continue`); E doesn't. → **E scored 0.0545, 18% of
   baseline.** Fix is unambiguous: skip, matching F.
4. **`C_gpl_marginmse`: `marginmse_scale` never applied.** Defined as 20.0, documented as
   "student cosine scale for margin matching", used in the **mnrl** branch as a softmax
   temperature, and **never referenced by the `MarginMSE` module**. C matches raw cosine
   margins (~±0.1) against teacher margins (~±1.0). → **−0.0977 SIG.** Fix needs a judgment
   call: scaling the student by 20 overshoots the other way. Decide deliberately.
5. **`D_listwise_kl`: teacher target unscaled.** `teacher_rel` enters `softmax` raw from
   [0,1] (max ratio *e* ≈ 2.7, near-uniform) while the student is ×20 and peaked. The KL
   teaches D to **flatten** its scores. → **+0.0091 ns, no effect.**
6. **`pack_listwise` isn't stratified.** Takes `pq.doc_ids[:max_list]` in pool order,
   unshuffled, no pos/neg guarantee. A/B/C get shuffled triples; D/E/F get the first 8
   candidates. Measured: 32.7% of queries have 0 positives in that window, 3.3% have 8/8.
   With the denser labels this gets **worse** (36% of queries now have ≥5 positives).
7. **`fn_flag` is global per-doc, not per-(query, doc).** `p1_1`: `fn_flag[r["document_id"]]
   = True`. 38,861 docs banned as negatives for **every** query.
8. **`build_qrels` mis-reports `mean_relevant_per_utility_query`.** It sums
   `len(v) for v in by_q_grades.values()`, which includes **grade-0** candidates — the
   `if g > 0` filter is applied only when writing. It reports ~50 ("candidates/query" is 49.5).
   **This is where the previous handoff's "~50 relevant docs/query" came from.** Real median:
   **1**.
9. **`pull_pipeline2_results.sh` loses the two most expensive artifacts.** It pulls
   `reports/ qrels/ train_labels/ manifests/ logs/` — **no `adjudication/`, no
   `teacher_scores/`**. And every rsync ends `2>/dev/null || true`, so with SSH broken it
   prints `==> pull complete` and **exits 0 having transferred nothing.** Use an explicit
   tarball. Scrub `dh2_env.sh` first — it holds the z.ai key.
10. **Unparseable LLM replies aren't masked.** `graded_relevance` catches a parse failure and
    caches `grade=1, relevance=0.5` with evidence `"[unparseable LLM reply masked]"` —
    nothing is masked. 42 occurrences; negligible, but the string lies.

---

## Corrections to the previous handoff

| claim | reality |
|---|---|
| "~50 relevant docs/query" | **median 1.** That number was candidates/query, from bug #8. |
| "52,707 cached verdicts" (implying they're in the labels) | Only **13,987** reached the labels. Labels were 17h stale. Now 55,605. |
| "B beat A by 2.2× — hard-negative filtering wins" | **Did not replicate. Reversed.** A > B (ns). And A's own gain is a lexical artifact. |
| "0.6B honest baseline 0.287, deployed 4B 0.445" | Different eval set — not comparable to anything here. **Z = 0.3023** is the only valid baseline on these 440 queries. |
| "bakeoff ~8h for 6 arms" | **~2.5h.** One 0.6B arm at step_cap=8 over 2,560 queries trains in **~25 min**. |

---

## Timings (corrected, measured)

- 0.6B screen train, 1 arm, 2,560 queries, step_cap=8, 1 epoch: **~25 min**
- `p1_3` eval, 1 model: **~43–60 min** (600,738 docs, batch 256 → 2,357 batches)
- 6-arm screen: **~2.5h** · 7-model eval: **~5h**
- `p0_3` re-merge from cache (no new calls): **~4 min**

---

## Open decisions (need a human call, deliberately not guessed)

### 1. Bug #4 — which side does `marginmse_scale` scale?

The fix is a knob (`DH2_MARGINMSE_MODE`), defaulting to `student_scale`. Both options are
defensible and **neither is validated**; C's number is uninterpretable until both run
(~25 min/arm on the 0.6B). Measured, not argued:

| mode | on a hard pair | cost | measured |
|---|---|---|---|
| `student_scale` (default) | margin 0.05 × 20 = 1.0 = teacher | overshoots easy pairs | an already-correct margin of 0.3 → 6.0 vs teacher max 1.0, gradient **+200 pushes it back down** |
| `teacher_scale` | teacher 1.0 / 20 = 0.05 = margin | trains ~**394×** slower at the shared lr=2e-5 | measured `|grad|` ratio 394× |
| `none` | — | reproduces the −0.0977 | error floor **0.9025** |

The 2×2 that would settle it: {student_scale, teacher_scale} × {lr 2e-5, lr scaled}.
Only worth running if C is still a live arm — and per the RASC recommendation, it isn't
before July 17.

### 2. Arm B's semantics changed under the bug #7 fix

`fn_flag` is now per-`(query, doc)`. B's hard-filter therefore removes a document only
from the query it was flagged for, not from all 3,000. That is the correct semantics, but
it means **the new B is not comparable to the old B**, and the "B beats A by 2.2×" headline
was already dead. Old flat sidecars still load (with a warning) if you want the old
behavior for a controlled comparison.

### 3. Global BM25 in production

`prod/07_retrieve_rank.py` fuses BM25 via RRF today. Under `DH_REGISTER_AWARE_CASCADE=1`
it is **dropped** from the union per the recommendation (measured cross-register
regression), replaced by the routed exact-identifier channel. The graph/R-GCN signal is
**kept** as a union channel — the recommendation doesn't mention it, it abstains when no
entity links, and adding a channel cannot lower union recall. Say so if you disagree.

### 4. GroupRank vs stage 09's determinism invariant

09 verifies Jaccard = 1.0 and tau = 1.0. Qwen reranking is a temperature-free forward pass
over a fixed candidate set and GroupRank's partitions are seeded — but vLLM's continuous
batching does not guarantee bitwise-identical logits across runs even at temperature 0.
**Validate 09 with the cascade ON and GroupRank OFF first.**

---

## Next

0. **Embed the corpus with the untouched 0.6B (~45 min). Nothing in the cascade runs
   without it.** `p0_1`, `p1_4`, and `07` all warn and fall back to the old
   fine-tuned-only pool if `--vectors-base-06b` / `DH_CASCADE_VECTORS_BASE_06B` is unset —
   which is the exact configuration measured to lose 0.089 R@10 on low-overlap queries.
   A "cascade" without the semantic hedge ships the regression under a new name.

1. **Fix the query generator — this is still the actual work.** Generate queries in *scout
   register*, not patent paraphrase. The few-shot prompt was supposed to do this and
   demonstrably doesn't: origin docs carry +0.100 more query-word overlap than equally-
   relevant alternatives. Options: generate from doc A, validate against doc B; force
   vocabulary substitution (drug class → mechanism → indication); or hand-write a small
   scout-register eval set as ground truth. **Nothing downstream matters until the training
   pairs contain the gap.** The cascade buys time; it does not fix the data.
2. **Stop evaluating on `qrels_exact_origin`.** It scores paraphrase retrieval. Use graded
   utility at full coverage. Report LOW/HIGH overlap buckets separately — the aggregate hides
   the trade.
3. **Full gray-zone adjudication on Drew, free.** 131,983 keys missing; p0_3 is network-bound
   and the GPU idles. 16 workers × ~2s ≈ **4.6h, one overnight, no H100 rent.**
4. **Then re-screen at power.** `p0_4 --eval-frac 0.50` → ~1,500 held out (SE ≈ 0.0094).
   Fix bugs #3–#6 first, or drop C/D/E and screen only A/B/F.
5. **Reconsider the teacher ensemble.** BGE is half the signal and it's measuring the wrong
   thing. Options: drop it; rank-normalize per query before comparing; or keep it explicitly
   as the lexical channel in a hybrid and stop treating its disagreement with Qwen as noise.

---

## Environment / access

- **Vast H100: DESTROYED.** Everything below is on Drew.
- **Drew**: `alex@aiserver`, artifacts in `~/dh2_takehome/`:
  - `data/llm_cache.jsonl` — **52,709 verdicts. The expensive artifact. Back this up.**
  - `data/teacher_scores_v1.jsonl` — 329,761 dual cross-encoder scores (GPU-hours)
  - `data/multi_positive_labels_v1.jsonl` — 329,761 rows @ 28.5% coverage
  - `data/qrels/` — incl. `held_out_query_ids.json` (440)
  - `data/reports/` — 8 evals (json + md)
  - `data/per_query_r10.json` — 8 arrays × 440, the paired-bootstrap substrate
  - `code/` — this tree, `dh2_env.sh` key-scrubbed
- **Models are NOT pulled** — reproducible in ~25 min each. Doc vectors ~45 min.
- Restart: `scripts/push_pipeline2_to_h100.sh <host>` then `source dh2_env.sh`.
  **`DH2_TEACHER_LLM_MODEL` in that script still defaults to `claude-sonnet-4-6` (dead
  1min.ai). Must be `glm-4.6` + z.ai base URL, or the cache keys won't match and you'll
  re-pay for 52,709 verdicts.**

## Operational lessons (carry forward)

- **`tmux new-window` does NOT inherit your shell env.** Put `cd` + `source dh2_env.sh`
  *inside* a wrapper script; don't pass a bare command. Cost ~40 min today.
- **A window that vanishes from the status bar didn't get killed — it exited.** tmux closes
  the window when the command returns. Read the log; don't re-guess.
- **`tmux capture-pane` looks empty because of tqdm.** Pipe through `tr '\r' '\n'`.
- **`scp` takes `-P`, `ssh` takes `-p`.**
- **Verify a script exists before launching it** (`ls -la` after the heredoc). A missing
  file makes the tmux window open and close instantly, looking identical to a crash.
- **The honest progress meter is the artifact, not the log.** `ls reports/*.json | wc -l`.
- Reading the loss source predicted C, D, and E's outcomes correctly. Extrapolating numbers
  from those readings ("40% zero-gradient" → 3.3%; "25% coverage" → 7.2%; "every gradient
  step was damage" → 4 of 6 arms beat Z) was wrong nearly every time. **Run the check.**
